use super::protocol::{Args, Reply};
use super::replication::{self, select_devices};
use super::AppState;
use crate::db::model::{Device, DeviceStatus, Host, HostStatus};
use crate::db::queries;
use crate::db::Db;
use crate::error::MogError;
use crate::storage::path as storepath;
use std::sync::Arc;

/// Maps an internal database error to the client-facing `ERR db` line. The
/// detailed error is logged server-side only — reflecting `e.to_string()` to
/// an unauthenticated client would leak SQL text, table/column/constraint
/// names, and (for SQLite) the database file path.
fn db_err(e: anyhow::Error) -> MogError {
    tracing::error!("database error: {e:#}");
    MogError::db()
}

fn req<'a>(args: &'a Args, name: &str) -> Option<&'a str> {
    args.get(name).map(|s| s.as_str()).filter(|s| !s.is_empty())
}

async fn resolve_domain(db: &Db, args: &Args) -> Result<crate::db::model::Domain, MogError> {
    let name = req(args, "domain").ok_or_else(MogError::no_domain)?;
    queries::get_domain_by_name(db, name)
        .await
        .map_err(db_err)?
        .ok_or_else(MogError::unreg_domain)
}

/// Resolves the `class` arg to a classid, treating missing/empty/"default" as
/// the implicit classid 0 (mirrors real MogileFS's implicit default class).
async fn resolve_class(db: &Db, dmid: i64, class_arg: Option<&str>) -> Result<i64, MogError> {
    match class_arg {
        None | Some("") | Some("default") => Ok(0),
        Some(name) => {
            let class = queries::get_class_by_name(db, dmid, name)
                .await
                .map_err(db_err)?
                .ok_or_else(MogError::unreg_class)?;
            Ok(class.classid)
        }
    }
}

type DevHost = (Device, Host);

fn build_url(hostip: &str, http_port: i64, devid: i64, fid: i64) -> String {
    format!("http://{hostip}:{http_port}{}", storepath::uri_path(devid, fid))
}

pub async fn dispatch(state: &Arc<AppState>, cmd: &str, args: &Args) -> Result<Reply, MogError> {
    match cmd {
        "noop" | "clear_cache" | "do_monitor_round" => Ok(Reply::new()),
        "sleep" => cmd_sleep(args).await,

        "create_open" => cmd_create_open(state, args).await,
        "create_close" => cmd_create_close(state, args).await,
        "get_paths" => cmd_get_paths(state, args).await,
        "get_domains" => cmd_get_domains(state, args).await,
        "list_keys" => cmd_list_keys(state, args).await,
        "delete" => cmd_delete(state, args).await,
        "rename" => cmd_rename(state, args).await,
        "list_fids" => cmd_list_fids(state, args).await,
        "file_info" => cmd_file_info(state, args).await,
        "replicate_now" => cmd_replicate_now(state, args).await,

        "set_weight" => cmd_set_weight(state, args).await,
        "set_state" => cmd_set_state(state, args).await,
        "get_hosts" => cmd_get_hosts(state, args).await,
        "get_devices" => cmd_get_devices(state, args).await,

        "create_domain" => cmd_create_domain(state, args).await,
        "delete_domain" => cmd_delete_domain(state, args).await,
        "create_class" => cmd_create_class(state, args).await,
        "update_class" | "updateclass" => cmd_update_class(state, args).await,
        "delete_class" => cmd_delete_class(state, args).await,
        "create_host" => cmd_create_host(state, args).await,
        "update_host" => cmd_update_host(state, args).await,
        "delete_host" => cmd_delete_host(state, args).await,
        "create_device" => cmd_create_device(state, args).await,

        "server_setting" | "server_settings" => cmd_server_setting(state, args).await,
        "set_server_setting" => cmd_set_server_setting(state, args).await,

        "httpcopy" => cmd_httpcopy(state, args).await,
        "edit_file" => cmd_edit_file(state, args).await,

        "fsck_start" => cmd_fsck_start(state, args).await,
        "fsck_stop" => cmd_fsck_stop(state, args).await,
        "fsck_reset" => cmd_fsck_reset(state, args).await,
        "fsck_clearlog" => cmd_fsck_clearlog(state, args).await,
        "fsck_getlog" => cmd_fsck_getlog(state, args).await,
        "fsck_status" => cmd_fsck_status(state, args).await,

        "rebalance_start" => cmd_rebalance_start(state, args).await,
        "rebalance_stop" => cmd_rebalance_stop(state, args).await,
        "rebalance_status" => cmd_rebalance_status(state, args).await,

        _ => Err(MogError::unknown_command()),
    }
}

async fn cmd_sleep(args: &Args) -> Result<Reply, MogError> {
    let dur: f64 = args.get("duration").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    if dur > 0.0 {
        tokio::time::sleep(std::time::Duration::from_secs_f64(dur.min(30.0))).await;
    }
    Ok(Reply::new())
}

// ---------------- create_open / create_close ----------------

async fn cmd_create_open(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let dom = resolve_domain(db, args).await?;
    let classid = resolve_class(db, dom.dmid, req(args, "class")).await?;
    let key = req(args, "key").ok_or_else(MogError::no_key)?;
    let multi_dest = args.get("multi_dest").map(|v| v == "1").unwrap_or(false);
    let want = if multi_dest { 3 } else { 1 };

    let candidates = select_devices(db, want, &[]).await?;

    let fid = if let Some(fid_str) = req(args, "fid") {
        let f: i64 = fid_str.parse().map_err(|_| MogError::bad_params_msg("invalid fid"))?;
        if queries::tempfile_or_file_fid_exists(db, f).await.map_err(db_err)? {
            return Err(MogError::fid_in_use());
        }
        let devids_csv = candidates.iter().map(|c| c.devid.to_string()).collect::<Vec<_>>().join(",");
        queries::create_tempfile_with_fid(db, f, dom.dmid, Some(key), classid, &devids_csv).await.map_err(db_err)?;
        f
    } else {
        let devids_csv = candidates.iter().map(|c| c.devid.to_string()).collect::<Vec<_>>().join(",");
        queries::create_tempfile(db, dom.dmid, Some(key), classid, &devids_csv).await.map_err(db_err)?
    };

    let mut reply = Reply::new();
    if multi_dest && candidates.len() > 1 {
        reply.insert("fid".into(), fid.to_string());
        reply.insert("dev_count".into(), candidates.len().to_string());
        for (i, c) in candidates.iter().enumerate() {
            reply.insert(format!("devid_{}", i + 1), c.devid.to_string());
            reply.insert(format!("path_{}", i + 1), build_url(&c.hostip, c.http_port, c.devid, fid));
        }
    } else {
        let c = &candidates[0];
        reply.insert("fid".into(), fid.to_string());
        reply.insert("devid".into(), c.devid.to_string());
        reply.insert("path".into(), build_url(&c.hostip, c.http_port, c.devid, fid));
    }
    Ok(reply)
}

async fn cmd_create_close(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let fid: i64 = req(args, "fid").ok_or_else(MogError::no_fid)?.parse().map_err(|_| MogError::bad_params_msg("invalid fid"))?;
    let devid: i64 = req(args, "devid").ok_or_else(MogError::no_devid)?.parse().map_err(|_| MogError::bad_params_msg("invalid devid"))?;
    let path_arg = req(args, "path").ok_or_else(MogError::no_path)?;

    let tf = queries::get_tempfile(db, fid).await.map_err(db_err)?.ok_or_else(MogError::no_temp_file)?;

    // Validate BEFORE consuming the tempfile, so a validation failure (wrong
    // devid/path/size/checksum) leaves the reservation intact and the client
    // can retry create_close — rather than destroying it on the first attempt.
    if !tf.devid_list().contains(&devid) {
        return Err(MogError::invalid_destdev());
    }

    let expected_uri = storepath::uri_path(devid, fid);
    if !path_arg.ends_with(&expected_uri) {
        return Err(MogError::bogus_args());
    }

    let key_arg = req(args, "key");

    let Some(key) = key_arg else {
        // Client is abandoning this upload: consume the tempfile and discard
        // the blob it PUT to its owning storage host.
        queries::delete_tempfile(db, fid).await.map_err(db_err)?;
        let _ = crate::storage::client::delete(db, devid, fid).await;
        return Ok(Reply::new());
    };

    // The blob was PUT by the client directly to the device's owning storage
    // host; verify size/checksum there over HTTP (works whether that host is
    // this process or a remote node).
    let actual_size = crate::storage::client::head(db, devid, fid)
        .await
        .map_err(|_| MogError::size_verify_error())?
        .ok_or_else(MogError::size_verify_error)? as i64;
    if let Some(size_str) = req(args, "size") {
        let expected: i64 = size_str.parse().map_err(|_| MogError::bad_params_msg("invalid size"))?;
        if expected != actual_size {
            return Err(MogError::size_mismatch());
        }
    }

    let checksum_to_store: Option<(String, String)> = if let Some(checksum_arg) = req(args, "checksum") {
        let (alg, hexval) = checksum_arg.split_once(':').ok_or_else(MogError::invalid_checksum_format)?;
        let verify = args.get("checksumverify").map(|v| v == "1").unwrap_or(true);
        if verify {
            let data = crate::storage::client::get_bytes(db, devid, fid)
                .await
                .map_err(|_| MogError::size_verify_error())?
                .ok_or_else(MogError::size_verify_error)?;
            let computed =
                crate::util::compute_checksum_hex(alg, &data).ok_or_else(MogError::invalid_checksum_format)?;
            if !computed.eq_ignore_ascii_case(hexval) {
                return Err(MogError::checksum_mismatch());
            }
        }
        Some((alg.to_string(), hexval.to_string()))
    } else {
        None
    };

    // Consume the tempfile now (all validation has passed). This is also the
    // race guard: if a concurrent create_close already consumed it, we deleted
    // nothing and must not double-finalize.
    if !queries::delete_tempfile(db, fid).await.map_err(db_err)? {
        return Err(MogError::no_temp_file());
    }

    super::store_ops::finalize_blob(state, fid, tf.dmid, key, tf.classid, devid, actual_size, checksum_to_store).await?;

    Ok(Reply::new())
}

// ---------------- get_paths ----------------

async fn cmd_get_paths(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let dom = resolve_domain(db, args).await?;
    let key = req(args, "key").ok_or_else(MogError::no_key)?;
    let file = queries::get_file_by_key(db, dom.dmid, key).await.map_err(db_err)?.ok_or_else(MogError::unknown_key)?;

    let pathcount: usize = args.get("pathcount").and_then(|s| s.parse().ok()).unwrap_or(2).max(2);
    let noverify = args.get("noverify").map(|v| v == "1").unwrap_or(false);

    let devids = queries::get_devids_for_fid(db, file.fid).await.map_err(db_err)?;
    let mut pairs: Vec<DevHost> = Vec::new();
    for devid in &devids {
        if let Ok(Some(dev)) = queries::get_device(db, *devid).await {
            if let Ok(Some(host)) = queries::get_host(db, dev.hostid).await {
                pairs.push((dev, host));
            }
        }
    }

    let (mut primary, mut backup): (Vec<DevHost>, Vec<DevHost>) = (Vec::new(), Vec::new());
    for (d, h) in pairs.into_iter() {
        let readable = DeviceStatus::parse(&d.status).map(|s| s.readable()).unwrap_or(false)
            && HostStatus::parse(&h.status).map(|s| s != HostStatus::Down).unwrap_or(false);
        if readable {
            primary.push((d, h));
        } else {
            backup.push((d, h));
        }
    }
    primary.sort_by(|a, b| b.0.weight.cmp(&a.0.weight));
    let mut chosen: Vec<DevHost> = primary.into_iter().take(pathcount).collect();
    if chosen.is_empty() {
        if let Some(b) = backup.drain(..).next() {
            chosen.push(b);
        }
    }
    if chosen.is_empty() {
        return Err(MogError::unknown_key());
    }

    if !noverify {
        if let Some((d, _h)) = chosen.first() {
            // Verify the first path really has the blob, over HTTP to its owning
            // host (which may be a remote storage node).
            let present = crate::storage::client::exists(db, d.devid, file.fid).await.unwrap_or(true);
            if !present && chosen.len() > 1 {
                chosen.remove(0);
            }
        }
    }

    let mut reply = Reply::new();
    reply.insert("paths".into(), chosen.len().to_string());
    for (i, (d, h)) in chosen.iter().enumerate() {
        let hostip = h.hostip.clone().unwrap_or_default();
        reply.insert(format!("path{}", i + 1), build_url(&hostip, h.http_port, d.devid, file.fid));
    }
    Ok(reply)
}

// ---------------- get_domains / list_keys / delete / rename / list_fids / file_info ----------------

async fn cmd_get_domains(state: &Arc<AppState>, _args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let domains = queries::list_domains(db).await.map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("domains".into(), domains.len().to_string());
    for (i, dom) in domains.iter().enumerate() {
        let n = i + 1;
        reply.insert(format!("domain{n}"), dom.namespace.clone());
        let classes = queries::list_classes(db, dom.dmid).await.map_err(db_err)?;
        reply.insert(format!("domain{n}classes"), classes.len().to_string());
        for (j, c) in classes.iter().enumerate() {
            let m = j + 1;
            reply.insert(format!("domain{n}class{m}name"), c.classname.clone());
            reply.insert(format!("domain{n}class{m}mindevcount"), c.mindevcount.to_string());
        }
    }
    Ok(reply)
}

async fn cmd_list_keys(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let dom = resolve_domain(db, args).await?;
    let prefix = args.get("prefix").map(|s| s.as_str()).unwrap_or("");
    let after = req(args, "after");
    let limit: i64 = args.get("limit").and_then(|s| s.parse().ok()).unwrap_or(1000).clamp(1, 1000);

    let rows = queries::list_keys(db, dom.dmid, prefix, after, limit).await.map_err(db_err)?;
    if rows.is_empty() {
        return Err(MogError::none_match());
    }
    let mut reply = Reply::new();
    reply.insert("key_count".into(), rows.len().to_string());
    for (i, (k, _len, _fid)) in rows.iter().enumerate() {
        reply.insert(format!("key_{}", i + 1), k.clone());
    }
    if rows.len() as i64 == limit {
        reply.insert("next_after".into(), rows.last().unwrap().0.clone());
    }
    Ok(reply)
}

async fn cmd_delete(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let dom = resolve_domain(db, args).await?;
    let key = req(args, "key").ok_or_else(MogError::no_key)?;
    let file = queries::get_file_by_key(db, dom.dmid, key).await.map_err(db_err)?.ok_or_else(MogError::unknown_key)?;
    queries::delete_file_row(db, file.fid).await.map_err(db_err)?;
    queries::dequeue_replicate(db, file.fid).await.map_err(db_err)?;
    queries::queue_delete(db, file.fid).await.map_err(db_err)?;
    Ok(Reply::new())
}

async fn cmd_rename(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let dom = resolve_domain(db, args).await?;
    let from_key = req(args, "from_key").ok_or_else(MogError::no_key)?;
    let to_key = req(args, "to_key").ok_or_else(MogError::no_key)?;
    let file = queries::get_file_by_key(db, dom.dmid, from_key).await.map_err(db_err)?.ok_or_else(MogError::unknown_key)?;
    if queries::get_file_by_key(db, dom.dmid, to_key).await.map_err(db_err)?.is_some() {
        return Err(MogError::key_exists());
    }
    queries::rename_file_key(db, dom.dmid, file.fid, to_key).await.map_err(db_err)?;
    Ok(Reply::new())
}

async fn cmd_list_fids(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let from: i64 = args.get("from").and_then(|s| s.parse().ok()).unwrap_or(1).max(0);
    let count: i64 = args.get("to").and_then(|s| s.parse().ok()).unwrap_or(100).clamp(1, 500);
    let rows = queries::list_fids_range(db, from, count).await.map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("fid_count".into(), rows.len().to_string());
    for (i, f) in rows.iter().enumerate() {
        let n = i + 1;
        let dom_name = queries::get_domain_by_id(db, f.dmid).await.map_err(db_err)?.map(|d| d.namespace).unwrap_or_default();
        let cls_name = replication::class_name(db, f.dmid, f.classid).await?;
        reply.insert(format!("fid_{n}_fid"), f.fid.to_string());
        reply.insert(format!("fid_{n}_domain"), dom_name);
        reply.insert(format!("fid_{n}_class"), cls_name);
        reply.insert(format!("fid_{n}_key"), f.dkey.clone());
        reply.insert(format!("fid_{n}_length"), f.length.unwrap_or(0).to_string());
        reply.insert(format!("fid_{n}_devcount"), f.devcount.to_string());
    }
    Ok(reply)
}

async fn cmd_file_info(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let dom = resolve_domain(db, args).await?;
    let key = req(args, "key").ok_or_else(MogError::no_key)?;
    let file = queries::get_file_by_key(db, dom.dmid, key).await.map_err(db_err)?.ok_or_else(MogError::unknown_key)?;

    let mut reply = Reply::new();
    reply.insert("fid".into(), file.fid.to_string());
    reply.insert("domain".into(), dom.namespace.clone());
    reply.insert("class".into(), replication::class_name(db, dom.dmid, file.classid).await?);
    reply.insert("key".into(), file.dkey.clone());
    reply.insert("length".into(), file.length.unwrap_or(0).to_string());
    reply.insert("devcount".into(), file.devcount.to_string());
    if args.get("devices").map(|v| v == "1").unwrap_or(false) {
        let devids = queries::get_devids_for_fid(db, file.fid).await.map_err(db_err)?;
        reply.insert("devids".into(), devids.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","));
    }
    if let Some((alg, hex)) = queries::get_checksum(db, file.fid).await.map_err(db_err)? {
        reply.insert("checksum".into(), format!("{alg}:{hex}"));
    }
    Ok(reply)
}

async fn cmd_replicate_now(state: &Arc<AppState>, _args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let pending = queries::next_to_replicate(db, 100000).await.map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("count".into(), pending.len().to_string());
    Ok(reply)
}

// ---------------- device / host admin ----------------

async fn cmd_set_weight(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let hostname = req(args, "host").ok_or_else(MogError::bad_params)?;
    let devid: i64 = req(args, "device").ok_or_else(MogError::bad_params)?.parse().map_err(|_| MogError::bad_params())?;
    let weight: i64 = req(args, "weight").ok_or_else(MogError::bad_params)?.parse().map_err(|_| MogError::bad_params())?;
    let host = queries::get_host_by_name(db, hostname).await.map_err(db_err)?.ok_or_else(MogError::unknown_host)?;
    let dev = queries::get_device(db, devid).await.map_err(db_err)?.ok_or_else(MogError::no_device)?;
    if dev.hostid != host.hostid {
        return Err(MogError::host_mismatch());
    }
    queries::update_device_weight(db, devid, weight).await.map_err(db_err)?;
    Ok(Reply::new())
}

async fn cmd_set_state(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let hostname = req(args, "host").ok_or_else(MogError::bad_params)?;
    let devid: i64 = req(args, "device").ok_or_else(MogError::bad_params)?.parse().map_err(|_| MogError::bad_params())?;
    let new_state = req(args, "state").ok_or_else(MogError::bad_params)?;
    let host = queries::get_host_by_name(db, hostname).await.map_err(db_err)?.ok_or_else(MogError::unknown_host)?;
    let dev = queries::get_device(db, devid).await.map_err(db_err)?.ok_or_else(MogError::no_device)?;
    if dev.hostid != host.hostid {
        return Err(MogError::host_mismatch());
    }
    let parsed = DeviceStatus::parse(new_state).ok_or_else(MogError::unknown_state)?;
    let current = DeviceStatus::parse(&dev.status);
    if current == Some(DeviceStatus::Dead) && parsed == DeviceStatus::Alive {
        return Err(MogError::state_too_high());
    }
    queries::update_device_status(db, devid, parsed.as_str()).await.map_err(db_err)?;
    Ok(Reply::new())
}

async fn cmd_get_hosts(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let filter: Option<i64> = args.get("hostid").and_then(|s| s.parse().ok());
    let hosts: Vec<Host> = queries::list_hosts(db).await.map_err(db_err)?.into_iter().filter(|h| filter.map(|f| f == h.hostid).unwrap_or(true)).collect();
    let mut reply = Reply::new();
    reply.insert("hosts".into(), hosts.len().to_string());
    for (i, h) in hosts.iter().enumerate() {
        let n = i + 1;
        reply.insert(format!("host{n}hostid"), h.hostid.to_string());
        reply.insert(format!("host{n}status"), h.status.clone());
        reply.insert(format!("host{n}hostname"), h.hostname.clone());
        reply.insert(format!("host{n}hostip"), h.hostip.clone().unwrap_or_default());
        reply.insert(format!("host{n}http_port"), h.http_port.to_string());
        reply.insert(format!("host{n}http_get_port"), h.http_port.to_string());
    }
    Ok(reply)
}

async fn cmd_get_devices(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let filter: Option<i64> = args.get("devid").and_then(|s| s.parse().ok());
    let devs: Vec<Device> = queries::list_devices(db).await.map_err(db_err)?.into_iter().filter(|d| filter.map(|f| f == d.devid).unwrap_or(true)).collect();
    let mut reply = Reply::new();
    reply.insert("devices".into(), devs.len().to_string());
    for (i, d) in devs.iter().enumerate() {
        let n = i + 1;
        reply.insert(format!("dev{n}id"), d.devid.to_string());
        reply.insert(format!("dev{n}hostid"), d.hostid.to_string());
        reply.insert(format!("dev{n}status"), d.status.clone());
        reply.insert(format!("dev{n}weight"), d.weight.to_string());
        reply.insert(format!("dev{n}mb_total"), d.mb_total.unwrap_or(0).to_string());
        reply.insert(format!("dev{n}mb_used"), d.mb_used.unwrap_or(0).to_string());
    }
    Ok(reply)
}

// ---------------- domain / class / host / device CRUD ----------------

async fn cmd_create_domain(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let name = req(args, "domain").ok_or_else(MogError::no_domain)?;
    if queries::get_domain_by_name(db, name).await.map_err(db_err)?.is_some() {
        return Err(MogError::domain_exists());
    }
    queries::create_domain(db, name).await.map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("domain".into(), name.to_string());
    Ok(reply)
}

async fn cmd_delete_domain(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let dom = resolve_domain(db, args).await?;
    if queries::count_files_in_domain(db, dom.dmid).await.map_err(db_err)? > 0 {
        return Err(MogError::domain_has_files());
    }
    queries::delete_domain(db, dom.dmid).await.map_err(db_err)?;
    Ok(Reply::new())
}

async fn cmd_create_class(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let dom = resolve_domain(db, args).await?;
    let name = req(args, "class").ok_or_else(MogError::no_class)?;
    let mindevcount: i64 = args
        .get("mindevcount")
        .and_then(|s| s.parse().ok())
        .unwrap_or(state.cfg.default_min_devcount as i64);
    if mindevcount < 1 {
        return Err(MogError::invalid_mindevcount());
    }
    if queries::get_class_by_name(db, dom.dmid, name).await.map_err(db_err)?.is_some() {
        return Err(MogError::class_exists());
    }
    let classid = queries::create_class(db, dom.dmid, name, mindevcount).await.map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("class".into(), name.to_string());
    reply.insert("classid".into(), classid.to_string());
    reply.insert("mindevcount".into(), mindevcount.to_string());
    Ok(reply)
}

async fn cmd_update_class(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let dom = resolve_domain(db, args).await?;
    let name = req(args, "class").ok_or_else(MogError::no_class)?;
    let class = queries::get_class_by_name(db, dom.dmid, name).await.map_err(db_err)?.ok_or_else(MogError::class_not_found)?;
    let mindevcount: i64 = req(args, "mindevcount").ok_or_else(MogError::bad_params)?.parse().map_err(|_| MogError::bad_params())?;
    if mindevcount < 1 {
        return Err(MogError::invalid_mindevcount());
    }
    queries::update_class_mindevcount(db, dom.dmid, class.classid, mindevcount).await.map_err(db_err)?;
    Ok(Reply::new())
}

async fn cmd_delete_class(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let dom = resolve_domain(db, args).await?;
    let name = req(args, "class").ok_or_else(MogError::no_class)?;
    let class = queries::get_class_by_name(db, dom.dmid, name).await.map_err(db_err)?.ok_or_else(MogError::class_not_found)?;
    if queries::count_files_in_class(db, dom.dmid, class.classid).await.map_err(db_err)? > 0 {
        return Err(MogError::class_has_files());
    }
    queries::delete_class(db, dom.dmid, class.classid).await.map_err(db_err)?;
    Ok(Reply::new())
}

async fn cmd_create_host(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let name = req(args, "host").ok_or_else(MogError::no_host)?;
    if queries::get_host_by_name(db, name).await.map_err(db_err)?.is_some() {
        return Err(MogError::host_exists());
    }
    let ip = req(args, "ip").ok_or_else(MogError::no_ip)?;
    let port: i64 = req(args, "port").ok_or_else(MogError::no_port)?.parse().map_err(|_| MogError::bad_params())?;
    let hostid = queries::create_host(db, name, Some(ip), port).await.map_err(db_err)?;
    if let Some(status) = req(args, "status") {
        if HostStatus::parse(status).is_some() {
            queries::update_host_status(db, hostid, status).await.map_err(db_err)?;
        }
    }
    let mut reply = Reply::new();
    reply.insert("hostid".into(), hostid.to_string());
    Ok(reply)
}

async fn cmd_update_host(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let name = req(args, "host").ok_or_else(MogError::no_host)?;
    let host = queries::get_host_by_name(db, name).await.map_err(db_err)?.ok_or_else(MogError::unknown_host)?;
    if let Some(ip) = req(args, "ip") {
        queries::update_host_ip(db, host.hostid, ip).await.map_err(db_err)?;
    }
    if let Some(port) = req(args, "port") {
        let port: i64 = port.parse().map_err(|_| MogError::bad_params())?;
        queries::update_host_port(db, host.hostid, port).await.map_err(db_err)?;
    }
    if let Some(status) = req(args, "status") {
        HostStatus::parse(status).ok_or_else(MogError::unknown_state)?;
        queries::update_host_status(db, host.hostid, status).await.map_err(db_err)?;
    }
    Ok(Reply::new())
}

async fn cmd_delete_host(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let name = req(args, "host").ok_or_else(MogError::no_host)?;
    let host = queries::get_host_by_name(db, name).await.map_err(db_err)?.ok_or_else(MogError::unknown_host)?;
    if queries::devices_for_host(db, host.hostid).await.map_err(db_err)? > 0 {
        return Err(MogError::host_not_empty());
    }
    queries::delete_host(db, host.hostid).await.map_err(db_err)?;
    Ok(Reply::new())
}

async fn cmd_create_device(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let hostname = req(args, "host").ok_or_else(MogError::no_host)?;
    let host = queries::get_host_by_name(db, hostname).await.map_err(db_err)?.ok_or_else(MogError::unknown_host)?;
    let devid: i64 = req(args, "devid").ok_or_else(MogError::no_devid)?.parse().map_err(|_| MogError::bad_params())?;
    if queries::get_device(db, devid).await.map_err(db_err)?.is_some() {
        return Err(MogError::device_exists());
    }
    queries::create_device(db, devid, host.hostid).await.map_err(db_err)?;
    if let Some(status) = req(args, "status") {
        if DeviceStatus::parse(status).is_some() {
            queries::update_device_status(db, devid, status).await.map_err(db_err)?;
        }
    }
    std::fs::create_dir_all(storepath::device_root(&state.cfg.docroot, devid)).ok();
    Ok(Reply::new())
}

// ---------------- server settings ----------------

async fn cmd_server_setting(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    if let Some(key) = req(args, "key") {
        let value = queries::get_setting(db, key).await.map_err(db_err)?.unwrap_or_default();
        let mut reply = Reply::new();
        reply.insert("key".into(), key.to_string());
        reply.insert("value".into(), value);
        return Ok(reply);
    }
    let all = queries::list_settings(db).await.map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("key_count".into(), all.len().to_string());
    for (i, (k, v)) in all.iter().enumerate() {
        reply.insert(format!("key_{}", i + 1), k.clone());
        reply.insert(format!("value_{}", i + 1), v.clone());
    }
    Ok(reply)
}

async fn cmd_set_server_setting(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let key = req(args, "key").ok_or_else(MogError::bad_params)?;
    let value = args.get("value").map(|s| s.as_str()).unwrap_or("");
    queries::set_setting(db, key, value).await.map_err(db_err)?;
    Ok(Reply::new())
}

// ---------------- httpcopy / edit_file ----------------

async fn cmd_httpcopy(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let sdevid: i64 = req(args, "sdevid").ok_or_else(MogError::bad_params)?.parse().map_err(|_| MogError::bad_params())?;
    let ddevid: i64 = req(args, "ddevid").ok_or_else(MogError::bad_params)?.parse().map_err(|_| MogError::bad_params())?;
    let fid: i64 = req(args, "fid").ok_or_else(MogError::no_fid)?.parse().map_err(|_| MogError::bad_params())?;

    queries::get_device(db, sdevid).await.map_err(db_err)?.ok_or_else(MogError::unknown_device)?;
    queries::get_device(db, ddevid).await.map_err(db_err)?.ok_or_else(MogError::unknown_device)?;

    // Copy across the (possibly different) owning hosts over HTTP.
    crate::storage::client::copy(db, sdevid, ddevid, fid).await.map_err(|e| {
        tracing::error!("httpcopy fid {fid} dev {sdevid}->{ddevid}: {e}");
        MogError::new("copy_err", "failed to copy file between devices")
    })?;
    queries::add_file_on(db, fid, ddevid).await.map_err(db_err)?;
    Ok(Reply::new())
}

/// Experimental in the reference tracker too: allocates a fresh temp
/// destination for an existing key so a client can read-modify-write it,
/// then close over the new copy exactly like a normal `create_open`.
async fn cmd_edit_file(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let dom = resolve_domain(db, args).await?;
    let key = req(args, "key").ok_or_else(MogError::no_key)?;
    let file = queries::get_file_by_key(db, dom.dmid, key).await.map_err(db_err)?.ok_or_else(MogError::unknown_key)?;
    let old_devids = queries::get_devids_for_fid(db, file.fid).await.map_err(db_err)?;
    let old_devid = *old_devids.first().ok_or_else(MogError::unknown_key)?;
    let old_dev = queries::get_device(db, old_devid).await.map_err(db_err)?.ok_or_else(MogError::unknown_device)?;
    let old_host = queries::get_host(db, old_dev.hostid).await.map_err(db_err)?.ok_or_else(MogError::unknown_host)?;
    let oldpath = build_url(&old_host.hostip.clone().unwrap_or_default(), old_host.http_port, old_devid, file.fid);

    let candidates = select_devices(db, 1, &[]).await?;
    let devids_csv = candidates.iter().map(|c| c.devid.to_string()).collect::<Vec<_>>().join(",");
    let new_fid = queries::create_tempfile(db, dom.dmid, Some(key), file.classid, &devids_csv).await.map_err(db_err)?;
    let c = &candidates[0];
    let newpath = build_url(&c.hostip, c.http_port, c.devid, new_fid);

    let mut reply = Reply::new();
    reply.insert("fid".into(), new_fid.to_string());
    reply.insert("devid".into(), c.devid.to_string());
    reply.insert("oldpath".into(), oldpath);
    reply.insert("newpath".into(), newpath);
    reply.insert("class".into(), replication::class_name(db, dom.dmid, file.classid).await?);
    Ok(reply)
}

// ---------------- fsck ----------------

async fn cmd_fsck_start(state: &Arc<AppState>, _args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    queries::set_setting(db, "fsck_running", "1").await.map_err(db_err)?;
    queries::set_setting(db, "fsck_cursor", "0").await.map_err(db_err)?;
    Ok(Reply::new())
}

async fn cmd_fsck_stop(state: &Arc<AppState>, _args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    queries::set_setting(db, "fsck_running", "0").await.map_err(db_err)?;
    Ok(Reply::new())
}

async fn cmd_fsck_reset(state: &Arc<AppState>, _args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    queries::set_setting(db, "fsck_running", "0").await.map_err(db_err)?;
    queries::set_setting(db, "fsck_cursor", "0").await.map_err(db_err)?;
    Ok(Reply::new())
}

async fn cmd_fsck_clearlog(state: &Arc<AppState>, _args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    queries::fsck_clear_log(db).await.map_err(db_err)?;
    Ok(Reply::new())
}

async fn cmd_fsck_getlog(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let limit: i64 = args.get("limit").and_then(|s| s.parse().ok()).unwrap_or(100).clamp(1, 1000);
    let entries = queries::fsck_log_entries(db, limit).await.map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("row_count".into(), entries.len().to_string());
    for (i, e) in entries.iter().enumerate() {
        let n = i + 1;
        reply.insert(format!("row_{n}_fid"), e.fid.to_string());
        reply.insert(format!("row_{n}_evcode"), e.evcode.clone());
        reply.insert(format!("row_{n}_devid"), e.devid.map(|d| d.to_string()).unwrap_or_default());
        reply.insert(format!("row_{n}_utime"), e.utime.to_string());
    }
    Ok(reply)
}

async fn cmd_fsck_status(state: &Arc<AppState>, _args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let running = queries::get_setting(db, "fsck_running").await.map_err(db_err)?.unwrap_or_else(|| "0".to_string());
    let log_count = queries::fsck_log_count(db).await.map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("running".into(), running);
    reply.insert("log_count".into(), log_count.to_string());
    Ok(reply)
}

// ---------------- rebalance ----------------

async fn cmd_rebalance_start(state: &Arc<AppState>, _args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    // Seed the queue: files on the currently most-utilized device are
    // candidates to move to less-utilized ones.
    let devices = queries::list_devices(db).await.map_err(db_err)?;
    if let Some(busiest) = devices
        .iter()
        .filter(|d| matches!((d.mb_total, d.mb_used), (Some(t), Some(_)) if t > 0))
        .max_by(|a, b| {
            let ua = a.mb_used.unwrap_or(0) as f64 / a.mb_total.unwrap_or(1) as f64;
            let ub = b.mb_used.unwrap_or(0) as f64 / b.mb_total.unwrap_or(1) as f64;
            ua.partial_cmp(&ub).unwrap()
        })
    {
        let fids: Vec<i64> = queries::get_fids_for_device(db, busiest.devid, 50).await.map_err(db_err)?;
        for fid in fids {
            queries::queue_add(db, fid, Some(busiest.devid), "rebalance", None).await.map_err(db_err)?;
        }
    }
    queries::set_setting(db, "rebal_running", "1").await.map_err(db_err)?;
    Ok(Reply::new())
}

async fn cmd_rebalance_stop(state: &Arc<AppState>, _args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    queries::set_setting(db, "rebal_running", "0").await.map_err(db_err)?;
    queries::queue_clear(db, "rebalance").await.map_err(db_err)?;
    Ok(Reply::new())
}

async fn cmd_rebalance_status(state: &Arc<AppState>, _args: &Args) -> Result<Reply, MogError> {
    let db = &state.db;
    let running = queries::get_setting(db, "rebal_running").await.map_err(db_err)?.unwrap_or_else(|| "0".to_string());
    let queued = queries::queue_count(db, "rebalance").await.map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("running".into(), running);
    reply.insert("queued".into(), queued.to_string());
    Ok(reply)
}
