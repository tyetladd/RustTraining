use super::protocol::{Args, Reply};
use super::replication::{self, select_devices};
use super::AppState;
use crate::db::model::{Device, DeviceStatus, Host, HostStatus};
use crate::db::queries;
use crate::db::Conn;
use crate::error::MogError;
use crate::storage::path as storepath;
use std::sync::Arc;

fn db_err(e: anyhow::Error) -> MogError {
    MogError::db_msg(e.to_string())
}

fn req<'a>(args: &'a Args, name: &str) -> Option<&'a str> {
    args.get(name).map(|s| s.as_str()).filter(|s| !s.is_empty())
}

fn resolve_domain(conn: &Conn, args: &Args) -> Result<crate::db::model::Domain, MogError> {
    let name = req(args, "domain").ok_or_else(MogError::no_domain)?;
    queries::get_domain_by_name(conn, name)
        .map_err(db_err)?
        .ok_or_else(MogError::unreg_domain)
}

/// Resolves the `class` arg to a classid, treating missing/empty/"default" as
/// the implicit classid 0 (mirrors real MogileFS's implicit default class).
fn resolve_class(conn: &Conn, dmid: i64, class_arg: Option<&str>) -> Result<i64, MogError> {
    match class_arg {
        None | Some("") | Some("default") => Ok(0),
        Some(name) => {
            let class = queries::get_class_by_name(conn, dmid, name)
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

        "create_open" => cmd_create_open(state, args),
        "create_close" => cmd_create_close(state, args),
        "get_paths" => cmd_get_paths(state, args),
        "get_domains" => cmd_get_domains(state, args),
        "list_keys" => cmd_list_keys(state, args),
        "delete" => cmd_delete(state, args),
        "rename" => cmd_rename(state, args),
        "list_fids" => cmd_list_fids(state, args),
        "file_info" => cmd_file_info(state, args),
        "replicate_now" => cmd_replicate_now(state, args),

        "set_weight" => cmd_set_weight(state, args),
        "set_state" => cmd_set_state(state, args),
        "get_hosts" => cmd_get_hosts(state, args),
        "get_devices" => cmd_get_devices(state, args),

        "create_domain" => cmd_create_domain(state, args),
        "delete_domain" => cmd_delete_domain(state, args),
        "create_class" => cmd_create_class(state, args),
        "update_class" | "updateclass" => cmd_update_class(state, args),
        "delete_class" => cmd_delete_class(state, args),
        "create_host" => cmd_create_host(state, args),
        "update_host" => cmd_update_host(state, args),
        "delete_host" => cmd_delete_host(state, args),
        "create_device" => cmd_create_device(state, args),

        "server_setting" | "server_settings" => cmd_server_setting(state, args),
        "set_server_setting" => cmd_set_server_setting(state, args),

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

fn cmd_create_open(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let dom = resolve_domain(&conn, args)?;
    let classid = resolve_class(&conn, dom.dmid, req(args, "class"))?;
    let key = req(args, "key").ok_or_else(MogError::no_key)?;
    let multi_dest = args.get("multi_dest").map(|v| v == "1").unwrap_or(false);
    let want = if multi_dest { 3 } else { 1 };

    let candidates = select_devices(&conn, want, &[])?;

    let fid = if let Some(fid_str) = req(args, "fid") {
        let f: i64 = fid_str.parse().map_err(|_| MogError::bad_params_msg("invalid fid"))?;
        if queries::tempfile_or_file_fid_exists(&conn, f).map_err(db_err)? {
            return Err(MogError::fid_in_use());
        }
        let devids_csv = candidates.iter().map(|c| c.devid.to_string()).collect::<Vec<_>>().join(",");
        queries::create_tempfile_with_fid(&conn, f, dom.dmid, Some(key), classid, &devids_csv).map_err(db_err)?;
        f
    } else {
        let devids_csv = candidates.iter().map(|c| c.devid.to_string()).collect::<Vec<_>>().join(",");
        queries::create_tempfile(&conn, dom.dmid, Some(key), classid, &devids_csv).map_err(db_err)?
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

fn cmd_create_close(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let fid: i64 = req(args, "fid").ok_or_else(MogError::no_fid)?.parse().map_err(|_| MogError::bad_params_msg("invalid fid"))?;
    let devid: i64 = req(args, "devid").ok_or_else(MogError::no_devid)?.parse().map_err(|_| MogError::bad_params_msg("invalid devid"))?;
    let path_arg = req(args, "path").ok_or_else(MogError::no_path)?;

    let tf = queries::get_tempfile(&conn, fid).map_err(db_err)?.ok_or_else(MogError::no_temp_file)?;
    queries::delete_tempfile(&conn, fid).map_err(db_err)?;

    if !tf.devid_list().contains(&devid) {
        return Err(MogError::invalid_destdev());
    }

    let expected_uri = storepath::uri_path(devid, fid);
    if !path_arg.ends_with(&expected_uri) {
        return Err(MogError::bogus_args());
    }

    let fs_path = storepath::fs_path(&state.cfg.docroot, devid, fid);
    let key_arg = req(args, "key");

    let Some(key) = key_arg else {
        // Client is abandoning this upload: discard the blob it PUT.
        let _ = std::fs::remove_file(&fs_path);
        return Ok(Reply::new());
    };

    let meta = std::fs::metadata(&fs_path).map_err(|_| MogError::size_verify_error())?;
    let actual_size = meta.len() as i64;
    if let Some(size_str) = req(args, "size") {
        let expected: i64 = size_str.parse().map_err(|_| MogError::bad_params_msg("invalid size"))?;
        if expected != actual_size {
            return Err(MogError::size_mismatch());
        }
    }

    if let Some(checksum_arg) = req(args, "checksum") {
        let (alg, hexval) = checksum_arg.split_once(':').ok_or_else(MogError::invalid_checksum_format)?;
        let verify = args.get("checksumverify").map(|v| v == "1").unwrap_or(true);
        if verify {
            let data = std::fs::read(&fs_path).map_err(|_| MogError::size_verify_error())?;
            let computed = match alg.to_ascii_uppercase().as_str() {
                "MD5" => {
                    use md5::{Digest, Md5};
                    let mut h = Md5::new();
                    h.update(&data);
                    hex::encode(h.finalize())
                }
                "SHA1" => {
                    use sha1::{Digest, Sha1};
                    let mut h = Sha1::new();
                    h.update(&data);
                    hex::encode(h.finalize())
                }
                _ => return Err(MogError::invalid_checksum_format()),
            };
            if !computed.eq_ignore_ascii_case(hexval) {
                return Err(MogError::checksum_mismatch());
            }
        }
        queries::set_checksum(&conn, fid, alg, hexval).map_err(db_err)?;
    }

    // Overwrite semantics: replacing an existing key retires the old fid.
    if let Some(old) = queries::get_file_by_key(&conn, tf.dmid, key).map_err(db_err)? {
        if old.fid != fid {
            queries::delete_file_row(&conn, old.fid).map_err(db_err)?;
            queries::dequeue_replicate(&conn, old.fid).map_err(db_err)?;
            queries::queue_delete(&conn, old.fid).map_err(db_err)?;
        }
    }

    queries::insert_file(&conn, fid, tf.dmid, key, actual_size, tf.classid).map_err(db_err)?;
    queries::add_file_on(&conn, fid, devid).map_err(db_err)?;

    let mindevcount = replication::resolve_class_mindevcount(&conn, tf.dmid, tf.classid, state.cfg.default_min_devcount as i64)?;
    if mindevcount > 1 {
        queries::queue_replicate(&conn, fid, Some(devid)).map_err(db_err)?;
    }

    Ok(Reply::new())
}

// ---------------- get_paths ----------------

fn cmd_get_paths(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let dom = resolve_domain(&conn, args)?;
    let key = req(args, "key").ok_or_else(MogError::no_key)?;
    let file = queries::get_file_by_key(&conn, dom.dmid, key).map_err(db_err)?.ok_or_else(MogError::unknown_key)?;

    let pathcount: usize = args.get("pathcount").and_then(|s| s.parse().ok()).unwrap_or(2).max(2);
    let noverify = args.get("noverify").map(|v| v == "1").unwrap_or(false);

    let devids = queries::get_devids_for_fid(&conn, file.fid).map_err(db_err)?;
    let mut pairs: Vec<DevHost> = Vec::new();
    for devid in &devids {
        if let Ok(Some(dev)) = queries::get_device(&conn, *devid) {
            if let Ok(Some(host)) = queries::get_host(&conn, dev.hostid) {
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
            let fs_path = storepath::fs_path(&state.cfg.docroot, d.devid, file.fid);
            if !fs_path.exists() && chosen.len() > 1 {
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

fn cmd_get_domains(state: &Arc<AppState>, _args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let domains = queries::list_domains(&conn).map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("domains".into(), domains.len().to_string());
    for (i, dom) in domains.iter().enumerate() {
        let n = i + 1;
        reply.insert(format!("domain{n}"), dom.namespace.clone());
        let classes = queries::list_classes(&conn, dom.dmid).map_err(db_err)?;
        reply.insert(format!("domain{n}classes"), classes.len().to_string());
        for (j, c) in classes.iter().enumerate() {
            let m = j + 1;
            reply.insert(format!("domain{n}class{m}name"), c.classname.clone());
            reply.insert(format!("domain{n}class{m}mindevcount"), c.mindevcount.to_string());
        }
    }
    Ok(reply)
}

fn cmd_list_keys(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let dom = resolve_domain(&conn, args)?;
    let prefix = args.get("prefix").map(|s| s.as_str()).unwrap_or("");
    let after = req(args, "after");
    let limit: i64 = args.get("limit").and_then(|s| s.parse().ok()).unwrap_or(1000).clamp(1, 1000);

    let rows = queries::list_keys(&conn, dom.dmid, prefix, after, limit).map_err(db_err)?;
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

fn cmd_delete(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let dom = resolve_domain(&conn, args)?;
    let key = req(args, "key").ok_or_else(MogError::no_key)?;
    let file = queries::get_file_by_key(&conn, dom.dmid, key).map_err(db_err)?.ok_or_else(MogError::unknown_key)?;
    queries::delete_file_row(&conn, file.fid).map_err(db_err)?;
    queries::dequeue_replicate(&conn, file.fid).map_err(db_err)?;
    queries::queue_delete(&conn, file.fid).map_err(db_err)?;
    Ok(Reply::new())
}

fn cmd_rename(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let dom = resolve_domain(&conn, args)?;
    let from_key = req(args, "from_key").ok_or_else(MogError::no_key)?;
    let to_key = req(args, "to_key").ok_or_else(MogError::no_key)?;
    let file = queries::get_file_by_key(&conn, dom.dmid, from_key).map_err(db_err)?.ok_or_else(MogError::unknown_key)?;
    if queries::get_file_by_key(&conn, dom.dmid, to_key).map_err(db_err)?.is_some() {
        return Err(MogError::key_exists());
    }
    queries::rename_file_key(&conn, dom.dmid, file.fid, to_key).map_err(db_err)?;
    Ok(Reply::new())
}

fn cmd_list_fids(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let from: i64 = args.get("from").and_then(|s| s.parse().ok()).unwrap_or(1).max(0);
    let count: i64 = args.get("to").and_then(|s| s.parse().ok()).unwrap_or(100).clamp(1, 500);
    let rows = queries::list_fids_range(&conn, from, count).map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("fid_count".into(), rows.len().to_string());
    for (i, f) in rows.iter().enumerate() {
        let n = i + 1;
        let dom_name = queries::get_domain_by_id(&conn, f.dmid).map_err(db_err)?.map(|d| d.namespace).unwrap_or_default();
        let cls_name = replication::class_name(&conn, f.dmid, f.classid)?;
        reply.insert(format!("fid_{n}_fid"), f.fid.to_string());
        reply.insert(format!("fid_{n}_domain"), dom_name);
        reply.insert(format!("fid_{n}_class"), cls_name);
        reply.insert(format!("fid_{n}_key"), f.dkey.clone());
        reply.insert(format!("fid_{n}_length"), f.length.unwrap_or(0).to_string());
        reply.insert(format!("fid_{n}_devcount"), f.devcount.to_string());
    }
    Ok(reply)
}

fn cmd_file_info(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let dom = resolve_domain(&conn, args)?;
    let key = req(args, "key").ok_or_else(MogError::no_key)?;
    let file = queries::get_file_by_key(&conn, dom.dmid, key).map_err(db_err)?.ok_or_else(MogError::unknown_key)?;

    let mut reply = Reply::new();
    reply.insert("fid".into(), file.fid.to_string());
    reply.insert("domain".into(), dom.namespace.clone());
    reply.insert("class".into(), replication::class_name(&conn, dom.dmid, file.classid)?);
    reply.insert("key".into(), file.dkey.clone());
    reply.insert("length".into(), file.length.unwrap_or(0).to_string());
    reply.insert("devcount".into(), file.devcount.to_string());
    if args.get("devices").map(|v| v == "1").unwrap_or(false) {
        let devids = queries::get_devids_for_fid(&conn, file.fid).map_err(db_err)?;
        reply.insert("devids".into(), devids.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","));
    }
    if let Some((alg, hex)) = queries::get_checksum(&conn, file.fid).map_err(db_err)? {
        reply.insert("checksum".into(), format!("{alg}:{hex}"));
    }
    Ok(reply)
}

fn cmd_replicate_now(state: &Arc<AppState>, _args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let pending = queries::next_to_replicate(&conn, 100000).map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("count".into(), pending.len().to_string());
    Ok(reply)
}

// ---------------- device / host admin ----------------

fn cmd_set_weight(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let hostname = req(args, "host").ok_or_else(MogError::bad_params)?;
    let devid: i64 = req(args, "device").ok_or_else(MogError::bad_params)?.parse().map_err(|_| MogError::bad_params())?;
    let weight: i64 = req(args, "weight").ok_or_else(MogError::bad_params)?.parse().map_err(|_| MogError::bad_params())?;
    let host = queries::get_host_by_name(&conn, hostname).map_err(db_err)?.ok_or_else(MogError::unknown_host)?;
    let dev = queries::get_device(&conn, devid).map_err(db_err)?.ok_or_else(MogError::no_device)?;
    if dev.hostid != host.hostid {
        return Err(MogError::host_mismatch());
    }
    queries::update_device_weight(&conn, devid, weight).map_err(db_err)?;
    Ok(Reply::new())
}

fn cmd_set_state(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let hostname = req(args, "host").ok_or_else(MogError::bad_params)?;
    let devid: i64 = req(args, "device").ok_or_else(MogError::bad_params)?.parse().map_err(|_| MogError::bad_params())?;
    let new_state = req(args, "state").ok_or_else(MogError::bad_params)?;
    let host = queries::get_host_by_name(&conn, hostname).map_err(db_err)?.ok_or_else(MogError::unknown_host)?;
    let dev = queries::get_device(&conn, devid).map_err(db_err)?.ok_or_else(MogError::no_device)?;
    if dev.hostid != host.hostid {
        return Err(MogError::host_mismatch());
    }
    let parsed = DeviceStatus::parse(new_state).ok_or_else(MogError::unknown_state)?;
    let current = DeviceStatus::parse(&dev.status);
    if current == Some(DeviceStatus::Dead) && parsed == DeviceStatus::Alive {
        return Err(MogError::state_too_high());
    }
    queries::update_device_status(&conn, devid, parsed.as_str()).map_err(db_err)?;
    Ok(Reply::new())
}

fn cmd_get_hosts(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let filter: Option<i64> = args.get("hostid").and_then(|s| s.parse().ok());
    let hosts: Vec<Host> = queries::list_hosts(&conn).map_err(db_err)?.into_iter().filter(|h| filter.map(|f| f == h.hostid).unwrap_or(true)).collect();
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

fn cmd_get_devices(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let filter: Option<i64> = args.get("devid").and_then(|s| s.parse().ok());
    let devs: Vec<Device> = queries::list_devices(&conn).map_err(db_err)?.into_iter().filter(|d| filter.map(|f| f == d.devid).unwrap_or(true)).collect();
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

fn cmd_create_domain(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let name = req(args, "domain").ok_or_else(MogError::no_domain)?;
    if queries::get_domain_by_name(&conn, name).map_err(db_err)?.is_some() {
        return Err(MogError::domain_exists());
    }
    queries::create_domain(&conn, name).map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("domain".into(), name.to_string());
    Ok(reply)
}

fn cmd_delete_domain(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let dom = resolve_domain(&conn, args)?;
    let has_files: i64 = conn
        .query_row("SELECT COUNT(*) FROM file WHERE dmid = ?1", [dom.dmid], |r| r.get(0))
        .map_err(|e| MogError::db_msg(e.to_string()))?;
    if has_files > 0 {
        return Err(MogError::domain_has_files());
    }
    queries::delete_domain(&conn, dom.dmid).map_err(db_err)?;
    Ok(Reply::new())
}

fn cmd_create_class(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let dom = resolve_domain(&conn, args)?;
    let name = req(args, "class").ok_or_else(MogError::no_class)?;
    let mindevcount: i64 = args
        .get("mindevcount")
        .and_then(|s| s.parse().ok())
        .unwrap_or(state.cfg.default_min_devcount as i64);
    if mindevcount < 1 {
        return Err(MogError::invalid_mindevcount());
    }
    if queries::get_class_by_name(&conn, dom.dmid, name).map_err(db_err)?.is_some() {
        return Err(MogError::class_exists());
    }
    let classid = queries::create_class(&conn, dom.dmid, name, mindevcount).map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("class".into(), name.to_string());
    reply.insert("classid".into(), classid.to_string());
    reply.insert("mindevcount".into(), mindevcount.to_string());
    Ok(reply)
}

fn cmd_update_class(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let dom = resolve_domain(&conn, args)?;
    let name = req(args, "class").ok_or_else(MogError::no_class)?;
    let class = queries::get_class_by_name(&conn, dom.dmid, name).map_err(db_err)?.ok_or_else(MogError::class_not_found)?;
    let mindevcount: i64 = req(args, "mindevcount").ok_or_else(MogError::bad_params)?.parse().map_err(|_| MogError::bad_params())?;
    if mindevcount < 1 {
        return Err(MogError::invalid_mindevcount());
    }
    queries::update_class_mindevcount(&conn, dom.dmid, class.classid, mindevcount).map_err(db_err)?;
    Ok(Reply::new())
}

fn cmd_delete_class(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let dom = resolve_domain(&conn, args)?;
    let name = req(args, "class").ok_or_else(MogError::no_class)?;
    let class = queries::get_class_by_name(&conn, dom.dmid, name).map_err(db_err)?.ok_or_else(MogError::class_not_found)?;
    let has_files: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM file WHERE dmid = ?1 AND classid = ?2",
            [dom.dmid, class.classid],
            |r| r.get(0),
        )
        .map_err(|e| MogError::db_msg(e.to_string()))?;
    if has_files > 0 {
        return Err(MogError::class_has_files());
    }
    queries::delete_class(&conn, dom.dmid, class.classid).map_err(db_err)?;
    Ok(Reply::new())
}

fn cmd_create_host(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let name = req(args, "host").ok_or_else(MogError::no_host)?;
    if queries::get_host_by_name(&conn, name).map_err(db_err)?.is_some() {
        return Err(MogError::host_exists());
    }
    let ip = req(args, "ip").ok_or_else(MogError::no_ip)?;
    let port: i64 = req(args, "port").ok_or_else(MogError::no_port)?.parse().map_err(|_| MogError::bad_params())?;
    let hostid = queries::create_host(&conn, name, Some(ip), port).map_err(db_err)?;
    if let Some(status) = req(args, "status") {
        if HostStatus::parse(status).is_some() {
            queries::update_host_status(&conn, hostid, status).map_err(db_err)?;
        }
    }
    let mut reply = Reply::new();
    reply.insert("hostid".into(), hostid.to_string());
    Ok(reply)
}

fn cmd_update_host(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let name = req(args, "host").ok_or_else(MogError::no_host)?;
    let host = queries::get_host_by_name(&conn, name).map_err(db_err)?.ok_or_else(MogError::unknown_host)?;
    if let Some(ip) = req(args, "ip") {
        conn.execute("UPDATE host SET hostip = ?2 WHERE hostid = ?1", rusqlite::params![host.hostid, ip])
            .map_err(|e| MogError::db_msg(e.to_string()))?;
    }
    if let Some(port) = req(args, "port") {
        let port: i64 = port.parse().map_err(|_| MogError::bad_params())?;
        conn.execute("UPDATE host SET http_port = ?2 WHERE hostid = ?1", rusqlite::params![host.hostid, port])
            .map_err(|e| MogError::db_msg(e.to_string()))?;
    }
    if let Some(status) = req(args, "status") {
        HostStatus::parse(status).ok_or_else(MogError::unknown_state)?;
        queries::update_host_status(&conn, host.hostid, status).map_err(db_err)?;
    }
    Ok(Reply::new())
}

fn cmd_delete_host(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let name = req(args, "host").ok_or_else(MogError::no_host)?;
    let host = queries::get_host_by_name(&conn, name).map_err(db_err)?.ok_or_else(MogError::unknown_host)?;
    if queries::devices_for_host(&conn, host.hostid).map_err(db_err)? > 0 {
        return Err(MogError::host_not_empty());
    }
    queries::delete_host(&conn, host.hostid).map_err(db_err)?;
    Ok(Reply::new())
}

fn cmd_create_device(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let hostname = req(args, "host").ok_or_else(MogError::no_host)?;
    let host = queries::get_host_by_name(&conn, hostname).map_err(db_err)?.ok_or_else(MogError::unknown_host)?;
    let devid: i64 = req(args, "devid").ok_or_else(MogError::no_devid)?.parse().map_err(|_| MogError::bad_params())?;
    if queries::get_device(&conn, devid).map_err(db_err)?.is_some() {
        return Err(MogError::device_exists());
    }
    queries::create_device(&conn, devid, host.hostid).map_err(db_err)?;
    if let Some(status) = req(args, "status") {
        if DeviceStatus::parse(status).is_some() {
            queries::update_device_status(&conn, devid, status).map_err(db_err)?;
        }
    }
    std::fs::create_dir_all(storepath::device_root(&state.cfg.docroot, devid)).ok();
    Ok(Reply::new())
}

// ---------------- server settings ----------------

fn cmd_server_setting(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    if let Some(key) = req(args, "key") {
        let value = queries::get_setting(&conn, key).map_err(db_err)?.unwrap_or_default();
        let mut reply = Reply::new();
        reply.insert("key".into(), key.to_string());
        reply.insert("value".into(), value);
        return Ok(reply);
    }
    let all = queries::list_settings(&conn).map_err(db_err)?;
    let mut reply = Reply::new();
    reply.insert("key_count".into(), all.len().to_string());
    for (i, (k, v)) in all.iter().enumerate() {
        reply.insert(format!("key_{}", i + 1), k.clone());
        reply.insert(format!("value_{}", i + 1), v.clone());
    }
    Ok(reply)
}

fn cmd_set_server_setting(state: &Arc<AppState>, args: &Args) -> Result<Reply, MogError> {
    let conn = state.db.conn().map_err(db_err)?;
    let key = req(args, "key").ok_or_else(MogError::bad_params)?;
    let value = args.get("value").map(|s| s.as_str()).unwrap_or("");
    queries::set_setting(&conn, key, value).map_err(db_err)?;
    Ok(Reply::new())
}
