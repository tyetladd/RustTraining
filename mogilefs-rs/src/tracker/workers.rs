use super::replication;
use super::AppState;
use crate::db::queries;
use crate::storage::path as storepath;
use std::sync::Arc;
use std::time::Duration;

/// Periodically stats each device's backing directory (equivalent to the
/// real tracker polling `GET /devN/usage` on the storage node) and records
/// `mb_total`/`mb_used`, used by device-selection freespace weighting.
pub async fn run_monitor(state: Arc<AppState>) {
    let mut interval = tokio::time::interval(Duration::from_secs(15));
    loop {
        interval.tick().await;
        let Ok(conn) = state.db.conn() else { continue };
        let Ok(devices) = queries::list_devices(&conn) else { continue };
        for d in devices {
            let dir = storepath::device_root(&state.cfg.docroot, d.devid);
            if std::fs::create_dir_all(&dir).is_err() {
                continue;
            }
            let total_kb = fs4::total_space(&dir).unwrap_or(0) / 1024;
            let avail_kb = fs4::available_space(&dir).unwrap_or(0) / 1024;
            let used_kb = total_kb.saturating_sub(avail_kb);
            let _ = queries::update_device_usage(&conn, d.devid, total_kb as i64, used_kb as i64);
        }
    }
}

/// Moves fids off the `file_to_replicate` queue by copying their bytes to
/// additional devices until the owning class's `mindevcount` is satisfied.
pub async fn run_replicate(state: Arc<AppState>) {
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    loop {
        interval.tick().await;
        let Ok(conn) = state.db.conn() else { continue };
        let Ok(pending) = queries::next_to_replicate(&conn, 20) else { continue };
        for fid in pending {
            let Ok(Some(file)) = queries::get_file_by_fid(&conn, fid) else {
                let _ = queries::dequeue_replicate(&conn, fid);
                continue;
            };
            let Ok(min) = replication::resolve_class_mindevcount(
                &conn,
                file.dmid,
                file.classid,
                state.cfg.default_min_devcount as i64,
            ) else {
                continue;
            };
            let have = queries::get_devids_for_fid(&conn, fid).unwrap_or_default();
            if have.len() as i64 >= min {
                let _ = queries::dequeue_replicate(&conn, fid);
                continue;
            }
            let need = (min as usize).saturating_sub(have.len());
            let Some(&src_devid) = have.first() else {
                let _ = queries::bump_replicate_failure(&conn, fid, 30);
                continue;
            };
            let src_path = storepath::fs_path(&state.cfg.docroot, src_devid, fid);
            match replication::select_devices(&conn, need, &have) {
                Ok(targets) => {
                    let mut all_ok = true;
                    for t in targets {
                        let dst_path = storepath::fs_path(&state.cfg.docroot, t.devid, fid);
                        if let Some(parent) = dst_path.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        if std::fs::copy(&src_path, &dst_path).is_ok() {
                            let _ = queries::add_file_on(&conn, fid, t.devid);
                        } else {
                            all_ok = false;
                        }
                    }
                    let have_now = queries::get_devids_for_fid(&conn, fid).unwrap_or_default();
                    if have_now.len() as i64 >= min {
                        let _ = queries::dequeue_replicate(&conn, fid);
                    } else if !all_ok {
                        let _ = queries::bump_replicate_failure(&conn, fid, 30);
                    }
                }
                Err(_) => {
                    let _ = queries::bump_replicate_failure(&conn, fid, 30);
                }
            }
        }
    }
}

/// Physically removes blobs for fids queued by `delete`/overwrite, mirroring
/// the reference tracker's asynchronous Delete worker.
pub async fn run_delete(state: Arc<AppState>) {
    let mut interval = tokio::time::interval(Duration::from_secs(3));
    loop {
        interval.tick().await;
        let Ok(conn) = state.db.conn() else { continue };
        let Ok(pending) = queries::next_to_delete(&conn, 50) else { continue };
        for fid in pending {
            let devids = queries::get_devids_for_fid(&conn, fid).unwrap_or_default();
            for devid in &devids {
                let path = storepath::fs_path(&state.cfg.docroot, *devid, fid);
                let _ = std::fs::remove_file(&path);
                let _ = queries::remove_file_on(&conn, fid, *devid);
            }
            let _ = queries::dequeue_delete(&conn, fid);
        }
    }
}
