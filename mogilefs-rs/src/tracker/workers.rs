use super::replication;
use super::AppState;
use crate::db::queries;
use crate::storage::client;
use std::sync::Arc;
use std::time::Duration;

/// Periodically polls `GET /dev<N>/usage` on each device's **owning host** (over
/// HTTP, exactly as the reference tracker polls `mogstored`) and records
/// `mb_total`/`mb_used`, used by device-selection freespace weighting. This
/// works whether the device is on this node or a remote storage node.
pub async fn run_monitor(state: Arc<AppState>) {
    let mut interval = tokio::time::interval(Duration::from_secs(15));
    loop {
        interval.tick().await;
        let db = &state.db;
        let Ok(devices) = queries::list_devices(db).await else { continue };
        for d in devices {
            if let Ok((total_mb, used_mb)) = client::usage(db, d.devid).await {
                let _ = queries::update_device_usage(db, d.devid, total_mb, used_mb).await;
            }
        }
    }
}

/// Moves fids off the `file_to_replicate` queue by copying their bytes to
/// additional devices until the owning class's `mindevcount` is satisfied.
pub async fn run_replicate(state: Arc<AppState>) {
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    loop {
        interval.tick().await;
        let db = &state.db;
        let Ok(pending) = queries::next_to_replicate(db, 20).await else { continue };
        for fid in pending {
            let Ok(Some(file)) = queries::get_file_by_fid(db, fid).await else {
                let _ = queries::dequeue_replicate(db, fid).await;
                continue;
            };
            let Ok(min) = replication::resolve_class_mindevcount(
                db,
                file.dmid,
                file.classid,
                state.cfg.default_min_devcount as i64,
            )
            .await
            else {
                continue;
            };
            let have = queries::get_devids_for_fid(db, fid).await.unwrap_or_default();
            if have.len() as i64 >= min {
                let _ = queries::dequeue_replicate(db, fid).await;
                continue;
            }
            let need = (min as usize).saturating_sub(have.len());
            let Some(&src_devid) = have.first() else {
                let _ = queries::bump_replicate_failure(db, fid, 30).await;
                continue;
            };
            match replication::select_devices(db, need, &have).await {
                Ok(targets) => {
                    let mut all_ok = true;
                    for t in targets {
                        // Copy from the source host to the target host over HTTP.
                        if client::copy(db, src_devid, t.devid, fid).await.is_ok() {
                            let _ = queries::add_file_on(db, fid, t.devid).await;
                        } else {
                            all_ok = false;
                        }
                    }
                    let have_now = queries::get_devids_for_fid(db, fid).await.unwrap_or_default();
                    if have_now.len() as i64 >= min {
                        let _ = queries::dequeue_replicate(db, fid).await;
                    } else if !all_ok {
                        let _ = queries::bump_replicate_failure(db, fid, 30).await;
                    }
                }
                Err(_) => {
                    let _ = queries::bump_replicate_failure(db, fid, 30).await;
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
        let db = &state.db;
        let Ok(pending) = queries::next_to_delete(db, 50).await else { continue };
        for fid in pending {
            let devids = queries::get_devids_for_fid(db, fid).await.unwrap_or_default();
            for devid in &devids {
                // Delete the blob on its owning host over HTTP.
                let _ = client::delete(db, *devid, fid).await;
                let _ = queries::remove_file_on(db, fid, *devid).await;
            }
            let _ = queries::dequeue_delete(db, fid).await;
        }
    }
}

/// When `fsck_running=1`, walks all `file` rows in fid order, verifying each
/// still has its expected replicas on disk (and matches its stored checksum,
/// if any), logging anomalies to `fsck_log` and re-queuing repairs.
pub async fn run_fsck(state: Arc<AppState>) {
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    const BATCH: i64 = 50;
    loop {
        interval.tick().await;
        let db = &state.db;
        let running = queries::get_setting(db, "fsck_running").await.ok().flatten().unwrap_or_default();
        if running != "1" {
            continue;
        }
        let cursor: i64 = queries::get_setting(db, "fsck_cursor")
            .await
            .ok()
            .flatten()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let rows = match queries::list_fids_range(db, cursor, BATCH).await {
            Ok(r) => r,
            Err(_) => continue,
        };

        for f in &rows {
            let min = replication::resolve_class_mindevcount(db, f.dmid, f.classid, state.cfg.default_min_devcount as i64)
                .await
                .unwrap_or(1);
            let devids = queries::get_devids_for_fid(db, f.fid).await.unwrap_or_default();

            let mut live = Vec::new();
            for devid in &devids {
                // Check existence on the device's owning host over HTTP.
                match client::exists(db, *devid, f.fid).await {
                    Ok(true) => live.push(*devid),
                    Ok(false) => {
                        let _ = queries::fsck_log(db, f.fid, "missing", Some(*devid)).await;
                        let _ = queries::remove_file_on(db, f.fid, *devid).await;
                    }
                    // On a transient host error, don't treat the replica as lost.
                    Err(_) => live.push(*devid),
                }
            }

            if let Ok(Some((alg, hexval))) = queries::get_checksum(db, f.fid).await {
                if let Some(&devid) = live.first() {
                    if let Ok(Some(data)) = client::get_bytes(db, devid, f.fid).await {
                        if let Some(computed) = crate::util::compute_checksum_hex(&alg, &data) {
                            if !computed.eq_ignore_ascii_case(&hexval) {
                                let _ = queries::fsck_log(db, f.fid, "checksum_mismatch", Some(devid)).await;
                            }
                        }
                    }
                }
            }

            if (live.len() as i64) < min {
                let _ = queries::fsck_log(db, f.fid, "under_replicated", None).await;
                let _ = queries::queue_replicate(db, f.fid, live.first().copied()).await;
            }
        }

        if rows.len() < BATCH as usize {
            // Reached the end of the fid space: pass complete.
            let _ = queries::set_setting(db, "fsck_cursor", "0").await;
            let _ = queries::set_setting(db, "fsck_running", "0").await;
        } else {
            let next_cursor = rows.last().map(|f| f.fid + 1).unwrap_or(cursor);
            let _ = queries::set_setting(db, "fsck_cursor", &next_cursor.to_string()).await;
        }
    }
}

/// When `rebal_running=1`, drains the `file_to_queue` "rebalance" queue
/// (seeded by `rebalance_start`), moving each fid's copy off its source
/// device onto a less-utilized one without changing its replica count.
pub async fn run_rebalance(state: Arc<AppState>) {
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    loop {
        interval.tick().await;
        let db = &state.db;
        let running = queries::get_setting(db, "rebal_running").await.ok().flatten().unwrap_or_default();
        if running != "1" {
            continue;
        }
        let Ok(batch) = queries::queue_next(db, "rebalance", 10).await else { continue };
        if batch.is_empty() {
            let _ = queries::set_setting(db, "rebal_running", "0").await;
            continue;
        }
        for entry in batch {
            let Some(src_devid) = entry.devid else {
                let _ = queries::queue_remove(db, entry.fid, "rebalance").await;
                continue;
            };
            let existing = queries::get_devids_for_fid(db, entry.fid).await.unwrap_or_default();
            match replication::select_devices(db, 1, &existing).await {
                Ok(targets) if !targets.is_empty() => {
                    let dst_devid = targets[0].devid;
                    // Only drop the source copy once the destination copy is
                    // both on its host AND recorded. If either fails, leave the
                    // source intact — otherwise a fid whose only replica was on
                    // src would end up tracked on neither device (lost file)
                    // even though the bytes exist on dst.
                    if client::copy(db, src_devid, dst_devid, entry.fid).await.is_ok()
                        && queries::add_file_on(db, entry.fid, dst_devid).await.is_ok()
                    {
                        let _ = client::delete(db, src_devid, entry.fid).await;
                        let _ = queries::remove_file_on(db, entry.fid, src_devid).await;
                        let _ = queries::queue_remove(db, entry.fid, "rebalance").await;
                    } else {
                        let _ = queries::queue_bump_failure(db, entry.fid, "rebalance", 30).await;
                    }
                }
                _ => {
                    let _ = queries::queue_bump_failure(db, entry.fid, "rebalance", 30).await;
                }
            }
        }
    }
}
