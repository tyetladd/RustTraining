//! Storage operations shared between the MogileFS wire protocol
//! (`create_close`) and the S3 gateway (`PutObject`). Both, after a blob's bytes
//! are on a device, need to finalize it into the `file`/`file_on` tables,
//! retire any prior object at the same key, record its checksum, and enqueue
//! replication — so that logic lives here once.

use super::replication;
use super::AppState;
use crate::db::queries;
use crate::error::MogError;

fn db_err(e: anyhow::Error) -> MogError {
    tracing::error!("database error: {e:#}");
    MogError::db()
}

/// Finalizes an already-written blob (fid on `devid`) as the current object for
/// `(dmid, key)` in one transaction, retiring any previous fid at that key,
/// storing its checksum, and queuing replication to reach the class's
/// `mindevcount`.
#[allow(clippy::too_many_arguments)]
pub async fn finalize_blob(
    state: &AppState,
    fid: i64,
    dmid: i64,
    key: &str,
    classid: i64,
    devid: i64,
    size: i64,
    checksum: Option<(String, String)>,
) -> Result<(), MogError> {
    let db = &state.db;

    // Overwrite semantics: replacing an existing key retires the old fid.
    let retire_old = queries::get_file_by_key(db, dmid, key)
        .await
        .map_err(db_err)?
        .map(|old| old.fid)
        .filter(|&old_fid| old_fid != fid);

    queries::finalize_file(db, fid, dmid, key, size, classid, devid, retire_old)
        .await
        .map_err(db_err)?;

    if let Some(old_fid) = retire_old {
        queries::dequeue_replicate(db, old_fid).await.map_err(db_err)?;
        queries::queue_delete(db, old_fid).await.map_err(db_err)?;
    }

    if let Some((alg, hexval)) = checksum {
        queries::set_checksum(db, fid, &alg, &hexval).await.map_err(db_err)?;
    }

    let mindevcount =
        replication::resolve_class_mindevcount(db, dmid, classid, state.cfg.default_min_devcount as i64).await?;
    if mindevcount > 1 {
        queries::queue_replicate(db, fid, Some(devid)).await.map_err(db_err)?;
    }

    Ok(())
}
