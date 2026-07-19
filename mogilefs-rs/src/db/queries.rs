use super::model::*;
use super::{pg, Db, Store};
use anyhow::Result;

/// Runs a `?`-style SELECT (fetching zero-or-one row) against whichever
/// backend `db` is pointed at, converting placeholders for Postgres.
macro_rules! q_fetch_optional {
    ($db:expr, $sql:expr, $ty:ty $(, $arg:expr)* $(,)?) => {{
        let result: ::std::result::Result<Option<$ty>, sqlx::Error> = match &$db.store {
            Store::Sqlite(p) => sqlx::query_as::<_, $ty>($sql) $(.bind($arg))* .fetch_optional(p).await,
            Store::MySql(p) => sqlx::query_as::<_, $ty>($sql) $(.bind($arg))* .fetch_optional(p).await,
            Store::Postgres(p) => sqlx::query_as::<_, $ty>(&pg($sql)) $(.bind($arg))* .fetch_optional(p).await,
        };
        result.map_err(::anyhow::Error::from)
    }};
}

macro_rules! q_fetch_all {
    ($db:expr, $sql:expr, $ty:ty $(, $arg:expr)* $(,)?) => {{
        let result: ::std::result::Result<Vec<$ty>, sqlx::Error> = match &$db.store {
            Store::Sqlite(p) => sqlx::query_as::<_, $ty>($sql) $(.bind($arg))* .fetch_all(p).await,
            Store::MySql(p) => sqlx::query_as::<_, $ty>($sql) $(.bind($arg))* .fetch_all(p).await,
            Store::Postgres(p) => sqlx::query_as::<_, $ty>(&pg($sql)) $(.bind($arg))* .fetch_all(p).await,
        };
        result.map_err(::anyhow::Error::from)
    }};
}

macro_rules! q_exec {
    ($db:expr, $sql:expr $(, $arg:expr)* $(,)?) => {{
        let result: ::std::result::Result<(), sqlx::Error> = async {
            match &$db.store {
                Store::Sqlite(p) => { sqlx::query($sql) $(.bind($arg))* .execute(p).await?; },
                Store::MySql(p) => { sqlx::query($sql) $(.bind($arg))* .execute(p).await?; },
                Store::Postgres(p) => { sqlx::query(&pg($sql)) $(.bind($arg))* .execute(p).await?; },
            }
            Ok(())
        }.await;
        result.map_err(::anyhow::Error::from)
    }};
}

macro_rules! q_scalar {
    ($db:expr, $sql:expr, $ty:ty $(, $arg:expr)* $(,)?) => {{
        let result: ::std::result::Result<$ty, sqlx::Error> = match &$db.store {
            Store::Sqlite(p) => sqlx::query_scalar::<_, $ty>($sql) $(.bind($arg))* .fetch_one(p).await,
            Store::MySql(p) => sqlx::query_scalar::<_, $ty>($sql) $(.bind($arg))* .fetch_one(p).await,
            Store::Postgres(p) => sqlx::query_scalar::<_, $ty>(&pg($sql)) $(.bind($arg))* .fetch_one(p).await,
        };
        result.map_err(::anyhow::Error::from)
    }};
}

// ---- domain ----

pub async fn create_domain(db: &Db, namespace: &str) -> Result<i64> {
    const SQL: &str = "INSERT INTO domain (namespace) VALUES (?)";
    Ok(match &db.store {
        Store::Sqlite(p) => sqlx::query(SQL).bind(namespace).execute(p).await?.last_insert_rowid(),
        Store::MySql(p) => sqlx::query(SQL).bind(namespace).execute(p).await?.last_insert_id() as i64,
        Store::Postgres(p) => {
            sqlx::query_scalar::<_, i64>("INSERT INTO domain (namespace) VALUES ($1) RETURNING dmid")
                .bind(namespace)
                .fetch_one(p)
                .await?
        }
    })
}

pub async fn get_domain_by_name(db: &Db, namespace: &str) -> Result<Option<Domain>> {
    q_fetch_optional!(db, "SELECT dmid, namespace FROM domain WHERE namespace = ?", Domain, namespace)
}

pub async fn get_domain_by_id(db: &Db, dmid: i64) -> Result<Option<Domain>> {
    q_fetch_optional!(db, "SELECT dmid, namespace FROM domain WHERE dmid = ?", Domain, dmid)
}

pub async fn delete_domain(db: &Db, dmid: i64) -> Result<()> {
    q_exec!(db, "DELETE FROM domain WHERE dmid = ?", dmid)
}

pub async fn list_domains(db: &Db) -> Result<Vec<Domain>> {
    q_fetch_all!(db, "SELECT dmid, namespace FROM domain ORDER BY dmid", Domain)
}

// ---- class ----

pub async fn create_class(db: &Db, dmid: i64, classname: &str, mindevcount: i64) -> Result<i64> {
    let next: i64 = q_scalar!(db, "SELECT COALESCE(MAX(classid), 0) + 1 FROM class WHERE dmid = ?", i64, dmid)
        .unwrap_or(1);
    q_exec!(
        db,
        "INSERT INTO class (dmid, classid, classname, mindevcount) VALUES (?, ?, ?, ?)",
        dmid,
        next,
        classname,
        mindevcount
    )?;
    Ok(next)
}

pub async fn get_class_by_name(db: &Db, dmid: i64, classname: &str) -> Result<Option<Class>> {
    q_fetch_optional!(
        db,
        "SELECT dmid, classid, classname, mindevcount FROM class WHERE dmid = ? AND classname = ?",
        Class,
        dmid,
        classname
    )
}

pub async fn get_class(db: &Db, dmid: i64, classid: i64) -> Result<Option<Class>> {
    q_fetch_optional!(
        db,
        "SELECT dmid, classid, classname, mindevcount FROM class WHERE dmid = ? AND classid = ?",
        Class,
        dmid,
        classid
    )
}

pub async fn list_classes(db: &Db, dmid: i64) -> Result<Vec<Class>> {
    q_fetch_all!(
        db,
        "SELECT dmid, classid, classname, mindevcount FROM class WHERE dmid = ? ORDER BY classid",
        Class,
        dmid
    )
}

pub async fn update_class_mindevcount(db: &Db, dmid: i64, classid: i64, mindevcount: i64) -> Result<()> {
    q_exec!(db, "UPDATE class SET mindevcount = ? WHERE dmid = ? AND classid = ?", mindevcount, dmid, classid)
}

pub async fn delete_class(db: &Db, dmid: i64, classid: i64) -> Result<()> {
    q_exec!(db, "DELETE FROM class WHERE dmid = ? AND classid = ?", dmid, classid)
}

// ---- host ----

pub async fn create_host(db: &Db, hostname: &str, hostip: Option<&str>, http_port: i64) -> Result<i64> {
    const SQL: &str = "INSERT INTO host (hostname, hostip, status, http_port) VALUES (?, ?, 'down', ?)";
    Ok(match &db.store {
        Store::Sqlite(p) => sqlx::query(SQL).bind(hostname).bind(hostip).bind(http_port).execute(p).await?.last_insert_rowid(),
        Store::MySql(p) => sqlx::query(SQL).bind(hostname).bind(hostip).bind(http_port).execute(p).await?.last_insert_id() as i64,
        Store::Postgres(p) => {
            sqlx::query_scalar::<_, i64>(
                "INSERT INTO host (hostname, hostip, status, http_port) VALUES ($1, $2, 'down', $3) RETURNING hostid",
            )
            .bind(hostname)
            .bind(hostip)
            .bind(http_port)
            .fetch_one(p)
            .await?
        }
    })
}

pub async fn get_host_by_name(db: &Db, hostname: &str) -> Result<Option<Host>> {
    q_fetch_optional!(
        db,
        "SELECT hostid, hostname, hostip, status, http_port FROM host WHERE hostname = ?",
        Host,
        hostname
    )
}

pub async fn get_host(db: &Db, hostid: i64) -> Result<Option<Host>> {
    q_fetch_optional!(
        db,
        "SELECT hostid, hostname, hostip, status, http_port FROM host WHERE hostid = ?",
        Host,
        hostid
    )
}

pub async fn list_hosts(db: &Db) -> Result<Vec<Host>> {
    q_fetch_all!(db, "SELECT hostid, hostname, hostip, status, http_port FROM host ORDER BY hostid", Host)
}

pub async fn update_host_status(db: &Db, hostid: i64, status: &str) -> Result<()> {
    q_exec!(db, "UPDATE host SET status = ? WHERE hostid = ?", status, hostid)
}

pub async fn update_host_ip(db: &Db, hostid: i64, ip: &str) -> Result<()> {
    q_exec!(db, "UPDATE host SET hostip = ? WHERE hostid = ?", ip, hostid)
}

pub async fn update_host_port(db: &Db, hostid: i64, port: i64) -> Result<()> {
    q_exec!(db, "UPDATE host SET http_port = ? WHERE hostid = ?", port, hostid)
}

pub async fn delete_host(db: &Db, hostid: i64) -> Result<()> {
    q_exec!(db, "DELETE FROM host WHERE hostid = ?", hostid)
}

pub async fn devices_for_host(db: &Db, hostid: i64) -> Result<i64> {
    q_scalar!(db, "SELECT COUNT(*) FROM device WHERE hostid = ?", i64, hostid)
}

// ---- device ----

pub async fn create_device(db: &Db, devid: i64, hostid: i64) -> Result<()> {
    q_exec!(db, "INSERT INTO device (devid, hostid, status, weight) VALUES (?, ?, 'down', 100)", devid, hostid)
}

pub async fn get_device(db: &Db, devid: i64) -> Result<Option<Device>> {
    q_fetch_optional!(
        db,
        "SELECT devid, hostid, status, weight, mb_total, mb_used FROM device WHERE devid = ?",
        Device,
        devid
    )
}

pub async fn list_devices(db: &Db) -> Result<Vec<Device>> {
    q_fetch_all!(db, "SELECT devid, hostid, status, weight, mb_total, mb_used FROM device ORDER BY devid", Device)
}

pub async fn update_device_status(db: &Db, devid: i64, status: &str) -> Result<()> {
    q_exec!(db, "UPDATE device SET status = ? WHERE devid = ?", status, devid)
}

pub async fn update_device_weight(db: &Db, devid: i64, weight: i64) -> Result<()> {
    q_exec!(db, "UPDATE device SET weight = ? WHERE devid = ?", weight, devid)
}

pub async fn update_device_usage(db: &Db, devid: i64, mb_total: i64, mb_used: i64) -> Result<()> {
    let now = now_ts();
    q_exec!(db, "UPDATE device SET mb_total = ?, mb_used = ?, mb_asof = ? WHERE devid = ?", mb_total, mb_used, now, devid)
}

// ---- tempfile / create_open ----

pub async fn create_tempfile(db: &Db, dmid: i64, dkey: Option<&str>, classid: i64, devids: &str) -> Result<i64> {
    let now = now_ts();
    const SQL: &str = "INSERT INTO tempfile (createtime, classid, dmid, dkey, devids) VALUES (?, ?, ?, ?, ?)";
    Ok(match &db.store {
        Store::Sqlite(p) => sqlx::query(SQL).bind(now).bind(classid).bind(dmid).bind(dkey).bind(devids).execute(p).await?.last_insert_rowid(),
        Store::MySql(p) => sqlx::query(SQL).bind(now).bind(classid).bind(dmid).bind(dkey).bind(devids).execute(p).await?.last_insert_id() as i64,
        Store::Postgres(p) => {
            sqlx::query_scalar::<_, i64>(
                "INSERT INTO tempfile (createtime, classid, dmid, dkey, devids) VALUES ($1, $2, $3, $4, $5) RETURNING fid",
            )
            .bind(now)
            .bind(classid)
            .bind(dmid)
            .bind(dkey)
            .bind(devids)
            .fetch_one(p)
            .await?
        }
    })
}

pub async fn create_tempfile_with_fid(db: &Db, fid: i64, dmid: i64, dkey: Option<&str>, classid: i64, devids: &str) -> Result<()> {
    let now = now_ts();
    q_exec!(
        db,
        "INSERT INTO tempfile (fid, createtime, classid, dmid, dkey, devids) VALUES (?, ?, ?, ?, ?, ?)",
        fid,
        now,
        classid,
        dmid,
        dkey,
        devids
    )?;
    // On Postgres `tempfile.fid` is BIGSERIAL and an explicit-id insert does NOT
    // advance the sequence, so a later auto-fid create_tempfile would eventually
    // collide with this row. Bump the sequence past the highest live fid.
    if let Store::Postgres(p) = &db.store {
        sqlx::query(
            "SELECT setval(pg_get_serial_sequence('tempfile','fid'), \
             GREATEST((SELECT COALESCE(MAX(fid), 0) FROM tempfile), $1))",
        )
        .bind(fid)
        .execute(p)
        .await?;
    }
    Ok(())
}

pub async fn tempfile_or_file_fid_exists(db: &Db, fid: i64) -> Result<bool> {
    let in_temp: i64 = q_scalar!(db, "SELECT COUNT(*) FROM tempfile WHERE fid = ?", i64, fid)?;
    if in_temp > 0 {
        return Ok(true);
    }
    let in_file: i64 = q_scalar!(db, "SELECT COUNT(*) FROM file WHERE fid = ?", i64, fid)?;
    Ok(in_file > 0)
}

pub async fn get_tempfile(db: &Db, fid: i64) -> Result<Option<TempfileRow>> {
    q_fetch_optional!(db, "SELECT fid, dmid, dkey, classid, devids FROM tempfile WHERE fid = ?", TempfileRow, fid)
}

/// Deletes the tempfile row and returns whether a row was actually removed.
/// `create_close` uses the `true`/`false` result as a race guard: if two
/// concurrent closes target the same fid, only the one that deletes the row
/// proceeds to finalize; the loser sees `false` and returns `no_temp_file`.
pub async fn delete_tempfile(db: &Db, fid: i64) -> Result<bool> {
    let affected = match &db.store {
        Store::Sqlite(p) => sqlx::query("DELETE FROM tempfile WHERE fid = ?").bind(fid).execute(p).await?.rows_affected(),
        Store::MySql(p) => sqlx::query("DELETE FROM tempfile WHERE fid = ?").bind(fid).execute(p).await?.rows_affected(),
        Store::Postgres(p) => sqlx::query("DELETE FROM tempfile WHERE fid = $1").bind(fid).execute(p).await?.rows_affected(),
    };
    Ok(affected > 0)
}

// ---- file / create_close ----

pub async fn get_file_by_key(db: &Db, dmid: i64, dkey: &str) -> Result<Option<FileRow>> {
    q_fetch_optional!(
        db,
        "SELECT fid, dmid, dkey, length, classid, devcount FROM file WHERE dmid = ? AND dkey = ?",
        FileRow,
        dmid,
        dkey
    )
}

pub async fn get_file_by_fid(db: &Db, fid: i64) -> Result<Option<FileRow>> {
    q_fetch_optional!(
        db,
        "SELECT fid, dmid, dkey, length, classid, devcount FROM file WHERE fid = ?",
        FileRow,
        fid
    )
}

pub async fn insert_file(db: &Db, fid: i64, dmid: i64, dkey: &str, length: i64, classid: i64) -> Result<()> {
    q_exec!(
        db,
        "INSERT INTO file (fid, dmid, dkey, length, classid, devcount) VALUES (?, ?, ?, ?, ?, 0)",
        fid,
        dmid,
        dkey,
        length,
        classid
    )
}

/// Atomically finalizes a `create_close`: within one transaction it retires any
/// pre-existing file at the same key (overwrite), inserts the new `file` row
/// (devcount = 1) and its `file_on` location. Doing this as a single unit means
/// a concurrent `get_paths` never observes the key transiently missing, and a
/// crash mid-close cannot leave a `file` row with no location (devcount 0,
/// unreadable). Returns the retired old fid, if any, so the caller can queue its
/// blobs for deletion outside the transaction.
#[allow(clippy::too_many_arguments)]
pub async fn finalize_file(
    db: &Db,
    fid: i64,
    dmid: i64,
    dkey: &str,
    length: i64,
    classid: i64,
    devid: i64,
    retire_old: Option<i64>,
) -> Result<()> {
    macro_rules! run_tx {
        ($pool:expr, $ph_old:literal, $insert_file:literal, $insert_on:literal) => {{
            let mut tx = $pool.begin().await?;
            if let Some(old) = retire_old {
                sqlx::query($ph_old).bind(old).execute(&mut *tx).await?;
            }
            sqlx::query($insert_file)
                .bind(fid).bind(dmid).bind(dkey).bind(length).bind(classid)
                .execute(&mut *tx).await?;
            sqlx::query($insert_on).bind(fid).bind(devid).execute(&mut *tx).await?;
            tx.commit().await?;
        }};
    }
    match &db.store {
        Store::Sqlite(p) => run_tx!(
            p,
            "DELETE FROM file WHERE fid = ?",
            "INSERT INTO file (fid, dmid, dkey, length, classid, devcount) VALUES (?, ?, ?, ?, ?, 1)",
            "INSERT OR IGNORE INTO file_on (fid, devid) VALUES (?, ?)"
        ),
        Store::MySql(p) => run_tx!(
            p,
            "DELETE FROM file WHERE fid = ?",
            "INSERT INTO file (fid, dmid, dkey, length, classid, devcount) VALUES (?, ?, ?, ?, ?, 1)",
            "INSERT INTO file_on (fid, devid) VALUES (?, ?) ON DUPLICATE KEY UPDATE fid = fid"
        ),
        Store::Postgres(p) => run_tx!(
            p,
            "DELETE FROM file WHERE fid = $1",
            "INSERT INTO file (fid, dmid, dkey, length, classid, devcount) VALUES ($1, $2, $3, $4, $5, 1)",
            "INSERT INTO file_on (fid, devid) VALUES ($1, $2) ON CONFLICT DO NOTHING"
        ),
    }
    Ok(())
}

pub async fn add_file_on(db: &Db, fid: i64, devid: i64) -> Result<()> {
    insert_file_on_ignore(db, fid, devid).await?;
    recount_devcount(db, fid).await
}

/// Inserts a (fid, devid) row, treating an existing row as success. Uses each
/// backend's narrow "ignore duplicate key only" form — notably NOT MySQL's
/// `INSERT IGNORE`, which would also swallow data-conversion / out-of-range /
/// NOT NULL errors and silently drop a genuine write failure.
async fn insert_file_on_ignore(db: &Db, fid: i64, devid: i64) -> Result<()> {
    match &db.store {
        Store::Sqlite(p) => {
            sqlx::query("INSERT OR IGNORE INTO file_on (fid, devid) VALUES (?, ?)").bind(fid).bind(devid).execute(p).await?;
        }
        Store::MySql(p) => {
            sqlx::query("INSERT INTO file_on (fid, devid) VALUES (?, ?) ON DUPLICATE KEY UPDATE fid = fid")
                .bind(fid).bind(devid).execute(p).await?;
        }
        Store::Postgres(p) => {
            sqlx::query("INSERT INTO file_on (fid, devid) VALUES ($1, $2) ON CONFLICT DO NOTHING").bind(fid).bind(devid).execute(p).await?;
        }
    }
    Ok(())
}

/// Recomputes `file.devcount` from `file_on` in a single statement so that
/// concurrent add/remove callers cannot lose each other's update (the previous
/// read-count-then-write-literal form raced and left devcount wrong, which in
/// turn made the replication policy over- or under-replicate).
async fn recount_devcount(db: &Db, fid: i64) -> Result<()> {
    q_exec!(
        db,
        "UPDATE file SET devcount = (SELECT COUNT(*) FROM file_on WHERE fid = ?) WHERE fid = ?",
        fid,
        fid
    )
}

pub async fn remove_file_on(db: &Db, fid: i64, devid: i64) -> Result<()> {
    q_exec!(db, "DELETE FROM file_on WHERE fid = ? AND devid = ?", fid, devid)?;
    recount_devcount(db, fid).await
}

pub async fn get_devids_for_fid(db: &Db, fid: i64) -> Result<Vec<i64>> {
    q_fetch_all!(db, "SELECT devid FROM file_on WHERE fid = ?", (i64,), fid).map(|v| v.into_iter().map(|(d,)| d).collect())
}

pub async fn delete_file_row(db: &Db, fid: i64) -> Result<()> {
    q_exec!(db, "DELETE FROM file WHERE fid = ?", fid)
}

pub async fn rename_file_key(db: &Db, dmid: i64, fid: i64, new_key: &str) -> Result<()> {
    q_exec!(db, "UPDATE file SET dkey = ? WHERE dmid = ? AND fid = ?", new_key, dmid, fid)
}

pub async fn list_keys(db: &Db, dmid: i64, prefix: &str, after: Option<&str>, limit: i64) -> Result<Vec<(String, i64, i64)>> {
    let after = after.unwrap_or("");
    // `!` as the escape char (not `\`) since MySQL requires backslashes
    // inside string literals to themselves be escaped by default, while
    // SQLite/Postgres do not — `!` needs no such dialect-specific handling.
    let like_pattern = format!("{}%", prefix.replace('!', "!!").replace('%', "!%").replace('_', "!_"));
    let sql = "SELECT dkey, length, fid FROM file WHERE dmid = ? AND dkey LIKE ? ESCAPE '!' AND dkey > ? ORDER BY dkey LIMIT ?";
    #[derive(sqlx::FromRow)]
    struct Row {
        dkey: String,
        length: Option<i64>,
        fid: i64,
    }
    let rows: Vec<Row> = q_fetch_all!(db, sql, Row, dmid, like_pattern, after, limit)?;
    Ok(rows.into_iter().map(|r| (r.dkey, r.length.unwrap_or(0), r.fid)).collect())
}

// ---- replication / deletion queues ----

pub async fn queue_replicate(db: &Db, fid: i64, fromdevid: Option<i64>) -> Result<()> {
    match &db.store {
        Store::Sqlite(p) => {
            sqlx::query("INSERT OR IGNORE INTO file_to_replicate (fid, fromdevid) VALUES (?, ?)").bind(fid).bind(fromdevid).execute(p).await?;
        }
        Store::MySql(p) => {
            sqlx::query("INSERT INTO file_to_replicate (fid, fromdevid) VALUES (?, ?) ON DUPLICATE KEY UPDATE fid = fid")
                .bind(fid).bind(fromdevid).execute(p).await?;
        }
        Store::Postgres(p) => {
            sqlx::query("INSERT INTO file_to_replicate (fid, fromdevid) VALUES ($1, $2) ON CONFLICT DO NOTHING").bind(fid).bind(fromdevid).execute(p).await?;
        }
    }
    Ok(())
}

pub async fn dequeue_replicate(db: &Db, fid: i64) -> Result<()> {
    q_exec!(db, "DELETE FROM file_to_replicate WHERE fid = ?", fid)
}

pub async fn next_to_replicate(db: &Db, limit: i64) -> Result<Vec<i64>> {
    let sql = "SELECT fid FROM file_to_replicate WHERE nexttry <= ? ORDER BY nexttry LIMIT ?";
    let rows: Vec<(i64,)> = q_fetch_all!(db, sql, (i64,), now_ts(), limit)?;
    Ok(rows.into_iter().map(|(f,)| f).collect())
}

pub async fn bump_replicate_failure(db: &Db, fid: i64, delay_secs: i64) -> Result<()> {
    let next = now_ts() + delay_secs;
    q_exec!(db, "UPDATE file_to_replicate SET failcount = failcount + 1, nexttry = ? WHERE fid = ?", next, fid)
}

pub async fn queue_delete(db: &Db, fid: i64) -> Result<()> {
    match &db.store {
        Store::Sqlite(p) => {
            sqlx::query("INSERT OR IGNORE INTO file_to_delete (fid) VALUES (?)").bind(fid).execute(p).await?;
        }
        Store::MySql(p) => {
            sqlx::query("INSERT INTO file_to_delete (fid) VALUES (?) ON DUPLICATE KEY UPDATE fid = fid")
                .bind(fid).execute(p).await?;
        }
        Store::Postgres(p) => {
            sqlx::query("INSERT INTO file_to_delete (fid) VALUES ($1) ON CONFLICT DO NOTHING").bind(fid).execute(p).await?;
        }
    }
    Ok(())
}

pub async fn next_to_delete(db: &Db, limit: i64) -> Result<Vec<i64>> {
    let rows: Vec<(i64,)> = q_fetch_all!(db, "SELECT fid FROM file_to_delete LIMIT ?", (i64,), limit)?;
    Ok(rows.into_iter().map(|(f,)| f).collect())
}

pub async fn dequeue_delete(db: &Db, fid: i64) -> Result<()> {
    q_exec!(db, "DELETE FROM file_to_delete WHERE fid = ?", fid)
}

// ---- checksum ----

pub async fn set_checksum(db: &Db, fid: i64, alg: &str, hex: &str) -> Result<()> {
    match &db.store {
        Store::Sqlite(p) => {
            sqlx::query(
                "INSERT INTO checksum (fid, hashtype, checksum) VALUES (?, ?, ?)
                 ON CONFLICT(fid) DO UPDATE SET hashtype = excluded.hashtype, checksum = excluded.checksum",
            )
            .bind(fid)
            .bind(alg)
            .bind(hex)
            .execute(p)
            .await?;
        }
        Store::MySql(p) => {
            sqlx::query(
                "INSERT INTO checksum (fid, hashtype, checksum) VALUES (?, ?, ?)
                 ON DUPLICATE KEY UPDATE hashtype = VALUES(hashtype), checksum = VALUES(checksum)",
            )
            .bind(fid)
            .bind(alg)
            .bind(hex)
            .execute(p)
            .await?;
        }
        Store::Postgres(p) => {
            sqlx::query(
                "INSERT INTO checksum (fid, hashtype, checksum) VALUES ($1, $2, $3)
                 ON CONFLICT(fid) DO UPDATE SET hashtype = excluded.hashtype, checksum = excluded.checksum",
            )
            .bind(fid)
            .bind(alg)
            .bind(hex)
            .execute(p)
            .await?;
        }
    }
    Ok(())
}

pub async fn get_checksum(db: &Db, fid: i64) -> Result<Option<(String, String)>> {
    let row: Option<(String, String)> =
        q_fetch_optional!(db, "SELECT hashtype, checksum FROM checksum WHERE fid = ?", (String, String), fid)?;
    Ok(row)
}

// ---- list_fids ----

pub async fn list_fids_range(db: &Db, from: i64, count: i64) -> Result<Vec<FileRow>> {
    q_fetch_all!(
        db,
        "SELECT fid, dmid, dkey, length, classid, devcount FROM file WHERE fid >= ? ORDER BY fid LIMIT ?",
        FileRow,
        from,
        count
    )
}

// ---- server_settings ----

pub async fn get_setting(db: &Db, key: &str) -> Result<Option<String>> {
    let row: Option<(String,)> = q_fetch_optional!(db, "SELECT value FROM server_settings WHERE field = ?", (String,), key)?;
    Ok(row.map(|(v,)| v))
}

pub async fn set_setting(db: &Db, key: &str, value: &str) -> Result<()> {
    match &db.store {
        Store::Sqlite(p) => {
            sqlx::query("INSERT INTO server_settings (field, value) VALUES (?, ?) ON CONFLICT(field) DO UPDATE SET value = excluded.value")
                .bind(key)
                .bind(value)
                .execute(p)
                .await?;
        }
        Store::MySql(p) => {
            sqlx::query("INSERT INTO server_settings (field, value) VALUES (?, ?) ON DUPLICATE KEY UPDATE value = VALUES(value)")
                .bind(key)
                .bind(value)
                .execute(p)
                .await?;
        }
        Store::Postgres(p) => {
            sqlx::query("INSERT INTO server_settings (field, value) VALUES ($1, $2) ON CONFLICT(field) DO UPDATE SET value = excluded.value")
                .bind(key)
                .bind(value)
                .execute(p)
                .await?;
        }
    }
    Ok(())
}

pub async fn list_settings(db: &Db) -> Result<Vec<(String, String)>> {
    let rows: Vec<(String, String)> = q_fetch_all!(db, "SELECT field, value FROM server_settings ORDER BY field", (String, String))?;
    Ok(rows)
}

// ---- fsck ----

pub async fn fsck_log(db: &Db, fid: i64, evcode: &str, devid: Option<i64>) -> Result<()> {
    let now = now_ts();
    q_exec!(db, "INSERT INTO fsck_log (utime, fid, evcode, devid) VALUES (?, ?, ?, ?)", now, fid, evcode, devid)
}

pub async fn fsck_log_entries(db: &Db, limit: i64) -> Result<Vec<FsckLogEntry>> {
    q_fetch_all!(db, "SELECT logid, utime, fid, evcode, devid FROM fsck_log ORDER BY logid DESC LIMIT ?", FsckLogEntry, limit)
}

pub async fn fsck_clear_log(db: &Db) -> Result<()> {
    q_exec!(db, "DELETE FROM fsck_log")
}

pub async fn fsck_log_count(db: &Db) -> Result<i64> {
    q_scalar!(db, "SELECT COUNT(*) FROM fsck_log", i64)
}

// ---- generic file_to_queue (rebalance) ----

pub async fn queue_add(db: &Db, fid: i64, devid: Option<i64>, qtype: &str, arg: Option<&str>) -> Result<()> {
    match &db.store {
        Store::Sqlite(p) => {
            sqlx::query("INSERT OR IGNORE INTO file_to_queue (fid, devid, type, arg) VALUES (?, ?, ?, ?)")
                .bind(fid).bind(devid).bind(qtype).bind(arg).execute(p).await?;
        }
        Store::MySql(p) => {
            sqlx::query("INSERT INTO file_to_queue (fid, devid, type, arg) VALUES (?, ?, ?, ?) ON DUPLICATE KEY UPDATE fid = fid")
                .bind(fid).bind(devid).bind(qtype).bind(arg).execute(p).await?;
        }
        Store::Postgres(p) => {
            sqlx::query("INSERT INTO file_to_queue (fid, devid, type, arg) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING")
                .bind(fid).bind(devid).bind(qtype).bind(arg).execute(p).await?;
        }
    }
    Ok(())
}

pub async fn queue_next(db: &Db, qtype: &str, limit: i64) -> Result<Vec<QueueEntry>> {
    q_fetch_all!(
        db,
        "SELECT fid, devid, type, failcount, arg FROM file_to_queue WHERE type = ? AND nexttry <= ? ORDER BY nexttry LIMIT ?",
        QueueEntry,
        qtype,
        now_ts(),
        limit
    )
}

pub async fn queue_remove(db: &Db, fid: i64, qtype: &str) -> Result<()> {
    q_exec!(db, "DELETE FROM file_to_queue WHERE fid = ? AND type = ?", fid, qtype)
}

pub async fn queue_bump_failure(db: &Db, fid: i64, qtype: &str, delay_secs: i64) -> Result<()> {
    let next = now_ts() + delay_secs;
    q_exec!(
        db,
        "UPDATE file_to_queue SET failcount = failcount + 1, nexttry = ? WHERE fid = ? AND type = ?",
        next,
        fid,
        qtype
    )
}

pub async fn queue_count(db: &Db, qtype: &str) -> Result<i64> {
    q_scalar!(db, "SELECT COUNT(*) FROM file_to_queue WHERE type = ?", i64, qtype)
}

pub async fn queue_clear(db: &Db, qtype: &str) -> Result<()> {
    q_exec!(db, "DELETE FROM file_to_queue WHERE type = ?", qtype)
}

// ---- misc ----

pub async fn count_files_in_domain(db: &Db, dmid: i64) -> Result<i64> {
    q_scalar!(db, "SELECT COUNT(*) FROM file WHERE dmid = ?", i64, dmid)
}

pub async fn count_files_in_class(db: &Db, dmid: i64, classid: i64) -> Result<i64> {
    q_scalar!(db, "SELECT COUNT(*) FROM file WHERE dmid = ? AND classid = ?", i64, dmid, classid)
}

pub async fn get_fids_for_device(db: &Db, devid: i64, limit: i64) -> Result<Vec<i64>> {
    let rows: Vec<(i64,)> = q_fetch_all!(db, "SELECT fid FROM file_on WHERE devid = ? LIMIT ?", (i64,), devid, limit)?;
    Ok(rows.into_iter().map(|(f,)| f).collect())
}

pub async fn all_fids(db: &Db, limit: i64) -> Result<Vec<FileRow>> {
    q_fetch_all!(db, "SELECT fid, dmid, dkey, length, classid, devcount FROM file ORDER BY fid LIMIT ?", FileRow, limit)
}

fn now_ts() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
