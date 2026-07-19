use super::model::*;
use super::Conn;
use anyhow::Result;
use rusqlite::{params, OptionalExtension};

// ---- domain ----

pub fn create_domain(conn: &Conn, namespace: &str) -> Result<i64> {
    conn.execute("INSERT INTO domain (namespace) VALUES (?1)", params![namespace])?;
    Ok(conn.last_insert_rowid())
}

pub fn get_domain_by_name(conn: &Conn, namespace: &str) -> Result<Option<Domain>> {
    conn.query_row(
        "SELECT dmid, namespace FROM domain WHERE namespace = ?1",
        params![namespace],
        |r| Ok(Domain { dmid: r.get(0)?, namespace: r.get(1)? }),
    )
    .optional()
    .map_err(Into::into)
}

pub fn get_domain_by_id(conn: &Conn, dmid: i64) -> Result<Option<Domain>> {
    conn.query_row(
        "SELECT dmid, namespace FROM domain WHERE dmid = ?1",
        params![dmid],
        |r| Ok(Domain { dmid: r.get(0)?, namespace: r.get(1)? }),
    )
    .optional()
    .map_err(Into::into)
}

pub fn delete_domain(conn: &Conn, dmid: i64) -> Result<()> {
    conn.execute("DELETE FROM domain WHERE dmid = ?1", params![dmid])?;
    Ok(())
}

pub fn list_domains(conn: &Conn) -> Result<Vec<Domain>> {
    let mut stmt = conn.prepare("SELECT dmid, namespace FROM domain ORDER BY dmid")?;
    let rows = stmt
        .query_map([], |r| Ok(Domain { dmid: r.get(0)?, namespace: r.get(1)? }))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

// ---- class ----

pub fn create_class(conn: &Conn, dmid: i64, classname: &str, mindevcount: i64) -> Result<i64> {
    let next: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(classid), 0) + 1 FROM class WHERE dmid = ?1",
            params![dmid],
            |r| r.get(0),
        )
        .unwrap_or(1);
    conn.execute(
        "INSERT INTO class (dmid, classid, classname, mindevcount) VALUES (?1, ?2, ?3, ?4)",
        params![dmid, next, classname, mindevcount],
    )?;
    Ok(next)
}

pub fn get_class_by_name(conn: &Conn, dmid: i64, classname: &str) -> Result<Option<Class>> {
    conn.query_row(
        "SELECT dmid, classid, classname, mindevcount FROM class WHERE dmid = ?1 AND classname = ?2",
        params![dmid, classname],
        |r| Ok(Class { dmid: r.get(0)?, classid: r.get(1)?, classname: r.get(2)?, mindevcount: r.get(3)? }),
    )
    .optional()
    .map_err(Into::into)
}

pub fn get_class(conn: &Conn, dmid: i64, classid: i64) -> Result<Option<Class>> {
    conn.query_row(
        "SELECT dmid, classid, classname, mindevcount FROM class WHERE dmid = ?1 AND classid = ?2",
        params![dmid, classid],
        |r| Ok(Class { dmid: r.get(0)?, classid: r.get(1)?, classname: r.get(2)?, mindevcount: r.get(3)? }),
    )
    .optional()
    .map_err(Into::into)
}

pub fn list_classes(conn: &Conn, dmid: i64) -> Result<Vec<Class>> {
    let mut stmt = conn.prepare("SELECT dmid, classid, classname, mindevcount FROM class WHERE dmid = ?1 ORDER BY classid")?;
    let rows = stmt
        .query_map(params![dmid], |r| {
            Ok(Class { dmid: r.get(0)?, classid: r.get(1)?, classname: r.get(2)?, mindevcount: r.get(3)? })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn update_class_mindevcount(conn: &Conn, dmid: i64, classid: i64, mindevcount: i64) -> Result<()> {
    conn.execute(
        "UPDATE class SET mindevcount = ?3 WHERE dmid = ?1 AND classid = ?2",
        params![dmid, classid, mindevcount],
    )?;
    Ok(())
}

pub fn delete_class(conn: &Conn, dmid: i64, classid: i64) -> Result<()> {
    conn.execute("DELETE FROM class WHERE dmid = ?1 AND classid = ?2", params![dmid, classid])?;
    Ok(())
}

// ---- host ----

pub fn create_host(conn: &Conn, hostname: &str, hostip: Option<&str>, http_port: i64) -> Result<i64> {
    conn.execute(
        "INSERT INTO host (hostname, hostip, status, http_port) VALUES (?1, ?2, 'down', ?3)",
        params![hostname, hostip, http_port],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn get_host_by_name(conn: &Conn, hostname: &str) -> Result<Option<Host>> {
    conn.query_row(
        "SELECT hostid, hostname, hostip, status, http_port FROM host WHERE hostname = ?1",
        params![hostname],
        |r| Ok(Host { hostid: r.get(0)?, hostname: r.get(1)?, hostip: r.get(2)?, status: r.get(3)?, http_port: r.get(4)? }),
    )
    .optional()
    .map_err(Into::into)
}

pub fn get_host(conn: &Conn, hostid: i64) -> Result<Option<Host>> {
    conn.query_row(
        "SELECT hostid, hostname, hostip, status, http_port FROM host WHERE hostid = ?1",
        params![hostid],
        |r| Ok(Host { hostid: r.get(0)?, hostname: r.get(1)?, hostip: r.get(2)?, status: r.get(3)?, http_port: r.get(4)? }),
    )
    .optional()
    .map_err(Into::into)
}

pub fn list_hosts(conn: &Conn) -> Result<Vec<Host>> {
    let mut stmt = conn.prepare("SELECT hostid, hostname, hostip, status, http_port FROM host ORDER BY hostid")?;
    let rows = stmt
        .query_map([], |r| Ok(Host { hostid: r.get(0)?, hostname: r.get(1)?, hostip: r.get(2)?, status: r.get(3)?, http_port: r.get(4)? }))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn update_host_status(conn: &Conn, hostid: i64, status: &str) -> Result<()> {
    conn.execute("UPDATE host SET status = ?2 WHERE hostid = ?1", params![hostid, status])?;
    Ok(())
}

pub fn delete_host(conn: &Conn, hostid: i64) -> Result<()> {
    conn.execute("DELETE FROM host WHERE hostid = ?1", params![hostid])?;
    Ok(())
}

// ---- device ----

pub fn create_device(conn: &Conn, devid: i64, hostid: i64) -> Result<()> {
    conn.execute(
        "INSERT INTO device (devid, hostid, status, weight) VALUES (?1, ?2, 'down', 100)",
        params![devid, hostid],
    )?;
    Ok(())
}

pub fn get_device(conn: &Conn, devid: i64) -> Result<Option<Device>> {
    conn.query_row(
        "SELECT devid, hostid, status, weight, mb_total, mb_used FROM device WHERE devid = ?1",
        params![devid],
        |r| Ok(Device { devid: r.get(0)?, hostid: r.get(1)?, status: r.get(2)?, weight: r.get(3)?, mb_total: r.get(4)?, mb_used: r.get(5)? }),
    )
    .optional()
    .map_err(Into::into)
}

pub fn list_devices(conn: &Conn) -> Result<Vec<Device>> {
    let mut stmt = conn.prepare("SELECT devid, hostid, status, weight, mb_total, mb_used FROM device ORDER BY devid")?;
    let rows = stmt
        .query_map([], |r| Ok(Device { devid: r.get(0)?, hostid: r.get(1)?, status: r.get(2)?, weight: r.get(3)?, mb_total: r.get(4)?, mb_used: r.get(5)? }))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn update_device_status(conn: &Conn, devid: i64, status: &str) -> Result<()> {
    conn.execute("UPDATE device SET status = ?2 WHERE devid = ?1", params![devid, status])?;
    Ok(())
}

pub fn update_device_weight(conn: &Conn, devid: i64, weight: i64) -> Result<()> {
    conn.execute("UPDATE device SET weight = ?2 WHERE devid = ?1", params![devid, weight])?;
    Ok(())
}

pub fn update_device_usage(conn: &Conn, devid: i64, mb_total: i64, mb_used: i64) -> Result<()> {
    conn.execute(
        "UPDATE device SET mb_total = ?2, mb_used = ?3, mb_asof = strftime('%s','now') WHERE devid = ?1",
        params![devid, mb_total, mb_used],
    )?;
    Ok(())
}

// ---- tempfile / create_open ----

pub fn create_tempfile(
    conn: &Conn,
    dmid: i64,
    dkey: Option<&str>,
    classid: i64,
    devids: &str,
) -> Result<i64> {
    conn.execute(
        "INSERT INTO tempfile (createtime, classid, dmid, dkey, devids) VALUES (strftime('%s','now'), ?1, ?2, ?3, ?4)",
        params![classid, dmid, dkey, devids],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn create_tempfile_with_fid(
    conn: &Conn,
    fid: i64,
    dmid: i64,
    dkey: Option<&str>,
    classid: i64,
    devids: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO tempfile (fid, createtime, classid, dmid, dkey, devids) VALUES (?1, strftime('%s','now'), ?2, ?3, ?4, ?5)",
        params![fid, classid, dmid, dkey, devids],
    )?;
    Ok(())
}

pub fn tempfile_or_file_fid_exists(conn: &Conn, fid: i64) -> Result<bool> {
    let in_temp: i64 = conn.query_row("SELECT COUNT(*) FROM tempfile WHERE fid = ?1", params![fid], |r| r.get(0))?;
    if in_temp > 0 {
        return Ok(true);
    }
    let in_file: i64 = conn.query_row("SELECT COUNT(*) FROM file WHERE fid = ?1", params![fid], |r| r.get(0))?;
    Ok(in_file > 0)
}

pub fn get_tempfile(conn: &Conn, fid: i64) -> Result<Option<TempfileRow>> {
    conn.query_row(
        "SELECT fid, dmid, dkey, classid, devids FROM tempfile WHERE fid = ?1",
        params![fid],
        |r| {
            Ok(TempfileRow {
                fid: r.get(0)?,
                dmid: r.get(1)?,
                dkey: r.get(2)?,
                classid: r.get(3)?,
                devids: r.get(4)?,
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

pub fn delete_tempfile(conn: &Conn, fid: i64) -> Result<()> {
    conn.execute("DELETE FROM tempfile WHERE fid = ?1", params![fid])?;
    Ok(())
}

// ---- file / create_close ----

pub fn get_file_by_key(conn: &Conn, dmid: i64, dkey: &str) -> Result<Option<FileRow>> {
    conn.query_row(
        "SELECT fid, dmid, dkey, length, classid, devcount FROM file WHERE dmid = ?1 AND dkey = ?2",
        params![dmid, dkey],
        |r| Ok(FileRow { fid: r.get(0)?, dmid: r.get(1)?, dkey: r.get(2)?, length: r.get(3)?, classid: r.get(4)?, devcount: r.get(5)? }),
    )
    .optional()
    .map_err(Into::into)
}

pub fn get_file_by_fid(conn: &Conn, fid: i64) -> Result<Option<FileRow>> {
    conn.query_row(
        "SELECT fid, dmid, dkey, length, classid, devcount FROM file WHERE fid = ?1",
        params![fid],
        |r| Ok(FileRow { fid: r.get(0)?, dmid: r.get(1)?, dkey: r.get(2)?, length: r.get(3)?, classid: r.get(4)?, devcount: r.get(5)? }),
    )
    .optional()
    .map_err(Into::into)
}

/// Finalize a tempfile into a real `file` row (this is what `create_close` does),
/// replacing dmid: caller must have already deleted any pre-existing file at dkey.
pub fn insert_file(conn: &Conn, fid: i64, dmid: i64, dkey: &str, length: i64, classid: i64) -> Result<()> {
    conn.execute(
        "INSERT INTO file (fid, dmid, dkey, length, classid, devcount) VALUES (?1, ?2, ?3, ?4, ?5, 0)",
        params![fid, dmid, dkey, length, classid],
    )?;
    Ok(())
}

pub fn add_file_on(conn: &Conn, fid: i64, devid: i64) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO file_on (fid, devid) VALUES (?1, ?2)",
        params![fid, devid],
    )?;
    conn.execute(
        "UPDATE file SET devcount = (SELECT COUNT(*) FROM file_on WHERE fid = ?1) WHERE fid = ?1",
        params![fid],
    )?;
    Ok(())
}

pub fn remove_file_on(conn: &Conn, fid: i64, devid: i64) -> Result<()> {
    conn.execute("DELETE FROM file_on WHERE fid = ?1 AND devid = ?2", params![fid, devid])?;
    conn.execute(
        "UPDATE file SET devcount = (SELECT COUNT(*) FROM file_on WHERE fid = ?1) WHERE fid = ?1",
        params![fid],
    )?;
    Ok(())
}

pub fn get_devids_for_fid(conn: &Conn, fid: i64) -> Result<Vec<i64>> {
    let mut stmt = conn.prepare("SELECT devid FROM file_on WHERE fid = ?1")?;
    let rows = stmt.query_map(params![fid], |r| r.get(0))?.collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn delete_file_row(conn: &Conn, fid: i64) -> Result<()> {
    conn.execute("DELETE FROM file WHERE fid = ?1", params![fid])?;
    Ok(())
}

pub fn rename_file_key(conn: &Conn, dmid: i64, fid: i64, new_key: &str) -> Result<()> {
    conn.execute(
        "UPDATE file SET dkey = ?3 WHERE dmid = ?1 AND fid = ?2",
        params![dmid, fid, new_key],
    )?;
    Ok(())
}

pub fn list_keys(conn: &Conn, dmid: i64, prefix: &str, after: Option<&str>, limit: i64) -> Result<Vec<(String, i64, i64)>> {
    let after = after.unwrap_or("");
    let like_pattern = format!("{}%", prefix.replace('%', "\\%").replace('_', "\\_"));
    let mut stmt = conn.prepare(
        "SELECT dkey, length, fid FROM file WHERE dmid = ?1 AND dkey LIKE ?2 ESCAPE '\\' AND dkey > ?3 ORDER BY dkey LIMIT ?4",
    )?;
    let rows = stmt
        .query_map(params![dmid, like_pattern, after, limit], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Option<i64>>(1)?.unwrap_or(0), r.get::<_, i64>(2)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

// ---- replication / deletion queues ----

pub fn queue_replicate(conn: &Conn, fid: i64, fromdevid: Option<i64>) -> Result<()> {
    conn.execute(
        "INSERT OR IGNORE INTO file_to_replicate (fid, fromdevid) VALUES (?1, ?2)",
        params![fid, fromdevid],
    )?;
    Ok(())
}

pub fn dequeue_replicate(conn: &Conn, fid: i64) -> Result<()> {
    conn.execute("DELETE FROM file_to_replicate WHERE fid = ?1", params![fid])?;
    Ok(())
}

pub fn next_to_replicate(conn: &Conn, limit: i64) -> Result<Vec<i64>> {
    let mut stmt = conn.prepare(
        "SELECT fid FROM file_to_replicate WHERE nexttry <= strftime('%s','now') ORDER BY nexttry LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit], |r| r.get(0))?.collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn bump_replicate_failure(conn: &Conn, fid: i64, delay_secs: i64) -> Result<()> {
    conn.execute(
        "UPDATE file_to_replicate SET failcount = failcount + 1, nexttry = strftime('%s','now') + ?2 WHERE fid = ?1",
        params![fid, delay_secs],
    )?;
    Ok(())
}

pub fn queue_delete(conn: &Conn, fid: i64) -> Result<()> {
    conn.execute("INSERT OR IGNORE INTO file_to_delete (fid) VALUES (?1)", params![fid])?;
    Ok(())
}

pub fn next_to_delete(conn: &Conn, limit: i64) -> Result<Vec<i64>> {
    let mut stmt = conn.prepare("SELECT fid FROM file_to_delete LIMIT ?1")?;
    let rows = stmt.query_map(params![limit], |r| r.get(0))?.collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn dequeue_delete(conn: &Conn, fid: i64) -> Result<()> {
    conn.execute("DELETE FROM file_to_delete WHERE fid = ?1", params![fid])?;
    Ok(())
}

// ---- checksum ----

pub fn set_checksum(conn: &Conn, fid: i64, alg: &str, hex: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO checksum (fid, hashtype, checksum) VALUES (?1, ?2, ?3)
         ON CONFLICT(fid) DO UPDATE SET hashtype = excluded.hashtype, checksum = excluded.checksum",
        params![fid, alg, hex],
    )?;
    Ok(())
}

pub fn get_checksum(conn: &Conn, fid: i64) -> Result<Option<(String, String)>> {
    conn.query_row(
        "SELECT hashtype, checksum FROM checksum WHERE fid = ?1",
        params![fid],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )
    .optional()
    .map_err(Into::into)
}

// ---- list_fids ----

pub fn list_fids_range(conn: &Conn, from: i64, count: i64) -> Result<Vec<FileRow>> {
    let mut stmt = conn.prepare(
        "SELECT fid, dmid, dkey, length, classid, devcount FROM file WHERE fid >= ?1 ORDER BY fid LIMIT ?2",
    )?;
    let rows = stmt
        .query_map(params![from, count], |r| {
            Ok(FileRow { fid: r.get(0)?, dmid: r.get(1)?, dkey: r.get(2)?, length: r.get(3)?, classid: r.get(4)?, devcount: r.get(5)? })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

// ---- server_settings ----

pub fn get_setting(conn: &Conn, key: &str) -> Result<Option<String>> {
    conn.query_row("SELECT value FROM server_settings WHERE field = ?1", params![key], |r| r.get(0))
        .optional()
        .map_err(Into::into)
}

pub fn set_setting(conn: &Conn, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO server_settings (field, value) VALUES (?1, ?2)
         ON CONFLICT(field) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

pub fn list_settings(conn: &Conn) -> Result<Vec<(String, String)>> {
    let mut stmt = conn.prepare("SELECT field, value FROM server_settings ORDER BY field")?;
    let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn devices_for_host(conn: &Conn, hostid: i64) -> Result<i64> {
    conn.query_row("SELECT COUNT(*) FROM device WHERE hostid = ?1", params![hostid], |r| r.get(0))
        .map_err(Into::into)
}
