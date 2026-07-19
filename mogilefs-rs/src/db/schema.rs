//! Schema mirrors the tables used by the reference Perl `MogileFS::Store`
//! (domain, class, file, tempfile, device, host, file_on, file_to_delete,
//! file_to_replicate, server_settings) adapted to SQLite types. Refined
//! against the upstream `lib/mogdbsetup.sql` / `Store/*.pm` definitions.

pub const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS domain (
    dmid    INTEGER PRIMARY KEY,
    namespace TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS class (
    dmid        INTEGER NOT NULL,
    classid     INTEGER NOT NULL,
    classname   TEXT NOT NULL,
    mindevcount INTEGER NOT NULL,
    hashtype    INTEGER,
    replpolicy  TEXT,
    PRIMARY KEY (dmid, classid),
    UNIQUE (dmid, classname)
);

CREATE TABLE IF NOT EXISTS host (
    hostid      INTEGER PRIMARY KEY,
    hostname    TEXT NOT NULL UNIQUE,
    hostip      TEXT,
    status      TEXT NOT NULL DEFAULT 'down',
    http_port   INTEGER NOT NULL DEFAULT 7500,
    http_get_port INTEGER,
    altip       TEXT,
    altmask     TEXT
);

CREATE TABLE IF NOT EXISTS device (
    devid       INTEGER PRIMARY KEY,
    hostid      INTEGER NOT NULL REFERENCES host(hostid),
    status      TEXT NOT NULL DEFAULT 'down',
    weight      INTEGER NOT NULL DEFAULT 100,
    mb_total    INTEGER,
    mb_used     INTEGER,
    mb_asof     INTEGER,
    devclass    TEXT
);

CREATE TABLE IF NOT EXISTS file (
    fid     INTEGER PRIMARY KEY,
    dmid    INTEGER NOT NULL,
    dkey    TEXT NOT NULL,
    length  INTEGER,
    classid INTEGER NOT NULL,
    devcount INTEGER NOT NULL DEFAULT 0,
    UNIQUE (dmid, dkey)
);
CREATE INDEX IF NOT EXISTS file_devcount ON file (dmid, classid, devcount);

CREATE TABLE IF NOT EXISTS tempfile (
    fid         INTEGER PRIMARY KEY AUTOINCREMENT,
    createtime  INTEGER NOT NULL,
    classid     INTEGER NOT NULL,
    dmid        INTEGER NOT NULL,
    dkey        TEXT,
    devids      TEXT NOT NULL,
    fixedkey    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS file_on (
    fid     INTEGER NOT NULL,
    devid   INTEGER NOT NULL,
    PRIMARY KEY (fid, devid)
);
CREATE INDEX IF NOT EXISTS file_on_devid ON file_on (devid);

CREATE TABLE IF NOT EXISTS file_to_replicate (
    fid         INTEGER PRIMARY KEY,
    fromdevid   INTEGER,
    failcount   INTEGER NOT NULL DEFAULT 0,
    flags       INTEGER NOT NULL DEFAULT 0,
    nexttry     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS file_to_delete (
    fid INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS file_to_delete_later (
    fid     INTEGER PRIMARY KEY,
    nexttry INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS checksum (
    fid         INTEGER PRIMARY KEY,
    hashtype    INTEGER NOT NULL,
    checksum    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS server_settings (
    field   TEXT PRIMARY KEY,
    value   TEXT
);
"#;
