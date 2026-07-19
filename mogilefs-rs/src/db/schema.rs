//! Schema mirrors the tables used by the reference Perl `MogileFS::Store`
//! (domain, class, file, tempfile, device, host, file_on, file_to_delete,
//! file_to_replicate, server_settings, fsck_log, file_to_queue) adapted per
//! backend. Each backend gets its own DDL because auto-increment syntax,
//! upsert syntax, and `IF NOT EXISTS` support for indexes differ.

pub const SCHEMA_SQLITE: &str = r#"
CREATE TABLE IF NOT EXISTS domain (
    dmid    INTEGER PRIMARY KEY,
    namespace TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS class (
    dmid        INTEGER NOT NULL,
    classid     INTEGER NOT NULL,
    classname   TEXT NOT NULL,
    mindevcount INTEGER NOT NULL,
    replpolicy  TEXT,
    hashtype    TEXT,
    PRIMARY KEY (dmid, classid),
    UNIQUE (dmid, classname)
);

CREATE TABLE IF NOT EXISTS host (
    hostid      INTEGER PRIMARY KEY,
    hostname    TEXT NOT NULL UNIQUE,
    hostip      TEXT,
    status      TEXT NOT NULL DEFAULT 'down',
    http_port   INTEGER NOT NULL DEFAULT 7500
);

CREATE TABLE IF NOT EXISTS device (
    devid       INTEGER PRIMARY KEY,
    hostid      INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'down',
    weight      INTEGER NOT NULL DEFAULT 100,
    mb_total    INTEGER,
    mb_used     INTEGER,
    mb_asof     INTEGER
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
    devids      TEXT NOT NULL
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
    nexttry     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS file_to_delete (
    fid INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS checksum (
    fid         INTEGER PRIMARY KEY,
    hashtype    TEXT NOT NULL,
    checksum    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS server_settings (
    field   TEXT PRIMARY KEY,
    value   TEXT
);

CREATE TABLE IF NOT EXISTS fsck_log (
    logid   INTEGER PRIMARY KEY AUTOINCREMENT,
    utime   INTEGER NOT NULL,
    fid     INTEGER NOT NULL,
    evcode  TEXT NOT NULL,
    devid   INTEGER
);
CREATE INDEX IF NOT EXISTS fsck_log_utime ON fsck_log (utime);

CREATE TABLE IF NOT EXISTS file_to_queue (
    fid         INTEGER NOT NULL,
    devid       INTEGER,
    type        TEXT NOT NULL,
    nexttry     INTEGER NOT NULL DEFAULT 0,
    failcount   INTEGER NOT NULL DEFAULT 0,
    arg         TEXT,
    PRIMARY KEY (fid, type)
);
CREATE INDEX IF NOT EXISTS file_to_queue_type ON file_to_queue (type, nexttry);
"#;

pub const SCHEMA_MYSQL: &str = r#"
CREATE TABLE IF NOT EXISTS domain (
    dmid    BIGINT PRIMARY KEY AUTO_INCREMENT,
    namespace VARCHAR(255) NOT NULL,
    UNIQUE KEY domain_namespace (namespace)
);

CREATE TABLE IF NOT EXISTS class (
    dmid        BIGINT NOT NULL,
    classid     BIGINT NOT NULL,
    classname   VARCHAR(255) NOT NULL,
    mindevcount BIGINT NOT NULL,
    replpolicy  VARCHAR(255),
    hashtype    VARCHAR(32),
    PRIMARY KEY (dmid, classid),
    UNIQUE KEY class_name (dmid, classname)
);

CREATE TABLE IF NOT EXISTS host (
    hostid      BIGINT PRIMARY KEY AUTO_INCREMENT,
    hostname    VARCHAR(255) NOT NULL,
    hostip      VARCHAR(64),
    status      VARCHAR(16) NOT NULL DEFAULT 'down',
    http_port   BIGINT NOT NULL DEFAULT 7500,
    UNIQUE KEY host_name (hostname)
);

CREATE TABLE IF NOT EXISTS device (
    devid       BIGINT PRIMARY KEY,
    hostid      BIGINT NOT NULL,
    status      VARCHAR(16) NOT NULL DEFAULT 'down',
    weight      BIGINT NOT NULL DEFAULT 100,
    mb_total    BIGINT,
    mb_used     BIGINT,
    mb_asof     BIGINT
);

CREATE TABLE IF NOT EXISTS file (
    fid     BIGINT PRIMARY KEY,
    dmid    BIGINT NOT NULL,
    dkey    VARCHAR(255) NOT NULL,
    length  BIGINT,
    classid BIGINT NOT NULL,
    devcount BIGINT NOT NULL DEFAULT 0,
    UNIQUE KEY file_key (dmid, dkey),
    KEY file_devcount (dmid, classid, devcount)
);

CREATE TABLE IF NOT EXISTS tempfile (
    fid         BIGINT PRIMARY KEY AUTO_INCREMENT,
    createtime  BIGINT NOT NULL,
    classid     BIGINT NOT NULL,
    dmid        BIGINT NOT NULL,
    dkey        VARCHAR(255),
    devids      VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS file_on (
    fid     BIGINT NOT NULL,
    devid   BIGINT NOT NULL,
    PRIMARY KEY (fid, devid),
    KEY file_on_devid (devid)
);

CREATE TABLE IF NOT EXISTS file_to_replicate (
    fid         BIGINT PRIMARY KEY,
    fromdevid   BIGINT,
    failcount   BIGINT NOT NULL DEFAULT 0,
    nexttry     BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS file_to_delete (
    fid BIGINT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS checksum (
    fid         BIGINT PRIMARY KEY,
    hashtype    VARCHAR(32) NOT NULL,
    checksum    VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS server_settings (
    field   VARCHAR(255) PRIMARY KEY,
    value   TEXT
);

CREATE TABLE IF NOT EXISTS fsck_log (
    logid   BIGINT PRIMARY KEY AUTO_INCREMENT,
    utime   BIGINT NOT NULL,
    fid     BIGINT NOT NULL,
    evcode  VARCHAR(32) NOT NULL,
    devid   BIGINT,
    KEY fsck_log_utime (utime)
);

CREATE TABLE IF NOT EXISTS file_to_queue (
    fid         BIGINT NOT NULL,
    devid       BIGINT,
    type        VARCHAR(32) NOT NULL,
    nexttry     BIGINT NOT NULL DEFAULT 0,
    failcount   BIGINT NOT NULL DEFAULT 0,
    arg         VARCHAR(255),
    PRIMARY KEY (fid, type),
    KEY file_to_queue_type (type, nexttry)
);
"#;

pub const SCHEMA_POSTGRES: &str = r#"
CREATE TABLE IF NOT EXISTS domain (
    dmid    BIGSERIAL PRIMARY KEY,
    namespace TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS class (
    dmid        BIGINT NOT NULL,
    classid     BIGINT NOT NULL,
    classname   TEXT NOT NULL,
    mindevcount BIGINT NOT NULL,
    replpolicy  TEXT,
    hashtype    TEXT,
    PRIMARY KEY (dmid, classid),
    UNIQUE (dmid, classname)
);

CREATE TABLE IF NOT EXISTS host (
    hostid      BIGSERIAL PRIMARY KEY,
    hostname    TEXT NOT NULL UNIQUE,
    hostip      TEXT,
    status      TEXT NOT NULL DEFAULT 'down',
    http_port   BIGINT NOT NULL DEFAULT 7500
);

CREATE TABLE IF NOT EXISTS device (
    devid       BIGINT PRIMARY KEY,
    hostid      BIGINT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'down',
    weight      BIGINT NOT NULL DEFAULT 100,
    mb_total    BIGINT,
    mb_used     BIGINT,
    mb_asof     BIGINT
);

CREATE TABLE IF NOT EXISTS file (
    fid     BIGINT PRIMARY KEY,
    dmid    BIGINT NOT NULL,
    dkey    TEXT NOT NULL,
    length  BIGINT,
    classid BIGINT NOT NULL,
    devcount BIGINT NOT NULL DEFAULT 0,
    UNIQUE (dmid, dkey)
);
CREATE INDEX IF NOT EXISTS file_devcount ON file (dmid, classid, devcount);

CREATE TABLE IF NOT EXISTS tempfile (
    fid         BIGSERIAL PRIMARY KEY,
    createtime  BIGINT NOT NULL,
    classid     BIGINT NOT NULL,
    dmid        BIGINT NOT NULL,
    dkey        TEXT,
    devids      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS file_on (
    fid     BIGINT NOT NULL,
    devid   BIGINT NOT NULL,
    PRIMARY KEY (fid, devid)
);
CREATE INDEX IF NOT EXISTS file_on_devid ON file_on (devid);

CREATE TABLE IF NOT EXISTS file_to_replicate (
    fid         BIGINT PRIMARY KEY,
    fromdevid   BIGINT,
    failcount   BIGINT NOT NULL DEFAULT 0,
    nexttry     BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS file_to_delete (
    fid BIGINT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS checksum (
    fid         BIGINT PRIMARY KEY,
    hashtype    TEXT NOT NULL,
    checksum    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS server_settings (
    field   TEXT PRIMARY KEY,
    value   TEXT
);

CREATE TABLE IF NOT EXISTS fsck_log (
    logid   BIGSERIAL PRIMARY KEY,
    utime   BIGINT NOT NULL,
    fid     BIGINT NOT NULL,
    evcode  TEXT NOT NULL,
    devid   BIGINT
);
CREATE INDEX IF NOT EXISTS fsck_log_utime ON fsck_log (utime);

CREATE TABLE IF NOT EXISTS file_to_queue (
    fid         BIGINT NOT NULL,
    devid       BIGINT,
    type        TEXT NOT NULL,
    nexttry     BIGINT NOT NULL DEFAULT 0,
    failcount   BIGINT NOT NULL DEFAULT 0,
    arg         TEXT,
    PRIMARY KEY (fid, type)
);
CREATE INDEX IF NOT EXISTS file_to_queue_type ON file_to_queue (type, nexttry);
"#;
