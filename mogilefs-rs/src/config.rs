use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Single-binary config covering both the tracker (mogilefsd-equivalent) and
/// the storage node (mogstored-equivalent), since we run both in one process.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Backend DSN, mirroring the real tracker's `db_dsn` config key.
    /// Scheme selects the backend: `sqlite://path/to/file.db`,
    /// `mysql://user:pass@host/db`, `postgres://user:pass@host/db`.
    #[serde(default)]
    pub db_dsn: Option<String>,

    /// Legacy/shorthand SQLite path, kept for convenience: if `db_dsn` is not
    /// set, this is used as `sqlite://<db_path>`.
    #[serde(default = "default_db_path")]
    pub db_path: String,

    #[serde(default = "default_tracker_ip")]
    pub tracker_listen_ip: String,
    #[serde(default = "default_tracker_port")]
    pub tracker_port: u16,

    #[serde(default = "default_storage_ip")]
    pub storage_listen_ip: String,
    #[serde(default = "default_storage_port")]
    pub storage_port: u16,

    #[serde(default = "default_s3_ip")]
    pub s3_listen_ip: String,
    #[serde(default = "default_s3_port")]
    pub s3_port: u16,

    /// Root directory on disk that backs every configured device.
    /// A device with id N stores files under `<docroot>/dev<N>/...`.
    #[serde(default = "default_docroot")]
    pub docroot: String,

    /// Default minimum replica count applied to newly created classes
    /// when the client doesn't request otherwise (mirrors mindevcount).
    #[serde(default = "default_min_devcount")]
    pub default_min_devcount: u32,

    /// Maximum size (bytes) accepted for a single storage-node upload. The
    /// storage server streams the body to disk and aborts once this is
    /// exceeded, so an unauthenticated client cannot exhaust memory/disk with
    /// one giant PUT. Defaults to 8 GiB.
    #[serde(default = "default_max_upload_bytes")]
    pub max_upload_bytes: u64,

    /// Maximum size (bytes) for a single S3 `PutObject`. The gateway buffers the
    /// object in memory before routing it to the owning storage host, so this is
    /// deliberately modest; larger objects are the job of multipart upload (a
    /// later phase). Defaults to 256 MiB.
    #[serde(default = "default_s3_max_single_put")]
    pub s3_max_single_put: u64,

    /// Which roles this process runs. A node can be a storage node in one
    /// location (`storage` only), a metadata tracker (`tracker` + `workers`), an
    /// S3 gateway, or — the default — all of them at once (single-host mode).
    /// Multiple nodes sharing one `db_dsn` form a cluster; blob I/O routes to
    /// each device's owning host over HTTP.
    #[serde(default = "default_true")]
    pub enable_tracker: bool,
    #[serde(default = "default_true")]
    pub enable_storage: bool,
    #[serde(default = "default_true")]
    pub enable_s3: bool,
    #[serde(default = "default_true")]
    pub enable_workers: bool,
}

fn default_db_path() -> String {
    "mogilefs.db".to_string()
}
fn default_tracker_ip() -> String {
    // Loopback by default: the tracker line protocol has no authentication, so
    // binding 0.0.0.0 out of the box would expose every admin/destructive
    // command to the network. Operators who front it with a network ACL can
    // override this explicitly.
    "127.0.0.1".to_string()
}
fn default_tracker_port() -> u16 {
    7001
}
fn default_storage_ip() -> String {
    // Loopback by default for the same reason: the storage HTTP server is
    // unauthenticated. Override explicitly once a network boundary is in place.
    "127.0.0.1".to_string()
}
fn default_storage_port() -> u16 {
    7500
}
fn default_s3_ip() -> String {
    // Loopback by default: the S3 gateway ships without request signing in this
    // phase, so binding 0.0.0.0 would expose an unauthenticated object store.
    "127.0.0.1".to_string()
}
fn default_s3_port() -> u16 {
    8333
}
fn default_docroot() -> String {
    "./mogdata".to_string()
}
fn default_min_devcount() -> u32 {
    2
}
fn default_max_upload_bytes() -> u64 {
    8 * 1024 * 1024 * 1024
}
fn default_s3_max_single_put() -> u64 {
    256 * 1024 * 1024
}
fn default_true() -> bool {
    true
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            tracing::warn!(
                "config file {} not found, using built-in defaults",
                path.display()
            );
            return toml::from_str("").context("building default config");
        }
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;
        let cfg: Config = toml::from_str(&text)
            .with_context(|| format!("parsing config file {}", path.display()))?;
        Ok(cfg)
    }

    /// The DSN actually used to open the store: `db_dsn` if set, else the
    /// `db_path` shorthand turned into a `sqlite://` DSN.
    pub fn resolved_db_dsn(&self) -> String {
        match &self.db_dsn {
            Some(dsn) => dsn.clone(),
            None => format!("sqlite://{}", self.db_path),
        }
    }
}
