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

    /// Root directory on disk that backs every configured device.
    /// A device with id N stores files under `<docroot>/dev<N>/...`.
    #[serde(default = "default_docroot")]
    pub docroot: String,

    /// Default minimum replica count applied to newly created classes
    /// when the client doesn't request otherwise (mirrors mindevcount).
    #[serde(default = "default_min_devcount")]
    pub default_min_devcount: u32,
}

fn default_db_path() -> String {
    "mogilefs.db".to_string()
}
fn default_tracker_ip() -> String {
    "0.0.0.0".to_string()
}
fn default_tracker_port() -> u16 {
    7001
}
fn default_storage_ip() -> String {
    "0.0.0.0".to_string()
}
fn default_storage_port() -> u16 {
    7500
}
fn default_docroot() -> String {
    "./mogdata".to_string()
}
fn default_min_devcount() -> u32 {
    2
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
