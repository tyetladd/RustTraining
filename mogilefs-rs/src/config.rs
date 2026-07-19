use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Single-binary config covering both the tracker (mogilefsd-equivalent) and
/// the storage node (mogstored-equivalent), since we run both in one process.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// SQLite database file backing all tracker metadata.
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
}
