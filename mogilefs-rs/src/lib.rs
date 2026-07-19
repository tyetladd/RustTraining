#![allow(dead_code)] // full MogError/model taxonomies are kept for spec parity even where not yet all wired up

pub mod config;
pub mod db;
pub mod error;
pub mod s3;
pub mod storage;
pub mod tracker;
pub mod util;

use anyhow::Result;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

/// Handle to a running tracker+storage instance. Background tasks are
/// aborted when this is dropped, which is what makes `spawn` convenient for
/// integration tests (each test gets an isolated, self-cleaning server).
pub struct ServerHandle {
    pub tracker_addr: SocketAddr,
    pub storage_addr: SocketAddr,
    pub s3_addr: SocketAddr,
    tasks: Vec<JoinHandle<()>>,
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        for t in &self.tasks {
            t.abort();
        }
    }
}

/// Starts the tracker line-protocol server, the storage HTTP server, and the
/// monitor/replicate/delete background workers, all as spawned tokio tasks,
/// then returns immediately with the bound addresses.
pub async fn spawn(cfg: config::Config) -> Result<ServerHandle> {
    std::fs::create_dir_all(&cfg.docroot)?;
    let db = db::Db::open(&cfg.resolved_db_dsn()).await?;
    db.migrate().await?;
    let state = Arc::new(tracker::AppState::new(db, cfg.clone()));

    let tracker_listener = TcpListener::bind((cfg.tracker_listen_ip.as_str(), cfg.tracker_port)).await?;
    let tracker_addr = tracker_listener.local_addr()?;

    let storage_listener = TcpListener::bind((cfg.storage_listen_ip.as_str(), cfg.storage_port)).await?;
    let storage_addr = storage_listener.local_addr()?;

    let s3_listener = TcpListener::bind((cfg.s3_listen_ip.as_str(), cfg.s3_port)).await?;
    let s3_addr = s3_listener.local_addr()?;

    let mut tasks = Vec::new();

    // Each role is independently enabled so one process can be a storage node in
    // a location, a metadata tracker, an S3 gateway, or (default) all at once.
    // The listeners are always bound so the handle reports stable addresses; a
    // disabled role simply doesn't serve.
    if cfg.enable_tracker {
        let s = state.clone();
        tasks.push(tokio::spawn(async move {
            tracker::serve(tracker_listener, s).await;
        }));
    }

    if cfg.enable_storage {
        let s = state.clone();
        tasks.push(tokio::spawn(async move {
            if let Err(e) = storage::serve(storage_listener, s).await {
                tracing::error!("storage server error: {e:#}");
            }
        }));
    }

    if cfg.enable_s3 {
        let s = state.clone();
        tasks.push(tokio::spawn(async move {
            if let Err(e) = s3::serve(s3_listener, s).await {
                tracing::error!("s3 gateway error: {e:#}");
            }
        }));
    }

    if cfg.enable_workers {
        let s = state.clone();
        tasks.push(tokio::spawn(async move { tracker::workers::run_monitor(s).await }));
        let s = state.clone();
        tasks.push(tokio::spawn(async move { tracker::workers::run_replicate(s).await }));
        let s = state.clone();
        tasks.push(tokio::spawn(async move { tracker::workers::run_delete(s).await }));
        let s = state.clone();
        tasks.push(tokio::spawn(async move { tracker::workers::run_fsck(s).await }));
        let s = state.clone();
        tasks.push(tokio::spawn(async move { tracker::workers::run_rebalance(s).await }));
    }

    tracing::info!(
        "roles[tracker={} storage={} s3={} workers={}] tracker={tracker_addr} storage={storage_addr} s3={s3_addr}",
        cfg.enable_tracker, cfg.enable_storage, cfg.enable_s3, cfg.enable_workers
    );

    Ok(ServerHandle {
        tracker_addr,
        storage_addr,
        s3_addr,
        tasks,
    })
}
