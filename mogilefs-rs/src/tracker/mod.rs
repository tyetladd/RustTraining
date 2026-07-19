pub mod commands;
pub mod protocol;
pub mod replication;
pub mod workers;

use crate::config::Config;
use crate::db::Db;
use std::sync::Arc;
use tokio::net::TcpListener;

pub struct AppState {
    pub db: Db,
    pub cfg: Config,
}

impl AppState {
    pub fn new(db: Db, cfg: Config) -> Self {
        Self { db, cfg }
    }
}

pub async fn serve(listener: TcpListener, state: Arc<AppState>) {
    loop {
        match listener.accept().await {
            Ok((sock, peer)) => {
                let state = state.clone();
                tokio::spawn(async move {
                    if let Err(e) = protocol::handle_connection(sock, state).await {
                        tracing::debug!("connection from {peer} ended: {e:#}");
                    }
                });
            }
            Err(e) => {
                tracing::error!("accept error: {e}");
            }
        }
    }
}
