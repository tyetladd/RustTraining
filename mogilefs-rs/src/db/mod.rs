use anyhow::{bail, Context, Result};
use sqlx::mysql::{MySqlPool, MySqlPoolOptions};
use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::str::FromStr;

pub mod model;
pub mod queries;
pub mod schema;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    Sqlite,
    MySql,
    Postgres,
}

#[derive(Clone)]
pub enum Store {
    Sqlite(SqlitePool),
    MySql(MySqlPool),
    Postgres(PgPool),
}

#[derive(Clone)]
pub struct Db {
    pub store: Store,
}

/// Rewrites `?`-style placeholders (SQLite/MySQL syntax) into Postgres's
/// `$1, $2, ...` positional syntax. Safe here because none of our static SQL
/// text contains a literal `?` inside a string/identifier.
pub fn pg(sql: &str) -> String {
    let mut out = String::with_capacity(sql.len() + 8);
    let mut n = 0usize;
    for ch in sql.chars() {
        if ch == '?' {
            n += 1;
            out.push('$');
            out.push_str(&n.to_string());
        } else {
            out.push(ch);
        }
    }
    out
}

impl Db {
    pub fn dialect(&self) -> Dialect {
        match &self.store {
            Store::Sqlite(_) => Dialect::Sqlite,
            Store::MySql(_) => Dialect::MySql,
            Store::Postgres(_) => Dialect::Postgres,
        }
    }

    /// Opens a connection pool from a DSN. Recognized schemes:
    /// `sqlite://path/to/file.db`, `mysql://user:pass@host/db`,
    /// `postgres://user:pass@host/db` (mirrors the real tracker's `db_dsn`
    /// config key, which likewise picks the backend from the DSN).
    pub async fn open(dsn: &str) -> Result<Self> {
        let store = if let Some(rest) = dsn.strip_prefix("sqlite://") {
            let opts = SqliteConnectOptions::from_str(rest)
                .with_context(|| format!("parsing sqlite dsn {dsn}"))?
                .create_if_missing(true);
            let pool = SqlitePoolOptions::new()
                .max_connections(16)
                .connect_with(opts)
                .await
                .with_context(|| format!("connecting to sqlite db {dsn}"))?;
            Store::Sqlite(pool)
        } else if dsn.starts_with("mysql://") {
            let pool = MySqlPoolOptions::new()
                .max_connections(16)
                .connect(dsn)
                .await
                .with_context(|| "connecting to mysql".to_string())?;
            Store::MySql(pool)
        } else if dsn.starts_with("postgres://") || dsn.starts_with("postgresql://") {
            let pool = PgPoolOptions::new()
                .max_connections(16)
                .connect(dsn)
                .await
                .with_context(|| "connecting to postgres".to_string())?;
            Store::Postgres(pool)
        } else {
            bail!("unrecognized db_dsn scheme (expected sqlite://, mysql://, or postgres://): {dsn}");
        };
        Ok(Db { store })
    }

    pub async fn migrate(&self) -> Result<()> {
        let sql = match self.dialect() {
            Dialect::Sqlite => schema::SCHEMA_SQLITE,
            Dialect::MySql => schema::SCHEMA_MYSQL,
            Dialect::Postgres => schema::SCHEMA_POSTGRES,
        };
        for stmt in sql.split(';') {
            let stmt = stmt.trim();
            if stmt.is_empty() {
                continue;
            }
            match &self.store {
                Store::Sqlite(p) => {
                    sqlx::query(stmt).execute(p).await.with_context(|| format!("applying: {stmt}"))?;
                }
                Store::MySql(p) => {
                    sqlx::query(stmt).execute(p).await.with_context(|| format!("applying: {stmt}"))?;
                }
                Store::Postgres(p) => {
                    sqlx::query(stmt).execute(p).await.with_context(|| format!("applying: {stmt}"))?;
                }
            }
        }
        Ok(())
    }
}
