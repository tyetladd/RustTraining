use anyhow::{Context, Result};
use r2d2_sqlite::SqliteConnectionManager;

pub type Pool = r2d2::Pool<SqliteConnectionManager>;
pub type Conn = r2d2::PooledConnection<SqliteConnectionManager>;

pub mod model;
pub mod queries;
pub mod schema;

#[derive(Clone)]
pub struct Db {
    pub pool: Pool,
}

impl Db {
    pub fn open(path: &str) -> Result<Self> {
        let manager = SqliteConnectionManager::file(path).with_init(|c| {
            c.execute_batch(
                "PRAGMA journal_mode=WAL;\
                 PRAGMA foreign_keys=ON;\
                 PRAGMA busy_timeout=5000;",
            )
        });
        let pool = r2d2::Pool::builder()
            .build(manager)
            .context("building sqlite connection pool")?;
        Ok(Db { pool })
    }

    pub fn conn(&self) -> Result<Conn> {
        Ok(self.pool.get()?)
    }

    pub fn migrate(&self) -> Result<()> {
        let conn = self.conn()?;
        conn.execute_batch(schema::SCHEMA_SQL)
            .context("applying schema")?;
        Ok(())
    }
}
