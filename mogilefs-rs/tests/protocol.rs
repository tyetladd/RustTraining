//! End-to-end tests that speak the real MogileFS tracker wire protocol and
//! the storage HTTP protocol directly (no client library), verifying this
//! server is byte-compatible with what `MogileFS::Backend` sends/expects.
//!
//! Runs against SQLite unconditionally. Also runs against MySQL/Postgres when
//! `MOGILEFS_TEST_MYSQL_DSN` / `MOGILEFS_TEST_POSTGRES_DSN` env vars point at
//! a live, empty-or-reusable database (skipped otherwise, e.g. in CI without
//! those services) — set them to something like
//! `mysql://mogile:mogilepass@127.0.0.1/mogilefs_test` /
//! `postgres://mogile:mogilepass@127.0.0.1/mogilefs_test` to exercise them.

use mogilefs_rs::config::Config;
use mogilefs_rs::util::{decode_args, encode_args};
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A name unique to this test process run, so tests can safely share a
/// persistent external MySQL/Postgres database without colliding on
/// domain/host uniqueness constraints.
fn uniq(prefix: &str) -> String {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    format!("{prefix}_{nanos}_{n}")
}

/// `devid` is a cluster-wide unique integer in the real schema (not scoped to
/// a domain/host), so tests sharing a persistent external MySQL/Postgres
/// database need non-colliding device ids too.
fn uniq_devid() -> i64 {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    // Keep well clear of concurrent/adjacent calls and well within i64/DB int range.
    ((nanos as u64).wrapping_mul(1000).wrapping_add(n) % 900_000_000) as i64 + 1_000
}

async fn test_config(tmp: &tempfile::TempDir, db_dsn: Option<String>) -> Config {
    let toml = format!(
        r#"
db_path = "{db}"
tracker_listen_ip = "127.0.0.1"
tracker_port = 0
storage_listen_ip = "127.0.0.1"
storage_port = 0
docroot = "{docroot}"
default_min_devcount = 1
"#,
        db = tmp.path().join("mogilefs.db").display(),
        docroot = tmp.path().join("data").display(),
    );
    let mut cfg: Config = toml::from_str(&toml).unwrap();
    cfg.db_dsn = db_dsn;
    cfg
}

struct Client {
    reader: BufReader<tokio::net::tcp::OwnedReadHalf>,
    writer: tokio::net::tcp::OwnedWriteHalf,
}

type CmdResult = Result<BTreeMap<String, String>, (String, String)>;

impl Client {
    async fn connect(addr: SocketAddr) -> Self {
        let sock = TcpStream::connect(addr).await.unwrap();
        let (r, w) = sock.into_split();
        Client { reader: BufReader::new(r), writer: w }
    }

    async fn cmd(&mut self, name: &str, pairs: &[(&str, &str)]) -> CmdResult {
        let argstr = encode_args(pairs.iter().map(|(k, v)| (*k, *v)));
        let line = format!("{name} {argstr}\r\n");
        self.writer.write_all(line.as_bytes()).await.unwrap();
        let mut resp = String::new();
        self.reader.read_line(&mut resp).await.unwrap();
        let resp = resp.trim_end();
        if let Some(rest) = resp.strip_prefix("OK ") {
            Ok(decode_args(rest).into_iter().collect())
        } else if resp == "OK" {
            Ok(BTreeMap::new())
        } else if let Some(rest) = resp.strip_prefix("ERR ") {
            let (code, text) = rest.split_once(' ').unwrap_or((rest, ""));
            Err((code.to_string(), mogilefs_rs::util::eurl_decode(text)))
        } else {
            panic!("unexpected response line: {resp:?}");
        }
    }
}

/// Minimal hand-rolled HTTP/1.1 client (PUT/GET) against the storage server,
/// good enough to exercise our own server without pulling in an HTTP client dep.
async fn http_put(addr: SocketAddr, path: &str, body: &[u8]) -> u16 {
    let mut sock = TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "PUT {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    sock.write_all(req.as_bytes()).await.unwrap();
    sock.write_all(body).await.unwrap();
    let mut resp = Vec::new();
    sock.read_to_end(&mut resp).await.unwrap();
    status_code(&resp)
}

async fn http_get(addr: SocketAddr, path: &str) -> (u16, Vec<u8>) {
    let mut sock = TcpStream::connect(addr).await.unwrap();
    let req = format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    sock.write_all(req.as_bytes()).await.unwrap();
    let mut resp = Vec::new();
    sock.read_to_end(&mut resp).await.unwrap();
    let code = status_code(&resp);
    let sep = b"\r\n\r\n";
    let pos = resp.windows(4).position(|w| w == sep).map(|p| p + 4).unwrap_or(resp.len());
    (code, resp[pos..].to_vec())
}

fn status_code(resp: &[u8]) -> u16 {
    let line_end = resp.iter().position(|&b| b == b'\r').unwrap_or(resp.len());
    let line = String::from_utf8_lossy(&resp[..line_end]);
    line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0)
}

/// Sends a raw HTTP request with an arbitrary method and request-target (so the
/// target can contain `..` without any client-side normalization) and returns
/// the status code.
async fn http_raw(addr: SocketAddr, method: &str, target: &str, body: &[u8]) -> u16 {
    let mut sock = TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "{method} {target} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    sock.write_all(req.as_bytes()).await.unwrap();
    sock.write_all(body).await.unwrap();
    let mut resp = Vec::new();
    sock.read_to_end(&mut resp).await.unwrap();
    status_code(&resp)
}

fn url_path(full_url: &str) -> String {
    let after_scheme = full_url.split_once("://").map(|(_, r)| r).unwrap_or(full_url);
    match after_scheme.split_once('/') {
        Some((_, path)) => format!("/{path}"),
        None => "/".to_string(),
    }
}

// ---------------- shared test bodies (parametrized by backend config) ----------------

async fn run_full_lifecycle(cfg: Config) {
    let handle = mogilefs_rs::spawn(cfg).await.unwrap();
    let mut c = Client::connect(handle.tracker_addr).await;
    let storage_port = handle.storage_addr.port();

    let domain = uniq("testdom");
    let host = uniq("h");

    c.cmd("noop", &[]).await.unwrap();

    c.cmd("create_domain", &[("domain", &domain)]).await.unwrap();
    assert!(c.cmd("create_domain", &[("domain", &domain)]).await.is_err());

    c.cmd("create_class", &[("domain", &domain), ("class", "c1"), ("mindevcount", "1")]).await.unwrap();
    c.cmd("create_host", &[("host", &host), ("ip", "127.0.0.1"), ("port", &storage_port.to_string())]).await.unwrap();
    c.cmd("update_host", &[("host", &host), ("status", "alive")]).await.unwrap();
    let dev1 = uniq_devid().to_string();
    c.cmd("create_device", &[("host", &host), ("devid", &dev1)]).await.unwrap();
    c.cmd("set_state", &[("host", &host), ("device", &dev1), ("state", "alive")]).await.unwrap();

    let open = c.cmd("create_open", &[("domain", &domain), ("key", "hello.txt"), ("class", "c1")]).await.unwrap();
    let fid = open["fid"].clone();
    let devid = open["devid"].clone();
    let path = open["path"].clone();
    let uri = url_path(&path);

    let body = b"Hello, MogileFS-in-Rust!";
    assert_eq!(http_put(handle.storage_addr, &uri, body).await, 201);

    c.cmd(
        "create_close",
        &[("fid", &fid), ("devid", &devid), ("path", &path), ("key", "hello.txt"), ("size", &body.len().to_string())],
    )
    .await
    .unwrap();

    let paths = c.cmd("get_paths", &[("domain", &domain), ("key", "hello.txt")]).await.unwrap();
    let get_uri = url_path(&paths["path1"]);
    let (code, data) = http_get(handle.storage_addr, &get_uri).await;
    assert_eq!(code, 200);
    assert_eq!(data, body);

    let info = c.cmd("file_info", &[("domain", &domain), ("key", "hello.txt")]).await.unwrap();
    assert_eq!(info["length"], body.len().to_string());

    let keys = c.cmd("list_keys", &[("domain", &domain)]).await.unwrap();
    assert_eq!(keys["key_count"], "1");

    c.cmd("rename", &[("domain", &domain), ("from_key", "hello.txt"), ("to_key", "hello2.txt")]).await.unwrap();
    assert!(c.cmd("get_paths", &[("domain", &domain), ("key", "hello.txt")]).await.is_err());
    assert!(c.cmd("get_paths", &[("domain", &domain), ("key", "hello2.txt")]).await.is_ok());

    c.cmd("delete", &[("domain", &domain), ("key", "hello2.txt")]).await.unwrap();
    assert!(c.cmd("get_paths", &[("domain", &domain), ("key", "hello2.txt")]).await.is_err());

    // httpcopy: copy fid's blob to a second device and confirm it lands there.
    let dev2 = uniq_devid().to_string();
    c.cmd("create_device", &[("host", &host), ("devid", &dev2)]).await.unwrap();
    c.cmd("set_state", &[("host", &host), ("device", &dev2), ("state", "alive")]).await.unwrap();
    let open2 = c.cmd("create_open", &[("domain", &domain), ("key", "copy-me"), ("class", "c1")]).await.unwrap();
    let (fid2, devid2, path2) = (open2["fid"].clone(), open2["devid"].clone(), open2["path"].clone());
    http_put(handle.storage_addr, &url_path(&path2), b"copy payload").await;
    c.cmd("create_close", &[("fid", &fid2), ("devid", &devid2), ("path", &path2), ("key", "copy-me")]).await.unwrap();
    c.cmd("httpcopy", &[("sdevid", &devid2), ("ddevid", &dev2), ("fid", &fid2)]).await.unwrap();
    let info2 = c.cmd("file_info", &[("domain", &domain), ("key", "copy-me"), ("devices", "1")]).await.unwrap();
    assert!(info2["devids"].split(',').any(|d| d == dev2), "httpcopy should add devid {dev2}: {info2:?}");
}

async fn run_checksum_verification(cfg: Config) {
    let handle = mogilefs_rs::spawn(cfg).await.unwrap();
    let mut c = Client::connect(handle.tracker_addr).await;
    let storage_port = handle.storage_addr.port();
    let domain = uniq("d");
    let host = uniq("h");

    c.cmd("create_domain", &[("domain", &domain)]).await.unwrap();
    c.cmd("create_class", &[("domain", &domain), ("class", "c"), ("mindevcount", "1")]).await.unwrap();
    c.cmd("create_host", &[("host", &host), ("ip", "127.0.0.1"), ("port", &storage_port.to_string())]).await.unwrap();
    c.cmd("update_host", &[("host", &host), ("status", "alive")]).await.unwrap();
    let dev1 = uniq_devid().to_string();
    c.cmd("create_device", &[("host", &host), ("devid", &dev1)]).await.unwrap();
    c.cmd("set_state", &[("host", &host), ("device", &dev1), ("state", "alive")]).await.unwrap();

    let open = c.cmd("create_open", &[("domain", &domain), ("key", "k"), ("class", "c")]).await.unwrap();
    let (fid, devid, path) = (open["fid"].clone(), open["devid"].clone(), open["path"].clone());
    let body = b"checksum-me-please";
    http_put(handle.storage_addr, &url_path(&path), body).await;

    let bad = c
        .cmd(
            "create_close",
            &[("fid", &fid), ("devid", &devid), ("path", &path), ("key", "k"), ("checksum", "MD5:00000000000000000000000000000000")],
        )
        .await;
    assert_eq!(bad.unwrap_err().0, "checksum_mismatch");

    // Good checksum should succeed and be echoed back by file_info.
    let open2 = c.cmd("create_open", &[("domain", &domain), ("key", "k2"), ("class", "c")]).await.unwrap();
    let (fid2, devid2, path2) = (open2["fid"].clone(), open2["devid"].clone(), open2["path"].clone());
    http_put(handle.storage_addr, &url_path(&path2), body).await;
    let md5 = {
        use md5::{Digest, Md5};
        let mut h = Md5::new();
        h.update(body);
        hex::encode(h.finalize())
    };
    c.cmd("create_close", &[("fid", &fid2), ("devid", &devid2), ("path", &path2), ("key", "k2"), ("checksum", &format!("MD5:{md5}"))])
        .await
        .unwrap();
    let info = c.cmd("file_info", &[("domain", &domain), ("key", "k2")]).await.unwrap();
    assert_eq!(info["checksum"], format!("MD5:{md5}"));
}

async fn run_replicates_to_mindevcount(cfg: Config) {
    let handle = mogilefs_rs::spawn(cfg).await.unwrap();
    let mut c = Client::connect(handle.tracker_addr).await;
    let storage_port = handle.storage_addr.port();
    let domain = uniq("d");
    let host = uniq("h");

    c.cmd("create_domain", &[("domain", &domain)]).await.unwrap();
    c.cmd("create_class", &[("domain", &domain), ("class", "c"), ("mindevcount", "2")]).await.unwrap();
    c.cmd("create_host", &[("host", &host), ("ip", "127.0.0.1"), ("port", &storage_port.to_string())]).await.unwrap();
    c.cmd("update_host", &[("host", &host), ("status", "alive")]).await.unwrap();
    for _ in 0..2 {
        let devid = uniq_devid().to_string();
        c.cmd("create_device", &[("host", &host), ("devid", &devid)]).await.unwrap();
        c.cmd("set_state", &[("host", &host), ("device", &devid), ("state", "alive")]).await.unwrap();
    }

    let open = c.cmd("create_open", &[("domain", &domain), ("key", "k"), ("class", "c")]).await.unwrap();
    let (fid, devid, path) = (open["fid"].clone(), open["devid"].clone(), open["path"].clone());
    http_put(handle.storage_addr, &url_path(&path), b"replicate me").await;
    c.cmd("create_close", &[("fid", &fid), ("devid", &devid), ("path", &path), ("key", "k")]).await.unwrap();

    let mut devcount = 0;
    for _ in 0..30 {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let info = c.cmd("file_info", &[("domain", &domain), ("key", "k")]).await.unwrap();
        devcount = info["devcount"].parse().unwrap();
        if devcount >= 2 {
            break;
        }
    }
    assert_eq!(devcount, 2, "background replicate worker should reach mindevcount=2");
}

async fn run_fsck_finds_missing_blob(cfg: Config) {
    let docroot = cfg.docroot.clone();
    let handle = mogilefs_rs::spawn(cfg).await.unwrap();
    let mut c = Client::connect(handle.tracker_addr).await;
    let storage_port = handle.storage_addr.port();
    let domain = uniq("d");
    let host = uniq("h");

    c.cmd("create_domain", &[("domain", &domain)]).await.unwrap();
    c.cmd("create_class", &[("domain", &domain), ("class", "c"), ("mindevcount", "1")]).await.unwrap();
    c.cmd("create_host", &[("host", &host), ("ip", "127.0.0.1"), ("port", &storage_port.to_string())]).await.unwrap();
    c.cmd("update_host", &[("host", &host), ("status", "alive")]).await.unwrap();
    let dev1: i64 = uniq_devid();
    let dev1s = dev1.to_string();
    c.cmd("create_device", &[("host", &host), ("devid", &dev1s)]).await.unwrap();
    c.cmd("set_state", &[("host", &host), ("device", &dev1s), ("state", "alive")]).await.unwrap();

    let open = c.cmd("create_open", &[("domain", &domain), ("key", "k"), ("class", "c")]).await.unwrap();
    let (fid, devid, path) = (open["fid"].clone(), open["devid"].clone(), open["path"].clone());
    http_put(handle.storage_addr, &url_path(&path), b"will go missing").await;
    c.cmd("create_close", &[("fid", &fid), ("devid", &devid), ("path", &path), ("key", "k")]).await.unwrap();

    // Corrupt state out from under the tracker: delete the blob directly on disk.
    let fs_path = mogilefs_rs::storage::path::fs_path(&docroot, devid.parse().unwrap(), fid.parse().unwrap());
    std::fs::remove_file(&fs_path).unwrap();

    c.cmd("fsck_start", &[]).await.unwrap();
    let mut found = false;
    for _ in 0..30 {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let log = c.cmd("fsck_getlog", &[]).await.unwrap();
        if log.get("row_count").map(|s| s.as_str()) != Some("0") {
            found = true;
            break;
        }
    }
    assert!(found, "fsck should log the missing blob");
}

async fn run_rebalance_drains_queue(cfg: Config) {
    let handle = mogilefs_rs::spawn(cfg).await.unwrap();
    let mut c = Client::connect(handle.tracker_addr).await;
    let storage_port = handle.storage_addr.port();
    let domain = uniq("d");
    let host = uniq("h");

    c.cmd("create_domain", &[("domain", &domain)]).await.unwrap();
    c.cmd("create_class", &[("domain", &domain), ("class", "c"), ("mindevcount", "1")]).await.unwrap();
    c.cmd("create_host", &[("host", &host), ("ip", "127.0.0.1"), ("port", &storage_port.to_string())]).await.unwrap();
    c.cmd("update_host", &[("host", &host), ("status", "alive")]).await.unwrap();
    for _ in 0..2 {
        let devid = uniq_devid().to_string();
        c.cmd("create_device", &[("host", &host), ("devid", &devid)]).await.unwrap();
        c.cmd("set_state", &[("host", &host), ("device", &devid), ("state", "alive")]).await.unwrap();
    }

    let open = c.cmd("create_open", &[("domain", &domain), ("key", "k"), ("class", "c")]).await.unwrap();
    let (fid, devid, path) = (open["fid"].clone(), open["devid"].clone(), open["path"].clone());
    http_put(handle.storage_addr, &url_path(&path), b"rebalance me").await;
    c.cmd("create_close", &[("fid", &fid), ("devid", &devid), ("path", &path), ("key", "k")]).await.unwrap();

    // Let the monitor worker record real disk-usage numbers before seeding
    // the rebalance queue from the busiest device.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    c.cmd("rebalance_start", &[]).await.unwrap();

    let mut drained = false;
    for _ in 0..30 {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let status = c.cmd("rebalance_status", &[]).await.unwrap();
        if status["running"] == "0" {
            drained = true;
            break;
        }
    }
    assert!(drained, "rebalance worker should finish draining its queue");

    // File should still be readable (and still have exactly 1 replica) after rebalancing.
    let info = c.cmd("file_info", &[("domain", &domain), ("key", "k")]).await.unwrap();
    assert_eq!(info["devcount"], "1");
    let paths = c.cmd("get_paths", &[("domain", &domain), ("key", "k")]).await.unwrap();
    let (code, data) = http_get(handle.storage_addr, &url_path(&paths["path1"])).await;
    assert_eq!(code, 200);
    assert_eq!(data, b"rebalance me");
}

// ---------------- sqlite (always runs) ----------------

#[tokio::test]
async fn full_lifecycle() {
    let tmp = tempfile::tempdir().unwrap();
    run_full_lifecycle(test_config(&tmp, None).await).await;
}

#[tokio::test]
async fn checksum_verification() {
    let tmp = tempfile::tempdir().unwrap();
    run_checksum_verification(test_config(&tmp, None).await).await;
}

#[tokio::test]
async fn replicates_to_mindevcount() {
    let tmp = tempfile::tempdir().unwrap();
    run_replicates_to_mindevcount(test_config(&tmp, None).await).await;
}

#[tokio::test]
async fn fsck_finds_missing_blob() {
    let tmp = tempfile::tempdir().unwrap();
    run_fsck_finds_missing_blob(test_config(&tmp, None).await).await;
}

#[tokio::test]
async fn rebalance_drains_queue() {
    let tmp = tempfile::tempdir().unwrap();
    run_rebalance_drains_queue(test_config(&tmp, None).await).await;
}

#[tokio::test]
async fn storage_rejects_path_traversal() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = test_config(&tmp, None).await;
    let handle = mogilefs_rs::spawn(cfg).await.unwrap();
    let addr = handle.storage_addr;

    // A PUT that tries to climb out of docroot (tmp/data) into the tempdir root
    // must be refused with 403 and must NOT create the target file.
    let escaped = tmp.path().join("pwned");
    assert_eq!(http_raw(addr, "PUT", "/dev1/../../pwned", b"evil").await, 403);
    assert!(!escaped.exists(), "traversal PUT must not write outside docroot");

    // GET and DELETE traversal are likewise refused.
    assert_eq!(http_raw(addr, "GET", "/dev1/../../../etc/passwd", b"").await, 403);
    assert_eq!(http_raw(addr, "DELETE", "/dev1/../../pwned", b"").await, 403);

    // A single '.' segment and a backslash are also rejected.
    assert_eq!(http_raw(addr, "GET", "/dev1/./0/000/000/x.fid", b"").await, 403);
    assert_eq!(http_raw(addr, "PUT", "/dev1/..\\x", b"evil").await, 403);
}

#[tokio::test]
async fn storage_rejects_oversize_upload() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = test_config(&tmp, None).await;
    cfg.max_upload_bytes = 16; // tiny cap for the test
    let handle = mogilefs_rs::spawn(cfg).await.unwrap();
    let addr = handle.storage_addr;

    // Under the cap succeeds; over the cap is refused with 413 and leaves no file.
    assert_eq!(http_raw(addr, "PUT", "/dev1/0/000/000/0000000001.fid", b"small").await, 201);
    let over = vec![b'x'; 64];
    assert_eq!(
        http_raw(addr, "PUT", "/dev1/0/000/000/0000000002.fid", &over).await,
        413
    );
    let leaked = tmp.path().join("data/dev1/0/000/000/0000000002.fid");
    assert!(!leaked.exists(), "over-cap upload must not leave a partial file");
}

// ---------------- mysql / postgres (only when a live DSN is provided) ----------------

/// MySQL/Postgres tests share one persistent external database across runs
/// (unlike SQLite, which gets a fresh tempfile per test), so each test wipes
/// it first for isolation — device ids are cluster-wide unique and device
/// *selection* considers every alive device in the DB, so leftover rows from
/// a previous run (or a concurrently-running test) would otherwise leak in.
async fn reset_shared_db(dsn: &str) {
    let db = mogilefs_rs::db::Db::open(dsn).await.expect("connect for reset");
    db.migrate().await.expect("migrate for reset");
    let tables = [
        "file_on",
        "file_to_replicate",
        "file_to_delete",
        "checksum",
        "file",
        "tempfile",
        "file_to_queue",
        "fsck_log",
        "class",
        "device",
        "host",
        "domain",
        "server_settings",
    ];
    for t in tables {
        let sql = format!("DELETE FROM {t}");
        match &db.store {
            mogilefs_rs::db::Store::Sqlite(p) => {
                let _ = sqlx::query(&sql).execute(p).await;
            }
            mogilefs_rs::db::Store::MySql(p) => {
                let _ = sqlx::query(&sql).execute(p).await;
            }
            mogilefs_rs::db::Store::Postgres(p) => {
                let _ = sqlx::query(&sql).execute(p).await;
            }
        }
    }
}

macro_rules! backend_tests {
    ($mod_name:ident, $env_var:literal) => {
        mod $mod_name {
            use super::*;

            // Serializes this backend's tests against each other, since they
            // all share one persistent external database.
            static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

            fn dsn() -> Option<String> {
                std::env::var($env_var).ok()
            }

            #[tokio::test]
            async fn full_lifecycle() {
                let Some(dsn) = dsn() else {
                    eprintln!("skipping: {} not set", $env_var);
                    return;
                };
                let _guard = LOCK.lock().await;
                reset_shared_db(&dsn).await;
                let tmp = tempfile::tempdir().unwrap();
                run_full_lifecycle(test_config(&tmp, Some(dsn)).await).await;
            }

            #[tokio::test]
            async fn checksum_verification() {
                let Some(dsn) = dsn() else {
                    eprintln!("skipping: {} not set", $env_var);
                    return;
                };
                let _guard = LOCK.lock().await;
                reset_shared_db(&dsn).await;
                let tmp = tempfile::tempdir().unwrap();
                run_checksum_verification(test_config(&tmp, Some(dsn)).await).await;
            }

            #[tokio::test]
            async fn replicates_to_mindevcount() {
                let Some(dsn) = dsn() else {
                    eprintln!("skipping: {} not set", $env_var);
                    return;
                };
                let _guard = LOCK.lock().await;
                reset_shared_db(&dsn).await;
                let tmp = tempfile::tempdir().unwrap();
                run_replicates_to_mindevcount(test_config(&tmp, Some(dsn)).await).await;
            }

            #[tokio::test]
            async fn fsck_finds_missing_blob() {
                let Some(dsn) = dsn() else {
                    eprintln!("skipping: {} not set", $env_var);
                    return;
                };
                let _guard = LOCK.lock().await;
                reset_shared_db(&dsn).await;
                let tmp = tempfile::tempdir().unwrap();
                run_fsck_finds_missing_blob(test_config(&tmp, Some(dsn)).await).await;
            }

            #[tokio::test]
            async fn rebalance_drains_queue() {
                let Some(dsn) = dsn() else {
                    eprintln!("skipping: {} not set", $env_var);
                    return;
                };
                let _guard = LOCK.lock().await;
                reset_shared_db(&dsn).await;
                let tmp = tempfile::tempdir().unwrap();
                run_rebalance_drains_queue(test_config(&tmp, Some(dsn)).await).await;
            }
        }
    };
}

backend_tests!(mysql_backend, "MOGILEFS_TEST_MYSQL_DSN");
backend_tests!(postgres_backend, "MOGILEFS_TEST_POSTGRES_DSN");
