//! End-to-end tests that speak the real MogileFS tracker wire protocol and
//! the storage HTTP protocol directly (no client library), verifying this
//! server is byte-compatible with what `MogileFS::Backend` sends/expects.

use mogilefs_rs::config::Config;
use mogilefs_rs::util::{decode_args, encode_args};
use std::collections::BTreeMap;
use std::net::SocketAddr;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

async fn test_config(tmp: &tempfile::TempDir) -> Config {
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
    toml::from_str(&toml).unwrap()
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

#[tokio::test]
async fn full_lifecycle() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = test_config(&tmp).await;
    let handle = mogilefs_rs::spawn(cfg).await.unwrap();

    let mut c = Client::connect(handle.tracker_addr).await;
    let storage_port = handle.storage_addr.port();

    c.cmd("noop", &[]).await.unwrap();

    c.cmd("create_domain", &[("domain", "testdom")]).await.unwrap();
    assert!(c.cmd("create_domain", &[("domain", "testdom")]).await.is_err());

    c.cmd("create_class", &[("domain", "testdom"), ("class", "c1"), ("mindevcount", "1")]).await.unwrap();
    c.cmd("create_host", &[("host", "h1"), ("ip", "127.0.0.1"), ("port", &storage_port.to_string())]).await.unwrap();
    c.cmd("update_host", &[("host", "h1"), ("status", "alive")]).await.unwrap();
    c.cmd("create_device", &[("host", "h1"), ("devid", "1")]).await.unwrap();
    c.cmd("set_state", &[("host", "h1"), ("device", "1"), ("state", "alive")]).await.unwrap();

    // no devices marked alive yet for a fresh domain/class pairing without a host -> now should work
    let open = c.cmd("create_open", &[("domain", "testdom"), ("key", "hello.txt"), ("class", "c1")]).await.unwrap();
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

    let paths = c.cmd("get_paths", &[("domain", "testdom"), ("key", "hello.txt")]).await.unwrap();
    let get_uri = url_path(&paths["path1"]);
    let (code, data) = http_get(handle.storage_addr, &get_uri).await;
    assert_eq!(code, 200);
    assert_eq!(data, body);

    let info = c.cmd("file_info", &[("domain", "testdom"), ("key", "hello.txt")]).await.unwrap();
    assert_eq!(info["length"], body.len().to_string());

    let keys = c.cmd("list_keys", &[("domain", "testdom")]).await.unwrap();
    assert_eq!(keys["key_count"], "1");

    c.cmd("rename", &[("domain", "testdom"), ("from_key", "hello.txt"), ("to_key", "hello2.txt")]).await.unwrap();
    assert!(c.cmd("get_paths", &[("domain", "testdom"), ("key", "hello.txt")]).await.is_err());
    assert!(c.cmd("get_paths", &[("domain", "testdom"), ("key", "hello2.txt")]).await.is_ok());

    c.cmd("delete", &[("domain", "testdom"), ("key", "hello2.txt")]).await.unwrap();
    assert!(c.cmd("get_paths", &[("domain", "testdom"), ("key", "hello2.txt")]).await.is_err());
}

#[tokio::test]
async fn checksum_verification() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = test_config(&tmp).await;
    let handle = mogilefs_rs::spawn(cfg).await.unwrap();
    let mut c = Client::connect(handle.tracker_addr).await;
    let storage_port = handle.storage_addr.port();

    c.cmd("create_domain", &[("domain", "d")]).await.unwrap();
    c.cmd("create_class", &[("domain", "d"), ("class", "c"), ("mindevcount", "1")]).await.unwrap();
    c.cmd("create_host", &[("host", "h"), ("ip", "127.0.0.1"), ("port", &storage_port.to_string())]).await.unwrap();
    c.cmd("update_host", &[("host", "h"), ("status", "alive")]).await.unwrap();
    c.cmd("create_device", &[("host", "h"), ("devid", "1")]).await.unwrap();
    c.cmd("set_state", &[("host", "h"), ("device", "1"), ("state", "alive")]).await.unwrap();

    let open = c.cmd("create_open", &[("domain", "d"), ("key", "k"), ("class", "c")]).await.unwrap();
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
}

#[tokio::test]
async fn replicates_to_mindevcount() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = test_config(&tmp).await;
    let handle = mogilefs_rs::spawn(cfg).await.unwrap();
    let mut c = Client::connect(handle.tracker_addr).await;
    let storage_port = handle.storage_addr.port();

    c.cmd("create_domain", &[("domain", "d")]).await.unwrap();
    c.cmd("create_class", &[("domain", "d"), ("class", "c"), ("mindevcount", "2")]).await.unwrap();
    c.cmd("create_host", &[("host", "h"), ("ip", "127.0.0.1"), ("port", &storage_port.to_string())]).await.unwrap();
    c.cmd("update_host", &[("host", "h"), ("status", "alive")]).await.unwrap();
    for devid in [1, 2] {
        c.cmd("create_device", &[("host", "h"), ("devid", &devid.to_string())]).await.unwrap();
        c.cmd("set_state", &[("host", "h"), ("device", &devid.to_string()), ("state", "alive")]).await.unwrap();
    }

    let open = c.cmd("create_open", &[("domain", "d"), ("key", "k"), ("class", "c")]).await.unwrap();
    let (fid, devid, path) = (open["fid"].clone(), open["devid"].clone(), open["path"].clone());
    http_put(handle.storage_addr, &url_path(&path), b"replicate me").await;
    c.cmd("create_close", &[("fid", &fid), ("devid", &devid), ("path", &path), ("key", "k")]).await.unwrap();

    let mut devcount = 0;
    for _ in 0..30 {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let info = c.cmd("file_info", &[("domain", "d"), ("key", "k")]).await.unwrap();
        devcount = info["devcount"].parse().unwrap();
        if devcount >= 2 {
            break;
        }
    }
    assert_eq!(devcount, 2, "background replicate worker should reach mindevcount=2");
}

fn url_path(full_url: &str) -> String {
    let after_scheme = full_url.split_once("://").map(|(_, r)| r).unwrap_or(full_url);
    match after_scheme.split_once('/') {
        Some((_, path)) => format!("/{path}"),
        None => "/".to_string(),
    }
}
