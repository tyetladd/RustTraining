//! Multi-location cluster test: two storage nodes in "different locations"
//! (separate docroots + ports), one tracker/S3 node, all sharing one metadata
//! DB. Verifies that blob I/O routes to each device's owning host over HTTP —
//! cross-host create_close verification, replication, get_paths, and S3.

use mogilefs_rs::config::Config;
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::io::AsyncBufReadExt;
use tokio::net::TcpStream;

fn base_config(dir: &std::path::Path, db_path: &std::path::Path) -> Config {
    let toml = format!(
        r#"
db_path = "{db}"
tracker_listen_ip = "127.0.0.1"
tracker_port = 0
storage_listen_ip = "127.0.0.1"
storage_port = 0
s3_listen_ip = "127.0.0.1"
s3_port = 0
docroot = "{docroot}"
default_min_devcount = 1
"#,
        db = db_path.display(),
        docroot = dir.display(),
    );
    toml::from_str(&toml).unwrap()
}

async fn tracker_cmd(sock: &mut BufReader<TcpStream>, line: &str) -> String {
    sock.get_mut().write_all(format!("{line}\r\n").as_bytes()).await.unwrap();
    let mut resp = String::new();
    sock.read_line(&mut resp).await.unwrap();
    resp.trim_end().to_string()
}

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
    parse_status(&resp)
}

async fn http_get(addr: SocketAddr, path: &str) -> (u16, Vec<u8>) {
    let mut sock = TcpStream::connect(addr).await.unwrap();
    let req = format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    sock.write_all(req.as_bytes()).await.unwrap();
    let mut resp = Vec::new();
    sock.read_to_end(&mut resp).await.unwrap();
    let code = parse_status(&resp);
    let pos = resp.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4).unwrap_or(resp.len());
    (code, resp[pos..].to_vec())
}

fn parse_status(resp: &[u8]) -> u16 {
    let end = resp.iter().position(|&b| b == b'\r').unwrap_or(resp.len());
    String::from_utf8_lossy(&resp[..end]).split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0)
}

fn url_path(full: &str) -> String {
    let after = full.split_once("://").map(|(_, r)| r).unwrap_or(full);
    match after.split_once('/') {
        Some((_, p)) => format!("/{p}"),
        None => "/".into(),
    }
}

fn parse_ok(resp: &str) -> std::collections::BTreeMap<String, String> {
    let rest = resp.strip_prefix("OK ").or_else(|| resp.strip_prefix("OK")).unwrap_or("");
    mogilefs_rs::util::decode_args(rest.trim())
}

#[tokio::test]
async fn two_storage_locations_one_cluster() {
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("cluster.db");

    // Two storage-only nodes, each its own docroot ("location").
    let mut cfg_a = base_config(&tmp.path().join("loc_a"), &db_path);
    cfg_a.enable_tracker = false;
    cfg_a.enable_s3 = false;
    cfg_a.enable_workers = false;
    let node_a = mogilefs_rs::spawn(cfg_a).await.unwrap();

    let mut cfg_b = base_config(&tmp.path().join("loc_b"), &db_path);
    cfg_b.enable_tracker = false;
    cfg_b.enable_s3 = false;
    cfg_b.enable_workers = false;
    let node_b = mogilefs_rs::spawn(cfg_b).await.unwrap();

    // One tracker + S3 + workers node (also sharing the DB).
    let cfg_t = base_config(&tmp.path().join("loc_t"), &db_path);
    let tracker = mogilefs_rs::spawn(cfg_t).await.unwrap();

    let storage_a = node_a.storage_addr;
    let storage_b = node_b.storage_addr;

    // Register the two storage nodes as two hosts, each with one device,
    // pointing at that node's real storage port.
    let mut t = BufReader::new(TcpStream::connect(tracker.tracker_addr).await.unwrap());
    assert!(tracker_cmd(&mut t, "create_domain domain=media").await.starts_with("OK"));
    assert!(tracker_cmd(&mut t, "create_class domain=media&class=c&mindevcount=2").await.starts_with("OK"));

    assert!(tracker_cmd(&mut t, &format!("create_host host=hostA&ip=127.0.0.1&port={}", storage_a.port())).await.starts_with("OK"));
    assert!(tracker_cmd(&mut t, "update_host host=hostA&status=alive").await.starts_with("OK"));
    assert!(tracker_cmd(&mut t, "create_device host=hostA&devid=1").await.starts_with("OK"));
    assert!(tracker_cmd(&mut t, "set_state host=hostA&device=1&state=alive").await.starts_with("OK"));

    assert!(tracker_cmd(&mut t, &format!("create_host host=hostB&ip=127.0.0.1&port={}", storage_b.port())).await.starts_with("OK"));
    assert!(tracker_cmd(&mut t, "update_host host=hostB&status=alive").await.starts_with("OK"));
    assert!(tracker_cmd(&mut t, "create_device host=hostB&devid=2").await.starts_with("OK"));
    assert!(tracker_cmd(&mut t, "set_state host=hostB&device=2&state=alive").await.starts_with("OK"));

    // create_open -> the tracker picks a device on one of the two nodes.
    let open = parse_ok(&tracker_cmd(&mut t, "create_open domain=media&key=clip.bin&class=c").await);
    let (fid, devid, path) = (open["fid"].clone(), open["devid"].clone(), open["path"].clone());

    // The returned path points at whichever storage NODE owns the device; PUT
    // the bytes there directly (as a real client would).
    let dst_storage = if devid == "1" { storage_a } else { storage_b };
    let body = b"cross-host object payload";
    assert_eq!(http_put(dst_storage, &url_path(&path), body).await, 201);

    // create_close verifies size over HTTP against the owning node (remote from
    // the tracker's perspective).
    let close = tracker_cmd(
        &mut t,
        &format!("create_close fid={fid}&devid={devid}&path={path}&key=clip.bin&size={}", body.len()),
    )
    .await;
    assert!(close.starts_with("OK"), "create_close failed: {close}");

    // The background replicate worker (mindevcount=2) must copy the blob to the
    // OTHER node over HTTP. Poll file_info until devcount reaches 2.
    let mut devcount = 0;
    for _ in 0..40 {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let info = parse_ok(&tracker_cmd(&mut t, "file_info domain=media&key=clip.bin&devices=1").await);
        devcount = info["devcount"].parse().unwrap_or(0);
        if devcount >= 2 {
            // Confirm the copy physically landed on both nodes' docroots.
            let devids = info["devids"].split(',').collect::<Vec<_>>();
            assert!(devids.contains(&"1") && devids.contains(&"2"), "should span both hosts: {devids:?}");
            break;
        }
    }
    assert_eq!(devcount, 2, "replicate worker should copy across hosts to mindevcount=2");

    // get_paths returns URLs on both hosts; fetch from each and confirm bytes.
    let paths = parse_ok(&tracker_cmd(&mut t, "get_paths domain=media&key=clip.bin&pathcount=2").await);
    let n: usize = paths["paths"].parse().unwrap();
    assert!(n >= 1);
    for i in 1..=n {
        let p = &paths[&format!("path{i}")];
        // Route to whichever node's port the URL names.
        let addr = if p.contains(&format!(":{}", storage_a.port())) { storage_a } else { storage_b };
        let (code, data) = http_get(addr, &url_path(p)).await;
        assert_eq!(code, 200);
        assert_eq!(data, body, "path{i} served wrong bytes");
    }

    // S3 gateway on the tracker node serves the same cluster: put an object, and
    // confirm it is readable back (bytes were routed to a remote storage node).
    let s3 = tracker.s3_addr;
    assert_eq!(http_put(s3, "/media", b"").await / 100, 2); // CreateBucket (bucket=domain "media" already exists -> 200)
    let s3body = b"s3 object stored on a remote node";
    let put = http_put(s3, "/media/report.txt", s3body).await;
    assert_eq!(put, 200);
    let (code, data) = http_get(s3, "/media/report.txt").await;
    assert_eq!(code, 200);
    assert_eq!(data, s3body);
}
