//! End-to-end tests for the S3 gateway, driving it over raw HTTP (no S3 SDK):
//! bucket create/list/head/delete and object put/get/head/range/list/delete,
//! plus ETag and traversal/robustness checks. SQLite backend, in-process.

use mogilefs_rs::config::Config;
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

async fn test_config(tmp: &tempfile::TempDir) -> Config {
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
        db = tmp.path().join("mogilefs.db").display(),
        docroot = tmp.path().join("data").display(),
    );
    toml::from_str(&toml).unwrap()
}

struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Resp {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// Sends one raw HTTP/1.1 request and parses the response (Content-Length or
/// connection-close framing — good enough for the gateway's responses).
async fn request(addr: SocketAddr, method: &str, target: &str, headers: &[(&str, &str)], body: &[u8]) -> Resp {
    let mut sock = TcpStream::connect(addr).await.unwrap();
    let mut req = format!("{method} {target} HTTP/1.1\r\nHost: s3\r\nConnection: close\r\n");
    req.push_str(&format!("Content-Length: {}\r\n", body.len()));
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    sock.write_all(req.as_bytes()).await.unwrap();
    sock.write_all(body).await.unwrap();
    let mut raw = Vec::new();
    sock.read_to_end(&mut raw).await.unwrap();

    let sep = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
    let head = String::from_utf8_lossy(&raw[..sep]).into_owned();
    let body = raw[sep + 4..].to_vec();
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap();
    let status: u16 = status_line.split_whitespace().nth(1).unwrap().parse().unwrap();
    let headers = lines
        .filter_map(|l| l.split_once(": ").map(|(k, v)| (k.to_string(), v.to_string())))
        .collect();
    Resp { status, headers, body }
}

#[tokio::test]
async fn s3_bucket_and_object_lifecycle() {
    let tmp = tempfile::tempdir().unwrap();
    let handle = mogilefs_rs::spawn(test_config(&tmp).await).await.unwrap();
    let s3 = handle.s3_addr;

    // A device is required for object storage; register one via the tracker,
    // pointing at this node's real (ephemeral) storage port so the gateway can
    // route blob I/O to it over HTTP.
    setup_device(handle.tracker_addr, handle.storage_addr.port()).await;

    // Bucket does not exist yet.
    assert_eq!(request(s3, "HEAD", "/mybucket", &[], b"").await.status, 404);

    // CreateBucket.
    assert_eq!(request(s3, "PUT", "/mybucket", &[], b"").await.status, 200);
    assert_eq!(request(s3, "HEAD", "/mybucket", &[], b"").await.status, 200);

    // ListBuckets shows it.
    let lb = request(s3, "GET", "/", &[], b"").await;
    assert_eq!(lb.status, 200);
    assert!(lb.text().contains("<Name>mybucket</Name>"), "{}", lb.text());

    // PutObject.
    let body = b"hello s3 world";
    let put = request(s3, "PUT", "/mybucket/greeting.txt", &[("Content-Type", "text/plain")], body).await;
    assert_eq!(put.status, 200);
    let etag = put.header("etag").unwrap().trim_matches('"').to_string();
    let expect_md5 = {
        use md5::{Digest, Md5};
        let mut h = Md5::new();
        h.update(body);
        hex::encode(h.finalize())
    };
    assert_eq!(etag, expect_md5, "ETag should be the object MD5");

    // GetObject returns the bytes + metadata.
    let get = request(s3, "GET", "/mybucket/greeting.txt", &[], b"").await;
    assert_eq!(get.status, 200);
    assert_eq!(get.body, body);
    assert_eq!(get.header("content-type"), Some("text/plain"));
    assert_eq!(get.header("etag").unwrap().trim_matches('"'), expect_md5);
    assert_eq!(get.header("accept-ranges"), Some("bytes"));

    // HeadObject: headers, no body.
    let head = request(s3, "HEAD", "/mybucket/greeting.txt", &[], b"").await;
    assert_eq!(head.status, 200);
    assert!(head.body.is_empty());
    assert_eq!(head.header("content-length"), Some(body.len().to_string().as_str()));

    // Range GET.
    let r = request(s3, "GET", "/mybucket/greeting.txt", &[("Range", "bytes=0-4")], b"").await;
    assert_eq!(r.status, 206);
    assert_eq!(r.body, b"hello");
    assert_eq!(r.header("content-range"), Some(format!("bytes 0-4/{}", body.len()).as_str()));

    // Suffix range.
    let r2 = request(s3, "GET", "/mybucket/greeting.txt", &[("Range", "bytes=-5")], b"").await;
    assert_eq!(r2.status, 206);
    assert_eq!(r2.body, b"world");

    // Another object for listing/prefix/delimiter.
    request(s3, "PUT", "/mybucket/dir/a.txt", &[], b"a").await;
    request(s3, "PUT", "/mybucket/dir/b.txt", &[], b"b").await;

    // ListObjectsV2 with a prefix.
    let list = request(s3, "GET", "/mybucket?list-type=2&prefix=dir/", &[], b"").await;
    assert_eq!(list.status, 200);
    let t = list.text();
    assert!(t.contains("<Key>dir/a.txt</Key>"), "{t}");
    assert!(t.contains("<Key>dir/b.txt</Key>"), "{t}");
    assert!(!t.contains("greeting.txt"), "prefix should exclude greeting.txt: {t}");

    // Delimiter rollup: top-level listing collapses dir/ into a CommonPrefix.
    let list2 = request(s3, "GET", "/mybucket?list-type=2&delimiter=/", &[], b"").await;
    let t2 = list2.text();
    assert!(t2.contains("<Key>greeting.txt</Key>"), "{t2}");
    assert!(t2.contains("<CommonPrefixes><Prefix>dir/</Prefix></CommonPrefixes>"), "{t2}");

    // Overwrite changes the ETag.
    let put2 = request(s3, "PUT", "/mybucket/greeting.txt", &[], b"different content").await;
    assert_ne!(put2.header("etag").unwrap().trim_matches('"'), expect_md5);

    // DeleteObject (idempotent).
    assert_eq!(request(s3, "DELETE", "/mybucket/greeting.txt", &[], b"").await.status, 204);
    assert_eq!(request(s3, "DELETE", "/mybucket/greeting.txt", &[], b"").await.status, 204);
    assert_eq!(request(s3, "GET", "/mybucket/greeting.txt", &[], b"").await.status, 404);

    // GET on a missing bucket / missing key returns S3 XML errors.
    let nb = request(s3, "GET", "/nope/x", &[], b"").await;
    assert_eq!(nb.status, 404);
    assert!(nb.text().contains("<Code>NoSuchBucket</Code>"), "{}", nb.text());

    // Non-empty bucket cannot be deleted.
    assert_eq!(request(s3, "DELETE", "/mybucket", &[], b"").await.status, 409);
    // Drain remaining objects then delete succeeds.
    request(s3, "DELETE", "/mybucket/dir/a.txt", &[], b"").await;
    request(s3, "DELETE", "/mybucket/dir/b.txt", &[], b"").await;
    assert_eq!(request(s3, "DELETE", "/mybucket", &[], b"").await.status, 204);
    assert_eq!(request(s3, "HEAD", "/mybucket", &[], b"").await.status, 404);
}

/// Registers one alive host+device via the tracker line protocol so the S3
/// gateway has somewhere to place object bytes.
async fn setup_device(tracker: SocketAddr, storage_port: u16) {
    use mogilefs_rs::util::encode_args;
    let mut sock = TcpStream::connect(tracker).await.unwrap();
    async fn cmd(sock: &mut TcpStream, name: &str, pairs: &[(&str, &str)]) {
        let line = format!("{name} {}\r\n", encode_args(pairs.iter().copied()));
        sock.write_all(line.as_bytes()).await.unwrap();
        let mut buf = [0u8; 1024];
        let n = sock.read(&mut buf).await.unwrap();
        let resp = String::from_utf8_lossy(&buf[..n]);
        assert!(resp.starts_with("OK"), "tracker {name} failed: {resp}");
    }
    let port = storage_port.to_string();
    cmd(&mut sock, "create_host", &[("host", "h1"), ("ip", "127.0.0.1"), ("port", &port)]).await;
    cmd(&mut sock, "update_host", &[("host", "h1"), ("status", "alive")]).await;
    cmd(&mut sock, "create_device", &[("host", "h1"), ("devid", "1")]).await;
    cmd(&mut sock, "set_state", &[("host", "h1"), ("device", "1"), ("state", "alive")]).await;
}
