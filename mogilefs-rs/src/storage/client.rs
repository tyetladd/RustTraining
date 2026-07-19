//! HTTP client the tracker and background workers use to reach blob storage on
//! the device's **owning host**, instead of touching a local `docroot`. This is
//! what lets `mogilefs-rs` run as a real multi-node cluster: a device lives on
//! exactly one host (`device.hostid` → `host.hostip`/`host.http_port`), and all
//! server-side blob operations route there over HTTP — exactly as the reference
//! tracker talks to `mogstored`.

use super::path as storepath;
use crate::db::queries;
use crate::db::Db;
use anyhow::{anyhow, bail, Context, Result};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use std::sync::OnceLock;

type HttpClient = Client<HttpConnector, Full<Bytes>>;

fn client() -> &'static HttpClient {
    static CLIENT: OnceLock<HttpClient> = OnceLock::new();
    CLIENT.get_or_init(|| Client::builder(TokioExecutor::new()).build_http())
}

/// The network location (ip + http port) of the host that owns a device.
#[derive(Debug, Clone)]
pub struct Location {
    pub hostip: String,
    pub http_port: i64,
}

pub async fn locate(db: &Db, devid: i64) -> Result<Location> {
    let dev = queries::get_device(db, devid).await?.ok_or_else(|| anyhow!("unknown device {devid}"))?;
    let host = queries::get_host(db, dev.hostid).await?.ok_or_else(|| anyhow!("unknown host {}", dev.hostid))?;
    let hostip = host.hostip.ok_or_else(|| anyhow!("host {} has no ip", dev.hostid))?;
    Ok(Location { hostip, http_port: host.http_port })
}

pub fn blob_url(loc: &Location, devid: i64, fid: i64) -> String {
    format!("http://{}:{}{}", loc.hostip, loc.http_port, storepath::uri_path(devid, fid))
}

async fn send(req: Request<Full<Bytes>>) -> Result<Response<Incoming>> {
    client().request(req).await.map_err(|e| anyhow!("storage request failed: {e}"))
}

/// HEAD the blob on its owning host; returns its content-length, or `None` if
/// the host reports it absent (404).
pub async fn head(db: &Db, devid: i64, fid: i64) -> Result<Option<u64>> {
    let loc = locate(db, devid).await?;
    let req = Request::builder()
        .method("HEAD")
        .uri(blob_url(&loc, devid, fid))
        .body(Full::new(Bytes::new()))?;
    let resp = send(req).await?;
    match resp.status() {
        StatusCode::OK => {
            let len = resp
                .headers()
                .get(hyper::header::CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            Ok(Some(len))
        }
        StatusCode::NOT_FOUND => Ok(None),
        s => bail!("HEAD {} returned {s}", blob_url(&loc, devid, fid)),
    }
}

/// Whether the blob exists on its owning host.
pub async fn exists(db: &Db, devid: i64, fid: i64) -> Result<bool> {
    Ok(head(db, devid, fid).await?.is_some())
}

/// GET the full blob from its owning host, or `None` if absent (404).
pub async fn get_bytes(db: &Db, devid: i64, fid: i64) -> Result<Option<Vec<u8>>> {
    let loc = locate(db, devid).await?;
    let req = Request::builder()
        .method("GET")
        .uri(blob_url(&loc, devid, fid))
        .body(Full::new(Bytes::new()))?;
    let resp = send(req).await?;
    match resp.status() {
        StatusCode::OK => {
            let body = resp.into_body().collect().await.context("reading blob body")?.to_bytes();
            Ok(Some(body.to_vec()))
        }
        StatusCode::NOT_FOUND => Ok(None),
        s => bail!("GET returned {s}"),
    }
}

/// Raw GET response (optionally range-limited) from the owning host, for the S3
/// gateway to proxy-stream back to its client without buffering.
pub async fn get_response(db: &Db, devid: i64, fid: i64, range: Option<&str>) -> Result<Response<Incoming>> {
    let loc = locate(db, devid).await?;
    let mut builder = Request::builder().method("GET").uri(blob_url(&loc, devid, fid));
    if let Some(r) = range {
        builder = builder.header(hyper::header::RANGE, r);
    }
    send(builder.body(Full::new(Bytes::new()))?).await
}

/// PUT a blob to its owning host.
pub async fn put_bytes(db: &Db, devid: i64, fid: i64, body: Vec<u8>) -> Result<()> {
    let loc = locate(db, devid).await?;
    let req = Request::builder()
        .method("PUT")
        .uri(blob_url(&loc, devid, fid))
        .body(Full::new(Bytes::from(body)))?;
    let resp = send(req).await?;
    if resp.status().is_success() {
        Ok(())
    } else {
        bail!("PUT returned {}", resp.status())
    }
}

/// DELETE a blob from its owning host (a 404 counts as success — idempotent).
pub async fn delete(db: &Db, devid: i64, fid: i64) -> Result<()> {
    let loc = locate(db, devid).await?;
    let req = Request::builder()
        .method("DELETE")
        .uri(blob_url(&loc, devid, fid))
        .body(Full::new(Bytes::new()))?;
    let resp = send(req).await?;
    if resp.status().is_success() || resp.status() == StatusCode::NOT_FOUND {
        Ok(())
    } else {
        bail!("DELETE returned {}", resp.status())
    }
}

/// Copies a blob from one device to another, GETting from the source host and
/// PUTting to the destination host (which may be different machines).
pub async fn copy(db: &Db, src_devid: i64, dst_devid: i64, fid: i64) -> Result<()> {
    let bytes = get_bytes(db, src_devid, fid).await?.ok_or_else(|| anyhow!("source blob missing"))?;
    put_bytes(db, dst_devid, fid, bytes).await
}

/// Polls `GET /dev<N>/usage` on the device's owning host, returning
/// (total_mb, used_mb). The storage node reports KB; we convert to MB.
pub async fn usage(db: &Db, devid: i64) -> Result<(i64, i64)> {
    let loc = locate(db, devid).await?;
    let url = format!("http://{}:{}/dev{}/usage", loc.hostip, loc.http_port, devid);
    let req = Request::builder().method("GET").uri(url).body(Full::new(Bytes::new()))?;
    let resp = send(req).await?;
    if resp.status() != StatusCode::OK {
        bail!("usage returned {}", resp.status());
    }
    let body = resp.into_body().collect().await?.to_bytes();
    let text = String::from_utf8_lossy(&body);
    let mut total_kb = 0i64;
    let mut used_kb = 0i64;
    for line in text.lines() {
        if let Some((k, v)) = line.split_once(':') {
            let v: i64 = v.trim().parse().unwrap_or(0);
            match k.trim() {
                "total" => total_kb = v,
                "used" => used_kb = v,
                _ => {}
            }
        }
    }
    Ok((total_kb / 1024, used_kb / 1024))
}
