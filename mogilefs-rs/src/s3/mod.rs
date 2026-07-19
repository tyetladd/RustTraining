//! Phase 1 S3-compatible gateway (in-process, anonymous, path-style addressing).
//!
//! Buckets map 1:1 onto MogileFS domains; object bytes live in the normal
//! `file`/`file_on` tables, and the extra S3 metadata (Content-Type, ETag,
//! Last-Modified, user metadata) lives in the `s3_object` sidecar table. The
//! data path reuses the tracker's device selection and `finalize_blob`, and
//! reads/writes blobs directly on the local docroot.
//!
//! Not yet implemented (later phases): SigV4 request signing, multipart upload,
//! copy, versioning, ACLs.

mod ops;
mod xml;

use crate::tracker::AppState;
use anyhow::Result;
use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::net::TcpListener;

/// Response body type: either a buffered `Full` (XML/empty) or a streamed file,
/// both boxed to a common type with `std::io::Error`.
pub type SBody = BoxBody<Bytes, std::io::Error>;

pub fn full_body(b: impl Into<Bytes>) -> SBody {
    Full::new(b.into()).map_err(|never: Infallible| match never {}).boxed()
}

pub fn empty_body() -> SBody {
    full_body(Bytes::new())
}

/// Standard URL percent-decoding (no `+` → space; that only applies to query
/// strings, handled separately). S3 object keys arrive percent-encoded in the
/// request path.
pub fn pct_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if let Some(hex) = s.get(i + 1..i + 3) {
                if let Ok(v) = u8::from_str_radix(hex, 16) {
                    out.push(v);
                    i += 3;
                    continue;
                }
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Parses a query string into a map, percent-decoding keys and values and
/// treating `+` as space (query-string convention).
fn parse_query(q: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    if q.is_empty() {
        return map;
    }
    for pair in q.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        map.insert(pct_decode(&k.replace('+', " ")), pct_decode(&v.replace('+', " ")));
    }
    map
}

async fn route(req: Request<Incoming>, state: Arc<AppState>) -> Result<Response<SBody>, Infallible> {
    let method = req.method().clone();
    let raw_path = req.uri().path().to_string();
    let query = parse_query(req.uri().query().unwrap_or(""));

    // Path-style addressing: /{bucket}/{key...}
    let decoded = pct_decode(raw_path.trim_start_matches('/'));
    let (bucket, key) = match decoded.split_once('/') {
        Some((b, k)) => (b.to_string(), Some(k.to_string())),
        None => (decoded.clone(), None),
    };

    use hyper::Method;
    let result = if bucket.is_empty() {
        match method {
            Method::GET => ops::list_buckets(&state).await,
            _ => Err(ops::S3Error::method_not_allowed()),
        }
    } else if key.as_deref().unwrap_or("").is_empty() {
        match method {
            Method::PUT => ops::create_bucket(&state, &bucket).await,
            Method::DELETE => ops::delete_bucket(&state, &bucket).await,
            Method::HEAD => ops::head_bucket(&state, &bucket).await,
            Method::GET => ops::list_objects(&state, &bucket, &query).await,
            _ => Err(ops::S3Error::method_not_allowed()),
        }
    } else {
        let key = key.unwrap();
        match method {
            Method::PUT => ops::put_object(&state, &bucket, &key, req).await,
            Method::GET => ops::get_object(&state, &bucket, &key, &req, false).await,
            Method::HEAD => ops::get_object(&state, &bucket, &key, &req, true).await,
            Method::DELETE => ops::delete_object(&state, &bucket, &key).await,
            _ => Err(ops::S3Error::method_not_allowed()),
        }
    };

    Ok(result.unwrap_or_else(|e| e.into_response(&raw_path)))
}

pub async fn serve(listener: TcpListener, state: Arc<AppState>) -> Result<()> {
    loop {
        let (stream, _peer) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let state = state.clone();
        tokio::spawn(async move {
            let svc = service_fn(move |req| route(req, state.clone()));
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, svc)
                .await
            {
                tracing::debug!("s3 http connection error: {e}");
            }
        });
    }
}

/// Helper used by ops to build a simple response with a status and headers.
pub(crate) fn response(status: StatusCode, body: SBody) -> Response<SBody> {
    Response::builder().status(status).body(body).unwrap()
}
