pub mod path;

use crate::tracker::AppState;
use anyhow::Result;
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::header::{CONTENT_LENGTH, CONTENT_TYPE};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;

type BoxBody = Full<Bytes>;

fn empty(status: StatusCode) -> Response<BoxBody> {
    Response::builder()
        .status(status)
        .body(Full::new(Bytes::new()))
        .unwrap()
}

fn text(status: StatusCode, body: String) -> Response<BoxBody> {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "text/plain")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

/// Parses a request path of the form `/dev<N>/...` and returns (devid, rest-of-path).
fn parse_dev_path(p: &str) -> Option<(i64, &str)> {
    let p = p.strip_prefix('/')?;
    let (dev_part, rest) = p.split_once('/').unwrap_or((p, ""));
    let devid_str = dev_part.strip_prefix("dev")?;
    let devid: i64 = devid_str.parse().ok()?;
    Some((devid, rest))
}

/// Rejects any request path that could escape the docroot. hyper does not
/// collapse `..` or decode `%2e`, so the raw path is concatenated onto docroot
/// verbatim — a segment of `..` (or an empty/`.`/backslash segment, or a NUL)
/// must be refused before the path reaches the filesystem, or a client could
/// read/write/delete arbitrary files with e.g. `PUT /dev1/../../etc/passwd`.
fn path_is_safe(p: &str) -> bool {
    if p.contains('\0') || p.contains('\\') {
        return false;
    }
    for seg in p.split('/') {
        if seg == ".." || seg == "." {
            return false;
        }
    }
    true
}

async fn handle(req: Request<Incoming>, state: Arc<AppState>) -> Result<Response<BoxBody>, Infallible> {
    let path = req.uri().path().to_string();
    let method = req.method().clone();

    let Some((devid, rest)) = parse_dev_path(&path) else {
        return Ok(empty(StatusCode::NOT_FOUND));
    };

    if rest == "usage" && method == Method::GET {
        return Ok(usage_response(&state, devid));
    }

    // Containment check MUST run before the path is joined onto docroot.
    if !path_is_safe(&path) {
        tracing::warn!("rejected traversal-unsafe storage path: {path:?}");
        return Ok(empty(StatusCode::FORBIDDEN));
    }

    // fid blob path: /devN/b/mmm/ttt/nfid.fid
    let fs_path = state.cfg.docroot.clone() + &path;
    let fs_path = std::path::PathBuf::from(fs_path);

    match method {
        Method::PUT => put_blob(req, &fs_path, state.cfg.max_upload_bytes).await,
        Method::GET => match tokio::fs::read(&fs_path).await {
            Ok(data) => Ok(Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_LENGTH, data.len())
                .body(Full::new(Bytes::from(data)))
                .unwrap()),
            Err(_) => Ok(empty(StatusCode::NOT_FOUND)),
        },
        Method::HEAD => match tokio::fs::metadata(&fs_path).await {
            Ok(meta) => Ok(Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_LENGTH, meta.len())
                .body(Full::new(Bytes::new()))
                .unwrap()),
            Err(_) => Ok(empty(StatusCode::NOT_FOUND)),
        },
        Method::DELETE => {
            match tokio::fs::remove_file(&fs_path).await {
                Ok(_) => Ok(empty(StatusCode::NO_CONTENT)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(empty(StatusCode::NO_CONTENT)),
                Err(_) => Ok(empty(StatusCode::INTERNAL_SERVER_ERROR)),
            }
        }
        _ => Ok(empty(StatusCode::METHOD_NOT_ALLOWED)),
    }
}

/// Streams the request body to disk frame-by-frame, enforcing `max_bytes` so an
/// unauthenticated client cannot exhaust memory (the old `collect()` buffered
/// the entire body in RAM) or fill the disk with one unbounded PUT. On overflow
/// or write error the partial file is removed and an error status returned.
async fn put_blob(
    req: Request<Incoming>,
    fs_path: &std::path::Path,
    max_bytes: u64,
) -> Result<Response<BoxBody>, Infallible> {
    if let Some(parent) = fs_path.parent() {
        if let Err(e) = tokio::fs::create_dir_all(parent).await {
            tracing::error!("mkdir {}: {e}", parent.display());
            return Ok(empty(StatusCode::INTERNAL_SERVER_ERROR));
        }
    }

    let mut file = match tokio::fs::File::create(fs_path).await {
        Ok(f) => f,
        Err(e) => {
            tracing::error!("create {}: {e}", fs_path.display());
            return Ok(empty(StatusCode::INTERNAL_SERVER_ERROR));
        }
    };

    let mut body = req.into_body();
    let mut written: u64 = 0;
    let mut failure: Option<StatusCode> = None;

    while let Some(frame) = body.frame().await {
        let frame = match frame {
            Ok(f) => f,
            Err(_) => {
                failure = Some(StatusCode::BAD_REQUEST);
                break;
            }
        };
        if let Ok(chunk) = frame.into_data() {
            written += chunk.len() as u64;
            if written > max_bytes {
                failure = Some(StatusCode::PAYLOAD_TOO_LARGE);
                break;
            }
            if file.write_all(&chunk).await.is_err() {
                failure = Some(StatusCode::INTERNAL_SERVER_ERROR);
                break;
            }
        }
    }

    if let Some(status) = failure {
        drop(file);
        let _ = tokio::fs::remove_file(fs_path).await;
        return Ok(empty(status));
    }

    if let Err(e) = file.flush().await {
        tracing::error!("flush {}: {e}", fs_path.display());
        let _ = tokio::fs::remove_file(fs_path).await;
        return Ok(empty(StatusCode::INTERNAL_SERVER_ERROR));
    }
    Ok(empty(StatusCode::CREATED))
}

/// `GET /devN/usage` — plain-text `key: value` lines, as polled by the
/// tracker's Monitor worker (`used:`/`total:` in KB).
fn usage_response(state: &Arc<AppState>, devid: i64) -> Response<BoxBody> {
    let dir = path::device_root(&state.cfg.docroot, devid);
    let _ = std::fs::create_dir_all(&dir);
    let total_kb = fs4::total_space(&dir).unwrap_or(0) / 1024;
    let avail_kb = fs4::available_space(&dir).unwrap_or(0) / 1024;
    let used_kb = total_kb.saturating_sub(avail_kb);
    let body = format!("total: {total_kb}\nused: {used_kb}\n");
    text(StatusCode::OK, body)
}

pub async fn serve(listener: TcpListener, state: Arc<AppState>) -> Result<()> {
    loop {
        let (stream, _peer) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let state = state.clone();
        tokio::spawn(async move {
            let svc = service_fn(move |req| handle(req, state.clone()));
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, svc)
                .await
            {
                tracing::debug!("storage http connection error: {e}");
            }
        });
    }
}
