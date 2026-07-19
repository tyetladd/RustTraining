//! S3 operation handlers. Bucket = MogileFS domain; object bytes live in the
//! normal file/file_on tables; extra metadata in the s3_object sidecar.

use super::xml;
use super::{empty_body, full_body, response, SBody};
use crate::db::queries;
use crate::storage::path as storepath;
use crate::tracker::{replication, store_ops, AppState};
use http_body_util::{BodyExt, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::header::{
    ACCEPT_RANGES, CONTENT_LENGTH, CONTENT_RANGE, CONTENT_TYPE, ETAG, LAST_MODIFIED, LOCATION, RANGE,
};
use hyper::{HeaderMap, Request, Response, StatusCode};
use std::collections::BTreeMap;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio_stream::StreamExt;
use tokio_util::io::ReaderStream;

/// An S3 error mapped to an HTTP status + error code, rendered as the standard
/// S3 XML error document.
pub struct S3Error {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

impl S3Error {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self { status, code, message: message.into() }
    }
    pub fn no_such_bucket() -> Self {
        Self::new(StatusCode::NOT_FOUND, "NoSuchBucket", "The specified bucket does not exist")
    }
    pub fn no_such_key() -> Self {
        Self::new(StatusCode::NOT_FOUND, "NoSuchKey", "The specified key does not exist")
    }
    pub fn bucket_not_empty() -> Self {
        Self::new(StatusCode::CONFLICT, "BucketNotEmpty", "The bucket you tried to delete is not empty")
    }
    pub fn method_not_allowed() -> Self {
        Self::new(StatusCode::METHOD_NOT_ALLOWED, "MethodNotAllowed", "The specified method is not allowed")
    }
    pub fn invalid_range() -> Self {
        Self::new(StatusCode::RANGE_NOT_SATISFIABLE, "InvalidRange", "The requested range is not satisfiable")
    }
    pub fn no_devices() -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, "ServiceUnavailable", "No storage devices available")
    }
    pub fn too_large() -> Self {
        Self::new(StatusCode::BAD_REQUEST, "EntityTooLarge", "The upload exceeds the maximum allowed size")
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "InternalError", msg)
    }

    pub fn into_response(self, resource: &str) -> Response<SBody> {
        let body = xml::error(self.code, &self.message, resource);
        Response::builder()
            .status(self.status)
            .header(CONTENT_TYPE, "application/xml")
            .body(full_body(body))
            .unwrap()
    }
}

type OpResult = Result<Response<SBody>, S3Error>;

fn db_err(e: anyhow::Error) -> S3Error {
    tracing::error!("s3 database error: {e:#}");
    S3Error::internal("internal error")
}

async fn resolve_bucket(state: &AppState, bucket: &str) -> Result<i64, S3Error> {
    queries::get_domain_by_name(&state.db, bucket)
        .await
        .map_err(db_err)?
        .map(|d| d.dmid)
        .ok_or_else(S3Error::no_such_bucket)
}

fn xml_response(body: String) -> Response<SBody> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "application/xml")
        .body(full_body(body))
        .unwrap()
}

// ---------------- bucket operations ----------------

pub async fn list_buckets(state: &AppState) -> OpResult {
    let buckets = queries::s3_list_buckets(&state.db).await.map_err(db_err)?;
    Ok(xml_response(xml::list_all_buckets(&buckets)))
}

pub async fn create_bucket(state: &AppState, bucket: &str) -> OpResult {
    if bucket.is_empty() || bucket.contains('/') || bucket.len() > 255 {
        return Err(S3Error::new(StatusCode::BAD_REQUEST, "InvalidBucketName", "Invalid bucket name"));
    }
    let dmid = match queries::get_domain_by_name(&state.db, bucket).await.map_err(db_err)? {
        Some(d) => d.dmid,
        None => queries::create_domain(&state.db, bucket).await.map_err(db_err)?,
    };
    queries::s3_bucket_create(&state.db, dmid).await.map_err(db_err)?;
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(LOCATION, format!("/{bucket}"))
        .body(empty_body())
        .unwrap())
}

pub async fn delete_bucket(state: &AppState, bucket: &str) -> OpResult {
    let dmid = resolve_bucket(state, bucket).await?;
    if queries::count_files_in_domain(&state.db, dmid).await.map_err(db_err)? > 0 {
        return Err(S3Error::bucket_not_empty());
    }
    queries::s3_bucket_delete(&state.db, dmid).await.map_err(db_err)?;
    queries::delete_domain(&state.db, dmid).await.map_err(db_err)?;
    Ok(response(StatusCode::NO_CONTENT, empty_body()))
}

pub async fn head_bucket(state: &AppState, bucket: &str) -> OpResult {
    let dmid = resolve_bucket(state, bucket).await?;
    // Confirm it is an S3-managed bucket, not just any domain.
    if queries::s3_bucket_created(&state.db, dmid).await.map_err(db_err)?.is_none() {
        return Err(S3Error::no_such_bucket());
    }
    Ok(response(StatusCode::OK, empty_body()))
}

pub async fn list_objects(state: &AppState, bucket: &str, query: &BTreeMap<String, String>) -> OpResult {
    let dmid = resolve_bucket(state, bucket).await?;
    let prefix = query.get("prefix").map(|s| s.as_str()).unwrap_or("");
    let delimiter = query.get("delimiter").map(|s| s.as_str()).unwrap_or("");
    let max_keys: i64 = query.get("max-keys").and_then(|s| s.parse().ok()).unwrap_or(1000).clamp(1, 1000);
    let continuation = query.get("continuation-token").map(|s| s.as_str());
    let start_after = query.get("start-after").map(|s| s.as_str()).unwrap_or("");
    let after = continuation.unwrap_or(start_after);

    // Fetch one extra row to detect truncation.
    let rows = queries::s3_list_objects(&state.db, dmid, prefix, after, max_keys + 1)
        .await
        .map_err(db_err)?;

    let mut contents = Vec::new();
    let mut common: Vec<String> = Vec::new();
    let mut seen_prefix = std::collections::BTreeSet::new();
    let mut count = 0i64;
    let mut is_truncated = false;
    let mut last_key = String::new();

    for obj in rows {
        if count == max_keys {
            is_truncated = true;
            break;
        }
        // Delimiter rollup: if the key beyond the prefix contains the delimiter,
        // collapse it into a CommonPrefix instead of listing it.
        if !delimiter.is_empty() {
            let after_prefix = &obj.dkey[prefix.len().min(obj.dkey.len())..];
            if let Some(pos) = after_prefix.find(delimiter) {
                let cp = format!("{}{}{}", prefix, &after_prefix[..pos], delimiter);
                if seen_prefix.insert(cp.clone()) {
                    common.push(cp);
                    last_key = obj.dkey.clone();
                    count += 1;
                }
                continue;
            }
        }
        last_key = obj.dkey.clone();
        contents.push(obj);
        count += 1;
    }

    let next_token = if is_truncated { Some(last_key.as_str()) } else { None };
    let params = xml::ListObjectsParams {
        bucket,
        prefix,
        delimiter,
        max_keys,
        is_truncated,
        next_token,
        continuation_token: continuation,
        start_after,
    };
    Ok(xml_response(xml::list_objects_v2(&params, &contents, &common)))
}

// ---------------- object operations ----------------

fn collect_user_meta(headers: &HeaderMap) -> Option<String> {
    let mut map = BTreeMap::new();
    for (name, value) in headers {
        let n = name.as_str();
        if let Some(k) = n.strip_prefix("x-amz-meta-") {
            if let Ok(v) = value.to_str() {
                map.insert(k.to_string(), v.to_string());
            }
        }
    }
    if map.is_empty() {
        return None;
    }
    // Compact JSON, escaping the few characters that matter for a flat string map.
    let mut s = String::from("{");
    for (i, (k, v)) in map.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{}:{}", json_str(k), json_str(v)));
    }
    s.push('}');
    Some(s)
}

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub async fn put_object(state: &AppState, bucket: &str, key: &str, req: Request<Incoming>) -> OpResult {
    let dmid = resolve_bucket(state, bucket).await?;

    let content_type = req
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let user_meta = collect_user_meta(req.headers());

    // Pick a destination device and reserve a fid (mirrors create_open).
    let candidates = replication::select_devices(&state.db, 1, &[])
        .await
        .map_err(|_| S3Error::no_devices())?;
    let devid = candidates[0].devid;
    let devids_csv = candidates.iter().map(|c| c.devid.to_string()).collect::<Vec<_>>().join(",");
    let fid = queries::create_tempfile(&state.db, dmid, Some(key), 0, &devids_csv)
        .await
        .map_err(db_err)?;

    let fs_path = storepath::fs_path(&state.cfg.docroot, devid, fid);
    if let Some(parent) = fs_path.parent() {
        if tokio::fs::create_dir_all(parent).await.is_err() {
            let _ = queries::delete_tempfile(&state.db, fid).await;
            return Err(S3Error::internal("failed to create storage directory"));
        }
    }

    // Stream the body to disk, hashing (for the ETag) and size-capping as we go.
    let write_result = stream_to_disk(req, &fs_path, state.cfg.max_upload_bytes).await;
    let (size, etag) = match write_result {
        Ok(v) => v,
        Err(e) => {
            let _ = tokio::fs::remove_file(&fs_path).await;
            let _ = queries::delete_tempfile(&state.db, fid).await;
            return Err(e);
        }
    };

    queries::delete_tempfile(&state.db, fid).await.map_err(db_err)?;
    store_ops::finalize_blob(state, fid, dmid, key, 0, devid, size as i64, Some(("MD5".to_string(), etag.clone())))
        .await
        .map_err(|_| S3Error::internal("failed to finalize object"))?;

    let mtime = now_ts();
    queries::s3_object_upsert(
        &state.db,
        dmid,
        key,
        content_type.as_deref(),
        &etag,
        size as i64,
        mtime,
        user_meta.as_deref(),
    )
    .await
    .map_err(db_err)?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(ETAG, format!("\"{etag}\""))
        .body(empty_body())
        .unwrap())
}

/// Streams a request body to `fs_path`, returning (bytes_written, md5_hex).
async fn stream_to_disk(
    req: Request<Incoming>,
    fs_path: &std::path::Path,
    max_bytes: u64,
) -> Result<(u64, String), S3Error> {
    use md5::{Digest, Md5};
    let mut file = tokio::fs::File::create(fs_path)
        .await
        .map_err(|_| S3Error::internal("failed to create object file"))?;
    let mut body = req.into_body();
    let mut hasher = Md5::new();
    let mut size: u64 = 0;

    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|_| S3Error::internal("error reading request body"))?;
        if let Ok(chunk) = frame.into_data() {
            size += chunk.len() as u64;
            if size > max_bytes {
                return Err(S3Error::too_large());
            }
            hasher.update(&chunk);
            file.write_all(&chunk).await.map_err(|_| S3Error::internal("write error"))?;
        }
    }
    file.flush().await.map_err(|_| S3Error::internal("flush error"))?;
    Ok((size, hex::encode(hasher.finalize())))
}

pub async fn delete_object(state: &AppState, bucket: &str, key: &str) -> OpResult {
    let dmid = resolve_bucket(state, bucket).await?;
    if let Some(file) = queries::get_file_by_key(&state.db, dmid, key).await.map_err(db_err)? {
        queries::delete_file_row(&state.db, file.fid).await.map_err(db_err)?;
        queries::dequeue_replicate(&state.db, file.fid).await.map_err(db_err)?;
        queries::queue_delete(&state.db, file.fid).await.map_err(db_err)?;
    }
    queries::s3_object_delete(&state.db, dmid, key).await.map_err(db_err)?;
    // S3 DeleteObject is idempotent: 204 whether or not the key existed.
    Ok(response(StatusCode::NO_CONTENT, empty_body()))
}

/// Handles both GET (head=false) and HEAD (head=true) for an object.
pub async fn get_object(
    state: &AppState,
    bucket: &str,
    key: &str,
    req: &Request<Incoming>,
    head: bool,
) -> OpResult {
    let dmid = resolve_bucket(state, bucket).await?;
    let file = queries::get_file_by_key(&state.db, dmid, key)
        .await
        .map_err(db_err)?
        .ok_or_else(S3Error::no_such_key)?;
    let total = file.length.unwrap_or(0).max(0) as u64;

    // Metadata (from the sidecar, or synthesized for objects created via the raw
    // MogileFS wire protocol without an s3_object row).
    let meta = queries::s3_object_get(&state.db, dmid, key).await.map_err(db_err)?;
    let content_type = meta
        .as_ref()
        .and_then(|m| m.content_type.clone())
        .unwrap_or_else(|| "application/octet-stream".to_string());
    let etag = match meta.as_ref().and_then(|m| m.etag.clone()) {
        Some(e) => e,
        None => queries::get_checksum(&state.db, file.fid)
            .await
            .map_err(db_err)?
            .map(|(_, hex)| hex)
            .unwrap_or_default(),
    };
    let mtime = meta.as_ref().map(|m| m.mtime).unwrap_or(0);

    // Locate a device that actually has the blob on local disk.
    let devids = queries::get_devids_for_fid(&state.db, file.fid).await.map_err(db_err)?;
    let mut fs_path = None;
    for devid in devids {
        let p = storepath::fs_path(&state.cfg.docroot, devid, file.fid);
        if p.exists() {
            fs_path = Some(p);
            break;
        }
    }
    let fs_path = fs_path.ok_or_else(|| S3Error::internal("object data not found on any device"))?;

    // Optional Range request.
    let range = req.headers().get(RANGE).and_then(|v| v.to_str().ok()).and_then(|r| parse_range(r, total));
    let (status, offset, length, content_range) = match range {
        Some((start, end)) => {
            if start > end || start >= total {
                return Err(S3Error::invalid_range());
            }
            let len = end - start + 1;
            (StatusCode::PARTIAL_CONTENT, start, len, Some(format!("bytes {start}-{end}/{total}")))
        }
        None => (StatusCode::OK, 0u64, total, None),
    };

    let last_modified = httpdate::fmt_http_date(std::time::UNIX_EPOCH + std::time::Duration::from_secs(mtime.max(0) as u64));

    let mut builder = Response::builder()
        .status(status)
        .header(CONTENT_TYPE, content_type)
        .header(CONTENT_LENGTH, length)
        .header(ACCEPT_RANGES, "bytes")
        .header(ETAG, format!("\"{etag}\""))
        .header(LAST_MODIFIED, last_modified);
    if let Some(cr) = content_range {
        builder = builder.header(CONTENT_RANGE, cr);
    }

    if head {
        return Ok(builder.body(empty_body()).unwrap());
    }

    let mut file = tokio::fs::File::open(&fs_path)
        .await
        .map_err(|_| S3Error::internal("failed to open object data"))?;
    if offset > 0 {
        file.seek(std::io::SeekFrom::Start(offset))
            .await
            .map_err(|_| S3Error::internal("seek error"))?;
    }
    let limited = file.take(length);
    let stream = ReaderStream::new(limited).map(|res| res.map(Frame::data));
    let body = StreamBody::new(stream).boxed();
    Ok(builder.body(body).unwrap())
}

/// Parses an HTTP `Range: bytes=...` header into an inclusive (start, end) byte
/// range, resolving suffix/open-ended forms against `total`. Only the first
/// range spec is honored (S3 does the same for object GETs).
fn parse_range(header: &str, total: u64) -> Option<(u64, u64)> {
    let spec = header.strip_prefix("bytes=")?;
    let spec = spec.split(',').next()?.trim();
    let (a, b) = spec.split_once('-')?;
    if total == 0 {
        return None;
    }
    if a.is_empty() {
        // suffix: last N bytes
        let n: u64 = b.parse().ok()?;
        if n == 0 {
            return None;
        }
        let n = n.min(total);
        Some((total - n, total - 1))
    } else {
        let start: u64 = a.parse().ok()?;
        let end = if b.is_empty() { total - 1 } else { b.parse::<u64>().ok()?.min(total - 1) };
        Some((start, end))
    }
}

fn now_ts() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
