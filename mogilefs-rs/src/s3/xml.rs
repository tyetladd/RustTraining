//! Minimal S3 XML response serialization. S3 responses are simple, fixed-shape
//! documents, so they're built by hand rather than pulling in an XML crate.

use crate::db::model::{S3Bucket, S3Object};

const NS: &str = "http://s3.amazonaws.com/doc/2006-03-01/";

/// XML-escapes text content (`&`, `<`, `>`).
pub fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn http_date(unix: i64) -> String {
    let t = std::time::UNIX_EPOCH + std::time::Duration::from_secs(unix.max(0) as u64);
    // ISO-8601 (S3 uses this form in listing/LastModified XML).
    // httpdate gives RFC1123; for the XML body S3 uses ISO-8601 UTC, so format
    // that explicitly from the civil date.
    iso8601_utc(t)
}

/// Formats a SystemTime as `YYYY-MM-DDTHH:MM:SS.000Z` (UTC), which is what S3
/// uses for `LastModified`/`CreationDate` in XML bodies.
pub fn iso8601_utc(t: std::time::SystemTime) -> String {
    let secs = t.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let days = secs.div_euclid(86400);
    let tod = secs.rem_euclid(86400);
    let (hh, mm, ss) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    let (y, mo, d) = civil_from_days(days);
    format!("{y:04}-{mo:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.000Z")
}

/// Converts a count of days since the Unix epoch to a (year, month, day) civil
/// date (Howard Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub fn list_all_buckets(buckets: &[S3Bucket]) -> String {
    let mut s = String::new();
    s.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    s.push_str(&format!(r#"<ListAllMyBucketsResult xmlns="{NS}">"#));
    s.push_str("<Owner><ID>mogilefs</ID><DisplayName>mogilefs</DisplayName></Owner><Buckets>");
    for b in buckets {
        s.push_str(&format!(
            "<Bucket><Name>{}</Name><CreationDate>{}</CreationDate></Bucket>",
            esc(&b.namespace),
            http_date(b.created)
        ));
    }
    s.push_str("</Buckets></ListAllMyBucketsResult>");
    s
}

pub struct ListObjectsParams<'a> {
    pub bucket: &'a str,
    pub prefix: &'a str,
    pub delimiter: &'a str,
    pub max_keys: i64,
    pub is_truncated: bool,
    pub next_token: Option<&'a str>,
    pub continuation_token: Option<&'a str>,
    pub start_after: &'a str,
}

pub fn list_objects_v2(
    p: &ListObjectsParams,
    contents: &[S3Object],
    common_prefixes: &[String],
) -> String {
    let mut s = String::new();
    s.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    s.push_str(&format!(r#"<ListBucketResult xmlns="{NS}">"#));
    s.push_str(&format!("<Name>{}</Name>", esc(p.bucket)));
    s.push_str(&format!("<Prefix>{}</Prefix>", esc(p.prefix)));
    s.push_str(&format!("<MaxKeys>{}</MaxKeys>", p.max_keys));
    s.push_str(&format!("<KeyCount>{}</KeyCount>", contents.len() + common_prefixes.len()));
    if !p.delimiter.is_empty() {
        s.push_str(&format!("<Delimiter>{}</Delimiter>", esc(p.delimiter)));
    }
    if let Some(ct) = p.continuation_token {
        s.push_str(&format!("<ContinuationToken>{}</ContinuationToken>", esc(ct)));
    }
    if !p.start_after.is_empty() {
        s.push_str(&format!("<StartAfter>{}</StartAfter>", esc(p.start_after)));
    }
    s.push_str(&format!("<IsTruncated>{}</IsTruncated>", p.is_truncated));
    if let Some(nt) = p.next_token {
        s.push_str(&format!("<NextContinuationToken>{}</NextContinuationToken>", esc(nt)));
    }
    for obj in contents {
        s.push_str("<Contents>");
        s.push_str(&format!("<Key>{}</Key>", esc(&obj.dkey)));
        s.push_str(&format!("<LastModified>{}</LastModified>", http_date(obj.mtime)));
        s.push_str(&format!("<ETag>&quot;{}&quot;</ETag>", esc(obj.etag.as_deref().unwrap_or(""))));
        s.push_str(&format!("<Size>{}</Size>", obj.size));
        s.push_str("<StorageClass>STANDARD</StorageClass>");
        s.push_str("</Contents>");
    }
    for cp in common_prefixes {
        s.push_str(&format!("<CommonPrefixes><Prefix>{}</Prefix></CommonPrefixes>", esc(cp)));
    }
    s.push_str("</ListBucketResult>");
    s
}

pub fn error(code: &str, message: &str, resource: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?><Error><Code>{}</Code><Message>{}</Message><Resource>{}</Resource><RequestId>0</RequestId></Error>"#,
        esc(code),
        esc(message),
        esc(resource)
    )
}
