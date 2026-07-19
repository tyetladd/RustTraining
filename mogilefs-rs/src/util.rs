use std::collections::BTreeMap;

/// `MogileFS::Util::eurl`: percent-encode everything except
/// `a-zA-Z0-9_,-./\: ` (note: backslash, colon and space are "safe"),
/// then turn literal spaces into `+`.
pub fn eurl_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        let safe = b.is_ascii_alphanumeric()
            || matches!(b, b'_' | b',' | b'-' | b'.' | b'/' | b'\\' | b':' | b' ');
        if safe {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{:02X}", b));
        }
    }
    // literal spaces -> '+'
    out.replace(' ', "+")
}

/// Reverse of `eurl_encode`: `+` -> space, then `%XX` -> byte.
pub fn eurl_decode(s: &str) -> String {
    let s = s.replace('+', " ");
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            // Use get() rather than slicing: a literal '%' followed by bytes
            // that fall inside a multi-byte UTF-8 char is not a char boundary,
            // and `&s[i+1..i+3]` would panic. get() returns None instead.
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

/// Computes the lowercase hex digest of `data` for a MogileFS checksum
/// algorithm name (`MD5` or `SHA1`, case-insensitive). Returns `None` for an
/// unrecognized algorithm. Shared by `create_close` verification and the fsck
/// worker so the two cannot drift on which algorithms they support.
pub fn compute_checksum_hex(alg: &str, data: &[u8]) -> Option<String> {
    match alg.to_ascii_uppercase().as_str() {
        "MD5" => {
            use md5::{Digest, Md5};
            let mut h = Md5::new();
            h.update(data);
            Some(hex::encode(h.finalize()))
        }
        "SHA1" => {
            use sha1::{Digest, Sha1};
            let mut h = Sha1::new();
            h.update(data);
            Some(hex::encode(h.finalize()))
        }
        _ => None,
    }
}

/// Encode a set of key/value pairs as `k1=v1&k2=v2...` using `eurl_encode`.
pub fn encode_args<'a, I: IntoIterator<Item = (&'a str, &'a str)>>(pairs: I) -> String {
    pairs
        .into_iter()
        .map(|(k, v)| format!("{}={}", eurl_encode(k), eurl_encode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// Decode a `k1=v1&k2=v2` argument string into a map (last value wins on dup keys).
pub fn decode_args(s: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    if s.is_empty() {
        return map;
    }
    for pair in s.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = match pair.split_once('=') {
            Some((k, v)) => (k, v),
            None => (pair, ""),
        };
        map.insert(eurl_decode(k), eurl_decode(v));
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let orig = "hello world/key:with,chars-and.stuff\\ünïcödé";
        let enc = eurl_encode(orig);
        assert_eq!(eurl_decode(&enc), orig);
    }

    #[test]
    fn space_becomes_plus() {
        assert_eq!(eurl_encode("a b"), "a+b");
        assert_eq!(eurl_decode("a+b"), "a b");
    }

    #[test]
    fn decode_percent_before_multibyte_does_not_panic() {
        // A literal '%' followed by a multi-byte UTF-8 char must not panic on
        // a non-char-boundary slice; the '%' is passed through unchanged.
        assert_eq!(eurl_decode("%\u{20ac}x"), "%\u{20ac}x");
        assert_eq!(eurl_decode("%"), "%");
        assert_eq!(eurl_decode("%4"), "%4");
        // A well-formed escape still decodes.
        assert_eq!(eurl_decode("%41"), "A");
    }

    #[test]
    fn args_roundtrip() {
        let mut m = BTreeMap::new();
        m.insert("domain".to_string(), "prod".to_string());
        m.insert("key".to_string(), "my file.jpg".to_string());
        let encoded = encode_args(m.iter().map(|(k, v)| (k.as_str(), v.as_str())));
        let decoded = decode_args(&encoded);
        assert_eq!(decoded, m);
    }
}
