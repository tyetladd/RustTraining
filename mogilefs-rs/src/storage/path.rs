use std::path::{Path, PathBuf};

/// Mirrors `MogileFS::DevFID::uri_path`:
/// ```perl
/// my $nfid = sprintf('%010d', $fid);
/// my ($b, $mmm, $ttt) = ($nfid =~ /(\d)(\d{3})(\d{3})(\d{3})/);
/// return "/dev$devid/$b/$mmm/$ttt/$nfid.fid";
/// ```
pub fn uri_path(devid: i64, fid: i64) -> String {
    let nfid = format!("{:010}", fid);
    let b = &nfid[0..1];
    let mmm = &nfid[1..4];
    let ttt = &nfid[4..7];
    format!("/dev{devid}/{b}/{mmm}/{ttt}/{nfid}.fid")
}

/// Filesystem location under `docroot` for a given device's directory
/// (used both for storing fid blobs and for reporting per-device disk usage).
pub fn device_root(docroot: &str, devid: i64) -> PathBuf {
    Path::new(docroot).join(format!("dev{devid}"))
}

/// Filesystem path backing a given (devid, fid) blob.
pub fn fs_path(docroot: &str, devid: i64, fid: i64) -> PathBuf {
    let uri = uri_path(devid, fid);
    Path::new(docroot).join(uri.trim_start_matches('/'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_formula() {
        assert_eq!(uri_path(3, 123), "/dev3/0/000/000/0000000123.fid");
    }
}
