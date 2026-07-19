use std::fmt;

/// A tracker-protocol error: an `ERR <code> <description>` line.
/// Codes and default messages mirror `MogileFS::Worker::Query::err_line`'s
/// `%errors` table so that `MogileFS::Backend`'s `errcode`/`errstr` behave
/// identically for unmodified clients.
#[derive(Debug, Clone)]
pub struct MogError {
    pub code: &'static str,
    pub description: String,
}

impl MogError {
    pub fn new(code: &'static str, description: impl Into<String>) -> Self {
        Self {
            code,
            description: description.into(),
        }
    }
}

impl fmt::Display for MogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.description)
    }
}

impl std::error::Error for MogError {}

pub type MogResult<T> = Result<T, MogError>;

macro_rules! mogerr_ctor {
    ($name:ident, $code:literal, $default:literal) => {
        pub fn $name() -> MogError {
            MogError::new($code, $default)
        }
    };
}

macro_rules! mogerr_ctor_msg {
    ($name:ident, $code:literal) => {
        pub fn $name(desc: impl Into<String>) -> MogError {
            MogError::new($code, desc.into())
        }
    };
}

impl MogError {
    // --- verbatim from Query.pm's %errors default table ---
    mogerr_ctor!(dup, "dup", "Duplicate name/number used.");
    mogerr_ctor!(after_mismatch, "after_mismatch", "Pattern does not match the after-value?");
    mogerr_ctor!(bad_params, "bad_params", "Invalid parameters to command; please see documentation");
    mogerr_ctor!(class_exists, "class_exists", "That class already exists in that domain");
    mogerr_ctor!(class_has_files, "class_has_files", "Class still has files, unable to delete");
    mogerr_ctor!(db, "db", "Database error");
    mogerr_ctor!(domain_has_files, "domain_has_files", "Domain still has files, unable to delete");
    mogerr_ctor!(domain_exists, "domain_exists", "That domain already exists");
    mogerr_ctor!(domain_not_found, "domain_not_found", "Domain not found");
    mogerr_ctor!(failure, "failure", "Operation failed");
    mogerr_ctor!(host_mismatch, "host_mismatch", "The device specified doesn't belong to the host specified");
    mogerr_ctor!(host_not_empty, "host_not_empty", "Unable to delete host; it contains devices still");
    mogerr_ctor!(invalid_mindevcount, "invalid_mindevcount", "The mindevcount must be at least 1");
    mogerr_ctor!(key_exists, "key_exists", "Target key name already exists; can't overwrite.");
    mogerr_ctor!(no_devices, "no_devices", "No devices found to store file");
    mogerr_ctor!(no_temp_file, "no_temp_file", "No tempfile or file already closed");
    mogerr_ctor!(none_match, "none_match", "No keys match that pattern and after-value (if any).");
    mogerr_ctor!(plugin_aborted, "plugin_aborted", "Action aborted by plugin");
    mogerr_ctor!(state_too_high, "state_too_high", "Status cannot go from dead to alive; must use down");
    mogerr_ctor!(unknown_command, "unknown_command", "Unknown server command");

    // --- additional codes referenced by individual cmd_* handlers ---
    mogerr_ctor!(no_domain, "no_domain", "No domain provided");
    mogerr_ctor!(unreg_domain, "unreg_domain", "Domain is not registered");
    mogerr_ctor!(no_class, "no_class", "No class provided");
    mogerr_ctor!(unreg_class, "unreg_class", "Class is not registered");
    mogerr_ctor!(class_not_found, "class_not_found", "Class not found");
    mogerr_ctor!(no_key, "no_key", "No key provided");
    mogerr_ctor!(unknown_key, "unknown_key", "Unknown key");
    mogerr_ctor!(no_fid, "no_fid", "No fid provided");
    mogerr_ctor!(unknown_fid, "unknown_fid", "Unknown fid");
    mogerr_ctor!(fid_in_use, "fid_in_use", "The fid specified is already in use");
    mogerr_ctor!(fid_exists, "fid_exists", "That fid already exists");
    mogerr_ctor!(no_devid, "no_devid", "No devid provided");
    mogerr_ctor!(no_device, "no_device", "No device provided");
    mogerr_ctor!(unknown_device, "unknown_device", "Unknown device");
    mogerr_ctor!(device_exists, "device_exists", "That device already exists");
    mogerr_ctor!(invalid_destdev, "invalid_destdev", "The devid specified is not a valid destination for this fid");
    mogerr_ctor!(no_path, "no_path", "No path provided");
    mogerr_ctor!(bogus_args, "bogus_args", "The path given does not match the fid/devid given");
    mogerr_ctor!(unknown_host, "unknown_host", "Unknown host");
    mogerr_ctor!(host_exists, "host_exists", "That host already exists");
    mogerr_ctor!(host_not_found, "host_not_found", "Host not found");
    mogerr_ctor!(no_host, "no_host", "No host provided");
    mogerr_ctor!(no_ip, "no_ip", "IP required to create host");
    mogerr_ctor!(no_port, "no_port", "Port required to create host");
    mogerr_ctor!(unknown_state, "unknown_state", "Unknown state");
    mogerr_ctor!(size_mismatch, "size_mismatch", "Size of file does not match what was expected");
    mogerr_ctor!(size_verify_error, "size_verify_error", "Couldn't verify size of file after upload");
    mogerr_ctor!(empty_file, "empty_file", "File is empty");
    mogerr_ctor!(unable_to_create_tempfile, "unable_to_create_tempfile", "Unable to create tempfile");
    mogerr_ctor!(unable_to_open_file, "unable_to_open_file", "Unable to open file");
    mogerr_ctor!(checksum_mismatch, "checksum_mismatch", "Checksum does not match data");
    mogerr_ctor!(invalid_checksum_format, "invalid_checksum_format", "Checksum format is not recognized");
    mogerr_ctor!(not_writable, "not_writable", "That setting is not writable");
    mogerr_ctor!(internal_error, "internal_error", "Internal server error");

    mogerr_ctor_msg!(bad_params_msg, "bad_params");
    mogerr_ctor_msg!(failure_msg, "failure");
    mogerr_ctor_msg!(db_msg, "db");
}
