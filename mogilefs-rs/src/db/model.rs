#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Domain {
    pub dmid: i64,
    pub namespace: String,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Class {
    pub dmid: i64,
    pub classid: i64,
    pub classname: String,
    pub mindevcount: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostStatus {
    Alive,
    Dead,
    Down,
}

impl HostStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            HostStatus::Alive => "alive",
            HostStatus::Dead => "dead",
            HostStatus::Down => "down",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "alive" => Some(HostStatus::Alive),
            "dead" => Some(HostStatus::Dead),
            "down" => Some(HostStatus::Down),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Host {
    pub hostid: i64,
    pub hostname: String,
    pub hostip: Option<String>,
    pub status: String,
    pub http_port: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceStatus {
    Alive,
    Dead,
    Down,
    Drain,
    ReadOnly,
}

impl DeviceStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceStatus::Alive => "alive",
            DeviceStatus::Dead => "dead",
            DeviceStatus::Down => "down",
            DeviceStatus::Drain => "drain",
            DeviceStatus::ReadOnly => "readonly",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "alive" => Some(DeviceStatus::Alive),
            "dead" => Some(DeviceStatus::Dead),
            "down" => Some(DeviceStatus::Down),
            "drain" => Some(DeviceStatus::Drain),
            "readonly" => Some(DeviceStatus::ReadOnly),
            _ => None,
        }
    }
    /// Devices we may write new file copies to.
    pub fn writeable(&self) -> bool {
        matches!(self, DeviceStatus::Alive)
    }
    /// Devices we may still read existing copies from.
    pub fn readable(&self) -> bool {
        matches!(self, DeviceStatus::Alive | DeviceStatus::Drain | DeviceStatus::ReadOnly)
    }
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Device {
    pub devid: i64,
    pub hostid: i64,
    pub status: String,
    pub weight: i64,
    pub mb_total: Option<i64>,
    pub mb_used: Option<i64>,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct FileRow {
    pub fid: i64,
    pub dmid: i64,
    pub dkey: String,
    pub length: Option<i64>,
    pub classid: i64,
    pub devcount: i64,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct TempfileRow {
    pub fid: i64,
    pub dmid: i64,
    pub dkey: Option<String>,
    pub classid: i64,
    pub devids: String,
}

impl TempfileRow {
    pub fn devid_list(&self) -> Vec<i64> {
        self.devids
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    }
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct FsckLogEntry {
    pub logid: i64,
    pub utime: i64,
    pub fid: i64,
    pub evcode: String,
    pub devid: Option<i64>,
}

/// A row in the generic `file_to_queue` table (used for rebalance jobs today;
/// mirrors the reference tracker's unified queue for maintenance workers).
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct QueueEntry {
    pub fid: i64,
    pub devid: Option<i64>,
    pub r#type: String,
    pub failcount: i64,
    pub arg: Option<String>,
}

/// S3 gateway per-object metadata that MogileFS itself doesn't model
/// (Content-Type, ETag, last-modified, user metadata). The object bytes live in
/// the `file`/`file_on` tables keyed by the same (dmid, dkey).
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct S3Object {
    pub dmid: i64,
    pub dkey: String,
    pub content_type: Option<String>,
    pub etag: Option<String>,
    pub size: i64,
    pub mtime: i64,
    pub user_meta: Option<String>,
}

/// An S3 bucket, i.e. a MogileFS domain plus its S3 creation timestamp.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct S3Bucket {
    pub namespace: String,
    pub created: i64,
}
