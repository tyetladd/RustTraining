//! Device-selection logic approximating `MogileFS::ReplicationPolicy::MultipleHosts`:
//! prefer spreading replicas across distinct hosts, weighted by free space and
//! the admin-assigned per-device `weight`.

use crate::db::model::{Device, DeviceStatus, HostStatus};
use crate::db::queries;
use crate::db::Conn;
use crate::error::MogError;
use anyhow::Result;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Candidate {
    pub devid: i64,
    pub hostid: i64,
    pub hostip: String,
    pub http_port: i64,
    pub weight: i64,
    pub percent_free: f64,
}

fn all_writable_candidates(conn: &Conn) -> Result<Vec<Candidate>> {
    let devices: Vec<Device> = queries::list_devices(conn)?;
    let mut out = Vec::new();
    for d in devices {
        let Some(status) = DeviceStatus::parse(&d.status) else { continue };
        if !status.writeable() {
            continue;
        }
        let Some(host) = queries::get_host(conn, d.hostid)? else { continue };
        let Some(hstatus) = HostStatus::parse(&host.status) else { continue };
        if hstatus != HostStatus::Alive {
            continue;
        }
        let Some(hostip) = host.hostip.clone() else { continue };
        let percent_free = match (d.mb_total, d.mb_used) {
            (Some(total), Some(used)) if total > 0 => (total - used).max(0) as f64 / total as f64,
            _ => 1.0, // unknown usage yet (e.g. before first monitor pass): assume free
        };
        out.push(Candidate {
            devid: d.devid,
            hostid: d.hostid,
            hostip,
            http_port: host.http_port,
            weight: d.weight.max(1),
            percent_free,
        });
    }
    Ok(out)
}

pub fn select_devices(conn: &Conn, want: usize, exclude_devids: &[i64]) -> Result<Vec<Candidate>, MogError> {
    let db_err = |e: anyhow::Error| MogError::db_msg(e.to_string());
    let exclude: HashSet<i64> = exclude_devids.iter().copied().collect();
    let mut pool: Vec<Candidate> = all_writable_candidates(conn)
        .map_err(db_err)?
        .into_iter()
        .filter(|c| !exclude.contains(&c.devid))
        .collect();

    if pool.is_empty() {
        return Err(MogError::no_devices());
    }

    // top 20 by free space
    pool.sort_by(|a, b| b.percent_free.partial_cmp(&a.percent_free).unwrap());
    pool.truncate(20);

    let mut rng = thread_rng();
    let mut chosen: Vec<Candidate> = Vec::new();
    let mut used_hosts: HashSet<i64> = HashSet::new();

    // Pass 1: at most one device per host, weighted by `weight`.
    let distinct_host_pool: Vec<&Candidate> = {
        let mut seen = HashSet::new();
        pool.iter().filter(|c| seen.insert(c.hostid)).collect()
    };
    if let Ok(picks) = distinct_host_pool.choose_multiple_weighted(&mut rng, want.min(distinct_host_pool.len()), |c| {
        c.weight as f64
    }) {
        for c in picks {
            chosen.push((*c).clone());
            used_hosts.insert(c.hostid);
        }
    }

    // Pass 2: fill remaining slots even if it means reusing a host.
    if chosen.len() < want {
        let remaining: Vec<&Candidate> = pool
            .iter()
            .filter(|c| !chosen.iter().any(|x| x.devid == c.devid))
            .collect();
        let need = want - chosen.len();
        if let Ok(picks) = remaining.choose_multiple_weighted(&mut rng, need.min(remaining.len()), |c| c.weight as f64) {
            for c in picks {
                chosen.push((*c).clone());
            }
        }
    }

    if chosen.is_empty() {
        return Err(MogError::no_devices());
    }
    Ok(chosen)
}

pub fn resolve_class_mindevcount(conn: &Conn, dmid: i64, classid: i64, default_min: i64) -> Result<i64, MogError> {
    let db_err = |e: anyhow::Error| MogError::db_msg(e.to_string());
    if classid == 0 {
        return Ok(default_min);
    }
    match queries::get_class(conn, dmid, classid).map_err(db_err)? {
        Some(c) => Ok(c.mindevcount),
        None => Ok(default_min),
    }
}

pub fn class_name(conn: &Conn, dmid: i64, classid: i64) -> Result<String, MogError> {
    let db_err = |e: anyhow::Error| MogError::db_msg(e.to_string());
    if classid == 0 {
        return Ok("default".to_string());
    }
    match queries::get_class(conn, dmid, classid).map_err(db_err)? {
        Some(c) => Ok(c.classname),
        None => Ok("default".to_string()),
    }
}
