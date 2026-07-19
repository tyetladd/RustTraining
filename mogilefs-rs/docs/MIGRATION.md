# Migrating an existing MogileFS deployment (≈100 TB) to mogilefs-rs

This guide describes how to migrate a production, multi-node Perl-MogileFS
deployment holding on the order of 100 TB to `mogilefs-rs` with **no bulk data
movement** and a short, reversible cutover.

---

## 0. Compatibility summary

What makes a cheap migration possible:

| Aspect | Status | Consequence for migration |
|---|---|---|
| **On-disk blob layout** | **Byte-identical** — `/dev<N>/<b>/<mmm>/<ttt>/<10-digit-fid>.fid`, same formula as `MogileFS::DevFID` | The 100 TB of blobs **never moves**. Migration is a *metadata* migration; the storage nodes' filesystems stay exactly as they are. |
| **Tracker wire protocol** | Compatible (`CMD k=v…\r\n` / `OK`/`ERR`, `eurl` encoding, error codes) | Existing `MogileFS::Client` applications keep working unchanged — you re-point them at new trackers, you do not rewrite them. |
| **Multi-host storage** | **Supported.** All server-side blob I/O routes to the device's owning host over HTTP (`http://host.hostip:host.http_port/dev<N>/…`), exactly as the reference tracker talks to `mogstored`. A process runs any subset of roles (`enable_tracker` / `enable_storage` / `enable_s3` / `enable_workers`), so you deploy storage nodes per location and tracker/S3 nodes centrally, all sharing one `db_dsn`. | You run a real multi-node cluster: register each storage host with its `hostip`/`http_port`, and the tracker/workers reach every device wherever it lives. |
| **Metadata DB schema** | *Similar but not identical* (types adjusted for SQLite/MySQL/Postgres; a few reference-only columns dropped; `checksum.hashtype` is text, not a numeric code) | The metadata cannot be used in place — it needs a one-time **ETL** (§3). Mechanically simple, mostly 1:1. |

### Cluster topology

Deploy the way real MogileFS does — decoupled roles sharing one metadata DB:

- **Storage node (per location):** `enable_storage = true`, everything else
  `false`. Serves its local `docroot` over HTTP for the devices physically on
  that host. Point its `hostip`/`http_port` registration at this process.
- **Tracker / gateway node(s):** `enable_tracker`, `enable_workers`, and
  optionally `enable_s3`. Holds no data; routes blob operations to the storage
  nodes over HTTP and coordinates replication/fsck/rebalance.
- All nodes set the **same `db_dsn`** (use MySQL or PostgreSQL for a 100 TB
  fleet — see §1).

Run several trackers for availability; they are stateless beyond the shared DB.

---

## 1. Assessment & capacity planning

Inventory the source deployment before touching anything:

```sql
-- against the existing MogileFS MySQL metadata DB
SELECT COUNT(*) AS files, MAX(fid) AS max_fid, SUM(length) AS bytes FROM file;
SELECT COUNT(*) FROM file_on;                     -- physical replicas
SELECT dmid, namespace FROM domain;
SELECT dmid, classid, classname, mindevcount, hashtype FROM class;
SELECT hostid, hostname, hostip, http_port, status FROM host;
SELECT devid, hostid, status, weight, mb_total, mb_used FROM device;
SELECT COUNT(*) FROM checksum;
```

Record: total `file` rows, `MAX(fid)` (**critical**, see §3.3), `file_on` rows,
the full host/device topology, and the class/replication policy.

**Metadata DB sizing.** The metadata size is driven by object *count*, not
bytes. Rough model:

```
file rows        ≈ number of live objects
file_on rows     ≈ objects × avg replica count (mindevcount, usually 2–3)
DB size          ≈ (file rows × ~120 B) + (file_on rows × ~40 B) + checksum rows
```

| Avg object size | Objects in 100 TB | Approx. metadata DB |
|---|---|---|
| 1 MB | ~100 M | ~20–40 GB |
| 256 KB | ~400 M | ~80–150 GB |
| 64 KB | ~1.6 B | multi-hundred GB — shard/tune, not SQLite |

**Pick the backend accordingly:** SQLite is fine for a lab or a small
single-host instance; **use MySQL or PostgreSQL for a 100 TB fleet.** Provision
the metadata DB for the row counts above plus headroom, and plan tracker count
for query throughput (each tracker holds one modest connection pool).

---

## 2. Migration strategy

Because blobs never move, the recommended strategy is **in-place, metadata-only
cutover**:

1. Deploy a `mogilefs-rs` **storage node on each existing storage host**
   (`enable_storage` only), pointing its `docroot` at the host's existing
   MogileFS data root. Because the layout is identical, it serves the files
   already there.
2. Stand up new `mogilefs-rs` **tracker nodes** alongside the existing ones,
   pointed at a **new** metadata DB populated by ETL from the old one (§3),
   preserving `fid`, `devid`, and `hostid` values exactly (so the identical path
   formula still resolves the same physical files).
3. Register the **same** hosts/devices with their real `hostip`/`http_port` (the
   new storage nodes), so the new trackers reach the **same** filesystems.
4. Verify (§4), then cut clients over (§5). Data is shared read-only during the
   overlap, so rollback is "point clients back."

Two strategies to avoid for 100 TB:

- **Full re-replication** (let the new cluster pull every object fresh) — moves
  100 TB across the network for no benefit, since the layout is already correct.
- **Reusing the old DB in place** — schemas differ; always ETL into a fresh DB
  so the source stays untouched and rollback-safe.

---

## 3. Metadata ETL (old MogileFS DB → mogilefs-rs DB)

Do this against a **snapshot/replica** of the source metadata DB, writing into a
fresh `mogilefs-rs` database that has been through `migrate()` (so the schema and
`s3_*` tables exist). The core tables map almost 1:1.

### 3.1 Table / column mapping

| Source (Perl MogileFS) | → mogilefs-rs | Transform |
|---|---|---|
| `domain(dmid, namespace)` | `domain` | copy verbatim |
| `class(dmid, classid, classname, mindevcount, replpolicy, hashtype)` | `class` | copy; keep `mindevcount`; `hashtype` numeric code → text (`1`→`MD5`, `2`→`SHA1`) or leave null |
| `host(hostid, hostname, hostip, http_port, status)` | `host` | copy these 5 columns; `http_get_port`/`altip`/`altmask` are not modeled — drop |
| `device(devid, hostid, status, weight, mb_total, mb_used, mb_asof)` | `device` | copy verbatim (**preserve `devid`/`hostid`**) |
| `file(fid, dmid, dkey, length, classid, devcount)` | `file` | copy verbatim (**preserve `fid`**) |
| `file_on(fid, devid)` | `file_on` | copy verbatim |
| `checksum(fid, hashtype, checksum)` | `checksum` | `hashtype` numeric → text; `checksum` binary → lowercase hex |
| `server_settings(field, value)` | `server_settings` | copy (optional) |
| `tempfile`, `file_to_delete`, `file_to_replicate`, `file_to_queue`, `unreachable_fids`, `fsck_log` | — | **do not migrate**; start empty. In-flight uploads (`tempfile`) should be drained before cutover, not carried over. |

Everything that determines *where a blob lives* (`fid`, `devid`, `hostid`, and
thus the path) is carried across unchanged — that is what lets the files stay put.

### 3.2 Doing the copy

For MySQL→MySQL the bulk tables (`file`, `file_on`) are best moved with
`SELECT … INTO OUTFILE` / `LOAD DATA INFILE` (or `mysqldump --no-create-info` of
just those tables) rather than row-by-row. The small tables (`domain`, `class`,
`host`, `device`) can be inserted directly. For cross-engine moves
(MySQL→Postgres), export to CSV and `COPY`/`LOAD` per table.

The `s3_bucket` / `s3_object` tables stay **empty** unless you are also adopting
the S3 gateway; existing MogileFS keys are served over the wire protocol without
any S3 metadata.

### 3.3 ⚠️ Advance the fid sequence (do not skip)

New uploads allocate a `fid` from `tempfile`'s auto-increment / sequence. After
loading historical rows it **must** start above the old `MAX(fid)`, or the first
new upload will collide with an existing object.

```sql
-- SQLite:   run a throwaway insert then reset, or set the sqlite_sequence row:
UPDATE sqlite_sequence SET seq = <MAX_FID> WHERE name = 'tempfile';

-- MySQL:
ALTER TABLE tempfile AUTO_INCREMENT = <MAX_FID + 1>;

-- PostgreSQL:
SELECT setval(pg_get_serial_sequence('tempfile','fid'), <MAX_FID>);
```

Verify with a probe `create_open`/`create_close` in staging that the returned
`fid` is greater than `MAX_FID`.

---

## 4. Verification (before any client touches it)

Run these against the new trackers pointed at the migrated DB and the real
storage hosts:

1. **Row reconciliation** — `file`, `file_on`, `domain`, `class`, `device`
   counts match the source snapshot.
2. **Path resolution** — for a random sample of keys per domain, `get_paths`
   returns URLs on the expected hosts, and an HTTP `GET`/`HEAD` to that URL
   returns the object with the right length.
3. **Checksum spot-check** — for keys that have `checksum` rows, fetch and
   confirm the digest matches (the gateway/tracker store the same MD5).
4. **fsck a subset** — point `fsck` at one domain and confirm it reports no
   `missing` / `under_replicated` events for data that is actually intact
   (validates that multi-host routing reaches every device).
5. **Write probe** — a full `create_open → PUT → create_close → get_paths → GET`
   cycle into a scratch domain, then delete it.

Do all of this on a **rehearsal** run against a single low-traffic domain first;
only widen to the full fleet once it is clean.

---

## 5. Cutover

Client applications speak the same protocol, so cutover is a matter of where
they connect. Minimize the write-inconsistency window:

1. **Freeze writes** briefly (or put the app in read-only mode). Reads can keep
   flowing to the old trackers.
2. **Final metadata delta** — re-run the ETL for rows changed since the initial
   snapshot (by `fid` range above the snapshot's max, plus any deletes). Because
   `fid` is monotonic, "new since snapshot" is a simple `fid > snapshot_max`
   range; reconcile deletes from the old `file_to_delete`/tombstones.
3. **Re-advance the fid sequence** (§3.3) to the new `MAX(fid)`.
4. **Flip clients** to the new trackers (update the tracker host list in the
   client config / load balancer). Existing sockets drain; new requests land on
   `mogilefs-rs`.
5. **Unfreeze writes.**
6. **Watch**: error rates, `get_paths` latency, replicate/fsck queue depth, DB
   connections, storage-host HTTP 5xx.

Keep the old trackers running but idle for the rollback window.

---

## 6. Rollback

Rollback is cheap precisely because **no data moved**:

- **Before cutover:** discard the new DB and trackers; the source is untouched.
- **After cutover, within the rollback window:** point clients back at the old
  trackers. The only data written to `mogilefs-rs`-only metadata in the interim
  are objects created after the flip — reconcile those back into the old DB (same
  ETL in reverse over the post-cutover `fid` range) before fully retiring
  `mogilefs-rs`, or accept the brief overlap window's writes as forward-only.

Only decommission the old trackers and metadata DB after the new stack has run
clean through at least one full backup/fsck cycle.

---

## 7. Risk register / known differences

| Risk / difference | Mitigation |
|---|---|
| **S3 single-PUT is buffered** (capped by `s3_max_single_put`, default 256 MiB) — multipart upload is a later phase | For large-object S3 workloads, wait for multipart, or raise the cap knowingly. The MogileFS wire path (what the migration uses) streams and is unaffected. |
| **Replication policy** approximates `MultipleHosts` (free-space + distinct-host weighting), not byte-identical device selection | Behavior converges to the same `mindevcount` guarantee; verify replica spread with `fsck` post-migration. |
| **Checksum side-channel** — reference `mogstored` exposes a mgmt-port digest protocol; `mogilefs-rs` computes checksums by reading the blob | Functionally equivalent for verification; no client-visible difference. |
| **`checksum.hashtype`** representation differs (numeric → text) | Handle in ETL (§3.1); confirm the mapping against your MogileFS version's hashtype codes. |
| **Dropped reference-only columns** (`http_get_port`, `altip`, `altmask`, `fixedkey`, …) | Not used by clients; safe to drop. If you rely on `altip`/`altmask` alternate-network routing, that is not yet modeled — evaluate before migrating. |
| **In-flight uploads** in `tempfile` at snapshot time | Drain (quiesce writers) before the final delta; do not migrate `tempfile`. |

---

## 8. Migration checklist (condensed)

- [ ] Storage nodes deployed per location (`enable_storage`), tracker/gateway nodes deployed, all sharing one `db_dsn`
- [ ] Source inventory captured: counts, `MAX(fid)`, topology, classes
- [ ] Metadata DB backend chosen (MySQL/Postgres for 100 TB) and provisioned
- [ ] Fresh `mogilefs-rs` DB created and `migrate()`d
- [ ] ETL of `domain`/`class`/`host`/`device`/`file`/`file_on`/`checksum` (ids preserved)
- [ ] **fid sequence advanced past `MAX(fid)`**
- [ ] New trackers stood up, storage hosts/devices registered `alive`
- [ ] Verification passed (reconcile, path resolution, checksums, fsck, write probe)
- [ ] Rehearsed on one low-traffic domain end-to-end
- [ ] Cutover runbook + rollback window agreed; monitoring dashboards ready
- [ ] Write-freeze → final delta → re-advance fid → flip clients → unfreeze
- [ ] Old stack kept idle for rollback; decommission only after a clean cycle
