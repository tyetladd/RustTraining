# mogilefs-rs

A single-binary Rust reimplementation of [MogileFS](https://github.com/mogilefs/MogileFS-Server): the
tracker (metadata server) and storage node run as one process (`mogilefsd`), wire-compatible with the
unmodified Perl client library (`MogileFS::Client` / `MogileFS::Backend`) so existing MogileFS clients can
point at this server without any client-side changes.

Implemented by reviewing the reference Perl tracker (`mogilefs/MogileFS-Server`) and client
(`mogilefs/perl-MogileFS-Client`) source directly — see "Wire compatibility" below for the exact contracts
this reimplementation matches byte-for-byte.

## Running

```
cargo run -p mogilefs-rs -- --config mogilefsd.toml
```

Example `mogilefsd.toml` (all fields optional, defaults shown):

```toml
db_dsn = "sqlite://mogilefs.db"  # or mysql://user:pass@host/db, or postgres://user:pass@host/db
tracker_listen_ip = "0.0.0.0"
tracker_port = 7001              # standard MogileFS tracker port
storage_listen_ip = "0.0.0.0"
storage_port = 7500              # standard mogstored HTTP port
docroot = "./mogdata"            # local dir backing every device's files: docroot/devN/...
default_min_devcount = 2         # replicas for the implicit "default" class
```

(`db_path = "mogilefs.db"` also works as shorthand for `db_dsn = "sqlite://mogilefs.db"` if you don't set
`db_dsn` at all.)

Since the tracker and storage node share one process/filesystem here, register a host pointing back at
this server's own storage port, then a device on that host, then flip both alive, e.g. over the wire
protocol (what `mogadm`/an admin client would send):

```
create_host host=h1 ip=127.0.0.1 port=7500
update_host host=h1 status=alive
create_device host=h1 devid=1
set_state host=h1 device=1 state=alive
```

## Wire compatibility

- **Tracker line protocol** (TCP, default port 7001): `CMD key1=val1&key2=val2\r\n` requests, `OK
  argline\r\n` / `ERR code text\r\n` responses, using the exact `eurl` percent-encoding charset from
  `MogileFS::Util` (safe chars `a-zA-Z0-9_,-./\: `, then literal spaces become `+`). An optional `N-M `
  request-id prefix is accepted and echoed back, matching `Backend.pm`.
- **Storage HTTP protocol** (default port 7500): plain `PUT`/`GET`/`HEAD`/`DELETE` on
  `/dev<N>/<b>/<mmm>/<ttt>/<10-digit-fid>.fid`, computed with the same formula as
  `MogileFS::DevFID::uri_path`. `GET /dev<N>/usage` returns `total:`/`used:` (KB) text lines, as polled by
  the reference tracker's Monitor worker.
- **Error codes**: `ERR` codes and default messages mirror `MogileFS::Worker::Query`'s `%errors` table
  (`dup`, `unreg_domain`, `unknown_key`, `no_devices`, `checksum_mismatch`, etc.) so `$mogc->errstr` /
  `errcode` on the client behave the same as against the real tracker.

## Database backends

Metadata storage is pluggable, selected by the `db_dsn` scheme, mirroring upstream's own
`MogileFS::Store::{MySQL,Postgres,SQLite}` split — each backend gets its own DDL (auto-increment/upsert
syntax differs) behind a common `Db`/`Store` abstraction (`src/db/`), built on `sqlx`:

| Scheme | Backend |
|---|---|
| `sqlite://path/to/file.db` | SQLite (bundled, no external server needed) |
| `mysql://user:pass@host/db` | MySQL / MariaDB |
| `postgres://user:pass@host/db` | PostgreSQL |

All three are exercised by the same integration test suite (see Tests below).

## Commands implemented

Core file lifecycle: `create_open`, `create_close` (including `multi_dest`, size verification, and
MD5/SHA1 `checksum`/`checksumverify`), `get_paths`, `file_info`, `file_debug`-equivalent via `file_info`,
`list_keys`, `list_fids`, `delete`, `rename`, `sleep`, `noop`.

Admin/topology: `create_domain`, `delete_domain`, `create_class`, `update_class`/`updateclass`,
`delete_class`, `create_host`, `update_host`, `delete_host`, `create_device`, `set_state`, `set_weight`,
`get_hosts`, `get_devices`, `get_domains`, `server_setting`/`server_settings`, `set_server_setting`,
`replicate_now`, `clear_cache`.

Maintenance: `httpcopy` (synchronous cross-device blob copy), `edit_file` (experimental
read-modify-write handshake — allocates a fresh tempfile for an existing key and returns both
`oldpath`/`newpath`), `fsck_start`/`fsck_stop`/`fsck_reset`/`fsck_clearlog`/`fsck_getlog`/`fsck_status`,
`rebalance_start`/`rebalance_stop`/`rebalance_status`.

Background workers mirror the reference tracker's Monitor/Replicate/Delete/Fsck/Rebalance workers:

- **Monitor**: periodic disk-usage stats per device, feeding free-space-weighted device selection.
- **Replicate**: async replication of newly-closed files up to each class's `mindevcount`, spread across
  distinct hosts and weighted by free space — approximating `ReplicationPolicy::MultipleHosts`.
- **Delete**: async physical deletion of unlinked/overwritten fids.
- **Fsck** (opt-in via `fsck_start`): walks all `file` rows in fid order, verifies each replica still
  exists on disk and (if a checksum was recorded) still matches it, logs anomalies to `fsck_log`, and
  re-queues under-replicated fids for repair. One full pass then stops itself automatically.
- **Rebalance** (opt-in via `rebalance_start`): seeds a queue from the currently most-utilized device's
  files and moves each one's copy onto a less-utilized device without changing its replica count.

### Simplifications versus a real multi-host deployment

Checksum verification and fsck both read files directly off local disk rather than through the mgmt-port
side-channel protocol real `mogstored` exposes — correct here since tracker and storage share one process,
but not something a genuinely distributed deployment could rely on. `edit_file` is deliberately minimal
(the real tracker's implementation is itself labeled experimental).

## Tests

```
cargo test -p mogilefs-rs
```

`tests/protocol.rs` drives the server through the raw tracker line protocol and raw HTTP (no client
library — the same bytes a real Perl client would send), covering: the full `create_open` → `PUT` →
`create_close` → `get_paths` → `GET` lifecycle plus `httpcopy`, checksum verification (both mismatch and
successful MD5), background replication to `mindevcount`, `fsck` detecting a blob deleted out from under
the tracker, and `rebalance` draining its queue.

These run against SQLite unconditionally. To also exercise them live against MySQL and/or PostgreSQL, point
`MOGILEFS_TEST_MYSQL_DSN` / `MOGILEFS_TEST_POSTGRES_DSN` at a reachable database:

```
export MOGILEFS_TEST_MYSQL_DSN="mysql://mogile:mogilepass@127.0.0.1/mogilefs_test"
export MOGILEFS_TEST_POSTGRES_DSN="postgres://mogile:mogilepass@127.0.0.1/mogilefs_test"
cargo test -p mogilefs-rs
```

(Each test wipes and reuses that database, so it's safe to point at a scratch DB you don't mind being
cleared.) Without those env vars set, the MySQL/Postgres variants skip themselves rather than failing.

### Not implemented (out of scope for this pass)

The mgmt-port checksum side-channel protocol (superseded here by direct filesystem access, see above).
Everything else in the reference tracker's client-facing and administrative command set now has at least a
basic implementation.
