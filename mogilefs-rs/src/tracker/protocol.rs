use super::commands;
use super::AppState;
use crate::error::MogError;
use crate::util::{decode_args, encode_args};
use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

/// Maximum length (bytes) of a single tracker request line. Keys can be long
/// but not unbounded; without this cap a client that never sends `\n` grows the
/// line buffer until the process runs out of memory.
const MAX_LINE_BYTES: usize = 1024 * 1024;

pub type Args = BTreeMap<String, String>;
pub type Reply = BTreeMap<String, String>;

/// Parses a request line of the form `[reqid ]CMD [args]`, mirroring the
/// tracker's `process_line` regex `^(\d+-\d+)?\s*(\S+)\s*(.*)`.
fn parse_request(line: &str) -> Option<(Option<&str>, &str, &str)> {
    let line = line.trim_end_matches(['\r', '\n']);
    let line = line.trim_start();
    if line.is_empty() {
        return None;
    }
    let (first, rest) = match line.split_once(char::is_whitespace) {
        Some((a, b)) => (a, b.trim_start()),
        None => (line, ""),
    };

    // Optional "N-M" reqid prefix before the actual command.
    if first.contains('-') && first.chars().all(|c| c.is_ascii_digit() || c == '-') {
        let (cmd, argstr) = match rest.split_once(char::is_whitespace) {
            Some((a, b)) => (a, b.trim_start()),
            None => (rest, ""),
        };
        if !cmd.is_empty() {
            return Some((Some(first), cmd, argstr));
        }
    }
    Some((None, first, rest))
}

fn ok_line(reqid: Option<&str>, reply: &Reply) -> String {
    let argline = encode_args(reply.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    match reqid {
        Some(id) => format!("{id} OK {argline}\r\n"),
        None => format!("OK {argline}\r\n"),
    }
}

fn err_line(reqid: Option<&str>, err: &MogError) -> String {
    let text = crate::util::eurl_encode(&err.description);
    match reqid {
        Some(id) => format!("{id} ERR {} {text}\r\n", err.code),
        None => format!("ERR {} {text}\r\n", err.code),
    }
}

/// Reads one `\n`-terminated line into `buf`, but errors out once more than
/// `cap` bytes have accumulated without a newline, so a client cannot force
/// unbounded buffer growth. Returns the number of bytes read (0 at EOF).
async fn read_line_capped<R>(reader: &mut R, buf: &mut Vec<u8>, cap: usize) -> Result<usize>
where
    R: AsyncBufReadExt + Unpin,
{
    let mut total = 0;
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok(total); // EOF
        }
        if let Some(pos) = available.iter().position(|&b| b == b'\n') {
            buf.extend_from_slice(&available[..=pos]);
            let consumed = pos + 1;
            reader.consume(consumed);
            return Ok(total + consumed);
        }
        buf.extend_from_slice(available);
        let consumed = available.len();
        reader.consume(consumed);
        total += consumed;
        if total > cap {
            anyhow::bail!("request line exceeded {cap} bytes without a newline");
        }
    }
}

pub async fn handle_connection(sock: TcpStream, state: Arc<AppState>) -> Result<()> {
    sock.set_nodelay(true).ok();
    let (read_half, mut write_half) = sock.into_split();
    let mut reader = BufReader::new(read_half);
    let mut buf = Vec::new();

    loop {
        buf.clear();
        let n = read_line_capped(&mut reader, &mut buf, MAX_LINE_BYTES).await?;
        if n == 0 {
            return Ok(()); // client closed connection
        }
        let line = String::from_utf8_lossy(&buf);
        let Some((reqid, cmd, argstr)) = parse_request(&line) else {
            continue;
        };
        let cmd_lc = cmd.to_ascii_lowercase();
        let args = decode_args(argstr);

        let out = match commands::dispatch(&state, &cmd_lc, &args).await {
            Ok(reply) => ok_line(reqid, &reply),
            Err(e) => err_line(reqid, &e),
        };
        write_half.write_all(out.as_bytes()).await?;
    }
}
