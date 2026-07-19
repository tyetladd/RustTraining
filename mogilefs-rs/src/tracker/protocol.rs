use super::commands;
use super::AppState;
use crate::error::MogError;
use crate::util::{decode_args, encode_args};
use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

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

pub async fn handle_connection(sock: TcpStream, state: Arc<AppState>) -> Result<()> {
    sock.set_nodelay(true).ok();
    let (read_half, mut write_half) = sock.into_split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            return Ok(()); // client closed connection
        }
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
