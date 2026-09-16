"""Running external toolchains (so-vits-svc-fork, Applio) as subprocesses.

Both trainers are command line tools, not libraries, so the integration is a
process driver: build the argument list, stream the output into the log, and
turn a non-zero exit into an error that still shows what the tool printed.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from voice_clone_tts.errors import VoiceCloneError

log = logging.getLogger(__name__)

TAIL_LINES = 80
POLL_SECONDS = 0.5
GRACE_SECONDS = 10.0


class ToolchainError(VoiceCloneError):
    """An external voice-conversion tool failed or is missing."""


@dataclass
class CommandResult:
    """Outcome of one external command."""

    args: list[str]
    returncode: int
    tail: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def output(self) -> str:
        return "\n".join(self.tail)


def find_executable(name: str) -> str | None:
    """Locate a console script, preferring the one next to this interpreter."""
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    if os.name == "nt":
        windows = candidate.with_suffix(".exe")
        if windows.exists():
            return str(windows)
    return shutil.which(name)


def _terminate_tree(process: subprocess.Popen) -> None:
    """Stop the process and everything it spawned.

    Trainers launch their own worker processes (Applio runs ``core.py``, which
    runs ``train.py``), so killing only the direct child would leave the real
    job running on the GPU. The child is started in its own process group,
    which makes the whole tree addressable.
    """
    if process.poll() is not None:
        return
    try:
        group = os.getpgid(process.pid)
    except (AttributeError, OSError, ProcessLookupError):
        group = None

    def signal_all(sig: int) -> None:
        if group is not None:
            try:
                os.killpg(group, sig)
                return
            except (OSError, ProcessLookupError):
                pass
        try:
            process.send_signal(sig)
        except (OSError, ProcessLookupError, ValueError):
            pass

    signal_all(getattr(signal, "SIGTERM", signal.SIGINT))
    try:
        process.wait(timeout=GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        log.warning("process did not stop on SIGTERM, killing it")
    signal_all(getattr(signal, "SIGKILL", signal.SIGTERM))
    try:
        process.wait(timeout=GRACE_SECONDS)
    except subprocess.TimeoutExpired:  # pragma: no cover - should not happen
        log.error("process %s survived SIGKILL", process.pid)


def _stream_reader(stream, sink: "queue.Queue[str | None]") -> None:
    """Split the output on newlines only, keeping ``\r`` inside a line.

    Text mode would translate every ``\r`` into a line break, turning one
    progress bar into thousands of log lines. Reading bytes and splitting on
    ``\n`` keeps a redrawn bar as a single line, of which the caller then
    logs only the final state.
    """
    buffer = b""
    try:
        while True:
            read1 = getattr(stream, "read1", None)
            chunk = read1(65536) if read1 else os.read(stream.fileno(), 65536)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, _, buffer = buffer.partition(b"\n")
                sink.put(line.decode("utf-8", "replace"))
        if buffer:
            sink.put(buffer.decode("utf-8", "replace"))
    except (OSError, ValueError):  # поток закрыли, пока мы читали
        pass
    finally:
        sink.put(None)


def run_command(
    args: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    label: str = "",
    check: bool = True,
    failure_patterns: Sequence[str] = (),
    stream_level: int = logging.INFO,
) -> CommandResult:
    """Run `args`, streaming its output to the log.

    `timeout` is a wall-clock budget for the whole command, enforced while the
    output is being read — cloud sessions end on a hard limit, so a training
    run has to be stopped in time for its checkpoints to be saved.

    `failure_patterns` catches tools that report failure in their output while
    still exiting 0: Applio's ``core.py`` swallows the exit code of the script
    it runs and merely prints "Training failed for model …", which would
    otherwise look like success.

    Output is logged at `stream_level` (INFO by default) because for a training
    run this stream *is* the progress report — and, when something breaks, the
    only place the real error appears.
    """
    args = [str(arg) for arg in args]
    label = label or Path(args[0]).name
    log.info("running %s: %s", label, " ".join(args))

    merged_env = {**os.environ, **(env or {})}
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    reported_failure: str | None = None
    try:
        process = subprocess.Popen(
            args,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group, so the tree can be killed
        )
    except FileNotFoundError as exc:
        raise ToolchainError(f"{label}: command not found ({args[0]})") from exc

    assert process.stdout is not None
    lines: "queue.Queue[str | None]" = queue.Queue()
    reader = threading.Thread(target=_stream_reader, args=(process.stdout, lines), daemon=True)
    reader.start()

    deadline = None if timeout is None else time.monotonic() + timeout
    finished = False
    try:
        while not finished:
            try:
                line = lines.get(timeout=POLL_SECONDS)
            except queue.Empty:
                line = ""
            else:
                if line is None:
                    finished = True
                    line = ""
            # Progress bars rewrite one line with \r; keep only its final state.
            stripped = line.rpartition("\r")[2].rstrip()
            if stripped:
                tail.append(stripped)
                log.log(stream_level, "[%s] %s", label, stripped)
                if reported_failure is None:
                    lowered = stripped.lower()
                    for pattern in failure_patterns:
                        if pattern.lower() in lowered:
                            reported_failure = stripped
                            break
            if deadline is not None and time.monotonic() > deadline and not finished:
                log.warning("%s: time budget of %.0fs is up, stopping it", label, timeout)
                _terminate_tree(process)
                raise ToolchainError(
                    f"{label}: stopped after the {timeout:.0f}s time budget.\n"
                    f"Last output:\n" + "\n".join(tail)
                )
        process.wait(timeout=GRACE_SECONDS)
    finally:
        reader.join(timeout=GRACE_SECONDS)
        try:
            process.stdout.close()
        except Exception:  # noqa: BLE001 - already closed
            pass
        if process.poll() is None:
            _terminate_tree(process)

    result = CommandResult(args=args, returncode=process.returncode, tail=list(tail))
    if check and result.ok and reported_failure is not None:
        raise ToolchainError(
            f"{label} reported a failure while still exiting 0:\n"
            f"  {reported_failure}\n"
            f"Last output:\n{result.output()}"
        )
    if check and not result.ok:
        raise ToolchainError(
            f"{label} exited with code {result.returncode}.\n"
            f"Last output:\n{result.output()}"
        )
    return result


def describe_commands(commands: Iterable[Sequence[str]]) -> str:
    """Render a command sequence the way a user would type it."""
    return "\n".join("  " + " ".join(str(part) for part in command) for command in commands)
