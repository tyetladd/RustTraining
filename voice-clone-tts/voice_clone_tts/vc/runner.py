"""Running external toolchains (so-vits-svc-fork, Applio) as subprocesses.

Both trainers are command line tools, not libraries, so the integration is a
process driver: build the argument list, stream the output into the log, and
turn a non-zero exit into an error that still shows what the tool printed.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from voice_clone_tts.errors import VoiceCloneError

log = logging.getLogger(__name__)

TAIL_LINES = 40


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


def run_command(
    args: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    label: str = "",
    check: bool = True,
) -> CommandResult:
    """Run `args`, streaming its output to the log at DEBUG level."""
    args = [str(arg) for arg in args]
    label = label or Path(args[0]).name
    log.info("running %s: %s", label, " ".join(args))

    merged_env = {**os.environ, **(env or {})}
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    try:
        process = subprocess.Popen(
            args,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise ToolchainError(f"{label}: command not found ({args[0]})") from exc

    assert process.stdout is not None
    try:
        for line in process.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
                log.debug("[%s] %s", label, line)
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        raise ToolchainError(f"{label}: timed out after {timeout:.0f}s") from None
    finally:
        process.stdout.close()

    result = CommandResult(args=args, returncode=process.returncode, tail=list(tail))
    if check and not result.ok:
        raise ToolchainError(
            f"{label} exited with code {result.returncode}.\n"
            f"Last output:\n{result.output()}"
        )
    return result


def describe_commands(commands: Iterable[Sequence[str]]) -> str:
    """Render a command sequence the way a user would type it."""
    return "\n".join("  " + " ".join(str(part) for part in command) for command in commands)
