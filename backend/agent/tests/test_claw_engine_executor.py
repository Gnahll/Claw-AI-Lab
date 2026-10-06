from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from researchclaw.pipeline.claw_engine.tools.executor import ToolExecutor

pytestmark = pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX only")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed child of an exited shell may linger as a zombie until reaped.
    try:
        with open(f"/proc/{pid}/stat") as handle:
            return handle.read().split()[2] != "Z"
    except FileNotFoundError:
        return True


def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_bash_returns_output_and_exit_code(tmp_path: Path) -> None:
    executor = ToolExecutor(tmp_path)

    output, is_error = executor.execute("bash", {"command": "echo hello; exit 3"})

    assert is_error is False
    assert "hello" in output
    assert "[exit_code: 3]" in output


def test_cleanup_terminates_background_jobs(tmp_path: Path) -> None:
    executor = ToolExecutor(tmp_path)

    output, _ = executor.execute(
        "bash", {"command": "nohup sleep 300 > /dev/null 2>&1 & echo $!"},
    )
    pid = int(output.strip())
    assert _alive(pid)

    assert executor.cleanup_processes(grace_sec=2) == 1
    assert _wait_dead(pid)


def test_cleanup_is_noop_when_nothing_is_running(tmp_path: Path) -> None:
    executor = ToolExecutor(tmp_path)
    executor.execute("bash", {"command": "true"})

    assert executor.cleanup_processes() == 0


def test_timeout_kills_child_processes(tmp_path: Path) -> None:
    executor = ToolExecutor(tmp_path, bash_timeout=1)

    output, _ = executor.execute(
        "bash", {"command": "sleep 300 & echo $! > child.pid; wait"},
    )

    assert "timed out" in output
    pid = int((tmp_path / "child.pid").read_text().strip())
    assert _wait_dead(pid)
