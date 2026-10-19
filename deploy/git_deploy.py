"""Fast-forward a clean server checkout with backup, validation and rollback."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time


TERMINAL = {"done", "completed", "failed", "cancelled", "canceled"}


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def service_record(root: Path, name: str) -> dict:
    pid = int((root / ".pids" / f"{name}.pid").read_text())
    token = identity(pid)
    if token is None:
        raise RuntimeError(f"Service is not running: {name}")
    process = Path("/proc") / str(pid)
    arguments = [os.fsdecode(value) for value in (process / "cmdline").read_bytes().split(b"\0") if value]
    if name == "agent_bridge" and str(root / "backend/services/agent_bridge.py") not in arguments:
        raise RuntimeError("Bridge PID does not belong to this checkout")
    environment = {}
    for item in (process / "environ").read_bytes().split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            environment[os.fsdecode(key)] = os.fsdecode(value)
    return {"pid": pid, "identity": token, "arguments": arguments,
            "cwd": str((process / "cwd").resolve()), "environment": environment}


def assert_idle(root: Path, bridge_pid: int) -> None:
    for path in (root / "backend/runs/queues").glob("*.json"):
        payload = json.loads(path.read_text())
        tasks = payload if isinstance(payload, list) else payload["tasks"]
        if any(task.get("status") not in TERMINAL for task in tasks):
            raise RuntimeError(f"Queue is not idle: {path.name}")
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            parent = next(line for line in (process / "status").read_text().splitlines() if line.startswith("PPid:")).split()[1]
            arguments = (process / "cmdline").read_bytes().split(b"\0")
            cwd = (process / "cwd").resolve()
        except (OSError, StopIteration):
            continue
        if int(parent) == bridge_pid:
            raise RuntimeError("Bridge has an active child process")
        if arguments and b"python" in arguments[0] and int(process.name) != bridge_pid:
            if cwd.is_relative_to(root / "backend/runs") or any(b"researchclaw.training.worker" in argument for argument in arguments):
                raise RuntimeError("A research/training worker is running")


def stop_bridge(root: Path, record: dict) -> None:
    if identity(record["pid"]) != record["identity"]:
        raise RuntimeError("Bridge identity changed")
    os.kill(record["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 20
    while identity(record["pid"]) == record["identity"] and time.monotonic() < deadline:
        time.sleep(.2)
    if identity(record["pid"]) == record["identity"]:
        raise RuntimeError("Bridge did not exit after SIGTERM; inspect before retrying")


def start_bridge(root: Path, record: dict):
    environment = dict(record["environment"], PYTHONDONTWRITEBYTECODE="1")
    local_env = root / ".env"
    if local_env.exists():
        result = subprocess.check_output(
            ["bash", "-c", 'set -a; source "$1"; env -0', "bash", str(local_env)],
            env=environment,
        )
        for item in result.split(b"\0"):
            if b"=" in item:
                key, value = item.split(b"=", 1)
                environment[os.fsdecode(key)] = os.fsdecode(value)
    with (root / "logs/agent_bridge.log").open("ab") as log:
        child = subprocess.Popen(record["arguments"], cwd=record["cwd"], env=environment,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    (root / ".pids/agent_bridge.pid").write_text(str(child.pid) + "\n")
    return child


def health(root: Path, *, timeout: int = 120) -> None:
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for name in list(environment):
        if name.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}:
            environment.pop(name)
    for line in (root / ".runtime_ports").read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() in {"FRONTEND_PORT", "RESOURCE_MONITOR_PORT", "AGENT_BRIDGE_PORT"}:
            environment[key.strip()] = value.strip().strip("\"'")
    deadline = time.monotonic() + timeout
    while True:
        result = subprocess.run([str(root / ".venv/bin/python"), str(root / "deploy/verify_services.py")],
                                cwd=root, env=environment, capture_output=True, text=True, timeout=90)
        if result.returncode == 0:
            print(result.stdout, end="", flush=True)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Service health check failed")
        time.sleep(1)


def tracked_clean(root: Path) -> None:
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked files have local changes; commit/review them before deployment")


def protected_files(root: Path) -> dict[str, str]:
    paths = [root / ".env", root / "examples/config_template.local.yaml", root / ".runtime_ports"]
    paths.extend((root / "backend/runs/queues").glob("*.json"))
    paths.extend((root / "backend/runs/project_configs").glob("*.yaml"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}


def validate_candidate(root: Path, revision: str, destination: Path) -> None:
    destination.mkdir()
    archive_path = destination.parent / "candidate.tar"
    subprocess.run(["git", "-C", str(root), "archive", "--format=tar", "-o", str(archive_path), revision], check=True)
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = destination / member.name
            if not target.resolve().is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                raise RuntimeError("Unexpected archive path or link")
        archive.extractall(destination)
    environment = dict(os.environ, PYTHONPATH=str(destination / "backend/agent"),
                       PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    tests = ["test_original_workflow_training.py", "test_claw_engine_executor.py", "test_coverage_data.py", "test_managed_training.py", "test_rc_config.py", "test_rc_cli.py",
             "test_resource_intake.py", "test_agent_bridge_reference_config.py", "test_bridge_auto_retry.py", "test_prediction_checks.py", "test_discussion_budget.py", "test_stage09_stage11_regressions.py",
             "test_git_deploy.py"]
    selected = [str(destination / "backend/agent/tests" / name) for name in tests
                if (destination / "backend/agent/tests" / name).exists()]
    with (destination.parent / "validation.log").open("wb") as log:
        subprocess.run([str(root / ".venv/bin/python"), "-m", "pytest", *selected, "-q", "-p", "no:cacheprovider"],
                       cwd=destination, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
    for script in ("start.sh", "deploy/install_server.sh"):
        subprocess.run(["bash", "-n", str(destination / script)], check=True)


def deploy(root: Path, *, check_only: bool = False) -> dict:
    root = root.resolve(strict=True)
    if Path(git(root, "rev-parse", "--show-toplevel")) != root:
        raise RuntimeError("Run this command from the deployed Git checkout")
    if git(root, "branch", "--show-current") != "preview-v1.1.0":
        raise RuntimeError("Expected deployment branch preview-v1.1.0")
    with (root / ".git/deploy.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tracked_clean(root)
        subprocess.run(["git", "-C", str(root), "fetch", "--prune", "origin"], check=True, timeout=120)
        before = git(root, "rev-parse", "HEAD")
        target = git(root, "rev-parse", "origin/preview-v1.1.0")
        subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", before, target], check=True)
        if before == target:
            health(root)
            return {"status": "already_current", "commit": before}
        changed = git(root, "diff", "--name-only", before, target).splitlines()
        if any(path.startswith("frontend/") or path in {"backend/agent/pyproject.toml", "start.sh", "backend/services/resource_monitor.py"}
               or Path(path).name in {"requirements.txt", "uv.lock", "poetry.lock"} for path in changed):
            raise RuntimeError("Frontend, dependency or launcher change requires a reviewed build/restart deployment")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup = root.parent.parent / "backups/claw-ai-lab" / ("git-deploy-" + timestamp)
        backup.mkdir(parents=True, mode=0o700)
        validate_candidate(root, target, backup / "candidate")
        if check_only:
            return {"status": "validated", "before": before, "target": target, "report": str(backup)}
        bridge = service_record(root, "agent_bridge")
        other = {name: service_record(root, name) for name in ("frontend", "resource_monitor")}
        assert_idle(root, bridge["pid"])
        snapshot = protected_files(root)
        subprocess.run(["git", "-C", str(root), "archive", "--format=tar", "-o", str(backup / "previous-source.tar"), before], check=True)
        (backup / "deployment.json").write_text(json.dumps({"before": before, "target": target, "changed": changed}, indent=2))
        stopped = False
        updated = False
        child = None
        try:
            os.kill(bridge["pid"], signal.SIGSTOP)
            try:
                assert_idle(root, bridge["pid"])
                tracked_clean(root)
                if protected_files(root) != snapshot:
                    raise RuntimeError("Runtime config changed during deployment preparation")
            finally:
                os.kill(bridge["pid"], signal.SIGCONT)
            stop_bridge(root, bridge)
            stopped = True
            assert_idle(root, bridge["pid"])
            subprocess.run(["git", "-C", str(root), "merge", "--ff-only", target], check=True)
            updated = True
            child = start_bridge(root, bridge)
            health(root)
            if child.poll() is not None or protected_files(root) != snapshot:
                raise RuntimeError("Service exited or protected runtime state changed")
            for record in other.values():
                if identity(record["pid"]) != record["identity"]:
                    raise RuntimeError("Unrelated service identity changed")
            tracked_clean(root)
            result = {"status": "deployed", "before": before, "commit": target, "backup": str(backup), "bridge_pid": child.pid}
            (backup / "result.json").write_text(json.dumps(result, indent=2))
            return result
        except BaseException:
            if child is not None and child.poll() is None:
                stop_bridge(root, service_record(root, "agent_bridge"))
            if updated:
                tracked_clean(root)
                subprocess.run(["git", "-C", str(root), "reset", "--keep", before], check=True)
            if stopped:
                start_bridge(root, bridge)
                health(root)
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fetch and validate; do not install or restart")
    parser.add_argument("--restart-bridge", action="store_true", help="Reload local .env after editing a key; requires idle queues")
    args = parser.parse_args()
    if args.check and args.restart_bridge:
        parser.error("--check and --restart-bridge cannot be combined")
    root = Path(__file__).resolve().parent.parent
    try:
        if args.restart_bridge:
            with (root / ".git/deploy.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                record = service_record(root, "agent_bridge")
                assert_idle(root, record["pid"])
                stop_bridge(root, record)
                start_bridge(root, record)
                health(root)
            result = {"status": "bridge_restarted"}
        else:
            result = deploy(root, check_only=args.check)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Deployment stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
