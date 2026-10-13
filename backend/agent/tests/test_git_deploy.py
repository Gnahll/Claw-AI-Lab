from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def deployment():
    if sys.platform != "linux":
        pytest.skip("Linux deployment process management")
    path = Path(__file__).resolve().parents[3] / "deploy/git_deploy.py"
    spec = importlib.util.spec_from_file_location("git_deploy_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nonterminal_queue_prevents_deployment(deployment, tmp_path):
    queues = tmp_path / "backend/runs/queues"
    queues.mkdir(parents=True)
    (queues / "jobs.json").write_text(json.dumps({"tasks": [{"status": "running"}]}))
    with pytest.raises(RuntimeError, match="not idle"):
        deployment.assert_idle(tmp_path, 99999999)


def test_dirty_tracked_files_block_deployment(deployment, tmp_path, monkeypatch):
    monkeypatch.setattr(deployment, "git", lambda *arguments: " M backend/services/agent_bridge.py")
    with pytest.raises(RuntimeError, match="local changes"):
        deployment.tracked_clean(tmp_path)


def test_failed_fetch_never_changes_checkout_or_restarts_service(deployment, tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()

    def git_result(root, *arguments):
        if arguments[0] == "rev-parse":
            return str(tmp_path)
        if arguments[0] == "branch":
            return "preview-v1.1.0"
        return ""

    commands = []

    def fail_fetch(command, **kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(deployment, "git", git_result)
    monkeypatch.setattr(deployment.subprocess, "run", fail_fetch)
    monkeypatch.setattr(deployment, "stop_bridge", lambda *args: pytest.fail("Must not restart after failed fetch"))
    with pytest.raises(subprocess.CalledProcessError):
        deployment.deploy(tmp_path)
    assert len(commands) == 1 and "fetch" in commands[0]


def test_noop_deployment_checks_health_without_restarting(deployment, tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()

    def git_result(root, *arguments):
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if arguments[0] == "branch":
            return "preview-v1.1.0"
        if arguments[0] == "status":
            return ""
        return "a" * 40

    health = []
    monkeypatch.setattr(deployment, "git", git_result)
    monkeypatch.setattr(deployment.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(deployment, "health", lambda root: health.append(root))
    monkeypatch.setattr(deployment, "stop_bridge", lambda *args: pytest.fail("No-op must not restart"))
    assert deployment.deploy(tmp_path)["status"] == "already_current"
    assert health == [tmp_path]


def test_local_runtime_settings_are_in_protected_snapshot(deployment, tmp_path):
    (tmp_path / "examples").mkdir()
    (tmp_path / ".env").write_text("RESEARCHCLAW_API_KEY=fixture-only\n")
    (tmp_path / "examples/config_template.local.yaml").write_text("llm: {}\n")
    before = deployment.protected_files(tmp_path)
    (tmp_path / ".env").write_text("RESEARCHCLAW_API_KEY=changed-fixture\n")
    assert before != deployment.protected_files(tmp_path)
    assert ".env" in before and "examples/config_template.local.yaml" in before
