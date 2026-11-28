import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.training import execution, method_review
from researchclaw.training.verified_run import file_hash, source_manifest


@pytest.fixture
def case(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.py").write_text("TARGET_STEPS = 3000")
    config = SimpleNamespace(experiment=SimpleNamespace(require_training=True, metric_key="AP", metric_direction="maximize"))
    return workspace, tmp_path / "supervision", config


def setup_runner(monkeypatch, *, optimizer_steps=None, fake_smoke=False):
    calls = []

    def launch(workspace, output_dir, config, *, pilot=False, step_limit=None, **options):
        calls.append((workspace, pilot, step_limit))
        if not pilot:
            return {"success": True}
        (workspace / "outputs").mkdir()
        ckpt = workspace / "outputs/model.pt"
        ckpt.write_bytes(b"weight fixture")
        result = {"pilot_validation": {"split": "validation", "samples": 2, "metrics": {"method": {"AP": 5.8},
                                                                                         "base": {"AP": 5.0}}}}
        (workspace / "results.json").write_text(json.dumps(result))
        return {"success": True, "smoke": fake_smoke, "require_training": True, "results": result, "world_size": 1,
                "training": {"optimizer_steps": step_limit if optimizer_steps is None else optimizer_steps,
                             "parameter_update_observed": True, "losses": [0.4], "checkpoint_verified": True,
                             "checkpoint_sha256": file_hash(ckpt)},
                "results_sha256": file_hash(workspace / "results.json"),
                "source_manifest": source_manifest(workspace), "predictions": {"groups": [{"checked": 2}]}}
    monkeypatch.setattr(execution, "_run_once", launch)
    return calls


class Review:
    def __init__(self, approved=True):
        self.approved = approved
        self.calls = 0

    def chat(self, messages, **kwargs):
        evidence = json.loads(messages[0]["content"])["observations"]
        assert evidence["validation"]["samples"] == 2
        self.calls += 1
        return SimpleNamespace(content=json.dumps({"approved": self.approved, "reason": "paired validation reviewed"}))


def test_automatic_isolated_short_execution_and_decision(case, monkeypatch):
    calls = setup_runner(monkeypatch)
    client = Review()
    assert execution.run_experiment(*case, review_client=client)["success"]
    assert client.calls == 1
    assert calls[0][0] != case[0] and calls[0][1:] == (True, 150)
    assert calls[1] == (case[0], False, None)
    assert list(case[1].rglob("method_review.json"))


@pytest.mark.parametrize("approved,bad_steps,smoke,expected_calls", [(False, None, False, 1),
                          (True, 1, False, 0), (True, None, True, 0)])
def test_rejections_do_not_launch_full(case, monkeypatch, approved, bad_steps, smoke, expected_calls):
    calls = setup_runner(monkeypatch, optimizer_steps=bad_steps, fake_smoke=smoke)
    client = Review(approved)
    report = execution.run_experiment(*case, review_client=client)
    assert not report["success"] and report["failure_type"] == "pretraining_blocked"
    assert len(calls) == 1 and client.calls == expected_calls


def test_project_threshold_cannot_be_overridden_by_reviewer(case, tmp_path, monkeypatch):
    setup_runner(monkeypatch)
    plan = tmp_path / "stage-09/exp_plan.yaml"
    plan.parent.mkdir()
    plan.write_text("short_training_review:\n  comparisons:\n    - {method: method, control: base, min_gain: 1}\n")
    client = Review()
    report = execution.run_experiment(*case, review_client=client, run_dir=tmp_path)
    assert not report["success"] and "required AP gain" in report["errors"][0]
    assert client.calls == 0


def test_unknown_budget_enforced_and_switches_to_review(case, monkeypatch):
    (case[0] / "main.py").write_text("steps = user_config()")
    normal = setup_runner(monkeypatch)
    launch = execution._run_once
    def bounded(*args, **kwargs):
        if kwargs.get("step_limit") == method_review.SHORT_LIMIT:
            args[1].mkdir()
            return {"success": False, "failure_type": "short_budget_exceeded"}
        return launch(*args, **kwargs)
    monkeypatch.setattr(execution, "_run_once", bounded)
    assert execution.run_experiment(*case, review_client=Review())["success"]
    assert [row[1] for row in normal] == [True, False]


def test_short_run_requires_no_ai(case, monkeypatch):
    (case[0] / "main.py").write_text("TARGET_STEPS = 200")
    calls = setup_runner(monkeypatch)
    assert execution.run_experiment(*case)["success"]
    assert calls == [(case[0], False, method_review.SHORT_LIMIT)]

