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
        result_report = {"success": True, "smoke": fake_smoke, "require_training": True, "results": result, "world_size": 1,
                "training": {"optimizer_steps": step_limit if optimizer_steps is None else optimizer_steps,
                             "parameter_update_observed": True, "losses": [0.4], "checkpoint_verified": True,
                             "checkpoint_sha256": file_hash(ckpt)},
                "results_sha256": file_hash(workspace / "results.json"),
                "source_manifest": source_manifest(workspace), "predictions": {"groups": [{"checked": 2}]}}
        (workspace / "outputs/val_predictions.csv").write_text("pred\n0\n1\n")
        output_dir.mkdir(parents=True)
        (output_dir / "execution_report.json").write_text(json.dumps(result_report))
        (output_dir / "training_observation.json").write_text(json.dumps(result_report["training"]))
        return result_report
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



def test_real_torch_pilot_before_full_method(tmp_path, monkeypatch):
    """Uses the same PyTorch observer, weight reload and prediction checks as a production launch."""
    import sys
    pytest.importorskip("torch")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.py").write_text("""from pathlib import Path
import json, os
import numpy as np
import torch
out = Path("outputs")
out.mkdir(exist_ok=True)
model = torch.nn.Linear(1, 1, bias=False)
torch.nn.init.constant_(model.weight, 0.2)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
data = torch.tensor([[1.0], [2.0], [3.0]])
for _ in range(int(os.getenv("RESEARCHCLAW_PILOT_UPDATES", "3"))):
    opt.zero_grad()
    (model(data) - data).square().mean().backward()
    opt.step()
val = torch.tensor([[1.5], [2.5]])
with torch.no_grad():
    pred = model(val)
    metric = (pred - val).abs().mean().item()
np.save(out / "val_predictions.npy", pred.numpy())
torch.save(model.state_dict(), out / "model.pt")
Path("results.json").write_text(json.dumps({"pilot_validation": {"split": "validation", "samples": 2,
        "metrics": {"method": {"error": metric}}}}))
""")
    stage = tmp_path / "stage-09"
    stage.mkdir()
    (stage / "exp_plan.yaml").write_text("training_phase: long\nshort_training_review: {pilot_steps: 2}\n")
    config = SimpleNamespace(experiment=SimpleNamespace(require_training=True, metric_key="error", metric_direction="minimize",
                  sandbox=SimpleNamespace(python_path=sys.executable), time_budget_sec=90, monitor_interval_sec=1))
    out = tmp_path / "stage-14/runs/attempt-1"
    client = Review()
    full = execution.run_experiment(workspace, out, config, run_dir=tmp_path, review_client=client)
    assert full["success"], full["errors"]
    assert full["training"]["optimizer_steps"] == 3 and client.calls == 1
    evidence_dir = Path(json.loads((out / "short_training_pilot.json").read_text())["pilot"])
    pilot = json.loads((evidence_dir / "execution/execution_report.json").read_text())
    assert pilot["training"]["optimizer_steps"] == 2
    assert full["training"]["checkpoint_sha256"] != pilot["training"]["checkpoint_sha256"]


class ReviewUnavailableOnce(Review):
    def __init__(self):
        super().__init__()
        self.unavailable = True

    def chat(self, messages, **kwargs):
        if self.unavailable:
            self.unavailable = False
            raise ConnectionError("review service unavailable")
        return super().chat(messages, **kwargs)


@pytest.mark.parametrize("change,expected_pilots", [
    (None, 1), ("yaml", 2), ("protocol", 2), ("prediction", 2), ("training_log", 2),
    ("checkpoint", 2), ("resource", 2), ("untracked_external", 2), ("pilot_yaml", 2),
])
def test_retry_validates_inputs_and_real_evidence(case, tmp_path, monkeypatch, change, expected_pilots):
    workspace, first_output, config = case
    calls = setup_runner(monkeypatch)
    client = ReviewUnavailableOnce()
    settings = workspace / "model.yaml"
    settings.write_text("width: 32")
    external = tmp_path / "dataset"
    external.mkdir()
    frame = external / "image.bin"
    frame.write_bytes(b"observed dataset")
    config.experiment.datasets_dir = str(external)
    if change == "untracked_external":
        (workspace / "main.py").write_text("TARGET_STEPS=3000\nsource='https://host/dynamic'\n")
    assert not execution.run_experiment(*case, run_dir=tmp_path, review_client=client)["success"]
    first = json.loads((first_output / "short_training_pilot.json").read_text())
    pilot = Path(first["pilot"])
    if change == "pilot_yaml":
        (pilot / "workspace/model.yaml").write_text("tampered: true")
    elif change == "yaml":
        settings.write_text("width: 64")
    elif change == "resource":
        frame.write_bytes(b"modified dataset")
    elif change == "protocol":
        (tmp_path / "stage-09").mkdir()
        (tmp_path / "stage-09/exp_plan.yaml").write_text("training_phase: long\nseed: 111")
    elif change == "checkpoint":
        (pilot / "workspace/outputs/model.pt").write_bytes(b"replaced model")
    elif change == "prediction":
        (pilot / "workspace/outputs/val_predictions.csv").write_text("pred\n0\n0\n")
    elif change == "training_log":
        (pilot / "execution/training_observation.json").write_text("{}")
    second_output = tmp_path / "new-attempt-dir"
    assert execution.run_experiment(workspace, second_output, config, run_dir=tmp_path, review_client=client)["success"]
    second = json.loads((second_output / "short_training_pilot.json").read_text())
    assert sum(pilot for _, pilot, _ in calls) == expected_pilots
    assert second["reused"] is (expected_pilots == 1)
    assert (first["pilot"] == second["pilot"]) is (expected_pilots == 1)
