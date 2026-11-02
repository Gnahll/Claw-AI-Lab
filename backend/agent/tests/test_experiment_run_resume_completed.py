"""A relaunch after training finished must not fail for lack of new updates or lose evidence."""
import json
from pathlib import Path

from researchclaw.pipeline.experiment_run.runtime import ExperimentRunRuntime
from researchclaw.training.verified_run import create_receipt, source_manifest


def _launch(stage: Path, attempt: str, workspace: Path, steps: int, *, jsonl_only: bool = False) -> Path:
    out = stage / "runs" / attempt
    out.mkdir(parents=True, exist_ok=True)
    (out / "execution_report.json").write_text(json.dumps(
        {"success": False, "smoke": False, "workspace": str(workspace.resolve())}), encoding="utf-8")
    if jsonl_only:
        (out / "training_observation.jsonl").write_text("".join(
            json.dumps({"step": i, "parameter_update_observed": True}) + "\n" for i in range(1, steps + 1)),
            encoding="utf-8")
    else:
        (out / "training_observation.json").write_text(json.dumps(
            {"optimizer_steps": steps, "parameter_update_observed": steps > 0}), encoding="utf-8")
    return out


def _resume(old: Path, new: Path) -> None:
    (new / "outputs").mkdir(parents=True, exist_ok=True)
    (new / "outputs" / "CARRIED_OVER_FROM.txt").write_text(str(old / "outputs") + "\n", encoding="utf-8")


def test_relaunch_archives_previous_attempts_instead_of_overwriting(tmp_path: Path) -> None:
    stage = tmp_path / "stage-14"
    ws1 = stage / "run_workspace_1"
    _launch(stage, "attempt-1", ws1, 20000)
    (stage / "runs" / "run_report.json").write_text("{}", encoding="utf-8")

    target = ExperimentRunRuntime._archive_previous_launch(stage / "runs")

    assert target == stage / "runs" / "history" / "run_workspace_1"
    assert (target / "attempt-1" / "training_observation.json").is_file()
    assert (target / "run_report.json").is_file()
    assert not (stage / "runs" / "attempt-1").exists()
    assert ExperimentRunRuntime._archive_previous_launch(stage / "runs") is None


def test_prior_training_follows_carried_over_lineage_only(tmp_path: Path) -> None:
    stage = tmp_path / "stage-14"
    ws1, ws2, ws3, other = (stage / f"run_workspace_{n}" for n in ("1", "2", "3", "x"))
    _launch(stage, "history/run_workspace_1/attempt-1", ws1, 20000, jsonl_only=True)  # killed mid-training
    _launch(stage, "history/run_workspace_2/attempt-1", ws2, 10150)
    _launch(stage, "history/run_workspace_x/attempt-1", other, 999)  # unrelated candidate
    _resume(ws1, ws2)
    _resume(ws2, ws3)

    prior = ExperimentRunRuntime._prior_training(stage, ws3)

    assert prior is not None and prior["prior_optimizer_steps"] == 30150
    assert {Path(item["workspace"]).name for item in prior["evidence"]} == {"run_workspace_1", "run_workspace_2"}


def test_no_prior_training_without_recorded_updates(tmp_path: Path) -> None:
    stage = tmp_path / "stage-14"
    ws1, ws2 = stage / "run_workspace_1", stage / "run_workspace_2"
    _launch(stage, "history/run_workspace_1/attempt-1", ws1, 0)
    _resume(ws1, ws2)
    assert ExperimentRunRuntime._prior_training(stage, ws2) is None


def test_receipt_accepts_resumed_completed_run(tmp_path: Path) -> None:
    run = tmp_path / "project"
    workspace = run / "stage-14/run_workspace_2"
    (workspace / "outputs/predictions").mkdir(parents=True)
    (workspace / "main.py").write_text("print(1)\n", encoding="utf-8")
    (workspace / "results.json").write_text(json.dumps({"pixel_f1": 0.7}), encoding="utf-8")
    (workspace / "outputs/model.pt").write_bytes(b"weights")
    (workspace / "outputs/predictions/a.npy").write_bytes(b"pred")
    report_path = run / "stage-14/runs/attempt-1/execution_report.json"
    report_path.parent.mkdir(parents=True)
    report = {"success": True, "smoke": False, "require_training": True,
              "source_manifest": source_manifest(workspace),
              "training": {"optimizer_steps": 0, "parameter_update_observed": False},
              "resumed_completed_run": {"prior_optimizer_steps": 30150, "evidence": []}}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    from types import SimpleNamespace
    config = SimpleNamespace(experiment=SimpleNamespace(require_training=True, metric_key="pixel_f1"))

    receipt = create_receipt(workspace, run, report_path, config)
    assert "outputs/model.pt" in receipt["artifacts"]

    report.pop("resumed_completed_run")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    try:
        create_receipt(workspace, run, report_path, config)
    except ValueError as exc:
        assert "successful supervised training" in str(exc)
    else:
        raise AssertionError("a run without any observed update must not get a receipt")
