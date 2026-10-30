import os
from pathlib import Path

from researchclaw.pipeline import candidate
from researchclaw.pipeline.experiment_run.runtime import ExperimentRunRuntime


def _touch(path: Path, mtime: float, data: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _freeze(run_dir: Path, plan: str = "plan: a\n") -> None:
    (run_dir / "stage-09").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage-09" / "exp_plan.yaml").write_text(plan, encoding="utf-8")
    candidate.freeze(run_dir)


def _stamp(run_dir: Path, *workspaces: Path) -> None:
    for ws in workspaces:
        ws.mkdir(parents=True, exist_ok=True)
        candidate.stamp_workspace(run_dir, ws)


def test_carries_over_newest_checkpoint_workspace(tmp_path: Path) -> None:
    _freeze(tmp_path)
    stage = tmp_path / "stage-14"
    older = stage / "run_workspace_1_1" / "outputs"
    _touch(older / "formal" / "A" / "checkpoint.pth", 1000, "old")
    good = stage / "run_workspace_2_2" / "outputs"
    _touch(good / "formal" / "A" / "results.json", 2000, "{}")
    _touch(good / "formal" / "B" / "checkpoint.pth", 3000, "epoch115")
    # A run that just started: newer logs but no checkpoint yet.
    _touch(stage / "run_workspace_3_3" / "outputs" / "formal" / "A" / "log_rank0.jsonl", 9000)
    _stamp(tmp_path, *(stage / f"run_workspace_{i}_{i}" for i in (1, 2, 3)))
    ws = stage / "run_workspace_4_4"
    (ws / "outputs").mkdir(parents=True)

    src = ExperimentRunRuntime._carry_over_outputs(stage, ws, tmp_path)

    assert src == good
    assert (ws / "outputs" / "formal" / "B" / "checkpoint.pth").read_text() == "epoch115"
    assert (ws / "outputs" / "formal" / "A" / "results.json").is_file()
    assert not (ws / "outputs" / "formal" / "A" / "log_rank0.jsonl").exists()
    assert (good / "formal" / "B" / "checkpoint.pth").is_file()  # copied, not moved
    assert (ws / "outputs" / "CARRIED_OVER_FROM.txt").read_text().strip() == str(good)


def test_no_prior_progress_leaves_outputs_empty(tmp_path: Path) -> None:
    _freeze(tmp_path)
    stage = tmp_path / "stage-14"
    _touch(stage / "run_workspace_1_1" / "outputs" / "train.log", 1000)
    _stamp(tmp_path, stage / "run_workspace_1_1")
    ws = stage / "run_workspace_2_2"
    (ws / "outputs").mkdir(parents=True)

    assert ExperimentRunRuntime._carry_over_outputs(stage, ws, tmp_path) is None
    assert list((ws / "outputs").iterdir()) == []


def test_other_candidate_checkpoints_are_not_resumed(tmp_path: Path) -> None:
    _freeze(tmp_path, "plan: old\n")
    stage = tmp_path / "stage-14"
    old = stage / "run_workspace_1_1"
    _touch(old / "outputs" / "formal" / "A" / "checkpoint.pth", 1000, "old candidate")
    _stamp(tmp_path, old)
    unstamped = stage / "run_workspace_2_2"
    _touch(unstamped / "outputs" / "formal" / "A" / "checkpoint.pth", 2000, "pre-candidate run")
    _freeze(tmp_path, "plan: new\n")
    ws = stage / "run_workspace_3_3"
    (ws / "outputs").mkdir(parents=True)

    assert ExperimentRunRuntime._carry_over_outputs(stage, ws, tmp_path) is None
    assert list((ws / "outputs").iterdir()) == []
