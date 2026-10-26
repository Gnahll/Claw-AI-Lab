import os
from pathlib import Path

from researchclaw.pipeline.experiment_run.runtime import ExperimentRunRuntime


def _touch(path: Path, mtime: float, data: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_carries_over_newest_checkpoint_workspace(tmp_path: Path) -> None:
    stage = tmp_path / "stage-14"
    older = stage / "run_workspace_1_1" / "outputs"
    _touch(older / "formal" / "A" / "checkpoint.pth", 1000, "old")
    good = stage / "run_workspace_2_2" / "outputs"
    _touch(good / "formal" / "A" / "results.json", 2000, "{}")
    _touch(good / "formal" / "B" / "checkpoint.pth", 3000, "epoch115")
    # A run that just started: newer logs but no checkpoint yet.
    _touch(stage / "run_workspace_3_3" / "outputs" / "formal" / "A" / "log_rank0.jsonl", 9000)
    ws = stage / "run_workspace_4_4"
    (ws / "outputs").mkdir(parents=True)

    src = ExperimentRunRuntime._carry_over_outputs(stage, ws)

    assert src == good
    assert (ws / "outputs" / "formal" / "B" / "checkpoint.pth").read_text() == "epoch115"
    assert (ws / "outputs" / "formal" / "A" / "results.json").is_file()
    assert not (ws / "outputs" / "formal" / "A" / "log_rank0.jsonl").exists()
    assert (good / "formal" / "B" / "checkpoint.pth").is_file()  # copied, not moved
    assert (ws / "outputs" / "CARRIED_OVER_FROM.txt").read_text().strip() == str(good)


def test_no_prior_progress_leaves_outputs_empty(tmp_path: Path) -> None:
    stage = tmp_path / "stage-14"
    _touch(stage / "run_workspace_1_1" / "outputs" / "train.log", 1000)
    ws = stage / "run_workspace_2_2"
    (ws / "outputs").mkdir(parents=True)

    assert ExperimentRunRuntime._carry_over_outputs(stage, ws) is None
    assert list((ws / "outputs").iterdir()) == []
