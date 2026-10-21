"""S20 uses only the latest round's experiment_summary.json."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from researchclaw.pipeline import executor as rc_executor


def _write(run_dir: Path, stage: str, metrics: int, paired: int) -> None:
    d = run_dir / stage
    d.mkdir(parents=True, exist_ok=True)
    (d / "experiment_summary.json").write_text(
        json.dumps({
            "stage": stage,
            "metrics_summary": {f"m{i}": {"mean": i} for i in range(metrics)},
            "paired_comparisons": [{} for _ in range(paired)],
        }),
        encoding="utf-8",
    )


def _picked(run_dir: Path) -> str | None:
    text = rc_executor._latest_experiment_summary_text(run_dir)
    return json.loads(text)["stage"] if text else None


def test_current_round_beats_richer_old_round(tmp_path: Path) -> None:
    # d9f4 layout: stage-16_v1 has more paired comparisons than stage-16.
    _write(tmp_path, "stage-16", metrics=5, paired=4)
    _write(tmp_path, "stage-16_v1", metrics=5, paired=5)
    assert _picked(tmp_path) == "stage-16"


def test_highest_version_when_no_current_dir(tmp_path: Path) -> None:
    _write(tmp_path, "stage-16_v1", metrics=9, paired=9)
    _write(tmp_path, "stage-16_v2", metrics=1, paired=0)
    _write(tmp_path, "stage-16_v10", metrics=1, paired=0)
    assert _picked(tmp_path) == "stage-16_v10"


def test_stage16_preferred_over_stage14(tmp_path: Path) -> None:
    _write(tmp_path, "stage-14", metrics=20, paired=5)
    _write(tmp_path, "stage-16", metrics=2, paired=0)
    assert _picked(tmp_path) == "stage-16"


def test_falls_back_to_stage14_and_skips_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "stage-16"
    bad.mkdir()
    (bad / "experiment_summary.json").write_text("not json", encoding="utf-8")
    _write(tmp_path, "stage-14_v1", metrics=3, paired=0)
    _write(tmp_path, "stage-14", metrics=1, paired=0)
    assert _picked(tmp_path) == "stage-14"


def test_none_when_missing(tmp_path: Path) -> None:
    assert rc_executor._latest_experiment_summary_text(tmp_path) is None


def test_paper_draft_uses_latest_helper() -> None:
    src = inspect.getsource(rc_executor._execute_paper_draft)
    assert "_latest_experiment_summary_text(run_dir)" in src
    assert "_best_metric_count" not in src
