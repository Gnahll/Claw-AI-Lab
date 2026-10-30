import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline import candidate, executor
from researchclaw.pipeline.stages import Stage, StageStatus
from .test_rc_config import _valid_config_data


def _plan(run_dir: Path, text: str = "plan: a\n") -> Path:
    path = run_dir / "stage-09" / "exp_plan.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _code(run_dir: Path, text: str = "print('train')\n") -> Path:
    path = run_dir / "stage-11" / "experiment" / "main.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _through_s12(run_dir: Path) -> None:
    _plan(run_dir)
    _code(run_dir)
    candidate.record_completion(run_dir, 9, True)
    for stage in (11, 12):
        (run_dir / f"stage-{stage}").mkdir(exist_ok=True)
        candidate.record_completion(run_dir, stage, True)


def test_s9_done_freezes_and_failure_withholds_plan(tmp_path):
    _plan(tmp_path)
    candidate.record_completion(tmp_path, 9, True)
    assert candidate.handoff_errors(tmp_path, 10) == []
    assert candidate.handoff_errors(tmp_path, 11) == []

    candidate.record_completion(tmp_path, 9, False)
    (tmp_path / "stage-09" / "decision.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    for stage in (10, 11):
        assert "S9 has not completed" in candidate.handoff_errors(tmp_path, stage)[0]


def test_new_candidate_is_recorded_in_history(tmp_path):
    _plan(tmp_path, "plan: a\n")
    first = candidate.freeze(tmp_path)
    assert candidate.freeze(tmp_path)["candidate_id"] == first["candidate_id"]
    _plan(tmp_path, "plan: b\n")
    second = candidate.freeze(tmp_path)
    lines = (tmp_path / candidate.HISTORY_FILE).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["candidate_id"] for line in lines] == [first["candidate_id"], second["candidate_id"]]
    assert json.loads(lines[1])["previous_candidate_id"] == first["candidate_id"]


def test_plan_edited_after_freeze_is_rejected(tmp_path):
    _plan(tmp_path)
    candidate.freeze(tmp_path)
    _plan(tmp_path, "plan: edited by hand\n")
    assert "changed after candidate" in candidate.handoff_errors(tmp_path, 11)[0]


def test_completed_legacy_plan_is_adopted(tmp_path):
    _plan(tmp_path)
    (tmp_path / "stage-09" / "decision.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")
    record = candidate.current(tmp_path)
    assert record["legacy"] is True
    assert (tmp_path / "stage-09" / candidate.CANDIDATE_FILE).is_file()


def test_s12_rejects_code_from_another_candidate(tmp_path):
    _plan(tmp_path, "plan: old\n")
    _code(tmp_path)
    candidate.record_completion(tmp_path, 9, True)
    (tmp_path / "stage-11").mkdir(exist_ok=True)
    candidate.record_completion(tmp_path, 11, True)
    assert candidate.handoff_errors(tmp_path, 12) == []

    _plan(tmp_path, "plan: redesigned\n")
    candidate.record_completion(tmp_path, 9, True)
    errors = candidate.handoff_errors(tmp_path, 12)
    assert "belongs to candidate" in errors[0] and "measured again" in errors[0]


def test_s12_requires_completed_s11(tmp_path):
    _plan(tmp_path)
    candidate.record_completion(tmp_path, 9, True)
    (tmp_path / "stage-11").mkdir()
    candidate.record_completion(tmp_path, 11, False)
    assert "S11 has no completed result" in candidate.handoff_errors(tmp_path, 12)[0]


def test_s14_rejects_code_changed_after_s12(tmp_path):
    _through_s12(tmp_path)
    assert candidate.handoff_errors(tmp_path, 13) == []
    assert candidate.handoff_errors(tmp_path, 14) == []
    # Run outputs are not code and do not invalidate the check.
    out = tmp_path / "stage-11" / "experiment" / "outputs" / "result.py"
    out.parent.mkdir()
    out.write_text("x = 1\n", encoding="utf-8")
    assert candidate.handoff_errors(tmp_path, 14) == []

    _code(tmp_path, "print('patched after the check')\n")
    assert "changed after S12" in candidate.handoff_errors(tmp_path, 14)[0]
    # Resource planning does not depend on the exact code.
    assert candidate.handoff_errors(tmp_path, 13) == []


def test_earlier_candidate_context_is_history(tmp_path):
    _plan(tmp_path)
    current = candidate.freeze(tmp_path)["candidate_id"]
    ctx = tmp_path / "revision-context"
    (ctx / "measured_candidate_source").mkdir(parents=True)
    (ctx / "measured_candidate_source" / "main.py").write_text("old\n", encoding="utf-8")
    (ctx / "CURRENT_CANDIDATE_PREFLIGHT.json").write_text("{}", encoding="utf-8")
    (ctx / "old_measurements.json").write_text(json.dumps({"candidate_id": "cand-old"}), encoding="utf-8")
    (ctx / "own_measurements.json").write_text(json.dumps({"candidate_id": current}), encoding="utf-8")
    (ctx / "FIXED_PROTOCOL.yaml").write_text("protocol: fixed\n", encoding="utf-8")
    (ctx / candidate.CONTEXT_REGISTRY).write_text(json.dumps(
        {"history": ["measured_candidate_source/", "CURRENT_CANDIDATE_PREFLIGHT.json"]}), encoding="utf-8")

    assert candidate.history_context(tmp_path) == {
        "measured_candidate_source/main.py", "CURRENT_CANDIDATE_PREFLIGHT.json", "old_measurements.json"}


def test_current_stage_dirs_skip_rolled_back_versions(tmp_path):
    for name in ("stage-11_v1", "stage-11", "stage-09", "stage-12_v2", "stage-14"):
        (tmp_path / name).mkdir()
    assert [d.name for d in candidate.current_stage_dirs(tmp_path)] == ["stage-14", "stage-11", "stage-09"]


def test_workspace_resume_only_within_candidate(tmp_path):
    ws = tmp_path / "stage-14" / "run_workspace_1_1"
    ws.mkdir(parents=True)
    candidate.stamp_workspace(tmp_path, ws)  # no candidate yet: left unstamped
    assert not (ws / candidate.WORKSPACE_STAMP).exists()

    _plan(tmp_path, "plan: a\n")
    candidate.freeze(tmp_path)
    candidate.stamp_workspace(tmp_path, ws)
    assert candidate.same_candidate_workspace(tmp_path, ws)
    _plan(tmp_path, "plan: b\n")
    candidate.freeze(tmp_path)
    assert not candidate.same_candidate_workspace(tmp_path, ws)


@pytest.fixture
def stage_runner(tmp_path, monkeypatch):
    cfg = RCConfig.from_dict(_valid_config_data(), project_root=tmp_path, check_paths=False)
    calls = []

    def run(stage):
        def fake(stage_dir, run_dir, config, adapters, **kwargs):
            calls.append(stage)
            return executor.StageResult(stage=stage, status=StageStatus.FAILED, error="fixture boundary")
        monkeypatch.setitem(executor._STAGE_EXECUTORS, stage, fake)
        return executor.execute_stage(stage, run_dir=tmp_path, run_id="candidate-test", config=cfg,
                                      adapters=AdapterBundle(), auto_approve_gates=True)

    monkeypatch.setattr(executor, "_read_prior_artifact", lambda *args: "fixture input")
    monkeypatch.setattr(executor.LLMClient, "from_rc_config", lambda *args: SimpleNamespace(
        config=SimpleNamespace(base_url="", api_key="")))
    return run, calls


def test_execute_stage_refuses_code_generation_without_completed_plan(tmp_path, stage_runner):
    run, calls = stage_runner
    _plan(tmp_path)
    (tmp_path / "stage-09" / "decision.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    result = run(Stage.CODE_GENERATION)

    assert calls == []
    assert result.status == StageStatus.FAILED and result.decision == "revise_design"
    assert result.error.startswith("Candidate handoff rejected:")
    decision = json.loads((tmp_path / "stage-11" / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "failed"


def test_execute_stage_refuses_sanity_check_on_other_candidate_code(tmp_path, stage_runner):
    run, calls = stage_runner
    _through_s12(tmp_path)
    _plan(tmp_path, "plan: redesigned\n")
    candidate.record_completion(tmp_path, 9, True)

    result = run(Stage.SANITY_CHECK)

    assert calls == [] and result.decision == "retry"
    assert "belongs to candidate" in result.error


def test_failed_stage_drops_its_binding(tmp_path, stage_runner):
    run, calls = stage_runner
    _through_s12(tmp_path)
    assert candidate.binding(tmp_path, 12) is not None

    result = run(Stage.SANITY_CHECK)

    assert calls == [Stage.SANITY_CHECK] and result.status == StageStatus.FAILED
    assert candidate.binding(tmp_path, 12) is None
    assert "S12 has no completed result" in candidate.handoff_errors(tmp_path, 14)[0]


def _checkpoint(run_dir: Path, stage: int) -> None:
    (run_dir / "checkpoint.json").write_text(json.dumps(
        {"last_completed_stage": stage, "last_completed_name": Stage(stage).name, "run_id": "r"}), encoding="utf-8")


def _last_completed(run_dir: Path) -> int:
    return json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))["last_completed_stage"]


def test_code_patched_by_s14_rewinds_resume_to_s12(tmp_path, stage_runner):
    run, calls = stage_runner
    _through_s12(tmp_path)
    _checkpoint(tmp_path, 13)
    _code(tmp_path, "print('fixed during training')\n")

    result = run(Stage.EXPERIMENT_RUN)

    assert calls == [] and "resume restarts at S12" in result.error
    assert _last_completed(tmp_path) == 11


def test_redesigned_candidate_rewinds_resume_to_s11(tmp_path, stage_runner):
    run, calls = stage_runner
    _through_s12(tmp_path)
    _checkpoint(tmp_path, 13)
    _plan(tmp_path, "plan: redesigned\n")
    candidate.record_completion(tmp_path, 9, True)

    result = run(Stage.EXPERIMENT_RUN)

    assert calls == [] and "resume restarts at S11" in result.error
    assert _last_completed(tmp_path) == 10


def test_unfinished_plan_rewinds_to_s9_and_never_moves_forward(tmp_path):
    _plan(tmp_path)
    (tmp_path / "stage-09" / "decision.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    assert candidate.rerun_stage(tmp_path, 12) == 9
    _checkpoint(tmp_path, 7)
    candidate.rewind_checkpoint(tmp_path, 9)
    assert _last_completed(tmp_path) == 7


def _legacy_run(run_dir: Path) -> Path:
    """A run that reached S14 before candidates existed."""
    _plan(run_dir)
    _code(run_dir)
    for stage in (9, 11, 12, 13):
        (run_dir / f"stage-{stage:02d}").mkdir(exist_ok=True)
        (run_dir / f"stage-{stage:02d}" / "decision.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")
    old = run_dir / "stage-14" / "run_workspace_1_1"
    (old / "outputs").mkdir(parents=True)
    (old / "outputs" / "checkpoint.pth").write_text("epoch 80", encoding="utf-8")
    return old


def test_legacy_run_resumes_training_instead_of_regenerating(tmp_path, stage_runner):
    run, calls = stage_runner
    old = _legacy_run(tmp_path)
    _checkpoint(tmp_path, 13)

    assert candidate.handoff_errors(tmp_path, 14) == []
    assert candidate.binding(tmp_path, 11)["legacy"] and candidate.binding(tmp_path, 12)["legacy"]
    assert candidate.same_candidate_workspace(tmp_path, old)
    result = run(Stage.EXPERIMENT_RUN)
    assert calls == [Stage.EXPERIMENT_RUN] and "handoff" not in (result.error or "")
    assert _last_completed(tmp_path) == 13


def test_legacy_adoption_requires_completed_stages(tmp_path):
    _legacy_run(tmp_path)
    (tmp_path / "stage-11" / "decision.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    assert "S12 has no completed result" in candidate.handoff_errors(tmp_path, 14)[0]
    assert candidate.rerun_stage(tmp_path, 14) == 11


def test_new_candidate_gets_no_legacy_adoption(tmp_path):
    old = _legacy_run(tmp_path)
    _plan(tmp_path, "plan: redesigned\n")
    candidate.record_completion(tmp_path, 9, True)
    assert "S12 has no completed result" in candidate.handoff_errors(tmp_path, 14)[0]
    assert not candidate.same_candidate_workspace(tmp_path, old)
