"""Only engineering checks and gates stated in the user's own words may stop training."""
from pathlib import Path
from types import SimpleNamespace

import yaml

from researchclaw.pipeline.acceptance import (
    ACCEPTANCE_POLICY, may_block_training, plan_issues, record_only_failures, release_record_only,
)


def _definition(**extra):
    return {"metric": "m", "inputs": "fixtures", "aggregation": "per head", "failure_action": "stop", **extra}


def _plan(quote: str) -> dict:
    return {
        "pretraining_acceptance": {"mandatory_thresholds": {"D2": {"L_max": 0.10, "R_min": 0.30}}},
        "acceptance_protocol": {"version": 1, "checks": [
            {"id": "finite_gradients", "kind": "engineering", "phase": "before_training", "blocking": True,
             "required_helpers": ["check_gradients"], "definition": _definition(metric="finite gradients")},
            {"id": "time_ratio", "kind": "engineering", "phase": "before_training", "blocking": True,
             "required_helpers": ["check_time"], "definition": _definition(metric="time_ratio", threshold=1.15)},
            {"id": "D2_connectivity", "legacy_gate": "D2", "kind": "user_required", "phase": "before_training",
             "blocking": True, "required_helpers": ["check_d2"], "user_requirement_quote": quote,
             "definition": _definition(thresholds={"L_max": 0.10, "R_min": 0.30})},
            {"id": "H1", "kind": "hypothesis", "phase": "after_training", "blocking": False,
             "definition": _definition(threshold=0.70)},
        ]},
    }


GENERIC = "保留其余强制检查，补齐H2/D2定义"


def test_policy_states_that_only_engineering_checks_stop_training():
    assert "Only these engineering checks may stop training" in ACCEPTANCE_POLICY
    assert "hypotheses.md are hypotheses, not gates" in ACCEPTANCE_POLICY


def test_gate_with_numbers_not_in_the_users_words_is_record_only():
    plan = _plan(GENERIC)
    status = {"finite_gradients": "PASS", "D2": "FAIL"}
    assert record_only_failures(plan, status, user_text="主题\n" + GENERIC) == ["D2"]


def test_gate_whose_numbers_the_user_wrote_still_blocks():
    quote = "D2 must hold before training: L <= 0.10 and R >= 0.30."
    plan = _plan(quote)
    assert may_block_training(plan["acceptance_protocol"]["checks"][2], quote)
    assert record_only_failures(plan, {"D2_connectivity": "FAIL"}, user_text=quote) == []


def test_engineering_failure_and_unknown_gates_keep_blocking():
    plan = _plan(GENERIC)
    assert record_only_failures(plan, {"finite_gradients": "FAIL", "D2": "FAIL"}, GENERIC) == []
    assert record_only_failures(plan, {"undeclared_check": "FAIL"}, GENERIC) == []
    assert record_only_failures(plan, {}, GENERIC) == []


def test_ratio_against_the_baseline_is_record_only():
    plan = _plan(GENERIC)
    assert record_only_failures(plan, {"time_ratio": {"status": "FAIL", "value": 1.289}}, GENERIC) == ["time_ratio"]


def test_preserved_gate_may_now_be_mapped_without_blocking():
    plan = _plan(GENERIC)
    plan["acceptance_protocol"]["checks"][2].update(blocking=False)
    plan["acceptance_protocol"]["checks"][2].pop("required_helpers")
    assert plan_issues(plan, user_text=GENERIC) == []
    plan["acceptance_protocol"]["checks"][2]["definition"]["thresholds"]["L_max"] = 0.5
    assert any("cannot change thresholds" in issue for issue in plan_issues(plan, user_text=GENERIC))


def test_release_turns_the_block_into_a_repair_request(tmp_path: Path):
    plan_path = tmp_path / "exp_plan.yaml"
    plan_path.write_text(yaml.safe_dump(_plan(GENERIC), allow_unicode=True), encoding="utf-8")
    config = SimpleNamespace(research=SimpleNamespace(topic=GENERIC))
    report = {"success": False, "failure_type": "pretraining_blocked", "retryable": False,
              "required_gate_status": {"D2": "FAIL", "finite_gradients": "PASS"}, "errors": ["blocked"]}
    assert release_record_only(report, plan_path, tmp_path, config) == ["D2"]
    assert report["failure_type"] == "record_only_gate_blocked" and report["retryable"] is True
    assert "FAIL_RECORDED" in report["errors"][-1] and not report["success"]


def test_s14_repairs_a_record_only_block_instead_of_returning_to_design(tmp_path: Path, monkeypatch):
    from researchclaw.pipeline.experiment_run import runtime as rt
    from .test_experiment_run_repair_guard import _Loop, _Session, _fake_runs
    plan_path = tmp_path / "exp_plan.yaml"
    plan_path.write_text(yaml.safe_dump(_plan(GENERIC), allow_unicode=True), encoding="utf-8")
    config = SimpleNamespace(research=SimpleNamespace(topic=GENERIC))
    calls = _fake_runs(monkeypatch, [
        {"success": False, "failure_type": "pretraining_blocked", "retryable": False,
         "required_gate_status": {"D2": "FAIL"}, "errors": ["D2 failed"]},
        {"success": True, "errors": []}])
    loop = _Loop([])
    report, _ = rt.ExperimentRunRuntime._supervised_run(
        tmp_path, tmp_path, config, loop, "repair", _Session(), plan_path=plan_path, run_dir=tmp_path)
    assert report["success"] and calls == ["attempt-1", "attempt-2"]
    assert "FAIL_RECORDED" in loop.messages[0]
