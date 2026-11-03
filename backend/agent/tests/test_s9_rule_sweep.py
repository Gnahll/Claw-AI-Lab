"""S9 corrects check labels itself; S11 sends a repeated helper gap back to S9."""
import json

from researchclaw.pipeline.acceptance import may_block_training, normalize_plan, plan_issues
from researchclaw.pipeline.codegen.readiness import repeated_helper_gaps


def _definition(**extra):
    return {"metric": "m", "inputs": "test sets", "aggregation": "macro", "failure_action": "report", **extra}


def _plan(*checks):
    return {"acceptance_protocol": {"version": 1, "checks": [
        {"id": "finite_gradients", "kind": "engineering", "phase": "before_training", "blocking": True,
         "required_helpers": ["check_gradients"], "definition": _definition()}, *checks]}}


def test_labelling_mistakes_are_corrected_not_failed():
    plan = _plan(
        {"id": "auroc_gain", "kind": "hypothesis", "phase": "after_training", "blocking": True, "definition": _definition()},
        {"id": "shapes", "kind": "engineering", "phase": "pre-training", "blocking": "false", "definition": _definition()},
        {"id": "shapes", "kind": "Scientific", "blocking": "yes", "definition": _definition()},
        {"id": "budget", "kind": "engineering", "phase": "before_training", "blocking": True,
         "required_helpers": ["check budget", "3d_mem"], "definition": _definition()},
        {"id": "user_gate", "kind": "user_required", "phase": "before_training", "blocking": True,
         "user_requirement_quote": "invented approval", "definition": _definition()})
    notes = normalize_plan(plan, "the user text")
    checks = plan["acceptance_protocol"]["checks"]
    assert checks[1]["blocking"] is False
    assert (checks[2]["phase"], checks[2]["blocking"], checks[2]["required_helpers"]) == (
        "before_training", True, ["check_shapes"])
    assert (checks[3]["id"], checks[3]["kind"], checks[3]["phase"], checks[3]["blocking"]) == (
        "shapes_2", "hypothesis", "after_training", False)
    assert checks[4]["required_helpers"] == ["check_budget", "check_3d_mem"]
    assert checks[5]["blocking"] is False
    assert len(notes) >= 7
    assert plan_issues(plan, require_protocol=True, user_text="the user text") == []


def test_after_training_engineering_check_is_recorded_not_blocking():
    check = {"id": "artifacts_written", "kind": "engineering", "phase": "after_training", "blocking": True,
             "definition": _definition()}
    plan = _plan(check)
    notes = normalize_plan(plan)
    assert any("recorded" in n for n in notes)
    assert plan_issues(plan, require_protocol=True) == []
    assert not may_block_training(check, "")


def test_unrunnable_definition_still_fails():
    plan = _plan({"id": "x", "kind": "hypothesis", "phase": "after_training", "blocking": False,
                  "definition": {"metric": "m"}})
    normalize_plan(plan)
    assert any("definition must specify" in i for i in plan_issues(plan))
    plan = {"acceptance_protocol": {"version": 1, "checks": [
        {"id": "artifacts", "kind": "engineering", "phase": "after_training", "blocking": True,
         "definition": _definition()}]}}
    normalize_plan(plan)
    assert any("retain engineering prerequisites" in i for i in plan_issues(plan))


def test_s11_rolls_back_only_when_same_helpers_are_missing_again(tmp_path):
    report = tmp_path / "implementation_readiness.json"
    fingerprint = {"plan_sha256": "a", "feedback_sha256": "b"}
    errors = ["Missing declared implementation helper: check_x", "Missing declared implementation helper: check_y"]
    assert repeated_helper_gaps(report, errors, fingerprint) == []
    report.write_text(json.dumps({"ready": False, "errors": errors[1:], **fingerprint}), encoding="utf-8")
    assert repeated_helper_gaps(report, errors, fingerprint) == ["check_y"]
    assert repeated_helper_gaps(report, errors, {**fingerprint, "feedback_sha256": "c"}) == []
    assert repeated_helper_gaps(report, ["No reachable training entrypoint"], fingerprint) == []
