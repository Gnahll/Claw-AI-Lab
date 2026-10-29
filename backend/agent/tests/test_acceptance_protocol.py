"""First-use acceptance policy and the real S14 failure shape, without a GPU."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from researchclaw.pipeline.acceptance import (
    ACCEPTANCE_POLICY, plan_issues, pretraining_blocker, registered_context,
)
from researchclaw.pipeline.stages import StageStatus


def protocol():
    return {"acceptance_protocol": {"version": 1, "checks": [
        {"id": "finite_gradients", "kind": "engineering", "phase": "before_training",
         "blocking": True, "required_helpers": ["check_gradients"],
         "definition": {"metric": "finite nonzero module gradients", "inputs": "training batch",
                        "aggregation": "every required module", "failure_action": "stop with evidence"}},
        {"id": "H1", "kind": "hypothesis", "phase": "after_training", "blocking": False,
         "definition": {"metric": "AUROC", "inputs": "registered diagnostic samples",
                        "aggregation": "macro over images", "threshold": 0.70,
                        "failure_action": "report the observed negative result"}},
    ]}}


def test_new_user_hypothesis_can_fail_without_a_training_prerequisite():
    plan = protocol()
    plan["common_training"] = {"epochs": 150, "per_gpu_batch": None, "precision": None}
    assert plan_issues(plan, require_protocol=True) == []
    assert "negative results are valid" in ACCEPTANCE_POLICY.lower()


def test_generated_hypothesis_must_not_be_a_blocking_gate():
    plan = protocol()
    plan["acceptance_protocol"]["checks"][1].update(phase="before_training", blocking=True)
    assert any("hypotheses cannot" in issue for issue in plan_issues(plan))


def test_explicit_user_gate_stays_mandatory_and_exact_quote_is_required():
    quote = "H1 AUROC must reach 0.70 before training begins."
    plan = protocol()
    gate = plan["acceptance_protocol"]["checks"][1]
    gate.update(kind="user_required", phase="before_training", blocking=True,
                required_helpers=["registered_H1"], user_requirement_quote=quote)
    original = deepcopy(plan)
    assert plan_issues(plan, user_text=quote) == []
    assert plan == original
    assert any("exact user_requirement_quote" in issue for issue in plan_issues(plan, user_text="No special gate"))


@pytest.mark.parametrize("change", [
    {"status": {"implementation_ready": False, "training_authorized": False}},
    {"pretraining_acceptance": {"blocking_missing_contract": {"scoring": ["H2 undefined"]}}},
    {"implementation_handoff": {"protocol_conflicts": ["FP16-first conflicts with BF16-first"]}},
])
def test_original_s14_unresolved_plan_is_rejected_before_codegen(change):
    plan = {**protocol(), **change}
    original = deepcopy(plan)
    assert plan_issues(plan)
    assert plan == original  # Never migrate or remove the old gates.


@pytest.mark.parametrize("value", [None, {}, {"version": True, "checks": []}, {"version": 1, "checks": []}])
def test_missing_or_malformed_new_protocol_cannot_pass(value):
    assert plan_issues({"acceptance_protocol": value}, require_protocol=True)


def test_legacy_plan_does_not_require_silent_protocol_migration():
    plan = {"pretraining_acceptance": {"mandatory_thresholds": {"H1": {"minimum_AUROC": .7}}}}
    assert plan_issues(plan) == []
    assert plan["pretraining_acceptance"]["mandatory_thresholds"]["H1"]["minimum_AUROC"] == .7


def test_new_contract_cannot_silently_downgrade_a_preserved_old_gate():
    plan = protocol()
    original = {"minimum_AUROC": .70, "scope": "original registered cohorts"}
    plan["pretraining_acceptance"] = {"mandatory_thresholds": {"H1": original}}
    assert any("legacy_gate" in error for error in plan_issues(plan))
    check = plan["acceptance_protocol"]["checks"][1]
    quote = "Keep H1 AUROC >= 0.70 as a mandatory pretraining gate."
    check.update(kind="user_required", phase="before_training", blocking=True,
                 legacy_gate="H1", required_helpers=["registered_H1"], user_requirement_quote=quote)
    check["definition"]["thresholds"] = dict(original)
    assert plan_issues(plan, user_text=quote) == []
    check["definition"]["thresholds"]["minimum_AUROC"] = .49
    assert any("cannot change thresholds" in error for error in plan_issues(plan, user_text=quote))


def test_unmeasured_readiness_is_not_an_unresolved_definition():
    plan = protocol()
    plan["status"] = {"training_authorized": False, "training_completed": False}
    assert plan_issues(plan) == []


def test_registered_definitions_are_supplied_whole_or_rejected(tmp_path):
    path = tmp_path / "revision-context/FIXED_PROTOCOL.yaml"
    path.parent.mkdir()
    path.write_text("registered_fixture: exact-original-32\nlast_required_definition: must survive\n", encoding="utf-8")
    text = registered_context(tmp_path)
    assert "last_required_definition: must survive" in text and "sha256=" in text
    with pytest.raises(ValueError, match="not a silently truncated"):
        registered_context(tmp_path, max_chars=20)


def test_s9_rejects_incomplete_definitions_and_includes_registered_context(tmp_path, monkeypatch):
    from researchclaw.pipeline import executor
    from researchclaw.config import RCConfig
    from .test_rc_config import _valid_config_data
    config = RCConfig.from_dict(_valid_config_data(), project_root=tmp_path, check_paths=False)
    config = replace(config, experiment=replace(config.experiment, require_training=True))
    plan = {**protocol(), "objectives": ["fair comparison"], "datasets": ["local"],
            "baselines": ["original"], "proposed_methods": ["complete"], "ablations": [],
            "metrics": ["F1"], "risks": ["may not improve"], "compute_budget": {"epochs": 150},
            "status": {"implementation_ready": False}}
    path = tmp_path / "revision-context/FIXED_PROTOCOL.yaml"
    path.parent.mkdir()
    path.write_text("frozen_fixture: original32", encoding="utf-8")
    prompts = []
    monkeypatch.setattr(executor, "_build_context_preamble", lambda *a, **k: "")
    def respond(llm, system, user, **kwargs):
        prompts.append(user)
        return SimpleNamespace(content=yaml.safe_dump(plan))
    monkeypatch.setattr(executor, "_chat_with_prompt", respond)
    stage = tmp_path / "stage-09"
    stage.mkdir()
    result = executor._execute_experiment_design(stage, tmp_path, config, object(), llm=object())
    assert result.status == StageStatus.FAILED and result.decision == "revise_design"
    assert not (stage / "exp_plan.yaml").exists()
    assert (stage / "exp_plan_draft.yaml").is_file()
    assert json.loads((stage / "acceptance_review.json").read_text())["retryable"] is False
    assert ACCEPTANCE_POLICY in prompts[0] and "frozen_fixture: original32" in prompts[0]


def test_s9_first_user_needs_no_incident_history_or_custom_gates(tmp_path, monkeypatch):
    from researchclaw.pipeline import executor
    from researchclaw.config import RCConfig
    from .test_rc_config import _valid_config_data
    data = _valid_config_data()
    data["research"]["topic"] = "Image localization with cross-branch adaptive attention"
    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)
    config = replace(config, experiment=replace(config.experiment, require_training=True,
                     benchmark_agent=replace(config.experiment.benchmark_agent, enabled=False)))
    plan = {**protocol(), "objectives": ["fair comparison"], "datasets": ["provided"],
            "baselines": ["original"], "proposed_methods": ["complete"], "ablations": [],
            "metrics": ["F1"], "risks": ["may not improve"], "compute_budget": {"epochs": 150}}
    monkeypatch.setattr(executor, "_build_context_preamble", lambda *a, **k: "first use")
    monkeypatch.setattr(executor, "_chat_with_prompt", lambda *a, **k: SimpleNamespace(content=yaml.safe_dump(plan)))
    stage = tmp_path / "stage-09"
    stage.mkdir()
    result = executor._execute_experiment_design(stage, tmp_path, config, object(), llm=object())
    assert result.status == StageStatus.DONE
    assert yaml.safe_load((stage / "exp_plan.yaml").read_text())["acceptance_protocol"] == plan["acceptance_protocol"]


@pytest.mark.parametrize("runtime_name,stage_name", [("sanity_check", "stage-12"), ("experiment_run", "stage-14")])
def test_resumed_legacy_plan_stops_before_llm_or_training(tmp_path, runtime_name, stage_name):
    from researchclaw.pipeline.sanity_check.runtime import SanityCheckRuntime
    from researchclaw.pipeline.experiment_run.runtime import ExperimentRunRuntime
    path = tmp_path / "stage-09/exp_plan.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump({"status": {"implementation_ready": False}}), encoding="utf-8")
    runtime = SanityCheckRuntime() if runtime_name == "sanity_check" else ExperimentRunRuntime()
    # No configured LLM/GPU: the definition check must happen before either is used.
    result = runtime.execute(tmp_path / stage_name, tmp_path, SimpleNamespace(), object())
    assert result.status == StageStatus.FAILED and result.decision == "revise_design"
    assert "implementation_ready=false" in result.error


def test_s11_missing_acceptance_checker_does_not_pass_as_complete():
    from researchclaw.pipeline.codegen.readiness import declared_interface_gaps
    plan = yaml.safe_dump(protocol())
    assert declared_interface_gaps({"main.py": "print('training')"}, plan) == [
        "Missing declared implementation helper: check_gradients"]
    assert declared_interface_gaps({"main.py": "def check_gradients(model):\n    return all(p.grad.isfinite().all() for p in model.parameters())"}, plan) == []


@pytest.mark.parametrize("exit_code", [0, 1])
def test_real_subprocess_blocker_survives_exit_without_becoming_success(tmp_path, monkeypatch, exit_code):
    from researchclaw.training.execution import run_experiment
    from .test_original_workflow_training import configuration
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    blocked = {"status": "blocked_pretraining", "completed_training_runs": 0,
               "blocker": {"reason": "retained_H1_failed", "required_gate_status": {"H1": "FAIL", "H2": "NOT_IMPLEMENTED"}}}
    (tmp_path / "main.py").write_text("import json\nfrom pathlib import Path\n"
        f"Path('results.json').write_text(json.dumps({blocked!r}))\nraise SystemExit({exit_code})\n", encoding="utf-8")
    report = run_experiment(tmp_path, tmp_path / "evidence", configuration())
    assert not report["success"] and report["failure_type"] == "pretraining_blocked"
    assert report["retryable"] is False and report["results"] == blocked
    assert "retained_H1_failed" in report["errors"][0]
    assert json.loads((tmp_path / "evidence/execution_report.json").read_text())["returncode"] == exit_code


def test_generic_failed_metrics_are_not_reclassified_as_pretraining():
    assert pretraining_blocker({"status": "failed", "reason": "CUDA error"}) is None
    assert pretraining_blocker({"status": "done", "H1": {"AUROC": .49}}) is None


def test_s14_skips_paid_repair_and_second_execution_for_acceptance_failure(tmp_path, monkeypatch):
    from researchclaw.pipeline.experiment_run import runtime as rt
    from .test_experiment_run_repair_guard import _Loop, _Session, _fake_runs
    calls = _fake_runs(monkeypatch, [{"success": False, "failure_type": "pretraining_blocked",
                                    "retryable": False, "errors": ["H1 failed"]}])
    loop = _Loop([])
    report, _ = rt.ExperimentRunRuntime._supervised_run(tmp_path, tmp_path, None, loop, "repair", _Session())
    assert not report["success"] and calls == ["attempt-1"] and loop.messages == []
