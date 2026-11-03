"""A user-required check that cannot stop training without a quote is a warning, not a failed S9 design."""
from researchclaw.pipeline.acceptance import may_block_training, plan_issues, quote_found


def _definition(**extra):
    return {"metric": "m", "inputs": "test sets", "aggregation": "macro", "failure_action": "report", **extra}


def _plan(**user_check):
    check = {"id": "no_ablation", "kind": "user_required", "phase": "after_training", "blocking": False,
             "definition": _definition(), **user_check}
    return {"acceptance_protocol": {"version": 1, "checks": [
        {"id": "finite_gradients", "kind": "engineering", "phase": "before_training", "blocking": True,
         "required_helpers": ["check_gradients"], "definition": _definition(metric="finite gradients")},
        check]}}


TOPIC = ("Optional experiments were selected by the user; follow this choice exactly.\n"
         "- Not requested: do not plan or run any ablation experiment; state its absence as a limitation.\n"
         "- Not requested: do not plan or run any visualization experiment; state its absence as a limitation.\n")


def test_nonblocking_check_without_quote_is_only_a_warning():
    warnings: list[str] = []
    assert plan_issues(_plan(), user_text=TOPIC, warnings=warnings) == []
    assert len(warnings) == 1 and "cannot stop training" in warnings[0]
    assert plan_issues(_plan(), user_text=TOPIC) == []


def test_quote_joined_across_list_lines_matches():
    quote = ("Not requested: do not plan or run any ablation experiment; state its absence as a limitation. "
             "Not requested: do not plan or run any visualization experiment; state its absence as a limitation.")
    assert quote_found(quote, TOPIC)
    warnings: list[str] = []
    assert plan_issues(_plan(user_requirement_quote=quote), user_text=TOPIC, warnings=warnings) == []
    assert warnings == []


def test_blocking_user_check_still_needs_its_quote():
    plan = _plan(blocking=True, phase="before_training", required_helpers=["check_user"])
    assert any("exact user_requirement_quote" in issue for issue in plan_issues(plan, user_text=TOPIC))


def test_after_training_check_marked_blocking_is_only_a_warning():
    warnings: list[str] = []
    plan = _plan(blocking=True, user_requirement_quote="保存训练损失、验证指标、模型权重、实验记录")
    assert plan_issues(plan, user_text=TOPIC, warnings=warnings) == []
    assert len(warnings) == 1 and "cannot stop training" in warnings[0]


def test_invented_quote_never_matches_and_cannot_block():
    assert not quote_found("The user approved a 0.30 threshold before training.", TOPIC)
    check = _plan(blocking=True, phase="before_training", required_helpers=["h"],
                  user_requirement_quote="The user approved R >= 0.30 before training.",
                  definition=_definition(threshold=0.30))["acceptance_protocol"]["checks"][1]
    assert not may_block_training(check, TOPIC)


def test_missing_helpers_are_named_from_check_id():
    from researchclaw.pipeline.acceptance import fill_missing_helpers
    plan = _plan()
    plan["acceptance_protocol"]["checks"][0].pop("required_helpers")
    plan["acceptance_protocol"]["checks"].append(
        {"id": "split-protocol", "kind": "engineering", "phase": "before_training", "blocking": True,
         "required_helpers": ["check_split"], "definition": _definition()})
    notes = fill_missing_helpers(plan)
    checks = plan["acceptance_protocol"]["checks"]
    assert checks[0]["required_helpers"] == ["check_finite_gradients"]
    assert checks[2]["required_helpers"] == ["check_split"] and "required_helpers" not in checks[1]
    assert len(notes) == 1 and plan_issues(plan, user_text=TOPIC) == []


def test_recorded_ratio_check_may_run_after_training():
    plan = _plan()
    ratio = {"id": "relative_resource_cost", "kind": "engineering", "phase": "after_training", "blocking": False,
             "definition": _definition(metric="Complete-method divided by baseline wall-time and peak-memory ratios.")}
    plan["acceptance_protocol"]["checks"].append(ratio)
    assert plan_issues(plan, user_text=TOPIC) == []
    ratio["definition"]["metric"] = "peak GPU memory"
    assert any("engineering prerequisites must block" in i for i in plan_issues(plan, user_text=TOPIC))
