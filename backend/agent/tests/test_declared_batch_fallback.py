"""An OOM with a fallback batch declared in the plan is a code repair, not a failed design."""
from types import SimpleNamespace

import yaml

from researchclaw.pipeline.acceptance import declared_batch_settings, release_batch_fallback

USER = "优先预检每卡batch50、累积1步，若任一条件无法稳定运行，则两个条件共同改用每卡batch25、累积2步"
CONFIG = SimpleNamespace(research=SimpleNamespace(prompt_topic=USER), experiment=SimpleNamespace(runs_per_condition=3))
PLAN = {"training_protocol": {
    "batching": {"preferred": {"devices": 2, "gradient_accumulation": 1, "per_device_batch": 50},
                 "authorized_fallback": {"devices": 2, "gradient_accumulation": 2, "per_device_batch": 25}},
    "common_settings": {"effective_batch": 100}}}
OOM = {"failure_type": "pretraining_blocked", "errors": ["blocked"],
       "reason": "CUDA out of memory at per-GPU batch 50 x accumulation 1 (78.1 GB allocated)"}


def _write(tmp_path, plan):
    path = tmp_path / "exp_plan.yaml"
    path.write_text(yaml.safe_dump(plan, allow_unicode=True), encoding="utf-8")
    return path


def test_settings_parsed_from_plan_mapping_and_user_text():
    assert sorted(declared_batch_settings(yaml.safe_dump(PLAN))) == [(25, 2), (50, 1)]
    assert declared_batch_settings(USER) == [(50, 1), (25, 2)]
    assert declared_batch_settings("try the jointly specified 25-per-GPU, accumulation-2 implementation") == [(25, 2)]


def test_oom_with_declared_fallback_becomes_code_repair(tmp_path):
    report = dict(OOM)
    settings = release_batch_fallback(report, _write(tmp_path, PLAN), tmp_path, CONFIG)
    assert settings == [(50, 1), (25, 2)]
    assert report["failure_type"] == "declared_fallback_oom" and report["retryable"] is True
    assert "per-GPU batch 25 x accumulation 2" in report["errors"][-1]


def test_fallback_only_in_user_text_is_enough(tmp_path):
    report = dict(OOM)
    assert release_batch_fallback(report, _write(tmp_path, {"plan": "no batching"}), tmp_path, CONFIG)


def test_without_same_effective_batch_fallback_stays_blocked(tmp_path):
    no_text = SimpleNamespace(research=SimpleNamespace(prompt_topic=""), experiment=SimpleNamespace(runs_per_condition=3))
    for plan in ({"batching": {"preferred": {"per_device_batch": 50, "gradient_accumulation": 1}}},
                 {"batching": {"preferred": {"per_device_batch": 50, "gradient_accumulation": 1},
                               "fallback": {"per_device_batch": 25, "gradient_accumulation": 1}}}):
        report = dict(OOM)
        assert release_batch_fallback(report, _write(tmp_path, plan), tmp_path, no_text) == []
        assert report["failure_type"] == "pretraining_blocked"


def test_non_oom_block_stays_blocked(tmp_path):
    report = {**OOM, "reason": "Measured gradients violate joint_gradient_connectivity"}
    assert release_batch_fallback(report, _write(tmp_path, PLAN), tmp_path, CONFIG) == []
    assert report["failure_type"] == "pretraining_blocked"
    assert release_batch_fallback({**OOM, "failure_type": "runtime_error"}, _write(tmp_path, PLAN), tmp_path, CONFIG) == []
