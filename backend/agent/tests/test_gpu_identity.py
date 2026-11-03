"""User GPU numbers are a count; the platform-allocated devices are never refused."""
from types import SimpleNamespace

from researchclaw.pipeline.acceptance import ACCEPTANCE_POLICY, normalize_gpu_budget, release_gpu_identity

CONFIG = SimpleNamespace(experiment=SimpleNamespace(gpu_uuid="GPU-aaa,GPU-bbb"))


def _blocked(reason):
    return {"failure_type": "pretraining_blocked", "reason": reason, "errors": ["blocked"]}


def test_physical_index_refusal_becomes_code_repair():
    for reason in ("Visible physical GPUs are not the explicitly authorized GPUs 1 and 2.",
                   "Allocated physical GPUs do not match explicitly authorized GPUs 1 and 2.",
                   "GPU UUIDs differ from the authorized devices",
                   "仅允许 GPU1、2"):
        report = _blocked(reason)
        assert release_gpu_identity(report, CONFIG), reason
        assert report["failure_type"] == "gpu_identity_blocked" and "2 device(s)" in report["errors"][-1]


def test_memory_or_scientific_blocks_stay_design_failures():
    for reason in ("Peak allocated GPU memory 83 GB exceeds the 80 GB per-GPU limit",
                   "Measured localization-only gradients violate joint_gradient_connectivity",
                   "Only 1 visible GPU; the plan needs 2"):
        report = _blocked(reason)
        assert not release_gpu_identity(report, CONFIG), reason
        assert report["failure_type"] == "pretraining_blocked"


def test_plan_budget_uses_platform_allocation():
    plan = {"compute_budget": {"authorized_gpus": [1, 2], "memory_limit_gb_per_gpu": 80}}
    notes = normalize_gpu_budget(plan, CONFIG)
    assert len(notes) == 1 and plan["compute_budget"]["gpu_count"] == 2
    assert "CUDA_VISIBLE_DEVICES" in plan["compute_budget"]["authorized_gpus"]
    assert normalize_gpu_budget(plan, CONFIG) == []
    assert normalize_gpu_budget({"compute_budget": {"authorized_gpus": [1]}},
                                SimpleNamespace(experiment=SimpleNamespace(gpu_uuid=""))) == []


def test_policy_tells_every_stage_numbers_are_a_count():
    assert "not which physical devices" in ACCEPTANCE_POLICY
