from researchclaw.pipeline.codegen.readiness import declared_interface_gaps
import json
from contextlib import nullcontext
from types import SimpleNamespace
import pytest


PLAN = '''proposed_methods:
  - name: two required mechanisms
    implementation_spec:
      required_distinct_helpers:
        - compute_directional_suppression
        - compute_joint_scale_key_attention
'''


def test_audit_only_main_does_not_satisfy_declared_method_interfaces():
    files = {'main.py': 'def main():\n    print("resource audit")\n    raise SystemExit("blocked")\n'}
    errors = declared_interface_gaps(files, PLAN)
    assert len(errors) == 2
    assert any('compute_joint_scale_key_attention' in error for error in errors)


def test_named_stubs_are_not_implementations():
    files = {'adapter.py': 'def compute_directional_suppression():\n    """todo"""\n    pass\n'
                          'def compute_joint_scale_key_attention():\n    raise NotImplementedError\n'}
    assert len(declared_interface_gaps(files, PLAN)) == 2


def test_declared_helpers_can_be_methods_or_real_imports():
    files = {'adapter.py': 'from real_implementation import compute_directional_suppression\n'
                          'class Adapter:\n    def compute_joint_scale_key_attention(self, logits):\n        return logits.softmax(-1)\n'}
    assert declared_interface_gaps(files, PLAN) == []


def test_generic_plan_without_named_interfaces_does_not_impose_architecture():
    assert declared_interface_gaps({'main.py': 'print("run actual baseline")'}, 'training: {epochs: 200}') == []


def test_resource_blocker_records_current_gpu_and_never_starts_code(tmp_path, monkeypatch):
    from researchclaw.training import resources
    from researchclaw.pipeline.stages import Stage, StageStatus
    gpu = 'GPU-267e1a41-2b4c-fe25-5291-0dc922c24454'
    monkeypatch.setattr(resources.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(resources, 'job_lock', lambda *a: nullcontext())
    def occupied(value):
        assert value == gpu
        raise RuntimeError('Allocated GPU is occupied; no training was started')
    monkeypatch.setattr(resources, 'require_idle_gpu', occupied)
    @resources.with_execution_resources(Stage.SANITY_CHECK)
    def stage(*args, **kwargs):
        pytest.fail('A resource-blocked stage must not execute code')
    output = tmp_path / 'stage-12'
    result = stage(output, tmp_path, SimpleNamespace(experiment=SimpleNamespace(gpu_uuid=gpu)), None)
    assert result.status == StageStatus.FAILED and result.decision == 'retry'
    report = json.loads((output / 'resource_blocker.json').read_text())
    assert report['checking_gpu_uuid'] == gpu and report['execution_started'] is False
