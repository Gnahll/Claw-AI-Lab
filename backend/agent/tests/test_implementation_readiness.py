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


@pytest.mark.parametrize('source', [
    'def score_H2(*args): return {"measured": False, "status": "blocked"}',
    'def score_H2(x): return x',
    'def score_H2(x):\n    report = {"measured": True, "status": "PASS"}\n    return report',
    'def score_H2(*args):\n    y = engineering_random_probe()\n    return {"status": "PASS", "response": float(y.mean())}',
])
def test_required_checker_rejects_metadata_identity_and_input_free_pass(source):
    plan = 'acceptance_protocol: {checks: [{required_helpers: [score_H2]}]}'
    errors = declared_interface_gaps({'main.py': source}, plan)
    assert len(errors) == 1 and 'score_H2' in errors[0]


@pytest.mark.parametrize('source', [
    'def score_H2(x): return {"status": "PASS" if x.mean() >= .5 else "FAIL"}',
    'def score_H2(x):\n    assert x.mean() >= .5\n    return {"status": "PASS"}',
    'def score_H2(x): return registered_checker(x)',
    'def score_H2():\n    fixtures = load_registered_fixtures()\n    return evaluate_cohorts(fixtures)',
])
def test_real_checkers_are_not_rejected_merely_for_returning_a_pass_record(source):
    plan = 'acceptance_protocol: {checks: [{required_helpers: [score_H2]}]}'
    assert declared_interface_gaps({'main.py': source}, plan) == []


def test_s12_does_not_execute_or_accept_a_checker_replaced_by_random_probe(tmp_path, monkeypatch):
    from researchclaw.pipeline.sanity_check.runtime import _run_checked_smoke
    from researchclaw.training import execution
    (tmp_path / 'main.py').write_text('def score_H2(*args):\n    y = random_probe()\n    return {"status": "PASS", "value": y.mean()}')
    monkeypatch.setattr(execution, 'run_experiment', lambda *a, **kw: pytest.fail('must not run incomplete code'))
    result = _run_checked_smoke(tmp_path, tmp_path/'attempt', None,
                               'checks: [{required_helpers: [score_H2]}]')
    assert not result['success'] and not result['execution_started']
    assert 'unconditional PASS' in result['errors'][0]
    assert (tmp_path/'attempt/execution_report.json').is_file()


def test_s12_keeps_measured_smoke_execution_for_real_implementation(tmp_path, monkeypatch):
    from researchclaw.pipeline.sanity_check.runtime import _run_checked_smoke
    from researchclaw.training import execution
    (tmp_path/'main.py').write_text('def score_H2(x): return evaluate_registered(x)')
    def run(workspace, output_dir, config, **kwargs):
        assert kwargs == {'smoke': True, 'timeout': 180}
        return {'success': True, 'measured': True}
    monkeypatch.setattr(execution, 'run_experiment', run)
    assert _run_checked_smoke(tmp_path, tmp_path/'attempt', None,
                             'checks: [{required_helpers: [score_H2]}]')['measured']


@pytest.mark.parametrize('source', [
    'def main(): print("audit only")\nmain()',
    'def train(): optimizer.step()\ndef main(): print("blocked")\nmain()',
    'def smoke(): loss.backward()\ndef main(): smoke()\nmain()',
    'from pretrained import load_model\nload_model()',
])
def test_training_task_rejects_no_update_path_even_with_named_train_function(source):
    errors = declared_interface_gaps({'main.py': source}, '{}', require_training=True)
    assert len(errors) == 1 and 'No reachable training entrypoint' in errors[0]


@pytest.mark.parametrize('source', [
    'def train(): optimizer.step()\ndef main(): train()\nmain()',
    'trainer.fit(model, data)',
    'trainer.train()',
    'from author.train import main as train_author\ntrain_author()',
    'import author.training as engine\nengine.run()',
    'import subprocess\nsubprocess.run(["python", "author_train.py"])',
])
def test_training_task_accepts_reachable_updates_or_author_training_delegation(source):
    assert declared_interface_gaps({'main.py': source}, '{}', require_training=True) == []


def test_training_reachability_does_not_start_from_an_unused_helper_module():
    files = {'main.py': 'print("audit")', 'unused.py': 'optimizer.step()'}
    assert declared_interface_gaps(files, '{}', require_training=True)


def test_archived_copies_do_not_shadow_the_code_that_runs():
    # Seen on the server: S11 kept earlier versions under revision-context/;
    # their run_full_training (no update) replaced the real one in the call graph.
    files = {'main.py': 'from training_impl import run_full_training\nrun_full_training()',
             'training_impl.py': 'def run_full_training():\n    optimizer.step()',
             'revision-context/repair_old/training_impl.py': 'def run_full_training():\n    return audit()',
             'revision-context/repair_old/main.py': 'print("old audit")'}
    assert declared_interface_gaps(files, '{}', require_training=True) == []
    del files['training_impl.py']
    assert declared_interface_gaps(files, '{}', require_training=True)


def test_same_named_functions_in_copied_author_scripts_do_not_replace_mains_own():
    # Seen on the server: upstream_contract/{train,test}.py each define main();
    # sorted after main.py, they replaced main.py's main() in the call graph.
    files = {'main.py': 'def train_condition():\n    scaler.step(optimizer)\n'
                        'def main():\n    train_condition()\n'
                        'if __name__ == "__main__":\n    main()',
             'upstream_contract/test.py': 'def main():\n    evaluate()',
             'upstream_contract/train.py': 'def main():\n    print("author entry")'}
    assert declared_interface_gaps(files, '{}', require_training=True) == []
    files['main.py'] = 'def main():\n    print("audit")\nmain()'
    files['upstream_contract/train.py'] = 'def main():\n    optimizer.step()'
    assert declared_interface_gaps(files, '{}', require_training=True)


def test_explicit_imports_resolve_to_the_named_local_module():
    files = {'main.py': 'from engine.loop import run\nrun()',
             'engine/loop.py': 'def run():\n    optimizer.step()',
             'tools/loop.py': 'def run():\n    print("probe")'}
    assert declared_interface_gaps(files, '{}', require_training=True) == []
    files['main.py'] = 'from tools.loop import run\nrun()'
    assert declared_interface_gaps(files, '{}', require_training=True)


def test_inference_tasks_do_not_require_optimizer_or_trainer():
    assert declared_interface_gaps({'main.py': 'model.predict(data)'}, '{}') == []


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
