import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.pipeline.iterative_refine import runtime
from researchclaw.pipeline.stages import REFINE_REQUEST_FILE, StageStatus
from researchclaw.training.verified_run import create_receipt, source_manifest


def verified(tmp_path):
    run = tmp_path / 'project'
    workspace = run / 'stage-14/run_workspace_1'
    experiment = run / 'stage-11/experiment'
    report_path = run / 'stage-14/runs/attempt-1/execution_report.json'
    for path in (workspace / 'outputs/predictions', experiment, report_path.parent, run / 'stage-09'):
        path.mkdir(parents=True)
    (workspace / 'main.py').write_text('from model import train\ntrain()\n')
    (workspace / 'model.py').write_text('def train(): pass\n')
    for filename in ('main.py', 'model.py'):
        (experiment / filename).write_bytes((workspace / filename).read_bytes())
    (run / 'stage-09/exp_plan.yaml').write_text('epochs: 200\n')
    (workspace / 'results.json').write_text(json.dumps({'pixel_f1': 0.75, 'completed_epochs': 200}))
    (workspace / 'outputs/model.pt').write_bytes(b'validated final weights')
    (workspace / 'outputs/predictions/mask.npy').write_bytes(b'validated raw predictions')
    report = {'success': True, 'smoke': False, 'require_training': True, 'source_manifest': source_manifest(workspace), 'training': {
        'parameter_update_observed': True, 'checkpoint_verified': True, 'optimizer_steps': 40200}}
    report_path.write_text(json.dumps(report))
    config = SimpleNamespace(experiment=SimpleNamespace(require_training=True, metric_key='pixel_f1',
        metric_direction='maximize', runs_per_condition=1, optional_experiments=()))
    receipt = create_receipt(workspace, run, report_path, config)
    path = run / 'stage-14/runs/verified_run_receipt.json'
    path.write_text(json.dumps(receipt))
    # Exercise JSON roundtrip of tuple-valued config fields.
    return run, workspace, experiment, config, json.loads(path.read_text())


def test_s15_integrity_review_never_launches_llm_or_retrains(tmp_path, monkeypatch):
    run, workspace, experiment, config, receipt = verified(tmp_path)
    monkeypatch.setattr(runtime, 'run_experiment', lambda *a, **kw: pytest.fail('Must not retrain'))
    monkeypatch.setattr(runtime, 'AgentTurnLoop', lambda *a, **kw: pytest.fail('No repair turn needed'))
    result = runtime.IterativeRefineRuntime().execute(run / 'stage-15', run, config, None)
    assert result.status == StageStatus.DONE
    assert (run / 'stage-15/results.json').read_bytes() == (workspace / 'results.json').read_bytes()
    assert (run / 'stage-15/experiment_final/model.py').read_bytes() == (experiment / 'model.py').read_bytes()
    log = json.loads((run / 'stage-15/refinement_log.json').read_text())
    assert log['adopted'] and not log['retrained']
    assert log['verification_mode'] == 'verified_s14_integrity_review'


@pytest.mark.parametrize('changed', ['code', 'executed_code', 'weights', 'predictions', 'results', 'plan', 'report', 'config'])
def test_changed_evidence_blocks_reuse_without_retraining(tmp_path, monkeypatch, changed):
    run, workspace, experiment, config, receipt = verified(tmp_path)
    paths = {'code': experiment / 'model.py', 'executed_code': workspace / 'model.py',
        'weights': workspace / 'outputs/model.pt', 'predictions': workspace / 'outputs/predictions/mask.npy',
        'results': workspace / 'results.json', 'plan': run / 'stage-09/exp_plan.yaml',
        'report': Path(receipt['report_path'])}
    if changed == 'config':
        config.experiment.runs_per_condition = 3
    else:
        paths[changed].write_bytes(b'changed')
    monkeypatch.setattr(runtime, 'run_experiment', lambda *a, **kw: pytest.fail('Must stop, not retrain'))
    result = runtime.IterativeRefineRuntime().execute(run / 'stage-15', run, config, None)
    assert result.status == StageStatus.FAILED
    assert (run / 'stage-15/verification_blocker.txt').is_file()
    assert not (run / 'stage-15/experiment_final').exists()


def test_missing_predictions_or_smoke_run_cannot_create_receipt(tmp_path):
    run, workspace, experiment, config, receipt = verified(tmp_path)
    (workspace / 'outputs/predictions/mask.npy').unlink()
    with pytest.raises(ValueError, match='predictions'):
        create_receipt(workspace, run, Path(receipt['report_path']), config)
    report = json.loads(Path(receipt['report_path']).read_text())
    report['smoke'] = True
    Path(receipt['report_path']).write_text(json.dumps(report))
    with pytest.raises(ValueError, match='successful supervised training'):
        create_receipt(workspace, run, Path(receipt['report_path']), config)


def test_refine_request_does_not_automatically_retrain_completed_models(tmp_path):
    run, workspace, experiment, config, receipt = verified(tmp_path)
    stage = run / 'stage-15'
    stage.mkdir()
    (stage / REFINE_REQUEST_FILE).write_text('Add another baseline')
    result = runtime.IterativeRefineRuntime().execute(stage, run, config, None)
    assert result.status == StageStatus.FAILED
    assert 'additional experiments' in result.error
    assert (workspace / 'outputs/model.pt').read_bytes() == b'validated final weights'


def test_real_cpu_training_receipt_can_be_reviewed_without_second_training(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from .test_original_workflow_training import configuration
    from researchclaw.training.execution import run_experiment

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    run = tmp_path / 'project'
    workspace = run / 'stage-14/run_workspace_cpu'
    workspace.mkdir(parents=True)
    script = '''import json, torch
from pathlib import Path
torch.manual_seed(42)
model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.tensor([[1., 2.], [-1., 0.], [0., 1.]])
y = torch.tensor([[1.], [0.], [0.3]])
for _ in range(2):
    optimizer.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()
Path('outputs').mkdir(exist_ok=True)
torch.save(model.state_dict(), 'outputs/model.pt')
Path('outputs/predictions.json').write_text(json.dumps({'predictions': model(x).detach().tolist()}))
Path('results.json').write_text(json.dumps({'metric': loss.item(), 'completed_epochs': 2}))
Path('trained_once.txt').write_text('one supervised execution')
'''
    (workspace / 'main.py').write_text(script)
    experiment = run / 'stage-11/experiment'
    experiment.mkdir(parents=True)
    (experiment / 'main.py').write_text(script)
    config = configuration(training=True)
    evidence = run / 'stage-14/runs/attempt-1'
    result = run_experiment(workspace, evidence, config)
    assert result['success'], result['errors']
    receipt = create_receipt(workspace, run, evidence / 'execution_report.json', config)
    (run / 'stage-14/runs/verified_run_receipt.json').write_text(json.dumps(receipt))
    marker_time = (workspace / 'trained_once.txt').stat().st_mtime_ns
    monkeypatch.setattr(runtime, 'run_experiment', lambda *a, **kw: pytest.fail('Second training forbidden'))
    reviewed = runtime.IterativeRefineRuntime().execute(run / 'stage-15', run, config, None)
    assert reviewed.status == StageStatus.DONE
    assert (workspace / 'trained_once.txt').stat().st_mtime_ns == marker_time
