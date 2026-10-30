import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

from researchclaw.pipeline.codegen.strategies.claw_agent import ClawAgentStrategy
from researchclaw.pipeline.codegen.types import CodegenContext


def test_registered_inputs_and_latest_feedback_survive_workspace_creation(tmp_path):
    run = tmp_path / 'run'
    registered = run / 'revision-context'
    registered.mkdir(parents=True)
    data = b'fixture_count: 32\nthreshold: 0.70\n'
    (registered / 'FIXED_PROTOCOL.yaml').write_bytes(data)
    (registered / 'scientific_readiness.json').write_text('{"status":"FAIL"}')
    (run / 'CODEGEN.md').write_text('Implement both conditions.')
    (run / 'human_feedback.jsonl').write_text(json.dumps({
        'targetLayer': 'all', 'content': 'Use GPUs6,7; retain both200epoch runs.'})+'\n')
    stage = run / 'stage-11'
    ctx = CodegenContext(topic='old four GPU topic', exp_plan='{}', metric='f1',
                         metric_direction='maximize', time_budget_sec=100,
                         mode='sandbox', run_dir=run, stage_dir=stage)
    ws = ClawAgentStrategy()._prepare_workspace(ctx, Mock())
    assert (ws / 'revision-context/FIXED_PROTOCOL.yaml').read_bytes() == data
    assert hashlib.sha256((ws / 'revision-context/FIXED_PROTOCOL.yaml').read_bytes()).digest() == hashlib.sha256(data).digest()
    assert json.loads((ws / 'revision-context/scientific_readiness.json').read_text())['status'] == 'FAIL'
    assert 'GPUs6,7' in (ws / 'CURRENT_FEEDBACK.md').read_text()
    assert (registered / 'FIXED_PROTOCOL.yaml').read_bytes() == data


def test_fresh_project_does_not_require_revision_context(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'CODEGEN.md').write_text('Implement both conditions.')
    ctx = CodegenContext(topic='fresh project', exp_plan='{}', metric='f1',
                         metric_direction='maximize', time_budget_sec=100,
                         mode='sandbox', run_dir=run, stage_dir=run/'stage-11')
    ws = ClawAgentStrategy()._prepare_workspace(ctx, Mock())
    assert (ws / 'EXPERIMENT_PLAN.yaml').is_file()
    assert not (ws / 'CURRENT_FEEDBACK.md').exists()


def test_registered_fixture_read_access_requires_matching_registration(tmp_path):
    import yaml
    bank = tmp_path / 'bank'
    bank.mkdir()
    data = b'{"images": []}'
    (bank / 'registration.json').write_bytes(data)
    spec = {'directory': str(bank), 'registration_sha256': hashlib.sha256(data).hexdigest()}
    ctx = CodegenContext(topic='fixture audit', exp_plan=yaml.safe_dump({'fixed_inputs': {'fixtures': spec}}),
                         metric='f1', metric_direction='maximize', time_budget_sec=100, mode='sandbox')
    assert bank.resolve() in ClawAgentStrategy._build_allowed_reads(ctx)
    (bank / 'registration.json').write_bytes(b'changed')
    assert bank.resolve() not in ClawAgentStrategy._build_allowed_reads(ctx)
