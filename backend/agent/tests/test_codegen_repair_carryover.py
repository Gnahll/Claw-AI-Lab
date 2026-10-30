import json
from types import SimpleNamespace
from unittest.mock import Mock

import yaml

from researchclaw.pipeline.codegen.readiness import codegen_inputs_fingerprint
from researchclaw.pipeline.codegen.strategies import claw_agent
from researchclaw.pipeline.codegen.strategies.claw_agent import ClawAgentStrategy
from researchclaw.pipeline.codegen.turn_loop import TurnResult
from researchclaw.pipeline.codegen.types import CodegenContext

PLAN = yaml.safe_dump({'proposed_methods': [{'required_helpers': ['alpha', 'beta', 'gamma']}]})
REAL = 'def {0}(x):\n    return x * 2 + 1\n'
STUB = 'def {0}(x):\n    pass\n'


def _context(tmp_path):
    run = tmp_path / 'run'
    stage = run / 'stage-11'
    stage.mkdir(parents=True)
    (run / 'CODEGEN.md').write_text('Implement the plan.')
    (run / 'human_feedback.jsonl').write_text(json.dumps({'targetLayer': 'all', 'content': 'Use GPUs 6,7.'}) + '\n')
    return CodegenContext(topic='t', exp_plan=PLAN, metric='f1', metric_direction='maximize',
                          time_budget_sec=100, mode='sandbox', run_dir=run, stage_dir=stage)


def _previous_attempt(ctx, **overrides):
    experiment = ctx.stage_dir / 'experiment'
    (experiment / 'lib').mkdir(parents=True)
    (experiment / 'main.py').write_text('from lib.ops import alpha\n')
    (experiment / 'lib/ops.py').write_text(REAL.format('alpha'))
    (experiment / 'EXPERIMENT_PLAN.yaml').write_text('stale: plan\n')
    report = {'ready': False, 'errors': ['Missing declared implementation helper: beta'],
              'files': ['main.py', 'lib/ops.py', 'EXPERIMENT_PLAN.yaml'],
              **codegen_inputs_fingerprint(ctx.exp_plan, ctx.run_dir), **overrides}
    (ctx.stage_dir / 'implementation_readiness.json').write_text(json.dumps(report))


def test_retry_with_unchanged_inputs_repairs_previous_files(tmp_path):
    ctx = _context(tmp_path)
    _previous_attempt(ctx)
    strategy = ClawAgentStrategy()
    ws = strategy._prepare_workspace(ctx, Mock())
    gaps = strategy._seed_previous_attempt(ctx, ws, Mock())
    assert gaps == ['Missing declared implementation helper: beta']
    assert (ws / 'lib/ops.py').read_text() == REAL.format('alpha')
    assert (ws / 'main.py').is_file()
    # Current inputs are never replaced by the old attempt's copies.
    assert (ws / 'EXPERIMENT_PLAN.yaml').read_text() == PLAN


def test_changed_feedback_or_plan_regenerates_from_scratch(tmp_path):
    ctx = _context(tmp_path)
    _previous_attempt(ctx)
    with open(ctx.run_dir / 'human_feedback.jsonl', 'a') as log:
        log.write(json.dumps({'targetLayer': 'all', 'content': 'Change the method.'}) + '\n')
    strategy = ClawAgentStrategy()
    ws = strategy._prepare_workspace(ctx, Mock())
    assert strategy._seed_previous_attempt(ctx, ws, Mock()) == []
    assert not (ws / 'main.py').exists()

    ctx2 = _context(tmp_path / 'other')
    _previous_attempt(ctx2, plan_sha256='0' * 64)
    ws2 = strategy._prepare_workspace(ctx2, Mock())
    assert strategy._seed_previous_attempt(ctx2, ws2, Mock()) == []


def test_ready_or_unsafe_reports_are_not_carried_over(tmp_path):
    ctx = _context(tmp_path)
    _previous_attempt(ctx, ready=True)
    strategy = ClawAgentStrategy()
    ws = strategy._prepare_workspace(ctx, Mock())
    assert strategy._seed_previous_attempt(ctx, ws, Mock()) == []

    ctx2 = _context(tmp_path / 'other')
    _previous_attempt(ctx2, files=['../outside.py', '/etc/passwd'])
    ws2 = strategy._prepare_workspace(ctx2, Mock())
    assert strategy._seed_previous_attempt(ctx2, ws2, Mock()) == []


class _ScriptedLoop:
    """Each run_turn writes the next scripted set of helper bodies."""
    script: list[dict[str, str]] = []
    created = 0
    messages: list[str] = []

    def __init__(self, workspace, **_):
        type(self).created += 1
        self.workspace = workspace

    def set_exp_plan(self, plan):
        pass

    def run_turn(self, message):
        type(self).messages.append(message)
        helpers = type(self).script.pop(0)
        source = ''.join((REAL if real else STUB).format(name) for name, real in helpers.items())
        (self.workspace / 'main.py').write_text(source)
        return TurnResult(files={'main.py': source}, iterations=1, tool_calls=1)


def _generate(tmp_path, monkeypatch, script):
    ctx = _context(tmp_path)
    _ScriptedLoop.script = list(script)
    _ScriptedLoop.created = 0
    _ScriptedLoop.messages = []
    monkeypatch.setattr(claw_agent, 'ClawTurnLoop', _ScriptedLoop)
    monkeypatch.setattr(claw_agent, 'build_system_prompt', lambda ctx: 'system')
    monkeypatch.setattr(claw_agent, 'build_user_message', lambda ctx: 'task')
    monkeypatch.setattr(claw_agent.single_run, 'is_single_run', lambda config: False)
    config = SimpleNamespace(experiment=SimpleNamespace(sandbox=SimpleNamespace(python_path=''),
                                                        require_training=False))
    llm = SimpleNamespace(config=SimpleNamespace(primary_model='m', base_url='http://x'))
    return ClawAgentStrategy().generate(ctx, config, llm, Mock(llm_calls=0)), ctx


def test_repair_continues_while_gaps_shrink(tmp_path, monkeypatch):
    result, ctx = _generate(tmp_path, monkeypatch, [
        {'alpha': False, 'beta': False, 'gamma': False},
        {'alpha': True, 'beta': False, 'gamma': False},
        {'alpha': True, 'beta': True, 'gamma': False},
        {'alpha': True, 'beta': True, 'gamma': True},
    ])
    assert not result.error
    assert len(_ScriptedLoop.messages) == 4
    # Later rounds start a fresh conversation over the files on disk.
    assert _ScriptedLoop.created == 3
    assert 'Edit them rather than starting over' in _ScriptedLoop.messages[2]
    log = json.loads((ctx.stage_dir / 'claw_agent_log.json').read_text())
    assert log['success'] and log['iterations'] == 4


def test_repair_stops_when_a_round_makes_no_progress(tmp_path, monkeypatch):
    result, _ = _generate(tmp_path, monkeypatch, [
        {'alpha': True, 'beta': False, 'gamma': False},
        {'alpha': True, 'beta': True, 'gamma': False},
        {'alpha': True, 'beta': True, 'gamma': False},
        {'alpha': True, 'beta': True, 'gamma': True},
    ])
    assert 'gamma' in result.error
    assert len(_ScriptedLoop.messages) == 3
