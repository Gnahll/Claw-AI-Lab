"""Separate display topics from durable, mandatory execution requirements."""
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from researchclaw.config import RCConfig
from researchclaw.pipeline import executor
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.adapters import AdapterBundle
from .test_rc_config import _valid_config_data
from .test_agent_bridge_reference_config import _load_agent_bridge_module
from .test_discussion_budget import dr, FakeLLM

DETAILS = '只训练原模型和完整方法，各150轮。\n有效batch64；GPU1—3。\n测试集不得调参。'

def config(tmp_path):
    data = _valid_config_data()
    data['research']['details'] = DETAILS
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

def test_details_roundtrip_keeps_display_topic_clean(tmp_path):
    original = config(tmp_path)
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(original.to_dict(), allow_unicode=True), encoding='utf-8')
    restored = RCConfig.load(path, check_paths=False)
    assert restored.research.topic == 'Test topic'
    assert restored.research.details == DETAILS
    assert restored.research.prompt_topic.count(DETAILS) == 1
    assert 'Research details and execution requirements' in restored.research.prompt_topic

@pytest.mark.parametrize('value', [None, 123, [], {}])
def test_invalid_details_rejected(value, tmp_path):
    data = _valid_config_data(); data['research']['details'] = value
    with pytest.raises(ValueError, match='research.details must be a string'):
        RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

@pytest.mark.parametrize('stage', [Stage.TOPIC_INIT, Stage.HYPOTHESIS_GEN,
    Stage.EXPERIMENT_DESIGN, Stage.CODE_GENERATION, Stage.SANITY_CHECK, Stage.PAPER_DRAFT])
def test_details_reach_stages_and_resume_without_mutating_original(tmp_path, monkeypatch, stage):
    original = config(tmp_path)
    observed = []
    def capture(stage_dir, run_dir, cfg, adapters, **kwargs):
        observed.append(cfg.research.topic)
        return executor.StageResult(stage=stage, status=StageStatus.FAILED, error='fixture boundary')
    monkeypatch.setitem(executor._STAGE_EXECUTORS, stage, capture)
    monkeypatch.setattr(executor, '_read_prior_artifact', lambda *args: 'fixture input')
    monkeypatch.setattr(executor.LLMClient, 'from_rc_config', lambda *args: SimpleNamespace(
        config=SimpleNamespace(base_url='', api_key='')))
    for _ in range(2):
        executor.execute_stage(stage, run_dir=tmp_path/'run', run_id='details-test',
                               config=original, adapters=AdapterBundle(), auto_approve_gates=True)
    assert len(observed) == 2 and all(t.count(DETAILS) == 1 for t in observed)
    assert original.research.topic == 'Test topic' and original.research.details == DETAILS

@pytest.mark.parametrize('discussion', [True, False])
def test_lab_submission_persists_details_in_common_and_direction_configs(tmp_path, discussion):
    bridge = _load_agent_bridge_module()
    state = bridge.BridgeState(runs_base_dir=str(tmp_path))
    state.queues['init_to_idea'] = bridge.TaskQueue('init_to_idea', tmp_path/'queue.json')
    bridge.quick_submit_project(state, '简洁的方法主题', 'details-lab',
        research_angles=['跨分支选择性交互', '空间多尺度注意力'],
        discussion_enabled=discussion, research_details=DETAILS,
        path_overrides={'require_training': False})
    project = state.projects_dir()/'details-lab'
    meta = json.loads((project/'project_meta.json').read_text(encoding='utf-8'))
    assert meta['topic'] == '简洁的方法主题' and meta['research_details'] == DETAILS
    assert meta['research_workflow'] == ('discussion_joint' if discussion else 'independent_joint')
    paths = [Path(meta['config_path']), *[Path(t.config_path) for t in state.queues['init_to_idea'].tasks]]
    assert len(paths) == 3
    for path in paths:
        saved = yaml.safe_load(path.read_text(encoding='utf-8'))
        assert saved['research']['details'] == DETAILS
        assert DETAILS not in saved['research']['topic']
        assert RCConfig.load(path, check_paths=False).research.prompt_topic.endswith(DETAILS)

def test_discussion_every_round_preserves_full_details(tmp_path):
    llm = FakeLLM()
    dr.run_discussion(llm, 'method\n\n'+DETAILS, {'A':'evidence a','B':'evidence b'}, 3,
                      tmp_path, retry_delay=0)
    assert len(llm.prompts) == 3
    assert all(DETAILS in user for _, user, _ in llm.prompts)

def test_oversized_discussion_requirements_are_not_silently_truncated(tmp_path):
    llm = FakeLLM()
    with pytest.raises(ValueError, match='topic/details exceed'):
        dr.run_discussion(llm, DETAILS*1000, {'A':'a','B':'b'}, 3, tmp_path, retry_delay=0)
    assert not llm.prompts

def test_joint_integration_cli_uses_details_after_restart(tmp_path, monkeypatch):
    from .test_research_integration import integration
    original = config(tmp_path)
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(original.to_dict()), encoding='utf-8')
    request = tmp_path/'request.json'; request.write_text('{"topic":"stale"}')
    seen=[]
    monkeypatch.setattr(integration, 'create_llm_client', lambda cfg: object())
    monkeypatch.setattr(integration, 'integrate', lambda req, out, llm: seen.append(req['topic']))
    monkeypatch.setattr(sys, 'argv', ['integrate', '--request', str(request), '--config', str(path),
                                    '--output', str(tmp_path/'output')])
    integration.main()
    assert seen == [original.research.prompt_topic]
