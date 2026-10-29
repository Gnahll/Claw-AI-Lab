from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from .test_agent_bridge_reference_config import _load_agent_bridge_module


services = Path(__file__).resolve().parents[2] / "services"
sys.path.insert(0, str(services))
spec = importlib.util.spec_from_file_location("hypothesis_integration_tests", services / "hypothesis_integration.py")
integration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integration)


def payload():
    return {"status": "ready", "reason": "Complementary mechanisms", "directions": [
        {"id": ident, "mechanism": mechanism, "evidence": "Supplied literature evidence",
         "novelty_risk": "Requires comparison with prior work", "integration": "Join in backbone",
         "verification": "Measure against the baseline"}
        for ident, mechanism in (("D1", "Reliability-gated cross-branch exchange"),
                                 ("D2", "Input-dependent attention scale weights"))],
        "joint_method": "Combine reliable cross-branch exchange and adaptive attention",
        "experiment_plan": "Use the same split, final checkpoint, and training budget",
        "readiness": "design_only", "training_completed": False,
        "interface_contract": "Keep branch-specific values; exchange constraints and route attention scales.",
        "reconciliations": [{"conflict": "Value transfer versus connectivity-only exchange",
                             "resolution": "Use directional connectivity constraints to preserve both direction goals"}],
        "pending_checks": [{"stage": "S9-S13 before full training", "check": "Verify source interface, gradients and memory",
                            "stop_condition": "No compatible interface or insufficient resources"}]}


@pytest.mark.parametrize("defect", ["missing", "duplicate", "blocked", "empty_evidence"])
def test_integration_rejects_omitted_or_unsupported_directions(defect):
    result = payload()
    if defect == "missing":
        result["directions"].pop()
    elif defect == "duplicate":
        result["directions"][1]["id"] = "D1"
    elif defect == "blocked":
        result["status"] = "blocked"
    else:
        result["directions"][1]["evidence"] = ""
    with pytest.raises(ValueError):
        integration.validate(result, ["D1", "D2"])


def test_joint_output_preserves_both_directions_literature_and_originals(tmp_path):
    sources = []
    originals = []
    for index in (1, 2):
        run = tmp_path / f"run-{index}"
        (run / "stage-08").mkdir(parents=True)
        original = "Hypothesis " + str(index) + (" background" * 200 if index == 1 else " adaptive attention")
        (run / "stage-08/hypotheses.md").write_text(original)
        originals.append(original)
        for stage, name in ((4, "candidates.jsonl"), (5, "shortlist.jsonl")):
            dest = run / f"stage-{stage:02d}"
            dest.mkdir()
            (dest / name).write_text(json.dumps({"title": f"Forensics {index}", "source": "arxiv"}) + "\n")
        (run / "stage-04/references.bib").write_text(f"@article{{ref{index}, title={{Forensics {index}}}}}")
        sources.append({"name": f"Mechanism {index}", "run_dir": str(run)})
    discussion = tmp_path / "discussion"
    discussion.mkdir()
    (discussion / "consensus_synthesis.md").write_text("Joint discussion consensus")
    prompts = []
    class LLM:
        def chat(self, messages, **kwargs):
            prompts.append(messages[0]["content"])
            return SimpleNamespace(content=json.dumps(payload()), total_tokens=10)
    output = tmp_path / "run-integrated"
    integration.integrate({"topic": "Image forgery localization, adaptive attention", "sources": sources,
                           "consensus_dir": str(discussion)}, output, LLM())
    text = (output / "stage-08/hypotheses.md").read_text()
    assert "D1: Mechanism 1" in text and "D2: Mechanism 2" in text
    assert "attention scale weights" in text
    assert "Pending checks before full training" in text
    assert "training are not completed" in text
    assert all(f"D{index}: Mechanism {index}" in prompts[-1] for index in (1, 2))
    assert len((output / "stage-04/candidates.jsonl").read_text().splitlines()) == 2
    assert "ref1" in (output / "stage-04/references.bib").read_text()
    assert "ref2" in (output / "stage-04/references.bib").read_text()
    assert (output / "stage-07/synthesis.md").read_text() == "Joint discussion consensus"
    assert json.loads((output / "checkpoint.json").read_text())["last_completed_stage"] == 8
    for index, source in enumerate(sources):
        assert (Path(source["run_dir"]) / "stage-08/hypotheses.md").read_text() == originals[index]


def test_bridge_integrates_before_queueing_and_resumes_integration(tmp_path, monkeypatch):
    bridge = _load_agent_bridge_module()
    state = bridge.BridgeState(runs_base_dir=str(tmp_path))
    project = state.projects_dir() / "p"
    project.mkdir(parents=True)
    bridge._save_project_meta(str(project), "p", "neutral.yaml", "Joint topic")
    agents = [bridge.create_agent(state, f"Researcher {i}", "idea") for i in (1, 2)]
    for index, agent in enumerate(agents, 1):
        agent.project_id = "p"
        agent.run_dir = str(project / f"run-{index}")
        agent.config_path = f"direction-{index}.yaml"
        agent.role_tag = f"Mechanism {index}"
    group = bridge.DiscussionGroup(project_id="p", topic="Wrong last direction", config_path="direction-2.yaml",
        agent_ids=[a.id for a in agents], run_dirs={a.id: a.run_dir for a in agents},
        direction_names={a.id: a.role_tag for a in agents}, discussion_output_dir=str(project / "discussion"))
    state.discussion_groups["p"] = group
    state.queues["idea_to_experiment"] = bridge.TaskQueue("idea_to_experiment", tmp_path / "queue.json")
    calls = []
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda cmd, **kwargs: calls.append(cmd) or SimpleNamespace(pid=123))
    bridge._on_discussion_s8_done(state, agents[0])
    assert not state.queues["idea_to_experiment"].tasks
    bridge._on_discussion_s8_done(state, agents[1])
    active = agents[1]
    assert active.run_dir == str(project / "run-integrated")
    assert active.config_path == "neutral.yaml"
    assert not state.queues["idea_to_experiment"].tasks
    request = json.loads((project / "run-integrated/discussion/integration_request.json").read_text())
    assert request["topic"] == "Joint topic"
    assert {s["name"] for s in request["sources"]} == {"Mechanism 1", "Mechanism 2"}
    selected, parked = bridge._lab_resume_angle_dirs(project, bridge._read_project_meta(str(project)))
    assert selected == [project / "run-integrated"]
    assert len(parked) == 0  # Source run dirs were not created by this unit fixture.
    task = bridge.Task(id="retry", project_id="p", run_dir=active.run_dir,
        config_path=active.config_path, topic="Joint topic", source_layer="init", target_layer="idea", created_at=0)
    state.discussion_groups.clear()  # Simulate restart: no in-memory discussion group.
    bridge.launch_agent_for_task(state, active, task)
    assert calls[-1][1].endswith("hypothesis_integration.py")
    stage = project / "run-integrated/stage-08"
    stage.mkdir()
    (stage / "integration_report.json").write_text(json.dumps(payload()))
    bridge._on_discussion_s8_done(state, active)
    tasks = state.queues["idea_to_experiment"].tasks
    assert len(tasks) == 1 and tasks[0].run_dir.endswith("run-integrated")


def test_role_prompt_includes_topic_and_complementary_responsibility():
    bridge = _load_agent_bridge_module()
    prompt = bridge._build_role_prompt("Adaptive attention", "Image forgery localization")
    assert "Image forgery localization" in prompt and "Adaptive attention" in prompt
    assert "统一方法" in prompt


def small_request(tmp_path):
    sources = []
    for index in (1, 2):
        run = tmp_path / f"input-{index}"
        (run / "stage-08").mkdir(parents=True)
        (run / "stage-08/hypotheses.md").write_text(f"D{index} candidate and unverified interface")
        sources.append({"name": f"Direction {index}", "run_dir": str(run)})
    consensus = tmp_path / "synthesis"
    consensus.mkdir()
    (consensus / "independent_synthesis.md").write_text("Independently researched mechanisms")
    return {"topic": "Image forgery localization", "sources": sources,
            "consensus_dir": str(consensus), "synthesis_file": "independent_synthesis.md",
            "workflow": "independent_joint"}


def test_blocked_design_keeps_model_response_and_does_not_advance(tmp_path):
    request = small_request(tmp_path)
    output = tmp_path / "joint"
    output.mkdir()
    (output / "checkpoint.json").write_text('{"last_completed_stage": 7}')
    result = {**payload(), "status": "blocked", "reason": "Irreconcilable explicit user constraints"}
    class LLM:
        def chat(self, messages, **kwargs):
            assert "Missing future code inspection" in kwargs["system"]
            assert "revise the latter" in kwargs["system"]
            return SimpleNamespace(content=json.dumps(result), total_tokens=1)
    with pytest.raises(ValueError, match="Irreconcilable"):
        integration.integrate(request, output, LLM())
    report = json.loads((output / "stage-08/integration_report.json").read_text())
    raw = json.loads((output / "stage-08/integration_response.json").read_text())
    assert report["status"] == "blocked" and raw == result
    assert json.loads((output / "checkpoint.json").read_text())["last_completed_stage"] == 7
    assert not (output / "stage-08/hypotheses.md").exists()


@pytest.mark.parametrize("change", [{"training_completed": True}, {"readiness": "trained"},
                                    {"pending_checks": []}, {"interface_contract": ""},
                                    {"reconciliations": [{"conflict": "values mismatch"}]}])
def test_s8_readiness_requires_untrained_design_and_pending_checks(change):
    with pytest.raises(ValueError):
        integration.validate({**payload(), **change}, ["D1", "D2"])


def test_doi_identity_accepts_url_variation_but_rejects_real_conflicts(tmp_path):
    request = small_request(tmp_path)
    template = ('@inproceedings{{zhang2026collaborative,\n'
                '  title = {{Collaborative Transformers}},\n  author = {{Jiwei Zhang}},\n'
                '  year = {{2026}},\n  doi = {{10.1609/aaai.v40i15.38250}},\n'
                '  url = {{{url}}},\n}}')
    urls = ['https://ojs.aaai.org/index.php/AAAI/article/view/38250', 'https://doi.org/10.1609/aaai.v40i15.38250']
    for source, url in zip(request['sources'], urls):
        path = Path(source['run_dir']) / 'stage-04'
        path.mkdir()
        (path / 'references.bib').write_text(template.format(url=url))
    entries, duplicates = integration._merge_bibliographies(request['sources'])
    assert len(entries) == 1 and len(duplicates) == 1
    assert all(url in str(duplicates) for url in urls)
    path = Path(request['sources'][1]['run_dir']) / 'stage-04/references.bib'
    path.write_text(path.read_text().replace('2026}', '2025}'))
    with pytest.raises(ValueError, match='Conflicting bibliography key'):
        integration._merge_bibliographies(request['sources'])


def test_no_doi_cannot_merge_different_url_records_by_guessing_identity():
    assert not integration._equivalent_bibliography_entries(
        '@article{x, title={Same title}, url={https://one.test}}',
        '@article{x, title={Same title}, url={https://two.test}}')


def test_retry_reuses_condensed_inputs_and_records_provenance(tmp_path, monkeypatch):
    request = small_request(tmp_path)
    condensed = []
    monkeypatch.setattr(integration.Discussion, 'condense',
                        lambda self, label, text, limit: condensed.append(label) or text)
    class LLM:
        def chat(self, messages, **kwargs):
            assert len(messages[0]['content']) + len(kwargs['system']) < 14000
            return SimpleNamespace(content=json.dumps(payload()), total_tokens=1)
    output = tmp_path / 'joint'
    integration.integrate(request, output, LLM())
    integration.integrate(request, output, LLM())
    assert len(condensed) == 2
    inputs = json.loads((output / 'stage-08/integration_inputs.json').read_text())
    assert len(inputs['directions']) == 2 and inputs['directions'][0]['sha256']


def test_failed_integration_displays_saved_reason_in_agent_log(tmp_path):
    bridge = _load_agent_bridge_module()
    state = bridge.BridgeState(runs_base_dir=str(tmp_path))
    agent = bridge.create_agent(state, 'integrator', 'idea')
    agent.project_id = 'p'
    agent.run_dir = str(tmp_path / 'joint')
    agent._integrating_hypotheses = True
    agent.process = SimpleNamespace(poll=lambda: 1)
    stage = Path(agent.run_dir) / 'stage-08'
    stage.mkdir(parents=True)
    (stage / 'integration_report.json').write_text(json.dumps({'status': 'blocked', 'reason': 'Unresolved interface conflict'}))
    messages = bridge.poll_agent(agent)
    assert agent.status == 'error' and 'Unresolved interface conflict' in agent.current_task
    assert any('Unresolved interface conflict' in str(message) for message in messages)
