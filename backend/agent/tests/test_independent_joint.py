import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from .test_agent_bridge_reference_config import _load_agent_bridge_module


def setup_lab(tmp_path, monkeypatch, *, discussion=False):
    bridge = _load_agent_bridge_module()
    state = bridge.BridgeState(runs_base_dir=str(tmp_path), discussion_mode=not discussion)
    for name in ("init_to_idea", "idea_to_experiment"):
        state.queues[name] = bridge.TaskQueue(name, tmp_path / f"{name}.json")
    def config(state, project_id, topic, *args, **kwargs):
        dest = tmp_path / "project_configs" / f"{project_id}.yaml"
        dest.parent.mkdir(exist_ok=True)
        dest.write_text("research:\n  topic: Image forgery localization\n")
        return str(dest)
    monkeypatch.setattr(bridge, "_generate_config_from_template", config)
    bridge.quick_submit_project(state, "Image forgery localization", "one-paper", research_angles=[
        "Consistency-guided cross-branch interaction", "Spatial adaptive attention"], discussion_enabled=discussion)
    project = state.projects_dir() / "one-paper"
    return bridge, state, project


@pytest.mark.parametrize("discussion", [False, True])
def test_submission_pins_mode_without_changing_other_projects(tmp_path, monkeypatch, discussion):
    bridge, state, project = setup_lab(tmp_path, monkeypatch, discussion=discussion)
    meta = bridge._read_project_meta(str(project))
    expected = "discussion_joint" if discussion else "independent_joint"
    assert meta["research_workflow"] == expected
    assert len(meta["research_directions"]) == 2
    assert state.discussion_mode is not discussion
    state.discussion_mode = not state.discussion_mode
    assert bridge._project_workflow(state, "one-paper") == expected


def test_independent_directions_join_once_without_debate_or_duplicate_experiments(tmp_path, monkeypatch):
    bridge, state, project = setup_lab(tmp_path, monkeypatch)
    inputs = list(state.queues["init_to_idea"].tasks)
    calls = []
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda cmd, **kwargs: calls.append(cmd) or SimpleNamespace(pid=123))
    monkeypatch.setattr(bridge, "_trigger_discussion", lambda *a: pytest.fail("Independent mode must not debate"))
    agents = []
    for index, task in enumerate(inputs):
        agent = bridge.create_agent(state, f"researcher {index}", "idea")
        agents.append(agent)
        bridge.launch_agent_for_task(state, agent, task)
        assert calls[-1][calls[-1].index("--to-stage") + 1] == "HYPOTHESIS_GEN"
        run = Path(task.run_dir)
        (run / "stage-07").mkdir()
        (run / "stage-07/synthesis.md").write_text(f"Evidence from direction {index}")
        (run / "stage-08").mkdir()
        (run / "stage-08/hypotheses.md").write_text(f"Candidate mechanism {index}")
        (run / "checkpoint.json").write_text(json.dumps({"last_completed_stage": 8}))
        bridge.on_agent_done(state, agent)
        assert not state.queues["idea_to_experiment"].tasks
    assert calls[-1][1].endswith("hypothesis_integration.py")
    target = project / "run-integrated"
    request = json.loads((target / "discussion/integration_request.json").read_text())
    assert request["workflow"] == "independent_joint"
    assert request["synthesis_file"] == "independent_synthesis.md"
    independent = project / "independent_synthesis/independent_synthesis.md"
    assert "No cross-direction debate" in independent.read_text()
    assert not (project / "discussion/discussion_transcript.md").exists()
    assert bridge._lab_resume_angle_dirs(project, bridge._read_project_meta(str(project)))[0] == [target]
    stage = target / "stage-08"
    stage.mkdir()
    (stage / "integration_report.json").write_text(json.dumps({"status": "ready"}))
    bridge._on_discussion_s8_done(state, agents[-1])
    outputs = state.queues["idea_to_experiment"].tasks
    assert len(outputs) == 1 and outputs[0].run_dir == str(target)


def test_restart_joins_saved_independent_hypotheses_without_rerunning_research(tmp_path, monkeypatch):
    bridge, state, project = setup_lab(tmp_path, monkeypatch)
    for task in state.queues["init_to_idea"].tasks:
        run = Path(task.run_dir)
        for stage, filename in ((7, "synthesis.md"), (8, "hypotheses.md")):
            (run / f"stage-{stage:02d}").mkdir()
            (run / f"stage-{stage:02d}" / filename).write_text("Saved evidence and mechanism")
        (run / "checkpoint.json").write_text(json.dumps({"last_completed_stage": 8}))
    calls = []
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(pid=42))
    state.lab_batches.clear()
    state.discussion_mode = True  # Different global default after restart.
    first, second = list(state.queues["init_to_idea"].tasks)
    a = bridge.create_agent(state, "a", "idea")
    b = bridge.create_agent(state, "b", "idea")
    bridge.launch_agent_for_task(state, a, first)
    bridge.launch_agent_for_task(state, b, second)
    assert len(calls) == 1 and calls[0][1].endswith("hypothesis_integration.py")
    assert state.discussion_groups["one-paper"].status == "integrating"
    assert not state.queues["idea_to_experiment"].tasks


def test_invalid_discussion_choice_is_rejected_before_creating_project(tmp_path):
    bridge = _load_agent_bridge_module()
    state = bridge.BridgeState(runs_base_dir=str(tmp_path))
    bridge.quick_submit_project(state, "topic", "invalid", discussion_enabled="false")
    assert not (state.projects_dir() / "invalid").exists()
