from __future__ import annotations

import json
from pathlib import Path

import pytest

from .test_agent_bridge_reference_config import _load_agent_bridge_module


@pytest.fixture
def bridge_state(tmp_path):
    bridge = _load_agent_bridge_module()
    state = bridge.BridgeState(runs_base_dir=str(tmp_path))
    for name in {"init_to_idea", "execution_feedback", *bridge.LAYER_INPUT_QUEUE.values()}:
        state.queues[name] = bridge.TaskQueue(name, tmp_path / f"{name}.json")
    project = tmp_path / "projects" / "proj"
    run_dir = project / "run-angle"
    run_dir.mkdir(parents=True)
    (project / "project_meta.json").write_text(json.dumps({"project_id": "proj", "config_path": "c.yaml"}))
    (run_dir / "checkpoint.json").write_text(json.dumps({"last_completed_stage": 13}))
    failed = bridge.Task(id="task-1", project_id="proj", run_dir=str(run_dir), config_path="c.yaml",
                         source_layer="coding", target_layer="execution", topic="[angle] topic")
    agent = bridge.LobsterAgent(id="L-1", name="L4", layer="execution", run_id="", run_dir=str(run_dir),
                                config_path="c.yaml")
    agent.project_id = "proj"
    return bridge, state, agent, failed, project


def queued(state):
    return [task for queue in state.queues.values() for task in queue.tasks if task.status == "pending"]


def test_crashed_layer_is_requeued_from_checkpoint_after_backoff(bridge_state, monkeypatch):
    bridge, state, agent, failed, _ = bridge_state
    messages = bridge._queue_auto_retry(state, agent, failed, 1)
    [task] = queued(state)
    assert task.run_dir == failed.run_dir and task.topic == failed.topic and task.target_layer == "execution"
    assert task in state.queues[bridge._queue_for_layer("execution")].tasks
    assert task.not_before > bridge._now_ms() + 60_000 and task.attempt == 1
    assert any("EXPERIMENT_RUN" in str(message) for message in messages)
    assert bridge._queue_auto_retry(state, agent, failed, 1) == [] and len(queued(state)) == 1  # no duplicates

    launched = []
    monkeypatch.setattr(bridge, "launch_agent_for_task", lambda state, agent, task: launched.append(task) or [])
    idle = bridge.LobsterAgent(id="L-2", name="L4b", layer="execution", run_id="", run_dir="", config_path="")
    state.agents[idle.id] = idle
    state._fail_counts["proj"] = 1
    bridge.schedule_idle_agents(state)
    assert launched == []                       # still backing off
    task.not_before = 0
    bridge.schedule_idle_agents(state)
    assert launched == [task]
    assert state._fail_counts["proj"] == 1      # only success or a manual resume resets the counter


def test_retry_limit_blocks_scheduling(bridge_state, monkeypatch):
    bridge, state, agent, failed, _ = bridge_state
    bridge._queue_auto_retry(state, agent, failed, 1)
    queued(state)[0].not_before = 0
    state._fail_counts["proj"] = bridge.AUTO_RETRY_LIMIT
    monkeypatch.setattr(bridge, "launch_agent_for_task", lambda *args: pytest.fail("retry limit exceeded"))
    state.agents["L-2"] = bridge.LobsterAgent(id="L-2", name="L4b", layer="execution", run_id="", run_dir="",
                                              config_path="")
    bridge.schedule_idle_agents(state)


@pytest.mark.parametrize("case", ["intervention", "discussion", "idea_factory", "no_task"])
def test_no_automatic_retry_where_a_human_or_peer_owns_recovery(bridge_state, case):
    bridge, state, agent, failed, project = bridge_state
    if case == "intervention":
        (project / "project_meta.json").write_text(json.dumps({"project_id": "proj", "intervention": "S12"}))
    elif case == "discussion":
        state.discussion_groups["proj"] = bridge.DiscussionGroup(project_id="proj", topic="", config_path="",
                                                                  agent_ids=[agent.id], run_dirs={})
    elif case == "idea_factory":
        agent._is_discussion_s8 = True
    elif case == "no_task":
        failed = None
    assert bridge._queue_auto_retry(state, agent, failed, 1) == []
    assert queued(state) == []


def test_pause_removes_project_tasks_from_queue_file(bridge_state):
    bridge, state, _, failed, _ = bridge_state
    queue = state.queues[bridge._queue_for_layer("execution")]
    other = bridge.Task(id="task-2", project_id="other", run_dir="", config_path="c.yaml",
                        source_layer="coding", target_layer="execution", topic="t")
    queue.push(failed)
    queue.push(other)
    bridge._pause_project(state, "proj")
    reloaded = bridge.TaskQueue(queue.name, queue.path)
    reloaded.load()
    assert [task.id for task in reloaded.tasks] == ["task-2"]


@pytest.mark.parametrize("error", [
    "Acceptance protocol needs design revision: H2 mapping is undefined",
    "Pretraining acceptance blocked: retained_H1_failed",
])
def test_acceptance_block_does_not_queue_an_unchanged_paid_retry(bridge_state, error):
    bridge, state, agent, failed, _ = bridge_state
    path = Path(failed.run_dir) / "stage-14/decision.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"status": "failed", "error": error}))
    messages = bridge._queue_auto_retry(state, agent, failed, 1)
    assert queued(state) == []
    assert any("验收协议" in str(message) for message in messages)


def test_old_acceptance_report_does_not_suppress_a_new_network_retry(bridge_state):
    bridge, state, agent, failed, _ = bridge_state
    from pathlib import Path
    path = Path(failed.run_dir) / "stage-14"
    path.mkdir()
    (path / "acceptance_review.json").write_text(json.dumps({"retryable": False}))
    (path / "decision.json").write_text(json.dumps({"status": "failed", "error": "HTTP Error 524"}))
    bridge._queue_auto_retry(state, agent, failed, 1)
    assert len(queued(state)) == 1
