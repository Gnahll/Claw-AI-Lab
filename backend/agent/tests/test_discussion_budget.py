from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from .test_agent_bridge_reference_config import _load_agent_bridge_module


def _load_runner():
    path = Path(__file__).resolve().parents[2] / "services" / "discussion_runner.py"
    spec = importlib.util.spec_from_file_location("discussion_runner_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dr = _load_runner()


class FakeLLM:
    """Answers like the gateway: HTTP 500 above the limit, long answers otherwise."""

    def __init__(self, limit=15000, failures=0):
        self.limit, self.failures, self.prompts = limit, failures, []

    def chat(self, messages, system=None, max_tokens=None):
        size = len(system or "") + sum(len(m["content"]) for m in messages)
        self.prompts.append((system, messages[0]["content"], size))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("All models failed. Last error: HTTP Error 500: Internal Server Error")
        if size > self.limit:
            raise RuntimeError("All models failed. Last error: HTTP Error 500: Internal Server Error")
        wanted = re.search(r"at most (\d+) characters", messages[0]["content"])
        text = "要点。" * (int(wanted.group(1)) // 3) if wanted else "分析结论。" * 2000   # 10k-char answers
        return SimpleNamespace(content=text, total_tokens=size // 2)


@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_every_call_fits_the_budget_with_long_syntheses_and_topic(tmp_path, rounds):
    syntheses = {"Agent A": "## 方向\n\n" + "跨分支特征交互的证据与风险。\n\n" * 1300,     # ~18.5k chars
                 "Agent B": "## 方向\n\n" + "自适应多尺度注意力的文献综述内容。\n\n" * 2100}  # ~38k chars
    llm = FakeLLM()
    transcript, consensus = dr.run_discussion(llm, "Topic. " * 1000, syntheses, rounds, tmp_path, retry_delay=0)
    assert consensus and transcript.startswith("# Discussion Transcript")
    assert all(size <= dr.DEFAULT_MAX_PROMPT_CHARS for _, _, size in llm.prompts)
    discussion = [user for system, user, _ in llm.prompts if system != dr.SYSTEM_CONDENSE]
    assert len(discussion) == {1: 2, 2: 3, 3: 3}[rounds]
    assert (tmp_path / "condensed_syntheses.md").is_file()
    assert "## Agent A" in discussion[0] and "## Agent B" in discussion[0]


def test_short_syntheses_are_passed_unchanged(tmp_path):
    syntheses = {"Agent A": "short a", "Agent B": "short b"}
    llm = FakeLLM()
    dr.run_discussion(llm, "topic", syntheses, 2, tmp_path, retry_delay=0)
    assert all(system != dr.SYSTEM_CONDENSE for system, _, _ in llm.prompts)
    assert "short a" in llm.prompts[0][1] and not (tmp_path / "condensed_syntheses.md").exists()


def test_transient_errors_are_retried_and_persistent_ones_reported(tmp_path, monkeypatch):
    steps = []
    llm = FakeLLM(failures=2)
    dr.run_discussion(llm, "topic", {"Agent A": "a", "Agent B": "b"}, 2, tmp_path, progress=steps.append,
                      retry_delay=0)
    assert steps == ["Round 1/2", "Round 2/2", "Consensus"]
    with pytest.raises(RuntimeError, match=r"Round 1/2 failed after 3 attempts: .*HTTP Error 500"):
        dr.run_discussion(FakeLLM(failures=3), "topic", {"Agent A": "a"}, 2, tmp_path, retry_delay=0)


def test_failed_discussion_leaves_a_failed_heartbeat(tmp_path, monkeypatch):
    (tmp_path / "s7").mkdir()
    (tmp_path / "s7" / "synthesis.md").write_text("synthesis", encoding="utf-8")
    config = SimpleNamespace(research=SimpleNamespace(topic="topic"))
    monkeypatch.setattr(dr.RCConfig, "load", staticmethod(lambda *a, **k: config))
    monkeypatch.setattr(dr, "create_llm_client", lambda config: FakeLLM(failures=99))
    monkeypatch.setattr(dr, "ensure_english_topic", lambda config, output_dir, llm: config)
    monkeypatch.setattr(dr.time, "sleep", lambda seconds: None)
    out = tmp_path / "discussion"
    monkeypatch.setattr(sys, "argv", ["runner", "--config", "c.yaml", "--synthesis-dirs", str(tmp_path / "s7"),
                                      "--output", str(out), "--rounds", "2"])
    with pytest.raises(RuntimeError):
        dr.main()
    heartbeat = json.loads((out / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["status"] == "failed" and "HTTP Error 500" in heartbeat["error"]
    assert heartbeat["step"] == "Round 1/2"


@pytest.fixture
def stuck_lab_project(tmp_path):
    """The 2026-10-19 state: discussion failed, agents in error/waiting, a task left pending."""
    bridge = _load_agent_bridge_module()
    state = bridge.BridgeState(runs_base_dir=str(tmp_path))
    for name in {"init_to_idea", "execution_feedback", *bridge.LAYER_INPUT_QUEUE.values()}:
        state.queues[name] = bridge.TaskQueue(name, tmp_path / f"{name}.json")
    project = tmp_path / "projects" / "proj"
    for slug in ("adaptive", "cross"):
        (project / f"run-{slug}").mkdir(parents=True)
        (project / f"run-{slug}" / "checkpoint.json").write_text(json.dumps({"last_completed_stage": 7}))
    (project / "project_meta.json").write_text(json.dumps({"mode": "lab", "config_path": "c.yaml"}))
    agents = [bridge.create_agent(state, f"L1{x}", "idea") for x in "ABC"]
    for agent, status in zip(agents, ("error", "error", "waiting_discussion")):
        agent.project_id, agent.status = "proj", status
    state.discussion_waiting[agents[2].id] = agents[2]
    leftover = bridge.Task(id="task-old", project_id="proj", run_dir=str(project / "run-cross"),
                           config_path="c.yaml", source_layer="init", target_layer="idea")
    state.queues["init_to_idea"].push(leftover)
    return bridge, state, agents, leftover


def test_resume_releases_stuck_agents_and_requeues_each_angle_once(stuck_lab_project, monkeypatch):
    bridge, state, agents, leftover = stuck_lab_project
    bridge.resume_project(state, "proj")
    assert all(a.status == "idle" and not a.project_id for a in agents)
    assert not state.discussion_waiting
    assert leftover.status == "cancelled"
    pending = [t for t in state.queues["init_to_idea"].tasks if t.status == "pending"]
    assert sorted(Path(t.run_dir).name for t in pending) == ["run-adaptive", "run-cross"]

    launched = []
    monkeypatch.setattr(bridge, "launch_agent_for_task", lambda state, agent, task: launched.append(task) or [])
    bridge.schedule_idle_agents(state)
    assert launched == pending


def test_resume_waits_for_a_running_discussion(stuck_lab_project):
    bridge, state, agents, leftover = stuck_lab_project
    group = bridge.DiscussionGroup(project_id="proj", topic="", config_path="")
    group.discussion_process = SimpleNamespace(poll=lambda: None)
    state.discussion_groups["proj"] = group
    messages = bridge.resume_project(state, "proj")
    assert "讨论进行中" in json.dumps(messages, ensure_ascii=False)
    assert agents[0].status == "error" and leftover.status == "pending"


def test_failed_discussion_marks_heartbeat_and_explains(stuck_lab_project, tmp_path):
    bridge, state, agents, _ = stuck_lab_project
    out = tmp_path / "discussion"
    out.mkdir()
    (out / "heartbeat.json").write_text(json.dumps({"status": "discussing"}))
    group = bridge.DiscussionGroup(project_id="proj", topic="", config_path="", agent_ids=[agents[0].id])
    group.status, group.discussion_output_dir = "discussing", str(out)
    group.discussion_process = SimpleNamespace(poll=lambda: -9)
    messages = bridge._poll_discussion(state, group)
    assert json.loads((out / "heartbeat.json").read_text())["status"] == "failed"
    assert "exited with code -9" in json.dumps(messages, ensure_ascii=False)
