"""experiment.runs_per_condition = 1: one run per condition, no multi-seed rules."""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from researchclaw import single_run
from researchclaw.config import RCConfig, validate_config
from researchclaw.prompts import PromptManager

PKG = Path(single_run.__file__).resolve().parent
SOURCES = (
    "prompts.py",
    "pipeline/executor.py",
    "pipeline/codegen/system_prompt.py",
    "pipeline/iterative_refine/system_prompt.py",
    "pipeline/result_analysis/system_prompt.py",
)
MULTI_SEED_RULES = (
    "AT LEAST 3 random seeds",
    "≥3 seeds",
    "at least 3 seeds per condition",
    "Use 3 seeds (42, 123, 456)",
    "Use 3 random seeds (42, 123, 456)",
    "MUST include 95% confidence intervals",
    "MUST cite p-value",
    "Is n>1 (multiple seeds)?",
    "Extra seeds may be appended",
)


def _string_constants(path: Path) -> list[str]:
    """Literal text of a source file; adjacent literals are already joined by the parser."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def test_every_rewrite_matches_a_current_prompt():
    texts = [t for name in SOURCES for t in _string_constants(PKG / name)]
    missing = [old for old, _ in single_run.REWRITES if not any(old in t for t in texts)]
    assert missing == []


def _all_prompt_text(pm: PromptManager) -> str:
    parts = []
    for stage in pm.stage_names():
        rendered = pm.for_stage(stage)
        parts += [rendered.system, rendered.user]
    parts += [pm.block(name) for name in pm._blocks]
    for name in pm._sub_prompts:
        rendered = pm.sub_prompt(name)
        parts += [rendered.system, rendered.user]
    return "\n".join(parts)


def test_single_run_removes_multi_seed_rules_from_every_prompt():
    default = _all_prompt_text(PromptManager())
    assert all(rule in default for rule in MULTI_SEED_RULES[:3] + MULTI_SEED_RULES[5:8])
    pm = PromptManager()
    pm.set_runs_per_condition(1)
    text = _all_prompt_text(pm)
    assert [rule for rule in MULTI_SEED_RULES if rule in text] == []
    for stage in ("experiment_design", "research_decision", "paper_draft", "peer_review"):
        assert single_run.SINGLE_RUN_POLICY in pm.for_stage(stage).user


def test_multi_run_keeps_default_prompts():
    pm = PromptManager()
    pm.set_runs_per_condition(3)
    assert _all_prompt_text(pm) == _all_prompt_text(PromptManager())
    assert single_run.SINGLE_RUN_POLICY not in _all_prompt_text(pm)


def test_agent_prompts_are_rewritten():
    from researchclaw.pipeline.iterative_refine.system_prompt import REFINE_LOOP_RULES

    out = single_run.apply(REFINE_LOOP_RULES, True, append_policy=True)
    assert "Extra seeds may be appended" not in out and "more seeds" not in out
    assert out.endswith(single_run.SINGLE_RUN_POLICY)
    assert single_run.apply(REFINE_LOOP_RULES, False, append_policy=True) == REFINE_LOOP_RULES


def _config(runs):
    return SimpleNamespace(experiment=SimpleNamespace(runs_per_condition=runs))


def test_config_parsing_and_validation():
    assert single_run.is_single_run(_config(1)) and not single_run.is_single_run(_config(3))
    assert not single_run.is_single_run(SimpleNamespace())
    base = {"project": {"name": "x"}, "research": {"topic": "t"}, "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "none"}, "knowledge_base": {"root": "kb"},
            "llm": {"provider": "openai-compatible", "base_url": "http://x", "api_key_env": "K"}}
    for runs, ok in ((1, True), (3, True), (5, True), (2, False), ("1", False)):
        data = {**base, "experiment": {"runs_per_condition": runs}}
        errors = [e for e in validate_config(data, check_paths=False).errors if "runs_per_condition" in e]
        assert (errors == []) is ok, (runs, errors)
    cfg = RCConfig.from_dict({**base, "experiment": {"runs_per_condition": 1}}, check_paths=False)
    assert cfg.experiment.runs_per_condition == 1
    assert RCConfig.from_dict(base, check_paths=False).experiment.runs_per_condition == 3


def test_validate_draft_skips_variance_directive_for_single_run():
    from researchclaw.pipeline.executor import _validate_draft_quality

    draft = "# T\n\n## Results\n\n" + " ".join(["The method reaches 0.71 F1 on the test set."] * 30) + "\n"
    directive = "Single-run results without variance reporting"
    multi = _validate_draft_quality(draft)
    single = _validate_draft_quality(draft, single_run_mode=True)
    assert any(directive in d for d in multi["revision_directives"])
    assert not any(directive in d for d in single["revision_directives"])


def _bridge():
    path = Path(__file__).resolve().parents[2] / "services" / "agent_bridge.py"
    spec = importlib.util.spec_from_file_location("agent_bridge_single_run", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bridge_writes_runs_per_condition(tmp_path):
    bridge = _bridge()
    state = SimpleNamespace(runs_base_dir=str(tmp_path / "runs"), agent_package_dir=str(tmp_path / "agent"))
    one = yaml.safe_load(Path(bridge._generate_config_from_template(
        state, "p1", "topic", runs_per_condition=1)).read_text(encoding="utf-8"))
    assert one["experiment"]["runs_per_condition"] == 1
    assert "每个实验条件只运行一次" in one["research"]["topic"]
    three = yaml.safe_load(Path(bridge._generate_config_from_template(
        state, "p3", "topic")).read_text(encoding="utf-8"))
    assert three["experiment"]["runs_per_condition"] == 3
    assert "只运行一次" not in three["research"]["topic"]
    with pytest.raises(ValueError, match="运行次数"):
        bridge._generate_config_from_template(state, "p2", "topic", runs_per_condition=2)
