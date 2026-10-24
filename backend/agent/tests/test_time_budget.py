"""Project form "experiment time limit" is written to experiment.time_budget_sec."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from researchclaw.config import RCConfig


def _bridge():
    path = Path(__file__).resolve().parents[2] / "services" / "agent_bridge.py"
    spec = importlib.util.spec_from_file_location("agent_bridge_time_budget", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _experiment(bridge, state, name, **kwargs):
    path = bridge._generate_config_from_template(state, name, "topic", **kwargs)
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_bridge_writes_time_budget(tmp_path):
    bridge = _bridge()
    state = SimpleNamespace(runs_base_dir=str(tmp_path / "runs"), agent_package_dir=str(tmp_path / "agent"))
    default = _experiment(bridge, state, "p0")
    template_budget = default["experiment"]["time_budget_sec"]
    assert _experiment(bridge, state, "p0b", time_budget_hours=0)["experiment"]["time_budget_sec"] == template_budget
    ten = _experiment(bridge, state, "p8", time_budget_hours=8)
    assert ten["experiment"]["time_budget_sec"] == 8 * 3600
    assert RCConfig.from_dict(ten, check_paths=False).experiment.time_budget_sec == 8 * 3600
    # Long options for full-length reproductions (e.g. 200 epochs on large balanced datasets).
    for hours in (48, 72):
        assert _experiment(bridge, state, f"p{hours}", time_budget_hours=hours)["experiment"]["time_budget_sec"] == hours * 3600
    for bad in (3, -1, 36, 100, "8", 8.0, True):
        with pytest.raises(ValueError, match="实验时间上限"):
            bridge._generate_config_from_template(state, "bad", "topic", time_budget_hours=bad)
