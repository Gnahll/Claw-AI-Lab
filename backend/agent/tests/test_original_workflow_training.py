from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from researchclaw.config import ExperimentConfig, SandboxConfig, _parse_experiment_config
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.training.execution import run_experiment
from researchclaw.training.resources import validate_gpu_uuid, with_execution_resources


GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"


def configuration(training=False, gpu="", metric="metric"):
    return SimpleNamespace(
        experiment=ExperimentConfig(require_training=training, gpu_uuid=gpu, time_budget_sec=30, metric_key=metric,
                                    sandbox=SandboxConfig(python_path=sys.executable)),
        llm=SimpleNamespace(coding_model=""),
    )


def test_config_preserves_original_workflow_settings():
    parsed = _parse_experiment_config({"require_training": True, "gpu_uuid": GPU_UUID})
    assert parsed.require_training and parsed.gpu_uuid == GPU_UUID
    assert not _parse_experiment_config({}).require_training


def test_cli_topic_override_cannot_erase_training_requirement(tmp_path, monkeypatch):
    import argparse
    from researchclaw import cli
    from researchclaw.pipeline import runner
    from .test_rc_cli import _write_valid_config

    config_path = tmp_path / "config.yaml"
    _write_valid_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["experiment"] = {"require_training": True}
    data["knowledge_base"]["root"] = str(tmp_path / "kb")
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    captured = []
    monkeypatch.setattr(runner, "execute_pipeline", lambda **kwargs: captured.append(kwargs["config"]) or [])
    args = argparse.Namespace(config=str(config_path), topic="override from discussion", output=str(tmp_path / "run"),
                              from_stage=None, to_stage=None, auto_approve=True, skip_preflight=True,
                              resume=False, skip_noncritical_stage=False, no_graceful_degradation=False)
    assert cli.cmd_run(args) == 0
    assert "override from discussion" in captured[0].research.topic
    assert "Mandatory training objective" in captured[0].research.topic


@pytest.mark.parametrize("mode,discussion", [("lab", True), ("lab", False), ("reproduce", True)])
def test_all_original_submissions_propagate_training_config(tmp_path, monkeypatch, mode, discussion):
    from .test_agent_bridge_reference_config import _load_agent_bridge_module

    bridge = _load_agent_bridge_module()
    state = bridge.BridgeState(runs_base_dir=str(tmp_path), discussion_mode=discussion)
    state.queues["init_to_idea"] = bridge.TaskQueue("init_to_idea", tmp_path / "queue.json")
    monkeypatch.setattr(bridge, "submit_new_project", lambda *args, **kwargs: [])
    bridge.quick_submit_project(state, "train a segmentation model", mode=mode,
                                research_angles=["CV", "VLM"], path_overrides={
                                    "require_training": True, "gpu_uuid": GPU_UUID,
                                    "datasets_dir": "/allocated/data", "codebases_dir": "/allocated/code",
                                })
    configs = list((tmp_path / "project_configs").glob("*.yaml"))
    assert len(configs) == (1 if mode == "reproduce" else 3)
    for config in configs:
        data = yaml.safe_load(config.read_text(encoding="utf-8"))
        assert data["experiment"]["require_training"] is True
        assert data["experiment"]["gpu_uuid"] == GPU_UUID
        assert data["experiment"]["datasets_dir"] == "/allocated/data"
        assert data["experiment"]["training_job_dir"] == ""
        assert "真实模型训练" in data["research"]["topic"]


@pytest.mark.parametrize("value", ["0", "all", "GPU-1", "GPU-../unsafe", "GPU-a,GPU-b"])
def test_gpu_requires_explicit_uuid(value):
    with pytest.raises(ValueError):
        validate_gpu_uuid(value)


def test_cpu_execution_never_autoselects_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    observed = []

    @with_execution_resources(Stage.EXPERIMENT_RUN)
    def execute(stage_dir, run_dir, config, adapters):
        observed.append(os.environ["CUDA_VISIBLE_DEVICES"])

    execute(tmp_path, tmp_path, configuration(), None)
    assert observed == [""]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_occupied_gpu_blocks_before_experiment(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from researchclaw.training import resources

    monkeypatch.setattr(resources, "job_lock", lambda path: nullcontext())

    def busy(gpu):
        assert gpu == GPU_UUID
        raise RuntimeError("Allocated GPU is occupied")

    monkeypatch.setattr(resources, "require_idle_gpu", busy)

    @with_execution_resources(Stage.EXPERIMENT_RUN)
    def execute(*args, **kwargs):
        pytest.fail("Must not execute on an occupied GPU")

    result = execute(tmp_path, tmp_path, configuration(gpu=GPU_UUID), None)
    assert result.status == StageStatus.FAILED
    assert "occupied" in result.error


def test_old_results_and_llm_prose_do_not_count_as_execution(tmp_path):
    (tmp_path / "main.py").write_text("print('training passed')\n")
    (tmp_path / "results.json").write_text('{"accuracy": 0.99}')
    result = run_experiment(tmp_path, tmp_path / "evidence", configuration())
    assert not result["success"]
    assert "fresh results.json" in result["errors"][0]


@pytest.mark.parametrize("payload", ['{}', '{"success": true}', '{"metric": NaN}', '{"metric": 1e999}'])
def test_invalid_metrics_are_rejected(tmp_path, payload):
    (tmp_path / "main.py").write_text(f"from pathlib import Path\nPath('results.json').write_text({payload!r})\n")
    assert not run_experiment(tmp_path, tmp_path / "evidence", configuration())["success"]


def test_process_exit_failure_overrides_written_metrics(tmp_path):
    (tmp_path / "main.py").write_text("from pathlib import Path\nPath('results.json').write_text('{\"metric\": 1}')\nraise SystemExit(2)\n")
    result = run_experiment(tmp_path, tmp_path / "evidence", configuration())
    assert not result["success"] and result["returncode"] == 2


def test_supervisor_removes_smoke_flag_for_full_run(tmp_path, monkeypatch):
    monkeypatch.setenv("SMOKE_TEST", "1")
    (tmp_path / "main.py").write_text("import os,json\nfrom pathlib import Path\nPath('results.json').write_text(json.dumps({'metric': int(os.environ.get('SMOKE_TEST','0'))}))\n")
    full = run_experiment(tmp_path, tmp_path / "full", configuration())
    smoke = run_experiment(tmp_path, tmp_path / "smoke", configuration(), smoke=True)
    assert full["success"] and full["results"]["metric"] == 0
    assert smoke["success"] and smoke["results"]["metric"] == 1


def test_hung_process_times_out(tmp_path):
    (tmp_path / "main.py").write_text("import time\ntime.sleep(60)\n")
    result = run_experiment(tmp_path, tmp_path / "evidence", configuration(), timeout=1)
    assert not result["success"] and result["elapsed_sec"] < 15


@pytest.mark.parametrize("train,save,correct,accepted", [(True, True, True, True), (False, False, False, False),
                                                        (True, False, False, False), (True, True, False, False)])
def test_real_cpu_torch_updates_and_matching_checkpoint(tmp_path, monkeypatch, train, save, correct, accepted):
    pytest.importorskip("torch")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    script = "import torch,json\nfrom pathlib import Path\nmodel=torch.nn.Linear(2,1)\noptimizer=torch.optim.SGD(model.parameters(),lr=0.1)\n"
    if train:
        script += "for step in range(3):\n optimizer.zero_grad()\n loss=(model(torch.ones(2,2))-2).square().mean()\n loss.backward()\n optimizer.step()\n"
    if save:
        script += "Path('outputs').mkdir(exist_ok=True)\n"
        script += "torch.save(" + ("model" if correct else "torch.nn.Linear(2,1)") + ".state_dict(),'outputs/model.pt')\n"
    script += "Path('results.json').write_text(json.dumps({'loss': float(model(torch.ones(2,2)).square().mean().item())}))\n"
    (tmp_path / "main.py").write_text(script)
    result = run_experiment(tmp_path, tmp_path / "evidence", configuration(training=True, metric="loss"))
    assert result["success"] == accepted, result
    if accepted:
        assert result["training"]["optimizer_steps"] == 3
        assert result["training"]["checkpoint_verified"]
        assert not result["training"]["cuda_update_observed"]
        assert len((tmp_path / "evidence/training_observation.jsonl").read_text().splitlines()) == 3


def test_s14_runs_actual_code_without_llm_claim(tmp_path, monkeypatch):
    from researchclaw.pipeline.experiment_run import runtime

    experiment = tmp_path / "stage-11/experiment"
    experiment.mkdir(parents=True)
    (experiment / "main.py").write_text("from pathlib import Path\nPath('results.json').write_text('{\"score\": 0.5}')\n")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(runtime.ExperimentRunRuntime, "_ensure_deps", lambda *args: None)
    monkeypatch.setattr(runtime, "AgentTurnLoop", lambda **kwargs: SimpleNamespace(run_turn=lambda message: pytest.fail("Passing experiment needs no LLM repair")))
    result = runtime.ExperimentRunRuntime().execute(tmp_path / "stage-14", tmp_path, configuration(metric="score"), None,
                                                  llm=SimpleNamespace(config=SimpleNamespace(primary_model="test")))
    assert result.status == StageStatus.DONE
    report = json.loads((tmp_path / "stage-14/runs/run_report.json").read_text())
    assert report["execution"]["returncode"] == 0
    assert report["execution"]["results"] == {"score": 0.5}


def test_s15_cannot_relabel_stale_baseline_as_success(tmp_path, monkeypatch):
    from researchclaw.pipeline.iterative_refine import runtime
    from researchclaw.pipeline.claw_engine.turn_loop import TurnResult

    experiment = tmp_path / "stage-11/experiment"
    experiment.mkdir(parents=True)
    (experiment / "main.py").write_text("print('improvement completed')\n")
    (experiment / "results.json").write_text('{"score": 0.99}')
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(runtime, "AgentTurnLoop", lambda **kwargs: SimpleNamespace(run_turn=lambda message: TurnResult(final_text="success")))
    result = runtime.IterativeRefineRuntime().execute(tmp_path / "stage-15", tmp_path, configuration(), None,
                                                    llm=SimpleNamespace(config=SimpleNamespace(primary_model="test")))
    assert result.status == StageStatus.FAILED


def test_removed_standalone_workflow_is_not_exposed():
    root = Path(__file__).resolve().parents[3]
    assert not (root / "frontend/src/components/TrainingPanel.tsx").exists()
    assert "training_start" not in (root / "backend/services/agent_bridge.py").read_text(encoding="utf-8")
    assert "TrainingPanel" not in (root / "frontend/src/App.tsx").read_text(encoding="utf-8")


def test_metadata_without_requested_metric_is_rejected(tmp_path):
    (tmp_path / "main.py").write_text("from pathlib import Path\nPath('results.json').write_text('{\"seed\":42}')\n")
    result = run_experiment(tmp_path, tmp_path / "evidence", configuration())
    assert not result["success"] and "missing measured metric" in result["errors"][0]


def test_s12_does_not_trust_llm_success_text(tmp_path, monkeypatch):
    from researchclaw.pipeline.sanity_check import runtime
    from researchclaw.pipeline.claw_engine.turn_loop import TurnResult

    experiment = tmp_path / "stage-11/experiment"
    experiment.mkdir(parents=True)
    (experiment / "main.py").write_text("raise SystemExit(7)\n")
    monkeypatch.setattr(runtime, "AgentTurnLoop", lambda **kwargs: SimpleNamespace(run_turn=lambda message: TurnResult(final_text="all smoke tests passed")))
    result = runtime.SanityCheckRuntime().execute(tmp_path / "stage-12", tmp_path, configuration(), None,
                                                llm=SimpleNamespace(config=SimpleNamespace(primary_model="test")))
    assert result.status == StageStatus.FAILED


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_stopping_original_stage_kills_its_experiment(tmp_path):
    import signal
    import subprocess
    import time

    child_pid = tmp_path / "child.pid"
    (tmp_path / "main.py").write_text("import os,time\nfrom pathlib import Path\nPath('child.pid').write_text(str(os.getpid()))\ntime.sleep(120)\n")
    launcher = tmp_path / "stage.py"
    launcher.write_text(
        "import sys\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
        "from researchclaw.config import ExperimentConfig,SandboxConfig\n"
        "from researchclaw.pipeline.stages import Stage\n"
        "from researchclaw.training.resources import with_execution_resources\n"
        "from researchclaw.training.execution import run_experiment\n"
        "@with_execution_resources(Stage.EXPERIMENT_RUN)\n"
        "def execute(stage_dir,run_dir,config,adapters):\n run_experiment(run_dir,stage_dir,config,timeout=120)\n"
        "config=SimpleNamespace(experiment=ExperimentConfig(sandbox=SandboxConfig(python_path=sys.executable)))\n"
        "execute(Path('evidence'),Path.cwd(),config,None)\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    process = subprocess.Popen([sys.executable, str(launcher)], cwd=tmp_path, env=environment)
    try:
        deadline = time.monotonic() + 20
        while not child_pid.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.05)
        assert child_pid.exists()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=15)
        assert not Path(f"/proc/{child_pid.read_text()}").exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
