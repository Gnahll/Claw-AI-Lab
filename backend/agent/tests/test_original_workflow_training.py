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
PREDICTIONS = ("Path('outputs').mkdir(exist_ok=True)\n"
               "Path('outputs/predictions.json').write_text(json.dumps({'predictions': [0.1, 0.7, 0.3]}))\n")


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
                                    "optional_experiments": ["visualization", "ablation"],
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
        assert data["experiment"]["optional_experiments"] == ["ablation", "visualization"]
        assert _parse_experiment_config(data["experiment"]).optional_experiments == ("ablation", "visualization")


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


def watched(stall=0, gpu_idle=0):
    config = configuration()
    config.experiment = ExperimentConfig(time_budget_sec=30, metric_key="metric", stall_timeout_sec=stall,
                                         gpu_idle_timeout_sec=gpu_idle, monitor_interval_sec=1,
                                         sandbox=SandboxConfig(python_path=sys.executable))
    return config


@pytest.mark.parametrize("script,accepted", [
    ("print('epoch 1', flush=True)\ntime.sleep(60)\n", False),                       # hung after start
    ("for epoch in range(8):\n print('epoch', epoch, flush=True)\n time.sleep(0.5)\n", True),
])
def test_watchdog_kills_a_stalled_run_but_not_a_progressing_one(tmp_path, script, accepted):
    (tmp_path / "main.py").write_text("import json,time\n" + script +
                                      "open('results.json','w').write(json.dumps({'metric': 1.0}))\n")
    result = run_experiment(tmp_path, tmp_path / "evidence", watched(stall=2))
    assert result["success"] == accepted, result
    assert result["elapsed_sec"] < 15 and result["monitor"]["samples"] >= 1
    if not accepted:
        assert "Training stalled" in result["errors"][0]
    assert (tmp_path / "evidence/training_monitor.jsonl").read_text().strip()


@pytest.mark.parametrize("sample,accepted", [
    ({"utilization": 0.0, "memory_mib": 0.0, "process_memory_mib": 0.0, "process_on_gpu": False}, False),
    ({"utilization": 87.0, "memory_mib": 9000.0, "process_memory_mib": 9000.0, "process_on_gpu": True}, True),
    ({"error": "nvidia-smi timed out"}, True),   # a monitoring failure must not kill training
])
def test_watchdog_kills_a_run_that_never_uses_its_gpu(tmp_path, monkeypatch, sample, accepted):
    from researchclaw.training import execution

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    monkeypatch.setattr(execution, "_gpu_sample", lambda uuid, pid: dict(sample))
    (tmp_path / "main.py").write_text("import json,time\nfor epoch in range(8):\n print(epoch, flush=True)\n"
                                      " time.sleep(0.5)\nopen('results.json','w').write(json.dumps({'metric': 1.0}))\n")
    result = run_experiment(tmp_path, tmp_path / "evidence", watched(gpu_idle=2))
    assert result["success"] == accepted, result
    if not accepted:
        assert "idle and unused" in result["errors"][0]


@pytest.mark.parametrize("script,accepted", [
    # Seen on the server: a resumable full data audit wrote its cache for 70 min before
    # touching the GPU, and was killed at 30 min as idle; each retry restarted the stage.
    ("for i in range(8):\n with open('outputs/audit.sqlite','a') as f: f.write('x'*100)\n time.sleep(0.5)\n", True),
    ("open('outputs/audit.sqlite','w').write('x')\ntime.sleep(60)\n", False),   # stopped writing: a hang
])
def test_watchdog_allows_cpu_preparation_that_keeps_writing_before_training(tmp_path, monkeypatch,
                                                                            script, accepted):
    from researchclaw.training import execution

    idle = {"utilization": 0.0, "memory_mib": 0.0, "process_memory_mib": 0.0, "process_on_gpu": False}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    monkeypatch.setattr(execution, "_gpu_sample", lambda uuid, pid: dict(idle))
    (tmp_path / "outputs").mkdir()
    (tmp_path / "main.py").write_text("import json,time\n" + script +
                                      "open('results.json','w').write(json.dumps({'metric': 1.0}))\n")
    result = run_experiment(tmp_path, tmp_path / "evidence", watched(stall=2, gpu_idle=2))
    assert result["success"] == accepted, result
    assert result["elapsed_sec"] < 15


def test_output_growth_does_not_excuse_an_idle_gpu_once_training_started(tmp_path, monkeypatch):
    from researchclaw.training import execution

    idle = {"utilization": 0.0, "memory_mib": 0.0, "process_memory_mib": 0.0, "process_on_gpu": False}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    monkeypatch.setattr(execution, "_gpu_sample", lambda uuid, pid: dict(idle))
    (tmp_path / "outputs").mkdir()
    steps = tmp_path / "evidence" / "training_observation.jsonl"
    (tmp_path / "main.py").write_text(
        "import json,time,pathlib\n"
        f"steps=pathlib.Path({str(steps)!r})\n"
        "for i in range(12):\n"
        " steps.open('a').write('{}\\n')\n"
        " open('outputs/epoch.log','a').write('x')\n"
        " time.sleep(0.5)\n"
        "open('results.json','w').write(json.dumps({'metric': 1.0}))\n")
    result = run_experiment(tmp_path, tmp_path / "evidence", watched(gpu_idle=2))
    assert not result["success"] and "idle and unused" in result["errors"][0]


GPU_UUID_2 = "GPU-66666666-7777-8888-9999-000000000000"


@pytest.mark.parametrize("value,expected", [
    (f"{GPU_UUID}, {GPU_UUID_2}", f"{GPU_UUID},{GPU_UUID_2}"),
    (f"{GPU_UUID},{GPU_UUID}", None),       # a GPU listed twice
    (f"{GPU_UUID},", None),                 # empty item
])
def test_one_experiment_may_use_several_gpus(value, expected):
    if expected is None:
        with pytest.raises(ValueError):
            validate_gpu_uuid(value)
    else:
        assert validate_gpu_uuid(value) == expected


def test_multi_gpu_execution_locks_and_checks_every_gpu(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from researchclaw.training import resources

    locked, checked, observed = [], [], []

    @contextmanager
    def lock(path):
        locked.append(path.stem)
        yield

    monkeypatch.setattr(resources, "job_lock", lock)
    monkeypatch.setattr(resources, "require_idle_gpu", checked.append)

    @with_execution_resources(Stage.EXPERIMENT_RUN)
    def execute(*args, **kwargs):
        observed.append(os.environ["CUDA_VISIBLE_DEVICES"])

    execute(tmp_path, tmp_path, configuration(gpu=f"{GPU_UUID_2},{GPU_UUID}"), None)
    assert locked == checked == sorted([GPU_UUID, GPU_UUID_2])
    assert observed == [f"{GPU_UUID_2},{GPU_UUID}"]


@pytest.mark.parametrize("training,expected", [(True, "torch.distributed.run"), (False, "main.py")])
def test_multi_gpu_training_starts_one_process_per_gpu(tmp_path, monkeypatch, training, expected):
    from researchclaw.training import execution

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{GPU_UUID},{GPU_UUID_2}")
    (tmp_path / "main.py").write_text("")

    def popen(cmd, **kwargs):
        raise OSError("not started in this test")

    monkeypatch.setattr(execution.subprocess, "Popen", popen)
    result = run_experiment(tmp_path, tmp_path / "evidence", configuration(training=training))
    assert expected in result["command"]
    if training:
        assert "--nproc_per_node=2" in result["command"] and result["world_size"] == 2
        assert result["command"][-2:] == ["researchclaw.training.observed",
                                          str((tmp_path / "evidence/training_observation.json").resolve())]


def _rank_file(path, gpu, steps=3, error=None):
    payload = {"optimizer_steps": steps, "cuda_update_observed": steps > 0, "last_loss": 0.5,
               "gpu_process": {"gpu_uuid": gpu, "pid": 1, "memory_mib": 900.0} if steps else None}
    if error:
        payload["error"] = error
    path.write_text(json.dumps(payload))


@pytest.mark.parametrize("ranks,problem", [
    ([GPU_UUID_2], None),
    ([], "Missing training evidence of rank 1"),
    ([(GPU_UUID_2, 0)], "no GPU parameter update"),
    ([GPU_UUID], "cover every allocated GPU"),          # both ranks on the same GPU
    ([(GPU_UUID_2, 3, "NCCL timeout")], "Rank 1 failed"),
])
def test_every_rank_must_train_on_its_own_gpu(tmp_path, ranks, problem):
    from researchclaw.training.execution import _check_ranks
    from researchclaw.training.observed import rank_evidence_path

    evidence = tmp_path / "training_observation.json"
    _rank_file(evidence, GPU_UUID)
    for rank, item in enumerate(ranks, start=1):
        _rank_file(rank_evidence_path(evidence, rank), *(item if isinstance(item, tuple) else (item,)))
    if problem is None:
        assert _check_ranks(evidence, [GPU_UUID, GPU_UUID_2])[0]["gpu_process"]["gpu_uuid"] == GPU_UUID_2
    else:
        with pytest.raises(ValueError, match=problem):
            _check_ranks(evidence, [GPU_UUID, GPU_UUID_2])


def test_watchdog_kills_a_multi_gpu_run_that_leaves_one_gpu_idle(tmp_path, monkeypatch):
    from researchclaw.training import execution

    busy = {"utilization": 90.0, "memory_mib": 9000.0, "process_memory_mib": 9000.0, "process_on_gpu": True}
    idle = {"utilization": 0.0, "memory_mib": 0.0, "process_memory_mib": 0.0, "process_on_gpu": False}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{GPU_UUID},{GPU_UUID_2}")
    monkeypatch.setattr(execution, "_gpu_sample", lambda uuid, pid: dict(busy if uuid == GPU_UUID else idle))
    (tmp_path / "main.py").write_text("import json,time\nfor epoch in range(8):\n print(epoch, flush=True)\n"
                                      " time.sleep(0.5)\nopen('results.json','w').write(json.dumps({'metric': 1.0}))\n")
    result = run_experiment(tmp_path, tmp_path / "evidence", watched(gpu_idle=2))
    assert not result["success"] and GPU_UUID_2 in result["errors"][0]


@pytest.mark.parametrize("drift,tensors,accepted", [
    ("1+1e-6", "weight", True),        # in-place renormalisation (BayarConv2d) after saving: float rounding
    ("1+1e-3", "weight", False),       # a real change is not rounding
    ("1+1e-6", "both", False),         # only max(1, 1%) tensors may differ from their saved bits
])
def test_checkpoint_tolerates_in_place_rounding_of_one_tensor(tmp_path, monkeypatch, drift, tensors, accepted):
    pytest.importorskip("torch")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # Saved from copies, so only the comparison with the parameters at exit can match.
    script = ("import torch,json\nfrom pathlib import Path\nmodel=torch.nn.Linear(2,1)\n"
              "optimizer=torch.optim.SGD(model.parameters(),lr=0.1)\n"
              "for step in range(3):\n optimizer.zero_grad()\n (model(torch.ones(2,2))-2).square().mean().backward()\n"
              " optimizer.step()\nPath('outputs').mkdir()\n"
              "torch.save({k:v.clone() for k,v in model.state_dict().items()},'outputs/model.pt')\n"
              "with torch.no_grad():\n model.weight.mul_(%s)\n" % drift)
    if tensors == "both":
        script += " model.bias.mul_(%s)\n" % drift
    script += PREDICTIONS + "Path('results.json').write_text(json.dumps({'loss': 0.5}))\n"
    (tmp_path / "main.py").write_text(script)
    result = run_experiment(tmp_path, tmp_path / "evidence", configuration(training=True, metric="loss"))
    assert result["success"] == accepted, result
    if accepted:
        assert result["training"]["checkpoint_matches"][0]["numeric_matches"] == 1


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
    script += PREDICTIONS + "Path('results.json').write_text(json.dumps({'loss': float(model(torch.ones(2,2)).square().mean().item())}))\n"
    (tmp_path / "main.py").write_text(script)
    result = run_experiment(tmp_path, tmp_path / "evidence", configuration(training=True, metric="loss"))
    assert result["success"] == accepted, result
    if accepted:
        assert result["training"]["optimizer_steps"] == 3
        assert result["training"]["checkpoint_verified"]
        assert not result["training"]["cuda_update_observed"]
        assert len((tmp_path / "evidence/training_observation.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("per_seed,final,accepted", [
    ("trained", "last", True),      # each seed saved; model.pt is the delivered last seed
    (None, "last", False),          # model.pt overwritten per seed and nothing else kept
    ("untrained", "last", False),   # per-seed files saved before training
    ("trained", "fresh", False),    # per-seed files fine but model.pt was never trained
])
def test_independent_seed_models_each_need_a_matching_checkpoint(tmp_path, monkeypatch, per_seed, final, accepted):
    pytest.importorskip("torch")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    script = ("import torch,json\nfrom pathlib import Path\nPath('outputs/full').mkdir(parents=True)\n"
              "for seed in (42,123,456):\n torch.manual_seed(seed)\n model=torch.nn.Linear(2,1)\n"
              " optimizer=torch.optim.SGD(model.parameters(),lr=0.1)\n")
    if per_seed == "untrained":
        script += " torch.save(model.state_dict(),f'outputs/full/final_{seed}.pt')\n"
    script += " for step in range(2):\n  optimizer.zero_grad()\n  (model(torch.ones(2,2))-2).square().mean().backward()\n  optimizer.step()\n"
    if per_seed == "trained":
        script += " torch.save(model.state_dict(),f'outputs/full/final_{seed}.pt')\n"
    script += " torch.save((model if %r=='last' else torch.nn.Linear(2,1)).state_dict(),'outputs/model.pt')\n" % final
    script += PREDICTIONS + "Path('results.json').write_text(json.dumps({'loss': 0.5}))\n"
    (tmp_path / "main.py").write_text(script)
    result = run_experiment(tmp_path, tmp_path / "evidence", configuration(training=True, metric="loss"))
    assert result["success"] == accepted, result
    matches = result["training"]["checkpoint_matches"]
    assert result["training"]["trained_optimizers"] == 3 and len(matches) == 3
    if accepted:
        assert [match["checkpoint"] for match in matches] == [
            "outputs/full/final_42.pt", "outputs/full/final_123.pt", "outputs/model.pt"]


@pytest.mark.parametrize("delivered,accepted", [
    ("final", True),        # model.pt = final weights, then best reloaded for testing
    ("renormalised", True), # 2026-10-19 S15: as above, but validation after the last step renormalised a
                            # weight in place (BayarConv2d assigns kernel.data on every forward pass)
    ("best", True),         # model.pt saved after reloading the best checkpoint
    ("fresh", False),       # model.pt never trained
    ("overwritten", False), # saved from the trained model, then replaced without torch.save
])
def test_model_saved_before_reloading_best_checkpoint_is_accepted(tmp_path, monkeypatch, delivered, accepted):
    pytest.importorskip("torch")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    script = ("import io,torch,json\nfrom pathlib import Path\nPath('outputs').mkdir()\n"
              "for seed in (42,123,456):\n torch.manual_seed(seed)\n model=torch.nn.Linear(2,1)\n"
              " optimizer=torch.optim.SGD(model.parameters(),lr=0.1)\n"
              " for step in range(3):\n  optimizer.zero_grad()\n  (model(torch.ones(2,2))-2).square().mean().backward()\n"
              "  optimizer.step()\n  if step==0:torch.save(model.state_dict(),f'outputs/best_{seed}.pt')\n"
              f" if {delivered!r}=='renormalised':model.weight.data=model.weight.data/model.weight.data.sum()\n"
              " torch.save(model.state_dict(),f'outputs/final_{seed}.pt')\n"
              f" if {delivered!r} in ('final','renormalised','overwritten'):torch.save(model.state_dict(),'outputs/model.pt')\n"
              " model.load_state_dict(torch.load(f'outputs/best_{seed}.pt'))\n"
              f" if {delivered!r}=='best':torch.save(model.state_dict(),'outputs/model.pt')\n"
              f" if {delivered!r}=='fresh':torch.save(torch.nn.Linear(2,1).state_dict(),'outputs/model.pt')\n"
              f"if {delivered!r}=='overwritten':\n buffer=io.BytesIO();torch.save(torch.nn.Linear(2,1).state_dict(),buffer)\n"
              " Path('outputs/model.pt').write_bytes(buffer.getvalue())\n")
    script += PREDICTIONS + "Path('results.json').write_text(json.dumps({'loss': 0.5}))\n"
    (tmp_path / "main.py").write_text(script)
    result = run_experiment(tmp_path, tmp_path / "evidence", configuration(training=True, metric="loss"))
    assert result["success"] == accepted, result
    matches = result["training"]["checkpoint_matches"]
    if accepted:
        assert [(match["checkpoint"], match["state"]) for match in matches] == [
            ("outputs/final_42.pt", "saved_from_trained_model"), ("outputs/final_123.pt", "saved_from_trained_model"),
            ("outputs/model.pt", "saved_from_trained_model")]
    else:
        assert "outputs/model.pt matches none of the trained models" in result["errors"][0]


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


def _run_s15_with_edit(tmp_path, monkeypatch, refined_lr):
    from researchclaw.pipeline.iterative_refine import runtime
    from researchclaw.pipeline.claw_engine.turn_loop import TurnResult

    script = ("import json\nfrom pathlib import Path\n"
              "Path('results.json').write_text(json.dumps({'metric': %s, 'protocol': {'learning_rate': %s, 'epochs': 10},"
              " 'split_ids': {'test': ['a', 'b']}}))\n")
    experiment = tmp_path / "stage-11/experiment"
    experiment.mkdir(parents=True)
    (experiment / "main.py").write_text(script % (0.5, 1e-05))
    runs = tmp_path / "stage-14/runs"
    runs.mkdir(parents=True)
    (runs / "results.json").write_text(json.dumps(
        {"metric": 0.5, "protocol": {"learning_rate": 1e-05, "epochs": 10}, "split_ids": {"test": ["a", "b"]}}))

    def loop(**kwargs):
        def run_turn(message):
            (kwargs["workspace"] / "main.py").write_text(script % (0.4, refined_lr))
            return TurnResult(final_text="done")
        return SimpleNamespace(run_turn=run_turn)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(runtime, "AgentTurnLoop", loop)
    stage = tmp_path / "stage-15"
    result = runtime.IterativeRefineRuntime().execute(stage, tmp_path, configuration(), None,
                                                    llm=SimpleNamespace(config=SimpleNamespace(primary_model="test")))
    assert result.status == StageStatus.DONE
    return stage, experiment, json.loads((stage / "refinement_log.json").read_text())


def test_s15_rejects_run_that_changes_hyperparameters(tmp_path, monkeypatch):
    stage, experiment, log = _run_s15_with_edit(tmp_path, monkeypatch, refined_lr=3e-05)
    assert log["adopted"] is False and log["refinement_rejected"] is True
    assert log["protocol_changes"] == {"protocol.learning_rate": {"initial": 1e-05, "refined": 3e-05}}
    assert log["final_metric"] == 0.5 and log["exploratory_metric"] == 0.4 and log["improved"] is False
    assert json.loads((stage / "results.json").read_text())["metric"] == 0.5
    assert json.loads((stage / "experiment_final/results.json").read_text())["metric"] == 0.5
    assert json.loads((stage / "exploratory_results.json").read_text())["metric"] == 0.4
    assert "1e-05" in (experiment / "main.py").read_text()
    assert "1e-05" in (stage / "experiment_final/main.py").read_text()


def test_s15_adopts_bug_fix_that_keeps_protocol(tmp_path, monkeypatch):
    stage, experiment, log = _run_s15_with_edit(tmp_path, monkeypatch, refined_lr=1e-05)
    assert log["adopted"] is True and "refinement_rejected" not in log
    assert log["final_metric"] == 0.4
    assert json.loads((stage / "results.json").read_text())["metric"] == 0.4
    assert "0.4" in (experiment / "main.py").read_text()


def test_s15_prompt_forbids_tuning():
    from researchclaw.pipeline.iterative_refine.system_prompt import build_system_prompt, build_user_message

    prompt = build_system_prompt() + build_user_message(experiment_dir="w", experiment_files=["main.py"])
    assert "lr tuning" not in prompt and "Hyperparameter tuning" not in prompt
    assert "NOT a tuning stage" in prompt and "Frozen protocol" in prompt


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


@pytest.mark.parametrize("status", ["blocked", "failed", "incomplete", "skipped",
                                    "blocked_architecture_prerequisite", "Failed: OOM"])
def test_supervisor_rejects_explicit_failure_status_even_with_metric(tmp_path, status):
    payload = json.dumps({"status": status, "metric": 0.5, "blocker": "approval required"})
    (tmp_path / "main.py").write_text(
        f"from pathlib import Path\nPath('results.json').write_text({payload!r})\n"
    )

    result = run_experiment(tmp_path, tmp_path / "evidence", configuration(), smoke=True)

    assert not result["success"]
    assert f"Experiment reported {status}" in result["errors"][0]
    assert result["results"]["blocker"] == "approval required"


def test_smoke_metadata_without_requested_metric_is_rejected(tmp_path):
    (tmp_path / "main.py").write_text(
        "from pathlib import Path\nPath('results.json').write_text('{\"seed\":42}')\n"
    )

    result = run_experiment(tmp_path, tmp_path / "evidence", configuration(), smoke=True)

    assert not result["success"]
    assert "missing measured metric" in result["errors"][0]


def test_s12_valid_source_needs_no_llm_repair(tmp_path, monkeypatch):
    from researchclaw.pipeline.sanity_check import runtime

    experiment = tmp_path / "stage-11/experiment"
    experiment.mkdir(parents=True)
    (experiment / "main.py").write_text(
        "from pathlib import Path\nPath('results.json').write_text('{\"metric\":0.5}')\n"
    )
    monkeypatch.setattr(runtime, "AgentTurnLoop", lambda **kwargs: SimpleNamespace(
        run_turn=lambda message: pytest.fail("Passing smoke needs no LLM repair"),
    ))

    result = runtime.SanityCheckRuntime().execute(
        tmp_path / "stage-12", tmp_path, configuration(), None,
        llm=SimpleNamespace(config=SimpleNamespace(primary_model="test")),
    )

    assert result.status == StageStatus.DONE
    report = json.loads((tmp_path / "stage-12/sanity_report.json").read_text())
    assert report["iterations"] == 0
    assert len(report["attempts"]) == 1
    assert report["execution"]["results"] == {"metric": 0.5}
    assert (tmp_path / "stage-12/smoke_execution/attempt-1/execution.log").is_file()


def test_s12_feedback_repairs_nested_result_save_in_source(tmp_path, monkeypatch):
    from researchclaw.pipeline.sanity_check import runtime
    from researchclaw.pipeline.claw_engine.turn_loop import TurnResult

    experiment = tmp_path / "stage-11/experiment"
    experiment.mkdir(parents=True)
    source = (
        "from pathlib import Path\n"
        "LR = 1e-5\n"
        "OUT = Path('outputs/smoke')\n"
        "OUT.mkdir(parents=True, exist_ok=True)\n"
        "RESULT = OUT / 'results.json'\n"
        "RESULT.write_text('{\"primary_metric\":0.4,\"status\":\"smoke_completed\"}')\n"
    )
    (experiment / "main.py").write_text(source)
    (experiment / "results.json").write_text('{"primary_metric":0.99}')
    messages = []

    def repair_loop(**kwargs):
        def repair(message):
            messages.append(message)
            path = kwargs["workspace"] / "main.py"
            path.write_text(path.read_text().replace("RESULT = OUT / 'results.json'", "RESULT = Path('results.json')"))
            return TurnResult(iterations=1, tool_calls=1)

        return SimpleNamespace(run_turn=repair)

    monkeypatch.setattr(runtime, "AgentTurnLoop", repair_loop)

    result = runtime.SanityCheckRuntime().execute(
        tmp_path / "stage-12", tmp_path, configuration(metric="primary_metric"), None,
        llm=SimpleNamespace(config=SimpleNamespace(primary_model="test")),
    )

    assert result.status == StageStatus.DONE
    assert len(messages) == 1
    assert "Experiment did not write a fresh results.json" in messages[0]
    assert "Execution log:" in messages[0]
    report = json.loads((tmp_path / "stage-12/sanity_report.json").read_text())
    assert not report["attempts"][0]["success"]
    assert report["attempts"][1]["success"]
    assert report["execution"]["results"]["primary_metric"] == 0.4
    assert report["files_fixed"] == 1
    assert "LR = 1e-5" in (experiment / "main.py").read_text()
    assert "RESULT = Path('results.json')" in (experiment / "main.py").read_text()
    latest = json.loads((tmp_path / "stage-12/smoke_execution/execution_report.json").read_text())
    assert latest == report["attempts"][1]


def test_s12_manually_written_results_cannot_pass_rerun(tmp_path, monkeypatch):
    from researchclaw.pipeline.sanity_check import runtime
    from researchclaw.pipeline.claw_engine.turn_loop import TurnResult

    experiment = tmp_path / "stage-11/experiment"
    experiment.mkdir(parents=True)
    source = "print('no measurements')\n"
    (experiment / "main.py").write_text(source)

    def repair_loop(**kwargs):
        def repair(message):
            (kwargs["workspace"] / "results.json").write_text('{"metric":0.99}')
            return TurnResult(final_text="smoke passed")

        return SimpleNamespace(run_turn=repair)

    monkeypatch.setattr(runtime, "AgentTurnLoop", repair_loop)

    result = runtime.SanityCheckRuntime().execute(
        tmp_path / "stage-12", tmp_path, configuration(), None,
        llm=SimpleNamespace(config=SimpleNamespace(primary_model="test")),
    )

    assert result.status == StageStatus.FAILED
    assert result.decision == "retry"
    assert "fresh results.json" in result.error
    assert (experiment / "main.py").read_text() == source
    report = json.loads((tmp_path / "stage-12/sanity_report.json").read_text())
    assert len(report["attempts"]) == 2
    assert not any(attempt["success"] for attempt in report["attempts"])


def test_s12_failed_fix_is_not_copied_back(tmp_path, monkeypatch):
    from researchclaw.pipeline.sanity_check import runtime
    from researchclaw.pipeline.claw_engine.turn_loop import TurnResult

    experiment = tmp_path / "stage-11/experiment"
    experiment.mkdir(parents=True)
    source = "raise SystemExit(7)\n"
    (experiment / "main.py").write_text(source)

    def repair_loop(**kwargs):
        def repair(message):
            assert "exited with 7" in message
            (kwargs["workspace"] / "main.py").write_text(
                "from pathlib import Path\nPath('results.json').write_text('{\"seed\":42}')\n"
            )
            return TurnResult(iterations=1)

        return SimpleNamespace(run_turn=repair)

    monkeypatch.setattr(runtime, "AgentTurnLoop", repair_loop)

    result = runtime.SanityCheckRuntime().execute(
        tmp_path / "stage-12", tmp_path, configuration(), None,
        llm=SimpleNamespace(config=SimpleNamespace(primary_model="test")),
    )

    assert result.status == StageStatus.FAILED
    assert "missing measured metric" in result.error
    assert (experiment / "main.py").read_text() == source
    report = json.loads((tmp_path / "stage-12/sanity_report.json").read_text())
    assert report["files_fixed"] == 0


def test_s12_prompt_describes_fresh_root_results_and_preserves_protocol():
    from researchclaw.pipeline.sanity_check.system_prompt import build_system_prompt, build_user_message

    prompt = build_system_prompt(python_path="/approved/python", workspace_path="/workspace", metric_key="loss")

    assert "SMOKE_TEST=1 /approved/python main.py" in prompt
    assert "workspace-root results.json" in prompt
    assert "written by main.py itself" in prompt
    assert "finite measured `loss`" in prompt
    assert "Preserve approved splits, optimizer, learning rate and schedule" in prompt
    assert "held-out test split" in prompt
    user_message = build_user_message(experiment_dir="/workspace", experiment_files=["main.py"], python_path="/approved/python")
    assert "SMOKE_TEST=1 /approved/python main.py" in user_message
    assert "SMOKE_TEST=1 python3 main.py" not in user_message


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


@pytest.mark.parametrize("status", ["completed", "ok", "unblocked", "success"])
def test_supervisor_accepts_non_failure_status(tmp_path, status):
    payload = json.dumps({"status": status, "metric": 0.5})
    (tmp_path / "main.py").write_text(
        f"from pathlib import Path\nPath('results.json').write_text({payload!r})\n"
    )

    result = run_experiment(tmp_path, tmp_path / "evidence", configuration(), smoke=True)

    assert result["success"], result["errors"]


def _run_s15_refine_loop(tmp_path, monkeypatch, refined_protocol):
    from researchclaw.pipeline.iterative_refine import runtime
    from researchclaw.pipeline.claw_engine.turn_loop import TurnResult

    script = "import json\nfrom pathlib import Path\nPath('results.json').write_text(json.dumps(%r))\n"
    initial = {"metric": 0.5, "protocol": {"learning_rate": 1e-05, "seeds": [1, 2, 3]}}
    experiment = tmp_path / "stage-11/experiment"
    experiment.mkdir(parents=True)
    (experiment / "main.py").write_text(script % initial)
    runs = tmp_path / "stage-14/runs"
    runs.mkdir(parents=True)
    (runs / "results.json").write_text(json.dumps({"metric": 0.9, "protocol": {"learning_rate": 1e-05}}))
    adopted = tmp_path / "stage-15_v1"
    adopted.mkdir()
    (adopted / "results.json").write_text(json.dumps(initial))
    (adopted / "refinement_log.json").write_text(json.dumps({"adopted": True}))
    stage = tmp_path / "stage-15"
    stage.mkdir()
    (stage / "refine_request.md").write_text("# REFINE request\nAdd seeds 4 and 5.")
    messages = []

    def loop(**kwargs):
        def run_turn(message):
            messages.append(message)
            (kwargs["workspace"] / "main.py").write_text(script % dict(initial, metric=0.4, protocol=refined_protocol))
            return TurnResult(final_text="done")
        return SimpleNamespace(run_turn=run_turn)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(runtime, "AgentTurnLoop", loop)
    result = runtime.IterativeRefineRuntime().execute(stage, tmp_path, configuration(), None,
                                                    llm=SimpleNamespace(config=SimpleNamespace(primary_model="test")))
    assert result.status == StageStatus.DONE
    return stage, messages, json.loads((stage / "refinement_log.json").read_text())


def test_s15_refine_loop_may_add_seeds(tmp_path, monkeypatch):
    stage, messages, log = _run_s15_refine_loop(
        tmp_path, monkeypatch, {"learning_rate": 1e-05, "seeds": [1, 2, 3, 4, 5]})
    assert "Add seeds 4 and 5." in messages[0]
    assert "REFINE loop" in (stage / "refine_system_prompt.md").read_text()
    assert log["refine_loop"] is True
    assert not log.get("protocol_changes") and log["adopted"] is True


def test_s15_refine_loop_still_rejects_retuning(tmp_path, monkeypatch):
    _, _, log = _run_s15_refine_loop(
        tmp_path, monkeypatch, {"learning_rate": 3e-05, "seeds": [1, 2, 3]})
    assert log["adopted"] is False
    assert list(log["protocol_changes"]) == ["protocol.learning_rate"]


def test_protocol_changes_allow_added_seeds_only_in_refine_loop():
    from researchclaw.pipeline.iterative_refine.runtime import IterativeRefineRuntime

    base = json.dumps({"protocol": {"seeds": [1, 2, 3]}})
    grown = json.dumps({"protocol": {"seeds": [1, 2, 3, 4]}})
    dropped = json.dumps({"protocol": {"seeds": [1, 2]}})
    assert IterativeRefineRuntime._protocol_changes(base, grown)
    assert IterativeRefineRuntime._protocol_changes(base, grown, allow_additions=True) == {}
    assert IterativeRefineRuntime._protocol_changes(base, dropped, allow_additions=True)
