from __future__ import annotations

import json
import os
import time

import pytest

from researchclaw.config import _parse_experiment_config
from researchclaw.training.execution import optional_experiments_directive
from researchclaw.training.predictions import check_predictions

np = pytest.importorskip("numpy")


def masks(folder, arrays, suffix=".npy"):
    folder.mkdir(parents=True, exist_ok=True)
    for index, array in enumerate(arrays):
        if suffix == ".npy":
            np.save(folder / f"{index}_probability.npy", np.asarray(array))
        else:
            from PIL import Image

            Image.fromarray(np.asarray(array, dtype=np.uint8)).save(folder / f"{index}_mask{suffix}")


@pytest.mark.parametrize("arrays,degenerate", [
    ([[[0.1, 0.9], [0.4, 0.2]], [[0.3, 0.3], [0.8, 0.0]]], False),
    ([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], True),                      # all-background masks
    ([[[0, 1], [1, 0]], [[0, 1], [1, 0]]], True),                      # same output for every image
    ([[[0, 0], [0, 0]], [[0, 1], [1, 0]]], False),                     # one empty image is legitimate
])
def test_array_predictions_flag_collapsed_output(tmp_path, arrays, degenerate):
    masks(tmp_path / "outputs" / "test_predictions", arrays)
    report = check_predictions(tmp_path / "outputs")
    assert bool(report["degenerate"]) == degenerate, report
    assert report["groups"][0]["checked"] == 2


def test_image_masks_and_non_finite_values(tmp_path):
    pytest.importorskip("PIL")
    masks(tmp_path / "outputs" / "seed_42" / "test_predictions", [[[0, 0], [0, 0]]] * 3, ".png")
    np.save(tmp_path / "outputs" / "pred_logits.npy", np.array([0.5, np.nan]))
    report = check_predictions(tmp_path / "outputs")
    assert report["degenerate"] == ["outputs/seed_42/test_predictions/*.png"]
    assert report["unreadable"][0]["error"] == "non-finite prediction values"


@pytest.mark.parametrize("payload,degenerate", [
    ({"predictions": [1, 0, 1]}, False),
    ({"predictions": [1, 1, 1]}, True),
    ({"predictions": []}, True),
    ([{"id": 1, "label": 0}, {"id": 2, "label": 1}], False),
])
def test_table_predictions(tmp_path, payload, degenerate):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "predictions.json").write_text(json.dumps(payload), encoding="utf-8")
    report = check_predictions(tmp_path / "outputs")
    assert bool(report["degenerate"]) == degenerate, report


def test_csv_and_unrelated_files(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "test_preds.csv").write_text("id,pred\n1,cat\n2,cat\n", encoding="utf-8")
    (tmp_path / "outputs" / "history.json").write_text("[1, 1, 1]", encoding="utf-8")
    report = check_predictions(tmp_path / "outputs")
    assert report["degenerate"] == ["outputs/test_preds.csv"]
    assert len(report["groups"]) == 1


def test_stale_predictions_from_an_earlier_run_are_ignored(tmp_path):
    masks(tmp_path / "outputs" / "predictions", [[[0.1, 0.9]], [[0.2, 0.3]]])
    old = time.time() - 3600
    for path in (tmp_path / "outputs" / "predictions").iterdir():
        os.utime(path, (old, old))
    assert check_predictions(tmp_path / "outputs", since=time.time() - 60)["groups"] == []


def test_training_run_without_predictions_is_rejected(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from .test_original_workflow_training import configuration
    from researchclaw.training.execution import run_experiment

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    script = ("import torch,json\nfrom pathlib import Path\nmodel=torch.nn.Linear(2,1)\n"
              "optimizer=torch.optim.SGD(model.parameters(),lr=0.1)\n"
              "for step in range(3):\n optimizer.zero_grad()\n (model(torch.ones(2,2))-2).square().mean().backward()\n"
              " optimizer.step()\nPath('outputs').mkdir()\ntorch.save(model.state_dict(),'outputs/model.pt')\n"
              "Path('outputs/predictions.json').write_text(json.dumps({'predictions': [0, 0, 0]}))\n"
              "Path('results.json').write_text(json.dumps({'loss': 0.5}))\n")
    (tmp_path / "main.py").write_text(script)
    result = run_experiment(tmp_path, tmp_path / "evidence", configuration(training=True, metric="loss"))
    assert not result["success"]
    assert "Degenerate test-set predictions" in result["errors"][0]


def test_optional_experiments_directive_and_config():
    assert _parse_experiment_config({}).optional_experiments is None
    assert _parse_experiment_config({"optional_experiments": []}).optional_experiments == ()
    none = optional_experiments_directive(())
    assert none.count("Not requested") == 3 and "Required" not in none
    some = optional_experiments_directive(("ablation",))
    assert "Required: ablation study" in some and some.count("Not requested") == 2


def test_cli_appends_optional_experiment_choice(tmp_path, monkeypatch):
    import argparse
    import yaml
    from researchclaw import cli
    from researchclaw.pipeline import runner
    from .test_rc_cli import _write_valid_config

    config_path = tmp_path / "config.yaml"
    _write_valid_config(config_path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["experiment"] = {"optional_experiments": []}
    data["knowledge_base"]["root"] = str(tmp_path / "kb")
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    captured = []
    monkeypatch.setattr(runner, "execute_pipeline", lambda **kwargs: captured.append(kwargs["config"]) or [])
    args = argparse.Namespace(config=str(config_path), topic="segmentation", output=str(tmp_path / "run"),
                              from_stage=None, to_stage=None, auto_approve=True, skip_preflight=True,
                              resume=False, skip_noncritical_stage=False, no_graceful_degradation=False)
    assert cli.cmd_run(args) == 0
    assert "do not plan or run any ablation experiment" in captured[0].research.topic


def test_bridge_rejects_unknown_optional_experiments(tmp_path):
    from .test_agent_bridge_reference_config import _load_agent_bridge_module

    bridge = _load_agent_bridge_module()
    state = bridge.BridgeState(runs_base_dir=str(tmp_path))
    with pytest.raises(ValueError):
        bridge._generate_config_from_template(state, "p", "topic", optional_experiments=["everything"])
