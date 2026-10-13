import json
import os
import sys
import time
from pathlib import Path

import pytest

from researchclaw.training import jobs
from researchclaw.training.worker import score_masks


def spec_fixture():
    return {
        "source_dir": "source", "split_manifest": "splits.json", "dataset_root": "data",
        "initial_checkpoint": "initial.pt", "initial_checkpoint_sha256": "a" * 64,
    }


@pytest.mark.parametrize("field,value", [
    ("epochs", 0), ("steps_per_epoch", True), ("learning_rate", float("nan")),
    ("checkpoint_every_steps", -1), ("initial_checkpoint_sha256", "unverified"),
    ("unknown", "ignored"), ("learning_rate", 0),
])
def test_invalid_training_specs_fail_closed(field, value):
    spec = spec_fixture()
    spec[field] = value
    with pytest.raises(ValueError):
        jobs.validate_spec(spec)


def test_gpu_requires_explicit_idle_uuid(monkeypatch):
    def inventory(query, *, applications=False):
        return [["GPU-test", "0", "0"]] if not applications else [["GPU-test", "99", "300"]]

    monkeypatch.setattr(jobs, "gpu_rows", inventory)
    with pytest.raises(ValueError, match="UUID"):
        jobs.require_idle_gpu("0")
    with pytest.raises(RuntimeError, match="occupied"):
        jobs.require_idle_gpu("GPU-test")
    monkeypatch.setattr(jobs, "gpu_rows", lambda query, applications=False: [] if applications else [["GPU-test", "0", "0"]])
    jobs.require_idle_gpu("GPU-test")
    with pytest.raises(ValueError, match="absent"):
        jobs.require_idle_gpu("GPU-other")


def test_gpu_probe_errors_do_not_fall_back_to_zero(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(jobs, "gpu_rows", unavailable)
    with pytest.raises(FileNotFoundError):
        jobs.require_idle_gpu("GPU-test")


def test_gpu_evidence_must_belong_to_this_process(monkeypatch):
    monkeypatch.setattr(jobs, "gpu_rows", lambda *args, **kwargs: [["GPU-test", "-1", "100"]])
    with pytest.raises(RuntimeError, match="attribute"):
        jobs.gpu_evidence("GPU-test")
    monkeypatch.setattr(jobs, "gpu_rows", lambda *args, **kwargs: [["GPU-test", str(os.getpid()), "100"]])
    assert jobs.gpu_evidence("GPU-test")["pid"] == os.getpid()


def test_status_detects_pid_reuse_and_stalled_progress(tmp_path, monkeypatch):
    jobs.write_json(tmp_path / "state.json", {
        "status": "running", "time": time.time(), "progress_time": time.time() - 1000,
        "pid": 123, "process_token": "original",
    })
    monkeypatch.setattr(jobs, "process_token", lambda pid: "replacement")
    assert jobs.job_status(tmp_path)["status"] == "interrupted"
    monkeypatch.setattr(jobs, "process_token", lambda pid: "original")
    assert "No training/evaluation progress" in jobs.job_status(tmp_path)["alert"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux worker lock")
def test_worker_lock_rejects_concurrent_owner(tmp_path):
    with jobs.job_lock(tmp_path / "worker.lock"):
        with pytest.raises(RuntimeError, match="Another worker"):
            with jobs.job_lock(tmp_path / "worker.lock"):
                pytest.fail("Second owner acquired lock")


def test_pixel_metrics_have_no_boundary_tolerance():
    import numpy as np

    target = np.array([[1, 0], [0, 0]])
    prediction = np.array([[0, 1], [0, 0]])
    assert score_masks(prediction, target) == {"pixel_f1": 0., "pixel_iou": 0.}
    assert score_masks(target, target) == {"pixel_f1": 1., "pixel_iou": 1.}


def test_pipeline_rejects_prepared_training_job(tmp_path, monkeypatch):
    from researchclaw.training import evidence

    monkeypatch.setattr(evidence, "verify_inputs", lambda path: ({}, {}))
    jobs.write_json(tmp_path / "state.json", {"status": "prepared", "time": time.time()})
    with pytest.raises(ValueError, match="prepared"):
        evidence.verify_completed_job(tmp_path)


def test_training_config_parser():
    from researchclaw.config import _parse_experiment_config

    assert _parse_experiment_config({"training_job_dir": "/jobs/baseline"}).training_job_dir == "/jobs/baseline"
    assert _parse_experiment_config({}).training_job_dir == ""


def test_pipeline_routes_without_llm_and_disables_refinement(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from researchclaw.pipeline.experiment_run import execute_experiment_run
    from researchclaw.pipeline.iterative_refine import execute_iterative_refine
    from researchclaw.training import evidence

    calls = []

    def importer(stage_dir, job_dir, *, refinement=False):
        calls.append((job_dir, refinement))
        return "verified"

    monkeypatch.setattr(evidence, "import_training_stage", importer)
    config = SimpleNamespace(experiment=SimpleNamespace(training_job_dir="/jobs/pilot"))
    assert execute_experiment_run(tmp_path, tmp_path, config, None) == "verified"
    assert execute_iterative_refine(tmp_path, tmp_path, config, None) == "verified"
    assert calls == [(Path("/jobs/pilot"), False), (Path("/jobs/pilot"), True)]


def test_cpu_training_resume_and_reloaded_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from researchclaw.training import worker

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = torch.nn.Conv2d(1, 1, 1)

        def forward(self, image, mask, edge_mask):
            prediction = torch.sigmoid(self.predictor(image))
            return {"pred_mask": prediction, "backward_loss": ((prediction - mask) ** 2).mean()}

    samples = [{"id": str(index), "image": torch.ones(1, 8, 8) * (index + 1),
                "mask": torch.ones(1, 8, 8), "edge_mask": torch.zeros(1, 8, 8)} for index in range(3)]
    spec = {"epochs": 2, "steps_per_epoch": 3, "learning_rate": .01}
    jobs.write_json(tmp_path / "spec.json", spec)
    jobs.write_json(tmp_path / "state.json", {"status": "running", "time": time.time()})
    original_save = worker.save_checkpoint

    def interrupted_save(path, payload):
        result = original_save(path, payload)
        if payload["global_step"] == 2:
            raise RuntimeError("simulated interruption after durable checkpoint")
        return result

    torch.manual_seed(42)
    initial_model = TinyModel()
    initial_state = {name: tensor.clone() for name, tensor in initial_model.state_dict().items()}
    monkeypatch.setattr(worker, "save_checkpoint", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        worker.fit(initial_model, samples, samples, spec, tmp_path, "cpu")
    assert worker.load_committed_checkpoint(tmp_path)["global_step"] == 2
    monkeypatch.setattr(worker, "save_checkpoint", original_save)
    restarted_model = TinyModel()
    proof = worker.fit(restarted_model, samples, samples, spec, tmp_path, "cpu", resume=True)
    assert proof["optimizer_steps"] == 6
    assert proof["initial_parameter_sha256"] != proof["trained_parameter_sha256"]
    assert proof["gpu_evidence"] is None
    assert len(list(tmp_path.glob("checkpoint-*.pt"))) <= 2
    results = worker.evaluate(restarted_model, samples, "cpu", output_dir=tmp_path / "predictions")
    assert len(results["records"]) == 3
    assert all((tmp_path / "predictions" / record["prediction"]).is_file() for record in results["records"])
    uninterrupted = tmp_path / "uninterrupted"
    uninterrupted.mkdir()
    jobs.write_json(uninterrupted / "spec.json", spec)
    jobs.write_json(uninterrupted / "state.json", {"status": "running", "time": time.time()})
    baseline_model = TinyModel()
    baseline_model.load_state_dict(initial_state)
    baseline = worker.fit(baseline_model, samples, samples, spec, uninterrupted, "cpu")
    assert proof["trained_parameter_sha256"] == baseline["trained_parameter_sha256"]
    selected = tmp_path / proof["selected_checkpoint"]["filename"]
    selected.write_bytes(b"damaged checkpoint")
    reference = jobs.read_json(tmp_path / "resume.json")
    (tmp_path / reference["filename"]).write_bytes(b"damaged checkpoint")
    with pytest.raises(ValueError, match="hash mismatch"):
        worker.load_committed_checkpoint(tmp_path)


def test_evaluation_rejects_nan_predictions():
    torch = pytest.importorskip("torch")
    from researchclaw.training.worker import evaluate

    class InvalidModel(torch.nn.Module):
        def forward(self, image, mask, edge_mask):
            return {"pred_mask": torch.full_like(mask, float("nan"))}

    samples = [{"id": "1", "image": torch.ones(1, 8, 8), "mask": torch.ones(1, 8, 8), "edge_mask": torch.zeros(1, 8, 8)}]
    with pytest.raises(ValueError, match="Invalid prediction"):
        evaluate(InvalidModel(), samples, "cpu")


def test_prepare_preserves_reference_and_rejects_unapproved_split(tmp_path, monkeypatch):
    source = tmp_path / "reference"
    source.mkdir()
    (source / "models").mkdir()
    (source / "common").mkdir()
    (source / "cotransformer.py").write_text("MODEL = 'reference'\n")
    (source / "models" / "model.py").write_text("VALUE = 1\n")
    (source / "common" / "helper.py").write_text("VALUE = 2\n")
    dataset = tmp_path / "data"
    dataset.mkdir()
    splits = tmp_path / "splits.json"
    jobs.write_json(splits, {"protocol_status": "proposal_only"})
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"checkpoint-not-deserialized-during-preparation")
    spec = {**spec_fixture(), "source_dir": str(source), "dataset_root": str(dataset),
            "split_manifest": str(splits), "initial_checkpoint": str(checkpoint),
            "initial_checkpoint_sha256": jobs.file_sha256(checkpoint)}
    spec_path = tmp_path / "spec.json"
    jobs.write_json(spec_path, spec)
    before = jobs.source_hashes(source)

    def check_manifest(manifest, **kwargs):
        if manifest["protocol_status"] != "approved":
            raise ValueError("Unapproved split")
        return {"train": 3}

    monkeypatch.setattr(jobs, "validate_splits", check_manifest)
    destination = tmp_path / "job"
    with pytest.raises(ValueError, match="Unapproved"):
        jobs.prepare_job(spec_path, destination)
    assert not destination.exists()
    jobs.write_json(splits, {"protocol_status": "approved"})
    jobs.prepare_job(spec_path, destination)
    assert before == jobs.source_hashes(source) == jobs.source_hashes(destination / "source")
    jobs.verify_inputs(destination)
    (destination / "source" / "models" / "model.py").write_text("VALUE = 99\n")
    with pytest.raises(ValueError, match="Model source changed"):
        jobs.verify_inputs(destination)
    assert jobs.source_hashes(source) == before


def test_evidence_recomputes_metrics_and_rejects_swapped_targets(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    from researchclaw.training import evidence

    masks = tmp_path / "mask"
    masks.mkdir()
    target = np.zeros((512, 512), dtype=np.uint8)
    target[:64, :64] = 255
    Image.fromarray(target).save(masks / "1forged.tif")
    predictions = tmp_path / "test_predictions"
    predictions.mkdir()
    Image.fromarray(target).save(predictions / "1-target.png")
    Image.fromarray(target).save(predictions / "1-prediction.png")
    checkpoint = tmp_path / "checkpoint-3-0-3.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    digest = jobs.file_sha256(checkpoint)
    proof = {
        "optimizer_steps": 3, "initial_parameter_sha256": "a" * 64,
        "trained_parameter_sha256": "b" * 64,
        "gpu_evidence": {"gpu_uuid": "GPU-fixture", "memory_mib": 100, "pid": 456},
        "selected_checkpoint": {"filename": checkpoint.name, "sha256": digest},
    }
    record = {"id": "1", "pixel_f1": 1., "pixel_iou": 1.,
              "prediction": "1-prediction.png", "target": "1-target.png",
              "prediction_sha256": jobs.file_sha256(predictions / "1-prediction.png"),
              "target_sha256": jobs.file_sha256(predictions / "1-target.png")}
    results = {"metrics": {"pixel_f1": 1., "pixel_iou": 1.}, "records": [record],
               "training_evidence": proof, "checkpoint_sha256": digest,
               "threshold": .5, "aggregation": "macro_per_image", "protocol": "fixture", "test_is_blind": False}
    spec = {"epochs": 1, "steps_per_epoch": 3, "dataset_root": str(tmp_path)}
    manifest = {"splits": {"test": [{"id": "1", "mask": "mask/1forged.tif"}], "train": [{"id": "2"}]}}
    monkeypatch.setattr(evidence, "verify_inputs", lambda path: (spec, manifest))
    (tmp_path / "events.jsonl").write_text("".join(json.dumps({"event": "optimizer_step", "sample_id": "2", "loss": .3, "global_step": step}) + "\n" for step in range(1, 4)))

    def commit_results():
        jobs.write_json(tmp_path / "test_results.json", results)
        jobs.write_json(tmp_path / "state.json", {"status": "completed", "gpu_uuid": "GPU-fixture", "test_results_sha256": jobs.file_sha256(tmp_path / "test_results.json")})

    commit_results()
    assert evidence.verify_completed_job(tmp_path)["metrics"]["pixel_f1"] == 1.
    results["metrics"]["pixel_f1"] = .9
    commit_results()
    with pytest.raises(ValueError, match="Aggregate"):
        evidence.verify_completed_job(tmp_path)
    results["metrics"]["pixel_f1"] = 1.
    results["records"][0]["pixel_f1"] = .9
    commit_results()
    with pytest.raises(ValueError, match="Per-image"):
        evidence.verify_completed_job(tmp_path)
    results["records"][0]["pixel_f1"] = 1.
    modified = np.zeros_like(target)
    modified[128:192, 128:192] = 255
    Image.fromarray(modified).save(predictions / "1-target.png")
    results["records"][0]["target_sha256"] = jobs.file_sha256(predictions / "1-target.png")
    commit_results()
    with pytest.raises(ValueError, match="source annotation"):
        evidence.verify_completed_job(tmp_path)


def test_official_dataset_adapter_uses_real_annotations(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("IMDLBenCo")
    import numpy as np
    from PIL import Image
    from researchclaw.training.worker import CoverageSamples

    (tmp_path / "image").mkdir()
    (tmp_path / "mask").mkdir()
    Image.new("RGB", (32, 32), (90, 100, 110)).save(tmp_path / "image" / "1t.tif")
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    Image.fromarray(mask).save(tmp_path / "mask" / "1forged.tif")
    sample = CoverageSamples(tmp_path, [{"id": "1", "image": "image/1t.tif", "mask": "mask/1forged.tif"}])[0]
    assert sample["image"].shape == (3, 512, 512)
    assert sample["mask"].shape == (1, 512, 512)
    assert torch.unique(sample["mask"]).tolist() == [0., 1.]
    assert sample["edge_mask"].sum() > 0
