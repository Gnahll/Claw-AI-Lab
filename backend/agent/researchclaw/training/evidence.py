"""Verify worker artifacts before making them available to paper stages."""

from __future__ import annotations

import json
import math
from pathlib import Path

from researchclaw.data.coverage import file_sha256
from researchclaw.training.jobs import job_status, read_json, verify_inputs, write_json
from researchclaw.training.worker import score_masks


def verify_completed_job(job_dir: Path) -> dict:
    import cv2
    import numpy as np
    from PIL import Image

    spec, manifest = verify_inputs(job_dir)
    state = job_status(job_dir)
    if state["status"] != "completed":
        raise ValueError(f"Training job is {state['status']}; inspect training status/logs before continuing")
    result_path = job_dir / "test_results.json"
    if file_sha256(result_path) != state["test_results_sha256"]:
        raise ValueError("Test result hash mismatch")
    results = read_json(result_path)
    proof = results["training_evidence"]
    expected_steps = spec.get("epochs", 1) * spec.get("steps_per_epoch", 3)
    if proof["optimizer_steps"] != expected_steps or expected_steps < 1:
        raise ValueError("Training did not complete the declared optimizer steps")
    if proof["initial_parameter_sha256"] == proof["trained_parameter_sha256"]:
        raise ValueError("No trainable parameter change")
    gpu = proof.get("gpu_evidence") or {}
    if gpu.get("gpu_uuid") != state.get("gpu_uuid") or gpu.get("memory_mib", 0) <= 0 or gpu.get("pid", 0) <= 0:
        raise ValueError("Missing process-attributed GPU evidence")
    selected = proof["selected_checkpoint"]
    if Path(selected["filename"]).name != selected["filename"]:
        raise ValueError("Invalid selected checkpoint path")
    digest = file_sha256(job_dir / selected["filename"])
    if digest != selected["sha256"] or digest != results["checkpoint_sha256"]:
        raise ValueError("Test did not reference the saved trained checkpoint")
    expected_ids = {record["id"] for record in manifest["splits"]["test"]}
    test_records = {record["id"]: record for record in manifest["splits"]["test"]}
    actual_ids = [record["id"] for record in results["records"]]
    if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
        raise ValueError("Test samples were missing, duplicated or substituted")
    if results.get("threshold") != .5 or results.get("aggregation") != "macro_per_image":
        raise ValueError("Unexpected evaluation protocol")
    recomputed = []
    for record in results["records"]:
        masks = {}
        for kind in ("prediction", "target"):
            filename = record[kind]
            if Path(filename).name != filename:
                raise ValueError("Invalid prediction artifact path")
            path = job_dir / "test_predictions" / filename
            if file_sha256(path) != record[f"{kind}_sha256"]:
                raise ValueError("Prediction artifact changed")
            with Image.open(path) as image:
                pixels = np.array(image.convert("L"))
                if pixels.shape != (512, 512) or not np.isin(pixels, [0, 255]).all():
                    raise ValueError("Unexpected saved mask encoding/geometry")
                masks[kind] = pixels > 0
        with Image.open(Path(spec["dataset_root"]) / test_records[record["id"]]["mask"]) as image:
            original_target = (np.array(image.convert("L")) > 0).astype(np.uint8)
        resized_target = cv2.resize(original_target, (512, 512), interpolation=cv2.INTER_NEAREST) > 0
        if not np.array_equal(resized_target, masks["target"]):
            raise ValueError("Saved test target differs from the approved source annotation")
        scores = score_masks(masks["prediction"], masks["target"])
        for key, value in scores.items():
            if not math.isclose(value, record[key], abs_tol=1e-12, rel_tol=0):
                raise ValueError("Per-image metric does not match saved masks")
        recomputed.append(scores)
    metrics = {key: sum(record[key] for record in recomputed) / len(recomputed) for key in ("pixel_f1", "pixel_iou")}
    for key, value in metrics.items():
        if not math.isclose(value, results["metrics"][key], abs_tol=1e-12, rel_tol=0):
            raise ValueError("Aggregate metric does not match per-image results")
    training_ids = {record["id"] for record in manifest["splits"]["train"]}
    witnessed_steps = set()
    with (job_dir / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "optimizer_step":
                if event["sample_id"] not in training_ids or not math.isfinite(event["loss"]):
                    raise ValueError("Invalid optimizer event")
                witnessed_steps.add(event["global_step"])
    if not set(range(1, expected_steps + 1)) <= witnessed_steps:
        raise ValueError("Missing optimizer step events")
    return {"metrics": metrics, "training_evidence": proof, "test_images": len(actual_ids),
            "protocol": results["protocol"], "test_is_blind": results["test_is_blind"]}


def import_training_stage(stage_dir: Path, job_dir: Path, *, refinement: bool = False):
    from researchclaw.pipeline.executor import StageResult
    from researchclaw.pipeline.stages import Stage, StageStatus

    stage = Stage.ITERATIVE_REFINE if refinement else Stage.EXPERIMENT_RUN
    stage_dir.mkdir(parents=True, exist_ok=True)
    try:
        verified = verify_completed_job(job_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return StageResult(stage=stage, status=StageStatus.FAILED, artifacts=(), error=str(exc))
    if refinement:
        write_json(stage_dir / "refinement_disabled.json", {
            "reason": "Baseline pilot only; no test-guided tuning or optional ablations",
            "training_job": str(job_dir),
        })
        return StageResult(stage=stage, status=StageStatus.DONE, artifacts=("refinement_disabled.json",))
    runs = stage_dir / "runs"
    runs.mkdir(exist_ok=True)
    write_json(runs / "run_report.json", {
        "success": True, "has_results": True, "results": verified["metrics"],
        "training_job": str(job_dir), "protocol": verified["protocol"],
        "test_is_blind": verified["test_is_blind"],
        "evidence": verified["training_evidence"],
        "run_count_basis": "single_trained_model",
    })
    (stage_dir / "training_limitations.md").write_text(
        "# Training evidence scope\n\nOne Co-Transformers COVERAGE fine-tuning pilot; "
        "not original-paper reproduction, a blind benchmark, or evidence for either proposed architecture innovation. "
        "Reported metrics use resized 512x512 masks, threshold 0.5, macro per-image pixel F1/IoU. "
        "No ablations or independent training repeats.\n",
        encoding="utf-8",
    )
    return StageResult(stage=stage, status=StageStatus.DONE,
                       artifacts=("runs/", "training_limitations.md"), evidence_refs=(str(job_dir),))
