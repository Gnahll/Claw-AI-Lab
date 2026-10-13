"""Dedicated Co-Transformers pilot worker; never launched through an agent shell."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from researchclaw.data.coverage import file_sha256
from researchclaw.training.jobs import (
    gpu_evidence, job_lock, process_token, read_json, require_idle_gpu,
    verify_inputs, write_json,
)


def parameter_digest(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def score_masks(prediction, target) -> dict[str, float]:
    import numpy as np

    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("Prediction/target geometry mismatch")
    predicted = prediction.astype(bool)
    expected = target.astype(bool)
    intersection = int(np.logical_and(predicted, expected).sum())
    predicted_count = int(predicted.sum())
    target_count = int(expected.sum())
    union = predicted_count + target_count - intersection
    return {
        "pixel_f1": 2 * intersection / (predicted_count + target_count) if predicted_count + target_count else 1.0,
        "pixel_iou": intersection / union if union else 1.0,
    }


class CoverageSamples:
    def __init__(self, root: Path, records: list[dict[str, Any]]):
        from IMDLBenCo.transforms import EdgeMaskGenerator, get_albu_transforms

        self.root = root
        self.records = records
        self.edge_generator = EdgeMaskGenerator(7)
        self.transform = get_albu_transforms(type_="resize", output_size=(512, 512))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image

        record = self.records[index]
        with Image.open(self.root / record["image"]) as image:
            pixels = np.array(image.convert("RGB"))
        with Image.open(self.root / record["mask"]) as image:
            mask = (np.array(image.convert("L")) > 0).astype(np.float32)
        edge = self.edge_generator(mask)[0][0]
        transformed = self.transform(image=pixels, masks=[mask, edge])
        return {
            "id": record["id"],
            "image": transformed["image"].float(),
            "mask": transformed["masks"][0].unsqueeze(0).float(),
            "edge_mask": transformed["masks"][1].unsqueeze(0).float(),
        }


def emit(job_dir: Path, event: str, **details) -> None:
    now = time.time()
    state = read_json(job_dir / "state.json")
    state.update(status="running", time=now, progress_time=now, phase=event, **details)
    write_json(job_dir / "state.json", state)
    with (job_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": now, "event": event, **details}, allow_nan=False) + "\n")
        stream.flush()


def evaluate(model, samples, device, *, output_dir: Path | None = None, progress=None):
    import numpy as np
    import torch
    from PIL import Image

    model.eval()
    records = []
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index in range(len(samples)):
            sample = samples[index]
            inputs = {key: sample[key].unsqueeze(0).to(device) for key in ("image", "mask", "edge_mask")}
            probability = model(**inputs)["pred_mask"].float()
            if not torch.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
                raise ValueError("Invalid prediction probabilities")
            prediction = (probability[0, 0].cpu().numpy() >= .5)
            target = sample["mask"][0].numpy() > .5
            record = {"id": sample["id"], **score_masks(prediction, target)}
            if output_dir:
                for kind, pixels in (("prediction", prediction), ("target", target)):
                    filename = f"{sample['id']}-{kind}.png"
                    Image.fromarray(pixels.astype(np.uint8) * 255).save(output_dir / filename)
                    record[kind] = filename
                    record[f"{kind}_sha256"] = file_sha256(output_dir / filename)
            records.append(record)
            if progress:
                progress(index + 1)
    if not records:
        raise ValueError("Evaluation split is empty")
    return {
        "metrics": {key: sum(record[key] for record in records) / len(records) for key in ("pixel_f1", "pixel_iou")},
        "records": records, "threshold": .5, "aggregation": "macro_per_image",
        "resolution": [512, 512], "drop_last": False,
    }


def load_committed_checkpoint(job_dir: Path):
    import torch

    reference = read_json(job_dir / "resume.json")
    filename = reference["filename"]
    if Path(filename).name != filename:
        raise ValueError("Invalid checkpoint filename")
    path = job_dir / filename
    if file_sha256(path) != reference["sha256"]:
        raise ValueError("Resume checkpoint hash mismatch")
    return torch.load(path, map_location="cpu", weights_only=True)


def save_checkpoint(job_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    name = f"checkpoint-{payload['global_step']}-{payload['epoch']}-{payload['offset']}.pt"
    path = job_dir / name
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    reference = {"filename": name, "sha256": file_sha256(path), "global_step": payload["global_step"]}
    write_json(job_dir / "resume.json", reference)
    keep = {name}
    if payload.get("best"):
        keep.add(payload["best"]["filename"])
    for previous in job_dir.glob("checkpoint-*.pt"):
        if re.fullmatch(r"checkpoint-\d+-\d+-\d+\.pt", previous.name) and previous.name not in keep:
            previous.unlink()
    return reference


def fit(model, training, validation, spec, job_dir, device, *, resume=False, gpu_uuid=None):
    import torch

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=spec.get("learning_rate", 1e-5), weight_decay=spec.get("weight_decay", .05),
    )
    use_amp = str(device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    seed = spec.get("seed", 42)
    epochs = spec.get("epochs", 1)
    steps_per_epoch = spec.get("steps_per_epoch", 3)
    checkpoint_interval = spec.get("checkpoint_every_steps", 1)
    global_step, start_epoch, offset = 0, 0, 0
    initial_digest = parameter_digest(model)
    best = None
    best_score = -1.0
    evidence = None
    if resume:
        saved = load_committed_checkpoint(job_dir)
        if saved["spec_sha256"] != file_sha256(job_dir / "spec.json"):
            raise ValueError("Resume protocol mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        global_step, start_epoch, offset = saved["global_step"], saved["epoch"], saved["offset"]
        initial_digest, best, best_score = saved["initial_digest"], saved["best"], saved["best_score"]
        evidence = saved["gpu_evidence"]
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if use_amp:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved

    def checkpoint(epoch_cursor, next_offset):
        return save_checkpoint(job_dir, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "global_step": global_step, "epoch": epoch_cursor, "offset": next_offset,
            "initial_digest": initial_digest, "best": best, "best_score": best_score,
            "gpu_evidence": evidence, "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all() if use_amp else [],
            "spec_sha256": file_sha256(job_dir / "spec.json"),
        })

    for epoch in range(start_epoch, epochs):
        order = list(range(len(training)))
        random.Random(seed + epoch).shuffle(order)
        model.train()
        for position in range(offset, steps_per_epoch):
            sample = training[order[position]]
            inputs = {key: sample[key].unsqueeze(0).to(device) for key in ("image", "mask", "edge_mask")}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                loss = model(**inputs)["backward_loss"]
            if loss.numel() != 1 or not torch.isfinite(loss):
                raise ValueError("Training loss is not a finite scalar")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise ValueError("Missing/non-finite gradients; no optimizer step was accepted")
            if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
                raise ValueError("All gradients are zero")
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if use_amp:
                torch.cuda.synchronize()
            if gpu_uuid:
                evidence = gpu_evidence(gpu_uuid)
            emit(job_dir, "optimizer_step", epoch=epoch + 1, global_step=global_step,
                 sample_id=sample["id"], loss=float(loss.detach().cpu()))
            if global_step % checkpoint_interval == 0:
                checkpoint(epoch, position + 1)
        validation_result = evaluate(
            model, validation, device,
            progress=lambda count: emit(job_dir, "validation", epoch=epoch + 1, evaluated=count),
        )
        score = validation_result["metrics"]["pixel_f1"]
        write_json(job_dir / f"validation-{epoch + 1}.json", validation_result)
        candidate = checkpoint(epoch, steps_per_epoch)
        if score > best_score:
            best, best_score = candidate, score
        checkpoint(epoch + 1, 0)
        offset = 0
    if not best or global_step < 1:
        raise ValueError("No trained validation-selected checkpoint")
    if file_sha256(job_dir / best["filename"]) != best["sha256"]:
        raise ValueError("Selected checkpoint hash mismatch")
    selected = torch.load(job_dir / best["filename"], map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model"], strict=True)
    del selected
    trained_digest = parameter_digest(model)
    if initial_digest == trained_digest:
        raise ValueError("Selected model has no trainable parameter changes")
    return {
        "optimizer_steps": global_step, "selected_checkpoint": best,
        "initial_parameter_sha256": initial_digest, "trained_parameter_sha256": trained_digest,
        "gpu_evidence": evidence,
    }


def run_reference(job_dir: Path, launch: dict[str, Any]) -> None:
    spec, manifest = verify_inputs(job_dir)
    gpu_uuid = launch["gpu_uuid"]
    require_idle_gpu(gpu_uuid)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated CUDA GPU must be visible")
    torch.manual_seed(spec.get("seed", 42))
    torch.cuda.manual_seed_all(spec.get("seed", 42))
    random.seed(spec.get("seed", 42))
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(job_dir / "source"))
    from cotransformer import CoTransformers

    emit(job_dir, "loading_reference")
    model = CoTransformers(segformer_pretrain_path=None, noiseprint_path=None)
    weights = torch.load(spec["initial_checkpoint"], map_location="cpu", weights_only=True)
    if "model" in weights:
        weights = weights["model"]
    if not isinstance(weights, dict):
        raise ValueError("Expected a tensor state dictionary")
    model.load_state_dict(weights, strict=True)
    del weights
    model.to("cuda:0")
    write_json(job_dir / "environment.json", {
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_uuid": gpu_uuid, "gpu_name": torch.cuda.get_device_name(0),
        "preprocessing": "IMDLBenCo resize 512x512, ImageNet normalization, EdgeMaskGenerator(7), grayscale mask > 0",
        "optimizer": "AdamW; constant learning rate; batch 1; AMP; no augmentation; pilot protocol",
        "loss": "Unmodified reference forward: 0.8 edge Dice + 0.16 pixel Dice",
    })
    samples = {name: CoverageSamples(Path(spec["dataset_root"]), records) for name, records in manifest["splits"].items()}
    training_evidence = fit(model, samples["train"], samples["validation"], spec, job_dir, "cuda:0",
                            resume=launch["resume"], gpu_uuid=gpu_uuid)
    test = evaluate(model, samples["test"], "cuda:0", output_dir=job_dir / "test_predictions",
                    progress=lambda count: emit(job_dir, "test", evaluated=count))
    test["checkpoint_sha256"] = training_evidence["selected_checkpoint"]["sha256"]
    test["training_evidence"] = training_evidence
    test["protocol"] = "coverage_finetuning_pilot_not_paper_reproduction"
    test["test_is_blind"] = False
    write_json(job_dir / "test_results.json", test)
    verify_inputs(job_dir)
    state = read_json(job_dir / "state.json")
    write_json(job_dir / "state.json", {**state, "status": "completed", "time": time.time(),
                                        "test_results_sha256": file_sha256(job_dir / "test_results.json")})


def main() -> int:
    job_dir = Path(sys.argv[1]).resolve(strict=True)
    launch_id = sys.argv[2]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if (job_dir / "ready.json").exists() and read_json(job_dir / "ready.json").get("launch_id") == launch_id:
            break
        time.sleep(.1)
    else:
        raise RuntimeError("Launcher did not commit the worker start")
    with job_lock(job_dir / "worker.lock"):
        launch = read_json(job_dir / "launch.json")
        state = read_json(job_dir / "state.json")
        if launch["launch_id"] != launch_id or state.get("process_token") != process_token(os.getpid()):
            raise RuntimeError("Worker launch identity mismatch")
        try:
            gpu_lock = Path.home() / ".cache" / "researchclaw" / "gpu-locks"
            gpu_lock.mkdir(parents=True, exist_ok=True)
            with job_lock(gpu_lock / f"{launch['gpu_uuid']}.lock"):
                run_reference(job_dir, launch)
            return 0
        except BaseException as exc:
            traceback.print_exc()
            state = read_json(job_dir / "state.json")
            write_json(job_dir / "state.json", {**state, "status": "failed", "time": time.time(), "error": str(exc)})
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
