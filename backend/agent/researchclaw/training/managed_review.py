"""Validation-only pilot and built-in AI review for detached Co-Transformers jobs."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

from researchclaw.data.coverage import file_sha256
from researchclaw.training.jobs import job_lock, process_token, read_json, require_idle_gpu, verify_inputs, write_json


def needs_review(spec):
    total = spec.get("epochs", 1) * spec.get("steps_per_epoch", 3)
    return spec.get("training_phase") == "long" or total > 1350


def pilot_budget(spec):
    per_epoch = spec.get("steps_per_epoch", 3)
    total = spec.get("epochs", 1) * per_epoch
    limit = min(spec.get("pilot_steps", 150), total - 1)
    if limit < 1:
        raise ValueError("Long training must leave at least one update after its pilot")
    return max(1, limit // per_epoch), min(per_epoch, limit)


def validation_pilot(job, folder, gpu):
    """Use the immutable model sources, train/validation splits and the native fit."""
    import torch
    from researchclaw.training.worker import CoverageSamples, evaluate, fit

    spec, manifest = verify_inputs(job)
    require_idle_gpu(gpu)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Pilot requires the allocated GPU")
    seed = spec.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sys.path.insert(0, str(job / "source"))
    from cotransformer import CoTransformers

    model = CoTransformers(segformer_pretrain_path=None, noiseprint_path=None)
    weights = torch.load(spec["initial_checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(weights.get("model", weights), strict=True)
    del weights
    model.to("cuda:0")
    root = Path(spec["dataset_root"])
    train = CoverageSamples(root, manifest["splits"]["train"])
    val = CoverageSamples(root, manifest["splits"]["validation"])
    before = evaluate(model, val, "cuda:0")
    epochs, steps = pilot_budget(spec)
    contract = dict(spec, epochs=epochs, steps_per_epoch=steps)
    write_json(folder / "spec.json", contract)
    write_json(folder / "state.json", {"status": "pilot"})
    proof = fit(model, train, val, contract, folder, "cuda:0", gpu_uuid=gpu)
    after = evaluate(model, val, "cuda:0")
    write_json(folder / "validation.json", {"before": before, "after": after})
    write_json(folder / "pilot.json", {"training": proof, "provenance_sha256": file_sha256(job / "provenance.json"),
                                       "validation_sha256": file_sha256(folder / "validation.json")})


def review_evidence(job, folder, spec, manifest):
    proof = read_json(folder / "pilot.json")
    if (proof["provenance_sha256"] != file_sha256(job / "provenance.json")
            or proof["validation_sha256"] != file_sha256(folder / "validation.json")):
        raise ValueError("Pilot inputs or validation evidence changed")
    training = proof["training"]
    epochs, per_epoch = pilot_budget(spec)
    count = epochs * per_epoch
    checkpoint = training["selected_checkpoint"]
    name = checkpoint["filename"]
    if (Path(name).name != name or not (folder / name).resolve().is_relative_to(folder.resolve())
            or file_sha256(folder / name) != checkpoint["sha256"]
            or training["initial_parameter_sha256"] == training["trained_parameter_sha256"]
            or training["optimizer_steps"] != count or not training.get("gpu_evidence")):
        raise ValueError("Pilot training evidence does not match the requested method")
    valid_train = {row["id"] for row in manifest["splits"]["train"]}
    events = [e for line in (folder / "events.jsonl").read_text().splitlines()
              if (e := json.loads(line)).get("event") == "optimizer_step"]
    if [e["global_step"] for e in events] != list(range(1, count + 1)) or any(
            e["sample_id"] not in valid_train or not math.isfinite(e["loss"]) for e in events):
        raise ValueError("Pilot optimizer ledger does not match the training split")
    data = read_json(folder / "validation.json")
    expected = [r["id"] for r in manifest["splits"]["validation"]]
    if not expected:
        raise ValueError("Empty pilot validation split")
    scores = {}
    for condition in ("before", "after"):
        records = data[condition]["records"]
        if [r["id"] for r in records] != expected:
            raise ValueError("Pilot validation coverage differs")
        scores[condition] = {}
        for key in ("pixel_f1", "pixel_iou"):
            points = [r[key] for r in records]
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in points):
                raise ValueError("Invalid pilot validation measurement")
            scores[condition][key] = sum(points) / len(points)
    gain = scores["after"]["pixel_f1"] - scores["before"]["pixel_f1"]
    minimum = spec.get("min_validation_f1_gain")
    if minimum is not None and gain < minimum:
        raise ValueError(f"Pilot pixel F1 change {gain:.6f} fell below the required {minimum}")
    return {"provenance_sha256": proof["provenance_sha256"], "pilot_sha256": file_sha256(folder / "pilot.json"),
            "optimizer_steps": count, "validation_count": len(expected), "scores": scores,
            "pixel_f1_gain": gain, "min_validation_f1_gain": minimum, "checkpoint_sha256": checkpoint["sha256"]}


def ask_reviewer(measured):
    from researchclaw.llm import resolve_provider_base_url
    from researchclaw.llm.client import LLMClient, LLMConfig

    env_file = Path(__file__).resolve().parents[4] / ".env"
    settings = {}
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.startswith("RESEARCHCLAW_"):
                settings[key] = value.strip().strip(chr(34) + chr(39))
    settings.update(os.environ)
    model = settings["RESEARCHCLAW_MODEL"]
    client = LLMClient(LLMConfig(primary_model=model, fallback_models=[],
        base_url=resolve_provider_base_url("openai-compatible", settings.get("RESEARCHCLAW_BASE_URL", "")),
        api_key=settings["RESEARCHCLAW_API_KEY"], timeout_sec=60))
    reply = client.chat([{"role": "user", "content": json.dumps(measured, allow_nan=False)}],
        system="Review this Co-Transformers validation-only pilot before long training. Consider actual training evidence, "
               "coverage, validation change and the specified threshold. Return JSON with decision approve or reject and a reason.",
        json_mode=True, max_tokens=400)
    answer = json.loads(reply.content)
    if (answer.get("decision") not in ("approve", "reject") or not isinstance(answer.get("reason"), str)
            or not answer["reason"].strip()):
        raise ValueError("Invalid response from the platform short-training reviewer")
    return {"decision": answer["decision"], "reason": answer["reason"][:2000], "model": model}


def review(job, gpu):
    spec, splits = verify_inputs(job)
    base = job / "long_review"
    base.mkdir(exist_ok=True)
    pointer = base / "pilot_run.json"
    if pointer.exists():
        name = read_json(pointer)["directory"]
        if Path(name).name != name or not (base / name).resolve().is_relative_to(base.resolve()):
            raise ValueError("Invalid pilot location")
        folder = base / name
    else:
        folder = Path(tempfile.mkdtemp(prefix="attempt-", dir=base))
        subprocess.run([sys.executable, "-m", __spec__.name, "--pilot", str(job), str(folder), gpu], check=True)
        write_json(pointer, {"directory": folder.name})
    measured = review_evidence(job, folder, spec, splits)
    digest = hashlib.sha256(json.dumps(measured, sort_keys=True, allow_nan=False).encode()).hexdigest()
    path = base / "decision.json"
    decision = read_json(path) if path.is_file() else {}
    if decision.get("evidence_sha256") != digest:
        decision = {**ask_reviewer(measured), "evidence_sha256": digest, "evidence": measured}
        write_json(path, decision)
    if decision.get("decision") != "approve":
        raise RuntimeError("Built-in AI declined long training; see long_review/decision.json")


def launch(job, launch_id):
    # The detached worker uses this same process identity after exec, retaining
    # the launcher's ordinary status, locks and resume behavior.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ready = job / "ready.json"
        if ready.is_file() and read_json(ready).get("launch_id") == launch_id:
            break
        time.sleep(.1)
    else:
        raise RuntimeError("Launcher did not commit the review start")
    launch_spec = read_json(job / "launch.json")
    state = read_json(job / "state.json")
    if launch_spec["launch_id"] != launch_id or state.get("process_token") != process_token(os.getpid()):
        raise RuntimeError("Unexpected long training launch identity")
    gpu = launch_spec["gpu_uuid"]
    try:
        with job_lock(job / "worker.lock"):
            locks = Path.home() / ".cache/researchclaw/gpu-locks"
            locks.mkdir(parents=True, exist_ok=True)
            with job_lock(locks / f"{gpu}.lock"):
                write_json(job / "state.json", dict(state, status="running", phase="ai_short_training", time=time.time()))
                review(job, gpu)
        os.execv(sys.executable, [sys.executable, "-u", "-m", "researchclaw.training.worker", str(job), launch_id])
    except Exception as error:
        write_json(job / "state.json", dict(state, status="failed", phase="ai_short_training", error=str(error), time=time.time()))
        raise


if __name__ == "__main__":
    if sys.argv[1] == "--pilot":
        validation_pilot(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
    else:
        launch(Path(sys.argv[1]), sys.argv[2])
