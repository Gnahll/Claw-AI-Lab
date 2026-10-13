"""Detached Linux workers with immutable inputs and explicit GPU allocation."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from researchclaw.data.coverage import file_sha256, validate_splits


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def job_lock(path: Path):
    if sys.platform != "linux":
        raise RuntimeError("Managed training workers currently require Linux")
    import fcntl

    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def gpu_rows(query: str, *, applications: bool = False) -> list[list[str]]:
    flag = "--query-compute-apps" if applications else "--query-gpu"
    result = subprocess.run(
        ["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    return [[value.strip() for value in row] for row in csv.reader(result.stdout.splitlines()) if row]


def require_idle_gpu(gpu_uuid: str) -> None:
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("An explicitly allocated physical GPU UUID is required")
    inventory = gpu_rows("uuid,memory.used,utilization.gpu")
    selected = [row for row in inventory if row[0] == gpu_uuid]
    if len(selected) != 1:
        raise ValueError("Allocated GPU UUID is absent or ambiguous")
    processes = gpu_rows("gpu_uuid,pid,used_gpu_memory", applications=True)
    if any(row[0] == gpu_uuid for row in processes):
        raise RuntimeError("Allocated GPU is occupied; no training was started")
    if float(selected[0][1]) > 256 or float(selected[0][2]) > 0:
        raise RuntimeError("Allocated GPU is not idle; no training was started")


def gpu_evidence(gpu_uuid: str) -> dict[str, Any]:
    rows = gpu_rows("gpu_uuid,pid,used_gpu_memory", applications=True)
    matches = [row for row in rows if row[0] == gpu_uuid and row[1] == str(os.getpid())]
    if len(matches) != 1 or float(matches[0][2]) <= 0:
        raise RuntimeError("Cannot attribute GPU memory to this training process")
    return {"gpu_uuid": gpu_uuid, "pid": os.getpid(), "memory_mib": float(matches[0][2]), "time": time.time()}


def process_token(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return f"{boot_id}:{fields[19]}"
    except (OSError, IndexError):
        return None


def validate_spec(spec: dict[str, Any]) -> None:
    required = {"source_dir", "split_manifest", "dataset_root", "initial_checkpoint", "initial_checkpoint_sha256"}
    options = {"epochs", "steps_per_epoch", "learning_rate", "weight_decay", "seed", "checkpoint_every_steps"}
    if set(spec) - required - options or not required <= set(spec):
        raise ValueError("Unknown or missing training specification fields")
    for key, default in (("epochs", 1), ("steps_per_epoch", 3), ("checkpoint_every_steps", 1), ("seed", 42)):
        value = spec.get(key, default)
        if type(value) is not int or value < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}")
    for key, default in (("learning_rate", 1e-5), ("weight_decay", .05)):
        value = spec.get(key, default)
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"Invalid {key}")
        if value < 0 or (key == "learning_rate" and value == 0):
            raise ValueError(f"Invalid {key}")
    digest = spec["initial_checkpoint_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Initial checkpoint requires an explicit SHA-256 digest")


def source_hashes(root: Path) -> dict[str, str]:
    paths = [root / "cotransformer.py"]
    for directory in ("models", "common"):
        paths.extend(sorted((root / directory).rglob("*.py")))
    if not (root / "models").is_dir() or not (root / "common").is_dir():
        raise ValueError("Expected the existing Co-Transformers source tree")
    result = {}
    for path in paths:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Reference source symlink escapes source root")
        result[path.relative_to(root).as_posix()] = file_sha256(path)
    return result


def prepare_job(spec_path: Path, job_dir: Path) -> None:
    spec = read_json(spec_path)
    validate_spec(spec)
    base = spec_path.resolve().parent
    for key in ("source_dir", "split_manifest", "dataset_root", "initial_checkpoint"):
        spec[key] = str((base / spec[key]).resolve(strict=True))
    source = Path(spec["source_dir"])
    manifest = read_json(Path(spec["split_manifest"]))
    counts = validate_splits(manifest, root=Path(spec["dataset_root"]))
    if spec.get("steps_per_epoch", 3) > counts["train"]:
        raise ValueError("steps_per_epoch exceeds available training samples")
    checkpoint = Path(spec["initial_checkpoint"])
    if file_sha256(checkpoint) != spec["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint hash mismatch")
    hashes = source_hashes(source)
    if job_dir.resolve().is_relative_to(source):
        raise ValueError("Job directory must be outside the reference repository")
    job_dir.mkdir(parents=True, exist_ok=False)
    workspace = job_dir / "source"
    workspace.mkdir()
    for relative, digest in hashes.items():
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
        if file_sha256(destination) != digest:
            raise RuntimeError("Reference source changed while copying")
    try:
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unknown; file hashes are authoritative"
    write_json(job_dir / "spec.json", spec)
    write_json(job_dir / "splits.json", manifest)
    write_json(job_dir / "provenance.json", {
        "source_revision": revision, "source_files": hashes,
        "spec_sha256": file_sha256(job_dir / "spec.json"),
        "splits_sha256": file_sha256(job_dir / "splits.json"),
        "model": "CoTransformers", "protocol": "coverage_finetuning_pilot",
        "worker_sha256": file_sha256(Path(__file__).with_name("worker.py")),
        "architecture_changes": [], "optional_ablations": False,
    })
    write_json(job_dir / "state.json", {"status": "prepared", "time": time.time()})


def verify_inputs(job_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = read_json(job_dir / "spec.json")
    provenance = read_json(job_dir / "provenance.json")
    validate_spec(spec)
    if file_sha256(Path(__file__).with_name("worker.py")) != provenance["worker_sha256"]:
        raise ValueError("Training adapter changed; create a new job")
    for filename in ("spec", "splits"):
        if file_sha256(job_dir / f"{filename}.json") != provenance[f"{filename}_sha256"]:
            raise ValueError(f"Immutable {filename} changed; create a new job")
    if source_hashes(job_dir / "source") != provenance["source_files"]:
        raise ValueError("Model source changed; create a new job")
    if file_sha256(Path(spec["initial_checkpoint"])) != spec["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint changed")
    manifest = read_json(job_dir / "splits.json")
    validate_splits(manifest, root=Path(spec["dataset_root"]))
    return spec, manifest


def job_status(job_dir: Path, *, stale_after: float = 600) -> dict[str, Any]:
    state = read_json(job_dir / "state.json")
    if state["status"] in {"starting", "running"}:
        token = process_token(state.get("pid", 0))
        if not token or token != state.get("process_token"):
            return {**state, "status": "interrupted", "alert": "Worker exited; inspect worker.log and resume from checkpoint"}
        progress_age = time.time() - state.get("progress_time", state["time"])
        if progress_age > stale_after:
            return {**state, "alert": "No training/evaluation progress; inspect logs", "progress_age_sec": progress_age}
    return state


def start_job(job_dir: Path, gpu_uuid: str, *, resume: bool = False) -> dict[str, Any]:
    job_dir = job_dir.resolve(strict=True)
    with job_lock(job_dir / "launch.lock"):
        state = job_status(job_dir)
        if state["status"] in {"starting", "running", "completed"}:
            raise ValueError(f"Cannot start job in state {state['status']}")
        if state["status"] != "prepared" and not resume:
            raise ValueError("Use explicit resume after an interrupted/failed job")
        if resume and not (job_dir / "resume.json").is_file():
            raise ValueError("No committed checkpoint exists; create a new job")
        if resume and state.get("gpu_uuid") != gpu_uuid:
            raise ValueError("Resume requires the original allocated GPU UUID")
        verify_inputs(job_dir)
        require_idle_gpu(gpu_uuid)
        launch_id = uuid.uuid4().hex
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        environment["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
        package_root = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = package_root
        write_json(job_dir / "launch.json", {"launch_id": launch_id, "gpu_uuid": gpu_uuid, "resume": resume})
        with (job_dir / "worker.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(
                [sys.executable, "-u", "-m", "researchclaw.training.worker", str(job_dir), launch_id],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                cwd=job_dir, env=environment, start_new_session=True, close_fds=True,
            )
        write_json(job_dir / "state.json", {
            "status": "starting", "pid": child.pid, "process_token": process_token(child.pid),
            "launch_id": launch_id, "time": time.time(), "gpu_uuid": gpu_uuid,
        })
        write_json(job_dir / "ready.json", {"launch_id": launch_id})
        return job_status(job_dir)
