"""Supervised execution and fail-closed evidence for S12, S14 and S15."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from researchclaw.training.observed import rank_evidence_path
from researchclaw.training.predictions import check_predictions


TRAINING_OBJECTIVE = (
    "Mandatory training objective: train the actual model on the approved data protocol, "
    "run independent evaluation, and retain losses, validation metrics and model checkpoints. "
    "Inference-only runs, smoke tests and written plans do not fulfill this task. "
    "Specify the full training budget and splits in the experiment plan; report resource blockers honestly."
)

OPTIONAL_EXPERIMENTS = {
    "ablation": "ablation study removing or replacing one proposed component at a time",
    "visualization": "visualization of qualitative predictions and intermediate features or attention maps",
    "analysis": "qualitative and quantitative analysis beyond the main comparison (failure cases, sensitivity)",
}


def optional_experiments_directive(selected) -> str:
    """Topic directive fixing which optional studies the user chose to run or skip."""
    chosen = [name for name in OPTIONAL_EXPERIMENTS if name in set(selected)]
    lines = ["Optional experiments were selected by the user; follow this choice exactly."]
    for name, description in OPTIONAL_EXPERIMENTS.items():
        if name in chosen:
            lines.append(f"- Required: {description}.")
        else:
            lines.append(f"- Not requested: do not plan or run any {name} experiment; "
                         "state its absence as a limitation.")
    return "\n".join(lines)


EXECUTION_CONTRACT = """
The pipeline runs main.py itself and does not accept prose as execution evidence.
main.py itself must write workspace-root results.json on every execution, containing
finite numeric measurements from the actual evaluation. A nested results file or
one manually written by an agent after the process exits does not satisfy this contract.
Never copy smoke/prior metrics into a full run. Keep full and SMOKE_TEST=1 paths separate.
For a training task: use real PyTorch optimizer updates in this process (do not start your own
subprocess or distributed launcher), loss.backward(), and save the FINAL trained model state_dict
to outputs/model.pt using torch.save. When several models are trained independently
(e.g. one per seed), also save each one's final state_dict under outputs/ (e.g.
outputs/final_<seed>.pt) before training the next; outputs/model.pt holds the delivered one.
Also save the final model's raw test-set predictions under outputs/ in a path containing
"pred" (e.g. outputs/test_predictions/<sample>.npy or .png masks, or outputs/predictions.json
with a "predictions" list or a CSV with a "pred" column). The runtime reloads them and rejects
missing, non-finite, or collapsed output (every prediction identical or constant).
The runtime checks observed parameter changes and reloads the checkpoints to verify that
every optimizer's final trained tensors were saved. Unsupported training backends must be reported
as blockers, not replaced with a toy implementation. Record per-epoch loss and validation
metrics, planned/completed epochs and steps, data split, sample counts, seed, evaluation
protocol and limitations in results.json; do not change test splits or reduce planned
training just to pass the gate. Parameter updates alone do not prove scientific validity.
Do not override CUDA_VISIBLE_DEVICES; an empty value means CPU only.
Multi-GPU: when CUDA_VISIBLE_DEVICES lists several GPUs, the runtime itself starts the full
training run with torchrun, one process per GPU (WORLD_SIZE, RANK, LOCAL_RANK are set). Then
use DistributedDataParallel: dist.init_process_group("nccl"), torch.cuda.set_device(LOCAL_RANK),
a DistributedSampler for training data, and sampler.set_epoch(epoch). The global batch is the
per-GPU batch times WORLD_SIZE; report both. Only rank 0 evaluates or gathers results, saves
checkpoints (model.module.state_dict()), predictions and results.json; other ranks wait at
dist.barrier() and must not write them. Every GPU must take part in training. When WORLD_SIZE
is unset (smoke test, single GPU, CPU), run as one ordinary process with no process group.
Long CPU-only preparation before the first optimizer step (e.g. a full data audit) must keep
writing measurable progress under outputs/ (a resumable cache or a progress file); until training
starts, that growth counts as progress and idle GPUs are not counted against it.
"""


def _has_measurement(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("Non-finite numeric result")
        return True
    if isinstance(value, dict):
        return any([_has_measurement(item) for item in value.values()])
    if isinstance(value, list):
        return any([_has_measurement(item) for item in value])
    return False


def run_experiment(workspace: Path, output_dir: Path, config, *, smoke=False, timeout=None):
    workspace = workspace.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = (output_dir / "training_observation.json").resolve()
    result_path = workspace / "results.json"
    training = bool(config.experiment.require_training) and not smoke
    if not (workspace / "outputs").resolve().is_relative_to(workspace):
        raise ValueError("outputs directory must remain inside the experiment workspace")
    gpus = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    world_size = len(gpus) if training and len(gpus) > 1 else 1
    stale_paths = [result_path, evidence_path, evidence_path.with_suffix(".jsonl")]
    for rank in range(1, len(gpus)):
        rank_path = rank_evidence_path(evidence_path, rank)
        stale_paths += [rank_path, rank_path.with_suffix(".jsonl")]
    if training:
        stale_paths.append(workspace / "outputs" / "model.pt")
    for stale in stale_paths:
        if stale.is_symlink() or stale.is_file():
            stale.unlink()
        elif stale.exists():
            raise ValueError(f"Expected an output file, not a directory: {stale}")
    python = config.experiment.sandbox.python_path or sys.executable
    if Path(python).is_file():
        python = str(Path(python).absolute())
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    if smoke:
        env["SMOKE_TEST"] = "1"
    else:
        env.pop("SMOKE_TEST", None)
    cmd = [python, "-m", "researchclaw.training.observed", str(evidence_path)] if training else [python, "main.py"]
    if world_size > 1:
        cmd = [python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world_size}",
               "-m", "researchclaw.training.observed", str(evidence_path)]
    budget = timeout if timeout is not None else config.experiment.time_budget_sec
    report = {"success": False, "smoke": smoke, "require_training": training, "command": cmd,
              "gpu_id": env.get("CUDA_VISIBLE_DEVICES", ""), "world_size": world_size, "results": {}, "errors": []}
    start = time.monotonic()
    started_at = time.time() - 2  # coarse filesystem mtime resolution
    process = None
    with (output_dir / "execution.log").open("w", encoding="utf-8") as log:
        try:
            report["source_sha256"] = hashlib.sha256((workspace / "main.py").read_bytes()).hexdigest()
            if training:
                from researchclaw.training.verified_run import source_manifest
                report["source_manifest"] = source_manifest(workspace)
            process = subprocess.Popen(cmd, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=os.name != "nt")
            report["pid"] = process.pid
            report["returncode"] = _supervise(process, max(1, budget), config.experiment, output_dir,
                                              evidence_path.with_suffix(".jsonl"), report,
                                              workspace / "outputs")
            # A deliberate, measured pretraining refusal commonly exits 1. Read
            # its fresh root report before replacing it with a generic exit code.
            if result_path.is_file() and not result_path.is_symlink():
                from researchclaw.pipeline.acceptance import pretraining_blocker
                try:
                    _blocked_results = json.loads(result_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
                except (OSError, ValueError):
                    _blocked_results = None
                blocker = pretraining_blocker(_blocked_results)
                if blocker:
                    report.update(results=_blocked_results, **blocker)
                    report["results_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
                    raise ValueError(f"Pretraining acceptance blocked: {blocker['reason']}; "
                                     f"checks={json.dumps(blocker['required_gate_status'], ensure_ascii=False)}. "
                                     "Revise the experiment design; mandatory gates were not changed.")
            if process.returncode:
                raise ValueError(f"Experiment process exited with {process.returncode}; see execution.log")
            if not result_path.is_file() or result_path.is_symlink():
                raise ValueError("Experiment did not write a fresh results.json")
            results = json.loads(result_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
            if not isinstance(results, dict) or not _has_measurement(results):
                raise ValueError("results.json has no finite numeric measurements")
            report["results"] = results
            report["results_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
            if _reports_failure_status(results.get("status")):
                blocker = results.get("blocker") or results.get("blockers") or "see results.json"
                raise ValueError(f"Experiment reported {results['status']}: {blocker}")
            if not _has_named_metric(results, config.experiment.metric_key):
                raise ValueError(f"results.json is missing measured metric: {config.experiment.metric_key}")
            if training:
                observation = json.loads(evidence_path.read_text(encoding="utf-8"))
                report["training"] = observation
                if not observation.get("parameter_update_observed") or observation.get("optimizer_steps", 0) < 1:
                    raise ValueError("No actual PyTorch parameter update observed")
                if not observation.get("losses"):
                    raise ValueError("Missing training loss: no scalar loss was passed to backward()")
                if not observation.get("checkpoint_verified"):
                    problem = (observation.get("checkpoint_problem") or observation.get("error")
                               or "outputs/model.pt was not saved or is empty")
                    raise ValueError(f"Missing checkpoint matching trained parameters: {problem}")
                if report["gpu_id"] and not observation.get("gpu_process"):
                    raise ValueError("GPU requested but no GPU training process observed")
                if world_size > 1:
                    report["training_ranks"] = _check_ranks(evidence_path, gpus)
                predictions = report["predictions"] = check_predictions(workspace / "outputs", started_at)
                if not predictions["groups"]:
                    raise ValueError("No saved test-set predictions found under outputs/ (path containing 'pred')")
                if predictions["unreadable"]:
                    raise ValueError(f"Unreadable or non-finite predictions: {predictions['unreadable'][0]['path']}")
                if predictions["degenerate"]:
                    raise ValueError("Degenerate test-set predictions (all identical or constant): "
                                     + ", ".join(predictions["degenerate"][:3]))
            report["success"] = True
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            report["errors"].append(str(exc))
        finally:
            if process is not None:
                if os.name != "nt":
                    # torchrun starts every rank in its own session; find them before the
                    # launcher dies, or they would be orphaned on their GPUs.
                    descendants = sorted(_descendants(process.pid))
                    for kill, target in [(os.killpg, process.pid), *[(os.killpg, pid) for pid in descendants],
                                         *[(os.kill, pid) for pid in descendants]]:
                        try:
                            kill(target, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
                elif process.poll() is None:
                    process.kill()
                process.wait()
            report["elapsed_sec"] = round(time.monotonic() - start, 2)
            (output_dir / "execution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _check_ranks(evidence_path, gpus):
    """Every torchrun rank must have trained on its own allocated GPU."""
    ranks = []
    used = [(json.loads(evidence_path.read_text(encoding="utf-8")).get("gpu_process") or {}).get("gpu_uuid")]
    for rank in range(1, len(gpus)):
        path = rank_evidence_path(evidence_path, rank)
        if not path.is_file():
            raise ValueError(f"Missing training evidence of rank {rank}; every GPU must train")
        observation = json.loads(path.read_text(encoding="utf-8"))
        if observation.get("error"):
            raise ValueError(f"Rank {rank} failed: {observation['error']}")
        if not observation.get("cuda_update_observed") or observation.get("optimizer_steps", 0) < 1:
            raise ValueError(f"Rank {rank} made no GPU parameter update; every GPU must train")
        gpu = observation.get("gpu_process") or {}
        if gpu.get("memory_mib", 0) <= 0:
            raise ValueError(f"Rank {rank} has no process-attributed GPU evidence")
        used.append(gpu.get("gpu_uuid"))
        ranks.append({key: observation.get(key) for key in ("rank", "optimizer_steps", "last_loss", "gpu_process")})
    if sorted(used) != sorted(gpus):
        raise ValueError("Training ranks did not cover every allocated GPU exactly once")
    return ranks


def _descendants(pid):
    """All live descendants of pid (Linux /proc), including those in their own sessions."""
    children = {}
    for entry in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = entry.read_text()
            children.setdefault(int(stat[stat.rfind(")") + 2:].split()[1]), []).append(int(entry.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    found, pending = set(), [pid]
    while pending:
        for child in children.get(pending.pop(), ()):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def _in_group(pid, group):
    """True for the experiment process or any process it started (torchrun ranks have their own sessions)."""
    try:
        return pid == group or os.getpgid(pid) == group or pid in _descendants(group)
    except (OSError, AttributeError):
        return pid in _descendants(group)


def _gpu_sample(gpu_uuid, pid):
    from researchclaw.training.jobs import gpu_rows

    try:
        inventory = [row for row in gpu_rows("uuid,utilization.gpu,memory.used") if row[0] == gpu_uuid]
        processes = gpu_rows("gpu_uuid,pid,used_gpu_memory", applications=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)[:200]}
    if len(inventory) != 1:
        return {"error": "Allocated GPU UUID is absent or ambiguous"}
    own = [row for row in processes if row[0] == gpu_uuid and row[1].isdigit() and _in_group(int(row[1]), pid)]
    return {"utilization": float(inventory[0][1]), "memory_mib": float(inventory[0][2]),
            "process_memory_mib": float(own[0][2]) if own else 0.0, "process_on_gpu": bool(own)}


def _outputs_marker(outputs: Path | None) -> tuple[int, int, int]:
    """File count, bytes and newest mtime under outputs/, without following links."""
    count = size = newest = 0
    if outputs is None or not outputs.is_dir():
        return count, size, newest
    for root, dirs, names in os.walk(outputs):
        for name in names:
            try:
                st = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            count, size, newest = count + 1, size + st.st_size, max(newest, st.st_mtime_ns)
    return count, size, newest


def _supervise(process, budget, experiment, output_dir, steps_path, report, outputs=None):
    """Wait for the experiment while checking that it is alive, progressing and using its GPU.

    Progress means new execution.log output or new optimizer steps. A hung run (deadlock,
    stuck data loader) or a run that silently never touches one of its allocated GPUs is
    killed instead of occupying the budget. Before the first optimizer step, growth under
    outputs/ is progress too: a resumable CPU preparation (a full data audit) that keeps
    writing is not a hang, so its idle GPUs are not counted until training starts.
    """
    stall_timeout = int(getattr(experiment, "stall_timeout_sec", 1800) or 0)
    gpu_timeout = int(getattr(experiment, "gpu_idle_timeout_sec", 1800) or 0)
    interval = max(1, int(getattr(experiment, "monitor_interval_sec", 30) or 30))
    gpu_uuids = [item for item in report["gpu_id"].split(",") if item.startswith("GPU-")]
    log_path = output_dir / "execution.log"
    start = last_progress = time.monotonic()
    gpu_last_used = dict.fromkeys(gpu_uuids, start)
    last_marker = last_outputs = None
    monitor = report["monitor"] = {"samples": 0, "gpu_seen": False, "max_steps": 0}
    with (output_dir / "training_monitor.jsonl").open("w", encoding="utf-8") as samples:
        while True:
            remaining = budget - (time.monotonic() - start)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(report["command"], budget)
            try:
                return process.wait(timeout=min(interval, remaining))
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            steps = 0
            if steps_path.is_file():
                with steps_path.open("rb") as handle:
                    steps = sum(1 for _ in handle)
            marker = (log_path.stat().st_size if log_path.is_file() else 0, steps)
            if marker != last_marker:
                last_marker, last_progress = marker, now
            preparing = False
            if steps == 0:
                written = _outputs_marker(outputs)
                preparing = last_outputs is not None and written != last_outputs
                last_outputs = written
                if preparing:
                    last_progress = now
            sample = {"elapsed_sec": round(now - start, 1), "alive": process.poll() is None,
                      "log_bytes": marker[0], "optimizer_steps": steps,
                      "since_progress_sec": round(now - last_progress, 1)}
            if preparing:
                sample["preparation_progress"] = True
            for gpu_uuid in gpu_uuids:
                gpu = _gpu_sample(gpu_uuid, process.pid)
                if len(gpu_uuids) == 1:
                    sample["gpu"] = gpu
                else:
                    sample.setdefault("gpus", {})[gpu_uuid] = gpu
                if "error" in gpu or gpu["process_on_gpu"] or gpu["utilization"] > 0 or preparing:
                    gpu_last_used[gpu_uuid] = now  # Do not kill on a monitoring failure.
                monitor["gpu_seen"] = monitor["gpu_seen"] or bool(gpu.get("process_on_gpu"))
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            monitor.update(samples=monitor["samples"] + 1, max_steps=steps, last=sample)
            if stall_timeout and now - last_progress > stall_timeout:
                raise ValueError(f"Training stalled: no new log output or optimizer step for "
                                 f"{int(now - last_progress)}s; see training_monitor.jsonl")
            for gpu_uuid, last_used in gpu_last_used.items():
                if gpu_timeout and now - last_used > gpu_timeout:
                    raise ValueError(f"GPU {gpu_uuid} requested but idle and unused by the experiment for "
                                     f"{int(now - last_used)}s; see training_monitor.jsonl")


def _reject_nonfinite(value):
    raise ValueError(f"Non-finite result: {value}")


_FAILURE_STATUSES = ("blocked", "failed", "incomplete", "skipped")


def _reports_failure_status(status: object) -> bool:
    """Treat qualified statuses such as ``blocked_architecture_prerequisite`` as failures."""
    if not isinstance(status, str):
        return False
    normalized = status.strip().lower()
    return any(normalized == word or normalized.startswith((f"{word}_", f"{word}-", f"{word}:", f"{word} "))
               for word in _FAILURE_STATUSES)


def _has_named_metric(value, metric):
    if isinstance(value, dict):
        found = value.get(metric)
        if isinstance(found, (int, float)) and not isinstance(found, bool) and math.isfinite(found):
            return True
        return any(_has_named_metric(item, metric) for item in value.values())
    if isinstance(value, list):
        return any(_has_named_metric(item, metric) for item in value)
    return False
