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
For a training task: use real PyTorch optimizer updates in this process (not a subprocess
or distributed launcher), loss.backward(), and save the FINAL trained model state_dict
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
    stale_paths = [result_path, evidence_path, evidence_path.with_suffix(".jsonl")]
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
    budget = timeout if timeout is not None else config.experiment.time_budget_sec
    report = {"success": False, "smoke": smoke, "require_training": training, "command": cmd,
              "gpu_id": env.get("CUDA_VISIBLE_DEVICES", ""), "results": {}, "errors": []}
    start = time.monotonic()
    started_at = time.time() - 2  # coarse filesystem mtime resolution
    process = None
    with (output_dir / "execution.log").open("w", encoding="utf-8") as log:
        try:
            report["source_sha256"] = hashlib.sha256((workspace / "main.py").read_bytes()).hexdigest()
            process = subprocess.Popen(cmd, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=os.name != "nt")
            report["pid"] = process.pid
            report["returncode"] = _supervise(process, max(1, budget), config.experiment, output_dir,
                                              evidence_path.with_suffix(".jsonl"), report)
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
                if not observation.get("losses") or not observation.get("checkpoint_verified"):
                    raise ValueError("Missing training loss or checkpoint matching trained parameters")
                if report["gpu_id"] and not observation.get("gpu_process"):
                    raise ValueError("GPU requested but no GPU training process observed")
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
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif process.poll() is None:
                    process.kill()
                process.wait()
            report["elapsed_sec"] = round(time.monotonic() - start, 2)
            (output_dir / "execution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _gpu_sample(gpu_uuid, pid):
    from researchclaw.training.jobs import gpu_rows

    try:
        inventory = [row for row in gpu_rows("uuid,utilization.gpu,memory.used") if row[0] == gpu_uuid]
        processes = gpu_rows("gpu_uuid,pid,used_gpu_memory", applications=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)[:200]}
    if len(inventory) != 1:
        return {"error": "Allocated GPU UUID is absent or ambiguous"}
    own = [row for row in processes if row[0] == gpu_uuid and row[1] == str(pid)]
    return {"utilization": float(inventory[0][1]), "memory_mib": float(inventory[0][2]),
            "process_memory_mib": float(own[0][2]) if own else 0.0, "process_on_gpu": bool(own)}


def _supervise(process, budget, experiment, output_dir, steps_path, report):
    """Wait for the experiment while checking that it is alive, progressing and using its GPU.

    Progress means new execution.log output or new optimizer steps. A hung run (deadlock,
    stuck data loader) or a run that silently never touches its allocated GPU is killed
    instead of occupying the budget.
    """
    stall_timeout = int(getattr(experiment, "stall_timeout_sec", 1800) or 0)
    gpu_timeout = int(getattr(experiment, "gpu_idle_timeout_sec", 1800) or 0)
    interval = max(1, int(getattr(experiment, "monitor_interval_sec", 30) or 30))
    gpu_uuid = report["gpu_id"] if report["gpu_id"].startswith("GPU-") else ""
    log_path = output_dir / "execution.log"
    start = last_progress = gpu_last_used = time.monotonic()
    last_marker = None
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
            sample = {"elapsed_sec": round(now - start, 1), "alive": process.poll() is None,
                      "log_bytes": marker[0], "optimizer_steps": steps,
                      "since_progress_sec": round(now - last_progress, 1)}
            if gpu_uuid:
                sample["gpu"] = gpu = _gpu_sample(gpu_uuid, process.pid)
                if "error" in gpu or gpu["process_on_gpu"] or gpu["utilization"] > 0:
                    gpu_last_used = now  # Do not kill on a monitoring failure.
                monitor["gpu_seen"] = monitor["gpu_seen"] or bool(gpu.get("process_on_gpu"))
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            monitor.update(samples=monitor["samples"] + 1, max_steps=steps, last=sample)
            if stall_timeout and now - last_progress > stall_timeout:
                raise ValueError(f"Training stalled: no new log output or optimizer step for "
                                 f"{int(now - last_progress)}s; see training_monitor.jsonl")
            if gpu_timeout and now - gpu_last_used > gpu_timeout:
                raise ValueError(f"GPU {gpu_uuid} requested but idle and unused by the experiment for "
                                 f"{int(now - gpu_last_used)}s; see training_monitor.jsonl")


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
