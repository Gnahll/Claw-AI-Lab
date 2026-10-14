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


TRAINING_OBJECTIVE = (
    "Mandatory training objective: train the actual model on the approved data protocol, "
    "run independent evaluation, and retain losses, validation metrics and model checkpoints. "
    "Inference-only runs, smoke tests and written plans do not fulfill this task. "
    "Specify the full training budget and splits in the experiment plan; report resource blockers honestly."
)

EXECUTION_CONTRACT = """
The pipeline runs main.py itself and does not accept prose as execution evidence.
main.py itself must write workspace-root results.json on every execution, containing
finite numeric measurements from the actual evaluation. A nested results file or
one manually written by an agent after the process exits does not satisfy this contract.
Never copy smoke/prior metrics into a full run. Keep full and SMOKE_TEST=1 paths separate.
For a training task: use real PyTorch optimizer updates in this process (not a subprocess
or distributed launcher), loss.backward(), and save the FINAL trained model state_dict
to outputs/model.pt using torch.save. For multiple optimizers include all their trained
parameters in that checkpoint. The runtime checks observed parameter changes and reloads
the checkpoint to compare trained tensors. Unsupported training backends must be reported
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
    process = None
    with (output_dir / "execution.log").open("w", encoding="utf-8") as log:
        try:
            report["source_sha256"] = hashlib.sha256((workspace / "main.py").read_bytes()).hexdigest()
            process = subprocess.Popen(cmd, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=os.name != "nt")
            report["pid"] = process.pid
            report["returncode"] = process.wait(timeout=max(1, budget))
            if process.returncode:
                raise ValueError(f"Experiment process exited with {process.returncode}; see execution.log")
            if not result_path.is_file() or result_path.is_symlink():
                raise ValueError("Experiment did not write a fresh results.json")
            results = json.loads(result_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
            if not isinstance(results, dict) or not _has_measurement(results):
                raise ValueError("results.json has no finite numeric measurements")
            report["results"] = results
            report["results_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
            if results.get("status") in ("blocked", "failed", "incomplete", "skipped"):
                raise ValueError(f"Experiment reported {results['status']}: {results.get('blocker', 'see results.json')}")
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


def _reject_nonfinite(value):
    raise ValueError(f"Non-finite result: {value}")


def _has_named_metric(value, metric):
    if isinstance(value, dict):
        found = value.get(metric)
        if isinstance(found, (int, float)) and not isinstance(found, bool) and math.isfinite(found):
            return True
        return any(_has_named_metric(item, metric) for item in value.values())
    if isinstance(value, list):
        return any(_has_named_metric(item, metric) for item in value)
    return False
