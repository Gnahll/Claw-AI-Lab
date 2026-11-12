"""Import a completed, explicitly registered external run for analysis only.

The original training workspace is never used as the pipeline output directory.
Completion is checked independently of the UI and no training is scheduled here.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def steps(value: dict) -> int:
    for row in (value, value.get("training", {})):
        for key in ("optimizer_steps_completed", "completed_optimizer_steps", "completed_steps", "unique_steps"):
            n = row.get(key)
            if isinstance(n, int) and not isinstance(n, bool):
                return n
    return 0


def _local(project: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.parent != project or not path.is_dir():
        raise ValueError("Expected a registered run directly under this project")
    return path


def _validate_source(project: Path, policy: dict) -> tuple[Path, dict]:
    source = _local(project, policy["source_run"])
    queue = read(source / "queue_status.json")
    target = policy["target_steps"]
    if queue.get("state") != "completed" or int((source / "exit_code.txt").read_text()) != 0:
        raise ValueError("Training has not exited successfully")
    if queue.get("target_steps") != target or queue.get("epochs") != policy["target_epochs"]:
        raise ValueError("Completed run differs from the authorized budget")
    pid = queue.get("supervisor_pid")
    if isinstance(pid, int):
        proc = Path("/proc", str(pid), "cmdline")
        if proc.exists():
            command = proc.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if str(source.resolve()) in command and "supervisor.py" in command:
                raise ValueError("Training supervisor is still running")
    result = read(source / "results.json")
    conditions = result.get("conditions", {})
    if set(conditions) != set(policy["conditions"]):
        raise ValueError("Completed condition set differs from the registered comparison")
    for name, row in conditions.items():
        if steps(row) != target:
            raise ValueError(f"Incomplete or over-budget condition: {name}")
        evaluation = row.get("evaluation", {})
        metrics = [evaluation.get(k) for k in ("iou_50_90", "iou_50_95", "AP_50_95_points")]
        metrics = [n.get("AP") if isinstance(n, dict) else n for n in metrics]
        if not any(isinstance(n, (float, int)) and math.isfinite(n) for n in metrics):
            raise ValueError(f"Missing finite evaluated AP: {name}")
    return source, result


def prepare(project: Path, policy: dict) -> Path:
    """Idempotently build a separate S14 import and checkpoint, then return it."""
    source, result = _validate_source(project, policy)
    key = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:12]
    dest = project / ("run-posttrain-" + key)
    if (dest / "external_result_import.json").exists():
        validate_import(dest)
        return dest
    context = _local(project, policy["context_run"])
    if not Path(policy["config_path"]).is_file():
        raise ValueError("Analysis configuration is missing")
    staging = project / (".posttrain-" + key)
    staging.mkdir(exist_ok=True)
    # Preserve compact research context; never copy data, weights, or old executions.
    for number in range(1, 14):
        old = context / f"stage-{number:02d}"
        if not old.is_dir():
            continue
        for item in old.iterdir():
            if item.is_file() and not item.is_symlink() and item.suffix in {".md", ".json", ".yaml", ".jsonl", ".bib"} and item.stat().st_size < 2_000_000:
                target = staging / old.name / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
    runs = staging / "stage-14/runs"
    runs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "results.json", runs / "results.json")
    artifacts = {}
    for relative in policy["artifacts"]:
        path = source / relative
        if not path.resolve().is_relative_to(source.resolve()) or not path.is_file():
            raise ValueError(f"Missing or escaping training artifact: {relative}")
        stat = path.stat()
        if stat.st_size == 0:
            raise ValueError(f"Empty training artifact: {relative}")
        artifacts[relative] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if not artifacts:
        raise ValueError("No final artifacts registered")
    receipt = {"schema": 1, "source_run": str(source), "results_sha256": digest(source / "results.json"),
               "policy": policy, "artifacts": artifacts, "training_allowed": False,
               "artifact_check": "existence_size_mtime; weights are not reloaded or content-hashed"}
    write(staging / "external_result_import.json", receipt)
    scope = (f"# Authoritative completed-run scope\n\nEach condition completed {policy['target_epochs']} cumulative epochs "
             f"and {policy['target_steps']} optimizer updates. This run imports existing results; no new training "
             "or hyperparameter search is authorized. Earlier stage documents are historical context.\n\n"
             + policy["analysis_scope"] + "\n\nUse stage-14/runs/results.json as measured evidence. "
             "Keep the two AP IoU ranges separate. A scene subset, one seed, or an engineering check "
             "does not establish full-benchmark superiority or statistical significance. Record missing "
             "experiments as future work; do not execute them.\n")
    (runs / "completed_run_scope.md").write_text(scope, encoding="utf-8")
    (staging / "stage-09").mkdir(exist_ok=True)
    (staging / "stage-09/completed_run_scope.md").write_text(scope, encoding="utf-8")
    write(staging / "project_meta.json", {**read(project / "project_meta.json"), "config_path": policy["config_path"]})
    write(staging / "checkpoint.json", {"last_completed_stage": 14, "last_completed_name": "EXPERIMENT_RUN",
          "run_id": dest.name, "timestamp": datetime.now(timezone.utc).isoformat(), "source": "verified_external_import"})
    if dest.exists():
        raise ValueError("Unrecognized existing post-training directory")
    staging.rename(dest)
    return dest


def validate_import(run_dir: Path) -> dict:
    receipt = read(run_dir / "external_result_import.json")
    if receipt.get("schema") != 1 or receipt.get("training_allowed") is not False:
        raise ValueError("Invalid external result receipt")
    source, _ = _validate_source(run_dir.parent, receipt["policy"])
    if digest(source / "results.json") != receipt["results_sha256"] or digest(run_dir / "stage-14/runs/results.json") != receipt["results_sha256"]:
        raise ValueError("Completed training results changed after import")
    for name, saved in receipt["artifacts"].items():
        path = source / name
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Artifact escaped its source run")
        stat = path.stat()
        if stat.st_size != saved["size"] or stat.st_mtime_ns != saved["mtime_ns"]:
            raise ValueError(f"Training artifact changed after import: {name}")
    return receipt


def review(run_dir: Path, stage_dir: Path) -> None:
    receipt = validate_import(run_dir)
    # Recover compact measured evidence before S16, including logs omitted by results.json.
    from researchclaw.pipeline.artifact_audit import collect_training_evidence
    collect_training_evidence(Path(receipt['source_run']), stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "stage-14/runs/results.json", stage_dir / "results.json")
    shutil.copy2(run_dir / "stage-14/runs/completed_run_scope.md", stage_dir / "completed_run_scope.md")
    final = stage_dir / "experiment_final"
    final.mkdir(exist_ok=True)
    shutil.copy2(stage_dir / "results.json", final / "results.json")
    (final / "SOURCE.md").write_text(
        "The executed source and weights remain in " + receipt["source_run"] +
        ". This directory contains imported results only, not regenerated experiment code.\n",
        encoding="utf-8")
    write(stage_dir / "refinement_log.json", {"adopted": True, "improved": False, "retrained": False,
          "verification_mode": "external_completed_run_review", "source_run": receipt["source_run"],
          "artifact_check": receipt["artifact_check"], "next_stage": "RESULT_ANALYSIS"})
