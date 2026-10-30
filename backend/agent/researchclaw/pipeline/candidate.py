"""Bind the plan, code and check reports of S9-S14 to one frozen candidate.

S9 freezes the completed experiment plan as a candidate. Every later stage
must run against that candidate: S11 code and S12/S14 reports record its id
and the code they used, and a stage refuses to start when the evidence it
builds on belongs to another candidate or to different code. A redesigned
candidate therefore has to be measured again instead of inheriting reports.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

CANDIDATE_FILE = "candidate.json"
BINDING_FILE = "candidate_binding.json"
HISTORY_FILE = "candidate_history.jsonl"
CONTEXT_REGISTRY = "REGISTRY.json"
# Executable code only; reports and outputs written by a run change legitimately.
_CODE_SUFFIXES = (".py", ".sh")
_SKIP_PARTS = ("outputs", "revision-context", "codebases", "datasets", "checkpoints", "__pycache__")

# Stage that produced the code/report a stage builds on, and whether the
# experiment code must still be exactly what that stage recorded. S12 checks
# and records whatever code it runs (including fixes made later by S14), so
# only S14 needs the exact code S12 checked.
_UPSTREAM = {12: (11, False), 13: (12, False), 14: (12, True)}
_BOUND_STAGES = (11, 12, 14)


class CandidateError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(run_dir: Path) -> Path:
    return run_dir / "stage-09" / "exp_plan.yaml"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def freeze(run_dir: Path, *, legacy: bool = False) -> dict[str, Any]:
    """Register stage-09/exp_plan.yaml as the current candidate."""
    plan = _plan(run_dir)
    if not plan.is_file():
        raise CandidateError("S9 produced no exp_plan.yaml; nothing can be frozen")
    digest = _sha256(plan)
    previous = _read_json(run_dir / "stage-09" / CANDIDATE_FILE) or {}
    record = {"candidate_id": "cand-" + digest[:12], "plan_sha256": digest,
              "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "legacy": legacy}
    if previous.get("candidate_id") == record["candidate_id"]:
        record["frozen_at"] = previous.get("frozen_at", record["frozen_at"])
    (run_dir / "stage-09" / CANDIDATE_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    if previous.get("candidate_id") != record["candidate_id"]:
        with open(run_dir / HISTORY_FILE, "a", encoding="utf-8") as log:
            log.write(json.dumps({**record, "previous_candidate_id": previous.get("candidate_id")}) + "\n")
    return record


def invalidate(run_dir: Path) -> None:
    """An unfinished S9 plan must not be handed to later stages."""
    (run_dir / "stage-09" / CANDIDATE_FILE).unlink(missing_ok=True)


def current(run_dir: Path) -> dict[str, Any]:
    plan = _plan(run_dir)
    if not plan.is_file():
        raise CandidateError("No completed S9 experiment plan; rerun S9")
    record = _read_json(run_dir / "stage-09" / CANDIDATE_FILE)
    if record is None:
        # Runs completed before candidates existed: adopt a plan S9 marked done.
        decision = _read_json(run_dir / "stage-09" / "decision.json") or {}
        if decision.get("status") != "done":
            raise CandidateError("S9 has not completed an experiment plan; an unfinished or "
                                 "rejected plan is not handed to later stages. Rerun S9")
        return freeze(run_dir, legacy=True)
    if record.get("plan_sha256") != _sha256(plan):
        raise CandidateError(f"stage-09/exp_plan.yaml changed after candidate {record.get('candidate_id')} "
                             "was frozen; rerun S9 so the changed plan becomes a new candidate")
    return record


def code_manifest(experiment_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    if not experiment_dir.is_dir():
        return files
    for path in sorted(experiment_dir.rglob("*")):
        rel = path.relative_to(experiment_dir)
        if (path.suffix not in _CODE_SUFFIXES or path.is_symlink() or not path.is_file()
                or any(part.startswith(".") or part in _SKIP_PARTS for part in rel.parts[:-1])):
            continue
        files[rel.as_posix()] = _sha256(path)
    return files


def history_context(run_dir: Path) -> set[str]:
    """Registered files that describe earlier candidates, not current inputs.

    ``revision-context/REGISTRY.json`` lists them under ``history`` (file or
    directory paths). A JSON file stamped with another candidate's id is
    history as well. They stay available to design stages but are not copied
    into the code workspace, where they could be read as current evidence.
    """
    root = run_dir / "revision-context"
    registry = _read_json(root / CONTEXT_REGISTRY) or {}
    listed = [str(item).strip("/") for item in registry.get("history") or [] if isinstance(item, str)]
    try:
        candidate_id = current(run_dir)["candidate_id"]
    except CandidateError:
        candidate_id = None
    history = set()
    for path in root.rglob("*") if root.is_dir() else ():
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(rel == item or rel.startswith(item + "/") for item in listed):
            history.add(rel)
        elif path.suffix == ".json":
            stamped = (_read_json(path) or {}).get("candidate_id")
            if stamped and stamped != candidate_id:
                history.add(rel)
    return history


def current_stage_dirs(run_dir: Path) -> list[Path]:
    """Live stage-NN directories, highest stage first.

    Rollback renames superseded stages to stage-NN_vK. A plain reverse sort
    puts stage-11_v1 before stage-11, so earlier candidates' files won.
    """
    dirs = [d for d in run_dir.glob("stage-*") if d.is_dir() and d.name[6:].isdigit()]
    return sorted(dirs, key=lambda d: int(d.name[6:]), reverse=True)


def experiment_dir(run_dir: Path) -> Path:
    return run_dir / "stage-11" / "experiment"


def bind(run_dir: Path, stage: int) -> dict[str, Any]:
    record = current(run_dir)
    binding = {"stage": stage, "candidate_id": record["candidate_id"], "plan_sha256": record["plan_sha256"],
               "code_sha256": code_manifest(experiment_dir(run_dir)),
               "bound_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (run_dir / f"stage-{stage:02d}" / BINDING_FILE).write_text(json.dumps(binding, indent=2), encoding="utf-8")
    return binding


def unbind(run_dir: Path, stage: int) -> None:
    (run_dir / f"stage-{stage:02d}" / BINDING_FILE).unlink(missing_ok=True)


def binding(run_dir: Path, stage: int) -> dict[str, Any] | None:
    return _read_json(run_dir / f"stage-{stage:02d}" / BINDING_FILE)


def handoff_errors(run_dir: Path, stage: int) -> list[str]:
    """Why ``stage`` must not start on the current plan, code and reports."""
    if not 10 <= stage <= 14:
        return []
    try:
        record = current(run_dir)
    except CandidateError as exc:
        return [str(exc)]
    if stage not in _UPSTREAM:
        return []
    upstream, same_code = _UPSTREAM[stage]
    bound = binding(run_dir, upstream)
    if bound is None:
        return [f"S{upstream} has no completed result for candidate {record['candidate_id']}; rerun S{upstream}"]
    if bound.get("candidate_id") != record["candidate_id"] or bound.get("plan_sha256") != record["plan_sha256"]:
        return [f"S{upstream} result belongs to candidate {bound.get('candidate_id')}, not the current candidate "
                f"{record['candidate_id']}; a new candidate must be regenerated and measured again from S{upstream}"]
    if same_code and bound.get("code_sha256") != code_manifest(experiment_dir(run_dir)):
        return [f"stage-11/experiment changed after S{upstream} recorded it; rerun S{upstream} so the checked "
                "code is the code that runs"]
    return []


def rerun_stage(run_dir: Path, stage: int) -> int | None:
    """Earliest stage that must run again before ``stage`` may start."""
    if not handoff_errors(run_dir, stage):
        return None
    if stage not in _UPSTREAM:
        return 9
    try:
        current(run_dir)
    except CandidateError:
        return 9
    upstream = _UPSTREAM[stage][0]
    return rerun_stage(run_dir, upstream) or upstream


def rewind_checkpoint(run_dir: Path, rerun: int) -> None:
    """Make the next resume start at ``rerun``; never moves the checkpoint forward.

    The runner stops on a failed stage and resumes after the last completed
    one, so without this a rejected handoff would be retried forever.
    """
    path = run_dir / "checkpoint.json"
    data = _read_json(path)
    if data is None or not isinstance(data.get("last_completed_stage"), int):
        return
    if data["last_completed_stage"] < rerun:
        return
    from researchclaw.pipeline.stages import Stage
    previous = Stage(rerun - 1)
    data.update(last_completed_stage=int(previous), last_completed_name=previous.name,
                rewound_for=f"candidate handoff: rerun S{rerun}",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    tmp = path.with_suffix(".rewind.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def record_completion(run_dir: Path, stage: int, done: bool) -> None:
    if stage == 9:
        freeze(run_dir) if done else invalidate(run_dir)
    elif stage in _BOUND_STAGES:
        bind(run_dir, stage) if done else unbind(run_dir, stage)


WORKSPACE_STAMP = ".candidate.json"


def stamp_workspace(run_dir: Path, workspace: Path) -> None:
    """Record which candidate an S14 run workspace trains.

    execute_stage refuses S14 without a current candidate; a runtime invoked
    directly simply leaves the workspace unstamped, so it is never resumed.
    """
    try:
        record = current(run_dir)
    except CandidateError:
        return
    (workspace / WORKSPACE_STAMP).write_text(json.dumps(
        {"candidate_id": record["candidate_id"], "code_sha256": code_manifest(experiment_dir(run_dir))},
        indent=2), encoding="utf-8")


def same_candidate_workspace(run_dir: Path, workspace: Path) -> bool:
    """Checkpoints may resume only within one candidate.

    Code fixed during S14 and smoke-checked again by S12 keeps the candidate,
    so the code hash is recorded for audit but does not block resuming.
    """
    try:
        candidate_id = current(run_dir)["candidate_id"]
    except CandidateError:
        return False
    return (_read_json(workspace / WORKSPACE_STAMP) or {}).get("candidate_id") == candidate_id
