"""Bind a paired pilot's native AI review to the evidence consumed by its dispatch."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path

from ai_experiment_loop import FAMILIES, POLICY, is_pass, read, terminal_result, write


REQUEST = "long_review_request.json"
DECISION = "long_review_decision.json"


def _hash(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Missing pilot artifact: {path.name}")
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _stamp(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=32)
def _hash_checkpoint_generation(path, stamp):
    # The long-running scheduler can retry the API without rereading multi-GB weights.
    # A newly launched dispatch has a fresh cache and independently verifies the bytes.
    digest = _hash(path)
    _check(_stamp(path) == stamp, "Checkpoint changed while hashing")
    return digest


def _paired_updates(run, folders, family, endpoint, budget):
    """Verify the two real four-rank ledgers and bind their contents in one pass."""
    _check(type(budget) is int and type(endpoint) is int and 0 < budget <= endpoint,
           "Missing or invalid pilot budget")
    first = endpoint - budget + 1
    keys = ("sample_indices", "frame_ids") if family == "cirb_mask_boundary" else ("sample_ids", "frame_ids")
    rank_key = "rank_records" if family == "osgk_depth_order" else "ranks"
    paths = [folder / "steps.jsonl" for folder in folders]
    hashes = [hashlib.sha256(), hashlib.sha256()]
    count = 0
    with paths[0].open("rb") as left, paths[1].open("rb") as right:
        for index, pair in enumerate(zip_longest(left, right)):
            _check(index < budget and None not in pair, "Unpaired or excess pilot updates")
            observations = []
            for raw, digest in zip(pair, hashes):
                digest.update(raw)
                row = json.loads(raw)
                ranks = row.get(rank_key)
                _check(row.get("step") == first + index and isinstance(ranks, list) and len(ranks) == 4,
                       "Missing, repeated or incomplete four-rank pilot update")
                participants = []
                for rank, item in enumerate(ranks):
                    _check(isinstance(item, dict) and item.get("rank", rank) == rank and all(item.get(k) for k in keys),
                           "Missing rank or sample provenance")
                    participants.append([item[k] for k in keys])
                observations.append(participants)
            _check(observations[0] == observations[1], "Pilot arms used different samples or frames")
            count += 1
    _check(count == budget, "Pilot ledgers do not cover the planned update budget")
    return {str(path.relative_to(run)): digest.hexdigest() for path, digest in zip(paths, hashes)}


def evidence(run: Path, family: str) -> dict:
    policy = read(run.parent / POLICY)
    _check(policy.get("enabled") is True and policy.get("family") == family, "Pilot policy mismatch")
    scenes = policy["scenes"]
    filename, _, names = FAMILIES[family]
    result = read(run / filename)
    metrics = terminal_result(run, scenes, family)
    _check(metrics and is_pass(metrics, policy), "Paired validation gate did not pass")
    setup = read(run / "setup.json")
    revision = read(run / "ai_revision.json")
    probe = read(run / "preflight_mechanism.json")
    _check(probe.get("passed") is True and probe.get("nonfinite_signal") is False,
           "Missing successful mechanism probe")
    artifacts = {}
    declared = revision.get("method_file_sha256")
    _check(declared and revision.get("family", family) == family, "Missing method provenance")
    for relative, expected in declared.items():
        file = (run / relative).resolve()
        _check(file.is_relative_to(run.resolve()) and _hash(file) == expected, "Method snapshot changed")
    for relative in ("setup.json", "ai_revision.json", "mechanism_plan.json", "preflight_mechanism.json", "dispatch.py", filename):
        artifacts[relative] = _hash(run / relative)
    a_plan = family == "osgk_depth_order"
    b_plan = family == "cirb_mask_boundary"
    steps = setup["short_endpoint" if a_plan else "target_steps"]
    budget = read(run / "mechanism_plan.json").get("steps_per_arm")
    if a_plan:
        _check(result.get("step") == steps and result.get("stage_updates") == budget,
               "Checkpoint and validation budgets differ")
    else:
        _check(budget == steps, "Planned and launched short budgets differ")
        ids = setup["scene_ids"]
        _check(len(ids) == len(set(ids)) == scenes, "Validation scene manifest differs")
        audit = read(run / "training_audit.json")
        paired = audit.get("matched_data_and_lr") if b_plan else audit.get("matched_actual_samples_frames_and_lr")
        updates = audit.get("optimizer_updates_each") if b_plan else audit.get("adapter_updates_per_condition")
        _check(paired is True and updates == steps, "Training audit or matched budget differs")
        artifacts["training_audit.json"] = _hash(run / "training_audit.json")
    folders = []
    for arm in names[:2]:
        if a_plan:
            folder = run / "short" / arm
            summary = read(folder / "pooled312.json")
            weight = folder / "training" / arm / "last_resume.pt"
            _check(summary.get("step") == steps and summary.get("AP_points") == result["metrics"][arm],
                   "Pooled arm metrics differ from comparison")
            expected = summary.get("checkpoint_sha256")
        elif b_plan:
            folder = run / arm
            weight = folder / "training" / "last_resume.pt"
            expected = None
            for scene in ids:
                _check(isinstance(scene, str) and Path(scene).name == scene, "Invalid scene ID")
                trace = read(folder / "scenes" / scene / "verified.json")
                expected = expected or trace.get("checkpoint_sha256")
                _check(trace == {"scene": scene, "checkpoint_sha256": expected, "protocol": setup["protocol_digest"]},
                       "Scene and checkpoint provenance differ")
        else:
            weight = run / arm / "training" / arm / "last_resume.pt"
            expected = result["weights"][arm]["sha256"]
        digest = _hash_checkpoint_generation(weight, _stamp(weight))
        _check(digest == expected, "Scored checkpoint differs from continuation checkpoint")
        artifacts[str(weight.relative_to(run))] = digest
        folders.append(weight.parent)
    artifacts.update(_paired_updates(run, folders, family, steps, budget))
    return {"run": run.name, "revision": revision["revision"], "family": family, "metrics": metrics,
            "short_endpoint": steps, "actual_updates_each": budget, "long_endpoint": setup["long_steps"], "scenes": scenes,
            "min_ap_gain": policy["min_ap_gain"], "max_ap25_drop": policy["max_ap25_drop"],
            "probe": probe, "artifacts": artifacts}


def fingerprint(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def may_start_long(run: Path, family: str) -> bool:
    """Called by the dispatch immediately before handing its weights to long training."""
    data = evidence(run, family)
    identity = fingerprint(data)
    request = read(run / REQUEST)
    if request.get("digest") != identity:
        write(run / REQUEST, {"digest": identity, "evidence": data})
    review = read(run / DECISION)
    if review.get("digest") != identity or review.get("action") != "approve_long":
        return False
    lineage = run / "long_start_lineage.json"
    if lineage.exists():
        _check(read(lineage).get("digest") == identity, "Long training belongs to another pilot")
    else:
        write(lineage, {"digest": identity, "run": run.name})
    return True


def review_request(repo, run, family, model):
    """Return an existing verdict or request one review for the dispatch's exact evidence."""
    request = read(run / REQUEST)
    data = evidence(run, family)
    identity = fingerprint(data)
    if request.get("digest") != identity or request.get("evidence") != data:
        raise ValueError("Pilot evidence changed after the handoff request")
    previous = read(run / DECISION)
    if previous.get("digest") == identity and previous.get("action") in ("approve_long", "revise_method"):
        return previous
    payload = {"outcome": "review_paired_pilot", "evidence_digest": identity, "evidence": data,
               "actions": {"approve_long": "同意按已验证的短训检查点进入长训", "revise_method": "修订方法并重新短训"}}
    from ai_experiment_loop import validated_choice
    try:
        reply = validated_choice(model(repo, payload), payload["actions"])
    except Exception as exc:
        # The next service tick retries the review. Never convert a model failure to a pass.
        return {"action": "await_ai", "error_type": type(exc).__name__}
    reply["digest"] = identity
    write(run / DECISION, reply)
    return reply
