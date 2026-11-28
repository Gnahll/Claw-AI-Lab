"""Source handoff and reporting semantics shared by analysis, writing and export.

Snapshots describe files captured at import time, not an immutable training-time
history. Reporting AP is distinct from minimizing a negative-AP objective.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SKIP = {"outputs", "datasets", "dataset", "checkpoints", "weights", "tests", "node_modules",
         "__pycache__", "codebases", "repair_backup"}
_CONFIGS = {"train.yaml", "train.yml", "config.yaml", "config.yml", "frozen_config.yaml", "frozen_config.yml"}


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(root: Path):
    """Small implementation/configuration files; no outputs, weights or symlinks."""
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "repair_"))
                         and d not in _SKIP and not (Path(folder) / d).is_symlink())
        for name in sorted(files):
            path = Path(folder) / name
            if (not path.is_symlink() and not name.startswith(".")
                    and (path.suffix == ".py" or name in _CONFIGS)
                    and not name.endswith(("_tests.py", "_test.py")) and not name.startswith("test_")
                    and path.stat().st_size <= 2_000_000):
                yield path


def snapshot_sources(source: Path, final_dir: Path) -> dict:
    """Capture an explicitly registered completed experiment, without touching it."""
    target = final_dir / "source"
    manifest_path = final_dir / "source_manifest.json"
    if manifest_path.exists():
        manifest = _read(manifest_path)
        validate_snapshot(final_dir)
        if manifest.get("source_root") != str(source.resolve()):
            raise ValueError("Existing implementation snapshot belongs to another run")
        return manifest
    if final_dir.resolve().is_relative_to(source.resolve()):
        raise ValueError("Implementation snapshot must be outside the training workspace")
    if target.exists():
        raise ValueError("Refusing to overwrite an unregistered source snapshot")
    target.mkdir(parents=True)
    records = {}
    candidates = list(source_files(source))
    # Executed configurations often live beside checkpoints; take only these
    # small named files, never traverse/copy prediction tensors or weight files.
    outputs = source / "outputs"
    if outputs.is_dir():
        for name in _CONFIGS:
            candidates.extend(p for p in outputs.glob("**/" + name)
                              if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(source.resolve())
                              and p.stat().st_size <= 2_000_000)
    for path in candidates:
        relative = path.relative_to(source).as_posix()
        original_relative = relative
        if path.is_relative_to(outputs):
            relative = "executed_configuration/" + path.relative_to(outputs).as_posix()
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        records[relative] = {"sha256": _sha(dest), "bytes": dest.stat().st_size, "original_relative": original_relative}
    manifest = {"schema": 1, "source_root": str(source.resolve()), "files": records,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "scope": "Import-time source snapshot, not independently verified training-time code history",
                "source_available": bool(records)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_snapshot(final_dir: Path) -> Path:
    manifest = _read(final_dir / "source_manifest.json")
    if manifest.get("schema") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Invalid implementation snapshot manifest")
    root = final_dir / "source"
    for name, row in manifest["files"].items():
        path = root / name
        if (not path.resolve().is_relative_to(root.resolve()) or path.is_symlink()
                or not path.is_file() or _sha(path) != row.get("sha256")):
            raise ValueError(f"Implementation snapshot changed or escaped its root: {name}")
    return root


def external_source_root(run_dir: Path) -> tuple[Path | None, str]:
    """Never interpret a free-form SOURCE.md as permission to read arbitrary paths."""
    receipt = run_dir / "external_result_import.json"
    if not receipt.is_file():
        return None, "not_external"
    final = run_dir / "stage-15/experiment_final"
    if (final / "source_manifest.json").exists():
        return validate_snapshot(final), "import_time_snapshot"
    # Backward compatibility for already-imported projects lacking the new snapshot.
    from researchclaw.pipeline.external_results import _local
    registered = _read(receipt)
    if (registered.get("schema") != 1 or registered.get("training_allowed") is not False
            or registered.get("source_run") != registered.get("policy", {}).get("source_run")):
        raise ValueError("Invalid registered external source")
    # This grants read access to registered source, not a fresh verification of
    # training/weight artifacts. S15/S16 keep their separate validate_import gate.
    source = _local(run_dir.parent, registered["source_run"])
    return source.resolve(), "registered_live_source_not_training_time_snapshot"


def load_measured_results(run_dir: Path) -> dict:
    def stage_key(path: Path) -> tuple[int, int]:
        if path.name == "stage-15":
            return (1, 0)
        match = re.fullmatch(r"stage-15_v(\d+)", path.name)
        return (0, int(match.group(1))) if match else (-1, 0)

    stages = sorted(run_dir.glob("stage-15*"), key=stage_key, reverse=True)
    for stage in stages:
        for relative in ["results.json", "experiment_final/results.json", "experiment_final/results_v0.json"]:
            data = _read(stage / relative)
            if data.get("conditions"):
                return data
    return _read(run_dir / "stage-14/runs/results.json")


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reporting_contract(results: dict, metric_key: str = "primary_metric", direction: str = "minimize") -> dict:
    """Recover explicit reporting values; never abs() a generic negative metric."""
    conditions = results.get("conditions", {})
    rows, ap_values, transforms, ranges = {}, {}, [], []
    declared = results.get("data_split", {}).get("val", {}).get("scene_count")
    for name, value in conditions.items():
        if not isinstance(value, dict):
            continue
        evaluation = value.get("evaluation", {})
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        scenes = evaluation.get("scenes")
        scene_ids = sorted(set(scenes)) if isinstance(scenes, list) and all(isinstance(x, str) for x in scenes) else None
        definitions = evaluation.get("metric_definitions", value.get("metric_definitions", {}))
        ap = evaluation.get("AP_percent")
        ap_source = "evaluation.AP_percent"
        raw95 = evaluation.get("iou_50_95", {}).get("AP") if isinstance(evaluation.get("iou_50_95"), dict) else None
        has95 = _number(raw95) and 0 <= raw95 <= 1
        if not _number(ap):
            if has95:
                ap, ap_source = 100 * raw95, "100 * evaluation.iou_50_95.AP"
        consistent = not has95 or (_number(ap) and math.isclose(ap, 100 * raw95, rel_tol=1e-7, abs_tol=1e-8))
        objective = value.get(metric_key)
        # Both the measured AP value and the objective relation must be present.
        if consistent and _number(ap) and 0 <= ap <= 100 and _number(objective):
            if direction == "minimize" and math.isclose(objective, -ap, rel_tol=1e-7, abs_tol=1e-8):
                ap_values[name] = ap
                transforms.append("negative_AP_objective_to_measured_AP_percent")
            elif direction == "maximize" and math.isclose(objective, ap, rel_tol=1e-7, abs_tol=1e-8):
                ap_values[name] = ap
                transforms.append("identity_AP_percent")
            if name in ap_values:
                ranges.append(".50:.95" if has95 else "not_recorded")
        scope = "not_recorded"
        if scene_ids is not None and isinstance(declared, int):
            scope = "subset" if len(scene_ids) < declared else "reported_count_matches_declared" if len(scene_ids) == declared else "count_exceeds_declared"
        rows[name] = {"optimizer_steps": value.get("optimizer_steps_completed"), "epochs": value.get("epochs_completed"),
                      "scene_count": len(scene_ids) if scene_ids is not None else None,
                      "scene_ids": scene_ids if scene_ids is not None and len(scene_ids) <= 8 else None,
                      "scene_ids_sha256": hashlib.sha256(json.dumps(scene_ids).encode()).hexdigest() if scene_ids is not None else None,
                      "declared_validation_scenes": declared, "evaluation_scope": scope,
                      "geometry": evaluation.get("geometry_label", "not_recorded"),
                      "reported_metric_source": ap_source if _number(ap) else None,
                      "AP_fields_consistent": consistent,
                      "metric_definitions": definitions,
                      "precision_recall_operating_point": definitions.get("precision_recall_operating_point", "not_recorded") if isinstance(definitions, dict) else "not_recorded",
                      "validation_loss_windows": value.get("validation", {}).get("windows") if isinstance(value.get("validation"), dict) else None}
    hashes = [row["scene_ids_sha256"] for row in rows.values()]
    comparable = bool(hashes) and all(hashes) and len(set(hashes)) == 1
    result = {"schema": 1, "measured_results_sha256": hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest(),
              "optimizer_objective": {"name": metric_key, "direction": direction},
              "report_metric": None, "conditions": rows, "same_reported_scene_ids": comparable,
              "seed": results.get("seed"),
              "rules": ["Do not confuse the optimizer objective with the reported metric.",
                        "Report scene counts separately from loss-window counts; matching counts alone do not verify official split membership.",
                        "Unknown precision/recall operating point is not a fixed confidence threshold; do not infer it from metric names.",
                        "A source file or a valid algebraic operator does not establish segmentation utility or statistical significance."]}
    if rows and len(ap_values) == len(rows):
        result["report_metric"] = {"name": "AP[.50:.95]" if all(v == ".50:.95" for v in ranges) else "AP (IoU range not recorded)", "unit": "percent", "direction": "maximize",
                                   "values": ap_values, "objective_transform": transforms[0]}
    return result


# Match affirmative measurement statements, so naming or discussing the full split is safe.
# Keep the object next to the measurement verb: "a subset of the full set" has a different object.
_SPLIT_WORDS = r"(?:[a-z0-9][\w.-]*\s+){0,4}validation\s+(?:split|set|scenes|benchmark)"
_FULL_SPLIT = rf"(?:the\s+)?(?:full|complete|entire)\s+{_SPLIT_WORDS}"
_NUMBERED_SPLIT = (
    rf"(?P<all>all\s+)?(?P<count>\d+)\s+(?:{_SPLIT_WORDS}|scenes\s+(?:of|in|from)\s+(?:the\s+)?{_SPLIT_WORDS})"
)
_SCOPE_OBJECT = rf"(?:{_FULL_SPLIT}|{_NUMBERED_SPLIT}|all\s+{_SPLIT_WORDS})"
_SCOPE_ACTIVE = re.compile(
    rf"\b(?:we\s+(?:evaluate[d]?|test(?:ed)?|benchmark(?:ed)?)"
    rf"(?P<active_subject>(?:\s+(?:(?:our|the|both|each|all)\s+)?(?:[a-z0-9_.-]+\s+)?(?:model|approach|method|variant|condition)s?)?)"
    rf"|(?:we\s+(?:report(?:ed)?|measure[d]?|compute[d]?)\s+(?:the\s+)?(?:instance\s+)?(?:mean\s+)?ap(?:\d+|\[[^\]]+\])?)"
    rf"|(?:ap(?:\d+|\[[^\]]+\])?)\s+(?:is|was)\s+(?:measured|evaluated|computed|reported))"
    rf"\s+(?:(?:on|over|across|using)\s+)?{_SCOPE_OBJECT}\b", re.I,
)
_SCOPE_ACTIVE_NAMED = re.compile(
    rf"\bwe\s+(?:evaluate[d]?|test(?:ed)?|benchmark(?:ed)?)\s+"
    rf"(?P<active_subject>.{{1,160}}?)\s+(?:on|over|across|using)\s+{_SCOPE_OBJECT}\b", re.I,
)
_SCOPE_SUBJECT = re.compile(
    rf"\b(?P<subject>(?:(?:our|the|both|each|all)\s+)?[a-z0-9_.-]+(?:\s+(?:model|approach|method|variant|condition))?)"
    rf"\s+(?:was|were|is|are|has\s+been|have\s+been)\s+(?:evaluated|tested|benchmarked)\s+(?:on|over|across|using)\s+{_SCOPE_OBJECT}\b",
    re.I,
)
_SCOPE_SUBJECT_LIST = re.compile(
    rf"\b(?P<subject>(?:(?:our|the|both|each|all)\s+)?[a-z0-9_.-]+"
    rf"(?:\s*,\s*[a-z0-9_.-]+)*(?:\s*,?\s+(?:and|or)\s+[a-z0-9_.-]+)"
    rf"(?:\s+(?:models?|approaches?|methods?|variants?|conditions?))?)"
    rf"\s+(?:was|were|is|are|has\s+been|have\s+been)\s+(?:evaluated|tested|benchmarked)\s+(?:on|over|across|using)\s+{_SCOPE_OBJECT}\b", re.I,
)
_SCOPE_OBJECT_FRONT = re.compile(
    rf"\b{_SCOPE_OBJECT}\s+(?:was|were|is|are)\s+(?:used\s+(?:to\s+evaluate|for\s+evaluation)|evaluated)\s+"
    rf"(?P<object_subject>(?:(?:our|the)\s+)?[a-z0-9_.-]+(?:\s+(?:and|or)\s+[a-z0-9_.-]+)?"
    rf"(?:\s+(?:model|approach|method|variant|condition))?)\b", re.I,
)
_SCOPE_PASSIVE = re.compile(
    rf"\b{_SCOPE_OBJECT}\s+(?:was|were|is|are)\s+(?:used\s+(?:to\s+evaluate|for\s+evaluation)|evaluated)\b", re.I,
)
_SCOPE_REPORTED = re.compile(
    rf"\b(?:results?\s+for\s+(?P<result_subject>.{{1,80}}?)\s+(?:is|are|was|were)\s+reported"
    rf"|(?:the\s+)?evaluation\s+(?:used|uses|was\s+(?:conducted|performed))"
    rf"|(?:(?:the\s+)?(?P<reported_subject>[a-z0-9_.-]+)\s+evaluation\s+(?:was|is|has\s+been)\s+(?:conducted|performed))"
    rf"|(?P<reported_results_subject>[a-z0-9_.-]+)\s+(?:performance|results?)\s+(?:is|are|was|were)\s+reported"
    rf"|we\s+report\s+(?P<reported_active_subject>[a-z0-9_.-]+)\s+results?)\s+(?:on|over|across|using)?\s*{_SCOPE_OBJECT}\b", re.I,
)
_SCOPE_CONDUCTED = re.compile(
    rf"\b(?P<conducted_subject>(?:(?:all|both|each)\s+)?(?:experiments?|evaluations?|benchmarks?|tests?))"
    rf"\s+(?:was|were|is|are|has\s+been|have\s+been)\s+(?:conducted|performed|run)"
    rf"\s+(?:on|over|across|using)\s+{_SCOPE_OBJECT}\b", re.I,
)
_SCOPE_USED_TO_EVALUATE = re.compile(
    rf"\bwe\s+(?:used|use)\s+{_SCOPE_OBJECT}\s+to\s+(?:evaluate|test|benchmark)\s+"
    rf"(?P<used_subject>(?:(?:our|the|both|each|all)\s+)?[a-z0-9_.-]+"
    rf"(?:\s+(?:model|approach|method|variant|condition))?)\b", re.I,
)
_SCOPE_ACHIEVED = re.compile(
    rf"\b(?P<achieved_subject>[a-z0-9_.-]+)\s+(?:achieve[d]?|obtain(?:ed)?|reach(?:ed)?)\s+"
    rf"(?:an?\s+)?(?:instance\s+)?ap(?:@\d+|_\{{?\d+\}}?|\d+|\[[^\]]+\])?(?:\s*=\s*|\s+(?:of\s+)?)[-+]?\d+(?:\.\d+)?"
    rf"\s+(?:on|over|across|using)\s+{_SCOPE_OBJECT}\b", re.I,
)
_CLAUSE_LIMIT = re.compile(r"(?<=[.!?;])\s+|\n\s*\n|\s+[—–]\s+")
_QUALIFIED_SCOPE = re.compile(
    r"\b(?:if|whether|planned|planning|would|could|will|future|wish|suppose|hypothetical|untrue|"
    r"incorrect(?:ly)?|false|not|no|neither|never|cannot|did(?:n['’]t| not)|was(?:n['’]t| not)|"
    r"were(?:n['’]t| not)|is(?:n['’]t| not)|are(?:n['’]t| not))\b", re.I,
)


def evaluation_scope_conflicts(paper: str, results: dict) -> list[str]:
    """Check asserted validation coverage against explicit per-condition scene IDs and split size."""
    conditions = results.get("conditions")
    split = results.get("data_split")
    if not isinstance(conditions, dict) or not isinstance(split, dict) or not isinstance(split.get("val"), dict):
        return []
    declared = split["val"].get("scene_count")
    if type(declared) is not int or declared <= 0:
        return []
    # A count match alone is not proof of official split membership. Only an
    # explicit official scene manifest can establish a full-split claim.
    official = split["val"].get("scene_ids")
    official_ids = (
        set(official) if isinstance(official, list) and len(official) == declared
        and all(isinstance(scene, str) for scene in official) and len(set(official)) == declared
        else None
    )
    rows = reporting_contract(results)["conditions"]
    recorded = {}
    for name, value in conditions.items():
        evaluation = value.get("evaluation", {}) if isinstance(value, dict) else {}
        scenes = evaluation.get("scenes") if isinstance(evaluation, dict) else None
        ids = set(scenes) if isinstance(scenes, list) and all(isinstance(scene, str) for scene in scenes) else None
        if ids is not None:
            recorded[name] = ids
    unsupported = {
        name: row["scene_count"] for name, row in rows.items()
        if name not in recorded or official_ids is None or recorded[name] != official_ids
    }
    conflicts = []
    text = re.sub(r"\x60\x60\x60.*?(?:\x60\x60\x60|$)", "", paper, flags=re.S)
    reference_heading = re.search(
        r"(?im)^#{1,3}\s+(?:references|bibliography)\s*$", text
    )
    if reference_heading:
        text = text[:reference_heading.start()]
    names = "|".join(re.escape(name) for name in rows)
    named_claim = re.compile(
        rf"(?<!\w)(?:we\s+(?:evaluated?|tested)\s+(?:{names})|(?:{names})\s+(?:was|is)\s+evaluated)"
        rf"\s+(?:on|across|over)\s+{_SCOPE_OBJECT}\b", re.I)

    for raw in _CLAUSE_LIMIT.split(text):
        clause = " ".join(raw.replace("**", "").split())
        for pattern in (
            _SCOPE_ACTIVE_NAMED, _SCOPE_ACTIVE, _SCOPE_SUBJECT_LIST, _SCOPE_SUBJECT,
            _SCOPE_OBJECT_FRONT, _SCOPE_PASSIVE, _SCOPE_REPORTED,
            _SCOPE_CONDUCTED, _SCOPE_USED_TO_EVALUATE, _SCOPE_ACHIEVED, named_claim,
        ):
            for match in pattern.finditer(clause):
                if pattern is _SCOPE_PASSIVE and _SCOPE_OBJECT_FRONT.search(clause):
                    continue
                lead = re.split(r"[,;]|\bbut\b|\band\b", clause[:match.start()], flags=re.I)[-1]
                # Only a qualifier immediately adjacent to the assertion can
                # negate it. Configuration prose such as "with no test-time
                # augmentation we evaluated ..." is still an affirmative claim.
                collective_exclusion_prefix = re.search(
                    r"\b(?:all|both|each)\s+(?:models?|methods?|approaches?|variants?|conditions?)\s+"
                    r"but\s+not\s*$", clause[:match.start()], re.I,
                )
                if (
                    _QUALIFIED_SCOPE.search(lead[-24:])
                    and not re.search(r"\bwe\s*$", lead, re.I)
                    and not collective_exclusion_prefix
                ):
                    continue
                if re.search(r"\b(?:only\s+for|for\s+(?:the\s+)?)\s*(?:training\s+|validation\s+|depth\s+)?loss(?:es)?\b", clause[match.end():], re.I):
                    continue
                claimed = int(match["count"]) if match["count"] else declared
                is_full_claim = (
                    match["count"] is None
                    or bool(match.groupdict().get("all"))
                    or claimed == declared
                )
                tail = clause[match.end():]
                if re.match(r"\s+(?:in|from|within)\s+(?:our|the|a)\s+subset\b", tail, re.I):
                    continue
                if re.search(r"^[\s\"'”’)]*(?:is|was|are|were)?\s*(?:false|incorrect|untrue)\b", tail, re.I):
                    continue
                # A plain numeric statement describes exactly the recorded subset;
                # it is not a claim that this number is the complete split.
                if not is_full_claim:
                    continue
                subject = (
                    match.groupdict().get("subject")
                    or match.groupdict().get("active_subject")
                    or match.groupdict().get("result_subject")
                    or match.groupdict().get("reported_subject")
                    or match.groupdict().get("reported_results_subject")
                    or match.groupdict().get("reported_active_subject")
                    or match.groupdict().get("conducted_subject")
                    or match.groupdict().get("used_subject")
                    or match.groupdict().get("achieved_subject")
                    or match.groupdict().get("object_subject")
                )
                if subject and _QUALIFIED_SCOPE.search(subject):
                    continue
                named = {
                    name for name in rows
                    if subject and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", subject, re.I)
                }
                except_tail = re.search(
                    r"\b(?:except|excluding|but\s+not)\s+(.+?)\s+(?:was|were|is|are)\s+"
                    r"(?:evaluated|tested|benchmarked)\b",
                    clause, re.I,
                )
                if except_tail is None:
                    except_tail = re.search(
                        r"\b(?:all|both|each)\s+(?:models?|methods?|approaches?|variants?|conditions?)\s+"
                        r"but\s+not\s+(.+?)\s+(?:was|were|is|are)\s+(?:evaluated|tested|benchmarked)\b",
                        clause, re.I,
                    )
                collective_clause = bool(except_tail and re.search(
                    r"\b(?:all|both|each)\s+(?:models?|methods?|approaches?|variants?|conditions?)\b",
                    clause[:match.start()] if match.start() else clause[:except_tail.start()], re.I,
                ))
                excluded = {
                    name for name in rows
                    if (
                        except_tail
                        and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", except_tail.group(1), re.I)
                    ) or re.search(
                        rf"\b(?:except|excluding|but\s+not)\s+(?:the\s+)?{re.escape(name)}(?!\w)",
                        clause, re.I,
                    )
                }
                if not named and pattern is named_claim:
                    named = {
                        name for name in rows
                        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", match.group(0), re.I)
                    }
                collective_subject = bool(subject and re.search(
                    r"\b(?:all|both|each)\s+(?:models?|methods?|approaches?|variants?|conditions?)\b",
                    subject, re.I,
                ))
                relevant = (
                    set(rows) if collective_clause or collective_subject or not named else named
                ) - excluded
                wrong = sorted(name for name in relevant if name in unsupported)
                if bool(match.groupdict().get("all")) and claimed != declared:
                    wrong = sorted(set(wrong) | set(relevant))
                if wrong:
                    counts = ", ".join(f"{name} {rows[name]['scene_count']}/{declared}" for name in wrong)
                    message = f"Evaluation scope: manuscript claims {claimed} validation scenes; recorded scene IDs cover {counts}."
                    if message not in conflicts:
                        conflicts.append(message)
    return conflicts


def emit_reporting_contract(run_dir: Path, destination: Path, metric_key: str = "primary_metric",
                            direction: str = "minimize", *, chart: bool = False) -> dict:
    contract = reporting_contract(load_measured_results(run_dir), metric_key, direction)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "paper_reporting_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    metric = contract["report_metric"]
    if chart and metric and contract["same_reported_scene_ids"]:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        out = destination / "charts/fig_main_comparison.png"
        out.parent.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        values = metric["values"]
        bars = ax.bar(list(values), list(values.values()))
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in values.values()], padding=4)
        ax.set_ylabel(metric["name"] + " (%) - higher is better")
        count = next(iter(contract["conditions"].values()))["scene_count"]
        ax.set_title(f"Recorded {count}-scene comparison (not an optimizer-objective plot)")
        ax.set_ylim(0, max(1, max(values.values()) * 1.16))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        (destination / "paper_metric_chart.json").write_text(json.dumps({"metric": metric, "source": "paper_reporting_contract.json",
            "image_sha256": _sha(out), "error_bars": "not inferred from a single stored condition value"}, indent=2), encoding="utf-8")
    return contract
