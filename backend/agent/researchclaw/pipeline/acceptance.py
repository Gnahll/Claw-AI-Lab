"""Design-time acceptance definitions, separate from measured research outcomes."""
from __future__ import annotations

import hashlib
import json
import keyword
import re
from pathlib import Path
from typing import Any

import yaml


ACCEPTANCE_POLICY = """
ACCEPTANCE PROTOCOL (platform default for first-time users):
Separate engineering prerequisites from scientific hypotheses. Before training, verify
real resources/data separation, shapes/alignment, finite forward/backward, nonzero new
module gradients, probability constraints when applicable, and memory/time feasibility.
Only these engineering checks may stop training. A predicted AUROC, expected superiority,
expected signal separability, or any measurement of how the new mechanism behaves
(attention/routing distribution, collapse, suppression, leakage between branches) is a
hypothesis: an untrained model cannot meaningfully pass or fail it. Record such values
and report them; negative results are valid. Do not invent an untrained quality threshold
that prevents the experiment intended to test that hypothesis. Thresholds predicted in
hypotheses.md are hypotheses, not gates. Absolute budgets (total time limit, GPU memory,
parameter count) block; ratios against the baseline (time_ratio, memory_ratio) are
recorded: declare them as their own engineering checks, separate from absolute budgets.
Preserve EVERY explicit user-mandated gate, including its metric, threshold, cohort and
timing. A numeric user gate stops training only when the user's own words
(user_requirement_quote) state its numbers; otherwise it is kept, measured and reported.

In the plan include acceptance_protocol: {version: 1, checks: [...]}.
Each check has id, kind (engineering | hypothesis | user_required), phase
(before_training | after_training), blocking (a YAML boolean), and definition (a mapping).
definition MUST specify metric, inputs, aggregation, failure_action; a numerical rule
also needs its threshold. A blocking before_training check MUST additionally declare
required_helpers: [Python function names] so S11 must implement its actual checker.
Hypothesis checks are nonblocking. A user_required check MUST contain user_requirement_quote,
an exact quotation of the user's requirement, not a claim of user approval invented by you.
Scientific gates already present in a preserved protocol keep their thresholds and are
measured and reported; they stop training only under the rule above. Engineering checks
remain mandatory and run before training.
When preserving pretraining_acceptance.mandatory_thresholds, map each entry to one
before_training user_required check via legacy_gate: <original key> and
definition.thresholds: <the entire unchanged threshold mapping>. Do not keep two
contradictory acceptance tables or quietly drop an existing entry.
When the platform reports that failed checks are record-only, write their measured values
to outputs/ and results.json, mark them FAIL_RECORDED and continue to training; do not
change thresholds, delete checks or alter the method.

Define candidate-specific operators, position alignment, references and cohorts BEFORE
measuring; do not transplant another method's diagnostic solely because its name matches.
Resolve definition/precision conflicts during design. Unmeasured memory/batch decisions
may wait for engineering preflight, but missing metric definitions or an implementation_ready
false plan cannot be handed to code generation as complete. A smoke pass proves only its
measured engineering scope, never all scientific prerequisites or completed training.
When blocked, write workspace-root results.json with status=blocked_pretraining, reason
and required_gate_status (keyed by the check ids above) in blocker, and measured evidence
paths. Stop without modifying the frozen protocol. Do not remove mandatory checks or label this as successful training.
GPU numbers written by the user ("GPU1、2", "GPUs 4 and 5") state how many GPUs and how much memory
the experiment may use, not which physical devices. The platform allocates the devices and exposes
exactly them through CUDA_VISIBLE_DEVICES. Never compare physical indices or UUIDs with the user's
numbers; resource checks verify the visible device count and the per-device memory budget.
A fallback batch declared in the plan or by the user (e.g. per-GPU batch 25 with accumulation 2
instead of 50 with accumulation 1, same effective batch) is part of the frozen plan. The code must
implement it: when the primary setting runs out of memory, switch every condition to the fallback
together, record both measurements, and continue. Report a resource blocker only when no fallback
is declared or the fallback also runs out of memory.
"""


def requirement_text(config: Any, run_dir: Path) -> str:
    from researchclaw import single_run

    research = getattr(config, "research", None)
    text = getattr(research, "prompt_topic", getattr(research, "topic", ""))
    if single_run.is_single_run(config):
        # runs_per_condition: 1 is the user's own setting; its policy text is quotable.
        text += "\n" + single_run.SINGLE_RUN_POLICY
    path = run_dir / "human_feedback.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and isinstance(row.get("content"), str):
                text += "\n" + row["content"]
    return text


def registered_context(run_dir: Path, *, max_chars: int = 60000, failure_chars: int = 8000) -> str:
    """Read authoritative definitions for plain-chat S9; never truncate requirements.

    Earlier candidates' measured failures get their own bounded budget: they
    are history, not requirements, so older entries drop their measured
    values first instead of pushing the protocol over its limit.
    """
    parts = []
    for name in ("FIXED_PROTOCOL.yaml", "scientific_readiness.json"):
        path = run_dir / "revision-context" / name
        if path.is_file():
            data = path.read_bytes()
            parts.append(f"Registered protocol {name} (sha256={hashlib.sha256(data).hexdigest()}):\n"
                         + data.decode("utf-8-sig"))
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        raise ValueError("Registered acceptance definitions exceed the design context budget; "
                         "supply a complete bounded protocol, not a silently truncated one")
    failures = design_failure_context(run_dir, max_chars=failure_chars)
    return "\n\n".join(part for part in (text, failures) if part)


def design_failure_context(run_dir: Path, *, max_chars: int = 8000) -> str:
    """Earlier candidates' failures, newest first, with their measured values."""
    path = run_dir / "revision-context" / "MEASURED_DESIGN_FAILURES.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return ""
    failures = [item for item in (data.get("failures") if isinstance(data, dict) else None) or []
                if isinstance(item, dict)]
    if not failures:
        return ""
    header = ("Measured failures of earlier candidate designs (MEASURED_DESIGN_FAILURES.json, newest first). "
              f"{data.get('note', '')}\nUse the measured values to identify how each design failed; "
              "do not resubmit a mechanism that failed the same way.")
    blocks, used = [], len(header)
    for number, item in enumerate(reversed(failures), 1):
        head = (f"\n[{number}] candidate {item.get('candidate_id')} failed at S{item.get('stage')} "
                f"({item.get('recorded_at')}): {str(item.get('measured_failure', ''))[:1500]}")
        values = item.get("measured_values")
        block = head + (f"\n  measured values:\n{values}" if isinstance(values, str) and values else "")
        if used + len(block) > max_chars:
            block = head + ("\n  (measured values omitted for space; see the file)" if values else "")
        if used + len(block) > max_chars:
            blocks.append(f"\n... {len(failures) - number + 1} older failures omitted for space; see the file")
            break
        blocks.append(block)
        used += len(block)
    return header + "".join(blocks)


_LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:[-*•]|\d+[.)])[ \t]+")


def _normalized(text: str) -> str:
    return " ".join(_LIST_MARKER.sub("", text).split())


def quote_found(quote: Any, user_text: str) -> bool:
    """The quote occurs in the user's text, ignoring line breaks, spacing and list markers."""
    return (isinstance(quote, str) and len(quote.strip()) >= 8
            and _normalized(quote) in _normalized(user_text))


def fill_missing_helpers(plan: Any) -> list[str]:
    """Name a checker for each blocking pretraining check that omits required_helpers.

    The name is derived from the check id, so S11 still has to implement and call it.
    Returns one note per filled check.
    """
    protocol = plan.get("acceptance_protocol") if isinstance(plan, dict) else None
    checks = protocol.get("checks") if isinstance(protocol, dict) else None
    notes = []
    for index, check in enumerate(checks if isinstance(checks, list) else []):
        if (not isinstance(check, dict) or check.get("blocking") is not True
                or check.get("phase") != "before_training" or check.get("required_helpers")):
            continue
        stem = re.sub(r"\W+", "_", str(check.get("id") or "")).strip("_").lower()
        if not stem.isascii() or not stem:
            stem = f"acceptance_{index}"
        check["required_helpers"] = [f"check_{stem}"]
        notes.append(f"acceptance_protocol.checks[{index}]: required_helpers missing, set to check_{stem}")
    return notes


_KINDS = {"engineering": "engineering", "hypothesis": "hypothesis", "scientific": "hypothesis",
          "research": "hypothesis", "user_required": "user_required", "user": "user_required"}
_PHASES = {"before_training": "before_training", "pre_training": "before_training", "pretraining": "before_training",
           "before": "before_training", "preflight": "before_training", "after_training": "after_training",
           "post_training": "after_training", "posttraining": "after_training", "after": "after_training"}
_BOOLS = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}


def _token(value: Any) -> str:
    return re.sub(r"[\s-]+", "_", str(value).strip().lower()) if isinstance(value, str) else ""


def normalize_plan(plan: Any, user_text: str = "") -> list[str]:
    """Correct how checks are labelled so S9 fails only on definitions that cannot run.

    Labels (kind, phase, blocking, ids, helper names) follow fixed platform rules,
    so they are set here instead of sending the whole design back. Definitions,
    missing protocols and preserved legacy gates are left to ``plan_issues``.
    Returns one note per correction, ending with ``fill_missing_helpers``.
    """
    protocol = plan.get("acceptance_protocol") if isinstance(plan, dict) else None
    if not isinstance(protocol, dict) or not isinstance(protocol.get("checks"), list):
        return []
    notes = []
    if protocol.get("version") != 1 or type(protocol.get("version")) is not int:
        notes.append(f"acceptance_protocol.version {protocol.get('version')!r} set to 1")
        protocol["version"] = 1
    ids: set[str] = set()
    for index, check in enumerate(protocol["checks"]):
        if not isinstance(check, dict):
            continue
        where = f"acceptance_protocol.checks[{index}]"

        def note(field: str, new: Any, why: str = "") -> None:
            notes.append(f"{where}.{field} {check.get(field)!r} set to {new!r}" + (f": {why}" if why else ""))
            check[field] = new

        identifier = check.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
            base = identifier.strip() if isinstance(identifier, str) and identifier.strip() else f"check_{index}"
            new, suffix = base, 2
            while new in ids:
                new, suffix = f"{base}_{suffix}", suffix + 1
            note("id", new, "ids must be unique")
        ids.add(check["id"])
        kind = check.get("kind")
        if kind not in ("engineering", "hypothesis", "user_required"):
            # An unknown kind is recorded only; it cannot become a gate by guesswork.
            note("kind", _KINDS.get(_token(kind), "hypothesis"))
            kind = check["kind"]
        phase = check.get("phase")
        if phase not in ("before_training", "after_training"):
            default = "before_training" if kind == "engineering" else "after_training"
            note("phase", _PHASES.get(_token(phase), default))
        blocking = check.get("blocking")
        if type(blocking) is not bool:
            text = _token(blocking) if isinstance(blocking, str) else str(blocking).lower()
            note("blocking", _BOOLS.get(text, kind == "engineering"))
        if kind == "hypothesis" and check["blocking"] is not False:
            note("blocking", False, "hypotheses are measured and reported, never gates")
        if (kind == "engineering" and check["phase"] == "before_training" and check["blocking"] is not True
                and not _relative_resource(check)):
            note("blocking", True, "engineering prerequisites stop training")
        if kind == "engineering" and check["phase"] == "after_training" and not _relative_resource(check):
            notes.append(f"{where}: engineering check measured after training; recorded, it cannot stop training")
        if (kind == "user_required" and check["phase"] == "before_training" and check["blocking"] is True
                and not quote_found(check.get("user_requirement_quote"), user_text)):
            note("blocking", False, "no exact user_requirement_quote, so it is measured and reported")
        helpers = check.get("required_helpers")
        if isinstance(helpers, str):
            helpers = [helpers]
        if isinstance(helpers, list) and helpers:
            fixed = []
            for name in helpers:
                stem = re.sub(r"\W+", "_", str(name)).strip("_")
                if stem and stem[0].isdigit():
                    stem = "check_" + stem
                if stem.isidentifier() and stem.isascii() and not keyword.iskeyword(stem):
                    fixed.append(stem)
            if fixed != check.get("required_helpers"):
                note("required_helpers", fixed or None, "helpers must be Python function names")
                if not fixed:
                    check.pop("required_helpers")
    return notes + fill_missing_helpers(plan)


def normalize_gpu_budget(plan: Any, config: Any) -> list[str]:
    """Replace physical GPU numbers in compute_budget with the platform allocation.

    The user's GPU numbers are a count; the platform picks the devices, so a
    plan pinned to physical indices refuses the GPUs it is actually given.
    """
    budget = plan.get("compute_budget") if isinstance(plan, dict) else None
    allocated = [item.strip() for item in
                 (getattr(getattr(config, "experiment", None), "gpu_uuid", "") or "").split(",") if item.strip()]
    if not isinstance(budget, dict) or not allocated:
        return []
    notes = []
    for key in ("authorized_gpus", "gpu_ids", "gpus"):
        value = budget.get(key)
        if isinstance(value, list) and value and all(isinstance(v, (int, str)) for v in value) and value != allocated:
            notes.append(f"compute_budget.{key} {value!r} replaced: the platform allocates "
                         f"{len(allocated)} GPU(s) via CUDA_VISIBLE_DEVICES; physical numbers are not checked")
            budget[key] = f"{len(allocated)} platform-allocated GPU(s) exposed via CUDA_VISIBLE_DEVICES"
            budget.setdefault("gpu_count", len(allocated))
    return notes


# Device identity (physical numbers, indices, UUIDs), not a memory or count measurement.
_GPU_IDENTITY = re.compile(r"physical\W+(\w+\W+){0,3}(GPU|device|ind(ex|ices))"
                           r"|GPU\w*\W+(\w+\W+){0,3}(ind(ex|ices)|uuids?)\b"
                           r"|GPUs?\s*#?\d+\s*(and|,|、|&|/)\s*\d+", re.I)


def release_gpu_identity(report: dict[str, Any], config: Any) -> bool:
    """Make a refusal of the platform-allocated GPUs a code repair, not a design failure.

    Generated code sometimes checks that the visible devices are the physical
    numbers in the user's text. The platform chooses the devices, so the fix is
    to drop that comparison and check count and memory; the plan stays frozen.
    """
    if report.get("failure_type") != "pretraining_blocked" or not _GPU_IDENTITY.search(str(report.get("reason", ""))):
        return False
    count = len([item for item in (getattr(getattr(config, "experiment", None), "gpu_uuid", "") or "").split(",")
                 if item.strip()])
    report.update(failure_type="gpu_identity_blocked", retryable=True, next_action="repair_gpu_check")
    report["errors"] = [*report.get("errors", []),
        "Platform GPU policy: the platform allocated the GPUs for this run and exposes exactly them via "
        f"CUDA_VISIBLE_DEVICES ({count or 'the inherited'} device(s)). GPU numbers in the user's text are a count, "
        "not physical devices. Remove every comparison of physical indices or UUIDs with those numbers; check "
        "only the visible device count and per-device memory, then continue. Do not change the frozen plan."]
    return True


# "每卡batch50、累积1步", "batch 25, accumulation 2", "25 per GPU with accumulation 2"
_BATCH_PAIR = (re.compile(r"batch\s*[=:]?\s*(\d+)[^\d\n]{0,30}?(?:accumulation|accum|累积)[\s:=-]*(\d+)", re.I),
               re.compile(r"(\d+)[- ]?per[- ]?(?:GPU|device)[^\d\n]{0,20}?(?:accumulation|accum)[\s:=-]*(\d+)", re.I))
_BATCH_KEYS = ("per_device_batch", "per_gpu_batch", "per_device_batch_size", "per_gpu_batch_size", "micro_batch")
_ACCUMULATION_KEYS = ("gradient_accumulation", "gradient_accumulation_steps", "accumulation", "accumulation_steps")
_OOM = re.compile(r"out of memory|\bOOM\b|OutOfMemory", re.I)


def _mapping_settings(value: Any, pairs: list[tuple[int, int]]) -> None:
    if isinstance(value, dict):
        batch = next((value[k] for k in _BATCH_KEYS if type(value.get(k)) is int), None)
        accumulation = next((value[k] for k in _ACCUMULATION_KEYS if type(value.get(k)) is int), None)
        if batch and accumulation and batch > 0 and accumulation > 0 and (batch, accumulation) not in pairs:
            pairs.append((batch, accumulation))
        for item in value.values():
            _mapping_settings(item, pairs)
    elif isinstance(value, list):
        for item in value:
            _mapping_settings(item, pairs)


def declared_batch_settings(text: str) -> list[tuple[int, int]]:
    """Distinct (per-GPU batch, accumulation) pairs declared in a plan or the user's text.

    Structured plan entries are read as YAML mappings; prose is matched by pattern.
    """
    pairs: list[tuple[int, int]] = []
    try:
        _mapping_settings(yaml.safe_load(text or ""), pairs)
    except yaml.YAMLError:
        pass
    for pattern in _BATCH_PAIR:
        for batch, accumulation in pattern.findall(text or ""):
            pair = (int(batch), int(accumulation))
            if pair[0] > 0 and pair[1] > 0 and pair not in pairs:
                pairs.append(pair)
    return pairs


def release_batch_fallback(report: dict[str, Any], plan_path: Path, run_dir: Path, config: Any) -> list[tuple[int, int]]:
    """Make an OOM block a code repair when the plan already declares a fallback batch.

    Switching to a declared setting with the same effective batch follows the
    frozen plan; redesigning from S9 would only write the same fallback again.
    Returns the declared settings, or [] when the block stays a design failure.
    """
    if report.get("failure_type") != "pretraining_blocked":
        return []
    if not _OOM.search(str(report.get("reason", "")) + " " + " ".join(map(str, report.get("errors") or []))):
        return []
    try:
        plan_text = plan_path.read_text(encoding="utf-8")
    except OSError:
        return []
    settings = declared_batch_settings(plan_text)
    settings += [pair for pair in declared_batch_settings(requirement_text(config, run_dir))
                 if pair not in settings]
    products = {batch * accumulation for batch, accumulation in settings}
    if len(settings) < 2 or len(products) != 1:
        return []
    settings.sort(key=lambda pair: -pair[0])
    listed = ", ".join(f"per-GPU batch {b} x accumulation {a}" for b, a in settings)
    report.update(failure_type="declared_fallback_oom", retryable=True, next_action="repair_use_declared_fallback",
                  declared_batch_settings=[list(pair) for pair in settings])
    report["errors"] = [*report.get("errors", []),
        f"Platform batch policy: the frozen plan declares these settings with the same effective batch: {listed}. "
        "The setting that was tried ran out of memory. Implement the declared fallback: switch every condition to "
        "the next smaller declared per-GPU batch together, record the measured peak memory of both settings in outputs/ and results.json, "
        "and continue to training. This follows the plan; do not change effective batch, learning rate, epochs or "
        "data. Report a resource blocker only if the last declared setting also runs out of memory."]
    return settings


def plan_issues(plan: Any, *, require_protocol: bool = False, user_text: str = "",
                warnings: list[str] | None = None) -> list[str]:
    """Validate definitions, not empirical pass/fail or unmeasured batch decisions.

    A user-required check that cannot stop training (nonblocking, or checked
    after training) without a matching quote is reported in ``warnings``
    instead of failing the design.
    """
    if not isinstance(plan, dict):
        return ["Experiment plan must be a mapping"]
    errors = []
    for section in ("status", "implementation_handoff", "pretraining_acceptance"):
        value = plan.get(section)
        if not isinstance(value, dict):
            continue
        if value.get("implementation_ready") is False:
            errors.append(f"{section}.implementation_ready=false: resolve definitions before S11")
        for key in ("blocking_missing_contract", "unresolved_definitions", "protocol_conflicts"):
            if value.get(key):
                errors.append(f"{section}.{key}: unresolved acceptance definitions")
    protocol = plan.get("acceptance_protocol")
    if protocol is None:
        if require_protocol:
            errors.append("Missing acceptance_protocol: classify engineering, hypothesis and user-required checks")
        return errors  # Existing plans keep all original mandatory gates; no auto-migration.
    if not isinstance(protocol, dict) or type(protocol.get("version")) is not int or protocol.get("version") != 1 or not isinstance(protocol.get("checks"), list):
        return errors + ["acceptance_protocol requires version: 1 and checks: a list"]
    ids = set()
    engineering = False
    for index, check in enumerate(protocol["checks"]):
        where = f"acceptance_protocol.checks[{index}]"
        if not isinstance(check, dict):
            errors.append(f"{where} must be a mapping")
            continue
        identifier = check.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in ids:
            errors.append(f"{where}.id must be a unique nonempty string")
        else:
            ids.add(identifier)
        kind, phase, blocking = (check.get(key) for key in ("kind", "phase", "blocking"))
        if kind not in ("engineering", "hypothesis", "user_required"):
            errors.append(f"{where}.kind is invalid")
        if phase not in ("before_training", "after_training") or type(blocking) is not bool:
            errors.append(f"{where} requires an explicit phase and boolean blocking")
        if kind == "hypothesis" and blocking is not False:
            errors.append(f"{where}: research hypotheses cannot become automatic training gates")
        if kind == "engineering":
            # An after-training engineering check (artifacts written, run completed) is
            # measured and reported; a baseline ratio is only recorded.
            if phase == "before_training" and blocking is not True and not _relative_resource(check):
                errors.append(f"{where}: engineering prerequisites must block before training")
            engineering = engineering or phase == "before_training"
        if kind == "user_required":
            if not quote_found(check.get("user_requirement_quote"), user_text):
                message = f"{where}: user-required check needs an exact user_requirement_quote from topic/details/feedback"
                # Only a blocking pretraining check can stop anything; after training the
                # check is measured and reported whatever the quote says.
                if blocking is False or phase == "after_training":
                    if warnings is not None:
                        warnings.append(message + " (cannot stop training: recorded, not a design failure)")
                else:
                    errors.append(message)
        definition = check.get("definition")
        if not isinstance(definition, dict) or any(not definition.get(key) for key in ("metric", "inputs", "aggregation", "failure_action")):
            errors.append(f"{where}.definition must specify metric, inputs, aggregation and failure_action")
        helpers = check.get("required_helpers")
        if blocking is True and phase == "before_training":
            if not isinstance(helpers, list) or not helpers or any(not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) for name in helpers):
                errors.append(f"{where}: blocking pretraining check needs required_helpers with Python function names")
    if not engineering:
        errors.append("acceptance_protocol must retain engineering prerequisites before training")
    legacy = plan.get("pretraining_acceptance")
    thresholds = legacy.get("mandatory_thresholds") if isinstance(legacy, dict) else None
    if isinstance(thresholds, dict):
        for name, original in thresholds.items():
            matching = [check for check in protocol["checks"] if isinstance(check, dict)
                        and check.get("legacy_gate") == name]
            if len(matching) != 1:
                errors.append(f"Preserved mandatory gate {name} needs exactly one explicit legacy_gate mapping")
                continue
            check = matching[0]
            definition = check.get("definition")
            if (check.get("kind") != "user_required" or check.get("phase") != "before_training"
                    or not isinstance(definition, dict) or definition.get("thresholds") != original):
                errors.append(f"Preserved mandatory gate {name} cannot change thresholds or phase")
    return errors


_PASSING = {"pass", "passed", "ok", "true", "success", "succeeded"}
_RELATIVE_RESOURCE = re.compile(r"(time|memory|mem|runtime|throughput|speed|cost|flops)[ _-]?ratio"
                                r"|relative[ _-](time|memory|runtime|throughput|cost)")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?%?")


def _quoted_numbers(text: str) -> set[float]:
    found = set()
    for token in _NUMBER.findall(text):
        number = float(token.rstrip("%"))
        found.add(number)
        if token.endswith("%"):
            found.add(number / 100)
    return found


def _numbers(value: Any) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return sorted(_quoted_numbers(value))
    if isinstance(value, dict):
        return [n for item in value.values() for n in _numbers(item)]
    if isinstance(value, list):
        return [n for item in value for n in _numbers(item)]
    return []


def _relative_resource(check: dict) -> bool:
    definition = check.get("definition") if isinstance(check.get("definition"), dict) else {}
    return bool(_RELATIVE_RESOURCE.search(f"{check.get('id', '')} {definition.get('metric', '')}".lower()))


def _user_states_numbers(check: dict, user_text: str) -> bool:
    """A numeric user gate blocks only when the user's quoted words state its numbers.

    The quote must occur in the topic or feedback, and every threshold number must
    occur in the quote: a generic quote ("keep the other gates") with numbers taken
    from S8 hypotheses or from a platform question does not qualify.
    """
    quote = check.get("user_requirement_quote")
    if not quote_found(quote, user_text):
        return False
    definition = check.get("definition") if isinstance(check.get("definition"), dict) else {}
    required = _numbers(definition.get("threshold")) + _numbers(definition.get("thresholds"))
    stated = _quoted_numbers(quote)
    return all(any(abs(n - s) <= 1e-9 * max(1.0, abs(n)) for s in stated) for n in required)


def may_block_training(check: dict, user_text: str) -> bool:
    """Whether a failed pretraining check may stop training; otherwise it is recorded."""
    if check.get("kind") == "engineering":
        # Existing plans may hold a ratio inside a blocking engineering check; it is still recorded.
        return (check.get("blocking") is not False and check.get("phase") != "after_training"
                and not _relative_resource(check))
    if check.get("kind") == "user_required" and check.get("blocking") is True:
        return not _relative_resource(check) and _user_states_numbers(check, user_text)
    return False


def _gate_key(name: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).lower()).strip("_")


def _failed(status: Any) -> bool:
    if isinstance(status, dict):
        status = status.get("status", status.get("passed", status.get("result")))
    if status is True:
        return False
    return str(status).strip().lower() not in _PASSING


def record_only_failures(plan: Any, gate_status: Any, user_text: str) -> list[str]:
    """The failed gates of a pretraining block, if every one of them is record-only.

    Returns [] when the block stands: no gate status was reported, a failed gate
    is an engineering check or a user gate stated in the user's own words, or a
    failed gate is not declared in the plan (unknown gates keep blocking). Old
    mandatory_thresholds entries without such a quote are scientific gates.
    """
    if not isinstance(plan, dict) or not isinstance(gate_status, dict):
        return []
    failed = [name for name, status in gate_status.items() if _failed(status)]
    if not failed:
        return []
    decisions: dict[str, bool] = {}
    legacy = plan.get("pretraining_acceptance")
    thresholds = legacy.get("mandatory_thresholds") if isinstance(legacy, dict) else None
    for name in thresholds if isinstance(thresholds, dict) else ():
        decisions[_gate_key(name)] = False
    protocol = plan.get("acceptance_protocol")
    checks = protocol.get("checks") if isinstance(protocol, dict) else None
    for check in checks if isinstance(checks, list) else ():
        if not isinstance(check, dict):
            continue
        blocks = may_block_training(check, user_text)
        for name in (check.get("id"), check.get("legacy_gate")):
            if name:
                decisions[_gate_key(name)] = decisions.get(_gate_key(name), False) or blocks
    if any(decisions.get(_gate_key(name), True) for name in failed):
        return []
    return failed


def release_record_only(report: dict[str, Any], plan_path: Path, run_dir: Path, config: Any) -> list[str]:
    """Turn a pretraining block whose failed gates are all record-only into a repairable failure.

    The experiment stopped before training on checks that cannot judge an untrained
    method. The measured block stays in the report; the next action is one code
    repair that records the values and continues, not a redesign from S9.
    """
    if report.get("failure_type") != "pretraining_blocked":
        return []
    try:
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return []
    names = record_only_failures(plan, report.get("required_gate_status"), requirement_text(config, run_dir))
    if names:
        report.update(failure_type="record_only_gate_blocked", retryable=True,
                      next_action="repair_record_and_continue", record_only_gates=names)
        report["errors"] = [*report.get("errors", []), record_only_instruction(names)]
    return names


def record_only_instruction(names: list[str]) -> str:
    return ("Platform acceptance policy: the failed checks " + ", ".join(map(str, names)) + " are record-only "
            "before training (scientific or mechanism checks, numeric gates not stated in the user's own "
            "words, or ratios against the baseline). Keep each check, its threshold and its measured values "
            "in outputs/ and results.json, mark it FAIL_RECORDED and continue to the formal training; do not "
            "change thresholds, delete checks or alter the method. Engineering checks still stop training.")


def check_plan_file(path: Path, run_dir: Path, config: Any, *, require_protocol: bool = False) -> list[str]:
    try:
        plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [f"Cannot read acceptance plan: {exc}"]
    return (plan_issues(plan, require_protocol=require_protocol, user_text=requirement_text(config, run_dir))
            + resource_plan_issues(plan, config))


def resource_plan_issues(plan: Any, config: Any) -> list[str]:
    """A changed GPU authorization must invalidate an old world-size/batch plan."""
    if not isinstance(plan, dict):
        return []
    allocated = getattr(getattr(config, "experiment", None), "gpu_uuid", "") or ""
    if not allocated:
        return []
    world = len([item for item in allocated.split(",") if item.strip()])
    errors = []
    for section in ("common_training", "training"):
        training = plan.get(section)
        if not isinstance(training, dict):
            continue
        declared = training.get("world_size")
        if declared is not None and declared != world:
            errors.append(f"{section}.world_size={declared} conflicts with {world} authorized GPUs; remeasure the resource plan")
        batch = training.get("per_gpu_batch_size", training.get("per_gpu_batch"))
        accumulation = training.get("gradient_accumulation_steps", training.get("accumulation"))
        effective = training.get("effective_batch_size", training.get("effective_batch"))
        if all(type(value) is int and value > 0 for value in (batch, accumulation, effective)):
            if world * batch * accumulation != effective:
                errors.append(f"{section}: {world} GPUs × batch {batch} × accumulation {accumulation} "
                              f"does not equal effective batch {effective}; preserve the budget and remeasure")
    return errors


def write_plan_blocker(stage_dir: Path, issues: list[str]) -> str:
    stage_dir.mkdir(parents=True, exist_ok=True)
    report = {"failure_type": "acceptance_definition_blocked", "retryable": False,
              "next_action": "revise_experiment_design", "errors": issues,
              "training_started": False, "gates_changed": False}
    (stage_dir / "acceptance_review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return "Acceptance protocol needs design revision (not an execution retry): " + "; ".join(issues)


def pretraining_blocker(results: Any) -> dict[str, Any] | None:
    """Classify explicit experiment refusals without ever making them successful."""
    if not isinstance(results, dict) or results.get("status") not in {
        "blocked_pretraining", "blocked_protocol", "protocol_resolution_required",
    }:
        return None
    blocker = results.get("blocker")
    blocker = blocker if isinstance(blocker, dict) else {}
    return {"failure_type": "pretraining_blocked", "retryable": False,
            "next_action": "revise_experiment_design", "status": results["status"],
            "reason": str(blocker.get("reason") or results.get("reason") or results["status"]),
            "required_gate_status": blocker.get("required_gate_status", {}),
            "message": "Training was blocked by its declared acceptance protocol. "
                       "Resolve failed checks or missing definitions in the design; "
                       "do not repeat execution or remove mandatory gates."}
