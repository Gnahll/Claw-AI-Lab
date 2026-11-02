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
"""


def requirement_text(config: Any, run_dir: Path) -> str:
    research = getattr(config, "research", None)
    text = getattr(research, "prompt_topic", getattr(research, "topic", ""))
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


def plan_issues(plan: Any, *, require_protocol: bool = False, user_text: str = "") -> list[str]:
    """Validate definitions, not empirical pass/fail or unmeasured batch decisions."""
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
            engineering = True
            if phase != "before_training" or blocking is not True:
                errors.append(f"{where}: engineering prerequisites must block before training")
        if kind == "user_required":
            quote = check.get("user_requirement_quote")
            if not isinstance(quote, str) or len(quote.strip()) < 8 or quote not in user_text:
                errors.append(f"{where}: user-required check needs an exact user_requirement_quote from topic/details/feedback")
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
    if not isinstance(quote, str) or len(quote.strip()) < 8 or quote not in user_text:
        return False
    definition = check.get("definition") if isinstance(check.get("definition"), dict) else {}
    required = _numbers(definition.get("threshold")) + _numbers(definition.get("thresholds"))
    stated = _quoted_numbers(quote)
    return all(any(abs(n - s) <= 1e-9 * max(1.0, abs(n)) for s in stated) for n in required)


def may_block_training(check: dict, user_text: str) -> bool:
    """Whether a failed pretraining check may stop training; otherwise it is recorded."""
    if check.get("kind") == "engineering":
        # Existing plans may hold a ratio inside a blocking engineering check; it is still recorded.
        return check.get("blocking") is not False and not _relative_resource(check)
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
