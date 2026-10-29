"""Design-time acceptance definitions, separate from measured research outcomes."""
from __future__ import annotations

import hashlib
import json
import keyword
from pathlib import Path
from typing import Any

import yaml


ACCEPTANCE_POLICY = """
ACCEPTANCE PROTOCOL (platform default for first-time users):
Separate engineering prerequisites from scientific hypotheses. Before training, verify
real resources/data separation, shapes/alignment, finite forward/backward, nonzero new
module gradients, probability constraints when applicable, and memory/time feasibility.
A predicted AUROC, expected superiority, or expected signal separability is a hypothesis,
not a default prerequisite for training. Record and evaluate these at the declared phase;
negative results are valid. Do not invent an untrained quality threshold that prevents
the experiment intended to test that hypothesis. Preserve EVERY explicit user-mandated
gate, including its metric, threshold, cohort and timing; never silently downgrade it.

In the plan include acceptance_protocol: {version: 1, checks: [...]}.
Each check has id, kind (engineering | hypothesis | user_required), phase
(before_training | after_training), blocking (a YAML boolean), and definition (a mapping).
definition MUST specify metric, inputs, aggregation, failure_action; a numerical rule
also needs its threshold. A blocking before_training check MUST additionally declare
required_helpers: [Python function names] so S11 must implement its actual checker.
Hypothesis checks are nonblocking. A user_required check MUST contain user_requirement_quote,
an exact quotation of the user's requirement, not a claim of user approval invented by you.
Scientific mandatory gates already present in a preserved protocol remain mandatory even
if they fail. Engineering checks remain mandatory and run before training.
When preserving pretraining_acceptance.mandatory_thresholds, map each entry to a
blocking before_training user_required check via legacy_gate: <original key> and
definition.thresholds: <the entire unchanged threshold mapping>. Do not keep two
contradictory acceptance tables or quietly drop an existing mandatory entry.

Define candidate-specific operators, position alignment, references and cohorts BEFORE
measuring; do not transplant another method's diagnostic solely because its name matches.
Resolve definition/precision conflicts during design. Unmeasured memory/batch decisions
may wait for engineering preflight, but missing metric definitions or an implementation_ready
false plan cannot be handed to code generation as complete. A smoke pass proves only its
measured engineering scope, never all scientific prerequisites or completed training.
When blocked, write workspace-root results.json with status=blocked_pretraining, reason
and required_gate_status in blocker, and measured evidence paths. Stop without modifying
the frozen protocol. Do not remove mandatory checks or label this as successful training.
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


def registered_context(run_dir: Path, *, max_chars: int = 60000) -> str:
    """Read authoritative definitions for plain-chat S9; never truncate requirements."""
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
    return text


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
                    or check.get("blocking") is not True or not isinstance(definition, dict)
                    or definition.get("thresholds") != original):
                errors.append(f"Preserved mandatory gate {name} cannot change thresholds, phase or blocking behavior")
    return errors


def check_plan_file(path: Path, run_dir: Path, config: Any, *, require_protocol: bool = False) -> list[str]:
    try:
        plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [f"Cannot read acceptance plan: {exc}"]
    return plan_issues(plan, require_protocol=require_protocol, user_text=requirement_text(config, run_dir))


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
