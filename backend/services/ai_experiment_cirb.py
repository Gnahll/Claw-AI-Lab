"""Validate and apply AI-authored tensor expressions for CIRB."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

SLOTS = {
    "edge_affinity": {
        "line": "condition=(1+evidence).reciprocal()*confidence_pair",
        "default": "(1+evidence).reciprocal()*confidence_pair",
        "variables": "rgb_gap error scale evidence confidence_pair color",
        "meaning": "Compute the reliability of the depth or luminance edge cue.",
    },
    "neighbor_message": {
        "line": "evidence=evidence+neighbor*(common+condition)",
        "default": "neighbor*(common+condition)",
        "variables": "neighbor common condition low",
        "meaning": "Compute one query-conditioned contribution from its neighbor.",
    },
    "mask_delta": {
        "line": "delta=.80*x.tanh()*boundary*present",
        "default": ".80*x.tanh()*boundary",
        "variables": "x boundary reference",
        "meaning": "Compute a signed mask-logit correction. It is clipped to ±0.80 and restricted to present masks.",
    },
    "training_loss": {
        "line": "targets.append((parts*wb).sum(1).mean()+.5*(1-current.mean())+guard)",
        "default": "classification+.5*(1-current.mean())+guard",
        "variables": "classification current guard parts wb p ref near y eligible_iou",
        "meaning": "Return a scalar loss. current and ref describe corrected IoU and original probabilities respectively.",
    },
}
TORCH_CALLS = {"abs", "clamp", "exp", "log1p", "maximum", "minimum", "relu", "sigmoid", "sqrt", "square", "tanh", "where"}
METHODS = {"abs", "clamp", "clamp_min", "clamp_max", "exp", "log1p", "mean", "reciprocal", "relu", "sigmoid", "sqrt", "square", "sum", "tanh"}
BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.BitAnd, ast.BitOr)
COMPARISONS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)


def validate_expression(slot: str, expression: str, slots=SLOTS) -> str:
    if slot not in slots or not isinstance(expression, str) or not 1 <= len(expression) <= 480 or "\n" in expression:
        raise ValueError("Invalid formula or insertion point")
    tree = ast.parse(expression.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("Formula is too complex")
    variables = set(slots[slot]["variables"].split())

    def check(node):
        if isinstance(node, ast.Name) and node.id in variables:
            return
        if (isinstance(node, ast.Constant) and type(node.value) in (float, int)
                and math.isfinite(node.value) and abs(node.value) <= 100):
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
            check(node.operand)
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, BINARY):
            check(node.left)
            check(node.right)
            return
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], COMPARISONS):
            check(node.left)
            check(node.comparators[0])
            return
        if isinstance(node, ast.Call) and not node.keywords and isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and receiver.id == "torch":
                if node.func.attr not in TORCH_CALLS:
                    raise ValueError("Unsupported Torch operator")
            else:
                if node.func.attr not in METHODS:
                    raise ValueError("Unsupported tensor method")
                check(receiver)
            for argument in node.args:
                check(argument)
            return
        raise ValueError(f"Unsupported formula expression: {type(node).__name__}")

    check(tree.body)
    return ast.unparse(tree.body)


def proposal(result: dict, previous: dict, used: list[str], slots=SLOTS) -> tuple[dict, str]:
    edits = result.get("edits")
    if result.get("parent") not in {"initial", "current"} or not isinstance(edits, dict) or not 1 <= len(edits) <= 2:
        raise ValueError("Expected a parent and one or two authored formulas")
    old = previous if result["parent"] == "current" else {}
    formulas = dict(old)
    for slot, expression in edits.items():
        canonical = validate_expression(slot, expression, slots)
        default = validate_expression(slot, slots[slot]["default"], slots)
        if canonical == default:
            formulas.pop(slot, None)
        else:
            formulas[slot] = canonical
    key = hashlib.sha256(json.dumps(formulas, sort_keys=True).encode()).hexdigest()[:16]
    if not formulas or formulas == previous or key in used:
        raise ValueError("Revision is unchanged or already tested")
    return formulas, key


def verify_numerics(formulas: dict) -> None:
    # The source templates cap runtime effects. This smoke check also verifies
    # shape, finiteness, zero initialization and the optimization signal.
    import torch

    samples = torch.linspace(-1.2, 1.2, 30).view(5, 6)
    j = samples.abs()
    old = samples.clone().requires_grad_(True)
    tests = {
        "edge_affinity": ({"rgb_gap": j, "error": j, "scale": j.mean(), "evidence": j,
                           "confidence_pair": (1-j).clamp_min(0), "color": (-j).exp()}, j.shape),
        "neighbor_message": ({"neighbor": samples, "common": old, "condition": old*.3, "low": samples}, j.shape),
        "mask_delta": ({"x": old, "boundary": (1-j/1.75).clamp_min(0), "reference": samples}, j.shape),
        "training_loss": ({"classification": old.square().mean(), "current": old.mean(1).sigmoid(),
                           "guard": old.relu().mean(), "parts": old.sigmoid(), "wb": j+1,
                           "p": old.sigmoid(), "ref": samples.sigmoid(), "near": (1-j).clamp_min(0),
                           "y": (samples>0).float(), "eligible_iou": j.mean(1)}, torch.Size([])),
    }
    for slot, expr in formulas.items():
        variables, shape = tests[slot]
        value = eval(compile(expr, "<reviewed-tensor-formula>", "eval"), {"__builtins__": {}, "torch": torch}, variables)
        if not isinstance(value, torch.Tensor) or value.shape != shape or not value.is_floating_point() or not value.isfinite().all():
            raise ValueError(f"{slot} has invalid output")
        if slot != "edge_affinity":
            gradient, = torch.autograd.grad(value.sum(), old, allow_unused=True)
            if gradient is None or not gradient.isfinite().all() or not gradient.abs().any():
                raise ValueError(f"{slot} has no finite learning signal")
        if slot in ("neighbor_message", "mask_delta"):
            blank = dict(variables)
            for name in ("x", "common", "condition"):
                if name in blank:
                    blank[name] = torch.zeros_like(old)
            result = eval(compile(expr, "<zero-readout>", "eval"), {"__builtins__": {}, "torch": torch}, blank)
            if result.abs().any():
                raise ValueError(f"{slot} changes the initially frozen prediction")


def prepare(source: Path, destination: Path, revision: str, decision: dict) -> None:
    if destination.exists():
        registered = json.loads((destination / "ai_revision.json").read_text(encoding="utf-8"))
        if registered.get("revision") != revision or registered.get("source") != source.name:
            raise ValueError("Existing experiment does not match this revision")
        return
    raw = decision["formulas"]
    formulas = {slot: validate_expression(slot, expr) for slot, expr in raw.items()}
    verify_numerics(formulas)
    before = json.loads((source / "setup.json").read_text(encoding="utf-8"))
    if before.get("method_version") != "cirb_fixed_score_conditional_mask_boundary_v1":
        raise ValueError("Unsupported CIRB source experiment")
    with tempfile.TemporaryDirectory(prefix=".ai-cirb-", dir=destination.parent) as tmp:
        staged = Path(tmp) / "trial"
        staged.mkdir()
        shutil.copytree(source / "experiment", staged / "experiment",
                        ignore=shutil.ignore_patterns("outputs", "__pycache__"))
        for filename in ("dispatch.py", "supervisor.py"):
            shutil.copy2(source / filename, staged / filename)
        model = staged / "experiment" / "directed_boundary.py"
        content = model.read_text(encoding="utf-8")
        replacements = {
            "edge_affinity": lambda expr: f"condition=({expr}).clamp(0.,1.)",
            "neighbor_message": lambda expr: f"evidence=evidence+({expr})",
            "mask_delta": lambda expr: f"delta=({expr}).clamp(-.80,.80)*present",
            "training_loss": lambda expr: f"classification=(parts*wb).sum(1).mean()\n            targets.append({expr})",
        }
        for slot, expression in formulas.items():
            line = SLOTS[slot]["line"]
            if content.count(line) != 1:
                raise ValueError(f"Source insertion point changed: {slot}")
            content = content.replace(line, replacements[slot](expression))
        compile(content, str(model), "exec")
        model.write_text(content, encoding="utf-8")
        setup = dict(before)
        setup["module_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
        setup["protocol_digest"] = hashlib.sha256(
            (setup["protocol_digest"] + setup["module_sha256"] + revision).encode()
        ).hexdigest()
        setup["method_version"] = before["method_version"] + "_ai_" + revision
        (staged / "setup.json").write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding="utf-8")
        (staged / "ai_revision.json").write_text(json.dumps({
            "revision": revision, "source": source.name, "decision": decision,
            "method_sha256": setup["module_sha256"], "short_steps_per_condition": setup["target_steps"],
            "long_steps_per_condition": setup["long_steps"], "validation_scenes": len(setup["scene_ids"]),
            "gpu_uuids": setup["gpu_uuids"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        staged.rename(destination)
