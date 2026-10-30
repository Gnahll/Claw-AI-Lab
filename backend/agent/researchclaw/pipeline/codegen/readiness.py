"""Check explicit implementation interfaces before accepting generated code."""
from __future__ import annotations

import ast
import re
import yaml


def _placeholder_reason(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    body = [s for s in node.body if not (
        isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
        and isinstance(s.value.value, str))]
    if not body or all(isinstance(s, (ast.Pass, ast.Raise)) for s in body):
        return "empty or raising stub"
    # Resolve simple local aliases, so assigning a constant report before returning
    # it is no different from returning the report directly. Do not execute code.
    literals: dict[str, object] = {}
    metadata_only = True
    for statement in body:
        if isinstance(statement, ast.Assign):
            try:
                value = ast.literal_eval(statement.value)
            except (ValueError, TypeError, SyntaxError):
                metadata_only = False
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    literals[target.id] = value
        elif not isinstance(statement, ast.Return):
            metadata_only = False
    returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
    if returns:
        constant = True
        for ret in returns:
            try:
                if isinstance(ret.value, ast.Name) and ret.value.id in literals:
                    value = literals[ret.value.id]
                else:
                    value = ast.literal_eval(ret.value)
            except (ValueError, TypeError, SyntaxError):
                constant = False
                break
        if constant and metadata_only:
            return "returns only constant metadata, not a computation"
    params = {a.arg for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
              if a.arg not in ('self', 'cls')}
    if len(body) == 1 and isinstance(body[0], ast.Return) and isinstance(body[0].value, ast.Name):
        if body[0].value.id in params:
            return "returns its input unchanged"
    # A regression seen in smoke repair: mandatory scientific checks ignored all
    # caller inputs, probed random tensors and returned unconditional PASS. An
    # asserted checker or a computed status is different and is not rejected here.
    args = params | {a.arg for a in (node.args.vararg, node.args.kwarg) if a}
    used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    fixed_pass = any(isinstance(n, ast.Dict) and any(
        isinstance(k, ast.Constant) and k.value == 'status'
        and isinstance(v, ast.Constant) and v.value == 'PASS'
        for k, v in zip(n.keys, n.values)) for n in ast.walk(node))
    guarded = any(isinstance(n, (ast.Assert, ast.If, ast.IfExp, ast.Raise)) for n in ast.walk(node))
    if args and not args.intersection(used) and fixed_pass and not guarded:
        return "ignores every input and returns unconditional PASS"
    return None


def declared_interface_gaps(files: dict[str, str], plan_text: str) -> list[str]:
    """Honor named helper contracts; an audit script is not their implementation.

    This is a structural handoff check, not proof of algorithm correctness,
    parameter updates, or scientific effectiveness. S12/S14 still execute code.
    """
    try:
        plan = yaml.safe_load(plan_text) or {}
    except yaml.YAMLError:
        return ["Experiment plan is not valid YAML"]
    if not isinstance(plan, dict):
        return ["Experiment plan must be a mapping"]
    required: set[str] = set()
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('required_distinct_helpers', 'required_helpers') and isinstance(item, list):
                    required.update(name for name in item if isinstance(name, str)
                                    and re.fullmatch(r'[A-Za-z_]\w*', name))
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(plan)
    implemented = set()
    placeholders: dict[str, str] = {}
    for name, source in files.items():
        if not name.endswith('.py'):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                reason = _placeholder_reason(node)
                if reason is None:
                    implemented.add(node.name)
                else:
                    placeholders[node.name] = reason
            elif isinstance(node, ast.ImportFrom):
                implemented.update(alias.asname or alias.name for alias in node.names)
    return [f"Missing declared implementation helper: {name}"
            + (f" ({placeholders[name]})" if name in placeholders else "")
            for name in sorted(required - implemented)]
