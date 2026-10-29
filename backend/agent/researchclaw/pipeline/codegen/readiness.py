"""Check explicit implementation interfaces before accepting generated code."""
from __future__ import annotations

import ast
import re
import yaml


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
    for name, source in files.items():
        if not name.endswith('.py'):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [statement for statement in node.body if not (
                    isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str))]
                if body and not all(isinstance(statement, (ast.Pass, ast.Raise)) for statement in body):
                    implemented.add(node.name)
            elif isinstance(node, ast.ImportFrom):
                implemented.update(alias.asname or alias.name for alias in node.names)
    return [f"Missing declared implementation helper: {name}" for name in sorted(required - implemented)]
