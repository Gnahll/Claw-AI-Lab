"""Bind native short-run reuse to frozen inputs and revalidated on-disk outputs."""
from __future__ import annotations

import ast
import dataclasses
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import stat
import sys

from researchclaw.training.verified_run import file_hash, protocol_manifest

IGNORED = {".git", ".snapshots", "__pycache__"}
_MAX_INPUT_FILES = 8192
_MAX_INPUT_BYTES = 512 * 1024 * 1024


def _hash_tree(path, *, budget=None, inputs=False):
    hashes = {}
    def visit(f, relative):
        if f.is_symlink():
            # A link can resolve differently on a restarted worker.
            raise ValueError("Mutable symbolic input or evidence")
        meta = f.stat()
        if stat.S_ISDIR(meta.st_mode):
            for child in sorted(f.iterdir()):
                if inputs and (child.name in IGNORED or child.suffix == ".pyc"):
                    continue
                visit(child, child.name if relative == "." else f"{relative}/{child.name}")
        elif stat.S_ISREG(meta.st_mode):
            if budget is not None:
                budget[0] -= 1
                budget[1] -= meta.st_size
                if min(budget) < 0:
                    raise ValueError("External inputs exceed the bounded fingerprint scan")
            hashes[relative] = file_hash(f)
            after = f.stat()
            if (meta.st_size, meta.st_mtime_ns, meta.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Input changed while hashing")
        else:
            raise ValueError("Input is not a regular file")
    visit(path, ".")
    return hashes


def _config(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Unsupported configuration")


def cache_key(workspace, config, budget, plan, run_dir):
    if run_dir is None or Path(run_dir).resolve().is_relative_to(workspace.resolve()):
        return None
    try:
        limits = [_MAX_INPUT_FILES, _MAX_INPUT_BYTES]
        resources = {}
        registered = {}
        for attr, name in (("existing_project_dir", "project"), ("datasets_dir", "datasets"),
                           ("checkpoints_dir", "checkpoints"), ("codebases_dir", "codebases")):
            value = getattr(config.experiment, attr, "")
            if value:
                path = Path(value).resolve(strict=True)
                resources[str(path)] = _hash_tree(path, budget=limits, inputs=False)
                registered[name] = path
        local = {}
        for entry in sorted(workspace.iterdir()):
            if entry.name in IGNORED or entry.suffix == ".pyc" or entry.name in ("outputs", "results.json"):
                continue
            if entry.is_symlink() and entry.name in registered and entry.resolve(strict=True) == registered[entry.name]:
                local[entry.name] = str(registered[entry.name])
            else:
                local[entry.name] = _hash_tree(entry, budget=limits, inputs=True)
        # Outside resources embedded in method code or manifests must also be fingerprintable.
        # A dynamic/remote loader should mark its inputs mutable when recording the plan.
        rules = plan.get("short_training_review", {})
        if rules.get("mutable_inputs") is True:
            return None
        explicit = rules.get("input_files", [])
        if not isinstance(explicit, list):
            return None
        for name in explicit:
            if not isinstance(name, str) or "://" in name:
                return None
            path = (workspace / name).resolve(strict=True)
            resources[str(path)] = _hash_tree(path, budget=limits)
        registered_paths = tuple(Path(x) for x in resources)
        for script in workspace.rglob("*.py"):
            rel = script.relative_to(workspace)
            if any(v in IGNORED or v in ("outputs", *registered) for v in rel.parts[:-1]):
                continue
            tree = ast.parse(script.read_text(encoding="utf-8-sig"))
            for n in ast.walk(tree):
                if isinstance(n, ast.Call):
                    fn = n.func
                    name = getattr(fn, "attr", getattr(fn, "id", ""))
                    if name in {"load_dataset", "hf_hub_download", "urlopen", "download_url", "home"}:
                        return None
                    if name in {"getenv", "get"} and (name == "getenv" or
                            getattr(getattr(fn, "value", None), "attr", None) == "environ"):
                        key = n.args[0].value if n.args and isinstance(n.args[0], ast.Constant) else None
                        if key not in {"RESEARCHCLAW_PILOT_UPDATES", "WORLD_SIZE", "LOCAL_RANK", "RANK",
                                       "CUDA_VISIBLE_DEVICES", "SMOKE_TEST"}:
                            return None
                if isinstance(n, ast.Subscript) and getattr(n.value, "attr", None) == "environ":
                    if getattr(n.slice, "value", None) not in {"RESEARCHCLAW_PILOT_UPDATES", "WORLD_SIZE",
                                                              "LOCAL_RANK", "RANK", "CUDA_VISIBLE_DEVICES", "SMOKE_TEST"}:
                        return None
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    token = n.value
                    if token.startswith(("http://", "https://", "s3://", "gs://", "hf://", "~/")):
                        return None
                    if len(token) < 512 and ".." in Path(token).parts:
                        return None
                    if len(token) < 512 and Path(token).is_absolute():
                        path = Path(token).resolve()
                        if not any(path == root or path.is_relative_to(root) for root in registered_paths):
                            return None
        interpreter = getattr(getattr(config.experiment, "sandbox", None), "python_path", "")
        if interpreter and Path(interpreter).absolute() != Path(sys.executable).absolute():
            return None
        packages = {}
        for name in ("torch", "torchvision", "numpy"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
        driver = Path(__file__).parent
        data = {"schema": 1, "method": local, "resources": resources, "plan": plan,
                "protocol": protocol_manifest(run_dir), "experiment": config.experiment,
                "pilot_steps": budget, "python": [sys.executable, sys.version], "libraries": packages,
                "environment": {k: os.getenv(k) for k in ("CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "PYTHONPATH")},
                "runner": {f: file_hash(driver / f) for f in
                           ("method_review.py", "pilot_cache.py", "execution.py", "observed.py", "predictions.py")}}
        encoded = json.dumps(data, sort_keys=True, ensure_ascii=False, default=_config, allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()
    except (OSError, ValueError, SyntaxError, TypeError):
        return None


def evidence_hashes(attempt):
    execution = attempt / "execution"
    pilot = attempt / "workspace"
    return {"execution": _hash_tree(execution), "results": file_hash(pilot / "results.json"),
            "outputs": _hash_tree(pilot / "outputs")}


def find_measurement(root, key):
    for attempt in sorted(root.glob("method-pilot-*"), reverse=True):
        receipt = attempt / "measured_pilot.json"
        try:
            if attempt.is_symlink() or receipt.is_symlink() or not receipt.is_file():
                continue
            record = json.loads(receipt.read_text(encoding="utf-8"))
            if record["key"] == key and record["artifacts"] == evidence_hashes(attempt):
                path = attempt / "execution/execution_report.json"
                return attempt, json.loads(path.read_text(encoding="utf-8"))
        except (OSError, KeyError, ValueError, TypeError):
            continue
    return None
