"""Bounded discovery of existing code, data, and checkpoints."""
from __future__ import annotations
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

ENTRYPOINTS = ("train.py", "test.py", "evaluate.py", "eval.py", "inference.py", "predict.py", "main.py", "run.py", "launch.py")
READMES = ("README.md", "README.rst", "README.txt", "README")
IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CHECKPOINTS = {".ckpt", ".pt", ".pth", ".bin", ".safetensors", ".onnx", ".pkl", ".npz"}


def _path_info(raw: str) -> dict[str, Any]:
    if not raw:
        return {"configured": False, "path": "", "exists": False, "is_dir": False, "readable": False}
    try:
        path = Path(raw).expanduser().resolve()
        return {"configured": True, "path": str(path), "exists": path.exists(), "is_dir": path.is_dir(), "readable": os.access(path, os.R_OK)}
    except OSError as exc:
        return {"configured": True, "path": raw, "exists": False, "is_dir": False, "readable": False, "error": str(exc)}


def _files(root: Path, limit: int = 5000) -> list[Path]:
    if not root.is_dir():
        return []
    result: list[Path] = []
    try:
        for current, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in {".git", ".venv", "venv", "__pycache__"})
            for name in sorted(names):
                if not name.startswith("."):
                    path = Path(current) / name
                    if path.is_file():
                        result.append(path)
                        if len(result) >= limit:
                            return result
    except OSError:
        pass
    return result


def _text(path: Path, limit: int = 128 * 1024) -> str:
    try:
        return path.read_bytes()[:limit].decode("utf-8", errors="replace")
    except OSError:
        return ""


def _candidates(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    markers = {".git", "pyproject.toml", "setup.py", "requirements.txt", "environment.yml"}
    names = {p.name for p in root.iterdir() if p.is_file()}
    if markers & names or (root / ".git").exists() or any((root / n).is_file() for n in ENTRYPOINTS):
        return [root]
    result = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            child_names = {p.name for p in child.iterdir() if p.is_file()}
            if markers & child_names or (child / ".git").exists() or any((child / n).is_file() for n in ENTRYPOINTS):
                result.append(child)
    return result


def _codebase(root: Path) -> dict[str, Any]:
    files = _files(root)
    rel = [str(p.relative_to(root)).replace("\\", "/") for p in files]
    readmes = [p for p in files if p.name in READMES]
    return {
        "root": str(root.resolve()),
        "file_count_scanned": len(files),
        "files": rel[:200],
        "readmes": [str(p.relative_to(root)).replace("\\", "/") for p in readmes[:20]],
        "entrypoints": [p for p in rel if Path(p).name in ENTRYPOINTS][:50],
        "config_files": [p for p in rel if Path(p).suffix.lower() in {".yaml", ".yml", ".toml", ".json", ".cfg", ".ini"}][:100],
        "ml_source_files": [p for p in rel if Path(p).suffix == ".py" and any(x in _text(root / p, 16000) for x in ("torch", "tensorflow", "transformers", "diffusers"))][:100],
        "readme_samples": {str(p.relative_to(root)).replace("\\", "/"): _text(p, 6000) for p in readmes[:3]},
    }


def _dataset(root: Path) -> dict[str, Any]:
    files = _files(root)
    splits: dict[str, int] = {}
    for path in files:
        parts = {part.lower() for part in path.relative_to(root).parts}
        for split in ("train", "val", "valid", "test", "validation"):
            if split in parts:
                splits[split] = splits.get(split, 0) + 1
    manifests: list[str] = []
    samples: dict[str, str] = {}
    for path in files:
        if path.suffix.lower() in {".csv", ".json", ".jsonl", ".txt", ".tsv"}:
            name = str(path.relative_to(root)).replace("\\", "/")
            manifests.append(name)
            if len(samples) < 5:
                samples[name] = _text(path, 4000)
    images = [p for p in files if p.suffix.lower() in IMAGES]
    masks = [p for p in images if any(x in p.stem.lower() for x in ("mask", "label", "seg"))]
    return {
        "root": str(root.resolve()), "file_count_scanned": len(files),
        "extension_counts": dict(Counter(p.suffix.lower() or "<none>" for p in files)),
        "split_file_counts": splits, "image_file_count": len(images),
        "mask_like_file_count": len(masks), "manifest_files": manifests[:100],
        "manifest_samples": samples,
        "sample_files": [{"path": str(p.relative_to(root)).replace("\\", "/"), "size_bytes": p.stat().st_size, "suffix": p.suffix.lower()} for p in files[:20]],
    }


def _checkpoint(root: Path) -> dict[str, Any]:
    files = _files(root)
    artifacts = []
    indexes = []
    for path in files:
        rel = str(path.relative_to(root)).replace("\\", "/")
        if path.suffix.lower() in CHECKPOINTS or path.name in {"model_index.json", "config.json", "tokenizer_config.json"}:
            artifacts.append({"path": rel, "size_bytes": path.stat().st_size, "suffix": path.suffix.lower()})
        if path.name == "model_index.json":
            try:
                indexes.append({"path": rel, "json": json.loads(_text(path, 32 * 1024))})
            except json.JSONDecodeError:
                indexes.append({"path": rel, "parse_error": True})
    return {"root": str(root.resolve()), "file_count_scanned": len(files), "artifact_files": artifacts[:200], "model_indexes": indexes[:20]}


def build_resource_manifest(*, existing_project_dir: str = "", codebases_dir: str = "", datasets_dir: str = "", checkpoints_dir: str = "") -> dict[str, Any]:
    """Build a bounded, JSON-serializable inventory of configured resources."""
    paths = {name: _path_info(value) for name, value in (("existing_project_dir", existing_project_dir), ("codebases_dir", codebases_dir), ("datasets_dir", datasets_dir), ("checkpoints_dir", checkpoints_dir))}
    project_root = Path(existing_project_dir).expanduser().resolve() if existing_project_dir and Path(existing_project_dir).is_dir() else None
    codebase_root = Path(codebases_dir).expanduser().resolve() if codebases_dir and Path(codebases_dir).is_dir() else None
    candidates = [project_root] if project_root else (_candidates(codebase_root) if codebase_root else [])
    selected = project_root or (candidates[0] if len(candidates) == 1 else None)
    errors = []
    for name, info in paths.items():
        if info["configured"] and not (info["exists"] and info["is_dir"] and info["readable"]):
            errors.append(f"{name} is not an existing readable directory: {info['path']}")
    if paths["existing_project_dir"]["configured"] and selected is None:
        errors.append("existing_project_dir has no recognizable project entrypoints or repository markers")
    if paths["codebases_dir"]["configured"] and not candidates:
        errors.append("codebases_dir contains no recognizable project candidate")
    if paths["codebases_dir"]["configured"] and not project_root and len(candidates) > 1:
        errors.append("codebases_dir contains multiple project candidates; set experiment.existing_project_dir explicitly")
    dataset_root = Path(datasets_dir).expanduser().resolve() if datasets_dir and Path(datasets_dir).is_dir() else None
    checkpoint_root = Path(checkpoints_dir).expanduser().resolve() if checkpoints_dir and Path(checkpoints_dir).is_dir() else None
    if dataset_root and not _files(dataset_root, 1):
        errors.append("datasets_dir is readable but contains no readable files")
    if checkpoint_root and not _files(checkpoint_root, 1):
        errors.append("checkpoints_dir is readable but contains no readable files")
    if selected and not _files(selected, 1):
        errors.append("selected existing project contains no readable files")
    manifest = {
        "schema_version": 1, "resource_type": "existing_project_intake", "paths": paths,
        "project": {"selected_root": str(selected) if selected else "", "selection_mode": "explicit" if project_root else ("single_candidate" if selected else "needs_selection"), "candidates": [_codebase(p) for p in candidates[:20]]},
        "dataset": _dataset(dataset_root) if dataset_root else {},
        "checkpoints": _checkpoint(checkpoint_root) if checkpoint_root else {},
        "preflight": {"required": any(info["configured"] for info in paths.values()), "ok": not errors, "errors": errors, "readable_sample_count": len(_files(dataset_root, 20)) if dataset_root else 0},
    }
    manifest["manifest_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    return manifest


def render_resource_manifest(manifest: dict[str, Any]) -> str:
    """Render the inventory as concrete instructions for the coding agent."""
    paths, project, preflight = manifest.get("paths", {}), manifest.get("project", {}), manifest.get("preflight", {})
    lines = ["# Existing project resource contract", "", f"Manifest ID: `{manifest.get('manifest_id', '')}`", f"Preflight: **{'PASS' if preflight.get('ok') else 'BLOCKED'}**"]
    lines.extend(f"- BLOCKER: {error}" for error in preflight.get("errors", []))
    lines.append("\n## Source paths")
    for name, info in paths.items():
        if info.get("configured"):
            lines.append(f"- `{name}`: `{info.get('path', '')}` (exists={info.get('exists')}, readable={info.get('readable')})")
    lines.extend(["\n## Project reuse rule", f"- Selected project root: `{project.get('selected_root', '') or 'none'}`", f"- Selection mode: `{project.get('selection_mode', 'none')}`", "- Reuse the selected project's loaders, model construction, evaluation, and configuration before adding new code.", "- Do not create a replacement model or synthetic dataset when a configured resource exists."])
    for candidate in project.get("candidates", []):
        lines.extend([f"\n### Candidate `{candidate.get('root', '')}`", f"- Entrypoints: {', '.join(candidate.get('entrypoints', [])) or 'none detected'}", f"- Configs: {', '.join(candidate.get('config_files', [])[:20]) or 'none detected'}", f"- ML source files: {', '.join(candidate.get('ml_source_files', [])[:20]) or 'none detected'}"])
        for name, sample in candidate.get("readme_samples", {}).items():
            lines.append(f"- README sample `{name}`:\n```text\n{sample[:2500]}\n```")
    dataset = manifest.get("dataset", {})
    if dataset:
        lines.extend(["\n## Dataset facts", f"- Scanned files: `{dataset.get('file_count_scanned', 0)}`; images: `{dataset.get('image_file_count', 0)}`; mask-like files: `{dataset.get('mask_like_file_count', 0)}`", f"- Split counts: `{json.dumps(dataset.get('split_file_counts', {}), sort_keys=True)}`", f"- Manifest files: {', '.join(dataset.get('manifest_files', [])[:30]) or 'none detected'}"])
        for name, sample in dataset.get("manifest_samples", {}).items():
            lines.append(f"- Sample `{name}`:\n```text\n{sample[:2000]}\n```")
    checkpoints = manifest.get("checkpoints", {})
    if checkpoints:
        lines.append("\n## Checkpoint facts")
        lines.append(f"- Artifact files: {', '.join(item.get('path', '') for item in checkpoints.get('artifact_files', [])[:30]) or 'none detected'}")
        for item in checkpoints.get("model_indexes", []):
            lines.append(f"- model index `{item.get('path', '')}`: `{json.dumps(item.get('json', item), sort_keys=True)[:3000]}`")
    lines.extend(["\n## Mandatory preflight", "Before writing experiment code, read this contract and at least one README/config/entrypoint from the selected project, plus one dataset manifest or sample and checkpoint metadata when configured.", "Record concrete files and APIs reused in PROJECT_CONTEXT.json. If a required resource is unavailable, stop with an explicit blocker instead of fabricating an experiment."])
    return "\n".join(lines) + "\n"
