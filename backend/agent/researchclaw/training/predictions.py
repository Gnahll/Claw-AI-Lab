"""Reload saved test-set predictions and reject missing, non-finite or collapsed output.

A model that diverged or collapsed still produces a finite metric (e.g. F1 with
zero_division=1 on an all-background prediction), so the metric alone cannot show
that the model output is meaningful. This check reads the prediction files
themselves: per directory and file type, the outputs must not all be identical or
all constant (blank masks, a single predicted class).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ARRAY_SUFFIXES = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
TABLE_SUFFIXES = {".json", ".csv"}
MAX_FILES_PER_GROUP = 500


def _is_prediction(relative: Path) -> bool:
    return any("pred" in part.lower() for part in relative.parts[:-1]) or "pred" in relative.stem.lower()


def _load_array(path: Path):
    import numpy as np

    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image)


def _table_values(path: Path) -> list | None:
    """Prediction values of a JSON/CSV table, or None when no prediction column is recognisable."""
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        columns = [name for name in (rows[0] if rows else {}) if name and "pred" in name.lower()]
        return [row[columns[0]] for row in rows] if columns else None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        keys = [key for key in data if "pred" in str(key).lower()]
        data = data[keys[0]] if keys else None
    if not isinstance(data, list):
        return None
    return [json.dumps(item, sort_keys=True) for item in data]


def check_predictions(outputs_dir: Path, since: float = 0.0) -> dict:
    """Summarise prediction files under outputs/ written at or after ``since`` and flag degenerate ones."""
    import numpy as np

    groups: dict[tuple[Path, str], list[Path]] = {}
    if outputs_dir.is_dir():
        for path in sorted(outputs_dir.rglob("*")):
            suffix = path.suffix.lower()
            if (path.is_file() and not path.is_symlink() and suffix in ARRAY_SUFFIXES | TABLE_SUFFIXES
                    and _is_prediction(path.relative_to(outputs_dir)) and path.stat().st_mtime >= since):
                groups.setdefault((path.parent, suffix), []).append(path)

    report = {"groups": [], "degenerate": [], "unreadable": []}
    for (folder, suffix), paths in sorted(groups.items()):
        relative = (folder.relative_to(outputs_dir.parent) / f"*{suffix}").as_posix()
        entry = {"path": relative, "files": len(paths)}
        if suffix in TABLE_SUFFIXES:
            for path in paths:
                try:
                    values = _table_values(path)
                except (OSError, ValueError, UnicodeDecodeError) as exc:
                    report["unreadable"].append({"path": path.as_posix(), "error": str(exc)[:200]})
                    continue
                if values is None:
                    continue
                item = {"path": path.relative_to(outputs_dir.parent).as_posix(), "predictions": len(values),
                        "distinct": len(set(values))}
                item["degenerate"] = len(values) == 0 or (len(values) > 1 and item["distinct"] == 1)
                report["groups"].append(item)
                if item["degenerate"]:
                    report["degenerate"].append(item["path"])
            continue
        digests, constant, checked = set(), 0, 0
        for path in paths[:MAX_FILES_PER_GROUP]:
            try:
                values = _load_array(path)
            except Exception as exc:  # corrupt image / pickled npy
                report["unreadable"].append({"path": path.as_posix(), "error": str(exc)[:200]})
                continue
            if values.dtype.kind == "f" and not np.isfinite(values).all():
                report["unreadable"].append({"path": path.as_posix(), "error": "non-finite prediction values"})
                continue
            checked += 1
            digests.add(hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest())
            constant += int(values.size == 0 or values.min() == values.max())
        entry.update(checked=checked, distinct=len(digests), constant=constant)
        entry["degenerate"] = checked > 0 and (constant == checked or (checked > 1 and len(digests) == 1))
        report["groups"].append(entry)
        if entry["degenerate"]:
            report["degenerate"].append(relative)
    report["checked"] = sum(group.get("checked", group.get("predictions", 0)) for group in report["groups"])
    return report
