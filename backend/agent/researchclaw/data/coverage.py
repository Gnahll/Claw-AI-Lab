"""Read-only COVERAGE audit and explicit, grouped split proposals."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_image(path: Path, *, mask: bool = False) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        converted = image.convert("L" if mask else "RGB")
        digest = hashlib.sha256()
        digest.update(f"{converted.mode}:{converted.size}".encode("ascii"))
        digest.update(converted.tobytes())
        result = {
            "size": list(converted.size),
            "file_sha256": file_sha256(path),
            "pixel_sha256": digest.hexdigest(),
        }
        if mask:
            histogram = converted.histogram()
            result["values"] = [value for value, count in enumerate(histogram) if count]
            result["foreground_pixels"] = sum(histogram[1:])
        return result


def dataset_root(root: Path) -> Path:
    root = root.expanduser().resolve(strict=True)
    if not (root / "image").is_dir() and (root / "COVERAGE" / "image").is_dir():
        root = root / "COVERAGE"
    if not (root / "image").is_dir() or not (root / "mask").is_dir():
        raise ValueError("Expected COVERAGE image/ and mask/ directories")
    return root


def _fingerprint(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_coverage(root: Path) -> dict[str, Any]:
    root = dataset_root(root)
    identifiers = set()
    for directory, suffix in (("image", "t.tif"), ("mask", "forged.tif")):
        for path in (root / directory).glob(f"*{suffix}"):
            identifier = path.name[:-len(suffix)]
            if identifier.isdecimal():
                identifiers.add(identifier)
    if not identifiers:
        raise ValueError("No COVERAGE forged image/mask pairs found")
    records = []
    excluded = []
    for identifier in sorted(identifiers, key=int):
        paths = {
            "image": f"image/{identifier}t.tif",
            "mask": f"mask/{identifier}forged.tif",
            "original": f"image/{identifier}.tif",
        }
        try:
            inspected = {}
            for role, relative_path in paths.items():
                path = _contained_path(root, relative_path)
                inspected[role] = _inspect_image(path, mask=role == "mask")
            if inspected["image"]["size"] != inspected["mask"]["size"]:
                raise ValueError(
                    f"Image/mask size mismatch: {inspected['image']['size']} "
                    f"vs {inspected['mask']['size']}"
                )
            width, height = inspected["mask"]["size"]
            foreground = inspected["mask"]["foreground_pixels"]
            if not 0 < foreground < width * height:
                raise ValueError("Forged mask has no foreground or no background")
            records.append({
                "id": identifier,
                "source_group": identifier,
                **paths,
                "files": inspected,
            })
        except (OSError, ValueError) as exc:
            excluded.append({"id": identifier, "paths": paths, "reason": str(exc)})
    return {
        "schema_version": 1,
        "dataset": "COVERAGE",
        "root": str(root),
        "mask_rule": "forged.tif; grayscale > 0 is foreground",
        "candidate_pairs": len(identifiers),
        "valid_pairs": len(records),
        "excluded_pairs": len(excluded),
        "dataset_fingerprint": _fingerprint(records),
        "records": records,
        "excluded": excluded,
        "limitations": [
            "COVERAGE contains copy-move manipulation, not all IML categories.",
            "This audit does not establish an official train/validation/test protocol.",
            "Previously evaluated samples are not an untouched test set.",
            "Pretrained model exposure to this dataset must be checked separately.",
        ],
    }


def _contained_path(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError("Dataset paths must be relative")
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes root: {relative_path}")
    return resolved


def _components(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    owners: dict[str, int] = {}
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, record in enumerate(records):
        keys = [f"group:{record['source_group']}"]
        keys.extend(
            f"pixels:{record['files'][role]['pixel_sha256']}"
            for role in ("image", "original")
        )
        for key in keys:
            if key in owners:
                parents[find(index)] = find(owners[key])
            else:
                owners[key] = index
    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(find(index), []).append(record)
    return list(grouped.values())


def propose_splits(
    audit: dict[str, Any],
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 42,
    previously_evaluated_groups: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not (
        math.isfinite(train_fraction) and math.isfinite(validation_fraction)
        and train_fraction > 0 and validation_fraction > 0
        and train_fraction + validation_fraction < 1
    ):
        raise ValueError("Train, validation and test fractions must all be positive")
    records = audit["records"]
    if audit["dataset_fingerprint"] != _fingerprint(records):
        raise ValueError("Audit fingerprint mismatch")
    components = _components(records)
    if len(components) < 3:
        raise ValueError("At least three independent source groups are required")
    exposed = set(previously_evaluated_groups)
    known = {record["source_group"] for record in records}
    if exposed - known:
        raise ValueError(f"Unknown previously evaluated groups: {sorted(exposed - known)}")
    random.Random(seed).shuffle(components)
    test_components = [
        group for group in components
        if any(record["source_group"] in exposed for record in group)
    ]
    available = [group for group in components if group not in test_components]
    train_count = max(1, int(len(components) * train_fraction))
    validation_count = max(1, int(len(components) * validation_fraction))
    if train_count + validation_count >= len(components):
        train_count = len(components) - validation_count - 1
    if train_count < 1 or len(available) < train_count + validation_count:
        raise ValueError("Not enough unexposed source groups for requested split")
    partitions = {
        "train": available[:train_count],
        "validation": available[train_count:train_count + validation_count],
        "test": test_components + available[train_count + validation_count:],
    }
    return {
        "schema_version": 1,
        "dataset": "COVERAGE",
        "root": audit["root"],
        "mask_rule": audit["mask_rule"],
        "dataset_fingerprint": audit["dataset_fingerprint"],
        "protocol": "coverage_grouped_finetuning_pilot_not_paper_reproduction",
        "protocol_status": "proposal_only",
        "seed": seed,
        "requested_fractions": {
            "train": train_fraction, "validation": validation_fraction,
            "test": 1 - train_fraction - validation_fraction,
        },
        "previously_evaluated_groups": sorted(exposed),
        "test_is_blind": False,
        "limitations": audit["limitations"],
        "splits": {
            name: sorted(
                [record for group in groups for record in group],
                key=lambda record: int(record["id"]),
            )
            for name, groups in partitions.items()
        },
    }


def validate_splits(
    manifest: dict[str, Any], *, root: Path | None = None,
    require_approval: bool = True,
) -> dict[str, int]:
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != "COVERAGE":
        raise ValueError("Unsupported dataset manifest")
    if require_approval and manifest.get("protocol_status") != "approved":
        raise ValueError("Training requires an explicitly approved split protocol")
    root = dataset_root(root or Path(manifest["root"]))
    owners: dict[str, str] = {}
    identifiers = set()
    records = []
    counts = {}
    splits = manifest["splits"]
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Exactly train, validation and test splits are required")
    for name, partition in splits.items():
        if not partition:
            raise ValueError(f"Empty {name} split")
        counts[name] = len(partition)
        for record in partition:
            if record["id"] in identifiers:
                raise ValueError(f"Duplicate sample ID: {record['id']}")
            identifiers.add(record["id"])
            keys = [f"group:{record['source_group']}"]
            if record["id"] != record["source_group"]:
                raise ValueError("COVERAGE ID must match its source group")
            for role in ("image", "mask", "original"):
                expected_path = {
                    "image": f"image/{record['id']}t.tif",
                    "mask": f"mask/{record['id']}forged.tif",
                    "original": f"image/{record['id']}.tif",
                }[role]
                if record[role] != expected_path:
                    raise ValueError(f"Unexpected COVERAGE {role} path")
                actual = _inspect_image(_contained_path(root, record[role]), mask=role == "mask")
                if actual != record["files"][role]:
                    raise ValueError(f"Dataset changed since audit: {record[role]}")
                if role != "mask":
                    keys.append(f"pixels:{actual['pixel_sha256']}")
            if record["files"]["image"]["size"] != record["files"]["mask"]["size"]:
                raise ValueError("Image/mask size mismatch")
            width, height = record["files"]["mask"]["size"]
            if not 0 < record["files"]["mask"]["foreground_pixels"] < width * height:
                raise ValueError("Invalid forged mask")
            for key in keys:
                if key in owners and owners[key] != name:
                    raise ValueError(f"Data leakage between {owners[key]} and {name}: {key}")
                owners[key] = name
            if name != "test" and record["source_group"] in manifest.get("previously_evaluated_groups", []):
                raise ValueError("Previously evaluated group appears in train/validation")
            records.append(record)
    records.sort(key=lambda record: int(record["id"]))
    if _fingerprint(records) != manifest["dataset_fingerprint"]:
        raise ValueError("Split fingerprint differs from audited dataset")
    return counts


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
