import copy
import json
from pathlib import Path

import pytest

from researchclaw.data.coverage import (
    audit_coverage, propose_splits, validate_splits, write_report,
)

Image = pytest.importorskip("PIL.Image")


def make_pair(root: Path, identifier: int, *, mask_size=(8, 8)) -> None:
    (root / "image").mkdir(parents=True, exist_ok=True)
    (root / "mask").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (identifier, 30, 40)).save(root / "image" / f"{identifier}t.tif")
    Image.new("RGB", (8, 8), (identifier, 50, 60)).save(root / "image" / f"{identifier}.tif")
    mask = Image.new("L", mask_size, 0)
    mask.putpixel((1, 1), 255)
    mask.save(root / "mask" / f"{identifier}forged.tif")


@pytest.fixture
def coverage(tmp_path):
    for identifier in range(1, 11):
        make_pair(tmp_path, identifier)
    return tmp_path


def test_audit_nested_root_and_relative_paths(coverage):
    parent = coverage / "outer"
    nested = parent / "COVERAGE"
    make_pair(nested, 12)
    report = audit_coverage(parent)
    assert report["valid_pairs"] == 1
    assert report["records"][0]["image"] == "image/12t.tif"
    assert report["records"][0]["files"]["mask"]["foreground_pixels"] == 1


def test_audit_reports_bad_pairs_without_modifying_data(coverage):
    make_pair(coverage, 11, mask_size=(7, 8))
    make_pair(coverage, 12)
    (coverage / "mask" / "12forged.tif").write_bytes(b"not a TIFF")
    before = {str(path): path.read_bytes() for path in coverage.rglob("*.tif")}
    report = audit_coverage(coverage)
    assert report["candidate_pairs"] == 12
    assert report["valid_pairs"] == 10
    assert report["excluded_pairs"] == 2
    assert "size mismatch" in report["excluded"][0]["reason"]
    assert before == {str(path): path.read_bytes() for path in coverage.rglob("*.tif")}


@pytest.mark.parametrize("foreground", [0, 255])
def test_rejects_empty_or_full_mask(coverage, foreground):
    Image.new("L", (8, 8), foreground).save(coverage / "mask" / "1forged.tif")
    report = audit_coverage(coverage)
    assert report["excluded"][0]["id"] == "1"
    assert report["valid_pairs"] == 9


def test_split_is_reproducible_proposal_not_training_permission(coverage):
    audit = audit_coverage(coverage)
    proposal = propose_splits(audit)
    assert proposal == propose_splits(audit)
    assert proposal != propose_splits(audit, seed=123)
    assert validate_splits(proposal, require_approval=False) == {"train": 7, "validation": 1, "test": 2}
    with pytest.raises(ValueError, match="explicitly approved"):
        validate_splits(proposal)
    proposal["protocol_status"] = "approved"
    assert validate_splits(proposal)["train"] == 7


def test_original_content_duplicates_stay_in_one_split(coverage):
    source = coverage / "image" / "1.tif"
    target = coverage / "image" / "2.tif"
    target.write_bytes(source.read_bytes())
    proposal = propose_splits(audit_coverage(coverage))
    owners = {record["id"]: name for name, records in proposal["splits"].items() for record in records}
    assert owners["1"] == owners["2"]
    assert validate_splits(proposal, require_approval=False)


def test_transitive_content_duplicates_stay_together(coverage):
    (coverage / "image" / "2.tif").write_bytes((coverage / "image" / "1.tif").read_bytes())
    (coverage / "image" / "3t.tif").write_bytes((coverage / "image" / "2t.tif").read_bytes())
    proposal = propose_splits(audit_coverage(coverage))
    owners = {record["id"]: name for name, records in proposal["splits"].items() for record in records}
    assert owners["1"] == owners["2"] == owners["3"]


def test_prior_evaluation_groups_not_used_for_training(coverage):
    proposal = propose_splits(audit_coverage(coverage), previously_evaluated_groups=("1",))
    assert any(record["id"] == "1" for record in proposal["splits"]["test"])
    assert not proposal["test_is_blind"]
    assert validate_splits(proposal, require_approval=False)


def test_duplicate_content_leakage_detected_after_repartition(coverage):
    (coverage / "image" / "2.tif").write_bytes((coverage / "image" / "1.tif").read_bytes())
    proposal = propose_splits(audit_coverage(coverage))
    source = next(name for name, records in proposal["splits"].items() if any(record["id"] == "2" for record in records))
    target = next(name for name in proposal["splits"] if name != source)
    record = next(record for record in proposal["splits"][source] if record["id"] == "2")
    proposal["splits"][source].remove(record)
    proposal["splits"][target].append(record)
    with pytest.raises(ValueError, match="Data leakage"):
        validate_splits(proposal, require_approval=False)


def test_file_changes_after_audit_rejected(coverage):
    proposal = propose_splits(audit_coverage(coverage))
    Image.new("RGB", (8, 8), (77, 22, 33)).save(coverage / "image" / "1t.tif")
    with pytest.raises(ValueError, match="changed since audit"):
        validate_splits(proposal, require_approval=False)


def test_removed_records_cannot_silently_change_audited_dataset(coverage):
    proposal = propose_splits(audit_coverage(coverage))
    proposal["splits"]["train"].pop()
    with pytest.raises(ValueError, match="fingerprint"):
        validate_splits(proposal, require_approval=False)


def test_changed_path_and_duplicate_ids_rejected(coverage):
    proposal = propose_splits(audit_coverage(coverage))
    invalid = copy.deepcopy(proposal)
    invalid["splits"]["train"][0]["image"] = "../outside.tif"
    with pytest.raises(ValueError, match="Unexpected"):
        validate_splits(invalid, require_approval=False)
    invalid = copy.deepcopy(proposal)
    invalid["splits"]["test"].append(invalid["splits"]["train"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_splits(invalid, require_approval=False)


@pytest.mark.parametrize("train,validation", [(0, .2), (.8, .3), (float("nan"), .1), (.7, float("inf"))])
def test_invalid_fractions_rejected(coverage, train, validation):
    with pytest.raises(ValueError, match="positive"):
        propose_splits(audit_coverage(coverage), train_fraction=train, validation_fraction=validation)


def test_insufficient_groups_and_unknown_prior_exposure_rejected(coverage):
    with pytest.raises(ValueError, match="Unknown"):
        propose_splits(audit_coverage(coverage), previously_evaluated_groups=("999",))
    with pytest.raises(ValueError, match="Not enough"):
        propose_splits(audit_coverage(coverage), previously_evaluated_groups=tuple(str(identifier) for identifier in range(1, 10)))
    small = coverage / "small"
    make_pair(small, 1)
    make_pair(small, 2)
    with pytest.raises(ValueError, match="three"):
        propose_splits(audit_coverage(small))


def test_report_refuses_overwrite(tmp_path):
    destination = tmp_path / "audit.json"
    write_report(destination, {"valid_pairs": 1})
    with pytest.raises(FileExistsError):
        write_report(destination, {"valid_pairs": 2})
    assert json.loads(destination.read_text())["valid_pairs"] == 1


def test_cli_audit_and_proposal_validation(coverage, capsys):
    from researchclaw.cli import main

    output = coverage / "splits.json"
    assert main(["dataset-split", "--root", str(coverage), "--output", str(output)]) == 0
    assert main(["dataset-check", "--manifest", str(output), "--proposal"]) == 0
    assert main(["dataset-check", "--manifest", str(output)]) == 1
    assert "explicitly approved" in capsys.readouterr().err
