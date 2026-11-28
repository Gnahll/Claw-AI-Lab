from pathlib import Path
import json

import pytest

from researchclaw.training import managed_review as gate
from researchclaw.training.jobs import file_sha256, write_json


def fixture(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    folder = job / "long_review" / "attempt-1"
    folder.mkdir(parents=True)
    write_json(job / "provenance.json", {"source": "fixed"})
    values = {"before": .3, "after": .4}
    write_json(folder / "validation.json", {name: {"records": [
        {"id": "v", "pixel_f1": v, "pixel_iou": v}]} for name, v in values.items()})
    checkpoint = folder / "checkpoint-2-0-2.pt"
    checkpoint.write_bytes(b"checkpoint from the pilot")
    proof = {"optimizer_steps": 2, "selected_checkpoint": {"filename": checkpoint.name, "sha256": file_sha256(checkpoint)},
             "initial_parameter_sha256": "a" * 64, "trained_parameter_sha256": "b" * 64,
             "gpu_evidence": {"gpu_uuid": "GPU-selected", "memory_mib": 500}}
    write_json(folder / "pilot.json", {"training": proof, "provenance_sha256": file_sha256(job / "provenance.json"),
                                      "validation_sha256": file_sha256(folder / "validation.json")})
    (folder / "events.jsonl").write_text("".join(json.dumps(dict(event="optimizer_step", global_step=n,
        loss=.3, sample_id="t"))+"\n" for n in (1, 2)))
    spec = {"epochs": 2, "steps_per_epoch": 3, "pilot_steps": 2, "training_phase": "long"}
    splits = {"splits": {"train": [{"id": "t"}], "validation": [{"id": "v"}], "test": [{"id": "unseen"}]}}
    write_json(folder.parent / "pilot_run.json", {"directory": folder.name})
    return job, folder, spec, splits


def test_budget_and_classification():
    assert gate.needs_review({"epochs": 1, "steps_per_epoch": 3}) is False
    assert gate.needs_review({"training_phase": "short", "epochs": 100, "steps_per_epoch": 20})
    assert gate.pilot_budget({"training_phase": "long", "epochs": 10, "steps_per_epoch": 10, "pilot_steps": 25}) == (2, 10)
    with pytest.raises(ValueError):
        gate.pilot_budget({"epochs": 1, "steps_per_epoch": 1})


def test_declared_threshold_and_tampering(tmp_path):
    job, folder, spec, splits = fixture(tmp_path)
    assert gate.review_evidence(job, folder, spec, splits)["pixel_f1_gain"] == pytest.approx(.1)
    with pytest.raises(ValueError, match="required"):
        gate.review_evidence(job, folder, dict(spec, min_validation_f1_gain=.2), splits)
    (folder / "checkpoint-2-0-2.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        gate.review_evidence(job, folder, spec, splits)


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_retry_reuses_real_evidence_and_decision(tmp_path, monkeypatch, decision):
    job, folder, spec, splits = fixture(tmp_path)
    monkeypatch.setattr(gate, "verify_inputs", lambda _: (spec, splits))
    def reviewer(data):
        assert data["validation_count"] == 1 and data["optimizer_steps"] == 2
        return {"decision": decision, "reason": "paired validation", "model": "service"}
    monkeypatch.setattr(gate, "ask_reviewer", reviewer)
    for _ in range(2):
        if decision == "approve":
            gate.review(job, "GPU-selected")
        else:
            with pytest.raises(RuntimeError, match="declined"):
                gate.review(job, "GPU-selected")
        monkeypatch.setattr(gate, "ask_reviewer", lambda _: pytest.fail("review was already recorded"))
