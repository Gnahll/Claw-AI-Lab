"""Regressions from externally imported RGB-D papers; no model/GPU/API needed."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.pipeline import executor
from researchclaw.pipeline.paper_evidence import (
    reporting_contract, snapshot_sources, validate_snapshot, emit_reporting_contract,
)
from researchclaw.pipeline.artifact_audit import manuscript_errors, audit_package
from researchclaw.templates.converter import markdown_to_latex
from researchclaw.templates.conference import NEURIPS_2025
from researchclaw.pipeline.result_analysis.runtime import ResultAnalysisRuntime
from researchclaw.pipeline import external_results


def measured():
    return {"data_split": {"val": {"scene_count": 312}}, "seed": 42, "conditions": {
        name: {"primary_metric": -ap, "optimizer_steps_completed": 11070, "epochs_completed": 5,
               "validation": {"windows": 312}, "evaluation": {"AP_percent": ap,
               "scenes": ["scene0203_00", "scene0222_00"], "geometry_label": "official-geometry-assisted"}}
        for name, ap in [("baseline", 11.732154), ("spectral", 12.630113)]}}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def registered_import(tmp_path):
    project = tmp_path / "project"
    source = project / "run-completed"
    source.mkdir(parents=True)
    (source / "fusion_adapter.py").write_text("def update(x):\n    return x + 0.01\n")
    run = project / "run-posttrain"
    write(run / "external_result_import.json", {"schema": 1, "training_allowed": False,
          "source_run": str(source), "policy": {"source_run": str(source)}})
    write(run / "stage-15/experiment_final/results.json", measured())
    (run / "stage-15/experiment_final/SOURCE.md").write_text("Only a historical pointer\n")
    return source, run


def test_legacy_external_import_gets_registered_source_not_empty_stub(tmp_path):
    source, run = registered_import(tmp_path)
    text = executor._implementation_sources(run)
    assert "def update" in text and "fusion_adapter.py" in text
    assert "registered_live_source_not_training_time_snapshot" in text
    block = executor._implementation_fidelity_block(run)
    assert "11.732154" in block and '"evaluation_scope": "subset"' in block
    assert '"validation_loss_windows": 312' in block


def test_import_snapshot_is_frozen_and_omits_outputs_and_links(tmp_path):
    source, run = registered_import(tmp_path)
    (source / "outputs").mkdir()
    (source / "outputs/junk.py").write_text("SHOULD_NOT_BE_INCLUDED=True")
    final = run / "stage-15/experiment_final"
    manifest = snapshot_sources(source, final)
    assert set(manifest["files"]) == {"fusion_adapter.py"}
    (source / "fusion_adapter.py").write_text("def update(x): return x + 999\n")
    text = executor._implementation_sources(run)
    assert "0.01" in text and "999" not in text and "import_time_snapshot" in text
    (final / "source/fusion_adapter.py").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        validate_snapshot(final)
    assert "SOURCE UNAVAILABLE" in executor._implementation_sources(run)
    assert "999" not in executor._implementation_sources(run)


def test_external_review_hands_source_and_executed_config_to_analysis(tmp_path, monkeypatch):
    source, run = registered_import(tmp_path)
    (source / "outputs/cond").mkdir(parents=True)
    (source / "outputs/cond/frozen_config.yaml").write_text("loss:\n  lambda_spec: 0.01\n")
    (source / "outputs/cond/large_weight.pt").write_bytes(b"do not copy")
    write(source / "results.json", measured())
    write(run / "stage-14/runs/results.json", measured())
    (run / "stage-14/runs/completed_run_scope.md").write_text("One completed paired run; no new training.")
    monkeypatch.setattr(external_results, "validate_import", lambda path: {
        "source_run": str(source), "artifact_check": "fixture"})
    external_results.review(run, run / "stage-15")
    text = executor._implementation_sources(run)
    assert "def update" in text and "lambda_spec: 0.01" in text
    assert not list((run / "stage-15/experiment_final").rglob("*.pt"))
    ws = ResultAnalysisRuntime._prepare_workspace(run / "stage-16", run, SimpleNamespace())
    assert (ws / "experiment_final/source/fusion_adapter.py").is_file()
    assert (ws / "experiment_final/source_manifest.json").is_file()
    reads = ResultAnalysisRuntime._build_allowed_reads(SimpleNamespace(experiment=SimpleNamespace()), run)
    assert run / "stage-15/experiment_final/source" in reads


def test_unregistered_path_is_not_used_as_writer_source(tmp_path):
    _, run = registered_import(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    write(run / "external_result_import.json", {"schema": 1, "training_allowed": False,
          "source_run": str(outside), "policy": {"source_run": str(outside)}})
    assert "SOURCE UNAVAILABLE" in executor._implementation_sources(run)


def test_negative_ap_objective_has_positive_report_and_explicit_subset():
    contract = reporting_contract(measured())
    assert contract["optimizer_objective"]["direction"] == "minimize"
    assert contract["report_metric"]["direction"] == "maximize"
    assert contract["report_metric"]["values"]["spectral"] == 12.630113
    row = contract["conditions"]["baseline"]
    assert row["scene_count"] == 2 and row["declared_validation_scenes"] == 312
    assert row["evaluation_scope"] == "subset" and row["validation_loss_windows"] == 312
    assert row["precision_recall_operating_point"] == "not_recorded"


def test_negative_generic_metric_and_invalid_ap_are_not_abs_converted():
    result = {"conditions": {"a": {"primary_metric": -.3}}}
    assert reporting_contract(result)["report_metric"] is None
    result["conditions"]["a"]["evaluation"] = {"AP_percent": float("nan")}
    assert reporting_contract(result)["report_metric"] is None
    result["conditions"]["a"]["evaluation"] = {"AP_percent": .9}
    assert reporting_contract(result)["report_metric"] is None


def test_reported_iou_range_must_be_explicit_and_fields_consistent():
    result = measured()
    assert reporting_contract(result)["report_metric"]["name"] == "AP (IoU range not recorded)"
    for row in result["conditions"].values():
        row["evaluation"]["iou_50_95"] = {"AP": row["evaluation"]["AP_percent"] / 100}
    assert reporting_contract(result)["report_metric"]["name"] == "AP[.50:.95]"
    result["conditions"]["spectral"]["evaluation"]["iou_50_95"]["AP"] = .99
    assert reporting_contract(result)["report_metric"] is None


def test_different_scene_sets_are_not_silently_declared_comparable():
    result = measured()
    result["conditions"]["spectral"]["evaluation"]["scenes"] = ["other_a", "other_b"]
    assert not reporting_contract(result)["same_reported_scene_ids"]


def test_canonical_chart_overwrites_legacy_negative_ap_chart(tmp_path):
    write(tmp_path / "stage-15/results.json", measured())
    out = tmp_path / "stage-25"
    contract = emit_reporting_contract(tmp_path, out, chart=True)
    receipt = json.loads((out / "paper_metric_chart.json").read_text())
    assert receipt["metric"] == contract["report_metric"]
    assert min(receipt["metric"]["values"].values()) > 0
    assert (out / "charts/fig_main_comparison.png").stat().st_size > 1000


def test_converter_keeps_nested_bracket_image_and_semicolon_citations():
    md = "# Test\n\n## Results\n\n![AP[.50:.95] comparison](charts/a.png)\n\n" \
         "As shown in [shuman2013emerging; ortega2018graph].\n"
    tex = markdown_to_latex(md, NEURIPS_2025, title="Test")
    assert "\\includegraphics" in tex and "![AP" not in tex
    assert "\\cite{shuman2013emerging,ortega2018graph}" in tex


def test_converter_preserves_real_caption_across_blank_line():
    md = "# Test\n\n## Method\n\n**Table 1. Executed implementation, not an ablation.**\n\n" \
         "| Component | Implementation |\n|---|---|\n| Graph | Eight-neighbor |\n"
    tex = markdown_to_latex(md, NEURIPS_2025, title="Test")
    assert "\\caption{Executed implementation, not an ablation.}" in tex
    assert "Ablation study results" not in tex
    assert tex.count("Executed implementation") == 1


def test_artifact_audit_checks_nested_image_and_dangling_reference(tmp_path):
    (tmp_path / "paper_final.md").write_text("# Test\n![AP[.50:.95]](missing.png)\nFigure 8 compares results.")
    write(tmp_path / "compile_report.json", {"success": True})
    write(tmp_path / "verification_report.json", {"summary": {"fully_verified": True}})
    (tmp_path / "paper.pdf").write_bytes(b"not used by this mechanical file-presence check")
    report = audit_package(tmp_path)
    assert not report["ready"]
    assert any("Missing figure" in x for x in report["errors"])
    assert any("Figure references" in x for x in report["errors"])
    assert not manuscript_errors("# Test\n![AP[.50:.95]](a.png)\nFigure 1 compares results.")
