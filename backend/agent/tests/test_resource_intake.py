from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.pipeline.claw_engine.tools.executor import ToolExecutor
from researchclaw.resources.intake import build_resource_manifest


def test_resource_manifest_discovers_project_dataset_and_checkpoint(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_text("Use the official loader.", encoding="utf-8")
    (project / "train.py").write_text("import torch\n", encoding="utf-8")
    (project / "config.yaml").write_text("model: official\n", encoding="utf-8")
    dataset = tmp_path / "dataset" / "train"
    dataset.mkdir(parents=True)
    (dataset / "samples.jsonl").write_text('{"image": "a.png", "label": 1}\n', encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model_index.json").write_text(json.dumps({"_class_name": "OfficialPipeline"}), encoding="utf-8")

    manifest = build_resource_manifest(
        existing_project_dir=str(project),
        datasets_dir=str(dataset.parent),
        checkpoints_dir=str(checkpoint),
    )

    assert manifest["preflight"]["ok"]
    assert manifest["project"]["selection_mode"] == "explicit"
    assert manifest["project"]["candidates"][0]["entrypoints"] == ["train.py"]
    assert "train/samples.jsonl" in manifest["dataset"]["manifest_files"]
    assert manifest["checkpoints"]["model_indexes"][0]["json"]["_class_name"] == "OfficialPipeline"


def test_resource_manifest_blocks_ambiguous_codebases(tmp_path: Path):
    codebases = tmp_path / "codebases"
    for name in ("one", "two"):
        repo = codebases / name
        repo.mkdir(parents=True)
        (repo / "train.py").write_text("print('train')\n", encoding="utf-8")

    manifest = build_resource_manifest(codebases_dir=str(codebases))

    assert not manifest["preflight"]["ok"]
    assert any("multiple project candidates" in error for error in manifest["preflight"]["errors"])


def test_glob_search_enters_authorized_workspace_symlink(tmp_path: Path):
    workspace = tmp_path / "workspace"
    dataset = tmp_path / "dataset"
    workspace.mkdir()
    dataset.mkdir()
    (dataset / "sample.txt").write_text("real data\n", encoding="utf-8")
    link = workspace / "datasets"
    try:
        link.symlink_to(dataset, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable in this environment")

    executor = ToolExecutor(workspace=workspace, allowed_read_dirs=[dataset])
    result, is_error = executor.execute("glob_search", {"path": str(workspace), "pattern": "datasets/**/*.txt"})

    assert not is_error
    assert "sample.txt" in result


@pytest.mark.parametrize("stage_number", [11, 14])
def test_missing_resource_blocks_stage_without_llm(tmp_path: Path, stage_number: int):
    from researchclaw.pipeline.codegen.runtime import CodegenRuntime
    from researchclaw.pipeline.experiment_run.runtime import ExperimentRunRuntime
    from researchclaw.pipeline.stages import StageStatus

    config = SimpleNamespace(
        research=SimpleNamespace(topic="resource preflight"),
        experiment=SimpleNamespace(
            existing_project_dir=str(tmp_path / "missing-project"),
            codebases_dir="", datasets_dir="", checkpoints_dir="",
            metric_key="primary_metric", metric_direction="minimize",
            time_budget_sec=10, mode="sandbox",
        ),
    )
    stage_dir = tmp_path / f"stage-{stage_number:02d}"
    stage_dir.mkdir()
    runtime = CodegenRuntime() if stage_number == 11 else ExperimentRunRuntime()
    result = runtime.execute(stage_dir, tmp_path, config, None, llm=None)

    assert result.status == StageStatus.FAILED
    assert "resource preflight blocked" in result.error
    assert result.evidence_refs
    for reference in result.evidence_refs:
        assert (tmp_path / reference).is_file()


def test_symlink_expansion_helper_ignores_unconfigured_directory(tmp_path: Path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "private.txt").write_text("not authorized\n", encoding="utf-8")
    try:
        (workspace / "untrusted").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable in this environment")
    executor = ToolExecutor(workspace=workspace)
    assert executor._workspace_symlink_glob("untrusted/**/*.txt") == []
