"""Regression checks for the S9 fallback and S11 nested generated files."""

from types import SimpleNamespace

import yaml

from researchclaw.pipeline.codegen.registry import StrategyRegistry
from researchclaw.pipeline.codegen.runtime import CodegenRuntime
from researchclaw.pipeline.codegen.types import CodegenContext, CodegenResult
from researchclaw.pipeline.executor import _execute_experiment_design
from researchclaw.pipeline.stages import StageStatus


def test_s9_no_llm_with_local_dataset(tmp_path, monkeypatch):
    from researchclaw.pipeline import executor

    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    stage_dir = tmp_path / "stage-09"
    stage_dir.mkdir()
    monkeypatch.setattr(executor, "_build_context_preamble", lambda *args, **kwargs: "")
    config = SimpleNamespace(
        research=SimpleNamespace(topic="image localization", domains=()),
        experiment=SimpleNamespace(
            datasets_dir=str(dataset_dir), codebases_dir="", checkpoints_dir="",
            metric_key="iou", mode="sandbox",
        ),
    )

    result = _execute_experiment_design(stage_dir, tmp_path, config, object(), llm=None)

    plan = yaml.safe_load((stage_dir / "exp_plan.yaml").read_text(encoding="utf-8"))
    assert result.status == StageStatus.DONE
    assert plan["local_resources"]["datasets_dir"] == str(dataset_dir)


def test_s11_writes_nested_generated_files(tmp_path, monkeypatch):
    class NestedFileStrategy:
        name = "test"

        def can_handle(self, ctx, config):
            return True

        def generate(self, ctx, config, llm, session, prompts=None):
            return CodegenResult(
                strategy_name=self.name,
                files={"main.py": "print(1)\n", "model/network.py": "class Net: pass\n"},
            )

    registry = StrategyRegistry()
    registry.register(NestedFileStrategy())
    runtime = CodegenRuntime(registry)
    stage_dir = tmp_path / "stage-11"
    stage_dir.mkdir()
    config = SimpleNamespace(
        llm=SimpleNamespace(primary_model="test", coding_model=""),
        experiment=SimpleNamespace(time_budget_sec=60, max_iterations=1),
    )
    monkeypatch.setattr(
        runtime, "_build_context",
        lambda *args: CodegenContext(
            topic="test", exp_plan="", metric="accuracy",
            metric_direction="maximize", time_budget_sec=60, mode="sandbox",
        ),
    )

    result = runtime.execute(stage_dir, tmp_path, config, object())

    assert result.status == StageStatus.DONE
    assert (stage_dir / "experiment/model/network.py").read_text(encoding="utf-8") == "class Net: pass\n"
