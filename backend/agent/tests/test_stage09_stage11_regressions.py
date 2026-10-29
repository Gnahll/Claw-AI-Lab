"""Regression checks for the S9 fallback and S11 nested generated files."""

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock
from dataclasses import replace

import pytest
import yaml

from researchclaw.pipeline.codegen.registry import StrategyRegistry
from researchclaw.pipeline.codegen.runtime import CodegenRuntime, _discover_data
from researchclaw.pipeline.codegen.types import CodegenContext, CodegenResult
from researchclaw.pipeline.executor import _execute_experiment_design
from researchclaw.pipeline.stages import StageStatus
from researchclaw.config import RCConfig
from .test_rc_config import _valid_config_data


@pytest.fixture
def rc_config(tmp_path):
    return RCConfig.from_dict(_valid_config_data(), project_root=tmp_path, check_paths=False)


def _complete_sections(plan):
    return {'objectives':['Compare the complete method fairly'], 'metrics':['pixel_f1'],
            'risks':['Numerical instability'], 'compute_budget':{'training_conditions':2}, **plan}


def _design_config(rc_config, tmp_path, *, fixed_local):
    for name in ('data', 'code'):
        (tmp_path / name).mkdir(exist_ok=True)
    return replace(rc_config, research=replace(rc_config.research, topic='Image manipulation localization'),
        experiment=replace(rc_config.experiment, mode='sandbox',
            datasets_dir=str(tmp_path / 'data') if fixed_local else '',
            codebases_dir=str(tmp_path / 'code') if fixed_local else '',
            benchmark_agent=replace(rc_config.experiment.benchmark_agent, enabled=True)))


def test_fixed_local_experiment_keeps_authored_baseline_spec_and_skips_recommender(tmp_path, rc_config, monkeypatch):
    from researchclaw.pipeline import executor
    from researchclaw.agents import benchmark_agent
    plan=_complete_sections({'baselines':[{'name':'original','implementation_spec':{'loss_function':'author loss'}}],
          'proposed_methods':[{'name':'joint D1 D2'}], 'datasets':['approved local data'], 'ablations':[]})
    monkeypatch.setattr(executor, '_build_context_preamble', lambda *a, **k:'')
    monkeypatch.setattr(executor, '_chat_with_prompt', lambda *a, **k:SimpleNamespace(content=yaml.safe_dump(plan)))
    called=[]
    monkeypatch.setattr(benchmark_agent, 'BenchmarkOrchestrator', lambda *a, **k:called.append(True))
    stage=tmp_path/'stage-09';stage.mkdir()
    result=executor._execute_experiment_design(stage,tmp_path,_design_config(rc_config,tmp_path,fixed_local=True),object(),llm=object())
    saved=yaml.safe_load((stage/'exp_plan.yaml').read_text())
    assert result.status==StageStatus.DONE and called==[]
    assert saved['baselines']==plan['baselines'] and saved['ablations']==[]


def test_authored_plan_survives_interrupted_optional_recommender(tmp_path, rc_config, monkeypatch):
    from researchclaw.pipeline import executor
    from researchclaw.agents import benchmark_agent
    plan=_complete_sections({'baselines':[{'name':'original','implementation_spec':{'loss_function':'author loss'}}],
          'proposed_methods':['complete'], 'datasets':['specified'], 'ablations':[]})
    monkeypatch.setattr(executor, '_build_context_preamble', lambda *a, **k:'')
    monkeypatch.setattr(executor, '_chat_with_prompt', lambda *a, **k:SimpleNamespace(content=yaml.safe_dump(plan)))
    class InterruptedRecommender:
        def __init__(self,*a,**k):pass
        def orchestrate(self,*a,**k):raise KeyboardInterrupt('Optional selection interrupted')
    monkeypatch.setattr(benchmark_agent, 'BenchmarkOrchestrator', InterruptedRecommender)
    stage=tmp_path/'stage-09';stage.mkdir()
    with pytest.raises(KeyboardInterrupt):
        executor._execute_experiment_design(stage,tmp_path,_design_config(rc_config,tmp_path,fixed_local=False),object(),llm=object())
    assert yaml.safe_load((stage/'exp_plan.yaml').read_text())['baselines']==plan['baselines']


def test_s9_invalid_reply_does_not_pass_with_placeholder_plan(tmp_path, rc_config, monkeypatch):
    from researchclaw.pipeline import executor
    monkeypatch.setattr(executor, '_build_context_preamble', lambda *a, **k:'')
    calls=[]
    def invalid(*args,**kwargs):
        calls.append((args,kwargs))
        return SimpleNamespace(content='I will prepare a plan.',finish_reason='length',truncated=True,completion_tokens=4096)
    monkeypatch.setattr(executor,'_chat_with_prompt',invalid)
    stage=tmp_path/'stage-09';stage.mkdir()
    result=executor._execute_experiment_design(stage,tmp_path,_design_config(rc_config,tmp_path,fixed_local=True),object(),llm=object())
    assert result.status==StageStatus.FAILED and result.decision=='retry'
    assert not (stage/'exp_plan.yaml').exists()
    assert 'placeholder' in result.error
    assert (stage/'plan_generation_response.json').is_file()
    assert (stage/'plan_generation_retry_response.json').is_file()
    assert calls[0][1]['max_tokens']==16384 and calls[1][1]['max_tokens']>=16384
    assert 'Image manipulation localization' in calls[1][0][2]


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


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "gbk", "gb18030"])
def test_s11_discovers_non_utf8_optional_context(tmp_path, encoding):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    codebase_dir = tmp_path / "codebase"
    codebase_dir.mkdir()
    text = "图像篡改定位数据集说明\n" + "\n".join(f"row {index}" for index in range(6))
    if encoding == "gb18030":
        text = "训练记录 😀\n" + text
    (dataset_dir / "readme.txt").write_bytes(text.encode(encoding))
    (codebase_dir / "README.md").write_bytes(text.encode(encoding))
    (codebase_dir / "train.py").write_text("print('train')\n", encoding="utf-8")
    session = Mock()

    discovered = _discover_data("", str(dataset_dir), str(codebase_dir), session)

    assert discovered.dataset_sample == "Sample from readme.txt:\n" + "\n".join(text.splitlines()[:5])
    assert discovered.codebase_readme == text
    assert discovered.codebase_files == ["train.py"]
    if encoding in ("gbk", "gb18030"):
        assert session.log.call_count == 2
        assert all("gb18030" in call.args[1] for call in session.log.call_args_list)
    else:
        session.log.assert_not_called()


@pytest.mark.parametrize("content", [b"\xff", b"binary\x00content"])
def test_s11_skips_unreadable_optional_context_and_keeps_resources(tmp_path, content):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    codebase_dir = tmp_path / "codebase"
    codebase_dir.mkdir()
    (dataset_dir / "readme.txt").write_bytes(content)
    (dataset_dir / "image.tif").write_bytes(b"image")
    (codebase_dir / "README.md").write_bytes(content)
    (codebase_dir / "train.py").write_text("print('train')\n", encoding="utf-8")
    nested_dir = codebase_dir / "reference"
    nested_dir.mkdir()
    (nested_dir / "README.md").write_text("Usable nested README", encoding="utf-8")
    session = Mock()

    discovered = _discover_data("", str(dataset_dir), str(codebase_dir), session)

    assert discovered.dataset_sample == ""
    assert "image.tif" in discovered.dataset_files
    assert discovered.codebase_files == ["train.py"]
    assert discovered.codebase_readme == "Usable nested README"
    assert session.log.call_count == 2
    assert all("Skipped optional context file" in call.args[1] for call in session.log.call_args_list)


def test_s11_logs_optional_context_io_errors(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    sample = dataset_dir / "readme.txt"
    sample.write_text("sample", encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def read_bytes(path):
        if path == sample:
            raise PermissionError("sample access denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    session = Mock()

    discovered = _discover_data("", str(dataset_dir), "", session)

    assert discovered.dataset_sample == ""
    assert discovered.dataset_files == ["readme.txt"]
    session.log.assert_called_once()
    assert "sample access denied" in session.log.call_args.args[1]


def test_s11_persists_optional_context_encoding_warning(tmp_path):
    from researchclaw.pipeline.codegen.session import CodegenSession

    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    sample = dataset_dir / "readme.txt"
    sample.write_bytes("数据集说明".encode("gbk"))
    stage_dir = tmp_path / "stage-11"
    stage_dir.mkdir()

    discovered = _discover_data("", str(dataset_dir), "", CodegenSession(stage_dir))

    assert discovered.dataset_sample == "Sample from readme.txt:\n数据集说明"
    live_log = (stage_dir / "codegen_live.log").read_text(encoding="utf-8")
    assert str(sample) in live_log
    assert "gb18030" in live_log
    assert (stage_dir / "codegen_session.json").is_file()
