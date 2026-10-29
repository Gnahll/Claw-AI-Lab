from researchclaw.pipeline.codegen.system_prompt import build_user_message, _intro_section
from researchclaw.pipeline.codegen.types import CodegenContext
from researchclaw.pipeline.experiment_run.system_prompt import build_system_prompt
from researchclaw.training.first_run import FIRST_RUN_CONTRACT


def test_execution_prompt_preserves_full_budget_and_declares_completion_artifacts():
    prompt = build_system_prompt(time_budget_sec=72 * 3600)
    assert FIRST_RUN_CONTRACT in prompt
    assert "50%" not in prompt and "Reduce training steps" not in prompt
    assert "you may reduce steps/epochs" not in prompt
    assert "workspace-root results.json" in prompt and "raw test predictions" in prompt
    assert "resumable per-condition checkpoints" in prompt


def test_codegen_does_not_assume_diffusers_when_a_model_index_is_missing():
    context = CodegenContext(topic="Image forgery localization", exp_plan="", metric="pixel_f1",
                             metric_direction="maximize", time_budget_sec=3600, mode="lab")
    prompt = build_user_message(context)
    assert "Missing model_index.json does NOT imply a Diffusers model" in prompt
    assert "use `torchvision.models` with pretrained=True" not in prompt
    assert FIRST_RUN_CONTRACT in _intro_section()
