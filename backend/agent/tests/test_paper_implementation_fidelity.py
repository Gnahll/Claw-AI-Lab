"""The paper writer and reviewer see the code that ran and everything that was measured."""
import json
from pathlib import Path

from researchclaw.pipeline import executor


def _project(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    code = run / "stage-15" / "experiment_final"
    code.mkdir(parents=True)
    (code / "main.py").write_text("from operator_impl import Op\nprint('train')\n", encoding="utf-8")
    (code / "operator_impl.py").write_text("TEMPERATURE = 0.25\nclass Op: heads = 4\n", encoding="utf-8")
    (code / "outputs").mkdir()
    (code / "outputs" / "junk.py").write_text("IGNORED = 1\n", encoding="utf-8")
    (run / "stage-15" / "results.json").write_text(json.dumps({
        "conditions": {"full": {"macro_f1": 0.7424}, "author_weights": {"jpeg": {"q50": 0.6912}}}}),
        encoding="utf-8")
    ws = run / "stage-14" / "run_workspace_1"
    (ws / "outputs" / "full").mkdir(parents=True)
    (ws / "outputs" / "full" / "bottom_decile_failures.json").write_text("[]", encoding="utf-8")
    (ws / "outputs" / "full" / "pred_0001.json").write_text("[]", encoding="utf-8")
    return run


def test_implementation_sources_put_operator_before_main_and_skip_outputs(tmp_path: Path) -> None:
    text = executor._implementation_sources(_project(tmp_path))
    assert text.index("operator_impl.py") < text.index("main.py")
    assert "TEMPERATURE = 0.25" in text
    assert "IGNORED" not in text


def test_result_inventory_lists_measurements_and_record_files(tmp_path: Path) -> None:
    text = executor._result_inventory(_project(tmp_path))
    assert "conditions.author_weights.jpeg.q50 = 0.6912" in text
    assert "outputs/full/bottom_decile_failures.json" in text
    assert "pred_0001" not in text


def test_fidelity_block_reaches_method_and_results_calls(tmp_path: Path) -> None:
    run = _project(tmp_path)

    class _LLM:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def chat(self, messages, **kwargs):
            from researchclaw.llm.client import LLMResponse
            self.prompts.append(messages[-1]["content"])
            return LLMResponse(content="## Title\nX", model="fake")

    llm = _LLM()
    from researchclaw.prompts import PromptManager
    executor._write_paper_sections(llm=llm, pm=PromptManager(), run_dir=run, preamble="", topic_constraint="",
                                   exp_metrics_instruction="", citation_instruction="", outline="")
    assert len(llm.prompts) == 3
    assert "IMPLEMENTATION FIDELITY" not in llm.prompts[0]
    for prompt in llm.prompts[1:]:
        assert "IMPLEMENTATION FIDELITY" in prompt and "TEMPERATURE = 0.25" in prompt
        assert "bottom_decile_failures.json" in prompt


def test_review_evidence_contains_sources_and_inventory(tmp_path: Path) -> None:
    text = executor._collect_experiment_evidence(_project(tmp_path))
    assert "operator_impl.py" in text
    assert "0.6912" in text
