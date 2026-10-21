"""FIGURE_PROMPT rendering: API key lookup, fallback diagrams, S25 conversion.

The image model is always mocked; no paid API is called.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.agents.figure_agent import nano_banana
from researchclaw.config import RCConfig
from researchclaw.pipeline import executor as rc_executor
from researchclaw.templates import get_template, markdown_to_latex

pytest.importorskip("matplotlib")

PROMPT_BLOCK = """<!-- FIGURE_PROMPT
figure_id: cisa_full_architecture
figure_type: architecture_diagram
section: Method
caption: "CISA keeps the encoders and inserts two modules."
aspect_ratio: 16:9
prompt: |
  Draw an architecture diagram. Begin with an input labeled “RGB image: 512 × 512.”
  Connect it to an orange “LocalOrderCBFI” box, then to “Image-wise AMSA,”
  and finally to “Prediction module.”
-->

*Figure 2. CISA preserves the encoders while inserting LocalOrderCBFI and AMSA.*
"""

PAPER = (
    "# CISA: Cross-Branch Interaction for Forgery Localization\n\n"
    "## Abstract\nShort abstract.\n\n"
    "## Method\nWe describe the method.\n\n"
    + PROMPT_BLOCK
    + "\nThe module is applied after stage 4.\n\n## Conclusion\nDone.\n"
)


@pytest.fixture()
def rc_config(tmp_path: Path) -> RCConfig:
    data = {
        "project": {"name": "rc-test", "mode": "docs-first"},
        "research": {"topic": "forgery localization"},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY",
            "api_key": "inline-test-key",
            "primary_model": "fake-model",
            "fallback_models": [],
        },
        "experiment": {"mode": "sandbox"},
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


@pytest.fixture()
def adapters() -> AdapterBundle:
    return AdapterBundle()


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    return path


def _config(api_key: str = "", api_key_env: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider="openai-compatible",
            base_url="http://example.invalid/v1",
            api_key=api_key,
            api_key_env=api_key_env,
            image_base_url="",
            image_api_key="",
            image_model="fake-image-model",
        ),
        experiment=SimpleNamespace(figure_agent=None),
    )


class _FakeAgent:
    calls: list[dict] = []
    fail = False

    def __init__(self, llm, *, base_url, openai_api_key, image_model, output_dir):
        _FakeAgent.calls.append({"api_key": openai_api_key, "output_dir": output_dir})
        self.output_dir = Path(output_dir)

    def execute(self, context):
        if _FakeAgent.fail:
            raise RuntimeError("image API down")
        generated = []
        for fig in context["image_figures"]:
            path = self.output_dir / f"{fig['figure_id']}.png"
            path.write_bytes(b"\x89PNG fake")
            generated.append({"figure_id": fig["figure_id"], "output_path": str(path), "success": True})
        return SimpleNamespace(data={"generated": generated})


@pytest.fixture()
def fake_agent(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAgent]:
    _FakeAgent.calls = []
    _FakeAgent.fail = False
    monkeypatch.setattr(nano_banana, "NanoBananaAgent", _FakeAgent)
    return _FakeAgent


def _prompts() -> list[dict]:
    return rc_executor._extract_figure_prompts(PAPER)


def test_render_reads_key_from_api_key_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_agent: type[_FakeAgent]
) -> None:
    monkeypatch.setenv("RC_TEST_IMAGE_KEY", "sk-from-env")
    prompts = rc_executor._render_figure_prompts(
        _prompts(), tmp_path, _config(api_key_env="RC_TEST_IMAGE_KEY"), llm=None
    )
    assert fake_agent.calls[0]["api_key"] == "sk-from-env"
    assert prompts[0]["success"] is True
    assert Path(prompts[0]["output_path"]).name == "cisa_full_architecture.png"
    assert not prompts[0].get("fallback")


def test_render_without_key_draws_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_agent: type[_FakeAgent]
) -> None:
    monkeypatch.delenv("RC_TEST_IMAGE_KEY", raising=False)
    prompts = rc_executor._render_figure_prompts(
        _prompts(), tmp_path, _config(api_key_env="RC_TEST_IMAGE_KEY"), llm=None
    )
    assert fake_agent.calls == []
    fp = prompts[0]
    assert fp["success"] is True and fp["fallback"] is True
    out = Path(fp["output_path"])
    assert out == tmp_path / "figures" / "cisa_full_architecture_fallback.png"
    assert out.read_bytes()[:4] == b"\x89PNG"


def test_render_failure_draws_fallback(
    tmp_path: Path, fake_agent: type[_FakeAgent]
) -> None:
    fake_agent.fail = True
    prompts = rc_executor._render_figure_prompts(
        _prompts(), tmp_path, _config(api_key="sk-test"), llm=None
    )
    assert prompts[0]["fallback"] is True
    assert Path(prompts[0]["output_path"]).is_file()


def test_fallback_nodes_come_from_quoted_labels() -> None:
    nodes = rc_executor._figure_nodes(_prompts()[0])
    assert nodes == ["RGB image: 512 × 512", "LocalOrderCBFI", "Image-wise AMSA", "Prediction module"]


def _run_export(tmp_path: Path, run_dir: Path, rc_config, adapters) -> Path:
    (run_dir / "stage-22").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage-22" / "paper_revised.md").write_text(PAPER, encoding="utf-8")
    stage_dir = run_dir / "stage-25"
    stage_dir.mkdir(parents=True, exist_ok=True)
    rc_executor._execute_export_publish(stage_dir, run_dir, rc_config, adapters, llm=None)
    return stage_dir


def test_export_turns_figure_prompt_into_fallback_figure(
    tmp_path: Path, run_dir: Path, rc_config, adapters
) -> None:
    stage_dir = _run_export(tmp_path, run_dir, rc_config, adapters)

    final_md = (stage_dir / "paper_final.md").read_text(encoding="utf-8")
    assert "FIGURE_PROMPT" not in final_md
    assert "Draw an architecture diagram" not in final_md
    assert (
        "![CISA preserves the encoders while inserting LocalOrderCBFI and AMSA.]"
        "(charts/cisa_full_architecture.png)"
    ) in final_md
    assert "*Figure 2." not in final_md
    assert (stage_dir / "charts" / "cisa_full_architecture.png").is_file()
    tex = (stage_dir / "paper.tex").read_text(encoding="utf-8")
    assert "FIGURE" not in tex
    assert "Draw an architecture diagram" not in tex
    assert "charts/cisa_full_architecture.png" in tex


def test_export_prefers_rendered_revision_figure(
    tmp_path: Path, run_dir: Path, rc_config, adapters
) -> None:
    figures = run_dir / "stage-22" / "figures"
    figures.mkdir(parents=True)
    (figures / "cisa_full_architecture.png").write_bytes(b"\x89PNG rendered")
    (run_dir / "stage-22" / "figure_prompts.json").write_text(
        json.dumps([{
            "figure_id": "cisa_full_architecture",
            "output_path": str(figures / "cisa_full_architecture.png"),
            "success": True,
        }]),
        encoding="utf-8",
    )
    stage_dir = _run_export(tmp_path, run_dir, rc_config, adapters)
    assert (stage_dir / "charts" / "cisa_full_architecture.png").read_bytes() == b"\x89PNG rendered"


def test_markdown_to_latex_drops_html_comments() -> None:
    tex = markdown_to_latex(
        "# T\n\n## Intro\nText.\n\n" + PROMPT_BLOCK, get_template("neurips_2025")
    )
    assert "FIGURE" not in tex
    assert "figure\\_id" not in tex
    assert "Text." in tex
