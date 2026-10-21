"""Title rules: no generic 'An Empirical Study of...' style titles."""

from __future__ import annotations

import inspect
import re

import pytest

from researchclaw import prompts as rc_prompts
from researchclaw.pipeline import executor as rc_executor

# Phrasings that used to *recommend* generic frames.
_OLD_RECOMMENDATIONS = [
    r"use a title like\s*['\"]?An Empirical Study",
    r"Use 'Toward\.\.\.', 'Investigating\.\.\.', or",
    r"Frame as 'An Empirical Study of",
]


@pytest.mark.parametrize("module", [rc_prompts, rc_executor])
def test_no_prompt_recommends_generic_title(module) -> None:
    src = re.sub(r'"\s*\n\s*"', "", inspect.getsource(module))
    for pat in _OLD_RECOMMENDATIONS:
        assert not re.search(pat, src), pat


def test_title_guidelines_require_method_name() -> None:
    block = rc_prompts._DEFAULT_BLOCKS["title_guidelines"]
    assert "REQUIRED structure: 'MethodName: Descriptive Phrase'" in block
    assert "also after a 'MethodName:' prefix" in block
    for frame in ("An Empirical Study of", "Towards", "Investigating", "On the"):
        assert frame in block


def test_negative_result_advisory_keeps_method_title() -> None:
    src = inspect.getsource(rc_executor)
    assert "Keep a method/mechanism title" in src
    assert "State the negative result" in src


@pytest.mark.parametrize(
    "title",
    [
        "CISA: An Empirical Study of Cross-Branch Interaction for Forgery Localization",
        "Towards Robust Cross-Branch Interaction in Forgery Localization",
        "Investigating Adaptive Multi-Scale Attention for Forgery Localization",
        "On the Role of Cross-Branch Features in Forgery Localization",
        "When Cross-Branch Gating Falls Short: Negative Results",
        "AMSA: Toward Adaptive Multi-Scale Attention",
    ],
)
def test_generic_title_frame_detected(title: str) -> None:
    assert rc_executor._generic_title_frame(title)


@pytest.mark.parametrize(
    "title",
    [
        "CISA: A Cross-Branch Interaction Mechanism for Forgery Localization",
        "AlphaEdit: Null-Space Knowledge Editing for LMs",
        "Vision Transformers Need Registers",
        "Ontology-Guided Fusion for Forgery Localization",
    ],
)
def test_method_title_not_flagged(title: str) -> None:
    assert rc_executor._generic_title_frame(title) == ""


def _draft(title: str) -> str:
    return f"# {title}\n\n# Abstract\nWe study things.\n\n# Method\nText.\n"


def test_validator_emits_retitle_directive() -> None:
    result = rc_executor._validate_draft_quality(
        _draft("CISA: An Empirical Study of Cross-Branch Interaction")
    )
    assert any("Generic title frame" in w for w in result["overall_warnings"])
    assert any(d.startswith("RETITLE") for d in result["revision_directives"])


def test_validator_accepts_mechanism_title() -> None:
    result = rc_executor._validate_draft_quality(
        _draft("CISA: A Cross-Branch Interaction Mechanism for Forgery Localization")
    )
    assert not any("Generic title frame" in w for w in result["overall_warnings"])
    assert not any(d.startswith("RETITLE") for d in result["revision_directives"])
