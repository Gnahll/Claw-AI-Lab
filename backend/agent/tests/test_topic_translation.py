"""Tests for researchclaw.topic_translation — entry-point topic translation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from researchclaw.topic_translation import (
    CACHE_FILENAME,
    check_translation,
    contains_cjk,
    ensure_english_topic,
)

SOURCE = (
    "研究课题：Co-Transformers 图像篡改定位。\n"
    "使用 /data/research-resources/checkpoints/cotransformers-104.pth 作为初始权重。\n"
    "batch size 1、学习率 1e-5，划分 63/13/15，不得修改原始数据。"
)
GOOD = (
    "Research topic: Co-Transformers image manipulation localization.\n"
    "Use /data/research-resources/checkpoints/cotransformers-104.pth as the initial weights.\n"
    "Batch size 1, learning rate 1e-5, split 63/13/15; the original data must not be modified."
)


@dataclass(frozen=True)
class _Research:
    topic: str


@dataclass(frozen=True)
class _Config:
    research: _Research


@dataclass
class _Resp:
    content: str


@dataclass
class _FakeLLM:
    replies: list[str]
    prompts: list[str] = field(default_factory=list)

    def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.prompts.append(messages[-1]["content"])
        return _Resp(self.replies.pop(0))


def test_contains_cjk() -> None:
    assert contains_cjk("图像篡改")
    assert not contains_cjk("Image manipulation localization (1e-5)")


def test_good_translation_passes() -> None:
    assert check_translation(SOURCE, GOOD) == []


def test_altered_path_and_number_are_caught() -> None:
    bad = GOOD.replace("cotransformers-104.pth", "cotransformers.pth").replace("1e-5", "0.00001")
    issues = check_translation(SOURCE, bad)
    assert issues and "'1e-5'" in issues[0] and "cotransformers-104.pth" in issues[0]


def test_leftover_chinese_is_caught() -> None:
    issues = check_translation(SOURCE, GOOD + " 不得")
    assert any("CJK" in i for i in issues)


def test_summary_is_caught() -> None:
    issues = check_translation(SOURCE, "Co-Transformers /data/research-resources/checkpoints/cotransformers-104.pth 1 1e-5 63/13/15")
    assert any("merged or dropped" in i for i in issues) or any("short" in i for i in issues)


def test_plain_english_words_may_inflect() -> None:
    source = "所有对比使用相同的 epoch 数和 Seed。"
    assert check_translation(source, "All comparisons use the same number of epochs and seeds.") == []


def test_splitting_a_paragraph_into_lines_is_allowed() -> None:
    source = SOURCE.replace("\n", "")
    assert check_translation(source, GOOD) == []


def test_typeset_multiplication_sign_is_accepted() -> None:
    assert check_translation("先完成 512x512 小样本检查。", "First complete the 512×512 small-sample check.") == []
    assert check_translation("使用 512×512 输入。", "Use 512×512 inputs.") == []
    assert check_translation("使用 512×512 输入。", "Use 512x512 inputs.") == []


def test_chinese_path_may_stay_verbatim() -> None:
    source = "读取 /data/项目/说明.md 中的要求。"
    assert check_translation(source, "Read the requirements in /data/项目/说明.md.") == []


def test_english_topic_skips_llm(tmp_path: Path) -> None:
    config = _Config(_Research("Image manipulation localization"))
    llm = _FakeLLM([])
    assert ensure_english_topic(config, tmp_path, llm) is config
    assert not (tmp_path / CACHE_FILENAME).exists()


def test_translates_retries_and_caches(tmp_path: Path) -> None:
    llm = _FakeLLM(["```\n" + GOOD.replace("1e-5", "0.00001") + "\n```", GOOD])
    config = ensure_english_topic(_Config(_Research(SOURCE)), tmp_path, llm)
    assert config.research.topic == GOOD
    assert len(llm.prompts) == 2
    assert "'1e-5'" in llm.prompts[1]  # rejection reason is fed back

    record = json.loads((tmp_path / CACHE_FILENAME).read_text(encoding="utf-8"))
    assert record["status"] == "translated"
    assert record["source_topic"] == SOURCE
    assert record["attempts"] == 2

    # A resumed launch reuses the cached text without calling the LLM.
    again = ensure_english_topic(_Config(_Research(SOURCE)), tmp_path, _FakeLLM([]))
    assert again.research.topic == GOOD


def test_failed_translation_keeps_original(tmp_path: Path) -> None:
    llm = _FakeLLM(["too short"] * 3)
    config = _Config(_Research(SOURCE))
    assert ensure_english_topic(config, tmp_path, llm) is config
    record = json.loads((tmp_path / CACHE_FILENAME).read_text(encoding="utf-8"))
    assert record["status"] == "failed" and record["issues"]
