"""Translate a non-English research topic into English once, at pipeline entry.

The pipeline's prompts, keyword extraction and literature search all assume
English input.  A Chinese topic makes every stage reinterpret it on its own
(and several ASCII-only parsers silently drop it), so the topic is translated
a single time, checked deterministically, and cached in the run directory so
resumed runs reuse the exact same text.  The original topic is archived in the
cache file only; it is never fed back into stage prompts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_FILENAME = "topic_translation.json"
MAX_ATTEMPTS = 3

# CJK ideographs plus CJK/full-width punctuation (a translation must not keep "。", "：" etc.).
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")
# Filesystem paths may legitimately contain CJK characters; they must survive verbatim.
_PATH_RE = re.compile(r"(?<![\w/])(?:~|\.{1,2})?/[^\s，。；：、！？“”‘’（）()\[\]{}<>,;\"'`]+")
_CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.+\-/:]*[A-Za-z0-9])?")

_SYSTEM_PROMPT = (
    "You are a professional translator for machine-learning research task "
    "specifications. You translate faithfully and never summarize."
)

_USER_PROMPT = """\
Translate the research task below into English. The translation becomes the
official task definition for an automated research pipeline, so it must carry
exactly the same meaning and the same requirements as the original.

Rules:
1. Translate sentence by sentence. Do not summarize, merge, drop, reorder or add anything.
2. Keep the original line structure: one output line for every input line, blank lines included.
3. Prohibitions and hard requirements keep their strength: "不得/不能/不要/禁止/严禁" become
   "must not" / "do not"; "必须/须/要求" become "must". Never soften them to "should" or "avoid".
4. Copy verbatim, character for character: file and directory paths, file names, URLs,
   code identifiers, model/dataset/method names, numbers, hyperparameters, units, version
   strings and IDs (e.g. GPU UUIDs). Paths stay unchanged even if they contain Chinese characters.
5. Use the standard English terminology of the research field.
6. Output only the translated text: no preface, no notes, no code fences.
{feedback}
Original task:
<<<
{topic}
>>>"""


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _protected_tokens(source: str) -> list[str]:
    """Spans of *source* that must appear unchanged in the translation."""
    text = unicodedata.normalize("NFKC", source)
    tokens: list[str] = []
    paths = _PATH_RE.findall(text)
    tokens.extend(p.rstrip(".") for p in paths)
    tokens.extend(m.strip() for m in _CODE_SPAN_RE.findall(text))
    # Strip paths first so their ASCII fragments are not counted twice.
    remainder = _PATH_RE.sub(" ", text)
    tokens.extend(_ASCII_TOKEN_RE.findall(remainder))
    return list(dict.fromkeys(t for t in tokens if t))


def _token_present(token: str, text: str) -> bool:
    if token.isalpha():
        # Plain English words embedded in the source may change case or inflect
        # in an English sentence ("epoch" -> "epochs"); only identifiers must be exact.
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}", text, re.IGNORECASE) is not None
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", text) is not None


def _nonempty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def check_translation(source: str, translated: str) -> list[str]:
    """Return a list of problems; an empty list means the translation passed."""
    issues: list[str] = []
    translated_n = unicodedata.normalize("NFKC", translated)
    if not translated_n.strip():
        return ["translation is empty"]

    # "512x512" and "512×512" are the same token; translators switch between them.
    comparable = translated_n.replace("×", "x")
    missing = [t for t in _protected_tokens(source.replace("×", "x")) if not _token_present(t, comparable)]
    if missing:
        shown = ", ".join(repr(t) for t in missing[:15])
        issues.append(f"{len(missing)} verbatim token(s) missing or altered: {shown}")

    leftover = _CJK_RE.findall(_PATH_RE.sub(" ", translated_n))
    if leftover:
        issues.append(f"untranslated CJK characters remain: {''.join(leftover[:20])!r}")

    # Splitting a long paragraph into one sentence per line is harmless; losing
    # lines means items were merged or dropped.
    src_lines, dst_lines = _nonempty_lines(source), _nonempty_lines(translated_n)
    if src_lines - dst_lines > max(2, src_lines // 10):
        issues.append(f"lines were merged or dropped: {src_lines} non-empty source lines vs {dst_lines} translated")

    # English text is longer than the Chinese it translates; a much shorter
    # result means content was summarized away.
    if len(translated_n) < 0.8 * len(source):
        issues.append(f"translation is suspiciously short ({len(translated_n)} vs {len(source)} chars)")
    return issues


def _strip_wrapping(text: str) -> str:
    text = text.strip()
    fence = re.fullmatch(r"```[\w-]*\n(.*?)\n?```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if text.startswith("<<<") and text.endswith(">>>"):
        text = text[3:-3].strip()
    return text


def translate_topic(source: str, llm: Any) -> tuple[str | None, list[str], int]:
    """Translate *source* with retries. Returns (translation or None, last issues, attempts)."""
    issues: list[str] = []
    max_tokens = min(16000, max(2048, len(source) * 4))
    for attempt in range(1, MAX_ATTEMPTS + 1):
        feedback = ""
        if issues:
            feedback = (
                "\nA previous translation was rejected for these reasons; fix them:\n- "
                + "\n- ".join(issues)
                + "\n"
            )
        prompt = _USER_PROMPT.format(topic=source, feedback=feedback)
        try:
            resp = llm.chat(
                [{"role": "user", "content": prompt}],
                system=_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=0.0,
                strip_thinking=True,
            )
        except Exception as exc:  # noqa: BLE001
            issues = [f"LLM call failed: {exc}"]
            logger.warning("Topic translation attempt %d failed: %s", attempt, exc)
            continue
        translated = _strip_wrapping(resp.content or "")
        issues = check_translation(source, translated)
        if not issues:
            return translated, [], attempt
        logger.warning("Topic translation attempt %d rejected: %s", attempt, "; ".join(issues))
    return None, issues, MAX_ATTEMPTS


def _read_cache(path: Path, source_hash: str) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and data.get("source_sha256") == source_hash and data.get("status") == "translated":
        return data
    return None


def ensure_english_topic(config: Any, cache_dir: Path, llm: Any = None) -> Any:
    """Return *config* with ``research.topic`` replaced by its English translation.

    English topics pass through untouched.  The translation is cached in
    ``cache_dir/topic_translation.json`` keyed by the source text hash, so every
    later launch for the same run reuses it.  If no translation passes the
    checks, the original topic is kept and a warning is printed.
    """
    source = config.research.topic
    if not contains_cjk(source):
        return config

    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cache_path = Path(cache_dir) / CACHE_FILENAME
    cached = _read_cache(cache_path, source_hash)
    if cached is not None:
        translated = cached["topic"]
        print(f"Topic translation: reusing {cache_path}")
    else:
        if llm is None:
            from researchclaw.llm import create_llm_client

            llm = create_llm_client(config)
        print("Topic translation: topic contains Chinese, translating to English...", flush=True)
        translated, issues, attempts = translate_topic(source, llm)
        record = {
            "status": "translated" if translated is not None else "failed",
            "source_sha256": source_hash,
            "source_topic": source,
            "topic": translated,
            "issues": issues,
            "attempts": attempts,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        if translated is None:
            print(
                f"WARNING: topic translation failed after {attempts} attempts "
                f"({'; '.join(issues)}). Continuing with the original topic. See {cache_path}",
                flush=True,
            )
            return config
        print(f"Topic translation: done in {attempts} attempt(s), saved to {cache_path}")

    preview = translated if len(translated) <= 400 else translated[:400] + " ..."
    print(f"  English topic: {preview}\n", flush=True)
    research = dataclasses.replace(config.research, topic=translated)
    return dataclasses.replace(config, research=research)
