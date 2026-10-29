#!/usr/bin/env python3
"""
L1 Discussion Runner — Multi-round LLM debate between agent perspectives.

Collects synthesis.md from multiple L1 agents who independently researched the
same topic, runs a structured multi-round debate via the LLM, and produces a
consensus synthesis that enriches all agents' context for hypothesis generation.

Usage:
    python discussion_runner.py \
        --config /path/to/config.arc.yaml \
        --synthesis-dirs /path/to/agent-a/stage-07 /path/to/agent-b/stage-07 \
        --output /path/to/discussion_output \
        --rounds 3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

AGENT_PACKAGE_DIR = str(Path(__file__).resolve().parent.parent / "agent")
if AGENT_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, AGENT_PACKAGE_DIR)

from researchclaw.config import RCConfig
from researchclaw.llm import create_llm_client
from researchclaw.topic_translation import ensure_english_topic


def _read_synthesis(stage_dir: Path) -> str | None:
    for name in ("synthesis.md", "synthesis.txt"):
        f = stage_dir / name
        if f.is_file():
            return f.read_text(encoding="utf-8")
    return None


def _label(index: int) -> str:
    return chr(ord("A") + index)


SYSTEM_PRESENT = """\
You are a senior research coordinator facilitating a multi-agent research discussion.
Multiple research agents have independently studied the same topic and produced their
own literature syntheses. Your task is to analyze each agent's perspective and identify:
1. Key findings unique to each perspective
2. Common themes across perspectives
3. Knowledge gaps that no perspective addressed
4. Contradictions or disagreements between perspectives

Be specific and cite which perspective (Agent A, Agent B, etc.) each point comes from.
Respond in the same language as the syntheses (Chinese if they are in Chinese)."""

SYSTEM_CRITIQUE = """\
You are a critical research reviewer. Given the initial analysis of multiple research
perspectives, your task is to:
1. Evaluate the strength of evidence for each key finding
2. Identify potential biases in individual perspectives
3. Find complementary findings that could be combined for stronger conclusions
4. Highlight the most promising research directions that emerge from combining perspectives
5. Note any methodological differences that explain contradictions

Be rigorous and constructive. Respond in the same language as the input."""

SYSTEM_CONSENSUS = """\
You are a research synthesis expert. Based on the multi-round discussion of independent
research perspectives, produce a unified consensus synthesis that:
1. Integrates the strongest findings from all perspectives
2. Resolves contradictions with reasoned explanations
3. Preserves novel insights that appeared in only one perspective
4. Identifies the most promising hypotheses suggested by the combined evidence
5. Notes remaining uncertainties and open questions

Format the output as a well-structured markdown document with clear sections.
This consensus will be used by all agents to generate research hypotheses.
Respond in the same language as the input."""


SYSTEM_CONDENSE = """\
You condense part of a research agent's literature synthesis so it can join a
multi-agent discussion under a strict input limit. Keep the research direction, the
proposed methods and hypotheses, key findings with their cited papers, datasets,
metrics, numbers, and stated gaps or risks. Drop repetition, generic background and
formatting. Output only the condensed markdown, in the same language as the input."""

# The model gateway rejects long requests with HTTP 500 (2026-10-19: ~15k chars of
# system + user prompt passed, ~27k failed), so every call is kept under this budget.
DEFAULT_MAX_PROMPT_CHARS = 14000
CALL_ATTEMPTS = 3
RETRY_DELAY_SEC = 30
TRUNCATION_NOTE = "\n\n[... truncated to fit the model input limit ...]"


def _fit(text: str, limit: int) -> str:
    """Cut text to at most ``limit`` chars, preferably at a line boundary."""
    if len(text) <= limit:
        return text
    cut = text[:max(0, limit - len(TRUNCATION_NOTE))]
    newline = cut.rfind("\n")
    if newline > len(cut) * 0.6:
        cut = cut[:newline]
    return cut + TRUNCATION_NOTE


def _split(text: str, size: int) -> list[str]:
    """Split text into chunks of at most ``size`` chars along paragraph boundaries."""
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        while len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:size])
            paragraph = paragraph[size:]
        if current and len(current) + 2 + len(paragraph) > size:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


class Discussion:
    """LLM calls of one discussion: prompt budget, retries, progress reporting."""

    def __init__(self, llm, max_prompt_chars: int, progress=None, retry_delay: float = RETRY_DELAY_SEC):
        self.llm = llm
        self.max_prompt_chars = max_prompt_chars
        self.progress = progress or (lambda step: None)
        self.retry_delay = retry_delay

    def chat(self, step: str, system: str, user: str, max_tokens: int) -> str:
        if len(system) + len(user) > self.max_prompt_chars:
            raise ValueError(f"{step}: prompt of {len(system) + len(user)} chars exceeds the "
                             f"{self.max_prompt_chars}-char budget")
        self.progress(step)
        for attempt in range(1, CALL_ATTEMPTS + 1):
            started = time.time()
            try:
                resp = self.llm.chat([{"role": "user", "content": user}], system=system, max_tokens=max_tokens)
            except Exception as exc:
                if attempt == CALL_ATTEMPTS:
                    raise RuntimeError(f"{step} failed after {CALL_ATTEMPTS} attempts: {exc}") from exc
                delay = self.retry_delay * attempt
                print(f"    {step}: attempt {attempt}/{CALL_ATTEMPTS} failed ({exc}); retrying in {delay:.0f}s",
                      flush=True)
                time.sleep(delay)
                continue
            if not (resp.content or "").strip():
                raise RuntimeError(f"{step}: model returned an empty response")
            print(f"    Done ({resp.total_tokens} tokens, {len(system) + len(user)} prompt chars, "
                  f"{time.time() - started:.0f}s)", flush=True)
            return resp.content
        raise AssertionError("unreachable")

    def condense(self, label: str, text: str, limit: int) -> str:
        """Shrink one synthesis to at most ``limit`` chars, chunk by chunk with the LLM."""
        chunk_size = self.max_prompt_chars - len(SYSTEM_CONDENSE) - 300
        for _ in range(3):
            if len(text) <= limit:
                return text
            chunks = _split(text, chunk_size)
            target = max(400, limit // len(chunks) - 50)
            parts = []
            for index, chunk in enumerate(chunks, 1):
                user = (f"Condense the following part {index}/{len(chunks)} of {label}'s synthesis "
                        f"to at most {target} characters.\n\n{chunk}")
                parts.append(self.chat(f"Condensing {label} ({index}/{len(chunks)})", SYSTEM_CONDENSE, user,
                                       max_tokens=min(4096, max(512, target))))
            text = "\n\n".join(parts)
        return _fit(text, limit)


def run_discussion(
    llm,
    topic: str,
    syntheses: dict[str, str],
    num_rounds: int,
    output_dir: Path,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    progress=None,
    retry_delay: float = RETRY_DELAY_SEC,
) -> tuple[str, str]:
    """Run the multi-round discussion and return (transcript, consensus)."""
    talk = Discussion(llm, max_prompt_chars, progress, retry_delay)
    transcript_parts: list[str] = []
    agent_labels = list(syntheses.keys())
    budget = max_prompt_chars - max(len(SYSTEM_PRESENT), len(SYSTEM_CRITIQUE), len(SYSTEM_CONSENSUS)) - 400
    # User-provided execution requirements must not disappear at a topic slice.
    block_limit = budget - len(topic) - 1400
    share = block_limit // max(1, len(syntheses)) - 40
    if share < 500:
        raise ValueError("Research topic/details exceed the discussion context budget; shorten them explicitly")
    condensed = {}
    for label, text in syntheses.items():
        condensed[label] = talk.condense(label, text, share)
        if condensed[label] != text:
            print(f"   ↓ {label}: condensed {len(text)} → {len(condensed[label])} chars", flush=True)
    if condensed != syntheses:
        (output_dir / "condensed_syntheses.md").write_text(
            "\n\n".join(f"## {label}\n\n{text}" for label, text in condensed.items()), encoding="utf-8")

    perspectives_block = "\n\n".join(
        f"## {label}\n\n{text}" for label, text in condensed.items()
    )

    def room(system: str, *fixed: str, parts: int = 1) -> int:
        """Chars left for each of ``parts`` earlier-round texts in one prompt."""
        return (max_prompt_chars - len(system) - sum(map(len, fixed)) - 300) // parts

    # ── Round 1: Present and analyze perspectives ──
    print(f"  [Round 1/{num_rounds}] Presenting perspectives...", flush=True)
    user_r1 = (
        f"Research topic: {topic}\n\n"
        f"The following are independent literature syntheses from {len(syntheses)} "
        f"research agents who studied this topic independently:\n\n"
        f"{perspectives_block}\n\n"
        f"Please analyze these perspectives following your instructions."
    )
    analysis = talk.chat(f"Round 1/{num_rounds}", SYSTEM_PRESENT, user_r1, max_tokens=4096)
    transcript_parts.append(f"# Discussion Transcript\n\nTopic: {topic}\n")
    transcript_parts.append(f"Participants: {', '.join(agent_labels)}\n")
    transcript_parts.append(f"## Round 1: Perspective Analysis\n\n{analysis}\n")

    # ── Round 2: Critical review ──
    if num_rounds >= 2:
        print(f"  [Round 2/{num_rounds}] Critical review...", flush=True)
        prior = _fit(analysis, room(SYSTEM_CRITIQUE, topic, perspectives_block))
        user_r2 = (
            f"Research topic: {topic}\n\n"
            f"Original perspectives:\n\n{perspectives_block}\n\n"
            f"---\n\nInitial analysis:\n\n{prior}\n\n"
            f"Please provide your critical review following your instructions."
        )
        critique = talk.chat(f"Round 2/{num_rounds}", SYSTEM_CRITIQUE, user_r2, max_tokens=4096)
        transcript_parts.append(f"## Round 2: Critical Review\n\n{critique}\n")
    else:
        critique = analysis

    # ── Round 3+: Consensus synthesis ──
    if num_rounds >= 3:
        print(f"  [Round 3/{num_rounds}] Building consensus...", flush=True)
        limit = room(SYSTEM_CONSENSUS, topic, perspectives_block, parts=2)
        user_r3 = (
            f"Research topic: {topic}\n\n"
            f"Original perspectives:\n\n{perspectives_block}\n\n"
            f"---\n\nPerspective analysis:\n\n{_fit(analysis, limit)}\n\n"
            f"---\n\nCritical review:\n\n{_fit(critique, limit)}\n\n"
            f"Please produce the consensus synthesis following your instructions."
        )
        consensus = talk.chat(f"Round 3/{num_rounds}", SYSTEM_CONSENSUS, user_r3, max_tokens=8192)
        transcript_parts.append(f"## Round 3: Consensus Synthesis\n\n{consensus}\n")
    else:
        print("  [Consensus] Building consensus synthesis...", flush=True)
        consensus_prompt = (
            f"Research topic: {topic}\n\n"
            f"Perspectives:\n\n{perspectives_block}\n\n"
            f"Discussion so far:\n\n{_fit(critique, room(SYSTEM_CONSENSUS, topic, perspectives_block))}\n\n"
            f"Please produce a unified consensus synthesis."
        )
        consensus = talk.chat("Consensus", SYSTEM_CONSENSUS, consensus_prompt, max_tokens=8192)
        transcript_parts.append(f"## Consensus\n\n{consensus}\n")

    transcript = "\n---\n\n".join(transcript_parts)
    return transcript, consensus


def main():
    parser = argparse.ArgumentParser(description="L1 Discussion Runner")
    parser.add_argument("--config", required=True, help="Path to config.arc.yaml")
    parser.add_argument(
        "--synthesis-dirs", nargs="+", required=True,
        help="Paths to stage-07 dirs containing synthesis.md",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--rounds", type=int, default=3, help="Number of discussion rounds")
    parser.add_argument("--topic", default="", help="Research topic override")
    parser.add_argument("--max-prompt-chars", type=int, default=DEFAULT_MAX_PROMPT_CHARS,
                        help="Upper bound on system + user prompt characters of every LLM call")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write heartbeat so the bridge can track progress
    heartbeat = {"status": "starting", "started_at": int(time.time() * 1000)}

    def beat(**fields):
        heartbeat.update(fields, updated_at=int(time.time() * 1000))
        (output_dir / "heartbeat.json").write_text(
            json.dumps(heartbeat, ensure_ascii=False), encoding="utf-8",
        )

    beat()
    try:
        _run(args, output_dir, beat)
    except BaseException as exc:
        if heartbeat.get("status") != "failed":
            beat(status="failed", error=f"{type(exc).__name__}: {exc}"[:1000])
        raise


def _run(args, output_dir: Path, beat) -> None:
    print(f"🦞 Discussion Runner starting")
    print(f"   Config:     {args.config}")
    print(f"   Dirs:       {args.synthesis_dirs}")
    print(f"   Rounds:     {args.rounds}")
    print(f"   Output:     {args.output}")
    print(f"   Max prompt: {args.max_prompt_chars} chars")

    # Load config and create LLM client
    config = RCConfig.load(args.config, check_paths=False)
    llm = create_llm_client(config)

    if args.topic:
        config = dataclasses.replace(config, research=dataclasses.replace(config.research, topic=args.topic))
    config = ensure_english_topic(config, output_dir, llm)
    topic = getattr(config.research, "prompt_topic", config.research.topic)
    print(f"   Topic:      {topic}")

    # Collect syntheses
    syntheses: dict[str, str] = {}
    for i, sd in enumerate(args.synthesis_dirs):
        label = f"Agent {_label(i)}"
        text = _read_synthesis(Path(sd))
        if text:
            syntheses[label] = text
            print(f"   ✓ {label}: {len(text)} chars from {sd}")
        else:
            print(f"   ✗ {label}: no synthesis.md found in {sd}")

    if len(syntheses) < 1:
        print("ERROR: No syntheses available for discussion")
        beat(status="failed", error="no syntheses available")
        sys.exit(1)

    if len(syntheses) == 1:
        print("   ℹ Single synthesis — will run critical self-review instead of multi-agent debate")

    beat(status="discussing", participants=len(syntheses))

    # Run discussion
    print(f"\n🗣️  Starting {args.rounds}-round discussion with {len(syntheses)} perspectives...\n", flush=True)
    transcript, consensus = run_discussion(
        llm, topic, syntheses, args.rounds, output_dir,
        max_prompt_chars=args.max_prompt_chars, progress=lambda step: beat(step=step),
    )

    # Write outputs
    (output_dir / "discussion_transcript.md").write_text(transcript, encoding="utf-8")
    (output_dir / "consensus_synthesis.md").write_text(consensus, encoding="utf-8")

    beat(status="completed", completed_at=int(time.time() * 1000))

    print(f"\n✅ Discussion complete")
    print(f"   Transcript: {output_dir / 'discussion_transcript.md'}")
    print(f"   Consensus:  {output_dir / 'consensus_synthesis.md'}")


if __name__ == "__main__":
    main()
