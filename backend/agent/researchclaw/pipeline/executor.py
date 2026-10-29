from __future__ import annotations

import os
import json
import logging
import math
import os
import re
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.hardware import HardwareProfile, detect_hardware, ensure_torch_available, is_metric_name
from researchclaw.llm import create_llm_client
from researchclaw.llm.client import LLMClient
from researchclaw.prompts import PromptManager
from researchclaw import single_run
from researchclaw.pipeline.stages import (
    NEXT_STAGE,
    Stage,
    StageStatus,
    TransitionEvent,
    TransitionOutcome,
    advance,
    gate_required,
)
from researchclaw.pipeline.contracts import CONTRACTS, StageContract
from researchclaw.experiment.validator import (
    CodeValidation,
    format_issues_for_llm,
    validate_code,
)

logger = logging.getLogger(__name__)

_PEER_REVIEW_CHUNK_CHARS = 24000
_PEER_REVIEW_SYNTHESIS_CHARS = 9000


# ---------------------------------------------------------------------------
# Domain detection — maps research topic to academic domain & venue context
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, tuple[list[str], str, str]] = {
    # domain_id: (keywords, display_name, top_venues)
    "ml": (
        ["machine learning", "deep learning", "neural network", "transformer",
         "reinforcement learning", "GAN", "diffusion model", "LLM", "language model",
         "computer vision", "NLP", "representation learning", "self-supervised",
         "federated learning", "meta-learning", "continual learning", "few-shot",
         "knowledge distillation", "attention mechanism", "fine-tuning", "RLHF",
         "vision transformer", "ViT", "BERT", "GPT", "autoencoder"],
        "machine learning",
        "NeurIPS, ICML, ICLR",
    ),
    "physics": (
        ["quantum", "thermodynamic", "electrodynamic", "particle physics",
         "condensed matter", "statistical mechanics", "cosmology", "astrophysics",
         "plasma", "optics", "photonics", "relativity", "gravitational",
         "PDE", "PINN", "physics-informed", "Burgers", "Navier-Stokes",
         "Darcy flow", "Schrödinger", "scientific computing", "operator learning",
         "neural operator", "Fourier neural", "DeepONet"],
        "physics",
        "Physical Review Letters, Nature Physics, JHEP",
    ),
    "chemistry": (
        ["molecular", "catalysis", "polymer", "organic chemistry", "inorganic",
         "electrochemistry", "spectroscopy", "crystallography", "drug discovery",
         "protein folding", "computational chemistry", "DFT", "force field"],
        "chemistry",
        "JACS, Nature Chemistry, Angewandte Chemie",
    ),
    "economics": (
        ["econometric", "macroeconomic", "microeconomic", "game theory",
         "market", "fiscal policy", "monetary", "behavioral economics",
         "causal inference", "panel data", "regression discontinuity",
         "instrumental variable", "supply chain", "auction"],
        "economics",
        "AER, Econometrica, QJE, Review of Economic Studies",
    ),
    "mathematics": (
        ["theorem", "proof", "prove", "conjecture", "topology", "algebra",
         "number theory", "combinatorics", "differential equation",
         "stochastic process", "functional analysis", "manifold",
         "Riemannian", "category theory", "graph theory",
         "neural ODE", "dynamical system", "Lorenz", "chaotic",
         "Lyapunov", "attractor", "ODE solver", "trajectory prediction",
         "mathematical formulation", "mathematical proof", "derivation",
         "Brownian motion", "branching process", "Galton-Watson",
         "Markov chain", "martingale", "ergodic", "convergence theorem",
         "marginal distribution", "extinction probability", "Feynman-Kac",
         "measure theory", "Hilbert space", "Banach space", "operator theory",
         "variational", "Euler-Lagrange", "calculus of variations"],
        "mathematics",
        "Annals of Mathematics, Inventiones Mathematicae, JAMS",
    ),
    "engineering": (
        ["robotics", "control system", "signal processing", "FPGA",
         "embedded system", "VLSI", "antenna", "fluid dynamics", "CFD",
         "finite element", "structural", "mechatronics", "autonomous"],
        "engineering",
        "IEEE Transactions, ASME journals, AIAA",
    ),
    "biology": (
        ["genomics", "proteomics", "transcriptomics", "CRISPR",
         "single-cell", "phylogenetic", "ecology", "neuroscience",
         "bioinformatics", "sequencing", "gene expression", "epigenetic"],
        "biology",
        "Nature, Science, Cell, PNAS",
    ),
}


def _detect_domain(topic: str, domains: tuple[str, ...] = ()) -> tuple[str, str, str]:
    """Detect research domain from topic string and config domains.

    Returns ``(domain_id, display_name, top_venues)``.
    Falls back to ``("ml", "machine learning", "NeurIPS, ICML, ICLR")``.
    """
    # If user explicitly specified domains, check them first
    for d in domains:
        d_lower = d.lower().strip()
        for did, (kws, dname, venues) in _DOMAIN_KEYWORDS.items():
            if d_lower in (did, dname) or any(k in d_lower for k in kws[:3]):
                return did, dname, venues

    # Auto-detect from topic text
    topic_lower = topic.lower()
    best_did, best_score = "ml", 0
    # BUG-101: Explicit theoretical intent words boost non-empirical domain scores.
    # Topics like "derive the mathematical formulation of X diffusion model"
    # should classify as math, not ML, even if "diffusion model" is an ML keyword.
    _theoretical_intent = any(
        w in topic_lower
        for w in ("derive", "prove", "mathematical formulation",
                  "mathematical proof", "formal proof", "formalism")
    )
    for did, (kws, dname, venues) in _DOMAIN_KEYWORDS.items():
        score = sum(1 for k in kws if k.lower() in topic_lower)
        # Boost non-empirical domains when theoretical intent is detected
        if _theoretical_intent and did in ("mathematics", "physics", "economics"):
            score += 1
        if score > best_score:
            best_score = score
            best_did = did

    did = best_did
    _, dname, venues = _DOMAIN_KEYWORDS[did]
    return did, dname, venues


def _is_ml_domain(domain_id: str) -> bool:
    """Check if the detected domain is ML/AI."""
    return domain_id == "ml"


@dataclass(frozen=True)
class StageResult:
    """Outcome of executing a single stage."""

    stage: Stage
    status: StageStatus
    artifacts: tuple[str, ...]
    error: str | None = None
    decision: str = "proceed"
    evidence_refs: tuple[str, ...] = ()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_CHINESE_ENGLISH_DOMAIN_MAP: dict[str, list[str]] = {
    "篡改": ["forgery", "tampering", "forensics"],
    "图像篡改定位": ["image manipulation localization"],
    "多尺度": ["multi-scale"],
    "注意力": ["attention"],
    "具身智能": ["embodied intelligence", "embodied AI"],
    "视觉语言动作": ["vision language action", "VLA"],
    "视觉语言模型": ["vision language model", "VLM"],
    "世界模型": ["world model"],
    "动作模型": ["action model"],
    "机器人": ["robot", "robotics"],
    "操控": ["manipulation"],
    "抓取": ["grasping"],
    "导航": ["navigation"],
    "模仿学习": ["imitation learning"],
    "强化学习": ["reinforcement learning"],
    "扩散策略": ["diffusion policy"],
    "视频生成": ["video generation"],
    "动作预测": ["action prediction"],
    "架构": ["architecture"],
    "最新": ["latest", "recent", "state of the art"],
    "评估": ["evaluation", "benchmark"],
    "应用": ["application"],
    "调研": ["survey"],
    "研究": ["research"],
    "方向": ["direction"],
    "联合建模": ["joint modeling", "unified model"],
    "联合训练": ["joint training"],
    "跨具身": ["cross-embodiment"],
    "灵巧操作": ["dexterous manipulation"],
    "双臂": ["bimanual", "dual-arm"],
    "长时序": ["long-horizon"],
    "泛化": ["generalization"],
    "零样本": ["zero-shot"],
    "预训练": ["pretraining", "pre-training"],
    "微调": ["fine-tuning"],
}


def _extract_english_from_mixed(topic: str) -> list[str]:
    """Extract English tokens and translate Chinese domain terms from mixed text."""
    english_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-_.]{1,}", topic)
    english_tokens = [t.strip("-_.") for t in english_tokens if len(t.strip("-_.")) > 1]

    translated: list[str] = []
    for zh, en_list in _CHINESE_ENGLISH_DOMAIN_MAP.items():
        if zh in topic:
            translated.extend(en_list)

    return english_tokens + translated


def _build_fallback_queries(topic: str) -> list[str]:
    """Extract meaningful search queries from a long topic string.

    Handles mixed Chinese-English topics by translating Chinese domain
    terms and extracting English keywords for academic search engines.
    """
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", topic))

    queries: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)

    if has_chinese:
        en_tokens = _extract_english_from_mixed(topic)
        en_tokens_dedup = list(dict.fromkeys(t.lower() for t in en_tokens))

        _skip_en = {"lab", "the", "and", "for", "with", "de", "成员"}
        core_en = [t for t in en_tokens_dedup if len(t) > 2 and t.lower() not in _skip_en]

        if core_en:
            _add(" ".join(core_en[:6]))
            _add(" ".join(core_en[:4]) + " survey")
            _add(" ".join(core_en[:4]) + " benchmark")
    else:
        chunks = re.split(r"[,:;()\[\]]+", topic)
        chunks = [c.strip() for c in chunks if len(c.strip()) > 8]
        cleaned_chunks = []
        for c in chunks:
            c = re.sub(
                r"^(and|or|the|a|an|in|of|for|with|across|multiple|three|various)\s+",
                "", c, flags=re.IGNORECASE,
            )
            c = c.strip()
            if len(c) > 8:
                cleaned_chunks.append(c)
        chunks = cleaned_chunks

        _stop = {
            "the", "and", "for", "with", "from", "that", "this", "into",
            "over", "across", "multiple", "three", "result", "comprehensive",
            "using", "based", "between", "various", "different", "several",
            "parameter", "parameters", "analysis", "approach", "method",
            "framework", "frameworks",
        }
        words = topic.lower().split()
        key_terms = [w for w in words if len(w) > 3 and w not in _stop]

        for chunk in chunks[:4]:
            if len(chunk) > 60:
                chunk = " ".join(chunk.split()[:6])
            _add(chunk)

        clean_terms = [t for t in key_terms if re.match(r"^[a-z]", t) and ":" not in t]
        for i in range(min(len(clean_terms) - 1, 4)):
            _add(f"{clean_terms[i]} {clean_terms[i + 1]}")

    topic_short = " ".join(re.findall(r"[a-zA-Z][a-zA-Z0-9\-_.]+", topic))[:60].strip()
    if not topic_short:
        topic_short = topic[:60].strip()
    for suffix in ("survey", "review", "benchmark", "state of the art", "recent advances"):
        if len(queries) >= 8:
            break
        _add(f"{topic_short} {suffix}")

    return queries[:12]


def _write_stage_meta(
    stage_dir: Path, stage: Stage, run_id: str, result: StageResult
) -> None:
    next_stage = NEXT_STAGE[stage]
    meta = {
        "stage_id": f"{int(stage):02d}-{stage.name.lower()}",
        "run_id": run_id,
        "status": result.status.value,
        "decision": result.decision,
        "output_artifacts": list(result.artifacts),
        "evidence_refs": list(result.evidence_refs),
        "error": result.error,
        "ts": _utcnow_iso(),
        "next_stage": int(next_stage) if next_stage is not None else None,
    }
    (stage_dir / "decision.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _read_prior_artifact(run_dir: Path, filename: str) -> str | None:
    # R14-2: Sort so non-versioned dirs (stage-13) come before versioned (stage-13_v1).
    # Within the same stage number, prefer the latest (non-versioned) copy.
    def _stage_sort_key(p: Path) -> tuple[str, int]:
        name = p.name
        # Extract base stage name and version
        if "_v" in name:
            base, _, ver = name.rpartition("_v")
            try:
                return (base, -int(ver))  # Versioned: lower priority (negative version)
            except ValueError:
                return (name, -999)
        return (name, 0)  # Non-versioned: highest priority

    for stage_subdir in sorted(run_dir.glob("stage-*"), key=_stage_sort_key, reverse=True):
        candidate = stage_subdir / filename
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
        if filename.endswith("/") and (stage_subdir / filename.rstrip("/")).is_dir():
            return str(stage_subdir / filename.rstrip("/"))
    return None


def _newest_stage_files(run_dir: Path, pattern: str) -> list[Path]:
    """Files matching *pattern* under *run_dir*, newest first.

    The current stage directory (``stage-15``) comes before older loop
    versions (``stage-15_v2``, then ``stage-15_v1``), and within a directory
    higher ``attempt-N`` numbers come first.  Plain string order would put
    ``stage-15_v1`` before ``stage-15`` and ``attempt-2`` before ``attempt-10``.
    """
    def key(path: Path) -> tuple[int, ...]:
        rank: list[int] = []
        for part in path.relative_to(run_dir).parts[:-1]:
            _, sep, ver = part.rpartition("_v")
            if part.startswith("stage-"):
                rank += (0, int(ver)) if sep and ver.isdigit() else (1, 0)
            elif part.startswith("attempt-") and part[8:].isdigit():
                rank.append(int(part[8:]))
        return tuple(rank)

    return sorted(run_dir.glob(pattern), key=key, reverse=True)


def _current_stage_dirs(run_dir: Path, *names: str) -> list[Path]:
    """Current round of each named stage dir, skipping older loop versions.

    ``stage-16`` wins over ``stage-16_v2``, which wins over ``stage-16_v1``.
    Figure plans and charts of an older REFINE round describe superseded
    results and must not reach the paper.
    """
    dirs: list[Path] = []
    for name in names:
        rounds = [
            p for p in run_dir.glob(f"{name}*")
            if p.is_dir() and (p.name == name or (
                p.name.startswith(f"{name}_v") and p.name[len(name) + 2:].isdigit()
            ))
        ]
        if rounds:
            dirs.append(max(rounds, key=lambda p: (
                p.name == name, 0 if p.name == name else int(p.name[len(name) + 2:]),
            )))
    return dirs


def _latest_experiment_summary_text(run_dir: Path) -> str | None:
    """Text of the newest experiment_summary.json (latest round only).

    Looks at stage-16 (RESULT_ANALYSIS) first, then the older stage-14
    layout, each ordered by ``_newest_stage_files``.  Older loop rounds
    (``stage-16_v1``...) are never preferred over the current round, even
    if they report more metrics: the paper must describe the latest run.
    """
    for pattern in ("stage-16*/experiment_summary.json", "stage-14*/experiment_summary.json"):
        for path in _newest_stage_files(run_dir, pattern):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if isinstance(_safe_json_loads(text, None), dict):
                logger.info("S20: Using %s", path.relative_to(run_dir).as_posix())
                return text
    return None


def _find_prior_file(run_dir: Path, filename: str) -> Path | None:
    """Like ``_read_prior_artifact`` but returns the *Path* instead of content."""
    def _stage_sort_key(p: Path) -> tuple[str, int]:
        name = p.name
        if "_v" in name:
            base, _, ver = name.rpartition("_v")
            try:
                return (base, -int(ver))
            except ValueError:
                return (name, -999)
        return (name, 0)

    for stage_subdir in sorted(run_dir.glob("stage-*"), key=_stage_sort_key, reverse=True):
        candidate = stage_subdir / filename
        if candidate.is_file():
            return candidate
    return None


def _experiment_summary_path(run_dir: Path, version: int | None = None) -> Path:
    """Locate experiment_summary.json written by RESULT_ANALYSIS (stage 16).

    ``version`` selects a rolled-back copy (``stage-16_v{version}``).  Older
    run layouts kept the summary under stage-14, so that location is checked
    as a fallback.  Returns the stage-16 path when nothing exists.
    """
    suffix = f"_v{version}" if version else ""
    candidates = [
        run_dir / f"stage-{int(Stage.RESULT_ANALYSIS):02d}{suffix}" / "experiment_summary.json",
        run_dir / f"stage-{int(Stage.EXPERIMENT_RUN):02d}{suffix}" / "experiment_summary.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _load_hardware_profile(run_dir: Path) -> dict[str, Any] | None:
    """Load hardware_profile.json from a prior stage (usually stage-01)."""
    raw = _read_prior_artifact(run_dir, "hardware_profile.json")
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_yaml_block(text: str) -> str:
    """Extract YAML from text that may contain ACP noise.

    Strips [thinking] blocks, insight blocks, and other ACP artifacts
    before looking for YAML in markdown fences or raw text.
    """
    # Strip ACP noise: [thinking]..., insight blocks, [plan]...
    cleaned = re.sub(
        r"\[thinking\].*?(?=\n```|\n[A-Z]|\Z)",
        "", text, flags=re.DOTALL,
    )
    cleaned = re.sub(r"\[plan\].*?\n\n", "", cleaned, flags=re.DOTALL)

    # Try markdown fences first (most reliable) — on cleaned text
    if "```yaml" in cleaned:
        return cleaned.split("```yaml", 1)[1].split("```", 1)[0].strip()
    if "```yml" in cleaned:
        return cleaned.split("```yml", 1)[1].split("```", 1)[0].strip()
    if "```" in cleaned:
        block = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
        if block:
            return block

    # Try the original text too (in case cleaning removed too much)
    if "```yaml" in text:
        return text.split("```yaml", 1)[1].split("```", 1)[0].strip()
    if "```yml" in text:
        return text.split("```yml", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        block = text.split("```", 1)[1].split("```", 1)[0].strip()
        if block:
            return block

    # Last resort: try to find YAML-like content (lines starting with key:)
    yaml_lines: list[str] = []
    in_yaml = False
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not in_yaml and re.match(r"^[a-z_]+:", stripped):
            in_yaml = True
        if in_yaml:
            if stripped and not stripped.startswith("#"):
                yaml_lines.append(line)
            elif not stripped and yaml_lines:
                yaml_lines.append(line)
    if yaml_lines:
        return "\n".join(yaml_lines).strip()

    return text.strip()


def _safe_json_loads(text: str, default: Any) -> Any:
    """Parse JSON from text, handling noisy ACP output.

    Tries multiple strategies: direct parse, markdown fence extraction,
    balanced brace matching (largest dict wins), and array brackets.
    """
    if not text or not text.strip():
        return default

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError):
        pass

    # Strategy 2: Find JSON in markdown code fences
    fence_pattern = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
    for match in fence_pattern.finditer(text):
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

    # Strategy 3: Find outermost balanced braces
    brace_depth = 0
    start = -1
    candidates: list[str] = []
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start >= 0:
                candidates.append(text[start : i + 1])
                start = -1

    # Try candidates from largest to smallest
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    # Strategy 4: Same for array [ ]
    bracket_depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "[":
            if bracket_depth == 0:
                start = i
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth == 0 and start >= 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    pass
                start = -1

    return default


_METACLAW_SKILLS_DIR = str(Path.home() / ".metaclaw" / "skills")


def _load_human_feedback(run_dir: Path | None, stage: "Stage") -> str:
    """Load pending human feedback from ``run_dir/human_feedback.jsonl``.

    Reads all feedback entries, formats them chronologically, and marks
    them as consumed by recording the current stage number in a stamp file.
    Returns empty string if no new feedback exists.
    """
    if run_dir is None:
        return ""
    fb_path = run_dir / "human_feedback.jsonl"
    if not fb_path.exists():
        return ""
    try:
        stamp_path = run_dir / ".feedback_consumed_up_to"
        consumed_ts = 0
        if stamp_path.exists():
            try:
                consumed_ts = int(stamp_path.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pass

        entries = []
        for line in fb_path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                continue

        if not entries:
            return ""

        new_entries = [e for e in entries if e.get("timestamp", 0) > consumed_ts]
        if not new_entries:
            all_text = "\n".join(
                f"- [{e.get('targetLayer', 'all')}] {e['content']}"
                for e in entries[-5:]
            )
            return f"(Previous feedback, still relevant)\n{all_text}"

        parts = []
        for e in new_entries:
            layer_tag = e.get("targetLayer", "all")
            content = e.get("content", "")
            parts.append(f"- [{layer_tag}] {content}")

        latest_ts = max(e.get("timestamp", 0) for e in new_entries)
        stamp_path.write_text(str(latest_ts), encoding="utf-8")

        return "\n".join(parts)
    except (OSError, json.JSONDecodeError):
        return ""


def _get_evolution_overlay(run_dir: Path | None, stage_name: str) -> str:
    """Load evolution lessons + MetaClaw skills for prompt injection.

    Combines intra-run lessons (from current run's evolution dir) with
    cross-run arc-* skills (from ~/.metaclaw/skills/).

    Returns empty string if no relevant lessons/skills exist or on any error.
    """
    if run_dir is None:
        return ""
    try:
        from researchclaw.evolution import EvolutionStore

        store = EvolutionStore(run_dir / "evolution")
        return store.build_overlay(
            stage_name, max_lessons=5, skills_dir=_METACLAW_SKILLS_DIR
        )
    except Exception:  # noqa: BLE001
        return ""


def _chat_with_prompt(
    llm: LLMClient,
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    max_tokens: int | None = None,
    retries: int = 0,
    strip_thinking: bool = True,
) -> Any:
    """Send a chat request with optional retry on timeout/transient errors.

    Parameters
    ----------
    retries:
        Number of extra attempts after the first failure (0 = no retry).
        Uses exponential backoff: 2s, 4s, 8s, ...
    strip_thinking:
        If True (default for pipeline usage), strip ``<think>`` tags from
        the LLM response.  This prevents chain-of-thought leakage from
        breaking YAML / JSON / LaTeX parsers downstream.
    """
    import time

    messages = [{"role": "user", "content": user}]
    last_exc: Exception | None = None
    for attempt in range(1 + retries):
        try:
            if json_mode and max_tokens is not None:
                return llm.chat(messages, system=system, json_mode=True, max_tokens=max_tokens, strip_thinking=strip_thinking)
            if json_mode:
                return llm.chat(messages, system=system, json_mode=True, strip_thinking=strip_thinking)
            if max_tokens is not None:
                return llm.chat(messages, system=system, max_tokens=max_tokens, strip_thinking=strip_thinking)
            return llm.chat(messages, system=system, strip_thinking=strip_thinking)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                delay = 2 ** (attempt + 1)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    1 + retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                raise last_exc from None
    raise last_exc  # type: ignore[misc]  # unreachable but satisfies type checker


def _split_markdown_for_peer_review(
    draft: str,
    max_chars: int = _PEER_REVIEW_CHUNK_CHARS,
) -> list[str]:
    """Split a long paper into reviewable chunks without dropping sections."""
    if len(draft) <= max_chars:
        return [draft]

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for block in re.split(r"(?=^#{1,6}\s+)", draft, flags=re.MULTILINE):
        if not block:
            continue
        if current and current_size + len(block) > max_chars:
            chunks.append("".join(current).strip())
            current = []
            current_size = 0
        if len(block) > max_chars:
            if current:
                chunks.append("".join(current).strip())
                current = []
                current_size = 0
            chunks.extend(
                block[start:start + max_chars].strip()
                for start in range(0, len(block), max_chars)
            )
        else:
            current.append(block)
            current_size += len(block)
    if current:
        chunks.append("".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _generate_neurips_checklist(
    has_experiments: bool = True,
    has_theory: bool = False,
    has_code: bool = True,
) -> str:
    """Generate a NeurIPS-style paper checklist appendix in markdown.

    This checklist is based on the NeurIPS 2025 submission requirements.
    It is appended to the paper before LaTeX conversion.
    """
    items = [
        ("Claims", "Do the main claims accurately reflect the paper's contributions and scope?", "Yes"),
        ("Limitations", "Does the paper discuss limitations of the work?", "Yes"),
    ]
    if has_theory:
        items.append(
            ("Theory", "Are all assumptions stated and proofs included?", "Yes")
        )
    items.extend([
        ("Experiments reproducibility", "Does the paper fully disclose experimental settings?", "Yes" if has_experiments else "NA"),
        ("Code and data", "Is code or data provided for reproducibility?", "Yes" if has_code else "No"),
        ("Experimental details", "Are training details and hyperparameters specified?", "Yes" if has_experiments else "NA"),
        ("Error bars", "Are error bars or confidence intervals reported?", "Yes" if has_experiments else "NA"),
        ("Compute resources", "Are compute requirements documented?", "Yes" if has_experiments else "NA"),
        ("Code of ethics", "Does the work comply with the code of ethics?", "Yes"),
        ("Broader impacts", "Are potential negative societal impacts discussed?", "Yes"),
        ("Licenses", "Are licenses for used assets respected?", "Yes"),
        ("New assets", "Are newly released assets documented?", "NA"),
        ("Human subjects", "Were IRB approvals obtained if applicable?", "NA"),
    ])

    lines = [
        "## NeurIPS Paper Checklist",
        "",
    ]
    for label, question, answer in items:
        lines.append(f"**{label}**: {question}")
        lines.append(f"Answer: [{answer}]")
        lines.append("")

    return "\n".join(lines)


def _extract_paper_title(md_text: str) -> str:
    """Extract paper title from markdown text for LaTeX generation.

    Prioritises H1 headings that appear *before* the abstract section and
    look like real titles (>= 4 words, starts with uppercase).  This avoids
    picking up pseudocode comments or algorithm step labels.
    """
    import re as _re

    # Limit search to content before Abstract heading
    abstract_pos = _re.search(
        r"^#{1,2}\s+(Abstract|ABSTRACT)", md_text, _re.MULTILINE
    )
    search_region = md_text[: abstract_pos.start()] if abstract_pos else md_text[:3000]

    _SKIP = {"title", "abstract", "references", "appendix"}
    candidates: list[str] = []

    for line in search_region.splitlines():
        line = line.strip()
        # Match H1 or H2 headings
        hm = _re.match(r"^(#{1,2})\s+(.+)$", line)
        if hm:
            heading = hm.group(2).strip()
            heading_lower = heading.lower()
            # Handle "## Title Actual Paper Title" pattern
            if heading_lower.startswith("title ") and len(heading) > 6:
                heading = heading[6:].strip()
                heading_lower = heading.lower()
            if heading_lower in _SKIP:
                continue
            candidates.append(heading)
        # Bold title line (e.g. **My Paper Title**)
        m = _re.match(r"\*\*(.+?)\*\*$", line)
        if m and len(m.group(1).split()) >= 3:
            candidates.append(m.group(1))

    # Prefer candidates that look like real titles (>= 4 words, capitalised)
    for c in candidates:
        words = c.split()
        if len(words) >= 4 and c[0].isupper():
            return c

    # Fallback: any candidate
    if candidates:
        return candidates[0]

    return "Untitled Paper"


# Generic title frames that hide the method name (checked on the whole title
# and on the subtitle after a "MethodName:" prefix).
_GENERIC_TITLE_FRAME_RE = re.compile(
    r"^\s*(?:"
    r"an?\s+(?:empirical|exploratory|systematic|comparative|preliminary)\s+"
    r"(?:study|analysis|investigation|evaluation)\b"
    r"|towards?\b|investigating\b|exploring\b|on\s+the\b"
    r"|when\b.*\bfalls?\s+short\b"
    r")",
    re.IGNORECASE,
)


def _generic_title_frame(title: str) -> str:
    """Return the generic frame matched by *title* (or its subtitle), else ''."""
    title = (title or "").strip().strip("*#").strip()
    parts = [title]
    if ":" in title:
        parts.append(title.split(":", 1)[1])
    for part in parts:
        m = _GENERIC_TITLE_FRAME_RE.match(part)
        if m:
            return m.group(0).strip()
    return ""


def _generate_framework_diagram_prompt(
    paper_text: str,
    config: "RCConfig",
    *,
    llm: "LLMClient | None" = None,
) -> str:
    """Generate a text-to-image prompt for a methodology framework diagram.

    Reads the paper's method section and produces a detailed prompt suitable
    for AI image generators (DALL-E, Midjourney, etc.).  The prompt describes
    an academic-style architecture/framework overview figure.

    Returns the prompt as a Markdown string, or empty string on failure.
    """
    import re as _re

    # Extract method/approach section from paper
    _method_section = ""
    _method_patterns = [
        r"(?:^#{1,3}\s+(?:Method(?:ology)?|Approach|Proposed\s+(?:Method|Framework|Approach)|Our\s+Method|Technical\s+Approach|Model\s+Architecture).*?)(?=^#{1,3}\s+|\Z)",
    ]
    for _pat in _method_patterns:
        _match = _re.search(_pat, paper_text, _re.MULTILINE | _re.DOTALL | _re.IGNORECASE)
        if _match:
            _method_section = _match.group(0)[:3000]
            break

    if not _method_section:
        # Fallback: use abstract + first 1500 chars
        _abs_match = _re.search(
            r"(?:^#{1,2}\s+Abstract\s*\n)(.*?)(?=^#{1,2}\s+|\Z)",
            paper_text, _re.MULTILINE | _re.DOTALL | _re.IGNORECASE,
        )
        _method_section = (_abs_match.group(1)[:1500] if _abs_match else paper_text[:2000])

    title = _extract_paper_title(paper_text)
    topic = config.research.topic

    # Use LLM to generate the prompt if available
    if llm is not None:
        _system = (
            "You are an expert academic figure designer. Generate a detailed text-to-image "
            "prompt for creating a methodology framework/architecture overview diagram.\n\n"
            "Requirements:\n"
            "- Academic style: clean, professional, suitable for a top-tier ML conference paper\n"
            "- Color palette: sophisticated and harmonious (suggest specific hex colors, "
            "prefer muted blues #4477AA, teals #44AA99, warm accents #CCBB44, soft purples #AA3377)\n"
            "- Layout: left-to-right or top-to-bottom data flow, with clearly labeled components\n"
            "- Components: boxes/modules with rounded corners, directional arrows, clear labels\n"
            "- Information density: high but not cluttered — each box should have a short label\n"
            "- Text on figure: minimal, only component names and key annotations\n"
            "- Background: white or very light grey\n"
            "- Style: vector-art look, flat design with subtle shadows, NO photorealism\n\n"
            "Output ONLY the prompt text (no markdown headers, no explanations). "
            "The prompt should be 150-300 words, highly specific and actionable."
        )
        _user = (
            f"Paper title: {title}\n"
            f"Research topic: {topic}\n\n"
            f"Method section excerpt:\n{_method_section}\n\n"
            "Generate a detailed text-to-image prompt for the methodology framework diagram."
        )
        try:
            resp = _chat_with_prompt(llm, _system, _user, max_tokens=1024)
            _llm_prompt = resp.content.strip()
            if len(_llm_prompt) > 50:
                return (
                    f"# Framework Diagram Prompt\n\n"
                    f"**Paper**: {title}\n\n"
                    f"## Image Generation Prompt\n\n"
                    f"{_llm_prompt}\n\n"
                    f"## Usage Instructions\n\n"
                    f"1. Copy the prompt above into an AI image generator "
                    f"(DALL-E 3, Midjourney, Ideogram, etc.)\n"
                    f"2. Generate the image at high resolution (2048x1024 or similar landscape)\n"
                    f"3. Save as `framework_diagram.png` in the same `charts/` folder\n"
                    f"4. Insert into the paper's Method section using:\n"
                    f"   - LaTeX: `\\includegraphics[width=\\textwidth]{{charts/framework_diagram.png}}`\n"
                    f"   - Markdown: `![Framework Overview](charts/framework_diagram.png)`\n"
                )
        except Exception:
            logger.debug("Framework prompt LLM generation failed, using template")

    # Fallback: template-based prompt without LLM
    _components = []
    _component_patterns = [
        (r"(?:encoder|decoder|transformer|attention|convolution|MLP|GNN|ResNet|ViT)", "Neural Network Module"),
        (r"(?:loss|objective|criterion|training|optimization)", "Training/Optimization"),
        (r"(?:data|dataset|input|preprocessing|augmentation)", "Data Pipeline"),
        (r"(?:output|prediction|inference|evaluation)", "Output/Evaluation"),
    ]
    _method_lower = _method_section.lower()
    for pat, label in _component_patterns:
        if _re.search(pat, _method_lower):
            _components.append(label)

    if not _components:
        _components = ["Input Processing", "Core Model", "Training Loop", "Evaluation"]

    return (
        f"# Framework Diagram Prompt\n\n"
        f"**Paper**: {title}\n\n"
        f"## Image Generation Prompt\n\n"
        f"Create a clean, academic-style methodology framework diagram for a research paper "
        f"titled \"{title}\". "
        f"The diagram should show a left-to-right data flow pipeline with these main components: "
        f"{', '.join(_components)}. "
        f"Use a professional color palette with muted blues (#4477AA), teals (#44AA99), "
        f"warm yellows (#CCBB44), and soft purples (#AA3377) on a white background. "
        f"Each component should be a rounded rectangle with a short label inside. "
        f"Connect components with clean directional arrows. "
        f"Add subtle shadows for depth. Flat vector-art style, no photorealism. "
        f"High information density but visually clean. "
        f"Suitable for a top-tier machine learning conference paper (ICML/NeurIPS/ICLR). "
        f"Landscape orientation, 2048x1024 resolution.\n\n"
        f"## Usage Instructions\n\n"
        f"1. Copy the prompt above into an AI image generator "
        f"(DALL-E 3, Midjourney, Ideogram, etc.)\n"
        f"2. Generate the image at high resolution (2048x1024 or similar landscape)\n"
        f"3. Save as `framework_diagram.png` in the same `charts/` folder\n"
        f"4. Insert into the paper's Method section using:\n"
        f"   - LaTeX: `\\includegraphics[width=\\textwidth]{{charts/framework_diagram.png}}`\n"
        f"   - Markdown: `![Framework Overview](charts/framework_diagram.png)`\n"
    )


def _safe_filename(name: str) -> str:
    name = name.replace("/", "_").replace("\\", "_").replace("..", "_")
    name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)
    return name[:100] or "unnamed"


def _collect_experiment_results(
    run_dir: Path,
    metric_key: str = "",
    metric_direction: str = "maximize",
) -> dict[str, Any]:
    """Aggregate experiment metrics from runs/ directory across prior stages.

    Returns a dict with ``runs``, ``metrics_summary``, ``best_run``,
    ``latex_table``, and optionally ``structured_results``. When final
    refinement results exist, only that version enters the statistics;
    ``result_versions`` retains earlier evidence without counting it as
    repeat trials. Agentic report counts are marked as result records.
    """
    runs_data: list[dict[str, Any]] = []
    result_versions: list[dict[str, Any]] = []
    selected_version = ""
    has_agentic_report = False
    structured_results: Any = None

    # Scan all stage dirs for runs/ subdirectory
    for stage_subdir in sorted(run_dir.glob("stage-*/runs")):
        # Check for structured results.json first
        results_json = stage_subdir / "results.json"
        if results_json.exists() and structured_results is None:
            try:
                structured_results = json.loads(
                    results_json.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                pass

        for run_file in sorted(stage_subdir.glob("*.json")):
            if run_file.name == "results.json":
                continue  # Already handled above
            parsed = _safe_json_loads(run_file.read_text(encoding="utf-8"), {})
            if isinstance(parsed, dict) and "metrics" in parsed:
                # Also check for structured_results inside run payload
                if "structured_results" in parsed and structured_results is None:
                    structured_results = parsed["structured_results"]
                runs_data.append(parsed)
            elif isinstance(parsed, dict) and "key_metrics" in parsed:
                # Simulated mode uses key_metrics
                parsed["metrics"] = parsed.pop("key_metrics")
                runs_data.append(parsed)
            elif isinstance(parsed, dict):
                # Agentic EXPERIMENT_RUN report (run_report.json)
                _agentic_metrics = _run_payload_metrics(parsed)
                if _agentic_metrics:
                    has_agentic_report = True
                    runs_data.append({
                        "metrics": _agentic_metrics,
                        "source": run_file.name,
                        "version": stage_subdir.parent.name,
                    })

    result_versions.extend(runs_data)

    # Agentic ITERATIVE_REFINE copies the refined harness output to results.json
    for _refined in sorted(run_dir.glob(f"stage-{int(Stage.ITERATIVE_REFINE):02d}*/results.json")):
        _refined_payload = _safe_json_loads(_refined.read_text(encoding="utf-8"), {})
        if isinstance(_refined_payload, dict):
            _refined_metrics = _run_payload_metrics(_refined_payload)
            if _refined_metrics:
                selected_version = _refined.parent.name
                refined_record = {
                    "metrics": _refined_metrics,
                    "source": str(_refined.relative_to(run_dir)),
                    "version": selected_version,
                }
                result_versions.append(refined_record)
                runs_data = [refined_record]
                structured_results = _refined_payload

    # Fallback: synthesise runs_data from structured_results if it has summaries
    if not runs_data and isinstance(structured_results, dict):
        _summaries = structured_results.get("summaries", [])
        if isinstance(_summaries, list):
            for _s in _summaries:
                if not isinstance(_s, dict):
                    continue
                _synth_metrics: dict[str, float] = {}
                for _mk, _mv in _s.items():
                    if _mk == "method":
                        continue
                    try:
                        _synth_metrics[_mk] = float(_mv)
                    except (ValueError, TypeError):
                        if isinstance(_mv, list):
                            continue
                if _synth_metrics:
                    runs_data.append({
                        "metrics": _synth_metrics,
                        "condition": _s.get("method", "unknown"),
                    })

    # Resumed/hand-finished runs: run_report.json is a failed attempt while
    # runs/results.json holds the finished experiment's real metrics.
    if not runs_data and isinstance(structured_results, dict):
        _structured_metrics = _run_payload_metrics(structured_results)
        if _structured_metrics:
            runs_data.append({"metrics": _structured_metrics, "source": "results.json"})
            result_versions.append(runs_data[0])

    if not runs_data:
        result: dict[str, Any] = {"runs": [], "metrics_summary": {}, "best_run": None, "latex_table": ""}
        if structured_results is not None:
            result["structured_results"] = structured_results
        return result

    # Aggregate metrics across runs
    all_metric_keys: set[str] = set()
    for r in runs_data:
        m = r.get("metrics") or {}
        if isinstance(m, dict):
            all_metric_keys.update(m.keys())

    metrics_summary: dict[str, dict[str, float | None]] = {}
    for key in sorted(all_metric_keys):
        values = []
        for r in runs_data:
            m = r.get("metrics") or {}
            if isinstance(m, dict) and key in m:
                try:
                    values.append(float(m[key]))
                except (ValueError, TypeError):
                    pass
        if values:
            metrics_summary[key] = {
                "min": round(min(values), 6),
                "max": round(max(values), 6),
                "mean": round(sum(values) / len(values), 6),
                "count": len(values),
            }

    # Find best run using metric_key and metric_direction
    best_run: dict[str, Any] | None = None
    if runs_data:

        def _primary_metric(r: dict[str, Any]) -> float:
            m = r.get("metrics") or {}
            if isinstance(m, dict):
                # Try specific metric_key first
                if metric_key and metric_key in m:
                    try:
                        return float(m[metric_key])
                    except (ValueError, TypeError):
                        pass
                # Fallback to first metric
                for v in m.values():
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        pass
            return 0.0

        _cmp = min if metric_direction == "minimize" else max
        best_run = _cmp(runs_data, key=_primary_metric)

    # Build LaTeX table
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Experiment Results}",
    ]
    if metrics_summary:
        cols = sorted(metrics_summary.keys())
        header = "Metric & Min & Max & Mean & N \\\\"
        latex_lines.append(r"\begin{tabular}{l" + "r" * 4 + "}")
        latex_lines.append(r"\hline")
        latex_lines.append(header)
        latex_lines.append(r"\hline")
        for col in cols:
            s = metrics_summary[col]
            row = f"{col} & {s['min']:.4f} & {s['max']:.4f} & {s['mean']:.4f} & {s['count']} \\\\"
            latex_lines.append(row)
        latex_lines.append(r"\hline")
        latex_lines.append(r"\end{tabular}")
    else:
        latex_lines.append(r"\begin{tabular}{l}")
        latex_lines.append("No experiment data available \\\\")
        latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    # R18-1: Extract paired statistical comparisons from stdout
    from researchclaw.experiment.sandbox import extract_paired_comparisons

    paired_comparisons: list[dict[str, object]] = []
    for r in runs_data:
        stdout = r.get("stdout", "")
        if stdout:
            paired_comparisons.extend(extract_paired_comparisons(stdout))

    collected: dict[str, Any] = {
        "runs": runs_data,
        "metrics_summary": metrics_summary,
        "best_run": best_run,
        "latex_table": "\n".join(latex_lines),
    }
    if selected_version or has_agentic_report:
        collected["result_versions"] = result_versions
        collected["selected_version"] = selected_version or runs_data[-1].get("version", "")
        collected["run_count_basis"] = "result_records"
    if paired_comparisons:
        collected["paired_comparisons"] = paired_comparisons
    if structured_results is not None:
        collected["structured_results"] = structured_results
    return collected


def _find_discussion_dir(run_dir: Path, config: RCConfig) -> Path | None:
    """Locate the discussion artifacts directory.

    Checks run_dir/discussion first, then falls back to
    shared_results_dir/idea_runs/<idea_id>/discussion for cases
    where L1 ran in shared_results but L5 runs in /dev/shm.
    """
    local = run_dir / "discussion"
    if local.is_dir() and (local / "consensus_synthesis.md").is_file():
        return local
    shared_dir = getattr(config.experiment, "shared_results_dir", "") or ""
    if shared_dir:
        fallback = Path(shared_dir) / "idea_runs" / run_dir.name / "discussion"
        if fallback.is_dir() and (fallback / "consensus_synthesis.md").is_file():
            return fallback
    return None


def _build_context_preamble(
    config: RCConfig,
    run_dir: Path,
    *,
    include_goal: bool = False,
    include_hypotheses: bool = False,
    include_synthesis: bool = False,
    include_exp_plan: bool = False,
    include_analysis: bool = False,
    include_decision: bool = False,
    include_experiment_data: bool = False,
    include_discussion: bool = False,
) -> str:
    parts = [
        "## Research Context",
        f"**Topic**: {config.research.topic}",
        f"**Domains**: {', '.join(config.research.domains) if config.research.domains else 'general'}",
    ]
    if include_goal:
        goal = _read_prior_artifact(run_dir, "goal.md")
        if goal:
            parts.append(f"\n### Goal\n{goal[:2200]}")
    if include_hypotheses:
        hyp = _read_prior_artifact(run_dir, "hypotheses.md")
        if hyp:
            parts.append(f"\n### Hypotheses\n{hyp[:2200]}")
    if include_synthesis:
        synthesis = _read_prior_artifact(run_dir, "synthesis.md")
        if synthesis:
            parts.append(f"\n### Synthesis\n{synthesis[:2200]}")
    if include_exp_plan:
        plan = _read_prior_artifact(run_dir, "exp_plan.yaml")
        if plan:
            parts.append(f"\n### Experiment Plan\n{plan[:2000]}")
    if include_analysis:
        analysis = _read_prior_artifact(run_dir, "analysis.md")
        if analysis:
            parts.append(f"\n### Result Analysis\n{analysis[:2500]}")
    if include_decision:
        decision = _read_prior_artifact(run_dir, "decision.md")
        if decision:
            parts.append(f"\n### Research Decision\n{decision[:1500]}")
    if include_discussion:
        disc_dir = _find_discussion_dir(run_dir, config)
        if disc_dir is not None:
            pre_synth = disc_dir / "pre_discussion_syntheses.md"
            consensus = disc_dir / "consensus_synthesis.md"
            transcript = disc_dir / "discussion_transcript.md"
            if pre_synth.exists() and consensus.exists():
                parts.append("\n### Multi-Agent Discussion (Ablation Data)")
                _pre = pre_synth.read_text(encoding="utf-8")
                parts.append(f"\n#### Pre-Discussion Individual Syntheses\n{_pre[:4000]}")
                _con = consensus.read_text(encoding="utf-8")
                parts.append(f"\n#### Post-Discussion Consensus\n{_con[:4000]}")
                if transcript.exists():
                    _tr = transcript.read_text(encoding="utf-8")
                    parts.append(f"\n#### Discussion Transcript (excerpt)\n{_tr[:3000]}")
    if include_experiment_data:
        hw_profile = _load_hardware_profile(run_dir)
        if hw_profile:
            hw_lines = ["### Hardware Environment"]
            for hk, hv in hw_profile.items():
                hw_lines.append(f"- **{hk}**: {hv}")
            parts.append("\n" + "\n".join(hw_lines))
        exp_summary = _read_prior_artifact(run_dir, "experiment_summary.json")
        if exp_summary:
            summary = _safe_json_loads(exp_summary, {})
            if isinstance(summary, dict) and summary.get("metrics_summary"):
                parts.append("\n### Experiment Results (Quantitative)")
                ms = summary["metrics_summary"]
                for mk, mv in ms.items():
                    if isinstance(mv, dict):
                        parts.append(
                            f"- **{mk}**: mean={mv.get('mean', '?')}, "
                            f"min={mv.get('min', '?')}, max={mv.get('max', '?')}, n={mv.get('count', '?')}"
                        )
                if summary.get("latex_table"):
                    parts.append(
                        f"\n### LaTeX Table\n```latex\n{summary['latex_table']}\n```"
                    )
    return "\n".join(parts)


# --- P1-1: Topic keyword extraction for domain pre-filter ---
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "of",
        "for",
        "to",
        "with",
        "by",
        "at",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "again",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "under",
        "over",
        "using",
        "based",
        "via",
        "toward",
        "towards",
        "new",
        "novel",
        "approach",
        "method",
        "study",
        "research",
        "paper",
        "work",
        "propose",
        "proposed",
    }
)


def _extract_topic_keywords(
    topic: str, domains: tuple[str, ...] | list[str] = ()
) -> list[str]:
    """Extract meaningful keywords from the research topic + domain list.

    Returns lowercased keyword list (2+ chars, no stop words).
    Used by the domain pre-filter to drop obviously irrelevant papers.
    Handles mixed Chinese-English topics by translating Chinese domain terms.
    """
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", topic.lower())
    keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) >= 3]
    if re.search(r"[\u4e00-\u9fff]", topic):
        translated = _extract_english_from_mixed(topic)
        for t in translated:
            t_lower = t.lower()
            if t_lower not in _STOP_WORDS and len(t_lower) >= 3:
                keywords.append(t_lower)
    for d in domains:
        for part in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", d.lower()):
            if part not in _STOP_WORDS and len(part) >= 2:
                keywords.append(part)
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


# --- P1-2: Topic constraint block for paper generation stages ---
def _topic_constraint_block(topic: str) -> str:
    """Return a hard constraint instruction that anchors paper content to the topic.

    Prevents the common LLM failure mode of drifting off-topic or
    presenting environmental/infrastructure issues as research contributions.
    """
    return (
        "\n\n=== HARD TOPIC CONSTRAINT ===\n"
        f"The paper MUST be about: {topic}\n"
        "PROHIBITED content (unless user explicitly specifies case-study mode):\n"
        "- Do NOT treat environment setup, dependency installation, or infrastructure "
        "failures as a research contribution.\n"
        "- Do NOT present debugging logs, system errors, or configuration issues "
        "as experimental findings.\n"
        "- Do NOT drift to tangential topics not directly related to the stated topic.\n"
        "- Every section MUST connect back to the core research question.\n"
        "- The Abstract and Introduction MUST clearly state the research problem "
        f"derived from: {topic}\n"
        "- The Method section MUST describe a technical approach, not a workflow.\n"
        "- The Results section MUST report quantitative outcomes of experiments, "
        "not environment status.\n"
        "=== END CONSTRAINT ===\n"
    )


def _detect_runtime_issues(sandbox_result: Any) -> str:
    """Detect NaN/Inf in metrics and extract stderr warnings from sandbox run.

    Returns a formatted string describing all runtime issues, or empty string
    if no issues are found.
    """
    import math

    issues: list[str] = []

    # Check metrics for NaN/Inf
    metrics = getattr(sandbox_result, "metrics", {}) or {}
    for key, val in metrics.items():
        try:
            fval = float(val)
            if math.isnan(fval):
                issues.append(f"METRIC NaN: '{key}' returned NaN — likely a division by zero or invalid computation in code")
            elif math.isinf(fval):
                issues.append(f"METRIC Inf: '{key}' returned Infinity — likely overflow or unbounded computation")
        except (TypeError, ValueError):
            pass

    # Check stdout for NaN values (word boundary to avoid matching "Nanotechnology" etc.)
    stdout = getattr(sandbox_result, "stdout", "") or ""
    _nan_re = re.compile(r"\bnan\b", re.IGNORECASE)
    if _nan_re.search(stdout):
        nan_lines = [
            line.strip()
            for line in stdout.splitlines()
            if _nan_re.search(line)
        ]
        if nan_lines:
            issues.append(
                f"NaN values detected in output:\n" + "\n".join(nan_lines[:10])
            )

    # Extract meaningful warnings from stderr
    stderr = getattr(sandbox_result, "stderr", "") or ""
    if stderr.strip():
        warning_lines = []
        for line in stderr.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Keep RuntimeWarning, ValueError, ZeroDivisionError, etc.
            if any(
                kw in line_stripped
                for kw in (
                    "Warning",
                    "Error",
                    "Traceback",
                    "Exception",
                    "divide",
                    "overflow",
                    "invalid value",
                    "NaN",
                    "inf",
                )
            ):
                warning_lines.append(line_stripped)
        if warning_lines:
            issues.append(
                "Runtime warnings/errors from stderr:\n"
                + "\n".join(warning_lines[:15])
            )

    # Check for identical metric values across all entries in stdout
    # (e.g., all algorithms reporting convergence_rate=1.0)
    stdout = getattr(sandbox_result, "stdout", "") or ""
    if stdout:
        from collections import Counter

        metric_values_by_name: dict[str, list[float]] = {}
        for line in stdout.splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            parts = line.rsplit(":", 1)
            if len(parts) != 2:
                continue
            try:
                fval = float(parts[1].strip())
            except (ValueError, TypeError):
                continue
            # Extract metric suffix (e.g. "convergence_rate" from "UCB (Stochastic) convergence_rate")
            name = parts[0].strip()
            metric_suffix = name.split()[-1] if name.split() else name
            metric_values_by_name.setdefault(metric_suffix, []).append(fval)

        for metric_name, vals in metric_values_by_name.items():
            if len(vals) >= 3:
                unique = set(vals)
                if len(unique) <= 2:
                    issues.append(
                        f"DUMMY METRIC: '{metric_name}' has only {len(unique)} unique value(s) "
                        f"across {len(vals)} entries ({unique}) — likely a placeholder. "
                        f"Implement real measurement logic (e.g., track iterations to convergence)."
                    )

    # R5-3: Check for diverging loss values (fast-fail indicator)
    for key, val in metrics.items():
        try:
            fval = float(val)
            if "loss" in key.lower() and fval > 100:
                issues.append(
                    f"DIVERGING LOSS: '{key}' = {fval} (>100) — the optimization is "
                    f"diverging. Reduce learning rate, check gradient computation, "
                    f"or add gradient clipping."
                )
        except (TypeError, ValueError):
            pass

    if not issues:
        return ""

    return (
        "## Runtime Issues Detected\n\n"
        "The experiment code ran but produced problematic results. "
        "Fix the ROOT CAUSE of these issues in the code:\n\n"
        + "\n\n".join(f"- {issue}" for issue in issues)
    )


def _parse_metrics_from_stdout(stdout: str) -> dict[str, Any]:
    """Parse ``name: value`` metric lines from experiment stdout.

    Handles formats like ``UCB (Stochastic) cumulative_regret: 361.9233``
    and simple ``loss: 0.0042``.  Returns a flat dict of metric_name → value.

    Filters out log/status lines (e.g. "Running experiments for support set
    size: 1") using :func:`is_metric_name`.
    """
    metrics: dict[str, Any] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        # Split on the LAST colon to handle names with colons
        parts = line.rsplit(":", 1)
        if len(parts) != 2:
            continue
        name_part = parts[0].strip()
        value_part = parts[1].strip()
        # Filter out log lines that look like status messages
        if not is_metric_name(name_part):
            continue
        try:
            fval = float(value_part)
            # Use the full name (e.g. "UCB (Stochastic) cumulative_regret")
            metrics[name_part] = fval
        except (ValueError, TypeError):
            pass
    return metrics


def _extract_code_block(content: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    if match is not None:
        return match.group(1).strip()
    # Handle truncated output: opening ``` without closing ```
    trunc = re.search(r"```(?:python)?\s*\n(.+)", content, flags=re.DOTALL)
    if trunc is not None:
        code = trunc.group(1).strip()
        if "\nimport " in code or "\ndef " in code or "\nclass " in code:
            return code
    # Last resort: if content looks like Python, return it; otherwise empty
    if content.strip().startswith(("import ", "from ", "#!/", "def ", "class ")):
        return content.strip()
    return ""


def _extract_multi_file_blocks(content: str) -> dict[str, str]:
    """Parse LLM response containing multiple files with filename markers.

    Expected format::

        ```filename:main.py
        import model
        ...
        ```

        ```filename:model.py
        class MyModel:
        ...
        ```

    Also handles common LLM format variations:
    - ````` ```python filename:main.py````` (space before filename)
    - ````` ``` filename:main.py````` (space after backticks)
    - ``filename:main.py`` on next line after backticks
    - ``# FILE: main.py`` comment markers inside code blocks

    Falls back to treating the entire code block as ``main.py`` if no
    ``filename:`` markers are found.

    Returns a dict mapping filename → code content.
    """
    # R13-2: Multiple patterns to handle LLM format variations
    patterns = [
        # Original: ```filename:xxx.py or ```python filename:xxx.py
        re.compile(
            r"```(?:python\s+)?filename:(\S+)\s*\n(.*?)```",
            flags=re.DOTALL,
        ),
        # Variation: ``` filename:xxx.py (space after backticks)
        re.compile(
            r"```\s+filename:(\S+)\s*\n(.*?)```",
            flags=re.DOTALL,
        ),
        # Variation: ```python\nfilename:xxx.py (filename on next line)
        re.compile(
            r"```(?:python)?\s*\nfilename:(\S+)\s*\n(.*?)```",
            flags=re.DOTALL,
        ),
        # Variation: ```python\n# filename: xxx.py (comment marker)
        re.compile(
            r"```(?:python)?\s*\n#\s*(?:FILE|filename)\s*:\s*(\S+\.py)\s*\n(.*?)```",
            flags=re.DOTALL,
        ),
    ]

    matches: list[tuple[str, str]] = []
    for pattern in patterns:
        matches = pattern.findall(content)
        if matches:
            break

    if matches:
        files: dict[str, str] = {}
        for fname, code in matches:
            fname = fname.strip()
            # Security: prevent path traversal
            if ".." in fname or fname.startswith("/"):
                continue
            # Normalise to flat filenames (strip leading ./ or subdirs for safety)
            fname = fname.replace("\\", "/").split("/")[-1]
            if fname and fname.endswith(".py"):
                files[fname] = code.strip()
        if files:
            # Ensure there is a main.py entry point
            if "main.py" not in files:
                # Pick the first file as main.py
                first_key = next(iter(files))
                files["main.py"] = files.pop(first_key)
            return files

    # Fallback: single code block → main.py
    code = _extract_code_block(content)
    if code.strip():
        return {"main.py": code}
    return {}


def _parse_jsonl_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = _safe_json_loads(line, {})
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _collect_json_context(
    directory: Path,
    *,
    max_files: int = 30,
    max_total_chars: int = 50_000,
) -> str:
    """Collect JSON context from a directory, with size limits.

    Large fields like ``stderr`` and ``stdout`` are stripped to avoid
    exceeding LLM token limits (the raw experiment output can be 5 MB+).
    """
    chunks: list[str] = []
    total = 0
    for file_path in sorted(directory.glob("*.json"))[:max_files]:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        # Strip verbose fields that bloat the context
        if isinstance(data, dict):
            for key in ("stderr", "stdout", "raw_output", "traceback"):
                if key in data and isinstance(data[key], str) and len(data[key]) > 500:
                    data[key] = data[key][:500] + f"\n... [truncated, {len(data[key])} chars total]"
        chunk = json.dumps(data, indent=2, ensure_ascii=False)
        if total + len(chunk) > max_total_chars:
            remaining = max_total_chars - total
            if remaining > 200:
                chunks.append(chunk[:remaining] + "\n... [truncated]")
            break
        chunks.append(chunk)
        total += len(chunk)
    return "\n\n".join(chunks)


def _default_hypotheses(topic: str) -> str:
    return f"""# Hypotheses

## H1
Increasing protocol control for {topic} improves metric stability across random seeds.

## H2
Adding robustness-aware objectives for {topic} improves out-of-domain performance without major in-domain regression.

## H3
The combined approach outperforms either component under fixed compute budget.

## Generated
{_utcnow_iso()}
"""


def _default_paper_outline(topic: str) -> str:
    return f"""# Paper Outline

## 1. Title
Focused title on {topic}

## 2. Abstract
- Problem framing
- Method overview
- Key quantitative result

## 3. Introduction
- Motivation
- Gap statement
- Contributions

## 4. Related Work
- Method families
- Evaluation practices

## 5. Method
- Problem setup
- Model/algorithm
- Complexity and constraints

## 6. Experiments
- Datasets and metrics
- Baselines and ablations
- Reproducibility protocol

## 7. Results
- Main table
- Robustness analysis
- Failure cases

## 8. Discussion
- Practical implications
- Limitations

## 9. Conclusion
- Findings and next steps

Generated: {_utcnow_iso()}
"""


def _default_quality_report(threshold: float) -> dict[str, Any]:
    # When LLM fails, return below-threshold score to force revision
    score = max(1.0, float(threshold) - 2.0) if threshold > 0 else 5.0
    score = max(1.0, min(10.0, score))
    verdict = "revise"
    return {
        "score_1_to_10": round(score, 2),
        "verdict": verdict,
        "criteria": {
            "novelty": round(min(10.0, score + 0.3), 2),
            "methodological_rigor": round(score, 2),
            "clarity": round(max(1.0, score - 0.2), 2),
            "reproducibility": round(min(10.0, score + 0.1), 2),
        },
        "strengths": [
            "Stage-by-stage evidence chain preserved",
            "Experiment artifacts are generated and archived",
        ],
        "weaknesses": [
            "Statistical significance may need stronger reporting",
            "Broader external validity remains partially evaluated",
        ],
        "required_actions": [
            "Report confidence intervals and seed variance",
            "Include at least one stronger external baseline",
        ],
        "generated": _utcnow_iso(),
    }


def _execute_topic_init(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    topic = config.research.topic
    domains = (
        ", ".join(config.research.domains) if config.research.domains else "general"
    )
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "topic_init")
        sp = _pm.for_stage(
            "topic_init",
            evolution_overlay=_overlay,
            topic=topic,
            domains=domains,
            project_name=config.project.name,
            quality_threshold=config.research.quality_threshold,
        )
        resp = llm.chat(
            [{"role": "user", "content": sp.user}],
            system=sp.system,
            max_tokens=8192,
        )
        goal_md = resp.content
    else:
        goal_md = f"""# Research Goal

## Topic
{topic}

## Scope
Investigate the topic with emphasis on reproducible methods and measurable outcomes.

## SMART Goal
- Specific: Build a focused research plan for {topic}
- Measurable: Produce literature shortlist, hypotheses, experiment plan, and final paper
- Achievable: Complete through staged pipeline with gate checks
- Relevant: Aligned with project {config.project.name}
- Time-bound: Constrained by pipeline execution budget

## Constraints
- Quality threshold: {config.research.quality_threshold}
- Daily paper target: {config.research.daily_paper_count}

## Success Criteria
- At least 2 falsifiable hypotheses
- Executable experiment code and results analysis
- Revised paper passing quality gate

## Generated
{_utcnow_iso()}
"""
    (stage_dir / "goal.md").write_text(goal_md, encoding="utf-8")

    # --- Hardware detection (GPU / MPS / CPU) ---
    hw = detect_hardware()
    (stage_dir / "hardware_profile.json").write_text(
        json.dumps(hw.to_dict(), indent=2), encoding="utf-8"
    )
    if hw.warning:
        logger.warning("Hardware advisory: %s", hw.warning)
    else:
        logger.info("Hardware detected: %s (%s, %s MB VRAM)", hw.gpu_name, hw.gpu_type, hw.vram_mb)

    # --- Optionally ensure PyTorch is available ---
    if hw.has_gpu and config.experiment.mode == "sandbox":
        torch_ok = ensure_torch_available(config.experiment.sandbox.python_path, hw.gpu_type)
        if torch_ok:
            logger.info("PyTorch is available for sandbox experiments")
        else:
            logger.warning("PyTorch could not be installed; sandbox will use CPU-only packages")
    elif hw.has_gpu and config.experiment.mode == "docker":
        logger.info("Docker sandbox: PyTorch pre-installed in container image")

    return StageResult(
        stage=Stage.TOPIC_INIT,
        status=StageStatus.DONE,
        artifacts=("goal.md", "hardware_profile.json"),
        evidence_refs=("stage-01/goal.md", "stage-01/hardware_profile.json"),
    )


def _execute_problem_decompose(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    goal_text = _read_prior_artifact(run_dir, "goal.md") or ""
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "problem_decompose")
        sp = _pm.for_stage(
            "problem_decompose",
            evolution_overlay=_overlay,
            topic=config.research.topic,
            goal_text=goal_text,
        )
        resp = llm.chat(
            [{"role": "user", "content": sp.user}],
            system=sp.system,
        )
        body = resp.content
    else:
        body = f"""# Problem Decomposition

## Source
Derived from `goal.md` for topic: {config.research.topic}

## Sub-questions
1. Which problem settings and benchmarks define current SOTA?
2. Which methodological gaps remain unresolved?
3. Which hypotheses are testable under realistic constraints?
4. Which datasets and metrics best discriminate method quality?
5. Which failure modes can invalidate expected gains?

## Priority Ranking
1. Problem framing and benchmark setup
2. Gap identification and hypothesis formulation
3. Experiment and metric design
4. Failure analysis and robustness checks

## Risks
- Ambiguous task definition
- Dataset leakage or metric mismatch

## Generated
{_utcnow_iso()}
"""
    (stage_dir / "problem_tree.md").write_text(body, encoding="utf-8")

    # IMP-35: Topic/title quality pre-evaluation
    # Quick LLM check: is the topic well-scoped for a conference paper?
    if llm is not None:
        try:
            _eval_resp = llm.chat(
                [
                    {
                        "role": "user",
                        "content": (
                            "Evaluate this research topic for a top ML conference paper. "
                            "Score 1-10 on: (a) novelty, (b) specificity, (c) feasibility. "
                            "If overall score < 5, suggest a refined topic.\n\n"
                            f"Topic: {config.research.topic}\n\n"
                            "Reply as JSON: {\"novelty\": N, \"specificity\": N, "
                            "\"feasibility\": N, \"overall\": N, \"suggestion\": \"...\"}"
                        ),
                    }
                ],
                system=(
                    f"You are a senior {_detect_domain(config.research.topic, config.research.domains)[1]} "
                    f"researcher evaluating research topic quality."
                ),
            )
            _eval_data = _safe_json_loads(_eval_resp.content, {})
            if isinstance(_eval_data, dict):
                overall = _eval_data.get("overall", 10)
                if isinstance(overall, (int, float)) and overall < 5:
                    logger.warning(
                        "IMP-35: Topic quality score %s/10 — consider refining: %s",
                        overall,
                        _eval_data.get("suggestion", ""),
                    )
                else:
                    logger.info("IMP-35: Topic quality score %s/10", overall)
                (stage_dir / "topic_evaluation.json").write_text(
                    json.dumps(_eval_data, indent=2), encoding="utf-8"
                )
        except Exception:  # noqa: BLE001
            logger.debug("IMP-35: Topic evaluation skipped (non-blocking)")

    return StageResult(
        stage=Stage.PROBLEM_DECOMPOSE,
        status=StageStatus.DONE,
        artifacts=("problem_tree.md",),
        evidence_refs=("stage-02/problem_tree.md",),
    )


_PLAN_QUERY_KEYS = frozenset({"queries", "query", "search_queries"})
# Longer values stored under a query key are retrieval instructions, not queries.
_MAX_PLAN_QUERY_CHARS = 150


def _plan_query_values(value: Any) -> list[str]:
    items = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        for part in item.split(";"):
            part = part.strip()
            if part and len(part) <= _MAX_PLAN_QUERY_CHARS and not re.search(r"[\u4e00-\u9fff]", part):
                out.append(part)
    return out


def _collect_plan_queries(plan: Any, llm_sources: list[dict[str, Any]]) -> list[str]:
    """Pull literature queries out of the S3 search plan.

    The prompt asks for ``search_strategies[].queries``, but LLMs regularly
    rename that structure (``search_tracks``, ``search_phases`` ...).  Instead
    of discarding such a plan and rebuilding queries from the topic, fall back
    to every ``queries``/``query`` field in the plan, then to the queries the
    LLM attached to its sources.
    """
    queries: list[str] = []
    if isinstance(plan, dict):
        strategies = plan.get("search_strategies", [])
        if isinstance(strategies, list):
            for strat in strategies:
                if isinstance(strat, dict):
                    qs = strat.get("queries", [])
                    if isinstance(qs, list):
                        queries.extend(str(q) for q in qs if q)
    if queries:
        return queries

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _PLAN_QUERY_KEYS:
                    queries.extend(_plan_query_values(value))
                else:
                    _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(plan)
    if queries:
        logger.info("Stage 03: plan has no search_strategies; using %d queries found elsewhere in the plan", len(queries))
        return list(dict.fromkeys(queries))
    for src in llm_sources:
        queries.extend(_plan_query_values(src.get("query")))
    if queries:
        logger.info("Stage 03: plan has no queries; using %d queries from the LLM's sources", len(queries))
    return list(dict.fromkeys(queries))


def _execute_search_strategy(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    problem_tree = _read_prior_artifact(run_dir, "problem_tree.md") or ""
    topic = config.research.topic
    plan: dict[str, Any] | None = None
    sources: list[dict[str, Any]] | None = None
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "search_strategy")
        sp = _pm.for_stage("search_strategy", evolution_overlay=_overlay, topic=topic, problem_tree=problem_tree)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict):
            yaml_text = str(payload.get("search_plan_yaml", "")).strip()
            if yaml_text:
                try:
                    parsed = yaml.safe_load(_extract_yaml_block(yaml_text))
                except yaml.YAMLError:
                    parsed = None
                if isinstance(parsed, dict):
                    plan = parsed
            src = payload.get("sources", [])
            if isinstance(src, list):
                sources = [item for item in src if isinstance(item, dict)]
    llm_plan = plan
    if plan is None:
        # Build smart fallback queries by extracting key terms from topic
        # instead of using the raw (often very long) topic string.
        _fallback_queries = _build_fallback_queries(topic)
        plan = {
            "topic": topic,
            "generated": _utcnow_iso(),
            "search_strategies": [
                {
                    "name": "keyword_core",
                    "queries": _fallback_queries[:5],
                    "sources": ["arxiv", "semantic_scholar", "openreview"],
                    "max_results_per_query": 60,
                },
                {
                    "name": "backward_forward_citation",
                    "queries": _fallback_queries[5:10] or _fallback_queries[:3],
                    "sources": ["semantic_scholar", "google_scholar"],
                    "depth": 1,
                },
            ],
            "filters": {
                "min_year": 2020,
                "language": ["en"],
                "peer_review_preferred": True,
            },
            "deduplication": {"method": "title_doi_hash", "fuzzy_threshold": 0.9},
        }
    llm_sources = list(sources or [])
    if not sources:
        sources = [
            {
                "id": "arxiv",
                "name": "arXiv",
                "type": "api",
                "url": "https://export.arxiv.org/api/query",
                "status": "available",
                "query": topic,
                "verified_at": _utcnow_iso(),
            },
            {
                "id": "semantic_scholar",
                "name": "Semantic Scholar",
                "type": "api",
                "url": "https://api.semanticscholar.org/graph/v1/paper/search",
                "status": "available",
                "query": topic,
                "verified_at": _utcnow_iso(),
            },
        ]
    if config.openclaw_bridge.use_web_fetch:
        for src in sources:
            try:
                response = adapters.web_fetch.fetch(str(src.get("url", "")))
                src["status"] = (
                    "verified"
                    if response.status_code in (200, 301, 302, 405)
                    else "unreachable"
                )
                src["http_status"] = response.status_code
            except Exception:  # noqa: BLE001
                src["status"] = "unknown"
    (stage_dir / "search_plan.yaml").write_text(
        yaml.dump(plan, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    (stage_dir / "sources.json").write_text(
        json.dumps(
            {"sources": sources, "count": len(sources), "generated": _utcnow_iso()},
            indent=2,
        ),
        encoding="utf-8",
    )

    # F1.5: Extract queries from plan for Stage 4 real literature search
    # The LLM's own queries (plan, then its sources) beat queries rebuilt from
    # the topic; a single YAML typo must not throw them all away.
    queries_list = _collect_plan_queries(llm_plan, llm_sources) or _collect_plan_queries(plan, [])
    year_min = 2020
    if isinstance(plan, dict):
        filters = plan.get("filters", {})
        if isinstance(filters, dict) and filters.get("min_year"):
            try:
                year_min = int(filters["min_year"])
            except (ValueError, TypeError):
                pass

    # --- Sanitize queries: shorten overly long queries ---
    # LLMs often produce the full topic title as a query, which is too long for
    # arXiv and Semantic Scholar (they work best with 3-8 keyword queries).
    _stop = {
        "a", "an", "the", "of", "for", "in", "on", "and", "or", "with",
        "to", "by", "from", "its", "is", "are", "was", "be", "as", "at",
        "via", "using", "based", "study", "analysis", "empirical",
        "towards", "toward", "into", "exploring", "comparison", "tasks",
        "effectiveness", "investigation", "comprehensive", "novel",
    }

    def _extract_keywords(text: str) -> list[str]:
        """Extract meaningful keywords from text, removing stop words.

        Handles mixed Chinese-English text by also extracting English tokens
        and translating Chinese domain terms.
        """
        en_words = [
            w for w in re.split(r"[^a-zA-Z0-9]+", text)
            if w.lower() not in _stop and len(w) > 1
        ]
        if re.search(r"[\u4e00-\u9fff]", text):
            en_words.extend(_extract_english_from_mixed(text))
            en_words = list(dict.fromkeys(w for w in en_words if w))
        return en_words

    _MAX_QUERY_LEN = 60  # characters — beyond this, shorten to keywords
    _SEARCH_SUFFIXES = ["benchmark", "survey", "seminal", "state of the art"]

    def _shorten_query(q: str, max_kw: int = 6) -> str:
        """Shorten a query to *max_kw* keywords, preserving any trailing suffix."""
        q_stripped = q.strip()
        # Check if query ends with a known search suffix
        suffix = ""
        q_core = q_stripped
        for sfx in _SEARCH_SUFFIXES:
            if q_stripped.lower().endswith(sfx):
                suffix = sfx
                q_core = q_stripped[: -len(sfx)].strip()
                break
        # Extract keywords from the core part
        kws = _extract_keywords(q_core)
        shortened = " ".join(kws[:max_kw])
        if suffix:
            shortened = f"{shortened} {suffix}"
        return shortened

    if queries_list:
        sanitized: list[str] = []
        for q in queries_list:
            if len(q) > _MAX_QUERY_LEN:
                shortened = _shorten_query(q)
                if shortened.strip():
                    sanitized.append(shortened)
            else:
                sanitized.append(q)
        queries_list = sanitized

    if not queries_list:
        # Build diverse keyword queries from the topic
        _words = _extract_keywords(topic)
        kw_primary = " ".join(_words[:6])
        kw_short = " ".join(_words[:4])
        queries_list = [
            kw_primary,
            f"{kw_short} benchmark",
            f"{kw_short} survey",
        ]

    # Ensure minimum query diversity — if dedup leaves too few, add variants
    _all_kw = _extract_keywords(topic)
    _seen_q: set[str] = set()
    unique_queries: list[str] = []
    for q in queries_list:
        q_lower = q.strip().lower()
        if q_lower and q_lower not in _seen_q:
            _seen_q.add(q_lower)
            unique_queries.append(q.strip())
    # If we have fewer than 5 unique queries, generate supplemental keyword variants
    if len(unique_queries) < 5 and len(_all_kw) >= 3:
        supplements = [
            " ".join(_all_kw[:4]) + " survey",
            " ".join(_all_kw[:4]) + " benchmark",
            " ".join(_all_kw[1:5]),  # shifted window for diversity
            " ".join(_all_kw[:3]) + " comparison",
            " ".join(_all_kw[:3]) + " deep learning",
            " ".join(_all_kw[2:6]),  # another shifted window
        ]
        for s in supplements:
            s_lower = s.strip().lower()
            if s_lower not in _seen_q:
                _seen_q.add(s_lower)
                unique_queries.append(s.strip())
            if len(unique_queries) >= 8:
                break
    queries_list = unique_queries
    (stage_dir / "queries.json").write_text(
        json.dumps({"queries": queries_list, "year_min": year_min}, indent=2),
        encoding="utf-8",
    )
    return StageResult(
        stage=Stage.SEARCH_STRATEGY,
        status=StageStatus.DONE,
        artifacts=("search_plan.yaml", "sources.json", "queries.json"),
        evidence_refs=(
            "stage-03/search_plan.yaml",
            "stage-03/sources.json",
            "stage-03/queries.json",
        ),
    )


def _expand_search_queries(queries: list[str], topic: str) -> list[str]:
    """Expand search queries for broader literature coverage.

    Generates additional queries by extracting key phrases from the topic
    and creating focused sub-queries. This ensures we find papers even when
    the original queries are too narrow or specific for arXiv.
    """
    expanded = list(queries)  # keep originals
    seen = {q.lower().strip() for q in queries}

    # Extract key phrases from topic by splitting on common delimiters
    # e.g. "Comparing A, B, and C on X with Y" → ["A", "B", "C", "X", "Y"]
    topic_words = topic.split()

    # Generate shorter, broader queries from the topic
    if len(topic_words) > 5:
        # First 5 words as a broader query
        broad = " ".join(topic_words[:5])
        if broad.lower().strip() not in seen:
            expanded.append(broad)
            seen.add(broad.lower().strip())

        # Last 5 words as another perspective
        tail = " ".join(topic_words[-5:])
        if tail.lower().strip() not in seen:
            expanded.append(tail)
            seen.add(tail.lower().strip())

    # Add "survey" and "benchmark" variants of the topic
    for suffix in ("survey", "benchmark", "comparison"):
        # Take first 4 content words + suffix
        short_topic = " ".join(topic_words[:4])
        variant = f"{short_topic} {suffix}"
        if variant.lower().strip() not in seen:
            expanded.append(variant)
            seen.add(variant.lower().strip())

    return expanded[:6]


def _check_literature_sites_reachable(timeout: float = 4.0) -> dict[str, bool]:
    """Quick connectivity check for literature API endpoints (parallel)."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, wait

    sites = {
        "openalex": "https://api.openalex.org/works?filter=title.search:test&per_page=1",
        "semantic_scholar": "https://api.semanticscholar.org/graph/v1/paper/search?query=test&limit=1",
        "arxiv": "https://export.arxiv.org/api/query?search_query=test&max_results=1",
    }

    def _probe(name: str, url: str) -> tuple[str, bool]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ResearchClaw/0.3"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return name, resp.status < 500
        except Exception as exc:
            logger.warning("Literature connectivity probe failed (%s): %s", name, type(exc).__name__)
            return name, False

    results: dict[str, bool] = {}
    pool = ThreadPoolExecutor(max_workers=len(sites))
    try:
        futures = {pool.submit(_probe, n, u): n for n, u in sites.items()}
        completed, pending = wait(futures, timeout=timeout + 2)
        for fut in completed:
            try:
                name, ok = fut.result()
                results[name] = ok
            except Exception:
                results[futures[fut]] = False
        if pending:
            logger.warning(
                "Literature connectivity checks exceeded %.1fs; skipping pending sources: %s",
                timeout + 2, ", ".join(sorted(futures[future] for future in pending)),
            )
        for future in pending:
            future.cancel()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    for name in sites:
        results.setdefault(name, False)
    return results


def _build_local_pdf_reference_entry(
    pdf_path: Path,
    *,
    title: str,
    abstract: str,
    source: str,
) -> dict[str, Any]:
    clean_title = title.strip() or pdf_path.stem
    clean_abstract = re.sub(r"\s+", " ", abstract).strip()[:1200]
    return {
        "id": f"user-ref-{_safe_filename(pdf_path.stem.lower())}",
        "title": clean_title,
        "source": source,
        "url": pdf_path.resolve().as_uri(),
        "year": datetime.now(timezone.utc).year,
        "abstract": clean_abstract,
        "authors": [],
        "collected_at": _utcnow_iso(),
    }


def _extract_full_pdf_text(pdf_path: Path, max_chars: int = 80_000) -> str:
    """Extract full text from a local PDF for downstream stages (S9, S11).

    Uses the PDFExtractor utility which reads all pages.  Falls back to
    the simpler fitz/pypdf page loop if the extractor is unavailable.
    Returns empty string on failure.
    """
    try:
        from researchclaw.web.pdf_extractor import PDFExtractor
        result = PDFExtractor(max_pages=0, extract_sections=False).extract(pdf_path)
        if result.success and result.text:
            return result.text[:max_chars]
    except Exception:  # noqa: BLE001
        pass

    try:
        import fitz
        doc = fitz.open(pdf_path)
        try:
            pages = []
            for idx in range(doc.page_count):
                text = doc.load_page(idx).get_text("text")
                if text:
                    pages.append(text)
            return "\n".join(pages)[:max_chars]
        finally:
            doc.close()
    except Exception:  # noqa: BLE001
        pass

    return ""


def _extract_local_pdf_reference(pdf_path: Path) -> dict[str, Any]:
    title = pdf_path.stem
    abstract = ""
    parser_errors: list[str] = []

    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            metadata = getattr(doc, "metadata", {}) or {}
            title = str(metadata.get("title") or title).strip() or title
            snippets: list[str] = []
            page_count = int(getattr(doc, "page_count", 0) or 0)
            for idx in range(min(page_count, 3)):
                page = doc.load_page(idx)
                text = page.get_text("text") if hasattr(page, "get_text") else ""
                text = re.sub(r"\s+", " ", text or "").strip()
                if text:
                    snippets.append(text)
            abstract = " ".join(snippets)
        finally:
            doc.close()
        return _build_local_pdf_reference_entry(
            pdf_path,
            title=title,
            abstract=abstract,
            source="user_reference_local_pdf",
        )
    except Exception as exc:  # noqa: BLE001
        parser_errors.append(f"PyMuPDF: {exc}")

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        metadata = getattr(reader, "metadata", None) or {}
        raw_title = getattr(metadata, "title", None)
        if not raw_title and isinstance(metadata, dict):
            raw_title = metadata.get("/Title")
        title = str(raw_title or title).strip() or title
        snippets = []
        for page in reader.pages[:3]:
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            if text:
                snippets.append(text)
        abstract = " ".join(snippets)
        return _build_local_pdf_reference_entry(
            pdf_path,
            title=title,
            abstract=abstract,
            source="user_reference_local_pdf",
        )
    except Exception as exc:  # noqa: BLE001
        parser_errors.append(f"pypdf: {exc}")

    logger.warning(
        "Failed to parse local PDF reference %s; falling back to weak entry (%s)",
        pdf_path,
        "; ".join(parser_errors) or "unknown parser error",
    )
    return _build_local_pdf_reference_entry(
        pdf_path,
        title=pdf_path.stem,
        abstract="",
        source="user_reference_local_pdf_parse_failed",
    )


def _resolve_user_reference(ref: str) -> dict | None:
    """Resolve a user-provided reference string to a candidate dict.

    Accepts arXiv IDs/URLs, publisher URLs, local PDFs, or plain titles.
    Unresolved URLs stay unresolved; URL slugs are never paper titles.
    """
    import re as _re_ref

    normalized_ref = ref.strip()
    if not normalized_ref:
        return None

    arxiv_id = ""
    m = _re_ref.search(r"(\d{4}\.\d{4,5})", normalized_ref)
    if m:
        arxiv_id = m.group(1)

    if arxiv_id:
        try:
            from researchclaw.literature.arxiv_client import get_paper_by_id
            paper = get_paper_by_id(arxiv_id)
            if paper:
                return {
                    "id": f"user-ref-{paper.paper_id}",
                    "title": paper.title,
                    "source": "user_reference",
                    "url": paper.url,
                    "year": paper.year,
                    "abstract": paper.abstract[:500] if paper.abstract else "",
                    "authors": [{"name": a.name} for a in paper.authors],
                    "arxiv_id": arxiv_id,
                    "venue": paper.venue,
                    "collected_at": _utcnow_iso(),
                }
        except Exception:  # noqa: BLE001
            pass

    pdf_path = Path(normalized_ref).expanduser()
    if pdf_path.is_file() and pdf_path.suffix.lower() == ".pdf":
        return _extract_local_pdf_reference(pdf_path)

    if normalized_ref.startswith(("https://", "http://")):
        from researchclaw.literature.publisher import resolve_publisher_reference
        paper = resolve_publisher_reference(normalized_ref)
        if paper is None:
            return None
        result = paper.to_dict()
        result["bibtex"] = paper.to_bibtex()
        result["collected_at"] = _utcnow_iso()
        return result
    title = normalized_ref
    return {
        "id": f"user-ref-{title[:40].replace(' ', '-').lower()}",
        "title": title,
        "source": "user_reference",
        "url": normalized_ref if normalized_ref.startswith("http") else "",
        "year": 0,
        "abstract": "",
        "authors": [],
        "collected_at": _utcnow_iso(),
    }


def _execute_literature_collect(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    """Stage 4: Collect real literature, retaining completed searches on timeout."""
    topic = config.research.topic

    # Pre-flight: check if any literature site is reachable
    site_status = _check_literature_sites_reachable(timeout=4.0)
    (stage_dir / "site_status.json").write_text(
        json.dumps({"sources": site_status, "checked_at": _utcnow_iso()}, indent=2), encoding="utf-8",
    )
    reachable = [name for name, ok in site_status.items() if ok]
    unreachable = [name for name, ok in site_status.items() if not ok]
    if unreachable:
        logger.warning(
            "Literature sites unreachable (will be skipped): %s — reachable: %s",
            ", ".join(unreachable), ", ".join(reachable) or "none",
        )
    if not reachable:
        logger.warning(
            "ALL literature sites unreachable (%s) — literature collection needs a retry",
            ", ".join(unreachable),
        )
        failure_note = (
            f"Literature connectivity checks failed or timed out for all sources "
            f"({', '.join(unreachable)}). Check server network/DNS and retry S4. "
            f"No literature search was completed."
        )
        (stage_dir / "candidates.jsonl").write_text("", encoding="utf-8")
        (stage_dir / "literature_error.txt").write_text(failure_note, encoding="utf-8")
        return StageResult(
            stage=Stage.LITERATURE_COLLECT,
            status=StageStatus.FAILED,
            artifacts=("candidates.jsonl", "site_status.json", "literature_error.txt"),
            evidence_refs=("stage-04/site_status.json", "stage-04/literature_error.txt"),
            decision="retry",
            error=failure_note,
        )

    # Read queries.json from Stage 3 (F1.5 output)
    queries_text = _read_prior_artifact(run_dir, "queries.json")
    queries_data = _safe_json_loads(queries_text or "{}", {})
    queries: list[str] = queries_data.get("queries", [topic])
    year_min: int = queries_data.get("year_min", 2020)

    # --- Try real API search first ---
    candidates: list[dict[str, Any]] = []
    bibtex_entries: list[str] = []
    real_search_succeeded = False

    _SEARCH_TIMEOUT_SEC = 60  # hard cap on API search phase

    try:
        from researchclaw.literature.bounded_search import bounded_search

        # Expand queries for broader coverage
        expanded_queries = _expand_search_queries(queries, config.research.topic)
        reachable_sources = tuple(reachable) if reachable else ("openalex", "semantic_scholar", "arxiv")
        logger.info(
            "[literature] Searching %d queries (expanded from %d) "
            "across %s (skipped unreachable: %s) — timeout %ds",
            len(expanded_queries),
            len(queries),
            " → ".join(reachable_sources),
            ", ".join(unreachable) if unreachable else "none",
            _SEARCH_TIMEOUT_SEC,
        )

        papers = bounded_search(
            expanded_queries, output_dir=stage_dir / "api_search",
            limit_per_query=40, sources=reachable_sources, year_min=year_min,
            s2_api_key=config.llm.s2_api_key, timeout=_SEARCH_TIMEOUT_SEC,
        )

        if papers:
            real_search_succeeded = True
            src_counts: dict[str, int] = {}
            for p in papers:
                src_counts[p.source] = src_counts.get(p.source, 0) + 1
                d = p.to_dict()
                d["collected_at"] = _utcnow_iso()
                candidates.append(d)
                bibtex_entries.append(p.to_bibtex())
            src_str = ", ".join(f"{s}: {n}" for s, n in src_counts.items())
            logger.info(
                "[literature] Found %d papers (%s)", len(papers), src_str
            )
    except Exception:  # noqa: BLE001
        logger.warning(
            "[rate-limit] Literature search failed — retaining resolved references only",
            exc_info=True,
        )

    # Foundational references are background, never substitutes for domain search.
    from researchclaw.data import load_seminal_papers
    background = load_seminal_papers(topic)
    (stage_dir / "background_references.json").write_text(
        json.dumps(background, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # --- Inject user-provided reference papers ---
    _user_refs = config.research.reference_papers
    unresolved_references: list[str] = []
    if _user_refs:
        _existing_titles = {c.get("title", "").lower() for c in candidates}
        _ref_injected = 0
        _ref_full_texts: list[str] = []
        for ref_str in _user_refs:
            ref_str = ref_str.strip()
            if not ref_str:
                continue
            _resolved = _resolve_user_reference(ref_str)
            if not _resolved:
                unresolved_references.append(ref_str)
            if _resolved and _resolved.get("title", "").lower() not in _existing_titles:
                _existing_titles.add(_resolved["title"].lower())
                candidates.append(_resolved)
                if _resolved.get("bibtex"):
                    bibtex_entries.append(_resolved["bibtex"])
                _ref_injected += 1

            # Extract full text for local PDFs (used by S9 experiment design)
            _pdf_path = Path(ref_str).expanduser()
            if _pdf_path.is_file() and _pdf_path.suffix.lower() == ".pdf":
                _full_text = _extract_full_pdf_text(_pdf_path)
                if _full_text:
                    _ref_title = (_resolved or {}).get("title", _pdf_path.stem)
                    _ref_full_texts.append(
                        f"# {_ref_title}\n\n{_full_text}"
                    )

        if _ref_full_texts:
            _combined = "\n\n---\n\n".join(_ref_full_texts)
            (stage_dir / "reference_paper_text.md").write_text(
                _combined, encoding="utf-8",
            )
            logger.info(
                "Stage 4: Extracted full text from %d reference PDF(s) "
                "(%d chars total)",
                len(_ref_full_texts), len(_combined),
            )

        if _ref_injected:
            logger.info("Stage 4: Injected %d user-provided reference papers", _ref_injected)
    (stage_dir / "unresolved_references.json").write_text(
        json.dumps(unresolved_references, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # --- arXiv recent papers auto-discovery (30s timeout) ---
    _ARXIV_DISC_TIMEOUT = 30
    try:
        from researchclaw.literature.bounded_search import bounded_search
        _topic_kw = _extract_keywords(topic)
        if _topic_kw:
            _arxiv_query = " AND ".join(
                f"all:{w}" for w in _topic_kw[:5]
            )

            _recent = bounded_search(
                [_arxiv_query], output_dir=stage_dir / "recent_search", sources=("arxiv",),
                limit_per_query=30, sort_by="submitted_date", year_min=2025,
                timeout=_ARXIV_DISC_TIMEOUT,
            ) if "arxiv" in reachable else []

            _existing_titles_lower = {c.get("title", "").lower() for c in candidates}
            _disc = 0
            for p in _recent:
                if p.title.lower() not in _existing_titles_lower:
                    _existing_titles_lower.add(p.title.lower())
                    candidates.append({
                        "id": p.paper_id,
                        "title": p.title,
                        "source": "arxiv_recent_discovery",
                        "url": p.url,
                        "year": p.year,
                        "abstract": p.abstract[:500] if p.abstract else "",
                        "authors": [{"name": a.name} for a in p.authors],
                        "arxiv_id": p.arxiv_id,
                        "venue": p.venue,
                        "collected_at": _utcnow_iso(),
                    })
                    _disc += 1
            if _disc:
                logger.info("Stage 4: Discovered %d recent arXiv papers (2025+)", _disc)
    except Exception:  # noqa: BLE001
        logger.debug("arXiv recent discovery skipped", exc_info=True)

    # --- Web search augmentation (Tavily/DDG + Google Scholar + Crawl4AI) — 60s timeout ---
    _WEB_SEARCH_TIMEOUT = 60
    web_context_parts: list[str] = []
    if config.web_search.enabled:
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
            from researchclaw.web.agent import WebSearchAgent
            import os

            tavily_key = config.web_search.tavily_api_key or os.environ.get(
                config.web_search.tavily_api_key_env, ""
            )
            exa_key = config.web_search.exa_api_key or os.environ.get(
                config.web_search.exa_api_key_env, ""
            )
            web_agent = WebSearchAgent(
                tavily_api_key=tavily_key,
                exa_api_key=exa_key,
                enable_scholar=config.web_search.enable_scholar,
                enable_crawling=config.web_search.enable_crawling,
                enable_pdf=config.web_search.enable_pdf_extraction,
                max_web_results=config.web_search.max_web_results,
                max_scholar_results=config.web_search.max_scholar_results,
                max_crawl_urls=config.web_search.max_crawl_urls,
            )

            with ThreadPoolExecutor(max_workers=1) as _pool:
                _fut = _pool.submit(
                    web_agent.search_and_extract,
                    topic, search_queries=queries,
                )
                try:
                    web_result = _fut.result(timeout=_WEB_SEARCH_TIMEOUT)
                except FuturesTimeout:
                    logger.warning(
                        "[web-search] Timed out after %ds — skipping",
                        _WEB_SEARCH_TIMEOUT,
                    )
                    web_result = None

            if web_result is None:
                raise RuntimeError("web search timed out")

            # Convert Google Scholar papers into candidates
            for sp in web_result.scholar_papers:
                _existing_titles = {
                    str(c.get("title", "")).lower().strip() for c in candidates
                }
                if sp.title.lower().strip() not in _existing_titles:
                    lit_paper = sp.to_literature_paper()
                    d = lit_paper.to_dict()
                    d["collected_at"] = _utcnow_iso()
                    candidates.append(d)
                    bibtex_entries.append(lit_paper.to_bibtex())

            # Save web search context for downstream stages
            web_context = web_result.to_context_string(max_length=20_000)
            if web_context.strip():
                (stage_dir / "web_context.md").write_text(
                    web_context, encoding="utf-8"
                )
                web_context_parts.append(web_context)

            # Save full web search metadata
            (stage_dir / "web_search_result.json").write_text(
                json.dumps(web_result.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )

            logger.info(
                "[web-search] Added %d scholar papers, %d web results, %d crawled pages",
                len(web_result.scholar_papers),
                len(web_result.web_results),
                len(web_result.crawled_pages),
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[web-search] Web search augmentation failed — continuing with academic APIs only",
                exc_info=True,
            )

    # Never fabricate papers to make a failed search look successful.
    online_sources = {"openalex", "semantic_scholar", "arxiv", "arxiv_recent_discovery", "google_scholar"}
    source_counts: dict[str, int] = {}
    for candidate in candidates:
        source = str(candidate.get("source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
    real_search_succeeded = any(source_counts.get(name, 0) for name in online_sources)
    if not candidates:
        note = "No verifiable literature candidates retrieved. Check queries/network and retry S4."
        (stage_dir / "candidates.jsonl").write_text("", encoding="utf-8")
        (stage_dir / "literature_error.txt").write_text(note, encoding="utf-8")
        (stage_dir / "search_meta.json").write_text(json.dumps({
            "real_search": False, "source_counts": source_counts,
            "background_count": len(background), "total_candidates": 0,
        }), encoding="utf-8")
        return StageResult(stage=Stage.LITERATURE_COLLECT, status=StageStatus.FAILED,
                           artifacts=("candidates.jsonl", "literature_error.txt", "search_meta.json",
                                      "background_references.json"), decision="retry", error=note)

    # Write candidates
    out = stage_dir / "candidates.jsonl"
    _write_jsonl(out, candidates)

    # BUG-50 fix: Generate BibTeX from candidates when real search failed
    # (LLM/placeholder fallback paths don't populate bibtex_entries)
    if not bibtex_entries and candidates:
        for c in candidates:
            if c.get("is_placeholder"):
                continue
            _ck = c.get("cite_key", "")
            if not _ck:
                # Derive cite_key from first author surname + year
                _authors = c.get("authors", [])
                _surname = "unknown"
                if isinstance(_authors, list) and _authors:
                    _a0 = _authors[0] if isinstance(_authors[0], str) else (_authors[0].get("name", "") if isinstance(_authors[0], dict) else "")
                    _surname = _a0.split()[-1].lower() if _a0.strip() else "unknown"
                _yr = c.get("year", 2024)
                _title_word = "".join(
                    w[0] for w in str(c.get("title", "study")).split()[:3]
                ).lower()
                _ck = f"{_surname}{_yr}{_title_word}"
            _title = c.get("title", "Untitled")
            _year = c.get("year", 2024)
            _author_str = ""
            _raw_authors = c.get("authors", [])
            if isinstance(_raw_authors, list):
                _names = []
                for _a in _raw_authors:
                    if isinstance(_a, str):
                        _names.append(_a)
                    elif isinstance(_a, dict):
                        _names.append(_a.get("name", ""))
                _author_str = " and ".join(n for n in _names if n)
            bibtex_entries.append(
                f"@article{{{_ck},\n"
                f"  title={{{_title}}},\n"
                f"  author={{{_author_str or 'Unknown'}}},\n"
                f"  year={{{_year}}},\n"
                f"  url={{{c.get('url', '')}}},\n"
                f"}}"
            )
        logger.info(
            "Stage 4: Generated %d BibTeX entries from candidates (fallback)",
            len(bibtex_entries),
        )

    # Write references.bib (F2.4)
    artifacts = ["candidates.jsonl", "site_status.json", "background_references.json"]
    if (stage_dir / "reference_paper_text.md").exists():
        artifacts.append("reference_paper_text.md")
    if web_context_parts:
        artifacts.append("web_context.md")
    if (stage_dir / "web_search_result.json").exists():
        artifacts.append("web_search_result.json")
    if bibtex_entries:
        bib_content = "\n\n".join(bibtex_entries) + "\n"
        (stage_dir / "references.bib").write_text(
            _escape_bib_specials(bib_content), encoding="utf-8"
        )
        artifacts.append("references.bib")
        logger.info(
            "Stage 4: Wrote %d BibTeX entries to references.bib", len(bibtex_entries)
        )

    # Write search metadata
    (stage_dir / "search_meta.json").write_text(
        json.dumps(
            {
                "real_search": real_search_succeeded,
                "source_counts": source_counts,
                "background_count": len(background),
                "queries_used": queries,
                "year_min": year_min,
                "total_candidates": len(candidates),
                "bibtex_entries": len(bibtex_entries),
                "ts": _utcnow_iso(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts.append("search_meta.json")

    return StageResult(
        stage=Stage.LITERATURE_COLLECT,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-04/{a}" for a in artifacts),
    )


def _execute_literature_screen(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    candidates_text = _read_prior_artifact(run_dir, "candidates.jsonl") or ""

    def fail_screen(note: str) -> StageResult:
        (stage_dir / "shortlist.jsonl").write_text("", encoding="utf-8")
        (stage_dir / "screening_error.txt").write_text(note, encoding="utf-8")
        return StageResult(stage=Stage.LITERATURE_SCREEN, status=StageStatus.FAILED,
                           artifacts=("shortlist.jsonl", "screening_error.txt"),
                           decision="retry", error=note)

    if not candidates_text.strip():
        return fail_screen("No literature candidates. Retry S4 before screening.")

    # --- P1-1: keyword relevance pre-filter ---
    # Before LLM screening, drop papers whose title+abstract share no keywords
    # with the research topic.  This catches cross-domain noise cheaply.
    topic_keywords = _extract_topic_keywords(
        config.research.topic, config.research.domains
    )
    filtered_rows: list[dict[str, Any]] = []
    dropped_count = 0
    for raw_line in candidates_text.strip().splitlines():
        row = _safe_json_loads(raw_line, {})
        if not isinstance(row, dict) or row.get("is_placeholder") or row.get("source") == "seminal_library":
            continue
        title = str(row.get("title", "")).lower()
        abstract = str(row.get("abstract", "")).lower()
        text_blob = f"{title} {abstract}"
        overlap = sum(1 for kw in topic_keywords if re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", text_blob))
        # T2.2: Relaxed from ≥2 to ≥1 keyword hit — previous threshold was
        # too aggressive (94% rejection rate).  Single-keyword matches are
        # still screened by the LLM in the next step.
        if overlap >= 1:
            row["keyword_overlap"] = overlap
            filtered_rows.append(row)
        else:
            dropped_count += 1
    if not filtered_rows:
        return fail_screen("No topic-relevant candidates remain. Revise queries and retry S4; unrelated papers are not restored.")
    logger.info(
        "Domain pre-filter: kept %d, dropped %d (keywords: %s)",
        len(filtered_rows),
        dropped_count,
        topic_keywords[:8],
    )

    shortlist: list[dict[str, Any]] = []
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "literature_screen")
        # The full topic also contains execution policies. Keep the scientific
        # focus, and assess every candidate in bounded batches rather than
        # sending hundreds of abstracts in a single gateway request.
        def make_prompt(rows: list[dict[str, Any]]) -> Any:
            return _pm.for_stage(
                "literature_screen", evolution_overlay=_overlay,
                topic=config.research.topic[:3500],
                domains=", ".join(config.research.domains)[:500] or "general",
                quality_threshold=config.research.quality_threshold,
                candidates_text="\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
            )

        batches: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        fields = ("paper_id", "id", "title", "abstract", "year", "venue", "citation_count", "doi", "url", "source", "cite_key")
        for original in filtered_rows:
            compact = {key: (value[:1400] if key == "abstract" else value[:500])
                       if isinstance(value, str) else value
                       for key in fields if (value := original.get(key)) is not None}
            trial = make_prompt([*current, compact])
            if current and (len(current) >= 6 or len(trial.system) + len(trial.user) > 12000):
                batches.append(current)
                current = []
            single = make_prompt([compact])
            if len(single.system) + len(single.user) > 12000:
                return fail_screen("Screening prompt instructions exceed 12000 characters even for one paper; shorten the custom prompt and retry S5.")
            current.append(compact)
        if current:
            batches.append(current)
        for index, batch in enumerate(batches, 1):
            sp = make_prompt(batch)
            (stage_dir / f"screening_batch_{index:03d}.json").write_text(
                json.dumps({"topic_focus": config.research.topic[:3500], "candidates": batch,
                            "prompt_chars": len(sp.system) + len(sp.user)}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            resp = _chat_with_prompt(llm, sp.system, sp.user,
                                     json_mode=sp.json_mode, max_tokens=sp.max_tokens)
            (stage_dir / f"screening_response_{index:03d}.json").write_text(resp.content, encoding="utf-8")
            payload = _safe_json_loads(resp.content, {})
            if not isinstance(payload, dict) or not isinstance(payload.get("shortlist"), list):
                return fail_screen(f"Invalid screening response in batch {index}; retry S5 without inventing assessments.")
            batch_titles = {str(row.get("title", "")).strip().casefold() for row in batch}
            shortlist.extend(row for row in payload["shortlist"] if isinstance(row, dict)
                             and str(row.get("title", "")).strip().casefold() in batch_titles)
    # Accept only assessed papers from the actual filtered input, never invented
    # entries or a fixed-size supplement. A small genuine shortlist is valid.
    by_title = {str(row.get("title", "")).strip().casefold(): row for row in filtered_rows}
    accepted = []
    seen = set()
    for assessment in shortlist:
        key = str(assessment.get("title", "")).strip().casefold()
        if key not in by_title or key in seen:
            continue
        try:
            relevance = float(assessment["relevance_score"])
        except (KeyError, ValueError, TypeError):
            continue
        if not 0.5 <= relevance <= 1.0:
            continue
        accepted.append({**by_title[key], **{k: assessment[k] for k in
                         ("relevance_score", "quality_score", "keep_reason") if k in assessment}})
        seen.add(key)
    shortlist = accepted
    if not shortlist:
        return fail_screen("No candidates passed relevance assessment. Retry screening or revise S4 queries; no template scores were assigned.")
    out = stage_dir / "shortlist.jsonl"
    _write_jsonl(out, shortlist)
    return StageResult(
        stage=Stage.LITERATURE_SCREEN,
        status=StageStatus.DONE,
        artifacts=("shortlist.jsonl",),
        evidence_refs=("stage-05/shortlist.jsonl",),
    )


def _execute_knowledge_extract(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    shortlist = _read_prior_artifact(run_dir, "shortlist.jsonl") or ""

    # Inject web context from Stage 4 if available
    web_context = _read_prior_artifact(run_dir, "web_context.md") or ""
    if web_context:
        shortlist = shortlist + "\n\n--- Web Search Context ---\n" + web_context[:10_000]

    cards_dir = stage_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, Any]] = []
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "knowledge_extract")
        sp = _pm.for_stage("knowledge_extract", evolution_overlay=_overlay, shortlist=shortlist)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict) and isinstance(payload.get("cards"), list):
            cards = [item for item in payload["cards"] if isinstance(item, dict)]
    if not cards:
        rows = _parse_jsonl_rows(shortlist)
        for idx, paper in enumerate(rows[:6]):
            title = str(paper.get("title", f"Paper {idx + 1}"))
            cards.append(
                {
                    "card_id": f"card-{idx + 1}",
                    "title": title,
                    "problem": f"How to improve {config.research.topic}",
                    "method": "Template method summary",
                    "data": "Template dataset",
                    "metrics": "Template metric",
                    "findings": "Template key finding",
                    "limitations": "Template limitation",
                    "citation": str(paper.get("url", "")),
                    "cite_key": str(paper.get("cite_key", "")),
                }
            )
    for idx, card in enumerate(cards):
        card_id = _safe_filename(str(card.get("card_id", f"card-{idx + 1}")))
        parts = [f"# {card.get('title', card_id)}", ""]
        for key in (
            "cite_key",
            "problem",
            "method",
            "data",
            "metrics",
            "findings",
            "limitations",
            "citation",
        ):
            parts.append(f"## {key.title()}")
            parts.append(str(card.get(key, "")))
            parts.append("")
        (cards_dir / f"{card_id}.md").write_text("\n".join(parts), encoding="utf-8")
    return StageResult(
        stage=Stage.KNOWLEDGE_EXTRACT,
        status=StageStatus.DONE,
        artifacts=("cards/",),
        evidence_refs=("stage-06/cards/",),
    )


def _execute_synthesis(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    cards_path = _read_prior_artifact(run_dir, "cards/") or ""
    cards_context = ""
    if cards_path:
        snippets: list[str] = []
        for path in sorted(Path(cards_path).glob("*.md"))[:24]:
            snippets.append(path.read_text(encoding="utf-8"))
        cards_context = "\n\n".join(snippets)
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "synthesis")
        sp = _pm.for_stage(
            "synthesis",
            evolution_overlay=_overlay,
            topic=config.research.topic,
            cards_context=cards_context,
        )
        resp = llm.chat(
            [{"role": "user", "content": sp.user}],
            system=sp.system,
            max_tokens=sp.max_tokens or 8192,
        )
        synthesis_md = resp.content
    else:
        synthesis_md = f"""# Synthesis

## Cluster Overview
- Cluster A: Representation methods
- Cluster B: Training strategies
- Cluster C: Evaluation robustness

## Gap 1
Limited consistency across benchmark protocols.

## Gap 2
Under-reported failure behavior under distribution shift.

## Prioritized Opportunities
1. Unified experimental protocol
2. Robustness-aware evaluation suite

## Generated
{_utcnow_iso()}
"""
    (stage_dir / "synthesis.md").write_text(synthesis_md, encoding="utf-8")
    return StageResult(
        stage=Stage.SYNTHESIS,
        status=StageStatus.DONE,
        artifacts=("synthesis.md",),
        evidence_refs=("stage-07/synthesis.md",),
    )


def _multi_perspective_generate(
    llm: LLMClient,
    roles: dict[str, dict[str, str]],
    variables: dict[str, str],
    perspectives_dir: Path,
) -> dict[str, str]:
    """Generate outputs from multiple debate perspectives.

    Each role has its own system/user prompt. Outputs are saved to
    *perspectives_dir* and returned as ``{role_name: response_text}``.
    """
    from researchclaw.prompts import _render  # noqa: PLC0415

    perspectives_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    for role_name, role_prompts in roles.items():
        try:
            system = _render(role_prompts["system"], variables)
            user = _render(role_prompts["user"], variables)
            resp = llm.chat(
                [{"role": "user", "content": user}],
                system=system,
            )
            results[role_name] = resp.content
            (perspectives_dir / f"{role_name}.md").write_text(
                resp.content, encoding="utf-8"
            )
            logger.info("Debate perspective '%s' generated (%d chars)", role_name, len(resp.content))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Debate perspective '%s' failed: %s", role_name, exc)
    if len(results) < 2:
        logger.error("Multi-perspective debate: only %d/%d roles succeeded", len(results), len(roles))
    return results


def _synthesize_perspectives(
    llm: LLMClient,
    perspectives: dict[str, str],
    sub_prompt_name: str,
    prompts: PromptManager,
) -> str:
    """Synthesize multiple perspective outputs into a unified result."""
    parts = []
    for role_name, text in perspectives.items():
        parts.append(f"### Perspective: {role_name}\n{text}")
    combined = "\n\n---\n\n".join(parts)
    sp = prompts.sub_prompt(sub_prompt_name, perspectives=combined)
    resp = llm.chat(
        [{"role": "user", "content": sp.user}],
        system=sp.system,
    )
    return resp.content


def _execute_hypothesis_gen(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    synthesis = _read_prior_artifact(run_dir, "synthesis.md") or ""

    # --- Inject prior knowledge from shared knowledge base ---
    _shared_dir = getattr(config.experiment, "shared_results_dir", "") or ""
    _prior_knowledge = ""
    if _shared_dir:
        _kb_index = Path(_shared_dir) / "knowledge_base" / "knowledge_index.jsonl"
        if _kb_index.exists():
            try:
                _kb_lines = _kb_index.read_text(encoding="utf-8").strip().split("\n")
                _kb_entries = []
                for _line in _kb_lines[-20:]:
                    _e = json.loads(_line)
                    _kb_entries.append(
                        f"- **{_e.get('topic', '?')}**: "
                        f"Conclusions: {'; '.join(_e.get('conclusions', [])[:3])}. "
                        f"Insights: {'; '.join(_e.get('insights', [])[:2])}. "
                        f"Directions: {'; '.join(_e.get('suggested_directions', [])[:2])}"
                    )
                if _kb_entries:
                    _prior_knowledge = (
                        "\n\n## PRIOR RESEARCH KNOWLEDGE (from completed projects)\n"
                        "Consider these findings when generating hypotheses. "
                        "Build upon successful approaches and avoid repeating failed ones.\n\n"
                        + "\n".join(_kb_entries) + "\n"
                    )
                    logger.info("S8: Injected %d prior knowledge entries", len(_kb_entries))
            except Exception:
                pass

    if llm is not None:
        _pm = prompts or PromptManager()
        from researchclaw.prompts import DEBATE_ROLES_HYPOTHESIS  # noqa: PLC0415

        # --- Multi-perspective debate ---
        perspectives_dir = stage_dir / "perspectives"
        variables = {"topic": config.research.topic, "synthesis": synthesis + _prior_knowledge}
        perspectives = _multi_perspective_generate(
            llm, DEBATE_ROLES_HYPOTHESIS, variables, perspectives_dir
        )
        # --- Synthesize into final hypotheses ---
        hypotheses_md = _synthesize_perspectives(
            llm, perspectives, "hypothesis_synthesize", _pm
        )
    else:
        hypotheses_md = _default_hypotheses(config.research.topic)
    (stage_dir / "hypotheses.md").write_text(hypotheses_md, encoding="utf-8")

    # --- Novelty check (non-blocking) ---
    novelty_artifacts: tuple[str, ...] = ()
    try:
        from researchclaw.literature.novelty import check_novelty  # noqa: PLC0415

        candidates_text = _read_prior_artifact(run_dir, "candidates.jsonl") or ""
        papers_seen = _parse_jsonl_rows(candidates_text) if candidates_text else []
        novelty_report = check_novelty(
            topic=config.research.topic,
            hypotheses_text=hypotheses_md,
            papers_already_seen=papers_seen,
            s2_api_key=getattr(config.llm, "s2_api_key", ""),
        )
        (stage_dir / "novelty_report.json").write_text(
            json.dumps(novelty_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        novelty_artifacts = ("novelty_report.json",)
        logger.info(
            "Novelty check: score=%.3f  assessment=%s  recommendation=%s",
            novelty_report["novelty_score"],
            novelty_report["assessment"],
            novelty_report["recommendation"],
        )
    except Exception:  # noqa: BLE001
        logger.warning("Novelty check failed (non-blocking)", exc_info=True)

    return StageResult(
        stage=Stage.HYPOTHESIS_GEN,
        status=StageStatus.DONE,
        artifacts=("hypotheses.md",) + novelty_artifacts,
        evidence_refs=("stage-08/hypotheses.md",),
    )


def _execute_experiment_design(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    hypotheses = _read_prior_artifact(run_dir, "hypotheses.md") or ""
    preamble = _build_context_preamble(
        config, run_dir, include_goal=True, include_hypotheses=True
    )
    plan: dict[str, Any] | None = None

    # These paths are also needed by the no-LLM fallback below.
    import os as _os_s9
    _datasets_dir_s9 = getattr(config.experiment, "datasets_dir", "") or ""
    _codebases_dir_s9 = getattr(config.experiment, "codebases_dir", "") or ""
    _checkpoints_dir_s9 = getattr(config.experiment, "checkpoints_dir", "") or ""

    # ── Load reference paper full text (produced by S4) ───────────────────
    _reference_paper_text = _read_prior_artifact(run_dir, "reference_paper_text.md") or ""
    if _reference_paper_text:
        _reference_paper_text = _reference_paper_text[:30_000]
        logger.info(
            "Stage 09: Found reference paper text (%d chars) — "
            "will inject into experiment design prompt",
            len(_reference_paper_text),
        )

    # ── Domain detection ──────────────────────────────────────────────────
    # Detect the research domain early so we can adapt experiment design
    # and code generation. For ML domains, existing behavior is unchanged.
    _domain_profile = None
    try:
        from researchclaw.domains.detector import detect_domain as _detect_domain_adv
        _domain_profile = _detect_domain_adv(
            topic=config.research.topic,
            hypotheses=hypotheses,
        )
        logger.info(
            "Domain detected: %s (%s)",
            _domain_profile.display_name,
            _domain_profile.domain_id,
        )
        # Persist domain profile for Stage 10
        import json as _json_dd
        (stage_dir / "domain_profile.json").write_text(
            _json_dd.dumps({
                "domain_id": _domain_profile.domain_id,
                "display_name": _domain_profile.display_name,
                "experiment_paradigm": _domain_profile.experiment_paradigm,
                "core_libraries": _domain_profile.core_libraries,
                "gpu_required": _domain_profile.gpu_required,
            }, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        logger.debug("Domain detection unavailable", exc_info=True)
    if llm is not None:
        _pm = prompts or PromptManager()
        # Pass dataset_guidance block for experiment design
        try:
            _dg_block = _pm.block("dataset_guidance")
        except (KeyError, Exception):  # noqa: BLE001
            _dg_block = ""
        # ── Inject project-specific local data paths into experiment design ──
        _local_data_parts: list[str] = []
        if _datasets_dir_s9 and _os_s9.path.isdir(_datasets_dir_s9):
            _ds_items = [d for d in sorted(_os_s9.listdir(_datasets_dir_s9)) if not d.startswith(".")]
            if _ds_items:
                _local_data_parts.append(
                    f"### LOCAL DATASETS (MUST USE — do NOT use synthetic data)\n"
                    f"Directory: `{_datasets_dir_s9}`\n"
                    f"Available: {', '.join(_ds_items)}\n"
                    f"In code: `DATASETS_DIR = '{_datasets_dir_s9}'`\n"
                    f"Design your experiment to use these REAL datasets. "
                    f"NEVER generate synthetic torch.randn() data when real data is available."
                )
                for _ds_name in _ds_items:
                    _ds_path = _os_s9.path.join(_datasets_dir_s9, _ds_name)
                    if _os_s9.path.isdir(_ds_path):
                        _ds_sub = []
                        for _root, _dirs, _files in _os_s9.walk(_ds_path):
                            _rel = _os_s9.path.relpath(_root, _ds_path)
                            for _f in _files[:5]:
                                _ds_sub.append(_os_s9.path.join(_rel, _f) if _rel != "." else _f)
                            if len(_ds_sub) > 20:
                                break
                        if _ds_sub:
                            _local_data_parts.append(
                                f"  Dataset `{_ds_name}` sample files: {', '.join(_ds_sub[:15])}"
                            )
        if _codebases_dir_s9 and _os_s9.path.isdir(_codebases_dir_s9):
            _cb_items = [d for d in sorted(_os_s9.listdir(_codebases_dir_s9))
                         if _os_s9.path.isdir(_os_s9.path.join(_codebases_dir_s9, d)) and not d.startswith(".")]
            if _cb_items:
                _local_data_parts.append(
                    f"\n### LOCAL CODEBASES (MUST BUILD ON TOP OF)\n"
                    f"Directory: `{_codebases_dir_s9}`\n"
                    f"Available: {', '.join(_cb_items)}\n"
                    f"**CRITICAL**: Your experiment MUST extend/wrap these existing codebases. "
                    f"Do NOT design a from-scratch implementation when a reference codebase exists."
                )
                for _cb_name in _cb_items:
                    _cb_path = _os_s9.path.join(_codebases_dir_s9, _cb_name)
                    try:
                        from researchclaw.utils.codebase_manifest import generate_manifest, manifest_to_prompt
                        _cb_manifest = generate_manifest(_cb_path)
                        _cb_prompt = manifest_to_prompt(_cb_manifest)
                        if len(_cb_prompt) > 6000:
                            _cb_prompt = _cb_prompt[:6000] + "\n  ... (truncated)"
                        _local_data_parts.append(_cb_prompt)
                    except Exception:  # noqa: BLE001
                        _local_data_parts.append(f"  Codebase `{_cb_name}` at `{_cb_path}`")
        if _checkpoints_dir_s9 and _os_s9.path.isdir(_checkpoints_dir_s9):
            _ck_items = [d for d in sorted(_os_s9.listdir(_checkpoints_dir_s9)) if not d.startswith(".")]
            if _ck_items:
                _local_data_parts.append(
                    f"\n### LOCAL CHECKPOINTS (pre-trained weights available)\n"
                    f"Directory: `{_checkpoints_dir_s9}`\n"
                    f"Available: {', '.join(_ck_items)}\n"
                    f"Use these checkpoints directly — do NOT plan to download them."
                )
        if _local_data_parts:
            _dg_block += (
                "\n\n## PROJECT-SPECIFIC LOCAL RESOURCES (HIGHEST PRIORITY)\n"
                + "\n".join(_local_data_parts)
                + "\n\nWhen designing the experiment, your `datasets` section MUST reference "
                "these local resources. The experiment code will have direct filesystem access "
                "to these paths.\n"
                "Your YAML MUST include a top-level `local_resources` mapping with "
                "`datasets_dir` equal to the exact local dataset path and a "
                "`usage_requirement` that says core experiments must use this local data.\n"
            )

        # I-08: Inject RL step guidance for RL topics
        _rl_kws = ("reinforcement learning", "ppo", "sac", "td3", "ddpg",
                    "dqn", "mujoco", "continuous control", "actor-critic",
                    "policy gradient", "exploration bonus")
        if any(kw in config.research.topic.lower() for kw in _rl_kws):
            try:
                _dg_block += _pm.block("rl_step_guidance")
            except Exception:  # noqa: BLE001
                pass
        # F-01: Inject framework docs for experiment design
        try:
            from researchclaw.data import detect_frameworks, load_framework_docs
            _fw_ids = detect_frameworks(config.research.topic, hypotheses)
            if _fw_ids:
                _fw_docs = load_framework_docs(_fw_ids, max_chars=4000)
                if _fw_docs:
                    _dg_block += _fw_docs
        except Exception:  # noqa: BLE001
            pass
        # ── Build reference paper block for reproduce-mode projects ────
        _ref_block = ""
        if _reference_paper_text:
            _ref_block = (
                "REFERENCE PAPER REPRODUCTION (HIGHEST PRIORITY):\n"
                "You are designing an experiment to REPRODUCE the following paper.\n"
                "Extract the EXACT algorithm, architecture, loss functions, training\n"
                "procedure, and hyperparameters from the paper text below.\n\n"
                "Your `proposed_methods` MUST include `implementation_spec` with:\n"
                "  - class_name matching the paper's described components\n"
                "  - algorithm_steps that faithfully reproduce the paper's method\n"
                "  - loss_function matching the paper's stated objective\n"
                "  - key_hyperparameters matching the paper's reported values\n"
                "Do NOT invent generic placeholders — extract details FROM THE PAPER.\n\n"
                "--- BEGIN REFERENCE PAPER TEXT ---\n"
                f"{_reference_paper_text}\n"
                "--- END REFERENCE PAPER TEXT ---\n\n"
            )

        _overlay = _get_evolution_overlay(run_dir, "experiment_design")
        sp = _pm.for_stage(
            "experiment_design",
            evolution_overlay=_overlay,
            preamble=preamble,
            hypotheses=hypotheses,
            dataset_guidance=_dg_block,
            time_budget_sec=config.experiment.time_budget_sec,
            metric_key=config.experiment.metric_key,
            metric_direction=config.experiment.metric_direction,
            reference_paper_block=_ref_block,
        )
        from researchclaw.pipeline.acceptance import ACCEPTANCE_POLICY, registered_context
        try:
            _registered_definitions = registered_context(run_dir)
        except (OSError, UnicodeError, ValueError) as exc:
            from researchclaw.pipeline.acceptance import write_plan_blocker
            error = write_plan_blocker(stage_dir, [str(exc)])
            return StageResult(stage=Stage.EXPERIMENT_DESIGN, status=StageStatus.FAILED,
                               artifacts=("acceptance_review.json",), decision="revise_design", error=error)
        _implementation_handoff = ACCEPTANCE_POLICY + "\n" + _registered_definitions + (
            "\nIMPLEMENTATION HANDOFF: Define proposed-method constants, every diagnostic metric's "
            "formula/aggregation/edge cases, and fixed fixture inputs/reference distributions. "
            "Do not leave these as 'recover definitions' or 'freeze later' instructions that prevent "
            "S11 from implementing the method. Distinguish later measured resource/gradient checks "
            "from definitions needed to write the code. Do not invent an author's missing protocol: "
            "identify it and request an explicit choice rather than claiming author fidelity. "
            "Respect all user budgets; no gate removal or validation/test-based definition selection.\n"
        )
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user + _implementation_handoff,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        def _record_plan_response(filename: str, response: Any) -> None:
            (stage_dir / filename).write_text(json.dumps({
                "content": response.content,
                "finish_reason": getattr(response, "finish_reason", ""),
                "truncated": getattr(response, "truncated", False),
                "completion_tokens": getattr(response, "completion_tokens", 0),
                "max_tokens": sp.max_tokens,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        _record_plan_response("plan_generation_response.json", resp)
        _plan_keys = {"objectives", "datasets", "baselines", "proposed_methods", "ablations", "metrics", "risks", "compute_budget"}
        def _complete_plan(value: Any, response: Any) -> bool:
            return (isinstance(value, dict) and _plan_keys <= value.keys()
                    and not getattr(response, "truncated", False))
        raw_yaml = _extract_yaml_block(resp.content)
        try:
            parsed = yaml.safe_load(raw_yaml)
        except yaml.YAMLError:
            parsed = None
        # Fallback: reasoning models sometimes emit the YAML without fences
        # or wrapped in prose. Try parsing the whole response as YAML.
        if not isinstance(parsed, dict):
            try:
                parsed = yaml.safe_load(resp.content)
            except yaml.YAMLError:
                pass
        # Last fallback: try to find any YAML-like dict in the response
        if not isinstance(parsed, dict):
            import re as _re_yaml

            # Look for lines starting with known keys
            _yaml_lines = []
            _capturing = False
            for line in resp.content.splitlines():
                if _re_yaml.match(
                    r"^(baselines|proposed_methods|ablations|datasets|"
                    r"metrics|objectives|risks|compute_budget)\s*:",
                    line,
                ):
                    _capturing = True
                if _capturing:
                    if line.strip() == "" or line.startswith("```"):
                        continue
                    if line.startswith("#") or line.startswith("**"):
                        continue
                    _yaml_lines.append(line)
            if _yaml_lines:
                try:
                    parsed = yaml.safe_load("\n".join(_yaml_lines))
                except yaml.YAMLError:
                    pass
        if _complete_plan(parsed, resp):
            plan = parsed
        else:
            logger.warning(
                "Stage 09: LLM response could not be parsed as YAML "
                "(len=%d, first 200 chars: %s). Content extraction method "
                "returned: %s",
                len(resp.content),
                resp.content[:200],
                raw_yaml[:200] if raw_yaml else "<empty>",
            )
            # BUG-12: Retry with a stricter, shorter prompt
            if llm is not None:
                logger.info("Stage 09: Retrying with strict YAML-only prompt...")
                _retry_prompt = (
                    "The previous reply was not a complete parseable experiment plan. "
                    "Return ONLY compact valid YAML; no preface or claims of work done. "
                    "Retain every required section, structured implementation specification, "
                    "source constraint and approved budget. Do not replace definitions with lists of names.\n\n"
                    + "Research topic and constraints:\n" + config.research.topic + "\n\n"
                    + sp.user + _implementation_handoff
                )
                _retry_resp = _chat_with_prompt(
                    llm,
                    "You output ONLY valid YAML. Nothing else.",
                    _retry_prompt,
                    max_tokens=max(sp.max_tokens or 0, 16384),
                )
                _record_plan_response("plan_generation_retry_response.json", _retry_resp)
                try:
                    _retry_parsed = yaml.safe_load(_retry_resp.content)
                    if _complete_plan(_retry_parsed, _retry_resp):
                        plan = _retry_parsed
                        logger.info("Stage 09: Strict YAML retry succeeded.")
                except yaml.YAMLError:
                    pass

    if plan is None and (llm is not None or getattr(config.experiment, "require_training", False)):
        failure = {"plan_accepted": False, "reason": "No complete experiment plan was generated; placeholder plans are not accepted"}
        (stage_dir / "plan_generation_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        return StageResult(stage=Stage.EXPERIMENT_DESIGN, status=StageStatus.FAILED,
            artifacts=("plan_generation_failure.json",), decision="retry", error=failure["reason"])

    # BUG-12: Fallback 4 — extract method/baseline names from Stage 8 hypotheses
    if plan is None:
        _hyp_text = _read_prior_artifact(run_dir, "hypotheses.md") or ""
        if _hyp_text:
            import re as _re_hyp
            # Extract method-like names from hypothesis text
            _method_candidates = _re_hyp.findall(
                r"(?:proposed|our|novel|new)\s+(?:method|approach|algorithm|framework|model)[:\s]+[\"']?([A-Za-z][\w-]+)",
                _hyp_text, _re_hyp.IGNORECASE,
            )
            _baseline_candidates = _re_hyp.findall(
                r"(?:baseline|compare|existing|standard|traditional)\s+(?:method|approach|model)?[:\s]+[\"']?([A-Za-z][\w-]+)",
                _hyp_text, _re_hyp.IGNORECASE,
            )
            if _method_candidates or _baseline_candidates:
                logger.info(
                    "Stage 09: Extracted names from hypotheses: methods=%s, baselines=%s",
                    _method_candidates[:3], _baseline_candidates[:3],
                )
                plan = {
                    "topic": config.research.topic,
                    "generated": _utcnow_iso(),
                    "objectives": ["Evaluate hypotheses with controlled experiments"],
                    "datasets": ["primary_dataset"],
                    "baselines": _baseline_candidates[:3] or ["baseline_1", "baseline_2"],
                    "proposed_methods": _method_candidates[:3] or ["proposed_method"],
                    "ablations": ["without_key_component", "simplified_version"],
                    "metrics": [config.experiment.metric_key, "secondary_metric"],
                    "risks": ["validity threats", "confounding variables"],
                    "compute_budget": {"max_gpu": 1, "max_hours": 4},
                }

    if plan is None:
        # BUG-12: Use domain-aware names instead of fully generic placeholders
        _topic_prefix = config.research.topic.split()[0] if config.research.topic else "method"
        logger.warning(
            "Stage 09: LLM failed to produce valid experiment plan YAML. "
            "Using topic-derived fallback."
        )
        plan = {
            "topic": config.research.topic,
            "generated": _utcnow_iso(),
            "objectives": ["Evaluate hypotheses with controlled experiments"],
            "datasets": ["primary_dataset", "secondary_dataset"],
            "baselines": [f"{_topic_prefix}_baseline_1", f"{_topic_prefix}_baseline_2"],
            "proposed_methods": [f"{_topic_prefix}_proposed", f"{_topic_prefix}_variant"],
            "ablations": ["without_key_component", "simplified_version"],
            "metrics": [config.experiment.metric_key, "secondary_metric"],
            "risks": ["validity threats", "confounding variables"],
            "compute_budget": {"max_gpu": 1, "max_hours": 4},
        }
    if isinstance(plan, dict):
        if _datasets_dir_s9 and _os_s9.path.isdir(_datasets_dir_s9):
            plan.setdefault("local_resources", {})
            if isinstance(plan["local_resources"], dict):
                plan["local_resources"]["datasets_dir"] = _datasets_dir_s9
                plan["local_resources"].setdefault(
                    "usage_requirement",
                    "All core experiments must use this local dataset path rather than unrelated generic benchmarks or synthetic substitutes.",
                )
            if _reference_paper_text:
                benchmark_suggestions = plan.get("benchmark_suggestions")
                if isinstance(benchmark_suggestions, dict) and "datasets" in benchmark_suggestions:
                    benchmark_suggestions.pop("datasets", None)
    # ── Validate implementation_spec presence ────────────────────────────
    # When reference paper text was provided (reproduce mode), the plan
    # MUST contain implementation_spec in proposed_methods.  If missing,
    # retry once with a targeted prompt.
    if (
        _reference_paper_text
        and isinstance(plan, dict)
        and llm is not None
    ):
        _methods = plan.get("proposed_methods", [])
        _has_impl = any(
            isinstance(m, dict) and "implementation_spec" in m
            for m in _methods
            if isinstance(m, dict)
        )
        if not _has_impl:
            logger.warning(
                "Stage 09: Plan missing implementation_spec for reproduce project "
                "— retrying with targeted prompt",
            )
            _impl_retry_prompt = (
                "The experiment plan you produced is missing `implementation_spec` "
                "under `proposed_methods`. This is required for code generation.\n\n"
                "Re-read the reference paper text and for EACH proposed method, "
                "add an `implementation_spec` block with:\n"
                "  - class_name\n"
                "  - algorithm_steps (3-10 concrete steps from the paper)\n"
                "  - loss_function (the exact loss from the paper)\n"
                "  - key_hyperparameters (values from the paper)\n"
                "  - required_loss_terms (symbolic loss/component names that must exist in code)\n"
                "  - required_distinct_helpers (non-generic helper methods that must be defined and called)\n"
                "  - required_data_pairing (how examples/prompts/pairs are grouped)\n"
                "  - required_model_edits (structural edits like monkey_patch_forward / replace_module / selective_unfreezing / attach_lora)\n"
                "  - required_runtime_hooks (hook targets or runtime interception points)\n"
                "  - key_methods\n"
                "  - differentiator\n\n"
                "Output the COMPLETE updated `proposed_methods` list as YAML.\n"
                "Include ONLY the `proposed_methods` key.\n\n"
                "Reference paper (first 15000 chars):\n"
                f"{_reference_paper_text[:15000]}\n\n"
                "Current proposed_methods:\n"
                f"{yaml.dump(_methods, default_flow_style=False)}"
            )
            try:
                _impl_resp = _chat_with_prompt(
                    llm,
                    "You are a research engineer. Output ONLY valid YAML.",
                    _impl_retry_prompt,
                    max_tokens=8192,
                )
                _impl_yaml = _extract_yaml_block(_impl_resp.content)
                _impl_parsed = yaml.safe_load(_impl_yaml)
                if not isinstance(_impl_parsed, dict):
                    _impl_parsed = yaml.safe_load(_impl_resp.content)
                if isinstance(_impl_parsed, dict):
                    _new_methods = _impl_parsed.get("proposed_methods", [])
                    if _new_methods and any(
                        isinstance(m, dict) and "implementation_spec" in m
                        for m in _new_methods
                        if isinstance(m, dict)
                    ):
                        plan["proposed_methods"] = _new_methods
                        logger.info(
                            "Stage 09: implementation_spec retry succeeded "
                            "(%d methods with specs)",
                            sum(1 for m in _new_methods
                                if isinstance(m, dict) and "implementation_spec" in m),
                        )
                    else:
                        logger.warning(
                            "Stage 09: implementation_spec retry returned no specs",
                        )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Stage 09: implementation_spec retry failed",
                    exc_info=True,
                )

    # Definitions must be implementable before spending a coding/training run.
    from researchclaw.pipeline.acceptance import plan_issues, resource_plan_issues, requirement_text, write_plan_blocker
    _acceptance_errors = plan_issues(
        plan, require_protocol=bool(llm is not None and getattr(config.experiment, "require_training", False)),
        user_text=requirement_text(config, run_dir),
    ) + resource_plan_issues(plan, config)
    if _acceptance_errors:
        (stage_dir / "exp_plan_draft.yaml").write_text(
            yaml.safe_dump(plan, allow_unicode=True), encoding="utf-8")
        error = write_plan_blocker(stage_dir, _acceptance_errors)
        return StageResult(stage=Stage.EXPERIMENT_DESIGN, status=StageStatus.FAILED,
                           artifacts=("exp_plan_draft.yaml", "acceptance_review.json"),
                           decision="revise_design", error=error)

    # ── BA: BenchmarkAgent — intelligent dataset/baseline selection ──────
    # Retain a completed design even if optional recommendation is interrupted.
    plan.setdefault("topic", config.research.topic)
    (stage_dir / "exp_plan.yaml").write_text(
        yaml.dump(plan, default_flow_style=False, allow_unicode=True), encoding="utf-8",
    )
    _benchmark_plan = None
    # BUG-40: Skip BenchmarkAgent for non-ML domains — it has no relevant
    # benchmarks for physics/chemistry/mathematics/etc. and would inject
    # wrong datasets (e.g., CIFAR-10 for PDE topics).
    _ba_domain_hint_ids: list[str] = []
    _ba_domain_label = ""
    if _domain_profile is not None:
        _ba_domain_hint_ids = [_domain_profile.domain_id]
        _ba_domain_label = _domain_profile.domain_id
        try:
            from researchclaw.domains.detector import is_ml_domain as _is_ml_domain_profile
            _ba_domain_ok = _is_ml_domain_profile(_domain_profile)
        except Exception:  # noqa: BLE001
            _ba_domain_ok = _domain_profile.domain_id.startswith("ml_")
    else:
        _ba_domain_id, _, _ = _detect_domain(
            config.research.topic,
            tuple(config.research.domains) if config.research.domains else (),
        )
        _ba_domain_label = _ba_domain_id
        _ba_domain_ok = _ba_domain_id == "ml"
    if not _ba_domain_ok:
        logger.info(
            "BenchmarkAgent skipped: domain '%s' is not ML (topic: %s)",
            _ba_domain_label, config.research.topic[:80],
        )
    _fixed_local_experiment = bool(
        getattr(config.experiment, "datasets_dir", "")
        and (getattr(config.experiment, "codebases_dir", "")
             or getattr(config.experiment, "existing_project_dir", ""))
    )
    if _fixed_local_experiment:
        logger.info("BenchmarkAgent skipped: configured local data and code define the experiment; preserving baseline specifications")
    if (
        not _fixed_local_experiment
        and _ba_domain_ok
        and config.experiment.benchmark_agent.enabled
        and config.experiment.mode in ("sandbox", "docker")
        and llm is not None
    ):
        try:
            from researchclaw.agents.benchmark_agent import BenchmarkOrchestrator
            from researchclaw.agents.benchmark_agent.orchestrator import (
                BenchmarkAgentConfig as _BACfg,
            )

            _ba_cfg_raw = config.experiment.benchmark_agent
            _ba_cfg = _BACfg(
                enabled=_ba_cfg_raw.enabled,
                enable_hf_search=_ba_cfg_raw.enable_hf_search,
                max_hf_results=_ba_cfg_raw.max_hf_results,
                enable_web_search=_ba_cfg_raw.enable_web_search,
                max_web_results=_ba_cfg_raw.max_web_results,
                web_search_min_local=_ba_cfg_raw.web_search_min_local,
                tier_limit=_ba_cfg_raw.tier_limit,
                min_benchmarks=_ba_cfg_raw.min_benchmarks,
                min_baselines=_ba_cfg_raw.min_baselines,
                prefer_cached=_ba_cfg_raw.prefer_cached,
                max_iterations=_ba_cfg_raw.max_iterations,
            )

            _hw = _load_hardware_profile(run_dir)
            _ba = BenchmarkOrchestrator(
                llm,
                config=_ba_cfg,
                gpu_memory_mb=(
                    _hw.get("gpu_memory_mb", 49000) if _hw else 49000
                ),
                time_budget_sec=config.experiment.time_budget_sec,
                network_policy=(
                    config.experiment.docker.network_policy
                    if config.experiment.mode == "docker"
                    else "full"
                ),
                stage_dir=stage_dir / "benchmark_agent",
            )
            _benchmark_plan = _ba.orchestrate({
                "topic": config.research.topic,
                "hypothesis": hypotheses,
                "experiment_plan": plan.get("objectives", "") if isinstance(plan, dict) else "",
                "domain_hints": _ba_domain_hint_ids,
            })

            # Inject BenchmarkAgent selections into experiment plan
            if isinstance(plan, dict) and _benchmark_plan.selected_benchmarks:
                _has_local_dataset_contract = bool(
                    getattr(config.experiment, "datasets_dir", "") or ""
                )
                if not _has_local_dataset_contract:
                    plan["datasets"] = [
                        b["name"] for b in _benchmark_plan.selected_benchmarks
                    ]
                else:
                    logger.info(
                        "BenchmarkAgent datasets preserved as suggestions only because project has local datasets_dir=%s",
                        getattr(config.experiment, "datasets_dir", ""),
                    )
                    plan.setdefault("benchmark_suggestions", {})
                    if isinstance(plan["benchmark_suggestions"], dict):
                        plan["benchmark_suggestions"]["datasets"] = [
                            b["name"] for b in _benchmark_plan.selected_benchmarks
                        ]
                # Normalize existing baselines to list of strings
                # BUG-35: LLM may emit baselines as dict, list of dicts,
                # or list of strings — normalize all to list[str].
                _baselines_from_plan = plan.get("baselines", [])
                if isinstance(_baselines_from_plan, dict):
                    _baselines_from_plan = list(_baselines_from_plan.keys())
                elif isinstance(_baselines_from_plan, list):
                    _baselines_from_plan = [
                        item["name"] if isinstance(item, dict) else str(item)
                        for item in _baselines_from_plan
                    ]
                else:
                    _baselines_from_plan = []
                plan["baselines"] = [
                    bl["name"] for bl in _benchmark_plan.selected_baselines
                ] + _baselines_from_plan
                # Deduplicate baselines
                plan["baselines"] = list(dict.fromkeys(plan["baselines"]))

            logger.info(
                "BenchmarkAgent: %d benchmarks, %d baselines selected (%d LLM calls, %.1fs)",
                len(_benchmark_plan.selected_benchmarks),
                len(_benchmark_plan.selected_baselines),
                _benchmark_plan.total_llm_calls,
                _benchmark_plan.elapsed_sec,
            )
        except Exception as _ba_exc:
            logger.warning("BenchmarkAgent failed (non-fatal): %s", _ba_exc)

    # Save benchmark plan for code_generation stage
    if _benchmark_plan is not None:
        try:
            (stage_dir / "benchmark_plan.json").write_text(
                json.dumps(_benchmark_plan.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

    plan.setdefault("topic", config.research.topic)
    (stage_dir / "exp_plan.yaml").write_text(
        yaml.dump(plan, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return StageResult(
        stage=Stage.EXPERIMENT_DESIGN,
        status=StageStatus.DONE,
        artifacts=("exp_plan.yaml",),
        evidence_refs=("stage-09/exp_plan.yaml",),
    )


# ---------------------------------------------------------------------------
# Stage 10: CODEBASE_SEARCH — find & download reusable codebases
# ---------------------------------------------------------------------------

def _execute_codebase_search(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    run_id: str = "",
    prompts: "PromptManager | None" = None,
    **kwargs: object,
) -> StageResult:
    """Search for reusable codebases based on the experiment plan.

    Reads exp_plan.yaml, asks LLM to identify relevant GitHub repos or
    papers-with-code, attempts to clone/download them, and records results
    in codebase_candidates.json.
    """
    from researchclaw.llm import create_llm_client

    exp_plan = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""

    llm = create_llm_client(config)

    search_prompt = (
        "Based on the following experiment plan, identify up to 5 existing open-source "
        "GitHub repositories or papers-with-code that could serve as a starting codebase "
        "for this experiment. For each candidate, provide:\n"
        "- repo_url: GitHub URL\n"
        "- description: why it is relevant\n"
        "- usability: 'direct' (can use as-is with minor changes) or 'reference' (useful as reference)\n"
        "- key_files: list of key files/modules to reuse\n\n"
        "Return a JSON array. If no suitable codebase exists, return an empty array [].\n\n"
        "Experiment Plan:\n"
        f"{exp_plan}"
    )

    import json as _json
    import os as _os_s10

    candidates: list[dict] = []

    # ── Phase 1: Index LOCAL codebases already on disk ────────────────
    _local_codebases_dir = getattr(config.experiment, "codebases_dir", "") or ""
    if _local_codebases_dir and _os_s10.path.isdir(_local_codebases_dir):
        _local_repos: list[dict] = []
        for name in sorted(_os_s10.listdir(_local_codebases_dir)):
            full = _os_s10.path.join(_local_codebases_dir, name)
            if _os_s10.path.isdir(full) and not name.startswith("."):
                _local_repos.append({"name": name, "path": full})

        # Use LLM to assess relevance when multiple local repos exist
        _relevance_map: dict[str, str] = {}
        if len(_local_repos) > 1 and llm is not None:
            _repo_names = [r["name"] for r in _local_repos]
            _relevance_prompt = (
                f"Research topic: {config.research.topic}\n\n"
                f"Local codebases available: {_repo_names}\n\n"
                "For each codebase, assess its relevance to the research topic. "
                "Return a JSON object mapping each codebase name to one of: "
                '"high" (directly related, should be used), '
                '"medium" (somewhat related, could be useful), '
                '"low" (unrelated to this research topic).\n\n'
                "Example: {\"RepoA\": \"high\", \"RepoB\": \"low\"}"
            )
            try:
                _rel_resp = llm.chat(
                    [{"role": "user", "content": _relevance_prompt}],
                    system="You are a research engineer. Return valid JSON only.",
                    json_mode=True,
                    max_tokens=256,
                )
                _relevance_map = _json.loads(_rel_resp.content)
                if not isinstance(_relevance_map, dict):
                    _relevance_map = {}
            except Exception:
                _relevance_map = {}

        for _repo in _local_repos:
            _rel = _relevance_map.get(_repo["name"], "high" if len(_local_repos) == 1 else "low")
            candidates.append({
                "repo_url": f"local://{_repo['path']}",
                "description": f"Local codebase: {_repo['name']}",
                "usability": "direct",
                "relevance": _rel,
                "key_files": [],
                "local_path": _repo["path"],
                "download_status": "success",
            })

    # ── Phase 2: LLM-based GitHub search ──────────────────────────────
    try:
        response = llm.chat(
            [{"role": "user", "content": search_prompt}],
            system="You are an expert research engineer. Return valid JSON only.",
            json_mode=True,
        )

        try:
            remote_candidates = _json.loads(response.content)
            if not isinstance(remote_candidates, list):
                remote_candidates = []
        except _json.JSONDecodeError:
            remote_candidates = []

        codebase_dir = stage_dir / "codebases"
        codebase_dir.mkdir(exist_ok=True)
        for i, cand in enumerate(remote_candidates):
            url = cand.get("repo_url", "")
            usability = cand.get("usability", "reference")
            if usability == "direct" and url and "github.com" in url:
                local_path = codebase_dir / f"repo_{i}"
                try:
                    import subprocess
                    result = subprocess.run(
                        ["git", "clone", "--depth", "1", url, str(local_path)],
                        capture_output=True, text=True, timeout=60,
                    )
                    if result.returncode == 0:
                        cand["local_path"] = str(local_path)
                        cand["download_status"] = "success"
                    else:
                        cand["download_status"] = f"failed: {result.stderr[:200]}"
                except Exception as e:
                    cand["download_status"] = f"error: {e}"
            else:
                cand["download_status"] = "skipped"
        candidates.extend(remote_candidates)

    except Exception as e:
        logger.warning("LLM codebase search failed: %s", e)

    # ── Phase 3: Filter out low-relevance candidates, then persist ────
    _filtered = [c for c in candidates if c.get("relevance", "high") != "low"]
    _dropped = len(candidates) - len(_filtered)
    if _dropped:
        logger.info("CODEBASE_SEARCH: dropped %d low-relevance candidates", _dropped)

    try:
        output = _json.dumps(_filtered, indent=2, ensure_ascii=False)
        (stage_dir / "codebase_candidates.json").write_text(output, encoding="utf-8")
    except Exception:
        (stage_dir / "codebase_candidates.json").write_text("[]", encoding="utf-8")

    n_local = sum(1 for c in _filtered if str(c.get("repo_url", "")).startswith("local://"))
    n_remote = sum(1 for c in _filtered if c.get("download_status") == "success") - n_local
    n_total = len(_filtered)
    summary = f"Found {n_total} candidates ({n_local} local, {n_remote} remote downloaded, {_dropped} low-relevance dropped)"

    return StageResult(
        stage=Stage.CODEBASE_SEARCH,
        status=StageStatus.DONE,
        artifacts=("codebase_candidates.json",),
        evidence_refs=("stage-09/exp_plan.yaml",),
        decision=summary,
    )


def _extract_selected_repos(codebase_info_json: str) -> list[str] | None:
    """Extract repo directory names from codebase_candidates.json that S10 marked as relevant.

    Returns None (= copy all) if no candidates have relevance info,
    or a list of directory basenames for repos marked relevant.
    """
    try:
        import json as _json_sr
        candidates = _json_sr.loads(codebase_info_json)
        if not isinstance(candidates, list) or not candidates:
            return None
    except Exception:
        return None

    has_relevance = any(c.get("relevance") for c in candidates if isinstance(c, dict))

    selected: list[str] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        if c.get("download_status") != "success":
            continue
        local_path = c.get("local_path", "")
        if not local_path:
            continue
        repo_name = local_path.rstrip("/").rsplit("/", 1)[-1]
        rel = c.get("relevance", "")
        if has_relevance and rel not in ("high", "medium"):
            continue
        if not has_relevance and c.get("usability") == "reference":
            continue
        selected.append(repo_name)

    return selected if selected else None


def _execute_code_generation(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    """S11 CODE_GENERATION — delegates to the refactored codegen package.

    The full implementation lives in ``researchclaw.pipeline.codegen``,
    structured around claw-code's harness engineering patterns:
    StrategyRegistry, CodegenRouter, CodegenRuntime, and PromptBuilder.
    """
    from researchclaw.pipeline.codegen import execute_code_generation
    return execute_code_generation(
                    stage_dir=stage_dir,
        run_dir=run_dir,
        config=config,
        adapters=adapters,
            llm=llm,
        prompts=prompts,
    )


def _execute_sanity_check(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    run_id: str = "",
    prompts: "PromptManager | None" = None,
    llm: "LLMClient | None" = None,
    **kwargs: object,
) -> StageResult:
    """S12 SANITY_CHECK — delegates to the refactored sanity_check package.

    The full implementation uses a claw-code agentic turn loop to run
    smoke tests, diagnose errors, and fix code iteratively.
    """
    from researchclaw.pipeline.sanity_check import execute_sanity_check
    return execute_sanity_check(
                stage_dir=stage_dir,
                run_dir=run_dir,
        config=config,
        adapters=adapters,
        llm=llm,
        prompts=prompts,
    )


def _execute_resource_planning(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    exp_plan = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
    schedule: dict[str, Any] | None = None
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "resource_planning")
        sp = _pm.for_stage("resource_planning", evolution_overlay=_overlay, exp_plan=exp_plan)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        parsed = _safe_json_loads(resp.content, {})
        if isinstance(parsed, dict):
            schedule = parsed
    if schedule is None:
        schedule = {
            "tasks": [
                {
                    "id": "baseline",
                    "name": "Run baseline",
                    "depends_on": [],
                    "gpu_count": 1,
                    "estimated_minutes": 20,
                    "priority": "high",
                },
                {
                    "id": "proposed",
                    "name": "Run proposed method",
                    "depends_on": ["baseline"],
                    "gpu_count": 1,
                    "estimated_minutes": 30,
                    "priority": "high",
                },
            ],
            "total_gpu_budget": 1,
            "generated": _utcnow_iso(),
        }
    schedule.setdefault("generated", _utcnow_iso())
    (stage_dir / "schedule.json").write_text(
        json.dumps(schedule, indent=2), encoding="utf-8"
    )
    return StageResult(
        stage=Stage.RESOURCE_PLANNING,
        status=StageStatus.DONE,
        artifacts=("schedule.json",),
        evidence_refs=("stage-11/schedule.json",),
    )


def _execute_experiment_run(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    """S14 EXPERIMENT_RUN — delegates to the refactored experiment_run package.

    The full implementation uses a claw-code agentic turn loop to execute
    the experiment, monitor for runtime errors, and apply fixes.
    """
    from researchclaw.pipeline.experiment_run import execute_experiment_run
    return execute_experiment_run(
        stage_dir=stage_dir,
        run_dir=run_dir,
        config=config,
        adapters=adapters,
        llm=llm,
        prompts=prompts,
    )


def _execute_iterative_refine(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    """S15 ITERATIVE_REFINE — delegates to the refactored iterative_refine package.

    The full implementation uses a claw-code agentic turn loop to analyze
    results, modify code, and re-run experiments iteratively.
    """
    from researchclaw.pipeline.iterative_refine import execute_iterative_refine
    return execute_iterative_refine(
        stage_dir=stage_dir,
        run_dir=run_dir,
        config=config,
        adapters=adapters,
        llm=llm,
        prompts=prompts,
    )


def _run_chart_generation(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    llm: Any | None = None,
) -> None:
    """Generate charts for S16 using visualize.py (shared by agentic and legacy paths)."""
    try:
        from researchclaw.experiment.visualize import (
            generate_all_charts as _gen_charts,
        )
        _charts_dir = stage_dir / "charts"
        _charts_dir.mkdir(exist_ok=True)
        _generated = _gen_charts(
            run_dir,
            _charts_dir,
            metric_key=config.experiment.metric_key,
        )
        if _generated:
            logger.info("S16: Generated %d charts", len(_generated))
    except Exception as _exc:
        logger.warning("S16: Chart generation failed: %s", _exc)


def _execute_result_analysis(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    # --- Agentic analysis: let the LLM read files, write scripts, produce summary ---
    from researchclaw.pipeline.result_analysis import ResultAnalysisRuntime
    _agent_rt = ResultAnalysisRuntime()
    _agent_result = _agent_rt.execute(
        stage_dir=stage_dir,
        run_dir=run_dir,
        config=config,
        adapters=adapters,
        llm=llm,
    )
    # If the agent produced both analysis.md and experiment_summary.json, use
    # them directly and skip the legacy hardcoded collection path.
    _agent_summary = stage_dir / "experiment_summary.json"
    _agent_analysis = stage_dir / "analysis.md"
    if _agent_summary.is_file() and _agent_analysis.is_file():
        try:
            _parsed = json.loads(_agent_summary.read_text(encoding="utf-8"))
            if isinstance(_parsed, dict) and _parsed.get("metrics_summary"):
                logger.info(
                    "S16: Using agentic experiment_summary.json (%d metric keys)",
                    len(_parsed["metrics_summary"]),
                )
                # Still run chart generation from the legacy path
                _run_chart_generation(stage_dir, run_dir, config, llm)
                artifacts = ["analysis.md", "experiment_summary.json"]
                if (stage_dir / "charts").is_dir() and any((stage_dir / "charts").iterdir()):
                    artifacts.append("charts/")
                return StageResult(
                    stage=Stage.RESULT_ANALYSIS,
                    status=StageStatus.DONE,
                    artifacts=tuple(artifacts),
                    evidence_refs=tuple(f"stage-16/{a}" for a in artifacts),
                )
        except (json.JSONDecodeError, OSError):
            pass
    logger.info("S16: Agentic analysis insufficient, falling back to legacy path")

    # --- Legacy: Collect experiment data ---
    exp_data = _collect_experiment_results(
        run_dir,
        metric_key=config.experiment.metric_key,
        metric_direction=config.experiment.metric_direction,
    )
    runs_dir = _read_prior_artifact(run_dir, "runs/") or ""
    context = ""
    if runs_dir:
        context = _collect_json_context(Path(runs_dir), max_files=30)
    if exp_data.get("run_count_basis") == "result_records":
        context = json.dumps({
            "selected_version": exp_data.get("selected_version"),
            "final_results": exp_data["runs"],
            "run_count_basis": "result_records",
        }, indent=2, default=str)

    # --- R13-1: Merge Stage 13 (ITERATIVE_REFINE) results if available ---
    # Stage 13 stores richer per-condition metrics in refinement_log.json
    # that _collect_experiment_results() misses (it only scans runs/ dirs).
    _refine_log_text = _read_prior_artifact(run_dir, "refinement_log.json")
    if _refine_log_text and not str(exp_data.get("selected_version", "")).startswith(
        f"stage-{int(Stage.ITERATIVE_REFINE):02d}"
    ):
        try:
            _refine_data = json.loads(_refine_log_text)
            _best_iter = None
            _best_ver = _refine_data.get("best_version", "")
            _refine_iterations = _refine_data.get("iterations", [])
            if not isinstance(_refine_iterations, list):
                logger.warning(
                    "R13-1: refinement_log.json has non-list iterations=%r; "
                    "skipping refinement merge",
                    type(_refine_iterations).__name__,
                )
                _refine_iterations = []
            for _it in _refine_iterations:
                if not isinstance(_it, dict):
                    continue
                _sbx = _it.get("sandbox", {})
                _it_metrics = _sbx.get("metrics", {})
                if _it.get("version_dir", "") == _best_ver and _it_metrics:
                    _best_iter = _it
                    break
            # If no version match, take the first iteration with metrics
            if _best_iter is None:
                for _it in _refine_iterations:
                    if not isinstance(_it, dict):
                        continue
                    _sbx = _it.get("sandbox", {})
                    if _sbx.get("metrics"):
                        _best_iter = _it
                        break
            if _best_iter is not None:
                _sbx = _best_iter.get("sandbox", {})
                _refine_metrics = _sbx.get("metrics", {})
                if _refine_metrics and (
                    not exp_data["metrics_summary"]
                    or len(_refine_metrics) > len(exp_data["metrics_summary"])
                ):
                    # Refinement has richer data — rebuild metrics_summary from it
                    _new_summary: dict[str, dict[str, float | None]] = {}
                    for _mk, _mv in _refine_metrics.items():
                        try:
                            _fv = float(_mv)
                            _new_summary[_mk] = {
                                "min": round(_fv, 6),
                                "max": round(_fv, 6),
                                "mean": round(_fv, 6),
                                "count": 1,
                            }
                        except (ValueError, TypeError):
                            pass
                    if _new_summary:
                        exp_data["metrics_summary"] = _new_summary
                        # Also update best_run with refinement data
                        exp_data["best_run"] = {
                            "run_id": "iterative-refine-best",
                            "task_id": "sandbox-main",
                            "status": "completed",
                            "metrics": {
                                k: v for k, v in _refine_metrics.items()
                            },
                            "elapsed_sec": _sbx.get("elapsed_sec", 0),
                            "stdout": "",  # omit for brevity
                            "stderr": _sbx.get("stderr", ""),
                            "timed_out": _sbx.get("timed_out", False),
                        }
                        # Rebuild latex table
                        _ltx = [
                            r"\begin{table}[h]", r"\centering",
                            r"\caption{Experiment Results (Best Refinement Iteration)}",
                            r"\begin{tabular}{lrrrr}", r"\hline",
                            r"Metric & Min & Max & Mean & N \\", r"\hline",
                        ]
                        for _col in sorted(_new_summary.keys()):
                            _s = _new_summary[_col]
                            _ltx.append(
                                f"{_col} & {_s['min']:.4f} & {_s['max']:.4f} "
                                f"& {_s['mean']:.4f} & {_s['count']} \\\\"
                            )
                        _ltx.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
                        exp_data["latex_table"] = "\n".join(_ltx)
                        # Count unique conditions (keys without 'seed' and not ending in _mean/_std)
                        _conditions = {
                            k for k in _refine_metrics
                            if "seed" not in k and not k.endswith("_std")
                        }
                        exp_data["runs"] = [exp_data["best_run"]]
                        # Store condition count for accurate reporting
                        exp_data["best_run"]["condition_count"] = len(_conditions)
                        if not context:
                            context = json.dumps(
                                {"refinement_best_metrics": _refine_metrics},
                                indent=2, default=str,
                            )
                        logger.info(
                            "R13-1: Merged %d metrics from refinement_log (best_metric=%.4f)",
                            len(_refine_metrics),
                            _refine_data.get("best_metric", 0),
                        )
        except (json.JSONDecodeError, OSError, KeyError):
            logger.warning("R13-1: Failed to parse refinement_log.json, using Stage 12 data")

    # --- R19-2: Extract PAIRED comparisons from refinement stdout ---
    from researchclaw.experiment.sandbox import extract_paired_comparisons as _extract_paired

    _all_paired: list[dict[str, object]] = []
    # First: from _collect_experiment_results (Stage 12 runs/)
    if exp_data.get("paired_comparisons"):
        _all_paired.extend(exp_data["paired_comparisons"])
    # Second: from refinement_log iterations (Stage 13)
    if _refine_log_text:
        try:
            _rl = json.loads(_refine_log_text)
            _refine_iterations = _rl.get("iterations", [])
            if not isinstance(_refine_iterations, list):
                _refine_iterations = []
            for _it in _refine_iterations:
                if not isinstance(_it, dict):
                    continue
                for _sbx_key in ("sandbox", "sandbox_after_fix"):
                    _sbx_stdout = (_it.get(_sbx_key) or {}).get("stdout", "")
                    if _sbx_stdout:
                        _all_paired.extend(_extract_paired(_sbx_stdout))
        except (json.JSONDecodeError, OSError):
            pass

    # --- R19-3: Build structured condition_summaries from metrics ---
    _condition_summaries: dict[str, dict[str, Any]] = {}
    _ms = exp_data.get("metrics_summary", {})
    _best_metrics = {}
    if exp_data.get("best_run") and isinstance(exp_data["best_run"], dict):
        _best_metrics = exp_data["best_run"].get("metrics", {})

    # Group metrics by condition prefix (e.g., "ppo/primary_metric" → condition "ppo").
    # Deeper paths also keep their sub-path ("test/clean/CASIA/pixel_f1"); the
    # bare leaf name is kept only when it is unambiguous within the condition.
    _leaf_counts: dict[tuple[str, str], int] = {}
    for _mk in _best_metrics:
        parts = _mk.split("/")
        if len(parts) >= 2:
            _leaf_counts[(parts[0], parts[-1])] = _leaf_counts.get((parts[0], parts[-1]), 0) + 1
    for _mk, _mv in _best_metrics.items():
        parts = _mk.split("/")
        if len(parts) >= 2:
            cond = parts[0]
            metric_name = parts[-1]
            if cond not in _condition_summaries:
                _condition_summaries[cond] = {"metrics": {}}
            try:
                _val = float(_mv)
            except (ValueError, TypeError):
                continue
            _cond_metrics = _condition_summaries[cond]["metrics"]
            if len(parts) > 2:
                _cond_metrics["/".join(parts[1:])] = _val
            if (
                len(parts) == 2
                or _leaf_counts[(cond, metric_name)] == 1
                or metric_name == config.experiment.metric_key
            ):
                _cond_metrics[metric_name] = _val

    # BUG-09 fix: If no condition summaries were built (metrics don't use
    # condition/metric format), try to extract from metrics_summary or
    # structured_results so FigureAgent has data to work with.
    if not _condition_summaries and _ms:
        # Try to parse condition data from metrics_summary keys
        for _mk, _mv in _ms.items():
            parts = _mk.split("/")
            if len(parts) >= 2:
                cond = parts[0]
                metric_name = parts[-1]
                if cond not in _condition_summaries:
                    _condition_summaries[cond] = {"metrics": {}}
                try:
                    _val = float(_mv) if not isinstance(_mv, dict) else None
                    if _val is not None:
                        _condition_summaries[cond]["metrics"][metric_name] = _val
                except (ValueError, TypeError):
                    pass
    if not _condition_summaries:
        # Last resort: build from structured_results condition keys
        _sr = exp_data.get("structured_results", {})
        if isinstance(_sr, dict):
            for _sk, _sv in _sr.items():
                if isinstance(_sv, dict) and _sk not in ("metadata", "config"):
                    _condition_summaries[_sk] = {"metrics": {}}
                    for _smk, _smv in _sv.items():
                        try:
                            _condition_summaries[_sk]["metrics"][_smk] = float(_smv)
                        except (ValueError, TypeError):
                            pass

    # R33: Build per-seed data structure (needed for CIs and paired tests below)
    _seed_data: dict[str, dict[int, float]] = {}  # {condition: {seed: value}}
    for _mk, _mv in _best_metrics.items():
        parts = _mk.split("/")
        # Pattern: condition/regime/seed_id/primary_metric
        if len(parts) >= 4 and parts[-1] == config.experiment.metric_key:
            cond = parts[0]
            try:
                seed_id = int(parts[2])
                val = float(_mv)
                _seed_data.setdefault(cond, {})[seed_id] = val
            except (ValueError, TypeError):
                pass

    # Enrich condition summaries with seed counts, success rates, and CIs
    for _ck, _cv in _condition_summaries.items():
        # Look for success_rate in metrics
        sr_key = f"{_ck}/success_rate"
        if sr_key in _best_metrics:
            try:
                _cv["success_rate"] = float(_best_metrics[sr_key])
            except (ValueError, TypeError):
                pass
        # Count seed-level entries to estimate n_seeds
        _seed_count = 0
        for _mk in _best_metrics:
            if _mk.startswith(f"{_ck}/") and "seed" in _mk.lower():
                _seed_count += 1
        if _seed_count > 0:
            _cv["n_seed_metrics"] = _seed_count

        # R33: Compute mean ± std and bootstrap 95% CI from per-seed data
        if _ck in _seed_data and len(_seed_data[_ck]) >= 3:
            _vals = list(_seed_data[_ck].values())
            import statistics as _stats_mod
            _mean = _stats_mod.mean(_vals)
            _std = _stats_mod.stdev(_vals)
            _cv["metrics"][f"{config.experiment.metric_key}_mean"] = round(_mean, 6)
            _cv["metrics"][f"{config.experiment.metric_key}_std"] = round(_std, 6)
            _cv["n_seeds"] = len(_vals)
            # Bootstrap 95% CI
            import random as _rng
            _rng.seed(42)
            _boot_means = []
            for _ in range(1000):
                _sample = [_rng.choice(_vals) for _ in range(len(_vals))]
                _boot_means.append(_stats_mod.mean(_sample))
            _boot_means.sort()
            _ci_low = round(_boot_means[int(0.025 * len(_boot_means))], 6)
            _ci_high = round(_boot_means[int(0.975 * len(_boot_means))], 6)
            # IMP-16: Sanity check — CI must contain the mean
            if _ci_low > _mean or _ci_high < _mean:
                logger.warning(
                    "Bootstrap CI [%.4f, %.4f] does not contain mean %.4f "
                    "for condition %s — replacing CI with mean ± 1.96*SE",
                    _ci_low, _ci_high, _mean, _ck,
                )
                _se = _std / (len(_vals) ** 0.5)
                _ci_low = round(_mean - 1.96 * _se, 6)
                _ci_high = round(_mean + 1.96 * _se, 6)
            _cv["ci95_low"] = _ci_low
            _cv["ci95_high"] = _ci_high

    # Count totals
    _total_conditions = len(_condition_summaries) if _condition_summaries else None
    _total_metrics = len(_best_metrics) if _best_metrics else None

    # --- R33: Pipeline-level paired computation as fallback ---
    # If the experiment code's PAIRED lines are sparse or suspicious (e.g.,
    # all identical t-stats), compute fresh paired tests from per-seed data.
    # (_seed_data was built above before condition summary enrichment)
    if len(_seed_data) >= 2:
        # Find common seeds across conditions
        _all_seeds_sets = [set(v.keys()) for v in _seed_data.values()]
        _common_seeds = set.intersection(*_all_seeds_sets) if _all_seeds_sets else set()

        if len(_common_seeds) >= 3:
            _cond_names_sorted = sorted(_seed_data.keys())
            _pipeline_paired: list[dict[str, object]] = []
            # Compare each condition against the first baseline (alphabetically)
            _baseline_cond = _cond_names_sorted[0]
            for _other_cond in _cond_names_sorted[1:]:
                _diffs = []
                for _sid in sorted(_common_seeds):
                    _diffs.append(
                        _seed_data[_other_cond][_sid] - _seed_data[_baseline_cond][_sid]
                    )
                if _diffs:
                    import statistics
                    _n = len(_diffs)
                    _mean_d = statistics.mean(_diffs)
                    _std_d = statistics.stdev(_diffs) if _n > 1 else 0.0
                    _t = (_mean_d / (_std_d / (_n ** 0.5))) if _std_d > 0 else 0.0
                    _df = _n - 1
                    # Two-tailed p-value using t-distribution
                    import math
                    try:
                        from scipy.stats import t as _t_dist
                        _p = float(2 * _t_dist.sf(abs(_t), _df))
                    except ImportError:
                        _p = 2 * (1 - 0.5 * (1 + math.erf(abs(_t) / (2 ** 0.5))))
                        if _df < 30:
                            _p = min(1.0, _p * (1 + 2.5 / max(_df, 1)))
                    _pipeline_paired.append({
                        "method": _other_cond,
                        "baseline": _baseline_cond,
                        "mean_diff": round(_mean_d, 6),
                        "std_diff": round(_std_d, 6),
                        "t_stat": round(_t, 4),
                        "p_value": round(_p, 6),
                        "n_seeds": _n,
                        "source": "pipeline_computed",
                    })

            # Use pipeline-computed if experiment code's are suspicious
            _exp_t_stats = {round(p.get("t_stat", 0), 4) for p in _all_paired}
            _all_identical = len(_exp_t_stats) <= 1 and len(_all_paired) > 1
            if _pipeline_paired and (_all_identical or len(_all_paired) < len(_pipeline_paired)):
                logger.info(
                    "R33: Using %d pipeline-computed paired tests (experiment code had %d, identical=%s)",
                    len(_pipeline_paired), len(_all_paired), _all_identical,
                )
                _all_paired = _pipeline_paired

    # --- P8: Detect identical conditions (broken ablations) ---
    _ablation_warnings: list[str] = []
    if _condition_summaries and len(_condition_summaries) >= 2:
        _cond_names = sorted(_condition_summaries.keys())
        for _i in range(len(_cond_names)):
            for _j in range(_i + 1, len(_cond_names)):
                _c1, _c2 = _cond_names[_i], _cond_names[_j]
                _s1 = _condition_summaries[_c1]
                _s2 = _condition_summaries[_c2]
                # Compare mean values for all shared metrics
                _shared_keys = set(_s1.keys()) & set(_s2.keys())
                if not _shared_keys:
                    continue
                _all_equal = True
                for _sk in _shared_keys:
                    _v1 = _s1[_sk].get("mean") if isinstance(_s1[_sk], dict) else _s1[_sk]
                    _v2 = _s2[_sk].get("mean") if isinstance(_s2[_sk], dict) else _s2[_sk]
                    if _v1 != _v2:
                        _all_equal = False
                        break
                if _all_equal and _shared_keys:
                    _warn = (
                        f"ABLATION FAILURE: Conditions '{_c1}' and '{_c2}' produce "
                        f"identical outputs across all {len(_shared_keys)} metrics. "
                        f"The ablation is invalid — the differentiating parameter "
                        f"is likely not used in the code."
                    )
                    _ablation_warnings.append(_warn)
                    logger.warning("P8: %s", _warn)
                elif _shared_keys:
                    # R5-BUG-03: Also flag near-identical conditions (< 1% relative diff)
                    _near_identical = True
                    for _sk in _shared_keys:
                        _v1 = _s1[_sk].get("mean") if isinstance(_s1[_sk], dict) else _s1[_sk]
                        _v2 = _s2[_sk].get("mean") if isinstance(_s2[_sk], dict) else _s2[_sk]
                        try:
                            _v1f, _v2f = float(_v1), float(_v2)
                            _denom = max(abs(_v1f), abs(_v2f), 1e-12)
                            if abs(_v1f - _v2f) / _denom > 0.01:
                                _near_identical = False
                                break
                        except (TypeError, ValueError):
                            _near_identical = False
                            break
                    if _near_identical:
                        _warn = (
                            f"ABLATION WARNING: Conditions '{_c1}' and '{_c2}' produce "
                            f"near-identical outputs (<1% relative difference) across "
                            f"all {len(_shared_keys)} metrics. The ablation may be trivial."
                        )
                        _ablation_warnings.append(_warn)
                        logger.warning("P8: %s", _warn)

    # --- Write structured experiment summary ---
    summary_payload = {
        "metrics_summary": exp_data["metrics_summary"],
        "total_runs": len(exp_data["runs"]),
        "best_run": exp_data["best_run"],
        "latex_table": exp_data["latex_table"],
        "generated": _utcnow_iso(),
    }
    for version_key in ("result_versions", "selected_version", "run_count_basis"):
        if version_key in exp_data:
            summary_payload[version_key] = exp_data[version_key]
    # R13-1: Detect zero-variance across conditions (all conditions identical primary metric)
    if _condition_summaries and len(_condition_summaries) >= 2:
        _primary_vals = []
        for _cs in _condition_summaries.values():
            if isinstance(_cs, dict):
                # Try 'metrics' dict first (actual structure), then 'primary_metric' fallback
                _metrics = _cs.get("metrics", {})
                if isinstance(_metrics, dict) and _metrics:
                    _pv_candidate = next(iter(_metrics.values()), None)
                    if isinstance(_pv_candidate, dict):
                        _pv_candidate = _pv_candidate.get("mean")
                    if isinstance(_pv_candidate, (int, float)):
                        _primary_vals.append(_pv_candidate)
                        continue
                _pm = _cs.get("primary_metric", {})
                _pv = _pm.get("mean") if isinstance(_pm, dict) else _pm
                if isinstance(_pv, (int, float)):
                    _primary_vals.append(_pv)
        if len(_primary_vals) >= 2 and len(set(_primary_vals)) == 1:
            _zv_warn = (
                f"ZERO VARIANCE: All {len(_primary_vals)} conditions have "
                f"identical primary_metric ({_primary_vals[0]}). "
                f"Experiment condition wiring is likely broken."
            )
            _ablation_warnings.append(_zv_warn)
            logger.warning("R13-1: %s", _zv_warn)

    if _ablation_warnings:
        summary_payload["ablation_warnings"] = _ablation_warnings
    if _all_paired:
        summary_payload["paired_comparisons"] = _all_paired
    if _condition_summaries:
        summary_payload["condition_summaries"] = _condition_summaries
        summary_payload["condition_metrics"] = _condition_summaries  # alias for quality gate
        summary_payload["total_conditions"] = _total_conditions
    if _total_metrics:
        summary_payload["total_metric_keys"] = _total_metrics
    (stage_dir / "experiment_summary.json").write_text(
        json.dumps(summary_payload, indent=2, default=str), encoding="utf-8"
    )
    if exp_data["latex_table"]:
        (stage_dir / "results_table.tex").write_text(
            exp_data["latex_table"], encoding="utf-8"
        )

    # --- Build data-augmented prompt ---
    preamble = _build_context_preamble(
        config, run_dir, include_goal=True, include_hypotheses=True
    )
    data_context = ""
    if exp_data["metrics_summary"]:
        lines = ["\n## Quantitative Results"]
        for mk, mv in exp_data["metrics_summary"].items():
            if isinstance(mv, dict):
                lines.append(
                    f"- {mk}: mean={mv.get('mean', '?')}, min={mv.get('min', '?')}, "
                    f"max={mv.get('max', '?')}, n={mv.get('count', '?')}"
                )
        data_context = "\n".join(lines)

    # Append structured results if available
    if exp_data.get("structured_results"):
        structured_text = json.dumps(
            exp_data["structured_results"], indent=2, default=str
        )
        # Truncate to avoid blowing up context
        if len(structured_text) > 6000:
            structured_text = structured_text[:6000] + "\n... (truncated)"
        data_context += (
            f"\n\n## Structured Experiment Results (from results.json)\n"
            f"```json\n{structured_text}\n```"
        )

    # P8: Inject ablation warnings into data context
    if _ablation_warnings:
        data_context += "\n\nCRITICAL ABLATION WARNINGS:\n"
        for _aw in _ablation_warnings:
            data_context += f"- {_aw}\n"
        data_context += (
            "\nYou MUST address these in your analysis. Identical conditions "
            "mean the ablation design is broken and the comparison is meaningless.\n"
        )

    if llm is not None:
        _pm = prompts or PromptManager()
        from researchclaw.prompts import DEBATE_ROLES_ANALYSIS  # noqa: PLC0415

        # --- Multi-perspective debate ---
        perspectives_dir = stage_dir / "perspectives"
        variables = {
            "preamble": preamble,
            "data_context": data_context,
            "context": context,
        }
        perspectives = _multi_perspective_generate(
            llm, DEBATE_ROLES_ANALYSIS, variables, perspectives_dir
        )
        # --- Synthesize into unified analysis ---
        analysis = _synthesize_perspectives(
            llm, perspectives, "analysis_synthesize", _pm
        )
    else:
        # Template with real data if available
        ms = exp_data["metrics_summary"]
        metrics_block = ""
        if ms:
            for mk, mv in ms.items():
                if isinstance(mv, dict):
                    metrics_block += (
                        f"- **{mk}**: mean={mv.get('mean')}, "
                        f"min={mv.get('min')}, max={mv.get('max')}, n={mv.get('count')}\n"
                    )
        else:
            metrics_block = f"- Primary metric key: `{config.experiment.metric_key}`\n- No quantitative data yet.\n"

        analysis = f"""# Result Analysis

## Metrics Summary
{metrics_block}
## Comparative Findings
- Proposed approach results from {len(exp_data["runs"])} run(s) collected.

## Statistical Checks
- Recommend confidence interval and seed-wise variance reporting.

## Limitations
- Limited runs and synthetic constraints.

## Conclusion
- Proceed to decision stage with moderate confidence.

Generated: {_utcnow_iso()}
"""
    (stage_dir / "analysis.md").write_text(analysis, encoding="utf-8")

    artifacts = ["analysis.md", "experiment_summary.json"]
    if (stage_dir / "results_table.tex").exists():
        artifacts.append("results_table.tex")

    # IMP-6 + FA: Generate charts early (Stage 14) so paper draft can reference them
    # Try FigureAgent first (multi-agent intelligent charts), fall back to visualize.py
    _figure_plan_saved = False
    if config.experiment.figure_agent.enabled and llm is not None:
        try:
            from researchclaw.agents.figure_agent import FigureOrchestrator
            from researchclaw.agents.figure_agent.orchestrator import FigureAgentConfig as _FACfg

            _fa_cfg = _FACfg(
                enabled=True,
                min_figures=config.experiment.figure_agent.min_figures,
                max_figures=config.experiment.figure_agent.max_figures,
                max_iterations=config.experiment.figure_agent.max_iterations,
                render_timeout_sec=config.experiment.figure_agent.render_timeout_sec,
                use_docker=config.experiment.figure_agent.use_docker,
                docker_image=config.experiment.figure_agent.docker_image,
                output_format=config.experiment.figure_agent.output_format,
                gemini_api_key=config.experiment.figure_agent.gemini_api_key,
                gemini_model=config.experiment.figure_agent.gemini_model,
                nano_banana_enabled=config.experiment.figure_agent.nano_banana_enabled,
                strict_mode=config.experiment.figure_agent.strict_mode,
                dpi=config.experiment.figure_agent.dpi,
            )
            _fa = FigureOrchestrator(llm, _fa_cfg, stage_dir=stage_dir)

            # Build conditions list from condition_summaries
            _fa_conditions = list(_condition_summaries.keys()) if _condition_summaries else []

            # BUG-09 fix: pass best_run metrics as fallback data if
            # structured_results is empty, so Planner has some data to chart
            _fa_exp_results = exp_data.get("structured_results", {})
            if not _fa_exp_results and _best_metrics:
                _fa_exp_results = {"best_run_metrics": _best_metrics}

            # Read paper draft for Decision Agent analysis
            _paper_draft = (
                _read_prior_artifact(run_dir, "paper_draft.md")
                or _read_prior_artifact(run_dir, "outline.md")
                or ""
            )

            _fa_plan = _fa.orchestrate({
                "experiment_results": _fa_exp_results,
                "condition_summaries": _condition_summaries,
                "metrics_summary": exp_data.get("metrics_summary", {}),
                "metric_key": config.experiment.metric_key,
                "conditions": _fa_conditions,
                "topic": _read_prior_artifact(run_dir, "topic.md") or config.research.topic,
                "hypothesis": _read_prior_artifact(run_dir, "hypotheses.md") or "",
                "paper_draft": _paper_draft,
                "output_dir": str(stage_dir / "charts"),
            })

            if _fa_plan.figure_count > 0:
                # Save figure plan for Stage 17 to read
                (stage_dir / "figure_plan.json").write_text(
                    json.dumps(_fa_plan.to_dict(), indent=2, default=str),
                    encoding="utf-8",
                )
                _figure_plan_saved = True
                for _cf_name in _fa_plan.get_chart_files():
                    artifacts.append(f"charts/{_cf_name}")
                logger.info(
                    "Stage 14: FigureAgent generated %d charts (%d passed review, %.1fs)",
                    _fa_plan.figure_count,
                    _fa_plan.passed_count,
                    _fa_plan.elapsed_sec,
                )
            else:
                logger.warning("Stage 14: FigureAgent produced no charts, falling back")
        except Exception as _fa_exc:
            logger.warning("Stage 14: FigureAgent failed (%s), falling back to visualize.py", _fa_exc)

    # Fallback: legacy visualize.py chart generation
    if not _figure_plan_saved:
        try:
            from researchclaw.experiment.visualize import (
                generate_all_charts as _gen_charts_early,
            )

            _charts_dir = stage_dir / "charts"
            _early_charts = _gen_charts_early(
                run_dir,
                _charts_dir,
                metric_key=config.experiment.metric_key,
            )
            if _early_charts:
                for _cp in _early_charts:
                    artifacts.append(f"charts/{_cp.name}")
                logger.info(
                    "Stage 14: Generated %d early charts (legacy) for paper embedding",
                    len(_early_charts),
                )
        except Exception as _chart_exc:
            logger.warning("Stage 14: Early chart generation failed: %s", _chart_exc)

    # --- Register baseline results to shared registry ---
    _shared_dir = getattr(config.experiment, "shared_results_dir", "") or ""
    if _shared_dir:
        try:
            import sys as _sys_rr
            _services_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "services")
            if _services_dir not in _sys_rr.path:
                _sys_rr.path.insert(0, _services_dir)
            from result_registry import ResultRegistry
            _registry = ResultRegistry(_shared_dir)
            _summary_path = stage_dir / "experiment_summary.json"
            if _summary_path.exists():
                _summary = json.loads(_summary_path.read_text(encoding="utf-8"))
                _best = _summary.get("best_run", {})
                _metrics = _best.get("metrics", {})
                if _metrics:
                    _registry.register(
                        project_id=getattr(config.project, "name", "unknown"),
                        description=f"{config.research.topic[:100]} - baseline results",
                        model="",
                        dataset="",
                        task=config.research.topic[:50],
                        metrics={k: float(v) for k, v in _metrics.items() if isinstance(v, (int, float))},
                        tags=[d.lower() for d in config.research.domains] + ["baseline"],
                        source_stage=int(Stage.RESULT_ANALYSIS),
                    )
                    logger.info("Registered %d metrics to shared results registry", len(_metrics))
                    print(f"[SHARED RESULTS] Registered baseline metrics: {list(_metrics.keys())}", flush=True)
        except Exception as _rr_exc:
            logger.debug("Shared results registration failed: %s", _rr_exc)

    return StageResult(
        stage=Stage.RESULT_ANALYSIS,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-14/{a}" for a in artifacts),
    )


def _parse_decision(text: str) -> str:
    """Extract PROCEED/PIVOT/REFINE from decision text.

    Looks for the first standalone keyword on its own line after a
    ``## Decision`` heading.  Falls back to a keyword scan of the first
    few lines after the heading, but only matches the keyword itself
    (not mentions inside explanatory prose like "PIVOT is not warranted").
    Returns lowercase ``"proceed"`` / ``"pivot"`` / ``"refine"``.
    Defaults to ``"proceed"`` if nothing matches.
    """
    import re as _re

    text_upper = text.upper()
    # Look in the first occurrence after "## Decision" heading
    decision_section = ""
    for keyword in ("## DECISION", "## Decision", "## decision"):
        if keyword.upper() in text_upper:
            idx = text_upper.index(keyword.upper())
            decision_section = text[idx : idx + 200]
            break
    search_text = decision_section or text[:500]

    # First try: look for a line that is just the keyword (possibly with
    # whitespace / markdown bold / trailing punctuation).
    for line in search_text.splitlines():
        stripped = line.strip().strip("*").strip("#").strip()
        if stripped.upper() in ("PROCEED", "PIVOT", "REFINE"):
            return stripped.lower()

    # Fallback: regex for standalone word boundaries so that
    # "PIVOT is not warranted" does NOT match as a decision.
    for kw in ("PIVOT", "REFINE", "PROCEED"):
        # Only match if the keyword appears as the FIRST keyword-class token
        # on its own (not embedded in a sentence saying "not PIVOT").
        pattern = _re.compile(
            r"(?:^|##\s*Decision\s*\n\s*)" + kw, _re.IGNORECASE | _re.MULTILINE
        )
        if pattern.search(search_text):
            return kw.lower()

    # Last resort: simple containment (original behavior)
    search_upper = search_text.upper()
    if "REFINE" in search_upper:
        return "refine"
    if "PIVOT" in search_upper:
        return "pivot"
    return "proceed"


def _decision_context(run_dir: Path, config: RCConfig) -> str:
    """Facts S17 needs to choose between PROCEED and REFINE."""
    from researchclaw.pipeline.stages import MAX_DECISION_LOOPS

    lines = ["\n\n## Decision context (from the platform)"]
    lines.append("User research scope and constraints take precedence over default baseline "
                 "recommendations. Do not expand a fixed training budget or rerun completed "
                 "conditions to satisfy an unrequested baseline.\n" + config.research.topic)
    selected = getattr(config.experiment, "optional_experiments", None)
    if selected is not None:
        from researchclaw.training.execution import optional_experiments_directive

        lines.append(optional_experiments_directive(selected))
        lines.append("An optional experiment marked 'Not requested' is never missing and is "
                     "not a reason for REFINE.")
    used = 0
    history_path = run_dir / "decision_history.json"
    if history_path.is_file():
        history = _safe_json_loads(history_path.read_text(encoding="utf-8"), [])
        if isinstance(history, list):
            used = sum(1 for h in history if isinstance(h, dict) and h.get("decision") == "refine")
    left = max(MAX_DECISION_LOOPS.get("refine", 0) - used, 0)
    lines.append(
        f"REFINE loops left in this run: {left}. A REFINE loop re-runs S15 to add the missing "
        "experiments under the same frozen protocol; it never re-tunes hyperparameters."
    )
    if left == 0:
        lines.append("No REFINE loop is left: unless results are unusable, choose PROCEED and list "
                     "the missing experiments as limitations.")
    if MAX_DECISION_LOOPS.get("pivot", 0) == 0:
        lines.append("PIVOT is not available in this pipeline; choose REFINE or PROCEED.")
    return "\n".join(lines) + "\n"


def _execute_research_decision(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    analysis = _read_prior_artifact(run_dir, "analysis.md") or ""

    # P6: Detect degenerate REFINE cycles — inject warning if metrics stagnate
    _degenerate_hint = ""
    _refine_log = _read_prior_artifact(run_dir, "refinement_log.json")
    if _refine_log:
        try:
            _rl = json.loads(_refine_log)
            _iters = _rl.get("iterations", [])
            if not isinstance(_iters, list):
                _iters = []
            _metrics = [it.get("metric") for it in _iters if isinstance(it, dict)]
            _valid = [m for m in _metrics if m is not None]
            _all_saturated = _valid and all(m <= 0.001 or m >= 0.999 for m in _valid)
            _all_identical = len(set(_valid)) <= 1 and len(_valid) >= 2
            if _all_saturated or _all_identical:
                _degenerate_hint = (
                    "\n\nSYSTEM WARNING — DEGENERATE REFINE CYCLE DETECTED:\n"
                    f"Metrics across {len(_valid)} iterations: {_valid}\n"
                    "All iterations produce identical/saturated results. Further REFINE "
                    "cycles CANNOT fix this — the underlying benchmark design is too "
                    "easy/hard. You SHOULD choose PROCEED with a quality caveat rather "
                    "than REFINE again.\n"
                )
                logger.warning("P6: Degenerate refine cycle detected, injecting PROCEED hint")
        except (json.JSONDecodeError, OSError):
            pass

    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "research_decision")
        sp = _pm.for_stage("research_decision", evolution_overlay=_overlay, analysis=analysis)
        _user = sp.user + _degenerate_hint + _decision_context(run_dir, config)
        resp = llm.chat(
            [{"role": "user", "content": _user}],
            system=sp.system,
        )
        decision_md = resp.content
    else:
        decision_md = f"""# Research Decision

## Decision
PROCEED

## Justification
Current evidence suggests measurable progress with actionable limitations.

## Next Actions
- Build detailed paper outline
- Expand ablation and uncertainty analysis in writing

Generated: {_utcnow_iso()}
"""
    (stage_dir / "decision.md").write_text(decision_md, encoding="utf-8")

    # --- Extract structured decision ---
    decision = _parse_decision(decision_md)

    # T3.1: Validate decision quality — check for minimum experiment rigor
    _quality_warnings: list[str] = []
    _dec_lower = decision_md.lower()
    if "baseline" not in _dec_lower and "control" not in _dec_lower:
        _quality_warnings.append("Decision text does not mention baselines")
    if "seed" not in _dec_lower and "replicat" not in _dec_lower and "run" not in _dec_lower:
        _quality_warnings.append("Decision text does not mention multi-seed/replicate runs")
    if "metric" not in _dec_lower and "accuracy" not in _dec_lower and "loss" not in _dec_lower:
        _quality_warnings.append("Decision text does not mention evaluation metrics")
    if _quality_warnings:
        logger.warning("T3.1: Decision quality warnings: %s", _quality_warnings)

    decision_payload = {
        "decision": decision,
        "raw_text_excerpt": decision_md[:500],
        "quality_warnings": _quality_warnings,
        "generated": _utcnow_iso(),
    }
    (stage_dir / "decision_structured.json").write_text(
        json.dumps(decision_payload, indent=2), encoding="utf-8"
    )
    logger.info("Research decision: %s", decision)

    return StageResult(
        stage=Stage.RESEARCH_DECISION,
        status=StageStatus.DONE,
        artifacts=("decision.md", "decision_structured.json"),
        evidence_refs=("stage-15/decision.md",),
        decision=decision,
    )


def _execute_knowledge_summary(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
    **kwargs: object,
) -> StageResult:
    """Summarize experiment findings into the shared knowledge base.

    Reads analysis, decision, and experiment plan, then produces a structured
    knowledge entry (JSON) and appends it to the shared knowledge base.
    L1 lobsters read this knowledge base when generating new hypotheses.
    """
    from researchclaw.llm import create_llm_client

    if llm is None:
        llm = create_llm_client(config)

    analysis = _read_prior_artifact(run_dir, "analysis.md") or ""
    decision = _read_prior_artifact(run_dir, "decision.md") or ""
    exp_plan = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
    hypotheses = _read_prior_artifact(run_dir, "hypotheses.md") or ""

    summary_prompt = (
        "Based on the following research experiment, create a structured knowledge entry.\n\n"
        "## Hypotheses\n" + hypotheses[:2000] + "\n\n"
        "## Experiment Plan\n" + exp_plan[:2000] + "\n\n"
        "## Analysis\n" + analysis[:3000] + "\n\n"
        "## Decision\n" + decision[:1000] + "\n\n"
        "Create a JSON object with these fields:\n"
        '- "topic": one-line research topic\n'
        '- "hypotheses": list of hypotheses tested\n'
        '- "method": experimental methodology summary (2-3 sentences)\n'
        '- "settings": key experimental settings (model, dataset, hyperparams, hardware)\n'
        '- "results": dict of metric_name -> value for main results\n'
        '- "conclusions": list of key conclusions (what worked, what didn\'t)\n'
        '- "insights": list of surprising or useful insights for future research\n'
        '- "limitations": list of limitations\n'
        '- "suggested_directions": list of promising follow-up directions\n'
        "Return valid JSON only."
    )

    try:
        resp = llm.chat(
            [{"role": "user", "content": summary_prompt}],
            system="You are a research scientist. Summarize experiment findings into a structured knowledge entry. Return valid JSON only.",
            json_mode=True,
        )
        content = resp.content if hasattr(resp, 'content') else str(resp)

        import json as _json_ks
        try:
            entry = _json_ks.loads(content)
        except _json_ks.JSONDecodeError:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            entry = _json_ks.loads(content)

        if not isinstance(entry, dict):
            entry = {"raw": str(entry)}

        entry["project_id"] = getattr(config.project, "name", "unknown")
        entry["research_topic"] = config.research.topic
        entry["domains"] = list(config.research.domains)
        entry["timestamp"] = _utcnow_iso()

        (stage_dir / "knowledge_entry.json").write_text(
            _json_ks.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8",
        )

        # Write to shared knowledge base
        _shared_dir = getattr(config.experiment, "shared_results_dir", "") or ""
        if _shared_dir:
            import os as _os_ks
            kb_dir = Path(_shared_dir) / "knowledge_base"
            kb_dir.mkdir(parents=True, exist_ok=True)
            kb_file = kb_dir / f"{entry['project_id']}_{_utcnow_iso().replace(':', '-')}.json"
            kb_file.write_text(
                _json_ks.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8",
            )

            # Append summary to knowledge_index.jsonl for quick L1 lookup
            index_file = kb_dir / "knowledge_index.jsonl"
            compact = {
                "project": entry.get("project_id", ""),
                "topic": entry.get("topic", entry.get("research_topic", "")),
                "conclusions": entry.get("conclusions", []),
                "insights": entry.get("insights", []),
                "suggested_directions": entry.get("suggested_directions", []),
                "results": entry.get("results", {}),
            }
            with open(index_file, "a", encoding="utf-8") as f:
                f.write(_json_ks.dumps(compact, ensure_ascii=False) + "\n")

            logger.info("Knowledge entry written to shared KB: %s", kb_file.name)
            print(f"[KNOWLEDGE] Written to shared KB: {len(entry.get('conclusions', []))} conclusions, {len(entry.get('insights', []))} insights", flush=True)

    except Exception as e:
        logger.warning("Knowledge summary generation failed: %s", e)
        (stage_dir / "knowledge_entry.json").write_text(
            json.dumps({"error": str(e), "topic": config.research.topic}, indent=2),
            encoding="utf-8",
        )

    return StageResult(
        stage=Stage.KNOWLEDGE_SUMMARY,
        status=StageStatus.DONE,
        artifacts=("knowledge_entry.json",),
        evidence_refs=("analysis.md", "decision.md"),
    )


def _execute_paper_outline(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    analysis = _read_prior_artifact(run_dir, "analysis.md") or ""
    decision = _read_prior_artifact(run_dir, "decision.md") or ""
    _disc_dir = _find_discussion_dir(run_dir, config)
    _has_discussion = _disc_dir is not None
    preamble = _build_context_preamble(
        config,
        run_dir,
        include_analysis=True,
        include_decision=True,
        include_experiment_data=True,
        include_discussion=_has_discussion,
    )

    # Build discussion ablation instruction if data exists
    discussion_ablation = ""
    if _has_discussion:
        discussion_ablation = (
            "\n\nIMPORTANT — MULTI-AGENT DISCUSSION ABLATION:\n"
            "The research idea for this paper was refined through a multi-agent "
            "discussion process (S8). Include an ablation section in the outline "
            "that compares pre-discussion individual syntheses vs post-discussion "
            "consensus. This should appear in the Discussion or Ablation Studies "
            "section of the paper. The discussion data is provided in the preamble.\n"
        )

    # WS-5.2: Read iteration feedback if available (multi-round iteration)
    feedback = ""
    iter_ctx_path = run_dir / "iteration_context.json"
    if iter_ctx_path.exists():
        try:
            ctx = json.loads(iter_ctx_path.read_text(encoding="utf-8"))
            iteration = ctx.get("iteration", 1)
            prev_score = ctx.get("quality_score")
            reviews_excerpt = ctx.get("reviews_excerpt", "")
            if iteration > 1 and reviews_excerpt:
                feedback = (
                    f"\n\n## Iteration {iteration} Feedback\n"
                    f"Previous quality score: {prev_score}/10\n"
                    f"Reviewer feedback to address:\n{reviews_excerpt[:2000]}\n"
                    f"\nYou MUST address these reviewer concerns in this revision.\n"
                )
        except (json.JSONDecodeError, KeyError):
            pass

    if llm is not None:
        _pm = prompts or PromptManager()
        # IMP-20: Pass academic style guide block for outline stage
        try:
            _asg = _pm.block("academic_style_guide")
        except (KeyError, Exception):
            _asg = ""
        _overlay = _get_evolution_overlay(run_dir, "paper_outline")
        sp = _pm.for_stage(
            "paper_outline",
            evolution_overlay=_overlay,
            preamble=preamble,
            topic_constraint=_pm.block("topic_constraint", topic=config.research.topic),
            feedback=feedback,
            analysis=analysis,
            decision=decision,
            academic_style_guide=_asg,
            discussion_ablation=discussion_ablation,
        )
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        outline = resp.content
        # Reasoning models may consume all tokens on CoT — retry with more
        if not outline.strip() and sp.max_tokens:
            logger.warning("Empty outline from LLM — retrying with 2x tokens")
            resp = _chat_with_prompt(
                llm,
                sp.system,
                sp.user,
                json_mode=sp.json_mode,
                max_tokens=sp.max_tokens * 2,
            )
            outline = resp.content
        if not outline.strip():
            logger.warning("LLM returned empty outline — using default")
            outline = _default_paper_outline(config.research.topic)
    else:
        outline = _default_paper_outline(config.research.topic)
    (stage_dir / "outline.md").write_text(outline, encoding="utf-8")
    return StageResult(
        stage=Stage.PAPER_OUTLINE,
        status=StageStatus.DONE,
        artifacts=("outline.md",),
        evidence_refs=("stage-16/outline.md",),
    )


def _load_experiment_summary(run_dir: Path) -> dict[str, Any] | None:
    """Load the latest experiment_summary.json written by RESULT_ANALYSIS."""
    text = _read_prior_artifact(run_dir, "experiment_summary.json")
    data = _safe_json_loads(text, None) if text else None
    return data if isinstance(data, dict) else None


def _compact_run_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep the scalar fields of one per-seed run (drop per-epoch arrays)."""
    compact: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, dict):
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float, bool))}
            if scalars:
                compact[key] = scalars
    return compact


def _verified_execution_audit(run_dir: Path) -> dict[str, Any]:
    """Summarise the platform's own check of the adopted experiment run.

    Reads the execution_report.json written by ``run_experiment`` for S15
    (``verified_execution/``) or, when S15 did not run or its result was not
    adopted, the last S14 attempt.  Returns {} when no report exists.
    """
    refine_log_path = _find_prior_file(run_dir, "refinement_log.json")
    refine_log = _safe_json_loads(refine_log_path.read_text(encoding="utf-8"), {}) if refine_log_path else {}
    if not isinstance(refine_log, dict):
        refine_log = {}

    candidates: list[Path] = []
    if refine_log.get("adopted", True):
        candidates += _newest_stage_files(run_dir, "stage-15*/verified_execution/execution_report.json")
    candidates += _newest_stage_files(run_dir, "stage-14*/runs/attempt-*/execution_report.json")
    report_path = next((c for c in candidates if c.is_file()), None)
    if report_path is None:
        return {}
    report = _safe_json_loads(report_path.read_text(encoding="utf-8"), {})
    if not isinstance(report, dict):
        return {}

    audit: dict[str, Any] = {
        "source": report_path.relative_to(run_dir).as_posix(),
        "verified_by": "platform run_experiment (re-executed main.py under supervision)",
    }
    for key in ("success", "returncode", "elapsed_sec", "gpu_id", "source_sha256", "results_sha256", "errors"):
        if key in report:
            audit[key] = report[key]
    training = report.get("training")
    if isinstance(training, dict):
        audit["training"] = {
            key: training[key]
            for key in (
                "optimizer_steps", "parameter_update_observed", "cuda_update_observed",
                "checkpoint_verified", "trained_optimizers", "last_loss", "scope", "loss_semantics",
            )
            if key in training
        }
        matches = training.get("checkpoint_matches")
        if isinstance(matches, list):
            audit["training"]["checkpoints"] = [
                {k: m[k] for k in ("checkpoint", "state") if k in m}
                for m in matches if isinstance(m, dict)
            ]
        gpu = training.get("gpu_process")
        if isinstance(gpu, dict):
            audit["training"]["gpu_process_memory_mib"] = gpu.get("memory_mib")
    predictions = report.get("predictions")
    if isinstance(predictions, dict):
        groups = [g for g in predictions.get("groups") or [] if isinstance(g, dict)]
        audit["predictions"] = {
            "groups": len(groups),
            "files_checked": sum(g.get("checked", 0) for g in groups),
            "degenerate": len(predictions.get("degenerate") or []),
        }
    results = report.get("results")
    if isinstance(results, dict):
        runs = results.get("runs")
        if isinstance(runs, list):
            audit["per_run_results"] = [_compact_run_record(r) for r in runs if isinstance(r, dict)]
        if isinstance(results.get("protocol"), dict):
            audit["protocol"] = results["protocol"]
    if refine_log:
        audit["iterative_refine"] = {
            key: refine_log[key]
            for key in ("adopted", "improved", "baseline_metric", "final_metric",
                        "protocol_changes", "exploratory_metric", "rejection_reason")
            if key in refine_log
        }
    return audit


def _execution_audit_section(run_dir: Path, heading: str) -> str:
    audit = _verified_execution_audit(run_dir)
    if not audit:
        return ""
    return (
        f"\n\n{heading}\n"
        "The platform re-executed the adopted experiment and recorded the checks below "
        "(parameter updates, saved checkpoints, test-set predictions, per-seed results). "
        "Treat them as verified; do not call training, checkpoints or per-seed results "
        "unverified when they are listed here. `scope` states what was NOT verified.\n"
        f"```json\n{json.dumps(audit, indent=2, ensure_ascii=False, default=str)[:8000]}\n```\n"
    )


_EVIDENCE_SKIP_KEYS = {"per_seed_results", "epoch_series", "latex_table", "generated_charts"}
_EVIDENCE_FIELD_MAX_CHARS = 6000


def _paper_experiment_evidence(run_dir: Path) -> str:
    """Build a compact, auditable experiment snapshot for paper writing."""
    summary = _load_experiment_summary(run_dir)
    if not summary:
        return _execution_audit_section(run_dir, "## VERIFIED EXECUTION AUDIT")

    snapshot: dict[str, Any] = {
        key: summary[key]
        for key in (
            "metrics_summary",
            "metrics_summary_scope",
            "total_runs",
            "total_conditions",
            "best_run",
            "best_condition",
            "paired_comparisons",
            "provenance",
            "audit",
            "limitations",
            "approval_compliance",
            "input_json_count",
            "smoke_runs_included",
        )
        if key in summary
    }

    condition_summaries = summary.get("condition_summaries")
    if isinstance(condition_summaries, dict):
        snapshot["condition_summaries"] = condition_summaries
    # Project-specific results (robustness grids, reference checkpoints, ...)
    # have no fixed key; keep every compact field, skip bulky series.
    for key, value in summary.items():
        if key in snapshot or key in _EVIDENCE_SKIP_KEYS:
            continue
        if len(json.dumps(value, ensure_ascii=False, default=str)) <= _EVIDENCE_FIELD_MAX_CHARS:
            snapshot[key] = value

    compact_runs: list[dict[str, Any]] = []
    per_seed_results = summary.get("per_seed_results")
    if isinstance(per_seed_results, list):
        for record in per_seed_results:
            if not isinstance(record, dict):
                continue
            compact_record = {
                key: record[key]
                for key in (
                    "condition", "seed", "best_epoch", "metrics",
                    "optimizer_steps", "backward_steps",
                )
                if key in record
            }
            compact_epochs: list[dict[str, Any]] = []
            epochs = record.get("epochs")
            if isinstance(epochs, list):
                for epoch in epochs:
                    if not isinstance(epoch, dict):
                        continue
                    compact_epoch = {
                        key: epoch[key]
                        for key in ("epoch", "mean_loss", "samples")
                        if key in epoch
                    }
                    validation = epoch.get("validation")
                    if isinstance(validation, dict):
                        compact_epoch["validation"] = {
                            key: validation[key]
                            for key in ("f1", "iou", "sample_count")
                            if key in validation
                        }
                    compact_epochs.append(compact_epoch)
            if compact_epochs:
                compact_record["epochs"] = compact_epochs
            compact_runs.append(compact_record)
    if compact_runs:
        snapshot["per_seed_results"] = compact_runs
    audit = _verified_execution_audit(run_dir)
    if audit:
        snapshot["verified_execution"] = audit

    summary_path = _find_prior_file(run_dir, "experiment_summary.json")
    source = str(summary_path.relative_to(run_dir)) if summary_path else "experiment_summary.json"
    payload = json.dumps(snapshot, indent=2, ensure_ascii=False)
    return (
        "\n\n## AUTHORITATIVE EXPERIMENT EVIDENCE FOR PAPER WRITING\n"
        f"Source: `{source}`\n"
        "The structured snapshot below is authoritative for completed runs, metrics, "
        "hardware, checkpoints, protocol deviations, and limitations. It is compacted "
        "only by removing per-image arrays and sample-loss arrays.\n"
        "```json\n"
        f"{payload}\n"
        "```\n"
        "EVIDENCE SYNCHRONIZATION RULES:\n"
        "- Update stale draft or review statements when this snapshot contains newer evidence.\n"
        "- Do not say that only aggregate means exist when per-seed results are listed.\n"
        "- Do not say training, GPU use, or checkpoints are unverified when the audit records "
        "them; describe the exact level of verification instead.\n"
        "- Preserve every limitation, approval deviation, test-exposure warning, and "
        "independent-replay caveat in the snapshot.\n"
        "- Never convert file-existence checks into claims of checkpoint reload or independent "
        "metric reproduction.\n"
        "- If the draft conflicts with this snapshot, revise the draft and retain the supported "
        "scientific boundary.\n"
        "- `verified_execution` is the platform's own re-execution of the adopted run; report its "
        "per-run results and checkpoint checks, and if `iterative_refine.adopted` is false, do not "
        "report the exploratory S15 metric as a result.\n"
        "- Fill every table cell whose value is in the snapshot (robustness, reference or "
        "public-checkpoint rows included); use a dash only for values the snapshot lacks.\n"
        "- Round reported metrics to at most 4 decimal places.\n"
        "- Do not call a difference statistically significant or not significant unless the "
        "snapshot contains that test; with single runs, say no significance test was possible.\n"
    )


def _paper_evidence_conflicts(paper: str, run_dir: Path) -> list[str]:
    """Detect claims that contradict the authoritative experiment summary."""
    summary = _load_experiment_summary(run_dir) or {}
    lowered = " ".join(paper.lower().split())
    conflicts: list[str] = []

    per_seed_results = summary.get("per_seed_results")
    if isinstance(per_seed_results, list) and per_seed_results:
        aggregate_only_patterns = (
            "only aggregate means",
            "contains aggregate means, not the full run-by-epoch",
            "does not contain the full run-by-epoch",
            "cannot be reconstructed from those means",
        )
        if any(pattern in lowered for pattern in aggregate_only_patterns):
            conflicts.append(
                "Paper says only aggregate means are available although per-seed results exist."
            )

    audit = summary.get("audit")
    completed_audits = []
    if isinstance(audit, dict):
        completed_audits = [
            value for value in audit.values()
            if isinstance(value, dict) and value.get("status") == "completed"
        ]
    if completed_audits:
        completion_patterns = (
            "no new execution occurred",
            "actual completed progress for the independent task therefore remain unverified",
            "does not establish completion of the newly authorized server run",
        )
        if any(pattern in lowered for pattern in completion_patterns):
            conflicts.append(
                "Paper denies completed execution although the audit contains completed runs."
            )

    return conflicts


def _paper_evidence_conflicts(paper: str, run_dir: Path) -> list[str]:
    """Detect claims that contradict the authoritative experiment summary."""
    summary = _load_experiment_summary(run_dir) or {}
    lowered = " ".join(paper.lower().split())
    conflicts: list[str] = []

    per_seed_results = summary.get("per_seed_results")
    if isinstance(per_seed_results, list) and per_seed_results:
        aggregate_only_patterns = (
            "only aggregate means",
            "contains aggregate means, not the full run-by-epoch",
            "does not contain the full run-by-epoch",
            "cannot be reconstructed from those means",
        )
        if any(pattern in lowered for pattern in aggregate_only_patterns):
            conflicts.append(
                "Paper says only aggregate means are available although per-seed results exist."
            )

    audit = summary.get("audit")
    completed_audits = []
    if isinstance(audit, dict):
        completed_audits = [
            value for value in audit.values()
            if isinstance(value, dict) and value.get("status") == "completed"
        ]
    if completed_audits:
        completion_patterns = (
            "no new execution occurred",
            "actual completed progress for the independent task therefore remain unverified",
            "does not establish completion of the newly authorized server run",
        )
        if any(pattern in lowered for pattern in completion_patterns):
            conflicts.append(
                "Paper denies completed execution although the audit contains completed runs."
            )

    return conflicts


_MAX_PAYLOAD_METRICS = 200
# Bookkeeping fields that appear next to metrics in results payloads
_NON_METRIC_KEYS = frozenset({
    "elapsed_sec", "elapsed_seconds", "training_steps", "seed", "seeds",
    "images_per_condition", "iterations", "tool_calls", "gpu_id", "smoke_test",
    "metadata", "config",
})
_METRIC_CONTAINER_KEYS = frozenset({
    "conditions", "condition_scores", "metrics", "key_metrics",
})


def _run_payload_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract flat numeric metrics from a run JSON payload.

    Supports the legacy sandbox record (``metrics`` / ``key_metrics``), the
    agentic EXPERIMENT_RUN ``run_report.json`` (metrics nested under
    ``results``) and a raw harness ``results.json``. Agent-written results
    use free-form layouts (``conditions``, ``condition_scores``, per-seed
    nesting, ...). Only known top-level wrappers are removed; condition
    names and other namespaces are preserved. Both ``{"ours": {"acc": 0.7}}``
    and ``{"conditions": {"ours": {"acc": 0.7}}}`` become ``{"ours/acc": 0.7}``.
    """
    for key in ("metrics", "key_metrics"):
        value = payload.get(key)
        if isinstance(value, dict) and value:
            return value

    if "results" in payload or "has_results" in payload:
        # run_report.json: only the nested results hold metrics
        results = payload.get("results")
        if not isinstance(results, dict):
            return {}
    else:
        results = payload

    def _is_num(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    flat: dict[str, Any] = {}

    # Held-out results are often nested as condition/split/dataset/(corruption/)metric
    def _walk(node: dict[str, Any], prefix: str, depth: int) -> None:
        for metric_name, metric_value in node.items():
            if metric_name in _NON_METRIC_KEYS:
                continue
            path = f"{prefix}/{metric_name}" if prefix else str(metric_name)
            if _is_num(metric_value):
                flat.setdefault(path, metric_value)
            elif isinstance(metric_value, dict) and depth < 8:
                container_path = (
                    "" if depth == 0 and metric_name in _METRIC_CONTAINER_KEYS else path
                )
                _walk(metric_value, container_path, depth + 1)

    _walk(results, "", 0)
    if len(flat) <= _MAX_PAYLOAD_METRICS:
        return flat

    # Over the cap: keep the declared metric (e.g. pixel_f1 inside
    # "macro_pixel_f1_error") and shallow paths before per-step diagnostics.
    declared = str(results.get("metric_name") or payload.get("metric_name") or "")

    def _rank(path: str) -> tuple[int, int]:
        leaf = path.rsplit("/", 1)[-1]
        is_target = leaf == "primary_metric" or (len(leaf) >= 3 and leaf in declared)
        return (0 if is_target else 1, path.count("/"))

    kept = set(sorted(flat, key=_rank)[:_MAX_PAYLOAD_METRICS])
    return {path: value for path, value in flat.items() if path in kept}


def _collect_raw_experiment_metrics(run_dir: Path) -> tuple[str, bool]:
    """Collect raw experiment metric lines from stdout for paper writing.

    Returns a tuple of (formatted block, has_parsed_metrics).
    ``has_parsed_metrics`` is True when at least one run had a non-empty
    ``metrics`` dict in its JSON payload — a reliable signal of real data.
    """
    metric_lines: list[str] = []
    run_count = 0
    has_parsed_metrics = False

    experiment_summary = _load_experiment_summary(run_dir)
    if experiment_summary and experiment_summary.get("metrics_summary"):
        summary_metrics = experiment_summary["metrics_summary"]
        if isinstance(summary_metrics, dict):
            for metric_name, metric_stats in summary_metrics.items():
                metric_value = (
                    metric_stats.get("mean") if isinstance(metric_stats, dict) else metric_stats
                )
                if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                    metric_lines.append(f"  {metric_name}: {metric_value}")
            if metric_lines:
                summary_run_count = experiment_summary.get("total_runs")
                if not isinstance(summary_run_count, int) or isinstance(summary_run_count, bool):
                    summary_run_count = 0
                return _format_experiment_metrics(
                    metric_lines, summary_run_count, True,
                    run_count_basis=experiment_summary.get("run_count_basis", ""),
                )

    version_data = _collect_experiment_results(run_dir)
    if version_data.get("run_count_basis") == "result_records":
        final_metrics = (version_data.get("best_run") or {}).get("metrics", {})
        if final_metrics:
            record_count = len(version_data["runs"])
            count_basis = "result_records"
            if experiment_summary:
                summary_run_count = experiment_summary.get("total_runs")
                if (
                    isinstance(summary_run_count, int)
                    and not isinstance(summary_run_count, bool)
                    and summary_run_count > 0
                ):
                    record_count = summary_run_count
                    count_basis = experiment_summary.get("run_count_basis", "")
            return _format_experiment_metrics(
                [f"  {metric_name}: {metric_value}" for metric_name, metric_value in final_metrics.items()],
                record_count, True, run_count_basis=count_basis,
            )

    for stage_subdir in sorted(run_dir.glob("stage-*/runs")):
        for run_file in sorted(stage_subdir.glob("*.json")):
            if run_file.name == "results.json":
                continue
            try:
                payload = json.loads(run_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, dict):
                continue

            # R10: Skip simulated data — only collect real experiment results
            if payload.get("status") == "simulated":
                continue

            run_count += 1

            # Extract from parsed metrics ('metrics', 'key_metrics', or the
            # agentic run_report.json 'results')
            metrics = _run_payload_metrics(payload)
            if metrics:
                has_parsed_metrics = True
                for k, v in metrics.items():
                    metric_lines.append(f"  {k}: {v}")

            # Also extract from stdout for full detail
            # BUG-23: Filter out infrastructure lines that are NOT experiment results
            _INFRA_KEYS = {
                "SEED_COUNT", "TIME_ESTIMATE", "TRAINING_STEPS",
                "REGISTERED_CONDITIONS", "METRIC_DEF", "GPU_MEMORY",
                "BATCH_SIZE", "NUM_WORKERS", "TOTAL_PARAMS",
                "time_budget_sec", "max_epochs", "num_seeds",
            }
            stdout = payload.get("stdout", "")
            if stdout:
                for line in stdout.splitlines():
                    line = line.strip()
                    if ":" in line:
                        parts = line.rsplit(":", 1)
                        try:
                            float(parts[1].strip())
                            key_part = parts[0].strip().split("/")[-1]  # last segment
                            if key_part in _INFRA_KEYS:
                                continue  # skip infrastructure lines
                            metric_lines.append(f"  {line}")
                        except (ValueError, TypeError, IndexError):
                            pass

    # R19-4 + R23-1: Collect metrics from the ITERATIVE_REFINE stage.
    # If refinement has richer data than EXPERIMENT_RUN runs/, REPLACE that
    # data to avoid confusing the paper writer with conflicting sources.
    _refine_lines: list[str] = []
    _refine_run_count = 0
    # Scan ALL refinement logs across versions, pick the richest
    _best_refine_metrics: dict[str, Any] = {}
    _best_refine_stdout = ""
    _refine_glob = f"stage-{int(Stage.ITERATIVE_REFINE):02d}*"
    # Agentic refinement copies the final harness output to results.json
    for _res_path in sorted(run_dir.glob(f"{_refine_glob}/results.json")):
        _res = _safe_json_loads(_res_path.read_text(encoding="utf-8"), None)
        if isinstance(_res, dict):
            _res_metrics = _run_payload_metrics(_res)
            if _res_metrics:
                has_parsed_metrics = True
            if len(_res_metrics) > len(_best_refine_metrics):
                _best_refine_metrics = _res_metrics
                _best_refine_stdout = ""
    for _rl_path in sorted(run_dir.glob(f"{_refine_glob}/refinement_log.json")):
        try:
            _rlog = json.loads(_rl_path.read_text(encoding="utf-8"))
            _best_ver = _rlog.get("best_version", "")
            _rlog_iters = _rlog.get("iterations", [])
            if not isinstance(_rlog_iters, list):
                _rlog_iters = []
            for _it in _rlog_iters:
                for _sbx_key in ("sandbox", "sandbox_after_fix"):
                    _sbx = _it.get(_sbx_key, {})
                    if not isinstance(_sbx, dict):
                        continue
                    _sbx_metrics = _sbx.get("metrics", {})
                    if isinstance(_sbx_metrics, dict) and len(_sbx_metrics) > len(_best_refine_metrics):
                        _best_refine_metrics = _sbx_metrics
                        _best_refine_stdout = _sbx.get("stdout", "")
        except (json.JSONDecodeError, OSError):
            pass

    if _best_refine_metrics and len(_best_refine_metrics) > len(metric_lines) // 2:
        # Refinement has richer data — REPLACE Stage 12 data to avoid conflicts
        metric_lines = []
        run_count = 1
        for k, v in _best_refine_metrics.items():
            metric_lines.append(f"  {k}: {v}")
        # Also extract PAIRED and metric lines from stdout
        if _best_refine_stdout:
            for _line in _best_refine_stdout.splitlines():
                _line = _line.strip()
                if _line.startswith("PAIRED:"):
                    metric_lines.append(f"  {_line}")
                elif ":" in _line:
                    parts = _line.rsplit(":", 1)
                    try:
                        float(parts[1].strip())
                        metric_lines.append(f"  {_line}")
                    except (ValueError, TypeError, IndexError):
                        pass
    elif _best_refine_metrics:
        # Refinement has some data but not richer — append to existing
        run_count += 1
        for k, v in _best_refine_metrics.items():
            metric_lines.append(f"  {k}: {v}")
        if _best_refine_stdout:
            for _line in _best_refine_stdout.splitlines():
                _line = _line.strip()
                if _line.startswith("PAIRED:"):
                    metric_lines.append(f"  {_line}")

    if not metric_lines:
        return "", has_parsed_metrics

    # Agentic runs write one run_report.json for all conditions/seeds; the
    # RESULT_ANALYSIS summary records the actual number of runs.
    _exp_summary = _load_experiment_summary(run_dir)
    _total_runs = _exp_summary.get("total_runs") if _exp_summary else None
    if isinstance(_total_runs, int) and not isinstance(_total_runs, bool) and _total_runs > 0:
        run_count = _total_runs

    return _format_experiment_metrics(metric_lines, run_count, has_parsed_metrics)


def _format_experiment_metrics(
    metric_lines: list[str],
    run_count: int,
    has_parsed_metrics: bool,
    *,
    run_count_basis: str = "",
) -> tuple[str, bool]:
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for line in metric_lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)

    # BUG-29: Reformat raw metric lines into human-readable condition summaries
    # to prevent LLM from pasting raw path-style lines into the paper
    _grouped: dict[str, list[str]] = {}
    _ungrouped: list[str] = []
    for line in unique[:200]:
        stripped = line.strip()
        # Match pattern: condition/env/step/metric: value
        parts = stripped.split("/")
        if len(parts) >= 3 and ":" in parts[-1]:
            cond = parts[0]
            detail = "/".join(parts[1:])
            _grouped.setdefault(cond, []).append(f"  - {detail}")
        else:
            _ungrouped.append(stripped)

    formatted_lines: list[str] = []
    if _grouped:
        for cond, details in sorted(_grouped.items()):
            formatted_lines.append(f"## Condition: {cond}")
            formatted_lines.extend(details[:30])
    if _ungrouped:
        formatted_lines.extend(_ungrouped)

    count_description = f"{run_count} run(s)"
    count_instruction = f"The experiment ran {run_count} time(s) — state this accurately in the methodology.\n"
    if run_count_basis == "result_records":
        count_description = f"{run_count} result record(s) for the selected version"
        count_instruction = (
            "Result records and code versions are NOT independent repeat trials. "
            "Only claim a repeat count supported by explicit condition and seed records.\n"
        )

    return (
        f"\n\nACTUAL EXPERIMENT DATA (from {count_description} — use ONLY these numbers):\n"
        "```\n"
        + "\n".join(formatted_lines[:200])
        + "\n```\n"
        "CRITICAL: Every number in the Results table MUST come from the data above. "
        "Do NOT round excessively, do NOT invent numbers, do NOT change values. "
        + count_instruction
        + "NEVER paste raw metric paths (like 'condition/env/step/metric: value') "
        "into the paper. Always convert to formatted LaTeX tables or inline prose.\n"
    ), has_parsed_metrics


def _write_paper_sections(
    *,
    llm: LLMClient,
    pm: PromptManager,
    run_dir: Path | None = None,
    preamble: str,
    topic_constraint: str,
    exp_metrics_instruction: str,
    citation_instruction: str,
    outline: str,
    model_name: str = "",
    discussion_ablation: str = "",
    figure_prompt_instruction: str = "",
) -> str:
    """Write a conference-grade paper in 3 sequential LLM calls.

    Call 1: Title + Abstract + Introduction + Related Work
    Call 2: Method + Experiments (with full experiment data)
    Call 3: Results + Discussion + Limitations + Conclusion

    Each call receives prior sections for coherence.
    """
    # Render writing_structure block for injection
    try:
        _writing_structure = pm.block("writing_structure")
    except (KeyError, Exception):  # noqa: BLE001
        _writing_structure = ""

    _overlay = _get_evolution_overlay(run_dir, "paper_draft")
    system = pm.for_stage(
        "paper_draft",
        evolution_overlay=_overlay,
        preamble=preamble,
        topic_constraint=topic_constraint,
        exp_metrics_instruction=exp_metrics_instruction,
        citation_instruction=citation_instruction,
        writing_structure=_writing_structure,
        outline=outline,
        discussion_ablation=discussion_ablation,
        figure_prompt_instruction=figure_prompt_instruction,
    ).system

    sections: list[str] = []

    # --- R4-3: Title guidelines and abstract structure ---
    try:
        title_guidelines = pm.block("title_guidelines")
    except (KeyError, Exception):  # noqa: BLE001
        title_guidelines = ""
    try:
        abstract_structure = pm.block("abstract_structure")
    except (KeyError, Exception):  # noqa: BLE001
        abstract_structure = ""

    # IMP-20/25/31/24: Academic style, narrative, anti-hedging, anti-repetition
    try:
        academic_style_guide = pm.block("academic_style_guide")
    except (KeyError, Exception):  # noqa: BLE001
        academic_style_guide = ""
    try:
        narrative_writing_rules = pm.block("narrative_writing_rules")
    except (KeyError, Exception):  # noqa: BLE001
        narrative_writing_rules = ""
    try:
        anti_hedging_rules = pm.block("anti_hedging_rules")
    except (KeyError, Exception):  # noqa: BLE001
        anti_hedging_rules = ""
    try:
        anti_repetition_rules = pm.block("anti_repetition_rules")
    except (KeyError, Exception):  # noqa: BLE001
        anti_repetition_rules = ""

    # --- Call 1: Title + Abstract + Introduction + Related Work ---
    call1_user = (
        f"{preamble}\n\n"
        f"{topic_constraint}"
        f"{citation_instruction}\n\n"
        f"{title_guidelines}\n\n"
        f"{academic_style_guide}\n"
        f"{narrative_writing_rules}\n"
        f"{anti_hedging_rules}\n"
        f"{anti_repetition_rules}\n\n"
        "Write the following sections of a NeurIPS/ICML-quality paper in markdown. "
        "Follow the LENGTH REQUIREMENTS strictly:\n\n"
        "1. **Title** (HARD RULE: MUST be 14 words or fewer. Create a catchy method name "
        "first, then build the title: 'MethodName: Subtitle'. If your title exceeds 14 words, "
        "it will be automatically rejected. NEVER use 'Untitled Paper'.)\n"
        f"2. **Abstract** (150-220 words — HARD LIMIT. Do NOT exceed 220 words. "
        f"Do NOT include raw metric paths or 16-digit decimals.){abstract_structure}\n"
        "3. **Introduction** (800-1000 words): real-world motivation, problem statement, "
        "research gap analysis with citations, method overview, 3-4 contributions as bullet points, "
        "paper organization paragraph. MUST cite 8-12 references.\n"
        "   **TEASER FIGURE (MANDATORY)**: Include a teaser figure (Figure 1) right after the "
        "first introductory paragraph using a `<!-- FIGURE_PROMPT ... -->` block. This figure "
        "should be a high-level conceptual illustration showing the core idea/motivation at a "
        "glance — it conveys the key insight without requiring method details. Use figure_type: "
        "concept_illustration, section: Introduction.\n"
        "4. **Related Work** (600-800 words): organized into 3-4 thematic subsections, each discussing "
        "4-5 papers with proper citations. Compare approaches, identify limitations, position this work.\n\n"
        f"{figure_prompt_instruction}\n\n"
        f"Outline:\n{outline}\n\n"
        "Output markdown with ## headers. Do NOT include a References section.\n"
        "IMPORTANT: Start DIRECTLY with '## Title'. Do NOT include any preamble, "
        "data verification, condition listing, or metric enumeration before the title. "
        "The paper should read like a published manuscript, not a data report."
    )
    # R14-1: Higher token limit for reasoning models
    _paper_max_tokens = 12000
    if any(model_name.startswith(p) for p in ("gpt-5", "o3", "o4")):
        _paper_max_tokens = 24000

    # T3.5: Retry once on failure, use placeholder if still fails
    try:
        resp1 = _chat_with_prompt(llm, system, pm.adapt(call1_user), max_tokens=_paper_max_tokens, retries=1)
        part1 = resp1.content.strip()
    except Exception:  # noqa: BLE001
        logger.error("Stage 17: Part 1 LLM call failed after retry — using placeholder")
        part1 = (
            "## Title\n[PLACEHOLDER — LLM call failed]\n\n"
            "## Abstract\n[This section could not be generated due to an LLM error. "
            "Please regenerate this stage.]\n\n"
            "## Introduction\n[PLACEHOLDER]\n\n"
            "## Related Work\n[PLACEHOLDER]"
        )
    sections.append(part1)
    logger.info("Stage 17: Part 1 (Title+Abstract+Intro+Related Work) — %d chars", len(part1))

    # --- Call 2: Method + Experiments ---
    call2_user = (
        f"{preamble}\n\n"
        f"{topic_constraint}"
        f"{exp_metrics_instruction}\n\n"
        f"{narrative_writing_rules}\n"
        f"{anti_hedging_rules}\n\n"
        # IMP-21: Citation instruction for Method + Experiments
        "CITATION REQUIREMENT: The Method section MUST cite at least 3-5 related "
        "technical papers (foundations your method builds on). The Experiments section "
        "MUST cite baseline method papers. Use [cite_key] syntax.\n"
        f"{citation_instruction}\n\n"
        "You are continuing a paper. The sections written so far are:\n\n"
        f"---\n{part1}\n---\n\n"
        "Now write the next sections, maintaining consistency with the above:\n\n"
        "5. **Method** (1000-1500 words): formal problem definition with mathematical notation "
        "($x$, $\\theta$, etc.), detailed algorithm description with equations, step-by-step procedure, "
        "complexity analysis, design rationale for key choices. Include algorithm pseudocode if applicable. "
        "Write as FLOWING PROSE — do NOT use bullet-point lists for method components.\n"
        "   **METHOD FIGURES (MANDATORY)**: Include at least TWO `<!-- FIGURE_PROMPT ... -->` blocks:\n"
        "   (a) A **framework / architecture overview** (figure_type: architecture_diagram or "
        "pipeline_overview) showing the full system pipeline.\n"
        "   (b) A **method detail figure** (figure_type: method_flowchart or comparison_illustration) "
        "illustrating a key algorithmic step or component.\n"
        "6. **Experiments** (800-1200 words): detailed experimental setup, datasets with statistics "
        "(size, splits, features), all baselines and their implementations, hyperparameter settings "
        "in a markdown table, evaluation metrics with mathematical definitions, hardware and runtime info.\n"
        "METHOD NAMES IN TABLES: Use SHORT abbreviations (4-8 chars) for method names "
        "in tables. Define abbreviation mappings in a footnote. "
        "NEVER put method names longer than 20 characters in table cells.\n\n"
        f"{figure_prompt_instruction}\n\n"
        f"Outline:\n{outline}\n\n"
        "Output markdown with ## headers. Continue from where Part 1 ended."
    )
    try:
        resp2 = _chat_with_prompt(llm, system, pm.adapt(call2_user), max_tokens=_paper_max_tokens, retries=1)
        part2 = resp2.content.strip()
    except Exception:  # noqa: BLE001
        logger.error("Stage 17: Part 2 LLM call failed after retry — using placeholder")
        part2 = (
            "## Method\n[PLACEHOLDER — LLM call failed. Please regenerate this stage.]\n\n"
            "## Experiments\n[PLACEHOLDER]"
        )
    sections.append(part2)
    logger.info("Stage 17: Part 2 (Method+Experiments) — %d chars", len(part2))

    # --- Call 3: Results + Discussion + Limitations + Conclusion ---
    call3_user = (
        f"{preamble}\n\n"
        f"{topic_constraint}"
        f"{exp_metrics_instruction}\n\n"
        f"{narrative_writing_rules}\n"
        f"{anti_hedging_rules}\n"
        f"{anti_repetition_rules}\n\n"
        # IMP-21: Citation instruction for Results + Discussion + Conclusion
        "CITATION REQUIREMENT: The Discussion section MUST cite at least 3-5 papers "
        "when comparing findings with prior work. The Conclusion may cite 1-2 "
        "foundational references.\n"
        f"{citation_instruction}\n\n"
        "You are completing a paper. The sections written so far are:\n\n"
        f"---\n{part1}\n\n{part2}\n---\n\n"
        "Now write the final sections, maintaining consistency:\n\n"
        "7. **Results** (600-800 words):\n"
        "   - START with an AGGREGATED results table (Table 1): rows = methods, columns = metrics.\n"
        "     Each cell = mean ± std across seeds. Bold the best value per column.\n"
        "     EVERY table MUST have a descriptive caption that allows understanding without "
        "     reading the main text. NEVER use just 'Table 1' as a caption.\n"
        "   - Follow with a PER-REGIME table (Table 2) breaking down by easy/hard regimes.\n"
        "   - Include a STATISTICAL COMPARISON table (Table 3): paired t-tests between key methods.\n"
        "   - NEVER dump raw per-seed numbers in the main text. Aggregate first, then discuss.\n"
        "   - MUST include at least 2 DATA figures using `![Caption](charts/filename.png)` syntax "
        "for pre-generated charts and plots. One MUST be a performance comparison chart.\n"
        "     Do NOT use `<!-- FIGURE_PROMPT -->` for data figures in Results/Experiments — "
        "only use `![Caption](charts/...)` here.\n"
        "     Figures MUST be referenced in text: 'As shown in Figure N, ...'\n"
        "8. **Discussion** (400-800 words): interpretation of key findings, unexpected results, "
        "comparison with prior work (CITE 3-5 papers here!), practical implications.\n"
        f"{discussion_ablation}"
        "9. **Limitations** (200-300 words): honest assessment of scope, dataset, methodology. "
        "ALL caveats consolidated HERE — nowhere else in the paper.\n"
        "10. **Conclusion** (100-200 words MAXIMUM — this is a HARD LIMIT): "
        "Summarize contributions in 2-3 sentences. State main finding in 1 sentence. "
        "Suggest 2-3 concrete future directions in 1-2 sentences. "
        "Do NOT repeat any specific numbers from Results. Do NOT restate the abstract. "
        "A good conclusion is SHORT and forward-looking.\n\n"
        "CRITICAL FORMATTING RULES FOR ALL SECTIONS:\n"
        "- Write as FLOWING PROSE paragraphs, NOT bullet-point lists\n"
        "- NEVER dump raw metric paths like 'config/method_name/seed_3/primary_metric'\n"
        "- All numbers must be rounded to 4 decimal places maximum\n"
        "- Every table MUST have a descriptive caption (not just 'Table 1')\n"
        "- Use \\begin{algorithm} or pseudocode notation, NOT \\begin{verbatim}\n\n"
        "Output markdown with ## headers. Do NOT include a References section."
    )
    try:
        resp3 = _chat_with_prompt(llm, system, pm.adapt(call3_user), max_tokens=_paper_max_tokens, retries=1)
        part3 = resp3.content.strip()
    except Exception:  # noqa: BLE001
        logger.error("Stage 17: Part 3 LLM call failed after retry — using placeholder")
        part3 = (
            "## Results\n[PLACEHOLDER — LLM call failed. Please regenerate this stage.]\n\n"
            "## Discussion\n[PLACEHOLDER]\n\n"
            "## Limitations\n[PLACEHOLDER]\n\n"
            "## Conclusion\n[PLACEHOLDER]"
        )
    sections.append(part3)
    logger.info("Stage 17: Part 3 (Results+Discussion+Limitations+Conclusion) — %d chars", len(part3))

    # Combine all sections
    draft = "\n\n".join(sections)

    # R32: Strip data verification preamble that LLMs sometimes emit before
    # the actual paper.  The preamble typically starts with "## Tested Conditions"
    # or similar headings and ends before "## Title".
    import re as _re_strip
    _title_match = _re_strip.search(r"^## Title\b", draft, _re_strip.MULTILINE)
    if _title_match and _title_match.start() > 200:
        _stripped = draft[_title_match.start():]
        logger.info(
            "R32: Stripped %d-char preamble before '## Title'",
            _title_match.start(),
        )
        draft = _stripped

    total_words = len(draft.split())
    logger.info("Stage 17: Full draft — %d chars, ~%d words", len(draft), total_words)

    return draft


# ---------------------------------------------------------------------------
# Figure prompt extraction from paper draft
# ---------------------------------------------------------------------------

_FIGURE_PROMPT_RE = re.compile(
    r"<!--\s*FIGURE_PROMPT\s*\n(.*?)-->",
    re.DOTALL,
)

_ACADEMIC_STYLE_SUFFIX = (
    " The image should be in a clean, professional ACADEMIC style suitable "
    "for a top-tier AI/ML research paper (NeurIPS, ICML, ICLR). "
    "Use a white or light background. Use clear labels and annotations. "
    "Avoid excessive decoration. Use a consistent color palette. "
    "Text should be legible at column width (~3.25 inches). "
    "Style: technical illustration, vector-like, clean lines."
)

_FIGURE_TYPE_STYLE: dict[str, str] = {
    "architecture_diagram": (
        "Show model layers, connections, and data flow with labeled boxes and arrows. "
        "Use a consistent left-to-right or top-to-bottom layout. "
        "Group related components with dashed borders."
    ),
    "method_flowchart": (
        "Step-by-step process flow with rounded rectangles for processes, "
        "diamonds for decision points, arrows with labels. "
        "Number sequential steps. Highlight novel steps with accent color."
    ),
    "pipeline_overview": (
        "Full pipeline from input to output with distinct visual blocks per stage. "
        "Include example inputs/outputs. Use consistent arrow style. "
        "Show parallel/branching paths if applicable."
    ),
    "concept_illustration": (
        "Simple, clean diagram illustrating a key concept or intuition. "
        "Include before/after or problem/solution comparison. "
        "Keep it understandable at a glance."
    ),
    "system_diagram": (
        "Overall system architecture with major components and interactions. "
        "Show data stores, APIs, external services with labeled connections."
    ),
    "comparison_illustration": (
        "Side-by-side comparison of approaches with consistent styling. "
        "Highlight key differences with visual cues (color, checkmarks/crosses)."
    ),
    "attention_visualization": (
        "Attention weights or patterns with heatmap-style coloring. "
        "Include input/output sequences, label attention heads, "
        "use a clear color scale legend."
    ),
}


def _extract_figure_prompts(draft: str, topic: str = "") -> list[dict]:
    """Parse ``<!-- FIGURE_PROMPT ... -->`` blocks from a paper draft.

    Returns a list of dicts, each with keys:
        figure_id, figure_type, section, caption, aspect_ratio,
        raw_prompt, full_prompt  (raw_prompt + academic style + type guidelines)
    """
    _top_level_keys = {
        "figure_id", "figure_type", "section", "caption", "aspect_ratio", "prompt",
    }

    results: list[dict] = []
    for m in _FIGURE_PROMPT_RE.finditer(draft):
        block = m.group(1)
        entry: dict[str, str] = {}
        current_key = ""
        current_lines: list[str] = []
        in_multiline = False

        for line in block.splitlines():
            stripped = line.strip()
            if not stripped:
                if in_multiline:
                    current_lines.append("")
                continue

            # Check if this line starts a new top-level key (not indented,
            # key name is a known field)
            is_new_key = False
            if ":" in stripped and not line.startswith((" ", "\t")):
                candidate_key = stripped.partition(":")[0].strip().lower()
                if candidate_key in _top_level_keys:
                    is_new_key = True

            if is_new_key:
                key, _, val = stripped.partition(":")
                key = key.strip().lower()
                val = val.strip().strip('"').strip("'")
                if current_key and current_lines:
                    entry[current_key] = "\n".join(current_lines).strip()
                current_key = key
                in_multiline = (val == "|" or val == "")
                if val and val != "|":
                    current_lines = [val]
                    in_multiline = False
                else:
                    current_lines = []
                    if val == "|":
                        in_multiline = True
            else:
                current_lines.append(stripped)
        if current_key and current_lines:
            entry[current_key] = "\n".join(current_lines).strip()

        fig_id = entry.get("figure_id", f"figure_{len(results) + 1}")
        fig_type = entry.get("figure_type", "concept_illustration")
        section = entry.get("section", "Method")
        caption = entry.get("caption", "")
        aspect_ratio = entry.get("aspect_ratio", "16:9")
        raw_prompt = entry.get("prompt", caption)

        type_style = _FIGURE_TYPE_STYLE.get(fig_type, _FIGURE_TYPE_STYLE["concept_illustration"])
        full_prompt = (
            f"Create a professional academic figure for the '{section}' "
            f"section of a research paper"
        )
        if topic:
            full_prompt += f" about: {topic}"
        full_prompt += (
            f".\n\nFigure description: {raw_prompt}\n\n"
            f"Style guidelines:\n{type_style}\n\n"
            f"{_ACADEMIC_STYLE_SUFFIX}"
        )

        results.append({
            "figure_id": fig_id,
            "figure_type": fig_type,
            "section": section,
            "caption": caption,
            "aspect_ratio": aspect_ratio,
            "raw_prompt": raw_prompt,
            "full_prompt": full_prompt,
        })

    return results


def _render_figure_prompts(
    fig_prompts: list[dict],
    stage_dir: Path,
    config: Any,
    llm: Any,
    topic: str = "",
) -> list[dict]:
    """Render extracted FIGURE_PROMPT entries via NanoBananaAgent.

    Uses the OpenAI-compatible proxy (same base_url/api_key as the LLM) to
    call an image-capable model (default ``gemini-3-pro-image-preview``).
    Gracefully skips if no API credentials are available.

    Returns the *same* ``fig_prompts`` list, augmented with ``output_path``
    and ``success`` fields for each entry.
    """
    if not fig_prompts:
        return fig_prompts

    from researchclaw.agents.figure_agent.nano_banana import NanoBananaAgent

    from researchclaw.llm import resolve_provider_base_url
    image_base_url = getattr(config.llm, "image_base_url", "") or ""
    image_api_key = getattr(config.llm, "image_api_key", "") or ""
    base_url = image_base_url or resolve_provider_base_url(
        getattr(config.llm, "provider", "openai-compatible"),
        getattr(config.llm, "base_url", ""),
    )
    api_key = (
        image_api_key
        or getattr(config.llm, "api_key", "")
        or os.environ.get(getattr(config.llm, "api_key_env", "") or "", "")
        or ""
    )
    if not api_key:
        logger.warning(
            "NanoBanana render skipped — no llm.api_key or llm.api_key_env"
        )
        return _fallback_unrendered_figures(fig_prompts, stage_dir / "figures")

    image_model = getattr(config.llm, "image_model", "") or ""
    if not image_model:
        fa_cfg = getattr(config.experiment, "figure_agent", None)
        image_model = (
            (getattr(fa_cfg, "gemini_model", "") if fa_cfg else "")
            or "gemini-3-pro-image-preview"
        )

    figures_dir = stage_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    image_figures = [
        {
            "figure_id": fp["figure_id"],
            "figure_type": fp.get("figure_type", "concept_illustration"),
            "section": fp.get("section", "Method"),
            "description": fp.get("raw_prompt", fp.get("caption", "")),
        }
        for fp in fig_prompts
    ]

    agent = NanoBananaAgent(
        llm,
        base_url=base_url,
        openai_api_key=api_key,
        image_model=image_model,
        output_dir=figures_dir,
    )

    try:
        result = agent.execute({
            "image_figures": image_figures,
            "topic": topic,
            "output_dir": str(figures_dir),
        })
    except Exception as exc:
        logger.warning("NanoBanana rendering failed: %s", exc)
        return _fallback_unrendered_figures(fig_prompts, figures_dir)

    gen_map = {
        g["figure_id"]: g for g in result.data.get("generated", [])
    }
    for fp in fig_prompts:
        gen = gen_map.get(fp["figure_id"], {})
        fp["output_path"] = gen.get("output_path", "")
        fp["success"] = gen.get("success", False)

    success_count = sum(1 for fp in fig_prompts if fp.get("success"))
    logger.info(
        "NanoBanana rendered %d/%d figure prompts → %s",
        success_count, len(fig_prompts), figures_dir,
    )
    return _fallback_unrendered_figures(fig_prompts, figures_dir)


_QUOTED_LABEL_RE = re.compile(r"[“\"]([^”\"\n]{2,60})[”\"]")


def _figure_nodes(fp: dict) -> list[str]:
    """Box labels for a fallback diagram: quoted labels from the prompt."""
    nodes: list[str] = []
    for label in _QUOTED_LABEL_RE.findall(fp.get("raw_prompt", "")):
        label = label.strip().rstrip(".,;:").strip()
        if 2 <= len(label) <= 48 and label not in nodes:
            nodes.append(label)
    if len(nodes) < 2:
        caption = fp.get("caption", "")
        nodes = [p.strip() for p in re.split(r"[;,:]| while | and ", caption) if p.strip()][:4]
    return nodes[:8] or [fp.get("figure_id", "figure")]


def _draw_fallback_figure(fp: dict, output_path: Path) -> bool:
    """Draw a schematic box diagram (caption title + labelled nodes).

    Used when the image model is unavailable so that a FIGURE_PROMPT still
    becomes a figure instead of leaking its prompt text into the paper.
    """
    try:
        import textwrap

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        return False

    nodes = _figure_nodes(fp)
    per_row = min(4, len(nodes))
    rows = (len(nodes) + per_row - 1) // per_row
    fig, ax = plt.subplots(figsize=(10, 1.6 + 1.5 * rows))
    ax.set_xlim(0, per_row)
    ax.set_ylim(-rows - 0.15, 0.55)
    ax.axis("off")
    title = textwrap.fill(fp.get("caption") or fp.get("figure_id", ""), 110)
    ax.set_title(title, fontsize=10, loc="left")
    centers: list[tuple[float, float]] = []
    for i, label in enumerate(nodes):
        row, col = divmod(i, per_row)
        x, y = col + 0.5, -row - 0.2
        centers.append((x, y))
        ax.add_patch(FancyBboxPatch(
            (x - 0.4, y - 0.3), 0.8, 0.6,
            boxstyle="round,pad=0.02", facecolor="#E8F0FA", edgecolor="#4477AA", linewidth=1.2,
        ))
        ax.text(x, y, textwrap.fill(label, 18), ha="center", va="center", fontsize=8.5)
    for (x0, y0), (x1, y1) in zip(centers, centers[1:]):
        if y0 == y1:
            start, end = (x0 + 0.4, y0), (x1 - 0.4, y1)
        else:
            start, end = (x0, y0 - 0.3), (x1, y1 + 0.3)
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.2})
    ax.text(0, -rows - 0.1, "Schematic overview", fontsize=7, color="#888888")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path.is_file()


def _fallback_unrendered_figures(fig_prompts: list[dict], figures_dir: Path) -> list[dict]:
    """Draw a fallback diagram for every prompt the image model did not render.

    Fallbacks are named ``<figure_id>_fallback.png`` so a later stage still
    retries the real render (S22 only reuses ``<figure_id>.png``).
    """
    for fp in fig_prompts:
        if fp.get("success") and fp.get("output_path") and Path(fp["output_path"]).is_file():
            continue
        out = figures_dir / f"{fp.get('figure_id', 'figure')}_fallback.png"
        try:
            ok = _draw_fallback_figure(fp, out)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fallback figure for %s failed: %s", fp.get("figure_id"), exc)
            ok = False
        fp["output_path"] = str(out) if ok else ""
        fp["success"] = ok
        fp["fallback"] = ok
    return fig_prompts


_FIGURE_CAPTION_LINE_RE = re.compile(r"\s*\n\*Figure\s+\d+[.:]\s*(.+?)\*[ \t]*(?=\n|\Z)")


def _figure_prompts_to_markdown(
    paper: str, run_dir: Path, chart_dir: Path
) -> tuple[str, list[str]]:
    """Replace ``<!-- FIGURE_PROMPT -->`` blocks with markdown images.

    Uses the newest rendered figure (S22, then S20); if none exists a
    fallback diagram is drawn.  The figure is copied into *chart_dir* and
    the hand-written ``*Figure N. ...*`` line after the block becomes its
    caption.  Returns the new paper and the copied file names.
    """
    import shutil

    meta: dict[str, dict] = {}
    # Older stages first so the revision (S22) overrides the draft (S20).
    for path in [
        *reversed(_newest_stage_files(run_dir, "stage-20*/figure_prompts.json")),
        *reversed(_newest_stage_files(run_dir, "stage-22*/figure_prompts.json")),
    ]:
        try:
            for fp in json.loads(path.read_text(encoding="utf-8")):
                if isinstance(fp, dict) and fp.get("figure_id"):
                    meta[fp["figure_id"]] = fp
        except (OSError, ValueError):
            continue

    copied: list[str] = []
    out: list[str] = []
    pos = 0
    for m in _FIGURE_PROMPT_RE.finditer(paper):
        out.append(paper[pos:m.start()])
        pos = m.end()
        parsed = (_extract_figure_prompts(m.group(0)) or [{}])[0]
        fid = parsed.get("figure_id") or f"figure_{len(copied) + 1}"
        fp = {**parsed, **meta.get(fid, {})}
        caption = fp.get("caption", "")
        cap = _FIGURE_CAPTION_LINE_RE.match(paper, pos)
        if cap:
            caption = cap.group(1).strip()
            pos = cap.end()

        source: Path | None = None
        candidates = [Path(fp["output_path"])] if fp.get("output_path") else []
        for name in (f"{fid}.*", f"{fid}_fallback.*"):
            for stage in ("stage-22*", "stage-20*"):
                candidates += _newest_stage_files(run_dir, f"{stage}/figures/{name}")
        for cand in candidates:
            if cand.is_file() and cand.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
                source = cand
                break
        target = chart_dir / f"{fid}{source.suffix.lower() if source else '.png'}"
        try:
            if source is not None:
                chart_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            elif not _draw_fallback_figure(fp, target):
                logger.warning("Stage 25: Dropped FIGURE_PROMPT %s (no figure available)", fid)
                continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 25: Dropped FIGURE_PROMPT %s: %s", fid, exc)
            continue
        copied.append(target.name)
        alt = caption.replace("[", "(").replace("]", ")") or fid
        out.append(f"![{alt}](charts/{target.name})")
    out.append(paper[pos:])
    return "".join(out), copied


def _read_figure_manifest(run_dir: Path) -> dict[str, dict]:
    """Read ``figure_manifest.json`` from stage-16/charts/ (or stage-14).

    Returns ``{filename: metadata_dict}`` for each chart entry.
    """
    for pattern in ("stage-16*", "stage-14*"):
        for d in _current_stage_dirs(run_dir, pattern.rstrip("*")):
            mf = d / "charts" / "figure_manifest.json"
            if mf.is_file():
                try:
                    entries = json.loads(mf.read_text(encoding="utf-8"))
                    return {
                        Path(e["file_path"]).name: e
                        for e in entries
                        if isinstance(e, dict) and e.get("file_path")
                    }
                except Exception:
                    pass
    return {}


# ---------------------------------------------------------------------------
# Draft quality validation (section balance + bullet-point density)
# ---------------------------------------------------------------------------

# Sections where bullets/numbered lists are acceptable.
_BULLET_LENIENT_SECTIONS = frozenset({
    "introduction", "limitations", "limitation",
    "limitations and future work", "abstract",
})

# Main body sections used for balance ratio check.
_BALANCE_SECTIONS = frozenset({
    "introduction", "related work", "method", "experiments", "results",
    "discussion",
})


def _validate_draft_quality(
    draft: str,
    stage_dir: Path | None = None,
    single_run_mode: bool = False,
) -> dict[str, Any]:
    """Validate a paper draft for section balance and prose quality.

    Checks:
    1. Per-section word count vs ``SECTION_WORD_TARGETS``.
    2. Bullet-point / numbered-list density per section.
    3. Largest-to-smallest main-section word-count ratio.

    Returns a dict with ``section_analysis``, ``overall_warnings``, and
    ``revision_directives``.  Optionally writes ``draft_quality.json`` to
    *stage_dir*.
    """
    from researchclaw.prompts import SECTION_WORD_TARGETS, _SECTION_TARGET_ALIASES

    _heading_re = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    matches = list(_heading_re.finditer(draft))

    sections_data: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(draft)
        body = draft[start:end].strip()
        sections_data.append({
            "heading": heading,
            "heading_lower": heading.strip().lower(),
            "level": level,
            "body": body,
        })

    section_analysis: list[dict[str, Any]] = []
    overall_warnings: list[str] = []
    revision_directives: list[str] = []
    main_section_words: dict[str, int] = {}

    _bullet_re = re.compile(r"^\s*[-*]\s+", re.MULTILINE)
    _numbered_re = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)

    # BUG-24: Accumulate subsection (H3+) word counts into parent H2 sections
    _subsection_words: dict[str, int] = {}
    _current_parent = ""
    for sec in sections_data:
        if sec["level"] <= 2:
            _current_parent = sec["heading_lower"]
            _subsection_words.setdefault(_current_parent, 0)
        else:
            # Add subsection words to parent
            _subsection_words[_current_parent] = (
                _subsection_words.get(_current_parent, 0) + len(sec["body"].split())
            )

    for sec in sections_data:
        if sec["level"] > 2:
            continue
        heading_lower: str = sec["heading_lower"]
        body: str = sec["body"]
        # BUG-24: Include subsection words in the parent's word count
        word_count = len(body.split()) + _subsection_words.get(heading_lower, 0)
        canon = heading_lower
        if canon not in SECTION_WORD_TARGETS:
            canon = _SECTION_TARGET_ALIASES.get(heading_lower, "")
        entry: dict[str, Any] = {
            "heading": sec["heading"],
            "word_count": word_count,
            "canonical": canon,
        }
        if canon and canon in SECTION_WORD_TARGETS:
            lo, hi = SECTION_WORD_TARGETS[canon]
            entry["target"] = [lo, hi]
            if word_count < int(lo * 0.7):
                overall_warnings.append(
                    f"{sec['heading']} is severely under target "
                    f"({word_count} words, target {lo}-{hi})"
                )
                revision_directives.append(
                    f"EXPAND {sec['heading']} from {word_count} to {lo}+ words. "
                    f"Add substantive content \u2014 do NOT pad with filler."
                )
                entry["status"] = "severely_short"
            elif word_count < lo:
                overall_warnings.append(
                    f"{sec['heading']} is under target "
                    f"({word_count} words, target {lo}-{hi})"
                )
                revision_directives.append(
                    f"Expand {sec['heading']} from {word_count} to {lo}+ words."
                )
                entry["status"] = "short"
            elif word_count > int(hi * 1.3):
                overall_warnings.append(
                    f"{sec['heading']} exceeds target "
                    f"({word_count} words, target {lo}-{hi})"
                )
                revision_directives.append(
                    f"Compress {sec['heading']} from {word_count} to {hi} words or fewer."
                )
                entry["status"] = "long"
            else:
                entry["status"] = "ok"
        if body:
            total_lines = len([ln for ln in body.splitlines() if ln.strip()])
            bullet_lines = len(_bullet_re.findall(body)) + len(_numbered_re.findall(body))
            density = bullet_lines / total_lines if total_lines > 0 else 0.0
            entry["bullet_density"] = round(density, 2)
            threshold = 0.50 if heading_lower in _BULLET_LENIENT_SECTIONS else 0.25
            if density > threshold and total_lines >= 4:
                overall_warnings.append(
                    f"{sec['heading']} has {bullet_lines}/{total_lines} "
                    f"bullet/numbered lines ({density:.0%} density, "
                    f"threshold {threshold:.0%})"
                )
                revision_directives.append(
                    f"REWRITE {sec['heading']} as flowing academic prose. "
                    f"Convert bullet points to narrative paragraphs."
                )
                entry["bullet_status"] = "high"
            else:
                entry["bullet_status"] = "ok"
        canon_balance = canon or heading_lower
        if canon_balance in _BALANCE_SECTIONS:
            main_section_words[canon_balance] = word_count
        section_analysis.append(entry)

    if len(main_section_words) >= 2:
        wc_values = list(main_section_words.values())
        max_wc = max(wc_values)
        min_wc = min(wc_values)
        if min_wc > 0 and max_wc / min_wc > 3.0:
            largest = max(main_section_words, key=main_section_words.get)  # type: ignore[arg-type]
            smallest = min(main_section_words, key=main_section_words.get)  # type: ignore[arg-type]
            overall_warnings.append(
                f"Section imbalance: {largest} ({max_wc} words) vs "
                f"{smallest} ({min_wc} words) \u2014 ratio {max_wc / min_wc:.1f}x"
            )
            revision_directives.append(
                f"Rebalance sections: expand {smallest} and/or compress {largest} "
                f"to achieve more even section lengths."
            )

    # --- C-4/C-5: Citation count and recency checks ---
    _cite_pattern = re.compile(r"\[([a-zA-Z][a-zA-Z0-9_-]*\d{4}[a-zA-Z0-9]*)\]")
    cited_keys = set(_cite_pattern.findall(draft))
    if cited_keys:
        n_citations = len(cited_keys)
        if n_citations < 15:
            overall_warnings.append(
                f"Only {n_citations} unique citations found (target: >=15 for a full paper)"
            )
            revision_directives.append(
                f"Add more references — a top-venue paper typically cites 25-40 works. "
                f"Currently only {n_citations} unique citations."
            )
        # Check recency: count citations with year >= current_year - 2
        _year_pat = re.compile(r"(\d{4})")
        import datetime as _dt_cit
        _cur_year = _dt_cit.datetime.now().year
        recent_count = sum(
            1 for k in cited_keys
            for m in [_year_pat.search(k)]
            if m and int(m.group(1)) >= _cur_year - 2
        )
        recency_ratio = recent_count / n_citations if n_citations > 0 else 0.0
        if recency_ratio < 0.3 and n_citations >= 10:
            overall_warnings.append(
                f"Citation recency low: only {recent_count}/{n_citations} "
                f"({recency_ratio:.0%}) from last 3 years (target: >=30%%)"
            )

    # --- Abstract and Conclusion length enforcement ---
    for sec in sections_data:
        hl = sec["heading_lower"]
        body_text: str = sec["body"]
        wc = len(body_text.split())
        if hl == "abstract" and wc > 250:
            overall_warnings.append(
                f"Abstract is too long: {wc} words (target: 150-220 words)"
            )
            revision_directives.append(
                f"COMPRESS the Abstract from {wc} to 150-220 words. "
                f"Remove raw metric values, redundant context, and self-references."
            )
        if hl in ("conclusion", "conclusions", "conclusion and future work"):
            if wc > 300:
                overall_warnings.append(
                    f"Conclusion is too long: {wc} words (target: 100-200 words)"
                )
                revision_directives.append(
                    f"COMPRESS the Conclusion from {wc} to 100-200 words. "
                    f"Do NOT repeat specific metric values from Results. "
                    f"Summarize findings in 2-3 sentences, then 2-3 future directions."
                )

    # --- Raw metric path detection (log dumps in prose) ---
    _raw_path_re = re.compile(
        r"\\texttt\{[a-zA-Z0-9_/.-]+(?:/[a-zA-Z0-9_/.-]+){2,}",
    )
    raw_path_count = len(_raw_path_re.findall(draft))
    if raw_path_count > 3:
        overall_warnings.append(
            f"Raw metric paths in prose: {raw_path_count} instances of "
            f"\\texttt{{config/path/metric}} style dumps"
        )
        revision_directives.append(
            "REMOVE raw experiment log paths from prose. Replace "
            "\\texttt{config/metric/path} with human-readable metric names "
            "and summarize values in tables, not inline text."
        )

    # --- Writing quality lint ---
    _weasel_words = re.compile(
        r"\b(various|many|several|quite|fairly|really|very|rather|"
        r"somewhat|relatively|arguably|interestingly|importantly|"
        r"it is well known that|it is obvious that|clearly)\b",
        re.IGNORECASE,
    )
    _duplicate_words = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
    weasel_count = len(_weasel_words.findall(draft))
    dup_matches = _duplicate_words.findall(draft)
    dup_count = len([d for d in dup_matches if d.lower() not in ("that", "had")])
    if weasel_count > 20:
        overall_warnings.append(
            f"High weasel-word count: {weasel_count} instances "
            f"(consider replacing vague words with precise language)"
        )
        revision_directives.append(
            "Replace vague hedging words (various, several, quite, fairly, "
            "rather, somewhat) with precise quantities or remove them."
        )
    if dup_count > 0:
        overall_warnings.append(
            f"Duplicate adjacent words found: {dup_count} instance(s) "
            f"(e.g., 'the the', 'is is')"
        )
        revision_directives.append(
            "Fix duplicate adjacent words (likely typos)."
        )

    # --- Generic title frame check ---
    _title = _extract_paper_title(draft)
    _frame = _generic_title_frame(_title)
    if _frame:
        overall_warnings.append(
            f"Generic title frame '{_frame}' in title: {_title!r}"
        )
        revision_directives.append(
            "RETITLE the paper: name the method or mechanism, e.g. "
            "'XXX: A Cross-Branch Gating Mechanism for Forgery Localization'. "
            "Do not use 'An Empirical Study of...', 'Towards...', "
            "'Investigating...', 'On the...' or 'When X Falls Short'. "
            "Negative or partial results are stated honestly in the abstract, "
            "results and conclusion, not in the title."
        )

    # --- AI-slop / boilerplate detection ---
    _BOILERPLATE_PHRASES = [
        "delves into", "delve into", "it is worth noting",
        "it should be noted", "it is important to note",
        "leverage the power of", "leverages the power of",
        "in this paper, we propose", "in this work, we propose",
        "to the best of our knowledge",
        "in the realm of", "in the landscape of",
        "plays a crucial role", "plays a pivotal role",
        "groundbreaking", "cutting-edge", "state-of-the-art",
        "game-changing", "paradigm shift",
        "a myriad of", "a plethora of",
        "aims to bridge the gap", "bridge the gap",
        "shed light on", "sheds light on",
        "pave the way", "paves the way",
        "the advent of", "with the advent of",
        "in recent years", "in recent times",
        "has gained significant attention",
        "has attracted considerable interest",
        "has emerged as a promising",
        "a comprehensive overview",
        "a holistic approach", "holistic understanding",
        "showcasing the efficacy", "demonstrate the efficacy",
        "multifaceted", "underscores the importance",
        "navigate the complexities",
        "harness the potential", "harnessing the power",
        "it is imperative to", "it is crucial to",
        "a nuanced understanding", "nuanced approach",
        "robust and scalable", "seamlessly integrates",
        "the intricacies of", "intricate interplay",
        "facilitate a deeper understanding",
        "a testament to",
    ]
    draft_lower = draft.lower()
    boilerplate_hits: list[str] = []
    for phrase in _BOILERPLATE_PHRASES:
        count = draft_lower.count(phrase)
        if count > 0:
            boilerplate_hits.extend([phrase] * count)
    if len(boilerplate_hits) > 5:
        unique_phrases = sorted(set(boilerplate_hits))[:5]
        overall_warnings.append(
            f"AI boilerplate detected: {len(boilerplate_hits)} instances "
            f"of generic LLM phrases (e.g., {', '.join(repr(p) for p in unique_phrases[:3])})"
        )
        revision_directives.append(
            "REWRITE sentences containing AI-generated boilerplate phrases. "
            "Replace generic language (e.g., 'delves into', 'it is worth noting', "
            "'leverages the power of', 'plays a crucial role', 'paves the way') "
            "with precise, specific academic language."
        )

    # --- Related work depth check ---
    _rw_headings = {"related work", "related works", "background", "literature review"}
    rw_body = ""
    for sec in sections_data:
        if sec["heading_lower"] in _rw_headings and sec["level"] <= 2:
            rw_body = sec["body"]
            break
    if rw_body and len(rw_body.split()) > 50:
        _comparative_pats = re.compile(
            r"\b(unlike|in contrast|whereas|while .+ focus|"
            r"however|differ(?:s|ent)|our (?:method|approach) .+ instead|"
            r"we (?:instead|differ)|compared to|as opposed to|"
            r"goes beyond|extends|improves upon|addresses the limitation)\b",
            re.IGNORECASE,
        )
        sentences = [s.strip() for s in re.split(r"[.!?]+", rw_body) if s.strip()]
        comparative_sents = sum(1 for s in sentences if _comparative_pats.search(s))
        ratio = comparative_sents / len(sentences) if sentences else 0.0
        if ratio < 0.15 and len(sentences) >= 5:
            overall_warnings.append(
                f"Related Work is purely descriptive: only {comparative_sents}/{len(sentences)} "
                f"sentences ({ratio:.0%}) contain comparative language (target: >=15%)"
            )
            revision_directives.append(
                "REWRITE Related Work to critically compare with prior methods. "
                "Use phrases like 'unlike X, our approach...', 'in contrast to...', "
                "'while X focuses on... we address...' for at least 20% of sentences."
            )

    # --- Statistical rigor check (result sections) ---
    _results_headings = {"results", "experiments", "experimental results", "evaluation"}
    results_body = ""
    for sec in sections_data:
        if sec["heading_lower"] in _results_headings and sec["level"] <= 2:
            results_body += sec["body"] + "\n"
    # One run per condition (user decision) has no variance to report.
    if results_body and len(results_body.split()) > 100 and not single_run_mode:
        has_std = bool(re.search(r"±|\\pm|\bstd\b|\\std\b|standard deviation", results_body, re.IGNORECASE))
        has_ci = bool(re.search(r"confidence interval|\bCI\b|95%|p-value|p\s*<", results_body, re.IGNORECASE))
        has_seeds = bool(re.search(r"(?:seed|run|trial)s?\s*[:=]\s*\d|averaged?\s+over\s+\d+\s+(?:seed|run|trial)", results_body, re.IGNORECASE))
        if not has_std and not has_ci and not has_seeds:
            overall_warnings.append(
                "No statistical measures found in results (no std, CI, p-values, or multi-seed reporting)"
            )
            revision_directives.append(
                "ADD error bars (±std), confidence intervals, or note the number of "
                "random seeds used. Single-run results without variance reporting "
                "are insufficient for top venues."
            )

    result: dict[str, Any] = {
        "section_analysis": section_analysis,
        "overall_warnings": overall_warnings,
        "revision_directives": revision_directives,
    }
    if stage_dir is not None:
        (stage_dir / "draft_quality.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if overall_warnings:
            logger.warning(
                "Draft quality: %d warning(s) \u2014 %s",
                len(overall_warnings),
                "; ".join(overall_warnings[:3]),
            )
        else:
            logger.info("Draft quality: all checks passed")
    return result


def _review_compiled_pdf(
    pdf_path: Path,
    llm: LLMClient,
    topic: str,
) -> dict[str, Any]:
    """Multi-dimensional LLM review of compiled paper (AI-Scientist style).

    Scores the paper on 7 academic review dimensions (1-10 each),
    identifies specific strengths/weaknesses, and provides an overall
    accept/reject recommendation with confidence.

    Returns a dict with dimensional scores, issues, and decision.
    """
    if not pdf_path.exists():
        return {}

    # Use source-based review since not all models support vision
    tex_path = pdf_path.with_suffix(".tex")
    if not tex_path.exists():
        return {}

    tex_content = tex_path.read_text(encoding="utf-8")[:12000]

    review_prompt = (
        "You are a senior Area Chair at a top AI conference (NeurIPS/ICML/ICLR) "
        "reviewing a paper submission. Provide a rigorous, structured review.\n\n"
        f"PAPER TOPIC: {topic}\n\n"
        f"LaTeX source:\n```latex\n{tex_content}\n```\n\n"
        "REVIEW INSTRUCTIONS:\n"
        "Score each dimension 1-10 (1=unacceptable, 5=borderline, 8=strong accept, "
        "10=best paper candidate). Be critical but fair.\n\n"
        "DIMENSIONS:\n"
        "1. SOUNDNESS: Are claims well-supported? Is methodology correct? "
        "Are there logical gaps or unsupported claims?\n"
        "2. PRESENTATION: Is the writing clear, flowing, and professional? "
        "Are there grammar errors, bullet lists in prose sections, or "
        "boilerplate phrases? Is it free of AI-generated slop?\n"
        "3. CONTRIBUTION: Is the contribution significant? Does it advance "
        "the field beyond incremental improvement?\n"
        "4. ORIGINALITY: Is the approach novel? Does it differentiate clearly "
        "from prior work?\n"
        "5. CLARITY: Are the method and results easy to understand? Are figures "
        "and tables well-designed with descriptive captions?\n"
        "6. SIGNIFICANCE: Would the community benefit from this work? Does it "
        "open new research directions?\n"
        "7. REPRODUCIBILITY: Are experimental details sufficient to reproduce "
        "results? Are hyperparameters, datasets, and metrics clearly stated?\n\n"
        "Also evaluate:\n"
        "- Are all figures referenced in the text?\n"
        "- Are tables properly formatted (booktabs style, no vertical rules)?\n"
        "- Does the related work critically compare, not just list papers?\n"
        "- Are statistical measures (std, CI, multiple seeds) reported?\n"
        "- Is there a clear limitations section?\n\n"
        "Return a JSON object:\n"
        "{\n"
        '  "soundness": N,\n'
        '  "presentation": N,\n'
        '  "contribution": N,\n'
        '  "originality": N,\n'
        '  "clarity": N,\n'
        '  "significance": N,\n'
        '  "reproducibility": N,\n'
        '  "overall_score": N,\n'
        '  "confidence": N,\n'
        '  "decision": "accept" or "reject",\n'
        '  "strengths": ["strength1", "strength2", ...],\n'
        '  "weaknesses": ["weakness1", "weakness2", ...],\n'
        '  "critical_issues": ["issue requiring revision", ...],\n'
        '  "minor_issues": ["formatting/typo issues", ...],\n'
        '  "summary": "2-3 sentence overall assessment"\n'
        "}\n"
    )

    try:
        resp = llm.chat(
            messages=[{"role": "user", "content": review_prompt}],
            system=(
                "You are a meticulous, critical academic reviewer. "
                "You have reviewed 100+ papers at top venues. "
                "Score honestly — most papers deserve 4-6, not 7-9. "
                "Flag any sign of AI-generated boilerplate."
            ),
        )
        review_data = _safe_json_loads(resp.content, {})
        if isinstance(review_data, dict) and "overall_score" in review_data:
            # Compute weighted aggregate if individual scores present
            dim_scores = {
                k: review_data.get(k, 0)
                for k in (
                    "soundness", "presentation", "contribution",
                    "originality", "clarity", "significance",
                    "reproducibility",
                )
            }
            valid = {k: v for k, v in dim_scores.items() if isinstance(v, (int, float)) and v > 0}
            if valid:
                review_data["mean_score"] = round(sum(valid.values()) / len(valid), 2)
            return review_data
    except Exception as exc:  # noqa: BLE001
        logger.debug("PDF review LLM call failed: %s", exc)

    return {}


def _check_ablation_effectiveness(
    exp_summary: dict[str, Any],
    threshold: float = 0.05,
) -> list[str]:
    """P7: Check if ablation results are within *threshold* of baseline.

    Returns a list of warning strings for ineffective ablations.
    """
    warnings: list[str] = []
    cond_summaries = exp_summary.get("condition_summaries", {})
    if not isinstance(cond_summaries, dict) or not cond_summaries:
        return warnings

    # Find baseline/control condition
    baseline_name = None
    baseline_mean = None
    for name, data in cond_summaries.items():
        if not isinstance(data, dict):
            continue
        name_lower = name.lower()
        if any(tag in name_lower for tag in ("baseline", "control", "vanilla", "standard")):
            metrics = data.get("metrics", {})
            # Use the first metric that has a _mean suffix or the first available
            for mk, mv in metrics.items():
                if mk.endswith("_mean"):
                    baseline_name = name
                    baseline_mean = float(mv)
                    break
            if baseline_mean is None:
                for mk, mv in metrics.items():
                    try:
                        baseline_name = name
                        baseline_mean = float(mv)
                        break
                    except (TypeError, ValueError):
                        continue
            if baseline_name:
                break

    if baseline_name is None or baseline_mean is None:
        return warnings

    # Check each ablation condition
    for name, data in cond_summaries.items():
        if not isinstance(data, dict):
            continue
        name_lower = name.lower()
        if name == baseline_name:
            continue
        if not any(tag in name_lower for tag in ("ablation", "no_", "without", "reduced")):
            continue
        metrics = data.get("metrics", {})
        for mk, mv in metrics.items():
            if not mk.endswith("_mean"):
                continue
            try:
                abl_val = float(mv)
            except (TypeError, ValueError):
                continue
            if baseline_mean != 0:
                rel_diff = abs(abl_val - baseline_mean) / abs(baseline_mean)
            else:
                rel_diff = abs(abl_val - baseline_mean)
            if rel_diff < threshold:
                warnings.append(
                    f"Ablation '{name}' {mk}={abl_val:.4f} is within "
                    f"{rel_diff:.1%} of baseline '{baseline_name}' "
                    f"{mk}={baseline_mean:.4f} — ablation may be ineffective"
                )
            break  # Only check the first _mean metric per condition

    return warnings


def _detect_result_contradictions(
    exp_summary: dict[str, Any],
) -> list[str]:
    """P10: Detect contradictions in experiment results before paper writing.

    Returns a list of advisory strings to inject into paper writing prompt.
    """
    advisories: list[str] = []
    cond_summaries = exp_summary.get("condition_summaries", {})
    if not isinstance(cond_summaries, dict) or not cond_summaries:
        return advisories

    # Collect primary metric means per condition
    means: dict[str, float] = {}
    for name, data in cond_summaries.items():
        if not isinstance(data, dict):
            continue
        metrics = data.get("metrics", {})
        for mk, mv in metrics.items():
            if mk.endswith("_mean"):
                try:
                    means[name] = float(mv)
                except (TypeError, ValueError):
                    pass
                break

    if len(means) < 2:
        return advisories

    # Check 1: All methods within noise margin (2% relative spread)
    vals = list(means.values())
    val_range = max(vals) - min(vals)
    val_mean = sum(vals) / len(vals)
    if val_mean != 0 and (val_range / abs(val_mean)) < 0.02:
        advisories.append(
            "NULL RESULT: All methods produce nearly identical primary metric values "
            f"(range={val_range:.4f}, mean={val_mean:.4f}). Frame this as a null result — "
            "the methods are statistically indistinguishable. Do NOT claim any method "
            "is superior. Discuss possible explanations (task too easy/hard, metric "
            "insensitive, insufficient differentiation in methods)."
        )

    # Check 2: Control/simple baseline outperforms proposed method
    baseline_val = None
    baseline_name = None
    proposed_val = None
    proposed_name = None
    for name, val in means.items():
        name_lower = name.lower()
        if any(tag in name_lower for tag in ("baseline", "control", "random", "vanilla")):
            if baseline_val is None or val > (baseline_val or 0):
                baseline_val = val
                baseline_name = name
        elif any(tag in name_lower for tag in ("proposed", "our", "novel", "method")):
            if proposed_val is None or val > (proposed_val or 0):
                proposed_val = val
                proposed_name = name

    if baseline_val is not None and proposed_val is not None:
        if baseline_val > proposed_val:
            advisories.append(
                f"NEGATIVE RESULT: Baseline '{baseline_name}' ({baseline_val:.4f}) "
                f"outperforms proposed method '{proposed_name}' ({proposed_val:.4f}). "
                "This is a NEGATIVE result. Do NOT claim the proposed method is superior. "
                "Keep a method/mechanism title (e.g. 'XXX: A ... Mechanism for ...'); "
                "never 'An Empirical Study of...', 'Towards...', 'Investigating...', "
                "'On the...' or 'When X Falls Short'. State the negative result "
                "honestly in the abstract, results and conclusion, and discuss why "
                "the baseline won and what this implies for future work."
            )

    return advisories


def _execute_paper_draft(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    outline = _read_prior_artifact(run_dir, "outline.md") or ""

    # ── SHORT paper fast-path: skip heavy validation, single LLM call ──
    _paper_len = getattr(config.experiment, "paper_length", "") or "full"
    if _paper_len == "short" and llm is not None:
        preamble = _build_context_preamble(
            config, run_dir, include_analysis=True, include_experiment_data=True,
        )
        analysis = _read_prior_artifact(run_dir, "analysis.md") or ""
        experiment_evidence = _paper_experiment_evidence(run_dir)
        _short_user = (
            f"{preamble}\n\n{experiment_evidence}\n\nOutline:\n{outline}\n\n"
            "Write a SHORT WORKSHOP PAPER (2-3 pages, ~1500-2000 words total) "
            "in markdown. Include these sections:\n"
            "1. **Title** (catchy, <=14 words)\n"
            "2. **Abstract** (100-150 words)\n"
            "3. **Introduction** (300-400 words): motivation, gap, approach, contributions\n"
            "4. **Method** (300-400 words): concise method description\n"
            "5. **Experiments** (300-400 words): setup, results with real numbers from data above\n"
            "6. **Conclusion** (100-150 words)\n\n"
            "Use ## headers. Be concise and direct. "
            "Do NOT include a References section. "
            "Start DIRECTLY with '## Title'."
        )
        _short_sys = (
            "You are an academic writer producing a concise short/workshop paper. "
            "Use formal academic tone. Report only real experimental numbers."
        )
        resp = _chat_with_prompt(llm, _short_sys, _short_user, max_tokens=4000)
        draft = resp.content
        logger.info("Stage 20: Short paper mode — single LLM call (%d words)", len(draft.split()))
        (stage_dir / "paper_draft.md").write_text(draft, encoding="utf-8")
        artifacts = ["paper_draft.md"]
        if experiment_evidence:
            (stage_dir / "experiment_evidence.md").write_text(
                experiment_evidence, encoding="utf-8"
            )
            artifacts.append("experiment_evidence.md")
        return StageResult(
            stage=Stage.PAPER_DRAFT,
            status=StageStatus.DONE,
            artifacts=tuple(artifacts),
            evidence_refs=("stage-20/paper_draft.md",),
        )
    # ── END short fast-path ──

    _disc_dir = _find_discussion_dir(run_dir, config)
    _has_discussion = _disc_dir is not None
    preamble = _build_context_preamble(
        config,
        run_dir,
        include_goal=True,
        include_hypotheses=True,
        include_analysis=True,
        include_experiment_data=True,  # WS-5.1: inject real experiment data
        include_discussion=_has_discussion,
    )

    # Build discussion ablation instruction for paper draft
    discussion_ablation = ""
    if _has_discussion:
        discussion_ablation = (
            "\n\n## MULTI-AGENT DISCUSSION ABLATION (MANDATORY SECTION)\n"
            "This paper's research idea was refined through a multi-agent collaborative "
            "discussion process. You MUST include an **ablation study** comparing:\n"
            "- **Without discussion**: Individual agent syntheses (each agent independently "
            "surveyed literature and generated a synthesis)\n"
            "- **With discussion**: Post-discussion consensus synthesis (agents debated, "
            "critiqued each other's views, and reached consensus)\n\n"
            "Analyze the following dimensions in the ablation:\n"
            "1. **Coverage**: How many unique research gaps/opportunities were identified "
            "before vs after discussion\n"
            "2. **Contradiction resolution**: What conflicting viewpoints existed and how "
            "the discussion resolved them\n"
            "3. **Novel insights**: Ideas that emerged ONLY from the discussion interaction "
            "(not present in any individual synthesis)\n"
            "4. **Hypothesis quality**: Compare the specificity and testability of hypotheses "
            "derived from individual vs consensus syntheses\n\n"
            "Place this ablation in the Discussion section or as a dedicated subsection "
            "titled 'Ablation: Multi-Agent Discussion Impact'. Use the pre-discussion "
            "and post-discussion data provided in the research context.\n"
        )

    exp_summary_text = _latest_experiment_summary_text(run_dir)
    # Fallback to standard artifact read
    if exp_summary_text is None:
        exp_summary_text = _read_prior_artifact(run_dir, "experiment_summary.json")
    exp_metrics_instruction = ""
    has_real_metrics = False
    if exp_summary_text:
        exp_summary = _safe_json_loads(exp_summary_text, {})
        if isinstance(exp_summary, dict) and exp_summary.get("metrics_summary"):
            has_real_metrics = True
            exp_metrics_instruction = (
                "\n\nIMPORTANT: Use the ACTUAL experiment results provided in the context. "
                "All numbers in the Results and Experiments sections MUST reference real data. "
                "Do NOT write 'no quantitative results yet' or use placeholder numbers. "
                "Cite specific metrics with their actual values.\n"
            )

    # Collect raw experiment stdout metrics as hard constraint for the paper
    raw_metrics_block, _has_parsed_metrics = _collect_raw_experiment_metrics(run_dir)
    if raw_metrics_block:
        # BUG-23: Raw stdout alone is not sufficient — require either
        # metrics_summary data, parsed metrics from run JSONs,
        # OR at least 3 condition= patterns in raw block
        _has_condition_pattern = len(re.findall(
            r"condition[=:]", raw_metrics_block, re.IGNORECASE
        )) >= 3
        if has_real_metrics or _has_parsed_metrics or _has_condition_pattern:
            has_real_metrics = True
        exp_metrics_instruction += raw_metrics_block

    # R18-1 + R19-6: Inject paired statistical comparisons AND condition summaries
    if exp_summary_text:
        exp_summary_parsed = _safe_json_loads(exp_summary_text, {})
        if isinstance(exp_summary_parsed, dict):
            # R19-6: Inject experiment scale header so LLM knows the data richness
            _total_conds = exp_summary_parsed.get("total_conditions")
            _total_mkeys = exp_summary_parsed.get("total_metric_keys")
            if _total_conds or _total_mkeys:
                scale_block = "\n\n## EXPERIMENT SCALE\n"
                if _total_conds:
                    scale_block += f"- Total conditions tested: {_total_conds}\n"
                if _total_mkeys:
                    scale_block += f"- Total metric keys collected: {_total_mkeys}\n"
                exp_metrics_instruction += scale_block

            # R19-6 + R33: Inject condition summaries with CIs
            cond_summaries = exp_summary_parsed.get("condition_summaries", {})
            if isinstance(cond_summaries, dict) and cond_summaries:
                cond_block = "\n\n## PER-CONDITION SUMMARY (use in Results tables)\n"
                for cname, cdata in sorted(cond_summaries.items()):
                    cond_block += f"\n### {cname}\n"
                    if not isinstance(cdata, dict):
                        continue
                    sr = cdata.get("success_rate")
                    if sr is not None:
                        cond_block += f"- Success rate: {sr:.1%}\n"
                    ns = cdata.get("n_seeds") or cdata.get("n_seed_metrics")
                    if ns:
                        cond_block += f"- Seeds: {ns}\n"
                    ci_lo = cdata.get("ci95_low")
                    ci_hi = cdata.get("ci95_high")
                    if ci_lo is not None and ci_hi is not None:
                        try:
                            cond_block += f"- Bootstrap 95% CI: [{float(ci_lo):.4f}, {float(ci_hi):.4f}]\n"
                        except (ValueError, TypeError):
                            cond_block += f"- Bootstrap 95% CI: [{ci_lo}, {ci_hi}]\n"
                    cm = cdata.get("metrics", {})
                    if cm:
                        for mk, mv in sorted(cm.items()):
                            if isinstance(mv, (int, float)):
                                cond_block += f"- {mk}: {mv:.4f}\n"
                            else:
                                cond_block += f"- {mk}: {mv}\n"
                exp_metrics_instruction += cond_block

            # R18-1: Inject paired statistical comparisons
            paired = exp_summary_parsed.get("paired_comparisons", [])
            if paired:
                paired_block = "\n\n## PAIRED STATISTICAL COMPARISONS (use these in Results)\n"
                paired_block += f"Total: {len(paired)} paired tests computed.\n"
                for pc in paired:
                    if not isinstance(pc, dict):
                        continue
                    method = pc.get("method", "?")
                    baseline = pc.get("baseline", "?")
                    regime = pc.get("regime", "all")
                    md = pc.get("mean_diff", "?")
                    sd = pc.get("std_diff", "?")
                    ts = pc.get("t_stat", "?")
                    pv = pc.get("p_value", "?")
                    ci_lo = pc.get("ci95_low")
                    ci_hi = pc.get("ci95_high")
                    ci_str = ""
                    if ci_lo is not None and ci_hi is not None:
                        try:
                            ci_str = f", 95% CI [{float(ci_lo):.3f}, {float(ci_hi):.3f}]"
                        except (ValueError, TypeError):
                            ci_str = f", 95% CI [{ci_lo}, {ci_hi}]"
                    paired_block += (
                        f"- {method} vs {baseline} (regime={regime}): "
                        f"mean_diff={md}, std_diff={sd}, "
                        f"t={ts}, p={pv}{ci_str}\n"
                    )
                exp_metrics_instruction += paired_block

            # R24: Method naming map — translate generic condition labels
            _cond_names = list(cond_summaries.keys()) if isinstance(cond_summaries, dict) and cond_summaries else []
            if _cond_names:
                naming_block = (
                    "\n\n## METHOD NAMING (CRITICAL — do NOT use generic labels in the paper)\n"
                    "The condition labels below come from the experiment code. In the paper, "
                    "you MUST use DESCRIPTIVE algorithm names, not generic labels.\n"
                    "- If a condition name is already descriptive (e.g., 'random_search', "
                    "'bayesian_optimization', 'ppo_policy'), use it directly as a proper name.\n"
                    "- If a condition name is generic (e.g., 'baseline_1', 'method_variant_1'), "
                    "you MUST infer the algorithm from the experiment code/context and use the "
                    "real algorithm name (e.g., 'Random Search', 'Bayesian Optimization', "
                    "'PPO', 'Curiosity-Driven RL').\n"
                    "- NEVER write `baseline_1` or `method_variant_1` in the paper text.\n"
                    f"- Conditions to name: {_cond_names}\n"
                )
                exp_metrics_instruction += naming_block

            # IMP-8: Inject broken ablation warnings
            abl_warnings = exp_summary_parsed.get("ablation_warnings", [])
            if abl_warnings:
                broken_block = (
                    "\n\n## BROKEN ABLATIONS (DO NOT discuss as valid results)\n"
                    "The following ablation conditions produced IDENTICAL outputs, "
                    "indicating implementation bugs. Do NOT present their differences "
                    "as findings. Mention them ONLY in a 'Limitations' sub-section "
                    "as known implementation issues:\n"
                )
                for _aw in abl_warnings:
                    broken_block += f"- {_aw}\n"
                broken_block += (
                    "\nIf you reference these conditions, state explicitly: "
                    "'Due to an implementation defect, conditions X and Y produced "
                    "identical outputs; their comparison is therefore uninformative.'\n"
                )
                exp_metrics_instruction += broken_block

            # R25: Statistical table format requirement
            if paired:
                stat_table_block = (
                    "\n\n## STATISTICAL TABLE REQUIREMENT (MANDATORY in Results section)\n"
                    "The Results section MUST include a statistical comparison table with columns:\n"
                    "| Comparison | Mean Diff | Std Diff | t-statistic | p-value | Significance |\n"
                    "Use the PAIRED STATISTICAL COMPARISONS data above to fill this table.\n"
                    "Mark significance: *** (p<0.001), ** (p<0.01), * (p<0.05), n.s.\n"
                    "This is non-negotiable — a top-venue paper MUST have statistical tests.\n"
                )
                exp_metrics_instruction += stat_table_block

            # R26: Metric definition requirement
            exp_metrics_instruction += (
                "\n\n## METRIC DEFINITIONS (MANDATORY in Experiments section)\n"
                "The Experiments section MUST define each metric:\n"
                "- **Primary metric**: what it measures, how it is computed, range, direction "
                "(higher/lower is better), and units if applicable.\n"
                "- **Secondary metric**: same details.\n"
                "- For time-to-event metrics: explain the horizon, what constitutes success, "
                "and how failures are handled (e.g., set to max horizon).\n"
                "- These definitions MUST appear BEFORE any results tables.\n"
            )

            # R27: Multi-seed framing enforcement
            _any_seeds = any(
                (cond_summaries.get(c) or {}).get("n_seed_metrics", 0) > 1
                for c in _cond_names
            ) if _cond_names else False
            if _any_seeds and not single_run.is_single_run(config):
                exp_metrics_instruction += (
                    "\n\n## MULTI-SEED EXPERIMENT FRAMING (CRITICAL)\n"
                    "This experiment uses MULTIPLE independent random seeds per condition.\n"
                    "- Report mean +/- std (or SE) for all metrics.\n"
                    "- NEVER describe this as 'a single run' or '1 benchmark-artifact run'.\n"
                    "- Frame as: 'We evaluate each method across N seeds per regime.'\n"
                    "- The seed-level data IS the evidence base — it is NOT a single observation.\n"
                    "- Include per-regime breakdowns (easy vs hard) as separate rows in tables.\n"
                )

    # BUG-003: Inject actual evaluated datasets as a hard constraint
    if exp_summary_text:
        _ds_parsed = _safe_json_loads(exp_summary_text, {})
        if isinstance(_ds_parsed, dict):
            _datasets: set[str] = set()
            # Extract from condition names (often contain dataset info)
            for _cname in (_ds_parsed.get("condition_summaries") or {}).keys():
                _datasets.add(str(_cname))
            # Extract from explicit "datasets" field if present
            for _ds in (_ds_parsed.get("datasets") or []):
                if isinstance(_ds, str):
                    _datasets.add(_ds)
            # Extract from "benchmark" or "dataset" fields
            for _key in ("benchmark", "dataset", "dataset_name"):
                _dv = _ds_parsed.get(_key)
                if isinstance(_dv, str) and _dv:
                    _datasets.add(_dv)
            if _datasets:
                exp_metrics_instruction += (
                    "\n\n## ACTUAL EVALUATED DATASETS (HARD CONSTRAINT)\n"
                    "The following datasets/conditions were ACTUALLY tested in experiments:\n"
                    + "".join(f"- {d}\n" for d in sorted(_datasets))
                    + "\nCRITICAL: Do NOT claim evaluation on any dataset not listed above.\n"
                    "Do NOT fabricate results for datasets you did not run experiments on.\n"
                    "If you reference other datasets, clearly state they are 'not evaluated "
                    "in this work' or are 'left for future work'.\n"
                )

    # P7: Ablation effectiveness check
    if exp_summary_text:
        _exp_parsed_p7 = _safe_json_loads(exp_summary_text, {})
        if isinstance(_exp_parsed_p7, dict):
            _abl_warnings = _check_ablation_effectiveness(_exp_parsed_p7)
            if _abl_warnings:
                _abl_block = (
                    "\n\n## ABLATION EFFECTIVENESS WARNINGS\n"
                    "The following ablations showed minimal effect (within 5%% of baseline). "
                    "Discuss this honestly — it may indicate the ablated component is not "
                    "important, or the ablation was not properly implemented:\n"
                )
                for _aw in _abl_warnings:
                    _abl_block += f"- {_aw}\n"
                exp_metrics_instruction += _abl_block
                logger.warning("P7: Ablation effectiveness warnings: %s", _abl_warnings)

    # P10: Contradiction detection
    if exp_summary_text:
        _exp_parsed_p10 = _safe_json_loads(exp_summary_text, {})
        if isinstance(_exp_parsed_p10, dict):
            _contradictions = _detect_result_contradictions(_exp_parsed_p10)
            if _contradictions:
                _contra_block = (
                    "\n\n## RESULT INTERPRETATION ADVISORIES (CRITICAL — read before writing)\n"
                )
                for _ca in _contradictions:
                    _contra_block += f"- {_ca}\n"
                exp_metrics_instruction += _contra_block
                logger.warning("P10: Contradiction advisories: %s", _contradictions)

    # R10: HARD BLOCK — refuse to write paper when all data is simulated
    all_simulated = True
    for stage_subdir in sorted(run_dir.glob("stage-*/runs")):
        for run_file in sorted(stage_subdir.glob("*.json")):
            if run_file.name == "results.json":
                continue
            try:
                _payload = json.loads(run_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(_payload, dict) and _payload.get("status") != "simulated":
                all_simulated = False
                break
        if not all_simulated:
            break

    if all_simulated:
        logger.error(
            "BLOCKED: All experiment data is simulated (mode='simulated'). "
            "Cannot write a paper based on formulaic fake data. "
            "Switch to experiment.mode='sandbox' and re-run."
        )
        (stage_dir / "paper_draft.md").write_text(
            "# Paper Draft Blocked\n\n"
            "**Reason**: All experiment results are from simulated mode "
            "(formulaic data: `0.3 + idx * 0.03`). "
            "These are not real experimental results.\n\n"
            "**Action Required**: Set `experiment.mode: 'sandbox'` in "
            "config.arc.yaml and re-run the pipeline.",
            encoding="utf-8",
        )
        return StageResult(
            stage=Stage.PAPER_DRAFT,
            status=StageStatus.FAILED,
            artifacts=("paper_draft.md",),
            evidence_refs=(),
        )

    # R4-2: HARD BLOCK — refuse to write paper with no real data (ML/empirical domains)
    # For non-empirical domains (math proofs, theoretical economics), allow proceeding
    _domain_id, _domain_name, _domain_venues = _detect_domain(
        config.research.topic, config.research.domains
    )
    _empirical_domains = {"ml", "engineering", "biology", "chemistry"}
    if not has_real_metrics:
        if _domain_id in _empirical_domains:
            logger.error(
                "BLOCKED: Cannot write paper — experiment produced NO metrics. "
                "The pipeline will not fabricate results."
            )
            (stage_dir / "paper_draft.md").write_text(
                "# Paper Draft Blocked\n\n"
                "**Reason**: Experiment stage produced no metrics (status: failed/timeout). "
                "Cannot write a paper without real experimental data.\n\n"
                "**Action Required**: Fix experiment execution or increase time_budget_sec.",
                encoding="utf-8",
            )
            return StageResult(
                stage=Stage.PAPER_DRAFT,
                status=StageStatus.FAILED,
                artifacts=("paper_draft.md",),
                evidence_refs=(),
            )
        else:
            logger.warning(
                "No experiment metrics found, but domain '%s' may be non-empirical "
                "(theoretical/mathematical). Proceeding with paper draft.",
                _domain_name,
            )

    # R11-5: Experiment quality minimum threshold before paper writing
    # Parse analysis.md for quality rating and condition completeness
    analysis_text = _read_prior_artifact(run_dir, "analysis.md") or ""
    _quality_warnings: list[str] = []

    # Check 1: Was the analysis quality rating very low?
    import re as _re_q
    _rating_match = _re_q.search(
        r"(?:quality\s+rating|result\s+quality)[:\s]*\**(\d+)\s*/\s*10",
        analysis_text,
        _re_q.IGNORECASE,
    )
    if _rating_match:
        _analysis_rating = int(_rating_match.group(1))
        if _analysis_rating <= 3:
            _quality_warnings.append(
                f"Analysis rated experiment quality {_analysis_rating}/10"
            )
        # BUG-23: If quality rating is ≤ 2, force has_real_metrics = False
        # to prevent fabricated results even if stdout had stray numbers.
        # R5-BUG-05: Skip override when _has_parsed_metrics is True — the
        # analysis.md may be stale (from pre-refinement Stage 14) while
        # Stage 13 refinement produced real parsed metrics.
        if _analysis_rating <= 2 and has_real_metrics and not _has_parsed_metrics:
            logger.warning(
                "BUG-23 guard: Analysis quality %d/10 ≤ 2 — "
                "overriding has_real_metrics to False (experiment likely failed)",
                _analysis_rating,
            )
            has_real_metrics = False

    # Check 2: Are baselines missing?
    _analysis_lower = analysis_text.lower()
    if "no" in _analysis_lower and "baseline" in _analysis_lower:
        if any(phrase in _analysis_lower for phrase in [
            "no baseline", "no bo", "no random", "baselines are missing",
            "missing baselines", "baseline coverage is missing",
        ]):
            _quality_warnings.append("Baselines appear to be missing from results")

    # Check 3: Is the metric undefined?
    if any(phrase in _analysis_lower for phrase in [
        "metric is undefined", "primary_metric is undefined",
        "undefined metric", "metric undefined",
    ]):
        _quality_warnings.append("Primary metric is undefined (direction/units/formula unknown)")

    # Check 4: Very few conditions completed
    _condition_count = len(_re_q.findall(
        r"condition[=:\s]+\w+.*?(?:mean|primary_metric)",
        raw_metrics_block or "",
        _re_q.IGNORECASE,
    ))

    if _quality_warnings:
        _warning_block = "\n".join(f"  - {w}" for w in _quality_warnings)
        logger.warning(
            "Stage 17: Experiment quality concerns detected before paper writing:\n%s",
            _warning_block,
        )
        # Inject quality warnings into the paper writing prompt so the LLM
        # writes an appropriately hedged paper
        exp_metrics_instruction += (
            "\n\n## EXPERIMENT QUALITY WARNINGS (address these honestly in the paper)\n"
            + "\n".join(f"- {w}" for w in _quality_warnings)
            + "\n\nBecause of these issues, the paper MUST:\n"
            "- Use hedged language ('preliminary', 'pilot', 'initial exploration')\n"
            "- NOT claim definitive comparisons between methods\n"
            "- Dedicate a substantial Limitations section to these gaps\n"
            "- Frame the contribution as methodology/framework, not empirical findings\n"
        )
        # Save warnings for tracking
        (stage_dir / "quality_warnings.json").write_text(
            json.dumps(_quality_warnings, indent=2), encoding="utf-8"
        )

    # R4-2: Anti-fabrication data integrity instruction
    experiment_evidence = _paper_experiment_evidence(run_dir)
    if experiment_evidence:
        exp_metrics_instruction += experiment_evidence

    exp_metrics_instruction += (
        "\n\n## CRITICAL: Data Integrity Rules\n"
        "- You may ONLY report numbers that appear in the experiment data above\n"
        "- If the experiment data is incomplete (fewer conditions than planned), report\n"
        "  ONLY the conditions that were actually run\n"
        "- Do NOT extrapolate, interpolate, or 'fill in' missing cells in tables\n"
        "- Do NOT invent confidence intervals, p-values, or statistical tests unless\n"
        "  the actual data supports them\n"
        "- If only N conditions completed, simply report results for those N conditions\n"
        "  without repeating apologies or disclaimers about missing conditions\n"
        "- Any table cell without real data must show '—' (not a plausible number)\n"
        "- FORBIDDEN: generating numbers that 'look right' based on your training data\n"
    )

    # IMP-6 + FA: Inject chart references into paper draft prompt
    # Prefer FigureAgent's figure_plan.json (rich descriptions) over raw file scan
    # BUG-FIX: figure_plan.json may be a list (from FigureAgent planner) or a dict
    # (from executor overwrite).  The orchestrator writes a list at planning time;
    # the executor overwrites with a dict only when figure_count > 0.  If the
    # FigureAgent renders 0 charts the list persists, and calling .get() on it
    # raises AttributeError.
    _fa_descriptions = ""
    # Charts and figure plans are generated in stage-16 (RESULT_ANALYSIS),
    # also check stage-14 as a legacy fallback.
    for _fig_stage_pattern in ("stage-16*", "stage-14*"):
        for _fig_stage_dir in _current_stage_dirs(run_dir, _fig_stage_pattern.rstrip("*")):
            for _fp_name in ("figure_plan_final.json", "figure_plan.json"):
                _fp_path = _fig_stage_dir / _fp_name
                if not _fp_path.exists():
                    continue
                try:
                    _fp_data = json.loads(_fp_path.read_text(encoding="utf-8"))
                    if isinstance(_fp_data, dict):
                        _fa_descriptions = _fp_data.get("figure_descriptions", "")
                    elif isinstance(_fp_data, list) and _fp_data:
                        _desc_parts = ["## PLANNED FIGURES (from figure plan)\n"]
                        for _fig in _fp_data:
                            if isinstance(_fig, dict):
                                _fid = _fig.get("figure_id", "unnamed")
                                _ftitle = _fig.get("title", "")
                                _fcap = _fig.get("caption", "")
                                _fsec = _fig.get("section", "results")
                                _desc_parts.append(
                                    f"- **{_fid}** ({_fsec}): {_ftitle}\n  {_fcap}"
                                )
                        if len(_desc_parts) > 1:
                            _fa_descriptions = "\n".join(_desc_parts)
                except (json.JSONDecodeError, OSError):
                    pass
                if _fa_descriptions:
                    break
            if _fa_descriptions:
                break
        if _fa_descriptions:
            break

    if _fa_descriptions:
        exp_metrics_instruction += "\n\n" + _fa_descriptions
        logger.info("Stage 17: Injected FigureAgent figure descriptions into paper draft prompt")
    else:
        # Fallback: scan for chart PNG files in stage-16 (RESULT_ANALYSIS) and stage-14
        _chart_files: list[str] = []
        for _chart_stage_pattern in ("stage-16*", "stage-14*"):
            for _chart_stage_dir in _current_stage_dirs(run_dir, _chart_stage_pattern.rstrip("*")):
                _charts_path = _chart_stage_dir / "charts"
                if _charts_path.is_dir():
                    for _cf in sorted(_charts_path.glob("*.png")):
                        if _cf.name not in _chart_files:
                            _chart_files.append(_cf.name)
        if _chart_files:
            _manifest = _read_figure_manifest(run_dir)
            _n = len(_chart_files)
            _chart_block = (
                f"\n\n## AVAILABLE DATA FIGURES — ALL {_n} MUST be embedded\n"
                f"The following {_n} figures were generated from actual experiment "
                "data. You MUST reference **ALL** of them in the Results / "
                "Analysis sections using markdown image syntax: "
                "`![Caption](charts/filename.png)`\n"
                "Each figure MUST have a descriptive caption and 2-3 sentences "
                "of discussion explaining what the figure shows.\n\n"
            )
            for _cf_name in _chart_files:
                _meta = _manifest.get(_cf_name, {})
                _title = _meta.get(
                    "title",
                    _cf_name.replace("_", " ").replace(".png", "").title(),
                )
                _caption = _meta.get("caption", "")
                _section = _meta.get("paper_section", "Results")
                _chart_block += (
                    f"- `charts/{_cf_name}` — **{_title}** "
                    f"(place in {_section})\n"
                )
                if _caption:
                    _chart_block += f"  Suggested caption: {_caption}\n"
            _chart_block += (
                "\nCRITICAL: A paper missing ANY of the above figures will be "
                "desk-rejected. Distribute them across Results and Analysis "
                "sections as indicated.\n"
            )
            exp_metrics_instruction += _chart_block
            logger.info(
                "Stage 17: Injected %d chart references (ALL required) "
                "into paper draft prompt",
                len(_chart_files),
            )

    # WS-5.5 → FIGURE_PROMPT: Build figure_prompt_instruction for non-data figures
    _venue_style = "NeurIPS / ICML / ICLR"
    _paper_len_cfg = getattr(config.experiment, "paper_length", "") or "full"
    if _paper_len_cfg == "short":
        _venue_style = "workshop / short paper"

    figure_prompt_instruction = (
        "\n\n## FIGURE GENERATION INSTRUCTIONS (MANDATORY)\n"
        "This paper uses a hybrid figure pipeline:\n\n"
        "### A. Non-data figures (architecture, method, pipeline, concept diagrams)\n"
        "For EVERY non-data figure, emit a structured FIGURE_PROMPT block "
        "using HTML comments **inline in the markdown** at the position where the "
        "figure should appear. Format:\n\n"
        "```\n"
        "<!-- FIGURE_PROMPT\n"
        "figure_id: unique_snake_case_id\n"
        "figure_type: architecture_diagram | method_flowchart | pipeline_overview | "
        "concept_illustration | system_diagram | comparison_illustration | "
        "attention_visualization\n"
        "section: Method | Results | Introduction | Discussion\n"
        "caption: \"Descriptive caption for the figure.\"\n"
        "aspect_ratio: 16:9\n"
        "prompt: |\n"
        "  A professional academic figure for a {venue} paper. "
        "[Describe: subject, layout (left-to-right / top-to-bottom), "
        "components (boxes, arrows, layers), labels, colors, annotations.] "
        "Use clean vector-style lines on a white background. "
        "Text must be legible at column width (~3.25 in). "
        "Color palette: blues and grays with one accent color for novelty.\n"
        "-->\n"
        "```\n"
        f"Replace {{venue}} with the target venue style: **{_venue_style}**.\n\n"
        "Then reference it in text:\n"
        "*[Figure N: <caption> — to be generated]*\n\n"
        "#### Required non-data figures:\n"
        "1. **Teaser figure** in the Introduction section — a high-level "
        "conceptual illustration showing the core idea/motivation at a glance "
        "(figure_type: concept_illustration). This is the FIRST figure in the "
        "paper and should convey the key insight without requiring method details. "
        "Place it right after the first introductory paragraph.\n"
        "2. **Framework / architecture overview** in the Method section "
        "(figure_type: architecture_diagram or pipeline_overview). This shows "
        "the full system/method pipeline with components, data flow, and key modules.\n"
        "3. **Method detail figure** — at least ONE additional figure in the Method "
        "section showing a key algorithmic step, comparison diagram, or attention "
        "visualization (figure_type: method_flowchart | comparison_illustration | "
        "attention_visualization).\n\n"
        "#### Prompt quality rules:\n"
        "- Prompt MUST be >= 50 words and self-contained.\n"
        "- Describe spatial layout (e.g., 'three boxes arranged left-to-right "
        "connected by arrows').\n"
        "- Specify labels for every component.\n"
        "- Mention the color scheme (e.g., 'blue boxes for encoder, orange for "
        "decoder, gray arrows for data flow').\n"
        "- Include the academic style: 'clean, professional, suitable for a "
        f"top-tier {_venue_style} paper'.\n\n"
        "### B. Data figures (charts, plots, heatmaps)\n"
        "Continue using `![Caption](charts/filename.png)` for pre-generated "
        "data charts, OR inline LaTeX/TikZ code blocks for simple data plots.\n"
        "Do NOT use FIGURE_PROMPT for data-driven visualizations.\n"
    )

    # P5: Extract hyperparameters from results.json for paper Method section
    _hp_table = ""
    for _s14_dir in sorted(run_dir.glob("stage-14*")):
        for _run_file in sorted(_s14_dir.glob("runs/*.json")):
            try:
                _run_data = json.loads(_run_file.read_text(encoding="utf-8"))
                if isinstance(_run_data, dict) and _run_data.get("hyperparameters"):
                    _hp = _run_data["hyperparameters"]
                    if isinstance(_hp, dict) and _hp:
                        _hp_table = "\n\n## HYPERPARAMETERS (include as a table in the Method section)\n"
                        _hp_table += "| Hyperparameter | Value |\n|---|---|\n"
                        for _hk, _hv in sorted(_hp.items()):
                            _hp_table += f"| {_hk} | {_hv} |\n"
                        _hp_table += (
                            "\nThis table MUST appear in the Method/Experiments section. "
                            "Include ALL hyperparameters used, with justification for key choices.\n"
                        )
                        break
            except (json.JSONDecodeError, OSError):
                continue
        if _hp_table:
            break
    # Also check staging dirs for results.json
    if not _hp_table:
        for _staging_dir in sorted(run_dir.glob("stage-*/runs/_docker_*")):
            _rjson = _staging_dir / "results.json"
            if _rjson.is_file():
                try:
                    _rdata = json.loads(_rjson.read_text(encoding="utf-8"))
                    if isinstance(_rdata, dict) and _rdata.get("hyperparameters"):
                        _hp = _rdata["hyperparameters"]
                        if isinstance(_hp, dict) and _hp:
                            _hp_table = "\n\n## HYPERPARAMETERS (include as a table in the Method section)\n"
                            _hp_table += "| Hyperparameter | Value |\n|---|---|\n"
                            for _hk, _hv in sorted(_hp.items()):
                                _hp_table += f"| {_hk} | {_hv} |\n"
                            _hp_table += (
                                "\nThis table MUST appear in the Method/Experiments section. "
                                "Include ALL hyperparameters used, with justification for key choices.\n"
                            )
                            break
                except (json.JSONDecodeError, OSError):
                    continue
    if _hp_table:
        exp_metrics_instruction += _hp_table

    # F2.6: Build citation list from references.bib / candidates with cite_keys
    citation_instruction = ""
    bib_text = _read_prior_artifact(run_dir, "references.bib")

    # P3: Pre-verify citations before paper draft — remove hallucinated refs
    if bib_text and bib_text.strip():
        from researchclaw.literature.verify import (
            filter_verified_bibtex,
            verify_citations as _verify_cit,
        )
        try:
            _pre_report = _verify_cit(bib_text, inter_verify_delay=0.5)
            _kept = _pre_report.verified + _pre_report.suspicious
            _removed = _pre_report.hallucinated
            if _removed > 0:
                bib_text = filter_verified_bibtex(
                    bib_text, _pre_report, include_suspicious=True
                )
                (stage_dir / "references_preverified.bib").write_text(
                    bib_text, encoding="utf-8"
                )
                logger.info(
                    "P3: Pre-verification kept %d/%d citations (removed %d hallucinated)",
                    _kept, _pre_report.total, _removed,
                )
        except Exception as exc:
            logger.warning("P3: Pre-verification failed, using original bib: %s", exc)

    candidates_text = _read_prior_artifact(run_dir, "candidates.jsonl")
    if candidates_text:
        cite_lines: list[str] = []
        for row_text in candidates_text.strip().splitlines():
            row = _safe_json_loads(row_text, {})
            if isinstance(row, dict) and row.get("cite_key"):
                authors_info = ""
                if isinstance(row.get("authors"), list) and row["authors"]:
                    first_author = row["authors"][0]
                    if isinstance(first_author, dict):
                        # BUG-38: name may be non-str (tuple/list) — force str
                        _name = first_author.get("name", "")
                        authors_info = _name if isinstance(_name, str) else str(_name)
                    elif isinstance(first_author, str):
                        authors_info = first_author
                    if len(row["authors"]) > 1:
                        authors_info += " et al."
                title = row.get("title", "")
                cite_lines.append(
                    f"- [{row['cite_key']}] → TITLE: \"{title}\" "
                    f"| {authors_info} "
                    f"({row.get('venue', '')}, {row.get('year', '')}, "
                    f"cited {row.get('citation_count', 0)} times) "
                    f"| ONLY cite this key when discussing: {title}"
                )
        if cite_lines:
            citation_instruction = (
                "\n\nAVAILABLE REFERENCES (use [cite_key] to cite in the text):\n"
                + "\n".join(cite_lines)
                + "\n\nCRITICAL CITATION RULES:\n"
                "- In the body text, cite using [cite_key] format, e.g. [smith2024transformer].\n"
                "- Do NOT write a References section — it will be auto-generated from the bibliography file.\n"
                "- Do NOT invent any references or arXiv IDs not in the above list.\n"
                "- You may cite a subset, but NEVER fabricate citations or change arXiv IDs.\n"
                "- SEMANTIC MATCHING: Before citing a reference, verify that its TITLE matches\n"
                "  the concept you are discussing. Do NOT use an unrelated cite_key just\n"
                "  because it sounds similar.\n"
                "- If no reference in the list matches the concept you want to cite,\n"
                "  write 'prior work has shown...' WITHOUT a citation, rather than using\n"
                "  a mismatched reference.\n"
                "- Each [cite_key] MUST correspond to the paper whose title is shown\n"
                "  next to that key in the list above. Cross-check before citing.\n"
                "\nCITATION QUANTITY & QUALITY CONSTRAINTS:\n"
                "- Cite 25-40 unique references in the paper body. The Related Work\n"
                "  section alone should cite at least 15 references.\n"
                "- Every citation MUST be directly relevant to the paper's topic.\n"
                "- DO NOT cite papers from unrelated domains (wireless communication, "
                "manufacturing, UAV, etc.).\n"
                "- Prefer well-known, highly-cited papers over obscure ones.\n"
                "- If unsure whether a paper exists or is relevant, DO NOT cite it.\n"
            )

    if llm is not None:
        _pm = prompts or PromptManager()
        topic_constraint = _pm.block("topic_constraint", topic=config.research.topic)

        # --- Section-by-section writing (3 calls) for conference-grade depth ---
        draft = _write_paper_sections(
            llm=llm,
            pm=_pm,
            run_dir=run_dir,
            preamble=preamble,
            topic_constraint=topic_constraint,
            exp_metrics_instruction=exp_metrics_instruction,
            citation_instruction=citation_instruction,
            outline=outline,
            model_name=config.llm.primary_model,
            discussion_ablation=discussion_ablation,
            figure_prompt_instruction=figure_prompt_instruction,
        )

        # R7: Strip LLM-generated References section — it often fabricates arXiv IDs.
        import re as _re_r7
        ref_pattern = _re_r7.compile(
            r'^(#{1,2}\s*References.*)', _re_r7.MULTILINE | _re_r7.DOTALL
        )
        ref_match = ref_pattern.search(draft)
        if ref_match:
            draft = draft[:ref_match.start()].rstrip()
            logger.info("Stage 20: Stripped LLM-generated References section (R7 fix)")
    else:
        # Build template with real data if available
        results_section = "Template results summary."
        if exp_summary_text:
            exp_summary = _safe_json_loads(exp_summary_text, {})
            if isinstance(exp_summary, dict) and exp_summary.get("metrics_summary"):
                lines = ["Experiment results:"]
                for mk, mv in exp_summary["metrics_summary"].items():
                    if isinstance(mv, dict):
                        lines.append(
                            f"- {mk}: mean={mv.get('mean')}, min={mv.get('min')}, "
                            f"max={mv.get('max')}, n={mv.get('count')}"
                        )
                results_section = "\n".join(lines)

        draft = f"""# Draft Title

## Abstract
Template draft abstract.

## Introduction
Template introduction for {config.research.topic}.

## Related Work
Template related work.

## Method
Template method description.

## Experiments
Template experimental setup.

## Results
{results_section}

## Limitations
Template limitations.

## Conclusion
Template conclusion.

## References
Template references.

Generated: {_utcnow_iso()}
"""
    (stage_dir / "paper_draft.md").write_text(draft, encoding="utf-8")

    # Extract FIGURE_PROMPT blocks, render via NanoBanana, write figure_prompts.json
    _topic = config.research.topic if hasattr(config, "research") else ""
    _fig_prompts = _extract_figure_prompts(draft, topic=_topic)
    _draft_artifacts = ["paper_draft.md"]
    if experiment_evidence:
        (stage_dir / "experiment_evidence.md").write_text(
            experiment_evidence, encoding="utf-8"
        )
        _draft_artifacts.append("experiment_evidence.md")
    if _fig_prompts:
        _fig_prompts = _render_figure_prompts(
            _fig_prompts, stage_dir, config, llm, topic=_topic,
        )
        _fp_path = stage_dir / "figure_prompts.json"
        _fp_path.write_text(
            json.dumps(_fig_prompts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _draft_artifacts.append("figure_prompts.json")
        for _fp in _fig_prompts:
            if _fp.get("success") and _fp.get("output_path"):
                _rel = Path(_fp["output_path"]).name
                _draft_artifacts.append(f"figures/{_rel}")
        logger.info(
            "Stage 20: Extracted %d figure prompts → %s",
            len(_fig_prompts), _fp_path,
        )

    # Validate draft quality (section balance + bullet density)
    _validate_draft_quality(draft, stage_dir=stage_dir, single_run_mode=single_run.is_single_run(config))

    return StageResult(
        stage=Stage.PAPER_DRAFT,
        status=StageStatus.DONE,
        artifacts=tuple(_draft_artifacts),
        evidence_refs=("stage-17/paper_draft.md",),
    )


def _collect_experiment_evidence(run_dir: Path) -> str:
    """Collect actual experiment parameters and results for peer review."""
    evidence_parts: list[str] = []
    exp_summary = _load_experiment_summary(run_dir)
    selected_version = str((exp_summary or {}).get("selected_version", ""))
    has_final_version = selected_version.startswith(f"stage-{int(Stage.ITERATIVE_REFINE):02d}")

    # 1. Read experiment code to find actual trial count, methods used.
    # Prefer the refined final code (ITERATIVE_REFINE) over the initial draft.
    for _code_dir_name in ("experiment_final/", "experiment/"):
        exp_dir = _read_prior_artifact(run_dir, _code_dir_name)
        if exp_dir and Path(exp_dir).is_dir():
            main_py = Path(exp_dir) / "main.py"
            if main_py.exists():
                code = main_py.read_text(encoding="utf-8")
                evidence_parts.append(f"### Actual Experiment Code (main.py)\n```python\n{code[:3000]}\n```")
                break

    # 2. Read sandbox run results (actual metrics, runtime, stderr)
    runs_text = _read_prior_artifact(run_dir, "runs/")
    if runs_text and Path(runs_text).is_dir():
        for run_file in sorted(Path(runs_text).glob("*.json"))[:5]:
            payload = _safe_json_loads(run_file.read_text(encoding="utf-8"), {})
            if isinstance(payload, dict):
                metrics = _run_payload_metrics(payload)
                summary = {
                    "metrics": dict(list(metrics.items())[:40]) or None,
                    "elapsed_sec": payload.get("elapsed_sec"),
                    "timed_out": payload.get("timed_out"),
                }
                # Agentic EXPERIMENT_RUN report (run_report.json)
                for _key in ("success", "has_results"):
                    if _key in payload:
                        summary[_key] = payload[_key]
                stderr = payload.get("stderr", "")
                if stderr:
                    summary["stderr_excerpt"] = stderr[:500]
                result_label = (
                    "Initial Version Result (not a final-version repeat)"
                    if has_final_version else "Run Result"
                )
                evidence_parts.append(
                    f"### {result_label}: {run_file.name}\n```json\n{json.dumps(summary, indent=2)}\n```"
                )

    # 3. Read refinement log for actual iteration count
    refine_log_text = _read_prior_artifact(run_dir, "refinement_log.json")
    if refine_log_text:
        try:
            rlog = json.loads(refine_log_text)
            _iters = rlog.get("iterations")
            summary = {
                # Legacy logs store a list of iterations; agentic logs store a count.
                "iterations_executed": len(_iters) if isinstance(_iters, list) else _iters,
                "converged": rlog.get("converged"),
                "stop_reason": rlog.get("stop_reason"),
                "best_metric": rlog.get("best_metric", rlog.get("final_metric")),
            }
            if "improved" in rlog:
                summary["improved"] = rlog.get("improved")
                summary["baseline_metric"] = rlog.get("baseline_metric")
            evidence_parts.append(
                f"### Refinement Summary\n```json\n{json.dumps(summary, indent=2)}\n```"
            )
        except (json.JSONDecodeError, TypeError):
            pass

    # 4. Final aggregated results from RESULT_ANALYSIS — authoritative record
    if exp_summary:
        _summary_view: dict[str, Any] = {
            k: exp_summary[k]
            for k in (
                "total_runs", "primary_metric_definition", "direction", "metrics_summary", "best_run",
                "selected_version", "run_count_basis",
            )
            if k in exp_summary
        }
        _conds = exp_summary.get("condition_summaries")
        if isinstance(_conds, dict):
            _summary_view["conditions"] = {
                c: v.get("metrics") for c, v in _conds.items() if isinstance(v, dict)
            }
        evidence_parts.append(
            "### Result Analysis Summary (experiment_summary.json)\n"
            f"```json\n{json.dumps(_summary_view, indent=2, ensure_ascii=False)[:4000]}\n```"
        )

    # 5. Count actual number of experiment runs
    actual_run_count = 0
    _total_runs = exp_summary.get("total_runs") if exp_summary else None
    if isinstance(_total_runs, int) and not isinstance(_total_runs, bool) and _total_runs > 0:
        actual_run_count = _total_runs
    else:
        for stage_subdir in sorted(run_dir.glob("stage-*/runs")):
            for rf in stage_subdir.glob("*.json"):
                if rf.name != "results.json":
                    actual_run_count += 1
    if actual_run_count > 0:
        if exp_summary and exp_summary.get("run_count_basis") == "result_records":
            evidence_parts.append(
                f"### Result Record Count\n"
                f"The selected version has {actual_run_count} result record(s). "
                "Code versions and aggregate reports are not independent repeat trials; "
                "verify repeat counts from explicit condition and seed records."
            )
        else:
            evidence_parts.append(
                f"### Actual Trial Count\n"
                + (
                    f"**The analysed experiment comprises {actual_run_count} run(s) "
                    f"(per experiment_summary.json total_runs).** "
                    if actual_run_count == _total_runs
                    else f"**The experiment was executed {actual_run_count} time(s).** "
                )
                + "If the paper claims a different number of trials, this is a CRITICAL discrepancy."
            )

    audit_section = _execution_audit_section(run_dir, "### Verified Execution Audit").strip()
    if audit_section:
        evidence_parts.append(audit_section)

    if not evidence_parts:
        return ""

    return (
        "\n\n## Actual Experiment Evidence\n"
        "Use the evidence below to verify the paper's methodology claims.\n\n"
        + "\n\n".join(evidence_parts)
    )


def _execute_peer_review(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    draft = _read_prior_artifact(run_dir, "paper_draft.md") or ""
    experiment_evidence = _collect_experiment_evidence(run_dir)

    # Load draft quality warnings from Stage 17 (if available)
    _quality_suffix = ""
    _quality_json_path = _find_prior_file(run_dir, "draft_quality.json")
    if _quality_json_path and _quality_json_path.exists():
        try:
            _dq = json.loads(_quality_json_path.read_text(encoding="utf-8"))
            _dq_warnings = _dq.get("overall_warnings", [])
            if _dq_warnings:
                _quality_suffix = (
                    "\n\nAUTOMATED QUALITY ISSUES (flag these in your review):\n"
                    + "\n".join(f"- {w}" for w in _dq_warnings)
                    + "\n"
                )
        except Exception:  # noqa: BLE001
            pass

    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "peer_review")
        chunks = _split_markdown_for_peer_review(draft)
        chunk_reviews: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            sp = _pm.for_stage(
                "peer_review",
                evolution_overlay=_overlay,
                topic=config.research.topic,
                draft=chunk,
                experiment_evidence=experiment_evidence,
            )
            review_user = sp.user + _quality_suffix
            if len(chunks) > 1:
                review_user = (
                    f"This is part {index} of {len(chunks)} of a long paper. "
                    "Review only the supplied part, and identify missing context "
                    "without inventing claims about unseen sections.\n\n"
                    + review_user
                )
            response = _chat_with_prompt(
                llm,
                sp.system,
                review_user,
                json_mode=sp.json_mode,
                max_tokens=sp.max_tokens,
                retries=2,
            )
            chunk_reviews.append(response.content)

        if len(chunk_reviews) == 1:
            reviews = chunk_reviews[0]
        else:
            review_digest = "\n\n".join(
                f"### Part {index} review\n{review[:_PEER_REVIEW_SYNTHESIS_CHARS]}"
                for index, review in enumerate(chunk_reviews, start=1)
            )
            synthesis_prompt = (
                "Synthesize the following section-level review notes into one final "
                "peer review of the complete paper. Output exactly Reviewer A "
                "(methodology expert), Reviewer B (domain expert), and Reviewer C "
                "(statistics/rigor expert), each with strengths, weaknesses, and "
                "actionable revisions. Reconcile duplicates and flag issues only "
                "when supported by the notes.\n\n"
                + review_digest
            )
            response = _chat_with_prompt(
                llm,
                "You are a senior editor consolidating peer-review notes.",
                synthesis_prompt,
                max_tokens=8192,
                retries=2,
            )
            reviews = response.content
    else:
        reviews = """# Reviews

## Reviewer A
- Strengths: Clear problem statement.
- Weaknesses: Limited ablation details.
- Actionable revisions: Add uncertainty analysis and stronger baselines.

## Reviewer B
- Strengths: Reproducibility focus.
- Weaknesses: Discussion underdeveloped.
- Actionable revisions: Expand limitations and broader impact.
"""
    (stage_dir / "reviews.md").write_text(reviews, encoding="utf-8")
    return StageResult(
        stage=Stage.PEER_REVIEW,
        status=StageStatus.DONE,
        artifacts=("reviews.md",),
        evidence_refs=("stage-18/reviews.md",),
    )


def _execute_paper_revision(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    draft = _read_prior_artifact(run_dir, "paper_draft.md") or ""
    reviews = _read_prior_artifact(run_dir, "reviews.md") or ""
    draft_word_count = len(draft.split())

    # R4-2: Collect real metrics for anti-fabrication guard in revision
    # BUG-47: _collect_raw_experiment_metrics returns tuple[str, bool], must unpack
    _raw_metrics_tuple = _collect_raw_experiment_metrics(run_dir)
    raw_metrics_revision = _raw_metrics_tuple[0] if isinstance(_raw_metrics_tuple, tuple) else (_raw_metrics_tuple or "")
    experiment_evidence = _paper_experiment_evidence(run_dir)
    data_integrity_revision = ""
    if raw_metrics_revision:
        data_integrity_revision = (
            raw_metrics_revision
            + "\nDATA INTEGRITY: Do NOT add new numbers that are not in the "
            "experiment data above. If a reviewer asks for additional results "
            "you do not have, state 'Due to computational constraints, "
            "this analysis was not conducted' instead of fabricating data.\n"
        )

    if llm is not None:
        _pm = prompts or PromptManager()
        try:
            _ws_revision = _pm.block("writing_structure")
        except (KeyError, Exception):  # noqa: BLE001
            _ws_revision = ""
        # IMP-20/25/31/24: Load style blocks for revision prompt
        _rev_blocks: dict[str, str] = {}
        for _bname in ("academic_style_guide", "narrative_writing_rules",
                        "anti_hedging_rules", "anti_repetition_rules"):
            try:
                _rev_blocks[_bname] = _pm.block(_bname)
            except (KeyError, Exception):  # noqa: BLE001
                _rev_blocks[_bname] = ""
        # Load draft quality directives from Stage 17
        _quality_prefix = ""
        _quality_json_path = _find_prior_file(run_dir, "draft_quality.json")
        if _quality_json_path and _quality_json_path.exists():
            try:
                _dq = json.loads(_quality_json_path.read_text(encoding="utf-8"))
                _dq_directives = _dq.get("revision_directives", [])
                if _dq_directives:
                    _quality_prefix = (
                        "MANDATORY QUALITY FIXES (address ALL of these):\n"
                        + "\n".join(f"- {d}" for d in _dq_directives)
                        + "\n\n"
                    )
            except Exception:  # noqa: BLE001
                pass

        _overlay = _get_evolution_overlay(run_dir, "paper_revision")
        sp = _pm.for_stage(
            "paper_revision",
            evolution_overlay=_overlay,
            topic_constraint=_pm.block("topic_constraint", topic=config.research.topic),
            writing_structure=_ws_revision,
            draft=draft,
            reviews=_quality_prefix + reviews + data_integrity_revision,
            experiment_evidence=experiment_evidence,
            **_rev_blocks,
        )
        # R10-Fix2: Ensure max_tokens is sufficient for full paper revision
        revision_max_tokens = sp.max_tokens
        if revision_max_tokens and draft_word_count > 0:
            # ~1.5 tokens per word, 20% headroom
            min_tokens_needed = int(draft_word_count * 1.5 * 1.2)
            if revision_max_tokens < min_tokens_needed:
                revision_max_tokens = min_tokens_needed
                logger.info(
                    "Stage 19: Increased max_tokens from %d to %d to fit full paper revision",
                    sp.max_tokens,
                    revision_max_tokens,
                )

        # R10-Fix4: Retry on timeout for paper revision (critical stage)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=revision_max_tokens,
            retries=2,
        )
        revised = resp.content
        revised_word_count = len(revised.split())
        # Length guard: if revision is shorter than 80% of draft, retry once
        if draft_word_count > 500 and revised_word_count < int(draft_word_count * 0.8):
            logger.warning(
                "Paper revision (%d words) is shorter than draft (%d words). "
                "Retrying with stronger length enforcement.",
                revised_word_count,
                draft_word_count,
            )
            retry_user = (
                f"CRITICAL LENGTH REQUIREMENT: The draft is {draft_word_count} words. "
                f"Your revision MUST be at least {draft_word_count} words — ideally longer. "
                f"Do NOT summarize or condense ANY section. Copy each section verbatim "
                f"and ONLY make targeted improvements to address reviewer comments. "
                f"If a section has no reviewer comments, include it UNCHANGED.\n\n"
                + sp.user
            )
            resp2 = _chat_with_prompt(
                llm, sp.system, retry_user,
                json_mode=sp.json_mode, max_tokens=revision_max_tokens,
            )
            revised2 = resp2.content
            revised2_word_count = len(revised2.split())
            if revised2_word_count >= int(draft_word_count * 0.8):
                revised = revised2
            elif revised2_word_count > revised_word_count:
                # Retry improved but still not enough — use the longer version
                revised = revised2
                logger.warning(
                    "Retry improved (%d → %d words) but still shorter than draft (%d).",
                    revised_word_count,
                    revised2_word_count,
                    draft_word_count,
                )
            else:
                # Both attempts produced short output — preserve full original draft
                logger.warning(
                    "Retry also produced short output (%d words). "
                    "Falling back to FULL ORIGINAL DRAFT to prevent content loss.",
                    revised2_word_count,
                )
                # Extract useful revision points as appendix
                revision_words = revised.split()
                revision_summary = (
                    " ".join(revision_words[:500]) + "\n\n*(Revision summary truncated)*"
                    if len(revision_words) > 500
                    else revised
                )
                if revision_summary.strip():
                    # Save revision notes to internal file, not paper body
                    (stage_dir / "revision_notes_internal.md").write_text(
                        revision_summary, encoding="utf-8"
                    )
                revised = draft
        sync_conflicts = _paper_evidence_conflicts(revised, run_dir)
        if sync_conflicts:
            correction_prompt = (
                "The proposed revision contradicts the authoritative experiment evidence. "
                "Correct every listed conflict while preserving all supported limitations. "
                "Return the complete revised markdown only.\n\n"
                + "\n".join(f"- {conflict}" for conflict in sync_conflicts)
                + "\n\n"
                + sp.user
            )
            corrected = _chat_with_prompt(
                llm,
                sp.system,
                correction_prompt,
                json_mode=sp.json_mode,
                max_tokens=revision_max_tokens,
                retries=1,
            ).content
            corrected_conflicts = _paper_evidence_conflicts(corrected, run_dir)
            draft_conflicts = _paper_evidence_conflicts(draft, run_dir)
            candidates = (
                (corrected, corrected_conflicts, "corrected_revision"),
                (revised, sync_conflicts, "initial_revision"),
                (draft, draft_conflicts, "original_draft"),
            )
            revised, final_conflicts, selected_source = min(
                candidates, key=lambda item: (len(item[1]), -len(item[0]))
            )
            sync_report = {
                "initial_conflicts": sync_conflicts,
                "corrected_conflicts": corrected_conflicts,
                "original_draft_conflicts": draft_conflicts,
                "selected_source": selected_source,
                "remaining_conflicts": final_conflicts,
            }
        else:
            sync_report = {
                "initial_conflicts": [],
                "corrected_conflicts": [],
                "original_draft_conflicts": _paper_evidence_conflicts(draft, run_dir),
                "selected_source": "initial_revision",
                "remaining_conflicts": [],
            }
    else:
        revised = draft
        draft_conflicts = _paper_evidence_conflicts(draft, run_dir)
        sync_report = {
            "initial_conflicts": [],
            "corrected_conflicts": [],
            "original_draft_conflicts": draft_conflicts,
            "selected_source": "original_draft_no_llm",
            "remaining_conflicts": draft_conflicts,
        }
    revised, _dup_figures = _dedupe_markdown_figures(revised)
    if _dup_figures:
        logger.info("Stage 22: Removed %d repeated figure reference(s)", _dup_figures)
    (stage_dir / "paper_revised.md").write_text(revised, encoding="utf-8")

    _revision_artifacts = ["paper_revised.md"]
    if experiment_evidence:
        (stage_dir / "experiment_evidence.md").write_text(
            experiment_evidence, encoding="utf-8"
        )
        _revision_artifacts.append("experiment_evidence.md")
        (stage_dir / "experiment_sync_report.json").write_text(
            json.dumps(sync_report, indent=2), encoding="utf-8"
        )
        _revision_artifacts.append("experiment_sync_report.json")
    _topic = config.research.topic if hasattr(config, "research") else ""

    # Reuse figures from S20 (PAPER_DRAFT) instead of regenerating
    _s20_figures_dir = None
    for _sd in sorted(run_dir.glob("stage-20*"), reverse=True):
        _candidate = _sd / "figures"
        if _candidate.is_dir() and any(_candidate.iterdir()):
            _s20_figures_dir = _candidate
            break

    _fig_prompts = _extract_figure_prompts(revised, topic=_topic)
    if _fig_prompts:
        _s22_figures_dir = stage_dir / "figures"
        _s22_figures_dir.mkdir(parents=True, exist_ok=True)

        # Copy existing figures from S20 before rendering new ones
        _reused_ids: set[str] = set()
        if _s20_figures_dir:
            import shutil as _shutil_fig
            for _existing_fig in _s20_figures_dir.iterdir():
                if _existing_fig.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".eps"}:
                    _dest = _s22_figures_dir / _existing_fig.name
                    if not _dest.exists():
                        _shutil_fig.copy2(_existing_fig, _dest)
            # Mark prompts as already rendered if their output file exists
            for _fp in _fig_prompts:
                _fid = _fp.get("figure_id", "")
                # Check common naming patterns
                for _ext in (".png", ".jpg", ".jpeg"):
                    _candidate_name = _s22_figures_dir / f"{_fid}{_ext}"
                    if _candidate_name.exists():
                        _fp["output_path"] = str(_candidate_name)
                        _fp["success"] = True
                        _reused_ids.add(_fid)
                        break
            if _reused_ids:
                logger.info(
                    "Stage 22: Reused %d/%d figures from S20",
                    len(_reused_ids), len(_fig_prompts),
                )

        # Only render prompts that don't already have figures
        _new_prompts = [fp for fp in _fig_prompts if fp.get("figure_id") not in _reused_ids]
        if _new_prompts:
            _new_prompts = _render_figure_prompts(
                _new_prompts, stage_dir, config, llm, topic=_topic,
            )
            # Merge back into _fig_prompts
            _new_map = {fp["figure_id"]: fp for fp in _new_prompts}
            for _fp in _fig_prompts:
                if _fp["figure_id"] in _new_map:
                    _fp.update(_new_map[_fp["figure_id"]])
            logger.info(
                "Stage 22: Rendered %d new figures (reused %d from S20)",
                len(_new_prompts), len(_reused_ids),
            )

        _fp_path = stage_dir / "figure_prompts.json"
        _fp_path.write_text(
            json.dumps(_fig_prompts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _revision_artifacts.append("figure_prompts.json")
        for _fp in _fig_prompts:
            if _fp.get("success") and _fp.get("output_path"):
                _rel = Path(_fp["output_path"]).name
                _revision_artifacts.append(f"figures/{_rel}")
        logger.info(
            "Stage 22: Total %d figure prompts → %s",
            len(_fig_prompts), _fp_path,
        )

    # --- Generate LaTeX package for Overleaf ---
    try:
        _latex_artifacts = _generate_latex_package(stage_dir, run_dir, config, revised, llm=llm)
        _revision_artifacts.extend(_latex_artifacts)
        logger.info("Stage 22: LaTeX package generated → %s", stage_dir / "latex_package.zip")
    except Exception:
        logger.warning("Stage 22: LaTeX package generation failed", exc_info=True)

    return StageResult(
        stage=Stage.PAPER_REVISION,
        status=StageStatus.DONE,
        artifacts=tuple(_revision_artifacts),
        evidence_refs=tuple(f"stage-22/{a}" for a in _revision_artifacts),
    )


_MD_IMAGE_LINE_RE = re.compile(r"^[ \t]*!\[[^\]]*\]\(\s*<?([^)\s>]+)>?[^)]*\)[ \t]*$", re.MULTILINE)
_TEX_FIGURE_RE = re.compile(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}\n?", re.DOTALL)
_TEX_GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def _figure_key(path: str) -> str:
    return Path(path.strip().replace("\\", "/")).name.lower()


def _dedupe_markdown_figures(paper: str) -> tuple[str, int]:
    """Drop repeated image lines that show an already embedded figure file."""
    seen: set[str] = set()
    removed = 0

    def _keep_first(match: re.Match[str]) -> str:
        nonlocal removed
        key = _figure_key(match.group(1))
        if key in seen:
            removed += 1
            return ""
        seen.add(key)
        return match.group(0)

    deduped = _MD_IMAGE_LINE_RE.sub(_keep_first, paper)
    embedded = set(re.findall(r"!\[\s*(?:Figure|Fig\.)\s*(\d+)", deduped))

    def _drop_orphan(match: re.Match[str]) -> str:
        nonlocal removed
        if embedded and match.group(1) not in embedded:
            removed += 1
            return ""
        return match.group(0)

    deduped = re.sub(
        r"^[ \t]*\*(?:Figure|Fig\.)\s*(\d+)[.:][^\n]*\*[ \t]*$",
        _drop_orphan, deduped, flags=re.MULTILINE,
    )
    if removed:
        deduped = re.sub(r"\n{3,}", "\n\n", deduped)
    return deduped, removed


def _dedupe_latex_figures(tex: str) -> tuple[str, int]:
    """Drop figure environments whose graphics were all included earlier."""
    seen: set[str] = set()
    removed = 0

    def _keep_first(match: re.Match[str]) -> str:
        nonlocal removed
        keys = {_figure_key(g) for g in _TEX_GRAPHIC_RE.findall(match.group(0))}
        if keys and keys <= seen:
            removed += 1
            return ""
        seen.update(keys)
        return match.group(0)

    return _TEX_FIGURE_RE.sub(_keep_first, tex), removed


def _generate_latex_package(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    paper_md: str,
    *,
    llm: "LLMClient | None" = None,
) -> list[str]:
    """Generate a complete LaTeX package (zip) ready for Overleaf upload.

    Uses LLM to convert markdown paper to LaTeX, then collects figures,
    references, and style files into a zip archive.
    """
    import shutil as _shutil_lp

    pkg_dir = stage_dir / "latex_package"
    if pkg_dir.exists():
        _shutil_lp.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True)
    fig_dir = pkg_dir / "figures"
    fig_dir.mkdir()

    # 1. Collect available figures first (need filenames for LLM prompt)
    _fig_files: list[str] = []
    for _fig_src in [
        stage_dir / "figures",
        run_dir / "stage-16" / "charts",
        run_dir / "stage-14" / "charts",
    ]:
        if _fig_src.is_dir():
            for _fig_file in sorted(_fig_src.iterdir()):
                if (_fig_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".eps"}
                        and _fig_file.name not in _fig_files):
                    _shutil_lp.copy2(_fig_file, fig_dir / _fig_file.name)
                    _fig_files.append(_fig_file.name)

    # 2. Collect references.bib
    bib_text = ""
    for _bib_candidate in [
        run_dir / "stage-04" / "references.bib",
        run_dir / "stage-22" / "references.bib",
    ]:
        if _bib_candidate.exists():
            bib_text = _bib_candidate.read_text(encoding="utf-8")
            break
    if not bib_text:
        bib_text = "% No references collected\n"

    # Extract cite keys from bib
    _bib_keys = re.findall(r"@\w+\{([^,]+),", bib_text)

    # 3. Use LLM to convert markdown to LaTeX
    if llm is not None:
        _system_prompt = (
            "You are an expert academic LaTeX typesetter. Convert the markdown paper "
            "into a complete .tex file. The references.bib file already exists and is "
            "used as is; do not write a .bib file.\n\n"
            "Output format: output the complete .tex inside ```latex ... ```.\n\n"
            "STRICT RULES:\n"
            "1. The .tex must be complete and compilable. No explanations outside code blocks.\n"
            "2. Use \\usepackage[preprint]{neurips_2025} as the style package.\n"
            "3. All citations [cite_xxx] in the markdown MUST become \\cite{xxx} in LaTeX, "
            "keeping the key 'xxx' unchanged.\n"
            "4. Cite only keys from the list of available BibTeX keys. Never add a new "
            "citation or invent a key; keep a citation whose key is missing unchanged so "
            "it can be reviewed.\n"
            "5. References section: Do NOT include a hand-written reference list in the .tex. "
            "Use \\bibliographystyle{plainnat} and \\bibliography{references} at the end.\n"
            "5. Section hierarchy: Use \\section{} for main sections (Introduction, "
            "Related Work, Method, Experiments, Results, Discussion, Limitations, "
            "Conclusion). Use \\subsection{} for sub-sections within them.\n"
            "6. All figures are in the figures/ directory. Use "
            "\\includegraphics[width=0.85\\textwidth]{figures/filename.png} with "
            "proper \\begin{figure}...\\end{figure} environments, \\caption{}, "
            "and \\label{fig:xxx}.\n"
            "7. Convert <!-- FIGURE_PROMPT --> comment blocks into \\begin{figure} "
            "environments. Match figure prompts to available figure files by their "
            "content/description.\n"
            "8. Include each figure the paper references exactly once; never repeat a "
            "figure file in a second figure environment.\n"
            "9. Preserve all math equations (both inline $...$ and display $$...$$).\n"
            "10. Convert markdown tables to LaTeX \\begin{table}...\\end{table} with "
            "\\begin{tabular}, \\toprule, \\midrule, \\bottomrule.\n"
            "11. Use \\textit{} for italics, \\textbf{} for bold.\n"
            "12. The preamble must include: hyperref, url, booktabs, amsfonts, "
            "amsmath, graphicx, natbib, microtype, inputenc (utf8), fontenc (T1), "
            "placeins (for \\FloatBarrier).\n"
            "13. Use \\begin{figure}[htbp] (NOT [t]) so figures stay near their text.\n"
            "14. Add \\FloatBarrier before \\section{Conclusion} and before "
            "\\bibliographystyle to prevent figures from floating past them.\n"
            "15. Spread figures evenly across sections. Do NOT cluster multiple "
            "figures together — place each figure near the text that discusses it.\n"
            "16. Figure placement priority: figure_1.png should be the teaser/overview "
            "figure (after abstract/intro). figure_2.png should be the architecture/method "
            "figure. fig_main_results and fig_radar are RESULTS figures — place in Results section. "
            "Use the figure descriptions below to determine correct placement.\n"
        )

        # Load figure descriptions from figure_prompts.json
        _fig_descriptions = ""
        _fp_path = stage_dir / "figure_prompts.json"
        if _fp_path.exists():
            try:
                _fp_data = json.loads(_fp_path.read_text(encoding="utf-8"))
                _fig_desc_lines = []
                for fp in _fp_data:
                    fname = Path(fp.get("output_path", "")).name if fp.get("output_path") else ""
                    desc = fp.get("caption", fp.get("raw_prompt", ""))
                    ftype = fp.get("figure_type", "")
                    section = fp.get("section", "")
                    if fname and desc:
                        _fig_desc_lines.append(
                            f"- {fname}: [{ftype}] for {section} section — {desc}"
                        )
                if _fig_desc_lines:
                    _fig_descriptions = (
                        "\n\n## Figure descriptions (use these to place figures correctly):\n"
                        + chr(10).join(_fig_desc_lines)
                    )
            except Exception:
                pass

        _user_prompt = (
            f"## Available figure files in figures/ directory:\n"
            f"{chr(10).join('- ' + f for f in _fig_files) if _fig_files else '(none)'}\n"
            f"{_fig_descriptions}\n\n"
            f"## Available BibTeX citation keys:\n"
            f"{chr(10).join('- ' + k for k in _bib_keys[:50]) if _bib_keys else '(none)'}\n\n"
            f"## Markdown paper to convert:\n\n{paper_md}\n"
        )

        try:
            resp = _chat_with_prompt(
                llm,
                _system_prompt,
                _user_prompt,
                max_tokens=32768,
            )
            raw_output = resp.content

            # Extract .tex from ```latex ... ``` block
            tex_content = ""
            if "```latex" in raw_output:
                tex_content = raw_output.split("```latex", 1)[1].split("```", 1)[0].strip()
            elif "```tex" in raw_output:
                tex_content = raw_output.split("```tex", 1)[1].split("```", 1)[0].strip()
            elif "```" in raw_output and "\\documentclass" in raw_output:
                tex_content = raw_output.split("```", 1)[1].split("```", 1)[0].strip()
            else:
                tex_content = raw_output

            # Extract .bib from ```bib ... ``` block (if LLM generated one)
            _llm_bib = ""
            if "```bib" in raw_output:
                _llm_bib = raw_output.split("```bib", 1)[1].split("```", 1)[0].strip()
            elif "```bibtex" in raw_output:
                _llm_bib = raw_output.split("```bibtex", 1)[1].split("```", 1)[0].strip()
            if _llm_bib and "@" in _llm_bib:
                # Entries written by the LLM are not from a literature source;
                # the collected bibliography is kept as is.
                logger.info("LaTeX package: Ignored LLM-generated bib (%d entries); "
                            "keeping the collected references.bib", _llm_bib.count("@"))

            # Validate
            if "\\documentclass" not in tex_content:
                raise ValueError("LLM output missing \\documentclass")
            if "\\end{document}" not in tex_content:
                tex_content += "\n\n\\end{document}\n"
            tex_content, _dup_figures = _dedupe_latex_figures(tex_content)
            if _dup_figures:
                logger.info("LaTeX package: Removed %d repeated figure environment(s)", _dup_figures)

            logger.info("LaTeX package: LLM conversion successful (%d chars)", len(tex_content))
        except Exception as e:
            logger.warning("LaTeX package: LLM conversion failed (%s), using fallback", e)
            llm = None  # Fall through to fallback

    if llm is None:
        # Fallback: minimal wrapping
        _title_match = re.search(r"^#\s+(.+)$", paper_md, re.MULTILINE)
        _title = _title_match.group(1) if _title_match else "Research Paper"
        tex_content = (
            "\\documentclass{article}\n"
            "\\usepackage[preprint]{neurips_2025}\n"
            "\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n"
            "\\usepackage{hyperref}\n\\usepackage{url}\n"
            "\\usepackage{booktabs}\n\\usepackage{amsfonts}\n"
            "\\usepackage{amsmath}\n\\usepackage{graphicx}\n"
            "\\usepackage{natbib}\n\\usepackage{microtype}\n\n"
            f"\\title{{{_title}}}\n"
            "\\author{{Anonymous}}\n\n"
            "\\begin{{document}}\n\\maketitle\n\n"
            "% LLM conversion failed. Paper content needs manual conversion.\n"
            "% See paper_revised.md for the original markdown.\n\n"
            "\\bibliographystyle{{plainnat}}\n"
            "\\bibliography{{references}}\n\n"
            "\\end{{document}}\n"
        )

    (pkg_dir / "main.tex").write_text(tex_content, encoding="utf-8")
    (pkg_dir / "references.bib").write_text(_escape_bib_specials(bib_text), encoding="utf-8")

    # 4. Copy style files
    try:
        from researchclaw.templates.conference import NEURIPS_2025 as _tpl
        for _sty_file in _tpl.get_style_files():
            _shutil_lp.copy2(_sty_file, pkg_dir / _sty_file.name)
    except Exception:
        pass

    # 5. Create zip archive
    zip_path = stage_dir / "latex_package"
    _shutil_lp.make_archive(str(zip_path), "zip", str(pkg_dir))

    return ["latex_package.zip", "latex_package/"]


def _execute_quality_gate(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    revised = _read_prior_artifact(run_dir, "paper_revised.md") or ""
    report: dict[str, Any] | None = None

    # BUG-25: Load experiment summary for cross-checking
    _exp_summary_text = _read_prior_artifact(run_dir, "experiment_summary.json") or ""
    _exp_summary = _safe_json_loads(_exp_summary_text, {}) if _exp_summary_text else {}
    _exp_failed = False
    if isinstance(_exp_summary, dict):
        _best_run = _exp_summary.get("best_run", {})
        if isinstance(_best_run, dict):
            _exp_failed = (
                _best_run.get("status") == "failed"
                and not _best_run.get("metrics")
            )
        # Also check if metrics_summary is empty
        if not _exp_summary.get("metrics_summary"):
            _exp_failed = True

    if llm is not None:
        _pm = prompts or PromptManager()
        paper_for_eval = revised

        # BUG-25: Inject experiment status into quality gate prompt
        _exp_context = ""
        if _exp_summary and isinstance(_exp_summary, dict):
            _exp_status_keys = {
                k: _exp_summary.get(k) for k in (
                    "total_conditions", "total_metric_keys",
                    "metrics_summary",
                ) if _exp_summary.get(k) is not None
            }
            if _best_run := _exp_summary.get("best_run"):
                _exp_status_keys["best_run_status"] = (
                    _best_run.get("status") if isinstance(_best_run, dict) else str(_best_run)
                )
            _exp_context = (
                "\n\nExperiment summary (for cross-checking reported numbers):\n"
                + json.dumps(_exp_status_keys, indent=2, default=str)[:4000]
                + "\n\nCross-check: If the experiment status is 'failed' with "
                "empty metrics, any numerical results in tables constitute "
                "fabrication. Penalize severely.\n"
            )
        _exp_context += _execution_audit_section(run_dir, "Verified execution audit:")

        _overlay = _get_evolution_overlay(run_dir, "quality_gate")
        sp = _pm.for_stage(
            "quality_gate",
            evolution_overlay=_overlay,
            quality_threshold=str(config.research.quality_threshold),
            revised=paper_for_eval + _exp_context,
        )
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        parsed = _safe_json_loads(resp.content, {})
        if isinstance(parsed, dict):
            report = parsed
    # BUG-25: If experiment failed with no metrics, cap the quality score
    if report is not None and _exp_failed:
        _orig_score = report.get("score_1_to_10", 5)
        if isinstance(_orig_score, (int, float)) and _orig_score > 3:
            report["score_1_to_10"] = min(_orig_score, 3.0)
            report.setdefault("weaknesses", []).append(
                "Experiment failed with no metrics — any reported numerical "
                "results are unsupported and likely fabricated."
            )
            logger.warning(
                "BUG-25: Experiment failed — capping quality score from %.1f to 3.0",
                _orig_score,
            )
    if report is None:
        report = _default_quality_report(config.research.quality_threshold)
    report["review_scope"] = {
        "paper_characters": len(revised),
        "reviewed_characters": len(revised) if llm is not None else 0,
        "full_paper_supplied": llm is not None,
    }
    report.setdefault("generated", _utcnow_iso())
    (stage_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # T2.1: Enforce quality gate — fail if score below threshold
    score = report.get("score_1_to_10", 0)
    verdict = report.get("verdict", "proceed")
    threshold = config.research.quality_threshold or 5.0

    # --- Fabrication flag: collect real metrics for Stage 22 sanitization ---
    _fabrication_info: dict[str, Any] = {
        "experiment_failed": _exp_failed,
        "quality_score": score,
        "real_metric_values": [],
    }
    if isinstance(_exp_summary, dict):
        # Collect ALL real numeric values from experiment_summary.json
        _cond_summaries = _exp_summary.get("condition_summaries", {})
        if isinstance(_cond_summaries, dict):
            for cond_name, cond_data in _cond_summaries.items():
                if not isinstance(cond_data, dict):
                    continue
                cond_status = cond_data.get("status", "")
                if cond_status == "failed":
                    continue  # skip failed conditions
                for k, v in cond_data.items():
                    if isinstance(v, (int, float)) and k not in (
                        "seed_count", "total_steps", "training_steps",
                    ):
                        _fabrication_info["real_metric_values"].append(
                            round(float(v), 4)
                        )
        _ms = _exp_summary.get("metrics_summary", {})
        if isinstance(_ms, dict):
            for _mk, _mv in _ms.items():
                if isinstance(_mv, dict):
                    for _stat in ("mean", "min", "max"):
                        _sv = _mv.get(_stat)
                        if isinstance(_sv, (int, float)):
                            _fabrication_info["real_metric_values"].append(
                                round(float(_sv), 4)
                            )
    _fabrication_info["has_real_data"] = bool(
        _fabrication_info["real_metric_values"]
    )
    _fabrication_info["fabrication_suspected"] = (
        _exp_failed and not _fabrication_info["has_real_data"]
    )
    (stage_dir / "fabrication_flags.json").write_text(
        json.dumps(_fabrication_info, indent=2), encoding="utf-8"
    )

    if isinstance(score, (int, float)) and score < threshold:
        if config.research.graceful_degradation:
            logger.warning(
                "Quality gate DEGRADED: score %.1f < threshold %.1f — "
                "continuing as a draft requiring review (graceful_degradation=True)",
                score, threshold,
            )
            # Write degradation signal for downstream stages
            signal = {
                "score": score,
                "threshold": threshold,
                "verdict": verdict,
                "weaknesses": report.get("weaknesses", []),
                "generated": _utcnow_iso(),
            }
            (run_dir / "degradation_signal.json").write_text(
                json.dumps(signal, indent=2), encoding="utf-8"
            )
            return StageResult(
                stage=Stage.QUALITY_GATE,
                status=StageStatus.DONE,
                artifacts=("quality_report.json",),
                evidence_refs=("stage-20/quality_report.json",),
                decision="degraded",
            )
        logger.warning(
            "Quality gate FAILED: score %.1f < threshold %.1f (verdict=%s)",
            score, threshold, verdict,
        )
        return StageResult(
            stage=Stage.QUALITY_GATE,
            status=StageStatus.FAILED,
            artifacts=("quality_report.json", "fabrication_flags.json"),
            evidence_refs=("stage-20/quality_report.json",),
            error=f"Quality score {score:.1f}/10 below threshold {threshold:.1f}. "
                  f"Paper needs revision before export.",
        )

    logger.info(
        "Quality gate PASSED: score %.1f >= threshold %.1f",
        score, threshold,
    )
    return StageResult(
        stage=Stage.QUALITY_GATE,
        status=StageStatus.DONE,
        artifacts=("quality_report.json", "fabrication_flags.json"),
        evidence_refs=("stage-20/quality_report.json",),
    )


def _execute_knowledge_archive(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    revised = _read_prior_artifact(run_dir, "paper_revised.md") or ""
    analysis = _read_prior_artifact(run_dir, "analysis.md") or ""
    decision = _read_prior_artifact(run_dir, "decision.md") or ""
    preamble = _build_context_preamble(config, run_dir, include_goal=True)
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "knowledge_archive")
        sp = _pm.for_stage(
            "knowledge_archive",
            evolution_overlay=_overlay,
            preamble=preamble,
            decision=decision,
            analysis=analysis,
            revised=revised[:15000],
        )
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        archive = resp.content
    else:
        archive = f"""# Knowledge Archive

## Lessons Learned
- Preserve strict metric reporting protocol.
- Keep refinement logs aligned with code changes.

## Reproducibility
- Include exact experiment script and schedule.
- Capture run-level JSON metrics.

## Future Work
- Extend robustness and external validity checks.

Generated: {_utcnow_iso()}
"""
    (stage_dir / "archive.md").write_text(archive, encoding="utf-8")

    files: list[str] = []
    for stage_subdir in sorted(run_dir.glob("stage-*")):
        for artifact in sorted(stage_subdir.rglob("*")):
            if artifact.is_file() and artifact != (stage_dir / "bundle_index.json"):
                files.append(str(artifact.relative_to(run_dir)))
    index = {
        "run_id": run_dir.name,
        "generated": _utcnow_iso(),
        "artifact_count": len(files),
        "artifacts": files,
    }
    (stage_dir / "bundle_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    return StageResult(
        stage=Stage.KNOWLEDGE_ARCHIVE,
        status=StageStatus.DONE,
        artifacts=("archive.md", "bundle_index.json"),
        evidence_refs=("stage-21/archive.md", "stage-21/bundle_index.json"),
    )


def _sanitize_fabricated_data(
    paper: str,
    run_dir: Path,
) -> tuple[str, dict[str, Any]]:
    """Replace unverified numerical data in markdown tables with '---'.

    Loads experiment_summary.json as ground truth, extracts all verified
    metric values, then scans markdown tables in Results/Experiment sections.
    Numbers not matching any verified value (within 1% relative tolerance)
    are replaced with ``---``.

    Returns (sanitized_paper, sanitization_report).
    """
    import re as _re_san

    # --- 1. Build verified values set from experiment_summary.json ---
    verified_values: set[float] = set()
    exp_path = run_dir / "stage-14" / "experiment_summary.json"
    if not exp_path.exists():
        # Try other common locations
        for candidate in sorted(run_dir.glob("stage-14*/experiment_summary.json")):
            exp_path = candidate
            break

    if exp_path.exists():
        try:
            exp_data = json.loads(exp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            exp_data = {}

        def _collect_numbers(obj: Any, depth: int = 0) -> None:
            if depth > 10:
                return
            if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                verified_values.add(float(obj))
            elif isinstance(obj, dict):
                for v in obj.values():
                    _collect_numbers(v, depth + 1)
            elif isinstance(obj, list):
                for v in obj:
                    _collect_numbers(v, depth + 1)

        # Extract from well-known keys
        for key in (
            "metrics_summary", "condition_summaries", "best_run",
            "condition_metrics", "conditions", "ablation_results",
        ):
            if key in exp_data:
                _collect_numbers(exp_data[key])

    if not verified_values:
        report: dict[str, Any] = {
            "sanitized": False,
            "reason": "no verified values found in experiment_summary.json",
            "tables_processed": 0,
            "numbers_replaced": 0,
        }
        return paper, report

    def _is_verified(num: float) -> bool:
        """Check if num matches any verified value within 1% relative tolerance."""
        for v in verified_values:
            if v == 0.0:
                if abs(num) < 1e-9:
                    return True
            elif abs(num - v) / abs(v) <= 0.01:
                return True
        return False

    # --- 2. Find and sanitize markdown tables ---
    # Match markdown table blocks (header + separator + data rows)
    table_pat = _re_san.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)+"  # one or more pipe-delimited lines
        r")",
        _re_san.MULTILINE,
    )
    # Match numbers in table cells (integers, decimals, percentages, scientific)
    num_pat = _re_san.compile(
        r"(?<![a-zA-Z_])"  # not preceded by letter/underscore
        r"(-?\d+\.?\d*(?:[eE][+-]?\d+)?)"
        r"(%?)"  # optional percent
        r"(?![a-zA-Z_])"  # not followed by letter/underscore
    )

    numbers_replaced = 0
    numbers_kept = 0
    tables_processed = 0
    replaced_values: list[str] = []

    def _sanitize_table(match: _re_san.Match[str]) -> str:
        nonlocal numbers_replaced, numbers_kept, tables_processed
        table_text = match.group(0)
        lines = table_text.split("\n")

        # Check if this looks like a results/experiment table
        # (heuristic: has a separator row with dashes)
        has_separator = any(
            _re_san.match(r"^[ \t]*\|[\s:|-]+\|[ \t]*$", line)
            for line in lines
        )
        if not has_separator:
            return table_text

        tables_processed += 1
        sanitized_lines: list[str] = []
        for i, line in enumerate(lines):
            # Skip header row and separator row
            is_separator = bool(
                _re_san.match(r"^[ \t]*\|[\s:|-]+\|[ \t]*$", line)
            )
            is_header = i == 0  # first line is typically the header
            if is_separator or is_header:
                sanitized_lines.append(line)
                continue

            def _replace_num(m: _re_san.Match[str]) -> str:
                nonlocal numbers_replaced, numbers_kept
                num_str = m.group(1)
                pct = m.group(2)
                try:
                    val = float(num_str)
                except ValueError:
                    return m.group(0)
                if _is_verified(val):
                    numbers_kept += 1
                    return m.group(0)
                numbers_replaced += 1
                replaced_values.append(num_str + pct)
                return "---"

            sanitized_lines.append(num_pat.sub(_replace_num, line))
        return "\n".join(sanitized_lines)

    sanitized = table_pat.sub(_sanitize_table, paper)

    report = {
        "sanitized": numbers_replaced > 0,
        "tables_processed": tables_processed,
        "numbers_replaced": numbers_replaced,
        "numbers_kept": numbers_kept,
        "verified_values_count": len(verified_values),
        "replaced_samples": replaced_values[:20],
        "generated": _utcnow_iso(),
    }
    return sanitized, report


def _paper_citation_keys(paper: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite\w*\{([^}]+)\}|\[([^\]\n]+)\]", paper):
        inner = match.group(1) or match.group(2)
        parts = re.split(r"[,;]\s*", inner)
        if match.group(1) is not None or all(
            re.fullmatch(r"[A-Za-z][A-Za-z0-9_:-]*\d{4}[A-Za-z0-9_:-]*", part.strip())
            for part in parts
        ):
            keys.update(part.strip() for part in parts)
    return keys


_BIB_TEXT_FIELD_RE = re.compile(
    r"\b(title|booktitle|journal|publisher|series|note|howpublished|school|"
    r"institution|organization|address|edition|author|editor)\s*=\s*([{\"])",
    re.IGNORECASE,
)
_BIB_SPECIAL_RE = re.compile(r"(?<!\\)([&%#])")


def _escape_bib_specials(bib_text: str) -> str:
    """Escape bare ``&``, ``%`` and ``#`` in BibTeX text fields.

    A title like ``Detection & Localization`` makes bibtex/LaTeX fail
    ("Misplaced alignment tab character &").  Already-escaped characters
    are left alone, and url/doi/eprint fields (where these characters are
    legal) are not touched.
    """
    out: list[str] = []
    pos = 0
    for m in _BIB_TEXT_FIELD_RE.finditer(bib_text):
        if m.start() < pos:
            continue
        start = m.end()  # just after the opening delimiter
        depth, i = 1, start
        while i < len(bib_text) and depth:
            ch = bib_text[i]
            if ch == "\\":
                i += 2
                continue
            if m.group(2) == "{":
                depth += {"{": 1, "}": -1}.get(ch, 0)
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == '"' and depth == 1:
                depth = 0
            i += 1
        end = i - 1 if not depth else len(bib_text)
        out.append(bib_text[pos:start])
        out.append(_BIB_SPECIAL_RE.sub(r"\\\1", bib_text[start:end]))
        pos = end
    out.append(bib_text[pos:])
    return "".join(out)


def _bibliography_entries(bib_text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text):
        depth = 1
        position = match.end()
        while position < len(bib_text) and depth:
            character = bib_text[position]
            if position == 0 or bib_text[position - 1] != "\\":
                if character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
            position += 1
        if depth == 0:
            entries.setdefault(match.group(1), bib_text[match.start():position])
    return entries


def _export_bibliography(run_dir: Path, paper: str) -> str:
    from zipfile import BadZipFile, ZipFile

    sources: list[str] = []
    stages = sorted(
        (
            path for path in run_dir.glob("stage-*")
            if path.name[6:].isdigit() and int(path.name[6:]) < int(Stage.EXPORT_PUBLISH)
        ),
        reverse=True,
    )
    for stage in stages:
        for candidate in (stage / "references.bib", stage / "latex_package/references.bib"):
            if candidate.is_file():
                sources.append(candidate.read_text(encoding="utf-8"))
        archive = stage / "latex_package.zip"
        if archive.is_file():
            try:
                with ZipFile(archive) as package:
                    for name in package.namelist():
                        if name == "references.bib" or name.endswith("/references.bib"):
                            sources.append(package.read(name).decode("utf-8"))
            except (BadZipFile, OSError, UnicodeError):
                logger.warning("Could not read bibliography from %s", archive)
    entries: dict[str, str] = {}
    cited_keys = _paper_citation_keys(paper)
    for source in sources:
        first_bibliography = not entries
        for key, entry in _bibliography_entries(source).items():
            if first_bibliography or key in cited_keys:
                entries.setdefault(key, entry)
    # Keys the writer cited from its own reference list (e.g. when literature
    # search fell back to an unrelated library) would print as [?].
    body = re.split(r"^#{1,3}[ 	]*(?:References|Bibliography)[ 	]*$", paper,
                    maxsplit=1, flags=re.MULTILINE | re.IGNORECASE)[0]
    body_keys = _paper_citation_keys(body)
    for key, entry in _reference_list_entries(paper).items():
        if key in body_keys:
            entries.setdefault(key, entry)
    return "\n\n".join(entries.values()) + "\n" if entries else ""


_REFERENCE_ITEM_RE = re.compile(
    r"^[ \t]*(?:[-*][ \t]+)?\[([A-Za-z][A-Za-z0-9_:-]*\d{4}[A-Za-z0-9_:-]*)\][ \t]+(.+)$",
    re.MULTILINE,
)


def _reference_list_entries(paper: str) -> dict[str, str]:
    """BibTeX for ``[key] Authors. Title. Venue, Year.`` lines under References."""
    heading = re.search(r"^#{1,3}[ \t]*(?:References|Bibliography)[ \t]*$", paper, re.MULTILINE | re.IGNORECASE)
    if not heading:
        return {}
    entries: dict[str, str] = {}
    for match in _REFERENCE_ITEM_RE.finditer(paper[heading.end():]):
        key, text = match.group(1), match.group(2).strip()
        fields = {"note": text}
        year = re.search(r"(?:19|20)\d{2}(?=\D*$)", text)
        # Authors, title and venue are separated by ". " not following an initial.
        parts = re.split(r"(?<![A-Z])\.\s+", text, maxsplit=2)
        if year and len(parts) == 3:
            fields = {
                "author": parts[0].replace(", and ", " and ").replace(", ", " and ")
                .replace(" et al", " and others"),
                "title": parts[1],
                "howpublished": re.sub(r",?\s*(?:19|20)\d{2}\.?$", "", parts[2]).rstrip("."),
            }
        if year:
            fields["year"] = year.group(0)
        body = ",\n".join(
            "  {} = {{{}}}".format(
                name, re.sub(r"(?<!\\)([&%#_])", r"\\\1", re.sub(r"[{}]", "", value)),
            )
            for name, value in fields.items()
        )
        entries[key] = f"@misc{{{key},\n{body}\n}}"
    return entries


def _execute_export_publish(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    revised = _read_prior_artifact(run_dir, "paper_revised.md") or ""
    final_paper = revised
    if not final_paper.strip():
        final_paper = "# Final Paper\n\nNo content generated."

    # --- Graceful degradation: sanitize fabricated data ---
    _degradation_signal_path = run_dir / "degradation_signal.json"
    if _degradation_signal_path.exists():
        try:
            _deg_signal = json.loads(
                _degradation_signal_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            _deg_signal = {}

        # Back up pre-sanitized version
        (stage_dir / "paper_presanitized.md").write_text(
            final_paper, encoding="utf-8"
        )

        _san_report = {
            "sanitized": False,
            "numbers_replaced": 0,
            "reason": "Original numerical claims preserved for independent review.",
            "requires_review": True,
        }
        (stage_dir / "sanitization_report.json").write_text(
            json.dumps(_san_report, indent=2), encoding="utf-8"
        )

        # Insert degradation notice after abstract
        _deg_score = _deg_signal.get("score", "N/A")
        _deg_threshold = _deg_signal.get("threshold", "N/A")
        _deg_notice = (
            "\n\n> **Note:** This paper was produced in degraded mode. "
            f"Quality gate score ({_deg_score}/{_deg_threshold}) was below "
            "threshold. This is a draft requiring revision. Original numerical "
            "claims have been preserved and require independent verification; "
            "no numbers have been replaced.\n\n"
        )
        # Try to insert after ## Abstract section
        _abstract_markers = ["## Abstract\n", "# Abstract\n"]
        _notice_inserted = False
        for _marker in _abstract_markers:
            if _marker in final_paper:
                _marker_end = final_paper.index(_marker) + len(_marker)
                # Find the end of the abstract paragraph
                _next_section = final_paper.find("\n## ", _marker_end)
                _next_heading = final_paper.find("\n# ", _marker_end)
                _insert_pos = min(
                    p for p in (_next_section, _next_heading)
                    if p > 0
                ) if any(p > 0 for p in (_next_section, _next_heading)) else len(final_paper)
                final_paper = (
                    final_paper[:_insert_pos]
                    + _deg_notice
                    + final_paper[_insert_pos:]
                )
                _notice_inserted = True
                break
        if not _notice_inserted:
            # Fallback: prepend to paper
            final_paper = _deg_notice + final_paper

        logger.info(
            "Stage 25: Preserved degraded draft — "
            "%d numbers replaced, %d kept",
            _san_report.get("numbers_replaced", 0),
            _san_report.get("numbers_kept", 0),
        )

    # IMP-19 Layer 2: Ensure at least figures are referenced in the paper
    import re as _re_fig
    chart_files = []
    for _chart_src_dir in [
        stage_dir / "charts",
        run_dir / "stage-16" / "charts",
        run_dir / "stage-14" / "charts",
    ]:
        if _chart_src_dir.is_dir():
            chart_files.extend(sorted(_chart_src_dir.glob("*.png")))
    if chart_files and "![" not in final_paper:
        # Distribute figures to relevant sections based on filename keywords
        _fig_placement: dict[str, list[str]] = {
            "method": [],       # architecture, method, model, pipeline diagrams
            "result": [],       # experiment, comparison, ablation charts
            "intro": [],        # concept, overview, illustration
        }
        _fig_counter = 0
        for cf in chart_files[:6]:
            _fig_counter += 1
            stem_lower = cf.stem.lower()
            label = cf.stem.replace("_", " ").title()
            fig_md = f"![Figure {_fig_counter}: {label}](charts/{cf.name})"
            if any(k in stem_lower for k in ("architecture", "model", "pipeline", "method", "flowchart")):
                _fig_placement["method"].append(fig_md)
            elif any(k in stem_lower for k in ("experiment", "comparison", "ablation", "result", "metric")):
                _fig_placement["result"].append(fig_md)
            elif any(k in stem_lower for k in ("concept", "overview", "illustration", "threat", "attack")):
                _fig_placement["intro"].append(fig_md)
            else:
                _fig_placement["result"].append(fig_md)  # default to results

        # Insert figures at relevant section boundaries
        _section_markers = {
            "method": ["## Method", "## Methodology", "## Approach", "## Framework",
                        "## 3. Method", "## 3 Method"],
            "result": ["## Results", "## Experiments", "## Evaluation",
                        "## 5. Results", "## 4. Experiments", "## 5 Results"],
            "intro": ["## Related Work", "## Background", "## 2. Related",
                       "## 2 Related Work"],
        }
        _total_inserted = 0
        for category, figs in _fig_placement.items():
            if not figs:
                continue
            fig_block = "\n\n" + "\n\n".join(figs) + "\n\n"
            inserted = False
            for marker in _section_markers.get(category, []):
                if marker in final_paper:
                    # Insert BEFORE the marker section (so figure appears at end of previous section)
                    final_paper = final_paper.replace(marker, fig_block + marker, 1)
                    inserted = True
                    _total_inserted += len(figs)
                    break
            if not inserted:
                # Fallback: insert before Conclusion/Limitations/Discussion
                for fallback in ["## Conclusion", "## Limitations", "## Discussion"]:
                    if fallback in final_paper:
                        final_paper = final_paper.replace(fallback, fig_block + fallback, 1)
                        inserted = True
                        _total_inserted += len(figs)
                        break
            if not inserted:
                final_paper += fig_block
                _total_inserted += len(figs)

        logger.info(
            "IMP-19: Injected %d figure references into paper_final.md (distributed across sections)",
            _total_inserted,
        )

    # IMP-24: Detect excessive number repetition
    _numbers_found = _re_fig.findall(r"\b\d+\.\d{2,}\b", final_paper)
    from collections import Counter as _Counter
    _num_counts = _Counter(_numbers_found)
    _repeated = {n: c for n, c in _num_counts.items() if c > 3}
    if _repeated:
        logger.warning(
            "IMP-24: Numbers repeated >3 times: %s",
            _repeated,
        )

    # WS-5.4: Generate result visualizations
    # Priority: FigureAgent charts (stage-16 RESULT_ANALYSIS) > stage-14 legacy > fallback
    try:
        chart_dir = stage_dir / "charts"
        chart_dir.mkdir(parents=True, exist_ok=True)
        # A resumed S25 reuses this dir; charts of the earlier attempt are stale.
        for _stale in chart_dir.glob("fig_*.png"):
            _stale.unlink()
        charts: list[Path] = []

        # Check if FigureAgent produced charts in stage-16 or stage-14
        _fa_charts_found = False
        # Only the current round: an older stage-16_vN holds superseded charts.
        for _fa_stage in _current_stage_dirs(run_dir, "stage-16", "stage-14"):
            if _fa_charts_found:
                break
            for _fa_dir in [_fa_stage / "charts"]:
                _fa_pngs = sorted(_fa_dir.glob("fig_*.png"))
                if _fa_pngs:
                    import shutil
                    for _fa_png in _fa_pngs:
                        dest = chart_dir / _fa_png.name
                        shutil.copy2(_fa_png, dest)
                        charts.append(dest)
                    _fa_charts_found = True
                    logger.info(
                        "Stage 22: Copied %d FigureAgent charts from %s",
                        len(_fa_pngs), _fa_dir,
                    )
                    break

        # Always generate structured charts from visualize.py (different names)
        from researchclaw.experiment.visualize import generate_all_charts
        _metric_dir = getattr(config.experiment, "metric_direction", "minimize")
        _viz_charts = generate_all_charts(
            run_dir,
            chart_dir,
            metric_key=config.experiment.metric_key,
            metric_direction=_metric_dir,
        )
        charts.extend(_viz_charts)

        if charts:
            logger.info("Stage 22: Generated %d chart(s) total", len(charts))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chart generation failed: %s", exc)

    # FIGURE_PROMPT blocks are instructions for the image model; turn them
    # into real figures (rendered or fallback) copied into charts/.
    final_paper, _prompt_figures = _figure_prompts_to_markdown(
        final_paper, run_dir, stage_dir / "charts"
    )
    if _prompt_figures:
        logger.info(
            "Stage 25: Converted %d FIGURE_PROMPT block(s) into figures: %s",
            len(_prompt_figures), _prompt_figures,
        )

    final_paper, _dup_figures = _dedupe_markdown_figures(final_paper)
    if _dup_figures:
        logger.info("Stage 25: Removed %d repeated figure reference(s)", _dup_figures)
    (stage_dir / "paper_final.md").write_text(final_paper, encoding="utf-8")

    # Initialize artifacts list
    artifacts = ["paper_final.md"]
    if any((stage_dir / "charts").glob("*.png")):
        artifacts.append("charts/")
    # F2.7: Post-process citations — [cite_key] → \cite{cite_key}
    # and copy final references.bib to export stage
    _ay_map: dict[str, str] = {}  # BUG-102: author-year → cite_key map
    bib_text = _export_bibliography(run_dir, final_paper)
    final_paper_latex = final_paper
    if bib_text:
        # Replace [cite_key] patterns in the final paper with \cite{cite_key}
        # Collect all valid cite_keys from the bib file
        import re as _re

        valid_keys = set(_re.findall(r"@\w+\{([^,]+),", bib_text))

        # BUG-102: Recover author-year citations → [cite_key] format.
        # When Stage 19 (paper_revision) converts [cite_key] to [Author et al., 2024],
        # the downstream regex can't match them. Build a reverse map from bib entries.
        def _build_author_year_map(bib: str, keys: set[str]) -> dict[str, str]:
            """Build mapping from author-year patterns to cite_keys.

            Returns dict like:
              "Raissi et al., 2019" → "raissi2019physicsinformed"
              "Tavella and Randall, 2000" → "tavella2000pricing"
            """
            mapping: dict[str, str] = {}
            # Parse each bib entry for author + year
            entry_pat = _re.compile(
                r"@\w+\{([^,]+),\s*(.*?)\n\}", _re.DOTALL
            )
            for m in entry_pat.finditer(bib):
                key = m.group(1).strip()
                if key not in keys:
                    continue
                body = m.group(2)
                # Extract author field
                author_m = _re.search(
                    r"author\s*=\s*[\{\"](.*?)[\}\"]", body, _re.IGNORECASE
                )
                year_m = _re.search(
                    r"year\s*=\s*[\{\"]?(\d{4})[\}\"]?", body, _re.IGNORECASE
                )
                if not author_m or not year_m:
                    continue
                author_raw = author_m.group(1).strip()
                year = year_m.group(1)
                # Parse author names (split on " and ")
                authors = [a.strip() for a in _re.split(r"\s+and\s+", author_raw)]
                # Extract last names
                last_names = []
                for a in authors:
                    if "," in a:
                        last_names.append(a.split(",")[0].strip())
                    else:
                        parts = a.split()
                        last_names.append(parts[-1] if parts else a)
                if not last_names:
                    continue
                # Generate author-year patterns:
                # 1 author: "Smith, 2024"
                # 2 authors: "Smith and Jones, 2024"
                # 3+ authors: "Smith et al., 2024"
                if len(last_names) == 1:
                    patterns = [f"{last_names[0]}, {year}"]
                elif len(last_names) == 2:
                    patterns = [
                        f"{last_names[0]} and {last_names[1]}, {year}",
                        f"{last_names[0]} \\& {last_names[1]}, {year}",
                    ]
                else:
                    patterns = [
                        f"{last_names[0]} et al., {year}",
                        f"{last_names[0]} et al. {year}",
                    ]
                    # Also add "Smith and Jones, 2024" for first two authors
                    patterns.append(
                        f"{last_names[0]} and {last_names[1]}, {year}"
                    )
                for pat in patterns:
                    mapping[pat] = key
            return mapping

        _ay_map = _build_author_year_map(bib_text, valid_keys)
        if _ay_map:
            # Count how many author-year citations exist in the paper
            _ay_found = 0
            for _ay_pat in _ay_map:
                if _ay_pat in final_paper:
                    _ay_found += 1
            if _ay_found > 0:
                logger.info(
                    "Stage 22: Found %d author-year citation patterns — "
                    "converting back to [cite_key] format.",
                    _ay_found,
                )
                # Sort by longest pattern first to avoid partial matches
                for _ay_pat in sorted(_ay_map, key=len, reverse=True):
                    _ay_key = _ay_map[_ay_pat]
                    # Match [Author et al., 2024] or [Author and Jones, 2024; ...]
                    # Handle single-citation brackets
                    final_paper = final_paper.replace(
                        f"[{_ay_pat}]", f"[{_ay_key}]"
                    )
                    # Handle within multi-citation brackets [A et al., 2020; B et al., 2021]
                    # Replace the author-year segment inside brackets
                    final_paper = final_paper.replace(_ay_pat, _ay_key)
                # Fix multi-key brackets: [key1; key2] → [key1, key2]
                # (author-year uses semicolons, cite-keys use commas)
                def _fix_semicolon_cites(m_sc: _re.Match[str]) -> str:
                    inner = m_sc.group(1)
                    # Only convert if ALL segments look like cite keys
                    parts = [p.strip() for p in inner.split(";")]
                    _ck = r"[a-zA-Z][a-zA-Z0-9_-]*\d{4}[a-zA-Z0-9_]*"
                    if all(_re.fullmatch(_ck, p) for p in parts):
                        return "[" + ", ".join(parts) + "]"
                    return m_sc.group(0)
                final_paper = _re.sub(
                    r"\[([^\]]+;[^\]]+)\]", _fix_semicolon_cites, final_paper
                )
                (stage_dir / "paper_final.md").write_text(
                    final_paper, encoding="utf-8"
                )

        # R10-Fix4: Citation cross-validation
        cited_keys_in_paper = _paper_citation_keys(final_paper)
        if valid_keys and cited_keys_in_paper:
            invalid_keys = cited_keys_in_paper - valid_keys
            if invalid_keys:
                logger.warning(
                    "Stage 22: Found %d citation keys in paper not in references.bib: %s",
                    len(invalid_keys),
                    ", ".join(sorted(invalid_keys)[:20]),
                )
                (stage_dir / "invalid_citations.json").write_text(
                    json.dumps(sorted(invalid_keys), indent=2), encoding="utf-8"
                )
                artifacts.append("invalid_citations.json")

        final_paper_latex = final_paper  # default: no citation conversion
        if valid_keys:
            _CITE_KEY_PAT = r"[a-zA-Z][a-zA-Z0-9_-]*\d{4}[a-zA-Z0-9]*"

            # Step 1: Convert multi-key brackets [key1, key2] → \cite{key1, key2}
            def _replace_multi_cite(m: _re.Match[str]) -> str:
                keys = [k.strip() for k in m.group(1).split(",")]
                matched = [k for k in keys if k in valid_keys]
                if len(matched) == len(keys):
                    return "\\cite{" + ", ".join(matched) + "}"
                return m.group(0)

            final_paper_latex = _re.sub(
                rf"\[({_CITE_KEY_PAT}(?:\s*,\s*{_CITE_KEY_PAT})+)\]",
                _replace_multi_cite,
                final_paper,
            )

            # Step 2: Convert single-key brackets [key] → \cite{key}
            def _replace_cite(m: _re.Match[str]) -> str:
                key = m.group(1)
                if key in valid_keys:
                    return f"\\cite{{{key}}}"
                return m.group(0)

            final_paper_latex = _re.sub(
                rf"\[({_CITE_KEY_PAT})\]", _replace_cite, final_paper_latex
            )

            # Step 3: Merge adjacent \cite{a} \cite{b} → \cite{a, b}
            def _merge_adjacent_cites(m: _re.Match[str]) -> str:
                keys = _re.findall(r"\\cite\{([^}]+)\}", m.group(0))
                return "\\cite{" + ", ".join(keys) + "}"

            final_paper_latex = _re.sub(
                r"\\cite\{[^}]+\}(?:\s*\\cite\{[^}]+\})+",
                _merge_adjacent_cites,
                final_paper_latex,
            )

            (stage_dir / "paper_final_latex.md").write_text(
                final_paper_latex, encoding="utf-8"
            )
            artifacts.append("paper_final_latex.md")
        # Write final references.bib
        (stage_dir / "references.bib").write_text(
            _escape_bib_specials(bib_text), encoding="utf-8"
        )
        artifacts.append("references.bib")
        logger.info(
            "Stage 22: Exported references.bib with %d entries",
            len(valid_keys) if valid_keys else 0,
        )

    # Conference template: generate .tex file
    try:
        from researchclaw.templates import get_template, markdown_to_latex

        tpl = get_template(config.export.target_conference)
        # Use the latex-citation-processed version if available
        tex_source = final_paper_latex
        # Append NeurIPS-style checklist if target is a ML conference
        if tpl.name in ("neurips_2024", "neurips_2025", "icml_2025", "icml_2026",
                         "iclr_2025", "iclr_2026"):
            _has_exp = bool(_read_prior_artifact(run_dir, "experiment_summary.json"))
            _checklist = _generate_neurips_checklist(
                has_experiments=_has_exp,
                has_code=True,
            )
            if "NeurIPS Paper Checklist" not in tex_source:
                tex_source = tex_source.rstrip() + "\n\n" + _checklist
        tex_content = markdown_to_latex(
            tex_source,
            tpl,
            title=_extract_paper_title(tex_source),
            authors=config.export.authors,
            bib_file=config.export.bib_file,
            bib_entries=_ay_map or None,
        )
        (stage_dir / "paper.tex").write_text(tex_content, encoding="utf-8")
        artifacts.append("paper.tex")
        logger.info(
            "Stage 22: Generated paper.tex for %s (%d chars)",
            tpl.display_name,
            len(tex_content),
        )
        # Copy bundled style files alongside paper.tex
        for sf in tpl.get_style_files():
            import shutil as _shutil_sty
            _shutil_sty.copy2(sf, stage_dir / sf.name)
        # Compile verification
        try:
            from researchclaw.templates.compiler import compile_latex
            _compile_result = compile_latex(stage_dir / "paper.tex", max_attempts=2)
            if _compile_result.success:
                logger.info("Stage 22: LaTeX compilation verification PASSED")
                artifacts.append("paper.pdf")
                # PDF-as-reviewer: LLM-based visual review of compiled PDF
                _pdf_path = stage_dir / "paper.pdf"
                if _pdf_path.exists() and llm is not None:
                    try:
                        _pdf_review = _review_compiled_pdf(
                            _pdf_path, llm, config.research.topic
                        )
                        if _pdf_review:
                            (stage_dir / "pdf_review.json").write_text(
                                json.dumps(_pdf_review, indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )
                            artifacts.append("pdf_review.json")
                            _pdf_score = _pdf_review.get("overall_score", 0)
                            if _pdf_score < 5:
                                logger.warning(
                                    "Stage 22: PDF visual review score %d/10 — %s",
                                    _pdf_score,
                                    _pdf_review.get("summary", ""),
                                )
                            else:
                                logger.info(
                                    "Stage 22: PDF visual review score %d/10",
                                    _pdf_score,
                                )
                    except Exception as _pdf_exc:  # noqa: BLE001
                        logger.debug("Stage 22: PDF review skipped: %s", _pdf_exc)
                # Post-compilation quality checks
                try:
                    from researchclaw.templates.compiler import check_compiled_quality
                    _qc = check_compiled_quality(stage_dir / "paper.tex")
                    if _qc.warnings_summary:
                        logger.warning(
                            "Stage 22: Quality checks: %s",
                            "; ".join(_qc.warnings_summary),
                        )
                    (stage_dir / "compilation_quality.json").write_text(
                        json.dumps({
                            "page_count": _qc.page_count,
                            "unresolved_refs": _qc.unresolved_refs,
                            "unresolved_cites": _qc.unresolved_cites,
                            "overfull_hboxes": len(_qc.overfull_hboxes),
                            "orphan_figures": _qc.orphan_figures,
                            "orphan_labels": _qc.orphan_labels,
                            "warnings": _qc.warnings_summary,
                        }, indent=2),
                        encoding="utf-8",
                    )
                    artifacts.append("compilation_quality.json")
                    # BUG-27: Warn if page count exceeds limit
                    _page_limit = 10
                    if _qc.page_count and _qc.page_count > _page_limit:
                        logger.warning(
                            "BUG-27: Paper is %d pages (limit %d). "
                            "Consider tightening content in revision.",
                            _qc.page_count, _page_limit,
                        )
                except Exception as _qc_exc:  # noqa: BLE001
                    logger.debug("Stage 22: Quality checks skipped: %s", _qc_exc)
            else:
                logger.warning("Stage 22: LaTeX compilation verification FAILED: %s", _compile_result.errors[:3])
                # Add compilation failure comment to .tex
                _tex_path = stage_dir / "paper.tex"
                if _tex_path.exists():
                    _tex_content = _tex_path.read_text(encoding="utf-8")
                    if "% WARNING: Compilation failed" not in _tex_content:
                        _tex_content = (
                            "% WARNING: Compilation failed. Errors:\n"
                            + "".join(f"% {e}\n" for e in _compile_result.errors[:5])
                            + _tex_content
                        )
                        _tex_path.write_text(_tex_content, encoding="utf-8")
        except Exception as _compile_exc:  # noqa: BLE001
            logger.debug("Stage 22: Compile verification skipped: %s", _compile_exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LaTeX generation skipped: %s", exc)

    # BUG-99: Validate that \includegraphics paths in .tex match actual files.
    # The LLM may write paper referencing a chart name that differs from the
    # actual generated filename (e.g. "performance_comparison.png" vs
    # "experiment_comparison.png").
    try:
        tex_path = stage_dir / "paper.tex"
        if tex_path.exists():
            tex_text = tex_path.read_text(encoding="utf-8")
            # Extract all \includegraphics{path} references
            _fig_refs = re.findall(
                r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_text
            )
            if _fig_refs:
                # Collect all actual chart files in stage_dir/charts/
                _chart_dir = stage_dir / "charts"
                _actual_files: dict[str, str] = {}  # lowercase stem → relative path
                if _chart_dir.is_dir():
                    for _af in _chart_dir.iterdir():
                        if _af.is_file() and _af.suffix.lower() in (
                            ".png", ".jpg", ".jpeg", ".pdf", ".svg",
                        ):
                            _actual_files[_af.stem.lower()] = f"charts/{_af.name}"
                            _actual_files[_af.name.lower()] = f"charts/{_af.name}"

                _fixes: dict[str, str] = {}
                for _ref in _fig_refs:
                    _ref_path = stage_dir / _ref
                    if _ref_path.exists():
                        continue  # File exists, no fix needed
                    # Try fuzzy matching against actual chart files
                    _ref_stem = Path(_ref).stem.lower()
                    _ref_name = Path(_ref).name.lower()
                    # Exact stem match (different extension or directory)
                    if _ref_stem in _actual_files:
                        _fixes[_ref] = _actual_files[_ref_stem]
                        continue
                    if _ref_name in _actual_files:
                        _fixes[_ref] = _actual_files[_ref_name]
                        continue
                    # Fuzzy match: find best match by keyword overlap
                    if _actual_files:
                        _ref_words = set(_ref_stem.replace("-", "_").split("_"))
                        _best_match, _best_overlap = "", 0
                        for _stem, _apath in _actual_files.items():
                            _a_words = set(
                                _stem.replace("-", "_").split("_")
                            )
                            _overlap = len(_ref_words & _a_words)
                            if _overlap > _best_overlap:
                                _best_overlap = _overlap
                                _best_match = _apath
                        if _best_overlap >= 1 and _best_match:
                            _fixes[_ref] = _best_match

                if _fixes:
                    for _old_path, _new_path in _fixes.items():
                        tex_text = tex_text.replace(
                            f"{{{_old_path}}}", f"{{{_new_path}}}"
                        )
                    tex_path.write_text(tex_text, encoding="utf-8")
                    logger.warning(
                        "BUG-99: Fixed %d figure path mismatch(es) in paper.tex: %s",
                        len(_fixes),
                        ", ".join(f"{k} → {v}" for k, v in _fixes.items()),
                    )
                # Warn about any remaining unresolved references
                _still_missing = [
                    r for r in _fig_refs
                    if r not in _fixes and not (stage_dir / r).exists()
                ]
                if _still_missing:
                    logger.warning(
                        "Stage 22: %d figure reference(s) have no matching file: %s",
                        len(_still_missing), _still_missing[:5],
                    )
    except Exception as _fig_exc:  # noqa: BLE001
        logger.debug("Stage 22: Figure path validation skipped: %s", _fig_exc)

    # --- Code packaging: multi-file directory or single file ---
    # Without ITERATIVE_REFINE output (skipped or stopped S15), package the
    # code that S14 actually ran from experiment/.
    exp_final_dir_path = None
    for _code_dir_name in ("experiment_final/", "experiment/"):
        _candidate = _read_prior_artifact(run_dir, _code_dir_name)
        if _candidate and Path(_candidate).is_dir():
            exp_final_dir_path = _candidate
            break
    if exp_final_dir_path:
        import ast

        code_dir = stage_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        all_code_combined = ""
        code_file_names: list[str] = []
        exp_root = Path(exp_final_dir_path)
        for src in sorted(exp_root.rglob("*.py")):
            rel = src.relative_to(exp_root)
            # source/ is copied below; outputs/ and backups are not project code
            if rel.parts[0] in _CODE_PACKAGE_SKIP_DIRS or any(
                part in _SOURCE_SKIP_DIRS for part in rel.parts[:-1]
            ):
                continue
            target = code_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(src.read_bytes())
            all_code_combined += src.read_text(encoding="utf-8", errors="replace") + "\n"
            code_file_names.append(rel.as_posix())

        # Detect dependencies from all files
        detected: set[str] = set()
        known_packages = {
            "numpy": "numpy",
            "torch": "torch",
            "tensorflow": "tensorflow",
            "sklearn": "scikit-learn",
            "scikit-learn": "scikit-learn",
            "scipy": "scipy",
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
            "transformers": "transformers",
            "datasets": "datasets",
            "jax": "jax",
        }
        try:
            tree = ast.parse(all_code_combined)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in known_packages:
                            detected.add(known_packages[top])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    if top in known_packages:
                        detected.add(known_packages[top])
        except SyntaxError:
            pass

        requirements = sorted(detected)
        (code_dir / "requirements.txt").write_text(
            "\n".join(requirements) + ("\n" if requirements else ""),
            encoding="utf-8",
        )

        source_summary = _copy_experiment_source(Path(exp_final_dir_path), code_dir)
        paper_title = _extract_paper_title(final_paper)
        file_list_md = "\n".join(f"- `{f}`" for f in code_file_names)
        readme = (
            f"# Code Package for {paper_title}\n\n"
            "## Description\n"
            "This directory contains the experiment project used for the paper.\n\n"
            "## Project Files\n"
            f"{file_list_md}\n\n"
            f"{_source_readme_section(source_summary)}"
            "## How to Run\n"
            "`python main.py`\n\n"
            "## Dependencies\n"
            "Install dependencies with `pip install -r requirements.txt` if needed.\n"
        )
        (code_dir / "README.md").write_text(readme, encoding="utf-8")
        artifacts.append("code/")
        logger.info(
            "Stage 22: Packaged multi-file code release (%d files, %d deps)",
            len(code_file_names),
            len(requirements),
        )
    else:
        # Backward compat: single-file packaging
        code_payload = _read_prior_artifact(run_dir, "experiment_final.py")
        if not code_payload:
            code_payload = _read_prior_artifact(run_dir, "experiment.py")
        if code_payload:
            import ast

            code_dir = stage_dir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "experiment.py").write_text(code_payload, encoding="utf-8")

            detected_single: set[str] = set()
            known_packages_single = {
                "numpy": "numpy",
                "torch": "torch",
                "tensorflow": "tensorflow",
                "sklearn": "scikit-learn",
                "scikit-learn": "scikit-learn",
                "scipy": "scipy",
                "pandas": "pandas",
                "matplotlib": "matplotlib",
                "seaborn": "seaborn",
                "transformers": "transformers",
                "datasets": "datasets",
                "jax": "jax",
            }
            try:
                tree = ast.parse(code_payload)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            top = alias.name.split(".")[0]
                            if top in known_packages_single:
                                detected_single.add(known_packages_single[top])
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        top = node.module.split(".")[0]
                        if top in known_packages_single:
                            detected_single.add(known_packages_single[top])
            except SyntaxError:
                pass

            requirements = sorted(detected_single)
            (code_dir / "requirements.txt").write_text(
                "\n".join(requirements) + ("\n" if requirements else ""),
                encoding="utf-8",
            )
            paper_title = _extract_paper_title(final_paper)
            readme = (
                f"# Code Package for {paper_title}\n\n"
                "## Description\n"
                "This directory contains the final experiment script used for the paper.\n\n"
                "## How to Run\n"
                "`python experiment.py`\n\n"
                "## Dependencies\n"
                "Install dependencies with `pip install -r requirements.txt` if needed.\n"
            )
            (code_dir / "README.md").write_text(readme, encoding="utf-8")
            artifacts.append("code/")
            logger.info(
                "Stage 22: Packaged single-file code release with %d deps",
                len(requirements),
            )
    # WS-5.5: Generate framework diagram prompt for methodology section
    try:
        _framework_prompt = _generate_framework_diagram_prompt(
            final_paper, config, llm=llm
        )
        if _framework_prompt:
            _chart_dir = stage_dir / "charts"
            _chart_dir.mkdir(parents=True, exist_ok=True)
            (_chart_dir / "framework_diagram_prompt.md").write_text(
                _framework_prompt, encoding="utf-8"
            )
            logger.info("Stage 22: Generated framework diagram prompt → charts/framework_diagram_prompt.md")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Stage 22: Framework diagram prompt generation skipped: %s", exc)

    return StageResult(
        stage=Stage.EXPORT_PUBLISH,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-22/{a}" for a in artifacts),
    )


_SOURCE_SKIP_DIRS = {"ckpt", "pretrained", "__pycache__", ".git"}
_SOURCE_SKIP_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors", ".bin", ".pyc"}
_SOURCE_MAX_BYTES = 20 * 1024 * 1024
_CODE_DIFFS = ("model_diff.patch", "full_code_diff.patch")
_CODE_PACKAGE_SKIP_DIRS = {"source", "outputs", "repair_backup", "codebases", "datasets", "checkpoints"}


def _copy_experiment_source(exp_final: Path, code_dir: Path) -> dict[str, list[str]]:
    """Copy the modified codebase copy (``source/``) and its diffs into *code_dir*.

    ``main.py`` imports the model from ``source/``, so without it the package
    hides every network change.  Weights and files above 20 MB are skipped.
    Returns the new, modified and skipped relative paths.
    """
    import shutil

    summary: dict[str, list[str]] = {"new": [], "modified": [], "skipped": [], "diffs": []}
    source = exp_final / "source"
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(source)
            if (
                any(part in _SOURCE_SKIP_DIRS for part in rel.parts[:-1])
                or path.suffix.lower() in _SOURCE_SKIP_SUFFIXES
                or path.stat().st_size > _SOURCE_MAX_BYTES
            ):
                summary["skipped"].append(rel.as_posix())
                continue
            target = code_dir / "source" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for name in _CODE_DIFFS:
        patch = exp_final / "outputs" / name
        if patch.is_file():
            shutil.copy2(patch, code_dir / name)
            summary["diffs"].append(name)

    original: set[str] = set()
    try:
        manifest = json.loads((source / "_manifest.json").read_text(encoding="utf-8"))
        original = {str(p) for p in manifest.get("file_tree", [])}
    except (OSError, ValueError, AttributeError):
        pass
    patch = next((code_dir / n for n in reversed(_CODE_DIFFS) if (code_dir / n).is_file()), None)
    if patch is not None:
        def strip(name: str) -> str:
            name = name.split("\t")[0].strip()
            for prefix in ("original/", "source/", "a/", "b/"):
                if name.startswith(prefix):
                    return name[len(prefix):]
            return name

        old = None
        for line in patch.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("--- "):
                old = line[4:]
            elif line.startswith("+++ ") and old is not None:
                new, before = strip(line[4:]), strip(old)
                is_new = before == "/dev/null" or (original and before not in original)
                bucket = summary["new" if is_new else "modified"]
                if new not in bucket:
                    bucket.append(new)
                old = None
    return summary


def _source_readme_section(summary: dict[str, list[str]]) -> str:
    if not any(summary.get(k) for k in ("new", "modified", "diffs")):
        return ""
    lines = [
        "## Modified Codebase",
        "`source/` is the experiment's copy of the original codebase, including",
        "every model change; `main.py` imports the model from it.",
        "",
    ]
    for key, label in (("new", "New files"), ("modified", "Modified files")):
        if summary.get(key):
            lines += [f"{label}:"] + [f"- `{p}`" for p in summary[key]] + [""]
    if summary.get("diffs"):
        lines += ["Diffs against the original codebase:"]
        lines += [f"- `{p}`" for p in summary["diffs"]] + [""]
    if summary.get("skipped"):
        lines += [
            f"Not included (weights or files over 20 MB): {len(summary['skipped'])} file(s),",
            "e.g. " + ", ".join(f"`{p}`" for p in summary["skipped"][:5]),
            "",
        ]
    return "\n".join(lines) + "\n"


def _check_citation_relevance(
    llm: Any,
    topic: str,
    results: list[Any],
) -> dict[str, float | None]:
    """Use LLM to assess relevance of each citation to the research topic.

    Returns a dict mapping cite_key → relevance score (0.0–1.0).
    Processes citations in batches of 30 to handle large bibliographies.
    """
    citation_lines = []
    for cr in results:
        citation_lines.append(f"- [{cr.cite_key}] \"{cr.title}\"")
    if not citation_lines:
        return {}

    all_scores: dict[str, float] = {}
    _BATCH_SIZE = 30

    for batch_start in range(0, len(citation_lines), _BATCH_SIZE):
        batch = citation_lines[batch_start:batch_start + _BATCH_SIZE]
        citations_text = "\n".join(batch)

        prompt = (
            f"Research topic: {topic}\n\n"
            f"Rate the relevance of each citation to the research topic "
            f"on a scale of 0.0 to 1.0.\n"
            f"Return ONLY a JSON object mapping cite_key to relevance score.\n"
            f"Example: {{\"smith2020\": 0.9, \"jones2019\": 0.2}}\n\n"
            f"Citations:\n{citations_text}"
        )

        try:
            resp = llm.chat(
                [{"role": "user", "content": prompt}],
                system="You assess citation relevance. Return only valid JSON.",
                json_mode=True,
            )
            parsed = _safe_json_loads(resp.content, {})
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(v, (int, float)):
                        all_scores[k] = max(0.0, min(1.0, float(v)))
        except Exception:  # noqa: BLE001
            logger.debug(
                "Citation relevance check failed for batch %d–%d, skipping",
                batch_start, batch_start + len(batch),
            )

    return all_scores


def _remove_bibtex_entries(bib_text: str, keys_to_remove: set[str]) -> str:
    """Remove BibTeX entries whose keys are in *keys_to_remove*."""
    kept: list[str] = []
    for m in re.finditer(r"@\w+\{([^,]+),", bib_text):
        key = m.group(1).strip()
        if key in keys_to_remove:
            continue
        # Find the full entry (from @ to the next @ or end)
        start = m.start()
        # Find balanced braces
        depth = 0
        end = start
        for i in range(start, len(bib_text)):
            if bib_text[i] == "{":
                depth += 1
            elif bib_text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            kept.append(bib_text[start:end])
    return "\n\n".join(kept) + "\n" if kept else ""


def _remove_citations_from_text(text: str, keys_to_remove: set[str]) -> str:
    """Remove \\cite{key} and [key] references for specified citation keys."""

    # Handle multi-key LaTeX cites: \cite{a,b,c} → filter keys inside braces
    def _filter_cite(m: re.Match[str]) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        kept = [k for k in keys if k not in keys_to_remove]
        if not kept:
            return ""
        return f"\\cite{{{','.join(kept)}}}"

    text = re.sub(r"\\cite\{([^}]+)\}", _filter_cite, text)

    # Markdown: [key]
    for key in keys_to_remove:
        text = re.sub(rf"\[{re.escape(key)}\]", "", text)
    return text


def _correct_bibtex_dois(bib_text: str, corrections: dict[str, str]) -> str:
    """Replace or drop the doi field of entries whose DOI points elsewhere.

    *corrections* maps cite key -> DOI of the matched paper ("" = drop).
    """
    doi_field = re.compile(r"^[ \t]*doi\s*=\s*(\{[^{}]*\}|\"[^\"]*\")[ \t]*,?[ \t]*\n?", re.IGNORECASE | re.MULTILINE)
    entries = _bibliography_entries(bib_text)
    for key, new_doi in corrections.items():
        entry = entries.get(key)
        if not entry:
            continue
        line = f"  doi = {{{new_doi}}},\n" if new_doi else ""
        fixed = doi_field.sub(lambda _m: line, entry, count=1)
        bib_text = bib_text.replace(entry, fixed, 1)
    return bib_text


def _execute_citation_verify(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    from researchclaw.literature.verify import (
        VerifyStatus,
        verify_citations,
    )

    bib_text = _read_prior_artifact(run_dir, "references.bib") or ""
    paper_text = _read_prior_artifact(run_dir, "paper_final.md") or ""

    if not bib_text.strip():
        report_data = {
            "summary": {
                "total": 0,
                "verified": 0,
                "suspicious": 0,
                "hallucinated": 0,
                "skipped": 0,
                "integrity_score": 0.0,
                "fully_verified": False,
                "verified_fraction": 0.0,
                "unresolved_citation_keys": sorted(_paper_citation_keys(paper_text)),
            },
            "results": [],
            "note": "No bibliography available. Citation verification is incomplete.",
        }
        (stage_dir / "verification_report.json").write_text(
            json.dumps(report_data, indent=2), encoding="utf-8"
        )
        (stage_dir / "references_verified.bib").write_text(
            "% No references to verify\n", encoding="utf-8"
        )
        return StageResult(
            stage=Stage.CITATION_VERIFY,
            status=StageStatus.DONE,
        artifacts=("verification_report.json", "references_verified.bib"),
            evidence_refs=(
                "stage-26/verification_report.json",
                "stage-26/references_verified.bib",
            ),
        )

    s2_api_key = getattr(config.llm, "s2_api_key", "") or ""

    from researchclaw.literature.verify import parse_bibtex_entries
    _n_entries = len(parse_bibtex_entries(bib_text))
    logger.info(
        "[citation-verify] Verifying %d references "
        "(DOI→CrossRef > OpenAlex > arXiv > S2)…",
        _n_entries,
    )
    report = verify_citations(bib_text, s2_api_key=s2_api_key)
    logger.info(
        "[citation-verify] Done: %d verified, %d suspicious, "
        "%d hallucinated, %d skipped (integrity: %.0f%%)",
        report.verified,
        report.suspicious,
        report.hallucinated,
        report.skipped,
        100 * report.verified / report.total if report.total else 0.0,
    )

    # --- Relevance check: assess topical relevance of verified citations ---
    if llm is not None and report.results:
        relevance_scores = _check_citation_relevance(
            llm, config.research.topic, report.results
        )
        for cr in report.results:
            score = relevance_scores.get(cr.cite_key)
            if score is not None:
                cr.relevance_score = score

    bibliography_keys = set(_bibliography_entries(bib_text))
    cited_keys = _paper_citation_keys(paper_text)
    verified_keys = {
        result.cite_key for result in report.results
        if result.status == VerifyStatus.VERIFIED
    }
    pending_keys = bibliography_keys - verified_keys
    missing_keys = cited_keys - bibliography_keys
    low_relevance_keys = {
        result.cite_key for result in report.results
        if result.relevance_score is not None and result.relevance_score < 0.5
    }
    report_data = report.to_dict()
    report_data["summary"].update({
        "bibliography_total": len(bibliography_keys),
        "cited_total": len(cited_keys),
        "integrity_score": len(verified_keys & bibliography_keys) / len(bibliography_keys) if bibliography_keys else 0.0,
        "verified_fraction": len(verified_keys & bibliography_keys) / len(bibliography_keys) if bibliography_keys else 0.0,
        "fully_verified": bool(bibliography_keys) and not pending_keys and not missing_keys and not low_relevance_keys,
        "pending_citation_keys": sorted(pending_keys),
        "unresolved_citation_keys": sorted(missing_keys),
        "low_relevance_citation_keys": sorted(low_relevance_keys),
    })
    # A DOI that resolves to a different paper is metadata, not a citation:
    # point it at the matched paper, or drop it when no DOI is known.
    doi_corrections = {
        result.cite_key: result.corrected_doi
        for result in report.results
        if result.bibtex_doi and result.status == VerifyStatus.VERIFIED
    }
    verified_bib = _correct_bibtex_dois(bib_text, doi_corrections) if doi_corrections else bib_text
    report_data["summary"]["corrected_doi_keys"] = sorted(doi_corrections)
    report_data["note"] = (
        "Original citations and bibliography are preserved. Pending and low-relevance entries "
        "require review; no references were removed."
        + (" DOIs that pointed to a different paper were corrected in references_verified.bib "
           "(original kept as references_original.bib)." if doi_corrections else "")
    )
    (stage_dir / "verification_report.json").write_text(
        json.dumps(report_data, indent=2), encoding="utf-8"
    )
    (stage_dir / "references_verified.bib").write_text(
        _escape_bib_specials(verified_bib), encoding="utf-8"
    )
    artifacts = ["verification_report.json", "references_verified.bib"]
    if doi_corrections:
        (stage_dir / "references_original.bib").write_text(bib_text, encoding="utf-8")
        artifacts.append("references_original.bib")
    if paper_text.strip():
        annotated = paper_text
        if pending_keys or missing_keys or low_relevance_keys:
            annotated = (
                "> **Citation review pending:** Original citations are preserved. "
                "See verification_report.json for unresolved references and relevance concerns.\n\n"
                + paper_text
            )
        (stage_dir / "paper_final_verified.md").write_text(annotated, encoding="utf-8")
        artifacts.append("paper_final_verified.md")

    logger.info(
        "Stage 26 citation verify: %d total, %d verified, %d suspicious, "
        "%d hallucinated, %d skipped (integrity=%.1f%%)",
        report.total,
        report.verified,
        report.suspicious,
        report.hallucinated,
        report.skipped,
        report_data["summary"]["integrity_score"] * 100,
    )

    return StageResult(
        stage=Stage.CITATION_VERIFY,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-26/{a}" for a in artifacts),
    )


_STAGE_EXECUTORS: dict[Stage, Callable[..., StageResult]] = {
    Stage.TOPIC_INIT: _execute_topic_init,
    Stage.PROBLEM_DECOMPOSE: _execute_problem_decompose,
    Stage.SEARCH_STRATEGY: _execute_search_strategy,
    Stage.LITERATURE_COLLECT: _execute_literature_collect,
    Stage.LITERATURE_SCREEN: _execute_literature_screen,
    Stage.KNOWLEDGE_EXTRACT: _execute_knowledge_extract,
    Stage.SYNTHESIS: _execute_synthesis,
    Stage.HYPOTHESIS_GEN: _execute_hypothesis_gen,
    Stage.EXPERIMENT_DESIGN: _execute_experiment_design,
    Stage.CODEBASE_SEARCH: _execute_codebase_search,
    Stage.CODE_GENERATION: _execute_code_generation,
    Stage.SANITY_CHECK: _execute_sanity_check,
    Stage.RESOURCE_PLANNING: _execute_resource_planning,
    Stage.EXPERIMENT_RUN: _execute_experiment_run,
    Stage.ITERATIVE_REFINE: _execute_iterative_refine,
    Stage.RESULT_ANALYSIS: _execute_result_analysis,
    Stage.RESEARCH_DECISION: _execute_research_decision,
    Stage.KNOWLEDGE_SUMMARY: _execute_knowledge_summary,
    Stage.PAPER_OUTLINE: _execute_paper_outline,
    Stage.PAPER_DRAFT: _execute_paper_draft,
    Stage.PEER_REVIEW: _execute_peer_review,
    Stage.PAPER_REVISION: _execute_paper_revision,
    Stage.QUALITY_GATE: _execute_quality_gate,
    Stage.KNOWLEDGE_ARCHIVE: _execute_knowledge_archive,
    Stage.EXPORT_PUBLISH: _execute_export_publish,
    Stage.CITATION_VERIFY: _execute_citation_verify,
}


def execute_stage(
    stage: Stage,
    *,
    run_dir: Path,
    run_id: str,
    config: RCConfig,
    adapters: AdapterBundle,
    auto_approve_gates: bool = False,
) -> StageResult:
    """Execute one pipeline stage, validate outputs, and apply gate logic."""

    # The file/UI topic stays short. Every stage receives the separately stored
    # details as labeled context, including resume and downstream writing stages.
    if getattr(config.research, "details", ""):
        from dataclasses import replace
        config = replace(config, research=replace(config.research,
            topic=config.research.prompt_topic, details=""))

    stage_dir = run_dir / f"stage-{int(stage):02d}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    _t_health_start = _time.monotonic()
    contract: StageContract = CONTRACTS[stage]

    if contract.input_files:
        for input_file in contract.input_files:
            found = _read_prior_artifact(run_dir, input_file)
            if found is None:
                result = StageResult(
                    stage=stage,
                    status=StageStatus.FAILED,
                    artifacts=(),
                    error=f"Missing input: {input_file} (required by {stage.name})",
                    decision="retry",
                )
                _write_stage_meta(stage_dir, stage, run_id, result)
                return result

    bridge = config.openclaw_bridge
    if bridge.use_message and config.notifications.on_stage_start:
        adapters.message.notify(
            config.notifications.channel,
            f"stage-{int(stage):02d}-start",
            f"Starting {stage.name}",
        )
    if bridge.use_memory:
        adapters.memory.append("stages", f"{run_id}:{int(stage)}:running")

    llm = None
    try:
        if config.llm.provider == "acp":
            llm = create_llm_client(config)
        else:
            candidate = LLMClient.from_rc_config(config)
            if candidate.config.base_url and candidate.config.api_key:
                llm = candidate
    except Exception:  # noqa: BLE001
        llm = None

    try:
        _ = advance(stage, StageStatus.PENDING, TransitionEvent.START)
        executor = _STAGE_EXECUTORS[stage]
        prompts = PromptManager(config.prompts.custom_file or None)  # type: ignore[attr-defined]
        prompts.set_runs_per_condition(single_run.runs_per_condition(config))

        human_fb = _load_human_feedback(run_dir, stage)
        if human_fb:
            prompts.set_human_feedback(human_fb)
            logger.info("Stage %s: injecting human feedback (%d chars)", stage.name, len(human_fb))

        try:
            result = executor(
                stage_dir, run_dir, config, adapters, llm=llm, prompts=prompts
            )
        except TypeError as exc:
            if "unexpected keyword argument 'prompts'" not in str(exc):
                raise
            result = executor(stage_dir, run_dir, config, adapters, llm=llm)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stage %s failed", stage.name)
        result = StageResult(
            stage=stage,
            status=StageStatus.FAILED,
            artifacts=(),
            error=str(exc),
            decision="retry",
        )

    if result.status == StageStatus.DONE:
        for output_file in contract.output_files:
            if output_file.endswith("/"):
                path = stage_dir / output_file.rstrip("/")
                if not path.is_dir() or not any(path.iterdir()):
                    result = StageResult(
                        stage=stage,
                        status=StageStatus.FAILED,
                        artifacts=result.artifacts,
                        error=f"Missing output directory: {output_file}",
                        decision="retry",
                        evidence_refs=result.evidence_refs,
                    )
                    break
            else:
                path = stage_dir / output_file
                if not path.exists() or path.stat().st_size == 0:
                    result = StageResult(
                        stage=stage,
                        status=StageStatus.FAILED,
                        artifacts=result.artifacts,
                        error=f"Missing or empty output: {output_file}",
                        decision="retry",
                        evidence_refs=result.evidence_refs,
                    )
                    break

    # --- MetaClaw PRM quality gate evaluation ---
    try:
        mc_bridge = getattr(config, "metaclaw_bridge", None)
        if (
            mc_bridge
            and getattr(mc_bridge, "enabled", False)
            and result.status == StageStatus.DONE
        ):
            mc_prm = getattr(mc_bridge, "prm", None)
            if mc_prm and getattr(mc_prm, "enabled", False):
                prm_stages = getattr(mc_prm, "gate_stages", (5, 9, 15, 20))
                if int(stage) in prm_stages:
                    from researchclaw.metaclaw_bridge.prm_gate import ResearchPRMGate

                    prm_gate = ResearchPRMGate.from_bridge_config(mc_prm)
                    if prm_gate is not None:
                        # Read stage output for PRM evaluation
                        output_text = ""
                        for art in result.artifacts:
                            art_path = stage_dir / art
                            if art_path.exists() and art_path.is_file():
                                try:
                                    output_text += art_path.read_text(encoding="utf-8")[:4000]
                                except (UnicodeDecodeError, OSError):
                                    pass
                        if output_text:
                            prm_score = prm_gate.evaluate_stage(int(stage), output_text)
                            logger.info(
                                "MetaClaw PRM score for stage %d: %.1f",
                                int(stage),
                                prm_score,
                            )
                            # Write PRM score to stage health
                            import json as _prm_json

                            prm_report = {
                                "stage": int(stage),
                                "prm_score": prm_score,
                                "model": prm_gate.model,
                                "votes": prm_gate.votes,
                            }
                            (stage_dir / "prm_score.json").write_text(
                                _prm_json.dumps(prm_report, indent=2),
                                encoding="utf-8",
                            )
                            # If PRM score is -1 (fail), mark stage as failed
                            if prm_score == -1.0:
                                logger.warning(
                                    "MetaClaw PRM rejected stage %d output",
                                    int(stage),
                                )
                                result = StageResult(
                                    stage=result.stage,
                                    status=StageStatus.FAILED,
                                    artifacts=result.artifacts,
                                    error="PRM quality gate: output below quality threshold",
                                    decision="retry",
                                    evidence_refs=result.evidence_refs,
                                )
    except Exception:  # noqa: BLE001
        logger.warning("MetaClaw PRM evaluation failed (non-blocking)")

    if gate_required(stage, config.security.hitl_required_stages):
        if auto_approve_gates:
            if bridge.use_memory:
                adapters.memory.append("gates", f"{run_id}:{int(stage)}:auto-approved")
        else:
            result = StageResult(
                stage=result.stage,
                status=StageStatus.BLOCKED_APPROVAL,
                artifacts=result.artifacts,
                error=result.error,
                decision="block",
                evidence_refs=result.evidence_refs,
            )
            if bridge.use_message and config.notifications.on_gate_required:
                adapters.message.notify(
                    config.notifications.channel,
                    f"gate-{int(stage):02d}",
                    f"Approval required for {stage.name}",
                )

    if bridge.use_memory:
        adapters.memory.append("stages", f"{run_id}:{int(stage)}:{result.status.value}")

    _write_stage_meta(stage_dir, stage, run_id, result)

    _t_health_end = _time.monotonic()
    stage_health = {
        "stage_id": f"{int(stage):02d}-{stage.name.lower()}",
        "run_id": run_id,
        "duration_sec": round(_t_health_end - _t_health_start, 2),
        "status": result.status.value,
        "artifacts_count": len(result.artifacts),
        "error": result.error,
        "timestamp": _utcnow_iso(),
    }
    try:
        (stage_dir / "stage_health.json").write_text(
            json.dumps(stage_health, indent=2), encoding="utf-8"
        )
    except OSError:
        pass

    return result
