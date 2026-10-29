"""Combine complementary L1 hypotheses, preserving every direction and its evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))

from discussion_runner import Discussion, SYSTEM_CONDENSE
from researchclaw.config import RCConfig
from researchclaw.llm import create_llm_client


SYSTEM = """You are coordinating complementary research directions into ONE experiment.
All supplied directions are required contributions, not competing proposals. Evaluate their
evidence, novelty relative to cited work, feasibility, interface compatibility, and testability.
Do not claim novelty or efficacy is proven. Do not invent references, results, or evidence.
If a required direction is unsupported or incompatible, set status=blocked and explain why;
never silently omit it. Each mechanism must be specific, grounded in the supplied hypothesis,
and mapped to an implementable part of the joint method. Preserve user constraints.
Return ONLY JSON with this schema:
{"status":"ready|blocked", "reason":"...", "directions":[
{"id":"D1", "mechanism":"...", "evidence":"...", "novelty_risk":"...",
"integration":"...", "verification":"..."}],
"joint_method":"...", "experiment_plan":"..."}.
Include exactly one entry for every supplied direction ID. If the research is about adaptive
attention, adaptation must act on attention scale selection or attention weight computation,
not just post-hoc feature fusion. Respond in the topic's language."""

SYSTEM += """
STAGE CONTRACT: This is S8, before S9 experiment design, S10 source inspection,
S11 implementation, and S12/S13 numerical and resource preflight. 'ready' means
an explicit, coherent DESIGN is ready for those stages; it never means trained,
empirically effective, proven novel, or cleared for immediate full training.
Missing future code inspection, fixtures, GPU access, or training results belongs
in pending_checks, not by itself in a blocked decision. Never claim those checks
were performed. Real unsupported scientific premises still require qualification.
Separate the user's hard constraints from candidate-specific implementation choices.
You may revise the latter to reconcile the proposals while retaining EVERY required
direction's scientific mechanism. Explicitly document each conflict and resolution;
do not silently discard a direction or pretend incompatible interfaces agree.
Specify one interface contract: where information flows, what is exchanged
(values, affinities, connectivity constraints or attention logits), how scale and
spatial attention adapt, and how gradients reach both mechanisms. There is no
requirement to exchange values if selective cross-branch information is communicated
through another explicit mechanism. Do not mandate an architecture based on a name.
If no coherent design can satisfy all required directions AND the user's constraints,
return blocked with the exact unresolved conflict. Lack of prior empirical success
alone is not such a conflict. Keep novelty and effectiveness as hypotheses.
Respect the requested training conditions, seeds, data rules, and optional experiment
scope; do not expand them to earn readiness. Put required source, mathematical,
gradient and resource checks BEFORE training in the plan, with stop conditions.
ADDITIONAL REQUIRED JSON FIELDS (both ready and blocked responses):
"readiness": "design_only", "training_completed": false,
"interface_contract": "the reconciled design and information-flow interface",
"reconciliations": [{"conflict":"...", "resolution":"..."}],
"pending_checks": [{"stage":"S9-S13 before full training", "check":"...",
"stop_condition":"..."}]. Do not relabel unperformed checks as completed evidence.
"""


def validate(payload: dict, expected: list[str]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Invalid joint hypothesis response")
    if payload.get("status") != "ready":
        raise ValueError("Joint hypothesis blocked: " + str(payload.get("reason", "invalid response")))
    rows = payload.get("directions", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Invalid direction coverage")
    ids = [row.get("id") for row in rows]
    if len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError("Joint hypothesis must cover every direction exactly once")
    for row in rows:
        for key in ("mechanism", "evidence", "novelty_risk", "integration", "verification"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing {key} for {row['id']}")
    for key in ("joint_method", "experiment_plan"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ValueError(f"Missing {key}")
    if payload.get("readiness") != "design_only" or payload.get("training_completed") is not False:
        raise ValueError("S8 readiness must describe an untrained design, not completed training")
    if not isinstance(payload.get("interface_contract"), str) or not payload["interface_contract"].strip():
        raise ValueError("Missing reconciled interface contract")
    checks = payload.get("pending_checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("Unperformed pre-training checks must be recorded")
    for check in checks:
        if not isinstance(check, dict) or any(not isinstance(check.get(k), str) or not check[k].strip()
                                              for k in ("stage", "check", "stop_condition")):
            raise ValueError("Invalid pending pre-training check")
    changes = payload.get("reconciliations")
    if not isinstance(changes, list) or any(not isinstance(row, dict) or any(
            not isinstance(row.get(k), str) or not row[k].strip() for k in ("conflict", "resolution")) for row in changes):
        raise ValueError("Invalid interface reconciliation record")


def _equivalent_bibliography_entries(left: str, right: str) -> bool:
    """Only URL/lookup-date variations of the same DOI record are harmless."""
    def doi(entry: str) -> str:
        match = re.search(r'\bdoi\s*=\s*[{"]([^}"]+)[}"]', entry, re.I)
        return re.sub(r'^https?://(?:dx\.)?doi\.org/', '', match.group(1).strip().lower()) if match else ''
    def core(entry: str) -> str:
        value = re.sub(r'(?im)^\s*(?:url|urldate|accessdate)\s*=\s*(?:\{[^{}]*\}|"[^"]*")\s*,?[ \t]*$', '', entry)
        value = re.sub(r',\s*}', '}', value)
        return re.sub(r'\s+', '', value).casefold()
    return bool(doi(left)) and doi(left) == doi(right) and core(left) == core(right)


def _merge_bibliographies(sources: list[dict]) -> tuple[dict[str, str], list[dict]]:
    from researchclaw.pipeline.executor import _bibliography_entries
    entries: dict[str, str] = {}
    duplicates = []
    for source in sources:
        path = Path(source["run_dir"]) / "stage-04/references.bib"
        if not path.is_file():
            continue
        for key, value in _bibliography_entries(path.read_text(encoding="utf-8")).items():
            if key in entries and entries[key] != value:
                if not _equivalent_bibliography_entries(entries[key], value):
                    raise ValueError(f"Conflicting bibliography key: {key}")
                duplicates.append({"cite_key": key, "source": str(path), "retained": entries[key],
                                   "equivalent": value, "reason": "same DOI and core fields; URL-only variation"})
            else:
                entries[key] = value
    return entries, duplicates


def integrate(request: dict, output: Path, llm) -> None:
    """Persist a reviewable failure as well as successful design integration."""
    stage = output / "stage-08"
    stage.mkdir(parents=True, exist_ok=True)
    try:
        _integrate(request, output, llm)
    except Exception as exc:
        (stage / "integration_report.json").write_text(json.dumps({
            "status": "blocked", "reason": str(exc), "readiness": "design_only",
            "training_completed": False, "sources": request.get("sources", []),
            "response_file": "integration_response.json",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


def _integrate(request: dict, output: Path, llm) -> None:
    talk = Discussion(llm, 14000)
    sources = request["sources"]
    if len(sources) < 2:
        raise ValueError("Integration needs at least two directions")
    # Detect genuine reference identity conflicts before spending model calls.
    entries, duplicate_refs = _merge_bibliographies(sources)
    topic = request["topic"]
    # Never silently truncate project constraints. Oversized input requires an explicit edit.
    if len(topic) > 8000:
        raise ValueError("Research topic exceeds integration budget; shorten it explicitly")
    topic_block = "Research topic and constraints:\n" + topic + "\n\n"
    budget = 14000 - len(SYSTEM) - len(topic_block) - 300
    share = budget // len(sources)
    if share < 500:
        raise ValueError("Too many directions for the integration input budget")
    blocks = []
    expected = []
    input_records = []
    for index, source in enumerate(sources, 1):
        ident = f"D{index}"
        expected.append(ident)
        text = (Path(source["run_dir"]) / "stage-08/hypotheses.md").read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Empty hypothesis for {ident}")
        novelty = Path(source["run_dir"]) / "stage-08/novelty_report.json"
        if novelty.is_file():
            text += "\n\nNovelty assessment (including uncertainty):\n" + novelty.read_text(encoding="utf-8")
        label = f"{ident}: {source['name']}"
        if len(label) > share // 2:
            raise ValueError("Direction name too long")
        digest = hashlib.sha256((text + SYSTEM + SYSTEM_CONDENSE).encode()).hexdigest()
        cache = output / "stage-08" / f"condensed_{ident}.json"
        try:
            cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
        except (ValueError, OSError):
            cached = {}
        if not isinstance(cached, dict):
            cached = {}
        limit = share - len(label) - 10
        if (cached.get("sha256") == digest and cached.get("limit") == limit
                and isinstance(cached.get("text"), str) and len(cached["text"]) <= limit):
            text = cached["text"]
        else:
            text = talk.condense(label, text, limit)
            cache.write_text(json.dumps({"sha256": digest, "limit": limit, "text": text}, ensure_ascii=False, indent=2), encoding="utf-8")
        input_records.append({"id": ident, "name": source["name"], "source": source["run_dir"],
                              "sha256": digest, "condensed": text})
        blocks.append(label + "\n" + text)
    (output / "stage-08/integration_inputs.json").write_text(json.dumps(
        {"topic": topic, "directions": input_records}, ensure_ascii=False, indent=2), encoding="utf-8")
    response = talk.chat("Integrating all directions", SYSTEM, topic_block + "\n\n".join(blocks), 8192)
    (output / "stage-08/integration_response.json").write_text(response, encoding="utf-8")
    raw = response.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    payload = json.loads(raw)
    validate(payload, expected)

    # Keep all original S1-S8 artifacts, and a neutral shared working tree for S9 onward.
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / "discussion/direction_artifacts"
    for index, source in enumerate(sources, 1):
        src = Path(source["run_dir"])
        for number in range(1, 9):
            stage = src / f"stage-{number:02d}"
            if stage.is_dir():
                shutil.copytree(stage, provenance / f"D{index}" / stage.name, dirs_exist_ok=True)
                if index == 1 and number <= 7:
                    shutil.copytree(stage, output / stage.name, dirs_exist_ok=True)
    # Union real candidates and screened papers from every direction, not just an anchor.
    for number, filename in ((4, "candidates.jsonl"), (5, "shortlist.jsonl")):
        unique = {}
        for source in sources:
            path = Path(source["run_dir"]) / f"stage-{number:02d}" / filename
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    key = str(row.get("title", row.get("id", ""))).casefold()
                    if key:
                        unique.setdefault(key, row)
        dest = output / f"stage-{number:02d}"
        dest.mkdir(exist_ok=True)
        (dest / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                             for row in unique.values()), encoding="utf-8")
    # Bibliographies remain attributable; combine entries by key for downstream citation lookup.
    (output / "stage-04/references.bib").write_text("\n\n".join(entries.values()), encoding="utf-8")
    (output / "stage-04/bibliography_merge.json").write_text(json.dumps(
        {"equivalent_duplicates": duplicate_refs}, ensure_ascii=False, indent=2), encoding="utf-8")
    discussion = output / "discussion"
    consensus = Path(request["consensus_dir"])
    synthesis_file = request.get("synthesis_file", "consensus_synthesis.md")
    for name in (synthesis_file, "discussion_transcript.md", "pre_discussion_syntheses.md"):
        path = consensus / name
        if path.is_file():
            shutil.copy2(path, discussion / name)
    pre_discussion = []
    for index, source in enumerate(sources, 1):
        path = Path(source["run_dir"]) / "stage-07/synthesis.md"
        if path.is_file():
            text = path.read_text(encoding="utf-8").split("# Multi-Agent Discussion Consensus", 1)[0]
            pre_discussion.append(f"## D{index}: {source['name']}\n\n{text}")
    (discussion / "pre_discussion_syntheses.md").write_text("\n\n".join(pre_discussion), encoding="utf-8")
    synthesis = output / "stage-07"
    synthesis.mkdir(exist_ok=True)
    (synthesis / "synthesis.md").write_text(
        (consensus / synthesis_file).read_text(encoding="utf-8"), encoding="utf-8")
    stage = output / "stage-08"
    stage.mkdir(exist_ok=True)
    method = ["# Joint research hypothesis", payload["joint_method"], "## Direction coverage"]
    by_id = {row["id"]: row for row in payload["directions"]}
    for index, source in enumerate(sources, 1):
        row = by_id[f"D{index}"]
        method.append(f"### D{index}: {source['name']}")
        method.extend(f"**{key}**: {row[key]}" for key in
                      ("mechanism", "evidence", "novelty_risk", "integration", "verification"))
    method.extend(["## Experiment plan", payload["experiment_plan"]])
    method.extend(["## Design readiness and interface contract",
                   "S8 design only. Implementation, preflight and training are not completed.",
                   payload["interface_contract"], "## Interface reconciliations"])
    method.extend(f"- Conflict: {row['conflict']}\n  Resolution: {row['resolution']}" for row in payload["reconciliations"])
    method.append("## Pending checks before full training")
    method.extend(f"- {row['stage']}: {row['check']}\n  Stop if: {row['stop_condition']}" for row in payload["pending_checks"])
    (stage / "hypotheses.md").write_text("\n\n".join(method) + "\n", encoding="utf-8")
    (stage / "integration_report.json").write_text(json.dumps(
        {**payload, "sources": sources, "mode": request.get("workflow", "discussion_joint")}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "checkpoint.json").write_text(json.dumps({"last_completed_stage": 8}), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = RCConfig.load(args.config, check_paths=False)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    request["topic"] = config.research.topic
    integrate(request, Path(args.output), create_llm_client(config))


if __name__ == "__main__":
    main()
