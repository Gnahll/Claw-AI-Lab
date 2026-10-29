"""Isolate slow API clients; retain each completed source response on timeout."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from researchclaw.literature.models import Paper
from researchclaw.literature.search import _deduplicate, _dicts_to_papers

logger = logging.getLogger(__name__)


def _worker_command(request: Path, output: Path) -> list[str]:
    return [sys.executable, "-m", "researchclaw.literature.bounded_search", str(request), str(output)]


def bounded_search(queries: list[str], *, output_dir: Path, sources: tuple[str, ...],
                   year_min: int, s2_api_key: str = "", limit_per_query: int = 40,
                   timeout: float = 60, sort_by: str = "relevance") -> list[Paper]:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["RESEARCHCLAW_SEARCH_S2_KEY"] = s2_api_key
    # The package may be imported via a launcher-specific sys.path.
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")])
    workers = []
    papers: list[Paper] = []
    deadline = time.monotonic() + timeout
    try:
        for index, source in enumerate(sources):
            request = output_dir / f"request-{index}.json"
            output = output_dir / f"partial-{index}.jsonl"
            request.write_text(json.dumps(dict(queries=queries, source=source, year_min=year_min,
                                               limit=limit_per_query, timeout=timeout, sort_by=sort_by)), encoding="utf-8")
            output.write_text("", encoding="utf-8")
            log = (output_dir / f"worker-{index}.log").open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(_worker_command(request, output), env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            workers.append((process, log, source, output))
        for process, _, source, _ in workers:
            try:
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                logger.warning("Literature %s exceeded %.1fs; preserving completed responses", source, timeout)
    finally:
        for process, log, _, _ in workers:
            if process.poll() is None:
                process.kill()
            process.wait()
            log.close()
    for process, _, source, output in workers:
        count = 0
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                batch = _dicts_to_papers([json.loads(line)])
                papers.extend(batch)
                count += len(batch)
            except (ValueError, TypeError, KeyError):
                # A killed writer can leave one incomplete trailing line.
                continue
        logger.info("Literature %s retained %d results (worker exit %s)", source, count, process.returncode)
    result = _deduplicate(papers)
    return sorted(result, key=lambda p: (p.citation_count, p.year), reverse=True)


def _main() -> None:
    from dataclasses import asdict
    from researchclaw.literature.search import search_papers_multi_query
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO)
    with Path(sys.argv[2]).open("a", encoding="utf-8") as stream:
        def persist(papers: list[Paper]) -> None:
            for paper in papers:
                stream.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")
            stream.flush()

        if request.get("sort_by") == "submitted_date":
            from researchclaw.literature.arxiv_client import search_arxiv
            for query in request["queries"]:
                persist(search_arxiv(query, limit=request["limit"], year_min=request["year_min"], sort_by="submitted_date"))
        else:
            search_papers_multi_query(request["queries"], sources=(request["source"],),
                                     limit_per_query=request["limit"], year_min=request["year_min"],
                                     s2_api_key=os.environ.get("RESEARCHCLAW_SEARCH_S2_KEY", ""),
                                     max_time_sec=request["timeout"], on_results=persist)


if __name__ == "__main__":
    _main()
