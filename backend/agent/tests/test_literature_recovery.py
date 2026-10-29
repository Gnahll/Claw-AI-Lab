import json
import gzip
import io
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from researchclaw.literature import bounded_search, publisher, search
from researchclaw.literature.models import Paper
from researchclaw.pipeline import executor
from researchclaw.pipeline.stages import StageStatus


def test_publisher_citation_metadata_preserves_real_title_authors_and_year():
    html = '''<meta name="citation_title" content="Collaborative Transformers With Multi-Level Forensic Attention for Image Manipulation Localization">
    <meta name="citation_author" content="A. Author"><meta name="citation_author" content="B. Author">
    <meta name="citation_date" content="2026/03/24"><meta name="citation_doi" content="10.1609/aaai.test">
    <meta name="DC.Description" content="&lt;p&gt;Image forgery localization.&lt;/p&gt;">
    <meta name="citation_conference_title" content="AAAI">'''
    paper = publisher.paper_from_html("https://publisher.example/article/38250", html)
    assert paper.title.startswith("Collaborative Transformers")
    assert paper.year == 2026 and len(paper.authors) == 2
    assert paper.doi == "10.1609/aaai.test" and paper.abstract == "Image forgery localization."
    assert "2026" in paper.to_bibtex()


@pytest.mark.parametrize("html", ["<title>Sign in</title>", '<meta name="citation_title" content="38250">', ""])
def test_unresolved_publisher_pages_are_not_numeric_title_candidates(monkeypatch, html):
    monkeypatch.setattr(publisher, "resolve_publisher_reference", lambda url: publisher.paper_from_html(url, html))
    assert executor._resolve_user_reference("https://publisher.example/article/38250") is None


def test_plain_title_does_not_invent_publication_year():
    assert executor._resolve_user_reference("Image forgery localization")["year"] == 0


def test_publisher_reads_unsolicited_gzip_response(monkeypatch):
    page = b'<meta name="citation_title" content="Real forensic attention paper"><meta name="citation_date" content="2026">'
    monkeypatch.setattr(publisher, "urlopen", lambda *a, **kw: io.BytesIO(gzip.compress(page)))
    paper = publisher.resolve_publisher_reference("https://publisher.example/38250")
    assert paper.title == "Real forensic attention paper" and paper.year == 2026


def test_source_response_is_flushed_before_later_source_blocks(monkeypatch):
    paper = Paper(paper_id="real", title="Real image forgery paper", source="openalex")
    observed = []
    monkeypatch.setattr(search, "_cache_api", lambda: (lambda *a: None, lambda *a: None))
    monkeypatch.setattr(search.time, "sleep", lambda _: None)
    monkeypatch.setattr(search, "search_openalex", lambda *a, **kw: [paper])
    def later(*args, **kwargs):
        assert observed == [paper]
        return []
    monkeypatch.setattr(search, "search_arxiv", later)
    assert search.search_papers("forgery", sources=("openalex", "arxiv"), on_results=observed.extend) == [paper]


def test_real_worker_timeout_retains_completed_results_and_reaps_process(tmp_path, monkeypatch):
    workers = []
    # Use a real OS process that flushes a result, then stalls like a slow API.
    def command(request, output):
        pid_file = output.with_suffix(".pid")
        workers.append(pid_file)
        program = (
            "import json,os,sys,time; from pathlib import Path; "
            "Path(sys.argv[2]).write_text(str(os.getpid())); "
            "f=open(sys.argv[1],'w'); f.write(json.dumps({'paper_id':'real','title':'Image forgery localization','source':'openalex'})+'\\n'); "
            "f.flush(); time.sleep(60)"
        )
        return [sys.executable, "-c", program, str(output), str(pid_file)]
    monkeypatch.setattr(bounded_search, "_worker_command", command)
    children = []
    original_popen = bounded_search.subprocess.Popen
    def popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(bounded_search.subprocess, "Popen", popen)
    started = time.monotonic()
    papers = bounded_search.bounded_search(["forgery"], output_dir=tmp_path, sources=("openalex",),
                                            year_min=2020, s2_api_key="secret-test-only", timeout=1)
    assert time.monotonic() - started < 4
    assert [paper.paper_id for paper in papers] == ["real"]
    assert children and all(child.poll() is not None for child in children)
    assert "secret-test-only" not in (tmp_path / "request-0.json").read_text()


def test_screening_batches_cover_all_candidates_within_gateway_budget(tmp_path, monkeypatch):
    stage4 = tmp_path / "stage-04"
    stage5 = tmp_path / "stage-05"
    stage4.mkdir()
    stage5.mkdir()
    rows = [{"title": f"Image forgery localization study {i}", "abstract": "Image forgery. " * 1200,
             "source": "openalex", "year": 2026, "url": f"https://example.test/{i}"} for i in range(17)]
    (stage4 / "candidates.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assessed = []
    def chat(llm, system, user, **kwargs):
        assert len(system) + len(user) <= 12000
        batch = [json.loads(line) for line in user.split("Candidates JSONL:\n", 1)[1].splitlines()]
        assessed.extend(row["title"] for row in batch)
        return SimpleNamespace(content=json.dumps({"shortlist": [{"title": row["title"], "relevance_score": 0.9} for row in batch]}))
    monkeypatch.setattr(executor, "_chat_with_prompt", chat)
    cfg = SimpleNamespace(research=SimpleNamespace(topic="Image forgery localization and adaptive attention. " * 250,
                                                   domains=(), quality_threshold=0.7))
    result = executor._execute_literature_screen(stage5, tmp_path, cfg, None, llm=object())
    assert result.status == StageStatus.DONE
    assert assessed == [row["title"] for row in rows]
    accepted = [json.loads(line) for line in (stage5 / "shortlist.jsonl").read_text().splitlines()]
    assert len(accepted) == 17 and accepted[0]["abstract"] == rows[0]["abstract"]
    assert len(list(stage5.glob("screening_response_*.json"))) > 1


def test_s4_unresolved_reference_is_explicit_and_does_not_call_llm_to_invent_papers(tmp_path, monkeypatch):
    from researchclaw import data
    monkeypatch.setattr(executor, "_check_literature_sites_reachable", lambda **kw: {"openalex": True})
    monkeypatch.setattr(bounded_search, "bounded_search", lambda *a, **kw: [])
    monkeypatch.setattr(publisher, "resolve_publisher_reference", lambda url: None)
    monkeypatch.setattr(data, "load_seminal_papers", lambda topic: [])
    monkeypatch.setattr(executor, "_chat_with_prompt", lambda *a, **kw: pytest.fail("must not invent papers"))
    cfg = SimpleNamespace(research=SimpleNamespace(topic="image forgery", reference_papers=("https://example.test/38250",)),
                          llm=SimpleNamespace(s2_api_key=""), web_search=SimpleNamespace(enabled=False))
    result = executor._execute_literature_collect(tmp_path, tmp_path, cfg, None, llm=object())
    assert result.status == StageStatus.FAILED
    assert json.loads((tmp_path / "unresolved_references.json").read_text()) == ["https://example.test/38250"]
    assert (tmp_path / "candidates.jsonl").read_text() == ""
