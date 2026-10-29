import json
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from researchclaw.pipeline import executor
from researchclaw.pipeline.stages import StageStatus


SOURCES = ("openalex", "semantic_scholar", "arxiv")


def test_pending_probe_preserves_available_sources(monkeypatch, caplog):
    import concurrent.futures

    futures = {name: Future() for name in SOURCES}
    futures["arxiv"].set_result(("arxiv", True))
    futures["openalex"].set_result(("openalex", False))
    shutdown_calls = []
    class Pool:
        def __init__(self, **kwargs):
            pass
        def submit(self, function, name, url):
            return futures[name]
        def shutdown(self, **kwargs):
            shutdown_calls.append(kwargs)
    def wait(submitted, timeout):
        assert timeout == 6
        assert set(submitted) == set(futures.values())
        return {futures["arxiv"], futures["openalex"]}, {futures["semantic_scholar"]}
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(concurrent.futures, "wait", wait)
    assert executor._check_literature_sites_reachable() == {
        "openalex": False, "semantic_scholar": False, "arxiv": True,
    }
    assert futures["semantic_scholar"].cancelled()
    assert shutdown_calls == [{"wait": False, "cancel_futures": True}]
    assert "skipping pending sources: semantic_scholar" in caplog.text


def test_running_probe_does_not_hold_stage_after_deadline(monkeypatch):
    import concurrent.futures
    import urllib.request

    release = threading.Event()
    finished = threading.Event()
    original_wait = concurrent.futures.wait
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    def urlopen(request, timeout):
        if "semanticscholar" in request.full_url:
            try:
                release.wait(5)
            finally:
                finished.set()
        return Response()
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(concurrent.futures, "wait", lambda futures, timeout: original_wait(futures, timeout=0.15))
    started = time.monotonic()
    try:
        result = executor._check_literature_sites_reachable()
        assert time.monotonic() - started < 2
        assert result == {"arxiv": True, "openalex": True, "semantic_scholar": False}
    finally:
        release.set()
        assert finished.wait(2)


@pytest.mark.parametrize("failed", [(), SOURCES, ("arxiv",)])
def test_probe_reports_network_errors_without_crashing(monkeypatch, failed):
    import urllib.request

    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    def urlopen(request, timeout):
        source = "semantic_scholar" if "semanticscholar" in request.full_url else "arxiv" if "arxiv" in request.full_url else "openalex"
        if source in failed:
            raise URLError("test network failure")
        return Response()
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert executor._check_literature_sites_reachable() == {name: name not in failed for name in SOURCES}


def test_all_unavailable_reports_retry_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "_check_literature_sites_reachable", lambda **kwargs: dict.fromkeys(SOURCES, False))
    result = executor._execute_literature_collect(
        tmp_path, tmp_path, SimpleNamespace(research=SimpleNamespace(topic="image forensics")), None,
    )
    assert result.status == StageStatus.FAILED
    assert result.decision == "retry"
    assert "retry S4" in result.error
    assert "No literature search was completed" in result.error
    assert (tmp_path / "candidates.jsonl").read_text() == ""
    assert json.loads((tmp_path / "site_status.json").read_text())["sources"] == dict.fromkeys(SOURCES, False)


@pytest.mark.parametrize("available", [("arxiv",), ("openalex", "semantic_scholar")])
def test_s4_searches_only_available_sources(tmp_path, monkeypatch, available):
    from researchclaw import data
    from researchclaw.literature import search, arxiv_client
    from researchclaw.literature.models import Paper

    monkeypatch.setattr(executor, "_check_literature_sites_reachable", lambda **kwargs: {name: name in available for name in SOURCES})
    calls = []
    def search_papers(*args, **kwargs):
        calls.append(kwargs["sources"])
        return [Paper(paper_id="real-paper", title="Retrieved paper", source=available[0])]
    from researchclaw.literature import bounded_search
    monkeypatch.setattr(bounded_search, "bounded_search", lambda *a, **kw:
                        [] if kw.get("sort_by") == "submitted_date" else search_papers(*a, **kw))
    monkeypatch.setattr(data, "load_seminal_papers", lambda topic: [])
    monkeypatch.setattr(arxiv_client, "search_arxiv", lambda *args, **kwargs: [])
    config = SimpleNamespace(research=SimpleNamespace(topic="image forensics", reference_papers=()),
                             llm=SimpleNamespace(s2_api_key=""), web_search=SimpleNamespace(enabled=False))
    result = executor._execute_literature_collect(tmp_path, tmp_path, config, None)
    assert result.status == StageStatus.DONE
    assert calls == [tuple(name for name in SOURCES if name in available)]
    assert "site_status.json" in result.artifacts
    assert json.loads((tmp_path / "search_meta.json").read_text())["real_search"]
