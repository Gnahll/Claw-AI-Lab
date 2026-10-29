import json
from types import SimpleNamespace

import pytest

from researchclaw import data
from researchclaw.literature import search, arxiv_client
from researchclaw.pipeline import executor
from researchclaw.pipeline.stages import StageStatus


def config():
    return SimpleNamespace(research=SimpleNamespace(topic="图像篡改定位与多尺度注意力 image forgery localization",
        reference_papers=(), domains=(), quality_threshold=0.7),
        llm=SimpleNamespace(s2_api_key=""), web_search=SimpleNamespace(enabled=False))


def offline_search(monkeypatch):
    monkeypatch.setattr(executor, "_check_literature_sites_reachable", lambda **kw: {"arxiv": True})
    monkeypatch.setattr(search, "search_papers_multi_query", lambda *a, **kw: [])
    monkeypatch.setattr(arxiv_client, "search_arxiv", lambda *a, **kw: [])


def test_seed_library_cannot_disguise_empty_search_or_create_placeholders(tmp_path, monkeypatch):
    offline_search(monkeypatch)
    monkeypatch.setattr(data, "load_seminal_papers", lambda topic: [{"title": "OpenVLA", "keywords": ["open source"]}])
    result = executor._execute_literature_collect(tmp_path, tmp_path, config(), None)
    assert result.status == StageStatus.FAILED and result.decision == "retry"
    assert (tmp_path / "candidates.jsonl").read_text() == ""
    meta = json.loads((tmp_path / "search_meta.json").read_text())
    assert not meta["real_search"] and meta["background_count"] == 1
    assert json.loads((tmp_path / "background_references.json").read_text())[0]["title"] == "OpenVLA"
    assert "Placeholder" not in (tmp_path / "literature_error.txt").read_text()


def test_user_reference_is_recorded_without_claiming_online_retrieval(tmp_path, monkeypatch):
    offline_search(monkeypatch)
    monkeypatch.setattr(data, "load_seminal_papers", lambda topic: [])
    monkeypatch.setattr(executor, "_resolve_user_reference", lambda ref: {
        "title": "Image forgery localization", "source": "user_reference", "authors": [], "year": 2025})
    cfg = config()
    cfg.research.reference_papers = ("reference-doi",)
    result = executor._execute_literature_collect(tmp_path, tmp_path, cfg, None)
    assert result.status == StageStatus.DONE
    meta = json.loads((tmp_path / "search_meta.json").read_text())
    assert not meta["real_search"] and meta["source_counts"] == {"user_reference": 1}


@pytest.mark.parametrize("topic", ["图像篡改定位 Cross-Branch Attention", "视频分割 video attention", "视觉语言模型 VLM 检测"])
def test_fallback_queries_never_append_robotics_to_unrelated_topics(topic):
    queries = executor._build_fallback_queries(topic)
    assert queries and all("robot" not in query.lower() for query in queries)


def test_foundational_matching_requires_whole_keywords_not_short_acronym_substrings(monkeypatch):
    monkeypatch.setattr(data, "_load_all", lambda: [
        {"cite_key": "short", "keywords": ["BN", "LN", "RL"]},
        {"cite_key": "robot", "keywords": ["robot"]},
        {"cite_key": "attention", "keywords": ["attention"]}])
    assert [p["cite_key"] for p in data.load_seminal_papers("learn attention without robotic control")] == ["attention"]


def screen(tmp_path, monkeypatch, candidates, assessments=None, use_llm=True):
    stage4 = tmp_path / "stage-04"
    stage4.mkdir()
    (stage4 / "candidates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in candidates))
    stage5 = tmp_path / "stage-05"
    stage5.mkdir()
    monkeypatch.setattr(executor, "_chat_with_prompt", lambda *a, **kw:
                        SimpleNamespace(content=json.dumps({"shortlist": assessments or []})))
    result = executor._execute_literature_screen(stage5, tmp_path, config(), None,
                                               llm=object() if use_llm else None)
    output = [json.loads(line) for line in (stage5 / "shortlist.jsonl").read_text().splitlines()]
    return result, output


def test_all_irrelevant_candidates_are_not_restored(tmp_path, monkeypatch):
    result, rows = screen(tmp_path, monkeypatch, [{"title": "OpenVLA robot policy", "abstract": "Robot manipulation"}])
    assert result.status == StageStatus.FAILED and not rows
    assert "unrelated papers are not restored" in result.error


@pytest.mark.parametrize("mode", ["empty", "invented", "no_llm", "low_relevance"])
def test_screening_never_fabricates_scores_or_uses_unassessed_papers(tmp_path, monkeypatch, mode):
    title = "Image forgery localization"
    assessed = [] if mode == "empty" else [{"title": "Invented study" if mode == "invented" else title,
                                            "relevance_score": 0.1 if mode == "low_relevance" else 0.9}]
    result, rows = screen(tmp_path, monkeypatch, [{"title": title, "source": "arxiv"}],
                          assessed, use_llm=mode != "no_llm")
    assert result.status == StageStatus.FAILED and not rows


def test_small_assessed_shortlist_is_not_padded_and_provenance_cannot_be_overwritten(tmp_path, monkeypatch):
    title = "Image forgery localization"
    candidates = [{"title": title, "source": "arxiv", "url": "https://arxiv.org/abs/1234"},
                  {"title": "Image tampering detection", "source": "openalex"}]
    assessed = [{"title": title, "source": "invented", "url": "https://example.org", "relevance_score": 0.9}]
    result, rows = screen(tmp_path, monkeypatch, candidates, assessed)
    assert result.status == StageStatus.DONE and len(rows) == 1
    assert rows[0]["source"] == "arxiv" and rows[0]["url"].startswith("https://arxiv.org")


def test_legacy_seeds_and_placeholders_cannot_reenter_screening(tmp_path, monkeypatch):
    result, rows = screen(tmp_path, monkeypatch, [
        {"title": "Image forgery localization", "source": "seminal_library"},
        {"title": "Image forgery placeholder", "is_placeholder": True}])
    assert result.status == StageStatus.FAILED and not rows
