import json
import sys
import time

from researchclaw.literature.bounded_novelty import _bounded_json


def worker(source):
    return [sys.executable, "-c", source]


def test_hung_optional_search_is_killed_and_never_reports_success():
    start = time.monotonic()
    report = _bounded_json(worker("import time; time.sleep(60)"), {}, .2)
    assert time.monotonic() - start < 5
    assert report["status"] == "incomplete" and report["novelty_score"] is None
    assert "timeout" in report["reason"]


def test_actual_worker_preserves_real_search_output():
    report = {"novelty_score": .42, "search_coverage": "full", "similar_papers": [{"title": "real match"}]}
    result = _bounded_json(worker("import json,sys; json.dump(json.load(sys.stdin),sys.stdout)"), report, 5)
    assert result["status"] == "completed" and result["similar_papers"] == report["similar_papers"]
    assert result["novelty_score"] == .42


def test_empty_search_is_not_perfect_novelty():
    result = _bounded_json(worker("print('{\"novelty_score\":1,\"search_coverage\":\"insufficient\"}')"), {}, 5)
    assert result["status"] == "incomplete" and result["novelty_score"] is None


def test_worker_errors_and_invalid_output_are_explicitly_incomplete():
    for code in ("raise SystemExit(1)", "print('not JSON')", "print('[]')"):
        result = _bounded_json(worker(code), {}, 5)
        assert result["status"] == "incomplete" and result["novelty_score"] is None


def test_s8_keeps_generated_hypotheses_when_optional_search_times_out(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from researchclaw.pipeline import executor
    from researchclaw.pipeline.stages import StageStatus
    from researchclaw.literature import bounded_novelty
    monkeypatch.setattr(bounded_novelty, "check_novelty_bounded", lambda **kw: {
        "novelty_score": None, "assessment": "insufficient_data", "recommendation": "verify_novelty",
        "status": "incomplete", "reason": "timeout", "search_coverage": "insufficient"})
    config = SimpleNamespace(research=SimpleNamespace(topic="Cross-branch image localization"),
                             experiment=SimpleNamespace(shared_results_dir=""), llm=SimpleNamespace(s2_api_key=""))
    result = executor._execute_hypothesis_gen(tmp_path, tmp_path, config, object(), llm=None)
    assert result.status == StageStatus.DONE
    assert (tmp_path / "hypotheses.md").stat().st_size > 0
    report = json.loads((tmp_path / "novelty_report.json").read_text())
    assert report["status"] == "incomplete" and report["novelty_score"] is None
