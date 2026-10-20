from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.literature import verify
from researchclaw.literature.models import Paper
from researchclaw.literature.verify import CitationResult, VerifyStatus
from researchclaw.pipeline import executor as rc_executor

from tests.test_rc_executor import FakeLLMClient, adapters, rc_config, run_dir  # noqa: F401

BIB = """@article{wen2016coverage,
  title = {COVERAGE: A Novel Database for Copy-Move Forgery Detection},
  author = {Wen, Bihan},
  doi = {10.1000/wrong},
  year = {2016},
}
"""


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify, "_read_cache", lambda title: None)
    monkeypatch.setattr(verify, "_write_cache", lambda title, result: None)
    monkeypatch.setattr(verify.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(verify, "verify_by_arxiv_id", lambda *a, **k: None)


def _result(status: VerifyStatus, method: str, confidence: float = 0.9, **kwargs) -> CitationResult:
    return CitationResult("", "t", status, confidence, method, **kwargs)


def _mismatch(doi: str, title: str) -> CitationResult:
    return _result(VerifyStatus.SUSPICIOUS, "doi", 0.0, details="DOI resolves but title mismatch (sim=0.00): 'Other'")


def test_doi_mismatch_falls_back_to_title_search(monkeypatch: pytest.MonkeyPatch) -> None:
    matched = Paper(paper_id="W1", title="COVERAGE", doi="10.1109/icip.2016.7532339")
    monkeypatch.setattr(verify, "verify_by_doi", _mismatch)
    monkeypatch.setattr(verify, "verify_by_openalex", lambda title: _result(
        VerifyStatus.VERIFIED, "openalex", 1.0, details="Confirmed via OpenAlex", matched_paper=matched))

    result = verify.verify_citations(BIB).results[0]

    assert result.status == VerifyStatus.VERIFIED
    assert result.method == "openalex+doi_mismatch"
    assert result.bibtex_doi == "10.1000/wrong"
    assert result.corrected_doi == "10.1109/icip.2016.7532339"
    assert result.to_dict()["corrected_doi"] == "10.1109/icip.2016.7532339"


def test_doi_mismatch_without_title_match_stays_suspicious(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify, "verify_by_doi", _mismatch)
    monkeypatch.setattr(verify, "verify_by_openalex", lambda title: _result(VerifyStatus.HALLUCINATED, "openalex"))
    monkeypatch.setattr(verify, "verify_by_title_search", lambda *a, **k: _result(VerifyStatus.HALLUCINATED, "title_search"))

    report = verify.verify_citations(BIB)

    assert report.results[0].status == VerifyStatus.SUSPICIOUS
    assert report.results[0].bibtex_doi == "10.1000/wrong"
    assert report.suspicious == 1


def test_unindexed_paper_is_verified_from_its_own_page(monkeypatch: pytest.MonkeyPatch) -> None:
    bib = """@inproceedings{zhang2026collaborative,
  title = {Collaborative Transformers with Multi-Level Forensic Attention
           for Image Manipulation Localization},
  url = {https://ojs.aaai.org/index.php/AAAI/article/view/38250},
}
"""
    monkeypatch.setattr(verify, "verify_by_openalex", lambda title: None)
    monkeypatch.setattr(verify, "verify_by_title_search", lambda *a, **k: _result(VerifyStatus.HALLUCINATED, "title_search"))

    class _Page:
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, limit: int) -> bytes:
            return (b"<html><h1 class='page_title'>Collaborative Transformers with "
                    b"Multi-Level Forensic Attention for Image Manipulation Localization</h1></html>")

    seen: list[str] = []
    monkeypatch.setattr(verify.urllib.request, "urlopen", lambda req, timeout: seen.append(req.full_url) or _Page())

    result = verify.verify_citations(bib).results[0]

    assert result.status == VerifyStatus.VERIFIED
    assert result.method == "url"
    assert seen == ["https://ojs.aaai.org/index.php/AAAI/article/view/38250"]


def test_cached_negative_verdict_is_checked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify, "_read_cache", lambda title: _result(VerifyStatus.HALLUCINATED, "title_search"))
    monkeypatch.setattr(verify, "verify_by_doi", lambda doi, title: _result(VerifyStatus.VERIFIED, "doi"))

    report = verify.verify_citations(BIB)

    assert report.verified == 1
    assert report.hallucinated == 0


def test_openalex_query_has_no_filter_separators(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def _fail(req, timeout):
        urls.append(req.full_url)
        raise OSError("offline")

    monkeypatch.setattr(verify.urllib.request, "urlopen", _fail)
    verify.verify_by_openalex("Attention, Please: A Survey")
    query = verify.urllib.parse.parse_qs(verify.urllib.parse.urlparse(urls[0]).query)
    assert "," not in query["filter"][0].removeprefix("title.search:")
    assert ":" not in query["filter"][0].removeprefix("title.search:")


def test_correct_bibtex_dois_replaces_or_drops_field() -> None:
    fixed = rc_executor._correct_bibtex_dois(BIB, {"wen2016coverage": "10.1109/icip.2016.7532339"})
    assert "doi = {10.1109/icip.2016.7532339}," in fixed
    assert "10.1000/wrong" not in fixed

    dropped = rc_executor._correct_bibtex_dois(BIB, {"wen2016coverage": ""})
    assert "doi" not in dropped
    assert "year = {2016}" in dropped


def test_citation_verify_corrects_doi_and_keeps_original(
    run_dir: Path, rc_config: RCConfig, adapters: AdapterBundle, monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    (run_dir / "stage-25").mkdir(parents=True)
    (run_dir / "stage-25" / "references.bib").write_text(BIB, encoding="utf-8")
    (run_dir / "stage-25" / "paper_final.md").write_text("# Paper\n\n[wen2016coverage]", encoding="utf-8")
    report = verify.VerificationReport(total=1, verified=1, results=[CitationResult(
        "wen2016coverage", "COVERAGE", VerifyStatus.VERIFIED, 1.0, "openalex+doi_mismatch",
        bibtex_doi="10.1000/wrong", corrected_doi="10.1109/icip.2016.7532339",
    )])
    monkeypatch.setattr(verify, "verify_citations", lambda *args, **kwargs: report)
    stage_dir = run_dir / "stage-26"
    stage_dir.mkdir()

    rc_executor._execute_citation_verify(stage_dir, run_dir, rc_config, adapters)

    verified_bib = (stage_dir / "references_verified.bib").read_text(encoding="utf-8")
    assert "10.1109/icip.2016.7532339" in verified_bib
    assert "10.1000/wrong" not in verified_bib
    assert (stage_dir / "references_original.bib").read_text(encoding="utf-8") == BIB
    data = json.loads((stage_dir / "verification_report.json").read_text(encoding="utf-8"))
    assert data["summary"]["corrected_doi_keys"] == ["wen2016coverage"]
    assert data["results"][0]["bibtex_doi"] == "10.1000/wrong"


def test_latex_package_keeps_collected_bibliography(
    run_dir: Path, rc_config: RCConfig,  # noqa: F811
) -> None:
    (run_dir / "stage-04").mkdir(parents=True)
    (run_dir / "stage-04" / "references.bib").write_text(BIB, encoding="utf-8")
    stage_dir = run_dir / "stage-22"
    stage_dir.mkdir()
    llm = FakeLLMClient(
        "```latex\n\\documentclass{article}\n\\begin{document}\n\\cite{wen2016coverage}\n"
        "\\end{document}\n```\n```bib\n@article{invented2025,\n  title={Made Up},\n}\n```\n"
    )

    rc_executor._generate_latex_package(stage_dir, run_dir, rc_config, "# Paper\n\n[wen2016coverage]", llm=llm)

    assert (stage_dir / "latex_package" / "references.bib").read_text(encoding="utf-8") == BIB
