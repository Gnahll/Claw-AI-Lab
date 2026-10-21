from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from researchclaw.pipeline import runner as rc_runner
from researchclaw.pipeline.executor import _bibliography_entries, _escape_bib_specials

BIB = r"""@article{mareen2023harmonizing,
  title = {Harmonizing Image Forgery Detection & Localization: Fusion of Complementary Approaches},
  author = {Hannes Mareen and Louis De Neve},
  year = {2023},
  journal = {Computers, materials & continua},
  doi = {10.3390/jimaging10010004},
  url = {https://example.org/paper?a=1&b=2#section%20one},
}

@inproceedings{done2024,
  title={Already \& escaped {100\%} and {Nested & braces}},
  booktitle = "Proc. of A & B #1",
  note = {50% of runs},
}
"""


def test_escapes_bare_specials_in_text_fields():
    out = _escape_bib_specials(BIB)
    assert r"Detection \& Localization" in out
    assert r"materials \& continua" in out
    assert r"{Nested \& braces}" in out
    assert r'"Proc. of A \& B \#1"' in out
    assert r"{50\% of runs}" in out


def test_does_not_double_escape_or_touch_urls():
    out = _escape_bib_specials(BIB)
    assert r"Already \& escaped {100\%}" in out
    assert r"\\&" not in out and r"\\%" not in out
    assert "url = {https://example.org/paper?a=1&b=2#section%20one}" in out
    assert "doi = {10.3390/jimaging10010004}" in out
    assert _escape_bib_specials(out) == out


def test_entries_still_parse_after_escaping():
    entries = _bibliography_entries(_escape_bib_specials(BIB))
    assert set(entries) == {"mareen2023harmonizing", "done2024"}


def test_deliverables_bib_is_escaped(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    verify_dir = run_dir / "stage-26"
    verify_dir.mkdir(parents=True)
    (verify_dir / "references_verified.bib").write_text(BIB, encoding="utf-8")
    (run_dir / "stage-25").mkdir()
    config = SimpleNamespace(export=SimpleNamespace(target_conference="neurips_2025"))
    try:
        rc_runner._package_deliverables(run_dir, "run-bib", config)
    except Exception:  # noqa: BLE001 - later packaging steps are irrelevant here
        pass
    text = (run_dir / "deliverables" / "references.bib").read_text(encoding="utf-8")
    assert r"Detection \& Localization" in text
    assert "a=1&b=2" in text
