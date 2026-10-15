from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from researchclaw.templates import compiler


def test_tectonic_compiles_with_bibliography_in_one_invocation(tmp_path, monkeypatch):
    tex_path = tmp_path / "paper.tex"
    tex_path.write_text("test", encoding="utf-8")
    monkeypatch.setattr(compiler.shutil, "which", lambda name: "/bin/tectonic" if name == "tectonic" else None)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["cwd"] == tmp_path
        tex_path.with_suffix(".pdf").write_bytes(b"%PDF-1.7 generated")
        return SimpleNamespace(returncode=0, stdout="compiled", stderr="")

    monkeypatch.setattr(compiler.subprocess, "run", run)
    assert compiler.compile_latex(tex_path).success
    assert commands == [["/bin/tectonic", "-X", "compile", "paper.tex", "--keep-logs", "--keep-intermediates"]]


def test_project_local_tectonic_is_discovered_without_path(tmp_path, monkeypatch):
    module_path = tmp_path / "backend/agent/researchclaw/templates/compiler.py"
    module_path.parent.mkdir(parents=True)
    executable = tmp_path / ".runtime/tectonic"
    executable.parent.mkdir()
    executable.write_bytes(b"test")
    tex_path = tmp_path / "paper.tex"
    tex_path.write_text("test", encoding="utf-8")
    monkeypatch.setattr(compiler, "__file__", str(module_path))
    monkeypatch.setattr(compiler.shutil, "which", lambda name: None)

    def run(command, **kwargs):
        assert command[0] == str(executable)
        tex_path.with_suffix(".pdf").write_bytes(b"%PDF-1.7 generated")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(compiler.subprocess, "run", run)
    assert compiler.compile_latex(tex_path).success


@pytest.mark.parametrize("outcome", ["missing_pdf", "timeout", "later_pass_failure"])
def test_failed_compile_never_reuses_an_old_pdf(tmp_path, monkeypatch, outcome):
    tex_path = tmp_path / "paper.tex"
    tex_path.write_text("test", encoding="utf-8")
    pdf_path = tex_path.with_suffix(".pdf")
    pdf_path.write_bytes(b"%PDF-1.7 old")
    monkeypatch.setattr(compiler.shutil, "which", lambda name: "/bin/pdflatex" if name == "pdflatex" else None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 120)
        if outcome == "later_pass_failure":
            pdf_path.write_bytes(b"%PDF-1.7 incomplete")
            return SimpleNamespace(returncode=0 if len(calls) == 1 else 1, stdout="", stderr="! Fatal error")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(compiler.subprocess, "run", run)
    result = compiler.compile_latex(tex_path)
    assert not result.success
    assert result.errors
    assert not pdf_path.exists()
