from pathlib import Path

from researchclaw.pipeline.claw_engine.turn_loop import TurnResult
from researchclaw.pipeline.experiment_run import runtime as rt
from researchclaw.pipeline.experiment_run.runtime import ExperimentRunRuntime


class _Session:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, phase: str, message: str) -> None:
        self.lines.append(message)


class _Loop:
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        self.messages: list[str] = []

    def run_turn(self, message: str) -> TurnResult:
        self.messages.append(message)
        return TurnResult(errors=list(self.errors))


def _fake_runs(monkeypatch, outcomes):
    calls: list[str] = []

    def fake(workspace, evidence_dir, config, **kwargs):
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "execution.log").write_text("log", encoding="utf-8")
        calls.append(evidence_dir.name)
        return outcomes[len(calls) - 1]

    monkeypatch.setattr(rt, "run_experiment", fake)
    return calls


def test_failed_repair_turn_keeps_attempt1_error_and_skips_rerun(tmp_path: Path, monkeypatch) -> None:
    calls = _fake_runs(monkeypatch, [{"success": False, "errors": ["missing outputs/model.pt"]}])
    loop = _Loop(["LLM call failed after 5 retries at iteration 2: HTTP Error 400"])

    run, _ = ExperimentRunRuntime._supervised_run(tmp_path, tmp_path, None, loop, "msg", _Session())

    assert calls == ["attempt-1"]
    assert not run["success"]
    assert run["errors"][0] == "missing outputs/model.pt"
    assert "HTTP Error 400" in run["errors"][1]


def test_second_failure_reports_both_attempts(tmp_path: Path, monkeypatch) -> None:
    calls = _fake_runs(monkeypatch, [
        {"success": False, "errors": ["missing outputs/model.pt"]},
        {"success": False, "errors": ["No actual PyTorch parameter update observed"]},
    ])

    run, _ = ExperimentRunRuntime._supervised_run(tmp_path, tmp_path, None, _Loop([]), "msg", _Session())

    assert calls == ["attempt-1", "attempt-2"]
    assert run["errors"] == ["No actual PyTorch parameter update observed", "attempt-1: missing outputs/model.pt"]


def test_resumed_repair_forbids_retraining_finished_conditions(tmp_path: Path, monkeypatch) -> None:
    _fake_runs(monkeypatch, [{"success": False, "errors": ["x"]}, {"success": True, "errors": []}])
    loop = _Loop([])

    run, _ = ExperimentRunRuntime._supervised_run(tmp_path, tmp_path, None, loop, "msg", _Session(), resumed=True)

    assert run["success"]
    assert "do not change the code to retrain conditions that already finished" in loop.messages[0]


def test_first_success_needs_no_repair(tmp_path: Path, monkeypatch) -> None:
    calls = _fake_runs(monkeypatch, [{"success": True, "errors": []}])
    loop = _Loop([])

    run, _ = ExperimentRunRuntime._supervised_run(tmp_path, tmp_path, None, loop, "msg", _Session())

    assert run["success"] and calls == ["attempt-1"] and loop.messages == []
