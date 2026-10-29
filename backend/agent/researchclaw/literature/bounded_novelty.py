"""Keep optional network novelty checks from hanging completed hypotheses."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from typing import Any


def _incomplete(reason: str) -> dict[str, Any]:
    return {"status": "incomplete", "reason": reason, "search_coverage": "insufficient",
            "novelty_score": None, "assessment": "insufficient_data", "similar_papers": [],
            "recommendation": "verify_novelty_before_claiming_it",
            "limitation": "Optional novelty search did not complete; this is not evidence of novelty."}


def _bounded_json(command: list[str], payload: dict, timeout_sec: float) -> dict:
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError("Novelty timeout must be positive and finite")
    try:
        # subprocess.run kills and waits for this dedicated worker on timeout.
        # The worker performs HTTP calls, never starts training or subprocesses.
        response = subprocess.run(command, input=json.dumps(payload).encode("utf-8"),
                                  capture_output=True, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired:
        return _incomplete(f"novelty_search_timeout_after_{timeout_sec:g}s")
    except OSError:
        return _incomplete("novelty_search_worker_unavailable")
    if response.returncode:
        return _incomplete(f"novelty_search_worker_exit_{response.returncode}")
    try:
        report = json.loads(response.stdout)
    except (ValueError, UnicodeError):
        return _incomplete("invalid_novelty_search_report")
    if not isinstance(report, dict) or "novelty_score" not in report or "search_coverage" not in report:
        return _incomplete("invalid_novelty_search_report")
    if report["search_coverage"] == "insufficient":
        report.update(novelty_score=None, status="incomplete", assessment="insufficient_data",
                      recommendation="verify_novelty_before_claiming_it")
    else:
        report["status"] = "completed"
    return report


def check_novelty_bounded(*, timeout_sec: float = 120, **kwargs: Any) -> dict:
    """Credentials travel through stdin, never command arguments or the report."""
    return _bounded_json([sys.executable, "-m", "researchclaw.literature.bounded_novelty"], kwargs, timeout_sec)


def main() -> None:
    from contextlib import redirect_stdout
    from researchclaw.literature.novelty import check_novelty

    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    # API libraries may print rate-limit notices; keep the result channel JSON.
    with redirect_stdout(sys.stderr):
        result = check_novelty(**payload)
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))


if __name__ == "__main__":
    main()
