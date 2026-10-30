"""S12 SANITY_CHECK runtime — orchestrates the agentic smoke test loop.

Follows the same pattern as S11's ClawAgentStrategy:
workspace preparation → system prompt → turn loop → result.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from researchclaw.config import RCConfig
from researchclaw.adapters import AdapterBundle
from researchclaw.pipeline.claw_engine import AgentTurnLoop, StageSession
from researchclaw.pipeline.claw_engine.turn_loop import TurnResult
from researchclaw.pipeline.sanity_check.system_prompt import (
    build_system_prompt,
    build_user_message,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw import single_run

logger = logging.getLogger(__name__)


def _run_checked_smoke(workspace: Path, output_dir: Path, config: RCConfig, plan_text: str) -> dict:
    """Do not accept a smoke repair that replaces mandatory checkers with stubs."""
    from researchclaw.pipeline.codegen.readiness import declared_interface_gaps
    from researchclaw.training.execution import run_experiment
    files = {str(p.relative_to(workspace)): p.read_text(encoding="utf-8")
             for p in workspace.rglob("*.py") if not p.is_symlink()
             and not any(part.startswith('.') or part in ('outputs', '__pycache__')
                         for part in p.relative_to(workspace).parts)}
    gaps = declared_interface_gaps(files, plan_text,
        require_training=bool(getattr(getattr(config, "experiment", None), "require_training", False)))
    if not gaps:
        return run_experiment(workspace, output_dir, config, smoke=True, timeout=180)
    report = {"success": False, "smoke": True, "require_training": False,
              "execution_started": False, "results": None,
              "errors": ["Implementation incomplete before smoke: " + gap for gap in gaps]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "execution.log").write_text("\n".join(report["errors"]), encoding="utf-8")
    (output_dir / "execution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _iter_plan_sources(run_dir: Path):
    """Yield candidate paths for the experiment plan, best match first."""
    # S11 claw_workspace may contain the LLM-created EXPERIMENT_PLAN.yaml
    s11 = run_dir / "stage-11"
    if s11.is_dir():
        for ws_dir in sorted(s11.glob("claw_workspace_*"), reverse=True):
            p = ws_dir / "EXPERIMENT_PLAN.yaml"
            if p.is_file():
                yield p
    # S9 canonical exp_plan.yaml
    s9_plan = run_dir / "stage-09" / "exp_plan.yaml"
    if s9_plan.is_file():
        yield s9_plan


class SanityCheckRuntime:
    """Orchestration for S12 SANITY_CHECK using claw-code turn loop."""

    def execute(
        self,
        stage_dir: Path,
        run_dir: Path,
        config: RCConfig,
        adapters: AdapterBundle,
        *,
        llm: Any | None = None,
    ) -> Any:
        from researchclaw.pipeline.executor import StageResult

        stage_dir.mkdir(parents=True, exist_ok=True)
        session = StageSession(stage_dir=stage_dir, stage_name="sanity_check")
        session.log("INIT", "SanityCheckRuntime started")

        from researchclaw.pipeline.acceptance import check_plan_file, write_plan_blocker
        plan = run_dir / "stage-09" / "exp_plan.yaml"
        if plan.is_file():
            issues = check_plan_file(plan, run_dir, config)
            if issues:
                error = write_plan_blocker(stage_dir, issues)
                return StageResult(stage=Stage.SANITY_CHECK, status=StageStatus.FAILED,
                                   artifacts=("acceptance_review.json",), decision="revise_design", error=error)

        # Resolve coding LLM
        llm = self._resolve_coding_llm(llm, config)
        if llm is None:
            session.log_error("INIT", "No LLM client available")
            return StageResult(
                stage=Stage.SANITY_CHECK,
                status=StageStatus.FAILED,
                artifacts=(),
                error="No LLM client available",
            )

        llm_config = llm.config
        session.log("INIT", f"LLM: {llm_config.primary_model}")

        # Find experiment directory from prior stages
        experiment_dir = self._find_experiment_dir(run_dir)
        if not experiment_dir:
            session.log_error("INIT", "No experiment directory found")
            return StageResult(
                stage=Stage.SANITY_CHECK,
                status=StageStatus.FAILED,
                artifacts=(),
                error="No experiment/ directory found in prior stages",
            )
        session.log("INIT", f"Experiment dir: {experiment_dir}")

        # Prepare workspace (copy experiment into a fresh workspace)
        workspace = self._prepare_workspace(stage_dir, experiment_dir, run_dir, config)
        session.log("INIT", f"Workspace: {workspace}")

        # List experiment files
        exp_files = [
            str(f.relative_to(workspace))
            for f in sorted(workspace.rglob("*.py"))
            if not f.is_symlink()
            and not any(p.startswith(".") or p == "__pycache__" for p in f.relative_to(workspace).parts)
        ]
        session.log("INIT", f"Experiment files: {exp_files}")

        # Build prompts
        python_path = getattr(config.experiment.sandbox, "python_path", "") or ""
        system_prompt = build_system_prompt(
            python_path=python_path,
            workspace_path=str(workspace),
            metric_key=config.experiment.metric_key,
        )

        exp_plan_summary = self._load_plan_summary(run_dir)
        user_message = build_user_message(
            experiment_dir=str(workspace),
            experiment_files=exp_files,
            exp_plan_summary=exp_plan_summary,
            python_path=python_path,
        )

        _single = single_run.is_single_run(config)
        system_prompt = single_run.apply(system_prompt, _single, append_policy=True)
        user_message = single_run.apply(user_message, _single)
        # Save system prompt for debugging
        (stage_dir / "sanity_system_prompt.md").write_text(system_prompt, encoding="utf-8")
        session.add_artifact("sanity_system_prompt.md")

        # Build allowed read dirs
        allowed_reads = self._build_allowed_reads(config)

        max_iters = getattr(config.experiment, "sanity_check_max_iterations", 18)
        session.log("INIT", f"Max iterations: {max_iters}")

        # Create and run turn loop
        loop = AgentTurnLoop(
            llm_config=llm_config,
            workspace=workspace,
            system_prompt=system_prompt,
            session=session,
            allowed_read_dirs=allowed_reads,
            bash_timeout=180,
            max_iterations=min(max_iters, 30),
            python_path=python_path,
            trace_prefix="sanity",
        )

        plan_text = plan.read_text(encoding="utf-8") if plan.is_file() else ""
        evidence_dir = stage_dir / "smoke_execution"
        session.log("EXECUTE", "Running main.py under pipeline supervision before repair")
        attempt_dir = evidence_dir / "attempt-1"
        measured_smoke = _run_checked_smoke(workspace, attempt_dir, config, plan_text)
        attempts = [measured_smoke]
        turn_result = TurnResult()
        if not measured_smoke["success"]:
            failure_log = (attempt_dir / "execution.log").read_text(encoding="utf-8", errors="replace")[-12000:]
            session.log("REPAIR", "Supervised smoke failed; sending measured errors to the repair agent")
            turn_result = loop.run_turn(
                user_message
                + "\nSupervised execution failed. Repair main.py or its helpers; the pipeline will rerun them.\n"
                + json.dumps(measured_smoke["errors"])
                + "\nExecution log:\n"
                + failure_log
            )
            if turn_result.errors:
                session.log("REPAIR", "Repair failed; preserving original failure without rerunning: " + "; ".join(turn_result.errors))
            else:
                attempt_dir = evidence_dir / "attempt-2"
                measured_smoke = _run_checked_smoke(workspace, attempt_dir, config, plan_text)
                attempts.append(measured_smoke)

        for filename in ("execution.log", "execution_report.json"):
            shutil.copy2(attempt_dir / filename, evidence_dir / filename)
        success = measured_smoke["success"]
        fixed_count = self._copy_fixes_back(workspace, experiment_dir, session) if success else 0
        session.log(
            "RESULT",
            f"Sanity check {'PASSED' if success else 'FAILED'}: "
            f"{turn_result.iterations} iters, {turn_result.tool_calls} tool calls, "
            f"{fixed_count} files fixed, {turn_result.elapsed_sec:.1f}s",
        )

        # Save report
        report = {
            "passed": success,
            "scope": "Engineering smoke only; not scientific acceptance or completed training",
            "scientific_acceptance_evaluated": False,
            "execution": measured_smoke,
            "attempts": attempts,
            "iterations": turn_result.iterations,
            "tool_calls": turn_result.tool_calls,
            "files_fixed": fixed_count,
            "errors": [*turn_result.errors, *measured_smoke["errors"]],
            "elapsed_sec": round(turn_result.elapsed_sec, 1),
        }
        (stage_dir / "sanity_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        session.add_artifact("sanity_report.json")

        return StageResult(
            stage=Stage.SANITY_CHECK,
            status=StageStatus.DONE if success else StageStatus.FAILED,
            artifacts=("sanity_report.json",),
            error=None if success else "Smoke test did not pass: " + "; ".join(measured_smoke["errors"]),
            decision="proceed" if success else "retry",
            evidence_refs=("experiment/",),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_coding_llm(llm: Any, config: RCConfig) -> Any:
        """Resolve to coding model if configured."""
        coding_model = getattr(config.llm, "coding_model", "") or ""
        if not coding_model or not llm:
            return llm

        if hasattr(llm, "config") and llm.config.primary_model == coding_model:
            return llm

        try:
            from researchclaw.llm import create_llm_client
            import dataclasses
            new_llm_cfg = dataclasses.replace(config.llm, primary_model=coding_model)
            new_config = dataclasses.replace(config, llm=new_llm_cfg)
            return create_llm_client(new_config)
        except Exception:
            return llm

    @staticmethod
    def _find_experiment_dir(run_dir: Path) -> Path | None:
        """Find the latest experiment/ directory from prior stages."""
        candidates = []
        for stage_d in sorted(run_dir.glob("stage-*"), reverse=True):
            exp_dir = stage_d / "experiment"
            if exp_dir.is_dir() and (exp_dir / "main.py").is_file():
                candidates.append(exp_dir)
        return candidates[0] if candidates else None

    @staticmethod
    def _prepare_workspace(
        stage_dir: Path,
        experiment_dir: Path,
        run_dir: Path,
        config: RCConfig,
    ) -> Path:
        """Create workspace with experiment code + symlinks to data."""
        ws = stage_dir / f"sanity_workspace_{int(time.time())}_{os.getpid()}"
        shutil.copytree(
            experiment_dir, ws,
            ignore=shutil.ignore_patterns("__pycache__", ".snapshots", "*.pyc"),
            dirs_exist_ok=True,
        )

        existing_project_dir = getattr(config.experiment, "existing_project_dir", "") or ""
        if existing_project_dir and Path(existing_project_dir).is_dir():
            link = ws / "project"
            if not link.exists():
                try:
                    link.symlink_to(Path(existing_project_dir).resolve())
                except OSError:
                    pass

        # Symlink datasets
        datasets_dir = getattr(config.experiment, "datasets_dir", "") or ""
        if datasets_dir and Path(datasets_dir).is_dir():
            link = ws / "datasets"
            if not link.exists():
                try:
                    link.symlink_to(Path(datasets_dir).resolve())
                except OSError:
                    pass

        # Symlink checkpoints
        checkpoints_dir = getattr(config.experiment, "checkpoints_dir", "") or ""
        if checkpoints_dir and Path(checkpoints_dir).is_dir():
            link = ws / "checkpoints"
            if not link.exists():
                try:
                    link.symlink_to(Path(checkpoints_dir).resolve())
                except OSError:
                    pass

        # Symlink codebases
        codebases_dir = getattr(config.experiment, "codebases_dir", "") or ""
        if codebases_dir and Path(codebases_dir).is_dir():
            link = ws / "codebases"
            if not link.exists():
                try:
                    link.symlink_to(Path(codebases_dir).resolve())
                except OSError:
                    pass

        # Copy experiment plan into workspace if not already present.
        # S11-generated code often references EXPERIMENT_PLAN.yaml; the plan
        # originates from S9 (exp_plan.yaml) or from S11's claw_workspace.
        plan_names = ("EXPERIMENT_PLAN.yaml", "exp_plan.yaml")
        has_plan = any((ws / n).exists() for n in plan_names)
        if not has_plan:
            for source in _iter_plan_sources(run_dir):
                target = ws / "EXPERIMENT_PLAN.yaml"
                try:
                    shutil.copy2(source, target)
                    break
                except OSError:
                    pass

        # Ensure outputs dir
        (ws / "outputs").mkdir(exist_ok=True)
        return ws

    @staticmethod
    def _load_plan_summary(run_dir: Path) -> str:
        """Load experiment plan summary from prior stages."""
        for stage_d in sorted(run_dir.glob("stage-*"), reverse=True):
            plan_file = stage_d / "exp_plan.yaml"
            if plan_file.is_file():
                try:
                    text = plan_file.read_text(encoding="utf-8")
                    return text[:2000]
                except OSError:
                    pass
        return ""

    @staticmethod
    def _build_allowed_reads(config: RCConfig) -> list[Path]:
        dirs: list[Path] = []
        for attr in ("existing_project_dir", "datasets_dir", "checkpoints_dir", "codebases_dir"):
            d = getattr(config.experiment, attr, "") or ""
            if d and Path(d).is_dir():
                dirs.append(Path(d))
        return dirs

    @staticmethod
    def _copy_fixes_back(workspace: Path, experiment_dir: Path, session: StageSession) -> int:
        """Copy fixed .py files back to the original experiment directory."""
        count = 0
        for py_file in workspace.rglob("*.py"):
            if py_file.is_symlink():
                continue
            rel = py_file.relative_to(workspace)
            if any(p.startswith(".") or p == "__pycache__" or p == "codebases" for p in rel.parts):
                continue
            dest = experiment_dir / rel
            try:
                new_content = py_file.read_text(encoding="utf-8")
                old_content = dest.read_text(encoding="utf-8") if dest.exists() else ""
                if new_content != old_content:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(new_content, encoding="utf-8")
                    count += 1
                    session.log("FIX", f"Updated {rel}")
            except OSError:
                pass
        return count

    @staticmethod
    def _check_success(
        turn_result: Any,
        workspace: Path,
        max_iterations: int = 30,
    ) -> bool:
        """Determine if the smoke test passed.

        Success criteria (in order):
        1. LLM final text explicitly declares the test passed → True
        2. LLM-level errors were recorded → False
        3. Loop was cut off at max_iterations (LLM still working) → False
        4. LLM final text contains failure language → False
        5. LLM stopped voluntarily (no more tool calls) with no errors → True
        """
        final = turn_result.final_text.lower()

        _PASS_PHRASES = (
            "smoke test pass", "sanity check pass", "test passed",
            "completed successfully", "all checks pass",
            "exit code 0", "smoke_test passed", "tests pass",
            "all tests pass", "successfully completed",
        )
        if any(phrase in final for phrase in _PASS_PHRASES):
            return True

        if turn_result.errors:
            return False

        if turn_result.iterations >= max_iterations:
            return False

        _FAIL_PHRASES = (
            "fail", "error", "traceback", "not pass", "does not pass",
            "exit code 1", "cannot", "unable to fix",
        )
        if any(phrase in final for phrase in _FAIL_PHRASES):
            return False

        if turn_result.iterations >= 2:
            return True

        return False
