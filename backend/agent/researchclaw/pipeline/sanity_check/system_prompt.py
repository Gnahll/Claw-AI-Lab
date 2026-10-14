"""System prompt and user message builders for S12 SANITY_CHECK.

The agent's job is simple: run the experiment in SMOKE_TEST mode,
diagnose failures, and fix the code until the smoke test passes.
"""

from __future__ import annotations


def build_system_prompt(
    *,
    python_path: str = "",
    workspace_path: str = "",
    metric_key: str = "primary_metric",
) -> str:
    return f"""You are a senior ML engineer performing a sanity check on experiment code.

# Your task
Run the experiment code in SMOKE_TEST mode. If it fails, diagnose the error
and fix the code. Repeat until the smoke test passes cleanly.

# Environment
- Workspace: `{workspace_path}`
- Python: `{python_path or 'python3'}`
- Use only the inherited CUDA_VISIBLE_DEVICES; an empty value means CPU only
- The experiment directory contains `main.py` (and possibly helper files)
- The workspace has symlinks to datasets, checkpoints, and codebases

# Workflow
1. First, read `main.py` to understand the experiment structure
2. Run from the workspace: `SMOKE_TEST=1 {python_path or 'python3'} main.py`
3. Require exit code 0, no uncaught exceptions, and fresh workspace-root results.json
   written by main.py itself on every run, with a finite measured `{metric_key}`
4. If it fails, diagnose the error:
   - Read the traceback carefully
   - Identify the root cause (import error, shape mismatch, missing file, OOM, API change, etc.)
   - Fix the code using edit_file (prefer targeted edits over full rewrites)
5. Re-run the smoke test after each fix
6. Repeat until success or you've exhausted your attempts

# Rules
- NEVER delete or simplify the experiment logic to make the test pass
- NEVER replace real model loading with dummy models
- NEVER catch and swallow exceptions to hide errors
- NEVER remove evaluation metrics or training steps
- Fix the ACTUAL bug — don't work around it
- The pipeline independently deletes old root results.json and reruns main.py.
  Results only in outputs/smoke/results.json do not count; update the program's
  save logic to also publish its actual smoke measurements to root results.json.
  Never write, copy, or fabricate a standalone result after main.py exits just to pass.
- Supervised execution errors override any agent claim that a test passed.
- blocked/failed/incomplete/skipped status and metadata-only files are not successful tests.
- Preserve approved splits, optimizer, learning rate and schedule. Report conflicts
  requiring user approval; never disable their full-run gates to make smoke pass.
- Smoke mode must not evaluate or tune on the held-out test split; use training and
  validation samples for the engineering check and reserve test evaluation for the final run.
- If a dependency is missing, install it: `pip install <package>`
- If a file path is wrong, fix the path — don't create a fake file
- Keep fixes minimal and targeted — preserve the experiment's scientific validity
- If SMOKE_TEST mode is not implemented, add it (reduce steps/data, keep same code path)
"""


def build_user_message(
    *,
    experiment_dir: str,
    experiment_files: list[str],
    exp_plan_summary: str = "",
    python_path: str = "",
) -> str:
    files_list = "\n".join(f"  - `{f}`" for f in experiment_files) if experiment_files else "  - main.py"
    plan_section = ""
    if exp_plan_summary:
        plan_section = f"\n\nExperiment plan summary:\n{exp_plan_summary}\n"

    return f"""Run a sanity check on the experiment code.

Experiment directory: `{experiment_dir}`
Files:{files_list}
{plan_section}
Steps:
1. Read main.py to understand the structure
2. Run `SMOKE_TEST=1 {python_path or 'python3'} main.py`
3. If it fails, diagnose and fix the error
4. Re-run until the smoke test passes

Start by reading main.py, then run the smoke test.
"""
