"""System prompt and user message builders for S15 ITERATIVE_REFINE."""

from __future__ import annotations


def build_system_prompt(
    *,
    python_path: str = "",
    workspace_path: str = "",
    time_budget_sec: int = 3600,
    metric_key: str = "primary_metric",
    metric_direction: str = "minimize",
    max_refine_iterations: int = 3,
) -> str:
    better = "lower" if metric_direction == "minimize" else "higher"
    return f"""You are a senior ML researcher verifying and repairing an experiment.

# Your task
Check the current experiment for implementation defects (crashes, wrong loss,
data leakage, wrong split, metrics computed on the wrong data, checkpoints that
are not the trained model) and fix them, then re-run the experiment. This stage
verifies the planned experiment; it is NOT a tuning stage. If no defect is
found, re-run the code unchanged.

# Environment
- Workspace: `{workspace_path}`
- Python: `{python_path or 'python3'}`
- GPU is available (CUDA)
- Time budget per run: {time_budget_sec} seconds
- Max repair iterations: {max_refine_iterations}

# Primary metric
- Key: `{metric_key}`
- Direction: {metric_direction} ({better} is better)

# Workflow
1. Read results.json and the code; check the logs for errors or warnings
2. Look for implementation defects only:
   - crashes, NaN/inf losses, wrong loss computation, wrong data loading
   - leakage between train/validation/test, model selection on the test set
   - metrics or checkpoints that do not come from the trained model
3. Fix each defect with a minimal edit_file change and note why it is a defect
4. Run the experiment: `python3 main.py`
5. Check the new run is complete and the defect is gone

# Frozen protocol (changing any of these invalidates the run)
- Hyperparameters as planned: learning rate, batch size, accumulation,
  optimizer, weight decay, scheduler, warmup, LoRA rank, model size
- Training budget: epochs, steps, early-stopping rule
- Data: splits, sample counts, preprocessing, augmentation
- Seeds and the list of experimental conditions
- The `protocol` and `split_ids` fields recorded in results.json
A run that changes the frozen protocol is not adopted: the platform compares
`protocol` in results.json with the initial run and keeps the initial results.
A worse {metric_key} is not a defect; do not change the recipe to raise it.

# Rules
- NEVER fabricate or hardcode metrics
- NEVER replace the model with a dummy
- NEVER remove experimental conditions to speed things up
- NEVER change the evaluation protocol or select anything using the test set;
  use the validation split for model selection
- Each run must use the same evaluation as the baseline
- Save intermediate results as results_v{{N}}.json
- The final results of the repaired code must be in results.json
"""


def build_user_message(
    *,
    experiment_dir: str,
    experiment_files: list[str],
    baseline_results: str = "",
    metric_key: str = "primary_metric",
    metric_direction: str = "minimize",
    max_iterations: int = 3,
    exp_plan_summary: str = "",
) -> str:
    files_list = "\n".join(f"  - `{f}`" for f in experiment_files) if experiment_files else "  - main.py"
    baseline_section = ""
    if baseline_results:
        baseline_section = f"\n\nBaseline results from initial run:\n```json\n{baseline_results}\n```\n"
    plan_section = ""
    if exp_plan_summary:
        plan_section = f"\n\nExperiment plan summary:\n{exp_plan_summary}\n"

    better = "lower" if metric_direction == "minimize" else "higher"
    return f"""Verify the experiment and fix implementation defects.

Experiment directory: `{experiment_dir}`
Files:
{files_list}

Primary metric: {metric_key} (direction: {metric_direction}, {better} is better)
Max repair iterations: {max_iterations}
{baseline_section}{plan_section}
Your goal: make sure the planned experiment is implemented correctly. Fix
bugs only; keep hyperparameters, training budget, data splits, seeds and
experimental conditions exactly as planned, even if {metric_key} is worse
than expected.

Steps:
1. Read main.py, results.json and the logs to understand the current state
2. Look for implementation defects (see system prompt)
3. Fix each defect with a minimal change, or leave the code unchanged
4. Re-run and confirm the run is complete
5. Repeat only if a defect remains (up to {max_iterations} iterations)

When done, ensure results.json contains the results of the final code.
Start by reading results.json and main.py.
"""
