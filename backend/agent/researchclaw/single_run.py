"""Single-run policy (experiment.runs_per_condition = 1).

The default prompts ask for >= 3 seeds per condition, mean +/- std, confidence
intervals and significance tests. When the user decides that every condition runs
once, those rules are rewritten in every prompt that carries them and a policy block
is appended, so planning, code generation, the PROCEED/REFINE decision and the paper
all follow the same protocol. With runs_per_condition >= 3 nothing changes.
"""
from __future__ import annotations

from typing import Any

SINGLE_RUN_POLICY = (
    "## Runs per condition (user decision; overrides any multi-seed rule above)\n"
    "Each experimental condition is trained and evaluated exactly ONCE with one fixed "
    "seed, the same seed for every condition. Do not add seeds or repeats, and never "
    "choose REFINE to add them. Do not report or invent mean +/- std, confidence "
    "intervals, p-values or significance tests. Report the single-run numbers as they "
    "are and state once, in the experimental setup, that each condition was run once."
)

# (default text, single-run text). Every default text must exist verbatim in a prompt
# source; tests/test_single_run.py checks this so prompt edits cannot silently skip it.
REWRITES: tuple[tuple[str, str], ...] = (
    # prompts.py: blocks
    ("- NO per-seed breakdowns or confidence intervals\n",
     "- NO confidence intervals (each condition was run once)\n"),
    ("- Do NOT list per-seed ranges (e.g., '0.71-0.73 across seeds') — use mean +/- std\n",
     "- Report the single-run values; do NOT invent mean +/- std or ranges\n"),
    ("- Tables: mean +/- std with 95% CI in parentheses\n",
     "- Tables: one value per cell from the single run (no std, no CI)\n"),
    ("- Every comparison claim must cite a p-value or note multiple seeds\n",
     "- Comparison claims quote the measured single-run numbers; no p-values\n"),
    ("- Report the number of random seeds/runs used\n",
     "- State once that each condition was trained and evaluated once with one fixed seed\n"),
    # prompts.py: code review sub-prompt
    ("3. **STATISTICAL RIGOR**: Multiple seeds? Results averaged and reported with std? "
     "Paired comparisons?\n",
     "3. **PROTOCOL**: Exactly one run per condition, with the same fixed seed for every "
     "condition?\n"),
    # prompts.py: S9 experiment_design
    ("- Under `compute_budget`, plan at least 3 seeds per condition (5 when the budget "
     "allows) to ensure valid statistical comparisons.\n",
     "- Plan exactly ONE run per condition (one fixed seed, the same for every condition); "
     "this is the user's decision.\n"),
    ("STATISTICAL POWER REQUIREMENTS (CRITICAL for publishability):\n",
     "RUNS PER CONDITION (user decision):\n"),
    ("- Use AT LEAST 3 random seeds per condition (5 preferred, more for cheap runs)\n",
     "- Each condition is trained and evaluated exactly once with one fixed seed\n"),
    ("- Report: mean ± std, 95% bootstrap CI, per-seed raw values\n",
     "- Report the single-run value of every condition (no std, no CI)\n"),
    ("- For method comparisons: use paired bootstrap or Wilcoxon signed-rank test "
     "(NOT paired t-test with n < 10)\n", ""),
    ("- Report effect sizes (Cohen's d or rank-biserial correlation)\n", ""),
    ("- With only 3 seeds, report it as a limitation and avoid strong significance claims\n",
     "- Make no significance claims; spend the saved compute on full training budgets\n"),
    ("num_regimes × num_seeds.\n", "num_regimes (one run each).\n"),
    # prompts.py: code generation stage
    ("4. Run all conditions with >=3 seeds; print per-run metrics",
     "4. Run every condition once (one fixed seed); print per-run metrics"),
    ("9. Breadth-first: run 1 seed per condition FIRST, then additional seeds.\n",
     "9. Run each condition exactly once; do not add seeds.\n"),
    ("11. Paired bootstrap CI, effect sizes.\n", ""),
    # prompts.py: S16 result_analysis
    ("4. REPLICATION: If results are from a single seed (n=1), explicitly note that no "
     "statistical significance claims can be made.\n",
     "4. REPLICATION: Each condition was run once by design (user decision); compare the "
     "single-run values directly, make no significance claims and do not ask for more "
     "seeds.\n"),
    # prompts.py: S17 research_decision
    ("incomplete (a baseline, a requested ablation, seeds); REFINE adds them",
     "incomplete (a baseline, a requested ablation); REFINE adds them"),
    ("3. Each condition has results from ≥3 seeds\n",
     "3. Each condition has results from its single run (one run per condition is the "
     "user's decision; never REFINE to add seeds)\n"),
    # prompts.py: S20 paper_draft
    ("23. EVERY result table MUST include 95% confidence intervals (mean +/- CI or "
     "[low, high]).\n",
     "23. Result tables report the single-run value of each condition (no CI, no std).\n"),
    ("24. EVERY comparison claim ('A outperforms B') MUST cite p-value. If p >= 0.05, "
     "write: 'The difference is not statistically significant.'\n",
     "24. Comparison claims quote the measured single-run numbers; no p-values.\n"),
    ("25. If the proposed method does NOT statistically significantly outperform a "
     "baseline, do NOT claim superiority.",
     "25. If the proposed method does NOT outperform a baseline on the reported numbers, "
     "do NOT claim superiority."),
    # prompts.py: S21 peer_review
    ("3. STATISTICAL VALIDITY: Are confidence intervals or error bars reported? Is n>1 "
     "(multiple seeds)? Are significance tests appropriate?\n",
     "3. STATISTICAL VALIDITY: Each condition was run once by design; do not ask for "
     "seeds, error bars or significance tests. Check that claims are no stronger than "
     "single-run numbers allow.\n"),
    # executor.py: S20 results call
    ("     Each cell = mean ± std across seeds. Bold the best value per column.\n",
     "     Each cell = the single-run value. Bold the best value per column.\n"),
    ("   - Include a STATISTICAL COMPARISON table (Table 3): paired t-tests between key "
     "methods.\n", ""),
    # codegen/system_prompt.py: S11
    ("- Use 3 seeds (42, 123, 456) for statistical robustness",
     "- Run each condition exactly once with seed 42; do not loop over seeds"),
    (" - Use 3 random seeds (42, 123, 456) for each condition\n",
     " - Run each condition exactly once with seed 42 (no seed loop)\n"),
    # iterative_refine/system_prompt.py: S15
    ("baseline, an ablation the user asked for, or more seeds).",
     "baseline or an ablation the user asked for)."),
    ("- Extra seeds may be appended to the planned seeds; keep the existing ones.\n",
     "- Do not add seeds: each condition runs exactly once (user decision).\n"),
    # result_analysis/system_prompt.py: S16
    ("   - Computes per-condition statistics (mean, std, CI95 if multiple seeds)",
     "   - Reports the per-condition values of the single run (no std/CI: one run per "
     "condition)"),
)


def runs_per_condition(config: Any) -> int:
    experiment = getattr(config, "experiment", None)
    return int(getattr(experiment, "runs_per_condition", 3) or 3)


def is_single_run(config: Any) -> bool:
    return runs_per_condition(config) == 1


def rewrite(text: str) -> str:
    for old, new in REWRITES:
        text = text.replace(old, new)
    return text


def apply(text: str, single_run: bool, *, append_policy: bool = False) -> str:
    """Rewrite the multi-seed rules in *text*; optionally append the policy block."""
    if not single_run:
        return text
    text = rewrite(text)
    if append_policy and SINGLE_RUN_POLICY not in text:
        text = f"{text}\n\n{SINGLE_RUN_POLICY}"
    return text
