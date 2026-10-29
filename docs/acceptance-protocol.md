# First-use experiment readiness

New users provide a research topic, research directions, local resources and their
training budget. The optional Details field holds execution requirements. Users do
not need to copy previous failures, platform internals or diagnostic formulas into
the topic to obtain the platform's standard engineering checks.

The platform distinguishes three kinds of checks:

| Kind | Default behavior |
| --- | --- |
| Engineering prerequisites | Check real data/model resources, data separation, shapes, numerical stability, gradients and resource feasibility before training. Failures block training. |
| Research hypotheses | Measure at the declared point and report the result, including negative results. Expected accuracy or separability is not automatically a prerequisite for training. |
| Explicit user requirements | Preserve the user's stated metric, threshold, data cohort and timing. A failed user-required gate is never silently removed or moved. |

S9 generates an `acceptance_protocol` alongside the experiment plan. It must define
each check's input, metric, aggregation and failure action. Blocking pretraining
checks name Python checker functions that S11 must implement. This structural
check does not prove the functions are correct; actual execution remains required.
Plans that explicitly declare missing definitions or an incomplete implementation
handoff stop during design, before spending a code-generation or training run.
Registered definitions in `revision-context` are included in the design input in
full, or rejected explicitly if they exceed the context budget.

A smoke test verifies only the engineering behavior it actually measured. It does
not prove that all scientific checks passed, that full-budget training finished or
that a new method improves the baseline. Changes to GPU authorization invalidate
an incompatible world-size or effective-batch plan; memory, batch/accumulation and
runtime must be measured again rather than inferred by scaling an old estimate.

The optional S8 novelty search has a 120-second total worker limit. A timeout,
worker failure or insufficient search coverage is explicitly reported as
incomplete, with no novelty score. Generated hypotheses are retained; an incomplete
optional search does not establish novelty or block the entire research workflow.

When an experiment deliberately returns `blocked_pretraining`, its measured
failure reason is retained even if the process exits with code 1. This is a failed
run requiring a design/protocol revision, not a completed experiment. The runtime
does not ask a repair agent to remove checks or automatically repeat the same
training attempt. Network and ordinary execution failures retain their existing
retry handling.

Existing projects retain their approved gates. The new default policy does not
rewrite old plans, change scientific thresholds or authorize a new training run.
No platform policy can guarantee a positive research result; the aim is to expose
missing definitions early and complete a faithful, reproducible experiment.
