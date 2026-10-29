"""Default first-run requirements belong in the platform, not in user incident reports."""

FIRST_RUN_CONTRACT = """
First-run readiness and frozen protocol:
- Inspect the configured repository's actual model, loader and evaluation entrypoints.
  Resolve required local checkpoint paths and symlinks and read real image/label pairs
  before allocating the full training budget. Missing resources are explicit blockers;
  do not switch to a generic architecture or an unrelated pretrained pipeline.
- Before formal training, smoke-test baseline AND proposed method with the real loss,
  augmentation and precision policy. Check finite losses/gradients, that the new modules
  participate in optimization, peak memory and throughput. Record measured readiness.
  Choose feasible microbatch/accumulation settings in the plan before freezing it;
  preserve any user-specified effective batch, data split and full training budget.
- Separate engineering prerequisites from research hypotheses in acceptance_protocol.
  Engineering readiness must pass before training. Hypothesis quality and expected
  improvements are measured research outcomes, not invented default training gates.
  Preserve explicit user-mandated gates verbatim. Define all checkers, alignment,
  reference distributions and aggregation during design; do not pass unresolved
  definitions to S11. A smoke pass never implies full scientific acceptance.
- Honor user-selected conditions, seeds and optional experiments. Extra baselines are
  recommendations when feasible, not permission to expand a fixed experiment budget.
  Distinguish retrained baselines, public-checkpoint references and literature values.
- Never select models, tune thresholds or change the method using test-set results.
  Apply the same data protocol and declared precision policy to the baseline and method.
- Save resumable per-condition checkpoints (model, optimizer, scheduler, completed epoch,
  RNG and sampler state), final weights, raw test predictions and workspace-root results.json.
  Export completion artifacts even on a checkpoint-resume path. Do not mark a condition
  complete until its required artifacts and measured results exist and are validated.
- Preserve completed work and its provenance. Missing exports must be derived from
  saved checkpoints, not solved by discarding completed runs or retraining from epoch 0.
  If runtime evidence cannot validate a resumed run, report the blocker; do not forge
  evidence or rewrite the code to force unnecessary retraining.
- OOM, numerical failure and timeout do not authorize reducing epochs, changing frozen
  batch/learning rate, dropping conditions or overwriting official results. Stop with a
  resource or protocol-change request when a faithful run cannot complete.
"""
