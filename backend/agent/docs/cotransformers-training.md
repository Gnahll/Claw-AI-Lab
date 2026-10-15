# Co-Transformers: real-training rollout

## Fixed research inputs

- Task: image manipulation localization, initially copy-move on COVERAGE.
- Reference: https://github.com/ProgrameThinking/Co-Transformers
- Inspected server reference revision: `1c9f2921e20f6727efbaf0d8826be0247a90e42f`.
- Preserve HES, noise fusion (SRM, Bayar, NoisePrint++), CE/CTE, MLF and prediction head. Do not substitute an unrelated model.
- Proposed extensions: Cross-Branch Feature Interaction and Adaptive Multi-Scale Attention. Establish a trained baseline before claiming improvement.
- No optional ablations, robustness sweeps or expensive visualization runs in the initial rollout.

The supplied presentation discusses original-paper experiments. Its reported scores, 200 epochs, batch size 4 and RTX 4090 are **not evidence of experiments performed by this platform**. The repository's launch script has different batch/accumulation settings. An exact reproduction requires checking the original training protocol and data, not blindly copying either set of parameters.

## 1. Audit and approve the data protocol

COVERAGE's README defines `image/<id>t.tif` as forged, `image/<id>.tif` as its original, and `mask/<id>forged.tif` as the forged-region mask. Do not use `copy.tif` or `paste.tif` interchangeably as the localization target.

Run from `backend/agent` using the existing Python environment with Pillow installed:

```powershell
python -m researchclaw.cli dataset-audit --root E:\COVERAGE --output coverage-audit.json
python -m researchclaw.cli dataset-split --root E:\COVERAGE --output coverage-splits-proposal.json
python -m researchclaw.cli dataset-check --manifest coverage-splits-proposal.json --proposal
```

The audit is read-only. It checks full image decoding, original/forged/mask pairing, dimensions, foreground/background and SHA-256 fingerprints, recording every excluded pair and reason. It does not resize mismatched masks to silently repair annotations. Nonzero grayscale mask pixels are foreground; this rule is recorded explicitly.

The split command produces a **proposal only**, defaulting to 70/15/15 with seed 42. Fractions apply to independent source/content groups; sample counts may differ when duplicate groups have different sizes. Original and forged variants, including identical decoded content under different IDs, stay together. This catches exact duplicates, not all near-duplicate images. Proposal validation never authorizes training. Reports refuse to overwrite existing files.

Add `--previously-evaluated-group <id>` for each source group examined in an earlier experiment. These groups remain out of training/validation. The manifest deliberately makes no blind-test claim; known exposure of the pretrained checkpoint and prior human/model inspection must be established separately. The current historical output has already been examined, so a reused test is not an untouched benchmark.

After explicit user agreement on the protocol, the reviewed manifest can be marked `protocol_status: approved`. The default `dataset-check` rejects proposals, missing/empty splits, duplicate IDs, content/group leakage, changed files and changed audited sample membership. `--root` allows the same relative manifest to be verified after transfer to the server; identical bytes and decoded content are required.

There are two different valid goals:

1. **COVERAGE pilot fine-tuning**: split COVERAGE for a small engineering experiment. It verifies the training chain but does not reproduce the paper's cross-dataset protocol or prove broad IML generalization.
2. **Original-paper reproduction**: retain COVERAGE as evaluation data and obtain the original training/validation datasets and protocol first. Do not invent an official split from the presentation.

## 2. Managed baseline pilot

The platform now provides an opt-in deterministic training adapter using the existing `CoTransformers` class and its unchanged loss. A job snapshots `cotransformer.py`, `models/*.py` and `common/*.py`, records file hashes/revision, and validates the full author checkpoint with `weights_only=True` and `strict=True`. It does not modify the shared reference repository. The worker passes no constructor pretraining paths, so the reference backbone's silent partial-loading path is bypassed; initialization uses the verified full state dictionary instead.

Use `examples/cotransformers-pilot.json` as the explicit job specification. The default budget is **one truncated epoch, three optimizer steps**, batch size one, AdamW with constant learning rate, AMP, no augmentation. Images use the installed official IMDLBenCo resize/normalization and width-7 edge generator. This deliberately simpler pilot training protocol is recorded in `environment.json`; it is not the paper's training recipe. Both original innovation proposals remain unimplemented until baseline validation succeeds.

On the Linux training host, using the existing Python environment (no torch/CUDA upgrade):

```bash
python -m researchclaw.cli training prepare --spec cotransformers-pilot.json --job-dir /absolute/path/to/new-job
python -m researchclaw.cli training start --job-dir /absolute/path/to/new-job --gpu-uuid GPU-ALLOCATED-UUID
python -m researchclaw.cli training status --job-dir /absolute/path/to/new-job --watch
python -m researchclaw.cli training resume --job-dir /absolute/path/to/new-job --gpu-uuid GPU-ALLOCATED-UUID
python -m researchclaw.cli training verify --job-dir /absolute/path/to/new-job
```

Preparation rejects unapproved splits, changed inputs and checkpoint hashes. Start/resume require an explicitly allocated UUID, and refuse busy GPUs or failed `nvidia-smi` detection. A per-user GPU lock prevents this runner's own concurrent jobs from selecting the same device. It is not a cluster-wide reservation system: the administrator/scheduler must still allocate the GPU exclusively, because unrelated processes do not honor this lock.

The worker runs in its own Linux session, independent of the CLI/LLM tool process. Closing the status watcher or restarting the platform does not intentionally stop it. A host reboot, logout policy or service/container cgroup termination can still stop it; resume is explicit and automatic reboot recovery is not claimed. Worker identity includes boot ID and process start time to avoid mistaking a recycled PID for a living training job.

`events.jsonl` records optimizer steps/losses and validation/test progress. `state.json` and `training status --watch` show current phase and detect dead workers/stalled progress. Alerts currently appear in status output, not through email, chat or frontend notifications. Each committed checkpoint includes model, optimizer, AMP scaler, cursor and Python/torch/CUDA RNG state. Resume checks input/adapter hashes and continues at the committed cursor. A crash may repeat work after the last checkpoint; repeated step numbers in the append-only log must not be counted as independent training. Storage keeps the latest committed checkpoint and the validation-selected best model, not every historical step. These full-state checkpoints can still consume several GB each.

Selection uses validation pixel F1. The worker reloads the selected newly saved checkpoint before testing every test image. It writes per-image binary prediction/target PNGs and macro pixel F1/IoU at threshold 0.5 on resized 512x512 masks. Verification checks complete test membership, original annotation correspondence, checkpoint/artifact hashes, parameter changes, process-attributed GPU memory evidence and optimizer events, and independently recomputes the scores from saved masks. This is an audit trail, not a tamper-proof security boundary against someone able to rewrite every artifact.

## 3. Original web workflows: discussion, independent research and reproduction

The separate GPU acceptance panel introduced in PR15 is removed. Keep using **Lab·讨论**, **Lab·独立**, or **论文复现**. They share the original S11 code generation, S12 smoke test, S14 experiment execution and S15 refinement; no fourth workflow or fixed three-step demo is substituted for research.

In the existing submission form, `Require real training` defaults to enabled. Uncheck it for evaluation-only experiments. Supply the original code, dataset and checkpoint paths, describe the approved data protocol and full training budget in the research requirements, and enter an **already authorized physical GPU UUID**. Submitting permits that task to use that GPU; leaving it empty permits CPU only, not automatic GPU selection. Busy devices fail the stage instead of being preempted. The original stage logs show errors and the original resume action retries the failed stage after resources are corrected.

The settings reach the project config and every research-direction config. They also apply when the discussion output enters the shared experiment layers. Training requirements are included in the research topic so planning and coding cannot treat an inference-only run as the requested training task. Original web submissions do not silently import an unrelated precompleted managed job.

These settings apply to newly submitted projects. Historical project YAML files are not silently rewritten; to exercise this change, submit a new project or explicitly configure `experiment.require_training: true` and the authorized `experiment.gpu_uuid` before resuming an old one. CLI topic overrides also preserve the mandatory training objective. WebSocket browser origins are restricted to localhost by default; an administrator can explicitly allow another origin using `CLAW_ALLOWED_ORIGINS`.

The experiment Python environment is preserved without resolving its executable symlink out of the virtual environment. S11/S12/S14/S15 acquire a per-user GPU lock shared with the existing managed worker, check that the authorized GPU is idle, and propagate that device to their subprocesses. CPU-only jobs are no longer blocked by the old nominal GPU-count scheduler gate. This is cooperative locking, **not a container security boundary or a cluster scheduler**; administrator allocation is still necessary and generated code remains subject to review.

S12 now executes `SMOKE_TEST=1 main.py` itself rather than believing a model's success statement. S14 first runs `main.py` directly, then invokes the existing agentic repair loop if it fails and reruns the repaired code. S15 still proposes/refines methods using its original agent loop, but must pass a new supervised full run before it can report success or improvement. Fresh metrics, a zero exit code, and the configured metric key are required; old smoke results and a reused S14 baseline cannot satisfy these gates. Each supervised full attempt uses `time_budget_sec`; a repair/retry or final S15 verification is an additional attempt, not a claim of a global wall-clock quota.

For training tasks the supervised process observes real PyTorch optimizer steps, scalar backward losses and parameter changes. Generated code must save the final model state dictionary to `outputs/model.pt`. The observer reloads it with `weights_only=True` and checks that every optimizer's final trained parameters are contained in some checkpoint under `outputs/`, with `outputs/model.pt` matching at least one of them. Independently trained models (for example one per seed) must therefore each save their final state dictionary under `outputs/`; the per-optimizer match is recorded as `checkpoint_matches`; GPU tasks additionally require process-attributed GPU evidence. This currently supports **single-process PyTorch**; distributed launchers, other frameworks and unsupported checkpoint formats fail explicitly rather than being counted as verified training. No fixed three-step training limit is inserted into the original workflows.

Evidence lives under the original run: S12 `smoke_execution/`, S14 `runs/attempt-*/`, and S15 `verified_execution/`. Each contains `execution.log` and `execution_report.json`; training additionally produces `training_observation.json` and the per-step `training_observation.jsonl`. The original stage artifacts retain results and model weights. Stopping an agent propagates termination and cleans up its owned experiment subprocesses. Original stage resume **reruns the stage**; it is not exact optimizer-step recovery. The separate managed CLI below retains its full-checkpoint resume feature.

**Limits:** this generic observer verifies execution and parameter updates, not scientific correctness. It does not independently recompute arbitrary user metrics, approve a COVERAGE split, prove absence of data leakage, guarantee all planned epochs/ablations ran, or prove a claimed architecture matches the paper. Review the plan, source, data protocol, complete learning curves and validation/test predictions. The stronger COVERAGE-specific mask verifier below remains available. No real Co-Transformers GPU training is implied by passing infrastructure tests.

## 4. Optional import of an already verified managed job

After `training verify` succeeds, set an **absolute** path in the run configuration:

```yaml
experiment:
  training_job_dir: /absolute/path/to/new-job
  metric_key: pixel_f1
  metric_direction: maximize
```

Start/resume the research pipeline at S14 (`EXPERIMENT_RUN`). In this opt-in CLI path, S14 verifies the managed job and imports its metrics without asking an LLM to fabricate or rerun the experiment. Incomplete or invalid jobs fail the stage. S15 records that optional refinement is disabled, preventing test-guided tuning. This path is intended for explicit entry at S14 after the managed pilot; it does not replace the original web workflows described above.

## Still required before scientific completion

1. Agree on pilot fine-tuning versus retaining COVERAGE solely for testing, and approve the corresponding dataset protocol.
2. Obtain an exclusively allocated idle GPU and execute the minimal real baseline. Confirm memory capacity, finite backward gradients, actual optimizer updates, checkpoint reload and every test prediction on the target model. A synthetic CPU recovery test does not establish these facts for Co-Transformers.
3. Review the resulting evidence and pilot limitations, then implement the agreed model innovations in an isolated source copy. Do not claim improved accuracy before training and comparing under the same protocol.
4. Verify a real authorized end-to-end project in each original web mode. Dedicated live learning-curve visualization, external alerts and administrator-approved automatic recovery remain follow-up work; stage resume is not optimizer-state resume.
5. Exercise the opt-in import through S16 and paper stages with genuine pilot results and review the written scientific claims. No LLM/paper run has been launched merely to test the infrastructure.

The adapter and original workflow execution gates can be tested without consuming shared GPUs. Their implementation does not establish that the target model has trained successfully: a real allocated-GPU baseline run on the approved split remains required. A CPU toy-model test is explicitly only an infrastructure test. Dedicated learning-curve monitoring, external alerts, automatic reboot recovery, NPU support and the two proposed model innovations remain outside this change.
