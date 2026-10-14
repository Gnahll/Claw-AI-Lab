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

## 3. Pipeline integration

After `training verify` succeeds, set an **absolute** path in the run configuration:

```yaml
experiment:
  training_job_dir: /absolute/path/to/new-job
  metric_key: pixel_f1
  metric_direction: maximize
```

Start/resume the research pipeline at S14 (`EXPERIMENT_RUN`). In this opt-in path, S14 verifies the managed job and imports its metrics without asking an LLM to fabricate or rerun the experiment. Incomplete or invalid jobs fail the stage. S15 records that optional refinement is disabled, preventing test-guided tuning. Existing workflows without `training_job_dir` retain their current behavior. This path does not constrain earlier S11/S12 generation, so explicit entry at S14 after the managed pilot remains necessary for a full paper pipeline.

## 4. Web acceptance (真实训练验收)

The frontend includes a separate **Real Training Acceptance (GPU)** panel, localized as **真实训练验收（GPU）**. This is deliberately separate from Lab / Reproduce quick submission: a natural-language topic records the purpose but does not authorize GPU use or modify the architecture. It does not promise autonomous model innovation, full paper training, or automatic paper writing.

1. Open the panel. Server defaults come from `examples/cotransformers-pilot.json` and `examples/cotransformers-web.json`; only server absolute resource paths are accepted. The administrator can set `CLAW_TRAINING_SPLIT_MANIFEST` to choose another audited manifest. Web jobs are limited to one truncated epoch and at most three optimizer steps.
2. **检查输入** validates the actual split, dataset, source and initial checkpoint, showing the train/validation/test counts and previously evaluated groups. No GPU is queried or launched. Editing inputs requires another check. A preflight expires after 24 hours.
3. Review and explicitly approve the pilot data protocol. **准备真实训练任务** writes an approved *copy* of the reviewed manifest, preserving the original proposal and shared code. This invokes the existing managed `prepare_job`, not LLM-generated substitute training code.
4. Obtain exclusive allocation from the administrator/scheduler. Inspect GPUs if desired, enter the allocated physical UUID and explicitly confirm allocation. **启动 GPU 真实训练** invokes the existing `start_job`; busy devices are rejected, never killed or silently replaced. `CLAW_WEB_TRAINING_DISABLED=1` lets the administrator disable web starts/resumes.
5. While the panel is open and connected it polls every five seconds, showing real worker state, phases, optimizer steps and bounded log tails. Closing the page does not stop the detached worker. Reopening or restarting the bridge recovers jobs from disk; it never automatically restarts interrupted GPU workers. Explicit resume requires a committed checkpoint and the original authorized GPU UUID.
6. On the next poll after completion, the service independently verifies the evidence, recalculates F1/IoU and imports results into the job's `pipeline/stage-14` and `pipeline/stage-15`. This exercises the managed S14/S15 importer, but does not launch S16–S22. A training process exiting successfully is **not** sufficient: the page displays success only for passed verification, with its timestamp. **重新核验** checks again and invalidates a previous pass if artifacts changed.
7. Inspect/download the verification JSON, test metrics, optimizer events and S14 report, and compare all test prediction/target masks. Large checkpoint files remain at the displayed server job directory, not loaded into the WebSocket or served eagerly by HTTP.

Jobs and preflights live under `backend/runs/training-jobs/` and are ignored by Git. Downloads are restricted to named evidence files and recorded test masks; private configuration, model source, arbitrary paths and oversized files are not exposed by this endpoint. GPU start and protocol approval are separate explicit actions. Training WebSocket commands reject foreign/missing origins; the Vite agent proxy must preserve Origin. Local SSH-tunnel origins are accepted; administrators may add exact origins with `CLAW_TRAINING_ALLOWED_ORIGINS`. This is a trusted single-user tunnel deployment, not a multi-tenant authentication system.

### 用户验收标准

网页的 **真实训练验收（GPU）** 入口是本次新增训练功能的专用入口，不是原来的“开始复现”。你在网页检查并确认数据划分，再确认 GPU 使用权限，之后由系统执行真实模型训练、验证和测试，保存权重，核验参数确实变化并重新计算指标。看到“真实训练核验通过”才算这条链路通过；只有报告、预训练推理结果或“任务结束”都不算通过。这次最多三次参数更新，是工程验收，不是完整训练或模型效果达标证明。

## Still required before scientific completion

1. Agree on pilot fine-tuning versus retaining COVERAGE solely for testing, and approve the corresponding dataset protocol.
2. Obtain an exclusively allocated idle GPU and execute the minimal real baseline. Confirm memory capacity, finite backward gradients, actual optimizer updates, checkpoint reload and every test prediction on the target model. A synthetic CPU recovery test does not establish these facts for Co-Transformers.
3. Review the resulting evidence and pilot limitations, then implement the agreed model innovations in an isolated source copy. Do not claim improved accuracy before training and comparing under the same protocol.
4. Exercise the dedicated web acceptance panel on a real allocated GPU, then add external alert delivery if requested and an administrator-approved service/scheduler policy for automatic host-reboot recovery. Explicit checkpoint resume exists; automatic recovery is not implemented.
5. Exercise the opt-in import through S16 and paper stages with genuine pilot results and review the written scientific claims. No LLM/paper run has been launched merely to test the infrastructure.

The adapter, worker lifecycle, web dispatch, resume, dataset safeguards and opt-in S14/S15 gate can be tested without consuming shared GPUs. Their implementation does not establish that the target model has trained successfully: a real allocated-GPU baseline run on the approved split remains required. Synthetic verifier fixtures and CPU toy-model recovery tests are explicitly only infrastructure tests. Full paper-pipeline autonomous managed training, external alert delivery, automatic reboot recovery, NPU support and the two proposed model innovations remain outside this bounded acceptance adapter.
