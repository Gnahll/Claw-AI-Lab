# Independent research with one joint paper

New Lab submissions persist their workflow per project. The frontend sends
`discussionEnabled` with the submission; choosing the next project's mode no
longer changes a server-wide switch for projects already running.

- Lab Discuss: independent S1–S7 research, cross-direction discussion, S8 proposals,
  joint integration, one experiment and one paper.
- Lab Independent: independent S1–S8 research, no cross-direction debate, joint
  integration of all proposals, one experiment and one paper.
- Reproduction retains its single-agent workflow. Legacy projects without the new
  metadata retain their existing behavior; deployment does not migrate or start them.

Independent projects persist direction names and paths in project metadata. They
recover completed S8 proposals after restart and create a single `run-integrated`
task. `independent_synthesis.md` is explicitly labeled as independent research;
it is not presented as a debate transcript or consensus reached through discussion.

First-run guidance now puts resource discovery, both-method smoke verification,
checkpoint paths, stable precision, full training budgets, fair baseline scope,
and final model/prediction exports in platform prompts. Missing model_index.json
does not cause a non-Diffusers repository to be replaced by a generic pipeline.
Timeout and OOM do not authorize silently changing frozen experimental settings.
An explicitly scoped single retrained baseline is not expanded into two required
training runs by S9/S17 defaults. These are guidance improvements; they do not
by themselves prove that generated code or scientific claims are correct.

Successful supervised S14 training now records a verified-run receipt: executed
Python source manifest, experiment plan/specification, selected configuration,
supervised evidence report, result JSON, weights, and raw prediction content hashes.
Sources must be unchanged from the start of supervised execution. On the ordinary
S15 transition, unchanged evidence can be reviewed without invoking the repair
model or training the experiment again. The refinement log explicitly records
`retrained: false` and the original evidence source. Hashing existing artifacts
uses disk/CPU but does not allocate training GPUs.

Missing/changed evidence blocks this new reuse path instead of manufacturing new
parameter-update evidence. A new S17 REFINE request also stops for explicit scope
review rather than retraining all completed conditions automatically. This receipt
does not implement arbitrary multi-process training-resume provenance, add missing
experiments automatically, or relax the original supervised training checks.
Legacy runs without receipts retain the old verification path.

Tests cover per-project mode isolation, independent S8 barriers, restart recovery,
single downstream experiment creation, complete-budget guidance, and integrity
review with mutations to code, weights, predictions, results, plan, config and
supervised reports. No live research task is created by these checks.
