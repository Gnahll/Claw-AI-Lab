# S12 supervised smoke repair

The reported S12 failure was an output-contract mismatch, not GPU scheduling:
main.py wrote measured smoke results under outputs/smoke/results.json, while the
supervisor required a fresh workspace-root results.json. Agent-generated result
files and prose claims were correctly rejected, but the measured rejection was
not fed back into the repair loop.

S12 now runs a supervised smoke attempt before requesting repair. If it fails,
the agent receives the actual errors and execution log, with explicit guidance
that main.py itself must publish root results.json on every invocation. The
supervisor reruns the repaired source in a second bounded attempt. It still
deletes stale root results and rejects process failures, nonfinite/missing named
metrics and explicit blocked/failed/incomplete/skipped status. A nested output
or manually written result is never silently accepted as substitute evidence.

Both attempts are retained under smoke_execution/attempt-1 and attempt-2. The
latest log and report remain at smoke_execution/ for compatibility. Only fixes
that pass the independent rerun are copied back to the experiment source. Failed
S12 decisions explicitly request retry and include the measured reason.

These changes do not approve learning-rate/scheduler changes, alter data splits,
remove full-run approval gates or certify a full training run. Smoke instructions
reserve the held-out test split for final evaluation; this is an instruction,
not an independent runtime audit of arbitrary generated data loaders. Existing
test exposure is not undone by this patch. Deployment does not resume tasks.
