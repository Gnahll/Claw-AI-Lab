# S4 connectivity failures

S4 checks OpenAlex, Semantic Scholar and arXiv before literature retrieval.
arXiv remains an optional source, not a required or exclusive provider.

If a probe exceeds the shared deadline, S4 retains completed results and skips
pending sources. It does not let an uncaught futures timeout abort the stage or
wait for unfinished network checks when leaving the executor context. Running
probe threads finish when their underlying network operations return; this is
not a process-level cancellation guarantee for DNS resolution.

Probe failures and pending source names appear in the task log. The stage also
writes `site_status.json`. If all checks fail or time out, S4 reports a retryable
failure in `literature_error.txt` rather than silently marking empty literature
collection successful. No later training stage is started by this error path.

After restoring connectivity, use the original task's resume/retry action. This
change does not automatically resume tasks, alter search queries, change GPU
authorization, or modify the existing literature retrieval/fallback algorithms.
