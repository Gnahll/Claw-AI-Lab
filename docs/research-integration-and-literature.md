# Complementary research directions and literature integrity

Lab projects with discussion enabled now combine all completed S8 proposals into
one joint hypothesis in `run-integrated`, instead of ranking text length,
headings, keyword counts, or the presence of a novelty report. Independent mode
and reproduction mode retain their separate workflows.

The integration uses the neutral project configuration and every direction's
hypothesis and available novelty assessment. It records mechanism, evidence,
novelty risk, interface with the joint method, and verification for each input
direction. Missing/duplicate directions, unsupported proposals, malformed output,
and model errors stop the transition to S9. The model must explain an incompatible
combination rather than silently removing a direction. These are proposed
mechanisms, not a guarantee of scientific novelty or empirical improvement.

`stage-08/integration_report.json` records coverage and source paths. All input
S1–S8 artifacts are retained under `discussion/direction_artifacts/`; candidate
papers, screened papers, and bibliography entries are combined for downstream
use. Conflicting citation keys block integration. The original direction
workspaces are preserved. A persisted integration request lets a manual resume
retry integration after interruption without restarting all literature research.
Existing projects already past S8 are not migrated or restarted by deployment.

Literature collection no longer adds robot-specific queries to unrelated mixed
Chinese/English topics. Seed references are saved as background only in
`background_references.json`. They do not count as retrieved related work, do not
populate the screened shortlist, and do not make `real_search` true. Source counts
separate online search from user-provided references. Empty retrieval produces a
retryable failure, never placeholder papers. The all-sites-unreachable failure
remains in place.

Screening rejects legacy seeds/placeholders, does not restore an entirely
irrelevant pool, and accepts only assessed candidates actually present in the
filtered input. Original bibliographic provenance cannot be overwritten by the
screening model. Empty/invalid/low-relevance screening results fail explicitly.
There is no minimum of 15 papers, no synthetic relevance scores, and no unassessed
supplements. A short genuine shortlist is allowed. Topic relevance still involves
model judgment and does not replace citation verification or human review.

Regression checks use fake model/search responses and do not train models or
consume GPUs. The deployment gate includes the new integration and literature
integrity tests as well as the connectivity tests.
