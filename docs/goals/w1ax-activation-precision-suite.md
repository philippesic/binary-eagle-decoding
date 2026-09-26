# Goal: all-layer W1Ax activation-precision suite on RTX 2080 Ti

**Opened:** 2026-09-25
**State:** active; protocol and implementation review
**GPU owner:** this goal; no remote job started yet

## Objective

Implement, validate, and run the full [W1Ax activation-precision study](../../experiments/w1ax-activation-precision-plan.md) on the RTX 2080 Ti. Compare W1A16, W1A8, W1A4, and W1A1 using identical one-bit weights across all nine drafter linears, with target-only, FP16, Q8_0, and Q4_0 anchors. Separate correctness and dispatch, acceptance, identical-input operator cost, full EAGLE round cost, and serving throughput. Preserve reproducible raw artifacts and a compact report.

## Checkpoint

- Native Codex Goal opened in this task on 2026-09-25.
- The frozen study protocol already exists. No new W1A16/W1A8/W1A4 artifact or measurement exists yet.
- Local `main` was clean at goal start. Shared host registry points to `192.168.4.29` for the RTX 2080 Ti, but SSH through tmux MCP timed out during banner exchange on 2026-09-25. The user has been asked for a current reachable address or WSL/SSH startup. GPU availability and process state remain unverified; no run has started.
- Keep the RTX 2080 Ti to one supervised experiment at a time. Use `scripts/remote_job.py` and unique run directories. Do not use the 24 QAT-final prompts in this untrained screen.

## Next actions

1. Verify host and GPU state through tmux MCP, plus local and remote Git/build state.
2. Implement and audit common W1Ax weight/export and numeric contracts; pass CPU/CUDA reference gates and real SM75 dispatch checks.
3. Run the frozen full matrix, operator replay, round tracing, and context/policy diagnostics; analyze both FP16 and Q4_0 comparisons.
4. Write a compact report with raw artifact hashes, update this checkpoint and status, integrate tested code into `main`, and push.

## Research review checkpoint (2026-09-25)

The user requested seven independent local-only deep analyses using Astra high:
QAT execution, non-EAGLE one-bit architectures, end-to-end drafting policies,
draft graph/kernel optimizations, precision allocation, training data and
vocabulary coverage, and evaluation/experiment design. These are advisory
reviews under the current goal, not a replacement goal or authorization for
new GPU runs. Agents read the shared checkout without modifying it; reports
are staged separately for synthesis. No web search or remote execution is
part of this review. Major research choices remain with the user.

Review agent ownership: `/root/qat_execution`, `/root/non_eagle_architectures`,
`/root/throughput_policies`, `/root/draft_pipeline_optimization`,
`/root/precision_allocation`, `/root/data_alignment`, and
`/root/evaluation_strategy`. All use `gpt-6-astra`, high reasoning.
The precision review was retried once after a transient model-capacity error.
Integration worktree: `/tmp/binary-eagle-deep-analysis-20260925`, branch
`research/deep-analysis-20260925`; per-category report staging:
`/tmp/binary-eagle-analysis-reports-20260925/`.

Review complete: all seven reports are preserved under
`experiments/research-review-2026-09-25/`, with the ranked synthesis at
[research-review-2026-09-25.md](../../experiments/research-review-2026-09-25.md).
The agents finished their assignments without repository code edits or remote
runs. Cross-review resolved a metadata-count discrepancy: development has nine
topic families and final has ten. Advisory findings and pending options are
recorded in `docs/DECISIONS.md`; the frozen primary study, QAT budget, final-set
reservation and next actions above remain unchanged. No new GPU state was
verified. Documentation link/whitespace checks precede integration into main.
