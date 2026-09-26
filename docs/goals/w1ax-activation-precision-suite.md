# Goal: all-layer W1Ax activation-precision suite on RTX 2080 Ti

**Opened:** 2026-09-25
**State:** active; primary matrices sealed, diagnostics in progress
**GPU owner:** `/root/gpu_monitor` supervises the authorized sequential diagnostic pipeline; root owns analysis/integration

## Objective

Implement, validate, and run the full [W1Ax activation-precision study](../../experiments/w1ax-activation-precision-plan.md) on the RTX 2080 Ti. Compare W1A16, W1A8, W1A4, and W1A1 using identical one-bit weights across all nine drafter linears, with target-only, FP16, Q8_0, and Q4_0 anchors. Separate correctness and dispatch, acceptance, identical-input operator cost, full EAGLE round cost, and serving throughput. Preserve reproducible raw artifacts and a compact report.

## Checkpoint

- Native Codex Goal opened in this task on 2026-09-25.
- At goal opening, the frozen study protocol existed and no new W1A16/W1A8/W1A4 artifact or measurement had been produced. Later milestones below supersede that initial state.
- Local `main` was clean at goal start and the goal checkpoint was pushed as `7ae1cc2`. A managed worktree at `/Users/pippo/.codex/worktrees/w1ax-suite/binary-eagle-decoding` holds the `w1ax-suite` branch.
- The shared `rtx2080ti` host-registry entry was used. The first SSH attempt through tmux MCP timed out during banner exchange; the user's requested retry connected. The actual RTX 2080 Ti (SM75) was idle at 0% with 672 MiB used and 10,356 MiB free, and the project process scan found no active supervisor/server/benchmark. Remote project parent was `11a25f5` and its llama.cpp checkout `34e21b7`. No new run had started at that checkpoint.
- Bounded workers own native W1Ax runtime, eight-path benchmark runner, and opt-in round tracing in separate files. The orchestrator owns GPU operations and integration. Workers are not to commit or use the GPU.
- Keep the RTX 2080 Ti to one supervised experiment at a time. Use `scripts/remote_job.py` and unique run directories. Do not use the 24 QAT-final prompts in this untrained screen.
- A supervised preflight on the actual RTX 2080 Ti (`runs/w1ax-preflight-w1a1-20260925/`) passed the existing W1A1 CUDA backend gate 5/5, including K=31/32/33/2560 and strided K=33. The supervisor ended with exit 0 and the GPU returned to 0% / 672 MiB used.
- The separate nine-prompt context diagnostic was generated under remote `runs/w1ax-context-20260925/` from committed script `e2ad359`; manifest SHA256 `5653cfe7599e5dd4ae44e057df27b816221f8ee89635056bc0fb24b9f44a21a3`. Qwen3-4B chat-template tokenization with thinking disabled passed all planned bins: short 195–207, medium 521–543, long 1131–1171 tokens. Audit JSON SHA256 `ff23798a72f3a782821008fba4ff43dcbe8a9e86e307fd77aa313b64585e4173`; supervised audit `runs/w1ax-context-audit-final-20260925/` exited 0. The audit script is committed as `69d29d4`. These prompts are a context/latency diagnostic, separate from the QAT development and final sets.
- Frozen QAT-revisit manifests were generated in remote `data/qat-revisit/` by supervised `runs/w1ax-qat-prompts-20260925/` (exit 0). The 24-prompt development manifest SHA256 is `a3b97d942a99f1bddd5bb97216c32a9920aaa50788baa5bdb92354842547e885` (8 prose, 8 code, 8 reasoning). The 24-prompt final manifest was generated and sealed; it has not been used in this study.
- The reused all-nine W1A1 GGUF SHA256 is `098e1ecbb299aa16e2c968663acc49e60c0fcf16b053766d9f558114f79d011c`, matching the prior all-row source audit. The fixed FP16 target GGUF SHA256 is `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`.
- Native W1Ax runtime commits `45b76c033`, `4c8767e4e`, `73caaa8ab`, and `3792aa79c` are published on the user's llama.cpp fork branch `w1ax-suite`. Local CPU operator gate passed 88/88; Mac `llama-server` with round tracing compiled. Parent runner, replay harness, and diagnostic analysis commits are on the parent `w1ax-suite` branch. The latest parent gitlink will be updated to `3792aa79c` before a sealed GPU run.
- Remote CUDA source worktree `runs/llama-w1ax-src/` started from published commit `4c8767e4e`, and remote parent runner worktree `runs/w1ax-project-src/` started at parent `b16d72e`; later updates are recorded below. Resolved historical/development configs are under ignored run paths. Only this orchestrator uses the GPU.

## SM75 correctness milestone

- The initial CUDA build completed 358/358 targets with exit 0 under `runs/w1ax-sm75-build-20260925/`. The source worktree was then updated to published llama.cpp `3792aa79c`; incremental `runs/w1ax-sm75-build-final-20260925/` rebuilt 33 targets and exited 0. This is an actual CUDA 12.8/SM75 build, not a cross-compile.
- Supervised CUDA backend gate `runs/w1ax-sm75-op-gate-20260925/` used `GGML_W1AX_ASSERT_INT_DOT=1` and passed 88/88 test cases, including distinct W1A1/W1A4 bit-serial/W1A8/W1A16 dispatch. The independent raw INT32 dot assertion was active for A4/A8. The conventional A4 comparator gate `runs/w1ax-sm75-a4-conventional-gate-20260925/` also passed 88/88 with its own dispatch marker and raw-dot assertion. Both exited 0; GPU returned to 0% / 672 MiB used.
- The eight-path historical dry run under remote parent worktree `runs/w1ax-project-src/` exited 0. Its manifest records target-only, FP16, Q8_0, Q4_0, W1A16, W1A8, W1A4, W1A1 with D=5, p_min=0, five repetitions, and a matching published llama.cpp gitlink. The dry-run artifact is `results/w1ax-historical-full-20260925/`; use a different run ID for actual measurements.
- Remote runner worktree was updated to parent commit `19c3064` and submodule `3792aa79c`. A bounded three-prompt all-nine activation capture is starting under `runs/w1ax-activation-capture-20260925/`; it must finish before operator replay or the timed matrix.

## Real-input operator milestone

- Supervised all-nine capture `runs/w1ax-activation-capture-20260925/` completed three historical requests with raw token IDs, all required loader/graph/CUDA markers, and clean server shutdown. It preserved 4,203 actual W1A1 activations across all nine packed linears; observed token-row counts include N=1, 2, and 34–38. The GPU returned idle. Capture synchronizes the stream and has no timing claim.
- Deterministic selector `runs/w1ax-capture-selection-20260925/selected/` preserved the full 73-row `(tensor,K,M,N,bits)` invocation histogram and symlinked 147 representative captures covering all nine linears and observed shapes. Its manifest SHA256 is `a923844dc8be80240c6b08ffda8499fa468291be6d9960e03a69b33c436c45c8`.
- Linked the native replay harness against the final `3792aa79c` CUDA libraries under supervised `runs/w1ax-replay-link-final-20260925/`. A one-capture SM75 smoke passed scalar parity for all four activation modes. Full selected-input replay `runs/w1ax-operator-replay-20260925/` completed with exit 0: 588 operator records (147 captured inputs × four modes), 435 full 32,000-row head comparisons, and all-nine coverage. It used two warmups and five synchronized samples per operator. The extracted JSONL SHA256 is `a06654a31f000bacf5988cda91201a100b57575fbead9548b66d614093c1afe5`.
- A separate real-input exact INT32 dot check initially failed because CUDA graph capture disallows its diagnostic stream synchronization; this was a diagnostic configuration error, not a dot mismatch. Retry `runs/w1ax-operator-real-dot-gate-retry-20260925/` disabled CUDA graphs, enabled `GGML_W1AX_ASSERT_INT_DOT=1`, replayed all 147 inputs through all four modes, and exited 0. These assertion runs are excluded from timing.
- CPU wall replay includes quantization/packing and output, while CUDA pack/dot substages remain to be traced separately. Capture files do not contain request/round IDs; the operator replay is linked by sequence and tensor/shape, not a proven per-request mapping.
- Replay-only analysis `runs/w1ax-project-src/runs/w1ax-replay-summary-20260925/summary.json` exited 0, SHA256 `e2ec2c6844642380dcbedda22492468068f8d7010d394abb979ce98b23200dc4`. On 145 selected head token rows from 16 captures, same-binary-weight W1A16 top-1 agreement was 97.2% for A8, 72.4% for A4, and 33.1% for A1. These are correlated operator inputs, not independent prompt acceptance. At observed head N=2, full synchronized operator medians were A16 978.7 µs, A8 331.4 µs, A4 117.8 µs, A1 94.8 µs; at N=37, 29,443/5,470/1,637/919 µs respectively. These totals include ggml graph dispatch and synchronization and are not isolated CUDA kernel events or serving rates.
- A CPU-draft versus CUDA-draft model parity check initially discovered that `--spec-draft-ngl 0` alone still selected CUDA for the custom op. The driver was corrected to use `--spec-draft-device none` for CPU and `CUDA0` for GPU. Supervised retry `runs/w1ax-model-parity-retry-supervisor-20260925/` exited 0; all four precisions emitted exactly the same 32 raw greedy token IDs on the frozen historical `prose-01` prompt between CPU and CUDA placement. The target was offloaded 37/37 layers to the GPU in both paths; the CPU draft showed no CUDA W1Ax dispatch, and each GPU draft showed its mode-specific dispatch. GPU returned idle afterward. This is one-prompt model parity, with full paired serving measurement still pending.

## Serving checkpoint

- The matched eight-path historical matrix `runs/w1ax-project-src/results/w1ax-historical-matrix-20260925/` completed 480/480 requests with exit 0 and all four W1Ax dispatch gates confirmed. It used the same FP16 target, 12 frozen prompts, two warmups, five repetitions, 128 output tokens, D=5, p_min=0, project `dc4ccdd`, and published llama.cpp `3792aa79c`. All speculative variants emitted identical raw IDs on all 60 paired requests; target-only differed on the same `reasoning-02` prompt in each repetition. See the [interim result report](../../experiments/w1ax-activation-precision-results.md) for rates, raw artifact hashes and limits. The four W1Ax decode rates were 29.43/42.12/44.44/45.39 tok/s for A16/A8/A4/A1, versus 80.67 FP16 EAGLE and 89.73 Q4_0 EAGLE. Accepted drafts/round fell to 0.106/0.102/0.041/0.055 versus ordinary 1.168. GPU returned idle after the historical matrix.
- The development dry run verified the frozen 24-prompt manifest SHA256 `a3b97d942a99f1bddd5bb97216c32a9920aaa50788baa5bdb92354842547e885`, the same model/gitlink hashes and D=5/p_min=0 policy. The full eight-path development matrix is running under tmux MCP session `w1ax-2080ti` and remote supervisor `runs/w1ax-project-src/runs/w1ax-development-matrix-supervisor-20260925/` (child PID/PGID `29140` at start). Raw results are `runs/w1ax-project-src/results/w1ax-development-matrix-20260925/`. It has 24 prompts × five repetitions × eight paths, 960 expected requests. Monitor `state.json` and `records.json`; on a pause request interrupt the supervisor through tmux MCP, wait for `state.json` to stop, inspect the recorded process group and `nvidia-smi` before reporting the GPU free. The reserved 24 QAT-final prompts have not been evaluated.
- The primary runner's `nvidia-smi` snapshots are marked unavailable because the WSL executable was outside its nonlogin PATH. Do not infer missing memory/clock values. Hardware and dispatch were verified independently; full explicit device snapshot `runs/w1ax-device-manifest-20260925/stdout.log` has SHA256 `bd87139e8ce478aef42c4b17fda2345891be9cd1c4ad02b826c47dee417af0c8`. Probe fallback fix `c6d1970` passed 38 local benchmark tests and will be used for subsequent diagnostics, without altering the sealed primary data. Root remains GPU owner; read-only worker `/root/gpu_monitor` watches the development run from its own tmux pane `%15`.

## Next actions

1. Verify host and GPU state through tmux MCP, plus local and remote Git/build state.
2. Implement and audit common W1Ax weight/export and numeric contracts; pass CPU/CUDA reference gates and real SM75 dispatch checks.
3. Run the frozen full matrix, operator replay, round tracing, and context/policy diagnostics; analyze both FP16 and Q4_0 comparisons.
4. Write a compact report with raw artifact hashes, update this checkpoint and status, integrate tested code into `main`, and push.

## Native runtime worker checkpoint

- Extended the existing packed-weight GGML operator with activation modes 1, 4, 8, and 16 selected by `GGML_W1AX_ACT_BITS` at EAGLE graph build. Default 1 preserves the old W1A1 path. All modes load the same nine packed signs and F32 row scales, with no dense selected-weight shadow.
- A16 casts each F32 activation to FP16 at the operator boundary, accumulates signed values in F32, then multiplies the row scale. A8/A4 use per-token F32 absmax scales, signed ranges ±127/±7, nearest-even rounding, exact integer dot accumulation, then row-scale and activation-scale multiplication. CUDA A4 defaults to a four-plane bit-serial dot; `GGML_W1AX_A4_KERNEL=conventional` selects a real quantized-code comparator.
- CPU build succeeded locally on Apple M3 Max; focused backend operator tests passed 88/88 across odd K, 32-bit boundaries, dirty tails, N=1/2/3, all-zero activations, negative zero, and strided views. This is a CPU correctness check, not an SM75 performance result. Python conversion tests were not run because the local environment lacks PyTorch.
- CUDA code is written but awaits CUDA compilation and actual 2080 Ti exact-dot/dispatch validation. Operator replay, all-nine real activation capture, full-model logits/IDs, and matched benchmark remain required gates. No GPU was used by this worker.
- Opt-in diagnostic capture: set `GGML_W1AX_CAPTURE_DIR` to an existing directory and disable CUDA graphs. Each `op-%012llu.bin` stores magic `W1AXACT1`, LE uint64 sequence/K/M/N, LE uint32 bits, a 128-byte NUL-padded packed tensor name, then N×K F32 activations. Capture synchronizes the CUDA stream and is excluded from timing.
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

## Research cross-reference checkpoint (2026-09-25)

The user now requests web research to cross-reference and revise the seven
local analyses. Reuse the seven Astra-high reviewers for bounded primary-source
checks; they write only isolated staging reports. The orchestrator owns
document integration in `/tmp/binary-eagle-cross-reference-20260925`, branch
`research/cross-reference-20260925`. Compare literature's actual precision,
architecture, verifier and hardware with this project; do not treat external
speedups as local measurements. Preserve the active W1Ax goal and frozen
protocol pending any explicit later research decision. No GPU run is planned.

Cross-reference complete: all seven reviewers returned primary-source checks.
The consolidated [evidence/revision record](../../experiments/research-cross-reference-2026-09-25.md)
and seven original reports now distinguish supported advice, narrowed claims
and rejected inferences. The original local review was integrated/pushed as
`b7ea617`; its temporary worktree and branch were removed. This follow-up is
documentation-only; no experiment, model, runtime gitlink or frozen protocol
changed. Development/final prompt contents were not used for new selection.
The next project action remains suite execution/analysis under the current
owner and run checkpoint above.
Review-stage notes remain under `/tmp/binary-eagle-cross-reference-reports-20260925/`;
all decision-bearing evidence and source versions are preserved in the repository.

Validation completed: 17 Markdown files, 75 local links and eight section
anchors checked; `git diff --check` passed. Architecture and evaluation
reviewers checked the integrated revisions; their concrete wording corrections
were applied. All seven review assignments are complete. Integration is
documentation-only; no model tests or GPU checks were run.

Integration preserved concurrent suite checkpoint commits `3f1b917` and
`57ea68b` by rebasing the documentation branch onto current main. The literature
review did not inspect, interrupt, or assume ownership of the recorded GPU run.
Its source/performance baseline remains explicit, pending the suite owner's
sealed result.

## Development completion (2026-09-26 UTC)

- The development matrix finished 960/960 requests with exit 0 at 01:53:03 UTC. All four W1Ax dispatch gates passed; all seven speculative variants matched target-only raw IDs on all 120 paired requests. Rates and hashes are in the linked result report. PGID 29140 was empty afterward and the GPU was at 0% with 687 MiB allocated.
- Exact primary binaries and all dependent libraries were copied to remote `runs/w1ax-primary-binaries-3792aa79c/bin/`; its `SHA256SUMS` contains ten file hashes. Primary measurements retain runtime `3792aa79c` and project `dc4ccdd`.
- `/root/gpu_monitor` has bounded exclusive GPU ownership for the diagnostic rebuild to published runtime `feba15698` and the 88-case CUDA gate only. New supervisors are `runs/w1ax-diagnostic-build-20260926/` and `runs/w1ax-diagnostic-op-gate-20260926/`. The runtime adds opt-in capture attribution and raw verifier logits; numerical kernels are unchanged. Root will resume ownership after the worker verifies stopped processes and idle GPU. No QAT-final evaluation or training has occurred.

## PrismML and quantization research session (2026-09-25)

The user requested 45 minutes of agent-led deep research into PrismML's
near-BF16 low-bit quality and implications for our quantizer/QAT. Start:
2026-09-26 01:22:50 UTC; intended research window ends at 02:07:50 UTC.
This advisory session verifies claims/artifacts, audits our quantizer against
weight-only methods, challenges causal explanations and proposes a bounded
redesign. It does not take GPU ownership or modify the active suite's frozen
protocol. At session start the main checkout had a concurrent uncommitted goal update;
it was preserved and its later committed progress is retained above. Reports stage in `/tmp/binary-eagle-prism-reports-20260925/`; integration
worktree `/tmp/binary-eagle-prism-qat-20260925`, branch
`research/prism-qat-20260925`.

Research milestones: seven primary-source/artifact analyses and cross-challenge
rounds completed; full BF16 selected-weight geometry (218,234,880 values),
Q1_0 export with exhaustive source-row audit, and ten total bounded local CPU
regression requests completed. The CPU tests used historical prompts only,
explicit CPU placement and the older `92bc706` binary; no SM75 result is claimed
from them. All owned local server processes/ports were verified stopped/closed.
Raw data and the 36,920,320-byte Q1 artifact are preserved under ignored
`results/prism-research-20260925/`. Report:
[PrismML and quantization/QAT](../../experiments/prism-quantization-research-2026-09-25.md).

The late-session concurrent commit `7ac0dd2` sealed development results; the
research report incorporates its A16 acceptance evidence while preserving the
current diagnostic owner and frozen protocol. New calibration/training budgets
remain proposed decisions, and no final-prompt evaluation or QAT was performed.

Research session completed at 2026-09-26 02:08:04 UTC: 45 minutes 14 seconds
from the recorded start, followed by publication. All seven reviewers completed
their work; claim, math, causal and CPU-result checks passed after corrections.
The report and two reproducible CPU audit scripts were checked for syntax,
links and whitespace. Integration preserves `7ac0dd2` and its diagnostic GPU
ownership; no new project goal or training experiment was opened.

## Diagnostic runtime checkpoint

- Published runtime `feba1569848e74996651a42a9751f8afa5958b1d` built successfully; `runs/w1ax-diagnostic-op-gate-20260926/` passed 88/88 CUDA cases with exact-dot assertion and exit 0. Both worker process groups stopped and ownership returned to root. The ten-file preserved primary checksum manifest verified. Parent gitlink is published in `019aa0c`.
- Supervised `runs/w1ax-project-src/runs/w1ax-verifier-divergence-supervisor-20260926/` exited 0. At historical reasoning-02 position 109, target-only raw logits favor token 12 over 29208 by 0.00348663; the emitted speculative verifier row favors 29208 over 12 by 0.00539780. The draft token is accepted because it matches the target verifier's own top choice. A preceding hypothetical row is explicitly not sampled and not emitted. This identifies a target ranking reversal, not its underlying numerical cause.
- CPU-only vocabulary audit `runs/w1ax-vocab-coverage-20260926/` exited 0: 22,460 of 22,800 target-only emitted IDs across historical/development requests are in the 32,000-entry draft vocabulary (98.51%). This is emitted-ID coverage, not target probability mass or acceptance. Report SHA256 `c341f882579b85b116c93e57d5b436a58565940eb17e0b69cf232a9846a7e922`.
- Root started `runs/w1ax-project-src/runs/w1ax-attributed-capture-20260926/` via tmux MCP `%12`; three short historical requests collect capture timestamps, per-request file inventories and round association. Do not run another GPU job until it stops. The extended replay harness built under `runs/w1ax-replay-anchor-build-20260926/` with exit 0; matched FP16/Q8/Q4/cast replay is next.

## Operator and pipeline milestone (2026-09-26 UTC)

- Attributed capture exited 0: 88 round rows across three 32-token historical requests. Each trace's concatenated emissions equal the response IDs after exactly one leading pre-round token. Analyzer `a841d66` now validates that explicit uniform offset, reports omitted-token totals and excludes warmups; arbitrary suffix matching is forbidden.
- `runs/w1ax-anchor-replay-20260926/` exited 0 with 588 W1Ax operator records, 441 FP16/Q8_0/Q4_0 anchor records, 435 W1Ax head comparisons, 147 FP16 cast controls and 145 cast head comparisons. Every cast control had zero output difference; all 145 head top-1/top-5 comparisons agreed. JSONL SHA256 `208b2e8f9caa526aeab4e9d66bfb88d0a375b14b1afbf7f95c864bc0f042f8cb`; analyzed report SHA256 `721708f3a669b0a935c8ad04223269da9b88212bf241b257a3bb0e2da62f0186`. The separate 147-input real conventional-A4 replay also exited 0.
- Four separate Nsight CUDA software traces (`runs/w1ax-nsys-a{16,8,4,1}-20260926/`) and an anchor trace (`runs/w1ax-nsys-anchors-20260926/`) completed, exported to SQLite and passed the read-only analyzer. Each W1Ax precision has 1,176 operator invocations (correctness, warmups and samples pooled), with CUDA graphs disabled. Matching graph-disabled unprofiled replay measured median per-capture profiler wall overhead factors 1.010/1.123/1.157/1.079 for A16/A8/A4/A1. These traces are separate diagnostics, not serving rates.
- Full historical round trace is active: runner cwd `runs/w1ax-project-src`, supervisor `runs/w1ax-round-matrix-supervisor-20260926/`, results `results/w1ax-round-matrix-20260926/`, PID/PGID35822, started 02:05:03UTC. Last root check60/480 requests. Runtimefeba15698/project2f6ab14 remain frozen while this and the following pipeline runs.
- Bounded experiment supervision was handed to `/root/gpu_monitor`, through its own tmux MCP pane. Root performs CPU/read-only analysis only. After successful round completion and stopped-process verification, worker runs sequentially: `w1ax-server-profile-baseline-supervisor-20260926` (120requests); `w1ax-server-nsys-supervisor-20260926` (same120 with CUDA software trace); `w1ax-streaming-supervisor-20260926` (240measured latency/telemetry requests); then `w1ax-context-policy-supervisor-20260926` using frozen `runs/w1ax-diagnostic-suite-20260926/suite.json` (two360-request context cells plus twelve960-request development policy cells). All paths are relative to the runner cwd. No GPU overlap, Git update, final-set use or training. Worker stops on failure or user pause and reports exact supervisor/child state.
- Pause procedure: local `scripts/agent_env.py pause rtx2080ti`, interrupt active remote supervisor through tmux MCP, wait for stopped state, inspect all recorded process groups and explicit WSL nvidia-smi before declaring GPU free. The launcher records its current command/result attempt in `progress.json`, not child PIDs; inspect its live descendants and confirm the benchmark/server also stop.

## Analysis validation checkpoint

- CUDA pair analyzer `61c1faf` validates adjacent same-device/stream pack→dot kernels with matching token-grid dimensions and no overlap; it reports paired sums, elapsed/gaps and unpaired launches without inventing capture/layer attribution. Actual four-mode SQLite exports passed.
- Fixed-trajectory analyzer `e865d93` reports trace-only exclusive draft-span removal and dual-anchor acceptance thresholds, with immutable proposal/cost assumptions. It never subtracts traced spans from uninstrumented decode times. First completed ordinary-EAGLE repetition passed real measured-request suffix mapping, warmup exclusion and strict span/emission accounting. Full480-row paired analysis remains pending.
- Local integrated W1Ax Python checks passed87/87; CUDA runtime remains88/88 on SM75. Concurrent Prism research documentation from main was preserved in merge `91ed411`; no frozen workload or QAT-final use changed.

- The tested implementation/analysis was integrated and pushed to main at `08383ce`; its published llama.cpp gitlink is `feba15698`. The main submodule checkout is clean. Local native benchmark integration checks passed38/38, alongside87/87 W1Ax checks. The managed worktree remains in use for ongoing analysis and will not be retired while workers need it. Remote measured runner stays frozen at2f6ab14.
- Native Goal API currently reports `paused` from the earlier interrupted connection attempt despite the user's retry authorization. Root requested that the user resume the Goal controls for automatic between-turn continuation; this API has no resume operation. The authorized GPU pipeline and this project's durable goal remain in progress. Do not infer a GPU pause from that app state; honor any explicit user pause immediately.

- Conditional secondary audit found W8A8/W4A4 all-nine artifacts are available but their loader/operators are absent from frozenfeba15698 (`d0724427b` is not an ancestor). The plan's combined-build condition is unmet; no new integration/secondaryjob is added and the authorized pipeline stays unchanged. Worker `/root/w1ax_benchmark` made no edits.

## Full round completion (2026-09-26 02:31 UTC)

- Historical round matrix completed480/480 with exit0 at02:31:21UTC; PGID35822 empty, GPU0%/687MiB before nextjob. All480 raw output sequences match their corresponding primary requests. Full analysis and break-even supervisors exited0.
- All420 speculative measured requests mapped across five repetitions, warmups excluded, with one leading token omitted per request. There are37,695 trace events. Accepted/proposed/verified-round counters exactly reconcile; extra trace rows are legitimate no-proposal events. FP16 andQ8 each have five accepted-but-unemitted tokens at stop boundaries. Span audits found zero invalid/outside/nested/overlapping/clamped rows. Exact artifact hashes and stage means/p95 are in the result report.
- Conditional removal of only recorded exclusive draft() spans leaves W1Ax trace-only emission proxies53.04–55.44tok/s, below both80.67/89.73 primary anchors. This holds proposals, emissions and other costs fixed; it is not an uninstrumented bound or full removal of all drafter work. Source helpers are separately frozen under `runs/w1ax-analysis-src/` while the measured runner remains2f6ab14.
- Worker `/root/gpu_monitor` started short server-profile baseline at02:32:41UTC, supervisor `runs/w1ax-project-src/runs/w1ax-server-profile-baseline-supervisor-20260926/`, PID/PGID39806, expected120 requests. It retains GPU ownership and the documented sequential pipeline. Root runs only CPU analysis and integrates reports.

## Full-server profiling repair checkpoint

- Short unprofiled server baseline completed120/120 requests at02:37:37UTC, exit0; PGID39806 stopped. Initial Nsight-wrapped120-request run also exited0 at02:43:39UTC (PGID42324 stopped), but its101KiB report/SQLite has no CUPTI kernel table. It is NOT valid CUDA profiling evidence. Raw requests and failed profile are preserved under runner `results/w1ax-server-nsys-20260926/` and `runs/w1ax-server-nsys-supervisor-20260926/`.
- Cause: the benchmark's intentional inherited-environment whitelist strips Nsight injection variables before spawning each server. New profile-only wrapper `b6453bc` writes a fresh configuration with exactly nine observed Nsight variables, checks injection-library containment, preserves numerical settings and records hashes/provenance. Nine focused tests passed; existing measured runner code remains2f6ab14. Wrapper is staged at remote `runs/w1ax-analysis-src/profile_w1ax_benchmark.py`, SHA256 `8f4f7f3f3ce59be917e245aee58e611cceb6e8af764ec9b45afbe952f01c0448`.
- `/root/gpu_monitor` is authorized to run only the two-request injection smoke next: supervisor `runs/w1ax-project-src/runs/w1ax-nsys-injection-smoke-supervisor-20260926/`, output `results/w1ax-nsys-injection-smoke-20260926/`, using `check_w1ax_target_divergence.py` and explicit Nsight `--cuda-graph-trace=node`. It must then hold for root's SQLite kernel validation before the full120 retry. Streaming/context/policy remain queued; no overlap or source checkout update.
- CUDA analyzer `989ed73` separates process/context identities and optionally infers40 serial server PIDs from the verified5×8 schedule, with exactcount/noninterleaving/signature checks and explicit unavailable reasons. This is chronological inference, not direct per-request or per-round CUDA attribution. Twenty-four focused tests passed.

## Profiler injection validated

- Injection smoke finished with exit 0 at 02:54:33 UTC; PGID 45161 stopped and the GPU returned idle. Exported SQLite contains 6,551 kernel activities and 532 aggregate CUDA graph activities across both server PIDs. The requested node setting did not yield replay-node activities in this child-process setup; the cause is not established. Actual granularity is recorded explicitly.
- CUDA analyzer `5f60909` includes graph, memcpy and memset activity spans alongside separate kernel-only summaries, preserves process/context identities, and unions intervals to avoid counting nested graph/child intervals twice. It labels graph interval coverage as such, not physical GPU utilization. Thirty-one focused tests and the actual smoke analysis passed; no missing process/context IDs were observed.
- Full 120-request profile retry started at 03:00:57 UTC under worker-owned supervisor `runs/w1ax-project-src/runs/w1ax-server-nsys-retry-supervisor-20260926/`, PID/PGID 45763. Results are `results/w1ax-server-nsys-retry-20260926/`; explicit `--cuda-graph-trace=graph` matches the validated observed mode. Individual kernel breakdowns remain in the separate graph-disabled operator profiles. Last worker check: 75/120 requests.
- After retry completion the worker holds before streaming so root can export and analyze the profile without competing CPU load. Root must explicitly release the remaining streaming/context/policy pipeline after that check. The measured runner remains at `2f6ab14`; only the separate ignored profile-driver configuration carries the Nsight settings.

## Full-server profile validated; streaming active

- Full retry completed 120/120 at 03:07:21 UTC with exit 0, PGID 45763 stopped and GPU idle. Export has 813,285 kernel activities and 29,470 graph replay activities across all 40 server PIDs. Combined analysis passed, including serial schedule/mode association; it remains labeled inference rather than direct PID provenance. All 120 output sequences match the short unprofiled baseline; observed decode-time ratios are 1.0053–1.0309. Profile analysis SHA256 `1be5bdef18d1356ccf300ea29d23e1027c78c121d17a1381495eb6bee365b37b`.
- CPU analysis barrier released. `runs/w1ax-conventional-real-int-gate-20260926/` then exited 0 at 03:14:35 UTC: independent exact INT32 assertions on all real conventional-A4 replay outputs, 147 captures/all nine linears, CUDA graphs disabled. Sampled scalar output errors were also zero. This is excluded from timing. PGID 48729 stopped; GPU returned idle.
- Streaming began 03:15:25 UTC, supervisor `runs/w1ax-project-src/runs/w1ax-streaming-supervisor-20260926/`, PID/PGID 48858; results `results/w1ax-streaming-20260926/`. Frozen catalog is nine prompts, SHA256 `5653cfe7599e5dd4ae44e057df27b816221f8ee89635056bc0fb24b9f44a21a3`. Selected IDs: context-prose-short, context-code-medium, context-reasoning-long; 3×2 caps×5 reps×8 paths =240 measured requests. Worker corrected a verbal prompt-count typo; manifest/configuration were correct.
- `/root/gpu_monitor` retains GPU ownership and proceeds from streaming to the full 14-cell context/policy launcher without another hold, unless a failure or explicit user pause requires stopping. Root performs only light read-only/CPU postprocessing.

## Streaming complete; context/policy suite active

- Streaming completed 240/240 with exit 0 at 03:34:10 UTC; PGID 48858 stopped and GPU returned idle. All 80 telemetry summaries completed with nonmissing loaded/peak memory. All paired completion-text hashes match target-only; the SSE API supplied neither raw IDs nor usage, so the diagnostic makes no raw-ID or decode-rate claim. The result report now includes TTFT, HTTP wall and sampled memory with tail/sampling limits and exact hashes.
- Final suite started 03:36:36 UTC under `runs/w1ax-project-src/runs/w1ax-context-policy-supervisor-20260926/`, PID/PGID 54155. Manifest `runs/w1ax-diagnostic-suite-20260926/suite.json` freezes two 360-request context cells followed by twelve 960-request development policy cells. `progress.json` under that suite directory records current config/attempt and child result path; last worker check has context-cap-32 at 198/360.
- `/root/gpu_monitor` owns the GPU and proceeds sequentially; root only reads/postprocesses completed artifacts. No final prompts or training are used. On failure or user pause, stop the outer supervisor and the active child/server process groups, confirmed from the live process tree, then verify GPU idle before reporting it free.
- Latest CPU policy analyzer `6392327` adds nonfatal exact raw-output comparisons within each cell versus FP16/Q4_0/target-only and across selected/fixed policies, with first divergence and source hashes. Mismatching cells remain timing observations; selection is exploratory development-only. Eight focused tests passed. Ship this helper separately for final postprocessing; do not update the running frozen source checkout.

## Context transition and handoff details

- Owning Codex chat ID: `01a0daf7-0c08-7310-8697-523744f96c71`; worker `/root/gpu_monitor` remains the sole GPU operator. The first context cell, `context-cap-32.toml`, completed 360/360 with exit 0 at 03:47:55 UTC. Its run ID is `w1ax-diagnostic-suite-20260926-context-cap-32-a1`. All seven speculative variants matched target-only raw IDs on all 45 paired requests. Records SHA256 `d4a8616c1dc8470cc8e1baac2d827a27c567c3a6a82e092a91a842887260d8fe`.
- `context-cap-128.toml` began at 03:48:13 UTC, run ID `w1ax-diagnostic-suite-20260926-context-cap-128-a1`; last check 63/360. Outer launcher PID/PGID remains 54155; at the 03:52:24 UTC snapshot benchmark PID/PGID was 56836 and server PID/PGID 57655. These child IDs change: inspect the live tree before manual signaling.
- Frozen launcher `progress.json` does NOT contain child PIDs. The operator now refreshes `runs/w1ax-project-src/runs/w1ax-context-policy-supervisor-20260926/process_snapshot.json` with timestamped PID/PPID/PGID/comm for the launcher and its project descendants. Graceful outer interruption forwards to the benchmark, whose cleanup stops its server; verify all live groups and GPU afterward.
- Completed small result files are mirrored via tmux-managed scp under ignored local `/Users/pippo/github/binary-eagle-decoding/runs/w1ax-analysis-mirror/` for CPU analysis without competing with timed GPU work. Frozen remote data/source are unchanged. Local helper worker `/root/context_analysis` owns only a new context-matrix analyzer/tests.
