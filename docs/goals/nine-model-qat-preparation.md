# Nine model QAT preparation

## Objective and authorization

October 4, 2026. The human requested a separate independent Codex task and
agent team to implement and test preparation for the nine-model campaign:
EAGLE, DSpark and DFlash, each with original frozen Q4 and fusion-calibrated
long-QAT W1A8/W1A1 candidates. Prepare code, data, configurations, artifacts
and launch/evaluation orchestration so no further integration is needed when
the RTX5080 becomes available. Training and model-quality/performance evaluation
are reserved for RTX5080. This objective authorizes preparation implementation
and testing, not starting a training/evaluation job now.

The human said they will open RTX2080Ti for development, including CUDA
testing/preflight. Until it is actually available, progress on local CPU/source
work. A stale unpaused host flag is not evidence of availability. Once available,
one operator may use it for bounded CUDA development tests, builds, native
pack/load/graph/operator checks, actual-model forward/backward smoke without
real-model optimizer updates, and necessary TRAIN teacher-data preparation.
Target-only TRAIN feature/logit capture is data preparation, not a drafter
quality/performance evaluation; record producer hardware and bound its resource
use. Do not run the long QAT, recipe-selection quality runs, held-out benchmark
or final evaluation on RTX2080Ti. Synthetic optimizer/save-resume fixtures are
development tests, not real-model training. Sealed-final content stays untouched.

RTX5080 is currently unavailable. Parent set its machine-local pause flag true,
which blocks new runs; no remote connection, interruption or availability
verification was performed. Do not query, stage to, resume or run on RTX5080
until the human announces availability. Preserve that pause and other teams'
processes, checkouts, captures and existing results. Read host addresses only
from the shared registry in AGENT_OPERATIONS; never persist IPs here or guess.

## Ownership and durable record

Sole preparation coordinator is `01a10903-1c7a-71b1-abb1-0de3ecc046b8`,
claimed October 4 from published handoff `da097e5`. Parent
recap/research chat is a launcher only and will stop editing these records after
dispatch. Do not reuse the completed A8 team's ownership or restart its monitors.
The previous A8 goal and eight-agent research slate remain COMPLETE.

Coordinator owns STATUS, this goal file, recipe/coverage decisions and integration.
Use the configured Sol-high coordinator/feature owners, Luna-high tests/operator,
Astra-medium focused advice, and Opus 5.5 peer review with exact requested model
claude-opus-5-5-high. Assign each mutable file/native module once; use isolated
worktrees, review/tests, frequent short commits and pushed main. Native commits
must be pushed to the fork before the parent gitlink. Workers are not alone and
must preserve each other's edits. Keep one RTX2080Ti operator and serialize GPU
use. Every SSH connection must use tmux MCP; detached jobs use the Linux tmux
socket and remote_job supervisor prescribed in AGENT_OPERATIONS.

## Authoritative inputs

Read the current main version and these records; do not reload every historical
goal document at startup:

- [Main nine-model plan](../../experiments/nine-model-qat-research-plan-2026-10-04.md).
- [Eight-agent additions and peer reconciliation](../../experiments/nine-model-research-slate-2026-10-04/synthesis.md).
- [Completed A8 comparison](../../experiments/a8-qat-recovery/comparison.md).
- [DSpark/DFlash native results](../../experiments/dspark-sm75-20261003/results.md) and
  [precision gap](../../experiments/dspark-sm75-20261003/precision-admission.md).
- [Real fusion fitting](../../experiments/fusion-binary-real-a8-2026-10-03/README.md) and
  [source-pinned zero-scale census](../../experiments/nine-model-research-slate-2026-10-04/zero-scale-census.json).
- [Evaluation protocol](../EVALUATION.md), [operations](../AGENT_OPERATIONS.md) and
  relevant [lessons](../USER_LESSONS.md).

Research-only baseline main d9b07a3 contains all reports. It is not a deployable
nine-model training bundle. Most block binary paths and training/data bridges
are still missing. Existing optimized EAGLE forward/backward/cache/head paths
have actual hardware evidence; old completion-audit pending claims can be stale.
The underperforming learned-A8/midpoint/low-inertia combination is not the default.

## Work packages

The independent coordinator should launch bounded feature owners and reviewers,
not another literature-only slate. Recommended ownership partitions, refined
against current files before edits:

1. Native DSpark/DFlash W1 export, loader and graph with honest selected-tensor
   manifests and CUDA dispatch. Start with fifteen FFN matrices, then admit
   architecture-specific fusion; prepare supported coverage options without
   pretending an activation flag makes dense weights binary. Preserve private
   embeddings/head, anchor-first slot mapping, Markov semantics, masks, cache
   repair and immutable target/verifier. No approximate head borrowing.
2. Training integration for EAGLE and block models: hard deployment-forward
   quantization, own-state/token-prefix contracts, correct full/pruned vocabulary,
   family-specific objective controls, parameter ownership and finite gradients,
   checkpoint/optimizer/RNG/cursor exact resume, export and stage transitions.
   Fixed reference, direct A1 and A8-to-A1 are supported profiles; optional probes
   must remain explicit and off by default until admitted, not stacked silently.
3. Data and fusion preparation: reuse authenticated existing EAGLE inputs;
   new five-tap/full-vocabulary block data where needed, balanced prompt-disjoint
   splits, bounded teacher storage, whole-chain order and label/cache ancestry.
   Fit/calibrate A8 and A1 using their actual arithmetic, with scale-only controls,
   row-orientation rescue and post-norm/raw diagnostics. Source derivative-one
   scales do not prove permanently dead QAT channels. Preserve historical fits.
4. Prepared pipeline and evaluator lifecycle: current-package source/config/data
   bindings, launch guards, cheap reuse of completed admission/audits, training
   ledgers, detached evaluator, process/context/resource release, result joins
   and automatic sequencing on successful committed endpoints. Avoid the old
   multi-hour repeated hashing/audits, in-process host-RAM evaluation failure,
   invented development-ID prefix checks and raw-tensor metadata collection.
5. Training/native performance engineering where applicable: one-line sign clone
   removal, typed metrics, paired-only upload reuse, source-correct logging
   cadence, admitted fused-optimizer option, grouped attention and static
   confidence. Profile-gated refinements are configurations with truthful
   evidence, not blanket assertions of benefit. Do not block readiness on the
   deferred formats/architectures or implementing every one of the new ideas.
6. Independent QA: own tests/acceptance ledger and review evidence separately
   from feature authors. A sole Luna operator dispatches coordinated CUDA tests;
   QA never creates a second host owner. Astra/Opus advise on difficult contracts.

## Readiness acceptance

Maintain a requirement-by-profile ledger: PASS with original evidence/hash and
hardware, PENDING with exact missing check, or EXCLUDED for a deliberately
unsupported option. Never claim a percentage from test count or that synthetic
fixtures prove real-model quality, movement or SM120 memory/performance.

Before declaring portable preparation complete, require:

- All declared profiles/configs parse and execute actual source APIs; loader/
  vocabulary/shape/padding/precision/export and bad-input tests pass. No mock
  evaluator or dense fallback can grant readiness.
- CPU and allowed SM75 pack/operator/graph and forward/backward fixtures,
  later-state/K/V gradients, serialization/resume/RNG and stage-transition
  tests pass with model/target storage ownership. Actual-model SM75 smoke uses
  zero real-model optimizer updates and no held-out quality evaluation.
- Data is materialized or its exact required capture stage and cost are
  exposed; missing block taps or full-distribution teachers cannot be hidden.
  Captures bind the actual producer GPU/native target and exact prefix.
  Native-captured teachers plus bounded trajectories are the numeric reference;
  timebox cross-backend diagnostics unless labels/decisions/gradients change.
- Memory and out-of-process lifecycle tests cover host/GPU floors, source/data
  integrity, STOP/caps/failures and supervisor/context cleanup. Preserve all
  protected files and failures. Empty WSL compute-app lists do not prove release.
- End-to-end synthetic pipeline tests prove checkpoint -> export -> resource
  release -> evaluator -> report, with frozen Q4 controls and target-only
  diagnostic; paused/unavailable/stale-source/failed-export scenarios refuse
  correctly. Real measured results remain reserved for RTX5080.
- Full applicable suites, build/lint and independent review pass on integrated
  source. Commit/push native and parent artifacts coherently; preserve raw
  models/captures/runs outside Git and record versions/hashes.

Prepare a frozen launch bundle and one concrete start command with no code edits
needed. It should run the minimal fresh RTX5080 source, resource, kernel/model/
backward/memory and capture-portability checks once availability is announced,
then start the approved QAT recipe/budget and automatically run native evaluation
after successful committed training and verified resource return. A failed guard
must retain the checkpoint and report the precise failure, not waive it or
blindly start evaluation. Clear ambient experimental flags and bind the actual
binaries/runtime libraries and current recipe. Use separately bounded checkpoint
quality and final timing stages, not intrusive profiler timings as throughput.

SM75 testing cannot certify SM120 execution or resource headroom. Report portable
software/data readiness separately from the pending fresh RTX5080 admission.
Do not repeat completed full corpus audits or synthetic suites just because the
host changes; reuse source-bound evidence where valid and perform only the
necessary fresh target-device checks. Any late dataset or artifact dependency
that prevents prompt QAT launch is a visible remaining preparation item.

## Decisions and launch boundary

Implement the user-authorized foundation and selected feasible preparation
options autonomously. Prepare explicit proposed tensor/exception coverage,
objective/quantizer profiles, balanced token/epoch milestones, wall caps and
evaluation allowance; record evidence and options in DECISIONS. The human owns
major recipe/coverage choices and long training compute allocation. Do not
quietly choose an unbounded budget or promote unmeasured recipe combinations.
The current task stops short of real QAT/evaluation while RTX5080 is unavailable.
When the human starts the prepared campaign with availability and a selected
budget, automatic training-to-evaluation sequencing is already authorized by
this request; no redundant confirmation or integration step is needed.

Keep brief milestone updates; update this file before handoffs/context rotation.
Final preparation report must show completed work/tests, exact commits/artifacts/
command, any excluded probes, and the narrowly defined remaining SM120 checks.
Never claim literal 100% hardware/quality certainty or immediate execution
without a fresh resource check. Complete all work that can be done independently
while a host or scientific decision is pending.

## Implementation team started — October 4, 2026

Native Goal is active in the coordinator chat. Source baseline parent `da097e5`,
native `fcdf5822`. All original untracked overnight files and older worktrees
are preserved. No host connection or training/evaluation was started.

Temporary worktree root: `/private/tmp/nine-model-qat-20261004/`.
Each author owns its isolated branch; coordinator reviews/tests/cherry-picks to
main and pushes before retiring any merged worktree.

| Native subagent | Worktree / branch suffix | Exclusive responsibility |
| --- | --- | --- |
| `/root/native` (Sol high) | native | native fork export/load/graphs, block export and TRAIN teacher capture producer |
| `/root/training` (Sol high) | training | block QAT, EAGLE continuous/recurrent QAT, exact resume/stages, training CLI |
| `/root/data_fusion` (Sol high) | data-fusion | five-tap/full-vocabulary data bridge, deployed-arithmetic fusion fitting |
| `/root/pipeline` (Sol high) | pipeline | frozen configs/launcher, process/resource/evaluator lifecycle and report aggregation |
| `/root/qa` (Luna high) | qa | independent requirement/profile ledger, CPU suites and failure-path tests |
| `/root/cuda_operator` (Luna high) | own report worktree pending | sole coordinated RTX2080Ti operator, initially local planning only |
| `/root/contracts_advisor` (Astra medium) | read-only | native architecture/data/mask/ownership advice |

Coordinator exclusively owns STATUS, this goal, DECISIONS and integration.
Training ownership additionally includes `recurrent_qat.py` for sign clone and
typed diagnostics. Workers must request overlapping expansions. Peer reviews
use exact `claude-opus-5-5-high` with bounded focused requests.

Early concrete dependency: existing EAGLE three-tap/pruned-vocabulary data cannot
supply block five-tap/full-vocabulary teachers. Native owns the real capture
producer; data owner defines ancestry/storage/cursor contracts. Materialized
block TRAIN captures and producer hardware/portability evidence remain PENDING.
The sole operator must plan their cost and identify source model dependencies.

RTX5080 remains locally paused/unavailable; no query/staging/resume authorized.
RTX2080Ti availability clarification requested; stale registry flags are ignored.
No remote operation until direct human availability, followed by sole-operator
fresh resource/ownership checks. CPU/source implementation progresses meanwhile.

Next: settle cross-owner interfaces, land focused tested implementation chunks,
review native arithmetic and block conditioning, build independent PASS/PENDING/
EXCLUDED ledger. Portable readiness and fresh SM120 admission remain separate.
