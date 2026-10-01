# QAT optimization readiness

## Objective and boundaries

User requested on 2026-10-01 via start-goal-team: implement and fully test faster
QAT computation, batching/cache construction, binary-weight optimization,
learned A4/A8 clipping and A1 thresholds, and curricula/trajectory refresh,
until ready for real GPU QAT. This is the one active project goal, continuing
Phase 1 of the one-bit plan. The previous [body/head goal](recurrent-binary-body-head.md)
retains historical experiment/data ancestry and preparation job ownership.

Preparation only: no real-data optimizer updates, no sealed-final reads, no
changes to frozen target/verifier or existing experiment source/config identity.
Tiny synthetic optimizer tests are permitted. RTX2080Ti remains paused.
RTX5080 belongs to the existing preparation operator in chat
01a0f47a-e246-75e1-a299-fcac42d34f8a; new GPU checks require explicit ownership
coordination and idle proof. Do not modify or stop that job from this goal.

## Completion evidence

- Computation: remove duplicate A1 forward work and validate custom backward,
  hard/native sign and scale order, zeros/subnormals, attached recurrent gradients.
- Cache/head: chunked K/V-only context and per-chain head batching, with exact
  token/feature/position ancestry, F16 cache casts, masks and detach boundaries.
  Investigate larger batches using measured memory/time; preserve update cadence
  by default and label experiments that change it.
- Binary optimizer: selectable initialization, LR/STE/optimizer controls,
  persistent-sign diagnostics, strict optimizer ownership and exact resume.
- Learned quantizers: effective trainable A4/A8 clipping and A1 thresholds;
  explicit checkpoint/export/native support or fail-closed deployment boundary.
  No simulation-only quantizer can receive native-ready status.
- Curricula: bounded A8→A1 transfer, depth weights, staged precision, fresh
  trajectory ancestry; deterministic resume and common-budget accounting.
- Independent CPU suite and integration review; actual RTX5080 bounded
  forward/backward, native export/decision gates and memory/timing evidence for
  enabled deployment options, without real optimizer updates. Preserve failures.
- Tested commits integrated/pushed on main, worker worktrees cleaned only after
  verification. Publish report and launch runbook with remaining limits stated.

100% ready is a testable gate, not an accuracy or speed guarantee. Q4_0 EAGLE
remains the acceptance/latency/throughput comparison target. Learned options may
require explicit experimental deployments; no unsupported feature silently
falls back to a different recipe. Measured GPU savings and quality improvements
must not be inferred from CPU tests or arithmetic counts.

## Initial team and ownership

Checkpoint baseline: main 5676254. Source audit:
experiments/training-optimization-audit-2026-10-01.md.

Planned independent temporary worktrees:
1. Computation owner: recurrent_qat.py A1 autograd implementation and its tests.
2. Cache/head owner: recurrent rollout/provider and native Torch adapter/tests.
3. Binary/curriculum owner: new optimization/curriculum modules and tests.
4. Learned-quantizer owner: new quantizer module, export contract and tests.
5. Luna: independent CPU integration checks and GPU readiness plan; no GPU use
   until assigned after existing preparation owner releases it.
Root owns continuous integration/config/checkpoints, reports, goal/status.
Every worker stops at a reviewed bounded deliverable, records checks/findings
under experiments/qat-optimization-readiness/, and commits its own files.

## Current state

Goal checkpoint committed before team launch. No implementation or new GPU
checks yet. Existing preparation is separate and remains preparation-only.
