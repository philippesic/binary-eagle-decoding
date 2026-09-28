# Joint binary EAGLE body and head

**Objective:** prepare a faithful W1A16 jointly trainable nine-linear EAGLE
draft body and binary head, then evaluate it against Q4_0 EAGLE when a later
user decision permits the required GPU work. The pinned FP16 target/verifier
and draft vocabulary map stay frozen. [Trial protocol](../../experiments/recurrent-binary-qat-plan.md).

**State (2026-09-28 UTC): active, CPU preparation.** The user explicitly
denies access to every GPU and accelerator, including remote hosts and local
Metal/MPS. No SSH or GPU work is permitted. No GPU owner, remote tmux session,
run directory, model training, final-set use, or model-quality result exists
for this goal. The native Codex Goal in this task has the same objective; this
file and `docs/STATUS.md` are the durable project checkpoint.

## Current findings and work

- Candidate D is the reference: packed signs, nonnegative fitted group-128
  scales, F16-cast inputs/F32 ordered arithmetic for all nine selected
  matrices. D scored 0.425 accepted drafts/round; Q4_0 scored 1.042 on the
  reused 24-prompt development set. The frozen-body fitted FP16 head was a
  diagnostic, not a binary training initialization.
- The existing `TrainableW1A1Head` detaches cached head inputs and uses BF16
  row scales. It cannot train the body or propagate later-position loss.
  Native recurrence uses `(token[P+1], target features[P])` at position `P`
  and feeds pre-norm student state into later positions; saved head captures
  contain output-normalized states only. A new feature/cache/verifier capture
  and a whole-drafter unroll remain needed.
- The native GGUF loader currently accepts mean-absolute or NNLS scale-rule
  metadata. Jointly learned scales require an honestly named rule and loader
  amendment before a trained export can load.
- CPU-only implementation is split into distinct bounded modules: trainable
  group binary arithmetic and a recurrent exact-prefix trace validator.
  Their review, integration and results will be recorded here.

## Research choices pending

The [decision log](../DECISIONS.md) records options for scale zero handling,
teacher-forced versus refreshed student prefix distribution, and the proposed
500-step/45-minute all-body budget. These do not prevent CPU contract work.
The user owns approval of a later GPU budget and any change to the no-GPU
restriction. Do not request or infer that change.

## CPU stop gate and next actions

1. Review and integrate the two CPU modules, with forward/backward, unroll,
   ancestry/mapping/mask and packed scalar-replay tests.
2. Add a bounded model-independent training driver or adapter only where it
   can preserve the real native recurrence contract. Keep any unavailable
   real-model steps explicit rather than substituting a cached-head trainer.
3. Test CPU-only commands with explicit CPU devices and record results,
   hashes and commits. Keep model files/raw captures out of Git.
4. Audit whether remaining local work can still advance the goal. This is the
   **first** goal turn under the no-GPU restriction. Do not mark the Goal
   blocked until the same GPU-only impasse persists across at least three
   consecutive goal turns and no meaningful CPU-only work remains.

**Completion evidence still required:** real-model target-prefix capture and
alignment, whole-drafter unroll parity including draft K/V and masks, trained
GGUF loader/export checks, native acceptance, graph and latency/full-throughput
measurements against matched Q4_0. None is established by CPU tests alone.
