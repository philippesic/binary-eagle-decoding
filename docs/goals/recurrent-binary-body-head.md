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
  The trace validator was integrated as `1e3742d`: eight synthetic ancestry,
  position, offset and mask tests passed. The masked CE objective now has
  three CPU gradient/mask tests. The learned nine-linear exporter was
  integrated as `a4dd003`: five synthetic GGUF serialization tests passed,
  including Q/K row order, frozen tensors and invalid inputs. It declares
  `f32_learned_nonnegative` and explicitly blocks native loading until the
  loader recognizes that rule. The hard-binary CPU reference was integrated
  as `fead52a`, then repaired locally after the project virtual environment
  exposed zero gradient at an exact zero scale. Eight arithmetic tests now
  pass, including a genuine two-call student recurrence and all-nine module
  installation. The CPU training step checks nine-linear optimizer ownership,
  projects scales and writes the 18-array exporter checkpoint; three tests
  pass. These modules still need a real-model recurrent graph and capture.
- A new CPU scalar replay of the archived candidate-D training-capture inputs
  matched **430/430 recorded native outputs exactly** across 16 captures and
  all nine projections. This reads old native outputs and performs no current
  GPU execution. Current host: Apple M3 Max CPU (`arm64`), PyTorch 2.14.0,
  NumPy 2.4.6; arithmetic uses F16-cast inputs with F32 signed/group sums.
  The ignored evidence is
  `results/binary-scale-fitting-5080/recurrent-binary-cpu-parity.json`
  (SHA256 `dc37bc4905ee81a0646ec7546c8ea050dbb4f9aa04193d8480780e03e26f46f2`).
  It validates D arithmetic on these sampled operands, not a trained export
  or whole-drafter recurrence.
- A CPU capture-metadata gate now enforces the frozen 96-prompt training hash,
  file hashes, exact-prefix ancestry, native offset mapping, `t2d` inverse
  consistency and valid/support masks. Four synthetic tests pass. It cannot create target features or
  certify K/V cache parity.

**Latest CPU checks:** 32 recurrent tests, eight existing scale-fitting tests,
Ruff lint/format and `git diff --check` pass. No local Metal/MPS or CUDA
runtime was selected. The complete native Q4_0 quality/throughput gates
remain unrun under this goal.

## Research choices pending

The [decision log](../DECISIONS.md) records options for scale zero handling,
teacher-forced versus refreshed student prefix distribution, and the proposed
500-step/45-minute all-body budget. These do not prevent CPU contract work.
The user owns approval of a later GPU budget and any change to the no-GPU
restriction. Do not request or infer that change.

## CPU stop gate and next actions

1. Review the corrected CPU arithmetic, recurrent loss, training-step and
   checkpoint/export integration; run their focused CPU gates together.
2. Prepare the missing real-model capture/recurrent graph contract without
   substituting saved normalized head states. The exact sequential CPU forward
   is intentionally slow and is not a practical full-model training kernel.
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
