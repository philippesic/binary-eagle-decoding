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
- The forked native GGUF loader now accepts the honest
  `f32_learned_nonnegative` rule in commit `7f23c89b3`. A CPU-only build with
  Metal, CUDA, Vulkan, SYCL, HIP, RPC, Accelerate and BLAS disabled passed
  eight native loader fixtures, including learned v2 and unknown-rule
  rejection. A real trained GGUF has not been loaded or checked numerically.
- CPU-only implementation is split into distinct bounded modules: trainable
  group binary arithmetic and a recurrent exact-prefix trace validator.
  The trace validator was integrated as `1e3742d`: eight synthetic ancestry,
  position, offset and mask tests passed. The masked CE objective now has
  three CPU gradient/mask tests. The learned nine-linear exporter was
  integrated as `a4dd003`: five synthetic GGUF serialization tests passed,
  including Q/K row order, frozen tensors and invalid inputs. It declares
  `f32_learned_nonnegative`; the loader now recognizes that rule. The
  hard-binary CPU reference was integrated
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
- A model-independent CPU rollout contract now feeds each proposed token and
  the previous **pre-norm** student state into the next step without detaching
  the functional cache. Three tiny causal-cache tests pass: a later-only CE
  loss reaches the first key, value, state, fusion and head; decoder
  memory/RoPE positions advance from `P` to `P+1` while shifted input-token
  positions are `P+1`, `P+2`; invalid terminal rows create no decoder call. This checks the
  training interface, not actual EAGLE mask or K/V byte parity.

**Latest CPU checks:** 47 recurrent tests, seven explicit decoder-step tests,
five frozen-operand tests, eight native loader fixtures, eight existing
scale-fitting tests, Ruff
lint/format and `git diff --check` pass. No local Metal/MPS or CUDA
runtime was selected. The complete native Q4_0 quality/throughput gates
remain unrun under this goal.

## Second goal turn: CPU alignment and arithmetic work

- Corrected a rollout position error: trace `input_position` is the shifted
  token index `P+d+1`; native decoder memory/RoPE position is `P+d`.
- Added accepted-prefix cache reconstruction from `(token[j+1], raw target
  feature[j])` at decoder position `j`, with a fresh cache for each round and
  the final feature row deferred for the seed. Rebuild is a declared
  truncated-gradient boundary; proposal states/K/V remain differentiable.
  Synthetic tests cover missing/misordered feature positions and discarded
  stale proposals. Actual EAGLE K/V, mask and RoPE parity remain unverified.
- Distinguished **raw target verifier logits** from the existing native head
  capture's draft logits mapped into target vocabulary. The trace validator
  now requires explicit raw-target provenance before computing mapped mass.
  This prevents using `EAGLE_CAPTURE_FULL_LOGITS` as a target-mass source.
- The capture audit now requires raw F32 7,680-wide feature rows with tap
  order `[2,18,33]`, accepted-prefix ancestry and absolute positions, joined
  to every round anchor. It rejects missing, misordered, rejected-branch and
  unjoined feature rows on synthetic fixtures. An audited CPU loader now
  supplies a round's prefix tokens and raw feature rows directly to the
  prefix-cache rebuilder after one manifest audit. The native writer does not yet
  produce this ledger, and numerical target-feature parity is unverified.
- Added an explicit CPU `group_matmul` training arithmetic option that keeps
  hard signs/A16/group scales but changes in-group F32 reduction order. On
  430 archived native D outputs, 395 matched exactly; max absolute discrepancy
  `0.0001220703125`, max scaled discrepancy `7.422315e-7`. The ignored
  diagnostic report is `results/binary-scale-fitting-5080/recurrent-binary-group-matmul-drift.json`
  (SHA256 `aa5f8c5de8789eaee1911264aec131d5be28ac60ca92f4268ef1c33bcdbf1550`).
  Checkpoint manifests now declare the training arithmetic. Its effect on
  argmax, gradients at scale, and native recurrence has no real-model gate.

This is the **second consecutive goal turn** under the user's no-GPU
restriction. It made substantive CPU progress; a GPU-only impasse has not
been established. Meaningful local work remains on a faithful full-drafter
adapter and native capture writer/parity fixtures. Do not
mark the native Goal blocked on this turn.

## Third goal turn: loader gate and full-drafter adapter

- Pushed llama.cpp fork commit `7f23c89b3` before advancing the parent
  gitlink. A plain CPU-only CMake build of `libllama` succeeded on Apple M3
  Max. All eight native loader fixtures passed with `n_gpu_layers=0` and
  `CUDA_VISIBLE_DEVICES=''`; the newly learned-scale v2 fixture loads, while
  an unknown rule is rejected. This closes the metadata-loader gap only.
- Read the pinned AngelSlim decoder source at revision
  `0358da9c651e6a7d7ccafea26ced4b9c98d11681` and verified its source
  hashes. The official forward reads `fc.weight.dtype`, which the new binary
  module does not expose; using it directly would also retain F32 K/V where
  the native graph writes F16 K/V before attention. A strict explicit CPU
  decoder-step adapter is now implemented. It preserves separate
  embedding/feature norms, embedding-first fusion, F16 K/V cache boundaries,
  RoPE, residuals and the final pre-norm state. Five synthetic CPU tests pass,
  including an independent two-step reference and a full prefix-rebuild to
  optimizer-step path; actual native full-drafter parity is still missing.
- A strict CPU initializer read the frozen D GGUF at SHA256
  `10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf`:
  all nine packed/sign-scale pairs, 32,000 absolute draft map IDs, shapes,
  metadata and tail bits passed. D contains **17,005 exact-zero group scales**.
  Two synthetic corruption/hash tests pass. Ignored audit report SHA256:
  `f91c3d09f71079019cf517c08805bb805fd7a5b1bfdb1ff6aea6d84617bcb4be`.
- A read-only GGUF view verified the actual pinned 7.5 GiB FP16 target file
  hash and the D draft hash, memory-mapped the 151,936-by-2,560 target
  embedding, copied only requested CPU F16 rows, and supplied the four
  F32 draft norm arrays. No target forward or full embedding copy ran.
  An adapter test binds copied norms and uses the external frozen embedding
  lookup even when the drafter's borrowed module is BF16; seven decoder-step
  tests and five frozen-operand tests pass on synthetic models. The ignored
  two-row/four-norm audit is
  `results/binary-scale-fitting-5080/recurrent-frozen-operands-audit.json`
  (SHA256 `5070724d9d2dc1a567c727980529bd15e26cbf0ff2372747b54e14c295ad076d`).

This is the **third consecutive goal turn**. It made meaningful CPU progress;
the no-GPU restriction has not created a true impasse. The native Goal stays
active. No actual target/drafter inference or accelerator execution occurred.
Third-turn parent commits `79208ef`, `7c91809`, and `cf593c8` are pushed to
`main`; the parent gitlink references the previously pushed llama.cpp fork
commit `7f23c89b3`. The temporary decoder and frozen-operand worktrees and
branches were retired after their reviewed content was integrated.

## Fourth goal turn: exact-prefix raw target logits

- Fork commit `c282087a9` was pushed before the parent gitlink update. It
  writes bounded raw target-verifier F32 logit rows **before** sampler
  processing under `EAGLE_CAPTURE_TARGET_LOGITS`, alongside existing
  `EAGLE_CAPTURE_PREFIX` head rows. The existing optional draft-head-logit
  file remains separate and explicitly labeled. A local `llama-server`
  compilation passed with Metal, CUDA, Vulkan, SYCL, HIP, RPC, Accelerate
  and BLAS disabled. No model/server inference was run.
- The CPU capture audit now joins the native raw-target file to exact-prefix
  rows by unique row index, target vocabulary width, SHA256 and source label.
  Synthetic tests reject missing/corrupt data and substitution of the old
  mapped draft-head logit file. `verifier_reached` is retained as a separate
  online diagnostic; valid supported cloned-verifier rows after a live
  rejection can still contribute later-position teacher-forced CE.
- The remaining native **raw target-feature** writer is unsolved. The target
  feature taps are available after target decode, but a raw dump could include
  rejected speculative branches. It must carry task/prefix ancestry and be
  filtered to the accepted-prefix ledger before the CPU audit can admit it.
  Real-model capture and numerical student/cache parity still need later
  user-authorized GPU execution.

This is the **fourth consecutive goal turn**. It made meaningful CPU-only
progress; no no-GPU impasse has been reached. The native Goal remains active.

**Published checkpoints:** `7744d8e` created the goal/protocol; `1e3742d`
integrated the trace contract; `a4dd003` integrated learned GGUF export;
`fead52a` integrated the binary CPU reference; `792b7e0` published the
corrected CPU path, capture audit and arithmetic replay; `0e0e14b` clarified
Q/K export row order; `6cfbe54` added the differentiable rollout contract.
All are on pushed `main`. The three temporary feature worktrees were clean and removed after
their reviewed content was integrated; their branches were removed. No
llama.cpp submodule commit or parent gitlink changed.

## Research choices pending

The [decision log](../DECISIONS.md) records options for scale zero handling,
teacher-forced versus refreshed student prefix distribution, and the proposed
500-step/45-minute all-body budget. These do not prevent CPU contract work.
The user owns approval of a later GPU budget and any change to the no-GPU
restriction. Do not request or infer that change.

## CPU stop gate and next actions

1. Implement a native capture writer for raw target feature taps with
   task/prefix ancestry, then filter to accepted-prefix rows and validate
   against the CPU ledger audit. The new raw verifier-logit writer is present
   but still needs real-model runtime validation. This requires careful
   llama.cpp changes and later model execution for parity.
2. Connect prefix rebuilding to the pinned full drafter only with verified
   attention/RoPE/KV rounding and mask semantics; do not substitute saved
   normalized head states. Keep the sequential reference and grouped-matmul
   training forward distinct in all reports.
3. Recheck CPU gates and record hashes/commits at the next milestone. Keep
   model files/raw captures out of Git. Audit whether remaining local work
   can still advance the goal. This is the
   **fourth** goal turn under the no-GPU restriction. Do not mark the Goal
   blocked until the same GPU-only impasse persists across at least three
   consecutive goal turns and no meaningful CPU-only work remains.

**Completion evidence still required:** real-model target-prefix capture and
alignment, whole-drafter unroll parity including draft K/V and masks, trained
GGUF loader/export checks, native acceptance, graph and latency/full-throughput
measurements against matched Q4_0. None is established by CPU tests alone.
