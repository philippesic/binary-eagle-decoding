# Joint binary EAGLE body and head

**Objective (amended 2026-09-28):** complete Phase 1 of the
[one-bit drafter plan](../W1_RESEARCH_PLAN.md): EAGLE W1 latency accounting and
bounded runtime improvements, an efficient jointly trainable body/head pipeline
for declared W1A16/A8/A4/A1 representations, and substantially expanded training
data. Evaluate native draft quality and speed against Q4_0 when GPU work is
explicitly permitted. Preserve the pinned FP16 target/verifier and draft map.
This continues the existing goal rather than opening a second goal.

**State: active CPU-only implementation and preparation; all GPU access paused.**
The user requested a fresh team after the audit. No joint GPU training or final
evaluation has run. The [fresh-team handoff](#fresh-team-handoff-eagle-w1-cpu-phase)
and overarching plan supersede older next-action lists and the old numerical
parity prerequisite. The original [trial protocol](../../experiments/recurrent-binary-qat-plan.md)
is retained as historical design; the 96-prompt/500-step recipe is a smoke
reference, not the enlarged full-body experiment. This file and STATUS.md remain
the durable record.

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
  contain output-normalized states only. Source capture and a CPU drafter
  unroll now exist; their real-model alignment and cache parity remain needed.
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
  consistency and valid/support masks. Synthetic tests pass. It cannot prove target features or
  certify K/V cache parity.
- A model-independent CPU rollout contract now feeds each proposed token and
  the previous **pre-norm** student state into the next step without detaching
  the functional cache. Three tiny causal-cache tests pass: a later-only CE
  loss reaches the first key, value, state, fusion and head; decoder
  memory/RoPE positions advance from `P` to `P+1` while shifted input-token
  positions are `P+1`, `P+2`; invalid terminal rows create no decoder call. This checks the
  training interface, not actual EAGLE mask or K/V byte parity.

**Latest CPU checks:** 91 recurrent tests, 16 capture-runner tests, ten explicit decoder-step tests,
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
- At this checkpoint, native **raw target-feature** capture was still open.
  The following milestone adds the source stream and accepted-prefix filter;
  real-model capture and numerical student/cache parity still need later
  user-authorized GPU execution.

This is the **fourth consecutive goal turn**. It made meaningful CPU-only
progress; no no-GPU impasse has been reached. The native Goal remains active.

## Fifth goal turn: raw feature stream and offline joins

- Fork commit `ddcf2a608` was pushed before updating the parent gitlink.
  It captures three draft-declared raw target layer-input taps after each
  successful target decode, before EAGLE fusion, with task/slot, exact token
  ancestry, position, batch-row identities and a later disposition per row.
  For `A` accepted drafts, input rows `j<=A` are retained; `j>A` are a
  rejected suffix. Capture is opt-in, bounded, and fails on overflow or
  unsupported reuse, shift, multimodal, replay or multi-slot paths. A
  CPU-only `llama-server` build passed with every optional accelerator and
  BLAS backend disabled. No target/drafter inference was run.
- Parent script `prepare_recurrent_native_rows.py` validates native
  proposal/round joins, frozen training split, verifier labels, support,
  offset mapping and optional raw target-logit indices. It emits rows,
  anchors, offsets and `t2d` with source hashes. A source cell manifest
  verifies request/task ownership; a plain task map is flagged unverified.
  Eight synthetic CPU tests pass.
- Parent script `prepare_recurrent_native_features.py` verifies one
  disposition per raw feature row, exact prefix and target tap order, and
  selects only retained rows needed by round anchors. Four synthetic tests
  cover zero and later-depth acceptance, rejected suffix exclusion,
  broken ancestry, missing disposition, nonfinite values and cell-manifest
  file hashes. The final capture audit still needs one combined native run;
  no real feature-value or microbatch row parity is claimed.

This is the **fifth consecutive goal turn**. It made meaningful CPU-only
progress. The native Goal remains active; the no-GPU restriction has not
created a complete impasse. Remaining CPU work includes an end-to-end
synthetic capture bundle, cross-round continuity checks and a bounded
native runtime capture protocol. GPU-dependent quality and timing gates
remain unauthorized.
Parent commits `cae7184` and `c51a8a7` are pushed to `main`; the latter
points to already-pushed llama.cpp fork commit `ddcf2a608`. Temporary native
row-preparer and submodule feature-capture worktrees/branches were clean and
retired after integration. No live process or remote session belongs to
this goal.

## Sixth goal turn: bounded capture assembly and continuity

- The native capture runner gained a `recurrent-train` mode in commit
  `3454f7a`. It requires the frozen 96-prompt hash, pinned FP16 target and
  candidate-D draft hashes, and the D absolute vocabulary map. Its D/D
  own-history policy requests bounded raw target logits and target features,
  disables context shift/cache reuse/checkpoint replay, and records task and
  request row ranges plus raw-file hashes. This mode has only been exercised
  against mocked CPU server output; no native server or model was run.
- `scripts/audit_recurrent_continuity.py` checks complete prefill, native
  speculative target input rows `j=0..len(D)`, retention through the
  accepted depth, and consecutive round prefixes and seeds. It rejects
  unexplained gaps, task/slot changes, misplaced target-only inputs and
  contradictory terminal input tokens. It verifies internal acceptance
  ancestry, not the initial target sample, final token emission, stopping,
  complete requests or numerical target features.
- The bundle builder in `be0e2af` joins cell-owned raw capture hashes, both
  preparer outputs and the final CPU capture audit. An optional continuity
  report must name the same round, event and task-map sources. A subsequent
  provenance gate in `20d6e26` checks pinned target/draft file hashes and
  the canonical little-endian D map digest. Its output is
  explicitly `training_eligible: false` because runtime feature parity,
  full-drafter numerical/cache parity and live model evidence remain open.
  The combined focused checks pass: 73 recurrent tests and 16 capture-runner
  tests, Ruff and diff hygiene. No accelerator, GPU host, native model
  inference, training or final-set use occurred.
- Parent commits `3454f7a`, `be0e2af` and `7f10643` are pushed to `main`.
  The two temporary capture worktrees were clean; their content matched the
  pushed integration and their worktrees/branches were retired. No native
  submodule revision or gitlink changed in this turn.

This is the **sixth consecutive goal turn** under the no-GPU restriction.
Meaningful CPU work closed the offline assembly and internal continuity
gaps. The native Goal remains active; a GPU-only impasse has not yet been
established.

## Seventh goal turn: actual CPU-only capture path

- The first CPU launch of pinned target plus D draft failed during draft
  loading because the `recurrent-train` runner omitted the mandatory
  `GGML_W1AX_ACT_BITS=16` setting. The runner and its mock test now set/check
  it. No request was served in that failed launch.
- A retry ran one frozen training prompt for eight output tokens with both
  model GPU-layer counts at zero and all accelerator/BLAS backends compiled
  out. The server stopped cleanly. It recorded four rounds, 16 head states,
  16 raw target-logit rows and 53 raw target-feature rows. The actual raw
  streams passed internal continuity, both native preparers and the final
  preparation-only bundle audit. A new response auditor joined all eight
  raw response IDs to the initial seed, four native round emissions and a
  terminal `no_proposal` trace. The terminal eighth token still has no
  independent raw target-logit check. The [CPU capture report](../../experiments/recurrent-binary-cpu-capture-smoke.md)
  has pinned identities, counts, SHA256s, hardware and limitations. Raw
  files remain ignored under `results/recurrent-binary-cpu-smoke-20260928-a16/`.
- A second CPU request with target and draft `n_ubatch=1` (versus 512)
  emitted identical raw IDs and canonical rounds. All 53 raw target-feature
  rows, 16 native head-state rows and 16 raw target-logit rows were bitwise
  identical, and both continuity/response audits passed. This is a bounded
  scheduling check, not independent numerical target-feature parity. The
  repeat remains ignored under
  `results/recurrent-binary-cpu-smoke-20260928-ubatch1/`.
- Real candidate-D operands and captured raw features now instantiate the
  full 2,560-hidden/4,096-Q CPU adapter. Its previous `hidden == Q width`
  guard rejected the valid native geometry and was corrected; a context
  decode path skips unused logits without changing state/KV values. The
  first round rebuilt 31 cache positions and replayed all five native
  proposal tokens. Both native-order and grouped-matmul arithmetic matched
  all five mapped top IDs, but normalized head-state max differences grew
  to about 0.0034. Numerical state/KV parity is **not** established; the
  two ignored per-depth reports and exact source hashes are in the CPU
  capture report.
- One diagnostic two-position CPU SGD step on the audited real capture
  produced finite nonzero sign and scale gradients in all nine binary
  projections. The 839 MiB checkpoint exported to a 38 MiB GGUF with
  truthful learned-scale metadata. All nine scale tensors changed versus D;
  the packed sign words remained unchanged after the small step. A CPU-only
  native server loaded all nine exported linears and returned the same eight
  raw IDs for this prompt. The [CPU capture report](../../experiments/recurrent-binary-cpu-capture-smoke.md)
  records hashes and limits. This is a one-step integration diagnostic, not
  approved full training or evidence against Q4_0.
- Seventh-turn parent commits `ca10b03`, `2b25d69`, `dca1487` and
  `1ce667c` are pushed to `main`. No llama.cpp submodule revision or parent
  gitlink changed. The CPU server process groups stopped with return code
  zero; no project remote session, GPU owner or temporary worktree exists
  for this turn.
- This closes a real runtime-format gap for a single request only. It does
  not establish numerical target-feature parity, whole-drafter state/KV
  parity, 96-prompt capture completeness, training quality or native Q4_0
  acceptance and throughput. The native Goal stays active. No GPU,
  accelerator, remote host, final prompt or approved training trial ran.

## Eighth goal turn: independent CPU target-feature comparison

- A CPU-only rerun with `--flash-attn off` retained the eight raw output
  IDs and passed continuity/response audits, but changed target-feature
  values versus `auto`. Grouped-matmul first/fifth-depth normalized state
  max differences fell from `0.002105/0.003462` to `0.001696/0.002143`;
  native scalar order under `off` measured `0.001756/0.002401`. All five
  mapped top IDs still matched. This implicates attention execution as a
  possible source of drift, without proving its cause or closing parity.
- An independent local Hugging Face Qwen3 forward on the Apple M3 Max CPU
  used F16-rounded source weights and F32 eager computation. The 32 prompt
  embedding rows and sampled target FFN gate matrices at layers 2, 18 and
  33 matched the pinned GGUF bitwise. Against the native `off` capture,
  median relative row L2 errors at those taps were 0.579%, 0.382% and
  0.280%; the largest relative row error was 1.628%. The exceptional
  first-token absolute error was small relative to that row's >335 RMS.
  The repeatable CPU checker hashes the source shards and rejects changed
  feature ancestry or payload size. Its three focused tests pass.
- The [capture report](../../experiments/recurrent-binary-cpu-capture-smoke.md)
  records exact SHA256s and numerical limits. This independent path is
  alignment evidence, not exact target-feature or full-drafter parity.
  No GPU, accelerator, remote host, final prompt, new training step or
  Q4_0 performance measurement ran. The native Goal remains active.
- Parent commit `c4f4849` is pushed to `main`. The optional
  `target-feature-cpu` dependency group pins Transformers 4.57.1; raw
  captures and source model weights remain outside Git. No remote session
  or native server process remains active for this turn.

## Ninth goal turn: native CPU draft graph boundaries

- Fork commit `b4df1b547` adds opt-in, CPU-host-only draft graph capture
  with bounded F32 payload, tensor/layout/execution JSONL and a complete
  footer. The source-level names `result_norm` and encoder `fc_out` did not
  appear in the scheduler callbacks during one real request; a bounded
  40-name node probe confirmed this. The callback does capture prenorm and
  `result_output`, correctly labeled as mapped **draft** logits.
- The final `auto` and `--flash-attn off` one-prompt CPU traces stopped
  cleanly and reproduced each untraced setting's output IDs, head states,
  target features and raw verifier logits bytewise. The `auto` graph trace
  recorded 26 decoder groups and 390 tensor rows under a 64 MiB limit.
  Prenorm plus the frozen output norm joined one graph column to native
  head row zero with RMS `6.74e-8`; the next nearest candidate was `0.577`.
  This is a numeric join because the normalized graph node was unavailable.
- The CPU adapter's seed-step taps matched native input embedding, both
  input norms, concatenation and Q/K/V **bitwise** after Q/K row-order
  conversion. RoPE Q/K maximum differences were at most `3.55e-6`.
  Attention output was the first substantial gap (max `0.009986` with
  Flash Attention auto, `0.005002` off). A separate replay using native
  Q/K/V and F16-rounded K/V reproduced each attention gap, constraining
  the cause to attention execution rather than binary projections.
  The mapped draft-head top ID remained `3070`; its 32,000 finite logits
  differed by RMS `0.001128` in the `auto` seed step.
- The [capture report](../../experiments/recurrent-binary-cpu-capture-smoke.md)
  records hashes, detailed limits and the unreconciled full-parity gate.
  No GPU, accelerator, remote host, final prompt, approved training trial
  or Q4_0 performance run occurred. The native Goal stays active.
- Fork commit `b4df1b547` was pushed to the user's llama.cpp fork before
  the parent gitlink update. The final diagnostic server process stopped
  with return code zero; no remote session or GPU owner exists. Parent
  integration commit `2c9dce2` is pushed to `main` with that gitlink.

## Tenth goal turn: broader CPU proposal-chain check

- A new bounded CPU diagnostic runner validates the pinned FP16 target,
  candidate-D draft, full 96-prompt training-file hash, native source
  revision and every disabled accelerator/BLAS backend before launching.
  It forces target and draft GPU layers to zero, captures one selected
  train-only prompt, stops the process group, and runs continuity plus
  response audits. Safety fixtures reject an enabled Metal build.
- The runner captured code and reasoning training prompts at eight output
  tokens each; the earlier prose capture supplied the third category.
  All three single-request captures passed the native continuity and
  response-emission checks. Code had four rounds with acceptance depths
  `[0,1,0,1]`; reasoning had three with `[0,1,2]`; prose had four with
  `[0,0,1,1]`.
- The CPU grouped-matmul student rebuilt a fresh accepted-prefix cache for
  every one of the 11 rounds using unique retained target-feature rows,
  then replayed the proposal chain. Its mapped top ID matched native D at
  **all 44/44 positions** (prose 16, code 15, reasoning 13). Maximum
  normalized state differences were `0.003462`, `0.004617` and `0.004564`
  respectively. Native-order first-round checks also matched 5/5 top IDs
  in each category. First-round graph taps in both new prompts again put
  the first material disagreement at attention, not Q/K/V projection.
- The [broader CPU report](../../experiments/recurrent-binary-cpu-broader-diagnostic.md)
  and ignored `results/recurrent-binary-cpu-broader-20260928/summary.json`
  record raw hashes, per-round joins and limitations. The three-prompt
  result does not establish the 96-prompt capture, numerical K/V parity,
  trained-model quality or native Q4_0 performance. No GPU/accelerator,
  remote host, development/final prompt or additional training step ran.
  The Goal remains active.
- The user noted that the RTX 5080 is available. The goal's explicit
  no-GPU restriction remains in force pending clarification; availability
  alone was not treated as authorization. CPU work continued independently.
- Parent commit `06a74a6` is pushed to `main`; the llama.cpp fork gitlink
  remains at the already-published `b4df1b547`. The two new CPU server
  process groups stopped cleanly, and no remote session or GPU owner was
  created.

## Eleventh goal turn: post-acceptance CPU graph joins

- The draft-graph comparator now selects each round's actual depth-zero
  head-state row. It joined prose round 3, code round 2 and reasoning round
  2 after earlier native accepted drafts. The frozen D output norm matched
  each native prenorm graph column to its head row at RMS below `1.3e-7`,
  with the next nearest candidate above `0.48`.
- Fresh CPU prefix reconstruction selected unique retained target-feature
  rows at every position. At the selected later seed steps, the native and
  CPU input embedding and unrotated Q/K/V were bitwise exact. Fused-input
  differences were zero to `2.4e-7`; RoPE Q/K stayed within `1.56e-5`.
  Attention output was again the first material gap, with maxima
  `0.011560`, `0.011746` and `0.009838` for prose/code/reasoning.
  The mapped seed top IDs matched in all three joins.
- The [broader CPU report](../../experiments/recurrent-binary-cpu-broader-diagnostic.md)
  records source hashes and per-round limits. This constrains accepted
  prefix ancestry and projection alignment; it does not directly read
  native stored K/V bytes or prove attention mask/reduction parity. The
  original no-GPU restriction remains in force pending the user's answer
  about the now-available RTX 5080. No accelerator work ran, and the Goal
  remains active.

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

1. Keep the 44/44 mapped-top diagnostic distinct from training quality.
   Directly compare F16-rounded draft K/V writes and cache positions for
   the post-acceptance joins, then expand fixed CPU comparisons before
   proposing any state tolerance. Keep the
   preparation bundle ineligible until full request, model and drafter
   parity gates pass; do not infer the 96-prompt training result from one
   prompt.
2. Keep accepted-prefix rebuilding tied to fresh current-student cache and
   recorded target features. Investigate native attention reduction/mask
   differences before treating PyTorch states as numerically interchangeable
   with llama.cpp; preserve the sequential and grouped-matmul modes.
3. Recheck CPU gates and record hashes/commits at the next milestone. Keep
   model files/raw captures out of Git. Audit whether remaining local work
   can still advance the goal. This is the
   **eleventh** goal turn under the no-GPU restriction. Do not mark the Goal
   blocked until the same GPU-only impasse persists across at least three
   consecutive goal turns and no meaningful CPU-only work remains.

**Completion evidence still required:** the full frozen 96-prompt real-model
target-prefix capture and alignment, whole-drafter unroll parity including
draft K/V and masks, trained GGUF loader/export checks, native acceptance,
graph and latency/full-throughput measurements against matched Q4_0. The
three existing single-prompt CPU captures do not establish these gates.

## Rotation handoff (2026-09-28 06:35 UTC)

- **Objective and restriction:** implement and CPU-validate the jointly
  trainable binary W1A16 EAGLE body and head against pinned FP16 target and
  candidate D, preserving Q4_0 EAGLE as the primary later acceptance,
  latency and total-throughput baseline. The user's explicit prohibition on
  every GPU and accelerator, including RTX 5080, Metal/MPS and remote hosts,
  is still in force. The user said “5080 available now”; a clarification
  asking whether that lifts the prohibition is pending, with no answer yet.
- **Completed this session:** round-aware graph comparison joined post-native-
  acceptance prose/code/reasoning seed steps at the exact pre-attention
  projection boundary. A new CPU diagnostic reconstructed all 37 code-round-2
  context positions from retained target features and joined one native graph
  column per position. After Q/K row conversion and F16 rounding, key write
  operands matched 37,681/37,888 elements and value operands matched
  37,723/37,888. The remaining 207/165 mismatches and unread native cache
  bytes keep K/V storage, rollback and attention parity open. See the
  [broader CPU report](../../experiments/recurrent-binary-cpu-broader-diagnostic.md)
  and ignored `results/recurrent-binary-cpu-broader-20260928/code-auto/round_02_projected_kv_writes_native_rows.json`
  (SHA256 `15e50ae2f8f385195c87ff752c68472884667aeb44a80f07e6fbbf65baf0403e`).
- **Published code:** parent `dce7f43` added the round-aware graph comparator
  and later-round report; parent `acfdab7` added the projected K/V comparison
  and result interpretation. Both are pushed to `origin/main` (remote HEAD
  verified at `acfdab7` before this handoff update). The llama.cpp gitlink
  stays at already-published fork commit `b4df1b547`. The following checkpoint
  commit records this handoff.
- **Checks:** seven focused graph-join unit tests pass; Ruff lint/format and
  `git diff --check` pass. The prior milestone's 91 recurrent, 16 capture-
  runner and ten decoder-step tests passed, as recorded above. The code-round-
  2 diagnostic completed on Apple M3 Max CPU with `CUDA_VISIBLE_DEVICES=''`
  and `PYTORCH_ENABLE_MPS_FALLBACK=0`; no accelerator, final prompt, training
  run or new Q4_0 measurement occurred.
- **Ownership and processes:** only the retiring orchestrator task is active.
  All listed subagents have completed; no worker owns unmerged work for this
  goal. No native server, local experiment, GPU process, remote tmux session,
  remote run directory or GPU owner exists. Main is clean before this handoff
  document edit. The earlier invalid ignored K/V report without Q/K row
  conversion must not be used.
- **Exact next CPU actions:** first, verify the handoff commit is pushed and
  that main is clean. Then run `scripts/check_recurrent_kv_writes.py` for the
  prose round 3 and reasoning round 2 joins already recorded in the broader
  CPU report, keeping all raw reports under ignored `results/`. Analyze the
  nonmatching F16 operands by position and whether attention arithmetic or
  row joins explain them; do not claim native cache-byte parity from graph
  write operands. Continue full frozen-96 capture preparation only under the
  CPU safety gate, with capture bundle ineligible until exact-prefix,
  model and drafter parity gates pass. Checkpoint and push the next milestone.
- **User-owned decisions:** the no-GPU restriction change, all-body training
  budget, scale boundary, prefix distribution and numeric tolerance remain
  pending in [DECISIONS.md](../DECISIONS.md). Keep independent CPU work moving
  while the 5080 clarification is unanswered. If the user authorizes GPU work,
  consult [AGENT_OPERATIONS.md](../AGENT_OPERATIONS.md) for the live host file,
  tmux-only SSH, unique remote run and owner protocol before any experiment;
  do not infer a training budget from hardware availability.

## Twelfth goal turn: native-style CPU norm and RoPE

- The handoff commit `6ca070f` was verified both locally and at
  `origin/main`; the inherited `main` checkout was clean. The successor
  created its own native Codex Goal with the same objective and used a
  temporary `kv-parity` worktree for changes.
- Prose round 3 and reasoning round 2 projected-write comparisons completed
  from the existing CPU captures. Before the correction, prose matched
  35,666/35,840 F16 key and 35,727/35,840 value elements; reasoning
  matched 49,688/50,176 key and 49,781/50,176 value elements. Each context
  position had one native graph-row join. All value mismatches in the three
  captures coincided with one or two F16 fused-input differences at that
  position, traced to tiny embedding RMS-norm rounding differences.
- The adapter now follows ggml's F64 sum of F32 squares, F32 mean and F32
  scale/product sequence, plus recurrent F32 RoPE frequencies. On Apple M3
  Max CPU, all 619,520 F16 fused-input elements and all 123,904 raw F32 K
  and V projection elements across 121 selected context positions matched
  native graph columns. All 123,904 F16 value write operands matched;
  123,900/123,904 F16 key write operands matched. The four remaining key
  differences are after RoPE, with maximum F16 difference `0.00048828125`.
  The [broader CPU report](../../experiments/recurrent-binary-cpu-broader-diagnostic.md)
  records individual counts and ignored report SHA256s.
- A fresh code round-2 full CPU proposal replay retained every mapped top
  ID and reduced the first-depth maximum state difference from `0.0025558472`
  to `0.0024642944`; whole-drafter numerical parity is still open. The
  focused 91 recurrent and 17 native-step/graph-join tests passed. Ruff
  and diff hygiene pass. No new native model inference, training, final
  prompt or GPU experiment ran.
- The user explicitly authorized use of the free RTX 5080. A tmux-only
  read-only host check found 0% utilization and no project experiment
  process. The remote `main` was behind and had an unrelated untracked
  rescue directory; neither was changed. The SSH/tmux session was closed.
  The proposed all-body budget, scale boundary, prefix distribution and
  tolerance decision remain user-owned.

**Next parity gate:** read native stored K/V bytes and check cache positions
and rollback against the projected operands, then isolate attention
mask/reduction arithmetic. Keep the preparation bundle ineligible and the
96-prompt capture/final-set gates separate. Use the authorized RTX 5080 only
for a specific bounded parity/capture need under the host and run protocol;
do not start the unapproved all-body training trial.

## Thirteenth goal turn: actual stored draft cache and mask

- The published fork commit `0abe6e586` adds an opt-in, bounded CPU-only
  capture after EAGLE draft graph completion. It synchronizes the graph,
  reads actual F16 K/V cache rows at physical slots after native writes,
  and records the causal mask input. It rejects nonhost, transposed,
  multi-sequence or unexpected geometry. The parent integration commit
  `0c2ddc1` points to the published fork commit and adds a strict offline
  byte/mask auditor, three corruption fixtures and the
  [cache report](../../experiments/recurrent-binary-cpu-cache-parity.md).
  Both commits are pushed to their respective remotes.
- Final committed-source CPU runs on the pinned prose and reasoning training
  prompts passed continuity, response and cache audits. Across 26/20
  decoder executions and 69/76 stored rows, **70,656/70,656** and
  **77,824/77,824** native F16 key and value elements matched the same
  graph execution's F16-rounded projected write operands. All 145 masks
  allowed exactly physical slots `0..position` and blocked later slots,
  including after eight rewritten positions in each request. Two dummy
  reserve rows per run are included in those totals. Both eight-token raw
  responses and all checked raw graph/feature/head/logit streams matched
  their earlier uninstrumented captures bitwise.
- The Apple M3 Max server build disabled every optional accelerator,
  BLAS and Accelerate backend; both model GPU layer counts were zero.
  Server process groups stopped with code zero. Synthetic audit corruption
  checks passed 3/3, CPU runner policy checks 2/2, Ruff and diff hygiene
  passed. The final manifests have SHA256
  `4ad342145a1b77deb5f48cac4e00671e26b691850f398be56687ef05830b7169`
  (prose) and
  `99b9003698bcdf5505421a1667dac4d21a8545471d91330c04181c9f2a931d40`
  (reasoning); raw files remain ignored. No GPU job, remote run, model
  training, final prompt or Q4_0 measurement ran.
- The stored-byte and simple contiguous-prefix mask gaps are closed for
  these two CPU requests only. Attention reduction still produces nonzero
  normalized state drift; general cache/rollback behavior beyond this
  single-sequence geometry, exact target-feature parity and full 96-prompt
  capture are open. The preparation bundle remains training-ineligible.

**Next action:** use the captured native mask and F16 cache bytes to isolate
the first attention-output discrepancy. Then prepare a bounded full frozen-96
capture under the user-authorized RTX 5080 with its live host registry,
tmux-only SSH, isolated run directory and one GPU owner. Do not infer the
all-body training budget or numeric tolerance from hardware availability.

## Fourteenth goal turn: CUDA capture path and full-run setup

- The user-authorized RTX 5080 passed a supervised eight-token capture smoke
  on one frozen prose training prompt. Both target and candidate-D draft
  models were fully offloaded under CUDA 13.1/SM120a with A16 draft
  activations. Four native rounds, 16 head rows, 16 raw target-logit rows
  and 53 target-feature rows passed continuity and response audits. The
  eight raw output IDs matched the earlier Apple M3 Max CPU request, while
  raw feature/logit/head payload hashes differed across backends. See the
  [CUDA smoke report](../../experiments/recurrent-binary-cuda-capture-smoke.md)
  (manifest SHA256
  `44b189aed1f603d9c56a8143c5c4e274d0f583ce5a4fe2198205def7435f267f`).
- Parent commits `d5a8374` and `f7e561b` are pushed to `main`. The first
  extends the pinned, bounded diagnostic runner with a CUDA-only build/device
  gate; four policy tests pass. The second permits explicit full-run caps
  of 65,536 verifier-logit rows and 131,072 target-feature rows; 16
  capture-runner tests pass. The former 8,192/65,536 caps could truncate
  the frozen 96-by-128 protocol. Ruff and diff hygiene pass.
- The remote GPU owner is this task. The live host registry was consulted;
  SSH uses tmux MCP session `recurrent-capture-5080` (`$15`). An isolated
  remote checkout at `checkouts/recurrent-gpu-capture-20260928/` has the
  published fork gitlink. The CUDA server build used the host's previously
  documented private CUDA/glibc compatibility include; no system header
  changed. The preserved remote D GGUF, frozen training file and D map
  passed their pinned file/raw digests. The smoke supervisor
  `recurrent-cuda-smoke-20260928` finished with code zero; afterward the
  GPU showed 0% utilization, 1,372 MiB whole-device use and no compute
  app or project server. No other GPU owner or run exists.

**Prepared next run:** update that remote checkout to `f7e561b`, then use
`scripts/remote_job.py recurrent-full96-20260928` for the frozen 96-prompt
D/D own-history cell at 128 output tokens per request, with explicit
65,536/131,072 raw-logit/feature limits. Keep its files under that run
directory. Stop and inspect its process group before releasing the GPU.
Audit request ownership, all raw file hashes, prompt/round continuity and
the preparation bundle before any training use. The all-body optimizer
budget and numeric tolerance remain user-owned; no training or reserved
final evaluation is authorized by this capture setup. The native Goal
remains active.

## Fifteenth goal turn: frozen 96-prompt CUDA capture, audit pending

- The user-authorized RTX 5080 ran the frozen 96-prompt, 128-output-token
  candidate-D own-history cell under supervisor
  `recurrent-full96-20260928` in the isolated remote checkout recorded
  above. Source checkout `dbdca38` includes the expanded explicit capture
  limits. The run started 2026-09-28 07:42:44 UTC and ended 07:50:00 UTC
  with supervisor exit code zero. All 96 requests have unique task/prompt
  ownership; target/draft/train/map identities passed the runner's pinned
  gates. Capture and cell manifests have SHA256
  `2b2f49861c010214d2e424ad49053c829acdd6c6dafccbad721406390ed888fd`
  and `f2f65984d6fc6857bc1857d9f484a0a2dbe371ea4c47a477a77687bdc621607b`.
- The raw capture has **40,815 head rows and 40,815 raw verifier-logit rows**
  (24,805,071,360 logit bytes), 52,297 raw target-feature rows
  (1,606,563,840 feature bytes), and 8,295 native rounds. Neither
  65,536/131,072 cap was reached. The native row preparer found all
  40,815 rows valid, 39,705 supported and 1,110 outside draft vocabulary;
  12,061 labels were reached by the live verifier. The feature preparer
  selected 15,042 rows for accepted-prefix anchors. Frozen-prompt
  continuity passed all 8,295 rounds. These are capture/alignment counts,
  not training quality or Q4_0 performance.
- The GPU supervisor and its process group are stopped. Final host check
  found 0% utilization, 1,372 MiB whole-device use, no compute app and no
  project server. No other GPU owner or remote job was started. The
  24.8 GB raw logit file and other source artifacts remain outside Git
  under this run directory.
- The final preparation bundle is still being built and audited. A new
  full-request auditor was pushed in parent commit `ed06972`; it has not
  yet run on these 96 requests. Do not mark the capture training-eligible
  from the runner's success alone. The all-body optimizer budget, numeric
  tolerance and reserved final evaluation remain user-owned.

**Immediate CPU actions:** finish the bundle audit, run the 96-request
raw-response auditor against the sealed files, record source/output hashes
and any rejection, then isolate native F16 attention arithmetic using the
captured CPU graph inputs. Keep the RTX 5080 free unless a specific new
bounded experiment is required.

## Sixteenth goal turn: full capture audited, training still gated

- A first bundle build was killed with exit 137 because the audit converted
  all 40,815-by-151,936 raw logits into Python float lists. Source captures
  and preparer outputs remained intact. Commit `51317fc` replaced that
  materialization with bounded F64 batches and preserved source/provenance
  and nonfinite-row checks. Eleven focused capture-audit tests and eight
  bundle tests passed. A second supervised CPU bundle build finished with
  code zero. Its manifest SHA256 is
  `8919cd05614f952f945bf6e1bf97a6bb8c30a13f47f5d02f3c7b169f93e8e31c`;
  audit SHA256 is
  `9a43b4a708664bb3f42287411595365b3b955fc4f83a67fab20010331909caff`.
  It verified all 40,815 raw target-logit rows and 15,042 selected feature
  rows, and measured mean mapped target probability mass `0.9719726884`.
  Its explicit `training_eligible: false` / `preparation_only` status remains.
- The all-request audit joined **96/96 responses**: 12,251 emitted tokens,
  8,295 rounds and 3,786 accepted drafts on training histories. One
  final round stopped at accepted EOS `151645`; its canonical verifier
  batch retained one un-emitted suffix token. Commit `bca41b1` allows
  only that final EOS prefix clipping, with focused positive/negative
  tests. The successful response-audit SHA256 is
  `8ebba18b41a58a886a71914298c139433832083bdac10a83001072f23afd4051`.
  Ninety-five requests ended at the 128-token cap and one stopped at EOS.
  The [full capture report](../../experiments/recurrent-binary-full-capture-5080.md)
  records sources, counts, hashes, the failed first attempts and limits.
- Independently, model-free Apple M3 Max CPU attention arithmetic ablations
  integrated as `4a448b8` reduced archived first-seed maximum native
  attention differences from `0.009986/0.005692/0.009899` to
  `0.001784/0.000000954/0.000847` for prose/code/reasoning by modeling
  F16 dot and online F16 value accumulation. Three focused tests, Ruff
  and source-hashed reports pass; this is an approximate numeric
  explanation, not exact whole-drafter parity. The
  [broader CPU report](../../experiments/recurrent-binary-cpu-broader-diagnostic.md)
  retains each stage and report hash. The temporary attention worker
  worktree/branch were retired after its pushed commit reached `main`.
- The full GPU capture and subsequent CPU audit supervisors exited zero;
  the failed audit runs are retained as evidence. The duplicate failed
  temporary bundle copy was removed only after the successful bundle
  manifest and audit were verified. Final RTX 5080 check again found no
  project process or compute app, 0% utilization and 1,372 MiB whole-device
  use. No training, development/final prompt, trained GGUF or Q4_0
  performance run occurred. The native Goal remains active.

**Next gate:** audit real-model target-feature numerical parity and the
remaining whole-drafter attention/state/logit gap before any training
eligibility declaration. Keep the user-owned all-body budget, scale
boundary, prefix distribution and numeric tolerance choices in
[DECISIONS.md](../DECISIONS.md). The full 96-prompt capture is complete
and should be reused; no repeat GPU capture is needed for the current
offline questions. Q4_0 remains the primary eventual acceptance, latency
and total-throughput baseline.

## Seventeenth goal turn: independent target-feature comparison

- The independent CPU Hugging Face target checker was run on existing code
  and reasoning native training captures after the earlier prose check.
  Source BF16 weights were rounded to F16; requested embedding rows and
  sampled gate weights matched the pinned GGUF. F32 eager computation on
  Apple M3 Max still differed from native CPU taps: median relative row
  L2 errors were 0.542/0.397/0.279% (code) and 0.501/0.569/0.490%
  (reasoning) for layers 2/18/33; largest rows reached 1.359% and
  2.088% at layer 33. The ignored reports are hashed in the
  [target-feature note](../../experiments/recurrent-binary-target-feature-parity.md).
- Parent commit `8e8f1a2` adds a bounded CUDA F16 mode to that checker,
  preserving the code-prompt CPU metrics exactly; four focused tests and
  Ruff passed. A supervised RTX 5080 run reused the eight-token prose CUDA
  native capture and local Hugging Face source shards. PyTorch 2.14.0+cu130
  and Transformers 4.57.6 in eager F16 computation matched all 32 prompt
  embedding rows, but median native-feature relative row L2 errors were
  0.311/0.426/0.385% at taps 2/18/33, with maxima
  0.789/1.367/1.753%. The ignored CUDA report SHA256 is
  `b2ee0c273b7b7a3a66abb61541ac8ee619728b43a04aabdf471fd0ad3b9c8a29`.
  It records source, model and capture hashes and device versions.
- That one-request diagnostic exited zero under remote supervisor
  `recurrent-target-feature-cuda-20260928`. Its process group stopped;
  the GPU returned to 0% utilization, 1,372 MiB whole-device use and no
  compute app. Native and independent forwards differ in attention and
  reduction arithmetic, and this F16 comparison is not a complete
  tensor-by-tensor target parity proof. Target-feature numeric parity and
  any predeclared tolerance remain open. No optimizer, development/final
  prompt or Q4_0 measurement ran. The native Goal remains active.
- A second supervised CUDA/F16 forward changed only Hugging Face attention
  to SDPA (parent commit `95b924d`, four focused tests). Its median
  native-feature relative row errors were 0.306/0.434/0.382%, versus
  0.311/0.426/0.385% under eager. The small mixed changes do not isolate
  the native discrepancy. The SDPA report SHA256 is
  `fb333b9e76238a46ebddb875bc2a351ae66f589dae8a433ba5314745197ddb60`.
  Its supervisor exited zero, process group stopped and RTX 5080 returned
  idle; no other project GPU job was created. Both tmux-only SSH sessions
  used for this turn were closed after process and device checks. The
  isolated remote checkout retains the ignored capture and diagnostic
  reports; no active GPU owner or remote experiment remains.

**Next action:** keep the full 96-prompt capture sealed; isolate the
remaining target-feature difference at the first divergent layer/row and
continue full-drafter attention/state/logit parity checks. Present a
concrete tolerance or exact-backend path with observed limits for the
user-owned research decision before any all-body training budget is used.

## Eighteenth goal turn: full training-prefix feature distribution and layer-0 probe

- Parent commit `4327679` added a strict full-capture feature diagnostic;
  six synthetic tests and Ruff passed. It verified all 96 frozen training
  request owners, raw hashes, complete prefill ancestry and first-round
  prefixes, selecting **3,112 prefill rows** of length 22–61. The supervised
  RTX 5080 CUDA/F16 eager forward finished with exit zero. Median relative
  row L2 discrepancy at target taps 2/18/33 was **0.293/0.538/0.488%**;
  p99 was 0.830/1.387/1.687%. Maxima were 2.433/10.014/2.779%.
  The 10.014% tap-18 row belongs to frozen training prompt
  `qat-revisit-train-code-data-validation-03`, position 3. All selected
  embedding rows and sampled gate weights matched the F16 GGUF. The ignored
  eager report SHA256 is
  `49d38214b1b59ad1597ee3111075f87fa05be9efa8ad24d91314c8f2374e93dd`.
- A second supervised full 96-prompt CUDA/F16 forward changed only independent
  attention to SDPA. Median errors became 0.288/0.531/0.480%; maxima became
  2.544/12.135/2.908%. The same tap-18 position rose to 12.135%.
  Source hashes were identical to the eager run. SDPA report SHA256 is
  `09a51dfb61361b2c3f7f6cb24aa2656d21a6673fc754090808353b0314cdceca`.
  Both report files remain outside Git in their separate supervised run
  directories beneath the isolated 5080 checkout. Both supervisors exited
  zero; no project process or compute app remained, and the GPU returned to
  0% utilization and 1,372 MiB whole-device use. The tmux SSH session was
  closed. These independent forwards do not establish a safe numeric gate.
- A read-only check of the same native capture found 103/427 cross-request
  prefill-row pairs with identical token prefixes but different feature bytes.
  First-token pairs matched; differences appeared at positions 1–5 and
  reached 0.473/0.801/0.821% relative row L2 at taps 2/18/33. The large
  tap-18 outlier's four-token prefix had no duplicate request. Numeric
  dispatch, batch shape and capture indexing remain open explanations; do
  not infer causal dependence on future tokens from this observation.
- A supervised one-prompt CUDA repeat of the 10–12% tap-18 outlier matched
  the full-run native prefill exactly: 29 token/position/prefix rows and all
  222,720 F32 feature values. The native binary SHA256 also matched the
  full capture. The passing eight-token repeat manifest SHA256 is
  `b5d4f5fae6b59c752742b9c45e6e770436c28fbbf1f48ba5fa3cc1cd148f6611`.
  An initial one-token capture produced identical prefill bytes but its
  runner audit failed because no speculative draft round was generated;
  failure logs remain. The second supervisor and native server exited zero,
  and the GPU returned idle with no compute app. The outlier is reproducible
  under this native backend, though its independent-forward cause remains
  unknown.
- Parent commit `c798fd3` adds an Apple M3 Max CPU ggml layer-0 norm/QKV
  operator probe. The pinned 32-token prose prefix embeddings, layer-0 norm
  and Q/K/V weights matched the F16 GGUF source. Native norm versus an F32
  reference differed by RMS `2.65e-9`; native pre-RoPE Q/K/V projections
  versus F32 matmul from the native norm differed by RMS
  `8.31e-5/9.16e-5/5.92e-5` and median relative row L2
  `0.086/0.087/0.142%`. Explicitly F16-casting the normalized input only
  reduced projection RMS to `8.27e-5/9.11e-5/5.88e-5`. Two focused
  native-fixture tests, Ruff and formatting passed. Ignored report SHA256:
  `d7bd1dd9b2b9beeca550682b67b142fdf8a29f92b1a5c82b60143edd696b2d62`.
  This identifies a CPU projection arithmetic difference before attention;
  it is not a native CUDA whole-model graph comparison and does not attribute
  the 10–12% tap-18 outlier. The probe worktree and branch were retired after
  equivalent content reached `main`.

**Next gate:** investigate native same-prefix differences, the first divergent
target operation and the large tap-18 outlier. Preserve the sealed
96-request source. Keep the full-drafter parity and user-owned numeric gate,
training budget, and final-set evaluation open. No optimization or Q4_0
performance comparison ran in this turn.

## Nineteenth goal turn: native tap-2 input intervention

- Parent commit `f5825e8` adds a bounded CUDA/F16 intervention on the sealed
  29-token outlier training prefix. It validates frozen capture ownership,
  source hashes and complete prefill, then replaces the input to independent
  Hugging Face decoder layer 2 with native tap-2 F32 rows cast to F16. Four
  synthetic tests, Ruff and formatting passed. The supervised RTX 5080 eager
  run exited zero; its ignored report SHA256 is
  `cebef815e90de17a8a5d4b597758422ef84f5bfe705e65a0032ea6a87c227a45`.
  At position 3, relative row L2 error fell from 0.324% to 0.020% at tap 2,
  but rose from 10.014% to 11.659% at tap 18. Median tap-18 error over all
  29 rows fell only from 0.611% to 0.573%. The early tap-2 difference alone
  does not explain the outlier under this independent forward. The F16 cast
  and downstream backend difference limit causal interpretation.
- The frozen full96 runner uses `--parallel 1`. Source tracing found that
  one-stream `split_simple` preserves caller row order through internal
  microbatches; sparse output reordering needs more scrutiny, but ordinary
  single-output prefill should not invoke its swap path. This lowers the
  likelihood of a prefill row permutation without proving capture indexing
  for every execution. The same-prefix differences remain unexplained.
- The CUDA supervisor stopped, GPU returned to 0% utilization and 1,372 MiB
  whole-device use with no compute app. No optimization or final-set prompt
  was used. The isolated intervention worktree was retired after equivalent
  code reached `main`. A separate isolated worker is preparing a strictly
  opt-in native target-layer input ladder to locate where layers 2–17 amplify
  the gap; it has not been integrated or run. The current tmux-only 5080
  session is `recurrent-tap2-intervention-5080` (`$20`) with no active job.

**Next gate:** review and CPU-build the bounded native ladder; then capture
one frozen 29-token prompt under supervision, prove original taps 2/18/33
unchanged, and compare layer-by-layer against the same F16 independent
forward. If the ladder changes old taps or fails its bounds, stop and keep
the existing capture sealed. Do not choose a numeric tolerance or begin the
all-body optimizer without the user's research decision.

## Twentieth goal turn: target layer-input ladder localizes the outlier

- Fork commit `87cdf11fb6fbfe5d35ab297ecae163c718a9593f` was pushed before
  parent gitlink commit `bfe5561`. The opt-in native target ladder captures
  only bounded prefill rows at layers `0–18,33`, separate from the unchanged
  EAGLE feature writer. Apple M3 Max CPU smoke passed 29 rows and 87/87
  byte-exact old-tap comparisons; the capture-off control preserved output
  and old feature bytes, and a 28-row limit rejected the 29-row request
  before ladder payload was written. The final source passed a full CPU
  server rebuild. Parent commit `ebbebe5` added an independent ladder auditor
  with 11 synthetic tests, source/row/hash checks and a bitwise old-tap gate.
  The existing one-prompt runner added explicit 20-layer/64-row/12.5 MiB caps;
  five focused runner tests and Ruff passed.
- The supervised RTX 5080 capture and CUDA/F16 independent comparison each
  exited zero on the frozen 29-token outlier prompt. All new native ladder
  taps 2/18/33 matched both same-run and sealed full96 raw feature bytes
  exactly. At prefill position 3, relative row L2 error rose from 1.209%
  at native layer-14 input to 4.148% at layer-15 input, then to 10.014% at
  layer-18 input. Absolute error RMS rose 0.01202→0.04218 across block 14
  while native row norms stayed near 50–51. This identifies the largest
  adjacent amplification at block 14, without attributing it to a particular
  suboperation. Another position has a distinct layer-6→7 jump.
- Capture manifest SHA256 is
  `72410d35fae0b1561fca0546e8e3b6e58a30506af75fd0802d10da865db6151d`;
  comparison report SHA256 is
  `9642d6c84e02377e8fcd83bc100be0cd57c3de0af9dbd663f9971cba0b1f79eb`.
  Both reports and raw ladder bytes remain outside Git in the isolated 5080
  checkout. The supervisors, native server and compute app stopped; device
  returned to 0% utilization and 1,372 MiB whole-device use. The remote
  tmux-only session `recurrent-tap2-intervention-5080` (`$20`) remains open
  for the bounded block-14 follow-up, with no active job. The full96 source
  remains sealed, and no optimization, final-set prompt or Q4_0 performance
  run occurred.

**Next gate:** inject captured native layer-14 input into the independent
block-14 forward to distinguish upstream perturbation amplification from a
block-local arithmetic gap. Then instrument only the implicated native block
operations if its same-input output still diverges. Keep the user-owned
numeric gate and all-body training budget unresolved.

## Twenty-first goal turn: block-14 amplifies upstream drift

- Parent commit `19b2246` adds a scoped native-input intervention to the
  audited ladder comparator; 13 focused synthetic tests, Ruff and formatting
  passed. The supervised RTX 5080 CUDA/F16 eager forward reused the exact
  29-row native ladder, again passing source hashes and old-tap byte identity.
  At position 3, the independent layer-14 input error of 1.209% fell to
  0.016% after replacing it with native F32 rows cast F16. The subsequent
  layer-15 input error fell from 4.148% to **0.149%**, and absolute RMS from
  0.04218 to 0.001516. This makes block 14 an amplifier of earlier state
  drift under this independent forward; it is not evidence of a large
  same-input block-14 operator mismatch. Exact native arithmetic still
  differs because the intervention casts F32 input to F16.
- Ignored report SHA256 is
  `e8945330179e87e1ac8ffa9d75124d19a372908a1c79b28f05ac490dfa30e8b5`.
  Its supervisor exited zero, no project GPU process remained, and the
  RTX 5080 returned to 0% utilization and 1,372 MiB whole-device use.
  The existing remote tmux session `$20` remains open, with no running job.
  No new native capture, optimizer, final-set or Q4_0 run occurred.

**Next gate:** compare every block `0..17` from its own captured native input
against the next native layer input, using the same bounded prefix and F16
cast. This local-block screen can rank where backend arithmetic first creates
material drift, separately from later amplification. Keep the user-owned
numeric tolerance and all-body training budget pending.

## Twenty-second goal turn: local target blocks and amplified state drift

- Parent commit `8c89e48` adds 18 independent CUDA/F16 no-gradient HF forwards,
  each replacing one target block's complete 29-row input with the captured
  native F32 ladder values cast F16. Baseline tensors remain immutable, hooks
  are scoped and removed, and only CPU metric summaries remain between
  forwards. Sixteen synthetic checks, Ruff and format passed. The supervised
  RTX 5080 run finished with exit zero; ignored report SHA256 is
  `e1ca603864bfd0454e14e5694e547c6bbfe1ba62dcffee5e17c1b6be7dab50f8`.
- At the position-3 outlier, **all 18 same-input block-output errors are at
  most 0.272% relative row L2**. Block 0 has that maximum from bitwise-exact
  native embeddings. Block 14's local output error is 0.149% and block 17's
  is 0.104%, versus their accumulated baseline errors of 4.148% and
  10.014%. Later native-input F16 casts add 0.016–0.026% at this position.
  The evidence supports distributed small local backend differences followed
  by amplification of state drift, rather than one 10% same-input block
  failure. The all-row RMS ranking is different: block 6 has 0.297 absolute
  RMS because of another position. This is a one-prompt diagnostic, not a
  general numeric tolerance or proof of exact operator parity.
- The new ladder run's first eight native output IDs matched the earlier
  no-ladder recapture. The local sweep reused the sealed ladder; no new
  native capture, optimization, final-set or Q4_0 run occurred. Its supervisor
  stopped and the GPU returned to 0% utilization, 1,372 MiB whole-device
  use and no compute app. Final remote state showed all four ladder-related
  supervisors `finished` with exit zero; the tmux-only SSH session `$20` was
  closed. The integrated target-ladder and auditor worktrees were archived.
  The fork commit remains published on `origin/research/target-layer-ladder`
  for the parent gitlink.

**Next gate:** audit the first local block and the largest all-row block-6
case at their exact native inputs, then decide with the user whether the
training gate requires exact target backend arithmetic or a predeclared
numeric/trajectory policy. Continue independent full-drafter attention,
state, logit and cache checks while that decision is pending. The captured
native target features and verifier logits remain sealed and are not yet
training-eligible.

## Twenty-third goal turn: native CPU attention oracle and real causal gradient

- Parent commit `22bf7ce` adds a standalone ggml CPU Flash Attention replay
  from archived same-run graph Q/output, actual F16 stored K/V cache bytes
  and exact-prefix masks. The Apple M3 Max helper matched **593,920/593,920
  native F32 attention elements bitwise** across 46 executions and 145
  query rows in the frozen prose and reasoning captures, including context
  batches and later physical slot rewrites. Maximum/RMS and every per-head
  relative L2 error were zero. Six focused tests with compiled native
  fixtures, Ruff and formatting passed. Reports SHA256:
  `b7ea8ca93f3e294770eb65d992ab2b3ce9b47735e2b13eb6fe264d89e2364fa4`
  (prose) and
  `b4cc1c95c17ec0326a17f23c56e8c159138fd2bbd182f6e1cfd7e021b02c47ee`
  (reasoning). The [oracle report](../../experiments/recurrent-binary-cpu-attention-oracle.md)
  records source/build hashes, 10-thread Apple CPU hardware and limits.
  This supplies an exact native operator reference for those captured
  operands; it does not yet replace the differentiable Python attention.
- Parent commit `ba96343` adds a no-optimizer, real-size depth-1-only CE
  probe on the audited one-prompt D bundle. On Apple M3 Max CPU, the
  later loss was `2.106572151184082`; the earlier proposal's pre-norm
  state had 2,560/2,560 nonzero gradient values and its appended F16-rounded
  key/value rows had 1,024/1,024 each. Depth-0 logits had zero gradient.
  All nine binary sign/scale groups had finite nonzero gradients in the
  shared graph, though those shared-weight gradients alone cannot attribute
  a depth-0 path. Six focused tests, Ruff and formatting passed. The ignored
  report SHA256 is
  `295c22adfae8b1bd0102769eddd0e5b51a0be6bf3c0807ef7d0ef3166101dc33`.
  The [causal-gradient report](../../experiments/recurrent-binary-real-later-gradient.md)
  records pinned source/map hashes and exact limitations. No optimizer,
  trained checkpoint, GPU work, final prompt or Q4_0 evaluation occurred.
- Both results were rerun from pushed `main` on Apple M3 Max, and their
  ignored reports were preserved under the main checkout's `results/`
  directory before the integrated temporary worktrees and branches were
  retired. The main checkout and published fork gitlink are clean. No local
  model server, remote session or GPU job remains active for this work.

**Next gate:** use the native CPU attention oracle to distinguish the
student's Q/K/V operand gap from its F32 attention arithmetic gap on the
same captured steps. If an optional exact-forward/surrogate-backward
attention mode is prepared, keep its arithmetic explicit and prove both
native output agreement and finite later-position Q/K/V/body gradients
before any training decision. The user still owns the full numeric gate
and all-body optimizer budget; captured native training data remains
preparation-only.

## Twenty-fourth goal turn: optional native-forward student attention

- Parent commits `c960078`, `24f140e`, `adf7166`, `2cb1d75` and `d286bf5`
  added an **opt-in CPU diagnostic** attention mode to the nine-linear
  student. Its forward calls the pinned ggml attention helper over 256
  physical slots; its backward explicitly recomputes the former F32
  attention derivative as a surrogate. The default student remains F32.
  Focused tests verified exact oracle output, strict mask/geometry and
  helper SHA, unchanged default behavior, Q/K row conversion and two-step
  causal derivatives. No optimizer or trained checkpoint used this mode.
- A seven-mode first-seed ablation isolated an initially missing row-order
  conversion. Student Q/K are half-split in Python and interleaved in
  native ggml. With both converted before the helper, the complete prose
  student attention matched all **4,096/4,096 F32 elements bitwise**;
  unconverted student Q/K had max error `0.007402`. Reasoning improved from
  max `0.009470` unconverted to `0.002172` converted, with 70 F16 key and
  66 value operand differences across its 47 visible cache positions.
  Student Q converted and paired with native K/V matched both first-seed
  native outputs bitwise. The ignored prose/reasoning ablation reports have
  SHA256 `3742bb7f6100e7877aa311d3cda1006599e9cc4257a7604d7d8ca6dc7997037f`
  and `3fe43dfd9b6bd76d49fcc8b92b37588a5215e94b948c2a237413a9a220fe4f42`.
- On the Apple M3 Max grouped-matmul D replay, corrected first-depth
  normalized-state maximum error fell from `0.002019` to `0.000184` for prose
  and from `0.002422` to `0.000511` for reasoning; all recorded top target
  IDs still matched native. The real-size depth-1-only CE check with this
  mode gave nonzero earlier state and appended K/V gradients, zero direct
  depth-0-logit gradient and finite nonzero sign/scale gradients in all nine
  shared linears, with **zero optimizer steps**. Its ignored report SHA256 is
  `1d50de7bfc48b6211ebc42272183b412b4cae13f0bfb1dc61375d396a814a892`.
  The [detailed diagnostic report](../../experiments/recurrent-binary-native-attention-student.md)
  records hardware, source hashes, the surrogate derivative and limits.
  This mode has not been selected as training arithmetic or validated on
  CUDA/SM75. Exact whole-drafter parity, the user-owned numeric gate and
  all-body training budget remain open; no final prompt or Q4_0 evaluation
  ran.
- The final diagnostic code and reports were checked from pushed `main`;
  all focused CPU gates passed. The integrated temporary attention worktree
  and its branch were retired after its content reached `main`. All raw
  capture and diagnostic reports remain outside Git under the main
  checkout's `results/` directory. No local server, GPU job or remote tmux
  session is active for this work.

**Next gate:** explain the remaining first-depth state drift after exact
prose attention—beginning at attention output projection/residual and FFN
math—and quantify which reasoning cache K/V F16 differences matter. If
the user chooses a bounded numeric/trajectory policy instead of exact
backend parity, freeze it before any optimizer run. Preserve the native
capture and keep the default training mode unchanged until that decision.

## Twenty-fifth goal turn: same-input binary FFN boundary

- Parent commit `3e2b73f` adds a CPU-only native graph FFN diagnostic with
  six synthetic tests, Ruff and formatting checks. It feeds the exact
  captured first-seed `post_attn_norm-0` F32 row to candidate D's three
  binary gate/up/down projections under both ordered sequential F32 and
  grouped F32 matmul arithmetic. The native graph/head join is unique at
  execution 2, column 0; no student context, attention or optimizer is run.
- On Apple M3 Max, native-order arithmetic matched all **2,560/2,560**
  prose `ffn_out-0` values bitwise. Grouped matmul differed by at most
  `1.4305e-6`. On reasoning, both arithmetic modes differed from native
  by at most `6.1035e-5`, while the two modes differed from each other by
  only `1.9073e-6`. Thus grouped reduction explains tiny prose FFN drift
  but not the reasoning residual. Native gate/up/SiLU intermediates were
  not captured, so its exact suboperation remains unidentified. The
  [FFN report](../../experiments/recurrent-binary-ffn-same-input.md)
  preserves hardware, source hashes and limits; ignored prose/reasoning
  report SHA256 values are
  `25a84157686595caa79d89d721a5f1b2fab41627e066f393001b7eeae9091c23`
  and `afe5faf6ca28dc55d3acbd267eaf0402bdd58ce0a1a5be9dfc8e921d0c39a887`.
  The main checkout reran both checks and preserved the ignored reports
  before the integrated temporary worktree and branch were retired. No
  local model server, GPU, training, final-set or Q4_0 run occurred.

**Next gate:** if exact CPU drafter parity is pursued, capture only the
native reasoning first-seed gate/up/SiLU/fused FFN intermediates and replay
them from the identical input to locate the F16 threshold or nonlinear
arithmetic gap. Separately, keep the earlier reasoning cache K/V F16
differences visible in the attention/state audit. The user still owns
the numeric/trajectory policy and all-body optimizer budget; neither
is inferred from this one-row FFN result.

## Twenty-sixth goal turn: FFN stage boundary and task rotation

**Objective:** prepare and CPU-validate a jointly trainable nine-linear
W1A16 EAGLE body and head with pinned FP16 target/verifier and D vocabulary
map. Q4_0 EAGLE remains the primary later acceptance, latency and total
throughput baseline. The user authorized the free RTX 5080 for needed work,
but has not chosen exact parity versus a numerical/trajectory tolerance or
approved an all-body optimizer budget, final-set use or Q4_0 gate.

- The fork's `21f617d4ef3f5dc383d3ab8dc619daaa87db7ff8` commit added
  four opt-in native FFN graph taps and was pushed before the parent gitlink.
  Parent `2abfe72` updated the gitlink and capture-runner revision; parent
  `5d336a8` added the sealed stage comparator and eight synthetic checks;
  parent `1256a8d` corrected its strict native four-dimensional graph shape
  check and recorded the outcome. All three parent commits are pushed to
  `main`.
- CPU-only fork build and `llama-server --help` passed. The eight stage
  comparator tests and Ruff passed after the geometry fix. The recapture
  completed eight tokens on Apple M3 Max CPU and its server exited with code
  zero. No local model server is running. The new manifest SHA256 is
  `97f9af6e03914edde054e123315d77ad85b518443b7c4be445fbd9f73f0224a7`;
  the archived baseline is `99b9003698bcdf5505421a1667dac4d21a8545471d91330c04181c9f2a931d40`.
  The ignored machine report SHA256 is
  `3136384dbebb39c1ca79361dbb1c0c1eec810e9e6acdbc7245c36c749b0774cd`.
- The recapture's prompt and request bytes, eight output IDs, first-seed
  head state, normalized FFN input and FFN output are byte-identical to the
  archived CPU reasoning control. On the joined execution-2/column-0 input,
  ordered binary gate and up are each 9,728/9,728 bitwise exact. The first
  mismatch is SiLU: 6,557/9,728 exact, maximum `4.76837158203125e-7`.
  The product is 6,772/9,728 exact, maximum `9.5367431640625e-7`; down is
  5/2,560 exact, maximum `6.103515625e-5`. Grouped matmul diverges at gate
  (9,101/9,728 exact). The
  [stage report](../../experiments/recurrent-binary-ffn-stage-parity.md)
  records complete metrics and limits. This is one Apple CPU column, not
  CUDA/SM75 or a trained quality result.
- All delegated agents are finished; their feature work has been integrated
  or superseded. No remote experiment or tmux GPU job is active; the RTX
  5080 is free. No training, final prompts or Q4_0 run occurred this turn.
  The local main checkout contains the integrated work. Both completed
  managed temporary worktrees are archived. The published fork commit
  referenced by the parent gitlink remains available.

**Exact next actions:** (1) inspect the pinned
ggml CPU SiLU implementation and run a bounded elementwise replay from the
captured raw gate and SiLU vectors, reporting exact count and max/RMS per
arithmetic choice; (2) update this file and `docs/STATUS.md` with evidence,
commit and push; (3) then consider the next drafter state/cache difference
without consuming training or final-set gates. The user still owns the
numeric/trajectory policy and all-body training budget.

## Twenty-seventh goal turn: CPU SiLU arithmetic replay

- Commit `2db7625` adds a sealed one-row SiLU arithmetic probe and
  [report](../../experiments/recurrent-binary-silu-arithmetic.md). It checks
  the capture ledger, prior stage-report identity, native revision and
  server/CMake build hashes, then invokes `ggml_vec_silu_f32` from that
  same pinned Apple M3 Max CPU build. The ggml vector call matches the
  native reasoning first-seed SiLU output **9,728/9,728 bitwise**. Torch
  SiLU matches 6,557; the ggml scalar tail 6,519; Torch sigmoid/multiply
  5,358; NumPy F32 exp/divide 6,519. All non-vector choices have maximum
  absolute error `4.76837158203125e-7`; the report gives RMS values.
- The pinned AArch64 ggml CPU path uses NEON vector `exp(-x)` approximation
  and F32 division; its scalar tail uses `expf`. The exact built-vector
  replay locates this one-row stage difference in activation arithmetic,
  after the previously exact ordered gate/up projections. It does not
  establish full FFN, whole-drafter or CUDA/SM75 parity. The ignored
  machine report is
  `results/recurrent-ffn-stage-capture-20260928/silu-arithmetic.json`,
  SHA256 `16e5f25b06d40ba039882dae801b3b9b7018b11483fbee59d6fb0d0cddfb479b`.
  Pinned CPU library SHA256 is
  `0d499359a40900596b172556bebd1aad1fa69aaf1cb3e46bf962a98ccf2a0600`.
  The probe did not alter ggml, the default student or the frozen capture.
- Ruff lint/format passed and the bounded real-input replay exited zero.
  No local server, GPU job, optimizer, final prompt or Q4_0 evaluation ran.
  The RTX 5080 remains free. Exact parity versus a numeric/trajectory
  tolerance and any all-body optimizer budget remain user-owned.

**Next gate:** feed the exact native SiLU vector into the captured up product
and ordered binary down projection to determine whether their remaining
one-row difference disappears; then resume the draft cache/state audit.
Keep results CPU-scoped and do not infer a training gate from this row.

## Twenty-eighth goal turn: native SiLU closes one-column FFN

- Commit `2960121` adds a sealed [native-SiLU downstream
  replay](../../experiments/recurrent-binary-ffn-native-silu.md). It reads
  the same reasoning execution-2/column-0 native up and SiLU vectors,
  checks the candidate-D and stage hashes, multiplies them in F32, and
  runs the existing ordered A16/W1 down projection. The product matches
  **9,728/9,728** and down matches **2,560/2,560** native F32 values
  bitwise. The Torch-SiLU control reproduces the prior 6,772/9,728
  product and 5/2,560 down counts, so the intervention isolates the
  change to SiLU input. The ignored machine report SHA256 is
  `3c19a62baff2b2afb744462def71709db998e407b6f1287c0b909ac7e2b9f7ef`.
- Ruff lint/format and the real-input replay passed. Together with the
  previously exact gate/up and pinned ggml SiLU vector replay, this
  accounts for the first-seed FFN arithmetic end to end on **one Apple
  M3 Max CPU column**. It does not certify another column, recurrent
  trajectory, CUDA/SM75 execution or training tolerance. No local model
  server, GPU job, optimizer, final prompt or Q4_0 evaluation ran. The
  5080 remains free; all-body optimizer budget and exact versus numeric
  policy are still user-owned.

**Next gate:** quantify the earlier reasoning stored-cache discrepancy
(70 F16 key and 66 F16 value differences across 47 visible positions) at
its first differing projection/write boundary. The native-attention
operand ablation already found a 0.002172 maximum attention difference
with student K/V and exact agreement with native K/V. Continue with the
sealed CPU reasoning capture and preserve the no-optimizer boundary.

## Twenty-ninth goal turn: reasoning cache projection boundary

- Successor native Goal task `01a0e7b2-e75d-7c00-8ad8-3101a73d8409`
  continued from pushed `c9b9c28`. Commit `d98b5aa` adds the bounded
  [cache-boundary probe and report](../../experiments/recurrent-binary-reasoning-cache-boundary.md).
  It verifies the frozen CPU reasoning capture and original grouped
  student cache hashes, rebuilds all 46 context K/V rows bitwise, and
  joins the first-round 47-slot native stored-cache prefix.
- The grouped student differs from native storage at 70/48,128 F16 keys
  and 66/48,128 F16 values; position 46 (first seed) matches. Grouped FC
  arithmetic produces only 14 differing F16-cast fused-input elements
  across 13 of the 46 context rows. At positions 0, 26 and 45, replacing
  just their two/one/one differing fused-input F16 coordinates and using
  ordered K/V projections makes both raw K and V 1,024/1,024 F32 exact.
  Ordered FC plus the existing norm reproduces native `g_norm`, fused
  input, raw K and raw V **bitwise at all 46 context positions**.
- With ordered FC/K/V, 47,104/47,104 context F16 values and 47,101/47,104
  context F16 keys match actual native stored bytes. The three remaining
  key bits are at positions 15, 20 and 40, after RoPE. The standalone
  RoPE replay reproduces the archived grouped-student key cache
  47,104/47,104, so the residual is native-versus-adapter key rotation
  arithmetic, not a cache-index join artifact. The ignored machine report
  SHA256 is `28aa44ed15f1798271f338338ea3cdb4821ea4f816aded38bc499a59955ca7f2`.
- Ruff lint/format and the sealed real-input probe passed on Apple M3 Max
  CPU. No local model server, GPU job, optimizer, final prompt or Q4_0
  evaluation ran. The RTX 5080 is free. Exact versus numeric/trajectory
  parity and the all-body optimizer budget remain user-owned.

**Next gate:** isolate the three F16 key thresholds in pinned ggml CPU
RoPE arithmetic, then recompute first-depth attention/state with corrected
cache operands. This one-prefix CPU result is not CUDA/SM75 evidence or a
training tolerance.

## Thirtieth goal turn: exact CPU RoPE and first-seed intervention

- Commit `52eb4f9` adds a bounded [ggml CPU key RoPE
  oracle](../../experiments/recurrent-binary-reasoning-rope-oracle.md).
  The native graph operator, on the identical 46 context raw K vectors,
  matches all **47,104/47,104** captured F32 rotated keys and actual
  stored F16 keys bitwise. The prior ordered FC/K/V replay matched those
  raw K vectors and all context V writes; the seed K/V write already
  matched. Together these boundaries account for the full first-round
  47-position draft cache (48,128/48,128 F16 keys and values). The
  ignored RoPE report SHA256 is
  `cf99b7bf89847a692800e8dc1c6870bd7e862476abeaa57fa3d9c1d5334a9f88`.
- Commit `e6f5c85` adds a [first-seed corrected-cache
  intervention](../../experiments/recurrent-binary-reasoning-seed-intervention.md).
  The ordered candidate-D CPU seed with the actual native K/V cache and
  pinned ggml attention helper matches attention output, FFN input and
  post-attention norm bitwise. Its ordinary Torch SiLU leaves the FFN
  output 5/2,560 exact (max `6.1035e-5`) and normalized head state
  5/2,560 exact (max `5.9009e-6`). Calling the pinned ggml vector SiLU
  on the same exact gate makes all gate/up/SiLU/product stages
  **9,728/9,728**, down/pre-norm/head state **2,560/2,560**, and eight
  captured draft-logit probes plus argmax/label logits bitwise exact.
  The mapped argmax target ID is 1477 and verifier-label rank is 4 in
  both paths. The ignored seed report SHA256 is
  `54c095813d5ea704f5b152b59b25dfb2c2e90a7d790fd0f68f11a2cc4909d9ed`.
- Both real-input CPU probes passed, as did Ruff lint/format. The RoPE
  helper links the same hashed ggml CPU library as the earlier attention
  ablation. No default student forward, training derivative, model weight
  or frozen capture was changed. No local server, GPU job, optimizer,
  target forward, final prompt or Q4_0 evaluation ran. The RTX 5080 is
  free. This is an **intervention on one Apple M3 Max CPU first seed**,
  not native whole-drafter/CUDA/SM75 parity or full-vocabulary logit proof.
  The user still owns exact versus numeric/trajectory acceptance and any
  all-body optimizer budget.

**Next gate:** test a multi-depth CPU proposal trajectory with ordered
binary projections and pinned native attention, RoPE and SiLU, including
cache writes and mapped draft logits. Separate the diagnostic native
forward from its F32 surrogate backward. Do not train or open final/Q4_0
gates before the user-owned numeric policy and optimizer budget.

## Thirty-first goal turn: five-depth CPU forward parity

- Commit `2afb0f9` adds a [sealed five-depth reasoning
  replay](../../experiments/recurrent-binary-reasoning-multidepth-cpu.md).
  It starts from the audited native 46-position context K/V cache, feeds
  the captured raw seed feature and native proposal tokens, and returns
  each corrected pre-norm state to the next depth. Ordered candidate-D
  binary projections, pinned ggml attention, query RoPE and vector SiLU
  then match all five depths: 5,120/5,120 new F16 keys and values each,
  20,480/20,480 F32 query-RoPE and attention values each,
  12,800/12,800 F32 FFN outputs and normalized head states each, and
  40/40 captured head-logit probes. All five mapped argmax IDs,
  argmax/label logits and verifier-label ranks match native.
- With the adapter's Python query RoPE, depths 0–3 remain exact at
  attention and normalized state. At depth 4, attention has 3,968/4,096
  exact values (one 128-wide head differs) and the state has 11/2,560
  exact values. One F16 threshold at query head 1/channel 52 is
  sufficient to restore that attention head and downstream checks.
  The pinned ggml RoPE helper reproduces all 20,480 native F32 query
  values and restores the complete five-depth diagnostic without using
  a captured-value patch. The ignored report SHA256 is
  `3259d01947dc5196b0f5eb46dd6cf02d4154e00d88165f90183cdde908bda455`.
- The real-input run and Ruff lint/format passed on Apple M3 Max CPU.
  The diagnostic used the pinned native attention forward with its
  declared F32 surrogate backward, but did not execute backward or an
  optimizer. The default student was not changed. This one
  native-token-forced first-round chain starts from captured context
  cache bytes, although the prior cache report verified their ordered
  reconstruction. Full mapped-vocabulary logits, another prefix,
  free-running recurrent trajectories, CUDA/SM75 behavior, training
  tolerance and Q4_0 evaluation remain open. No GPU, local server,
  final prompt or training run occurred; the 5080 is free.

**Next gate:** replay a second frozen training prefix or post-acceptance
round with the same CPU operator oracles and reconstruct its context cache
from ordered candidate-D arithmetic. Then assess whether the CPU forward
can be integrated without captured-cache injection before proposing any
user-owned numeric policy or optimizer budget. Keep final prompts and
Q4_0 acceptance/speed evaluation gated.

## Thirty-second goal turn: independent prose-prefix parity

- Commit `ca157ef` adds an [independent prose-prefix CPU
  replay](../../experiments/recurrent-binary-prose-multidepth-cpu.md).
  It seals all 23 capture files against prose manifest SHA256
  `4ad342145a1b77deb5f48cac4e00671e26b691850f398be56687ef05830b7169`,
  then constructs the 31-position context cache from frozen raw target
  features, borrowed embeddings, ordered candidate-D projections,
  native-style norms and pinned ggml key RoPE. Native graph embeddings,
  norms, fused inputs, raw K/V and rotated K are bitwise exact; all
  **31,744/31,744** stored F16 context keys and values each match.
- With no native-cache injection, the computed prose seed and four later
  corrected pre-norm states feed five draft depths. All **5,120/5,120**
  new F16 key and value writes each, **20,480/20,480** F32 query RoPE and
  attention values each, **12,800/12,800** F32 FFN outputs and
  normalized head states each, and **40/40** captured head-logit probes
  match native. All five mapped argmax IDs, argmax/label logits and
  verifier-label ranks match. The ignored machine report SHA256 is
  `e1896769591a8c85a9fb8ed418f12ee2624bc801409eced39e84e0e3d9bdae60`.
- The real-input run and Ruff lint/format passed on Apple M3 Max CPU.
  It joins the prior reasoning first-round replay as a second frozen
  training prefix, but both are native-token-following diagnostics.
  No backward, optimizer, local server, GPU job, target forward, final
  prompt or Q4_0 evaluation ran; the RTX 5080 is free. The F32
  surrogate derivative, exact versus numeric/trajectory gate, and
  all-body budget remain user-owned. Full mapped-vocabulary logits,
  post-acceptance cache scheduling, broad training trajectories and
  CUDA/SM75 parity remain open.

**Next gate:** exercise a post-acceptance round on a frozen training
prompt. Rebuild its shifted accepted-prefix context, including cache
rewrites, then run the pinned CPU operator oracles through its native
proposal chain. Keep the diagnostic forward separate from any training
derivative or budget decision.

## Thirty-third goal turn: post-acceptance CPU parity

- Commit `6230b5c` adds a [sealed reasoning round-2 CPU
  replay](../../experiments/recurrent-binary-postacceptance-cpu.md).
  The prior round accepted one draft. Round 2's 50-token accepted prefix
  requires 49 draft context rows. Its native cache has speculative
  rewrites: position 47 was written three times, with token 525 from
  execution 8 retained; position 48 was written four times, with
  accepted token 2661 from five-column catch-up execution 13 retained.
  The probe joins each accepted-prefix position to its latest physical
  write before the round-2 seed and reconstructs K/V independently from
  frozen target features and ordered candidate-D arithmetic.
- All context embeddings and normalized inputs match native
  125,440/125,440 F32 each, fused input 250,880/250,880, raw K/V and
  ggml-rotated K 50,176/50,176 each, and actual stored F16 context K/V
  50,176/50,176 each. The three draft depths have 3,072/3,072 new F16
  key and value writes each, 12,288/12,288 query-RoPE and attention
  F32 values each, 7,680/7,680 FFN outputs and normalized head states
  each, and 24/24 captured head-logit probes. Their mapped argmax IDs
  (12, 3070, 362), argmax/label logits and verifier-label ranks match.
  The ignored machine report SHA256 is
  `31acfdb2c62fb314bced1d4ed188db1855f38fe53ebdd4d02c5aa2a7bce45a4d`.
- The real-input run and Ruff lint/format passed on Apple M3 Max CPU.
  No captured-cache injection, GPU, local server, target forward,
  backward, optimizer, final prompt or Q4_0 evaluation ran. The RTX
  5080 is free. This verifies one post-acceptance training round under
  the diagnostic native forward, not general cache scheduling,
  free-running trajectories, full mapped logits or CUDA/SM75 parity.
  The F32 surrogate derivative, exact/numeric policy and all-body
  optimizer budget remain user-owned.

**Next gate:** integrate the validated native CPU RoPE, attention and
SiLU arithmetic into one opt-in diagnostic student forward, then test
its unmodified five-depth calls on the sealed prose and reasoning
training rounds, including accepted-prefix reconstruction. Keep the
default differentiable path unchanged until the user chooses a numeric
policy and training budget.

## Thirty-fourth goal turn: integrated exact CPU diagnostic forward

- Commit `7cf1ef7` adds an opt-in `native_cpu_diagnostic` path to
  `NativeStepAdapter` with pinned ggml CPU RoPE, Flash Attention and
  vector SiLU. It requires Apple arm64, exact helper/library hashes,
  the frozen 32-Q/8-KV × 128-head EAGLE geometry, ordered candidate-D
  binary projections and `torch.no_grad()`. The path rejects
  gradient-enabled calls; it supplies no chosen training derivative.
  The existing F32 default and attention-only/F32-surrogate modes remain
  available. The [integrated report](../../experiments/recurrent-binary-integrated-cpu-diagnostic.md)
  gives implementation scope and limits.
- The adapter's ordinary `rebuild_prefix_cache` and `decode_step` calls
  passed three sealed Apple M3 Max CPU training cases, with no
  captured-cache injection or per-stage substitutions: prose first
  round (31,744/31,744 F16 context K/V each, five depths), reasoning
  first round (47,104/47,104 each, five depths), and reasoning
  post-acceptance round (50,176/50,176 each, three depths). All 13
  depths match every checked native graph tap and K/V write, all
  **33,280/33,280** normalized head-state F32 values, **104/104**
  captured head-logit probes, and captured argmax/label logits,
  mapped argmax IDs and verifier-label ranks. The ignored report
  SHA256 values are `9b6744fde999fcd19f901bacd4dd9dee3005c6d66fe7b2118374d7ec230973a0`
  (prose), `8b358c1398b62bff05f5f9723510b1b0aa5d1191f447c6f6e852cc8d7636e1e8`
  (reasoning first) and `a5595125ba24e95b13b6b0a565f71b66bc42a31293a315ee420caadc1ddd398a`
  (reasoning post-acceptance).
- The real-input gates, 62 native CPU unit tests, three real-step input
  checks and Ruff lint/format passed. No target forward, backward,
  optimizer, GPU job, local server, final prompt or Q4_0 evaluation
  ran; the RTX 5080 is free. The diagnostics cover two frozen prompts
  and one accepted-draft catch-up, not all 96 training trajectories,
  full mapped-vocabulary logits, CUDA/SM75 execution, target-feature
  alignment or a training tolerance. The user still owns exact versus
  numerical/trajectory acceptance and any all-body optimizer budget.

**Next gate:** broaden the opt-in CPU forward across another sealed
training case or later cache rewrite, and examine whether target-feature
alignment on RTX 5080 needs an exact backend path or a predeclared
numeric/trajectory gate. Keep this no-grad CPU diagnostic separate from
any QAT recipe; Q4_0 remains the primary future acceptance, latency and
throughput comparison.

## Thirty-fifth goal turn: rejected-draft rewind parity

- Commit `1bb2a10` extends the opt-in integrated CPU runner to the
  sealed reasoning middle round and records the [rewind
  result](../../experiments/recurrent-binary-reasoning-middle-cpu.md).
  Its first-round drafts were rejected; native had written proposal
  token 1477 at position 47 before the next seed overwrote that slot
  with verifier token 525. The student's ordinary context rebuild
  matches **48,128/48,128** F16 keys and values each, and its five
  subsequent steps match **5,120/5,120** new F16 keys and values each.
  All checked native graph taps, **12,800/12,800** normalized head-state
  F32 values, **40/40** captured logit probes, mapped argmax IDs,
  argmax/label logits and verifier-label ranks are bitwise exact. The
  ignored report SHA256 is
  `1bda1318f85169f586490e62dd9220692824d7f4242c24aa224f02a7aaf26128`.
- The real-input gate and Ruff checks passed on Apple M3 Max CPU. Across
  the four integrated cases, 18 native-token-following draft depths
  now include first-round, rejected-draft rewind and accepted-draft
  catch-up behavior on two frozen training prompts. No gradient,
  optimizer, target forward, GPU job, local server, final prompt or
  Q4_0 evaluation ran; the RTX 5080 is free. The diagnostic remains
  forward-only and does not establish full-vocabulary, all-96,
  free-running or CUDA/SM75 parity. The exact/numeric policy and all-body
  budget remain user-owned.

**Next gate:** use the authorized RTX 5080 for a bounded, no-optimizer
target-feature arithmetic diagnostic on frozen training inputs, using
the existing native layer ladder and independent forward as controls.
Choose the operator boundary and measurement before starting a remote
run; keep the final split and Q4_0 serving gate sealed.

## Thirty-sixth goal turn: safe target block-zero taps on 5080

- The bounded target block-0 helper, runner and auditors were pushed in
  commits `7663ec0`, `c14d7f9`, `9824c80`, `395ead3`, `02b06ac`,
  `7c7c085`, `65ae6d5`, `47f27ae`, `09f1c4d`, `835f183`, `fe8dfc3` and
  `80d6170`. Local C++ syntax and Ruff checks passed. No target ggml
  submodule change was needed; the published fork gitlink remains
  `21f617d4ef3f5dc383d3ab8dc619daaa87db7ff8`.
- The host registry supplied the RTX 5080 address; SSH used only the tmux
  MCP. A new detached remote checkout
  `checkouts/target-block0-operator-20260928` preserved the older native
  ladder checkout and the frozen 29-token **code/data-validation training
  prompt**. The first attempts were preserved: run `a` lacked CMake on
  PATH, `b` lacked the GPU host's pinned CUDA/glibc compatibility include,
  and `c` built successfully but rejected a mistakenly named prompt
  before model execution. Later runs reused `c`'s 54 MiB compiled CUDA
  libraries by hash in separate supervised directories.
- An output-only CUDA callback matched the sealed native layer-1 input
  **74,240/74,240 F32 bitwise**. Attention-norm, K-norm, V, and FFN-side
  callbacks each preserved that complete block output. Q-normalization
  alone and the all-tap callback produced **72,329/74,240** exact values
  (maximum `0.000244140625`), so their Q data are excluded from
  server-path attribution. Every run's process group stopped and the
  5080 returned to 0% utilization and 1,372 MiB whole-device baseline
  use. The tmux session was closed after the audit.
- The [safe-tap report](../../experiments/recurrent-target-block0-safe-taps-5080.md)
  compares only output-preserving native CUDA/F32 taps against a pinned
  HF CUDA/F16 eager forward. On outlier position 3, relative row L2 is
  0.0325% after attention RMS norm, 0.0922/0.0994% at K/V, 0.2547%
  after attention residual, and 0.2721% at full block output. Supplying
  native attention-norm values cast to F16 changes K/V errors only to
  0.0916/0.0934%. This rules out ordinary norm-input drift as the sole
  explanation under that F16 intervention, but it does not identify a
  particular Q or Flash Attention instruction. The valid HF report is
  `checkouts/target-block0-operator-20260928/runs/target-block0-safe-hf-c-20260928/comparison.json`
  on the registered host, SHA256
  `e2012372501f4f9e0595465e4e69e7656330917bdf8c0547c22bd521bb1b9b43`.
  An earlier HF report with an EAGLE-style K row permutation was corrected
  and excluded; Qwen3 target uses NeoX half-split rows.
- No optimizer, training, development/final prompt or Q4_0 serving
  evaluation ran. This is one SM120/RTX 5080 F16 target-operator
  diagnostic, not SM75 performance or a global target-feature gate. The
  full96 native capture remains training-ineligible; the user still owns
  exact versus predeclared numeric/trajectory acceptance and the
  all-body budget.

**Next gate:** isolate the remaining same-input K/V projection arithmetic
from F16 input-cast effects using an operator replay with the pinned
ggml CUDA backend, while requiring an output-preserving callback or a
separate equivalence check. Q-normalization cannot be attributed from
the intrusive callback. Keep CUDA target-feature and all-body training
decisions separate from the exact Apple CPU drafter diagnostic.

## Thirty-seventh goal turn: same-input target V CUDA arithmetic

- Commit `b9320fa` adds a bounded standalone [ggml CUDA V
  replay](../../experiments/recurrent-target-v-cuda-projection.md) and
  source/HF operand audit. It takes the 29 F32 block-0 native
  attention-norm rows from an output-preserving capture and the pinned
  F16 `blk.0.attn_v.weight` GGUF tensor, whose bytes match the HF source
  shard. It runs the same 29-token V `ggml_mul_mat` geometry on the RTX
  5080, then an explicit F16-cast-input variant and a Torch CUDA/F16
  linear control. The run uses no target full forward or optimizer.
- The F32-input ggml output matches all **29,696/29,696** actual native
  server V values bitwise, satisfying its fidelity gate. Explicit F16
  input cast changes **0/29,696** values. Torch on the same F16 input
  and F16 weight matches only **5,351/29,696** native values (maximum
  `0.00146484375`, position-3 relative row L2 **0.0934%**), reproducing
  the earlier full-model same-input intervention. This isolates the V
  residual under this geometry to ggml-versus-Torch projection backend
  arithmetic/dispatch; it does not identify an instruction or settle
  K/Q or later target blocks. The ignored machine report at
  `checkouts/target-block0-operator-20260928/runs/target-v-cuda-20260928/comparison.json`
  on the registered host has SHA256
  `39cd41fe114e0c42b8cd31598cbd54efef1abf97057428c0e55d15d45ede9747`.
- The supervised WSL run exited zero and released its process group;
  final RTX 5080 use was 0% GPU and 1,372 MiB whole-device baseline.
  The tmux session was closed. Local C++ syntax and Ruff checks passed.
  No training, target full-model forward, final prompt or Q4_0 serving
  evaluation ran. This is one SM120 operator case, not SM75 performance,
  general target-feature parity or a user-approved training tolerance.
  The full96 capture remains training-ineligible and the all-body
  optimizer budget remains user-owned.

**Next gate:** replay the pinned K projection and per-head RMS norm on
the same output-preserving native input and weights, with an exact native
K tap as the fidelity gate. Keep the intrusive Q tap excluded. Only
after these local operators are understood should the team revisit a
predeclared target-feature/trajectory tolerance or exact training backend.

## Thirty-eighth goal turn: same-input target K CUDA arithmetic

- Commits `1d02161`, `86d1b67` and `c6b63b0` add and refine the
  bounded K operator replay. The [report](../../experiments/recurrent-target-k-cuda-projection.md)
  covers block-0 `ggml_mul_mat`, eight-head reshape and weighted RMS norm
  on the frozen 29-token code/data-validation training prompt. The F16
  K matrix and F32 head-norm weight match the pinned GGUF/HF sources;
  the input is an output-preserving native attention-norm capture.
  Local C++ syntax and Ruff lint/format checks passed. The standalone
  ggml CUDA F32-input graph matches **29,696/29,696** output-preserving
  native `Kcur_normed-0` F32 values bitwise, satisfying the fidelity
  gate. Explicitly casting its input to F16 changes no raw or normed K
  values.
- Torch CUDA/F16 raw K projection matches **6,530/29,696** ggml raw
  values; after the exact Qwen3 HF norm it matches only **2/29,696**
  native F32 normed values, with position-3 relative row L2 **0.0916%**.
  This reproduces the earlier same-input full-model HF intervention
  exactly. Giving the Torch norm the identical ggml raw F32 K yields
  **24,456/29,696** exact values, maximum absolute gap
  `4.57763671875e-5` and zero position-3 gap. Casting that same raw K
  to F16 before the Torch norm yields 0.0338% position-3 error. The
  tested K gap therefore includes projection backend arithmetic and
  F16 intermediate precision; same-raw F32 norm arithmetic is much
  smaller. These effects are not presumed additive.
- The corrected supervised run
  `checkouts/target-block0-operator-20260928/runs/target-k-cuda-c-20260928`
  on the registered RTX 5080/SM120 host exited zero; ignored report
  SHA256 `58c67405a21576200ba5ce0a1db4675e39c674ee43b27f2e9e448dcedcf8400f`.
  The first generic-Torch-norm run and the second corrected-HF-norm run
  are preserved as ignored diagnostics. The final process group stopped
  and the GPU returned to 0% utilization and 1,372 MiB whole-device
  baseline. No target full-model forward, optimizer, final prompt or
  Q4_0 serving evaluation ran. This one SM120 operator case does not
  establish SM75 performance or a global target-feature tolerance.
  Training remains gated by the user-owned exact/numeric policy and
  all-body budget; the 96-prompt capture is still training-ineligible.

**Next gate:** investigate an output-preserving native Q/attention
boundary, since the existing Q-normalization callback changes the
block output. Keep the sealed training prefix and same-input checks;
do not treat an intrusive tap as native server-path evidence. Revisit
the user-owned numeric/trajectory policy only after the operator
evidence is adequate.

## Thirty-ninth goal turn: output-preserving Q boundary and CUDA replay

- Commits `af7fa14`, `453f558` and `dae1137` add a deferred native Q
  callback mode, its safe-capture audit and a same-input ggml CUDA Q
  replay. The [Q report](../../experiments/recurrent-target-q-deferred-cuda.md)
  records the method, hashes and limits. The mode observes the
  post-RoPE `Qcur-0` pointer without requesting a scheduler stop at Q,
  then copies it when the previously safe K-norm callback stops the
  graph. The supervised 29-token native RTX 5080/SM120 run captured
  one finite 118,784-value F32 Q tensor. Its K payload matches the old
  safe K capture bytewise, and its **74,240/74,240** block-output
  values match the sealed ladder and output-only capture bitwise.
- The safe deferred Q differs slightly from the old intrusive
  post-RoPE Q capture: **102,401/118,784** F32 values exact, maximum
  absolute difference `9.5367431640625e-7`, with only three different
  F16-cast values. The old direct Q callback changed 1,911 block-output
  values; the small Q discrepancy alone does not prove that later
  change's cause. A standalone ggml CUDA graph on identical native
  F32 norm input and pinned F16 Q/F32 head-norm weights applies Q
  projection, 32-head RMS norm and NeoX RoPE. It matches the safe
  post-RoPE Q **118,784/118,784 F32 values bitwise**. Explicit F16
  input cast changes no raw, normed or rotated Q value.
- Ignored comparison SHA256 values on the registered host are
  `ae5f934444e64b4b3a589b5be2bdcf67cfa55611e64c425d3d41411d437a969f`
  (deferred native Q),
  `d1da2dd8ca27477e7e25992be60c789296572a9fe8adee864a303b11c62c7f1d`
  (safe K/output and intrusive-Q audit), and
  `febaced1bd0e7ad5d6d7471693e3408788ae98cd276e017dcbab07e723e8979b`
  (standalone Q replay). All supervisors exited zero, process groups
  stopped and the 5080 returned to 0% utilization and 1,372 MiB
  whole-device baseline. Local C++ syntax and Ruff checks passed.
  No optimizer, final prompt or Q4_0 serving evaluation ran. This
  single SM120 target case does not prove general target-feature or
  SM75 parity. The full96 capture remains training-ineligible; the
  exact/numeric training policy and all-body budget remain user-owned.

**Next gate:** compare a pinned Torch CUDA/F16 same-input Q path
against the newly safe post-RoPE Q boundary, then attribute block-0
attention/residual error using output-preserving taps. Keep all
probes on frozen training prefixes and separate from training and
Q4_0 gates.

## Fortieth goal turn: same-input Torch Q stages

- Commits `ba17621`, `927ff19` and `55c6125` add the bounded
  [Torch Q control](../../experiments/recurrent-target-q-torch-control.md)
  and separate an exact ggml Q graph from a retained-intermediate
  diagnostic graph. The Torch path uses the installed Qwen3 RMS norm,
  rotary embedding and RoPE application modules on the same native
  F32 attention-norm rows explicitly cast to F16, with exact source
  GGUF/HF Q weights. The final supervised RTX 5080/SM120 run retains
  the **118,784/118,784** exact ggml post-RoPE Q fidelity gate.
- Torch F16 raw Q projection matches **14,513/118,784** exact ggml
  raw values, with position-3 relative row L2 **0.2880%**. Torch
  normalized Q matches **3/118,784** F32 values of the diagnostic
  retained ggml norm, position-3 error **0.2337%**. Torch post-RoPE Q
  matches **2/118,784** safe native F32 values, position-3 error
  **0.2347%** and maximum absolute gap `0.08107709884643555`. The
  Q discrepancy is present before norm and persists; the stage errors
  are not assumed additive.
- A ggml graph retaining its raw/normed intermediates leaves raw Q
  bitwise identical to the exact graph, but changes 16,383 final
  post-RoPE F32 values by at most `9.5367431640625e-7`. That graph
  supplies only a diagnostic norm-stage comparison. The original
  unretained norm readback was invalid because its buffer was reused,
  and the retained-only run failed the strict server-Q fidelity gate;
  both are preserved as ignored diagnostics, not results. The final
  ignored report at
  `checkouts/target-block0-operator-20260928/runs/target-q-cuda-d-20260928/comparison.json`
  has SHA256 `849b6c26b40261062f2664783c2c0b7672dca681510cebfd7687364f8d0e5474`.
  Its supervisor exited zero and the 5080 returned to 0% utilization
  and 1,372 MiB whole-device baseline. Local C++ syntax and Ruff
  lint/format passed. No optimizer, final prompt or Q4_0 serving
  evaluation ran; this single SM120 prefix does not set a global
  target-feature tolerance or validate SM75 performance. The full96
  capture remains training-ineligible; exact/numeric policy and the
  all-body budget remain user-owned.

**Next gate:** attribute the block-0 attention and residual boundary
using same-input, output-preserving native Q/K/V and FFN-input taps.
Separate source Q/K/V projection precision from attention backend
arithmetic before choosing a target-feature/trajectory policy.

## Forty-first goal turn: safe post-RoPE K boundary

- Commit `abedf46` adds a `k_rope` native callback mode that requests
  only the RoPE-operation `Kcur-0` tensor, skipping earlier nodes with
  the same name. The [report](../../experiments/recurrent-target-k-rope-safe.md)
  records one supervised RTX 5080/SM120 run on the frozen 29-token
  code/data-validation training prefix. It captured one 29,696-value
  F32 post-RoPE K tensor and preserved all **74,240/74,240** sealed
  block-output F32 values bitwise. The output payload matches the
  earlier output-only capture bytewise. The old intrusive all-tap K
  payload differs and remains excluded from native-path attribution.
- Ignored machine report
  `checkouts/target-block0-operator-20260928/runs/target-k-rope-a-20260928/comparison.json`
  has SHA256 `c3fc11f35c995c95b75302906f37a4bf52489c70ca3351fc053a6b7b1fa6a2cf`.
  The supervised process group stopped; final RTX 5080 use was 0%
  utilization and 1,372 MiB whole-device baseline. Local C++ syntax
  and Ruff checks passed. No optimizer, final prompt or Q4_0 serving
  evaluation ran. This single SM120 capture does not prove attention,
  later-block, SM75 or general target-feature parity. The full96
  capture remains training-ineligible and the user-owned numeric
  policy and all-body budget remain open.

**Next gate:** join the output-preserving post-RoPE Q/K and V tensors
with the safe FFN-input residual tap on the same sealed prefix. Replay
the attention and output projection under pinned ggml CUDA and Torch
controls, requiring exact native output before attributing backend
arithmetic.

## Forty-second goal turn: native-QKV Torch attention intervention

- Commits `724b70d` and `1d90531` add a bounded source-Qwen3 eager
  [attention intervention](../../experiments/recurrent-target-attention-intervention.md)
  on the frozen 29-token training prefix. The runner joins five
  output-preserving native captures (attention norm, post-RoPE Q/K,
  V and `ffn_inp-0`) with matching target GGUF/CUDA identities and
  audits seven source weights. A full HF CUDA/F16 forward supplies the
  exact block-0 causal mask, rotary cos/sin and attention input. A
  manual source-module replay matches its own **74,240/74,240**
  block-0 residual F16 values bitwise, satisfying the HF fidelity
  gate. The isolated Q and V same-input controls reproduce their
  earlier reports.
- Against the safe native F32 `ffn_inp-0`, position-3 relative row L2
  is **0.2547%** for the loaded HF forward, **0.2434%** with native
  attention-norm input but source Q/K/V, and **0.2048%** after
  substituting safe native post-RoPE Q/K and V cast to F16. The
  corresponding median row errors are 0.2269%, 0.2257% and 0.2038%.
  Projection substitution reduces error but leaves a material
  attention/residual difference. It cannot be assigned solely to
  Flash Attention because the Torch path casts native Q to F16 and
  retains Torch eager attention, F16 output projection and F16
  residual arithmetic.
- The loaded HF model's rotary inverse-frequency buffer is F16,
  while a fresh isolated Qwen3 rotary module's is F32. Their cosines
  match 3,264/3,712 F16 values, maximum gap `0.00439453125`.
  The corrected run accounts for this and reproduces the earlier
  isolated Q control. Its ignored report at
  `checkouts/target-block0-operator-20260928/runs/target-attention-intervention-b-20260928/comparison.json`
  has SHA256 `9b125cb4c8f73eade7bfc1127495479ac88e4ff7d1fbfae097337be30d9b46d8`.
  The first attempt is preserved as a diagnostic control mismatch.
  The final supervised RTX 5080/SM120 process exited zero and stopped;
  GPU use returned to 0% and 1,372 MiB whole-device baseline. Ruff,
  Python compilation and diff checks passed. No optimizer, final
  prompt or Q4_0 serving evaluation ran. This single prefix does not
  set a training tolerance, prove later-block parity or validate
  SM75 performance; the full96 capture remains training-ineligible.

**Next gate:** replay pinned ggml CUDA Flash Attention and output
projection on the same safe Q/K/V operands, requiring the native
attention-residual tap as a fidelity gate. Quantify separately the
effects of F32 Q versus F16 casting, F16 K/V cache storage and
output-projection backend arithmetic. The exact/numeric policy and
all-body optimizer budget remain user-owned.

## Forty-third goal turn: exact same-input Flash Attention residual

- Commits `e2c4bbe` and `27ec5ec` add a bounded [ggml CUDA Flash
  Attention replay](../../experiments/recurrent-target-flash-attention-cuda.md)
  on the frozen 29-token training prefix. It takes output-preserving
  F32 post-RoPE Q/K and V, reconstructs 256-slot F16 K/V cache and
  F16 causal mask, uses F32 Q and F32-accumulation Flash Attention,
  then the pinned F16 O projection and F32 residual. The O weight
  matches the HF source bytes. The standalone graph matches all
  **74,240/74,240** safe native `ffn_inp-0` F32 values bitwise,
  satisfying the attention-residual fidelity gate.
- An explicit F16 roundtrip of Q before the CUDA kernel changes no
  residual value on this prefix. Direct F16 Q was rejected by the
  pinned CUDA Flash Attention implementation's F32-Q assertion; that
  first supervised attempt is preserved as an ignored diagnostic.
  The corrected `target-flash-attn-b-20260928` run exited zero and
  has ignored `comparison.json` SHA256
  `ae265f07976acdd5954023a5697bb5ffe6bf7cde38101f051198dad8fef2a0a1`.
  Its process group stopped and the RTX 5080/SM120 returned to 0%
  utilization and 1,372 MiB whole-device baseline. Local C++ syntax
  and Ruff checks passed. No optimizer, final prompt or Q4_0 serving
  evaluation ran; the full96 capture remains training-ineligible.
  This one case does not validate SM75 performance, later target
  blocks or a global target-feature tolerance.

**Next gate:** obtain an output-preserving pre-O attention tensor
from native block 0, then compare Torch eager attention and Torch/
ggml O projection on identical operands. Keep the full residual replay
as an exact fidelity gate; distinguish attention-kernel differences
from O-projection and residual rounding. The exact/numeric policy and
all-body budget remain user-owned.

## Forty-fourth goal turn: safe pre-O attention tensor

- Commit `d1e16bb` adds an `attn_output` callback mode for native
  `kqv_out-0`, immediately after Flash Attention and before O
  projection. The [report](../../experiments/recurrent-target-attn-output-safe.md)
  records one supervised RTX 5080/SM120 run on the frozen 29-token
  training prefix. It captured one 118,784-value F32 pre-O tensor
  while preserving all **74,240/74,240** sealed block-output F32
  values bitwise; the block output is byte-identical to the prior
  output-only capture. This supplies a safe stage boundary between
  the already exact combined ggml attention/O replay and source
  Torch attention/O arithmetic.
- Ignored machine report
  `checkouts/target-block0-operator-20260928/runs/target-attn-output-a-20260928/comparison.json`
  has SHA256 `3150ec96dbca949ef0564deff898fb65cef2b8fd4ae3eaea339651e43b7ac35d`.
  The supervised process group stopped and the GPU returned to 0%
  utilization and 1,372 MiB whole-device baseline. Local C++ syntax
  and Ruff checks passed. No optimizer, final prompt or Q4_0 serving
  evaluation ran. This one SM120 capture does not validate SM75,
  later-block or general target-feature parity. The full96 capture
  remains training-ineligible and the numeric policy and all-body
  budget remain user-owned.

**Next gate:** replay the pinned ggml Flash Attention output alone
against the safe `kqv_out-0` tensor, then feed that same tensor through
ggml and Torch O projections plus residual. Compare the source Torch
eager attention output on safe Q/K/V with this native boundary so
attention-kernel and O-projection differences are measured separately.

## Forty-fifth goal turn: attention and O projection split

- Commits `bb0e796` and `0318f50` add a bounded [stage-split
  report](../../experiments/recurrent-target-attention-stage-split.md)
  for the frozen 29-token training prefix on RTX 5080/SM120. The
  standalone ggml CUDA full residual matches **74,240/74,240**
  native F32 values, Flash Attention alone matches **118,784/118,784**
  safe pre-O F32 values, and O plus F32 residual on captured native
  pre-O input matches **74,240/74,240**. Separately emitted ggml O
  projection plus the frozen F32 input reconstructs the same native
  residual bitwise. Explicit F16 rounding of the ggml O input changes
  no residual value. The constructed source Torch eager combined
  path reproduces the earlier native-Q/K/V intervention exactly.
- On identical native operands, Torch eager attention has 0.1058%
  position-3 relative row error at pre-O. Feeding Torch attention
  output through ggml O/F32 residual yields 0.1253% error. On the
  **native** pre-O tensor, Torch F16 O projection differs from ggml
  by 0.2071% relative row L2 (7,451/74,240 F32 exact); with F32
  residual addition the error against native `ffn_inp-0` is 0.1947%,
  and with source F16 residual it is 0.1941%. The full Torch
  attention/O/F16-residual path is 0.2048%. F16 versus F32 residual
  addition on the same Torch O output differs by 0.0212% position-3
  row L2. These are separate interventions and not additive terms.
- The final ignored report at
  `checkouts/target-block0-operator-20260928/runs/target-attention-stages-b-20260928/comparison.json`
  has SHA256 `c9c24997d2f3f6bcc1d1bf9b734cd5ffa2bbb1072f35bba795c2a499e9c2164e`.
  The first successful stage run is retained as an ignored diagnostic;
  the second added projection-only and residual-precision checks.
  Its supervisor exited zero, process group stopped and the GPU
  returned to 0% utilization and 1,372 MiB whole-device baseline.
  Local C++ syntax, Ruff and Python checks passed. No optimizer,
  final prompt or Q4_0 serving evaluation ran. This single SM120
  case does not validate SM75 or later target blocks, set a global
  feature tolerance or make the full96 capture training-eligible.

**Next gate:** extend exact target-feature attribution beyond block 0
using the frozen native layer ladder and safe callback discipline,
prioritizing the earliest material later-layer divergence and the
block-14 outlier amplification. Keep this independent of the
user-owned exact/numeric policy and all-body optimizer budget.

## Forty-sixth goal turn: output-preserving block-14 stages

- Commit `24d740c` extends the bounded native CUDA callback and
  auditor with a `block14_stages` mode. The [report](../../experiments/recurrent-target-block14-safe-stages.md)
  records six F32 tensors on the frozen 29-token code/data-validation
  training prefix: attention norm, pre-O attention output,
  post-attention residual, FFN norm, FFN output and `l_out-14`.
  The supervised RTX 5080/SM120 run captured 1,959,936 bytes and
  matched **74,240/74,240** sealed native layer-15 input F32 values
  bitwise, satisfying the block-output fidelity gate. The existing
  block-0 `all` mode retains its previous selection.
- Ignored machine report
  `checkouts/target-block0-operator-20260928/runs/target-block14-stages-a-20260928/comparison.json`
  has SHA256 `8caeabd9313f090725de7b611848216d60fee6e603f81ca89b77775ee4686827`.
  The supervisor exited zero, its process group stopped and the GPU
  returned to 0% utilization and 1,372 MiB whole-device baseline.
  Local C++ syntax, Ruff and Python checks passed. No optimizer,
  final prompt or Q4_0 serving evaluation ran. This single SM120
  capture does not set a training tolerance, validate SM75 or make
  the full96 bundle training-eligible.

**Next gate:** compare the same block-14 native stage tensors against
source HF CUDA/F16 eager stages under both accumulated HF input and
captured native layer-14 input. Reproduce the earlier 4.148% versus
0.149% position-3 output errors before attributing amplification to
attention or FFN. The numeric policy and all-body budget remain
user-owned.

## Forty-seventh goal turn: block-14 stage attribution; GPU paused

- Commit `6212603` adds the bounded [block-14 stage
  comparison](../../experiments/recurrent-target-block14-stage-intervention.md).
  A supervised RTX 5080/SM120 run on the frozen 29-token training
  prefix compared six output-preserving native F32 taps against source
  HF CUDA/F16 eager under accumulated and native-cast layer-14 inputs.
  Eleven source weights match the pinned GGUF. The prior position-3
  layer-14 input/output intervention reproduces exactly: baseline
  block output error **4.1478%**, native-input error **0.1491%**.
- At position 3, accumulated HF error is 1.2157% at the
  post-attention residual/FFN input, 2.1931% at FFN norm, 12.1527%
  at FFN branch output and 4.1478% at block output. With native block
  input cast to F16, these are 0.1015%, 0.1904%, 0.5045% and
  0.1491%. The accumulated absolute RMS error rises from `0.011395`
  at FFN input to `0.033592` at FFN branch output and `0.042182`
  at complete output; with native input these are `0.000952`,
  `0.001394` and `0.001516`. This localizes the observed
  amplification chiefly to the FFN path on that row; it does not
  prove which FFN operation or input direction is causal.
- Ignored report
  `checkouts/target-block0-operator-20260928/runs/target-block14-hf-stages-a-20260928/comparison.json`
  has SHA256 `ad8e719a77667da86b74fc269d2a40cddab1b7cb66df84f12792a1b85d7c95c6`.
  The supervisor finished with exit zero at 2026-09-28 15:10 UTC;
  process group 417 is absent. The user then said the RTX 5080 is
  unavailable. `scripts/agent_env.py pause rtx5080` now blocks new
  project runs. The later device observation was 95% utilization and
  8,011 MiB whole-device use; no project process was restarted or
  interrupted. Local Ruff and Python checks passed. No optimizer,
  final prompt or Q4_0 serving evaluation ran. The full96 bundle
  remains training-ineligible; numeric policy and all-body budget
  remain user-owned.

**Next gate when GPU access resumes:** intervene at native block-14
FFN input and inspect FFN gate/up/activation/down boundaries to
separate sensitivity to upstream state from local arithmetic. Also
broaden target-feature parity across representative training rows and
later blocks before proposing a numeric/trajectory gate. While the
5080 is paused, continue only CPU analysis and checkpoint work; do
not start GPU jobs without an explicit availability update.

## Fresh-team handoff: EAGLE W1 CPU phase

The user's audit in chat `01a0e9aa-3619-7012-9b4a-e4002a36901a` changed the
execution priority: account for actual drafter latency and avoidable work,
finish practical joint QAT, and expand the data beyond 96 prompts. The user
then requested a consolidated plan and a fresh team, explicitly **no GPU access**.
The [overarching plan](../W1_RESEARCH_PLAN.md) is the controlling work breakdown.
The forty-seventh-turn target-FFN investigation above is preserved as history
and is not the next assignment.

Starting state: parent `d111335` on clean main, predecessor checkpoint
`5110257`, native submodule `21f617d4ef3f5dc383d3ab8dc619daaa87db7ff8`.
The predecessor chat **Continue joint binary EAGLE parity**
(`01a0e7b2-e75d-7c00-8ad8-3101a73d8409`) is idle. Its last reported supervised
GPU run had exited and process group 417 was absent; tmux session `$33` was
closed. No remote job is inherited. Both GPU host pause flags were set locally
for this handoff; no remote availability check or model run was performed.

CPU work starts immediately in parallel file partitions: native runtime and
latency tooling; efficient device-configurable W1Ax joint trainer; larger-data
ingestion/split and compact-teacher tools; Luna validation. Use a short Astra
consultation for representation/gradient risks. The fresh coordinator owns
integration and these durable records; it must register its chat and worker
IDs, worktrees and owned files here before handing out conflicting edits.

Reuse all previously audited capture, masks/cache, gradients and exporter
work. Native captured target values are the teacher contract; no exact HF
target reproduction is required. Keep structural correctness and practical
native export/trajectory checks. The inherited 96-prompt eligibility flag
requires a documented replacement readiness assessment, not silent editing.
Preserve legacy final prompts unopened.

Explicit remaining design work includes row versus group scale compatibility
(group-128 currently runs only with A16), hard-label versus target-probability
loss, prompt/token scaling, and later GPU budgets based on measured training
step time. These do not block CPU implementation. No DFlash/DSpark work starts.
Stop at a concrete GPU-only validation/training boundary with a reviewable
checkpoint; do not create more open-ended numerical investigations.

## Fresh-team execution: CPU Phase 1A

- Coordinator task `01a0e9d0-1273-70a1-972e-8d1381f72701` resumes this same
  project goal from plan commit `8bd1b91`. Its native task Goal covers only
  CPU Phase 1A. Main checkout was clean at takeover. No GPU or remote model
  operations are assigned.
- Boundaries: legacy final-set contents stay sealed; the 96-prompt capture is
  smoke/regression evidence only. Group-128/A16 and row-scale W1Ax are distinct
  representation contracts. GPU timing, capture and training wait for explicit
  restored access.
- Initial file ownership: coordinator owns `docs/STATUS.md`, this goal file,
  `docs/DECISIONS.md`, integration and final checks. The runtime owner alone
  writes the `llama.cpp` submodule, and owns new latency tooling and its tests.
  The QAT owner owns `src/w1a1_eagle/recurrent_*` changes, a new joint trainer
  entry point and its tests. The data owner owns new data ingestion, manifests,
  compact-teacher tools and their tests. The CPU verification owner reads
  archived traces, owns a separate accounting/validation report and adds
  disjoint tests by coordination. The representation advisor is read-only.
  Fresh bounded subagents: `/root/runtime` (Sol high, sole native writer),
  `/root/qat` (Sol high), `/root/data` (Sol high), `/root/verify` (Luna high),
  and `/root/advice` (Astra medium, read-only). Each implementation owner is
  preparing an isolated managed worktree; paths/branches will be recorded at
  the first implementation checkpoint.
- Worktrees assigned: runtime `/Users/pippo/.codex/worktrees/w1-runtime-latency/binary-eagle-decoding`
  (branch being named); QAT
  `/Users/pippo/.codex/worktrees/joint-w1ax-qat/binary-eagle-decoding`
  (`work/joint-w1ax-qat`);
  data `/Users/pippo/.codex/worktrees/w1a-data-prep/binary-eagle-decoding`
  (branch being named); verifier
  `/Users/pippo/.codex/worktrees/cpu-verification/binary-eagle-decoding`
  (`verify-archived-evidence`).
  Runtime initializes its own native submodule checkout; no other worker
  writes that submodule.
- Early compatibility finding: native row-scale kernels cover A16/A8/A4/A1,
  but the v2/v3 learned-scale loader metadata gate currently rejects values
  below A16 even for row scales. Runtime owner is assessing a bounded gate
  fix, with group-128 still A16-only; QAT will label export boundaries
  honestly until the native gate is validated.
- The Luna verification owner completed its separate archival report on branch
  `verify-archived-evidence` at `e347449`; reviewed and integrated on main as
  `e5bd3dc` and pushed. [The report](../../experiments/cpu-archived-latency-and-readiness.md)
  distinguishes historical CPU-wall, request and profiler scopes and classifies
  the old 96-prompt bundle as an audited smoke fixture without changing its
  inherited `training_eligible: false` metadata. No new benchmark or model
  run occurred. Its test plan awaits the implementation owners' CLIs.
- The data owner committed `8dd23cb` on `w1a-data-prep`, reviewed and
  integrated as main `0b5fce7` and pushed. It provides source-hashed catalog
  ingestion, deterministic exact/near deduplication, grouped nested train
  tiers plus independent development/final splits, and compact native teacher
  shards with exact-prefix and outside-draft mass checks. Four synthetic CPU
  tests and Ruff passed in the repository environment. No public source or
  target-token counts are claimed yet; the owner is preparing pinned real
  source choices and a catalog under ignored data storage.
- The QAT owner committed `9d58840` on `work/joint-w1ax-qat`, reviewed and
  integrated as main `6db186f` and pushed. Row-scale A16/A8/A4/A1 hard
  forwards, a separate group-128/A16 path, joint optimizer and sign/scale
  metrics, compact teacher loss, schema-v2 row export and synthetic CPU runner
  are present. Thirteen focused CPU tests and Ruff passed on main. The owner is
  adding a captured-data/real-drafter adapter; the current runner alone is a
  smoke fixture, not a real training command.
- The runtime owner's first native `25e31b6`/parent `4a6eb78` candidate placed
  the opt-in unused-head branch in the encoder graph, which has no draft head;
  it was **not integrated**. Corrected native `14c188e` was pushed to the
  user's fork before parent `d484b6e`; reviewed and integrated on main as
  `112693f` and pushed. The branch now prunes the decoder output norm/head/map
  only for zero-logit, no-embedding calls under `GGML_EAGLE_PRUNE_UNUSED_HEAD=1`.
  The shared graph covers FP16, Q4_0 and W1Ax. Learned row-scale v2/v3 loader
  metadata accepts A16/A8/A4/A1, while group-128 remains A16-only. The offline
  [latency report](../../experiments/eagle-latency-phase1a-cpu.md) preserves
  residual CPU-wall time and a fixed-input later A/B protocol. On main, an
  Apple M3 Max arm64 CPU build with CUDA/Metal off, 11 native loader tests, two
  accounting tests, Ruff and archived analysis passed. No model inference or
  accelerator test ran; cache/trajectory equivalence and speed of the opt-in
  path remain unverified until native GPU A/B.
- The data follow-up `3d40344` on `w1a-data-sources` was reviewed and
  integrated as main `c4f6764`, pushed. The [candidate source freeze](../../experiments/w1a-public-source-freeze.md)
  pins Dolly, GSM8K and MBPP revisions, terms, raw/normalized hashes and
  transformations. The ignored main-checkout catalog SHA256 is
  `498017fd13a49c97f782ff50035d544dcf98bb70583ffadf68d61236c2415647`;
  the candidate split manifest SHA256 is
  `dc37f752bb137054183162dcb7ca96004edabeb752bd611996d9cb289bfdc667`.
  It contains 2,000 train prompts (667 prose, 667 code, 666 reasoning), 192
  independent development and 192 sealed new final prompts (64/domain each).
  Metadata audit found zero cross-split ID/group/exact-content-hash overlap;
  final text was not opened. Five focused CPU tests and Ruff pass. This is a
  candidate, not tokenized or natively captured training data; the three
  sources cannot supply a balanced 10,000-prompt tier. Ignored raw/normalized
  data and freeze were copied from the temporary worktree to main checkout
  `data/` with hashes rechecked before worktree retirement.
- During the Luna integrated-verification pass, an environment discovery step
  accidentally called the local PyTorch CUDA/MPS availability APIs once. CUDA
  reported unavailable and MPS available. No accelerator tensor/kernel/model
  operation, remote check or inference ran. The worker stopped availability
  probing and continues with explicit CPU commands only. This is recorded as
  an instruction deviation, not GPU validation evidence.
- The Luna pass on main `1e32901` ran the archived analyzer to ignored
  `results/cpu-integrated-verification-20260928/eagle-latency-accounting.json`
  (SHA256 `2b6c2a36708d96b0ecfd0f75dc6477673f9defef2596e0f067d7f7dc8554df12`).
  It reconciled every round and draft-stage partition across 11 variants;
  Q4_0 had 1,493 rounds/7,320 proposals/1,555 accepted/3,048 emitted,
  D group-128/A16 had 2,139/10,449/909/3,048. Their unassigned CPU-wall
  round time was 103,568 and 151,330 µs respectively. Two analyzer, four data
  and six QAT tests passed; explicit CPU one-step smoke ran row A16/A8/A4/A1
  and group-128/A16, each with finite loss and 18 gradient tensors. No sign
  flips occurred in those single steps; they are not convergence evidence.
  This Luna worktree lacked a CPU `libllama` and skipped its loader class,
  while the coordinator's separate main CPU build and 11 loader tests passed.
- The completed data worktree was archived after copying ignored raw/normalized
  files and the candidate freeze to main; temporary data branches were removed
  after patch-equivalence checks. Native and verification worktree cleanup is
  pending; QAT provider work remains active.
- The QAT provider addition `2e31437` was reviewed, integrated on main as
  `29f0e96` and pushed. It installs all nine row modules from pinned original
  dense weights or group modules from explicit D arrays; audits native-prefix
  rounds and exact-prefix compact teacher binding; rebuilds the current student
  context cache and retains state/K/V gradients. The CLI accepts an injected
  provider factory and rejects `training_eligible:false` before loading
  models. Twenty-one focused CPU tests and Ruff passed on main. A concrete
  model/capture-backed factory is still being implemented so the documented
  command will exist when larger eligible native captures arrive; no model
  weights were loaded or inference run in this phase.
- Runtime and both Luna verification worktrees were archived after tracked
  integration checks; their temporary parent branches were removed. The
  native fork branch `research/w1-phase1a-runtime` remains published because
  main's gitlink references `14c188e`. Ignored analyzer JSON and five tiny
  QAT smoke artifacts were preserved in main `results/` before archival.
- The concrete capture factory `e792493` was reviewed, integrated as main
  `83e72d6` and pushed. It pins capture/prompt/map/model/teacher hashes,
  rejects inherited ineligible bundles before loading weights, validates
  compact teacher prefix/logit-row/label joins, and exposes the real
  `w1ax_capture_provider:create_provider` CLI path. Fourteen focused CPU tests
  and Ruff passed on main using injected model doubles. No weights were loaded
  in this check. The QAT owner is adding a device-agnostic Torch rollout under
  an explicit accelerator guard; actual accelerator execution stays paused.
- A new bounded data subtask owns sharded capture preparation in managed
  worktree `/Users/pippo/.codex/worktrees/w1ax-capture-shards/binary-eagle-decoding`
  on branch `w1ax-capture-shards` from `83e72d6`. It owns only a new shard
  planner, tests and report; QAT owner alone edits provider files. The reason
  is the 24.8 GB raw full-vocabulary logit file from just 96 old prompts.
  Shards must cap both prompts and raw logit rows/bytes, audit and compact
  exact-prefix teacher data before any raw retirement, and preserve hashes.
- The shard audit found the inherited capture runner/preparers/bundle/auditor
  still hardcode the old 96-prompt hash/count. A separate Sol owner has sole
  ownership of those five existing scripts and disjoint tests in managed
  worktree `/Users/pippo/.codex/worktrees/capture-shard-contract/binary-eagle-decoding`
  on `codex/capture-shard-contract` (started from `78d98f2`). Its bounded
  change will accept an explicit frozen prompt file/hash/count while keeping
  the 96 defaults. The data owner keeps only the new shard planner/report;
  raw-logit retirement remains disabled until compact provenance is verified.
- The device-aware QAT follow-up `2b19b2f` was reviewed, integrated as main
  `4d0684d` and pushed. Row-scale standard Torch attention/rollout follows
  the declared device, with F16 K/V cache casts and exact prefix/mask/position
  structure; group-128 and native CPU-oracle modes remain CPU-only. The CLI
  still requires `--allow-accelerator` for CUDA, and neither CUDA nor MPS was
  exercised. Twenty-nine focused CPU tests and Ruff passed on main; a separate
  meta-device shape check by the owner used no accelerator. CUDA memory,
  training throughput and native trajectories remain hardware-only gates.
  A single eligible shard can support the first 100-step calibration; the QAT
  owner is coordinating a multi-shard iterator with the data planner for a
  continuous 2k-prompt training run and optimizer state.
- The data shard planner `f1f7276` was reviewed, integrated as main `08c70b0`
  and pushed. Ignored `data/w1ax-capture-shards/plan-003/plan.json` has SHA256
  `a20a9f8e3a48c65dfc754c0598c1c256f77c9754d874caa04dabcb60b4095c1d`:
  65 ordered shards cover all 2,000 frozen train prompts, preserving declared
  conversation/topic families. Per-shard caps are 96 prompts, 16,384 target
  logit rows and 12 GiB raw bytes; the current largest estimated shard is
  32 prompts/16,384 rows/9.96 GB. Estimates are planning values; the `observe`
  command fails on actual row/byte cap or ancestry mismatch. Four synthetic
  CPU tests and Ruff pass. [The report](../../experiments/w1ax-sharded-capture-plan.md)
  estimates 633.26 GB raw across 2k at its conservative row assumption and
  19.91 GB transient peak for the largest shard when source+bundle raw copies
  coexist. Current v1 bundle re-audit requires raw logits, so the planner
  explicitly prohibits raw retirement; full-tier storage remains an open
  capture-schema decision, not a solved compact-storage claim.
- The five-script manifest-aware capture extension `71f152f` was reviewed,
  integrated as main `a18e5a9` and pushed. Runner, native row/feature
  preparers, bundle builder and independent auditor now accept explicit frozen
  prompt SHA/count with old 96 defaults preserved. The runner binds an ordered
  shard manifest and enforces full raw target-logit coverage; the bundle embeds
  the shard manifest and retains raw bytes with retirement prohibited. The
  independent audit checks embedded provenance and hard row/byte caps. On main,
  35 affected synthetic CPU tests and Ruff passed. The [shard runbook](../../experiments/w1ax-sharded-capture-plan.md)
  command was corrected in `6e8a58d` to pass `--shard-manifest` to the bundle
  builder. No model capture or GPU run has tested these new CLI paths.

## CPU Phase 1A completion and GPU-only boundary

Phase 1A implementation is integrated on pushed main through `d9f3ae9`; the
native gitlink is published fork commit `14c188e`. Main was clean at this
checkpoint. All temporary team worktrees were archived after integration and
ignored data preservation; their temporary parent branches were removed. The
same project goal remains active for Phase 1B. Both GPU hosts remain paused;
no remote check, CUDA/Metal/MPS model run, target inference, training or native
performance measurement was performed by this team. One Luna worker's single
accidental local CUDA/MPS *availability query* is recorded above; no accelerator
compute followed it.

- Latency: the [offline analyzer](../../scripts/account_eagle_latency.py) and
  [archival report](../../experiments/cpu-archived-latency-and-readiness.md)
  separate nested draft/process CPU-wall spans from request rates and lifetime
  kernel sums, retaining unassigned time. No per-projection or CUDA-graph-node
  attribution exists yet. The opt-in shared decoder head-pruning patch is
  source-reviewed and CPU built, but its native trajectory and speed effect
  remain unmeasured. Paired A/B configs are
  [off](../../configs/w1_phase1b_prune_off.json) SHA256
  `38e44ad17c5a32c1fd7d0d52d13d618d8254a238d94dfdbe46980233d6d842c1`
  and [on](../../configs/w1_phase1b_prune_on.json) SHA256
  `ad69c379def326c169d00cbb6750553841fddc0dbf9e066862931b0d9e68e8eb`.
  They differ only by `GGML_EAGLE_PRUNE_UNUSED_HEAD=0/1`, use Q4_0 as primary,
  D group-128/A16 and FP16 as diagnostics, and keep target-only as control.
  Local pinned target/Q4_0/D/FP16 draft GGUF hashes are respectively
  `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`,
  `2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280`,
  `10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf`,
  `c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1`.
- QAT: row-scale A16/A8/A4/A1 hard-forward training, separate candidate-D
  group-128/A16, nine-linear installation, exact-prefix provider, compact
  teacher option, guarded Torch-device rollout, checkpoint/export and
  multi-shard one-optimizer streaming are implemented. Hard-label CE is the
  first objective; top-k/tail conditional probability loss is an explicitly
  approximate option. The native learned-row loader passes CPU metadata tests
  for all four widths. Actual CUDA numeric/throughput and native proposal/cache
  checks are pending. The old 96-prompt bundle keeps
  `training_eligible:false`; the provider rejects it before model loading.
- Data: the pinned candidate 2k/192/192 train/dev/sealed-final freeze and
  65-shard plan remain under ignored main `data/` with manifest SHA256
  `dc37f752bb137054183162dcb7ca96004edabeb752bd611996d9cb289bfdc667`
  and plan SHA256
  `a20a9f8e3a48c65dfc754c0598c1c256f77c9754d874caa04dabcb60b4095c1d`.
  This source set is narrow and not target-tokenized or captured. Full-tier
  v1 raw retention is potentially above 1 TB with bundle copies; the
  [storage choice](../DECISIONS.md#phase-1a-implementation-defaults-and-pending-research-choices)
  remains user-owned before a full 2k capture. The first capped shard can be
  retained raw for a bounded calibration without deciding the full tier.

**Integrated CPU checks:** Apple M3 Max arm64 `GGML_CUDA=OFF`,
`GGML_METAL=OFF` llama shared-library build; 11 native loader tests; 36
focused integrated Python tests after formatting; changed Python files pass
Ruff lint/format and `git diff --check`. Archived analyzer totals reconcile;
single-step CPU QAT fixtures cover row A16/A8/A4/A1 and group-128/A16 with
finite loss and all 18 parameter-gradient tensors. A meta-device shape check
by the QAT owner used no accelerator. Single-step sign flips were zero, so
these checks do not establish learning. Whole-repository Ruff lint reports
legacy errors in untouched files; only changed-file lint is claimed green.

**Exact first GPU commands, only after the user explicitly restores access:**
the coordinator first runs `python3 scripts/agent_env.py resume rtx5080`, reads
the saved host registry, and connects solely through tmux MCP. On that saved
host's project checkout, one GPU owner fast-forwards main and updates the
submodule, verifies the four GGUF hashes above and the two config hashes, and
builds the pinned CUDA binary. If any artifact or hash differs, stop and
resolve it before running. Start each command below inside its own tmux
MCP-managed session and supervise it with `scripts/remote_job.py`:

```sh
python3 scripts/remote_job.py w1-prune-off-quality -- python3 scripts/run_binary_rescue_benchmark.py --config configs/w1_phase1b_prune_off.json --mode quality --output runs/w1-prune-off-quality/benchmark
python3 scripts/remote_job.py w1-prune-on-quality -- python3 scripts/run_binary_rescue_benchmark.py --config configs/w1_phase1b_prune_on.json --mode quality --output runs/w1-prune-on-quality/benchmark
python3 scripts/remote_job.py w1-prune-off-timed -- python3 scripts/run_binary_rescue_benchmark.py --config configs/w1_phase1b_prune_off.json --mode timed --output runs/w1-prune-off-timed/benchmark
python3 scripts/remote_job.py w1-prune-on-timed -- python3 scripts/run_binary_rescue_benchmark.py --config configs/w1_phase1b_prune_on.json --mode timed --output runs/w1-prune-on-timed/benchmark
```

Run the two quality commands first; require identical emitted IDs, proposed/
accepted/round counts and verifier outcomes before exploratory timing. Cache/
state equivalence needs a separate bounded native trace before enabling the
opt-in path by default or claiming an output-preserving speedup. Alternate
timed order in a second paired block if the first is clean, and report
absolute draft/process/request times and acceptance versus Q4_0. Do not
sum overlapping CPU/GPU spans or claim SM75 speed from RTX 5080. The
[first-shard capture command block](../../experiments/w1ax-sharded-capture-plan.md#native-command-sequence-and-current-interface-gate)
is next after fixed native timing, with `shard-0000` capped at 16,384 raw
logit rows. Its model/variant/map inputs must be pinned on the remote host.
The builder still emits preparation-only bundles; a larger capture's
eligibility/readiness gate and a measured 100-step budget must be recorded
before substantive QAT. The later 2080 Ti SM75 comparison remains separate.

**Stop condition:** CPU Phase 1A is complete. Do not manufacture more backend
parity diagnostics or start another architecture while GPU access is paused.
The next action is the supervised native A/B and first-shard calibration after
explicit access restoration and user-owned research choices at their stated
gates.

## GPU Phase 1B startup

On 2026-09-28 the user said the GPU is free and authorized continuing this
same EAGLE goal. The coordinator task
`01a0e9d0-1273-70a1-972e-8d1381f72701` owns the sole RTX 5080 experiment;
its native task Goal is now scoped to the first native A/B and bounded shard
calibration. The local `rtx5080` pause flag was resumed; `rtx2080ti` remains
paused. The saved host registry supplied the current RTX 5080 address and
workdir. All SSH is through tmux MCP session `$34` (`w1-phase1b-5080`), pane
`%54`, into WSL. First `nvidia-smi` showed RTX 5080, 3,365 MiB whole-device
use, 2% utilization and no listed running GPU processes; WSL process check
found no project `remote_job.py`, `llama-server` or joint trainer. These are
fresh startup observations, not a reserved-capacity guarantee.

The remote project checkout fast-forwarded from `6a8e3b3` to `81a2a96` and
updated the native submodule to published `14c188e`. Existing untracked
`checkouts/` and `rescue-head-20260927/` directories were left untouched.
The remote D group-128/A16 and Q4_0 artifacts were found in the preserved
scale-fitting archive, with SHA256
`10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf`
and `2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280`;
the pinned A/B config paths need verified local links to those archive files.
Target, FP16 and both A/B config SHA256 values matched the Phase 1A checkpoint.
The remote pinned Q4_0 and D artifacts had different ignored directory paths;
after hash verification, the coordinator added symlinks at the tracked-config
paths pointing into their preserved scale-fitting archive. Existing artifacts
were not overwritten. The first model run will be the off/on quality pair;
timing follows only after comparing outputs and counts. Each run gets its own
remote supervisor directory, and the process group must stop before the GPU
is reported free.

The first supervised build attempt `runs/w1-phase1b-cuda-build-20260928/`
exited code 2 before compilation because plain `python3` had no `cmake` on
PATH. The locked project environment resolved `cmake`, `ninja` and `nvcc`.
Corrected supervised build
`runs/w1-phase1b-cuda-build-uv-20260928/` started at 2026-09-29 05:22:50 UTC
in tmux MCP pane `%54`, PID/process group 3001, with
`uv run --locked python scripts/build_llama.py cuda --cuda-arch 120
--cuda-include-root results/cuda-glibc-compat/include --with-tests --jobs 4`.
It finished with exit 0 at 2026-09-29 05:29:22 UTC; its supervised process
group stopped. The build used 342 Ninja steps and produced the current
`build/llama-cuda/bin/llama-server` plus backend tests. The bounded CUDA
operator job `runs/w1-phase1b-w1ax-ops-20260928/` then finished exit 0:
**112/112 W1A1_MUL_MAT cases passed** on RTX 5080/SM120, including A1/A4/A8/A16
and group-128/A16 variants. This is operator correctness, not whole-model
acceptance or speed.

The remote current-source regeneration of the pinned Dolly/GSM8K/MBPP data
produced byte-identical train/dev/sealed-final prompt files but different
catalog/manifest hashes because the later format-only source commit changed
the transformation script's own byte hash. The coordinator reconstructed the
original frozen catalog and manifest metadata, verified exact SHA256
`498017fd13a49c97f782ff50035d544dcf98bb70583ffadf68d61236c2415647`
and `dc37f752bb137054183162dcb7ca96004edabeb752bd611996d9cb289bfdc667`,
and preserved the regenerated metadata under `runs/w1-data-freeze-20260928/`.
The supervised shard planner `runs/w1-shard-plan-20260928/` finished exit 0;
plan and first child hashes match
`a20a9f8e3a48c65dfc754c0598c1c256f77c9754d874caa04dabcb60b4095c1d`
and `17b8c65c47b335449e7573e42ec644f0682a7b5dfb97bb1874f1bc8eb05cd8b6`.
No final prompt content was inspected.

Before a model run, fresh RTX 5080 readings rose from the initial 2% to
55–61% utilization, about 4,136 MiB whole-device use and 204–206 W while no
WSL GPU process was listed; this appears to be non-project load. It was not
interrupted. The operator test is valid as a correctness check, but these
conditions preclude a clean timing claim. An earlier reply misread the user's
follow-up as a direction not to pause; the later pause checkpoint below
corrects this. The completed supervised quality checks are retained, while
throughput timing remains unrun.

The first off-path native quality job started at 2026-09-29 05:33:09 UTC as
`runs/w1-prune-off-quality-20260928/`, supervised in tmux MCP pane `%54` with
PID/process group 7482. It uses the pinned `prune_off` config, 24 old
development prompts, Q4_0/D group-128/A16/FP16/target-only and one diagnostic
repetition. The job is progressing through requests; this is quality and
trajectory evidence, not an uncontended timing measurement. Monitor from pane
`%56`; interrupt the supervisor and verify process group/GPU state if the
user asks to pause or the job must stop.

The off-path quality supervisor finished exit 0 at 2026-09-29 05:38:04 UTC.
Its manifest is complete with 96 measured requests (24 for each variant)
and reproduces archived Q4_0/D/FP16 totals: respectively
1,493/2,139/1,496 rounds; 7,320/10,449/7,329 proposals;
1,555/909/1,552 accepted; 3,048 emitted tokens per variant. These are
quality counts on the heavily reused old development suite, not held-out
promotion evidence or speed. Fresh GPU reading after this project job stopped
was 86% utilization, 4,365 MiB whole-device use and 275.6 W from workload
outside visible WSL processes. It remains a timing contamination.

The on-path quality job `runs/w1-prune-on-quality-20260928/` started at
2026-09-29 05:39:47 UTC, tmux pane `%54`, PID/process group 7800. It is
progressing through the same pinned prompts under
`GGML_EAGLE_PRUNE_UNUSED_HEAD=1`; pane `%56` monitors its state and log.
Compare exact per-request token IDs, proposals, accepted counts, round rows
and verifier outcomes against the off-path manifest after exit. Do not start
another project GPU job concurrently.

The on-path supervisor finished exit 0 at 2026-09-29 05:43:22 UTC. The
[paired quality report](../../experiments/eagle-prune-quality-5080.md) records
the result: **96/96 measured request pairs matched** in raw generated IDs,
completion hashes, speculative counters and checked per-round proposal,
acceptance and emission fields. Q4_0/D/FP16 totals stayed exactly
1,555/909/1,552 accepted drafts with 3,048 emitted tokens per variant. The
comparison JSON in `runs/w1-prune-ab-compare-20260928/report.json` has SHA256
`891e8cfb979752cb49c77f5c3fa130142a2d40f568e94abfe5dc84ab079c5f63`.
Both model supervisors and the CPU comparison supervisor exited zero; no
project process group remains active. Exact cache bytes were not captured.
Afterward the device still showed external 87% utilization, 4,755 MiB use and
323 W. The user was asked asynchronously whether this competing workload
can be cleared; timed A/B and a large capture remain resource-gated, while
CPU preparation continues. The local GPU pause flag remains resumed.

First-shard readiness work continued without another GPU job. The remote
`w1a1` Python environment synced from the lockfile, providing NumPy 2.4.6 and
Torch 2.14.0+cu130. The new [D-only capture variant](../../configs/w1_phase1b_capture_d.json)
has SHA256 `6d44e89844f38f98b72fa2b05eb6a4d44d090b4feb5f3066a47d36f9f1f1b120`;
the archived candidate-D absolute I64 d2t `.npy` bytes match local SHA256
`6dcd8cadd270000775cb04278c5f994647cc1ab2e2e762982c5ff7f31efc78f4`.
The runner's static `validate_inputs` passed on shard-0000 with its 31 exact
train IDs, prompt SHA `968ffbb21b23f934912862ef6f7add7d03bf0cfbbb3910492f2f0f50075fc18a`,
target/D hashes, vocabulary map, 16,384 raw-logit-row limit and 131,072
target-feature-row limit. This did not start inference or write a capture.
Remote filesystem space was 795 GiB free. The archived quality run's GPU
telemetry reached 13,415 MiB whole-device memory used while external load
was present; a fresh post-run reading remained at 4,731 MiB, 91% utilization
and 265.6 W. These facts make a 31-prompt feature/logit capture risky until
the non-project workload clears. No timed A/B or shard capture has begun.

Latest resource gate: with every project supervisor stopped, a later RTX 5080
reading was 4,824 MiB whole-device use, 93% utilization and 308.4 W.
`ps` found no project supervisor, `llama-server` or joint trainer process.
An asynchronous resource question was sent under the earlier mistaken reading
of the user's follow-up; the pause checkpoint below supersedes it. At this
historical observation the local RTX 5080 flag was still resumed and tmux MCP
session `$34` remained open. Fair timed A/B and the memory-heavy first-shard
capture were held for capacity.
The off/on quality evidence and CPU first-shard preflight are preserved, so no
completed experiment needs rerunning merely to resume.

## GPU Phase 1B pause checkpoint

At 2026-09-29 05:54 UTC the coordinator recognized that the user's “No pause
gpu work actually” corrected the earlier go-ahead and requested a pause. The
prior response had misinterpreted it. The local `rtx5080` flag was immediately
set to paused; `rtx2080ti` was already paused. No project GPU job was running
at that point. Through tmux MCP, a fresh WSL process check found no
`remote_job.py`, `llama-server`, joint trainer or backend-ops process; all
Phase 1B `w1-*` supervisor `state.json` records were terminal. The paired
quality supervisors, comparison, CUDA build, operator test, data preparation
and shard-plan jobs exited zero; the initial plain-Python build exited code 2
before compilation and was superseded by the successful locked build.

The final GPU reading after project jobs stopped was 4,760 MiB whole-device
use, 75% utilization and 231 W. Windows-side `nvidia-smi` listed game and
display processes, including Fortnite; those were not interrupted. **Project
GPU use is stopped, but the whole device is not idle.** The tmux MCP session
`$34` was closed after verification. The local checkout is clean, and the
completed [quality comparison](../../experiments/eagle-prune-quality-5080.md)
plus ignored remote run artifacts remain preserved. No timed A/B, first-shard
capture or 100-step QAT calibration ran. The first-shard static preflight had
passed, including the exact D map and 31 training prompt identities.

Next action only after the user explicitly resumes GPU work: use the saved
host registry and a new tmux MCP session, check fresh resource/process state,
fast-forward the remote checkout, then perform the first-shard capture or a
clean timed A/B as appropriate. Keep the old 24 development prompts as
regression data and the new final set sealed. The active project goal remains
the joint binary EAGLE body/head study; this task's native GPU Goal is paused.

## GPU Phase 1B resume and timed A/B

At 2026-09-29 06:53 UTC the user explicitly restored GPU access. The native
task Goal resumed and the local `rtx5080` pause flag was cleared; the
`rtx2080ti` flag remains paused. The coordinator remains the sole project GPU
owner. New tmux MCP session `$35` (`w1-phase1b-resume-5080`), execution pane
`%57` and monitor pane `%58`, reconnected through the current saved host
registry into WSL. A fresh pre-run device reading was **0% utilization**,
2,899 MiB whole-device memory and 48 W; WSL showed no project supervisor,
server, trainer or backend test. The remote checkout fast-forwarded to
`a8492ee` without touching ignored artifacts or untracked directories.

The first clean timed run started at 2026-09-29 06:54:48 UTC:
`runs/w1-prune-off-timed-20260928/`, supervised through
`scripts/remote_job.py` in pane `%57`, PID/process group 8574. It uses the
pinned prune-off config, Q4_0/D group-128/A16/FP16/target-only, 24 old
development prompts, two warmups and five measured repetitions. Pane `%58`
monitors `state.json`, stdout, server blocks and GPU state. Do not start a
second project GPU job while it runs. On completion, verify terminal state,
process-group absence and resource use before the matched prune-on run. If
the user requests another pause, mark the host paused first, interrupt this
supervisor through tmux MCP, then verify its process group has stopped.

The prune-off supervisor finished exit 0 at 2026-09-29 07:08:41 UTC. Its
manifest is complete: 480 measured requests across Q4_0, D group-128/A16,
FP16 and target-only (24 prompts × 5 repetitions each); all 20 server blocks
record `verified_launches` for CUDA graphs. WSL showed no remaining project
supervisor/server process, and the GPU returned to 0% utilization, 2,899 MiB
whole-device baseline and 40.5 W. The matched prune-on timed run started at
2026-09-29 07:09:52 UTC as
`runs/w1-prune-on-timed-20260928/`, tmux MCP pane `%57`, PID/process group
9554. Monitor from pane `%58`; do not start another project GPU job until it
finishes and the process group is absent. An offline paired comparator was
committed as `d9c5e96` and pulled into the remote checkout before this run.

The first matched prune-on supervisor finished exit 0 at 2026-09-29
07:23:19 UTC, again with 480 measured requests and 20/20 verified CUDA-graph
blocks. Both runs returned the GPU to 0% and 2,899 MiB baseline with no
remaining project process. [The paired timing report](../../experiments/eagle-prune-timing-5080.md)
records zero behavior mismatches across 480 request pairs. On/off server
decode throughput ratios were Q4_0 **1.0088×**, D **1.0528×**, FP16 **1.0113×**
and target-only **1.0026×**. D remained only about 0.403× Q4_0 decode rate;
acceptance recovery is still the principal gap. Target-only drift and the
off-then-on ordering prohibit a final speed claim from this one block.

The reverse-order confirmation started with prune **on** at 2026-09-29
07:26:10 UTC: `runs/w1-prune-on-timed-b-20260929/`, tmux MCP pane `%57`,
PID/process group 10756. GPU preflight was 0%, 2,899 MiB and 39 W. This is
the sole active project GPU job. After its terminal state and process-group
check, run prune **off** with the same config and a new supervisor ID, then
compare both blocks and their target-only drift before deciding what to keep.

The reverse-order prune-on supervisor finished exit 0 at 2026-09-29
07:39:35 UTC: 480 measured requests and all 20 graph blocks marked
`verified_launches`. The process group stopped and the GPU returned to 0%,
2,899 MiB and 40 W. The reverse-order prune-off supervisor then started at
2026-09-29 07:40:52 UTC as `runs/w1-prune-off-timed-b-20260929/`, pane `%57`,
PID/process group 11639. It is the sole active project GPU job; pane `%58`
monitors it. Once finished, run the same behavior/timing comparator with
these second-block manifests and assess order-dependent drift before deciding
whether to retain the opt-in patch.

The reverse-order prune-off supervisor finished exit 0 at 2026-09-29
07:54:35 UTC: 480 measured requests and 20/20 verified graph blocks. The
process group stopped and the GPU returned to 0%, about 2,900 MiB and 40 W.
The second paired comparison finished exit zero with **480/480** exact raw
outputs and speculative counters. [The timing report](../../experiments/eagle-prune-timing-5080.md)
now includes both orders: order-balanced on/off server decode ratios Q4_0
**1.0085×**, D **1.0513×**, FP16 **1.0112×**, target-only **1.0013×**. The
reverse comparison JSON SHA256 is
`e452374ceaa9c125310ff79eb9f1ec25424916e9f253455a2567c93499104488`.
These data support keeping the shared patch opt-in for this RTX 5080 workload;
exact cache bytes and `process()` stage attribution remain open before a
default change. D still trails Q4_0 strongly because of acceptance.

With the GPU clear, the first bounded larger-data capture started at
2026-09-29 07:57:32 UTC as
`runs/w1-shard0000-capture-20260929/`, supervised in tmux MCP pane `%57`,
PID/process group 12597. It uses the frozen shard-0000 plan (31 training
prompts; SHA256
`17b8c65c47b335449e7573e42ec644f0682a7b5dfb97bb1874f1bc8eb05cd8b6`),
the pinned F16 target and D group-128/A16, absolute I64 d2t map, 16,384
raw target-logit-row cap and 131,072 target-feature-row cap. The runner's
static preflight passed before launch. Pane `%58` monitors `state.json`,
stdout, raw row/byte growth and GPU use. No other project GPU job may start
while capture is active. If interrupted, stop this supervisor and its process
group before reporting the GPU free; retain any partial raw artifacts for
diagnosis, never silently promote an incomplete bundle.

## GPU Phase 1B first-shard audit

The sole supervised shard-0000 capture finished exit zero at 2026-09-29
07:59:33 UTC. All 31 frozen train prompts completed. The WSL project process
group stopped and RTX 5080 returned to 0% utilization, about 2,900 MiB
whole-device baseline. CPU follow-ups under unique `w1-shard0000-*` run IDs
then passed: static/observed cell cap and SHA checks, native row preparation
(2,562 rounds, 12,610 logit rows), feature preparation (7,796 selected
accepted-prefix rows), bundle build, independent byte-identical re-audit,
12,610-row compact teacher creation and verification, internal round
continuity, and 31/31 full-request emission audit. Raw F32 logits are
7,663,651,840 bytes and remain in the source cell and bundle; no raw
retirement is allowed. The [capture report](../../experiments/w1ax-shard0000-capture-5080.md)
records artifact hashes, counts, hardware, precision and limits.

The bundle manifest SHA256 is
`3ee7a8f4526f1dbcca1b6e0ea0756e42eff81333d7a88137afebe7213d3c1947`.
Independent audit SHA256 is
`832325813eefea67dc97dc0d251b1e37b3a7b4a7349a4e26eb3aa9663198508b`;
it matched builder audit byte for byte. Compact teacher manifest SHA256 is
`2d4b39867e2199c39116d98491baf5b46c706c17709b13d0efdebf910255fc52`.
The full-request auditor was extended to accept an explicit frozen prompt
hash/count while preserving its old 96-prompt defaults (`4db0440`, pushed
and pulled to WSL). It verified 3,836 response tokens, 1,220 accepted drafts
and 2,562 rounds; initial/terminal sampler parity is still unverified.

The bundle deliberately remains `training_eligible:false` and
`readiness:preparation_only`. Its declared gaps are model execution identity,
numeric target-feature parity, full drafter mask/position/KV parity and
cross-round ancestry. The new continuity and full-request evidence narrows
the last item but cannot silently promote this inherited v1 manifest. The
provider rejects it before model loading, so no captured-data 100-step QAT
has started. A model-independent 100-step row-A4 CUDA trainer fixture ran as
the sole GPU job in tmux MCP pane `%58`:
`runs/w1-joint-cuda-fixture-a4-20260929/`. It finished exit zero at 08:08:33
UTC in 19.5 seconds with finite metrics and 18 gradient tensors at every
step; its loss moved 1.40130 to 1.40134 and no signs flipped. This checks
the accelerator training path only, not native model memory, real step rate,
convergence or quality. Its `training_run.json` SHA256 is
`4515af9a865a3dab6607b310de68d735ffeb855e04acf0e3d455b1154c2dcda8`.
Its process group stopped; RTX 5080 returned to 0%, about 2,900 MiB and
41 W, with no project supervisor/server/trainer process. RTX 2080 Ti remains
paused. Next: record concrete readiness options, then run only the bounded
native checks needed for the selected eligibility contract before any real
100-step calibration. Keep the raw bundle and final set untouched.

An independent read-only identity check matched the capture cell's native
server SHA256 `a57f9e784eb2528d2de954d12ebcb9ff57c1ee24cd8b11e3df55e526125e52f0`
to the current CUDA binary. The cell pins target/D hashes, CUDA device 0,
all-GPU model layers, F16 K/V cache and graph-disabled capture. Both pinned
Hugging Face model snapshots passed a full file-list and SHA256 recheck
against archived snapshot manifest SHA256
`2db1c860f059bd8702ca6bb523e64f0c7d064c7a95f59919a001fc88168a91e3`.
The prepared 31-prompt provider manifest SHA256 is
`4ce8a76f41f1aeecd7f953951dbe3782d9ae2c27b6794234c7c28d39dabc7d47`;
it copied `training_eligible:false`. A supervised one-step CPU attempt exited
as expected with `training manifest is not eligible` before model loading.
The user-owned practical-versus-strict first-shard readiness choice is now
recorded in [DECISIONS.md](../DECISIONS.md#phase-1a-implementation-defaults-and-pending-research-choices).
