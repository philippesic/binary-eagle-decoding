# Joint binary EAGLE body and head

**Objective:** prepare a faithful W1A16 jointly trainable nine-linear EAGLE
draft body and binary head, then evaluate it against Q4_0 EAGLE when a later
user decision permits the required GPU work. The pinned FP16 target/verifier
and draft vocabulary map stay frozen. [Trial protocol](../../experiments/recurrent-binary-qat-plan.md).

**State (2026-09-28 UTC): active, parity and capture preparation.** The user
lifted the earlier GPU restriction for the free RTX 5080. The frozen training
capture and bounded feature diagnostics have run there; no model training,
final-set use, or model-quality result exists for this goal. The native Codex
Goal in this task has the same objective; this file and `docs/STATUS.md` are
the durable project checkpoint.

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
