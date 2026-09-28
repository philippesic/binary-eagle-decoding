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
