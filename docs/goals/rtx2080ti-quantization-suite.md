# Goal: benchmark the RTX 2080 Ti quantization suite

**Opened:** 2026-09-24  
**State:** active; WSL access established, toolchain and models pending
**Orchestrator:** current Codex task  
**GPU owner:** `/root/sm75_operator`, exclusive 2080 Ti access

## Objective

Measure the complete relevant EAGLE draft precision matrix on the RTX 2080 Ti
under matched target, prompt, and server settings. Include every implemented
native W1A1 coverage setting and the Turing-supported FP16, INT8, INT4, and
binary arithmetic paths. Distinguish actual two-operand low-bit execution from
weight-only GGUF storage and from PyTorch numerical simulations. Produce a
reproducible comparison, including a quantified negative result if appropriate.

## Required matrix and evidence

| Track | Variants | Required execution evidence |
| --- | --- | --- |
| Anchors | target-only; ordinary FP16 EAGLE | Same target, settings, device, and paired prompts |
| W1A1 | fusion, attention, FFN, head, all nine linears | Packed weight and activation operands, CUDA dispatch for each selected group |
| Binary Tensor Core | at least head portable XOR/POPCOUNT versus opt-in SM75 MMA | Real SM75 correctness, `BMMA.88128.XOR.POPC` SASS and runtime dispatch, matched end-to-end comparison |
| Native INT8 and INT4 | W8A8 and W4A4 on the same eligible draft layer coverage, if a correct native path can be built | Actual operand format and operator/SASS evidence, numerical and model-level checks before timing |
| Weight-only controls | Q8_0 and Q4_0 draft GGUFs | Actual CUDA operator and activation precision labels; never call these W8A8/W4A4 |

Use [NVIDIA's Turing precision set](https://docs.nvidia.com/cuda/turing-tuning-guide/#tensor-core-operations)
(FP16, INT8, INT4, binary) as the hardware inventory, then audit which formats
the pinned runtime can actually execute.
Do not fill a native INT8/INT4 row with a simulation or weight-only result. If
a true path proves infeasible, document the concrete blocker, its attempted
check, and the available weight-only control; keep the row visibly unmeasured.
Do not expand to unsupported newer formats such as FP8 or FP4.

For each timed variant preserve at least five alternating measured repetitions
over the frozen 12-prompt suite after warmups, with completion lengths,
request/decode throughput, accepted/proposed/emitted drafts, per-round draft and
verifier cost, memory, text/token parity, and raw artifacts. Record host GPU,
driver, CUDA compiler, build and model hashes, CUDA dispatch, and precision
coverage. Measure packing-inclusive component cost for real EAGLE shapes.
Compare pooled rates and prompt/repetition spread against both anchors.

## Boundaries and decision gates

- Check the shared local host registry for the current address; never commit
  it. SSH to WSL only through tmux MCP. If Ubuntu or SSH is absent, establish
  access before claiming any SM75 result. All remote builds, tests, conversions,
  and inference use unique supervised `scripts/remote_job.py` runs.
- Give the 2080 Ti one GPU owner. Check GPU and other processes before and
  after every experiment. Keep models, raw results, and caches outside Git.
- First establish real SM75 W1A1 backend correctness, standalone binary-MMA
  correctness, and model-level dispatch. Then measure the existing five-group
  plus weight-only matrix. Pursue native INT8/INT4 operator paths only with a
  pinned arithmetic contract and checks; profile them against the same anchors.
- Try the existing FP16 target track subject to measured VRAM. If it does not
  fit, define a separate quantized-target track and run every anchor and
  candidate with the identical target model, KV precision, and context.
- The earlier 5080 target-only versus speculative output mismatch remains a
  correctness limitation. Report paired ordinary/W1A1 text parity and exact
  target divergence evidence; do not call a divergent target-only ratio a
  strict lossless speedup.
- QAT is outside this measurement goal until the untrained post-training
  matrix is established. Its separate plan remains in
  `experiments/qat-revisit-plan.md`.

## First independent work units

1. **Orchestrator (docs and integration):** own this file, `docs/STATUS.md`,
   decisions, runbook, final analysis, review, and main integration. Checkpoint
   commits and raw artifact hashes; give brief user updates.
2. **Sol high precision owner (isolated worktree, no GPU):** audit llama.cpp
   CUDA paths for genuine SM75 W8A8/W4A4 on the eligible EAGLE linears. Deliver
   a format/operator feasibility map and, if bounded, a tested implementation
   proposal or reviewable commit. Own only new precision-path code/report;
   stop before any GPU run or shared-file edit.
3. **Luna high GPU operator (exclusive 2080 Ti owner):** via tmux MCP inspect
   WSL access and hardware, prepare the pinned remote checkout and models,
   run supervised SM75 correctness gates, then the existing five-W1A1 plus
   weight-only paired suite. Own remote sessions/jobs/raw logs only. Stop at
   access failure, correctness failure, or resource contention; record exact
   state, commands, hashes, and results in an experiment report.
4. **Astra medium advisor if needed:** answer a focused INT4/INT8 kernel or
   memory-fit question without GPU use or code edits.

## Opening checkpoint

- Parent `main` was clean and matched `origin/main` at goal creation. The
  expanded llama.cpp gitlink is `8d2b18a`; all five W1A1 variants have local
  conversion, row audit, and CPU smoke evidence in
  `experiments/native-w1a1-groups-local.md`. The nine-variant runner has a
  local dry-run only. Q4_0 and Q8_0 drafts are prepared and audited in
  `experiments/native-weight-only-draft-prep.md`.
- The optional integrated SM75 binary-MMA branch is `9bb01a6`, based on the
  older head-only production revision; it passed 5080 proxy checks only. It
  needs real SM75 correctness and a carefully matched portable comparison.
- The shared host registry lists RTX 2080 Ti user `philip`, port 22, and the
  user-supplied current address. Earlier SSH attempts timed out while Ubuntu
  WSL was absent. No 2080 Ti runtime or benchmark result exists.
- Next: commit and push this checkpoint, dispatch independent precision audit
  and GPU access work, and update this file after the first hardware gate.

## Access milestone

- Opening checkpoint `af5f337` and Turing scope clarification `beffcc8` were
  pushed to `main` before worker access. `/root/int_precision_audit` owns the
  no-GPU native INT8/INT4 feasibility audit; `/root/sm75_operator` alone owns
  the 2080 Ti remote session, setup, raw artifacts, and experiment report.
- tmux-MCP SSH now reaches Ubuntu 24.04 WSL2 at the registered host. The GPU
  is an RTX 2080 Ti, compute capability 7.5, with 11,264 MiB VRAM. The first
  sample showed 855 MiB used, 0% utilization, Xwayland only, and no project
  compute process. The host has about 955 GiB free disk and 15 GiB RAM.
- The fresh WSL environment has no system nvcc, CMake, Ninja, or G++ and no
  passwordless sudo. HTTPS is available. The operator cloned clean parent
  `beffcc883b1a595ef1861971d4ee19559145a6f1` with pinned llama.cpp
  `8d2b18a9c9b3a42927404a91799a823c758c09b6` and bootstrapped
  micromamba 2.9.0 under ignored `runs/toolchain-bootstrap` (archive SHA256
  `8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd`).
  CUDA 12.8/build tools are being resolved in user space. No models or SM75
  run results are staged yet.
- The precision audit found that Q4_0/Q8_0 GGUF formats alone do not specify
  CUDA activation arithmetic. It is tracing the actual small-batch EAGLE
  operators before assigning any INT4/INT8 execution label.

## Precision audit milestone

- The static [INT8/INT4 path audit](../../experiments/rtx2080ti-int4-int8-path.md)
  was integrated at parent `c425ca2`. At the frozen single-sequence decode
  shape, source control flow predicts MMVQ with one token column, subject to
  runtime source/output/layout gates. Q8_0 storage with live Q8_1 activation
  would use signed-byte `DP4A`; Q4_0 storage with Q8_1 activation would extract
  packed nibbles and compute with byte `DP4A`. Neither follows the prior
  whole-row/whole-token F32-scale W8A8/W4A4 simulation contract, and neither
  demonstrates INT4 Tensor Core execution. These are predictions until an
  actual SM75 dispatch trace is captured.
- The same precision owner has started a separate no-GPU native W8A8
  numerical/export/CPU stage in a new isolated worktree. It must retain the
  earlier signed-code and scale contract across all nine eligible linears.
  W4A4 remains a separate subsequent implementation gate. The 2080 Ti owner
  continues user-space toolchain setup, model staging, and the existing
  nine-variant SM75 suite independently.

## Export and harness milestone

- The W8A8 [export and CPU gate](../../experiments/w8a8-export-cpu-gate.md)
  was integrated as parent report `5c02865`. The published llama.cpp fork
  commit `bad469841` is deliberately not the parent gitlink yet. A full local
  EAGLE conversion produced all nine I8-code/F32-row-scale tensors with no
  dense shadows; a fresh BF16-source audit matched all arrays exactly. The
  GGUF SHA256 is
  `48d8c517253ee24278412efc18eaf38340d6e9eede4fc819f64ab268dab590d8`.
  Focused W8A8/W1A1 converter and CPU tests passed 7/7. This is storage and
  numerical evidence, not a loadable or timed runtime. The same feature owner
  now owns strict loader/graph and CPU operator work in its isolated submodule
  checkout, without GPU use or a parent gitlink update.
- Optional native W8A8/W4A4 benchmark rows were integrated in parent
  `97c85e0` and `64e3cb5`. The two formats can run independently or together
  with the existing anchors and five W1A1 groups. Each selected row requires
  a present GGUF and exact loader/CUDA operator marker; a missing marker leaves
  failure artifacts but no complete report. The integrated fake-server and
  analysis suite passed 25/25, and the temporary worktree and branch were
  removed after content verification. Real operand rows remain disabled until
  their runtime correctness gates pass.
- The remote user-space CUDA 12.8.93/GNU 13.4 toolchain configured an SM75
  production build. The first supervised configure attempt identified missing
  cuBLAS development libraries; the operator added them locally. The retry
  `sm75-production-build-retry-20260924` advanced through CUDA compilation
  and runtime linking and finished 368/368 targets, exit 0, with no GPU process.
  It produced `llama-server` and `test-backend-ops`; raw state/stdout are kept
  under that run directory. Model staging remains pending.
- The first real SM75 native W1A1 backend gate then passed 5/5 cases under
  supervised run `sm75-w1a1-op-gate-20260924`: K=31, 32, 33, 2560, and a
  strided K=33 input. The log explicitly showed CUDA packed W1A1
  XOR/POPCOUNT dispatch on the RTX 2080 Ti. The process exited zero and a
  fresh sample found the GPU idle. This validates the core operation, not the
  five EAGLE group graphs or end-to-end timing. The standalone binary-MMA
  probe and model-level checks are next.

## Binary MMA and W8A8 CPU milestones

- The standalone binary-MMA probe compiled on the real 2080 Ti using CUDA
  12.8. `cuobjdump` found six `BMMA.88128.XOR.POPC` static instruction sites;
  default runtime mode passed 21 cases and 880 exact integer dots against the
  dense CPU reference. The supervised `sm75-mma-probe-20260924` exited zero,
  its process group ended, and the GPU returned to its 855 MiB/0% Xwayland
  baseline. This proves the standalone probe, not integrated EAGLE dispatch or
  throughput. The operator is building the opt-in integrated candidate next.
- The opt-in integrated candidate branch `9bb01a6` built 358/358 SM75 targets
  and exited zero. Supervised runs
  `sm75-integrated-mma-portable-gate-20260924` and
  `sm75-integrated-mma-tensorcore-gate-20260924` each passed 8/8 backend
  cases and exited zero. Their respective logs show portable
  XOR/POPCOUNT and binary-MMA dispatch; the latter reported `cc=750` on a
  K=31, seven-row, three-token case. The GPU returned to its 855 MiB/0%
  Xwayland baseline after each run. A separate supervised SASS dump
  `sm75-integrated-mma-sass-20260924` exited zero: the integrated library
  (SHA256 `1985378050b10c13958dd3c2394cc28bcc940a03c0d1f49b98f6b84891e679f3`)
  contains seven static `BMMA.88128.XOR.POPC` sites. The seven-line extract
  SHA256 is `c272c95c3de53b7304e3db4bc9e84f325898ef24250fc98a27bfa32885727f82`;
  the preserved full SASS dump SHA256 is
  `9a4eb5027d197146337b5ba0ceea2655c276eba4d26701afc1ebc09a1c37c1da`.
  Model-level parity and timing remain unmeasured.
- Supervised download `fetch-pinned-models-20260924` is active. Its first
  Qwen target shard matched the local source manifest SHA256. It subsequently
  completed: all 18 pinned target/drafter source files matched local source
  hashes. The remote model manifest is
  `runs/fetch-pinned-models-20260924/model-manifest.json`, SHA256
  `e665b63bc4470f1a3d59015fe641eedb3e4ce824894c6ebae9d7eb0a7aa2d539`.
  GGUF conversion remains pending. The remote worktree for the integrated
  candidate is isolated from the production gitlink `8d2b18a`.
- The W8A8 [strict loader and CPU operator gate](../../experiments/w8a8-cpu-loader-gate.md)
  was integrated as parent report `2006582`; its published llama.cpp feature
  commit is `492818599` on top of exporter `bad469841`. The parent gitlink
  remains `8d2b18a` until CUDA integration is reviewed. CPU backend scalar
  oracle passed, a complete W8A8 GGUF loaded, malformed rounding/missing
  tensor cases were rejected, and an eight-token CPU speculative smoke
  generated 16 draft tokens. Ordinary and W1A1 smoke runs also passed. The
  feature owner now owns CUDA W8A8 source/tests without GPU use; the operator
  retains exclusive 2080 Ti access.
- W8A8 [CUDA source](../../experiments/w8a8-cuda-source-gate.md) was published
  at llama.cpp fork commit `a68968fad`; its parent report was integrated at
  `4075ae2` without a gitlink change. The source packs F32 activations into
  signed I8 once per token and uses signed byte-dot/I32 accumulation plus
  ordered F32 scales for one or multiple tokens. Expanded CPU-oracle tests
  passed K=8, 33, and 9,728. Mac lacks nvcc, so SM75 compilation, dispatch,
  SASS, and model-level parity are still required. The operator has the branch
  and gate instructions, after the integrated MMA check. A separate no-GPU
  worker owns W4A4 export/CPU format work in its own checkout; it does not
  touch GGML/CUDA or the 2080 Ti.
- The W4A4 [export and CPU gate](../../experiments/w4a4-export-cpu-gate.md)
  was integrated as parent report `c173233`. Published llama.cpp fork commit
  `f0cf0cf` adds an opt-in all-nine-linears format with two signed I4 codes
  per I8 storage byte, F32 row scales, explicit nibble order and odd-K padding.
  The full source audit matched all nine packed matrices and scales exactly;
  GGUF SHA256 is
  `0471dd2a1ac7628ae97018dad5d24aaf08cbfc6a258758a975d4e60b0d40beed`.
  Focused W4A4/W8A8/W1A1 converter and numerical tests passed 12/12. This
  format has no runtime loader or SM75 result yet. Its owner now proceeds to
  a separate strict loader/CPU operator gate, without GPU use or gitlink
  update.
- The W4A4 [strict loader and CPU operator gate](../../experiments/w4a4-cpu-loader-gate.md)
  was integrated as parent report `8b40f34`; published llama.cpp fork commit
  `edd3615` remains outside the parent gitlink. The complete all-nine GGUF
  loaded, malformed metadata and forbidden -8 nibbles were rejected, two
  numerical backend cases passed (including K=9 odd/strided and K=9,728),
  ordinary/W1A1/W8A8 model loads still passed, and a short CPU speculative
  request completed. It proposed 20 W4A4 draft tokens and accepted zero on
  that one prompt; this is an execution smoke, not a quality estimate. Its
  owner now works on W4A4 CUDA source without GPU use. A focused advisor is
  checking the SM75 signed INT4 Tensor Core layout; actual instruction,
  dispatch, and timing evidence remain pending.
- W4A4 [CUDA vector source and standalone signed-I4 MMA probe](../../experiments/w4a4-cuda-source-gate.md)
  were published at llama.cpp fork `3087139`, with source reports integrated
  at parent `0e741b7` and `7c5478c`. The vector source explicitly labels
  scalar integer multiply/add over packed nibbles, while the separate
  `m8n8k32` MMA probe has no production dispatch. A follow-up pins F32
  round-nearest quantization and scale order under CUDA fast-math; CPU oracle
  tests passed 2/2 with non-unit near-half cases. Neither CUDA path has
  compiled or run on SM75 yet.
- A combined [W8A8/W4A4 CUDA source branch](../../experiments/w8a8-w4a4-cuda-integration.md)
  was published at llama.cpp fork `b881a6346`, with parent note `59a4d90`.
  It retains distinct runtime operators and labels, allows one future binary
  to load both nine-linear formats, and passed CPU build, W8A8 3/3, W4A4 2/2,
  and both real GGUF load checks. The parent gitlink remains pinned to
  `8d2b18a`. The GPU operator will build and validate this exact branch in an
  isolated remote checkout after the existing nine-variant suite.

## Production model and link milestone

- On WSL, F16 target GGUF SHA256
  `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`
  and ordinary F16 draft GGUF SHA256
  `c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1`
  exactly match local artifacts. All five W1A1 group GGUFs were converted;
  each corresponding source-row audit passed with zero packed-word mismatches.
  The all-group audit `audit-w1a1-all-20260924` checked nine tensors and
  65,280 rows, with maximum F32 scale difference `7.45e-9` from source.
  Non-head GGUF bytes can vary slightly across hosts despite matching signs
  and this bounded scale difference; full host hashes are in the operator's
  experiment report.
- Q4_0 and Q8_0 controls were produced with the already built older standard
  quantizer and exactly match local SHA256 values
  `2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280`
  and `29ee91f09971555458cf420461fdfeeeabc9236f8a2c20662b7333a980a49507`.
  The original production `8d2b18a` shared-library `llama-quantize` failed
  before reading a GGUF due to a missing exported string-array loader symbol.
- Minimal published llama.cpp fix `34e21b7d85c17e25d5a91ce2ab1074d4c18bfe39`
  explicitly instantiates that loader template. A local shared build exported
  it; an incremental SM75 build (28/28) exported the Linux symbol. Supervised
  real Q4_0 conversion `quantizer-link-fix-q4-20260924` exited zero and
  reproduced the expected hash. The combined W8A8/W4A4 source branch includes
  the same fix at `d0724427b`, with local shared-build and both draft-load
  checks; SM75 validation of that branch is pending.
- A short fixed-binary four-path model smoke
  `sm75-integrated-mma-model-gpu-20260924` completed 20 requests over five
  repetitions: target-only, ordinary EAGLE, portable head W1A1, and MMA head
  W1A1 all loaded. Both packed CUDA dispatch flags were true. Each speculative
  variant's greedy text matched target-only 5/5 on this short suite; MMA and
  portable matched 5/5. API token IDs were unavailable. Loaded memory samples
  were 9,074 / 10,110 / 9,962 / 9,962 MiB respectively on the 11,264 MiB
  card; minimum observed free was 1,154 MiB. All servers stopped and GPU
  returned to 855 MiB/0%. Raw manifest/report/records SHA256 values are
  `416e40a136c42d496bd909e75ae9d30510d3df7b2bc6b35490f49891221ed573`,
  `3517a7f6c1661ccbdc9668e0631093c4127027da3dffcd14773d31488fa176b7`,
  and `08732c541ffcd7911b203fc333fdec2f8f09703f50dce86e3adebdc32df212a2`.
  This establishes FP16 target fit and short model-level dispatch/parity, not
  the frozen 12-prompt throughput comparison.
- The fixed llama.cpp commit was published to the user's fork before this
  parent gitlink update. Next: pin it in `main`, run the production five-W1A1
  plus Q4/Q8 model smokes and matched benchmark, then test the combined native
  INT branch in a separate isolated build.
