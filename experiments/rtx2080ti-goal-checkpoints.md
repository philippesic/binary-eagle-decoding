# Goal: benchmark the RTX 2080 Ti quantization suite

This file archives the long chronological goal checkpoints through the main
benchmark and Q4_0/Q8_0 trace. It preserves exact intermediate commits,
owners, runs, and decisions. The concise completed goal file is
[here](../docs/goals/rtx2080ti-quantization-suite.md), and the final result is
in the [synthesis](rtx2080ti-synthesis.md) and
[experiment provenance](rtx2080ti-quantization-suite.md).

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

## Nine-variant model smoke milestone

- Parent `main` commit `f99affa` now pins the WSL-validated llama.cpp fix
  `34e21b7`; the fork's `w1a1-integrated` branch was advanced to that commit
  before the temporary fix branch/worktree were removed.
- Both Q-format controls have the expected GGUF inventory: Q4_0 has nine
  Q4_0 linears, Q8_0 has nine Q8_0 linears, and each retains four F32 other
  tensors plus one I64 mapping.
- Supervised production smoke `sm75-native-nine-variant-smoke-20260924`
  completed 45 measured requests (one held-out prompt × five repetitions)
  after two warmups/server. Target-only, ordinary EAGLE, fusion/attention/FFN/
  head/all W1A1, and Q4_0/Q8_0 all loaded and generated. Each W1A1 group
  showed its expected loader marker and packed CUDA XOR/POPCOUNT dispatch.
  Every candidate's greedy text matched target-only 5/5 in this short check;
  the API did not return token IDs. GPU processes exited and the host returned
  idle. This is a model/dispatch gate, not the frozen 12-prompt timing result.
- Q4_0 and Q8_0 server logs show stored GGUF types but not selected kernel
  names. At decode batch one, the pinned GGML source predicts Q8_1 activation
  quantization and MMVQ/byte `DP4A`; prefill may take MMQ. Keep those labels
  explicitly source-inferred until a kernel trace confirms actual execution.
  They do not implement the prior per-row/per-token W4A4/W8A8 contract.
- The operator is starting the full 12-prompt, five-repetition paired matrix
  with one GPU/server owner. Raw smoke hashes and final run IDs will be added
  to the experiment report and the next checkpoint.

## Full run in progress

- The production matrix is supervised as
  `native-nine-quantization-full-supervisor-20260924`, writing ignored raw
  artifacts to `results/native-nine-quantization-full-run-20260924/` on the
  2080 Ti host. It pins parent `f99affa`/llama.cpp `34e21b7`, 12 frozen
  prompts, two warmups/server, five alternating repetitions, 128 output-token
  cap, and one server at a time. Preflight was 855 MiB used / 0% GPU use.
  Repetition 0 target-only and ordinary EAGLE each completed all 12 prompts;
  ordinary loaded at about 10,108 MiB used with 920 MiB free. The run remains
  active; no throughput conclusion is available yet.
- A separate opt-in Q4_0/Q8_0 dispatch trace source was published on
  llama.cpp branch `feat/qformat-cuda-dispatch-trace` at
  `5c52067`. With `GGML_CUDA_QFORMAT_DISPATCH_TRACE=1`, it will log bounded
  actual MMVQ/MMQ/cuBLAS branch, activation format, and shapes. It is disabled
  by default and has not been compiled on CUDA or run during timed requests.
  A short isolated post-suite check will resolve the source-inferred labels.

## Sealed nine-variant result

- The supervised full matrix `native-nine-quantization-full-supervisor-20260924`
  finished exit 0. Its ignored raw directory is
  `/home/philip/binary-eagle-decoding/results/native-nine-quantization-full-run-20260924/`.
  It pinned parent `f99affa`/llama.cpp `34e21b7`, the F16 target/KV and
  2,048 context, 12 prompts, five alternating repetitions, two warmups/server,
  and one server at a time. All 540/540 measured records are complete, 60 per
  variant; all five W1A1 CUDA dispatch flags were true. The GPU returned to
  its 855 MiB/0% desktop baseline and no project process remained. The largest
  live memory reading was 10,160 MiB used / 868 MiB free.
- The compact [experiment report](../../experiments/rtx2080ti-quantization-suite.md)
  was committed/pushed at parent `d871064`. Raw manifest/report/records and
  analysis SHA256 values are respectively
  `8fe6358677f4891b71e78f0f6be7a23211979764a24ac6d915a8711d09333fd9`,
  `54e4ba42f4130856000db25ce36cf1b93026bc66a76dc839339e9b5d0114d13f`,
  `04c4a1a71244f8b8f18077e5d0a192455d36625d780c990a1742da6628507889`,
  and `9e1a8b5b2f48394b927e6825927edb8f86d58bf2908eec52af8777f9d20d14a1`.
  Its paired analysis used 2,000 prompt/repetition bootstrap resamples.

| Variant | Request tok/s | Decode tok/s | Decode ratio vs ordinary | Accepted drafts/round |
| --- | ---: | ---: | ---: | ---: |
| Target only | 59.79 | 61.67 | 0.748 | — |
| Ordinary EAGLE | 77.35 | 82.49 | 1.000 | 1.168 |
| Q4_0 draft | 85.20 | 91.51 | 1.109 | 1.182 |
| Q8_0 draft | 81.95 | 87.73 | 1.064 | 1.168 |
| W1A1 fusion | 47.44 | 49.32 | 0.598 | 0.271 |
| W1A1 attention | 47.31 | 49.17 | 0.596 | 0.238 |
| W1A1 FFN | 56.32 | 59.05 | 0.716 | 0.472 |
| W1A1 head | 70.51 | 74.81 | 0.907 | 0.860 |
| W1A1 all | 44.65 | 46.31 | 0.561 | 0.055 |

- Every speculative variant produced identical completion text to the others
  on all 60 prompt/repetition pairs. Each differed from target-only only on
  `reasoning-02`, consistently in all five repetitions, by a capitalization/
  formatting choice beginning at character 440. Target-only output was stable
  across repetitions. Request bodies, seed 42, greedy sampling, and output cap
  matched; the server omitted generated token IDs. Compare speculative variants
  directly; label target-only ratios as timing observations rather than strict
  lossless speedups. Client TTFT is unavailable because responses were not
  streamed, while server decode durations were positive for all records.
- Q4_0 and Q8_0 are verified GGUF weight formats and showed 1.109× and
  1.064× ordinary decode throughput respectively (paired 95% intervals
  1.081–1.135 and 1.053–1.075). Their exact CUDA activation/operator path is
  still source-inferred. All five native W1A1 scopes lost to ordinary; head
  was 0.907× (0.869–0.946) and all-group 0.561× (0.511–0.617). The acceptance
  collapse, especially all-group's 0.055 versus ordinary's 1.168 accepted
  drafts/round, is the primary measured reason to investigate before any
  speedup claim. The server's host draft span per verification round fell
  from 6.471 ms ordinary to 3.558 ms all-group, but verification rounds
  rose 3,400→6,940 and pooled draft time rose 22.0→24.7 s. The saved
  `accept_ms` counter covers an acceptance hook, not an isolated target
  verifier span; do not infer missing GPU kernel/verification durations.
- Next: run a full paired portable-versus-integrated-binary-MMA head matrix
  on this device; then build/check the combined W8A8/W4A4 branch and use the
  opt-in Q-format trace after sealed timing. Keep the one-GPU-owner rule.

## Sealed binary-MMA head comparison

- A separate supervised four-path run
  `native-mma-head-full-supervisor-20260925` completed 240/240 measured
  requests (12 prompts × five repetitions × target-only, ordinary EAGLE,
  portable W1A1 head, integrated binary-MMA W1A1 head). It used the same F16
  target, ordinary draft, frozen packed-head GGUF, context and prompts as the
  production matrix. The isolated candidate llama.cpp revision was `9bb01a6`;
  the runtime binary SHA256 was
  `365edec352eb3cc57fa7649317ea3354410363a475afdeced0ea345d748355cf`.
  Each portable request recorded selector `0`, each MMA request selector `1`,
  and both CUDA dispatch markers were confirmed in all five repetitions.
- Pooled request/decode rates were portable **70.388/74.591** tok/s and MMA
  **70.523/74.720** tok/s. MMA/portable ratios were 1.0019 request and 1.0017
  decode, with paired 95% bootstrap intervals 0.9966–1.0068 and
  0.9973–1.0062. This experiment resolved no end-to-end MMA gain. Both paths
  accepted the same 3,405/19,480 draft tokens, used 3,960 rounds, and emitted
  identical text on all 60 paired requests. Each was about 0.91× ordinary
  EAGLE decode rate on this candidate binary. Target-only differed by the
  same stable `reasoning-02` formatting choice, so its ratios remain timing
  observations.
- The raw directory is
  `/home/philip/binary-eagle-decoding/results/native-mma-head-full-run-20260925/`.
  Manifest/report/records/analysis SHA256 values are
  `ca7f6647b1731fcac8b6d6494cf3a7d0d348fb8e34bed7b2be17ce9cdb48b8a1`,
  `0ee964ea70cc76ab7a479eea6291334bc783af370b36c8d07dabe1f12afba665`,
  `ad53f8f9875b786117894e1dd54a637d7bea04976aead6609631a5746ef9ce3f`,
  and `9a996f457fa5864166e920dd08f185596b80da75e1009585fc1f8d2b9ad229c2`.
  GPU returned to 855 MiB/0% with no project process. Full commands and
  variance are in the [experiment report](../../experiments/rtx2080ti-quantization-suite.md).
- Next: the exclusive operator is building combined W8A8/W4A4 source
  `d0724427b` in an isolated SM75 checkout, then will run exact backend
  oracles, the standalone signed-I4 MMA probe, model dispatch/parity, and
  only then optional paired timing. The parent gitlink remains the validated
  production W1A1 revision `34e21b7` until the low-bit branch passes.

## Native INT8/INT4 SM75 arithmetic milestone

- The isolated combined llama.cpp revision `d0724427b` compiled for SM75
  (360/360 targets, exit 0) with its own `llama-server` and backend tests.
  The production parent gitlink remains `34e21b7`.
- On the actual RTX 2080 Ti, the W8A8 CUDA backend passed 3/3 independent
  oracle cases (K=8, 33, and 9,728) with explicit signed INT8 dot/I32
  dispatch. W4A4 passed 2/2 (K=9 and 9,728) with an explicit packed signed-I4
  vector marker. This W4 production path uses scalar integer multiply/add,
  so it is not labeled INT4 Tensor Core execution.
- The separate signed-I4 `m8n8k32` Tensor Core probe passed all six SM75
  cases, 308 exact I32 outputs, covering basis/random tiles, N=1/N>1, and
  K=9/33/9,728. Its executed SASS contains `IMMA.8832.S4.S4.SAT`. This proves
  the standalone instruction/layout, not an integrated EAGLE speedup.
- The operator is converting W8A8/W4A4 draft GGUFs on WSL from pinned BF16
  source and auditing every code/scale tensor before model-level CUDA checks.
  Independent no-GPU owners are preparing opt-in production W4A4 and W8A8
  Tensor Core candidates; each must pass its own real SM75 backend/model
  dispatch and paired timing before any live Tensor Core benefit is claimed.

## Native INT8/INT4 artifact milestone

- Supervised WSL conversions `convert-w8a8-d0724427b-20260925` and
  `convert-w4a4-d0724427b-20260925` produced the full nine-linear draft
  GGUFs: W8A8 224,730,560 bytes, SHA256
  `48d8c517253ee24278412efc18eaf38340d6e9eede4fc819f64ab268dab590d8`;
  W4A4 115,613,408 bytes, SHA256
  `0471dd2a1ac7628ae97018dad5d24aaf08cbfc6a258758a975d4e60b0d40beed`.
  Both exactly match their earlier local CPU-gate artifacts.
- Supervised source audits `audit-w8a8-source-d0724427b-20260925` and
  `audit-w4a4-source-d0724427b-20260925` checked every code and F32 scale
  byte for all nine linears against the pinned BF16 safetensors after Q/K
  permutation: zero code mismatches and zero scale-byte mismatches. Each GGUF
  has 23 expected tensors, strict metadata/pair names and shapes, and no
  dense shadow. The next gate is short all-nine CUDA model dispatch and
  text/acceptance validation, before any low-bit timed row.
- A separate opt-in live W4A4 SM75 Tensor Core candidate is published at
  llama.cpp fork `eaa7fb1`, based on the combined vector branch. Local CPU
  W4A4 backend cases passed 4/4; no production CUDA compile/run yet. A
  separate W8A8 signed-INT8 MMA candidate is published at `fbaa9e2`, with
  local CPU W8A8 cases 5/5; no SM75 probe/production run yet. Both retain the
  validated default vector/DP4A paths and require exact instruction/model
  evidence before an end-to-end hardware claim.

## Native low-bit model smoke milestone

- A short one-prompt, five-repetition validation with the isolated combined
  `d0724427b` server completed 25/25 requests over target-only, ordinary
  EAGLE, portable W1A1 head, default W8A8, and default W4A4. Both W8A8/W4A4
  model paths logged strict all-nine loader markers and their distinct CUDA
  vector/dot dispatch markers; no CPU fallback or load failure was recorded.
  All five paths matched target completion text 5/5 on `prose-01`; generated
  token IDs were not returned. GPU returned to 855 MiB/0% with no project
  process. Raw results are under ignored remote
  `results/native-lowbit-model-smoke-d0724427b-20260925/`; supervisor
  `sm75-eagle-int-lowbit-model-smoke-d0724427b-20260925` exited zero.
  Manifest/report/records/prompt SHA256 values are
  `e14a98240c09a7013dc1e8a9f6b92b57f790c6f560b4585d481c2c05ba898694`,
  `d8cbc1a2b1a212c4c3ebdcb77a324e18ee2082768521d39c6c89cd081c985183`,
  `a1d772d94f73b33177885dd823d676346184f59724137aefa3d0775d3d542693`,
  and `ddb4868a2717c813a852713a7c455c35438d784ed6e334c70b2ae2b93af68164`.
  The largest loaded GPU snapshot was 10,110 MiB.
- In this one-prompt smoke, ordinary EAGLE and W8A8 each accepted 230/1,960
  draft tokens, or 0.575 per verification round; W4A4 accepted 45/2,875,
  or 0.077/round; packed-head W1A1 accepted 180/2,200, or 0.400/round.
  These are a small functional/quality screen, not the frozen suite's
  acceptance conclusion or timing result. The operator is starting the
  12-prompt, five-repetition vector-path matrix to quantify both formats.

## Sealed W8A8/W4A4 vector result

- The isolated combined low-bit server at llama.cpp `d0724427b` completed
  300/300 measured requests: target-only, ordinary EAGLE, portable W1A1 head,
  genuine all-nine W8A8 default signed-INT8 dot, and genuine all-nine W4A4
  default packed-nibble vector, each on 12 prompts × five repetitions. Both
  MMA selectors were off. W8A8/W4A4 loader and CUDA dispatch gates held in
  every repetition. The committed parent gitlink remained production
  `34e21b7`; the temporary checkout used for manifest provenance was restored
  to it after the run. The GPU returned idle with no server process.

| Variant | Request tok/s | Decode tok/s | Decode ratio vs ordinary | Accepted drafts/round |
| --- | ---: | ---: | ---: | ---: |
| Target only | 58.67 | 60.60 | 0.748 | — |
| Ordinary EAGLE | 76.01 | 80.99 | 1.000 | 1.168 |
| Portable W1A1 head | 69.40 | 73.58 | 0.909 | — |
| Native W8A8 vector | 75.14 | 80.18 | 0.990 | 1.127 |
| Native W4A4 vector | 40.05 | 41.40 | 0.511 | 0.092 |

- W8A8/ordinary request and decode ratios were 0.989 and 0.990, with paired
  95% intervals 0.974–1.003 and 0.975–1.005: no rate difference was resolved.
  W8A8 accepted 3,905/17,055 proposed drafts (22.90%) versus ordinary's
  3,970/16,740 (23.72%). W4A4 accepted only 620/33,055 (1.88%); request and
  decode ratios were 0.527 and 0.511, with paired intervals 0.484–0.575 and
  0.466–0.561. Its quality loss dominates the default vector result.
- All four speculative paths produced identical completion text on all 60
  paired requests. Each differed from target-only only on the established
  stable `reasoning-02` formatting prompt. Target-only ratios remain timing
  observations. The complete [experiment report](../../experiments/rtx2080ti-quantization-suite.md)
  was committed/pushed at `8fc61d1`. Its raw run under
  `/home/philip/binary-eagle-decoding/results/native-lowbit-vector-five-rep-d0724427b-20260925/`
  has manifest/report/records/analysis SHA256 values
  `4dd772b7efd3bcc113528f51502e89fa763f362c4fb01b1bae7569091eb2051f`,
  `72eb46d17da9d67cdc5f0211d3184eb53653cb66cde231b6323f18156aec1f27`,
  `39bb90a4c105de4056a8104bd592cfd61524982aa6cf5f6c28d4392a0243eb12`,
  and `08302212b00d2f9e3ced34645f5d29a722c0a7feab5fa8ee6d881212b5035208`.
  Largest live use was 10,160 MiB / 868 MiB free; the GPU returned to its
  855 MiB/0% baseline after cleanup.
- Host draft-call time per verification round was 6.617 ms ordinary,
  6.329 ms W8A8, and 6.661 ms W4A4. W8A8's small draft-cost reduction was
  offset by its acceptance/round difference. W4A4 neither reduced this
  per-round cost nor retained acceptance; rounds rose 3,400→6,710. The logged
  acceptance-hook span does not measure total verifier latency.
- Next: build/check the separate combined opt-in INT8/INT4 Tensor Core source
  `2d9712cde` on SM75, first exact probe/backend/model parity and then paired
  timing if those pass. Trace actual Q4_0/Q8_0 CUDA operator paths afterward.

## Opt-in INT8/INT4 Tensor Core arithmetic milestone

- The isolated combined opt-in branch `2d9712cde` built 360/360 SM75 targets
  under a supervised WSL run; parent production gitlink remains `34e21b7`.
  W8A8 default signed-byte DP4A and selector-enabled signed-INT8 MMA each
  passed 5/5 CUDA backend cases with distinct markers. W4A4 default packed
  vector and selector-enabled signed-I4 MMA each passed 4/4, also with
  distinct markers. Every gate returned the GPU to the 855 MiB/0% baseline.
- Standalone W8A8 MMA probe passed 63/63 exact I32 dot cases on SM75 and its
  SASS contains `IMMA.8816.S8.S8`. The W4A4 probe passed six cases/308 exact
  I32 outputs and contains `IMMA.8832.S4.S4.SAT`. Targeted SASS extracts of
  the **built live candidate library** confirm the W8A8 MMA function contains
  `IMMA.8816.S8.S8` (excerpt SHA256
  `2d9c549f1df31c39122891a70f36cee5bed076f98c8d63c9df10b55217c1dfe1`)
  and the W4A4 MMA function contains `IMMA.8832.S4.S4.SAT` (excerpt SHA256
  `0b83964bb3d96ee47df168b3fbd35c75fe8967bee3db0ff17d4038a3c9999782`).
  These prove numerical operator and instruction evidence, not real EAGLE
  parity or end-to-end speed.
- Next: short same-GGUF, all-nine model smoke with selector 0 versus 1 for
  each format; require matching text/acceptance and actual dispatch before
  using the integrated optional MMA benchmark rows. A full-library SASS dump
  was stopped after exceeding 1 GiB; targeted function extracts are retained.

## Opt-in Tensor Core model gate and full run

- A one-prompt, five-repetition, seven-path model smoke completed 35/35
  requests on the combined opt-in candidate `2d9712cde`. Target-only,
  ordinary EAGLE, W1A1 head, W8A8 default/MMA, and W4A4 default/MMA loaded;
  all intended CUDA markers appeared and opposite-path markers were absent.
  The W8 pair emitted identical text and identical 46/392 accepted/proposed
  counts per repeated request; the W4 pair likewise matched text and 9/575
  counts. Every path matched target text 5/5 on `prose-01`. This is a model
  parity gate, not a representative rate or quality result. The GPU returned
  to 855 MiB/0% after the smoke.
- The full seven-path paired comparison uses the 12
  frozen prompts, five repetitions, two warmups/server, one server at a time,
  and the same audited GGUF for each default/MMA pair. Its resolved config
  SHA256 is `9084a6e0059da4ac478a03c5096680f34102b712a67749f0b4a27bed4140f6e5`;
  candidate server binary SHA256 is
  `07bb2339b000ba63f71cb4c5c7b74304a0816e466b5796d40f7e09712ff33b5a`.
  The actual candidate checkout is `2d9712cde8d7808bb869e59e56a565e0e4fa2918`;
  the parent committed gitlink remains `34e21b7`. Preflight found no project
  process and 855 MiB GPU memory used/0% utilization. The first supervisor
  `native-lowbit-mma-full-supervisor-2d9712cde-20260925` exited 1 before any
  GPU work because its run ID reused the successful dry-run output directory;
  that dry-run artifact is preserved. The timed run was relaunched cleanly
  as supervisor `native-lowbit-mma-full-supervisor-timed-2d9712cde-20260925`
  with fresh result ID `native-lowbit-mma-full-timed-2d9712cde-20260925`.
  It is active; do not interpret partial rates.

## Sealed live INT8/INT4 Tensor Core result

- The full seven-path run completed 420/420 requests, 12 prompts × five
  repetitions, with supervisor exit 0 and no error rows. Both MMA selectors
  and CUDA dispatch markers were confirmed; forbidden default-path markers
  were absent in candidate server logs. The GPU returned to 855 MiB/0% and
  the process group ended. The operator is completing artifact hashes and
  restoring the working submodule to the production `34e21b7` gitlink.
- W8A8 default DP4A versus opt-in signed-INT8 MMA emitted identical text on
  all 60 paired requests and identical 3,905/17,055 accepted/proposed drafts
  over 3,465 rounds. W8A8 MMA/default request and decode ratios were
  **0.8302** and **0.8205**; paired 95% intervals were 0.8255–0.8356 and
  0.8172–0.8236. The specialized INT8 instruction was slower end-to-end.
- W4A4 default packed-nibble vector versus opt-in signed-I4 MMA likewise
  matched text 60/60 and had identical 620/33,055 accepted/proposed drafts
  over 6,710 rounds. W4A4 MMA/default request and decode ratios were
  **0.8844** and **0.8801**, with paired intervals 0.8801–0.8888 and
  0.8759–0.8844. The specialized INT4 instruction also lost end-to-end.
- Each speculative path matched target-only text 55/60, with the same stable
  `reasoning-02` heading-format difference; generated token IDs were absent.
  The paired MMA/default conclusions are clean because draft weights,
  accepted counts, and emitted text are equal within each format. These are
  measurements of this single-sequence EAGLE workload, not general Tensor
  Core throughput claims. The detailed [experiment report](../../experiments/rtx2080ti-quantization-suite.md)
  was committed/pushed at `37197fe`.
- Absolute decode rates were W8A8 default 80.301 versus MMA 65.889 tok/s,
  and W4A4 default 41.502 versus MMA 36.526. Host draft-call time per
  verification round rose from 6.352→12.054 ms for W8A8 and 6.638→10.189 ms
  for W4A4. This explains the cost loss while accepted tokens and rounds were
  unchanged. These spans are host draft-call time, not isolated GPU kernel or
  total target-verifier time.
- Raw results are at
  `/home/philip/binary-eagle-decoding/results/native-lowbit-mma-full-timed-2d9712cde-20260925/`.
  Manifest/report/records/analysis SHA256 values are
  `c1accfcc98eafb63230470beea84d149fee9bce8ed1708850de43ca4eb224779`,
  `e51ce493b519aa5e6e6c37c23eb86aad836f98491015aa2b0ec97fd2b42a825d`,
  `c254f858db9273054f71a24d3198dbcf0454f3da3792f4e1cb735df9198152e9`,
  and `92e77f518c32bfa575b5946306da1690e775d8c28beb1dd21fad8232c12eb632`.
  The highest live GPU sample was 10,160 MiB used / 868 MiB free; final GPU
  was 855 MiB/0% and the production checkout/gitlink was restored cleanly.
- A separate short Q4_0/Q8_0 trace is complete below. The remaining action
  is to finish the consolidated result, verify clean/published project state,
  and close the goal only when that report is committed.

## Q4_0/Q8_0 executed-path trace

- An isolated SM75 server build from published llama.cpp diagnostic branch
  `5c52067` completed 356/356 targets. The trace-only server SHA256 was
  `0fd2d394a5136526c0e5fce96e0d704a27d53a3445d0d0135386034c02425199`.
  Its short supervised run `native-qformat-dispatch-trace-supervisor-5c52067-20260925`
  completed 25/25 measured requests with one prompt, one warmup, five
  repetitions, and 32-token cap; GPU returned to 855 MiB/0% with no server.
- Actual CUDA trace lines for **both** Q4_0 and Q8_0 recorded F32 input
  activations quantized internally to **Q8_1**. One-token fused decode
  `(K=4096,M=2560,N=1)` and two-token work `(K=5120,M=4096,N=2)` selected
  **MMVQ**; an observed 38-token operation `(K=7680,M=2560,N=38)` selected
  **MMQ**. No cuBLAS branch appeared in this diagnostic. The selected MMVQ
  source uses byte `DP4A`, extracting Q4_0 nibbles for the Q4_0×Q8_1 path;
  this instruction property is source-backed rather than disassembled in the
  diagnostic. These are block-scale mixed-format operators, not the separate
  whole-row/whole-token W4A4/W8A8 contracts.
- The exact run commands, six branch/shape records, cleanup, and raw hashes
  are in the [trace section](../../experiments/rtx2080ti-quantization-suite.md)
  committed at `2b5c192`. Raw manifest/report/records SHA256 values are
  `28c40fcf79cb570b349bab254e0b07f56db0672b513202a514fbac98952fc3d7`,
  `56d28fe2323d5f14747911c1c8a59aa7a84faaa5c0e976129a3ec7622698c411`,
  and `ca9566b44c74fa9cece3ca84122a0029a02cad88f8a691b46aa64c5bbb887f90`.
