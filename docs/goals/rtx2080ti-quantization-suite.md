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
