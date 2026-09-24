# RTX 2080 Ti execution gate

**Access state:** the user supplied the current RTX 2080 Ti address and it is
stored only in the shared local `hosts.toml`; the username is `philip`.
Ubuntu 24.04 WSL2 and SSH became reachable on 2026-09-24. The fresh host lacks
system CUDA and build tools, so a user-space toolchain is being prepared before
the SM75 correctness gate.
All SSH to this WSL GPU must use tmux MCP and all experiments must use
`scripts/remote_job.py` with unique run IDs.

The [active benchmark goal](goals/rtx2080ti-quantization-suite.md) and
[evaluation protocol](EVALUATION.md) govern this gate. The 5080
[native end-to-end comparison](../experiments/native-end-to-end-5080.md)
found head-only packed W1A1 slower than ordinary EAGLE, but 5080 behavior
cannot establish SM75 correctness, utilization, or speed. A standalone
[binary-MMA probe](../experiments/sm75-binary-mma-plan.md) cross-compiled to
SM75 and passed numerical checks only on an SM120 proxy path.

## Access and preflight

1. Register the user-supplied IP with `python3 scripts/agent_env.py set-host
   rtx2080ti <ip> philip --port 22`; verify the pause flag and recorded
   workdir. Use tmux MCP for the SSH session. Check actual GPU name, compute
   capability 7.5, VRAM, driver, toolkit, compiler, glibc, free disk, and
   other active processes. Give the 2080 Ti one experiment owner.
2. Fetch pinned parent `main` and its llama.cpp submodule. Hash the exact
   source, model snapshots, config and prompts. Start every compile, test,
   conversion, and inference under a supervised unique run directory. Keep
   raw artifacts outside Git.
3. Use this host's own toolkit and headers. The 5080 needed a private CUDA
   13.1/glibc 2.43 header patch; apply it only if the 2080 Ti has the same
   demonstrated conflict. Never modify the system CUDA header in place.

## Correctness before timing

1. Build llama.cpp CUDA for `sm_75` with backend tests enabled, recording
   CMake cache, nvcc command/version and binary hashes. Run
   `test-backend-ops test -b CUDA0 -o W1A1_MUL_MAT`; require all five
   scalar-reference cases to pass.
2. Compile the standalone `kernels/sm75_mma_probe.cu` for SM75, inspect
   `cuobjdump` for `BMMA.88128.XOR.POPC`, and execute the **default** mode on
   the real 2080 Ti. Require all 21 cases/880 exact integer dots to pass.
   The 5080 proxy pass is not a substitute. If this fails, stop the MMA claim
   and keep the portable `__popc` path separate.
   The opt-in [integrated MMA branch](../experiments/integrated-binary-mma-5080.md)
   is published as `feat/w1a1-sm75-mma` at
   `9bb01a682ed4ba5e506870c8a38338589830b164`, based on production
   `92bc706`. Build its SM75 CUDA backend in a separate tree, verify SASS
   contains `BMMA.88128.XOR.POPC`, and run the expanded
   `test-backend-ops test -b CUDA0 -o W1A1_MUL_MAT` suite with the default
   portable path and then `GGML_CUDA_W1A1_MMA=1`. Require 8/8 scalar-reference
   cases and distinct dispatch logs for each, plus a model-level generated-ID
   parity smoke. The 5080 proxy passed these checks; SM75 runtime remains
   untested. Keep this branch separate from the pinned production gitlink
   until real 2080 Ti correctness and timing are reviewed.
3. Convert the pinned target and ordinary/packed EAGLE draft on this host,
   auditing all 32,000 packed head rows and hashing the three GGUFs. A short
   native server smoke must show the packed-head loader and explicit CUDA
   XOR/POPCOUNT dispatch before a timed run.

## Matched hardware comparison

The user requested all five W1A1 coverage settings previously evaluated in
PyTorch: fusion-only (one linear), attention-only (Q/K/V/O), FFN-only
(gate/up/down), head-only, and all nine linears. Test these as **native packed
execution** on the 2080 Ti, with ordinary FP16 EAGLE and target-only anchors,
the same target and evaluation settings, and explicit per-group dispatch and
precision coverage. The 5080-tested runtime at `92bc706` supported only
head-only; expanded revision `8d2b18a` passed local conversion, exhaustive
packed-row audit, and CPU graph generation for all groups. Do not report the
other four as **CUDA** results until their actual SM75 backend correctness,
model-level dispatch, and timing are verified. Treat the existing BF16
PyTorch counts as quality screening, not SM75 timing.

Revisit QAT as a separate, versioned experiment after establishing the
post-training matrix. The previous head-only 500-step teacher-KL reconstruction
improved validation KL but lowered held-out acceptance. Choose a new objective
and a new untouched held-out prompt set before training; record which drafter
groups change and compare each trained path with its untrained counterpart.
Do not tune on the original 12 held-out prompts.

The 2080 Ti has a different memory budget. Try the same FP16 target/draft,
context and KV settings as the 5080 comparison only after checking actual
free VRAM. If they do not fit, define a **separate quantized-target track**
and use the identical target GGUF, KV precision, placement and context for
target-only, ordinary EAGLE and packed-head W1A1. Record conversion command,
target quantization and resulting hash; do not compare its throughput ratio
against the 5080 FP16 track.

Run the paired harness with the frozen 12 prompt IDs, greedy settings, two
warmups/server, at least five alternating repetitions, and one variant/server
at a time. Preserve per-request wall/prefill/decode time, completion lengths,
accepted/proposed/round counts, draft-time stats, GPU memory snapshots,
text/token parity evidence, and explicit W1A1 CUDA dispatch. Analyze pooled
token/time ratios and paired prompt/repetition spread. Profile packing and
the portable binary kernel separately; do not add overlapping GPU spans as
though serial. A negative result is valid if its bottleneck and limitations
are quantified. If the portable path is negative, only claim that path is
negative. Compare the opt-in integrated MMA path against portable under the
same target, draft weights, prompts, server binary, and settings before ruling
out Turing binary Tensor Cores. Run both paths with five alternating measured
repetitions if the MMA correctness gates pass; identify the selector in every
raw record. Do not compare the 2080 Ti speed ratios to the 5080 track.
The paired runner supports this directly: use a run-specific copy of
`configs/native_benchmark.toml` with `binary_mma = true` under `[evaluation]`
and the candidate branch's `llama-server` binary. This selects target-only,
ordinary EAGLE, portable packed-head W1A1, and MMA packed-head W1A1 in
balanced order. It sets `GGML_CUDA_W1A1_MMA=0` for the first three server
processes and `1` for MMA, even if an inherited/configured value differs;
the selector is saved with every raw record and per-server environment. Require
`native_cuda_dispatch_confirmed` and `mma_cuda_dispatch_confirmed` in
`report.json`, inspect `greedy_text_match_mma_vs_portable`, then run
`scripts/analyze_native_benchmark.py` on the completed run to obtain pooled
MMA/portable ratios and paired bootstrap intervals. With the flag absent,
the existing three-variant behavior remains the default.
The run manifest records both the parent gitlink and the actual llama.cpp
checkout commit, status, and diff hash. For a candidate-branch trial, confirm
the actual checkout is `9bb01a6` while the pinned production gitlink remains
`92bc706`; preserve that distinction when reporting binary provenance.

For the five W1A1 groups plus Q4_0/Q8_0 weight-only draft comparison, use a
run-specific copy of `configs/native_benchmark_five_w1a1_groups.toml`. Its
opt-in matrix has target-only, ordinary FP16, five packed W1A1 group drafts,
and two weight-only drafts. The harness requires each packed variant's exact
loader group marker and CUDA W1A1 dispatch marker, preserves generated token
IDs where the server returns them, and records pooled ratios versus both
anchors. The environment manifest captures GPU/driver/toolkit/compiler/build
information before timed requests, with per-server GPU samples. Resolve the
Q4_0/Q8_0 activation and operator precision labels from actual 2080 Ti
backend evidence; do not call them W4A4/W8A8. A local nine-variant dry-run
validated model paths and hashes, command assembly, and the parent/submodule
commit record; it is not a runtime result.
The current activation packer accumulates magnitudes in F64 before writing
an F32 scale; measure that reduction separately on Turing rather than
carrying over the 5080 pack cost. The 5080 integrated-kernel NCU attempt was
denied GPU performance counters, so its standalone CUDA-event result is the
available component baseline, not an integrated per-kernel profile.

After each run, stop the supervised process group, check its state and GPU
process list, record final memory/utilization, close SSH/tmux, and checkpoint
the goal/status with exact commit, artifact paths, and hashes.
