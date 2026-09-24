# W4A4 CUDA source and SM75 MMA probe gate

**Status:** packed signed-I4 CUDA vector path and an isolated SM75 signed-I4 MMA correctness probe are published as source on 2026-09-24. Neither was compiled or executed on CUDA because this local Apple M3 Max host has no `nvcc` or NVIDIA device. No SM75 correctness, Tensor Core dispatch, or speed claim is made.

**Code:** llama.cpp fork branch `feat/w4a4-eagle-export`, commit `e98ef51f34faa882ef33f0ab2aaa7dff6fe754fd`, atop the [W4A4 loader and CPU gate](w4a4-cpu-loader-gate.md) commit `edd3615561519f7b1537b4b215d5c58eb822e7f5`. The parent gitlink is unchanged by this report.

## Production CUDA path in source

The `GGML_OP_W4A4_MUL_MAT` CUDA dispatch uses a separate activation-pack kernel and a signed-nibble vector-dot kernel. Packing computes one F32 absmax/7 scale per token, divides in F32, rounds to nearest even, clamps to `[-7,7]`, stores two signed nibbles per byte, zeroes a zero token, and handles strided token rows and odd-K zero padding. A nonfinite activation traps rather than silently producing codes. The dot kernel reads both operands in packed I4 storage, sign-extends each nibble, accumulates an exact I32 dot, then applies F32 row and token scales in the declared order. Its grid covers `N=1` decode and `N>1` work, including row and token tails.

The first dispatch in each N class logs **“CUDA W4A4 signed-nibble vector dot, scalar integer MUL/ADD; no INT4 Tensor Core MMA.”** This labels the source path; executed instruction evidence remains pending. It is distinct from the pinned Q4_0/Q8_1 control and from true signed-I4 Tensor Core MMA. No CUDA timing is available.

## Separate SM75 instruction probe

`tests/test-w4a4-sm75-mma.cu` is a standalone, opt-in probe and is not wired into production dispatch. It packs the existing even-low signed-I4 layout into the A/B per-lane registers for NVIDIA's `mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32`. A warp computes an 8-row × 8-token tile, with guarded K, row, and token tails. It compares every I32 output to an independent scalar dot for an 8×8×32 basis/random tile and M=9 cases with N=1/9 and K=9/33/9,728, including a zero token. The register mapping follows the [PTX `mma.m8n8k32` fragment specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-8832). The mapping is still a hypothesis until the probe runs on SM75.

On the 2080 Ti, the first correctness command is:

```sh
nvcc -std=c++17 -arch=sm_75 tests/test-w4a4-sm75-mma.cu -o test-w4a4-sm75-mma
./test-w4a4-sm75-mma
```

Require all exact I32 comparisons to pass, inspect the executed kernel's SASS, then compare the production vector path against the CPU oracle at `N=1` and representative `N>1` EAGLE shapes. The current backend oracle cases are K=9/N=3 (ties, signs, zero, stride) and K=9,728/N=2 (maximal reduction); both are registered for CUDA as well as CPU. Only after the probe and model-level checks should an MMA branch be considered for production dispatch. Keep vector and MMA instruction classes separate in logs and benchmark labels.

## Checks and integration

- CPU-only CMake rebuilt `test-backend-ops` and `test-w4a4-eagle-load`; `test-backend-ops -b CPU -o 'W4A4.*'` passed **2/2** on Apple M3 Max. Focused W4A4 Python converter/reference tests passed **5/5**; `git diff --cached --check` passed.
- CUDA compilation and execution remain **unverified**: local `nvcc` is unavailable. The probe has no SM75 result or SASS yet. No 2080 Ti session or other GPU was used.
- This W4 branch is based on the published W8A8 CPU/loader commit `492818599`; concurrent W8A8 CUDA source is on another branch. Integrating both will require review of shared `ggml-cuda.cu` dispatch/supports-op edits. The first runtime gate is an SM75 CUDA build and exact backend oracle run, followed by the isolated MMA probe, graph-dispatch trace across all nine linears, logits/acceptance checks, and only then packing-inclusive timing.
