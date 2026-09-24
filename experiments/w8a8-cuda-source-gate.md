# W8A8 CUDA source gate

**Status:** reviewable source published, 2026-09-24. No CUDA build, GPU execution, SM75 SASS, or timing was performed in this stage. Local host is an Apple M3 Max without `nvcc`.

**Code:** llama.cpp fork branch `feat/w8a8-eagle-export`, commit `a68968fad`, based on the published W8A8 CPU/loader commit `492818599`. The parent repository gitlink remains pinned at `8d2b18a`; this report does not update it.

## CUDA path

`GGML_OP_W8A8_MUL_MAT` now has an NVIDIA CUDA dispatch. A launch first quantizes each F32 activation token/vector once to signed I8, using one F32 absmax/127 scale and nearest-even conversion; an all-zero vector produces zero codes. It accepts a strided token row (`nb[1]`) with contiguous F32 elements. A row-parallel kernel then packs four two's-complement bytes per operand, uses GGML's signed `dp4a` wrapper, reduces I32 partial sums across a warp, and computes `((F32)dot * weight_scale) * token_scale` in that order. It handles K tails, N=1 decode, and N>1 encoder/reconciliation shapes with the same path. An explicit one-time log marker reads `CUDA W8A8 signed INT8 dot/I32 accumulation dispatch`.

GGML CUDA normally compiles with `-use_fast_math`. The new pack uses explicit PTX `div.rn.f32` before `__float2int_rn`, and the output uses ordered `mul.rn.f32` operations, to avoid approximate reciprocal and reassociation. The absmax reduction also uses PTX F32 abs/max without FTZ. Nonfinite activations or weight scales trigger a device trap, matching the CPU operator's finite-input requirement. This is a **signed INT8 DP4A/SIMT design**; it makes no INT8 Tensor Core claim. HIP/MUSA report this op unsupported rather than run an unvalidated path.

## Source and CPU checks

- `git diff --cached --check` passed before the submodule commit.
- CPU-only CMake build of `test-backend-ops` passed. `test-backend-ops -b CPU -o W8A8_MUL_MAT` passed **3/3** on Apple M3 Max: K=8 contiguous, K=33 strided/tail, and K=9,728 strided/multi-token. Each has an all-zero token. The independent scalar oracle now divides in F32 before nearest-even rounding and includes positive and negative near-half cases where double division would choose differently.
- The prior [loader/CPU gate](w8a8-cpu-loader-gate.md) loaded all nine real export tensors, rejected malformed metadata and missing codes, and completed a short CPU EAGLE run. Those results do not validate the new CUDA source.

## Required SM75 gate

On the RTX 2080 Ti, configure and build the exact submodule revision with CUDA architecture 75, then run the expanded W8A8 backend cases against the independent oracle. Check K=2,560, 4,096, 5,120, 7,680, and 9,728 at real row counts and representative N=1/N>1 shapes; include zero, signed extremes, near-half boundaries, cancellation, and strided rows. Compare packed codes, scales, exact I32 dots, and scaled outputs separately where possible. Trace a short real EAGLE request to show all nine nodes execute on CUDA with no CPU fallback; preserve the dispatch marker and disassemble the executed kernel to establish signed DP4A or equivalent integer instructions. Only after logits, acceptance, and text checks should the paired benchmark time the variant, including activation packing. The native W8A8 result remains unmeasured until these checks pass.
