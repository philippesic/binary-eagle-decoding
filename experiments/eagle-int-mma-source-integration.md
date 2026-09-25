# Combined opt-in low-bit MMA source

**Status:** source integration and CPU checks passed, 2026-09-24. No CUDA build, SM75 execution, SASS, model parity, or timing is claimed.

The llama.cpp fork branch `feat/eagle-int-mma-integration` starts at combined vector/DP4A baseline `d0724427b` and applies the published W4A4 opt-in source as `f48971ba7` (from `eaa7fb18`), then W8A8 opt-in source as `efded1819` and `2d9712cde` (from `4025d037` and `fbaa9e239`). The parent repository gitlink is unchanged. The commits applied without conflicts. The focused backend test registry contains both format families: five W8A8 cases and four W4A4 cases, including N=1 and N=9 shape cases. Existing default vector/DP4A paths, environment selectors, arithmetic contracts, and dispatch markers were preserved.

On Apple M3 Max, a CPU-only CMake build of `test-backend-ops` passed with `GGML_CUDA=OFF`, `GGML_METAL=OFF`, and `LLAMA_BUILD_TESTS=ON`. `test-backend-ops -b CPU -o W8A8_MUL_MAT` passed **5/5**; `test-backend-ops -b CPU -o W4A4_MUL_MAT` passed **4/4**. `git diff d0724427b..HEAD --check` passed. These tests verify the combined source and numerical CPU oracles; they do not execute either CUDA MMA kernel.

The GPU operator must build this exact combined tip for `sm_75` after the sealed vector run. For each opt-in candidate, separately run backend correctness, the standalone instruction probe, actual EAGLE dispatch and token/logit parity, and executed SASS inspection before comparing packing-inclusive and end-to-end timing. Keep the default paths and both opt-in variants separately labeled; a source branch alone establishes no Tensor Core speed or correctness result.
