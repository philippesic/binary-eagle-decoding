# Combined W8A8 and W4A4 CUDA source

**Status:** combined source published, 2026-09-24. This is a CPU build and integration check only; no CUDA build, SM75 execution, SASS, or timing is claimed.

**llama.cpp fork:** branch `feat/eagle-int-cuda-integration`, commit `b881a6346`, starting from W4A4 commit `308713988d5b2863a974d5edbaf7409b8d32b1d5` and cherry-picking W8A8 CUDA commit `a68968fad`. The parent repository gitlink remains at its pinned revision; this note is the only parent-repository change.

## Resolution

The cherry-pick overlapped in `ggml/src/ggml-cuda/ggml-cuda.cu` and `tests/test-backend-ops.cpp`. The resolved CUDA file includes both headers and has distinct compute and `supports_op` cases for `GGML_OP_W4A4_MUL_MAT` and `GGML_OP_W8A8_MUL_MAT`. The W4A4 case retains packed nibble geometry and its original marker; the W8A8 case retains full-width signed I8 geometry and the `CUDA W8A8 signed INT8 dot/I32 accumulation dispatch` marker. No quantization, scale, or integer-dot kernel contract was changed. The backend test registry keeps both W4A4 cases and the three expanded W8A8 cases.

## Local checks

- `git diff HEAD^ HEAD --check`: passed in the combined submodule branch.
- CPU-only CMake configure/build with `GGML_METAL=OFF`, `GGML_CUDA=OFF`, `LLAMA_BUILD_TESTS=ON` succeeded for `test-backend-ops`, `test-w8a8-eagle-load`, and `test-w4a4-eagle-load` on Apple M3 Max.
- `test-backend-ops -b CPU -o W8A8_MUL_MAT`: **3/3 passed**, K=8, K=33 strided/tail, K=9,728 strided/multi-token.
- `test-backend-ops -b CPU -o W4A4_MUL_MAT`: **2/2 passed**, including the K=9 strided/tail and K=9,728 cases.
- The focused load binaries each accepted their previously exported draft GGUF. Logs identified W8A8 and W4A4 all-nine-linear coverage, respectively. Ignored local logs are under `results/int-integration/`.

## Remaining validation

Build the exact combined commit for `sm_75` and run both CUDA backend-op families on the 2080 Ti under the GPU operator's ownership. Trace separate EAGLE requests to prove all nine eligible linears use CUDA for each format, preserve operand/scale audits, and disassemble executed kernels. The W8A8 path is signed INT8 DP4A until SASS shows otherwise; the W4A4 path must be labeled by its actual signed nibble implementation. Check logits, accepted drafts, target text parity, and packing-inclusive cost before adding timed rows to the shared suite. One combined source tree prepares the binary; it does not validate either native precision result.
