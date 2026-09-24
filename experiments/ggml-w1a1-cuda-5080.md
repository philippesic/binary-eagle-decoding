# Integrated GGML W1A1 CUDA on RTX 5080

**Date:** 2026-09-24. **State:** CUDA build and focused backend correctness
passed; model-level CUDA smoke and paired end-to-end benchmark are next.
**Hardware:** RTX 5080, SM120a codegen, driver 616.92, CUDA 13.1.115,
WSL GCC/G++ 15.2 and glibc 2.43. This is not an SM75 or 2080 Ti result.

The parent checkout was `b7e7f73` for the successful build and the pinned
llama.cpp submodule was `92bc70602e13214d6db94c007894261b51f5f36c`.
That submodule has a dedicated `GGML_OP_W1A1_MUL_MAT`: GGUF I32 little-bit
packed weight words, F32 per-output-row scales, runtime F32 activation sign
packing and mean-absolute scales, a masked XOR/`__popc` integer dot, and F32
output. The EAGLE graph routes only the drafter vocabulary head to it when
the versioned packed GGUF companions are present. The target and verifier
paths remain ordinary.

## Host toolchain compatibility

Plain CUDA 13.1 compilation on glibc 2.43 conflicts with glibc's GNU/C23
`rsqrt`/`rsqrtf` declarations. The standalone prototype could suppress
`_GNU_SOURCE`, but the integrated llama.cpp build then lost pthread
clockwait/clocklock declarations needed by GCC 15 `<mutex>` and stopped at
27/368 objects. That failed supervised run is preserved under
`runs/native-ggml-cuda-build-20260924/`; it was a host-header error before
W1A1 validation.

A private copy of CUDA's include tree was placed under the ignored
`results/cuda-glibc-compat/include/`. Only the two copied
`crt/math_functions.h` device declarations for `rsqrt(double)` and
`rsqrtf(float)` received `noexcept(true)`. The system toolkit was not changed.
A supervised nvcc probe including CUDA runtime, `<mutex>`, and
`<condition_variable>` passed with normal GNU macros and confirmed the
private header was resolved. The project build uses
`scripts/build_llama.py cuda --cuda-arch 120 --cuda-include-root
results/cuda-glibc-compat/include --with-tests --jobs 4` under
`scripts/remote_job.py`; the CMake cache records the copied include root
and no `-U_GNU_SOURCE`. The patched build completed all 341 objects under
`runs/native-ggml-cuda-build-patched-20260924/`. Exact private header and
build-log hashes remain in the remote run artifacts; they will be added after
the operator seals the model-level run.

## Backend correctness

The supervised `test-backend-ops test -b CUDA0 -o W1A1_MUL_MAT` passed 5/5 on
the 5080: logical K=31/32/33/2560 and a strided K=33 activation. These tests
compare against an independent scalar sign reference, including signed zeros,
dirty tail bits, and zero scales. The CUDA log emitted
`CUDA packed W1A1 XOR/POPCOUNT dispatch (K=31, rows=7, tokens=3)`, confirming
the new operation ran on the GPU. The test supervisor exited zero and GPU
memory returned to its 3,050 MiB Windows idle baseline at 0% utilization.

The five tests cover arithmetic/layout edge cases, not the full 32,000-row
EAGLE head or GPU graph replay. Model-level native dispatch, conversion
hashes, matched target-only/ordinary/packed acceptance, and end-to-end timing
are separate gates. The portable `__popc` implementation is true binary
execution but does not exercise Turing binary Tensor Cores; the
[SM75 MMA check](sm75-binary-mma-plan.md) remains pending actual 2080 Ti
access.
