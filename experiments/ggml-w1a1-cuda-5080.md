# Integrated GGML W1A1 CUDA on RTX 5080

**Date:** 2026-09-24. **State:** CUDA build, focused backend correctness,
and one-prompt model-level CUDA dispatch passed; paired end-to-end benchmark
is next.
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
The original/system header SHA256 was
`decdc28efcfaf0aaf806abc96d7bba9cb84b37c6e83cb82cda59b6aa59916ff8`;
the patched private header SHA256 was
`13256b220d400a5665cde8bc87c21b0188944390c6a945a7fffa2827254b305a`.
The bounded local preparation was:

```sh
mkdir -p results/cuda-glibc-compat
cp -aL /usr/local/cuda/targets/x86_64-linux/include results/cuda-glibc-compat/include
python3 - <<'PY'
from pathlib import Path
import re
p = Path('results/cuda-glibc-compat/include/crt/math_functions.h')
s = p.read_text()
for name in ('rsqrt', 'rsqrtf'):
    pattern = rf'(\b{name}\(float x\)|\b{name}\(double x\));'
    s, count = re.subn(pattern, r'\1 noexcept (true);', s)
    assert count == 1, (name, count)
p.write_text(s)
PY
```

The copied header was verified by an nvcc probe including `<cuda_runtime.h>`,
`<mutex>`, and `<condition_variable>` with normal GNU macros before
attempting the full build. This workaround is tied to the recorded CUDA/glibc
combination; do not apply it blindly on the 2080 Ti host if its toolchain
differs. The project build uses
`scripts/build_llama.py cuda --cuda-arch 120 --cuda-include-root
results/cuda-glibc-compat/include --with-tests --jobs 4` under
`scripts/remote_job.py`; the CMake cache records the copied include root
and no `-U_GNU_SOURCE`. The patched build completed all 341 objects under
`runs/native-ggml-cuda-build-patched-20260924/`. Exact private header and
build-log hashes are in the sealed remote validation manifest cited below.

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

## Pinned GGUF conversion

A separate pinned Python 3.11.15 conversion environment used torch 2.11.0,
Transformers 4.57.6, and GGUF 0.19.0. Its package list hash matches the local
reference `d5269b8b800401897f8c6519b745236f1882663c295b6c50891bf9df1fa8af69`.
Every conversion set `CUDA_VISIBLE_DEVICES=` and left the GPU idle.

| RTX 5080 host artifact | Bytes | SHA256 |
| --- | ---: | --- |
| Target F16 GGUF | 8,051,285,280 | `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6` |
| Ordinary EAGLE F16 GGUF | 442,700,800 | `c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1` |
| Original-head packed W1A1 EAGLE GGUF | 289,229,312 | `b2095130b5196574a9a08a88d2fb9a32ac1ea870ff3cf587ae7d6e7f64e7819f` |

Target and ordinary draft are byte-identical to the local Mac conversions.
The packed file's whole hash differs from the local file because the F32
weight-scale reduction is platform sensitive. The packed I32 tensor SHA256 is
identical across hosts,
`7e3c5642ec462e7a01691db84c4d25bbbd4a262977dc59794e441fc502a49b1b`;
the 5080 host's F32 scale tensor SHA256 is
`9683db664835fb5273fef0a2e8696da14cd224938cc9cc3b59bb8ba0b3d7460c`
versus local
`93c714eb6c21ce41ee27e8aeb6aad877f63e398a874167ce5f65f3a9cd2c56ea`.
All 32,000 signs and scales exactly match the corresponding host's BF16
source-to-F32-mean exporter; all 13 shared non-head tensors match the ordinary
draft exactly, and dense `output.weight` is absent. Against a F64-summed/F32
mean reference, remote F32 weight scales differ in 12,883 rows by at most
three ULP or 5.59e-9 absolute. This is a small, recorded conversion
rounding difference, not an integer packing error. Native results must cite
the 5080 host packed GGUF hash rather than the Mac hash.

## Model-level CUDA smoke

A single supervised packed-draft `llama-server` request on held-out
`prose-01` returned HTTP 200 with 16 generated tokens. The log showed the
packed EAGLE-head loader and the explicit
`CUDA packed W1A1 XOR/POPCOUNT dispatch` marker. `/metrics` deltas were 50
proposed tree nodes, two accepted draft tokens, and 12 verification rounds:
0.1667 accepted drafts/round for this short prompt. Loaded GPU memory sampled
at 12,269 MiB used with 3,709 MiB free; after clean shutdown it returned to
the 3,050 MiB/0% Windows baseline. Supervisor
`native-cuda-packed-head-smoke-20260924` exited zero. Raw log/request/model
hashes are sealed in the ignored host manifest
`results/native-ggml-cuda-validation-20260924/run-manifest.json`, SHA256
`8707fc2e774508a8945211168841edeafea15774ddd8b4ddd4c4ba9b572bab3d`.
It covers the failed/successful build, patched header, backend test,
conversions/audits, smoke, and final GPU state. One short prompt is only a
dispatch/counter gate, not an acceptance distribution or speed measurement.
The native server uses `--spec-draft-n-max 5`; its proposals differ from the
AngelSlim PyTorch acceptance sweep's 59-node tree. Native same-device
comparisons must use the target-only and ordinary llama.cpp anchors rather
than transplanting PyTorch accepted/round counts.

After the matched comparison, one bounded Nsight Compute 2025.4.1 attempt
filtered to the activation-pack and XOR/POPCOUNT kernels. The packed request
completed and logged CUDA dispatch, but NCU returned `ERR_NVGPUCTRPERM`
(performance counters unavailable), so no integrated kernel timings or
launch counts were obtained. The partial trace/log artifacts are preserved
under remote `results/packed-w1a1-ncu-profile-20260924/`, manifest SHA256
`1f3268c4dce267609fdc7cec9b355dcd14d382cab780e1539c81e36def2cd031`.
No second profiler attempt was made; the GPU returned to idle and SSH/tmux
closed.
