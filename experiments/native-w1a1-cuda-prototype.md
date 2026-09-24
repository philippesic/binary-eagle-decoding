# Packed W1A1 CUDA prototype on RTX 5080

**Date:** 2026-09-24. **Status:** standalone kernel correctness and component
latency, before GGML dispatch and end-to-end decoding. **Device:** NVIDIA RTX
5080, SM 12.0; Windows driver 616.92, CUDA toolkit 13.1.115/runtime 13010,
WSL glibc 2.43. No RTX 2080 Ti/SM75 result is claimed here.

The prototype packs F32 activation signs (zero maps to +1) into little-bit-order
uint32 words, computes one F32 mean-absolute activation scale per token, and
uses XOR/`__popc` for the integer dot. F32 row weight scales and activation
scales multiply the integer result. Source commits are `44d1c4b` (prototype),
`26484b7` (host math-header fix), and `37ebbc6` (documented build flags).
The tested remote source/binary came from the worker's pushed branch
`feat/cuda-w1a1-prototype` at `8950822`; its binary SHA256 is
`c0f4d0ab97a86058920b47298f4208311553b02d6725d0c1f973d43af6f030e3`.

The WSL CUDA 13.1/glibc 2.43 headers disagree on `rsqrt` declarations with
plain `nvcc`. The working command used
`nvcc -std=c++17 -O3 -arch=sm_120 -U_GNU_SOURCE -D_DEFAULT_SOURCE -Xcompiler=-Wall,-Wextra`
as recorded in `kernels/README.md`. This host-specific fix must be retested on
SM75 and should not be mistaken for a kernel correctness issue.

The supervised correctness run
`runs/cuda-w1a1-correctness-20260924-c` exited zero. It covered all eight
audited reduction widths (31, 32, 33, 2560, 4096, 5120, 7680, 9728), with
21 row/token pairs each, signed zeros, subnormals, and dirty tails. Integer
dots were exact; packed activation signs and F32 scale/output passed the
declared tolerance. Correctness JSON SHA256:
`72a39773faacf23efd2d725f69c8fbce1ff15611c397fe7d386e624ccd5f81c5`.

Two supervised benchmarks ran all nine EAGLE shapes at 1 and 10 tokens,
alternating 30 CUDA-event samples per prepacked and packing-inclusive path.
The paths exclude H2D transfers, allocation, graph replay, remaining drafter
work, and target verification. Raw JSON lives under
`runs/cuda-w1a1-bench-20260924/` (SHA256
`15a0ef3b69f3a83b839e562014b26d6e238d00ea597ff35b7baa154c723e11c0`)
and `runs/cuda-w1a1-bench-repeat-20260924/` (SHA256
`2eaa20c224c4109bd84e02a48f3be562d4f5b947193de14780379f68c253c9b4`).
The table shows repeat-run medians for 10 tokens:

| Linear | Prepacked ms | Packing-inclusive ms |
| --- | ---: | ---: |
| Fusion | 0.016128 | 0.026400 |
| Attention Q | 0.016608 | 0.026016 |
| Attention K | 0.010112 | 0.019360 |
| Attention V | 0.009760 | 0.019360 |
| Attention O | 0.011168 | 0.021280 |
| FFN gate | 0.025856 | 0.035520 |
| FFN up | 0.025376 | 0.035488 |
| FFN down | 0.015840 | 0.025760 |
| Vocabulary head | 0.070016 | 0.079136 |

For one token, repeat-run prepacked medians ranged from 0.006944 to 0.013280
ms and packing-inclusive medians from 0.017056 to 0.023424 ms across the nine
shapes. Attention Q's first-run one-token packing-inclusive median was
0.032832 ms versus 0.017696 ms on repeat, with bimodal raw samples. This
variance and the missing graph/H2D/system costs preclude a component speedup
or end-to-end claim from these numbers alone.

The standalone packer accumulates activation magnitudes in F32 before storing
an F32 scale. The integrated GGML CPU/CUDA operation accumulates magnitudes in
F64 and rounds the mean once to F32. Integer dots are the same, but close
floating results can differ. The native-contract acceptance check must use
the integrated GGML reduction policy; the standalone timing cannot by itself
validate exact GGML head logits.

The CUDA prototype's supervisors are terminal, PIDs gone, and GPU memory
returned to its 3,050 MiB Windows baseline at 0% utilization before QAT GPU
handoff. Next, compile and validate the GGML CUDA dispatch, then measure the
packed head in direct same-device decoding comparisons.
