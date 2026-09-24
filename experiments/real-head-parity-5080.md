# Captured-input packed-head numerical parity on RTX 5080

**Date:** 2026-09-24. **Scope:** standalone portable CUDA W1A1 kernels with
the published full 32,000-row EAGLE vocabulary head and eight actual
captured BF16 pre-head activation vectors. This is a correctness check, not
a timing result or an SM75 execution result.

Parent commit `1e4e76a` added a versioned fixture creator and CUDA checker.
The ignored fixture was generated from
`results/qat-head-capture-20260924/train.pt` (the disjoint QAT training
capture, never the held-out prompts) and the 5080 host's
`models/gguf/Qwen3-4B-eagle3-head-w1a1.gguf` (SHA256
`b2095130b5196574a9a08a88d2fb9a32ac1ea870ff3cf587ae7d6e7f64e7819f`).
It contains eight deterministic captured vectors of K=2560, all 32,000
packed weight rows/scales, expected activation sign words, F64-summed/F32
activation scales, exact integer dots, and ordered F32 outputs. The binary
header and JSON sidecar record version, dimensions, source/code hashes and
selected row indices. The creator and checker commands are in
[kernels/README.md](../kernels/README.md).
Selected capture row indices were `[0, 4681, 9362, 14043, 18724, 23405,
28086, 32767]`; source capture SHA256 was
`62565c9b675e19bdddc3fb0a963128c98df7539640f6ba6c5b4361a7131ef2e9`.

The supervised CUDA checker ran once on RTX 5080 SM120 and passed:

| Comparison | Checked | Mismatches | Maximum absolute error |
| --- | ---: | ---: | ---: |
| Activation packed sign words | 8 × 80 | 0 | exact |
| Integer XOR/POPCOUNT dots | 8 × 32,000 | 0 | exact |
| F32 activation scales | 8 | 0 outside tolerance | 1.1920929e-7 |
| Ordered F32 outputs | 8 × 32,000 | 0 outside tolerance | 9.53674316e-7 |

The standalone CUDA activation packer accumulates F32 partial sums, while
the reference sums magnitudes in F64 and rounds once to F32; the scale/output
tolerances explicitly cover this reduction-order difference. Exact sign and
integer-dot equality ensure the packed binary arithmetic itself agrees for
every sampled real-head output. This result complements the five integrated
GGML CUDA backend tests and the full native server dispatch smoke, but it
does not directly read out integrated GGML head logits for these captured
vectors. It does not measure latency or establish 2080 Ti/SM75 behavior.

The fixture, compile, check logs, environment, and final GPU/process-state
hashes are preserved under remote `results/real-head-fixture/` with
`artifact-manifest.json` SHA256
`dd3b732e5f9188589b12700f3e059275a6bc1b7e81e407655cc066854b7a6d98`.
Fixture SHA256 was
`daf1dfac478f583f09612492698f4d1c427c14fd75cf12dd3cef4d0fd0d8e2d5`,
checker binary
`f5106030e586b806bcae89527c44e23d6d02eada301a16c0f9232a6b4c60ba3e`,
and checker JSON/stdout
`5bff12e3f38833983473f1ff15a0b89ba7594db5f3d031c39c14e94e942361c5`.
The remote code was parent `6a8e3b3` with llama.cpp submodule `92bc706`,
CUDA 13.1.115, Python 3.11.15, torch 2.11.0+cu130, NumPy 2.2.6, and packed
contract v1. Fixture, compile and checker supervisors all exited zero. Three
post-run GPU samples were at 3,046 MiB used / 12,932 MiB free / 0%, with no
project process; SSH/tmux closed.
