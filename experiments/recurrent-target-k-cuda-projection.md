# Same-input target K projection and head norm on RTX 5080

This bounded, no-optimizer probe replays Qwen3 target block-0 K on the
RTX 5080 (SM120). It uses the 29 F32 attention-norm rows from an
output-preserving native capture, the pinned F16 GGUF K matrix, and the
F32 GGUF per-head norm weight. GGUF values match the published HF
source. The graph performs `ggml_mul_mat`, reshapes its 1,024 outputs
into eight 128-wide heads, then applies ggml RMS norm and weight with
epsilon `1e-6`, matching the native Qwen3 graph before RoPE. The frozen
training prompt, target/candidate GGUFs, layer ladder, safe K/norm taps
and earlier HF comparison are checked by hash.

| Same native input and weights, versus output-preserving K tap | Exact F32 / 29,696 | Position-3 relative row L2 |
| --- | ---: | ---: |
| Standalone ggml CUDA K projection + norm, F32 input | **29,696** | 0 |
| Standalone ggml CUDA K projection + norm, explicit F16 input cast | **29,696** | 0 |
| Torch CUDA/F16 K projection + Qwen3 HF norm | 2 | 0.0916% |
| Torch Qwen3 norm, same ggml raw K as F32 | 24,456 | 0 |
| Torch Qwen3 norm, same ggml raw K cast to F16 | 4 | 0.0338% |

The two ggml paths are bitwise identical at both raw projection and
post-norm K. Torch's F16 raw projection matches 6,530/29,696 ggml raw
F32 values; its maximum absolute raw gap is `0.00244140625`. The full
Torch comparison reproduces the prior same-input HF intervention
exactly: 5,928/29,696 values agree after F16 casting, maximum absolute
post-norm gap `0.557708740234375`, and 0.0785% median relative row
L2. With identical ggml raw F32 K, the HF norm's maximum absolute gap
is only `4.57763671875e-5` and position 3 is bitwise exact. Casting
that same raw K to F16 before HF norm increases position-3 error to
0.0338%. Thus the tested K gap includes projection backend arithmetic
and F16 intermediate precision; same-raw F32 norm arithmetic is much
smaller. The components are not assumed additive. This does not
identify a particular CUDA instruction.

The final supervised run is
`checkouts/target-block0-operator-20260928/runs/target-k-cuda-c-20260928`
on the registered WSL host. Its ignored `comparison.json` SHA256 is
`58c67405a21576200ba5ce0a1db4675e39c674ee43b27f2e9e448dcedcf8400f`.
It records operands, source/capture/helper/library hashes, software
versions and full metrics. The ggml CUDA library SHA256 is
`59c8b5c4cccaaab2fd3721ee6f8d50b37b8b7ab6b62b6c995978fa84e8b9ef1f`.
The earlier generic `torch.nn.functional.rms_norm` control did not
match the prior HF result; a corrected Qwen3 F32-variance,
cast-before-weight control did. The first two run directories are
retained as ignored diagnostics, while the corrected third run is the
result above.

The final supervisor exited zero and released its process group. The
RTX 5080 returned to 0% utilization and 1,372 MiB whole-device
baseline use. Local C++ syntax and Ruff lint/format passed. No target
full-model forward, optimizer, development/final prompt or Q4_0
serving evaluation ran. This is one frozen prefix on SM120, not SM75
performance, global target-feature parity or a training tolerance.
The Q callback remains intrusive; exact versus predeclared numeric
policy and any all-body optimizer budget remain user-owned.
