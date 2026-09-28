# Output-preserving target Q capture and replay on RTX 5080

The earlier direct `Qcur_normed-0` callback changed 1,911 of 74,240
block-0 output values on the frozen 29-token code/data-validation
training prefix, so its Q data could not identify the normal server
path. The ggml scheduler computes a graph segment and synchronizes at
each callback tensor requested by `cb_eval`. A new bounded mode records
the post-RoPE `Qcur-0` tensor pointer while declining a callback there,
then reads it when the previously output-preserving K-norm callback
runs. Q is still live for subsequent attention at that point.

The supervised native RTX 5080 (SM120) run captured one F32 post-RoPE
Q tensor of shape `[128,32,29]`. Its K-norm payload is byte-identical
to the prior safe K capture; its complete block-0 output is
**74,240/74,240 F32 bitwise** identical to the sealed layer ladder and
to the output-only capture. The deferred Q is finite. Relative to the
earlier intrusive post-RoPE Q tensor, **102,401/118,784** F32 values
match; maximum absolute difference is `9.5367431640625e-7`, and all
but three values agree after F16 casting. This small Q difference does
not by itself prove why the intrusive run's later output changed.

A standalone ggml CUDA replay used the output-preserving native F32
attention-norm rows, the pinned F16 GGUF Q matrix and F32 head-norm
weight, and positions 0–28. The graph applies `ggml_mul_mat`, 32-head
128-wide RMS norm and NeoX RoPE with source theta 1,000,000. Both
operands match the HF source weights. The replay matches the deferred
native post-RoPE Q **118,784/118,784 F32 values bitwise**. Explicitly
casting the input to F16 changes none of the raw projection,
normalized Q or post-RoPE Q values. This validates the deferred capture
as a faithful same-input ggml Q boundary for this geometry. It does
not yet compare Torch Q arithmetic or identify attention/FFN causes
of target-feature drift.

The ignored remote reports are:

| Run under `checkouts/target-block0-operator-20260928/runs/` | `comparison.json` SHA256 |
| --- | --- |
| `target-q-deferred-a-20260928` | `ae5f934444e64b4b3a589b5be2bdcf67cfa55611e64c425d3d41411d437a969f` |
| `target-q-deferred-audit-20260928` | `d1da2dd8ca27477e7e25992be60c789296572a9fe8adee864a303b11c62c7f1d` |
| `target-q-cuda-a-20260928` | `febaced1bd0e7ad5d6d7471693e3408788ae98cd276e017dcbab07e723e8979b` |

The reports seal the frozen ladder and GGUF, output/K/Q payloads,
helper source and binary, source weights, ggml CUDA library and
software versions. The CUDA library SHA256 is
`59c8b5c4cccaaab2fd3721ee6f8d50b37b8b7ab6b62b6c995978fa84e8b9ef1f`.
All three supervised runs exited zero and released their process
groups; the final RTX 5080 use was 0% and 1,372 MiB whole-device
baseline. Local C++ syntax and Ruff lint/format passed. No optimizer,
final prompt, Q4_0 serving evaluation or target full-model forward
ran in the standalone replay. This is one SM120 training-prefix
operator case, not SM75 performance, general target-feature parity or
a training tolerance. The all-body budget and exact-versus-numeric
policy remain user-owned.
