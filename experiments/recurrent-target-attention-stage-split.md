# Attention and O projection stage split on RTX 5080

This bounded same-input diagnostic uses output-preserving native
post-RoPE Q/K, V, pre-O `kqv_out-0` and `ffn_inp-0` tensors from one
frozen 29-token code/data-validation training prefix. The RTX 5080
(SM120) ggml path uses F32 Q, a padded 256-slot F16 K/V cache, F16
causal mask, F32-accumulation Flash Attention, pinned F16 O weight and
F32 residual. The Torch control uses the source Qwen3 eager attention
and F16 O projection. The target GGUF, source O weight, captures and
prior Torch intervention are checked by hash.

Three independent ggml CUDA gates are **bitwise native**: full
attention/O/residual 74,240/74,240 F32 values, attention alone
118,784/118,784, and O plus residual from captured native pre-O
input 74,240/74,240. The separately emitted ggml O projection plus
the frozen F32 layer input also reconstructs all 74,240 native
`ffn_inp-0` values. Explicit F16 rounding of the ggml O input changes
none. The constructed Torch native-Q/K/V path reproduces the earlier
combined intervention metrics exactly.

| Same-input intervention | Comparison | Position-3 relative row L2 |
| --- | --- | ---: |
| Torch eager attention on native Q/K/V | Native pre-O attention output | 0.1058% |
| Torch attention output through ggml O plus F32 residual | Native `ffn_inp-0` | 0.1253% |
| Torch O projection on native pre-O output | ggml O projection | 0.2071% |
| Torch O on native pre-O output plus F32 residual | Native `ffn_inp-0` | 0.1947% |
| Torch O on native pre-O output plus F16 residual | Native `ffn_inp-0` | 0.1941% |
| Torch attention and Torch O/F16 residual | Native `ffn_inp-0` | 0.2048% |

Under this geometry, both source attention and source O/output
precision contribute. The O projection discrepancy is already
present on identical native pre-O input: only 7,451/74,240 projected
F32 values match, maximum absolute gap `0.005859375`. Torch eager
attention matches only 3/118,784 native pre-O F32 values, with
maximum absolute gap `0.0007233619689941406`. F16 versus F32
residual addition on the same Torch O output differs by 0.0212%
position-3 relative row L2;
all 74,240 results agree after F16 casting. These errors are not
additive and do not establish one universal cause outside the tested
prefix.

The final supervised run is
`checkouts/target-block0-operator-20260928/runs/target-attention-stages-b-20260928`
on the registered WSL host. Its ignored `comparison.json` SHA256 is
`c9c24997d2f3f6bcc1d1bf9b734cd5ffa2bbb1072f35bba795c2a499e9c2164e`.
The first run already passed attention, O-residual and Torch control
gates; the second added the projection-only and residual-precision
checks. The report records full metrics, operand/capture and code
hashes, hardware, software and precision. The final supervisor exited
zero and released its process group; GPU use returned to 0% and
1,372 MiB whole-device baseline. Local C++ syntax, Ruff lint/format
and Python compilation passed. No optimizer, final prompt or Q4_0
serving evaluation ran. This one SM120 operator case does not prove
SM75 performance, later-block parity, accepted-trajectory parity or
a training tolerance.
