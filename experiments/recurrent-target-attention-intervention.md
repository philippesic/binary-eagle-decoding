# Same-input block-0 attention intervention on RTX 5080

This bounded RTX 5080 (SM120) diagnostic joins five output-preserving
native captures on the frozen 29-token code/data-validation training
prefix: attention-norm input, post-RoPE Q and K, V, and the
attention-residual `ffn_inp-0`. All captures use the same pinned F16
target GGUF, CUDA library and sealed ladder. The full Hugging Face
Qwen3-4B F16 model runs once in eager mode on that prefix to capture
its actual block-0 attention input, causal mask, rotary cos/sin and
residual. An independent call through the same source Qwen3 modules
then matches its own full-model residual **74,240/74,240 F16 values
bitwise**. Seven block-0 source weights match the GGUF values.

| Torch CUDA/F16 eager path versus safe native `ffn_inp-0` | Position-3 relative row L2 | Median relative row L2 |
| --- | ---: | ---: |
| Loaded HF model, source input and Q/K/V | 0.2547% | 0.2269% |
| Source Q/K/V on native attention-norm rows cast to F16 | 0.2434% | 0.2257% |
| Safe native post-RoPE Q/K and V cast to F16 | 0.2048% | 0.2038% |

The intervention reduces the position-3 row error, but a material
residual remains. This does **not** assign the remainder solely to
Flash Attention: the source eager path casts native Q to F16, uses
Torch attention and F16 output projection, and adds its residual in
F16. Native ggml Flash Attention takes F32 Q and stored F16 K/V, then
projects through its own backend. Those precision and operator
boundaries need separate same-input replay before a numeric policy
can be selected. The stage errors are not additive.

The loaded HF model's rotary inverse-frequency buffer is F16. A fresh
isolated Qwen3 rotary module has an F32 buffer; their captured cosines
match 3,264/3,712 F16 values, with maximum absolute difference
`0.00439453125`. The isolated rotary path reproduces the earlier
same-input Torch Q report exactly, while V also reproduces its earlier
control. The intervention uses the **loaded model's** cos/sin and
matches its full forward bitwise, so this distinction is accounted
for rather than silently changing the HF baseline.

The final supervised run is
`checkouts/target-block0-operator-20260928/runs/target-attention-intervention-b-20260928`
on the registered WSL host. Its ignored `comparison.json` SHA256 is
`9b125cb4c8f73eade7bfc1127495479ac88e4ff7d1fbfae097337be30d9b46d8`.
It records every stage metric, safe-capture and weight identities,
model/source hashes, software versions and precision. The first
supervised attempt preserved its report but failed only the isolated
Q-control equality because its loaded-model rotary cos/sin were
different; the corrected run reproduces both Q variants and V. The
final supervisor exited zero and released its process group; the GPU
returned to 0% utilization and 1,372 MiB whole-device baseline.
Local Ruff lint/format and Python compilation passed. No optimizer,
final prompt or Q4_0 serving evaluation ran. This one SM120 prefix
does not establish SM75 performance, later target-feature parity,
accepted-trajectory parity or a training tolerance.
