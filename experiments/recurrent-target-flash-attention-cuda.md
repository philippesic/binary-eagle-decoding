# Same-input target Flash Attention and O projection on RTX 5080

This bounded ggml CUDA replay consumes the output-preserving native
block-0 post-RoPE Q/K and V tensors for the frozen 29-token
code/data-validation training prefix. It reconstructs the native
256-slot F16 K/V cache layout, F16 causal mask with `-inf` for future
or padded slots, F32 Q, F32-accumulation Flash Attention, pinned F16
`blk.0.attn_output.weight`, and F32 layer-input residual. The O weight
bytes match the HF source. The input captures, target GGUF, candidate
D and old layer ladder are checked by hash.

| Standalone RTX 5080/SM120 output vs output-preserving native `ffn_inp-0` | Exact F32 / 74,240 | Position-3 relative row L2 |
| --- | ---: | ---: |
| ggml CUDA, native F32 Q | **74,240** | 0 |
| ggml CUDA, Q rounded to F16 then returned to required F32 | **74,240** | 0 |

The two outputs are bitwise identical. This validates the combined
Flash Attention, O projection and residual graph under these frozen
operands and geometry. It does not separately identify which of
attention, projection or residual arithmetic accounts for the earlier
Torch eager path's 0.2048% position-3 error after native Q/K/V
substitution. The Q-roundtrip observation applies to this ggml path
and prefix; it is not a general permission to change target precision.
An attempted direct F16-Q diagnostic was stopped by the pinned CUDA
kernel's assertion requiring F32 Q. Its supervised failure is preserved
as an ignored diagnostic; the valid test rounds Q to F16 and back to
F32 before entering the kernel.

The final supervised run is
`checkouts/target-block0-operator-20260928/runs/target-flash-attn-b-20260928`
on the registered WSL host. Its ignored `comparison.json` SHA256 is
`ae265f07976acdd5954023a5697bb5ffe6bf7cde38101f051198dad8fef2a0a1`.
The report records exact operands, source/capture/helper/library
hashes, hardware, precision and both full-row metrics. The used ggml
CUDA library SHA256 is
`59c8b5c4cccaaab2fd3721ee6f8d50b37b8b7ab6b62b6c995978fa84e8b9ef1f`.
The final supervisor exited zero and released its process group; GPU
use returned to 0% utilization and 1,372 MiB whole-device baseline.
Local C++ syntax and Ruff checks passed. No optimizer, standalone
target full-model forward, final prompt or Q4_0 serving evaluation
ran. This is one SM120 training-prefix operator case, not SM75
performance, later target-feature parity or a training tolerance.
