# Output-preserving post-RoPE target K on RTX 5080

This bounded native Qwen3 block-0 probe observes one post-RoPE
`Kcur-0` tensor on the frozen 29-token code/data-validation training
prefix. The callback requests only the tensor whose operation is RoPE;
earlier projection and reshape nodes share the name and are skipped.
The capture includes the complete block-0 output as a fidelity gate.

The supervised RTX 5080 (SM120) F16-target run captured one F32
post-RoPE K tensor with 29,696 values. Its block output matches the
sealed native layer-1 input **74,240/74,240 F32 values bitwise**, and
the output payload is byte-identical to the prior output-only capture.
This makes the new K tensor usable with the existing output-preserving
post-RoPE Q and V captures for same-input attention attribution. An
earlier all-tap K-RoPE payload has different bytes; that run's Q
callback altered block output, so its K values remain excluded from
server-path attribution.

The ignored report is
`checkouts/target-block0-operator-20260928/runs/target-k-rope-a-20260928/comparison.json`
on the registered WSL host, SHA256
`c3fc11f35c995c95b75302906f37a4bf52489c70ca3351fc053a6b7b1fa6a2cf`.
It seals the frozen ladder, GGUF, callback mode, tensor payload and
compiled CUDA libraries. The K payload SHA256 is
`5263643cd41dff2f4b3313b693df706cc406c56f698956dc4bbbe28d6b34a24b`.
The ggml CUDA library SHA256 is
`59c8b5c4cccaaab2fd3721ee6f8d50b37b8b7ab6b62b6c995978fa84e8b9ef1f`.
The supervisor exited zero and released its process group; GPU use
returned to 0% utilization and 1,372 MiB whole-device baseline.
Local C++ syntax and Ruff checks passed. No optimizer, final prompt
or Q4_0 serving evaluation ran. This single SM120 training-prefix
capture does not establish general target-feature parity, attention
backend parity, SM75 performance or a training tolerance.
