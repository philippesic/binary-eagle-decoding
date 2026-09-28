# Output-preserving pre-O attention tensor on RTX 5080

This bounded native Qwen3 block-0 probe captures `kqv_out-0`, the
F32 Flash Attention result immediately before the O projection, on
the frozen 29-token code/data-validation training prefix. The
callback also captures the complete block output as its fidelity
gate. The target GGUF, prefix and prior layer ladder remain frozen.

The supervised RTX 5080 (SM120) run captured one 118,784-value F32
attention-output tensor. All **74,240/74,240** block-output F32 values
match the sealed native layer-1 input bitwise, and the output payload
is byte-identical to the earlier output-only capture. The pre-O
tensor is therefore suitable for same-input attention and O
projection comparisons, alongside the separately output-preserving
post-RoPE Q/K and V tensors. Its own arithmetic has not yet been
replayed independently in this report.

The ignored report is
`checkouts/target-block0-operator-20260928/runs/target-attn-output-a-20260928/comparison.json`
on the registered WSL host, SHA256
`3150ec96dbca949ef0564deff898fb65cef2b8fd4ae3eaea339651e43b7ac35d`.
The `kqv_out-0` payload SHA256 is
`ba9d7fe1f867f31f2bd918512a7cf9ee55a974a0b816a1d89925067a842368b2`.
The report seals the callback source, payloads, target GGUF, ladder,
compiled helper and CUDA library identities. The CUDA library SHA256
is `59c8b5c4cccaaab2fd3721ee6f8d50b37b8b7ab6b62b6c995978fa84e8b9ef1f`.
The supervisor exited zero and released its process group; GPU use
returned to 0% and 1,372 MiB whole-device baseline. Local C++
syntax and Ruff checks passed. No optimizer, final prompt or Q4_0
serving evaluation ran. This one SM120 training-prefix capture does
not establish SM75 performance, later-block parity or a training
tolerance.
