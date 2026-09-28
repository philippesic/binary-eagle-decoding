# Output-preserving target block-14 stages on RTX 5080

This bounded native Qwen3 capture extends the existing 29-token
code/data-validation training-prefix probe to block 14. It records
F32 attention norm, pre-O Flash Attention output, post-attention
residual, FFN norm, FFN output and complete block output. The frozen
layer ladder supplies an independent layer-15 input for the full
block-output fidelity gate. The target GGUF and old block-0 modes
remain pinned.

The supervised RTX 5080 (SM120) F16-target run captured all six
requested tensors (1,959,936 bytes). Its `l_out-14` payload matches
the sealed native layer-15 input **74,240/74,240 F32 values
bitwise**. This makes the other five stage tensors usable for a
same-input block-14 intervention. It does not yet say whether the
position-3 amplification occurs in attention, FFN or their residuals.

The ignored report is
`checkouts/target-block0-operator-20260928/runs/target-block14-stages-a-20260928/comparison.json`
on the registered WSL host, SHA256
`8caeabd9313f090725de7b611848216d60fee6e603f81ca89b77775ee4686827`.
It records the frozen ladder, target GGUF, capture mode, tensor
payloads and compiled CUDA library hashes. The ggml CUDA library
SHA256 is
`59c8b5c4cccaaab2fd3721ee6f8d50b37b8b7ab6b62b6c995978fa84e8b9ef1f`.
The supervisor exited zero and released its process group; GPU use
returned to 0% utilization and 1,372 MiB whole-device baseline.
Local C++ syntax and Ruff checks passed. No optimizer, final prompt
or Q4_0 serving evaluation ran. This one SM120 prefix does not
establish SM75 performance, later-block parity or a training
tolerance.
