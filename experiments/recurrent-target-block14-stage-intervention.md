# Block-14 stage comparison on RTX 5080

This bounded diagnostic compares the six output-preserving native
block-14 F32 tensors on the frozen 29-token code/data-validation
training prefix with source Qwen3 CUDA/F16 eager execution. The first
HF forward uses its accumulated layer-14 input. The second replaces
only that block input with the complete captured native F32 rows cast
to F16. Eleven block-14 weights and norms match the pinned GGUF
source. The prior layer-14 input/output intervention metrics reproduce
exactly, including the position-3 4.148% versus 0.149% output errors.

| Position-3 boundary vs native | Accumulated HF input | Native block input cast to F16 |
| --- | ---: | ---: |
| Block-14 input | 1.2092% | 0.0160% |
| Attention norm | 1.4334% | 0.0322% |
| Pre-O attention output | 2.6507% | 0.5099% |
| Post-attention residual / FFN input | 1.2157% | 0.1015% |
| FFN norm | 2.1931% | 0.1904% |
| FFN branch output | 12.1527% | 0.5045% |
| Complete block output | 4.1478% | 0.1491% |

These are relative row L2 errors with each native stage as its own
denominator; they are not additive. In absolute RMS units at position
3, the accumulated HF error is `0.011395` at FFN input,
`0.033592` at FFN branch output and `0.042182` at complete block
output. With native block input, those three errors are `0.000952`,
`0.001394` and `0.001516`. The post-attention residual remains near
the incoming error before the large FFN-branch increase, so the
observed block-14 amplification is concentrated in the FFN path on
this row. Attention also differs at pre-O; this table alone does not
prove which FFN operation or input direction creates the sensitivity.
A future native-FFN-input intervention could isolate that causal step.

The supervised RTX 5080 (SM120) run is
`checkouts/target-block0-operator-20260928/runs/target-block14-hf-stages-a-20260928`
on the registered WSL host. Its ignored `comparison.json` SHA256 is
`ad8e719a77667da86b74fc269d2a40cddab1b7cb66df84f12792a1b85d7c95c6`.
The report records all 29 row metrics at every stage, source/capture
hashes, eleven weight identities, hardware and software versions.
The supervisor exited zero and its process group stopped before the
user made the RTX 5080 unavailable. New GPU runs are locally paused;
the device's later 95% utilization was not caused by this stopped
supervised process. Local Ruff lint/format and Python
compilation passed. No optimizer, final prompt or Q4_0 serving
evaluation ran. This one SM120 prefix does not establish general
target-feature parity, SM75 performance or a training tolerance.
