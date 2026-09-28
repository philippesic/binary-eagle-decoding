# Same-input target V projection on RTX 5080

This no-optimizer diagnostic replays Qwen3 target block-0 V projection
on the RTX 5080 (SM120). Its inputs are the 29 captured F32
attention-normalized rows from an **output-preserving** native callback
and the pinned F16 `blk.0.attn_v.weight` GGUF tensor. The F16 GGUF
weight bytes match the published HF source shard. The frozen code/data
training prompt, target GGUF, candidate D, old native layer ladder and
safe V capture are hash-checked before execution.

| Identical native input and weight, V output versus server | Bitwise exact / 29,696 | Position-3 relative row L2 |
| --- | ---: | ---: |
| Standalone ggml CUDA, F32 input | **29,696/29,696** | 0 |
| Standalone ggml CUDA, explicit F16-cast input | **29,696/29,696** | 0 |
| Torch CUDA/F16 `linear`, explicit F16-cast input | **5,351/29,696** | 0.0934% |

The two ggml outputs are also bitwise identical to each other. Torch's
maximum absolute difference from native is `0.00146484375`, with
0.1063% median relative row L2 across the 29 positions. Its metrics
reproduce the prior full-model same-input V intervention exactly. The
standalone ggml F32-input graph matching the output-preserving server V
tap is the fidelity gate: under this geometry, the residual Torch V gap
comes from the projection backend arithmetic/dispatch, not from
rounding the native norm input to F16. This observation does not identify
a specific CUDA instruction or establish the same explanation for K/Q
or later target blocks.

The supervised run `checkouts/target-block0-operator-20260928/runs/target-v-cuda-20260928`
on the registered WSL host exited zero and released its process group.
The ignored report `comparison.json` has SHA256
`39cd41fe114e0c42b8cd31598cbd54efef1abf97057428c0e55d15d45ede9747`.
It records the source/capture report hashes, exact F16 V weight and
native-norm operand hashes, compiled helper and CUDA library hashes,
and per-row/max/RMS metrics. The used CUDA library SHA256 is
`59c8b5c4cccaaab2fd3721ee6f8d50b37b8b7ab6b62b6c995978fa84e8b9ef1f`.
The run ended at 0% GPU utilization and 1,372 MiB whole-device baseline
use. Local C++ syntax and Ruff checks passed. No optimizer, target
full-model forward, development/final prompt or Q4_0 evaluation ran.

This result is one frozen training prefix on SM120, not SM75 speed,
general target-feature parity, an accepted trajectory, or a training
tolerance. Exact versus predeclared numerical/trajectory policy and any
all-body budget remain user-owned.
