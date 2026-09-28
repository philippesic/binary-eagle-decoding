# Candidate-D FFN replay with captured native SiLU

On the same sealed reasoning first-seed CPU column, the earlier ordered
binary gate and up projections each matched all 9,728 native F32 values.
The [SiLU arithmetic check](recurrent-binary-silu-arithmetic.md) matched
all native SiLU values by calling the pinned ggml CPU vector function.
This follow-up feeds the **captured native SiLU** into the F32 product with
the captured native up vector, then runs the candidate-D ordered A16/W1
down projection.

| SiLU input | Product bitwise exact | Product max / RMS | Down bitwise exact | Down max / RMS |
| --- | ---: | ---: | ---: | ---: |
| Torch SiLU control | 6,772/9,728 | 9.5367e-7 / 3.3643e-8 | 5/2,560 | 6.1035e-5 / 1.0246e-5 |
| Captured native SiLU | 9,728/9,728 | 0 / 0 | 2,560/2,560 | 0 / 0 |

The Torch control reproduces the prior sealed stage metrics exactly. With
the native SiLU vector, both remaining stages match bitwise. This closes
the **one-column Apple M3 Max CPU FFN arithmetic comparison** from the
identical captured FFN input through gate, up, SiLU, product and down. It
does not establish recurrent whole-drafter, CUDA/SM75, or training
trajectory parity.

The probe verifies the capture ledger, candidate-D GGUF SHA256
`10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf`,
the native stage hashes, and the prior SiLU report. Pinned fork revision is
`21f617d4ef3f5dc383d3ab8dc619daaa87db7ff8`. The ignored machine
report is `results/recurrent-ffn-stage-capture-20260928/native-silu-product-down.json`
in the main checkout, SHA256
`3c19a62baff2b2afb744462def71709db998e407b6f1287c0b909ac7e2b9f7ef`.
The run used NumPy 2.4.6 and Torch 2.14.0. Ruff lint/format and the
real-input replay passed. No local model server, GPU job, optimizer, final
prompts or Q4_0 evaluation ran.
