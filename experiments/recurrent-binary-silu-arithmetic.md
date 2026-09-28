# Candidate-D first-seed CPU SiLU arithmetic

The sealed reasoning FFN stage capture supplies one native F32 gate vector
(9,728 values) and its native SiLU output. Its gate matches the candidate-D
ordered binary projection bitwise. This probe calls `ggml_vec_silu_f32`
directly from the same pinned Apple M3 Max CPU build as the capture and
compares five elementwise choices on that unchanged gate input.

| F32 arithmetic | Bitwise exact / 9,728 | Maximum absolute error | RMS error |
| --- | ---: | ---: | ---: |
| ggml CPU vector | 9,728 | 0 | 0 |
| ggml CPU scalar tail (`n=1`) | 6,519 | 4.7684e-7 | 2.01198e-8 |
| Torch SiLU | 6,557 | 4.7684e-7 | 2.01279e-8 |
| Torch sigmoid then multiply | 5,358 | 4.7684e-7 | 3.44875e-8 |
| NumPy F32 exp then divide | 6,519 | 4.7684e-7 | 2.01198e-8 |

The pinned `eagle3.cpp` graph applies `ggml_silu` to the gate before its
product with up. On this CPU build, F32 `ggml_compute_forward_silu_f32`
calls `ggml_vec_silu_f32`; its AArch64 NEON branch evaluates a vector
approximation of `exp(-x)` followed by F32 addition and division. The scalar
tail uses `expf`. Calling the built vector function on the captured gate
reproduces every native SiLU output bitwise. Thus the previously observed
Torch SiLU difference is explained at this operator boundary on this row;
it is not evidence of a gate projection or packed-weight difference.

The probe verifies the capture file ledger, archived stage-report identity,
native revision and captured server/CMake hashes before reading the graph.
The ignored machine report is
`results/recurrent-ffn-stage-capture-20260928/silu-arithmetic.json` in the
main checkout, SHA256
`16e5f25b06d40ba039882dae801b3b9b7018b11483fbee59d6fb0d0cddfb479b`.
The native CPU library SHA256 is
`0d499359a40900596b172556bebd1aad1fa69aaf1cb3e46bf962a98ccf2a0600`;
the server and CMake hashes match the sealed stage report. Pinned fork
revision: `21f617d4ef3f5dc383d3ab8dc619daaa87db7ff8`; `vec.cpp` and
`vec.h` SHA256 values are
`cda0babb53f9c8ff1faa7459b2c49019d15e5970dfe0c613d58e2659e919f524`
and `926330bae1c5d003bd654035426e31381fafcdca23ffcc23201d219dbb97cbeb`.
The probe used NumPy 2.4.6 and Torch 2.14.0. Ruff lint and format checks
passed.

This is one Apple M3 Max CPU column, with no optimizer, CUDA/SM75, final
prompts or Q4_0 evaluation. The captured library hash is recorded here but
was not part of the original capture manifest. Whole-drafter parity and a
training tolerance remain open. A bounded next check can feed this exact
native SiLU vector into the product and ordered down projection to see
whether those stages then match, before returning to cache/state drift.
