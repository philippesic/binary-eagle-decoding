# Candidate-D first-seed FFN stage parity

The bounded Apple M3 Max CPU diagnostic recaptured the frozen reasoning
training prompt with four additional native draft FFN graph taps. It used the
pinned FP16 target, candidate D, eight output tokens, CPU ggml backend and
the existing first-seed graph/head join (execution 2, token column 0).
The new capture's prompt and request bytes, eight raw output IDs, joined
head state, `post_attn_norm-0` input and `ffn_out-0` output match the archived
reasoning capture bitwise. Its manifest SHA256 is
`97f9af6e03914edde054e123315d77ad85b518443b7c4be445fbd9f73f0224a7`;
the archived control manifest is
`99b9003698bcdf5505421a1667dac4d21a8545471d91330c04181c9f2a931d40`.

From that one identical native F32 FFN input, candidate D's **native-order**
binary gate and up projections each match **9,728/9,728** F32 values bitwise.
The first difference is the gate SiLU: **6,557/9,728** values match, with
maximum absolute difference `4.76837158203125e-7` and RMS
`2.0127936319918706e-8`. The post-SiLU product matches **6,772/9,728**
(maximum `9.5367431640625e-7`), and the down output matches **5/2,560**
(maximum `6.103515625e-5`, RMS `1.0246000302166139e-5`). With grouped
matmul, the first difference is already at gate (**9,101/9,728** exact,
maximum `9.5367431640625e-7`); up is **9,049/9,728** exact. The ordered
mode therefore isolates the first reasoning FFN difference to the SiLU
implementation on this input, rather than packed weights or gate/up reduction.

The ignored machine-readable report is
`results/recurrent-ffn-stage-capture-20260928/stage-parity.json` in the main
checkout; it records all stage metrics, tensor hashes, source/build hashes,
versions and capture ledgers. The stage comparator checks every sealed file,
the pinned target/D/train identities, the archived control manifest, raw IDs,
and unchanged first-seed boundaries before comparing arithmetic. Eight
synthetic tests, Ruff, and the CPU native build passed. The local capture
server exited cleanly. No GPU, training, final prompts or Q4_0 evaluation
were used. This single-column Apple CPU result does not establish CUDA or
SM75 parity, a training tolerance, or an all-body optimizer budget.

Next, inspect the pinned ggml CPU SiLU path and run a small same-input
elementwise arithmetic probe against the captured gate and SiLU vectors.
Keep any training tolerance and exact-versus-trajectory choice with the user.
