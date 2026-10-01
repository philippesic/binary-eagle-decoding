# Learned scalar quantizer export and native contract

The learned row export uses manifest schema 3 with `activation_rule`
`learned_scalar_a1_threshold_a4a8_clip_v1` and `activation_quantizers` version 1.
Exactly six shared boundaries are required: `fc`, `qkv`, `attn_output`,
`gate_up`, `down`, `head`. Each declares matching `bits`, effective F32
`threshold_delta`, and effective F32 `clip_ratio`. A1 permits finite delta and
requires clip 1. A4/A8 require delta 0 and clip in (0,1]. Missing/unknown fields,
nonfinite, non-F32-effective, incompatible bits/scalars fail closed. A16 cannot
silently accept this learned contract. The manifest pins the NPZ hash; quantizer
scalars live in the manifest, with the existing 18 weight arrays unchanged.

GGUF stores these values as F32 under `eagle3.w1a1.activation_quantizer.*`, with
strict version and six-boundary inventory. Existing row metadata remains full
version 2; its strict optional learned metadata activates the learned APIs.
Q/K row permutation, nine projection signs and scales, and frozen tensor bytes
retain the existing serialization audit. Native load rejects an incomplete,
unknown, or incompatible learned contract. Runtime activation bits must match.

A1 computes original F64 meanabs -> F32 beta, F32 delta*beta threshold, then
raw sign of the rounded F32 subtraction; delta zero bypasses subtraction to
preserve signed-zero and negative-subnormal behavior. Reconstruction retains
original beta. CPU and CUDA direct/shared paths carry scalars in operator
parameters. Shared EAGLE graph packing keys input, bits and both scalars, keeping
QKV and gate/up packs shared. Integer dot -> row scale -> token scale is intact.
A4/A8 compute F32 limit=absmax*clip, beta=limit/qmax, and normal F32 x*(qmax/limit)
RNE symmetric codes. Learned-only reciprocal overflow uses F64 x/limit*qmax ->
F32 before RNE/clamp; limit zero produces zero codes. Legacy wrappers preserve
existing reciprocal-overflow behavior. No graph subtraction shortcut is used.

Native commit: `cce962891` on `feature/learned-w1ax`, to be pushed to the user's
fork before integration. CPU build used Apple Silicon, release, Metal/Accelerate/
BLAS off, two compiler jobs. `test-backend-ops -b CPU -o W1A1_MUL_MAT` passed
187/187 (54 learned cases across direct/shared, default/nondefault, zero/tiny
inputs, K=1/33/2560). `test-eagle3-learned` passed 24 tiny actual-loader cases,
including valid legacy/default/nondefault encoder arithmetic and invalid
version/nonfinite/missing/extra/incompatible cases. Python learned export suite
passed 9 tests including imported legacy serialization fixtures.

Remaining gate: CUDA source is implemented but not compiled/run locally; no GPU
or remote action occurred. CPU does not establish SM75 behavior or performance.
Before deployment run bounded actual CUDA code/bit, trajectory numeric and
native-choice gates (relative RMS <=0.10; no choice change with margin >0.02),
then packing-inclusive native acceptance/latency against Q4_0. No model data,
sealed finals, or real-data optimizer updates were used.
