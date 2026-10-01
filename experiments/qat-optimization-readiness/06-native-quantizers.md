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

## Optional fusion correction extension

Manifest schema 4 adds the seven-field `fusion_correction` v1 descriptor and
`activation_quantizers` (null for the fixed quantizer, otherwise the same v1
object). Correction arrays are exactly named `fc.correction_u.weight` F16
[out,rank], `fc.correction_v.weight` F16 [rank,in], and optionally
`fc.correction_bias` F32 [out]. Rank is 1 or 4. The descriptor pins names,
rank, arithmetic and the optional positive bias bound. Extra/missing arrays,
wrong shapes/dtypes, nonfinite factors, and out-of-bound bias fail export.
GGUF stores descriptor fields under `eagle3.fusion_correction.*`, with an
explicit empty bias name and zero bound when disabled. Native load rejects
unknown/malformed descriptors, missing/type/shape mismatches and unversioned
correction tensors. Enabling correction forces finite tensor validation;
after load, the bias is checked against its bound.

The encoder retains raw prequantization FC input and adds U(Vx) after the binary
projection, then bias. Stored F16 factors are promoted to F32 for the ordinary
MUL_MAT graphs with F32 precision: CPU F16 MUL_MAT otherwise rounds its F32
right-hand side to F16. Raw input and rank intermediate therefore retain F32,
matching the Torch correction contract, without introducing a custom op.
No correction metadata/tensors means the existing graph remains unchanged.

Final CPU fixtures pass 60 actual loader cases and 25 valid encoder numeric
checks across A1/A4/A8. Valid cases include default/fixed/learned quantizers,
zero correction, rank1/rank4 and optional bias. Invalid payloads include rank,
version, missing factor, shape, unknown field, nonfinite factor and exceeded
bias bound. Raw inputs include non-F16-exact values; absolute tolerance 1e-4
allows scalar versus SIMD F32 accumulation order. Nine standalone pack fixtures
add exact scale bytes, A1 signs, A4/A8 codes, A4 planes and A1 tail checks for
normal/zero/subnormal tokens, positive/negative thresholds and clipping; these
avoid tiny reconstructed outputs hiding bit errors in a floating tolerance.
The 187 operator fixtures still pass. Final Python export checks pass 7 legacy
and 4 focused learned/correction tests; Ruff and diff checks pass.

The correction's CPU loader/graph checks do not validate CUDA compilation,
SM75 behavior, deployment trajectories, GPU timing or quality. Those remain
integration/deployment gates, with no inference of a measured speed or quality
gain. Native commit and parent export commit are provided to the orchestrator
for integration; no primary checkout or submodule gitlink was changed here.

Post-review CUDA fixes are published as `01eb425e4` and `dc6d178b5`, following
`69a3697f7`. The CUDA build uses `-use_fast_math`; learned absmax reductions now
compare raw positive IEEE magnitude bits, and learned zero-limit/finite-inverse
checks use raw bits too. A1 delta-zero detection uses raw bits and F64-to-F32
beta/fallback conversion explicitly uses round-nearest intrinsics. This prevents
FTZ comparisons from turning a learned subnormal limit/threshold into zero.
Legacy A4/A8 reductions and reciprocal-overflow behavior are retained. Threshold
multiplication/subtraction already uses non-FTZ RN intrinsics. These are source
review fixes, with CUDA compilation and actual tiny-token bit/code checks still
required on the designated GPU; the local CPU evidence is unchanged.
