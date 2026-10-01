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

## Opt-in affine binary weight extension

The user-approved affine weight path is `mu + alpha*sign(z)` per output row,
with `mu` initially zero. Manifest schema 5 requires `activation_quantizers`
(null or the existing v1 object), `fusion_correction` (null or its descriptor),
and `affine_weights` exactly `{version:1, coverage:'fusion'|'all', arithmetic,
tensors}`. Arithmetic is
`integer_dot_alpha_beta_plus_integer_sum_midpoint_beta_before_bias_f32`.
Coverage is exactly FC or all nine selected GGUF bases. Each maps to its named
F32 row vector `<base>.w1ax_midpoint`; extra/missing names, nonfinite values and
shape/type mismatches fail closed. Q/K midpoint rows use exactly the alpha
permutation. The existing checkpoint hash covers midpoint payloads; the export
report records hashes of the permuted GGUF midpoint arrays. Training midpoint
regularization has no inference/export effect.

The native v1 contract under `eagle3.affine_weights.*` pins version, coverage,
arithmetic, bases and midpoint tensor names. Native load rejects missing,
unknown, unversioned, malformed or nonfinite payloads and forces data validation.
The affine-only pack extension appends one F32 S slot per token after the
existing scale slots; disabled layout/headers and arithmetic stay unchanged.
A1/A4/A8 S is an integer sum of the exact emitted sign/code values, converted to
F32. A16 S sums the same rounded F16 boundary inputs in sequential F32 order,
with beta 1. The fused epilogue is `(D*alpha)*beta + (S*mu)*beta`; A16 uses
`D*alpha + S*mu`. No extra dense matrix is introduced. CUDA uses one bounded
code-sum kernel per shared pack and explicit RN epilogue products/additions;
CPU computes the sum inside the token pack. Affine graphs always share the
pack and S for QKV and gate/up by input/quantizer/affine identity, regardless of
the optional legacy shared-pack environment toggle. Optional FC correction and
its bounded output bias are applied after the affine binary result.

Native commit `3db933346` is published on the existing worker branch. CPU
checks pass 220 operator cases (33 affine additions), 108 actual EAGLE loader
cases, 53 valid encoder numeric checks, and 21 standalone pack cases with exact
code-sum/scale bytes, sign/code/plane/tail checks. They cover fixed/learned
quantizers, fusion/all coverage, optional correction, A16, zero mu, alpha zero
with nonzero mu, zeros/subnormals, and malformed/nonfinite midpoint payloads.
Python passes 7 legacy and 6 learned/correction/affine serialization checks;
Ruff and diff checks pass. Raw native outputs are
`/private/tmp/eagle-native-affine-ops.log` and
`/private/tmp/eagle-native-affine-final-fixture.log`; compilation log is
`/private/tmp/eagle-native-affine-final-build.log`.

Remaining gates are unchanged: CUDA source compilation, actual GPU exact
code/S/epilogue checks and bounded native trajectories/choices, then actual
packing-inclusive latency/acceptance/throughput against Q4_0. CPU evidence does
not establish CUDA/SM75 correctness or speed. No GPU/remote actions, real-data
updates, or sealed-final reads occurred. Both isolated worktrees remain intact
for integration or validation repairs.

## Explicit backend fixture gate

Native follow-up `524427ed3` adds `test-eagle3-learned --backend CPU|CUDA`,
defaulting to CPU. CUDA selection requires a CUDA-enabled fixture and a device
from the `CUDA` backend registration; unavailable or incompatible selection
exits 2 and refuses fallback. Direct exact-pack graphs allocate their buffers
on the selected device and execute on its explicit backend. Valid encoder
models pin that device with full offload; a scheduler evaluation callback
checks every pack/W1Ax/matmul/conversion/add node's actual execution backend
and cancels on fallback. Invalid metadata/data loader cases remain CPU checks.
The summary separates exact pack cases, loader cases, actual encoder graph
cases and audited arithmetic nodes, and prints device/hardware identity.

The operator's actual GPU command after compiling with `GGML_CUDA=ON` is:

```sh
./build/bin/test-eagle3-learned --backend CUDA
```

This is the exact sign/code/scale/S gate, complementing the floating operator
suite; `test-backend-ops` alone cannot establish packed-byte correctness.
The explicit CPU command and unchanged default invocation both passed locally:
33 exact pack cases, 108 loader cases, 53 encoder graphs, 179 audited arithmetic
nodes on Apple M3 Max CPU. The operator suite still passes 220/220. The CPU
build's explicit CUDA invocation was checked to reject before backend
initialization with exit 2. No actual CUDA compilation or execution occurred.
Logs are `/private/tmp/eagle-native-backend-final-cpu.log`,
`/private/tmp/eagle-native-backend-default.log`,
`/private/tmp/eagle-native-backend-final-ops.log`, and
`/private/tmp/eagle-native-backend-final-build.log`.

Exact packs now include 12 additional fixed-affine variants and tokens mixing
signed zeros with positive/negative subnormals. The opt-in affine pack uses the
same safe raw-magnitude/reciprocal-overflow normalization as the learned pack,
even with fixed quantizer parameters; this avoids CUDA fast-math FTZ changing
S. Legacy non-affine packing remains unchanged. A1 F64 beta and RN conversion
already use raw magnitude bits/non-FTZ intrinsics. Affine fixed hard-code
reference implementations must match this explicit safe tiny-token rule;
ordinary finite-range codes are unchanged. Native CUDA correctness remains an
open gate until the designated GPU operator runs the explicit CUDA command.

## Granular measurement JSON

Published native commits `0e37ba740` and `8025a0777` add optional
`--json-report <newpath>`. It creates exclusively and checkpoints completed
measurements with status `failed` until all checks and backend cleanup complete;
then status becomes `passed`. Existing report files are refused without change.
Schema 1 is explicitly `input_scope: synthetic_operator`, with little-endian
raw hex, command argv, actual requested/backend/device/hardware identity and
informational compiled runtime version/short commit/compiler/target. It invents
no train prompt, capture ancestry or deployment context. The collector must
bind the actual full source, executable, recipe and deployment hashes separately.

`pack_cases` contains expected/native complete packed bytes, layout offsets,
raw input and quantizer scalars. `projection_cases` measures all nine selected
base identities with six actual shared packs at odd synthetic widths; it includes
code/scale/S/output hex, alpha/midpoint/weight values, paired baseline and zero-mu
outputs, and `pack_case_index` linking the same input and quantizer. A16 records
its explicit backend F16 boundary cast, F32 value sum and beta-one exception,
without treating those values as integer codes or adding an int64 kernel.
`encoder_cases` records expected/native output, raw FC input, expected base,
correction and bias, actual GGUF correction rank/bias, and actual arithmetic
node outputs/source bytes/device. Named `fc_correction_latent` and
`fc_correction_delta` provide raw-input and nonzero-delta witnesses. Zero-U cases
retain nonzero V; rank4 zero-U and rank4+bias cases are included. `loader_cases`
records actual admission/rejection. Consumers must derive checks from data,
never turn reported counters or mode labels into a numerical proof.

Final local CPU JSON `/private/tmp/eagle-native-json-cpu-6.json` is approximately
1.49 MiB and identifies compiled native head `8025a0777`, Apple M3 Max CPU.
57 pack byte pairs (33 scalar plus 24 six-boundary packs), 36 projection cases,
114 loader cases, 59 encoder graphs and 218 audited arithmetic nodes pass.
An independent local JSON check verified exact bytes/S, nonzero alpha-zero
midpoint outputs, exact zero-midpoint/baseline identity, raw FC input equality,
and relative RMS <=1e-4. Overwrite refusal preserves the prior SHA and exits 2.
The collector correctly refuses this CPU report as CUDA evidence. Raw logs are
`/private/tmp/eagle-native-json-final-pinned-cpu.log` and
`/private/tmp/eagle-native-json-final-pinned-build.log`.

CPU command: `test-eagle3-learned --backend CPU --json-report <newpath>`.
Future GPU command, **only after the user resumes GPU work**:
`test-eagle3-learned --backend CUDA --json-report <newpath>` with a CUDA-enabled
build. Both GPUs remain paused; no CUDA compilation, GPU execution or remote
queries occurred during this work. GPU numeric/trajectory/performance gates
remain pending.
