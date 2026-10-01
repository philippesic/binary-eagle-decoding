# Learned activation delivery

2026-10-01, bounded CPU implementation at baseline parent `5676254`. Ownership:
`src/w1a1_eagle/learned_activation.py`, `tests/test_learned_activation.py`, this
report. No existing QAT recipe, recurrent implementation, native submodule,
target/verifier, real data, or experiment ancestry was changed by this worker.
The orchestrator owns integration and the durable goal checkpoint.

## Delivered contract

`LearnedActivationQuantizer(bits, boundary, in_features)` owns exactly one F32
scalar: `threshold_delta` at A1, `clip_ratio` at A4/A8. The latter starts at one
and is projected to `[2^-16, 1]` after updates; the former starts at zero.
Projected direct clipping retains a nonzero recovery derivative at initializer
one. Optimizer integration must include each parameter once, without decay,
and call `bank.project_()` after the optimizer update. Finite checks fail rather
than silently replacing nonfinite parameters.

For A4/A8, hard arithmetic is `M=absmax(F32(x))`, `L=F32(M*c)`,
`s=F32(L/qmax)`, `u=F32(x*F32(qmax/L))`, then nearest-even rounding and symmetric
clamping, with qmax 7 or 127. Codes change when c changes. They are not unchanged
codes multiplied by an amplitude. When the F32 reciprocal overflows, the
versioned learned rule uses `F32((F64(x)/F64(L))*qmax)` before rounding. An exactly
zero limit yields zero codes/scales; this includes a nonzero subnormal whose
limit underflows. The ordinary F32 branch preserves baseline bytes at c=1.
The old Torch/native invalid-reciprocal behavior differs on pathological
subnormal rows and is not silently presented as parity.

For A1, `beta=F32(mean_F64(abs(F32(x))))`, `t=F32(delta*beta)`, then raw sign of
`F32(x-t)` determines ordinary +/-1 codes. Both signed zeros and exact threshold
ties are positive. At delta zero, subtraction is bypassed and the original raw
negative-subnormal sign is retained. Reconstruction remains +/-beta computed
from the original input; there is no reconstruction offset or new dot epilogue.

The attached functional API is
`learned_activation(input, bits, parameter, valid_mask=None, normalization_count=None)`.
It returns `ActivationResult(values, scale, codes, saturated, clipped)`; only
values differentiate. `saturated` identifies endpoint codes, while `clipped`
identifies inputs outside the learned interval (A1: outside sign-surrogate
support). `learned_activation_reference` returns the same hard arithmetic
without a gradient graph. `native_order_linear` provides a hard reference of
integer dot, then row scale, then token scale, then frozen bias. The I8 codes
are ordinary symmetric integer codes and retain the existing bit packing.

## Backward recipe and normalization

A4/A8 use clipped input identity, with input derivative one inside `abs(x)<=L`
and zero outside. The learned scalar surrogate is LSQ:
`dQ/dc=(M/qmax)*(q-u)` inside, `(M/qmax)*q` outside, normalized by
`1/sqrt(N*qmax)`. The implementation uses algebraically equivalent
`(Q-x)/c` or `Q/c`, avoiding infinity-times-zero on subnormal rows.

A1 uses input derivative one when `abs(x-t)<=beta`, zero outside, and threshold
derivative `-beta*support/sqrt(N)`. Zero rows have zero output, zero parameter
gradient, and input derivative one. These surrogates intentionally differ from
the baseline's unconditional identity activation STE; future quality comparisons
must label the whole recipe or add a frozen-parameter control with the same
surrogate.

N counts valid feature elements in one quantization invocation. QKV/gate-up
consumer count is never included. Contributions from shared consumers accumulate
on the same parameter. Passing the same full-batch N to chunks reproduces the
unchunked parameter gradient. A boolean token-shaped valid mask excludes padding
from both N and backward; it leaves forward arithmetic explicit and unchanged.
Repeated recurrent invocations normalize individually unless the caller supplies
a common larger N. This choice is serialized in checkpoint identity.

## Ownership and integration APIs

`LearnedActivationBank(bits, in_features_by_linear_path)` requires all nine row
projections. Six boundaries are `fc`, `qkv`, `attn_output`, `gate_up`, `down`,
and `head`. Q/K/V share one module object and parameter; gate/up share another.
`bank.attach(linears)` validates row precision, widths, devices and the entire
graph before mutation. `bank.validate_attachment(linears)` catches broken
consumer aliases. `LearnedActivationBank.from_attached(linears)` reconstructs a
bank using the existing six object references, with no new parameters; it rejects
unknown types, mixed precision, mismatched boundaries, broken sharing, and
cross-boundary parameter aliases. Root can register `drafter.qat_activation_bank`
for canonical checkpoint/optimizer ownership.

`bank.checkpoint()` stores scalar F32 tensors and exact version/recipe/bits/
boundary/width/consumer/gradient-normalization identity. `load_checkpoint()`
validates every key/value before copying any parameter. Standard quantizer
state_dicts also serialize strict extra-state identity and reject incompatible
loads before copying the parameter. Optimizer moments belong to the surrounding
optimizer checkpoint; CPU tests demonstrate exact continuation with AdamW.

Root integration still must route RowBinaryLinear through attached quantizers,
keep the native hard epilogue and surrogate gradients attached, extend config and
optimizer allowlists, deduplicate shared ownership, project after updates, and
serialize canonical bank payload plus optimizer state. No automatic monkey patch
of linear forward or optimizer construction is performed by this module.

## Native encoding and deployment gate

Aligned with the native owner: schema-v3 row checkpoint manifests add
`activation_quantizers={version:1,boundaries:{...}}`. Each of the six boundary
entries has `bits`, `threshold_delta`, `clip_ratio`; A1 requires clip one,
A4/A8 require threshold zero. `bank.native_parameters()` emits exactly this
payload and does not claim readiness. Native metadata must encode/load the
scalars and their shared consumer identity, use the original beta for A1,
apply the threshold before packing, apply the relative clip before A4/A8 codes,
and preserve integer-dot -> row-scale -> token-scale order. Delta-zero raw signs
and nonzero round-nearest subtraction/multiplication must survive CUDA FTZ;
native owner is implementing explicit rounding intrinsics and learned-only
overflow handling. Existing target/verifier and Q/K row permutation remain part
of the surrounding contract.

`bank.require_native_ready(evidence, backend=...)` fails without evidence bound
to SHA256 of the current canonical payload, a native Git revision, hardware and
matching backend. Evidence must report exact packing, relative RMS <=0.10, and
zero changed native choices whose margin exceeds 0.02. A CPU result authorizes
only CPU. Updating any scalar invalidates previous evidence. This module has
no flag that blesses defaults as CUDA-ready and no unsupported fallback recipe.
The tests use deliberately synthetic evidence only to test validation plumbing;
no actual native-readiness evidence is claimed here.

## Acceptance evidence and remaining work

Executed with `PYTHONPATH=src` and the project's existing `.venv/bin/python`,
macOS 27.0 arm64, Python 3.11.15, Torch 2.14.0; CPU F32/F64 only. The 21
focused tests pass: ordinary baseline initialization,
round-even ties, actual code/bit changes, signed zeros, negative subnormals,
zero/underflow/overflow rows, tail packing, analytical finite meaningful scalar
gradients, recoverable clip initializer, tiny synthetic optimizer effects,
masking/chunk normalization, shared gradient accumulation, attachment ownership,
strict/atomic checkpoint loads, exact AdamW parameter/moment resume, native-order
integer reference and fail-closed/stale evidence. Six existing recurrent QAT
tests also pass. `git diff --check` passes. No CUDA/Metal discovery, GPU work,
datasets, sealed evaluations, model training, or accelerator benchmark occurred.

Remaining acceptance belongs to integration: native exporter round trip and CPU
packer equality, attached recurrent/later state/K/V gradients, current-student
native decision gates, optimizer/resume graph audits, and authorized GPU tests
on declared hardware. CUDA Torch reference currently computes the F64 reciprocal
overflow fallback for all rows to avoid a host synchronization; measure its
cost before enabling large workloads or specialize it safely after evidence.
No training speed, memory saving, quality gain, online acceptance, or SM75
throughput is inferred from these tests. Q4_0 EAGLE remains the primary native
comparison baseline.
