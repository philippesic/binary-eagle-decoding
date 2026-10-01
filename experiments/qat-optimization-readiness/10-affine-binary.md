# Affine binary midpoint preparation

The human authorized optional affine asymmetric binary **weights** on
2026-10-01: `w[r,i] = mu[r] + alpha[r] sign(z[r,i])`, with nonnegative alpha
and one F32 trainable midpoint per output row. This deliverable owns the new
`affine_binary.py` helper and synthetic CPU tests. It introduces no activation
threshold or extra sign plane. Q4_0 EAGLE remains the primary comparison target.

## Implemented contract

`AffineBinaryConfig` defaults to disabled, fusion coverage, midpoint LR 1e-5,
and L2 coefficient 1e-6. Coverage is `fusion` or `all`; invalid switches,
nonfinite rates and negative regularization fail closed. Midpoints initialize
to exact F32 zero. They are unrestricted unless the user explicitly configures
`midpoint_bound`; projection rejects nonfinite/F32-invalid masters and clips
only an explicit bound. The regularizer is `mild_l2 * sum(mu**2)`.

`install_affine_binary(linears, target=..., config=...)` validates the exact
nine-path row-binary inventory and target module/storage aliases before any
mutation. Enabled rows receive the registered child `linear.affine_binary`,
whose `midpoint` parameter is also owned by the returned `AffineBinaryBank`.
The bank exposes `declared_paths`, `midpoints`, `parameter_group`, `project_`,
`regularization_loss`, strict state load/save and native/manifest payloads.
Disabled installation adds no child, trainable or native tensor.

The row-forward integration calls
`affine_binary_projection(values, signs, alpha, midpoint, codes=codes,
beta=beta, bias=frozen_bias, single_forward=...)` with the **same** quantized
values, integer codes and beta produced by the existing input boundary.
The helper never requantizes and never materializes dense centered weights.
Native arithmetic is `(D * alpha) * beta + (S * mu) * beta`, then frozen bias,
where D is the existing code/sign dot and S is the exact code sum. For fixed
A16, the already F16-rounded input values supply D and S; beta is omitted.
All accumulation and epilogue arithmetic is F32 with autocast disabled.

`shared_affine_input_sums()` bounds the cache to a forward scope. Identical
quantizer results share a sum; distinct equivalent Q/K/V or gate/up result
nodes can use `cache_key=(original_input_tensor, bits, quantizer_identity,
parameter_versions...)`. Source identity, tensor mutation counters, precision
and complete quantizer metadata must describe an identical boundary. Strong
references prevent tensor identity reuse; separate scopes and changed source,
result or quantizer versions do not share. Inference tensors without mutation
counters bypass the cache. The root scopes this helper around a single round.

One **native code sum** is performed per shared boundary. Training also saves
one shared actual-Q sum to preserve midpoint gradients if native codes and Q
differ through device subnormal handling or rounding. A16 uses one sum for both.
The single-forward implementation performs one existing projection GEMM and
no midpoint GEMM. Its custom VJP gives midpoint/input/quantizer gradients
through actual Q, plus the original alpha/sign surrogate gradients. In
particular, nonzero mu still supplies input gradients when alpha is zero.
Equivalent quantizer nodes share the midpoint branch through their first
attached Q node; this can reassociate aggregate F32 gradients. The practical
CPU gate is `rtol=1e-5, atol=2e-7` versus the attached reference, not a bit-exact
training-gradient claim. Native forward arithmetic matches exactly in fixtures.

## Checkpoint and native handoff

State payload fields are exactly `identity`, `state`, `state_sha256`; identity
includes helper version 1, affine-binary-core kind, configuration, arithmetic
and declared dimensions. Every F32 row vector, inventory and content hash is
validated before mutation. Changed coverage/configuration and corrupt,
nonfinite, wrong-dtype or out-of-explicit-bound state fail closed.
`validate_affine_optimizer` requires exactly the supplied base parameters plus
each midpoint once, protecting target storage views. Root owns the additional
midpoint optimizer family and exact-resume runner plumbing.

Native descriptor is exactly:

```python
{
    "version": 1,
    "coverage": "fusion",  # or "all"
    "arithmetic": "integer_dot_alpha_beta_plus_integer_sum_midpoint_beta_before_bias_f32",
    "tensors": {"fc": "fc.w1ax_midpoint"},  # all nine GGUF bases for coverage=all
}
```

Native payload tensors are contiguous F32 output-row vectors named
`<GGUF base>.w1ax_midpoint`; they are independent midpoint vectors, never
binary sign-pack shadows. The native/export owner applies the same Q/K row
permutation as alpha. All-nine reference dimensions contain 65,280 output rows,
so additional midpoint storage is **261,120 bytes**, labeled
**affine binary core**. Small fixtures report their own actual storage.
The payload reports `requires_native_validation`; it cannot claim GPU readiness.
Root and native/export owner coordinate outer checkpoint schema 5.

## Acceptance checks and remaining work

Apple M3 Max CPU, F32 training masters/arithmetic with fixed A16 F16 boundary
casts; Python 3.11 shared local venv, Torch 2.14.0. The bounded command passed:

```sh
PYTHONPATH=src:tests /Users/pippo/github/binary-eagle-decoding/.venv/bin/python \
  -m unittest test_affine_binary test_recurrent_qat -q
```

**22 tests passed** (16 affine tests plus 6 existing recurrent-QAT checks).
The affine tests cover validated configuration; disabled/fusion/all inventory;
atomic target alias protection; unrestricted/default and explicit-bounded mu;
all A1/A4/A8/A16 arithmetic; zero activation, signed zeros and subnormal signs;
zero-mu forward identity including output bits; reference/single-forward VJPs;
zero-alpha input and scale revival; actual-Q versus intentionally differing
native codes; learned-factor gradients; shared code/Q reductions and GEMM
counts; mutation/version/quantizer cache separation; inference safety;
later-state/K/V gradients; exact optimizer ownership; strict state/native
inventory and tiny synthetic AdamW uninterrupted/resumed equality.

Root still owns row-forward/config/checkpoint/optimizer integration, combined
learned-quantizer and fusion-correction validation, and durable active-goal
checkpoint. Native/export owner owns schema 5 and midpoint runtime epilogue.
Actual GPU/native acceptance, memory, latency and throughput remain unmeasured.
No GPU, remote, real-data optimizer update, calibration fitting, sealed-final
read or deployment occurred in this feature.
