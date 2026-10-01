# A1 computation implementation

Date: 2026-10-01. Preparation-only CPU work, Apple M3 Max arm64,
PyTorch 2.14.0, CPython 3.11.15, F32 projections and gradients. No CUDA
discovery, accelerator execution, model/data loading, or real-data updates.
The only optimizer update was one tiny synthetic AdamW equivalence check.
Q4_0 EAGLE remains the inference acceptance/latency/throughput baseline.

## Deliverable and API

`RowBinaryLinear(..., a1_computation="reference")` remains the default. Its
attached A1 computation retains the historical two-forward-product surrogate
arithmetic and bit-exact gradient tests. A4/A8/A16 computation is unchanged.

All A1 calls under `torch.no_grad()` now use one native projection. They skip
the unused surrogate product and dequantized activation tensor/saturation
mask, retaining input validation, the detached F64 mean-absolute reduction
rounded to F32, raw F32 sign-bit handling, and the saturation diagnostic zero.
Both signed zeros map positive; negative subnormals remain negative. The
order is integer-valued F32 sign dot, weight scale, activation scale, bias.
A positive-zero epilogue preserves the historical surrogate cancellation
epilogue's output zero bits, including negative dots at zero weight scale.

`a1_computation="single_forward"` explicitly enables the custom autograd
implementation for attached A1 calls. The selected attribute can be set before
a fresh round/graph. Configuration, installation, checkpoint metadata and
experiment identity wiring belong to integration; this worker did not change
those systems. The attribute is Python configuration, not a state-dict tensor.
Persist its identity and reject mismatched resume before exposing this option
in a persistent training launcher. This is an experimental computation option,
not a native-ready or exact-resume claim.

The custom function saves actual dequantized activation Q, attached shared hard
weight signs S, and effective weight scale alpha. With upstream G it computes
T=G-transpose Q, dS=alpha T, dAlpha=row-dot(T,S), and dQ=(G alpha) S. Gradients
pass through the original activation identity STE, inclusive latent mask
`abs(latent)<=1`, and effective-scale node with derivative one at raw scale
zero. Raw/native activation signs and detached beta never supply backward
values. There is no division by alpha or beta. Zero weight scales can revive;
zero activation scales still permit input gradients. Optional trainable bias
is supported by the private function, while module bias remains frozen.

## Acceptance checks and results

Independent integer/raw-bit references cover vectors, multiple leading
dimensions, noncontiguous inputs, signed zeros, positive/negative subnormals,
default F64 dtype, tails and scale/bias order. F32 forward values are exact;
the zero-output fixture additionally compares raw integer bits. Instrumented
forward counts show reference attached A1=2 products, custom attached A1=1,
and no-grad A1=1. A no-grad mock rejects any attempted activation-Q creation.

The declared custom-gradient gate is relative L2 <=1e-5, plus absolute <=1e-6
for near-zero entries (`assert_close` rtol=1e-5/atol=1e-6). First gradients
match the historical surrogate within this gate, including negative/raw-zero
scale rows, zero inputs, latent values at and outside +/-1, requested gradient
subsets, and intentionally different surrogate-Q/native signs. A 30-seed tiny
fixture scan found maximum relative L2 error input/latent/scale of
0 / 9.18e-8 / 6.79e-7; maximum absolute errors were
0 / 1.19e-7 / 1.91e-6 (the latter was away from the near-zero region).
This is bounded F32 drift, not bitwise equivalence.

One clipped synthetic AdamW step retains latent signs, effective scales within
the declared tolerance, and toy output argmax. A final-position-only CE reaches
earlier recurrent state and attached K/V through explicit F16-write/F32-read
casts, with shared round signs cleared after the round. Private-function
double backward matches the defined surrogate's second derivatives on a tiny
continuous fixture; finite-difference gradcheck of the hard forward is not a
valid check of the specified STE.

Commands (run in `/private/tmp/eagle-qat-computation`):

```sh
PYTHONPATH=src /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest tests.test_qat_computation tests.test_recurrent_a1_native_forward tests.test_recurrent_qat tests.test_recurrent_binary tests.test_recurrent_continuity
PYTHONPATH=src /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest discover -s tests -p test_continuous_qat.py
/Users/pippo/github/binary-eagle-decoding/.venv/bin/ruff check src/w1a1_eagle/recurrent_qat.py tests/test_qat_computation.py
git diff --check
```

Results: 38 focused tests and 14 continuous-QAT tests passed; lint and whitespace
checks passed. An initial package-style invocation of `tests.test_continuous_qat`
failed importing its existing sibling `test_recurrent_provider`; using the
suite's discovery invocation resolved that harness-path issue. No source fix
was required for the failed invocation.

## Small-operation reduction/fusion evaluation

The custom scale VJP uses a batched row dot through `torch.bmm`, avoiding the
explicit O-by-K `T*S` product tensor. It still needs dense T and dS; no peak
memory benefit is asserted without a real workload measurement. Compared with
`einsum`, the explicit bmm avoids extra dispatcher/shape-processing overhead.

Bounded CPU microcheck: one thread, 20 warmups, median of seven groups of 200
calls, F32 random operands, O/K=96/129 and 192/256. Median microseconds:

| O/K | `(T*S).sum(-1)` | einsum row dot | explicit bmm row dot |
| --- | ---: | ---: | ---: |
| 96/129 | 2.989 | 9.024 | 5.789 |
| 192/256 | 9.624 | 24.955 | 21.277 |

The explicit product sizes avoided by row dot are 49,536 and 196,608 bytes,
respectively; these are tensor-size counts, not allocator peak measurements.
Maximum reduction difference between row dot and multiplication/sum was
6.10e-5 and 1.53e-4 for the positive reductions, within relative F32 rounding
but not bit-exact. CPU row dot is slower than multiplication/sum at these
shapes. Retain it only as an explicit experimental memory/performance tradeoff
until real GPU checks decide. No regional compilation, autocast, TF32 change,
packed forward kernel, or whole-loop fusion was enabled. Sharing quantized
Q/K/V or gate/up inputs may be a future regional optimization, but needs an
explicit quantizer/observer interface and a measured launch/traffic bottleneck.

## Remaining integration and deployment gates

1. Root integrates the three owned files and records the computation selection
   in config/checkpoint/runtime identity; keep reference as the default.
2. After GPU ownership is separately granted, validate native prefix state,
   F16 K/V, proposal decisions, and gradients on actual RTX5080 SM120. Preserve
   the existing held-out ancestry and frozen target/verifier. CPU arithmetic
   counts do not establish GPU latency or SM75 performance.
3. On matched representative synthetic/captured train rounds without persistent
   real optimizer updates, enforce finite gradients, the declared numerical
   gate, native state/logit RMS and material-choice gates, and reserved-memory
   limits. Measure full paired-step median/p95 and peak allocated/reserved
   memory, including the dense T/gradient buffers. The audit proposes >=10%
   median paired-step improvement without p95 regression for enabling it.
4. Do not silently reinterpret a frozen reference-arithmetic experiment as
   custom arithmetic. Reassociation can alter scale updates near cancellation;
   CPU gates alone do not authorize default deployment or exact resume.
