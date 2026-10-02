# Fixed-state binary-head flip oracle

October 2, 2026. Bounded synthetic CPU diagnostic, source base `bb0d304`.
No model, dataset, capture, GPU, Metal, SSH, training, optimizer update, or
decision policy was used. This is telemetry for nominated candidates, not a
new recipe or an acceptance/throughput result. Q4_0 EAGLE remains the primary
comparison baseline; no baseline was evaluated here.

## Deliverable and acceptance check

`research/parallel20261002/head_flip_oracle/reference/oracle.py` captures an
actual `RowBinaryLinear` head and evaluates at most 32 independent candidates.
Each candidate flips a unique nonempty coordinate set in one head row. All
body states, quantizer parameters, row/token scales, affine midpoints and bias
stay fixed. The capture copies inputs to detached integer codes/signs and
F32 epilogue parameters; evaluation never updates the module or optimizer.

The exact I64 update is `dot_new = dot - 2*sum(sign*code)` over nominated
coordinates. Width times maximum code must be below `2^24`, matching the
source F32 exact-integer accumulation bound. Recompute `(dot_new*alpha)*beta`,
then add `(sumcodes*midpoint)*beta` if affine, then bias. Do not add an
algebraic logit delta to a previously rounded F32 logit or factor beta over
the affine addition. Every candidate is relative to the same baseline;
candidates are neither cumulative nor selected by the diagnostic.

One baseline integer head projection is cached. Capture also runs the actual
module once to retain trainer-baseline drift; it restores the saturation
telemetry field afterward. Candidate evaluation performs no full head
projection. An owner acceptance check blocks `F.linear` and verifies that
the CPU operator trace contains no `aten::mm`, `aten::addmm` or `aten::bmm`.
The structural work counter is one avoided candidate head GEMM per candidate.
This counter is not a speedup measurement: cached-logit reductions, row
statistics and rank telemetry still have cost, and capture has additional cost.

Hard CE deltas use F64, shifted logsumexp terms from the *same native-order*
cached baseline as the candidate. Algebraically canceled label/max changes
are subtracted before small relative-logsum terms, and identical logits return
an exact zero delta. Unsupported target labels have NaN CE and label margin,
but retain mapped top1 telemetry; their acceptance denominator is not removed.
The d2t API takes unique absolute target IDs, not native offset encoding.
Ties use the declared lowest draft row before d2t mapping. The code flags
top1 margins at or below `8*F32_epsilon*max(1, abs(top_logit))`; it makes no
claim that a native C++/CUDA comparator uses that tie rule.

## Source arithmetic and numerical gate

A1 signed zeros are positive, including the source raw-bit activation rule.
Learned A1 uses the actual frozen threshold; A4/A8 use actual clip ratio and
round-to-nearest-even. Fixed affine uses the source safe reciprocal rule.
Legacy fixed symmetric A4/A8 uses its own `_HardActivationSTE` quantizer;
nonfinite/invalid codes from reciprocal overflow fail closed. Its trainer
forward uses dequantized dense dots, so `module_logits` and
`baseline_max_drift` are retained separately. The diagnostic baseline instead
uses native-order integer-dot arithmetic. It never borrows the trainer's
softmax denominator when those baselines differ. A16 is excluded.

The independent validator gate is changed-row F32 error at most
`8*epsilon*max(1, abs(reference))`, ordinary F64 CE error at most
`2e-10 + 2e-12*abs(reference)`, and exact decision classification only above
the declared F32 margin gate. Tiny CE changes inside the F64 gate and F32
rounding ties are numerically ambiguous; exact real-arithmetic ordering is
not claimed there. Extreme shifted-logsumexp cases are checked separately.
These are CPU F32 reference gates, not CUDA or SM75 validation.

Source SHA256 at capture implementation base:

| Source | SHA256 |
| --- | --- |
| `recurrent_qat.py` | `f2b6d83b8929807b65b6f21826cc99097a6ddd70f99ff23ca0836aebe4db1224` |
| `learned_activation.py` | `65b7b0863bb67806991ee4860ce327bf46bfbdf0e8d4e44226aa1ca268e880db` |
| `affine_binary.py` | `edad3b513413eac5dc027cb949da6ae4e8d3ca4184d66b7d0f97b880c6252ebc` |
| `recurrent_binary.py` | `7624fa4486bcb130440cbc6b84c2463af01f0750e0145236063a125ab8b3de06` |

## Constructive overshoot

The advisor's variable row has logit `2*s`; the competing row has logit zero.
Four fixed examples have labels `[0,0,0,1]`. At `s=+1`, mean CE is
0.6269280110 and the relaxed derivative is +0.2615941560. A complete flip to
`s=-1` has linear prediction -0.5231883119, but actual mean CE becomes
1.6269280110: increase +1, accuracy 75% to 25%. Owner tests reproduce the
valid source `RowBinaryLinear` STE gradient and show the candidate oracle
reports the harmful complete flip. This is finite-step overshoot, not an
incorrect autograd derivative.

## Checkpoint and integration

Owner checks: 6/6 unittest tests pass across all twelve A1/A4/A8 combinations
of fixed/learned quantization and absent/present affine midpoint. Cases include
five-feature tails, zero scale, signed zeros, mapped unsupported labels,
multiple flips per row, independent candidates, bounded nomination count,
invalid legacy reciprocal overflow, nonmutation and the overshoot example.
Ruff and `git diff --check` pass. Environment: macOS arm64, Apple M3 Max,
Python 3.11, PyTorch 2.14.0, CPU F32/F64/I64 arithmetic.

Independent Luna validation owns `validation/`, its unique test file, and
`validation.md`; final results are recorded there. No production source is
edited. The branch is `research/20261002-head-flip-oracle`, isolated worktree
`/private/tmp/eagle-parallel-20261002/head-flip-oracle`.

Remaining work at this checkpoint: independent validator result, combined
acceptance run, final commit and root integration/push/cleanup. Root owns the
active-goal checkpoint under `docs/goals/qat-optimization-readiness.md`.
Integration is report/prototype only; adoption as training telemetry would
require explicit source binding and separate recipe/runtime review. A useful
fixed-prefix decision does not predict subsequent embeddings, private states,
native acceptance, or future trajectory quality.
