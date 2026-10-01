# W1Ax training optimization audit

Date: 2026-10-01. Five user-requested GPT-6.1 Sol/high agents independently
audited source and primary research, then the root reconciled their findings.
Source baseline: `d725fc88d52fc49841c2b20128d176c1e7f486d8`.
This is an applicability audit, not an implemented optimization or a measured
speed/acceptance result. No model workloads, accelerator/remote actions,
installations, large artifact reads or sealed-final access occurred. The live
continuous A8/A1 experiment and its monitor remain unchanged.

## Recommendation

**First remove structurally unnecessary training computation.** The best fit is
bulk K/V-only prefix construction, followed by batching the output head within
one captured proposal chain and skipping the unused A1 surrogate forward during
no-gradient context work. These can preserve the existing one-round optimizer
cadence, training examples, hard-forward quantizer, loss and gradient boundary.
Bulk matrix shapes can change F32 accumulation, so deployment/gradient gates
remain necessary.

For quality, preserve the current learning curve as the control, then prepare
one bounded optimization comparison. A lower-inertia latent initialization is
the smallest intervention with identical checkpoint-zero deployment. A
ParetoQ-inspired optimizer/scale-gradient study has stronger causal-LLM precedent
but more coupled changes. Learned A1 thresholds or A4 clipping are subsequent
representation experiments, with explicit native/export work. A short A8-to-A1
warm-start is worth a separate compute-matched comparison, not a default long
precision ladder.

## Findings across the five areas

| Area | Best fit | Verdict | Why / remaining uncertainty |
| --- | --- | --- | --- |
| [Batching and cache](training-audit-2026-10-01/02-batching-cache.md) | Chunked bulk K/V-only prefix; then stacked per-chain head | Highest implementation priority | Source proves context outputs are discarded and cache rows are independent in this single-layer model. Removes known work without changing Adam cadence; total wall-time fraction is unmeasured. |
| [Custom QAT and fusion](training-audit-2026-10-01/01-qat-kernels.md) | A1 native-only no-gradient forward; then measured regional fusion | Pursue simple path; custom backward conditional | A1 performs two forward products even under no-grad. Full custom autograd can remove an attached forward product but introduces reduction/temporary and numeric risks. |
| [Binary optimization](training-audit-2026-10-01/03-binary-optimization.md) | Inertia control; then one ParetoQ-inspired component | Conditional quality experiment | Current row latents start at ±0.5. Inertia is real, but no baseline training evidence yet shows it is harmful; useful flips must improve native acceptance. |
| [Learned activation quantizers](training-audit-2026-10-01/04-learned-quantizers.md) | A4 learned relative clipping; A1 learned threshold | Conditional new quantizer | Multi-bit clipping and binary thresholds can change codes. Positive A1 amplitude alone is redundant with learned row scales. Native shared packing and parameter/export ownership require changes. |
| [Curricula](training-audit-2026-10-01/05-curricula.md) | Short A8 warm-start with substantial hard A1 phase | Conditional separate comparison | Published binary-transformer evidence supports staged initialization, but does not establish equal-GPU-hour EAGLE gains. Mild early-depth weighting is a cheaper separate alternative. |

## Why structural cache removal ranks first

Context rebuilding currently runs feature fusion, Q/K/V, attention, O and FFN
for every accepted-prefix position, and keeps only K/V. Each next context feature
is a captured target feature rather than the discarded previous student output.
In this pinned one-layer decoder, retained K/V depend only on that position's
shifted token, target feature, norms, K/V projections and K-RoPE. The existing
native K/V-only graph independently has the same guarded architectural premise;
the Torch trainer still uses the full context step.

The batching agent counts **136,314,880 logical linear MACs per context row**,
of which only **30,146,560** are needed for K/V: **77.88% can be removed from
that stage's linear arithmetic**. Context attention and growing cache-copy work
can also disappear. This is not a 78% training-speed claim. If prefix time is
fraction `f` of the measured step and its measured speed ratio is `r`, the
whole-step ratio is `1 / ((1-f) + f/r)` before additional overhead.

Use bounded chunks to cap temporary memory. Preserve the shifted-token/feature
join, absolute RoPE positions, F16 cache writes, deferred attached seed and
no-gradient context boundary. It must be gated to one decoder layer with no
consumed context output. Do not reuse caches after an optimizer update, across
lanes or across changed token/feature ancestry.

## Two complementary speed opportunities

**Stack the head within a round.** Training feeds attached student states and
K/V back, but selects the next token from the capture. Current head logits do
not choose the next training token. Collect attached pre-norm proposal states,
then compute their output norm/head together before the same CE and optimizer
update. Up to five small head products become one small matrix product. The
recurrent body remains sequential; unsupported valid states still run when a
later supported loss depends on them. This preserves update cadence, unlike
batching several rounds into a new Adam update.

**Remove A1's no-gradient surrogate product.** The corrected A1 module computes
a native-order hard forward and a separate differentiable surrogate. The latter
is unused under no-grad but still executes. Skip it there while retaining the
raw-bit sign rule, F64 meanabs scale and integer-dot → row-scale → token-scale
order. This remains useful in the K/V projections left after context pruning.
Do not count its saving again inside the 77.88% single-pass MAC bound.

For attached projections, a custom backward can mathematically retain the
current surrogate with one forward product. Under equal product costs, its
projection arithmetic changes from two forward/two backward products to one
forward/two backward: an ideal **4/3 ratio**, not 2. Scale gradients at zero,
attached recurrent inputs and F32 reassociation matter; an extra full-weight
temporary could erase the benefit. Native-only no-grad work is the simpler gate.

## What the quality research changes

**Binary optimization:** the fresh row initializer is ±0.5, not ±1 or original
dense magnitudes. Projection and the inclusive clipped STE already keep latents
inside their trainable interval. Removing clipping or importing ReCU is therefore
not an immediate cure for a demonstrated dead zone. Lowering initialization
magnitude preserves deployed signs/scales while changing the cost of flipping a
bit. Compare one inertia change, not simultaneous LR/STE/scale changes, and judge
persistent flips and native acceptance rather than cumulative flips or CE alone.

[ParetoQ v2](https://arxiv.org/html/2502.02631v2) supplies causal-LLM evidence for
hard binary weights with learned scales, but its weight-unit gradients,
initialization, scale-gradient normalization, wider activations, head exceptions
and large training budget differ. The reports pin author code to
`7b36b8de958aada6508e620c8dc544e6ed6c39b4` and reconcile the v2 benchmark scope.
[Bop](https://proceedings.neurips.cc/paper_files/paper/2019/file/9ca8c9b0996bbf05ae7753d34667a6fd-Paper.pdf)
could remove one dense optimizer state per sign tensor, about 1.63 GiB across
both lanes with F32 signs/EMA, but its efficacy evidence is computer vision.
It does not accelerate dense training products by itself.

**Quantizers:** A4 has documented local zero-code concentration and therefore a
concrete clipping/resolution hypothesis. A learned relative step must change
code selection; a multiplier applied after fixed codes adds only amplitude.
An A1 threshold changes bits; a common positive A1 amplitude cannot directly
repair bias-free head greedy rank and can be absorbed into learned row scales.
[LSQ](https://arxiv.org/pdf/1902.08153) and [BiT](https://arxiv.org/html/2205.13016v2)
are primary precedents, with vision/BERT transfer limitations. Share learned
parameters across Q/K/V and gate/up unless the packing-cache identity is extended.
The current parameter allowlist and exporter do not support new activation
parameters; these are not live-compatible drop-ins.

**Curricula:** the current trainer already exposes attached student states to
later losses, while forcing captured D tokens. Official EAGLE-3 training also
distinguishes state recurrence from own-token sampling. Fresh student-token
trajectories need new native target labels/features; old labels cannot follow
changed prefixes. BiT supports staged initialization, but its longer precision
ladder did not improve its two-stage recipe. Charge any reused A8 checkpoint's
training cost to an A8-to-A1 comparison and retain substantial hard-A1 exposure.
Mild full-horizon early-depth CE weighting has official EAGLE-3 precedent, but
no isolated ablation establishes its advantage here. Shorter unrolls leave the
prefix cost intact and sacrifice later-state supervision.

## Deferred work and decisive next gate

Defer a full packed training/backward stack, blanket low-precision/autocast,
optimizer-in-backward, a ragged-attention framework, cache reuse across updates,
large minibatch/LR sweeps, long precision ladders and mandatory feature-state
distillation. They either solve a bottleneck not yet measured, alter the current
gradient/clipping/ancestry contract, or add experiments before the simpler
opportunities are exhausted.

The recommended implementation package is the **K/V-only chunked prefix,
stacked per-round head and A1 no-gradient shortcut**, reviewed independently and
tested first on tiny CPU fixtures. Validate exact ancestry and cache casts,
context detachment, attached seed and later-only state/K/V gradients; gate
avoidable numerical drift against the current Torch reference. A later bounded
RTX 5080 phase profile and order-balanced step comparison must establish actual
savings and reserved-memory fit. Re-profile before choosing fusion or larger
minibatches. No GPU profile was run in this audit.

For quality, first use the running recipe's predeclared native development
checkpoints to choose one mechanism test: inertia if persistent sign movement
lags, quantizer clipping/threshold if activation information is lost, or a
curriculum/trajectory intervention if teacher-forced improvement fails to
transfer to native rollouts. Do not turn these diagnostics into an unbounded
search. Each comparison needs equal GPU-time accounting plus common-token
learning-curve points, fixed data and loss semantics, and independent native
development acceptance. Q4_0 remains the final acceptance/latency/throughput
target; neither faster training nor published task accuracy is an inference
success claim. Keep sealed final prompts unopened.

These are recommendations for user selection. The audit launches no new
training experiment and changes no live source, optimizer or precision contract.
