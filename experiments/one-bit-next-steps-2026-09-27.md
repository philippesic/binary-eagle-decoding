# One-bit speculative decoding after the W1Ax suite

**Status:** consolidated recommendation, 2026-09-27. No new goal, training,
GPU run or reserved-final evaluation is started by this review. The completed
suite remains sealed. Research scope and budget choices remain user-owned.

## Recommendation

Prioritize **binary-weight representation and acceptance recovery**, using
high-precision activations to isolate weight damage. Then test activation
binarization as a separate gate toward the original W1A1 endpoint. Do not spend
the next phase on another binary kernel or another depth/confidence sweep of
the unchanged model.

This is a recommended intermediate milestone, not a redefinition of W1A1.
A practical W1-weight/A8 model, an all-selected-linear W1A1 model, and a
mixed-precision model are separate outcomes. Report all scales, exceptions,
storage and executed operand widths. For example, group-128 Q1_0 is 1.125
stored bits per matrix weight including its FP16 scales, with wider activations.

## What the completed results establish

The user's fixed **development policy-grid D=5, p_min=0** cell gives:

| Path | Decode tok/s | Accepted drafts/round |
| --- | ---: | ---: |
| Target-only | 60.49 | — |
| FP16 EAGLE | 76.83 | 1.047 |
| Q4_0 EAGLE | 84.58 | 1.036 |
| W1A16 | 29.99 | 0.117 |
| W1A8 | 42.68 | 0.111 |
| W1A4 | 44.85 | 0.047 |
| W1A1 | 45.50 | 0.055 |

These paths match raw emitted tokens in this cell. That confirms the observed
output agreement, not draft quality: the verifier can reject poor proposals.
W1A16 retains only about 11% of FP16's accepted drafts/round despite restoring
activation precision. Thus A1 alone does not explain the collapse. This does
not isolate every cause: weight approximation, recurrent propagation, interface
mismatch and graph fidelity remain distinguishable hypotheses. A16's current
scalar sign-add kernel is also a reference cost, not an optimized W1A16 ceiling.

The twelve-cell grid is now complete: every W1Ax mode loses to both same-cell
anchors. Highest observed W1A1 decode rate is 55.315 tok/s, versus selected
FP16/Q4_0 rates of 92.184/98.068. Those cross-cell comparisons have timing-epoch,
selection and output-difference limitations; they are not causal policy gains.
The robust conclusion is that the tested policy grid did not rescue this recipe.

The separate **historical instrumented trace** makes the mechanism clearer:
W1A1 reaches prefix depth one in 5.403% of eligible rounds and depth two in
0.145%; no W1Ax prefix reaches depth three. W1A16 reaches depth one in 10.574%.
Most extra proposals therefore cannot yield useful accepted prefixes.

In that trace, W1A1 averages 23.441 ms/round, of which 3.815 ms is draft()
and 18.061 ms target decode plus synchronization. Removing only recorded
draft() spans at a fixed trajectory gives a 53.76 tok/s proxy, below that
historical primary Q4_0 anchor's 89.73. Holding traced round cost fixed gives
approximately 1.103 accepted drafts/trace round to match Q4_0, versus roughly
0.055 today. These are conditional planning calculations, not attainable
speedups or universal acceptance thresholds. Quality changes alter costs and
trajectories; the final gate must use new matched end-to-end measurements.

Source: [completed results](w1ax-activation-precision-results.md) and
[policy appendix](w1ax-policy-grid-results.md). Keep the fixed development
cell, historical trace and older 5080 BF16 tree experiments separate.

## First phase: determine the smallest useful recovery

| Order | Experiment | Decision it enables |
| --- | --- | --- |
| 1 | Complete a same-prefix ordinary → cast-only → dense binary reference → packed binary bridge. | Separate representation loss from graph/export/cache defects. |
| 2 | Fit row/group scales against actual training activations, with A16 fixed. | Determine whether better magnitude representation is sufficient before changing signs or training scope. |
| 3 | If scale recovery is insufficient, cross ordinary/binary bodies and heads on forced common histories; fit one regularized readout on frozen binary-body training states. | Distinguish a recoverable readout mismatch from a body requiring adaptation. |
| 4 | Freeze one recovery recipe selected by these diagnostics. | Spend the training budget on the smallest supported intervention. |

**Bridge.** Existing integer-dot, operator replay and model-token checks are
valuable and should be reused. Cast-only FP16 replay already changed no ordinary
operator outputs on 147 captures; this is not full recurrent parity. The remaining bridge should compare retained
K/V, pre/post-normalization states, logits and proposals across initial,
zero-accept, partial-accept and full-accept transitions. Preserve source casts
and scale application order; do not materialize F32-scaled weights in F16 and
accidentally add a quantizer. Fix material divergence before fitting it away.

**Scale fitting.** Use four controlled references: row versus group-128 scales,
each with mean-absolute versus activation-output-fitted values. Hold original
signs, source weights, A16 inputs and F32 reference scale arithmetic fixed;
then evaluate actual rounded exports separately. Minimize linear output error
on designated training inputs, using their uncentered second moment rather
than weight error alone. Positive constrained fitting with regularization is
the small baseline; clipping an unconstrained solution is not the same solve.
Select by online development proposals/acceptance after the local error screen.

Grouping is not a promised cure. The prior full weight audit found only about
0.74–0.77% residual-SSE reduction for the head/FFN, although several fusion and
attention matrices improved more. A naive Q1_0 CPU smoke already had poor
proposals. Its two-prompt body/head factorial retained considerably more quality
with a binary head on an ordinary body than with a binary body and ordinary head.
These CPU results motivate diagnosis; they do not establish SM75 performance
or show that a fitted readout cannot compensate.

**Readout diagnostic.** A train-only fitted dense readout asks whether binary
body states still contain information the old head cannot use. It is an oracle
for choosing the next training scope, not a deployable all-binary success.
If it works, prioritize interface/head adaptation and then compress that head.
If it does not, consider recurrent body-and-head adaptation; a failed small
readout fit is not proof of information-theoretic impossibility.

These proposals consolidate the [Prism/representation review](prism-quantization-research-2026-09-25.md)
and [precision review](research-review-2026-09-25/05-precision.md).

## Training: change the experiment, not just the step count

The old pilot reduced cached KL from 2.506 to 1.017 but reduced online BF16
PyTorch acceptance from 1.677 to 1.565. It trained only the head and presented
about 32,000 cached rows. It does not establish that binary QAT fails, that KL
is intrinsically wrong, or that longer training alone will solve the problem.
Poor ordinary execution of QAT latent weights is a diagnostic, not a rejection
criterion for a model deployed with a hard quantized forward.

Before any recipe, capture the actual target verifier label/probabilities at
each exact student prefix with ancestry, positions, recurrent depth and validity
masks. Match native chain semantics or demonstrate evaluator equivalence;
BF16 tree acceptance is not interchangeable with native FP16-target acceptance.
Teacher-head logits on arbitrary student features are not target supervision.

The suite now finds 98.51% target-only emitted-token vocabulary coverage on the
primary prompts. This makes vocabulary expansion a lower priority explanation
for the gross collapse on those trajectories, but does not measure target
probability mass or all draft prefixes. Complete the aligned support audit and
declare unsupported-label treatment; preserve those rows in denominators.

The existing [target-aligned head pilot](qat-revisit-plan.md) remains a useful
bounded test of supervision, with its documented 500-step/45-minute cap. It
cannot establish recovery of the all-binary body. A wider recipe requires a
separate explicit amendment: hard-forward signs/scales, recurrent student
states, separate teacher/student caches, valid masks and target-aligned token
supervision. Hard CE with the declared regularizer is a hypothesis; capture
soft targets without silently adding an objective sweep. Any comparison meant
to isolate teacher choice must keep the loss family and masks fixed.

Select checkpoints by live development acceptance/prefix survival at fixed
policy, not reconstruction loss. Track root and conditional deeper acceptance,
sign flips, scale drift, gradient-mask activity, unique states and effective
passes. Improvements confined to cached KL do not pass the gate. Use the
frozen 96 train / 24 development / 24 final split; the development set is now
reused evidence, and five repetitions do not create new independent prompts.
Keep final prompts sealed until model, policy and evaluation are frozen.

OneBit provides a concrete W1A16 representation/distillation precedent, while
ParetoQ releases learned low-bit models and training code. Neither demonstrates
our W1A1 EAGLE endpoint. EAGLE-3's own multi-step training motivates recurrent
fidelity when widening training. Sources rechecked for this synthesis:
[OneBit v3](https://arxiv.org/html/2402.11295v3),
[ParetoQ](https://github.com/facebookresearch/ParetoQ),
[EAGLE-3 v3](https://arxiv.org/html/2503.01840v3).

## After weight recovery: activation precision, then execution

Re-run A16/A8/A1 quality on the recovered binary weights. If A8 retains useful
acceptance, existing Q1_0 execution offers a practical weight-compression track;
its Q8_1 arithmetic is distinct from our custom whole-token A8. Native speed
must still be measured against FP16 and Q4_0.

If A1 now becomes the main loss, choose one capture-supported mechanism:
learned thresholds, coarse scales preserving fusion taps/QKV input streams,
or activation-aware recurrent QAT. Positive rescaling immediately before sign
does not change nonzero sign bits, so it cannot restore discarded magnitudes
by itself. Extra block reductions, affine corrections, outlier paths and
low-rank residuals must pay their full runtime and precision cost. Mixed
precision is an explicitly labeled fallback, not an all-W1A1 result.

Do native optimization only after quality permits plausible headroom. Prior
source audits suggest unused-head removal during catch-up, genuinely early
draft caps and shared input packing for Q/K/V and gate/up. Validate each in
isolation, including cache boundaries, and apply general optimizations to
FP16/Q4_0 too. The measured process() span near 1 ms/round limits enthusiasm
for catch-up cleanup as the main cure; it cannot be counted as wholly removable.
Binary MMA already tied the portable head path on SM75, and the tested INT MMA
paths lost at this workload's small shapes.

## Architecture fallback and deferred ideas

If the bounded EAGLE recovery has no credible quality/cost frontier, screen
released **DSpark with a matched DFlash control** before building kernels.
Measure normal precision, then simulated W1A16 and selective W1A1 at the same
target, hardware and single-request workload. Include actual head precision,
memory, serial work and prefix survival. DSpark's Markov head retains sequential
work; block generation does not guarantee useful SM75 batching. A shared target
head must remain unchanged, and its use must be counted as a draft precision
exception. [DSpark v1](https://arxiv.org/html/2607.05147v1) supports this as an
architecture hypothesis, not a local one-bit speed result.

Defer broad rotations, sparse/low-rank corrections, ternary or multiple binary
bases, larger vocabulary, foundation-model training, and a full Bonsai draft
integration until a specific diagnosis justifies their cost. The previous
Prism review's near-98% headline concerns ternary task-score retention, not
binary next-token agreement; it supplies no acceptance guarantee here.

## Decision and success gates

Recommended next scope: finish the missing bridge checks and run one train-only
fixed-sign scale screen. Readout fitting is a conditional follow-up if scale
recovery is insufficient, before an explicitly budgeted QAT choice. Recommend
accepting W1 with wider activations as an intermediate milestone while retaining
W1A1 as the research endpoint. This updates recommendation order after the
completed suite; it does not expand the existing training authorization.

Stop a recipe when deployment-matched development acceptance does not improve,
or measured complete-round costs leave no credible advantage over Q4_0.
Approximately one useful accepted draft per round is a planning aspiration
from the historical trace, not a universal admission threshold. Passing an
acceptance gate only earns a native timing test.

For the selected survivor, require actual binary dispatch, numerical/export
checks, same-device target-only/FP16/Q4_0 controls, alternating repetitions,
decode and request rates, memory and precision accounting, and explicit raw
output comparisons. A practical speed claim must beat both draft anchors;
matching the target on one cell is not a general proof of numerical equivalence.
Recompute headroom from the candidate's own emitted-token and complete-round
counts. Keep a reproducible negative result if the gain does not materialize.

## Consolidated prior research

This note updates the [seven-agent synthesis](research-review-2026-09-25.md),
its [primary-source corrections](research-cross-reference-2026-09-25.md),
the [Prism audit](prism-quantization-research-2026-09-25.md),
the [QAT revisit](qat-revisit-plan.md), and the completed W1Ax reports.
Earlier instructions to finish the suite, collect round spans or investigate
the redundant 0.1 floor are historical: those measurements now exist. The
state-label audit, fitted representation, recurrent recovery and architecture
screen remain proposals. Two focused read-only advisor reviews checked quality
and execution/architecture recommendations for this consolidation; no new
model-quality or GPU-performance result is claimed.
