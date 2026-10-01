# CPU-first work to advance the binary drafter project

Date: 2026-10-01. Ten user-requested GPT-6.1 Sol/high agents audited distinct
areas using project source and primary research. Source baseline:
`f0bb92dda687d908b3cffe084510b3be757df54f`. The user requested non-Fast agents;
the launcher exposes model and reasoning effort but no service-tier toggle.
Model/effort were explicitly selected, with that limitation disclosed.

This is a research shortlist. No implementation, model/test run, installation,
GPU/remote action, large weight/capture read or sealed evaluation occurred.
This audit changes no live experiment. During publication another chat recorded
`553244e`: finish data/QAT preparation and stop before optimizer updates. That
boundary is preserved. The previous training audit is complete; this broader
audit neither implements its recommendations nor starts another project goal.

## What we are trying to improve

The drafter proposes tokens for a larger, unchanged model to check. Smaller
weights help only when the proposals remain useful enough to save checking
rounds. Our reference is Q4_0 EAGLE, the standard four-bit drafter. The goal is
better accepted proposals and total decoding speed with honest storage and
precision accounting.

The list below ranks concrete work we can prepare locally, rather than
published headline speedups. Prototype preparation can be CPU-only. Actual
acceptance and GPU speed still require later, bounded measurements.

## Ranked shortlist

### 1. Fit better binary weights without a long training run

Instead of keeping each original weight's sign, choose signs and scales that
better reproduce the layer's output on recorded training examples. Earlier
scale fitting never changed the signs. A direct search can produce a different
model with the same packed representation and inference cost.

Start with feature fusion, the projection that combines target-model signals.
Build a bounded alternating sign/scale fitter and the existing export bridge.
Keep a stronger scale-only control: the old fusion scale solve had not fully
converged. Check candidate moves against the real objective; simultaneous sign
flips interact and individually good stale proposals can be bad together.

**CPU work now:** fitter, arithmetic fixtures and a small eligible calibration
package. Full calibration captures are mostly remote; local diagnostic samples
are not a replacement for training data. **GPU later:** native proposal quality
and whole-request comparison. Reconstruction improvement alone is insufficient.

See [discrete fitting](broader-audit-2026-10-01/05-discrete-fitting.md).

### 2. Give binary weights a little more expressive power

There are two related routes. First, let each row's two weight values be
asymmetric instead of equally far from zero. This keeps the main binary dot
and adds a shared input-sum correction. Second, add a tiny bias or rank-one/four
correction to the sensitive body. A small correction could recover information
that binary signs discard.

The affine row representation adds about 255 KiB of F32 centers across the
nine matrices. A rank-four floating fusion correction adds about 80 KiB of
factors. Their purpose and cost differ: the affine input-sum term is constrained;
a correction reading raw input is a more flexible mixed-precision bypass.
Neither is the frozen symmetric W1A1 contract. Tiny storage does not prove tiny
latency, especially when extra kernel launches are required.

**CPU work now:** compare a mean/output-bias control, affine rows and one small
fusion residual on the same designated training operands. **GPU later:** verify
that recurrent proposal gains outweigh correction cost. The completed dense
head rescue makes another large head-quality patch a weaker first choice.

See [representations](broader-audit-2026-10-01/03-binary-representations.md) and
[small corrections](broader-audit-2026-10-01/04-small-corrections.md).

### 3. Use existing fast kernels for fitted binary weights

The stronger group-scaled binary model currently uses a slow reference path.
The runtime already supports Q1_0, a packed binary-weight format with optimized
integer kernels. The missing piece is a direct exporter for learned signs and
scales, plus a numerical reference for its activation conversion.

**CPU work now:** codec/exporter and small round-trip fixtures. Preserve the
learned signs directly; reconstructing a dense matrix and requantizing can
erase signs in zero-scale groups or refit the scales. **GPU later:** test the
new scale rounding and blockwise eight-bit activations, then measure acceptance
and total speed. This route is binary weights with wider activations, not an
unchanged W1A1 deployment. Old D's same-trajectory zero-draft-cost proxy trails
Q4_0, so improved quality or a valid new trajectory is needed for a win.

Joining the packed gate/up projections is a secondary source-supported speed
candidate: shared packing exists, but separate dot launches remain. A new
persistent decoder or generic kernel rewrite is lower priority.

See [native execution](broader-audit-2026-10-01/09-native-execution.md).

### 4. Remove less useful FFN neurons

The FFN is the drafter's wide internal transformation. Remove matching neurons
from its gate/up outputs and down inputs while preserving the main hidden-state
width. That makes the actual matrices smaller and reduces training state too.
Removing 25% of FFN neurons eliminates 18.68M selected weights, about 8.56% of
the selected total.

**CPU work now:** a dependency-safe slicer/export contract and contribution
ranking from small training activation samples. Keep only one width candidate
first. **GPU later:** check quality before and after bounded recovery training.
Pruning changes activation scales and group boundaries; it is not an exact
transformation. It also cannot accelerate the FFN already removed by the
previous audit's K/V-only training-prefix proposal.

See [smaller drafters](broader-audit-2026-10-01/02-smaller-drafters.md).

### 5. Make the token-prediction head cheaper

The head scores 32,000 possible draft tokens and owns about 37.5% of selected
weights. A low-rank head replaces one large projection with two thinner ones.
A dynamic shortlist instead scores only likely vocabulary rows, retaining an
exhaustive fallback.

[SlimSpec](https://arxiv.org/html/2605.10453) provides directly relevant EAGLE-3
head evidence. [SpecVocab](https://arxiv.org/html/2602.13836v2) supplies a learned
shortlist alternative and author code. Both need local precision and quality
checks. Rank-320 FP16 factors would occupy about 21.1 MiB versus 9.77 MiB of
current packed head signs, so a smaller parameter count can increase bandwidth.
The draft head is already vocabulary-pruned; NanoSpec's released implementation
explicitly limits its applicability to such heads.

**CPU work now:** factor/export fixtures and a shortlist coverage screen using
only information available before verification. **GPU later:** charge routing,
gathering, added projections and fallback, and test accepted proposals. This
is a promising research branch, with a weaker immediate fit than the first
four because it adds structure and may need substantial training.

See [output head](broader-audit-2026-10-01/08-output-head.md).

### 6. Get more value from fewer training rounds

Some recorded examples may repeat common easy behavior while expensive long
prefixes consume much more training time. A selector could retain whole rounds
that preserve source, length, depth and token coverage at a smaller compute
budget. Select complete recurrent chains: unsupported earlier positions can
still teach through later supported losses.

**CPU work now:** an auditable selection manifest and provider wrapper, first
using coverage and estimated prefix cost. **GPU later:** compare with a
stratified random subset at the same budget and the full-data control. Hardest
or highest-loss examples are not automatically the most useful.

Shared teacher capture and accepted-feature deduplication already exist. An
indexed provider can remove an additional F32 selected-feature copy, saving
30,720 bytes per selected row, but that saves storage rather than teacher or
training arithmetic. Keep these benefits separate.

See [data value](broader-audit-2026-10-01/10-data-value.md).

### 7. Prepare a block-parallel drafter comparison

EAGLE proposes tokens sequentially. DFlash and DSpark predict a block with more
parallel work; this may use hardware better and reduce accumulated proposal
error. Compatible released Qwen3 checkpoints and native paths exist, so a
small compatibility fixture is more useful than another architecture survey.

**CPU work now:** prove feature indexing, masks, token alignment, cache rollback
and private draft-head ownership with tiny tensors. **GPU later:** a normal-
precision baseline screen before any binary training. DSpark reads different
intermediate target features from our EAGLE corpus and borrows a much larger
full-vocabulary target head. Existing saved data is therefore not sufficient,
and the target head must remain unchanged. No paper establishes its W1Ax or
2080 Ti speed. This is a high-upside later branch, not a current pivot.

See [architectures](broader-audit-2026-10-01/01-architectures.md).

### 8. Spend draft effort only when it is likely to pay off

A controller can stop proposing before deeper, unreliable steps waste work.
The useful question is whether one more proposal saves more target work than
it costs. Raising acceptance percentage by dropping proposals is not itself
a speed result.

**CPU work now:** a bounded cost/survival controller and an optimistic feasibility
test from the available traces. **GPU later:** collect missing per-step scores
and evaluate the frozen controller against both the binary candidate and Q4_0.
Existing traces cannot exactly replay new round boundaries or alternate branches.
Old fixed policy sweeps failed to rescue W1Ax; prioritize this only after an
improved candidate has realistic speed headroom. Learned large-tree controllers
and repeated depth grids are lower priority.

See [draft policies](broader-audit-2026-10-01/07-draft-policies.md).

### 9. Use a two-bit or ternary control to find the practical frontier

A little more weight information may preserve much more proposal quality while
remaining smaller than Q4_0. Existing IQ2_XS/Q2_K kernels offer a practical
starting point; a balanced four-level oracle distinguishes that capacity from
ternary or two binary residual planes.

**CPU work now:** exporter/grid/byte accounting and a small reconstruction
oracle. **GPU later:** one controlled acceptance comparison. The current Q2_0
converter does not use its fourth level normally, so it is not a clean balanced
two-bit control. Ternary storage support also does not prove fast CUDA dispatch.
These are intermediate comparisons, not a replacement of the strict W1A1 goal.

See [low-bit formats](broader-audit-2026-10-01/06-lowbit-formats.md).

## What to choose first

For a two-day research/implementation window, the strongest model-quality track
is **one fusion sign/scale fitter plus one cheap correction control**. They use
a common small train-only calibration package and ask whether lost information
can be recovered without another long training run. The most direct deployment
track is the **learned Q1_0 export bridge**. FFN pruning is the best independent
structural-compression candidate.

Do not combine all these changes into one experiment. Each needs a baseline,
fixed provenance, a small rejection gate and native development acceptance.
Rotations, sub-bit factorization, exact binary vocabulary search, unrestricted
tree expansion and new deep-teacher/distillation systems remain conditional
or deferred because their cost or quality transfer is weakly supported here.
No CPU-only test establishes GPU speed or final held-out acceptance.

The root reviewed all ten reports, reconciled the smaller/head precision
overlap and corrected an early-cap inconsistency: current per-round EAGLE caps
are applied after generation; earlier unused-head and K/V pruning are separate.
Those previously reviewed caps are not counted as a new discovery.
