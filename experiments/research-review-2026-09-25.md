# Research review: improving binary speculative decoding

**Date:** 2026-09-25. **Type:** advisory source/evidence review, not a new experiment.
**Scope:** seven independent `gpt-6-astra` agents at high reasoning. The initial
review used local files only; a user-requested primary-source web cross-reference
was completed afterward on the same date. No new GPU runs were performed.
**Evidence update:** [cross-reference, verdicts and source versions](research-cross-reference-2026-09-25.md). The seven reports below incorporate its corrections.
**Baseline:** project `7ae1cc2`, inspected llama.cpp `34e21b7`; published same-device measurements remain
authoritative. Proposed improvements below are hypotheses until tested.

The active W1Ax study remains the project goal. This review identifies bounded
follow-ups and research choices; it does not authorize an architecture pivot,
change the frozen protocol, or establish new performance results.

## Reports

| Category | Analysis |
| --- | --- |
| QAT execution and objectives | [QAT](research-review-2026-09-25/01-qat.md) |
| Non-EAGLE one-bit architectures | [Architectures](research-review-2026-09-25/02-architectures.md) |
| Drafting, token batching and scheduling | [Throughput policies](research-review-2026-09-25/03-throughput.md) |
| Fusion, attention and draft graph costs | [Pipeline](research-review-2026-09-25/04-pipeline.md) |
| Quantizer design and precision allocation | [Precision](research-review-2026-09-25/05-precision.md) |
| Training data, alignment and vocabulary coverage | [Data](research-review-2026-09-25/06-data.md) |
| Evaluation and experiment prioritization | [Evaluation](research-review-2026-09-25/07-evaluation.md) |

## What the web cross-reference changed

- **QAT:** replace the claim that the old compression proxy was inherently wrong
  with the measured conclusion that it failed this pilot. Target-greedy CE is a
  bounded hypothesis; clipped STE and recomputed scales are not demonstrated defects.
- **Architectures:** DSpark now leads the later quality screen, with a matched
  DFlash control. Neither establishes W1A1 robustness or single-request SM75 speed.
- **Precision:** structured scales are a conditional candidate alongside thresholds,
  not an established best remedy. W1A8, ternary-A8 and W1A16 successes do not
  establish W1A1.
- **Systems/evaluation:** keep the source-supported cleanup candidates, but qualify
  K/V-only to eligible single-layer calls; measure context-dependent scheduling
  rather than assuming batching helps or hurts. Distinguish target matching from
  probability-ratio verification before importing an acceptance formula.

The [primary-source review](research-cross-reference-2026-09-25.md) records
what was kept, revised or rejected, with publication versions, implementation
commits, precision and hardware limits. It strengthens the diagnosis-first
sequence without changing the active goal or frozen experiment budget.

## Evidence that constrains the recommendations

The [RTX 2080 Ti synthesis](rtx2080ti-synthesis.md) reports Q4_0 at 1.109×
ordinary EAGLE decode throughput, head-only W1A1 at 0.907×, and all-nine
W1A1 at 0.561× in their matched production comparison. All-nine W1A1
shortened the measured draft call from 6.471 to 3.558 ms/round but lowered
accepted drafts from 1.168 to 0.055/round. The measured draft call became cheaper while
useful progress per verification round collapsed.

The [first QAT pilot](qat-head-pilot-results.md) lowered cached validation
KL from 2.506 to 1.017 while held-out acceptance fell from 1.677 to 1.565
drafts/round on the RTX 5080 PyTorch BF16 track. Those acceptance values
must not be mixed with the native FP16-target 2080 Ti numbers.

The current `draft_ms` timing excludes feature processing and cache catch-up;
`accept_ms` is an acceptance hook, not target verification time. The published
tables cannot supply a reliable numerical zero-total-draft-cost ceiling.
Record emitted token IDs/counts and complete round costs before calculating
such a ceiling. Preserve the target-only equivalence caveat in all speed claims.

## Most actionable new findings

| Area | Finding | Concrete next test |
| --- | --- | --- |
| QAT | The pilot reconstructed the old draft head; it had no target-verifier labels. Cached KL selected a worse live drafter. | Audit state/label joins, then test the bounded target-aligned objective with online development selection; CE is not a proven optimal loss. |
| Data | Capture lacks parent-path/verifier IDs, weights many off-path tree rows, and can retain a terminal tree never verified. The vocabulary table itself checks out. | Matched ordinary/binary trajectories on the same prompts; explicit valid/unsupported/unverified masks and depth counts. |
| Scheduling | The native loop uses global maximum depth; per-sequence caps truncate afterward. Source predicts wasted proposals near caps. | Count computed versus returned tokens, then enforce caps before the next decode and test cache/output boundaries. |
| Graph work | Logits-disabled cache catch-up still builds an output head and vocabulary expansion. Existing large-N head traces corroborate execution, but lack stage attribution. | Tag process/draft stages; skip unused head output first, then separately test single-layer K/V-only catch-up. |
| Packing | Q/K/V share one input and FFN gate/up share another, but each custom operation packs separately. | Share codes/scales for identical graph values with exact numerical/lifetime checks; measure packing-inclusive benefit. |
| Precision | Fusion joins three feature taps; Q/K/V join two normalized streams. One common activation scale may discard useful relative magnitudes. | Diagnose group energy/errors after W1Ax; screen one structured-scale contract if supported, accounting for every partial dot and scale. |
| Alternative architectures | Local DFlash/DSpark paths exist; block generation changes available matrix columns. DSpark retains a sequential Markov head. | A later DSpark-led, matched-DFlash ordinary/W1A16/W1A1 quality screen before binary kernels, including head coverage and prefix survival. |
| Evaluation | Native confidence can be normalized over only top-10 candidates, making a 0.1 floor potentially redundant. Full round accounting is incomplete. | Audit actual backend candidate normalization before any declared policy-grid amendment; preserve separate counts and timing boundaries. |

These are source-supported mechanisms and proposed tests, not measured speedups.
Apply generally useful runtime improvements to the FP16 and Q4_0 anchors too;
otherwise a runtime cleanup could be misattributed to binary arithmetic.

The all-nine run accepted only 385 drafts in 6,940 rounds: at least 94.45%
of rounds accepted none. Increasing depth alone cannot fix these first-token
failures. At unchanged output count, the paired head-only and all-nine rates
would need roughly 18.2% and 49.4% less total decode time respectively just to
match Q4_0. Those are arithmetic deficits, not predicted optimization gains.

## Recommended order

1. **Finish the W1Ax diagnosis and complete-round instrumentation.** Holding
   binary weights fixed while changing activation precision identifies whether
   weights, activations, or both need recovery. Time feature processing, draft
   catch-up, proposal generation, target verification, sampling and cache repair
   separately. This is already the active study, not an additional architecture
   project.
2. **Audit avoidable native work before designing another matrix kernel.**
   Investigate whether logits-disabled catch-up executes an unused output head;
   whether per-sequence length caps are enforced before computation; and whether
   rejected suffixes are processed unnecessarily. Start with counters and a
   small isolated change. Require actual graph execution evidence, token/KV
   correctness, and matched complete-request timing.
3. **Test target-aligned supervision before tuning the optimizer.** Build new state-aligned
   captures that identify the parent path, target prefix and matching verifier
   row. Measure vocabulary support and the native-chain versus PyTorch-tree
   distribution difference. Select the bounded target-aligned head checkpoint
   by online development acceptance, with cached reconstruction KL retained as a diagnostic.
   Preserve target probabilities for analysis; do not silently add a loss sweep.
4. **Use precision allocation as a measured decision.** Diagnose row/token
   outliers and group sensitivity; preserve the frozen W1Ax contract in the
   primary comparison. Block scales, learned thresholds and correction paths
   are conditional candidates; literature does not identify an EAGLE winner. They need explicit packing/export and cost accounting.
5. **Screen a non-EAGLE block drafter only after the quality gate or a user
   decision to change direction.** Local DFlash/DSpark support makes this more
   concrete than a new foundation-model training project. Lead with released
   DSpark and matched DFlash as control; first measure their
   ordinary and simulated-binary quality/cost; architecture-matched low-bit
   controls must precede claims that one-bit arithmetic is responsible for a win.

## Decisions to preserve for the user

- Whether to advance a bounded native scheduling/graph cleanup alongside the
  existing W1Ax study. This is a source-supported opportunity, not yet a measured
  speed improvement.
- How to treat target labels outside the fixed 32,000-token draft vocabulary
  after measuring their frequency and mass. Silently dropping them would hide
  an acceptance limitation.
- Whether later quality evidence justifies broader recurrent QAT, a new
  quantizer contract, or a non-EAGLE architecture screen. These are distinct
  research forks; this review does not choose all of them at once.

## Acceptance and performance gates

For a fixed workload, use total emitted tokens divided by total wall time.
With measured mean emitted tokens per round E and full mean round cost C,
throughput is E/C. A candidate beats anchor b only if E_candidate/C_candidate
exceeds E_b/C_b. Do not replace E with accepted-drafts-plus-one unless token
accounting proves that equality for the actual stopping and bonus-token rules.

For an optimization that removes measured nonoverlapping cost D while holding
the emitted-token trajectory and every other cost fixed, E/(C-D) is a
conditional upper bound. A quality change can also change verifier shapes,
round count and cache work, so it needs a new matched measurement.

Preserve the 12 historical prompts for regression, use the 24 development
prompts for selection, and keep the 24 final prompts sealed until both the
trained candidate and policy are frozen. Five timing repetitions are not five
independent quality samples. Every practical W1A1 speed claim must beat both
FP16 EAGLE and Q4_0 EAGLE on the same hardware and target/verifier track.

## Subsequent PrismML-focused investigation

The [extended report](prism-quantization-research-2026-09-25.md) incorporates the
new W1Ax historical result and deeper artifact/quantizer audits. It narrows the
next recommendation to graph/cast parity, output-aware scale fitting and
head/body adaptation. It also corrects an interpretation of the earlier pilot:
worse ordinary execution of trained latent weights is a diagnostic, not an
independent QAT failure criterion; the quantized path is the deployed endpoint.
