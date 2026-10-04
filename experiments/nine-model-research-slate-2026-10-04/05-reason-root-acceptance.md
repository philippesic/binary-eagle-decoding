# Root acceptance and quantized representation: focused research report

October 4, 2026. Research lane 5. Research was read-only: no edits, model runs, tests, benchmarks, installations, downloads, GPU/Metal/SSH operations, or child agents. This report was subsequently persisted as a separately authorized documentation-only write. Recommendations are untested and allocate no compute.

The existing plan addresses most plausible causes. Recommend **one concrete calibration refinement and two bounded probes**, without adding another quantization format or expanding the nine final models.

## Evidence and limits

Fixed-A8 reached 0.639 accepted drafts/round versus Q4's 1.306; the combined learned-A8/midpoint/low-inertia arm reached 0.248. Their CE measurements cover 229 supported labels, while native acceptance comes from different, live trajectories; neither CE nor accepted/round identifies the root-token failure mechanism. Source: `experiments/a8-qat-recovery/comparison.md:62`.

Historical forced-history interventions implicate the body more than the head: replacing Q4 body with D body under the same Q4 head costs approximately 19.8 percentage points of position-one agreement, versus 5.9 points for replacing only the head. Those are older group-scale/A16 results, not a diagnosis of current row-A8. Unsupported labels were 2.02% across the historical states; that overall rate cannot be substituted for a root-specific ceiling. Source: `experiments/binary-rescue-head-5080.md:197`.

## 1. Incorporate into the next calibration development screen: whole-row sign reversal for zero-scale rows

**Derivation.** The fitter computes

`a* = max(0, dot(z,y) / dot(z,z))`,

where `z` is the signed integer-dot output times activation scale. It then skips every row with scale zero. If `dot(z,y) < 0`, reversing all signs makes `z' = -z`, producing positive correlation and a potentially useful nonnegative scale. This avoids a dead end of the current alternating search without changing the representable model.

**Actual evidence.** The real fit has **383 zero-scale rows out of 2,560**, explicitly unable to move through fixed-scale sign search. The count caused specifically by negative correlation is unmeasured; zero design or exactly zero correlation also need distinction. Sources: `experiments/fusion-binary-real-a8-2026-10-03/README.md:82`; scale solver `scripts/fit_fusion_binary_discrete.py:696`; skip condition `scripts/fit_fusion_binary_discrete.py:808`.

**Mechanism and cost.** For zero-scale rows, propose a whole-row inversion, refit scale, and accept only an improvement under existing finite exported arithmetic. Recompute subsequent proposals from the accepted state. Small fitter change; **zero additional inference operators, parameters or bytes**. Record row reversals separately from local flip budgets so this is not silently represented as the historical 32-flip search.

**Expected benefit, hypothesis.** Recover usable fusion coordinates and avoid starting QAT with avoidable dead rows. No acceptance gain is established.

**Strongest failure argument.** Negative calibration correlation may reflect an unrepresentative small sample; reversing an entire row could generalize poorly. Scale zero may be the right regularizer.

**Decisive development test.** Compare the unchanged fitter with inversion-enabled fitting on identical, expanded all-domain data. Report reactivated rows, prompt-disjoint raw/norm-space errors, root agreement and native prefix survival; promote only on native quality and complete-request cost. The algebra is directly derived here. [Least-squares binary quantization](https://arxiv.org/abs/2001.02786) supports the broader scaling framework, not this project-specific remedy or its acceptance efficacy.

## 2. Probe: make fusion calibration selection sensitive to both normalized direction and residual amplitude

**Derivation.** EAGLE applies `hidden_norm` to the fused feature before Q/K/V, but also adds the raw feature into the residual. Thus raw magnitude and normalized direction affect different paths. Optimizing only raw fusion SSE can favor shrinkage while leaving attention inputs poorly aligned.

**Actual evidence.** Real fitting reduces validation reconstruction error by 26.98% relative to scale-only, yet cosine **0.665599** still trails the mean-absolute initializer's **0.667877**. Largest-coordinate agreement also trails initialization. These metrics concern fusion coordinates, not vocabulary rankings. Source: `experiments/fusion-binary-real-a8-2026-10-03/README.md:18`. The two paths are explicit in normalization at `src/w1a1_eagle/native_step.py:580` and residual addition at `src/w1a1_eagle/native_step.py:649`.

**Cheap refinement.** First score the existing initializer/scale-only/fitted candidates after the actual frozen `hidden_norm`, while retaining raw residual error. If those rankings disagree with raw SSE and track root decisions better, screen one fitting objective combining normalized-output error with raw error. A cheaper intermediate candidate uses teacher-token-energy-weighted SSE; this preserves rowwise fitting, but is only a proxy for actual normalized error.

**Implementation/deployment cost.** Diagnostics are easy. Exact normalized fitting couples output rows and therefore needs more solver work than weighted SSE. Both leave deployment unchanged.

**Expected benefit, hypothesis.** Choose calibration that preserves attention-relevant direction without destroying residual magnitude.

**Strongest failure argument.** Neither local metric accounts for attention/cache errors or downstream vocabulary boundaries. Pure cosine optimization can make residual amplitude wrong; do **not** replace raw SSE with cosine alone.

**Decisive development test.** One matched raw-SSE versus combined-objective screen, then compare supported root-token signed margins, unconditional root agreement, and live early-prefix survival on independent development prompts. [RMSNorm's original paper](https://arxiv.org/abs/1910.07467) supports the scaling-invariance motivation, not acceptance improvement. Preserve architecture-specific norm ordering for DSpark/DFlash; EAGLE's objective cannot be copied blindly.

## 3. Probe for A8; defer blanket A1 inclusion: tune existing drafter normalization gains

**Derivation.** Channelwise gains could compensate systematic channel imbalance and reduce an A8 vector's outlier-to-typical-value ratio before shared absmax quantization, without adding a residual branch. This addresses a plausible mechanism, **not an observed current activation pathology**.

**Actual source evidence.** All original norms are explicitly frozen today: `src/w1a1_eagle/native_step.py:323`. A8 uses one absmax-derived scale per token, so changed relative channel amplitudes can change codes: `src/w1a1_eagle/learned_activation.py:93`.

**Implementation/deployment cost.** Start with only EAGLE's drafter `hidden_norm` gain, initialized unchanged and trained with ordinary hard-target CE alongside the fixed-A8 reference. Existing normalization means essentially no additional inference cost. Training ownership, checkpoint/export, and alias checks require explicit changes: frozen parameters are currently shared across lanes (`src/w1a1_eagle/continuous_qat.py:339`). Do not unfreeze target norms or borrowed embeddings.

**Important A1 limitation.** With fixed zero threshold and positive gains, `sign(gamma*x) = sign(x)`. A directly following W1A1 projection therefore sees unchanged bits; gains mainly change its common mean-absolute amplitude. This cannot generally restore lost channel detail. Learned thresholds or intervening operations change the argument, but introduce interactions. Source: `src/w1a1_eagle/learned_activation.py:73`.

**Expected benefit, hypothesis.** A low-capacity correction to A8 channel distortion. [Norm Tweaking](https://arxiv.org/abs/2309.02784) reports benefits from updating normalization parameters in low-bit LLMs, including weight/activation quantization; it does not establish W1A1 or speculative-acceptance efficacy.

**Strongest failure argument.** Binary direction/capacity loss dominates, and gain tuning merely rescales noise or worsens outliers.

**Decisive development test.** Matched exposure with frozen-versus-trainable `hidden_norm`; inspect code changes/outlier ratios, then native root agreement, later survival and request throughput. Retain the existing floating norm precision and declare changed trainability honestly.

## Diagnostic prerequisite and disposition

Before allocating these probes, request one diagnostic decomposition on the current A8 checkpoint: root-label support, supported root top-one agreement, signed target-label margins, and conditional survival by depth on correctly joined prefixes. Unsupported roots cannot be fixed by CE; small-margin supported errors and large-margin body failures call for different interventions. Do not change the vocabulary, verifier, target precision or sampler to improve these numbers.

The zero-scale refinement is the strongest concrete addition. The normalized-calibration and A8 norm-gain ideas are optional bounded probes, not long-run defaults. For DSpark/DFlash, retain architecture-specific confirmation and honest tensor-by-tensor precision/trainability manifests; where fusion remains floating, binary-fusion fitting is not yet applicable. None of these hypotheses overturns the priority of calibrated initialization, balanced exposure, admitted native arithmetic, and direct native acceptance/throughput checks already in the nine-model plan.
