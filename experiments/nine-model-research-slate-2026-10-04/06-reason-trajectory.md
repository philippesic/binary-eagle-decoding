# Recurrent trajectory, cache exposure and replay refinements

Research completed October 4, 2026, by researcher 06 (reasoning-first trajectory lane).
The scientific work was read-only: source/document inspection and primary-source web validation. No implementation, model/test/benchmark runs, downloads, GPU/Metal/SSH, host control or child-agent work occurred. The later reporting-only step created this uniquely assigned Markdown report; it did not extend the research or change shared plans, status, checkpoints or implementation. Root owns integration and commits.

## Recommendation and established boundary

Recommend **one new learning probe—an attached EAGLE context tail—and two inexpensive refinements to already planned refresh/batching work**. No measured gain is established.

The current code already covers much of the suspected mismatch: it rebuilds context using current student weights, feeds current student states through each proposal chain, retains within-chain state/K/V gradients, and includes native F16 cache casts. Captured proposal **tokens**, however, remain forced. See `src/w1a1_eagle/recurrent_rollout.py:153` for rollout construction, `recurrent_rollout.py:172` for token/state recurrence, and `src/w1a1_eagle/native_step.py:625` for cache casts. These are existing protections, not missing fixes.

The completed comparison does not establish that longer training alone closes the gap. Its fixed-A8 reference finished at 0.638568 accepted drafts/round and 95.689 request tokens/s, versus paired Q4_0 at 1.306255 and 135.787. Only 206,163 of 3,899,930 available supervised rows were presented. Those are RTX5080/SM120 observations with F16 target/KV and a frozen greedy verifier; they establish neither SM75 performance nor a stochastic acceptance claim. The combined candidate changed multiple factors and cannot isolate their effects. Source: `experiments/a8-qat-recovery/comparison.md`, especially lines 1–24 and 36–45. Q4_0 remains the primary baseline.

## 1. Probe: attach a short accepted-context K/V tail to EAGLE’s loss

**Derivation.** Accepted-context reconstruction explicitly runs under `no_grad`; the proposal chain is attached afterward. Thus downstream CE differentiates through proposal K/V but cannot tell fusion/K/V parameters how their earlier accepted-context representations affected that CE. This is an intentional approximation, not a bug: `src/w1a1_eagle/recurrent_rollout.py:46` declares the truncation boundary; line 75 implements it.

Rebuild all but the last small number of context rows detached, then form the last **1–4 rows with gradients**, preserving their exact tokens, raw target features, positions, hard quantization and F16 cache casts. The single-layer EAGLE context path needs only fusion, normalization and K/V projections; it does not need context attention, Q/O/FFN or the head (`src/w1a1_eagle/native_step.py:435`, particularly lines 466–481). This is substantially narrower than full-prefix backpropagation.

**Hypothesis.** Credit assignment to context-forming parameters could improve root and early-depth decisions, particularly after fusion becomes binary. It does not solve changed-token exposure.

**Primary validation.** Truncation introduces gradient bias; [Aicher, Foti and Fox, Adaptively Truncating Backpropagation Through Time to Control Gradient Bias](https://proceedings.mlr.press/v115/aicher20a.html) motivate measuring the useful gradient horizon. Their smooth-RNN guarantees do **not** establish guarantees for hard-binary STE training. [EAGLE-3](https://arxiv.org/html/2503.01840v3) supports retaining student-produced feature trajectories and avoiding unnecessary feature-regression constraints; our existing proposal recurrence already does this.

**Integration and compute cost.** Small provider/cache-builder change; extra backward storage/work for a few FC/K/V rows, no new teacher capture, no inference operators. Splitting the detached and attached segments requires explicit autograd-safe cache construction.

**Minimal test.** First establish unchanged forward logits and finite nonzero gradient from proposal CE into the attached tail while earlier rows remain detached. Then compare tail 0 versus one predeclared tail length at matched exposure, recording trainer rate/memory and native prefix survival. Promote only with native quality/complete-request evidence, not gradient magnitude alone.

**Failure arguments.** Existing seed/proposal gradients may already suffice; long-range context may dominate over recent rows; extra gradient can increase clipping or destabilize STE learning. Learned quantizers require invocation-preserving normalization—splitting a 64-row chunk must not accidentally introduce a second recipe change.

**Disposition:** bounded probe for EAGLE A8 first; A1 confirmation only if promising. **Defer block transfer.** Their native injection is fusion→encoder norm→separate per-layer K/V, so any analogous gradient boundary must be established in their new trainer: `third_party/llama.cpp/src/models/dflash.cpp:626` (per-layer projections at 635–637).

## 2. Incorporate into refresh design: retain original and newly captured whole rounds as separately identified replay sources

**Derivation.** Current training uses current continuous state but captured discrete actions. Replacing actions with student argmax while keeping the old later labels would change target conditioning and invalidate the data. The provider correctly rejects mismatched capture/prefix/label joins: `src/w1a1_eagle/recurrent_provider.py:190`, especially lines 200–212.

Refine the planned own-prefix refresh into **dataset aggregation**: keep a fixed share of original rounds and a fixed share of properly recaptured current-student rounds. Select complete rounds; never splice states, tokens or teacher rows across sources. Deduplicate exact examples for unique-exposure reporting and identify repeated presentations separately.

**Hypothesis.** Fresh rounds address action-distribution shift while original replay limits forgetting and repeated concentration on a small refresh subset. This is a refinement of the existing refresh experiment, not a separate claim of novelty.

**Primary validation.** [Ross, Gordon and Bagnell, A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger)](https://arxiv.org/abs/1011.0686) motivates collecting expert supervision on learner-induced states and training on accumulated data. Its guarantees do not automatically cover this approximate neural/QAT optimizer.

**Integration and compute cost.** Little additional training machinery once recapture exists; recapture itself remains the main cost. Teacher/verifier weights, precision and sampling policy remain immutable. Each new capture needs its own provenance and exact labels.

**Minimal test.** Use the already planned refresh comparison with a declared mixture, retaining matched no-refresh continuation. If its result is ambiguous, compare replacement versus aggregation using the same captured pool and total presented-token budget—no second recapture is necessary.

**Failure arguments.** Too much old replay dilutes the desired distribution correction; insufficient fresh-prompt diversity makes replay merely repeated memorization. A fixed mixture is an experimental choice, not a justified optimum.

**Disposition:** incorporate ancestry and replay accounting now; **probe** mixture efficacy within the existing refresh allowance. For DSpark/DFlash, construct their own five-tap/full-vocabulary blocks and exact slot mapping. EAGLE teacher material is insufficient.

## 3. Incorporate an objective audit; conditionally probe supported-token normalization across accumulated rounds

**Derivation.** EAGLE currently averages CE within a round and performs a corresponding optimizer update: `src/w1a1_eagle/recurrent_loss.py:40` and `src/w1a1_eagle/continuous_qat.py:1187`. Consequently, within a round a supported token has coefficient `1/n_r`; short rounds receive more weight per token. Domain balancing alone does not remove this.

When accumulation/batching is introduced, explicitly distinguish the mean of round means from the global mean over supported tokens:

\[
\frac1R\sum_r\frac{\sum_jm_{rj}\ell_{rj}}{\sum_jm_{rj}}
\quad\text{versus}\quad
\frac{\sum_{r,j}m_{rj}\ell_{rj}}{\sum_{r,j}m_{rj}}.
\]

With depth weighting, use the actual weighted valid-token denominator. Do not silently switch objectives during a speed optimization. This does not imply equivalence between accumulated updates and the historical serial AdamW updates; update cadence is part of the recipe.

**Primary validation.** The author [DeepSpec loss implementation at revision 005e03b81cec38b7da6399833d609ee89a2587f2](https://raw.githubusercontent.com/deepseek-ai/DeepSpec/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/loss.py) accumulates weighted CE numerators and globally reduced denominators. It also defines overlap acceptance from full distributions; that metric is not our native greedy prefix survival.

**Hypothesis and cost.** Token normalization may reduce unintended emphasis on short or heavily masked rounds. It needs no extra teacher data or inference cost and fits planned accumulation work.

**Minimal test.** First report supported-length distribution and induced source/depth weights from existing metadata. If variation is negligible, stop. Otherwise compare the two normalizations with identical accumulation batches/update cadence.

**Failure argument.** Equal-round weighting could be preferable because short/root-heavy rounds matter to acceptance; token normalization is not automatically better.

**Disposition:** incorporate explicit semantics and accounting; probe only if the audit shows material reweighting.

## Unresolved block conditioning contract and scope limits

An additional block risk remains for the orchestrator: native DSpark’s Markov predecessor is generated by sequential argmax (`third_party/llama.cpp/src/models/dflash.cpp:346`, with predecessor update at 387–389). Admission of its new QAT trainer should state whether that discrete conditioning is captured, teacher-forced, or current-student-generated, and how later teacher labels remain valid. I verified the author loss contract, but did not verify its complete forward conditioning implementation; this is an unresolved contract check, **not evidence of an existing DSpark bug**. Two attempted upstream forward-file URLs returned 404 and supplied no evidence.

Do not call generic quantized-cache exposure a new fix: current EAGLE already rebuilds current-student quantized context and performs F16 writes/reads. Do not reinterpret the repaired block output-reservation/setup-mask issues as latent new training bugs. Do not replace the private full-vocabulary block heads or use the EAGLE 32k map. Keep confidence scheduling off initially: author distribution overlap and local greedy matching answer different questions.

These recommendations refine `experiments/nine-model-qat-research-plan-2026-10-04.md`; they do not select a new active goal, allocate compute, authorize training, or alter the nine frozen-control/candidate deliverables. Effects are hypotheses, and source-level gradient reasoning alone does not establish acceptance, latency or throughput improvements.
