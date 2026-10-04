# Reasoning-first cost/quality controls

Research completed October 4, 2026. The scientific investigation was read-only: source/report inspection and primary-source browsing, with no edits, tests, model runs, downloads, GPU/Metal or remote work. This document was subsequently written under separate reporting-only authorization. No implementation or experiment is authorized by this report.

Recommend **one fitter refinement, one objective control, and one conditional norm-gain probe**. All preserve current tensor shapes and deployment arithmetic; none establishes a quality gain without native validation. Initial locally derived ideas were sent to the orchestrator before web research.

## 1. Probe whole-row sign reversal to rescue zero-scale fusion rows

The current nonnegative scale solver computes a = max(0, <z,y>/||z||²), then the sign search skips every row with a=0: `scripts/fit_fusion_binary_discrete.py:696` and `:808`. The real fit has **383 zero-scale rows with no sign moves**: `experiments/fusion-binary-real-a8-2026-10-03/independent-validation.md:78`.

**Derivation:** when c=<z,y><0, reversing all weight signs changes z to -z. Its optimal nonnegative scale becomes -c/||z||², reducing continuous fitting SSE from ||y||² by c²/||z||². This escapes an alternating-optimization dead point while retaining exactly the same binary representation. Zero correlation or zero design gives no benefit. The report does not expose those correlations, so the number of rescuable rows remains unknown.

**Minimal test:** on the existing authenticated fitting inputs, classify zero rows by denominator and correlation; evaluate the original and reversed orientation using the existing exported-F32 scale/neighbour gate. Count a row reversal separately from single-bit moves, preserve the original candidate, and report validation SSE, post-normalization direction, and surviving zero rows. Admit a winner only after native prefix survival. Repeat using actual A1 operands before claiming A1 applicability.

**Cost/ease:** small fitter change and bounded CPU arithmetic; no new inference bytes, operators, precision exceptions, or kernel. Expected benefit is better initialization and fewer disabled output channels—not demonstrated acceptance. **Strongest objection:** negative correlation on eight training prompts may reverse channels incorrectly on other domains. The existing fit/validation split and later expanded all-domain validation remain essential.

**Disposition: probe first.** This is a direct algebraic finding from current source, not an efficacy claim borrowed from another paper.

## 2. Probe example-normalized fusion reconstruction instead of simply increasing the raw-SSE search budget

Current fitting cuts validation SSE by 27%, yet its cosine, coordinate-sign agreement and coordinate argmax trail the initializer: `experiments/fusion-binary-real-a8-2026-10-03/independent-validation.md:85`. EAGLE normalizes the encoded feature before concatenating it into Q/K/V input: `src/w1a1_eagle/native_step.py:580`. Raw magnitude error can therefore receive more fitting attention than downstream directional error warrants.

**Concrete control:** retain the existing discrete fitter but use fixed per-example weights

```
w_t = 1 / (mean_j(y_tj²) + epsilon)
L   = sum_tj w_t * (prediction_tj - y_tj)²
```

Use a predeclared floor/cap estimated exclusively from fitting data. This approximately balances relative error across examples while preserving separable row fitting. It is **not equivalent to optimizing post-RMSNorm error**: the student denominator and channel gains matter. Measure exact downstream RMSNorm error alongside raw SSE.

Merely multiplying each output row's independent SSE by a constant “importance” weight cannot change its minimizer. A meaningful modification must vary across examples or introduce coupling/downstream information.

**Minimal test:** one weighted fit versus the existing unweighted fit, equal sign-move budget and identical initial signs. Select using independent prompts and actual post-norm metrics, then native decisions. Do not stack this with row rescue until each singleton has evidence.

**Cost/ease:** modest CPU fitter work, no deployment change. Hypothesis: less domination by high-energy examples and better useful direction at comparable compute. **Strongest objection:** fusion also contributes beyond the normalized attention input; discarding amplitude importance could worsen recurrence/residual behavior. Consequently this is a control, not a replacement objective by assumption.

Primary evidence supports downstream-sensitive reconstruction in general: [GuidedQuant](https://arxiv.org/abs/2505.07004) incorporates end-loss gradients into quantization objectives, and [BRECQ's author implementation](https://github.com/yhhhli/BRECQ/blob/main/quant/block_recon.py) provides Fisher-weighted reconstruction. Neither validates this particular weighting or W1 EAGLE acceptance.

**Disposition: probe; defer full gradient-guided fitting unless this cheap diagnostic identifies a real mismatch.**

## 3. Conditionally probe existing drafter RMSNorm gains, starting with A8

Four EAGLE norm vectors already exist in the graph: `src/w1a1_eagle/native_step.py:235`. Their multiplication is explicit at `:79`. Current optimizer ownership omits norms, and deployment identity treats them as frozen source tensors: `src/w1a1_eagle/recurrent_qat.py:631` and `src/w1a1_eagle/qat_state.py:114`.

**Hypothesis:** allowing a small existing gain vector to adapt can reweight quantization-sensitive channels without another residual branch or extra inference operation. Prefer a localized vector supported by channel-error evidence, rather than unfreezing every norm.

There is a crucial A1 limitation. For positive channel multipliers g_j, sign(g_j*x_j)=sign(x_j). At the current global sign/mean-absolute boundary, these gains leave codes unchanged and affect only the shared activation scale: `src/w1a1_eagle/recurrent_qat.py:66`. They cannot generally restore channel-specific amplitudes lost by A1. A8 offers a less degenerate first test.

**Minimal test:** ordinary A8 QAT versus identical QAT plus one existing norm vector, initialized identically; unchanged signs/scales recipe and matched token exposure. Log gain drift, activation-code changes, channel error and native prefix survival. Explicitly extend ownership, checkpoint/export and deployment hashing; “already in graph” does not make current infrastructure support it automatically.

**Cost/ease:** tiny parameter/moment storage and unchanged inference operation count, but moderate contract/export work. **Strongest objection:** available evidence does not show enough capacity to repair extreme quantization. [Norm Tweaking](https://arxiv.org/abs/2309.02784) supports this mechanism, while [AQLM Table 7](https://arxiv.org/html/2401.06118v3) finds RMSNorm-only tuning roughly comparable to no tuning at 2.02 bits. Neither is binary-drafter evidence.

**Disposition: defer until localized channel-error evidence; then one A8 probe.** Retain full binary QAT as the control. This is not a proposal for norm-only rescue across all six long runs.

## What not to pursue and unresolved decisions

Do not simply increase sign activity, increase fusion move caps, or make norm-only training the cheap replacement for full QAT. The bad combined A8 arm made 1.12 billion flips versus reference's 137 million; those counters do not identify useful changes: `experiments/a8-qat-recovery/comparison.md:132`. Most active fusion rows hit the cap, but improved SSE already fails to guarantee improved direction. Likewise, existing negative rotation evidence (`experiments/overnight20261002/representation/graph-results.md:102`) and cold-head factorization evidence (`experiments/overnight20261002/head_compression/decision.md:12`) provide no reason to reopen those paths.

The initial layer-freezing and prompt-anchor-cap ideas are lower priority: the plan already covers allocation and balanced exposure, while current evidence does not identify a safe layer to freeze or prove adjacent anchors are redundant. The completed reference exposed 531 prompts and 206,163 supervised rows, only 5.29% of available rows (`experiments/a8-qat-recovery/comparison.md:15`), which motivates broader exposure but does not by itself validate dropping anchors.

The orchestrator should first resolve the zero-row correlation counts and whether raw versus post-norm error predicts native quality. No local evidence licenses transfer of an EAGLE winner directly to DSpark/DFlash: their fusion/norm ordering and actual deployed A8/A1 operands must be inspected separately. No claims here change target/verifier precision, introduce mixed-precision exceptions, imply SM75 performance, or treat an adaptive controller as mathematically unchanged optimization.
