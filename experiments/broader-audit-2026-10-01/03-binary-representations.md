# Binary representations through coordinate and geometry changes

Date: 2026-10-01. Source: `f0bb92dda687d908b3cffe084510b3be757df54f`. This is a source/primary-research audit, not a new quality measurement. No model workloads, weights, large captures, accelerator/remote operations, installations, tests or final prompts were accessed. The current row-W1A8/W1A1 training remains unchanged; RTX 2080 Ti work remains paused.

## Verdict and evidence

Prepare three bounded CPU diagnostics: **affine two-value weights**, **compressed/coarse scale structure**, then **one local signed-Hadamard transform**. These respectively add a mean direction, retain magnitude differences cheaply, and change the coordinates entering sign quantization. None has established EAGLE acceptance or throughput. Avoid importing OneBit’s column vectors into A1 without accounting for their weighted reduction.

The [scale screen](../binary-scale-fitting-5080.md) establishes a useful quality/cost tradeoff: fitted row C accepted 0.290 drafts/round, group128/A16 D 0.425, Q4_0 1.042. Instrumented draft time was 7.79/14.49/3.49 ms respectively, not a controlled performance benchmark. D has 1,704,960 F32 scales versus 65,280 row scales and 218,234,880 binary weights. It stores 1.25 bits/selected weight before other tensors. Better scaling matters, but reproducing that expensive kernel is an unattractive endpoint.

## What information actually survives

For column-vector input, current row W1A1 computes `y_r = a_r beta_t B_r·s_t`, with `B,s∈{−1,+1}`. Positive row scaling before weight sign changes no bits and duplicates `a_r`. Positive channel balancing before activation sign also changes no bits; it changes only dynamic `beta_t`. That token-dependent amplitude may affect a recurrent body, but cannot restore discarded relative channel magnitudes. At a bias-free head, a common positive logit multiplier preserves greedy rank. Global amplitude can also be largely suppressed by subsequent RMS normalization; epsilon and intervening residuals prevent blanket exact invariance claims.

[OneBit v6](https://arxiv.org/html/2402.11295v6) uses `W_hat=diag(h) B diag(g)`, rank-one magnitude initialization, W1A16, knowledge transfer and added post-LayerNorm. Its reported ≥81% LLaMA task-performance retention belongs to that package. The representation is useful, but the published result does not isolate scaling and does not validate W1A1. Its sign matrix remains high rank; only its magnitude approximation is rank one.

Applying positive `g` **before** A1 gives `sign(g⊙x)=sign(x)`. Applying it **after** A1 retains `g_i` inside `sum_i B_ri g_i s_i`, which cannot become one ordinary XOR/popcount dot unless `g` is constant. Group-constant `g` needs partial dots; arbitrary `g` requires weighted/wider arithmetic. Signed `g` adds channel sign flips that can be folded into B, without rescuing magnitude information. Metadata savings alone do not imply arithmetic compatibility.

An exact wider transformation `x'=D⁻¹x, W'=WD` preserves `Wx` for invertible diagonal D, provided both occur at the same linear boundary. Binarizing afterward generally breaks equality. For positive D, the signs of both operands stay unchanged, so this chiefly alters amplitudes under global A1. It can change A8 codes by redistributing range, making channel balancing more credible for the wider-activation lane than for A1.

## 1. Affine two-value weights: best new low-cost diagnostic

Use `W_hat[r,i]=a_r B[r,i]+m_r`, with `a_r≥0`; each row’s two levels are `m_r±a_r`. Then

`y_r = a_r (B_r·x_hat) + m_r sum_i x_hat_i`.

For A1, this becomes `beta_t [a_r d_rt + m_r S_t]`, where both d and S are exact integer sign sums. The main matrix operation stays binary; S is one shared O(K) token reduction and the correction costs O(M) multiply-adds. Storage is `1+64/K` bits/weight with two F32 row values, versus `1+32/K` today, excluding padding. Across all nine matrices, the additional row centers occupy 261,120 bytes. The represented row mean is `m_r+a_r mean(B_r)`, so m alone is its mean only for balanced codes.

This is new information: the row-specific input-sum direction cannot generally be absorbed into a row scale or fixed bias. RMSNorm does not subtract input means. Fit fixed B first using a regularized two-feature least-squares solve against original output, with features `B_r·x_hat` and `sum x_hat`. Then optionally compare one frozen centered-code initialization `B=sign(W−m)`; centering without reconstructing m loses the mean rather than preserving it. Asymmetric levels do not require a balanced positive/negative bit count.

**Two-day CPU feasibility:** tiny synthetic proofs plus fitting/replay on already-small training operand subsets; no full-model evaluation is necessary. Require ≥10% lower holdout output SSE than the same-width row fit in at least two prespecified affected projections, no >5% regression elsewhere, and inspect output direction/rank rather than weight SSE alone. These are proposed triage thresholds, not inferred acceptance guarantees. If small source-row operands are unavailable, the empirical gate needs a separately prepared subset; do not silently load full checkpoints.

**Integration:** moderate. Add a row-mean tensor, shared token sum, versioned arithmetic/export rule and CPU/CUDA epilogue; preserve Q/K row permutation, zero/subnormal signs, tails and explicit scaling order. **Strongest counterargument:** pretrained rows may be nearly symmetric, making m negligible; calibration may overfit recurrent input means. Stop if the correction merely fits ordinary-state bias and fails on independent recurrent samples.

## 2. Compress the scale structure before adding more groups

First factor the existing D scale matrix approximately as `s_rg≈a_r c_g`, fixing its scale gauge, e.g. mean(c)=1. This is OneBit-like magnitude structure at **block** rather than channel granularity. Storage changes from `32MG` to `32(M+G)` scale bits, but **G partial dots remain**. Exact equivalence to D requires its scale matrix to be separable; approximation changes the model. Fit using output error, not just scale-matrix Frobenius error. A good metadata fit can still fail because group signed sums are correlated.

Next restrict G to semantic partitions: fusion’s three target taps; Q/K/V’s two separately normalized streams. Their widths are 2560 and match packed-word boundaries. The prior Prism geometry audit attributed most fusion/QKV grouping gains to these boundaries, but did not establish acceptance for this coarse candidate. Extend low-activation math explicitly:

`y_rt = sum_g a_rg beta_tg d_rtg`.

For A1 every partial dot remains XOR/popcount. A8 uses integer signed sums with per-group quantization. This preserves relative group amplitudes that global sign scaling loses; it is a new packed contract, not the current group128/A16 path. Shared Q/K/V packing requires identical partitions, quantizers and parameters; gate/up likewise share an input boundary.

**Two-day feasibility:** scale-only decomposition and synthetic scalar/packed algebra are immediate; sampled output fitting requires small operands. Gate on retaining at least 80% of D’s output-error improvement over C on the chosen A16 subset, then independently assess same-width A1/A8 error. Charge G output accumulations, scale loads and pack reductions. **Integration:** moderate/high because exporter and `W1AxContract` currently admit group128 only with A16. **Counterargument:** independent row/group adaptation, not shared channel magnitude, may explain D’s gain; coarsening can remove precisely that benefit. No acceptance estimate follows by interpolation.

## 3. Local signed-Hadamard: conditional bit-changing candidate

For orthogonal R, `x'=Rx, W'=WRᵀ` gives `W'x'=Wx` exactly in real arithmetic. Insert R immediately before one linear, after all preceding nonlinear operations. FFN down is a clear site: rotate `SiLU(gate)⊙up` and its down weights, not the gate before SiLU. With block128 R, runtime work is about `K log₂128` butterfly additions plus normalization/sign operations; signs for randomized diagonals require K stored bits unless reproducibly generated. A learned dense block rotation costs O(K·128) online arithmetic and dense metadata unless a graph-specific cancellation is proven.

[QuaRot v2](https://arxiv.org/html/2404.00456v2) reports 99% Llama-2-70B zero-shot retention at W4A4/KV4; its large-batch prefill acceleration uses INT4, not binary MMA. [QuIP#](https://arxiv.org/abs/2402.04396) combines randomized Hadamard processing, lattice vector codebooks and fine-tuning in weight-only low-bit quantization, not a sign codebook. [SpinQuant v4](https://arxiv.org/html/2405.16406v4) reports a 2.9-point W4A4/KV4 gap for LLaMA-2-7B and substantial random-rotation variance; learned rotations plus GPTQ differ from this binary drafter.

**Two-day feasibility:** one fixed seed, one projection, small-row transform/replay. Require ≥15% holdout output-SSE improvement over the best same-width scale fit before native work. The prior plain H512 weight-only probe mostly worsened error; test actual joint weight/activation output loss. **Integration:** high relative to its uncertain return. Cast order matters: `F16(Rx)` differs from `R F16(x)`. RMSNorm rotation needs norm-weight absorption; residual branches need coordinated transforms; generic rotations do not commute with RoPE; GQA/cache layouts constrain head rotations. **Counterargument:** spreading outliers helps multibit grids but can erase asymmetric structure useful to binary codes while adding launch/packing cost.

BiLLM’s residual binary planes and split membership genuinely add levels, but its approximately 1.1 parameter-bit headline excludes identification overhead and uses wider activations; it is not strict one-bit-per-weight execution. Treat it as a later precision exception. Neither OneBit nor BiLLM supplies small-N SM75 EAGLE timing. [BiLLM v2](https://arxiv.org/html/2402.04291v2)

Promote only a CPU-surviving candidate to bounded native development acceptance, depth survival and complete-operator timing. Later matched RTX 2080 Ti throughput against Q4_0, with fixed target/verifier, is decisive; no CPU result establishes SM75 speed.

The drafter is a small model that proposes tokens for the unchanged target model to check. Binary signs discard both magnitude differences and uneven distributions around zero. A cheap correction for those uneven distributions is the best new representation to examine first. A few shared scales are another possibility, while rotations need stronger evidence because they add work and can make binary approximation worse. CPU checks can reject weak ideas cheaply, but success still means beating the existing four-bit drafter in accepted proposals and total decoding speed.
