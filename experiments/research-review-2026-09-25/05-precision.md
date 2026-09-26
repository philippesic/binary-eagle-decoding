# Precision allocation: where practical one-bit benefit could remain

Local-only analysis, 2026-09-25. No web, GPU, SSH, experiments, or repository edits. Inspected parent revision `7ae1cc2e4e5ce5f1e3070377722d679d15bc486e` and local llama.cpp `34e21b7d85c17e25d5a91ce2ab1074d4c18bfe39`. Statements about unmeasured designs below are hypotheses or mathematical consequences, not results. The active goal remains the frozen all-nine W1Ax study; this note does not amend its numerical contract or authorize another goal.

## Assessment

The strongest next precision experiment is already the active plan: hold binary weights byte-identical and measure A16/A8/A4/A1. Existing results establish that the current broad post-training W1A1 recipe destroys useful proposals, but do not establish whether binary weights, discarded activation magnitudes, or their interaction dominates. Successful Q4_0 versus failed whole-row/whole-token W4A4 also makes granularity and outlier sensitivity credible suspects; that comparison does not isolate them because both weight and activation formats and kernels differ.

For preserving genuine one-bit arithmetic, the most promising bounded extensions after that result are structured block scales and narrow scale/threshold calibration, chosen from measured activation/error diagnostics. For practical compression regardless of purity, keeping sensitive fusion/attention operations at higher precision and testing a binary head on a Q4_0 body is a useful candidate, but the existing head-only result supplies a negative prior: the head already lost against FP16 and has not beaten Q4_0. Sparse or low-rank floating corrections are later escape routes with real integration and coverage costs, not free ways to claim all-W1A1.

Do not run a large quantizer grid or silently replace absmax A4/A8, identical W1 weights, or the explicit A16 cast. The untouched 24 QAT-final prompts remain reserved for the eventual frozen trained candidate.

## What local evidence actually establishes

1. **All-nine native W1A1 is primarily a quality problem in the measured workload.** On RTX 2080 Ti / SM75, native acceptance fell from 1.168 to 0.055 accepted drafts/round, while measured `draft()` time improved from 6.471 to 3.558 ms/round. More rounds made total draft-call time increase from 22.0 to 24.7 s. Decode rate was 0.561× FP16. This is not an end-to-end cost decomposition; omitted feature processing and verification costs must be measured by the new plan. [Local source: `experiments/rtx2080ti-synthesis.md`.]
2. **Sensitivity is very nonuniform.** Native W1A1 fusion/attention/FFN/head accepted 0.271/0.238/0.472/0.860 per round, versus ordinary 1.168. Their decode ratios were 0.598/0.596/0.716/0.907. Head is the least damaging tested group, but still lost. Fusion is only 9.0% of eligible parameter count, attention 19.2%, FFN 34.2%, head 37.5%. These are single-group interventions into an otherwise ordinary draft, not leave-one-group-high-precision rescues of an all-binary draft; interactions cannot be inferred by addition. [Sources: synthesis and `experiments/drafter-static-coverage.md`.]
3. **Four-bit failure is format-specific evidence.** Native whole-row/whole-token W4A4 accepted 0.092/round and reached 0.511× ordinary in its paired track. Native W8A8 accepted 1.127/round and tied ordinary at 0.990×. Standard Q4_0 accepted 1.182/round and reached 1.109× ordinary in the nine-path track. Q4_0 uses block-scale weights and observed Q8_1 activation conversion with MMVQ/MMQ. It is not the same A4 quantizer or a controlled isolation of weight block size. [Source: synthesis.]
4. **The older PyTorch results support direction, not numerical interchangeability.** BF16 fake W1A1 all/head accepted 0.202/1.677 versus ordinary 2.317 on the 5080; tree topology and arithmetic differ from native. The nine linears accounted for only 38.32% of one instrumented PyTorch draft profile, and fusion for 1.03%. Those percentages do not supply native SM75 latency shares. [Source: `experiments/pytorch-w1a1-cuda-acceptance.md`.]
5. **Better local reconstruction need not mean better proposals.** The head QAT pilot improved teacher KL 2.506→1.017 and top-1 agreement 0.477→0.542 but reduced held-out acceptance 1.677→1.565. Quantizer decisions therefore need online development acceptance after local error screening, not weight MSE or cached head KL alone. [Source: `experiments/qat-head-pilot-results.md`.]
6. **No W1A16/W1A8/W1A4 result exists in the inspected record.** Any assertion that activations dominate weights, or that A4 suffices with W1, is premature. Status/goal/plan are authoritative over older reports' historical “next step” wording.

## Decomposing weight versus activation damage

For one linear operation on an identical captured input x, let ΔW = Wq − W and Δx = xq − x. Then

`Wq xq − W x = ΔW x + W Δx + ΔW Δx`.

The planned W1Ax matrix measures the extra activation approximation conditional on binary weights; W1A16 is a weight-damage anchor plus the explicitly specified FP16 input cast. It is not a full factorial weight/activation causality study. The plan's optional W8A8/W4A4 controls help distinguish W1 versus W8 at A8 and W1 versus W4 at A4, but not pure activation-only damage with dense weights.

Use existing planned captures for per-op reference outputs `Wx`, `W1·cast16(x)`, `W1·q8(x)`, `W1·q4(x)`, and `W1·q1(x)`. If a later diagnostic adds `W·qb(x)`, label it a captured-input diagnostic, not another native primary row. Report normalized output error, direction/cosine change, norm change, and top-token margin flips where meaningful. Weight and activation MSE alone miss alignment with the output-sensitive directions. Independent op replay identifies local error; only full-model evaluation captures propagation, recurrent draft-state drift, and changed proposal paths.

Interpret the sweep conditionally:

- **A16 remains near the A1 quality floor:** changing activation bitwidth is insufficient for these fixed W1 weights. Prioritize weight representation/calibration or sensitive-layer protection. This does not mathematically rule out W1A1 QAT, but rejects further A-bit kernel work justified solely by hoped-for quality recovery.
- **A16/A8 recover, A4/A1 collapse:** magnitudes matter. W1A8 becomes a practical one-bit-weight candidate; block/clipped A4 is a distinct later experiment. Pure W1A1 requires learning or additional representation capacity, not an assumption that its packer is slow.
- **A4 tracks A8/A16:** W1A4 is a plausible frontier point; judge its actual bit-plane and full-round cost against both anchors. Four activation bit planes are not one binary dot and should not be credited W1A1 throughput.
- **Quality improves but neither anchor can be beaten even with the measured candidate draft computation free:** stop precision-specific performance investment for that configuration; another study must change quality or non-draft cost.

No strict monotonicity of acceptance with bitwidth is guaranteed: A1 uses mean-absolute sign quantization, A4/A8 use absmax grids, and proposal trajectories differ. Diagnose surprising reversals rather than forcing a monotone story.

## Structural causes worth measuring before changing formats

The graph makes three targeted granularity hypotheses plausible:

- **Feature fusion combines three different target taps** into K=7680. A single activation scale discards differences in their magnitudes under A1. Native fusion-only damage is severe, despite modest parameter share. Capture scale/energy separately for the three 2560-wide source blocks. Three group scales have much smaller metadata and reduction complexity than arbitrary small blocks; no data yet establishes that the tap distributions differ enough to help.
- **Q/K/V consume two independently RMS-normalized 2560-wide streams** (borrowed token embedding and fused state) concatenated into K=5120. RMSNorm does not center coordinates, and learned norm weights do not guarantee identical channel/tail distributions. Capture halves separately; two scales may preserve balance that one token scale discards. Q/K are also followed by RoPE and attention, where directional error can matter more than scalar norm error. Existing attention-group results do not identify which of Q, K, V, O dominates.
- **FFN down sees `SiLU(gate) * up`, K=9728**, rather than the normalized FFN input seen by gate/up. Measure skew, sparsity, tail energy, sign balance and zero-code fraction there. Calling this tensor “positive” would be wrong: up and SiLU outputs can be negative. Product formation makes a different distribution plausible, not proven. The largest K also gives a whole-vector absmax more opportunities to be set by a rare extreme.

For each layer and input slice, record mean/RMS/mean-absolute/absmax, absmax-to-RMS ratio, fixed quantiles, zero-code fraction under A4/A8, fraction of squared energy in extreme coordinates, and clipping rate under any *diagnostic* trial threshold. Under ordinary absmax scaling the clipping rate is mostly zero by construction; “zero clipping” is not evidence that precision is sufficient. At A4 every |x| below roughly absmax/14 rounds to zero (ties depend on nearest-even), whereas A8's analogous threshold is absmax/254. Static recurring outlier channels and token-specific spikes require different remedies. Do not assume either exists without captures.

## Ranked candidate extensions and stop gates

### 1. Preserve the common-weight W1Ax sweep; collect actionable precision diagnostics

**Priority: highest; already authorized by the frozen plan.** Implement exactly the frozen codes/scales/casts, capture per-layer statistics and same-input output errors, and compare online quality. This is the cheapest way to decide whether the next intervention should be in weights or activations. No weight-sign adjustment, clipping search, changed granularity, or selective-coverage row belongs inside this primary comparison.

**Gate:** finish numerical/native dispatch gates before speed claims. If an implementation cannot match its own scalar contract, there is no precision result to interpret. Keep the original/development/context diagnostics distinct and final prompts unopened.

### 2. Structured group scales, then one coarse block-size candidate

**Priority: highest plausible new true-binary representation experiment after the sweep; moderate implementation cost.** First screen semantically natural activation partitions: fusion's three taps and Q/K/V's two streams. If error is concentrated within these partitions, use `sum_g α[row,g] β[token,g] dot(sign_w_g, sign_x_g)` for paired block weight/activation scaling; activation-only group scales can retain the existing row weight scale. Every dot remains genuinely packed W1A1, followed by normal-precision group scaling, but this is a new block-scaled W1A1 contract and no longer the frozen single-dot/row-token-scale format.

A coarse G=128 or 256 diagnostic is more credible as a first general block test than starting at G=32 everywhere. With one F32 scale per G binary weights, weight storage is `1 + 32/G` bits/weight before padding/metadata: G=32 gives 2.0, G=64 1.5, G=128 1.25, G=256 1.125. Activation block scales have the same overhead per packed activation, plus reduction/packing work. A scale per input channel would largely remove the compression advantage and, with both sides scaled, replace a single popcount dot by weighted terms.

**Counterargument:** global mean-absolute scaling is already the least-squares optimal scalar for a fixed sign vector. Gains require heterogeneity the old scalar misses, not merely renaming it RMS. Finer blocks add output accumulations and metadata loads; small N may erase the arithmetic benefit. **Gate:** choose one partition from captures, require material development acceptance improvement and favorable complete-operator cost, and stop if only tensor reconstruction improves. Do not conduct all combinations of group sizes/layers.

### 3. Protect sensitive groups; binary head with a quantized ordinary body

**Priority: strong pragmatic hypothesis, explicitly mixed precision; moderate export/loader work.** Fusion and attention show a poor acceptance-versus-parameter tradeoff; head is the least damaging tested binary group and owns 37.5% of eligible weight parameters. A bounded follow-up could compare a Q4_0 body + W1A1 head against Q4_0 everywhere, or preserve fusion/attention while applying the best W1Ax representation to less sensitive groups. This is an additional candidate after the all-layer screen, not a replacement for it.

Selection should maximize full-round savings per acceptance loss, not number of binary tensors. Protecting one group in an all-binary model is a distinct intervention from binarizing only that group in FP16; do not infer the rescue from the old ablation. Candidate groups interact through residuals, normalization and recurrent draft states. One development-screen combination followed by native cost screening is defensible; an unconstrained 2^9 search is not.

**Counterargument:** untrained head-only already reached only 0.907× FP16; Q4_0 is a stronger anchor, and shrinking an already Q4 head has less headroom than shrinking an FP16 head. A higher-precision body may improve acceptance but also remove most binary cost savings. **Gate:** no native implementation unless measured development acceptance plus a same-input cost bound leaves credible headroom versus Q4_0. Report exactly which groups/parameters/dynamic calls remain binary.

### 4. Calibrated scales or thresholds without adding dense residual computation

**Priority: narrow, inexpensive numerical screen; contract changes require review.** Distinguish four mechanisms:

- Weight row scale α can be fit to linear-output error rather than weight MSE, or learned in a target-aligned recipe. For captured pre-scale outputs z and desired row output y, the unconstrained least-squares scalar is Σ(yz)/Σ(z²), with an explicit positivity rule if positive scales are required. This changes row-relative logits and can affect head ordering. It is not proof of acceptance recovery.
- A learned global positive activation scale β on a bias-free head cannot change exact-real greedy argmax: all output rows receive the same factor. It can affect softmax confidence/branch ordering across contexts, sampling and floating rounding; it is not a general route to repair head top-1 choices. Under the native common greedy/p_min=0 policy its direct head-quality potential is especially limited.
- Symmetric clipping before zero-threshold A1 keeps all nonzero sign bits unchanged. If β is recomputed after clipping it changes a token-wide scalar; it does not recover lost activation direction. A8/A4 clipping can instead devote more codes to the bulk at the expense of tails, making it worth a separate later calibrated candidate if captures show absmax domination.
- A nonzero activation threshold `sign(x−τ)` actually changes bits and can be shared per layer, per semantic block, or per channel. Small threshold arrays are cheap storage but add reads/subtraction/comparison during packing. Weight thresholds are offline sign choices and need no runtime thresholding, but changed signs/scales break the common-weight experiment and the current audit provenance. Median thresholds maximize bit balance, not necessarily useful information; choosing them from sign entropy alone is unjustified.

**Counterargument/gate:** train/calibrate only on allowed calibration/development data, keep one declared candidate and online acceptance endpoint, and stop if gains are confined to MSE/KL or confidence rescaling. Existing `W1A1Linear` caches signs/scales by weight version/dtype/device but not config identity; mutating its config in place can silently reuse stale weight quantization. Construct fresh wrappers or explicitly invalidate caches before any future config ablation. The fake-uniform wrapper has the stronger cache key. This is a future-experiment correctness risk, not an observed fault in sealed runs.

### 5. Affine centering with explicit correction terms

**Priority: conditional on measured sign imbalance/mean structure; medium-to-high contract cost.** A nonzero mean is not removed by RMSNorm. Simply subtracting means and discarding them changes the model; represent both binary codes and the correction explicitly. For scalar-per-row/token means, write `w≈μw+α sw`, `x≈μx+β sx`, giving

`dot≈αβ dot(sw,sx) + α μx sum(sw) + β μw sum(sx) + K μw μx`.

Weight sign sums and μw can be stored; activation mean/sign sums can be computed during packing. The expensive cross term remains a binary dot, but extra normal-precision terms carry information absent from the current symmetric representation. Call it an affine binary-core operator and disclose auxiliary values/cost. It is not the current W1A1 reference with a harmless preprocessing step. Fixed per-channel activation centering can alternatively induce a precomputed output offset, but channel-dependent dynamic scales/means generally do not collapse so cheaply.

**Counterargument:** good reconstruction of common-mode components may not recover acceptance; an added bias at a formerly bias-free head can change token preferences substantially. Per-block affine corrections further multiply metadata/work. **Gate:** require a captured error decomposition showing systematic mean-related residual energy and an online improvement before any native extension.

### 6. Static outlier-channel rescue or diagonal rebalancing

**Priority: conditional on concentrated, stable outliers; mixed precision or transformed binary contract.** A small fixed channel subset may be handled at A8/FP16 while the remainder stays binary. Keep the subset fixed from calibration, preserve its exact channel map, and measure gathers and correction arithmetic. Binary weights with an FP16 activation side path are W1 with mixed activation precision; FP16 weight/activation outlier columns are mixed on both sides. Dynamic top-k outlier selection incurs per-token selection/index traffic and is a poor first candidate for N=1.

Replacing fraction f of binary weights with FP16 has a lower-bound payload near `1+15f` bits/weight before scales/indices (1% gives 1.15). It does not imply only 1% runtime overhead. A diagonal transformation `W D` with input `D^-1 x` is exact before quantization and may rebalance outliers, but adds online scaling unless a graph-aware fold is proved. Folding through SiLU, residual sums, concatenations or shared input consumers is not generally free. Start only if captures show repeatable channel concentration.

**Counterargument/gate:** diffuse errors cannot be fixed by a tiny subset; rescue may require so many coordinates that Q4_0 is simpler/faster. Stop if a predeclared small budget does not recover development acceptance or if the correction's measured complete cost consumes available headroom.

### 7. Low-rank residual correction

**Priority: later fallback, not part of frozen W1Ax or the unchanged-forward QAT revisit.** Approximate a residual `R=W−α sign(W)` using `UV^T`, then execute `binary_base(x)+U(V^T x)`. The floating correction can restore directions lost by binary weights; using original x can additionally restore activation magnitude information, whereas correction fed only binary x cannot generally recover that information. Identify which variant is meant.

For the 32000×2560 head, rank 8 adds 276,480 FP16 parameters, 552,960 bytes, roughly 5.3% of the current packed-head signs+F32-row-scales payload. This small parameter budget still adds two floating operations, launches, intermediate storage, and input traffic at N=1. A residual approximation chosen by weight SVD is not necessarily optimal under actual activation covariance or target-token acceptance. This may yield practical binary-plus-FP16 hybrid execution, never all-W1A1 coverage.

The local packed EAGLE graph explicitly rejects draft LoRA adapters. This is not supported by attaching an adapter to the present packed artifact; it requires a versioned export/loader/graph branch, source-to-GGUF layout audits, correction-path precision and dispatch proof. Merging the correction into dense weights and binarizing again removes the explicitly retained residual and is another experiment.

**Gate:** one tiny rank budget only after evidence that residual error concentrates in a few output-relevant directions. Reject if online acceptance does not improve enough to pay the measured branch cost. A cheaper Q4_0 solution remains the practical comparator.

### 8. Ternary/multiple binary bases

**Priority: lowest in this goal; explicit scope decision.** Adding a zero state or multiple binary weight bases can reduce distortion, but ternary is not one bit per independently addressable weight, and two binary bases are not the present single W1 dot. A dense ternary representation generally needs masks or multi-bit codes; two binary bases require additional dots/scales. A4 activation bit planes likewise perform multiple binary suboperations with signed combination. Such designs may be useful, but require their own stored-bit count, effective precision, arithmetic count and full-round measurements. Do not compare them to W1A1 while hiding the extra capacity as “scaling.”

## Export and numerical feasibility

The current converter `third_party/llama.cpp/conversion/llama.py:17` packs original BF16 signs after source layout mapping and computes F32 mean-absolute row scales. The EAGLE loader `src/models/eagle3.cpp:68–140` accepts only version 1, `nonnegative_is_one`, `f32_mean_abs`, arithmetic F32, exact group-to-tensor membership, I32 packed signs/F32 scales, and no dense shadow. Its metadata currently groups all Q/K/V/O and all gate/up/down together; per-projection allocation needs a deliberate schema change, not false group declarations.

The native CUDA W1A1 implementation in `ggml/src/ggml-cuda/w1a1.cu` consumes F32 inputs, handles signed zero as +1, computes mean magnitude using double accumulation then an F32 token scale, masks tails, computes integer XOR/POPCOUNT, and applies weight then activation scale. The older fake-binary module computes in operand dtype and therefore is not an exact native F32 oracle. `kernels/binary_reference.py` explicitly documents this distinction. A new threshold/scale/block method must freeze zero/tie handling, finite-value policy, rounding, logical K/tails, correction order, and final dtype before exporter/model comparisons.

Arbitrary positive learned row scales are mechanically close to the present storage, but labeling them `f32_mean_abs` is false and conversion currently recomputes rather than reads them. Learned activation thresholds, block scales, offsets, outlier maps and residual matrices need new metadata and loader validation. Preserve separate source hashes and quantizer parameter hashes; validate every transformed row, not just a few logits. Changing scale precision to FP16 saves little for one row scale (head overhead is only 32/2560=0.0125 bits/weight); it becomes material for fine blocks and then changes the arithmetic contract.

For W1A8 and W1A4 specifically, one-bit weights remain shared and packed, but activations and arithmetic are higher precision. Signed-code bit-plane representation must document sign extension, forbidden codes (−8/−128 in the current symmetric ranges), zero, padding and negative contributions. These are exact-integer contract issues, not reasons to relax the primary sweep. W1A16 sign-add execution is one-bit-weight execution with FP16 input values, not true one-bit activation execution.

## Honest accounting and a bounded sequence

For every candidate report: weight payload including all scales/masks/indices/residuals, runtime activation payload, temporary buffers, loaded and peak model memory, group-level precision, logical versus physical binary coverage, observed invocation/N histograms, pack/transform/dot/correction/rescale complete cost, and emitted tokens per measured complete round. Parameter share is not compute share; a “99% binary weights” hybrid can spend most draft time elsewhere. Latent full-precision training weights are acceptable for QAT but should not stay loaded or execute as shadow matrices during a claimed packed benchmark.

A compact sequence compatible with current decisions is:

1. Finish the frozen W1Ax numerical and same-device measurement matrix and its already planned diagnostics. No trained or new-quantizer row enters this baseline.
2. From those captures, select **one** mechanism: weight damage → row/block weight representation; activation magnitude damage → structured A scales or one A4 clipping candidate; sensitive early-layer propagation → one mixed allocation. Document a pre-run amendment/separate follow-up with immutable references before inspecting its selection results.
3. Screen local output error and development online acceptance, then calculate a conditional full-round headroom bound using both FP16 and Q4_0 anchors. Do not derive a universal acceptance threshold from old `draft()` timing.
4. Implement one native survivor only if its contract/export are explicit and its headroom survives estimated overhead. Stop on failure rather than cycling across all eight families. The existing target-aligned head-QAT revisit stays separate and retains its fixed forward rule unless the user approves a recorded amendment.

The most useful likely outcome is an attribution: whether binary weights can support useful proposals at higher activation precision, and which small amount of additional information buys the best measured full-round benefit. A negative result is still decisive if it eliminates precision/kernel combinations without consuming the untouched final set.

## Local reading map

- `AGENTS.md`; `docs/STATUS.md`; `docs/PROJECT_OVERVIEW.md`; `docs/EVALUATION.md`; `docs/DECISIONS.md`.
- `docs/goals/w1ax-activation-precision-suite.md`; `experiments/w1ax-activation-precision-plan.md`; `experiments/qat-revisit-plan.md`.
- `experiments/rtx2080ti-synthesis.md`; `experiments/drafter-static-coverage.md`; `experiments/eagle3-graph-audit.md`.
- `experiments/pytorch-w1a1-cuda-acceptance.md`; `experiments/pytorch-int4-int8-cuda-acceptance.md`; `experiments/qat-head-pilot-results.md`.
- `src/w1a1_eagle/fake_binary.py`; `src/w1a1_eagle/fake_uniform.py`; `src/w1a1_eagle/qat_head.py`; `kernels/binary_reference.py`.
- `third_party/llama.cpp/conversion/llama.py`; `third_party/llama.cpp/src/models/eagle3.cpp`; `third_party/llama.cpp/ggml/src/ggml-cuda/w1a1.cu`.
