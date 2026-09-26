# PrismML and a better quantization/QAT strategy

**Session:** 2026-09-26 01:22:50–02:08:04 UTC (September 25 local time), **45 minutes 14 seconds** before publication. Seven Astra-high agents audited claims, artifacts, optimization, quantizer mathematics, supervision, causal alternatives and execution feasibility, then challenged each other's recommendations. Completion and final checks are recorded below. No remote GPU work is part of this review; the concurrent W1Ax suite retains its owner.

## Answer

The concern about our quantizer is justified. Our current all-nine W1 weights are the original signs with one mean-absolute scale per output row. That is a useful arithmetic baseline, but it is a very weak test of what an optimized binary-weight drafter can achieve. The previous 500-step experiment trained only the output head against its original-head outputs; it was not whole-drafter recovery.

There is also an important correction: the checked **98.2%** claim belongs to **ternary Bonsai 2 27B**, with higher-precision activations, grouped scales, rotations and some higher-precision parameters. It is aggregate task-score retention, not token agreement. Earlier true-binary Bonsai releases report materially lower retention. The released weights and runtime are inspectable; the procedure that produced those weights is not reproducibly disclosed in the sources checked. We cannot honestly say its secret is ordinary QAT, GPTQ, or group-128 scaling alone. [Official Bonsai 2 announcement](https://prismml.com/news/bonsai-2-27b), [versioned whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/9ef32054fe44797376792c869891163083d64bd0/bonsai-2-27b-whitepaper.pdf).

My recommendation is to replace a kernel-first binary recipe with a **representation and recovery study**: verify the graph bridge, use the existing grouped Q1 control, calibrate against actual activations, and determine whether head/interface adaptation or wider QAT is needed. Keep activation precision high during diagnosis; return to A1 as a separate gate. This is experimental triage, not a theorem about the best achievable W1A1 model. A two-prompt CPU smoke of naive Q1_0 already produced weak proposals, so grouping alone is not a demonstrated recovery. This is a proposed next research phase, not a change to the frozen suite or permission to start extra training.

## 1. What PrismML actually demonstrates

All external scores here are author-reported; we did not reproduce their benchmark suite.

| Release | Relevant weight/activation distinction | Reported score comparison | Interpretation |
| --- | --- | --- | --- |
| Original binary Bonsai 8B | Binary weight codes, scale per 128 values; wider activations | Six-task headline 70.5 / 79.3 ≈ 88.9% | Useful compressed model, not a 98% result. |
| Original binary Bonsai 4B | Same family; nearest parent family to our Qwen3-4B | Ten-task mean 55.39 / 68.31 ≈ 81.1% | Some capability losses remain large; not a drop-in EAGLE model. |
| Binary Bonsai 27B | Binary grouped weights; wider activations | 15-task mean 76.11 / 85.07 ≈ 89.5% | Different parent, size and inference protocol. |
| Ternary Bonsai 27B | Three-valued weights | Same table 80.49 / 85.07 ≈ 94.6% | Higher quality at a larger weight footprint; not an isolated capacity ablation. |
| Ternary Bonsai 2 27B | Rotated ternary weights; precision exceptions | Approximately 98.2% aggregate retention | Does not establish W1A1 quality or speculative acceptance. |

Sources: [binary 8B announcement](https://prismml.com/news/bonsai-8b), [8B/4B whitepaper, Tables 5–8](https://github.com/PrismML-Eng/Bonsai-demo/blob/9ef32054fe44797376792c869891163083d64bd0/1-bit-bonsai-8b-whitepaper.pdf), [27B whitepaper, Table 14](https://github.com/PrismML-Eng/Bonsai-demo/blob/9ef32054fe44797376792c869891163083d64bd0/bonsai-27b-whitepaper.pdf).

The current Bonsai 2 PDF uses 20 `xhigh` tasks; its model card uses a separately presented 14-task average. Both are near 98.2%, but their rows must not be combined. Medium-effort and separately reported agentic results are weaker. Equal token caps also do not establish equal consumed reasoning compute. The headline is reasonably supported as an approximate aggregate, not a uniform capability guarantee. [Pinned card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/b072e1d3b35a0a630cece372c2127528e0994386/README.md), [whitepaper §4, Tables 10–11](https://github.com/PrismML-Eng/Bonsai-demo/blob/9ef32054fe44797376792c869891163083d64bd0/bonsai-2-27b-whitepaper.pdf).

**Why this is not our acceptance metric:** two models can answer the same question correctly using different first tokens. Their task scores can match while greedy speculative acceptance is zero at that position. No positive lower bound on accepted-prefix length follows from a task-score retention percentage. We need paired conditional logits, selected-token agreement and live prefix survival. A poor drafter can still produce correct final answers because the target rejects its proposals; the cost is more rounds. Our acceptance metric measures that acceleration opportunity, not standalone answer accuracy.

### Known representation; unknown checkpoint-producing recipe

The original Q1_0 format stores 128 sign bits and one FP16 scale: **1.125 bits per matrix weight**. Its MLX encoding stores an additional mathematically redundant affine bias, yielding 1.25 bits. Norms and runtime state remain floating point. CUDA's vector path expands signs and uses Q8_1 activations with integer byte arithmetic; MLX has floating-activation paths. This is weight compression, not our W1A1 XOR/popcount operator. [Pinned format/code](https://github.com/PrismML-Eng/llama.cpp/blob/adfffbe41b2cabcd51fff326ab045662265062bb/ggml/src/ggml-common.h), [CUDA dot implementation](https://github.com/PrismML-Eng/llama.cpp/blob/adfffbe41b2cabcd51fff326ab045662265062bb/ggml/src/ggml-cuda/vecdotq.cuh), [MLX artifact card](https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit/blob/019934f87a61a654e3960ea22f53688e0d2c49ba/README.md), [MLX arithmetic](https://github.com/PrismML-Eng/mlx/blob/752b2d1999f82d81a7ec8021123e87f1b408617e/mlx/backend/metal/kernels/quantized.h).

Bonsai 2's PTQ1_0 uses ternary packing, FP16 scales and a signed block-Hadamard transform. The artifact also preserves small state-path projections and other floating tensors. Its codec name does not make it binary. Author quality suites and device throughput measurements use different serving paths; do not combine them into a single measured quality-at-latency result without a matched runtime evaluation. [Versioned format specification](https://github.com/PrismML-Eng/Bonsai-demo/blob/9ef32054fe44797376792c869891163083d64bd0/MODEL-FORMATS.md).

The generic public packer computes signs and mean-absolute scales. That only explains storage conversion. To test the distinction, we read **63 fixed, predeclared groups** from released binary Bonsai 8B and the named public Qwen3-8B family checkpoint, avoiding Q/K layout ambiguity. Among 8,064 values, 1,697 signs differed; none of the 63 groups matched naive packing. Exact source-parent revision remains unverified, and the sample is not random or representative of the whole model. It nevertheless rejects literal RTN of that specific public checkpoint as an explanation for those groups. It does **not** identify an optimizer. Range-read revisions, offsets and hashes are recorded with the artifacts below.

Official releases describe TPU training, while the whitepaper identifies proprietary compression IP. Related [Caltech mirror-descent research](https://azizan.mit.edu/papers/RMD.pdf) and [patent application 20260220467](https://patents.justia.com/patent/20260220467) disclose possible ideas but do not identify themselves as the released Bonsai recipe. We found no checked first-party basis for the blanket claim that this was achieved with no retraining. [Official binary launch](https://prismml.com/news/prismml-launches-worlds-first-1-bit-ai-model), [27B launch](https://prismml.com/news/prismml-releases-bonsai-27b).

## 2. What the new project evidence changes

The [historical SM75 matrix](w1ax-activation-precision-results.md) reports all-nine W1A16 acceptance **0.106 drafts/round**, W1A8 **0.102**, and ordinary FP16 EAGLE **1.168**. Restoring activation precision therefore does not rescue the current row-scaled binary-weight recipe. This rejects “A1 activations alone caused the collapse,” not “one-bit weights cannot work.” Coverage, weight fitting, head/body compatibility and graph fidelity remain distinct explanations.

**Late-session evidence update:** concurrent commit `7ac0dd2` sealed the
24-prompt development matrix (960 requests, same SM75 primary build/policy).
W1A16 accepted **0.117 drafts/round**, W1A8 **0.111**, W1A1 **0.055**,
versus ordinary **1.047** and Q4_0 **1.036**. Thus poor quality with A16
persists beyond the historical prompts. All speculative variants matched
target-only raw IDs on all 120 development pairs; the historical mismatch
remains a separate unresolved observation. We reviewed the committed report,
not independently reprocessed remote raw data. Remaining attributed traces
and other diagnostics are still owned by the suite task.

The historical speculative paths still differ from target-only on one prompt; identical speculative outputs do not establish strict target equivalence.

The serving timings also have a narrower meaning than the bit labels imply. The current A16 custom kernel performs scalar sign-add work, and its A8 counterpart is not the existing Q1_0/Q8_1 vector path. Slow A16 reference execution is not a performance ceiling for optimized one-bit-weight inference. Likewise, all variants emitting the same verified text says little about draft quality: the target verifier can reject bad drafts and still emit the same output.

### New local measurement: every selected source weight

We performed a read-only CPU geometry audit of **65,280 output rows and 218,234,880 BF16 weight values** across all nine selected matrices. No activations, prompts, model inference or GPU were used for this measurement. NMSE below is weight squared error divided by source weight squared norm, not accuracy loss.

| Matrix | Row-scale binary NMSE | Group-128/F16-scale binary NMSE | Reduction of weight residual SSE |
| --- | ---: | ---: | ---: |
| Feature fusion | 0.4290 | 0.3670 | 14.45% |
| Attention Q | 0.3732 | 0.3613 | 3.19% |
| Attention K | 0.4263 | 0.3591 | 15.76% |
| Attention V | 0.4205 | 0.3685 | 12.37% |
| Attention output | 0.4065 | 0.3540 | 12.90% |
| FFN gate | 0.3619 | 0.3592 | 0.74% |
| FFN up | 0.3618 | 0.3591 | 0.74% |
| FFN down | 0.3661 | 0.3633 | 0.77% |
| Vocabulary head | 0.3549 | 0.3522 | 0.74% |

This is a direct reason not to assume group-128 alone reproduces Prism. It changes little weight geometry in the largest head/FFN matrices. Conversely, it is meaningful in several early matrices. Fusion's three 2560-wide source partitions capture 87% of its group-128 gain; the two input streams capture 75%/96%/93% for Q/K/V. These are useful calibration structures, not measured acceptance improvements.

For unrounded least-squares scales and fixed original signs, the exact weight-error reduction is

`SSE_row - SSE_group = sum_g |g| * (mean_abs_g - mean_abs_row)^2`.

Thus grouping restores variation in magnitude between groups. It does not optimize signs against the actual input distribution. A synthetic correlated-input counterexample confirmed that lower weight SSE can even accompany higher output error. The row reference here sums in higher precision then rounds its scale to F32; it is a geometry reference, not a claim of byte-identical native reduction.

A small prespecified plain normalized block-H512 probe, without sign diagonals, on 128 rows each of head/fusion/K/O did not justify mandatory rotation: groupwise weight residual error improved 1.45% in fusion but worsened 0.42–2.21% in the other three (relative changes, not percentage points). This does not test activation outlier or covariance benefits. Rotation remains a conditional experiment, with transform cost and FP16-cast order explicitly included.

## 3. A closer control is already available

Our pinned llama.cpp already contains **Q1_0 group-128 storage and CPU/CUDA paths**. A Prism fork transplant is unnecessary for the first unrotated control.

We executed a local CPU `--pure Q1_0` conversion of the existing ordinary FP16 EAGLE GGUF. All nine matrices became Q1_0, and the other five tensors were preserved. An exhaustive source-row audit found **zero mismatched packed sign bytes and zero mismatched FP16 scale words**. The resulting file is **36,920,320 bytes**, with **30,689,280 bytes** of quantized matrix payload. The current row-binary matrix payload is 27,540,480 bytes: group scales add about 3.15 MB, not another model-sized residual.

This artifact is a practical control, not Prism's optimized checkpoint and not a W1A1 result. It derives from the existing **F16 GGUF**, whereas the custom W1Ax study derives signs/scales from the pinned BF16 source. Source casts, row mappings and scale rounding must be controlled before attributing differences. Standard Q1's CUDA activation contract also differs from our whole-token A8 rule: Q8_1 uses smaller blocks and different rounding/scale storage. `GGML_W1AX_ACT_BITS=16` does not turn standard Q1_0 into A16.

The export audit proves storage/numerical conversion only. Any local CPU regression smoke is separately recorded below and cannot validate SM75 dispatch or throughput.

## 4. Diagnose the failure before increasing training scope

### Gate 0: ordinary / cast-only / dense binary / packed binary bridge

Compare four versions on the same exact prefixes and inputs:

- **O:** ordinary graph and ordinary weights.
- **C:** ordinary weights with the proposed explicit activation casts, separating cast effects from weight approximation.
- **B:** ordinary floating operations implementing the same signs, scales and activation boundary casts as the quantized candidate.
- **P:** native packed candidate.

Compare draft-layer outputs, pre/post-normalization state, retained K/V, draft logits and proposed IDs—not only final target tokens. Test initial, zero-accept, partial-accept and full-accept transitions. O versus C diagnoses cast effects; C versus B diagnoses weight approximation; B versus P diagnoses graph/export/packing divergence. Materialize historical F32-scaled effective binary weights in F32, or keep the scale as an explicit F32 post-reduction operation; materializing them in F16 adds another quantizer. Scaling before versus after a reduction can change rounding, so define tolerance and near-tie diagnostics rather than promising bitwise equivalence from algebra alone.

### Gate 1: separate head error from body/head incompatibility

Under a common forced token history, compare ordinary/binary body states crossed with ordinary/binary heads. A poor old head on binary states does not prove the body destroyed useful information: an invertible change of basis could require a new readout.

Before all-body QAT, fit a **training-only regularized readout** on frozen binary-body states, with development evaluation. Treat a dense or higher-precision readout as a diagnostic oracle, not a deployable strict-binary success. If it restores proposals, prioritize interface/head adaptation. If it fails, the test has not proved information-theoretic impossibility; it has narrowed the cheap remedies.

For supervision, distinguish three teachers:

1. The old head on student states: operator reconstruction, as in the pilot.
2. The whole ordinary drafter on its own state at the same token prefix: drafter distillation.
3. The actual target verifier at that prefix: target alignment.

Applying the target LM head directly to arbitrary EAGLE states is not target supervision. Cache ancestry and state/label alignment must be explicit.

## 5. Fit the representation to activations

The relevant linear reconstruction error is `E[||(W-Q) x||²]`, governed by the **uncentered second moment** `H=E[xxᵀ]`. Here x is the actual input after the fixed boundary cast; use the cast-only reference C when isolating weight error. Weight MSE corresponds to an isotropic-input surrogate. Subtracting the activation mean changes this bias-free objective.

The smallest interpretable scale screen holds A16, source weights, original signs, designated training calibration data, scale dtype and reference arithmetic fixed. Use F32 scales throughout the four-cell reference, keep the historical native anchor separate, and evaluate rounded F16 exports afterward:

| | Mean-absolute scale | Activation-output-fitted scale |
| --- | --- | --- |
| One scale per row | Row representation control | Tests objective without extra group capacity |
| Scale per 128 values | Tests group capacity alone | Tests both, with the other cells separating effects |

For a fixed output row, form `z[t,g] = sum_{i in g} sign(w_i)*x[t,i]`, then fit positive group scales against the reference output. Streaming `ZᵀZ` and `Zᵀy` avoids a full K×K Hessian. Use nonnegative least squares or a valid constrained method; clipping an unconstrained solution is not equivalent. Ridge regularization toward the baseline keeps that baseline feasible and can control drift.

The continuous calibration objective can have a non-increase guarantee relative to its anchor. That guarantee does not survive arbitrary FP16 rounding, approximate optimization, held-out inputs or nonlinear recurrence. Evaluate the actual exported scales, retain a baseline fallback, and gate on development logits/proposals. Do not convert an MSE guarantee into an acceptance claim.

If this fails, **sign optimization** is a separate next intervention: covariance-aware error compensation or learned signs/scales with task supervision. Released Bonsai codes differing from naive source signs makes this worth investigating, but does not select its secret algorithm. Public low-bit methods support the possibility while differing in activations, exceptions and budgets. [OneBit](https://arxiv.org/html/2402.11295v3), [BiLLM](https://arxiv.org/html/2402.04291v2), [ParetoQ author implementation](https://github.com/facebookresearch/ParetoQ).

### A concrete sign-fitting alternative

For one row, fixed positive group scales and fixed calibration inputs, let
`r = sum_j alpha_g(j)*s_j*x_j - y`. Flipping one sign changes mean squared
output error by

`Delta L = -4*alpha_g*s_j*E[x_j*r] + 4*alpha_g^2*E[x_j^2]`.

A flip helps this calibration objective when the first correlation term wins.
This is our algebraic proposal, independently checked by the quantizer reviewer,
not an assertion about Prism's method. With diagonal input second moments and
a linear teacher using the original weights, original weight signs already
admit no improving fixed-scale single flip; correlated inputs enable the
difference. Update residuals after a flip or evaluate proposed groups jointly:
simultaneous flips add cross-terms and cannot be approved independently from
one stale gradient. Export rounding, development quality and recurrent
behavior still require checks. The final code/scale layout can remain Q1_0;
better fitted signs do not inherently require another inference format.

## 6. A better bounded QAT experiment

The old pilot presented approximately **32,000 cached rows** (500×64), against 32,768 cached training rows. This was below one nominal pass, not extensive whole-model binary adaptation. It also trained just the 81.92M-weight head. These facts do not prove undertraining caused failure; they prevent treating that failure as a capacity limit.

Recommended controls for a new, separately scoped recipe:

- Start with the smallest scope supported by the preceding tests: fitted scales, head/readout adaptation, or body plus head. Do not automatically optimize all 218M weights.
- Keep teacher and loss effects separate. Compare two teachers with the **same loss family and valid-row mask**; old-draft KL versus target CE changes two factors. Capture target probabilities even if the first training loss is hard CE.
- Use a deployment-matched hard forward, explicit zero-sign rule and export/rounding audit. Preserve the distinction between BF16 tree simulation and the native FP16-target chain; neither its acceptance numbers nor teacher states are interchangeable.
- Select checkpoints by online development acceptance at fixed policy, using the native chain or a demonstrated equivalent validator. Log calibration, ranking, root/depth survival and fresh-state drift separately. For the fixed-length greedy chain at `p_min=0`, a token-wide positive temperature change can lower KL without changing selected tokens. Tree pruning and confidence-based stopping can react differently.
- Use exposure counts, unique states, effective passes and a declared wall cap, not steps alone. A proposal in this review is a bounded 128K valid-position budget with fixed checkpoints, but it **does not replace the currently authorized 500-step/45-minute pilot** without a research decision.
- Use a hard-forward learned-scale STE as the first sign-optimization baseline, preserving explicit zero-sign and scale/export rules. Do not implement a patent-based optimizer on the assumption that it is Prism's method. Keep hard quantization visible during optimization and checkpoint selection. Soft continuation may help optimization yet fail at the hard endpoint. A mirror-descent or relaxation control belongs only after sign/mask diagnostics justify it. Hold teacher/data/representation fixed, select using the hard path, and define conversion from dual/soft parameters into the common hard-stage latent state; resetting optimizer moments alone does not equalize that stage.
- Wider body QAT must use faithful recurrent states, masks, positions and separate teacher/student cache histories. Add auxiliary hidden/attention losses only after an output-loss baseline identifies a need. Published methods disagree on which auxiliary alignment helps. [EAGLE-3](https://arxiv.org/html/2503.01840v3), [DistillSpec](https://arxiv.org/html/2310.08461v2).

Do not reject a QAT checkpoint because its latent weights perform worse when executed as an ordinary floating model. Those weights were optimized for a quantized forward. The deployed quantized path's acceptance is the decisive gate; the ordinary latent forward is a diagnostic.

Practical arithmetic: FP32 latent weights, gradients and two Adam moments for all selected matrices occupy approximately **3.25 GiB**, excluding activations, teacher, logits, framework temporaries and allocator overhead. Cached 32k-wide teacher targets for 12,000 states need about **0.768 GB** at 16 bits or **1.536 GB** at F32. Sparse top-k targets save space but cannot reconstruct exact dense KL from a normalizer alone. These estimates motivate streamed capture/training, not a claim that the full target and training graph fit together.

## 7. What not to prioritize yet

- **Another custom binary kernel:** the existing Q1 control removes the immediate format-engineering prerequisite. Native speed work follows quality and complete-round evidence.
- **A compulsory Hadamard transform:** it is part of newer Bonsai's representation, not proven necessary here. For a proposed block-512 right rotation, `Wx=(WRᵀ)(Rx)` holds before quantization; casts, SiLU, norms, residuals and RoPE cannot be crossed casually. Share compatible transforms for Q/K/V and gate/up and measure their cost.
- **A full Bonsai-4B draft integration:** its tokenizer basis is promising, but loaded vocabulary lengths, EOS/chat behavior and configuration differ. It is a 36-layer model versus the one-layer EAGLE component. A small common-prefix quality diagnostic may be useful later; an architecture change is not required to test grouped weights now.
- **A claimed reproduction of Prism's secret method:** one public reproduction repository evaluates calibration and perplexity on the same short text in several scripts, changes text/baselines between methods, and folds transformed weights back into dense floating storage. Its sign/scale fingerprint does not uniquely establish GPTQ or absence of training. It is not evidence for near-lossless generalization or deployable one-bit throughput. [Audited source snapshot](https://github.com/ThakiCloud/bonsai-1bit-repro/tree/a55e89a90894a3bb8b30deba48fdc0b29d55cbe7).

## Recommended sequence and decisions

1. Finish and preserve the remaining diagnostics of the concurrent W1Ax suite; its historical and development primary matrices are now sealed. Reuse historical/development captures only for parity and diagnosis. NNLS, readout fitting and QAT are training: collect them from designated training prompts, evaluate on development and keep the reserved final set untouched.
2. Pass the O/C/B/P graph bridge and evaluate the existing grouped Q1 control under explicit source/activation contracts.
3. Run the small fixed-sign scale screen and body/head/readout diagnostics. Choose the smallest remedy supported by results.
4. Freeze one recovery recipe, loss/teacher contract, exposure budget and stop rule. Only then run a fresh development-selected and final-evaluated QAT experiment.
5. If weight-only quality recovers, assess grouped A8/native Q1 execution against both FP16 and Q4_0. Investigate A1 separately; never relabel a successful weight-only result as W1A1.

The main open research choice is whether the next phase prioritizes **a practical one-bit-weight drafter** or immediately insists on **one-bit weights and activations**. The former provides a cleaner recovery target and a readily available kernel path. The latter remains the original research endpoint, but Prism's headline does not supply evidence that it should be reached with the current quantizer or small head-only pilot.

## Reproducibility and review record

- Starting project revision: `66192ae`; `7ac0dd2` development results were reviewed during the session. Concurrent updates are integrated without taking ownership of the GPU suite.
- Source checkpoint: `models/hf/Qwen3-4B_eagle3/model.safetensors`, SHA256 `58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e`.
- Full geometry script: [prism-weight-geometry.py](prism-weight-geometry.py). Raw JSON SHA256 `74120b5bfbbc50f59e6e33b4a2e637cc8b34c88534a71ac19884db613c734d50`. Full coverage independently reviewed by quantizer and skeptic agents.
- Q1 audit reproducer: [prism-q1-export-audit.py](prism-q1-export-audit.py), using the pinned `gguf-py` reader and NumPy.
- Q1 export/source-row audit: `results/prism-research-20260925/q1-export-audit.json`; records source GGUF, binary and artifact hashes. The artifact and raw logs stay outside Git. Preserved artifact: `results/prism-research-20260925/Qwen3-4B-eagle3-q1_0-control.gguf`, SHA256 `0c28d7c02b83c19cc66421dd2d9f29e22f3b3b483e448dc9a030f8b3337d18b7`.
- External range audit: Qwen3-8B `b968826d9c46dd6066d109eabc6255188de91218`; Bonsai MLX `019934f87a61a654e3960ea22f53688e0d2c49ba`; 63-group sample JSON SHA256 `40260dae9955f43b1602698c231e6195bf63ad00714632f6093b593f88e13e80`. Only small byte ranges were downloaded, not full external weights.
- Prism demo/PDF snapshot `9ef32054fe44797376792c869891163083d64bd0`; Prism llama.cpp `adfffbe41b2cabcd51fff326ab045662265062bb`. Publication statements, artifact observations, mathematical deductions and our CPU diagnostics are kept separate throughout.
- Raw geometry, H512 probe and range-sample manifests are preserved under ignored `results/prism-research-20260925/`, indexed by `artifact-manifest.json`. They contain no final-prompt evaluation.
- Seven working analyses and review exchanges staged under `/tmp/binary-eagle-prism-reports-20260925/`; decision-bearing conclusions and reproducible diagnostics are preserved here.

### Bounded CPU regression smoke

A matched six-request regression used Apple M3 Max **CPU only**, the existing
`92bc706` server binary, the same FP16 target, historical `prose-01` and
`code-01`, 32 output tokens, greedy decoding, D=5, confidence floor zero,
no prompt-cache reuse and one server at a time. Both target and draft were
explicitly placed on CPU; this is not the newer SM75 suite build.

| Draft | Prose accepted/proposed | Code accepted/proposed |
| --- | ---: | ---: |
| FP16 EAGLE | 13 / 81 | 21 / 44 |
| Q4_0 EAGLE | 12 / 82 | 21 / 44 |
| Pure Q1_0 EAGLE | 0 / 140 | 3 / 125 |

Every variant emitted the same 32 raw IDs as the FP16 EAGLE anchor for each prompt. This CPU diagnostic did not include target-only. The
Q1 artifact therefore loads and executes, while its proposals remain weak on
these two examples. This is evidence against presenting naive group-128 as
an already successful recovery, not a held-out conclusion or a Turing speed
result. Q1 also changes the activation arithmetic, so it is not the isolated
A16 scale experiment described above. All owned server processes were stopped
and their ports closed. Raw manifests, commands, model/library hashes, logs
and IDs are under `results/prism-research-20260925/cpu-smoke/`; summary SHA256
`2c71bb6f37009846dd968e68c0db17df80e66ff623de0f14e9f48a392e9a7f6a`.

### Final bounded coverage discriminator

The last four CPU requests completed the body/head 2×2 with exactly the same
source, target, server, prompts and settings. Tensor types and payloads were
audited before requests; no further model smoke is planned.

| Body / output head | Prose accepted/proposed | Code accepted/proposed |
| --- | ---: | ---: |
| F16 / F16 | 13 / 81 | 21 / 44 |
| Q1 / F16 | 3 / 125 | 2 / 130 |
| F16 / Q1 | 11 / 91 | 19 / 50 |
| Q1 / Q1 | 0 / 140 | 3 / 125 |

Replacing the head with the original dense head did not rescue the naive Q1
body on these examples. Head-only Q1 retained much more acceptance. This
prioritizes body representation/interface diagnosis over assuming the output
head alone explains all-group failure. It is an online coverage intervention,
not a same-fixed-state causal decomposition: trajectories can differ, and an
original dense head is not a fitted readout. It does not rule out compensation
by a trained readout or calibrating the body. No Q4-body/Q1-head result is claimed.

The first attempt to request a Q1 head under an overall-F16 conversion was
bypassed by the quantizer; the tensor audit caught it before inference. The
corrected export used explicit overrides under the Q1 base format and passed
payload checks. This is why the actual tensor inventory is part of admission.
Raw factorial artifacts are under `results/prism-research-20260925/cpu-factorial/`;
summary SHA256 `7127193f3779717663909987a77cf57bdccc2747b0e45cfccb40a94fc9b58f70`.
All four new responses matched the anchor's raw emitted IDs and all owned
servers/ports were stopped/closed. These remain CPU diagnostics only.

### Session completion

Completed the requested window at **02:08:04 UTC**, 45 minutes 14 seconds
after the first checkpoint. All seven assignments and their cross-reviews are
complete. Claim/source corrections, mathematical checks and raw CPU table/hash
checks were incorporated. Nine changed files passed relative-link, Python
syntax, whitespace and conflict-marker checks; the reusable Q1 audit also
reproduced zero mismatches. The remaining project GPU diagnostics retain their
existing owner. No training or reserved-final evaluation was performed.
