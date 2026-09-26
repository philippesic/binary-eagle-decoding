# Research cross-reference and revised recommendations

**Reviewed:** 2026-09-25 (America/Los_Angeles), following the [seven local analyses](research-review-2026-09-25.md). Seven Astra-high reviewers searched and opened primary papers, author artifacts and official implementation documentation. This is a literature/source audit, not a reproduction or a new GPU result. Local evidence remains pinned to project `b7ea617` and llama.cpp `34e21b7`; newer upstream sources are explicitly identified below.

**Concurrent project progress:** while this review ran, the suite owner updated
`main` through `57ea68b`, recording W1Ax SM75 operator/replay gates and a running
historical comparison. That checkpoint is preserved in the [active goal](../docs/goals/w1ax-activation-precision-suite.md). This literature review does not independently validate or analyze those new runs; its historical performance arguments remain tied to the stated baseline. Revisit the recommendations when the suite owner seals the comparison.

## Changes to the advice

| Category | Verdict | Revised recommendation |
| --- | --- | --- |
| QAT execution | Keep the bounded pilot; narrow objective claims | Original-draft KL is a valid compression proxy that failed this run. Target-greedy CE is a controlled next hypothesis, not an established optimum. Keep online development selection and numerical/export checks. |
| Non-EAGLE architectures | Change the lead candidate | Favor **DSpark with matched DFlash as control** for the later quality screen. This is a normal-precision quality prior, not a ranking of W1A1 robustness. Start from released weights. |
| Throughput policies | Keep measured policy screening; revise controller assumptions | Choose from a small declared policy set using complete round cost. Context and active batch shape matter; neither more batching nor shorter depth under load is universally better. |
| Pipeline | Keep staged graph pruning; strengthen conditions | Unused-head removal, explicit single-layer K/V-only reconstruction, and acceptance-dependent deferral are separate experiments. Graph dependency reasoning does not certify cache or floating-point parity. |
| Precision | Downgrade a presumed block-scale winner | Block scales and learned thresholds remain capture-led hypotheses. Fine grouping can impair binary kernels; learned scales can help or hurt. One-bit-weight language models do not establish W1A1. |
| Data and vocabulary | Strengthen diagnosis; reject causal overstatement | Separate recurrent-state exposure, learner-token trajectories and cached-data refresh. Measure support and ranking error before changing the vocabulary; larger support can improve quality while losing speed. |
| Evaluation | Keep full-round gates; qualify acceptance formulas | Native target-sample-and-match and probability-ratio/residual verification have different acceptance laws. Exact-sampling proofs do not establish local batched/incremental numerical equivalence. |

No reviewed primary source establishes a W1A1 EAGLE speedup on this project's RTX 2080 Ti workload. None overturns the sealed local negative results. The frozen W1Ax study, one-run QAT budget, target/verifier track and final-prompt reservation remain unchanged.

## 1. QAT: improve the experiment without certifying a loss

DistillSpec finds that data and divergence choices depend on task and decoding. This supports capturing target probabilities as well as labels, while training only the prespecified recipe. It does not show that our teacher choice caused the pilot's failure. [DistillSpec, §5.2](https://arxiv.org/html/2310.08461v2).

EAGLE-3's training-time test feeds predicted draft states back with corresponding attention relationships and removes mandatory target-feature regression. Wider body QAT therefore needs recurrence/mask/position fidelity; an independent cached-head trainer is insufficient. Do not interpret “alignment loss” as a requirement to restore target-hidden-state MSE. [EAGLE-3, §§3.1–3.2](https://arxiv.org/html/2503.01840v3).

Clipped STE is established binary-network practice, not inherently a bug. XNOR-Net also supplies a counterexample to assuming separately learned scales improve accuracy. Keep mask activity, sign changes and scale drift as diagnostics; do not change them simultaneously with supervision. [BNN, §1.3](https://arxiv.org/html/1602.02830v3), [XNOR-Net, §4.3](https://arxiv.org/html/1603.05279v4).

MiniCPM4 is actual four-bit EAGLE-2 QAT precedent, with a different precision contract. QSpec instead uses complementary drafting/verification precision; its EAGLE comparison retains an FP16 drafter. Neither establishes one-bit-activation recovery. [MiniCPM4, §4.1.3](https://arxiv.org/html/2506.07900v2), [QSpec, §4.1](https://aclanthology.org/2025.emnlp-main.240.pdf).

**Local decision:** retain the 500-step/45-minute target-aligned head diagnostic. The suggested CE/regularizer setting is experimental. If a later separately scoped extra arm is warranted, a hard-versus-soft target comparison may be more informative than another old-draft-KL run; use the latter only for causal diagnosis. VSD motivates path-utility diagnostics, but its p/q assumptions and EM/Monte Carlo machinery are not a drop-in objective or a reason to expand this pilot. [VSD v6, §4.4](https://arxiv.org/html/2602.05774v6).

## 2. Architecture: DSpark leads a paired screen

DSpark reports a matched Qwen3-4B advantage over DFlash with scheduling disabled. Its temperature-1 accepted-length measure includes the bonus token; its serial-overhead analysis uses batch 128. These qualify, rather than establish, applicability to our greedy N=1 workload. [DSpark, §4.2/Table 1 and §4.3.2](https://arxiv.org/html/2607.05147v1).

Use paired DeepSpec releases to reduce training-data confounding. Its default large-cache, multi-GPU training pipeline is outside this bounded project; released weights followed by a small quality screen are the feasible starting point. Do not treat a published two-layer ablation as an acquired checkpoint or truncate a five-layer model and claim reproduction. [DeepSpec, pinned README](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/README.md).

DFlash's shared target head and feature-conditioned block prediction remain relevant cost/coverage questions. Keep the shared target unchanged; a separate binary draft head needs explicit plumbing and memory accounting. [DFlash, §§4.1–4.2](https://arxiv.org/html/2602.06036v1).

For the third-ranked family, include prefix-conditioned lightweight heads rather than favoring independence by default. Medusa-1 respects frozen-target training, whereas joint-target variants do not fit that boundary; Hydra provides contrary evidence to the assumption that independent parallel heads always offer the best quality/cost tradeoff. Their relaxed “typical acceptance” results are not exact-verifier anchors. [Medusa, §2](https://arxiv.org/html/2401.10774v3), [Hydra, §§3,5–6](https://arxiv.org/html/2402.05109v2).

**Local decision:** later screen released DSpark/DFlash reference quality with scheduling disabled, then W1A16 and selective W1A1. Compare actual proposed counts, prefix survival and full round time, including Markov/head work. The architecture recommendation does not open a new goal.

## 3. Scheduling: optimize measured complete policies

Sequoia supports hardware-aware selection using measured verification and draft costs. The local marginal ratio inequality remains correct for a specified pair of policies, but it is not proof that greedy addition/removal of one node finds the global best policy. Dispatch thresholds and graph reuse can make costs nonconvex. [Sequoia, §4.1 and Appendix G.5](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea1f5f0878d43ff4fb8bf64ef4a2326c-Paper-Conference.pdf).

SpecInfer and MagicDec examine different operating regimes: concurrency can consume spare compute, while long-context KV traffic can preserve speculative benefits at larger batches. Their devices, contexts and serving regimes do not predict our 2048-context SM75 result. Retain context as an independent policy dimension. [SpecInfer, §6.2](https://arxiv.org/html/2305.09781v4), [MagicDec v4, §3](https://arxiv.org/html/2408.11049v4).

External dynamic-lookahead work motivates confidence stopping, not importing its threshold into a top-10-normalized sampler. Our late per-sequence cap and possible redundant 0.1 floor remain pinned-source findings requiring execution counters and backend checks. [Official dynamic-lookahead article](https://huggingface.co/blog/dynamic_speculation_lookahead), [Transformers v4.45.0 confidence criterion](https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/generation/stopping_criteria.py).

**Local decision:** keep fixed-policy and tuned-policy results separate. Full-round attribution determines whether early caps, graph pruning or deferred reconciliation deserves the first isolated implementation; none is already a measured winner.

## 4. Pipeline: stronger source support, unchanged performance uncertainty

The inspected upstream graph still builds the head/scatter, and orchestration still documents rejected-row processing. Updating to that revision is not an identified fix. K/V-only reconstruction remains our single-layer dependency inference, restricted to calls with no consumed decoder hidden output or logits. Preserve explicit cache-write roots, boundary state and sampler plumbing. [Upstream EAGLE graph](https://github.com/ggml-org/llama.cpp/blob/171e8846b4af9766c354064cb776cb34a50f053f/src/models/eagle3.cpp), [upstream orchestration](https://github.com/ggml-org/llama.cpp/blob/171e8846b4af9766c354064cb776cb34a50f053f/common/speculative.cpp).

CUDA capture restrictions rule out simply wrapping the existing synchronizing host-driven loop in a capture. Existing per-decode graph reuse and a device-resident recurrent-state redesign are separate projects. [CUDA 12.8 graph documentation](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#cuda-graphs).

Microsoft's implementation offers concrete shared-input quantization prior art through joint QKV/FFN projections, but uses a different packed-weight/INT8-activation contract. Reuse the design principle, not its numeric rules. [Pinned BitNet GPU implementation](https://github.com/microsoft/BitNet/blob/0b341e582afbf9e1011f24744b554c96a3477eb5/gpu/model.py).

The official FlashAttention documentation now points to a separate Turing implementation; that implementation lacks KV-cache support in the inspected revision. It is a design reference, not a validated llama.cpp decode replacement. Profile current attention before raising its priority. [Official support documentation](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/README.md), [Turing implementation](https://github.com/ssiu/flash-attention-turing/blob/9ef98fcb506bb1e2fe3cece50935e2935bf6b124/README.md).

## 5. Precision: preserve operand distinctions and kernel feasibility

| Primary result | Actual relevant contract | Implication here |
| --- | --- | --- |
| [BitNet b1](https://arxiv.org/html/2310.11453v1) | W1A8, pretrained; centered weights and normalization | Supports a distinct W1A8 hypothesis, not W1A1 transfer. |
| [BitNet b1.58](https://arxiv.org/html/2402.17764v1) | Ternary weights, A8; packed execution differs from one binary dot | Entropy/weight labels cannot substitute for executed operand widths. |
| [OneBit](https://arxiv.org/html/2402.11295v3) | W1A16 with separable input/output scales and substantial QAT | Add a conditional W1Ax scaling diagnostic; it is not a cheap proven W1A1 remedy. |
| [BiLLM](https://arxiv.org/html/2402.04291v2) | Weight-only grouping and extra binary residuals; about 1.1 weight bits | Its stated binary-GEMM difficulty is counterevidence to assuming fine grouping is free. |
| [ReActNet](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590137.pdf) | W1A1 vision layers with learned channel thresholds and precision exceptions | Gives threshold experiments cross-domain precedent, not EAGLE acceptance evidence. |

**Local decision:** choose one representation follow-up from W1Ax captures. Positive channel scaling immediately followed by sign leaves A1 bits unchanged; retaining channel magnitudes inside a dot needs additional structure/computation. Symmetric clipping also preserves nonzero A1 signs. Keep structured scales and thresholds as alternatives rather than declaring a winner. Account for all codes, scales, corrections, packing and launches; do not import CPU convolution or large-batch accelerator speed factors.

## 6. Data: distinguish exposure, support and vocabulary design

DAgger motivates collecting learner-visited states with expert labels, while Online Speculative Decoding offers rejection-location supervision precedent. Neither licenses updating weights on held-out requests or simultaneous training on our constrained experiment GPU. Recurrent-state exposure, token-level on-policy training and a train-only capture refresh are different interventions. [DAgger, Algorithm 3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf), [Online Speculative Decoding, §4](https://arxiv.org/html/2310.07177v2).

SpecVocab v2 directly examines Qwen3 and distinguishes training with reduced support from pruning afterward. Its full-vocabulary comparison also changes head sharing, and greater acceptance can coexist with lower throughput. It motivates measuring missing support and ranking error separately; it does not prove that this checkpoint's 32k support is its bottleneck. [SpecVocab v2, §§4–6 and Appendix B.1](https://arxiv.org/html/2602.13836v2).

**Local decision:** retain target probability mass, mapped argmax, ancestry and valid/unverified status in capture. For unmapped hard labels, use an explicit mask and retain denominators; restricted soft targets may still carry information. Full-vocabulary forward KL is infinite when the draft assigns zero probability to target-supported tokens; a restricted/renormalized KL is a different objective. Vocabulary expansion or full-support training followed by pruning is a separate decision after the audit.

## 7. Evaluation: name the actual verifier and service metric

Classical corrected speculative sampling uses proposal-ratio acceptance and residual resampling. Its overlap formula and simplified iid/constant-cost speed model have explicit assumptions. Numerical target equivalence remains a separate implementation premise. [Leviathan et al., §§2–3](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf), [Chen et al., §4.2](https://arxiv.org/html/2302.01318v1).

**Our derivation from the pinned native code:** at a reached prefix, independently sampling a target token and comparing with a deterministic candidate d matches with probability p(d). With an independently sampled draft q, it is sum p*q. Ratio/residual verification instead gives sum min(p,q). Each requires correct conditionals/state; these are not interchangeable acceptance metrics. If candidate d lies in S, or q is supported on S, target mass p(S) bounds the corresponding acceptance probability, but may be loose for sample-and-match. Matching target draws can preserve the target distribution without computing p/q; that reasoning does not certify cache, masking or floating-point premises.

DSpark's token-dependent admission warning concerns its p/q correction law. It does not by itself invalidate arbitrary deterministic candidate truncation under independent target matching. Check the actual sampler before importing a paper's correctness condition or training theorem.

SmartSpec's useful-token rate and DistServe's latency-constrained request capacity are different notions of goodput. Our current rates are concurrency-one measurements, not maximum online arrival capacity. Future serving claims need arrivals, latency tails and resource accounting; this review does not expand the current study into a serving system. [SmartSpec, §§3–4](https://arxiv.org/html/2406.14066v1), [DistServe, §2](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf).

## Source versions and transfer boundaries

The links above identify inspected versions, not promises about later upstream state. The source register below records publication/artifact status and the most material transfer limit; it is not a list of reproduced results.

| Source/version | Status/date | Setting or limitation retained |
| --- | --- | --- |
| EAGLE-3 v3 | arXiv, 2025-04-23 | Recurrent training; paper's H100 serving experiment is not binary SM75. |
| DistillSpec v2 | ICLR 2024; revision 2024-03-31 | Smaller dense drafters and decoding-dependent objectives; no binary QAT. |
| MiniCPM4 v2 and author QAT card | technical report, 2025-09-04; card release 2025-06-06 | Four-bit draft QAT; inspected paragraph does not specify draft activation width. Orin/4090 system results. |
| QSpec | EMNLP 2025 proceedings | W4A4/W4A16 self-speculation; L20 and A100 experiments. |
| BNN v3 / XNOR-Net v4 | 2016-03-17 / 2016-08-02; vision work | Binary training mechanisms; no language-model throughput transfer. |
| VSD v6 | arXiv, 2026-08-12 | Probability-ratio assumptions; lower-bound results and different accelerators. |
| DFlash v1 / DSpark v1 | arXiv, 2026-02-05 / 2026-07-06 | Non-binary block-drafter evidence; paper metrics/settings need conversion. |
| DeepSpec `005e03b` | author artifact, 2026-07-09 | Paired releases; BF16 training config, not an audit of all serving arithmetic. |
| Medusa v3 / Hydra v2 | arXiv, 2024-06-14 / 2024-10-07 | Target-training and verifier variants differ; no compatible binary result. |
| SpecInfer v4 / Sequoia proceedings | ASPLOS / NeurIPS 2024 | A10, A100/L40 and offloading regimes differ from this resident model. |
| MagicDec v4 | arXiv, 2025-03-26 | Multi-GPU, long-context experiments; no monotonic batch rule. |
| HF dynamic speculation / Transformers v4.45.0 | official implementation, 2024 | Different confidence-score construction and model pairs. |
| llama.cpp `171e8846` | upstream source, 2026-09-25 | A source cross-check only; project gitlink was not changed. |
| CUDA 12.8 guide / [Turing tuning guide](https://docs.nvidia.com/cuda/archive/12.8.0/turing-tuning-guide/index.html) | NVIDIA, 2025 release | Runtime/hardware constraints, not measured EAGLE performance. |
| BitNet GPU `0b341e5` | author code, 2026-07-27 | Packed two-bit weights/INT8 activations/BF16 output differ from our operator. |
| FlashAttention `e9cf2c1` / Turing `9ef98fc` | implementation snapshots, 2026-09-25 / 2026-09-14 | Mainline architecture support and separate no-KV-cache Turing path. |
| BitNet v1 / b1.58 v1 | arXiv, 2023 / 2024 | W1A8 versus ternary-A8; pretraining differs from small-drafter QAT. |
| OneBit v3 / BiLLM v2 / ReActNet | 2024-05-22 / 2024-05-15 / ECCV 2020 | Different operand widths, training costs and kernel feasibility. |
| SpecVocab v2 | Findings ACL 2026; revision 2026-07-17 | Qwen3 support/head tradeoff; no one-bit experiment. |
| DAgger / Online Speculative Decoding v2 | AISTATS 2011 / arXiv 2023-10-17 | Data-collection principles, not QAT convergence guarantees. |
| Leviathan / Chen v1 | ICML 2023 / arXiv 2023 | Sampling theorems require their algorithmic/numerical premises. |
| SmartSpec v1 / DistServe | arXiv / OSDI 2024 | Useful-token rate versus SLO-constrained request capacity. |

## Revised execution order and unresolved decisions

1. Finish the frozen common-weight W1Ax study with complete-round/count instrumentation and same-build FP16/Q4_0 anchors.
2. Audit state-label alignment, reachable-prefix support, native arithmetic and confidence semantics. Use attribution to choose one isolated runtime cleanup.
3. Run the bounded target-aligned head pilot, treating objective settings as hypotheses. Select online on development; keep the final set sealed until model and policy are frozen.
4. Let measured activation/weight damage choose one representation follow-up. Do not automatically add scales, thresholds, residuals and a new loss together.
5. Only after the quality gate or a user direction change, admit paired released DSpark/DFlash models. Screen reference and simulated-binary quality before packed-kernel work.

The literature changes confidence and ranking, not experiment authorization. Open choices remain unsupported-label treatment, any extra QAT arm, a new quantizer contract, policy-grid amendment and the later architecture pivot. The zero-removable-work ceiling still needs missing local timings; no paper fills those measurements in.
