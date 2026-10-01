# Extreme low-bit formats: quality before a new kernel

Audit date: 2026-10-01. Project source `f0bb92dda687d908b3cffe084510b3be757df54f`; llama.cpp `b4e366d4f0a30cac07f14d51c54c5b1329b3f485`. This is a read-only source/literature assessment and proposed work, not a measured prototype. No models, accelerators, remote hosts, full weights, large captures or sealed final prompts were used. The active strict W1A1 goal and unchanged target/verifier remain in force.

## Verdict and local evidence

**The best practical intermediate is an existing block-scaled approximately two-bit weight format with its native Q8_1 activation path, screened against Q4_0.** Prefer IQ2_XS, with Q2_K as a simpler control, before engineering ternary or sub-bit inference. This is a diagnostic compression branch, not W1A1 success. A balanced four-level CPU oracle is the most informative new representation experiment; two binary residual planes provide a useful matched-capacity comparator.

The [W1Ax suite](../w1ax-activation-precision-results.md) already establishes the problem: on 24 development prompts, Q4_0 accepted 1.036 drafts/round at 83.93 decode tok/s; all-layer W1A16 accepted 0.117 at 29.86 tok/s and W1A1 0.055 at 45.36 tok/s. Restoring activation precision does not repair the existing row-scaled signs. At head N=1, packing-inclusive W1A1 took 70.8 µs versus Q4_0's 140.7 µs, but this component saving did not compensate for extra verification rounds. Larger N=37 reverses that comparison: 966.4 versus 225.1 µs. Stronger quantization currently hurts more than it saves; the relevant uncertainty is whether a better representation recovers acceptance.

The [2080 Ti suite](../rtx2080ti-quantization-suite.md) does not condemn every four-bit design: its research W4A4 uses different scaling and activations from successful block-scaled Q4_0. Likewise, actual binary MMA tied portable W1A1 at 1.002× decode rate. Reusing that instruction with more planes is not a demonstrated escape. The [Prism audit](../prism-quantization-research-2026-09-25.md) already found group-128 signs and a working Q1_0 export, but grouping reduced head/FFN weight residual SSE by only about 0.74–0.77%. Do not duplicate that control or infer optimized signs from a format label.

## Evidence and mechanism table

Bits below describe matrix payload, including stated scales, before file headers, alignment, floating exceptions and runtime state. Whole-draft precision must be parameter-weighted across all nine linears; the head alone contains 37.5% of their 218.23M parameters ([coverage](../drafter-static-coverage.md)).

| Family | Mechanism / honest storage | Counterargument and verdict |
| --- | --- | --- |
| W1A2 / W1A4 | Same signs; two/four activation planes. Row F32 scale adds `32/K` weight bits; activation scale and padded tails also count. | A4 is already implemented and measured; A2 is absent. Two planes save work but lose resolution and require a declared signed grid. Defer A2 until A4 clipping/quality improves. |
| Ternary, W2A8 | Zero suppresses weak weights. Entropy `log2(3)` is 1.585, but fixed two-bit codes are two bits. Native TQ1_0/TQ2_0 are 1.6875/2.0625 bpw. | BitNet's quality is trained-from-scratch WternaryA8, not binary activations or EAGLE PTQ. CPU oracle first; no immediate kernel transplant. |
| Balanced W2A4/A8 | Four symmetric levels preserve magnitude discrimination. Two bits plus F16 group-128 scales gives 2.125 bpw. | W2A4 adds activation damage; W2A8 is the cleaner weight-capacity diagnosis. Existing IQ2_XS/Q2_K storage is 2.3125/2.625 bpw with native CUDA paths. Highest practical priority. |
| Two residual sign planes | `W ≈ αB₁+βB₂`; two row planes cost `2+64/K` bpw with F32 scales. A partial residual fraction `f` costs at least `1+f`, plus selectors/scales. | Two reductions and scale accumulation; arbitrary corrections can dominate metadata. A quality oracle, not strict W1 or a free second bit. |
| Structured sparse binary | Prune signs and store positions. Arbitrary 2:4 signs need two sign bits plus at least `log2(6)` selector bits per four weights: over 1.146 bpw before scales; fixed three-bit selectors give 1.25. | Nominal nonzero-bit counts are not packed payload. SM75 has no later-generation structured sparse Tensor Core route. Low priority. |
| NanoQuant low-rank binary | Two factors plus channel scales: F16-scale bpw `(r+16)(M+K)/(MK)`; rounding/alignment increase it. | Two sequential products and a wider intermediate; unpacked floating arithmetic, not XOR/POPCOUNT W1A1. Geometry feasibility only. |

## What the primary research actually supports

[BitNet b1.58 v1](https://arxiv.org/html/2402.17764v1) explicitly trains ternary weights with eight-bit activations and measures latency using a two-bit kernel. The [official GPU implementation](https://github.com/microsoft/BitNet/blob/0b341e582afbf9e1011f24744b554c96a3477eb5/gpu/README.md) names W2A8 GEMV. This is useful packed decode precedent, but importing its normalization, quantizer and trained model would change several mechanisms simultaneously.

[ParetoQ v2](https://arxiv.org/html/2502.02631v2) favors balanced, learned grids for ternary/two-bit weights and finds binary less attractive on its accuracy–size frontier. Its lower-bit models require substantial representation reconstruction and training. Its scaling comparison can vary model size, whereas this project fixes a small EAGLE component. Therefore it supports a symmetric-grid comparator and scale optimization, not a promise that two-bit RTN will preserve speculative prefixes within our budget.

[BiLLM v2](https://arxiv.org/html/2402.04291v2) applies residual binarization selectively and distribution splitting. [STBLLM's ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/ff997469ac66cf893c4183efeb22212a-Paper-Conference.pdf) combines N:M sparsity, importance allocation and multiple quantization regions. Their nominal near/sub-one-bit labels need independent packed-byte accounting. [NanoQuant v1, Appendix F](https://arxiv.org/html/2602.06694v1) disputes their effective storage after masks and scales; those are competing author calculations, not a local reproduction. Require an executable codec and byte inventory before claiming compression. For a masked binary dot, both mismatch popcount and surviving-count correction are needed; dense sign XOR alone is wrong.

NanoQuant instead factorizes a weight matrix into two packed binary matrices with channel scales. Its GEMV unpacks signs into FP16/BF16 arithmetic; GEMM dequantizes into floating Tensor Core operations. A scaled wide intermediate follows the first product, so binarizing it creates a different approximation. On the 32000×2560 head, its ideal F16-scale payload falls below one bpw only for rank below roughly 2354, excluding padding and exceptions. That arithmetic possibility says nothing about retained rank utility or prefix acceptance.

## Native implementation and hardware costs

The pinned `ggml-common.h`, `ggml-quants.c`, CUDA `mmvq.cu`, `mmq.cu` and `vecdotq.cuh` already contain Q1_0, Q2_0, Q2_K and IQ2 formats with byte expansion and Q8_1 dot paths. TQ types exist in the common/CPU codec, but inspected CUDA MMVQ/MMQ dispatch does not include them; codec availability is not fast-GPU support.

**Q2_0 is a misleading balanced-W2 control.** Its reference converter sets `d=amax` and rounds `w/d` into codes for `{-1,0,1,2}`. Ordinary finite inputs never reach +2, so the fourth level is unused and it behaves as coarse ternary. Changing its export alone to symmetric levels would violate existing decode semantics. A new symmetric oracle must explicitly encode its level mapping.

W1A4 currently performs four weighted plane reductions; W1A2 would perform two. A straightforward W2A4 decomposition needs up to eight plane products, W2A8 sixteen, versus one for W1A1. Shared activation-only reductions can reduce constants, but launches, packing, metadata loads and rescaling remain. Ternary requires masks or two indicators. Group scales interrupt one long reduction with partial sums; mixed row/group layouts need an explicit scale-owner contract.

On SM75, [NVIDIA documents](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html) integer Tensor Core widths of 1/4/8, not native INT2. Expansion to bytes plus DP4A is a credible decode path because the project already exercises it. SM120 has different occupancy/memory resources ([Blackwell guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)); neither Turing binary performance nor datacenter Blackwell claims transfer automatically. Use architecture-specific dispatch/instruction inspection and small real shapes. Memory savings against Q4_0's 4.5 bpw are upper bounds on traffic savings, not end-to-end speedups.

## Ranked two-day work and later gates

1. **Native format/export audit:** on synthetic tensors and small predeclared weight slices, inventory Q1_0, IQ2_XS, Q2_K and Q4_0 payloads, grids, scale rounding and dispatch. Produce a codec oracle and source-pinned eligibility table. Later, admit one two-bit control only after native parity and fixed-policy development acceptance; compare complete rounds against same-device Q4_0 with the target untouched.
2. **Balanced W2 versus ternary/residual oracle:** implement explicit symmetric four-level, zero-containing ternary and two-plane row/group reconstructions with matched byte budgets, tails and dense-versus-packed scalar dots. Compare slice geometry, not model quality. Later obtain designated training calibration inputs and assess output error/ranking before one bounded GPU acceptance test. Learned scales require exporter/gradient ownership; this audit does not modify the live trainer.
3. **Sub-bit geometry screen:** derive rank/storage breakpoints for actual shapes and test bounded sampled blocks, including factorization versus sparse-mask overhead. Stop if rank or metadata erases savings. Later GPU work is conditional on a quality case and full intermediate/launch costs; no full NanoQuant/STBLLM reproduction is justified now.

Q4_0 remains the model to beat. Two-bit weights with wider activations are the best practical intermediate. Better grids may preserve information that the current signs discard. Extra planes and metadata must earn their cost through acceptance. Keep W1A1 as the endpoint and keep the target unchanged.
