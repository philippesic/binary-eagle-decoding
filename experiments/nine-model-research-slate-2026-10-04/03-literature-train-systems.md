# Practical QAT execution and memory refinements

October 4, 2026. Agent 3 literature and training-systems report for the nine-model research slate.

The research phase was read-only: no edits, commits, tests, model loads, accelerator activity, or remote work. This subsequent uniquely assigned report write is documentation only. All tests below are proposals, not executed results. Root owns integration and commit.

Four bounded refinements are worth considering. None requires changing the objective, quantizer normalization, optimizer batch, or frozen target.

Existing evidence already admits full-shape EAGLE forward/backward and 90 B1/B2/B4 graph measurements on RTX5080/SM120; it does **not** establish larger-batch learning equivalence. See [actual-model record](../qat-optimization-readiness/first-1000-paired-steps-2026-10-03.md). Verified local hook: `experiments/qat-optimization-readiness/first-1000-paired-steps-2026-10-03.md:55`.

## 1. Remove unnecessary diagnostic copies, then consolidate telemetry transfers

**Verdict: incorporate clone removal after a small equivalence check; probe telemetry consolidation.**

Current sign snapshot at `src/w1a1_eagle/recurrent_qat.py:760` evaluates `latent_sign.detach().clone() < 0`. The comparison already creates an independent Boolean tensor, so `latent_sign.detach() < 0` preserves the snapshot without the intermediate FP32 clone. Retain the scale snapshots and all finite-gradient/parameter gates.

**Projection:** the existing EAGLE scope has 218,234,880 latent signs. This removes 872,939,520 bytes—832.5 MiB—of cumulative FP32 clone allocations per lane/update, not 832.5 MiB of resident peak memory. The largest head clone alone is 312.5 MiB; allocator overlap determines the actual peak benefit. The existing Boolean snapshots remain.

Next, assemble diagnostic scalars on-device and transfer them together after the mandatory gates. Current metrics at `src/w1a1_eagle/recurrent_qat.py:842` and return conversions make several separate `int`/`float` reads. PyTorch explicitly identifies scalar reads and CPU copies as synchronization points. Do not remove the pre-update finite checks or change their failure ordering. Primary sources: [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization), [clone semantics](https://docs.pytorch.org/docs/2.14/generated/torch.clone.html).

**Ease/risk:** low. No arithmetic, mask, LSQ, state ownership, or checkpoint change is necessary for clone removal. Consolidated telemetry needs schema-compatible results and must retain integer counters without converting large counts through FP32.

**Proposed test, not run:** compare diagnostic values and unchanged gradients/updates on the same disposable actual-model rounds, then measure diagnostic-stage and whole-step time plus allocated/reserved peaks. Applicable to all six runs wherever comparable diagnostics exist; no direct SM75 inference speedup follows.

## 2. Transfer immutable captured features once; prefetch only if stalls remain

**Verdict: incorporate reuse after equivalence admission; probe one-round pinned prefetch.**

The paired loop at `src/w1a1_eagle/continuous_qat.py:1142` presently transfers `raw_target_features` separately for each lane. Move the transfer outside the lane loop and share that immutable input tensor. Keep each lane's FC output, quantization, hard signs, student state, K/V and graph independent.

**Projection:** at the existing maximum EAGLE prefix, approximately 60 MiB of F32 raw features are uploaded per lane. One shared transfer removes approximately 60 MiB of host-to-device traffic per paired round. This is a byte saving, not a latency estimate. Retain the shared tensor through both lanes and release it afterward.

If measured transfer or preparation stalls remain, use a bounded background producer with one next-round pinned buffer and an explicit copy stream/event. PyTorch documents that compute overlap requires pinned source memory, a separate stream and available DMA capacity; pinning in the main training thread can erase the benefit. Primary source: [transfer tutorial](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).

**Ease/risk:** reuse is low effort; prefetch is moderate. Preserve audited ordering, exact cursor/resume behavior, STOP handling and host-memory floors. Do not retain another entire capture shard: the existing streaming policy at `scripts/w1ax_multishard_provider.py:352` intentionally bounds active shards. Never overwrite/free a pinned buffer until its copy completes. No full target residency is required.

**Proposed test, not run:** compare byte-identical device inputs, labels and paired results; profile H2D traffic and compute overlap across short/long audited rounds, including shard transitions and interrupted/resumed iteration. On RTX5080, verify actual overlap rather than assuming it. For DSpark/DFlash, share only genuinely identical architecture-specific captures. This is training infrastructure; nine final native tests retain their existing data/runtime contracts.

## 3. Use grouped F32 attention contractions without physically repeating K/V

**Verdict: probe.**

The current attention branch at `src/w1a1_eagle/native_step.py:642` repeats each of eight K/V heads four times before QK and AV. Reshape Q into `[8,4,128]`, retain K/V as `[8,L,128]`, and express:

- scores: `einsum("grd,gtd->grt", q_grouped, key)`
- output: `einsum("grt,gtd->grd", probabilities, value)`

This implements the existing query-to-K/V grouping; it does not change the architecture. Keep the original F32 scale/softmax and F16-write/F32-read K/V boundaries. Primary sources: [original GQA paper](https://arxiv.org/abs/2305.13245), [PyTorch contraction documentation](https://docs.pytorch.org/docs/2.14/generated/torch.einsum.html).

**Projection:** at `L=2048`, the two repeated F32 tensors occupy `2×32×2048×128×4 = 64 MiB` per proposal depth. Avoiding these tensors removes that explicit materialization; five depths represent approximately 320 MiB of repeated K/V tensor storage if backward retains every pair. These are tensor-size counts, not a guaranteed allocator-peak reduction. Original caches and their attached concatenations remain; dot-product counts are unchanged.

**Ease/risk:** low/moderate for this F32 branch. Preserve contiguous groups matching `repeat_interleave`; gradients must sum the four query-head contributions into each K/V head. Different contractions can change reduction ordering or introduce backend layout copies. Keep diagnostic/native attention branches untouched. This also avoids accidentally selecting a fused SDPA backend with different precision or short-query mask semantics.

**Proposed test, not run:** actual A8/A1 rounds at representative prefix lengths, comparing state/logits, all parameter VJPs and later-only gradients through first K/V. Measure saved tensors, backward peaks and whole-step timing. Standard F32 contractions are compatible in principle with SM120, but require the local runtime check. Block-model portability needs its own bidirectional noise/target-injection masks. This trainer change supplies no SM75 runtime acceleration claim.

## 4. Compare an explicit fused FP32 AdamW backend

**Verdict: probe; retain the current backend until admitted.**

Current optimizer construction at `src/w1a1_eagle/recurrent_qat.py:673` forces the single-tensor backend with `foreach=False`; the optional recipe constructor at `src/w1a1_eagle/qat_optimization.py:196` does likewise. Test `fused=True` while retaining FP32 parameters/moments, betas, epsilon, zero weight decay, clipping, learning-rate schedule and post-update projection.

PyTorch supports fused FP32 AdamW and documents vertical/horizontal fusion, but that is not a local speed guarantee. Its foreach alternative can add approximately one parameter set of peak intermediates; therefore do not silently enable default foreach. Primary source: [AdamW documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).

**Projection:** persistent masters and two moments remain `12P` bytes per model; fused AdamW does not reduce optimizer storage. For the block models' currently proposed 373,555,200 FFN weights, two FP32 moments alone are approximately 2.99 GB per model, before scales or other trainables.

**Integration risk:** this is not merely a flag change. Smoke initialization at `src/w1a1_eagle/continuous_qat.py:623` manually puts Adam step scalars on CPU, whereas upstream fused initialization places them on the parameter device. Admit fresh initialization and checkpoint restoration explicitly. The recipe audit at `src/w1a1_eagle/qat_recipe_audit.py:297` also needs an explicit backend identity. Primary source: [pinned upstream implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/torch/optim/adam.py).

**Proposed test, not run:** isolated matched FP32 optimizer steps followed by a bounded disposable actual-model continuation; compare moments, scale updates, latent signs, clipping and native proposal choices. Measure optimizer-stage and whole-step improvements on RTX5080. Floating reassociation near a sign boundary can matter; do not treat this as exact resume of the historical backend. SM75 still needs native quality/export validation, not optimizer benchmarking.

## Defer tempting generic optimizations

Defer blanket activation checkpointing, whole-loop `torch.compile`, and low-precision moments. Current callbacks/cache lifetimes make replay ownership nontrivial, and recomputation must preserve invocation-local LSQ normalization and attached recurrence. Primary source: [PyTorch checkpoint warnings](https://docs.pytorch.org/docs/2.14/checkpoint.html).

Head/loss chunking is also low priority: one F32 EAGLE logits tensor at five rows ×32k is only 0.61 MiB; seven rows ×151,936 is 4.06 MiB for a block model. These are buffer arithmetic projections, not allocator measurements. Revisit if admitted batching or full-distribution objectives make these buffers material. Published hardware speedups cannot establish savings here.

The current masks select supported valid direct labels while preserving unsupported valid states for later recurrence: `src/w1a1_eagle/recurrent_loss.py:17`. Learned quantizer normalization counts valid feature elements of each invocation: `src/w1a1_eagle/learned_activation.py:161`. The learned-head serial safeguard is explicit at `src/w1a1_eagle/recurrent_provider.py:258`; do not remove it to make a systems optimization appear faster. Shared hard signs belong to one forward/backward round: `src/w1a1_eagle/recurrent_qat.py:368`.

Promote each singleton on measured supervised tokens/s and memory without changing supported-row means, chronology, invocation-local normalization or update cadence. Faster training alone does not beat Q4_0; the final nine-model native acceptance and complete-request throughput comparisons remain decisive. No new SM120 result, SM75 performance claim, quality winner, implementation, or integration commit is supplied by this report.
