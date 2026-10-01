# Audit: custom QAT operators and training fusion

Date: 2026-10-01. Source inspected: `d725fc88d52fc49841c2b20128d176c1e7f486d8` in the original checkout. This is a source-and-primary-literature audit; no tests, model workloads, installations, GPU/Metal activity, remote actions, or live-pipeline changes were performed. Q4_0 EAGLE remains the acceptance, latency and total-throughput baseline. Training acceleration cannot establish any of those inference improvements.

## Verdict and ranked suboptions

**Pursue a bounded measurement and A1 no-gradient fastpath; conditionally pursue a Python custom-autograd projection; defer a complete custom GEMM/backward stack.** The parent hypothesis is correct mathematically, with important zero-scale, floating-point and memory qualifications. There is no measured current A1 training bottleneck or demonstrated gain.

| Rank | Suboption | Verdict | Implementation complexity / reason |
| --- | --- | --- | --- |
| 1 | Skip the unused A1 surrogate GEMM when gradients are disabled | Pursue first | Low. Prefix reconstruction already declares a truncated-gradient boundary; preserve the existing native-order forward exactly. |
| 2 | Regional pointwise/reduction fusion, including shared activation quantization for Q/K/V and gate/up | Conditional on launch/quantization profile | Low–medium. Benefits both lanes without changing the optimizer or STE. Avoid compiling the data/audit/checkpoint control plane. |
| 3 | Python custom autograd retaining native A1 forward, using two ordinary F32 backward products | Conditional | Medium. One fewer trainable forward GEMM, but scale-gradient reduction and dense temporaries can erase the saving. |
| 4 | Built-in fused AdamW, then compiled update/projection if necessary | Conditional; coordinate with optimizer audit | Low initially. Current explicit single-tensor optimizer is a concrete fusion candidate; preserve full-gradient clipping and parameter-update order. |
| 5 | Triton M=1 projection/backward epilogues, fused STE masks and scale reduction | Conditional after ranks 1–3 | Medium–high. Actual vector/outer-product shapes matter more than large-GEMM tutorials. |
| 6 | Packed binary forward plus custom backward; complete replacement of GEMMs | Defer | High. F32 gradients and Adam states remain dense; packing, custom operator registration and validation add substantial work. |

Quality benefit for ranks 1–5 is **more valid training progress per wall-clock budget**, not a supported change in optimum acceptance. At equal token/update budgets, the intended quality is unchanged apart from explicitly gated numerical reassociation.

This ranking is within the fusion audit. The independent batching/cache audit identifies a stronger structural candidate: prefix context only needs bulk feature fusion, norms, K/V, K-RoPE and stored F16 K/V; its source arithmetic estimate removes approximately 77.9% of linear MACs per context row before fusion. That is a separate worker's finding, not independently measured here. **Rank full custom autograd behind that structural removal**, then re-profile; less retained prefix arithmetic means a smaller absolute benefit from the A1-only shortcut. The native-only shortcut remains compatible with the surviving A1 prefix projections.

## What the project source establishes

`src/w1a1_eagle/recurrent_qat.py:156–184` first forms dequantized activations and computes `F.linear(quantized, signs)` for the differentiable surrogate. For A1 it then computes `F.linear(activation_signs, signs.detach())` under `no_grad`, applies weight scale, activation scale, then bias, and substitutes that result into the forward while keeping surrogate derivatives. **Two forward products execute even if the caller is already under `torch.no_grad()`.** This is source evidence, not a latency measurement.

`recurrent_rollout.py:71–85` reconstructs every accepted context position under `no_grad`; `native_step.py:483–487` skips the vocabulary head in that context path. The valid proposal unroll retains pre-norm state and F16-written K/V links. `continuous_qat.py:817–831` trains lanes serially on the same audited round, with attached hard signs shared once per round. Its `step_seconds` includes transfer, prefix rebuilding, forward, diagnostics, backward, optimizer and synchronization; it is not a phase profile. The optimizer explicitly sets `foreach=False`, with no fused flag (`continuous_qat.py:276–285`). `joint_train_step` scans/clones large parameters for finite checks, sign-flip diagnostics and projection, in addition to ordinary training arithmetic.

The preparation report counts **218,234,880 latent weights and 65,280 scales per lane**, approximately **5.61 GiB** dual persistent state, **0.813 GiB** active gradients, and an assumed—not measured—3 GiB graph budget. Signs are already shared per unroll, so a second sign-cache optimization is not available. The historical row-A16 calibration measured median **0.737187 s/step**, peak **8.494 GiB allocated / 10.049 GiB reserved** on RTX 5080 SM120. Its 100 steps included **9,393 prefix joins and 495 supported labels**. These counts suggest prefix work deserves profiling; they do not establish the present A1/A8 timing distribution. The latest durable status reports zero paired optimizer steps. See `experiments/continuous-w1ax-preparation.md`, `experiments/w1ax-row-a16-calibration-5080.md`, and `docs/STATUS.md`.

## Exact forward versus surrogate gradient

Flatten leading dimensions to M rows. Let latent weights be L in R^(O×K), attached hard weight signs S=sign₊(L), raw weight scale r=initial+offset and α=max₊(r). The project uses derivative 1 at r=0. Let β be the detached per-input-row F64 mean absolute activation, rounded to F32, and Q the actual dequantized activation returned by `_HardActivationSTE`. Its derivative is **identity through Q**, not sign clipping and not differentiation of β.

The surrogate is

```text
Z = Q Sᵀ
Ys = Z ⊙ α + b
```

The deployment-faithful A1 forward is

```text
A = sign from raw F32 bits; signed zeros map positive
D = A Sᵀ                  # exact integer-valued F32 dot at deployed K < 2^24
Yn = (D ⊙ α) ⊙ β + b     # preserve this multiplication order
Y = Yn.detach() + (Ys - Ys.detach())
```

Raw-bit sign extraction deliberately preserves negative subnormals that CUDA floating comparisons can flush. Therefore Q/β's sign and A's raw-bit sign must not be assumed identical in every edge case. Use **actual Q** for surrogate gradients, even when native A is saved or packed. The existing native forward remains the reference; do not replace its output with `Q Sᵀ`, change F64 β reduction, combine scaling order, or let a fused multiply-add absorb the bias.

For upstream gradient G=dLoss/dY, ordinary surrogate first derivatives are

```text
H = G ⊙ α
dQ = H S
dS = Hᵀ Q
dα[o] = Σm G[m,o] Z[m,o]
dL = dS ⊙ 1{|L| <= 1}
dOffset = dα ⊙ 1{r >= 0}
dX = dQ                   # A1 identity activation STE
```

Frozen bias receives no gradient; if a generic trainable bias were supported, its derivative would be `Σm G[m,o]`. No derivative of detached β or native A is permitted. Preserve A16 cast gradients and A8/A4 identity/dequantized STE independently if extending the operator.

One forward GEMM is sufficient **mathematically**. Reuse the weight-backward contraction:

```text
T = Gᵀ Q
dS = T ⊙ α[:,None]
dα[o] = Σk T[o,k] S[o,k]
dQ = (G ⊙ α) S
```

The custom forward computes only Yn. Save Q, attached S and α as inputs through `save_for_backward`; backward computes T and dQ. Passing Q/S/effective α through the existing activation, shared sign and scale nodes is safer than reimplementing all their STE rules. Their autograd nodes supply clipping, shared accumulation and derivative-at-zero semantics. Clear the per-round cache as today; never reuse its graph after a backward or optimizer update.

**Zero-scale requirements:** α=0 suppresses that row's dS and dQ contribution, but dα can remain nonzero and revive a zero row. Never reconstruct dα from `dS/α` or `Y/α`. When β=0, Q=0 gives dS=dα=0, yet dX can remain nonzero through the identity activation STE. Differentiating the integer-dot/scaled native expression directly would instead introduce a β factor and lose that required gradient.

**Floating-point qualification:** `α*(GᵀQ)` versus `(G*α)ᵀQ`, and `row_sum((GᵀQ)*S)` versus `column_sum(G*(QSᵀ))`, change F32 rounding/reduction order. This is the same algebraic surrogate, **not bitwise-equivalent implementation gradients**. Recomputing old Z during backward restores that scale-gradient contraction but merely moves the removed forward GEMM. A separate sign-dot-derived dα also changes its rounding and can mishandle subnormals. The appropriate gate is finite, bounded gradient/update drift and unchanged material drafter decisions, not an unsupported exact-resume claim.

For later-position losses, return dQ without detaching it, and retain state/K/V casts and concatenations. `ctx.needs_input_grad` may omit truly unnecessary work but cannot be inferred from loss placement. A later-only CE must still reach the earlier state, K and V. Higher-order autograd is a separate contract: saved **inputs** and ordinary Torch backward formulas permit recording a backward graph when `create_graph=True`; saving detached intermediates or using opaque Triton kernels requires an explicit double-backward implementation or `once_differentiable`. The existing diagnostic uses retained first-order graph traversal, not evidence of second-order correctness. See [PyTorch double-backward guidance](https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html) and [save_for_backward](https://docs.pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.save_for_backward.html).

## Performance bound and memory caveat

For an attached projection needing both operand gradients, current A1 has approximately **2 forward + 2 backward matrix products**; a custom operator has **1 forward + 2 backward products**, plus scale reduction. Equal-cost arithmetic gives **4/3×**, not 2×. A prefix projection has two forward products and no backward, giving an ideal local **2×** arithmetic reduction. If C is prefix projection work and U attached projection work, an idealized lane ratio is `(2C+4U)/(C+3U)`, between 4/3 and 2. It assumes equal costs, ignores norms/attention/launches/optimizer/extra reductions, and is not an end-to-end latency prediction. If a profile attributes fraction f of total time to removable work, the optimistic ceiling is `1/(1-f)` before replacement overhead. Paired-lane gain also depends on the A1 share of pair time; A8 is unaffected by an A1-only operator.

At M=1, T is a full O×K outer product. A naive Torch `T*S` materializes another weight-sized temporary before reduction, and `T*α` creates another dense tensor; all nine weights total 0.813 GiB. These extra buffers and reads can be material under the 12 GiB reserved-memory gate. A fused row reduction/scaling epilogue avoids materializing `T*S`; avoid cloning S per invocation. Full-size backward products and Adam moments remain F32 even if forward signs are packed. Benchmark memory as well as GEMM count.

## Primary evidence and its limits

| Method/source | Documented result | Applicability and limit |
| --- | --- | --- |
| [TorchAO QAT overhead](https://pytorch.org/blog/quantization-aware-training/) | Llama3 8da4w fine-tuning: 546.314 versus 359.637 tokens/s; 67.501 versus 69.850 GB per GPU on six 80GB A100s. | Confirms fake-quant overhead is real. The table is a ~34% throughput decrease, not a 34% increase in time. Different model, quantizer, batch and hardware; no W1A1 gain prediction. |
| [PyTorch compile/FSDP study](https://pytorch.org/blog/maximizing-training-throughput/) | Reported MFU gains 10–23%; 7B A100 MFU 0.57→0.68. | Large batched Llama2 on 128 A100/464 H100 GPUs, with selective checkpointing and graph-break fixes. Supports compile exploration, not transfer of the percentage to one-token recurrent W1Ax. |
| [Triton quantization fusion upstream report](https://pytorch.org/blog/fp8-training-on-amd-gpus-with-torchtitan-and-torchao-upstreaming-performance-improvements/) | DeepSeek-V3 MoE forward fusion recovered 89% of FP8 overhead, +17% end-to-end (5,996→7,027 tokens/s versus 7,156 BF16) on 8×MI325X. | Direct quantization-fusion evidence, but enormous AMD FP8 MoE matrices are unlike SM120 vector projections. It supports eliminating measured launch/traffic overhead, not low-bit substitution. |
| [Triton fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | Approximately 4× versus unfused Torch JIT example; native Torch is already competitive across several table sizes. | Illustrates DRAM-traffic savings. Benchmarks use 4,096 rows; rows must fit SRAM. Neither a universal 4× claim nor evidence for a one-row A1 reduction. |
| [AdamW official API](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html) | Fused/foreach are typically faster than single-tensor; foreach uses approximately one parameter-size extra peak memory. No universal measured gain. | `fused=True` supports F32 and preserves AdamW intent; test installed CUDA implementation, memory and update drift. Distinguish it from optimizer-in-backward. |
| [Optimizer-in-backward tutorial](https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html) | ViT-L/Adam demonstration drops roughly 6→4GB peak by freeing gradients and reducing intermediates. | Deferred: updating each parameter before all gradients exist breaks the current global clipping/check-before-step semantics and complicates recurrent/shared graphs. The project already disables foreach. |
| [TorchAO II / pinned FP-Quant autograd source](https://github.com/IST-DASLab/FP-Quant/blob/293b2bf60d5628513532d2594652293dbddae224/inference_lib/src/fp_quant/module/linear_fns.py) | [TorchAO II](https://pytorch.org/blog/quantization-aware-training-in-torchao-ii/) points to MXFP4 forward/MXFP8 backward as a QAT kernel approach; does not report a W1Ax training gain. | Architectural example only. Those precisions and gradient quantizers do not implement the current one-bit native forward/F32 surrogate. |

Do not adopt the FP16 Triton matmul tutorial as a drop-in: its stated cuBLAS-comparable result is **FP16 GEMM**, not this F32 workload. [Triton `dot`](https://triton-lang.org/main/python-api/generated/triton.language.dot.html) defaults F32 tensor-core inputs to TF32 and warns of truncation; specify IEEE when preserving this contract. ±1 operands are representable in reduced formats, but that alone does not prove the entire scale, norm, residual, attention and backward path preserves cancellation/sign decisions. Any exact integer-dot specialization must validate integer accumulation and the separate F32 scale/bias order; blanket autocast/TF32/FP8 substitutions are rejected.

## Smallest decisive experiment, if separately authorized

1. Obtain one bounded phase profile on actual RTX 5080 SM120 from representative **training** rounds: short/median/long prefix, every domain, rejected and accepted continuation roots. Separate prefix, proposal, backward, optimizer, diagnostics, transfer, synchronization and checkpoints; record M/K/O, allocator allocated/reserved peaks and source/runtime identity. Ten warmed rounds per shape class are enough to decide whether to prototype; no sealed evaluation is necessary.
2. A/B the no-gradient native-only path first. Require exact per-projection output and later raw sign agreement on zero/cancellation, signed-zero, negative-subnormal, tails and bias fixtures; then same prefix state/stored F16 K/V and proposal decisions on the bounded train roots. Count removed products and time the full lane, not just a large synthetic GEMM.
3. Separately prototype the custom-autograd formulas with ordinary F32 Torch products. Compare dX/dL/dOffset and one clipped AdamW update against the existing surrogate, including α=0, r<0/r=0, β=0, |L|=1/outside clip, repeated attached signs and later-only state/K/V loss. Finite-difference gradcheck against the discontinuous hard forward is inappropriate; compare against the defined surrogate VJP. Predeclare tolerances, e.g. gradient relative L2≤1e-5 plus abs≤1e-6 for near-zero entries initially; if exceeded, examine whether sign flips, scale projection or drafter decisions change rather than silently relaxing them. Exact thresholds are a proposed gate, not measured evidence.
4. Run an order-balanced warmed A/B of 30 matched train rounds with no persistent optimizer-state change, followed by a short separate smoke update sequence if numerics pass. Require no material margin>0.02 native choice change, preserve the existing state/logit RMS gate, and fit the reserved-memory limit. A suggested implementation acceptance gate is ≥10% median **paired-step** reduction with consistent improvement across prefix classes and no p95 regression; it is a decision threshold, not an expected gain.

For regional compile, start with pure quantization/scale/STE helpers and report compile cost, recompilations and graph breaks separately. Variable prefix lengths, Python position branches, observer hooks and diagnostic tensor-to-host conversions make whole-loop capture unattractive. [PyTorch dynamic-shape guidance](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html) explains shape guards/recompilation; [the tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) supports pointwise/reduction fusion. Q/K/V and gate/up can reuse one attached quantized input while autograd sums branch gradients; do not detach that shared activation or change per-projection monitoring semantics.

The live runtime records math-source hashes and CUDA precision settings in `continuous_runtime.py`; implementation changes need a new explicitly identified experiment, not an invisible resume of the frozen pipeline. Stop after the phase profile/no-grad A/B if removable arithmetic is a small fraction, or if custom temporaries exceed memory. Batching/cache-prefix work, optimizer algorithms/STE choice, learned quantizers and curricula are separate audits; this report does not authorize those changes or conflate their gains with fusion.
