# Project overview: W1A1 EAGLE speculative decoding

## Purpose and central question

Investigate whether an EAGLE speculative drafter with true one-bit weights and
one-bit activations improves end-to-end inference throughput. The target model,
verification, and sampling semantics remain unchanged.

The central question is whether reduced draft execution time outweighs reduced
speculative acceptance. A faster binary matrix multiplication alone is not a
successful outcome. A reproducible negative result that identifies the limiting
cost is also a successful research outcome.

Treat this document as the project's overarching guidance. Update it when an
experiment changes a major assumption, and record the evidence in `experiments/`.
The initial time budget is approximately two to three weeks of experimental work.

## Starting point and boundaries

- Candidate target: `Qwen/Qwen3-4B`.
- Candidate drafter: `AngelSlim/Qwen3-4B_eagle3`.
- Runtime: extend llama.cpp/ggml rather than building an inference engine.
- Day-to-day experiment hardware: RTX 5080. Use it for conversion, W1A1
  simulation/QAT, and general iteration when available.
- Binary Tensor Core measurement hardware: RTX 2080 Ti, Turing, compute
  capability 7.5. Re-run all comparison anchors on this device before making
  an SM75 end-to-end speedup claim.
- Start from existing draft weights. Fine-tune the small EAGLE component with
  quantization-aware training if needed; do not train a foundation model.
- Keep target precision, weights, sampling, and verifier settings fixed within
  each comparison. If memory requires a quantized target, label that experimental
  track explicitly and use the same target in every baseline in that track.

Out of scope: broad architecture support, a general binary CUDA library, a new
serving engine, extensive production integration, and foundation-model training.

## What W1A1 means here

For selected major drafter linear layers, approximate weights and activations as

```text
W_b = alpha * sign(W)
X_b = beta  * sign(X)
```

The sign convention at zero, scaling granularity, scale precision, clipping, and
STE gradient rule are explicit experiment parameters. QAT retains latent
full-precision weights while the forward pass uses binarized operands. Begin
with PyTorch simulation to evaluate acceptance before substantial kernel work.

The final accelerated path packs both operands and performs binary arithmetic.
For equal bit encodings of ±1 values and a logical reduction length K:

```text
dot(a, b) = K - 2 * popcount(packed(a) XOR packed(b))
```

Padding bits must be masked or accounted for; alpha and beta are applied to the
integer dot product. One-bit storage followed by floating-point dequantization
does not qualify as true binary execution. Normalization, attention softmax,
residual additions, bias, and scaling may remain at normal precision.

Audit the actual drafter graph before deciding coverage. Record each linear
layer's shape, latency, precision, and binary eligibility. Treat feature fusion,
attention/MLP projections, and the vocabulary head separately. If a large head
remains FP16, report the mixed-precision coverage and its share of total cost.

## System design

```text
unchanged Qwen3 target -> target hidden states -> EAGLE drafter
                                                |
                              activation sign extraction + packing
                                                |
                                packed W1A1 linear operations
                                                |
                              scaling + normal-precision scaffolding
                                                |
                                           draft tokens
                                                |
                                  unchanged target verification
```

Reuse upstream target inference, EAGLE feature plumbing, verification, sampling,
KV caches, CLI, and server. Custom runtime work is limited to a packed W1 tensor
representation/export, activation packing, an SM75 binary matrix operation, and
dispatch for the selected EAGLE linear layers. Keep development prototypes in
this repository; integrate graph/backend changes in the llama.cpp submodule and
preserve them as reviewable commits or patches before sharing experiments.

## Milestones and decision gates

| Stage | Work and deliverable | Decision gate |
| --- | --- | --- |
| 0: preparation, days 1–3 | Pin and convert the target/draft pair; audit drafter layer shapes, precision, and binary eligibility. Resolve conversion or memory issues that block W1A1 work. | The model artifacts and graph information needed for simulation are available. A separate FP16 EAGLE benchmark is not a prerequisite. |
| 1: acceptance, days 3–7 | Simulate W1A1 in PyTorch; ablate linear groups and scaling; use bounded QAT if initial acceptance is poor. | Measure held-out W1A1 acceptance and choose a candidate for native execution. Defer relative throughput and break-even conclusions until paired comparison runs. |
| 2: kernels, days 7–11 | Implement a numerical reference, packer, and SM75 XOR/POPCOUNT kernel for measured shapes; benchmark packing-inclusive latency. | Correctness covers tails, scales, layouts, and zero-sign behavior. Real layer shapes show useful savings after packing and launch costs. |
| 3: integration and comparison, days 11–15 | Route selected draft operations to binary execution. Run target-only, FP16 EAGLE, and native W1A1 under matched settings on each relevant GPU; measure acceptance and end-to-end latency together, profile bottlenecks, and write a report. | Report repeatable throughput gains or a quantified negative result with artifacts sufficient to reproduce it. |

These dates are planning estimates, not a reason to expand scope. If a gate fails,
run a small diagnostic or ablation, record the outcome, and favor a clear result
over adding more architectures or training infrastructure.

### Evidence at the current gate (2026-09-24)

The pinned PyTorch fake-binary drafter and 12-prompt held-out RTX 5080 sweep are
implemented. Under the AngelSlim BF16 verifier, ordinary EAGLE accepted 2.317
draft tokens/round, head-only W1A1 1.677, and all selected W1A1 groups 0.202.
W4A4 and W8A8 simulations over all candidate linears accepted 0.288 and 2.182
respectively. The nine candidate linears occupied 38.32% of one-prompt
instrumented draft time. See the [W1A1 CUDA report](../experiments/pytorch-w1a1-cuda-acceptance.md)
and [W4A4/W8A8 report](../experiments/pytorch-int4-int8-cuda-acceptance.md).

Strict target-only versus ordinary EAGLE greedy parity fails at generated
token 4 on CUDA BF16 through a target-selected verifier tie. A bounded 5080
replay found that off-path sibling tokens do not alter the selected hidden
state or logits; the tree row's 21.0 tie versus the incremental/full-prefix
target's 20.875 logit for ID 11 is the immediate mismatch source. The
acceptance counts are verifier-relative exploratory evidence, not a
target-equivalent speed claim.
The current autonomous goal investigates that discrepancy, tests a bounded
head-only QAT recovery path, and builds a packed native numerical path before
end-to-end comparison. The [native integration plan](native-w1a1-path.md)
records the proposed narrow GGML/GGUF changes.

The standalone packed CUDA kernel has since passed eight-width correctness
and packing-inclusive component timing on the 5080; see the
[native prototype report](../experiments/native-w1a1-cuda-prototype.md). A
dedicated GGML operation and EAGLE packed-head bridge are in the pinned
llama.cpp fork, with CPU tests and local GGUF load checks. GGML CUDA execution,
head-QAT held-out acceptance, and matched end-to-end throughput remain the
open gates.

An approximate planning model is `throughput = emitted tokens per round / round
time`, with round time including draft, verification, and all other overhead.
Use values measured in the paired comparison, including target-emitted tokens,
when estimating the break-even draft cost. The final decision comes from
end-to-end timing.

## Comparison and evidence

Mandatory anchors are no speculation and normal FP16 EAGLE. Desired additional
comparisons are W8A8, W4A4, and W1A1 EAGLE. Record the actual execution path for
every precision; weight-only GGUF quantization is not automatically W8A8/W4A4.
If a genuine low-bit baseline is unavailable within scope, state that limitation.
Keep 5080 development results separate from 2080 Ti binary-speed conclusions;
each hardware track needs its own same-device anchors. Measure the anchors in
direct, matched comparisons with native W1A1 rather than as a separate early
baseline milestone.

Measure draft latency, activation packing, binary kernels, output-head cost,
verification latency, accepted/drafted tokens, accepted tokens per round,
end-to-end output tokens/sec, and speedup over both mandatory anchors. Include
memory usage and runtime/environment metadata. Separate prefill from decode and
report request time as well as decode throughput. See [evaluation](EVALUATION.md).

Success requires correct binary math, preserved target/verifier behavior, clear
precision coverage, and repeatable measurements on the RTX 2080 Ti. Validation
includes greedy equivalence checks and appropriate sampling correctness checks;
do not assume identical seeded sampled sequences across different execution paths.

## Principal risks and responses

| Risk | Evidence to collect / bounded response |
| --- | --- |
| Acceptance collapses | Layer-group and scaling ablations, then limited QAT; evaluate held-out prompts. |
| Packing costs exceed savings | Time sign extraction, packing, scales, and GEMM together; investigate fusion/reuse only if profiling supports it. |
| Small matrices underutilize hardware | Use measured batch/token shapes; include launch and synchronization costs rather than large-GEMM peak claims. |
| Vocabulary head dominates | Profile and ablate head precision explicitly; calculate the maximum gain from accelerating the remaining layers. |
| Hardware/toolchain mismatch | Compile an SM75 instruction probe, validate numerical behavior, and inspect generated instructions on the benchmark host. |
| Target, draft, and caches exceed memory | Establish the memory budget early; reduce fixed context/batch settings or define a separately labeled target-precision track. |

## Novelty and final deliverables

The candidate contribution is the combination of EAGLE drafting, W1A1 operands,
packed native binary execution, and an end-to-end acceptance/latency evaluation.
Do not claim to invent binary inference or quantized speculative decoding, and
treat novelty as provisional until a focused related-work review is complete.

Deliver a pinned runtime, conversion/quantization configuration, bounded training
recipe if used, binary correctness checks and kernels, reproducible benchmark
commands/raw results, and a short report explaining gains or failure modes.

## Verified starting references

At repository setup, llama.cpp revision
`6e60f35608ec6918b44a9839c0c433687165f086` documents this target/drafter pair and
`draft-eagle3`. That establishes upstream support, not a successful local model
run. The starting scaffold had no W1A1 implementation or measured results;
the current state is summarized above and in `docs/STATUS.md`.

- [Pinned upstream EAGLE-3 instructions](https://github.com/ggml-org/llama.cpp/blob/6e60f35608ec6918b44a9839c0c433687165f086/docs/speculative.md)
- [Candidate draft checkpoint](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3)
- [Candidate target checkpoint](https://huggingface.co/Qwen/Qwen3-4B)
- [NVIDIA PTX ISA 8.5](https://docs.nvidia.com/cuda/archive/12.6.0/parallel-thread-execution/index.html): binary `mma` XOR/POPCOUNT and architecture requirements. Validate the exact instruction shape and toolkit on SM75; do not assume all binary instruction variants work on Turing.
