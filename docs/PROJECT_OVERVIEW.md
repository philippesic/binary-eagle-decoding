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
- Primary measurement hardware: RTX 2080 Ti, Turing, compute capability 7.5.
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
| 0: baseline, days 1–3 | Build on SM75; convert the pair; validate target-only and FP16 EAGLE generation; record memory, layer shapes, acceptance and latency. | Both runs work under the same workload and fit memory. Resolve compatibility before quantization. |
| 1: acceptance, days 3–7 | Simulate W1A1 in PyTorch; ablate linear groups and scaling; use bounded QAT if initial acceptance is poor. | Measure held-out accepted tokens per round. Estimate the draft-time reduction needed to beat normal EAGLE; stop or narrow scope if even optimistic savings cannot help. |
| 2: kernels, days 7–11 | Implement a numerical reference, packer, and SM75 XOR/POPCOUNT kernel for measured shapes; benchmark packing-inclusive latency. | Correctness covers tails, scales, layouts, and zero-sign behavior. Real layer shapes show useful savings after packing and launch costs. |
| 3: integration, days 11–15 | Route selected draft operations to binary execution; compare end-to-end runs; profile bottlenecks and write a report. | Report repeatable throughput gains or a quantified negative result with artifacts sufficient to reproduce it. |

These dates are planning estimates, not a reason to expand scope. If a gate fails,
run a small diagnostic or ablation, record the outcome, and favor a clear result
over adding more architectures or training infrastructure.

An approximate planning model is `throughput = emitted tokens per round / round
time`, with round time including draft, verification, and all other overhead.
Use measured values, including target-emitted tokens, when estimating the
break-even draft cost. The final decision comes from end-to-end timing.

## Comparison and evidence

Mandatory anchors are no speculation and normal FP16 EAGLE. Desired additional
comparisons are W8A8, W4A4, and W1A1 EAGLE. Record the actual execution path for
every precision; weight-only GGUF quantization is not automatically W8A8/W4A4.
If a genuine low-bit baseline is unavailable within scope, state that limitation.

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
run. The current scaffold has no W1A1 implementation or measured results.

- [Pinned upstream EAGLE-3 instructions](https://github.com/ggml-org/llama.cpp/blob/6e60f35608ec6918b44a9839c0c433687165f086/docs/speculative.md)
- [Candidate draft checkpoint](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3)
- [Candidate target checkpoint](https://huggingface.co/Qwen/Qwen3-4B)
- [NVIDIA PTX ISA 8.5](https://docs.nvidia.com/cuda/archive/12.6.0/parallel-thread-execution/index.html): binary `mma` XOR/POPCOUNT and architecture requirements. Validate the exact instruction shape and toolkit on SM75; do not assume all binary instruction variants work on Turing.
