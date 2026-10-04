# Additional speculative inference deployment probes

Research completed October 4, 2026. Reporting-only documentation persisted after the read-only research phase. No implementation edits, commits, tests, model execution, downloads or host operations were performed. The root agent owns integration and commits.

Recommend three additional deployment probes, ranked below.

Current evidence remains bounded: RTX2080Ti/SM75, F16 target/verifier and target/draft KV; released DSpark/DFlash floating matrices have BF16 storage with F32 norm widening. Their Q4 control changes exactly fifteen FFN matrices. Both D3 and D7 still compute seven noise rows. Intrusive head/Markov totals identify investigation targets but do not predict clean request speed. Resident state was about 1% slower; shared-pack/warp changes were near control noise. Local evidence: `experiments/dspark-sm75-20261003/results.md`, `experiments/eagle-runtime-optimizations.md`, and `experiments/nine-model-qat-research-plan-2026-10-04.md`.

## 1. Skip unused DSpark confidence computation and transfer at p_min=0

**Disposition: incorporate after native admission.**

**Mechanism and primary evidence:** Static verification need not compute scheduling confidence. SGLang already uses this distinction: its DSpark-V4 implementation returns no confidence head for static verification. This is a serving precedent, not evidence about Qwen3 model numerics or performance. [Primary implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v4_dspark.py).

**Verified local hooks:** [speculative.cpp:1633](/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/common/speculative.cpp:1633) reads confidence only when `params.p_min > 0`, yet [speculative.cpp:1361](/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/common/speculative.cpp:1361) enables nextn extraction unconditionally. [dflash.cpp:304](/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/src/models/dflash.cpp:304) decides to build confidence from weight presence alone. Lines 370–406 construct per-position projections, sigmoid, concatenation, layout conversion and an `n_embd`-wide broadcast. [llama-context.cpp:2387](/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/src/llama-context.cpp:2387) then permits its host copy.

**Integration cost:** Small-to-medium. Add a runtime “confidence output requested” graph parameter and suppress both graph construction and extraction when absent. Preserve confidence weights and restore the path for `p_min>0`, requested diagnostics and DFlash2’s separate selector lattice. Include the parameter in graph reuse compatibility.

**Performance hypothesis:** Saves small projections, graph nodes, broadcast work and an unused device-to-host copy. Potentially useful when launch/host cost dominates. The synchronized profile’s approximately 0.8–1.1 ms confidence totals cannot establish a request gain.

**Counterargument:** Confidence arithmetic is small; removing nodes can change allocator placement and scheduling sufficiently to expose decision drift. Simply disabling extraction leaves most arithmetic intact.

**Minimal acceptance check:** Native on/off D7 pairs for all three DSpark precision cells once implemented, including root rejection, deep acceptance, EOS and longer contexts. Require identical proposals, complete generated IDs, accepted-prefix/round counters, target-feature ancestry, masks and cache rollback. Confirm the confidence nodes/copy disappear; retain a `p_min>0` restoration check. Run clean order-balanced request timing with events/captures off and matched warmups. Apply it to DSpark Q4 too; DFlash/EAGLE are unaffected controls.

## 2. Reduce repeated copying in DSpark’s full-vocabulary output assembly

**Disposition: probe a small graph-only version first.**

**Mechanism and primary evidence:** The author’s Markov implementation collects corrected columns and concatenates once after its sequential conditioning loop. Its previous-token dependency remains essential. [Pinned author code](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/markov_head.py).

**Verified local hooks:** [dflash.cpp:368](/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/src/models/dflash.cpp:368) grows `cat` after each corrected column; [dflash.cpp:388](/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/src/models/dflash.cpp:388) conditions the next projection on the current argmax. Final layout restoration occurs at [dflash.cpp:394](/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/src/models/dflash.cpp:394). These concats dispatch copying kernels through [concat.cu:197](/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/ggml/src/ggml-cuda/concat.cu:197).

**Integration cost:** Small for a balanced concatenation tree built after collecting the columns. GGML concat is binary, so the author’s single `torch.cat` is not a literal drop-in. For seven columns, the current growing chain writes 27 column-equivalents; a balanced tree can write 20, with the same six concat nodes. This is graph arithmetic, not a measured saving. Investigate avoiding the final permutation copy for the concurrency-one identity layout separately.

A preallocated output with indexed column writes could approach linear copying, but allocator alias/dependency handling raises the cost; defer that extension until the simpler probe matters.

**Performance hypothesis:** Reduces full-vocabulary temporary traffic and may shorten assembly dependencies without changing head/Markov dot products or the seven-row backbone.

**Counterargument:** Seven is small, kernel launches remain, and preserving several columns longer can increase peak allocation. The profile combines Markov projections with assembly, so its material total cannot be attributed to concat alone.

**Minimal acceptance check:** Compare corrected logits and per-position greedy IDs under matching conditioning history, then complete native trajectories and serialized used-cache state. Cover one and multiple blocks, source slot mapping and EOS. Test every applicable DSpark Q4/W1A8/W1A1 cell; keep confidence suppression off to isolate this change. Clean balanced timing decides promotion. Probe, then incorporate only if the native gain survives allocation and quality checks.

## 3. SM75-only floating-exception BF16→F16 storage/dispatch probe

**Disposition: conditional development ablation outside frozen controls.**

**Mechanism and primary evidence:** Turing supports FP16 Tensor Core inputs, while native BF16 Tensor Core support starts with Ampere. [NVIDIA hardware documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html). Locally, BF16 seven-column work can select a vector path where F16 can select existing matrix-MMA code.

**Verified local hooks:** [mmvf.cu:859](/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/ggml/src/ggml-cuda/mmvf.cu:859) permits BF16 MMVF through eight columns on hardware without BF16 MMA. F16’s Tensor Core-capable branch permits only up to three columns. [mmf.cu:187](/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/ggml/src/ggml-cuda/mmf.cu:187) admits F16 on Turing; its BF16 branch requires Ampere. Dispatch ordering is at [ggml-cuda.cu:2130](/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:2130). Thus eligible seven-row floating heads/fusion/attention may reach a different existing kernel without writing new MMA code.

**Integration cost:** Small exporter/storage experiment; medium admission burden. Start with one large floating projection, preferably the private full head. Audit that every source value survives BF16→F16→F32 exactly; BF16’s larger exponent range means this cannot be assumed. Preserve the private source values and immutable teacher.

**Performance hypothesis:** Better seven-row Tensor Core utilization for floating exceptions left outside fifteen-FFN Q4/W1 coverage. Memory remains two bytes per weight, so this is a dispatch hypothesis, not compression.

**Counterargument:** F16 dispatch changes activation rounding and accumulation. [mmf.cu:19](/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/ggml/src/ggml-cuda/mmf.cu:19) records F16 operand rounding/F32 MMA; [mmvf.cu:656](/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/ggml/src/ggml-cuda/mmvf.cu:656) exposes different vector accumulation contracts. Exact stored values therefore do not prove identical proposals. Thin GEMM can also remain bandwidth-bound.

**Minimal acceptance check:** Require source-value preservation, actual SM75 dispatch evidence, finite logits and unchanged greedy proposals under matching histories, followed by native acceptance and clean timing. Failure is a rejected deployment policy, not permission to broaden numerical tolerances. Test matched Q4 and binary candidates; label floating-exception storage and actual activation/accumulation explicitly. Probe outside the frozen nine-model controls; any final policy change needs the user’s scope decision and matching QAT arithmetic. No SM120 benefit follows from this argument.

## Rejected tempting shortcut

Set global `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16`. [ggml-cuda.cu:1926](/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1926) applies this override across cuBLAS work, potentially touching the verifier and explicit F32 choices. It also does not redirect BF16 MMVF operations. It neither isolates the proposed draft improvement nor preserves the frozen arithmetic contract.

## Handoff

Completed: read-only report with primary citations, verified local hooks, dispatch contracts, evidence limits and proposed admission gates. No new performance is claimed. Remaining work is an orchestrator/user scope decision followed by isolated implementation and native quality/clean timing probes if authorized. Apply every shared improvement to applicable Q4 controls. Target/verifier/RNG, masks, cache rollback, source ancestry and actual greedy behavior remain protected.
