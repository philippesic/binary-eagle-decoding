# RTX 2080 Ti quantization benchmark synthesis

**Hardware:** NVIDIA RTX 2080 Ti, SM75, 11,264 MiB; Ubuntu 24.04 WSL2; CUDA 12.8.93.
**Workload:** Qwen3-4B target and AngelSlim EAGLE-3 draft, 12 frozen prompts, greedy decoding, 128-token cap, F16 target/KV, context 2,048, one server at a time, two warmups and five measured repetitions per variant.
**Evidence:** [raw provenance and per-run reports](rtx2080ti-quantization-suite.md); [active goal checkpoint](../docs/goals/rtx2080ti-quantization-suite.md). Raw captures remain on the WSL host under the report's hashed result directories.

## Result

All five **native packed W1A1** layer settings were slower than ordinary EAGLE on the same 2080 Ti. Genuine all-nine **W8A8** was effectively tied with ordinary EAGLE. Genuine all-nine **W4A4** was much slower because accepted drafts collapsed. Existing **Q4_0/Q8_0** draft GGUF controls were faster than ordinary EAGLE, but their block-scale arithmetic is distinct from the W4A4/W8A8 research quantizers. Moving the binary head or the native INT8/INT4 paths to Turing's specialized matrix instructions did not improve whole-request speed in this single-sequence workload.

Each ratio below comes from a **direct paired run with its own ordinary or same-format default anchor**. Absolute rates from different run tracks are not compared to one another.

| Comparison track | Candidate / anchor decode-rate ratio | Paired 95% interval | Accepted drafts/round for candidate / anchor |
|---|---:|---:|---:|
| W1A1 fusion / ordinary EAGLE | 0.598 | 0.545–0.660 | 0.271 / 1.168 |
| W1A1 attention / ordinary EAGLE | 0.596 | 0.552–0.646 | 0.238 / 1.168 |
| W1A1 FFN / ordinary EAGLE | 0.716 | 0.677–0.756 | 0.472 / 1.168 |
| W1A1 head / ordinary EAGLE | 0.907 | 0.869–0.946 | 0.860 / 1.168 |
| W1A1 all nine / ordinary EAGLE | 0.561 | 0.511–0.617 | 0.055 / 1.168 |
| Q4_0 draft GGUF / ordinary EAGLE | 1.109 | 1.081–1.135 | 1.182 / 1.168 |
| Q8_0 draft GGUF / ordinary EAGLE | 1.064 | 1.053–1.075 | 1.168 / 1.168 |
| Genuine W8A8 DP4A / ordinary EAGLE | 0.990 | 0.975–1.005 | 1.127 / 1.168 |
| Genuine W4A4 vector / ordinary EAGLE | 0.511 | 0.466–0.561 | 0.092 / 1.168 |
| Binary Tensor Core head / portable W1A1 head | 1.002 | 0.997–1.006 | 0.860 / 0.860 |
| W8A8 INT8 Tensor Core / W8A8 DP4A | 0.821 | 0.817–0.824 | 1.127 / 1.127 |
| W4A4 INT4 Tensor Core / W4A4 vector | 0.880 | 0.876–0.884 | 0.092 / 0.092 |

The W1A1 all-group drafter cut measured host draft-call time per verification round from 6.471 to 3.558 ms, but accepted drafts fell from 1.168 to 0.055 per round. Verification rounds rose from 3,400 to 6,940, and total draft-call time rose from 22.0 to 24.7 seconds. A faster binary dot therefore did not translate to a faster decoder. W8A8 shortened draft-call time slightly (6.617 to 6.329 ms/round) but accepted slightly fewer drafts, yielding no resolved throughput change. W4A4 had both poor acceptance and no per-round draft-cost saving (6.661 versus ordinary's 6.617 ms).

In the opt-in Tensor Core comparison, each MMA path used the **same GGUF, draft codes, scales, target, prompts, and accepted-draft totals** as its default counterpart. Yet host draft-call time per round rose from 6.352 to 12.054 ms for W8A8 and from 6.638 to 10.189 ms for W4A4.

An independent [Nsight Systems CUDA trace](rtx2080ti-quantization-suite.md) measured activation packing and matrix-dot kernels on real EAGLE shapes. The table reports the median of **same-stream, sequential pack-plus-dot pairs** over three short requests; these are profiler diagnostics in microseconds, not the end-to-end benchmark timings.

| Actual layer shape | Default pack+dot | Tensor Core pack+dot | Pair counts |
|---|---:|---:|---:|
| W1A1 head, N=1, M=32,000 | 31.4 µs | — | 15 |
| W1A1 all-group FFN output, N=1, M=9,728 | 14.7 µs | — | 36 |
| W8A8 head, N=1, M=32,000 | 149.4 µs | 434.7 µs | 15 per path |
| W8A8 FFN output, N=1, M=9,728 | 51.9 µs | 180.5 µs | 30 per path |
| W4A4 head, N=1, M=32,000 | 152.7 µs | 362.4 µs | 15 per path |
| W4A4 FFN output, N=1, M=9,728 | 53.3 µs | 129.0 µs | 30 per path |

For larger N=37 head launches, the corresponding W8A8 pack-plus-dot medians were **5.236 ms default versus 3.045 ms MMA**, and W4A4 **5.359 ms versus 2.942 ms** (three pairs per path). Thus these MMA kernels can help at a larger token batch while losing in the repeated N=1 decode work. The 8×8 tile layout's unused columns at N=1 are a plausible cause, inferred from the layout and measured durations; the trace does not prove a unique cause. The profiler adds instrumentation overhead, so the paired request throughput above remains the performance conclusion.

## Execution and correctness evidence

- All five W1A1 variants used packed one-bit weights and runtime-packed one-bit activations with logged CUDA XOR/POPCOUNT dispatch. Exhaustive GGUF audits found zero sign-word mismatches, including all 65,280 rows in the all-group model. Five SM75 backend cases passed. A standalone binary-MMA probe passed 21 cases/880 exact integer dots; integrated portable and MMA paths passed 8/8 cases each, with distinct dispatch and `BMMA.88128.XOR.POPC` instruction evidence.
- W8A8 and W4A4 exporters matched every original BF16 weight code and F32 scale byte across all nine eligible draft linears. Their CUDA backend oracles passed 3/3 and 2/2 cases on the 2080 Ti. The signed-INT8 MMA candidate passed 5/5 backend cases and a 63-case exact-dot probe; the signed-INT4 candidate passed 4/4 and a six-case/308-output probe. Targeted candidate-library disassembly contains `IMMA.8816.S8.S8` and `IMMA.8832.S4.S4.SAT`, respectively. All default and opt-in model runs required their own loader and CUDA markers and rejected the opposite path's marker.
- Every speculative variant within the production nine-path run emitted the same text on all 60 paired requests. The separate low-bit and MMA runs also showed pairwise text identity and identical acceptance totals within each same-format default/MMA pair. Target-only text differed from speculative text on the same `reasoning-02` heading capitalization in all five repetitions. Target-only ratios are therefore timing observations, not strict lossless speedups. The server omitted generated token IDs; token-level identity was not available.

## Precision labels and limits

The Q4_0/Q8_0 control labels name stored GGUF weight formats. A separate short [executed-path trace](rtx2080ti-quantization-suite.md) confirmed `src1=f32` is quantized to `q8_1` on CUDA for both formats. One- and two-token operations selected **MMVQ**; a 38-token operation selected **MMQ**. No cuBLAS path appeared in that diagnostic. The pinned MMVQ source computes Q8_0×Q8_1 with signed-byte `DP4A` and Q4_0×Q8_1 by extracting 4-bit weight codes for byte `DP4A`; the MMQ source can use Turing signed-INT8 MMA after expanding Q4_0 codes. This short trace identifies the runtime branch and activation format, not the full instruction mix of every timed request. The controls must not be relabeled as the whole-row/whole-token W4A4/W8A8 quantizers. The opt-in INT4 Tensor Core path is separate from the default packed-nibble vector implementation.

Request throughput includes prefill; decode throughput uses server-predicted decode duration. Nonstreaming responses did not expose client time to first token. Host draft-call spans include more than the kernel and do not isolate total target verification time. The paired intervals are descriptive for this fixed prompt set, device, and runtime. The evaluation does not establish a general disadvantage for binary, INT8, or INT4 Tensor Cores at larger batch sizes or after different packing/layout work.
