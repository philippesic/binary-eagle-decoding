# Recurrent D CPU proposal-chain diagnostic across three training prompts

**2026-09-28 UTC.** This is an instrumented CPU-only numerical and capture
check, not a model-training, quality or throughput experiment. It used one
frozen training prompt from each of prose, code and reasoning, with eight
output tokens per request. The target was pinned FP16 Qwen3-4B (SHA256
`05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`);
the untrained own-history draft was candidate D (SHA256
`10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf`).
The native fork revision was `b4df1b547`. The Apple M3 Max CPU server build
had CUDA, Metal, Vulkan, SYCL, HIP, RPC, OpenCL, BLAS and Accelerate off,
and both model GPU layer counts were zero. Every server process group stopped
cleanly. No accelerator, remote host, development/final prompt or approved
training trial was used.

The new `run_recurrent_cpu_diagnostic.py` pins those sources and settings,
captures bounded native target and draft graph streams, and runs the
accepted-prefix continuity and raw-response audits. The earlier prose
capture was produced manually under the same pinned CPU policy. For every
recorded round, the current-student CPU adapter rebuilt its cache from
retained raw target features, then followed the native D proposal ancestry
with hard binary signs, group-128 scales and A16 inputs. The summary script
rechecked source hashes, feature/round continuity, response emissions and
each per-depth native proposal ID before counting matches.

| Frozen training prompt | Native rounds | Proposal positions | CPU mapped top IDs matching native | Largest normalized state difference | Accepted drafts by round |
| --- | ---: | ---: | ---: | ---: | --- |
| Prose: urban waterways 01 | 4 | 16 | 16/16 | 0.003462 | 0, 0, 1, 1 |
| Code: streaming text 01 | 4 | 15 | 15/15 | 0.004617 | 0, 1, 0, 1 |
| Reasoning: rate and work 01 | 3 | 13 | 13/13 | 0.004564 | 0, 1, 2 |
| **Total** | **11** | **44** | **44/44** | **0.004617** | — |

All 44 mapped top IDs match along the recorded proposal chains under the
practical grouped-F32-matmul arithmetic. The native-order CPU reference
also matched all five first-round top IDs for each prompt. Since every
earlier proposed token in each chain matched, the fixed-prefix greedy
proposal chains agree for these 11 rounds. The nonzero state drift is
measured, not accepted as a full-drafter parity tolerance. No Python
verifier acceptance or trained-model improvement is inferred.

First-round native graph taps further constrained the drift. Prose and
reasoning input embedding, input norms, fused input and unrotated Q/K/V
matched the CPU adapter bitwise after Q/K row-order conversion. In code,
unrotated Q/K/V maximum differences were at most `1.93e-5`. Across the
three prompts, RoPE differences were micro-scale; attention output was the
first larger gap, with maximum absolute differences of `0.009986`
(prose), `0.005725` (code) and `0.009904` (reasoning). Replaying attention
from **native-captured** Q/K/V and F16-rounded K/V reproduced gaps of
`0.009986`, `0.005692` and `0.009899` respectively. This points to
attention execution as the dominant first-round arithmetic mismatch; it
does not identify one kernel instruction or prove later-depth K/V parity.

The ignored audit summary is
`results/recurrent-binary-cpu-broader-20260928/summary.json` (SHA256
`e37aaab03197a473a2ad749c2d37f88302b35fed8b7d642a1db133f931a050f2`).
The two new cell manifests have SHA256
`733bf519c1ff2cd5e47661d9bf96a5d30b9398330ff1a4e4abee16e240257abd`
(code) and
`d08e1e1bb97ce8852f4718c4b631883f0df945181e7634ec370c86a620bbfadd`
(reasoning). Their graph comparisons and native-Q/K/V replay reports have
SHA256s, respectively,
`92bc9d18d5d4555be28121edf9da8ea497db9fd96cf932b166d1b5be89340dcf` /
`8433b263101f299b8e36227f07eb18cb88c56c8d4f6222e3dc8aef137664d377`
and
`4ddd12785176e998a47834692f16972a5a7c184cedea35e037cd01b2ab3e493b` /
`6ccc64aa07eac40f3a68bfae22d35d22c464ea57cd9b4fe34b17de8102d6d101`.
The [first prose capture report](recurrent-binary-cpu-capture-smoke.md)
records its raw file and graph hashes.

This three-prompt training subset does not establish the proposed 96-prompt
capture, target-feature exactness, full-drafter state/K/V parity, any
trained-model quality gain, or native acceptance/latency/throughput against
Q4_0 EAGLE. Those remain explicit gates.

## Later-round decoder boundary after native acceptance

The CPU adapter and native graph were also joined at one later round in
each prompt, chosen after an earlier native accepted draft. Each adapter
run rebuilt its context cache from the retained raw target-feature rows;
the native graph execution was selected by the round's depth-zero head
state and the frozen D output norm. The join RMS was below `1.3e-7` in
all three cases, with the next nearest graph candidate above `0.48`.

| Prompt/round | Native group | Fused-input max difference | Unrotated Q/K/V | RoPE Q/K largest difference | Attention-output max difference | Prenorm max difference | Mapped seed top ID |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Prose round 3 | 20 | `4.8e-7` | bitwise exact | `1.16e-5` | `0.011560` | `0.014145` | 2176 |
| Code round 2 | 14 | 0 | bitwise exact | `1.56e-5` | `0.011746` | `0.014496` | 13027 |
| Reasoning round 2 | 14 | `2.4e-7` | bitwise exact | `1.38e-5` | `0.009838` | `0.018600` | 12 |

The Python and native mapped seed top IDs matched in all three joined
rounds. This shows the captured accepted-prefix feature ancestry and
current-student rebuild reach the same **pre-attention projection
operands** after those acceptance histories. It still does not directly
read the native stored K/V cache bytes or prove an identical attention
mask/reduction. The attention gap remains the first material numerical
boundary. These three later-round graph comparison JSON files have SHA256
`c606881019b82feed24100e95789f5ffcdd9d7222965d640ae2f7dd7aa0faec9`,
`cc4c2e17a0f7c5493287fd7d0db317ca5f062e01ffd50c112524cabea09e0af1`,
and `68c87c953799bd6fcd457e10405f92571a3ac8adcc5105ee23acdaaa7d867c8a`
for prose/code/reasoning respectively. Their native-order adapter tap
NPZ SHA256s are
`a3db7ca216a1bf7afcbcf3b5954f495064af28de5c1cbecae8ef83ba6332d9b3`,
`f7f791c812b5b1d79bb5ce908c77b7e60d4494aae60f2f7fe2b487d9c40f7748`,
and `bbdb9bbcdad84b15baee8e1d5203af98fe101705e5253ff239e48883585fc6c8`.

## Code round 2 projected K/V write operands

A follow-up CPU comparison rebuilt all 37 decoder context positions preceding
code round 2 from retained target features. Each adapter input embedding and
fused-feature norm joined one native decoder graph column. After converting
the adapter key to native Q/K row order and rounding both projected operands
to F16, **37,681/37,888 key elements** and **37,723/37,888 value elements**
matched bitwise. Every position had exactly one qualifying native graph
column. This confirms close projected K/V write values through the accepted
prefix, while the 207 key and 165 value mismatches rule out bitwise operand
parity. Native stored cache bytes were not captured or read; neither cache
layout/rollback parity nor attention mask/reduction parity follows.

The ignored comparison is
`results/recurrent-binary-cpu-broader-20260928/code-auto/round_02_projected_kv_writes_native_rows.json`
(SHA256 `15e50ae2f8f385195c87ff752c68472884667aeb44a80f07e6fbbf65baf0403e`).
The diagnostic script explicitly distinguishes projected write operands from
native cache storage. An earlier ignored exploratory report without the Q/K
row conversion is invalid and must not be used.

## CPU attention arithmetic ablation

A model-free replay used the archived first prose seed graph row, with 32
attended positions and native-captured Q/K/V. The earlier replay kept Q,
dot products, softmax and value accumulation in F32. Source inspection of
the Apple M3 Max native CPU Flash Attention path found F16 query conversion,
NEON F16 dot accumulation, and an online F16 value numerator. A bounded
arithmetic ablation changed these stages one at a time:

| Replay arithmetic | Maximum absolute difference from native attention | RMS difference |
| --- | ---: | ---: |
| Prior F32 softmax/value replay | 0.0099864 | 0.00106654 |
| F16 query, F32 softmax/value | 0.0108323 | 0.00106794 |
| Modeled NEON F16 dot, F32 softmax/value | 0.00926089 | 0.000915012 |
| F16 dot, ordered F32 online value sum | 0.00926042 | 0.000915017 |
| F16 dot and online F16 value numerator | **0.00178361** | **0.0000366726** |

This strongly implicates native F16 attention arithmetic as the principal
source of the first-row gap. The remaining difference is real; NumPy half
rounding and `math.exp` are approximations of NEON FMA and native `expf`, so
the table does not declare exact attention or whole-drafter parity. The
CPU-only script and three tests were integrated as `4a448b8`. Its ignored
report SHA256 is
`d38190805a8a4cb1a5441d0b8b7000b31a529bcf6155f2c6d6fcd2634b1b9a1e`,
with graph, model-reference and kernel-source hashes. No model inference,
accelerator or performance run was used for this ablation.

The same bounded ablation on the first code and reasoning seed rows reduced
maximum/RMS differences from `0.00569153/0.000875192` to
`0.000000954/0.0000000856` (code) and from `0.00989914/0.00108065` to
`0.000846684/0.0000154912` (reasoning). Their ignored report SHA256s are
`561bb21e1b7b4dca59483c641dc5d5acb415b272fff05f8475ab076aa8a5452b`
and `e3eafe47f2ce68202d9c541ec0d9784758a87f1ac2ff5b6b1554c96de982ec44`.
The residual gap varies by prompt; these three rows do not set a safe state
or logit tolerance for recurrent training.

## Native-style norm and RoPE replay across later rounds

The earlier projected-write discrepancies were localized with the same three
accepted-prefix captures. Every decoder position had one native graph-row
join. Before the adapter change, all native value-write mismatches occurred at
positions with one or two different F16-cast fused-input elements. The raw
target-feature norm joined exactly; the differing elements came from the
borrowed embedding norm near F16 rounding boundaries. The native CPU RMS norm
sums F32 squares in F64, rounds the mean to F32, then applies F32 square root,
reciprocal and weight products. The adapter had used a PyTorch F32 mean.

The corrected adapter now follows that reduction and builds RoPE frequencies
by recurrent F32 multiplication, as the native CPU graph does. A fresh replay
gave the following **projected operands**, before native cache storage:

| Prompt/round | Context positions | F16 fused input | Raw F32 K and V | F16 key write | F16 value write |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose 3 | 35 | 179,200/179,200 | 35,840/35,840 each | 35,840/35,840 | 35,840/35,840 |
| Code 2 | 37 | 189,440/189,440 | 37,888/37,888 each | 37,887/37,888 | 37,888/37,888 |
| Reasoning 2 | 49 | 250,880/250,880 | 50,176/50,176 each | 50,173/50,176 | 50,176/50,176 |

The four remaining F16 key differences occur after RoPE at code position 34
and reasoning positions 15, 20 and 40; the largest difference is
`0.00048828125`. A code round-2 full CPU proposal replay still matched all
native mapped top IDs, and its first-depth maximum state difference changed
from `0.0025558472` to `0.0024642944`. The normalized state is not numerically
interchangeable with native execution. Native stored K/V bytes, rollback,
attention mask/reduction, complete 96-prompt capture and Q4_0 quality/timing
remain unverified.

The ignored corrected reports for prose/code/reasoning have SHA256
`931fb363b4dae81c8540ee36e6a6242c54f7bfac5561f34ada14682c3e686731`,
`0d15d3e197c565139898687ae885779bdc0527e8ea23b18907be93e640e86ab5`
and `d3051b34c71922c92406e65b9fb4b4c9104a90d1f5db4d2c03ec156a4531cf7e`.
The code full-round replay has SHA256
`e0d4e1c2a0bcd3b2a56bc2bac13ef591dedffee9583be20fa3cb113737ee3412`.
All used Apple M3 Max CPU, the pinned FP16 target and candidate-D draft, and
the previously hashed native capture streams. No new model inference or
training was run.
