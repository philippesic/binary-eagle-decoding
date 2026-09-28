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
