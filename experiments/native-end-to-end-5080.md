# Matched native EAGLE comparison on RTX 5080

**Date:** 2026-09-24. **Result:** the packed W1A1 vocabulary head is slower
end-to-end than ordinary FP16 EAGLE under this fixed 5080 configuration,
although both speculative variants are faster than target-only. This is an
RTX 5080 result, not an SM75 or 2080 Ti claim.

## Fixed comparison

The supervised `native-eagle-5080-20260924` run compared target-only,
ordinary EAGLE-3, and EAGLE-3 with only its vocabulary head executed by
packed I32 signs/F32 scales and CUDA XOR/`__popc`. The target weights,
precision, KV type, target verifier, prompts, sampling, and server settings
were held fixed on one RTX 5080. The pinned target and ordinary draft used
FP16 GGUF; the packed draft kept its other weights FP16. The backend and a
one-prompt CUDA dispatch smoke passed before this run, documented in the
[integration report](ggml-w1a1-cuda-5080.md).

| Path | Target | Draft fusion/attention/FFN | Draft output head |
| --- | --- | --- | --- |
| Target-only | FP16 GGUF, F16 KV | none | none |
| Ordinary EAGLE | same target | FP16 GGUF, F16 KV | FP16 weights, ordinary matmul |
| Packed-head W1A1 | same target | FP16 GGUF, F16 KV | I32 packed signs + F32 row scales; F32 activation packing and XOR/`__popc` |

The harness used 12 fixed held-out prompts (four prose, code, and reasoning
each), five alternating variant orders, two warmup requests after each server
load, then one measured request per prompt/server/repetition: **180 measured
requests**. Greedy generation used temperature 0, seed 42, a 128-token cap,
`--spec-draft-n-max 5`, context 2048, one parallel slot, F16 target/draft KV,
and full GPU offload. Each server ran separately, with readiness and cleanup
checks. Request throughput divides generated completion tokens by client wall
time including prefill; decode throughput divides them by server
`predicted_ms`. These are pooled token/time ratios, not means of per-request
speedups. Raw requests/responses, timing/counter deltas, GPU snapshots,
commands, model/binary hashes, and server logs remain under the ignored remote
`results/native-eagle-5080-20260924/` and supervised
`runs/native-eagle-5080-20260924/` directories.
The timed run's project commit was `ef2f47b1e169ff38331af553387ba48ffe9ef1d2`
with llama.cpp gitlink `92bc70602e13214d6db94c007894261b51f5f36c`;
the later analysis used the same raw data and newer analysis script.

## Pooled results

| Variant | Completion tokens | Request tokens/s | Decode tokens/s | Request speedup vs target-only | Decode speedup vs target-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target-only FP16 | 7,440 | 96.6686 | 99.3866 | 1.000× | 1.000× |
| Ordinary FP16 EAGLE | 7,465 | 123.9626 | 132.8794 | 1.282× | 1.337× |
| Packed-head W1A1 EAGLE | 7,465 | 115.3805 | 122.9426 | 1.194× | 1.237× |

Packed W1A1 reached **0.931× ordinary request throughput** and **0.925×
ordinary decode throughput**, a 6.9% and 7.5% loss respectively. Both remain
faster than target-only in this run. The packed head saves about 153.5 MB of
draft weights, but only one drafter linear is binary and packing/kernel work
is nonzero. Every packed server log confirmed actual CUDA W1A1 dispatch;
this is native packed execution rather than PyTorch fake quantization.
The target-only ratios are timing observations rather than a clean lossless
speedup claim because both speculative paths differed from target-only on two
prompts, as detailed below. The packed-versus-ordinary negative comparison
uses identical decoded outputs across all 60 matched requests.

| Repetition | Packed/ordinary request TPS | Packed/ordinary decode TPS |
| --- | ---: | ---: |
| 0 | 0.9293× | 0.9258× |
| 1 | 0.9314× | 0.9258× |
| 2 | 0.9323× | 0.9241× |
| 3 | 0.9315× | 0.9261× |
| 4 | 0.9293× | 0.9244× |

All five repetitions favored ordinary EAGLE. A deterministic 2,000-draw
paired prompt/repetition bootstrap of the raw records gave descriptive 95%
intervals for packed/ordinary throughput of **[0.891, 0.973] request** and
**[0.884, 0.970] decode**; packed/target-only intervals were **[1.100,
1.296] request** and **[1.127, 1.357] decode**. The intervals resample
prompt IDs and repetitions while preserving all three variants in each draw;
they do not remove systematic model or hardware bias. The analysis artifact
`results/native-eagle-5080-20260924/analysis.json` has SHA256
`2c5b59b0b3de54b4acb240a955c8f78e3345804a337a2edf4c52dabab89795fb`.
It was produced by `scripts/analyze_native_benchmark.py --samples 2000
--seed 42` after the measured run completed, with no inference rerun.

| Prompt category (20 requests each) | Ordinary request TPS | Packed request TPS | Packed/ordinary request ratio | Packed/ordinary decode ratio |
| --- | ---: | ---: | ---: | ---: |
| Code | 138.959 | 122.613 | 0.882× | 0.878× |
| Prose | 103.498 | 101.893 | 0.984× | 0.983× |
| Reasoning | 133.754 | 123.041 | 0.920× | 0.907× |

All three categories favored ordinary EAGLE on both timing definitions;
the loss was smallest on prose and largest on code. Loaded GPU samples across
the 15 server instances were stable within each variant:

| Variant | GPU memory used while loaded |
| --- | ---: |
| Target-only | 11,381 MiB |
| Ordinary EAGLE | 12,417 MiB |
| Packed-head W1A1 | 12,269 MiB |

These samples include the 3,050 MiB Windows idle baseline. The packed draft
used **148 MiB less** than ordinary EAGLE, consistent with replacing the
FP16 vocabulary head with packed signs and F32 scales; the exact file-level
head saving is 153,472,000 bytes before runtime buffers/alignment. Memory
samples are point-in-time loaded readings, not a peak-allocation profile.

| Speculative variant | Accepted draft tokens | Proposed draft nodes | Verification rounds | Accepted/round | Emitted/round | Node acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary FP16 EAGLE | 3,970 | 16,845 | 3,420 | 1.16082 | 2.18275 | 23.57% |
| Packed-head W1A1 EAGLE | 3,480 | 19,185 | 3,900 | 0.89231 | 1.91410 | 18.14% |

The packed drafter accepted about 23% fewer draft tokens per round and needed
480 more verification rounds to emit the same 7,465 tokens. The server's
cumulative EAGLE timing traces, after subtracting warmup calls, were available
in all ten speculative server runs:

| Draft measurement | Ordinary EAGLE | Packed-head W1A1 |
| --- | ---: | ---: |
| Total draft-generation host time | 14.998 s | 13.707 s |
| Draft-generation time per verification round | 4.386 ms | 3.515 ms |
| Total server decode time | 56.179 s | 60.719 s |

The packed path reduced draft-generation time **19.9% per round** and saved
1.291 s of draft work across the run, but the extra rounds coincided with
4.541 s more total decode time. Subtracting draft from decode leaves an
additional roughly 5.832 s of non-draft decode work for packed W1A1; this
residual includes target verification, scheduling, and other work and is not
a direct verifier timer. The observed draft saving did not recover the lost
emissions per round. The host timing inside `common_speculative_impl` is not
the standalone CUDA kernel duration or a packing-only timer.

At this run's pooled decode rate, ordinary EAGLE spent about 16.427 ms per
verification round and packed W1A1 about 15.569 ms. Holding the packed
round-time fixed, it would need about **2.069 emitted tokens/round** instead
of 1.914 to match ordinary throughput; holding its emissions fixed, it would
need to reduce round time to about **14.405 ms** (another 1.164 ms/round).
These are same-run break-even calculations, not predictions for the 2080 Ti.

These native figures must not be compared numerically with the earlier
AngelSlim PyTorch acceptance table: that sweep used BF16 and a 59-node tree
per round, while this llama.cpp run used FP16 GGUF and at most five draft
tokens per round.

## Greedy correctness boundary

Decoded response text matched target-only on **50/60 matched requests** for
each speculative variant. Ordinary and packed EAGLE decoded texts matched
each other on **all 60/60** prompt/repetition pairs. The ten target-only
mismatches were the same two prompt IDs in all five repetitions:
`prose-04` diverged at decoded character 185 and ended with 80 target-only
versus 85 speculative tokens (`stop` in both); `reasoning-04` diverged at
character 406 and reached the 128-token `length` cap with 417 target-only
versus 423 speculative characters. Thus the ordinary EAGLE anchor itself is
not strictly text-identical to target-only on this suite, while the packed
path introduced no additional decoded-text divergences relative to ordinary.

A **separate**, supervised two-prompt diagnostic requested raw generated IDs
with `return_tokens:true, verbose:true`; it did not change or rerun the timed
comparison. Ordinary and packed token arrays matched exactly on both prompts.
Against target-only, `prose-04` first differed at 0-based generated token
index 33 (target ID 438 versus both speculative ID 264); `reasoning-04`
first differed at index 123 (target 362 versus both speculative 5737). These
IDs establish baseline-native EAGLE token divergence and no extra difference
from the packed head on the two flagged prompts. They do not identify the
underlying target-logit cause or validate stochastic distributional behavior.
The diagnostic summary SHA256 is
`1806e2f230d0a07f4957bce7a3ae8e7ba6a79ee4b6b982fe24dec5c585043bf6`;
its 50-file artifact seal SHA256 is
`bcbf739dfbd4409059ff333bf36977f2a70d02d217193a568fe028c68165ad68`.

One further bounded `n_probs:5` check reproduced the same target-only and
ordinary token IDs. At the first `prose-04` divergence, target-only ranked
its winning ID 438 at logprob −0.6927611 and speculative ID 264 second at
−0.6937244, a **0.0009633-nat** gap. At `reasoning-04`, target-only ranked
winning ID 362 at −0.6957814 and speculative ID 5737 second at −0.7122894,
a **0.0165080-nat** gap. These small target-only margins make numerical
sensitivity plausible. The ordinary EAGLE API returned placeholder logprob
zero and an empty top-candidate list at the accepted-draft positions, so the
verifier's corresponding logits were not available; this diagnostic cannot
prove that rounding, rather than a cache/verification issue, caused the
native mismatch. Its summary SHA256 is
`cba59b6090bfacdf5713482f1b134ace97b2583ab195d521031008912696e0fe`
and artifact manifest SHA256 is
`693415914997891220807069260c9f6df38e8d48f3a984ea7bb0661b5d27b723`.
No full benchmark was rerun or altered for this check. After the diagnostic,
three GPU samples were idle at 3,047 MiB used / 12,931 MiB free / 0%; no
project process or SSH/tmux session remained.
The benchmark otherwise had no error or OOM. All completed requests were
128 tokens except five target-only 80-token stops and five 85-token stops
for each speculative variant; these actual lengths are included in the pooled
rates rather than replaced with nominal caps.

This is a quantified negative **relative to ordinary EAGLE on the RTX 5080**
for head-only W1A1 with the original published draft head. It does not rule
out other coverage or retraining recipes, and it does not establish a Turing
Tensor Core result. The bounded head-only QAT pilot was stopped after its
separate held-out gate failed to recover acceptance; see its
[report](qat-head-pilot-results.md). A focused SM75 binary-MMA probe is
cross-compiled with BMMA SASS and passed 880 integer-dot checks on a 5080
proxy path, but awaits actual RTX 2080 Ti execution and
measurements; see the [probe record](sm75-binary-mma-plan.md) and
[2080 Ti runbook](../docs/RTX2080TI_RUNBOOK.md).

The sealed raw run hashes are: `manifest.json`
`86f5cd6b931132a41e4cf4132a0f490aefc13baa0753c66e987af458b0c9e24b`,
`report.json`
`7986e79b409e5565d6323f69ee183f42dd1aff0b291ee48488c4447cf4aab27e`,
and `records.json`
`463b5505561af566691ed1c552a022a609b3a914c97bb1e303cc4baf88193d9e`.
The tested CUDA `llama-server` binary SHA256 is
`0494fe48e653fb950a92301674b77ecd1641ac6c6b59eb7ee2aae9b82981a33c`;
the target/ordinary/packed GGUF SHA256 values are respectively
`05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`,
`c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1`,
and `b2095130b5196574a9a08a88d2fb9a32ac1ea870ff3cf587ae7d6e7f64e7819f`.
The supervisor and all server processes exited; the GPU returned to its
3,050 MiB/0% baseline before the separate correctness diagnostic.
