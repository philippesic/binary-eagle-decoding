# W1A1 quality versus draft-cost screen

**Date:** 2026-09-24. This is a deliberately optimistic arithmetic screen,
not an end-to-end throughput measurement. It combines the fixed 12-prompt
RTX 5080 AngelSlim BF16 verifier-relative accepted/round counts from
[the acceptance report](pytorch-w1a1-cuda-acceptance.md) with one-prompt
instrumented draft-time shares from that same report. The trained-head point
comes from [the bounded QAT pilot](qat-head-pilot-results.md).

If each speculative round emits its accepted draft tokens plus one target
token, a candidate's emitted-token ratio to ordinary EAGLE is
`(1 + candidate accepted/round) / (1 + 2.31659)`. If its selected linears
could be made completely free while all other measured draft work stayed
fixed, the most favorable draft speedup is `1 / (1 - selected share)`. The
product is an optimistic throughput ratio in a hypothetical system with no
target verification or other overhead:

| W1A1 coverage | Accepted/round | Profiled draft-time share | Emitted-token ratio | Zero-cost draft speedup | Optimistic ratio to ordinary |
| --- | ---: | ---: | ---: | ---: | ---: |
| Feature fusion | 0.683 | 1.03% | 0.507 | 1.010× | 0.513× |
| Attention | 0.771 | 10.08% | 0.534 | 1.112× | 0.594× |
| FFN | 1.067 | 14.29% | 0.623 | 1.167× | 0.727× |
| Vocabulary head | 1.677 | 12.93% | 0.807 | 1.149× | 0.927× |
| All nine candidate linears | 0.202 | 38.32% | 0.362 | 1.621× | 0.588× |
| Trained vocabulary head | 1.565 | 12.93% | 0.773 | 1.149× | 0.888× |

No observed BF16-verifier-relative W1A1 variant clears 1.0 even in this
zero-cost draft-only calculation. The trained head worsens the quality/cost
screen despite improved validation KL. This argues against expanding the
current fake-binary rule to more linears or tuning this QAT recipe on the
held-out prompts.

The calculation is not a bound on actual native throughput: layer shares are
from one instrumented prompt, the native GGUF path uses F16 upstream and F32
packed-head arithmetic rather than the BF16 PyTorch simulation, acceptance
streams and proposal topology can differ (the PyTorch sweep uses a 59-node
tree per round, while the planned llama.cpp server uses
`--spec-draft-n-max 5`), and actual packing/kernel costs are nonzero. It also
omits the target/verifier cost, which would ordinarily reduce the benefit of
draft-only acceleration. The decisive test remains a matched same-device
target-only/ordinary/native run with real accepted/emitted tokens and wall
time. The 2080 Ti must be measured separately before any SM75 claim.
