# Native EAGLE target-verifier logit trace on RTX 5080

**Date:** 2026-09-24. **Scope:** two fixed ordinary-EAGLE prompts that
diverged from target-only in the sealed [end-to-end
comparison](native-end-to-end-5080.md). This is a separate diagnostic, not a
timed benchmark or a W1A1 code change.

The ordinary and packed native variants produced identical decoded text on
all 60 matched benchmark requests. Both differed from target-only on
`prose-04` and `reasoning-04`. Earlier token-ID and `n_probs:5` diagnostics
showed the target-only output favored IDs 438 and 362 at zero-based generated
positions 33 and 123, while both speculative variants emitted 264 and 5737.
The target-only winning/runner-up log-probability gaps were 0.0009633 and
0.0165080 nats. The normal server API omits the verifier's top candidates at
accepted-draft positions.

An isolated opt-in llama.cpp diagnostic commit
`b06892b697cba17a1e47ec3fc770716a91a8e699` (fork branch
`feat/verify-logit-trace`, based on pinned `92bc706`) logged raw target
verifier top-two logits from `llama_get_logits_ith` immediately before
`common_sampler_sample_and_accept_n`. It scanned only generated positions
33 and 123 with `W1A1_TRACE_VERIFY_LOGITS=1`, then recorded sampled/draft
IDs and whether the row was emitted. No sampler state or logit array was
modified. The pinned parent gitlink and measured CUDA server binary were
unchanged; a separate diagnostic server build ran the two prompts once each
with the original FP16 target/draft, greedy seed, template, context and
draft length. Both generated ID arrays reproduced the sealed ordinary-EAGLE
baseline exactly.

| Prompt, generated position | Emitted row | Draft ID | Verifier choice | Raw target top 1 | Raw target top 2 | Margin |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `prose-04`, 33 | 2 | 54112 | 264, rejected | 264: 28.9808998 | 438: 28.9728260 | 0.0080738 |
| `reasoning-04`, 123 | 0 | 362 | 5737, rejected | 5737: 31.7031441 | 362: 31.7024155 | 0.0007286 |

Six additional scanned rows were not emitted because an earlier verifier
row rejected the draft; their raw logits are preserved but cannot explain
the generated divergence. No scanned row had `logits=null`. At each emitted
row, the ordinary target verifier's raw argmax was the token it emitted.
Thus the immediate mismatch is **not a wrongly accepted draft**: the batched
target verification path and the target-only incremental path choose a
different argmax at the same generated position. Both candidate pairs are
close in target-only, and the verifier's own gaps are small. Batch-dependent
numeric sensitivity is plausible; this trace alone does not rule out a
target KV/cache or position-state difference, so strict target-equivalent
decoding remains unproven. Because ordinary and packed native outputs matched,
this limitation belongs to the pinned ordinary EAGLE baseline, not a new
W1A1-specific acceptance failure.

The build and two-prompt run were supervised separately on the 5080; both
exited zero. The raw response, server log, summary, exact commands,
environment, and file hashes are sealed in remote
`/home/philip/binary-eagle-decoding/results/verify-logit-trace-20260924/`.
Its `artifact-manifest.json` SHA256 is
`9e64ede2a31c50417b843d5245ab45aff206453f1df61eee36bd001835aac898`.
The remote llama.cpp checkout was restored clean to pinned
`92bc70602e13214d6db94c007894261b51f5f36c`; no project processes or
tmux session remained. Repeated final GPU samples showed 3,046 MiB used,
12,932 MiB free and 0% utilization. No timed result was rerun.
