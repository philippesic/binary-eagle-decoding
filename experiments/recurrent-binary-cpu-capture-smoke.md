# Recurrent binary capture: one real CPU request

**2026-09-28 UTC.** This is an instrumented capture-path diagnostic, not a
training, acceptance or throughput experiment. The frozen training prompt was
`qat-revisit-train-prose-urban-waterways-01`, capped at eight output tokens.
The target was the pinned FP16 Qwen3-4B GGUF (SHA256
`05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`)
and the own-history draft was candidate D (SHA256
`10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf`).
The absolute D map had canonical I64-content SHA256
`03d2f0e3420175955c14a43a1c1dda360cde26c8a03318ffb1a013bb06d0fe0a`.

The run used the forked llama.cpp revision `ddcf2a608` on the Apple M3 Max
CPU (`arm64`). Its CMake cache has CUDA, Metal, Vulkan, SYCL, HIP, RPC,
Accelerate and BLAS **OFF**. Target and draft GPU layer counts were both zero;
`CUDA_VISIBLE_DEVICES` was empty. The local `llama-server` binary SHA256 was
`69655583e268a30e5477192ec7b48adba519ec2b424d8ac9577a871a44250d75`.
The server's process group stopped with return code zero. No GPU, accelerator,
remote host, final prompt or training step was used.

The first launch exposed a real runner defect: candidate D's v2 loader
requires `GGML_W1AX_ACT_BITS=16`, but the new `recurrent-train` mode omitted it.
It exited before serving. Setting that variable produced the successful CPU
capture; the runner and its mock test now set/check it.

| Captured item | Count or result |
| --- | ---: |
| Prompt input tokens | 32 |
| Response output tokens | 8 |
| Native proposal rounds | 4 |
| Head state/target-logit rows | 16 / 16 |
| Raw target-feature rows | 53: 32 prefill, 20 speculative, 1 target-only |
| Prepared accepted-prefix feature rows | 36 |
| Accepted draft depths by round | 0, 0, 1, 1 |

The internal continuity auditor passed all four rounds. Both native
preparers accepted the raw streams, and the final capture audit accepted a
self-contained bundle with `training_eligible: false`. The capture audit
reported 16 valid supported proposal rows, six live reached verifier rows,
and mean mapped target probability mass `0.999897388061748` across the 16
sampled raw target-logit rows. That mass is for this single prompt and cannot
be used to choose a training loss or claim population coverage.

The recorded seed plus each round's emitted verifier IDs equal the first
seven response IDs: `[32, 3283, 38921, 646, 1824, 2176, 7557]`. The eighth
response ID was `323` from the terminal target-only path. The response reported
`finish_reason=length` and eight completion tokens. The fifth native round
trace has `status=no_proposal` and `emitted_token_ids=[323]`; a separate
CPU response audit joined all eight response IDs to the four canonical
rounds, that terminal trace, request cap and complete prefill. There is no
independent raw target-logit row for that terminal sample, nor a numerical
target-feature reference. Full request sampling correctness, whole-drafter numeric/KV parity,
trained-export loading, native Q4_0 acceptance and timing remain open.

Ignored evidence directory:
`results/recurrent-binary-cpu-smoke-20260928-a16/`. Key SHA256s:

| File | SHA256 |
| --- | --- |
| `forced-rounds.jsonl` | `1847aa3fc6e0364a12ab88c965c75d91d54eef76e51a4baba8d79f65333bed78` |
| `rounds.jsonl` | `f7a81beba2e94dd99b3ae8c368eea60697730f925c11c14809c861bfa8ad2133` |
| `heads.jsonl` | `14f3da6379b7888a223bc1abb0126edd5a088764adf37dff229259e492fe3d60` |
| `heads.target_features.jsonl` | `6ea22d4c62a3bcfbb32194e3eb2d49b1cbd379c693be9a202fd72d6c8caf6a17` |
| `heads.target_features.f32` | `f5c4189cae56f9f6470e44b1ddd285788c4b95d8974dd261c323bd610aa793dc` |
| `heads.target_logits.f32` | `8ff85286da40171bd91f352d1ff34584d6761a86cfd2bd4b38418d29f42f5c1d` |
| `response.json` | `fcb1cd319a93574bb2fc9e109d074a95aeacbf82b9fdd8727d02b959304cf4c5` |
| `continuity.json` | `4370502acd45c76947f7cc6752d1b99d3967d5f3023347d1ec3a8a362ac3e7de` |
| `bundle/manifest.json` | `52a6f718922c15c46c7bfaa2a2754795322652f7a7ee0d975a21af19c2f51649` |
| `bundle/audit.json` | `58fdb01e7d1a917a05f384957051fe51874ffc16dbb25c5ad83e6b77eeefe369` |
| `response_round_join.json` | `60548c5ccfcfe50249cf3ed5a9b31e664e1d44efb80e18483c86afe36d8222e1` |

This one-request diagnostic used an explicit one-prompt audit override. The
future frozen training capture still requires all 96 prompts, the full
request ownership manifest, and every later parity and quality gate.
