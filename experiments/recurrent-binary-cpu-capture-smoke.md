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

## One-row microbatch repeat

The same request was repeated with `--ubatch-size 1`, versus the default
`n_ubatch=512`. Both target and draft context logs reported the changed
setting. The eight output IDs and all four canonical rounds were identical.
The raw feature event file and every F32 value in the 53-by-7,680 feature
matrix were **bitwise identical**. All 16 native head-state rows and all 16
raw target-logit rows were also bitwise identical. Both internal continuity
and response-emission audits passed on the repeat. This checks this prompt
under two native microbatch configurations; it does not prove feature
values against an independent target forward or establish all possible
batching paths.

The repeat is ignored at
`results/recurrent-binary-cpu-smoke-20260928-ubatch1/`. Its feature-event,
feature-value, head-state and raw-target-logit SHA256s match the corresponding
files in the table above; `heads.f32` is
`585d15f04c9b5dd2584884b1d4d8aebf50966f6927af83a63d04fc7d9de1e982`.

## First-round CPU drafter replay

The captured raw target features, pinned F16 target embedding, four D norms
and all nine packed D linears were loaded into the recurrent CPU adapter.
The frozen prompt rebuilt 31 context cache positions; the seed and four
teacher-forced proposals then produced all five first-round head states.
The pinned model's attention projects 2,560 hidden values to 4,096 Q values;
the adapter incorrectly rejected this valid geometry before this check and
was corrected. Context rebuild now skips unused head logits while retaining
the same state and F16-rounded K/V cache.

| Arithmetic | First-depth max state difference | Fifth-depth max state difference | Fifth-depth RMS difference | Mapped top IDs matching native |
| --- | ---: | ---: | ---: | ---: |
| Native scalar order | 0.0020966 | 0.0034338 | 0.0008022 | 5/5 |
| Grouped F32 matmul | 0.0021046 | 0.0034620 | 0.0008022 | 5/5 |

The five native proposals were `[3070, 3070, 8926, 334, 32]`; both CPU
paths selected those target IDs after the frozen D vocabulary map. These
are top-one matches on one teacher-forced chain. The nonzero state drift is
similar under both arithmetic orders and remains unexplained. It may affect
other argmax decisions or training gradients; whole-drafter numeric/cache
parity and any quality claim remain unverified. No optimizer step ran.
The ignored reports are `real_round_native_order.json` (SHA256
`ff3890aaf14225c68955c178d2b520b13f8384321b5338d69172251ecf577b7a`)
and `real_round_group_matmul.json` (SHA256
`061215c1900fad504af406481f97534593dbbb8034c842b649b01d489119c239`).

## Two-position diagnostic update and native export

The audited one-prompt bundle supplied the first two proposal rows and
frozen cloned-verifier labels. One CPU grouped-matmul student unroll rebuilt
the accepted-prefix cache, kept the next two proposal steps differentiable,
and applied one SGD step (`lr=1e-5`) to exactly the nine binary sign/scale
parameter pairs. The supported-label CE was `2.4826886653900146`; all nine
linears had finite nonzero sign and scale gradients. This tests the real-size
joint gradient path; the earlier tiny causal-cache test remains the evidence
that a *later-only* loss reaches an earlier K/V/state. The two-position
diagnostic did not establish that specific real-size later-only derivative.

The resulting 839 MiB checkpoint exported through the learned-scale GGUF
serializer. The 38 MiB artifact (SHA256
`111458a74c8eed4daaef572c3f60097907aff13b3af55aa350ae00834abdd53b`)
retained the frozen map and four draft norms. Against candidate D, all nine
scale tensors changed, with **1,092,652 changed F32 scale entries**; no
packed sign word changed after this single small step. The accelerator-disabled
native server loaded all nine linears under `f32_learned_nonnegative` and A16,
served the same eight raw output IDs on the one prompt, and stopped cleanly.
That output agreement is a loader/export smoke result, not improved acceptance
or throughput. The project's Q4_0 gate and complete training budget remain
unrun.

Ignored files are under `results/recurrent-binary-cpu-smoke-20260928-a16/diagnostic-train-step/`:

| Artifact | SHA256 |
| --- | --- |
| `diagnostic_step.npz` | `caf2753f9acee468eae95af5b873e49fe8095ea9c0c9005381b144095c3cf86e` |
| `checkpoint_manifest.json` | `c896b83954c5c7c1403613f51a0395a1fbb4273e0c250d3a1e0c1cab30abd590` |
| `report.json` | `76d28ac1ae0137e078121e633a858c9395e888733beb67b880aaf34bc7c2ad63` |
| `export_audit.json` | `39a497299912090c6332b36c70b3d282a33d6116bea0e76813b2fc4898389c4e` |
| `export_delta.json` | `ea7e70763d8e34be62b90e7349631cf546fc547f405ef2e8e7448ae728080167` |
| `native-smoke/response.json` | `3abf697d9cc6ad50f4e50baf8dc392154399c623e248902c536e824e69428c3b` |

## Attention-path and independent target-feature checks

The identical CPU request was captured with `--flash-attn off` in both
contexts. It emitted the same eight raw IDs and passed continuity and
response-emission audits. Native raw feature values changed versus the
default `flash_attn=auto` capture (53-by-7,680 maximum absolute difference
`0.308945`), so these are distinct arithmetic paths. Replaying each
capture's own features with the grouped-matmul CPU drafter reduced the
first/fifth-depth head-state maximum differences from `0.002105/0.003462`
under `auto` to `0.001696/0.002143` under `off`; all five mapped top IDs
still matched. The native-order path under `off` measured
`0.001756/0.002401`. The attention path plausibly contributes to drift,
but both settings remain numerically different from the adapter.

An independent Hugging Face Qwen3 CPU forward used the local BF16 source
weights rounded to F16, then computed in F32 with eager attention. All 32
prompt embedding rows and sampled FFN gate
weight matrices at target layers 2, 18 and 33 matched the pinned GGUF
**bitwise**. The native capture's declared prefill ancestry was checked
before comparing its ordered `[2,18,33]` feature taps. The median relative
row L2 differences versus the `off` capture were 0.579%, 0.382% and
0.280% respectively; the worst row was 1.628% at layer 33. The largest
absolute difference (`64.785`) was at the first prompt position, whose
native layer-18/33 row RMS exceeded 335; its relative row error was about
0.397%. These measurements support feature identity and order, but do not
meet an exact target-feature parity gate. The Python and GGML forwards use
different arithmetic paths, and no acceptance or training decision is based
on this single prompt.

The ignored `--flash-attn off` capture is
`results/recurrent-binary-cpu-smoke-20260928-no-flash/`. Its grouped and
native-order replay reports have SHA256
`1de6ce3a7be4a147ad7779d48b1d80110811b4aa294b988bfdc9dd3bdbbb95c9`
and `2f8269635a4ff19511c2b373d7259b17044ee3342d53245e227569bdb6ca6d64`.
The independent comparison report is
`results/recurrent-binary-cpu-smoke-20260928-a16/hf_target_feature_comparison_audit.json`
(SHA256 `4820b720d0f7b1e27296760f9380ec515fc7dfd71aab062c66eff45814dca719`).
It records source hashes, Transformers 4.57.1, PyTorch 2.14.0, per-tap
absolute and relative errors, and both capture identities.
