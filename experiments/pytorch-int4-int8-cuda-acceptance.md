# PyTorch INT4 and INT8 drafter acceptance on RTX 5080

**Status:** fixed 12-prompt held-out sweep and artifact audit complete. This
experiment measures post-training W4A4 and W8A8 numerical simulations; it
does not execute native INT4/INT8 kernels or measure inference speed.

## Matched setup and quantization rule

- Target `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` and
  drafter `AngelSlim/Qwen3-4B_eagle3@fd331e59626c8e95c392381a16ee59d518727fbb`.
  All 18 snapshot files match the SHA256 hashes used in the prior W1A1 CUDA
  experiment. The AngelSlim source commit remains
  `0358da9c651e6a7d7ccafea26ced4b9c98d11681`.
- Remote project code `92c91a4cecdd0123a2e1d7bf7c6d394fa673dabe`, clean
  checkout. Config `configs/pytorch_int4_int8.toml` SHA256
  `7ac30ee431e9b2583598ab68cb28731daca6edb5e453ebd1c300578621752c3a`;
  remote model manifest `results/model-manifest-int4-int8-20260924.json`
  SHA256 `872cc50776ce1e5adb9494e1822cadaec37fbc81338dd1a7b455057907c1f807`.
  The 12-prompt manifest SHA256 remains
  `0d6a698d6816592c6ff435fed2fea4cdafe9f5248393d2a5ac091e1551919476`.
- Greedy Qwen3 non-thinking chat template, target and verifier unchanged,
  `max_new_tokens=128`, 60-token EAGLE tree budget (59 proposed draft nodes
  per round), depth 5, top-k 10. Ordinary BF16 EAGLE is rerun in the same
  process as both low-bit variants.
- W4A4 and W8A8 cover the nine drafter-owned feature-fusion, attention, FFN,
  and vocabulary-head linears. Weights use symmetric signed codes `[-7,7]`
  or `[-127,127]` with one absmax scale per output row. Activations use the
  same code range with one absmax scale per input token vector. Rounding is
  nearest-even; zero vectors produce zero codes. PyTorch performs FP32
  `F.linear` on integer-valued floats, applies FP32 scales/bias, and casts
  output to the drafter dtype (BF16). TF32 is disabled and FP32 matmul precision
  is `highest`. Embeddings, normalization, softmax, residuals, and target
  verification retain their ordinary precision. This simulation is not a
  bit-packed kernel or exact INT32 accumulator.

## Parity limitation and one-prompt smoke

The prior strict CUDA check found ordinary BF16 EAGLE diverges from target-only
greedy output at generated token 4 due a target-selected verifier-logit tie;
the tree-versus-prefix logit shift remains unexplained. The new run records
target-greedy mismatches and continues only as an explicitly exploratory,
same-verifier acceptance comparison. It cannot establish target-equivalent
output or a clean quantization-only quality effect.

The supervised `cuda-int4-int8-smoke-20260924` run finished with exit 0 on
`prose-01`. The disabled quantizer wrapper matched ordinary EAGLE exactly.
All three variants recorded the known first target-greedy mismatch at index 3.

| Variant | Accepted / proposed draft nodes | Rounds | Accepted/round |
| --- | ---: | ---: | ---: |
| Ordinary BF16 EAGLE | 84 / 2,773 | 47 | 1.787 |
| W4A4 simulation | 25 / 6,195 | 105 | 0.238 |
| W8A8 simulation | 80 / 3,068 | 52 | 1.538 |

The one-prompt result checked the path and gave a preliminary quality signal.

## Full held-out accepted drafts

The supervised `cuda-int4-int8-12prompt-20260924` run completed all 12 prompts
for ordinary BF16 EAGLE, W4A4, and W8A8 under the same verifier. It produced
36 unique variant/prompt acceptance rows, 12 target-reference rows, and 19
greedy-mismatch records. Each round proposed 59 draft tree nodes. The primary
metric below is total accepted draft tokens divided by total verification
rounds, so prompts with more rounds contribute proportionally.

| Drafter setting | Accepted / proposed nodes | Rounds | Accepted/round | Accepted/proposed nodes | % of ordinary | Target-greedy matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary BF16 EAGLE | 1,061 / 27,022 | 458 | 2.3166 | 3.926% | 100% | 5/12 |
| W4A4 simulation | 338 / 69,207 | 1,173 | 0.2882 | 0.488% | 12.4% | 6/12 |
| W8A8 simulation | 1,045 / 28,261 | 479 | 2.1816 | 3.698% | 94.2% | 6/12 |

W8A8 preserved nearly all observed ordinary EAGLE acceptance on this suite.
W4A4 lost most accepted drafts, though it was above the prior all-group W1A1
result of 0.2022 accepted/round under the same pinned BF16 verifier. These are
acceptance measurements, not kernel timing or end-to-end speed. Node acceptance
ratios are small because each round proposes 59 branching tree nodes and only
one path can be accepted. The ordinary baseline exactly reproduced the prior
W1A1 CUDA run's 1,061 accepted drafts over 458 rounds.

The known BF16 target-only versus ordinary EAGLE mismatch remains. Ordinary
matched target-greedy output on 5/12 prompts; W4A4 and W8A8 matched on 6/12
each. Raw outputs preserve all 19 mismatch records and each prompt's accepted
counts. These rates are **exploratory verifier-relative evidence**; they do not
prove target-equivalent decoding or isolate quantization as the sole cause of
every output difference.

Prompt-level accepted-per-round values vary by content. The median and range
below are across 12 prompts; category values pool accepted drafts and rounds
within four prose, four code, or four reasoning prompts. There was one greedy
run per prompt/variant, so these show prompt variation rather than confidence
intervals.

| Setting | Prompt median (range) | Prose | Code | Reasoning | Zero-accept rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary BF16 | 2.467 (1.729–3.000) | 1.856 | 2.568 | 2.593 | 9.2% |
| W4A4 | 0.328 (0.112–0.449) | 0.193 | 0.315 | 0.360 | 75.5% |
| W8A8 | 2.263 (1.538–2.694) | 1.747 | 2.390 | 2.470 | 10.9% |

Generated output lengths ranged from 85 to 133 tokens for ordinary, 87 to
130 for W4A4, and 85 to 133 for W8A8 because generation can stop at a
verification-round boundary. The per-prompt raw rows preserve actual lengths,
token IDs, accepted counts, and mismatch flags.

## Raw artifacts and replay

The fixed 12-prompt run used this exact command under the configured remote
workdir:

```sh
python3 scripts/remote_job.py cuda-int4-int8-12prompt-20260924 -- \
  .venv/bin/python scripts/evaluate_pytorch_w1a1.py \
  --config configs/pytorch_int4_int8.toml \
  --model-manifest results/model-manifest-int4-int8-20260924.json \
  --run-id cuda-int4-int8-12prompt-20260924 \
  --variants ordinary int4_w4a4 int8_w8a8 \
  --allow-greedy-mismatch
```

The code revision was `92c91a4cecdd0123a2e1d7bf7c6d394fa673dabe`.
Raw files remain outside Git under remote
`results/cuda-int4-int8-12prompt-20260924/`, with supervisor state and log
under `runs/cuda-int4-int8-12prompt-20260924/`. SHA256:

| Artifact | SHA256 |
| --- | --- |
| `acceptance.jsonl` | `e95e658068e2c9d6c6dff53c6ea50cd72a2daf5783888260aaa1439cd50a02ea` |
| `summary.json` | `669a560264b66f4c7403ee9b22d06d0408d7002daa7d40091f62f0bbefbd20b7` |
| `target-reference.jsonl` | `7cafdfff4e7bf728264021c4ac65f571419648637121fb73da9cb3426cafe0c8` |
| `greedy-mismatches.jsonl` | `d5aed64bf1e2729a64bfc1836d4f8979cf67ee18a9d7645bc97bf34673a8ff13` |
| `greedy-parity.json` | `7e9d7694774349163d96c8080cdd120158e7209d57fcd0a19201e296eaa49742` |
| `derived-metrics.json` | `24fe557e02b3e46bd9fb791f6d9f90f75bbc84b2b1f4893cbae74f343d682ee4` |
| `environment.json` | `c8ea7026b493b411208d4e741a272934d3fc796a2642531edc2399f6d77876fe` |
| supervisor `state.json` | `6a3bbd2e83e11820f3a5261537189ac5612247f13c017f11c524b9561d69df35` |
| supervisor `stdout.log` | `b741d06270181873280775c762fcbde044ad12ed49fecc90cd0a5716293b6de3` |

The remote environment used RTX 5080 / Windows driver 616.92 / CUDA UMD
13.4, Python 3.11.15, PyTorch 2.14.0+cu130 (CUDA build 13.0), Transformers
4.57.6, and AngelSlim at the pinned commit. The package freeze from the prior
matched CUDA setup is SHA256
`d0027ce02dd9c4929338edecd66c58e2258da0bf3056d5c62713381d94db619e`.
The supervisor finished with exit 0. Its PID/PGID 866 and all project
evaluator processes were absent afterward; three Windows samples showed 0%
GPU use and 3,050 MiB used / 12,928 MiB free. Both tmux SSH connections were
closed. The ignored local command and cleanup record is
`results/preflight-5080-20260924T034000Z/host-preflight.txt`, SHA256
`505280ce549ca001d0c63a70be054b3a59a7ca5748a22cc7658b43b565506f0e`.

## Interpretation for later hardware benchmarks

W8A8 retained 94.2% of ordinary BF16's accepted drafts per round under this
simulated verifier path. W4A4 kept 12.4%, and zero-accept rounds rose to
75.5%. The earlier all-group W1A1 simulation kept
8.7% under the same BF16 verifier. This is evidence about draft acceptance,
not real INT4/INT8 throughput. Before using these numbers to predict or
compare a native kernel, match its signed range, scales, rounding, accumulation,
output casting, and layer coverage to this run, then validate numerical
agreement and measure end-to-end timing on the same hardware. A weight-only
INT4/INT8 kernel would require a separate W4A16/W8A16 acceptance measurement.
