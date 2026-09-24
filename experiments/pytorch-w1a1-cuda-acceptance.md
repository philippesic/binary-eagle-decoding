# PyTorch W1A1 acceptance: RTX 5080 CUDA confirmation

**Status:** strict parity failed; the full 12-prompt exploratory acceptance
sweep and ordinary drafter profile are complete. Results are verifier-relative
and are not a clean target-equivalent comparison.
This is a BF16 PyTorch fake-binary acceptance experiment, not native one-bit
execution or an end-to-end throughput comparison.

## Fixed inputs and environment

- Remote project code: strict parity and profiling at
  `daad1571f2f2be8404912c66cc19754e9f0daa00`; the exploratory runner at
  `ae471198911fe7e093a4e26ff95655105843b725`. Both checkouts were clean
  for their runs. AngelSlim source:
  `0358da9c651e6a7d7ccafea26ced4b9c98d11681`.
- Target: `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`,
  13 files / 8,060,926,626 bytes. Drafter:
  `AngelSlim/Qwen3-4B_eagle3@fd331e59626c8e95c392381a16ee59d518727fbb`,
  5 files / 436,987,505 bytes. Every remote file SHA256 matched the prior
  local reference manifest.
- Remote manifest: `results/model-manifest-5080-20260924.json`, SHA256
  `2db1c860f059bd8702ca6bb523e64f0c7d064c7a95f59919a001fc88168a91e3`.
  Resolved config SHA256:
  `41e4509a5a444357b9c6c3079569ea9649a6ce9a34752b0d5908913d9062d6a2`.
  Prompt manifest SHA256:
  `0d6a698d6816592c6ff435fed2fea4cdafe9f5248393d2a5ac091e1551919476`.
- Hardware: RTX 5080, 16,303 MiB reported; WSL NVIDIA-SMI 615.71.08,
  Windows driver/KMD 616.92, CUDA UMD 13.4. Python 3.11.15,
  PyTorch 2.14.0+cu130, Transformers 4.57.6, Hugging Face Hub 0.36.2,
  Accelerate 1.15.0, `datasets` 5.0.1, `shortuuid` 1.0.13, Pillow 12.3.0,
  and the pinned AngelSlim commit. The CUDA runtime package is 13.0.96,
  cuDNN 9.24.0.43, cuBLAS 13.1.1.3, and Triton 3.8.0. Target and drafter
  were BF16. No CUDA compiler or native kernel was invoked in this phase.
- Greedy non-thinking prompts and the 60-token EAGLE tree settings are fixed
  in `configs/pytorch_w1a1.toml`. The sweep included ordinary EAGLE
  and five W1A1 layer-group variants over the same 12 held-out prompts.
  Fake W1A1 uses `sign(0)=+1`, mean-absolute scales per output weight row and
  per input token vector; only selected drafter linear groups are quantized.

The pinned environment and model snapshots were prepared under supervised
CPU-only runs before GPU work. Exact setup commands and run states are preserved
in ignored remote `runs/` and the local
`results/preflight-5080-20260924T034000Z/host-preflight.txt` record (SHA256
`394838155567ba0e463df47bd9bdecf7b98349c2341aba04092f6557229303a9`).
The RTX 5080 was initially occupied by a Windows game; CUDA work began only
after repeated idle samples and a process check found the game closed. Idle
overlays still reserved about 3,107 MiB.

## CUDA parity gate

The first supervised smoke stopped before model loading because AngelSlim's
import path required `datasets`. After that dependency and two additional
import-path requirements were installed, the supervised
`cuda-parity-smoke3-20260924` run loaded the pinned BF16 models and performed
the strict ordinary-EAGLE versus target-only greedy check on `prose-01`.

It exited 1 at generated index 3 (the fourth token). The first three generated
IDs agreed: `[785, 7406, 41506]`. Target-only next produced ID 272; ordinary
EAGLE produced ID 11. The raw remote file is
`results/cuda-parity-smoke3-20260924/greedy-parity.json`, SHA256
`592278d6ca4d1f23c3890f50cecd0cd0473de0f985d07c75f7e28a9c2c8c53a4`.
The bounded BF16 trace shows that token 11 was a **target-selected seed** after
two accepted draft positions, not a wrongly accepted draft. The verifier tree
row at absolute position 40 tied token IDs 11, 272, and 7578 at logit 21.0;
`argmax` selected the lowest ID, 11. A full-prefix target pass on the same
41-token prefix also evaluated position 40, but ranked ID 272 at 21.0 and IDs
11 and 7578 at 20.875. It selected 272. Both passes used the pinned target in
BF16. Thus the immediate token-choice mechanism is known; the cause of the
tree/cache versus full-prefix logit shift is still unresolved. A mask/cache
path difference is possible but not established. The trace is at remote
`results/cuda-parity-diagnostic-20260924/trace.json` (SHA256
`05563bd35b5f502c77a0a4485779e02bf74e8e07bda25d6f0a17341ac09a65ac`).
Its exact diagnostic script has SHA256
`7d148d4714e85962ca387053357d9592866c5b52c3aa7a77a04cf02d02f53963`.
The diagnostic supervisor finished with exit 0 and released its GPU memory.
This divergence occurs earlier than the Metal BF16 divergence at index 20.

The disabled W1A1 wrapper, all five W1A1 variants, and the full held-out sweep
were **not measured** by the strict run because the ordinary baseline failed
the parity gate. The supervised process exited and GPU memory returned to the
idle baseline.

## Ordinary drafter layer-cost diagnostic

A separate supervised ordinary EAGLE profile on `prose-01` used a 38-token
prompt, a 32-token generation cap, two warmups, and five measured repetitions.
It recorded 70 `topK_genrate` draft invocations. CUDA events on the current
stream bracketed each draft invocation and each of the nine eligible linear
calls; the instrumented intervals were synchronized before reduction. The
aggregate draft-event time was 628.69 ms. Candidate linears accounted for
240.95 ms (38.32%); the residual was 387.75 ms (61.68%).

| Group | Linear calls | Event ms | Share of draft-event time |
| --- | ---: | ---: | ---: |
| Feature fusion | 70 | 6.475 | 1.03% |
| Attention Q/K/V/O | 1,680 | 63.345 | 10.08% |
| FFN gate/up/down | 1,260 | 89.846 | 14.29% |
| Drafter vocabulary head | 420 | 81.280 | 12.93% |
| Remaining graph and instrumentation | — | 387.75 | 61.68% |

The five measured draft-event totals were 126.349, 124.037, 126.925,
125.227, and 126.155 ms (range 124.037–126.925). Common observed BF16
inputs included fusion `[1,3,7680]` into `(2560,7680)`, attention Q
`[1,10,5120]` into `(4096,5120)` with K/V using `(1024,5120)`, FFN gate/up
`[1,10,2560]` into `(9728,2560)`, and head `[10,2560]` into `(32000,2560)`.
The raw profile lists all observed sequence lengths and weight shapes, including
the attention output and FFN down projections. These nine linears are the
binary-eligible candidates; embedding lookup, RMSNorm, RoPE, attention
softmax, residuals, tree selection, and target verification remain ordinary
operations.

Peak PyTorch allocation for the profiled generation was 9,781,691,904 bytes.
The remote artifact is
`results/cuda-drafter-profile-20260924/layer-profile.json`, SHA256
`1677a3dd0b0cce78cec1e7296c6b18a30ed51e5e780fcd53509b8faed473c15f`.
Its supervisor finished with exit 0, no project process remained, and three
post-run GPU samples returned to 0% use and the 3,107 MiB Windows idle memory
baseline.
These are one-prompt, event-instrumented draft timings. The residual includes
non-linear graph work, tree selection, launch gaps, and instrumentation cost;
the shares are not native binary-kernel savings or end-to-end speedups.

As a deliberately optimistic planning model, hold all other measured work
fixed and reduce a selected group's time to zero. The resulting draft-event
speedup is `1 / (1 - group_share)`: fusion 1.010×, attention 1.112×, FFN
1.167×, head 1.148×, or all listed linears 1.621×. Real W1A1 execution retains
packing, scales, and kernel time, and end-to-end inference also includes target
verification. Acceptance changes can outweigh these draft-time savings.

## Exploratory runner validation

The strict smoke cannot establish a CUDA W1A1 acceptance rate or a clean
quantization-induced acceptance loss. The immediate mismatch mechanism is
established,
but the upstream logit shift remains a blocker to clean target-equivalent
acceptance interpretation. An opt-in, mismatch-recording diagnostic mode was
checked on one prompt under supervised
`cuda-exploratory-ordinary-fusion-20260924` at code commit
`ae471198911fe7e093a4e26ff95655105843b725`. The disabled wrapper matched
ordinary EAGLE exactly; both generated outputs were flagged as mismatching
target-only greedy output. Observed verifier acceptance was:

| Variant | Accepted drafts | Proposed tree nodes | Rounds | Accepted/round |
| --- | ---: | ---: | ---: | ---: |
| Ordinary BF16 EAGLE | 84 | 2,773 | 47 | 1.787 |
| Fusion-only W1A1 simulation | 42 | 5,133 | 87 | 0.483 |

The run finished exit 0 and all project processes stopped. It preserved
`acceptance.jsonl` (SHA256
`517a38441c8a8d7501bfdb6e5aa18b3a51752de1ea5234a1b3a29fb6117d01fe`),
`greedy-mismatches.jsonl` (SHA256
`6c8dd7ba2365f1b5ac3570422a7ae976b52d1861aeb691b2d87d376becd8361e`),
and `summary.json` (SHA256
`c62aeaacb4a666b92c8a97238fe848d6219b196f8cbf87b7888ef301bcde4bda`)
under remote `results/cuda-exploratory-ordinary-fusion-20260924/`. This is one
prompt and two variants, so it cannot replace the fixed 12-prompt comparison.

## Full held-out exploratory acceptance

The supervised `cuda-exploratory-12prompt-20260924` run used the same pinned
CUDA code, models, prompt manifest, and `--allow-greedy-mismatch` policy. It
finished with exit 0 and produced 72 unique variant/prompt rows: 12 prompts
each for ordinary EAGLE and five W1A1 layer-group settings (four prose, four
code, four reasoning prompts). All 12 target-reference rows were preserved.
There were 43 target-greedy mismatch records across the 72 rows. The table
uses total accepted drafts divided by total verification rounds; each round
proposed 59 tree nodes. Percent of ordinary uses the ordinary variant's
accepted-per-round value on this same run. Actual generated outputs ranged
from 85 to 133 tokens because stopping occurred at verification-round
boundaries; each raw row preserves its output length and token IDs.

| Drafter setting | Accepted / proposed nodes | Rounds | Accepted/round | Node acceptance | % of ordinary | Target-greedy matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary BF16 EAGLE | 1,061 / 27,022 | 458 | 2.317 | 3.93% | 100% | 5/12 |
| W1A1 feature fusion | 616 / 53,218 | 902 | 0.683 | 1.16% | 29.5% | 5/12 |
| W1A1 attention | 656 / 50,209 | 851 | 0.771 | 1.31% | 33.3% | 5/12 |
| W1A1 FFN | 783 / 43,306 | 734 | 1.067 | 1.81% | 46.0% | 4/12 |
| W1A1 vocabulary head | 949 / 33,394 | 566 | 1.677 | 2.84% | 72.4% | 4/12 |
| W1A1 all listed groups | 256 / 74,694 | 1,266 | 0.202 | 0.34% | 8.7% | 6/12 |

The five CUDA W1A1 accepted-per-round values closely reproduce the Metal
development pattern: fusion 0.683 versus 0.712, attention 0.771 versus 0.773,
FFN 1.067 versus 1.057, head 1.677 versus 1.683, and all groups 0.202 versus
0.202. Ordinary EAGLE was not measured across the Metal prompt suite, so the
same-device ordinary baseline here is new. The combined post-training W1A1
setting retained only 8.7% of ordinary EAGLE's accepted drafts per round under
this verifier. Head-only retained the most acceptance among the tested W1A1
groups. These ratios describe accepted drafts, not end-to-end speed: draft
latency, target verification, emitted target tokens, and packing costs also
matter.

Prompt variation is substantial, and prose has lower acceptance than code or
reasoning in this suite. The median and range below are across 12 prompt-level
accepted-per-round values; category values pool accepted tokens and rounds
within each four-prompt category. There was one deterministic greedy run per
prompt/variant, so these ranges describe prompt variation, not confidence
intervals or repetition variance.

| Setting | Prompt median (range) | Prose | Code | Reasoning | Zero-accept rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary | 2.467 (1.729–3.000) | 1.856 | 2.568 | 2.593 | 9.2% |
| Fusion | 0.700 (0.483–0.870) | 0.600 | 0.702 | 0.747 | 55.7% |
| Attention | 0.792 (0.554–0.955) | 0.699 | 0.789 | 0.823 | 38.2% |
| FFN | 1.055 (0.724–1.339) | 0.868 | 1.093 | 1.262 | 29.3% |
| Head | 1.758 (1.224–2.095) | 1.319 | 1.821 | 1.938 | 15.5% |
| All | 0.217 (0.121–0.303) | 0.129 | 0.231 | 0.249 | 81.3% |

Every variant has some prompts that differ from target-only greedy output.
The BF16 tree-versus-prefix logit shift is unresolved, and generated paths can
diverge across variants. Therefore these are **observed acceptance counts under
one verifier**, not a clean causal estimate of quantization-induced acceptance
loss or evidence of target-equivalent speculative decoding. The prior Metal
development counts are in `experiments/pytorch-w1a1-metal-acceptance.md`.
No W8A8 or W4A4 acceptance track was measured here; "ordinary" means the
unmodified BF16 EAGLE drafter.

## Decision implications

The combined post-training W1A1 setting loses 91.3% of ordinary EAGLE's
accepted drafts per round under this verifier. Its five-group Metal result was
nearly identical. That is strong evidence that the initial broad fake-binary
rule damages draft quality, even though the parity issue prevents a clean
target-equivalent attribution. The one-prompt profile shows all nine candidate
linears account for 38.32% of instrumented draft time, so native binary math
would still leave most observed draft work and all target verification.

- **Selective coverage:** head-only W1A1 retained 72.4% of ordinary acceptance,
  the best of the tested groups. The head accounted for 12.93% of measured
  draft time on the profiled prompt. Its optimistic zero-cost draft-time model
  was 1.148× before packing or acceptance effects. FFN-only retained 46.0%
  of ordinary acceptance for 14.29% of measured draft time. More combinations
  could be ablated without training, but no selective variant has demonstrated
  an end-to-end speedup.
- **Bounded QAT:** a short drafter-only fine-tuning run with fake W1A1 forward
  operations could test whether broader binary coverage regains acceptance.
  Use separate calibration/training prompts and re-evaluate this held-out
  suite. No QAT result exists, and training cost or success is uncertain.

The user owns that fork. Before either path supports a target-equivalent or
speedup claim, the BF16 tree-verifier versus full-prefix target discrepancy
needs a bounded resolution or a protocol that explicitly carries the
limitation. The RTX 2080 Ti native-binary timing and paired end-to-end
comparisons remain later stages.

## Raw artifacts and replay

The full sweep used this exact command under the configured remote workdir:

```sh
python3 scripts/remote_job.py cuda-exploratory-12prompt-20260924 -- \
  .venv/bin/python scripts/evaluate_pytorch_w1a1.py \
  --model-manifest results/model-manifest-5080-20260924.json \
  --run-id cuda-exploratory-12prompt-20260924 \
  --variants ordinary fusion attention ffn head all \
  --allow-greedy-mismatch
```

The code revision was `ae471198911fe7e093a4e26ff95655105843b725`.
Raw files are ignored by Git under remote
`results/cuda-exploratory-12prompt-20260924/`, with supervisor state and log
under `runs/cuda-exploratory-12prompt-20260924/`. The results directory also
contains the resolved config, prompt copy, and model-manifest copy. SHA256:

| Artifact | SHA256 |
| --- | --- |
| `acceptance.jsonl` | `83f1442ad157c3e2c8b53ed9f4edcaee2bfdb64226d04776b4bb873bab402963` |
| `summary.json` | `f881d367d535b3b9de5ac72a0572ba0540eeb798f06d15ea93d0ff3a68defb2a` |
| `greedy-mismatches.jsonl` | `f2576dae0ff4c4b0ef2888a6d5ab61b7d683a357547854bfd97f3b600dcbd596` |
| `greedy-parity.json` | `7e9d7694774349163d96c8080cdd120158e7209d57fcd0a19201e296eaa49742` |
| `target-reference.jsonl` | `7cafdfff4e7bf728264021c4ac65f571419648637121fb73da9cb3426cafe0c8` |
| `derived-metrics.json` | `c6aee99e843af727283968054b762cbe6898c39c045d3bfc2fe8a734ab7852a1` |
| `zero-accept-rounds.json` | `96062da51512ef6c4e63be5a1639008b2ac07431b3f791860d3fabe7267f3524` |
| `environment.json` | `fcd54fb9f52eb6982128e4436a76f5c83688a5e4357432bcb69d29390dab382a` |
| `environment-freeze.txt` | `d0027ce02dd9c4929338edecd66c58e2258da0bf3056d5c62713381d94db619e` |
| supervisor `state.json` | `60bbddb5b1b5298a7f30b56456b9e06d0311ffa62674f1bba7cbb5cc4136762c` |
| supervisor `stdout.log` | `d6639e02b977b1641a39e25c732494f37522d7fee4983f9715ee16d94120ac39` |

The supervisor finished with exit 0; PID/PGID 494 and all project evaluator
processes were absent afterward. Three Windows samples showed 0% GPU use and
memory back at the 3,108 MiB idle baseline. No model, cache, or raw run file
was added to Git. The ignored local command and cleanup record is
`results/preflight-5080-20260924T034000Z/host-preflight.txt`, SHA256
`394838155567ba0e463df47bd9bdecf7b98349c2341aba04092f6557229303a9`.
