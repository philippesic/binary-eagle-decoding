# PyTorch W1A1 acceptance: RTX 5080 CUDA confirmation

**Status:** parity diagnostic, ordinary drafter profile, and one-prompt
exploratory acceptance check complete. The held-out 12-prompt sweep has not run.
This is a BF16 PyTorch fake-binary acceptance experiment, not native one-bit
execution or an end-to-end throughput comparison.

## Fixed inputs and environment

- Remote project code: `daad1571f2f2be8404912c66cc19754e9f0daa00`, clean
  checkout. AngelSlim source:
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
  PyTorch 2.14.0+cu130, Transformers 4.57.6, BF16 target and drafter.
  The import path also required `datasets`, `shortuuid`, and Pillow.
- Greedy non-thinking prompts and the 60-token EAGLE tree settings are fixed
  in `configs/pytorch_w1a1.toml`. The intended sweep includes ordinary EAGLE
  and five W1A1 layer-group variants over the same 12 held-out prompts.

The pinned environment and model snapshots were prepared under supervised
CPU-only runs before GPU work. Exact setup commands and run states are preserved
in ignored remote `runs/` and the local
`results/preflight-5080-20260924T034000Z/host-preflight.txt` record (SHA256
`11904b197bb51c43ac1bd3e752bc9eb4ca03407c7a77b0625d23bb7e23d0f495`).
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
the attention output and FFN down projections.

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

## Interpretation pending

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
The full suite will be run in the same explicitly exploratory mode. The prior
Metal development counts are in
`experiments/pytorch-w1a1-metal-acceptance.md`; they are not a CUDA baseline.
