# All-layer W1Ax activation precision on RTX 2080 Ti

**State:** historical and development matrices complete; diagnostics in progress.
**Protocol:** [frozen W1Ax study](w1ax-activation-precision-plan.md).
**Hardware:** NVIDIA RTX 2080 Ti, SM75, 11,264 MiB VRAM under Ubuntu 24.04 WSL2.

## Frozen comparison

The same FP16 Qwen3-4B target and verifier serve target-only, ordinary FP16
EAGLE, Q8_0 EAGLE, Q4_0 EAGLE, and four all-nine-linear W1Ax drafts. The W1Ax
modes use byte-identical packed weight signs and F32 row scales from one GGUF;
`GGML_W1AX_ACT_BITS` changes only the activation path. A16 explicitly rounds
incoming F32 values to FP16 before signed accumulation. A8/A4 use per-token
absmax quantization with signed ranges ±127/±7 and nearest-even rounding. A4
uses the four-plane bit-serial CUDA kernel; A1 uses sign packing and mean
absolute scaling. The target, embeddings, normalization, attention scores,
residuals, and verifier remain at their ordinary precision.

The FP16 target GGUF SHA256 is
`05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`.
The shared all-nine W1A1 draft GGUF SHA256 is
`098e1ecbb299aa16e2c968663acc49e60c0fcf16b053766d9f558114f79d011c`,
matching the prior all-row source audit. The benchmark uses published llama.cpp
fork commit `3792aa79c` and project checkout `dc4ccdd`, CUDA 12.8 on SM75,
F16 KV, context 2048, concurrency one, D=5, confidence floor zero, greedy
decoding, two warmups, five measured repetitions, and one server at a time.
Each run manifest preserves exact commands and model/binary/config hashes.
The primary runner's GPU snapshots were unavailable because its nonlogin WSL
PATH omitted `nvidia-smi`; they are not zero-valued measurements. Hardware and
dispatch were independently verified by the SM75 gates and explicit WSL GPU
queries. A separate full device snapshot is preserved in
`runs/w1ax-device-manifest-20260925/` (stdout SHA256
`bd87139e8ce478aef42c4b17fda2345891be9cd1c4ad02b826c47dee417af0c8`).
The probe is being corrected for separate telemetry diagnostics; primary
per-variant peak-memory/clock claims are not made. Q8_0/Q4_0 name standard GGUF block weight formats; their live
activation path is Q8_1, not the research W8A8/W4A4 format.
Acceptance is the pinned native server's target-sample-and-match count, not
classical probability-ratio/residual speculative sampling. All comparisons
use that same verifier and its unchanged target sampling path.

## Correctness gates

- Local CPU operator tests passed 88/88. On the actual 2080 Ti, native A1,
  A16, A8, and A4 bit-serial CUDA tests passed 88/88 with an independent
  INT32-dot assertion for A8/A4. The conventional A4 CUDA comparator passed
  another 88/88. Distinct mode dispatch markers were observed.
- A three-request diagnostic captured 4,203 real activations from all nine
  selected linears. A frozen selector retained 147 captures over every
  observed tensor/shape group; its full invocation histogram has 73 rows.
  Identical-input replay completed 147 × 4 native W1Ax operators with scalar
  output parity, and a separate assertion run passed exact A8/A4 INT32 dots
  on those real inputs. The diagnostic assertion requires CUDA graphs disabled;
  it is excluded from timing.
- One frozen 32-token request per mode gave identical raw CPU-draft and
  CUDA-draft token IDs for A1/A4/A8/A16, with the FP16 target on GPU in both
  paths. The selected-token log probabilities also matched exactly on that
  request. `--spec-draft-device none` was required to keep the reference
  draft on CPU; setting its GPU layer count to zero alone still dispatched the
  custom operation on CUDA.

Real-input replay measured the complete ggml operation, including activation
quantization/packing, native dot, output, graph dispatch and synchronization.
At the observed head N=2 shape, medians were 978.7/331.4/117.8/94.8 µs for
A16/A8/A4/A1. At N=37 they were 29,443/5,470/1,637/919 µs. These are
synchronized host-wall operator timings, not isolated CUDA kernel events or
serving rates. Across 145 selected head token rows, A8/A4/A1 agreed with the
same-binary-weight A16 top choice 97.2%/72.4%/33.1% of the time. The rows are
correlated diagnostic inputs, not independent prompt examples.

## Historical paired matrix

The 12 frozen historical prompts × five repetitions × eight paths completed
480/480 requests on the RTX 2080 Ti with no failed request. All four W1Ax
loader, graph-selector, and CUDA-dispatch gates passed in every repetition;
every response included raw generated token IDs. The GPU was idle afterward.

| Draft | Decode tok/s | Request tok/s | Accepted drafts/round | Decode vs FP16 | Decode vs Q4_0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target-only | 60.56 | 58.61 | — | — | — |
| FP16 EAGLE | 80.67 | 75.71 | 1.168 | 1.000× | 0.899× |
| Q8_0 EAGLE | 86.16 | 80.47 | 1.168 | 1.068× | 0.960× |
| Q4_0 EAGLE | 89.73 | 83.57 | 1.182 | 1.112× | 1.000× |
| W1A16 EAGLE | 29.43 | 28.73 | 0.106 | 0.365× | 0.328× |
| W1A8 EAGLE | 42.12 | 40.70 | 0.102 | 0.522× | 0.469× |
| W1A4 EAGLE | 44.44 | 42.89 | 0.041 | 0.551× | 0.495× |
| W1A1 EAGLE | 45.39 | 43.74 | 0.055 | 0.563× | 0.506× |

Paired prompt/repetition bootstrap 95% intervals for W1Ax decode rate versus
Q4_0 were 0.304–0.354 (A16), 0.434–0.508 (A8), 0.454–0.541 (A4), and
0.466–0.549 (A1), using 2,000 resamples. Repetitions estimate timing
variation; the 12 prompts, not the 60 repeated requests, supply the quality
examples.

All speculative variants emitted identical raw IDs on all 60 paired
prompt/repetition requests. They differed from target-only on the same
`reasoning-02` request in each repetition, first at generated ID position
109. Ratios to target-only are timing observations, not strict lossless speedup
claims. W1A16 preserves more activation precision than A8/A4/A1, yet accepts
only 0.106 drafts/round versus FP16 EAGLE's 1.168. Its slow sign-add kernel
also makes it the slowest W1Ax serving path. The development matrix below provides the next quality check; the 24 QAT-final prompts remain reserved.

Historical raw artifacts reside on the 2080 Ti under ignored
`runs/w1ax-project-src/results/w1ax-historical-matrix-20260925/`. The
`records.json` SHA256 is
`160c2bced47d98cf641bc4e5fecb7b46caf23c3442ad2e75cb6d7891f11869e7`;
`report.json` is
`83299ab3ce612e75dcae839ab4221b8514dc8792bc1b6073d041369d97307fbf`;
the 2,000-resample paired analysis is
`b646e3df7bbb04764067386ecc1bf868011cc0fb764296debb47ddc20661a7f2`.
The paired intervals, per-prompt data, actual proposal/acceptance counts,
model hashes and dispatch evidence are preserved in those files. The selected
operator replay JSONL SHA256 is
`a06654a31f000bacf5988cda91201a100b57575fbead9548b66d614093c1afe5`.

## Development paired matrix

The frozen 24 development prompts × five repetitions × eight paths completed
960/960 requests with exit 0. All dispatch gates passed, and every speculative
variant matched target-only raw token IDs on all 120 paired requests. Each
request reached the 128-token cap. The primary build and policy were unchanged.

| Draft | Decode tok/s | Request tok/s | Accepted drafts/round | Decode vs FP16 | Decode vs Q4_0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target-only | 60.45 | 58.58 | — | — | — |
| FP16 EAGLE | 76.55 | 72.40 | 1.047 | 1.000× | 0.912× |
| Q8_0 EAGLE | 81.56 | 76.94 | 1.046 | 1.065× | 0.972× |
| Q4_0 EAGLE | 83.93 | 79.05 | 1.036 | 1.096× | 1.000× |
| W1A16 | 29.86 | 29.20 | 0.117 | 0.390× | 0.356× |
| W1A8 | 42.49 | 41.19 | 0.111 | 0.555× | 0.506× |
| W1A4 | 44.68 | 43.26 | 0.047 | 0.584× | 0.532× |
| W1A1 | 45.36 | 43.91 | 0.055 | 0.593× | 0.540× |

Paired prompt/repetition bootstrap 95% decode-ratio intervals versus Q4_0
were A16 0.343–0.368, A8 0.488–0.523, A4 0.509–0.555, and A1 0.520–0.559.
Against FP16 they were 0.375–0.404, 0.534–0.575, 0.558–0.608, and
0.570–0.613, respectively (2,000 resamples, seed 42).

The same acceptance loss persists with A16: restoring activation precision does
not recover the all-layer binary draft's quality. None of the W1Ax paths
approaches either throughput anchor on this development set.

Raw artifacts: remote `runs/w1ax-project-src/results/w1ax-development-matrix-20260925/`.
SHA256: `records.json`
`a3ac1f91d249a3edd7c54d07ff69a457d49dbd0efaabb317a45a4c0149ab04f8`;
`report.json` `f653acbb50bf490da3d589932d201f9d7f59bb4e795968eade0e9344de07e7ea`;
2,000-resample `analysis.json`
`f175094050ca1d807e2f3a6b963b44f97b14b8a7e2d39ab8d2eee5d2f49f9592`;
`manifest.json` `e7c2c55fca7a609bce162fa348d37bc93da2181707b5eb24d21c0be6b7a22cab`.

## Outstanding measurements

Identical-input FP16/Q8_0/Q4_0 anchor replay, per-round trace, context/output-cap diagnostic, D/p_min policy
grid, and separate CUDA stage profiles are pending. Do not treat the historical
screen as a final trained-QAT result or use the reserved 24-prompt final set.
