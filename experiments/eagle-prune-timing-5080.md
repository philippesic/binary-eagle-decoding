# EAGLE unused-head pruning: paired native timing on RTX 5080

The opt-in `GGML_EAGLE_PRUNE_UNUSED_HEAD=1` skips decoder head work on
zero-logit catch-up. It applies to Q4_0, fitted D group-128/A16 and FP16
EAGLE through one shared native graph. Q4_0 is the primary comparison
baseline. Target-only controls time drift. This report separates full client
request throughput from server decode throughput; neither is a `process()`
stage duration or serving-capacity result.

## First paired block: pruning off, then on

Both supervised RTX 5080/SM120 jobs finished exit zero on 2026-09-29 UTC.
Each used the same SHA-pinned F16 target, Q4_0/D/FP16 drafts, binary,
24 old development prompts, 2 warmups and 5 measured repetitions per
variant: **480 measured requests and 20 verified CUDA-graph server blocks**
per condition. No other project GPU job ran concurrently. The GPU read 0%
utilization and 2,899 MiB baseline memory before the block and after each
supervisor stopped. The [quality A/B](eagle-prune-quality-5080.md) had
already matched all 96 single-pass request pairs.

The [paired timing comparator](../scripts/compare_eagle_prune_timing.py)
matched **480/480** measured prompt/repetition/variant pairs in generated
raw IDs and speculative counters; behavior mismatch count was zero. Ratios
below use pooled token/time totals. The 95% intervals resample 24 prompt IDs
and retain their five repetitions; they do not account for all time drift.

| Variant | Off decode tok/s | On decode tok/s | On/off decode | Off request tok/s | On request tok/s | On/off request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q4_0 EAGLE** | 141.55 | 142.80 | 1.0088 (1.0066–1.0109) | 133.72 | 134.98 | 1.0094 (1.0070–1.0118) |
| D group-128/A16 | 54.68 | 57.57 | 1.0528 (1.0521–1.0537) | 53.43 | 56.19 | 1.0516 (1.0510–1.0525) |
| FP16 EAGLE | 126.91 | 128.35 | 1.0113 (1.0106–1.0120) | 120.56 | 121.94 | 1.0115 (1.0106–1.0123) |
| Target-only | 99.61 | 99.87 | 1.0026 (1.0011–1.0049) | 96.93 | 97.24 | 1.0032 (1.0018–1.0056) |

Q4_0 still leads D strongly on both acceptance and throughput. Across five
repetitions, Q4_0 accepted 7,775 of 36,600 drafts; D accepted 4,545 of
52,245. D's on-path decode rate is about 0.403× Q4_0 on the same target
and hardware, so this runtime patch alone does not meet the primary goal.

Ignored remote evidence under `~/binary-eagle-decoding/runs/`:

- `w1-prune-off-timed-20260928/benchmark/manifest.json` SHA256
  `0c3e396bbabb018318858507da50d920ed50edd13ec410763857cf4e1a230cf0`.
- `w1-prune-on-timed-20260928/benchmark/manifest.json` SHA256
  `6e64dfddbad2b00da0043269957e67081fe8ab963903b1302e6a24eeff357d40`.
- `w1-prune-timing-compare-20260928/report.json` SHA256
  `44368dc46ced468d42d23f77bd4bed7410e11aa290066b824d8c78730a8521df`.

The first block ran all off repetitions before all on repetitions. Target-only
also improved 0.26–0.32%, so an ordering effect was present.

## Reverse-order confirmation: pruning on, then off

A second pair ran **on before off** with the same configs and artifacts. Both
supervised runs exited zero with 480 measured requests and 20/20 verified
CUDA-graph blocks each. The GPU returned to 0% utilization and about 2,900
MiB baseline after each run. The paired comparator again found **zero behavior
mismatches across 480 request pairs**, including raw outputs and speculative
counters.

| Variant | Reverse on/off decode | Reverse on/off request | Order-balanced decode | Order-balanced request |
| --- | ---: | ---: | ---: | ---: |
| **Q4_0 EAGLE** | 1.0081 (1.0073–1.0088) | 1.0073 (1.0061–1.0085) | **1.0085** | **1.0084** |
| D group-128/A16 | 1.0498 (1.0494–1.0502) | 1.0483 (1.0478–1.0487) | **1.0513** | **1.0499** |
| FP16 EAGLE | 1.0111 (1.0104–1.0117) | 1.0105 (1.0094–1.0116) | **1.0112** | **1.0110** |
| Target-only | 0.9999 (0.9997–1.0002) | 0.9992 (0.9985–1.0000) | **1.0013** | **1.0012** |

The order-balanced columns are geometric means of the two block ratios,
descriptive rather than a new independent confidence interval. The repeat
supports a throughput benefit for the shared opt-in graph on this RTX 5080
development workload: roughly 0.8% for Q4_0, 1.1% for FP16 and 5.1% for D
in server decode. Target-only changed about 0.1%. The effect does not repair
D's much lower acceptance or its throughput deficit to Q4_0.

Reverse-order ignored remote evidence:

- `w1-prune-on-timed-b-20260929/benchmark/manifest.json` SHA256
  `9f5323589f919108d9e8d6c1110abcc34574912810e2eb5ee8aa93f36f05518e`.
- `w1-prune-off-timed-b-20260929/benchmark/manifest.json` SHA256
  `71a19f9aab269d1f3cba45cb9177d336e40ed66ce9d9e54186084e11cfdfe3f3`.
- `w1-prune-timing-compare-b-20260929/report.json` SHA256
  `e452374ceaa9c125310ff79eb9f1ec25424916e9f253455a2567c93499104488`.

`process()` stage attribution and exact cache-byte equivalence remain separate
validation gates before making pruning the default. This is concurrency-one
native request/decode timing on heavily reused development prompts, not
saturated serving capacity, untouched final quality or SM75 performance.
