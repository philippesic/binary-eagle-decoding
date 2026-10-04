# Released-reference result on RTX 2080 Ti

DFlash7 and DSpark7 beat the primary Q4_0 EAGLE baseline on all six paired
repeats. DFlash7 achieved132.37 request tokens/s versus90.80 (+45.8%);
DSpark7 achieved129.49 (+42.6%). Both seven-token variants matched Q4_0 EAGLE
raw output IDs on all144 measured requests each. This is a completed native
released-checkpoint result, not a W1A8/W1A1 or FFN-Q4 result.

| Arm | Request tok/s | Decode tok/s | Mean client request ms | p50 / p95 ms | Useful accepted / active draft round |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target-only |63.75|66.23|1994.64|2008.19 /2029.08|—|
| Q4_0 EAGLE D5 |90.80|96.12|1400.49|1382.63 /1690.65|1.135|
| DSpark D3, released BF16 |99.30|105.91|1280.65|1197.01 /1631.94|2.087|
| DSpark D7, released BF16 |129.49|140.87|982.08|871.22 /1536.03|3.254|
| DFlash D3, released BF16 |107.13|114.88|1187.02|1148.39 /1546.97|1.995|
| DFlash D7, released BF16 |132.37|144.18|960.73|870.05 /1576.83|2.903|

These are pooled output-count/time rates at concurrency1, with fixed24 unsealed
prompts (eight prose/code/reasoning each), six balanced repeats, two warmups per
cell, greedy seed42, nonthinking, max128 outputs, context2048, batch/microbatch32,
F16 target and F16 KV. DSpark/DFlash compute seven bidirectional noise rows,
read from author slot0, and propose3 or7. Regular native round tracing is enabled;
no admission, raw-logit, synchronized component or CUDA-event profiling runs are
included. Latency is whole nonstreaming client HTTP latency after loading and
warmup; TTFT and maximum concurrent serving capacity were not measured.
BF16 names source matrix storage: SM75 does not establish BF16 Tensor Core use.

The DFlash7 request-rate ratio to Q4 spans1.4504–1.4666 across the six repeats;
DSpark7 spans1.4178–1.4409. Their mean client request latencies are31.4% and29.9%
lower than Q4 respectively. All within-arm full-output repeatability lists are
empty: no variable prompt IDs were observed. Each arm emitted18,312 API-counted
completion tokens. Active-round counts/proposed/raw-accepted/useful-accepted are
Q4:8484/41850/9636/9630; DS3:5862/17460/12234/12234;
DS7:4260/29310/13860/13860; DF3:6048/18054/12066/12066;
DF7:4644/31968/13482/13482. No-proposal terminal records are separately60,72,48,
60,48 respectively. The initial seed can lie outside native round traces.

## Correctness scope

Q4 and both D7 arms differ from target-only on prose-01 at generated index89
in every repeat. The original full-request raw target-logit gate passed at that
shared prefix with opposing near-tie margins0.008221/0.003122 and centered
common-top-five difference0.008618; target-only versus speculative paths are
not bit-exact. Both D7 arms nevertheless match the primary Q4 output on all24
prompts and all six repeats.

Both D3 arms additionally differ from Q4 on prose-01 at94 (scoped near-tie gate
passed) and prose-06 at90 (16062 versus28071). The latter is newly observed and
not covered by the five-case admission receipt; one bounded original-order
raw-logit check is being prepared. Their measured rates remain preserved, but
no lossless-output claim is made for short arms. No threshold or native verifier
behavior is changed to obtain the reported rates.

## Pins and budget

Actual hardware: RTX2080Ti/SM75, UUID GPU-35b7b96c-d577-95f0-0050-40699c9faef7,
11264MiB, driver610.74, WSL2. Native sourcefcdf5822c5b78f9dbcfd1f5c7106f09c3f0c9b1a;
CUDA binary1bd67cdf74d6ced49454ca2546d9a462e71d4e695b8fb5e52d7359fa68473822.
Target05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6;
Q4 EAGLE2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280;
DSparkdc5299bdb1e7e906b003334a9b806ba502e72111341432ac5a9fd11de22f520c;
DFlash92925d9e4be49a82d1e4f9d9671aae0e3647c1146fccbd4004bfc8afe0c9896f.

Run dspark-reference-sixrep-fcdf-20261004 used immutable parent
 a9dded1bb2e554d7394472f93d6160db24f333a9. Supervisor state confirms exit0,
started2026-10-04T02:43:24.061417Z, ended2026-10-04T03:10:43.771869Z.
Measurements:936 records (864 measured+72 warmups), inference1246.091775789s,
prior0; SHA74b9c9dfc4f16bf549502efe0fe779f22ac7e1a8db5839e008a9ee655aa31381.
CPU analysis source5c8fac35c3cb043b926960207e53f5c5fa62ff32,
summary SHA07c85d24f846c06fd9eea4a1b9188d28dd9d5548efb008fa8620d87472b78f99.
Artifacts live under the remote immutable checkout's
runs/reference-sixrep-fcdf-20261004/results; see operator.md for exact launch.

All owned reference groups and native contexts stopped; GPU returned to366MiB
desktop baseline with empty compute-app query before Q4 admission. Request
inference1246.09s is charged to the shared7200s budget. Source/config/model
hashing, downloads, builds, admissions and later profiles are separate overhead;
the supervised reference transaction took1639.71s total wall, including391s
of non-request overhead. FFN-Q4 admission and timing are still incomplete.
W1A8/W1A1 need the pending human implementation scope choice; no training or
RTX5080/QAT/fusion work is part of this study.
