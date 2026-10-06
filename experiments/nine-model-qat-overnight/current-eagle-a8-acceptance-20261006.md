# Current EAGLE A8 native acceptance

Measured October 5–6, 2026 on RTX5080 / SM120. Current trained snapshot is
step 344,724 after **14.43 trainer hours** (51,950.921720 seconds), preserved
from the human pause. Its old 86,400-second receipt remains interrupted.
The newly selected 12-hour rule adds no further EAGLE A8 training.

| Drafter | Accepted / proposed | Acceptance | Accepted / round | Request tokens/s |
|---|---:|---:|---:|---:|
| Original Q4_0 EAGLE | 8,025 / 30,290 | 26.4939% | 1.3038 | 143.87 |
| Calibrated untrained A8 | 1,485 / 62,050 | 2.3932% | 0.1180 | 70.46 |
| Trained A8 | 4,765 / 46,125 | 10.3306% | 0.5080 | 92.85 |

Training improves acceptance **4.32×** over the exact zero-update
initialization. Trained A8 remains below Q4; its measured request throughput is
**0.645×** its paired Q4 control. These are negative primary-baseline
observations, not a deployment or total-throughput success claim.

Each A8 comparison uses its own fresh Q4/target-only paired sweep: 24 original
unsealed development prompts, eight per domain, five clean repetitions and two
warmups per cell; 128-output cap, ctx 2048, batch/ubatch 32, seed 42, greedy,
thinking disabled, F16 target/verifier/KV and maximum draft length 5. Each cell
emits 14,290 tokens across 120 requests. Timing covers complete HTTP requests
(prefill plus decode), excluding server loading/warmups. Diagnostic logging and
memory sampling run separately. The table's Q4 timing is from the trained
sweep; the initial sweep's Q4 is 143.01 tokens/s with identical acceptance.

**Strict target-only output parity FAILED and remains failed.** Each drafter
and Q4 differs from target-only on one of 24 prompts at token 98, consistently
in all five repetitions. Trained/Q4 and untrained/Q4 complete token arrays,
output counts and finish reasons match all 120 pairs in their respective
sweeps. All nine candidate CUDA projection dispatches were verified in the
separate diagnostic pass. The bounded raw-logit probe reproduced the original complete outputs and
validated identical reached prefixes through token 97. At token 98, A8 and Q4
have identical competing logits: 31.634883881 for token 1519 and 31.629983902
for token 12 (gap +0.004899979). Target-only scores them 31.624923706 and
31.630243301 (gap -0.005319595). Raw argmax, sampled and emitted IDs agree
within every arm, with no NaNs. This supports numerical sensitivity; batching
is a source-backed hypothesis, not a proven cause or harmlessness claim.
No numeric threshold has been selected or original quality gate relaxed.
Probe summary SHA `175303f902f286e4e163b56825ce3a472f365a3fa1a2abb2b3dc74c0ece858be`,
under `results/twelve-hour-qa/native-raw/eagle-verifier-parity-20261006-01/`.

Source/runtime: frozen native `cc9cab3c64f61580cf63e5ef050b075b11cd1fb9`,
helper `c2544aa7928b0d0c454099a56ae912262c6b0ab5`; exact original Q4 SHA
`2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280`.
Initial model SHA `b2f6dddcf2eaad503a52523ed6e02061c388c3c92aa60241e212aab49611aee8`;
trained model SHA `6d3a8c1bc8b1c634f00a677ad466c00944c00f8c1accac4ff06e82403bdf60c9`.
Resume SHA `34978ea458afdc43794edf4a26e5d80a860ce0fd6c1d16ac1a3961203a591423`.

Raw runs: `eagle-a8-paused-native-20261005-02` (trained) and
`eagle-a8-paused-native-20261006-03` (initial). Preserved locally under
`results/twelve-hour-qa/native-raw/`; original remote paths remain in the records.
Every clean measurement's declared SHA256 was verified after transfer. Original
progress hashes: trained `b954bb86c4da3cbca10380c7fa2c15c00e87100d48a514a744faa1e450d1a213`;
initial `5fcbbf46dafa62fd3b21fe3c9066306142bb2ab5576731fd308221d89e665ebd`.
Derived counts/provenance: `results/twelve-hour-qa/eagle-current-native-summary.json`.
Raw files are ignored by Git; this report does not open sealed final inputs.
