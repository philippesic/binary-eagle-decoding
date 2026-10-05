# Calibrated FC16 native CPU preparation

Status on 2026-10-04: **PASS, all four CPU instrumented operator cells**.
After root reviewed the exact plan/composition proof and training released its
slot, four sequential exact compositions, exports and standalone native graphs
completed successfully. This report extends `native-cpu-artifacts.md`; original
prototypes and failed runs remain unchanged. No deployment source changed.

The four cells are DSpark/DFlash W1A8/W1A1 `ffn15_fusion`. Each combines
the previously verified original FFN15 latent/scale arrays with the corresponding
actual TRAIN, scale-only fitted FC initializer. The source FFN state is untrained;
the FC scales are fitted, with rescue disabled and zero coordinate/sign events.
The explicit FC policy is `preserve_reference_magnitudes`, reference kind
`block_source_weight_magnitudes`. Latent magnitudes, including values outside
the STE interval, are copied exactly; there is no normalization, refit, clipping,
sign change or optimizer update.

The ignored plan is
`runs/nine-model-native-cpu-20261004/calibrated-fc16-plan/plan.json`, SHA256
`892860d7431e36fbcbe06ee6721f536412e1b0bb282347b997334e420711a930`.
Fit index SHA256 is
`e59edcd2c235c4fbca800892dde1bc86199db57729177e93371b006a8c490800`.

| Cell | Fitted FC NPZ SHA256 |
|---|---|
| DSpark A8 | `caa9d4337015271d69fbfc56a8ef71f9d7d6883b21058b9340b3a9c20ac79c32` |
| DSpark A1 | `06652d0812173994e6d7b3cc830f7b047fb53e4e9c5e19f2818f2211da7c47e0` |
| DFlash A8 | `effb7421beada9addd90f9b73b6146126f7380d5b800b45ac7315fd603a40085` |
| DFlash A1 | `4c392657542350803afa7a41da47228fec084f829ecf0c2463d4b798efeb51fd` |

The composition helper copies 30 FFN and two FC NPY members directly through
uncompressed ZIP streams and independently rereads their SHA256s. The full
manifest retains the exporter's exact schema v1 fields with sixteen projections;
checkpoint SHA256 is filled only after composition. Source fit/report hashes,
member byte hashes and magnitude policy are recorded in a separate provenance
receipt. Candidate/control FC files have identical hashes. Small header checks
confirm exact F32 FC shapes `[2560,12800]` and `[2560]`.

New artifacts use distinct ignored directories
`models/nine-model-calibrated-native-03-scale-only/{family}-a{bits}/`.
The base models, private embeddings/head, norm, attention and Markov tensors
are preserved by the existing serialization audit. Source/base and old FFN15
hashes are pinned in the plan and previous report. Original frozen Q4 controls
remain absent locally and are not replaced by these new exports.

The CPU build uses native source `624f50e74f51b6af93bf6b879f84703e726df172`,
CPU enabled and CUDA/Metal/BLAS/backend-DL disabled. Runtime packet
`runs/nine-model-native-cpu-20261004/cpu-native-packet/rebuilt-624/packet.json`
has SHA256 `9782bb05e7fa43acb54471f5ad32519ee8673be13676556734c7fa7b07c78c1a`.
The graph helper binary SHA256 is
`e0c0acf4c8de3016195b132583926a5f3c31a5b2f496aee9ff0fb1fd1f5b766d`.
Mac runtime mapping remains explicitly unchecked in NativeTeacher; this native
operator check does not turn it into a CUDA or teacher-portability proof.

All jobs ran sequentially on the Apple M3 Max CPU, each with an 8 GiB process
group RSS cap. Export timeout is 600 seconds; graph timeout is 180 seconds.
The graph uses three synthetic context rows and seven intact anchor/MASK slots,
actual MASK151669 and all sixteen selected I32/W1 projections. Passing proves calibrated weight serialization/load/operator execution with
synthetic activations. It does not prove native TRAIN trajectory quality,
convergence, CUDA execution, latency or throughput. Optimizer updates were zero.

Acceptance checks are exact member preservation, protected tensor preservation,
exact sixteen named projections/bits/shapes, CPU buffers, intact slot/mask layout,
finite full-vocabulary outputs, exit0 and empty owned process groups. All checks passed; actual receipts and new GGUF hashes follow below.

A bounded synthetic composition review passed all 32 exact NPY member byte
matches and sixteen canonical projections. A missing FFN member was rejected
before reserving its output directory. Receipt:
`runs/nine-model-native-cpu-20261004/calibrated-fc16-plan/composition-fixture-review.json`.
Composition source SHA256 is
`5d2c29a46ef544044b0e27ca3d1ef1f0ba4a69e8403e3af486e078712b5a1c0f`.
The fixture used tiny synthetic arrays only and grants no actual-model readiness.


## Actual calibrated CPU exports and graphs

Every composed checkpoint contains 32 exact original NPY members: thirty FFN
members plus the two fitted FC members. The protected tensor serialization audit
passed for every export, including private `token_embd.weight` and `output.weight`
remaining BF16 with raw SHA256
`eabe5625fc0575bf517c424041e9701c0fd521889e0f547c8522d2aa20e8c0f8`.
Norm, attention and DSpark Markov tensors retain their base bytes/type.

Every graph observed exactly sixteen named selected I32/W1 projections, matching
its manifest and activation bits, with actual CPU result buffers and no selected
dense fallback. All use three synthetic context rows and seven anchor-first
noise inputs `[2,151669,151669,151669,151669,151669,151669]` at positions3..9.
The actual native helper verified finite full-vocabulary outputs. No target
co-load or captured native TRAIN trajectory was repeated for these cells.

| Cell | Calibrated GGUF SHA256 | CPU graph receipt SHA256 |
|---|---|---|
| Dspark A8 | `f508b201a263ab2e3769f94a1eaf3696db9a1b7aaccbf73ea98307509df920f6` | `762d38a5d75d468635106b5c29baa7c6e37c03f58145062094dfa50763bb7cd9` |
| Dspark A1 | `9a59e2a8f933f50631e1da61ab025fc0ab00333c57b3d2ba4773997f3fdf4c47` | `7a4d7d0745d361118c4beb522be535a57d1700da21047716cbc1d0fdc12d2377` |
| Dflash A8 | `2f76dc6e6e2a031cdcbe3dd2350d8305bfc9205958ce964e20fb6c11126232aa` | `9192385570baed74a39fd4a0ab1a6d2ff5ebcc256c4e83a1a74612669ed21e9e` |
| Dflash A1 | `c6f87378de90ddfc86a011572680cf01a9223ddc35be5f85dcb592ea8a1c9973` | `af9cb5893cf210a1a89515fa5dd538696f5bf42db5157cc5a6493e7b2f23509b` |

Machine-readable source/composition/manifest/export/graph/resource joins are in
`models/nine-model-calibrated-native-03-scale-only/actual-cpu-summary.json`,
SHA256 `77ba22ef6c1ddb3e00f2d891c68ea1be809e28482ae1d6ae819739151c70b009`.
The reviewed plan remains immutable, including its pre-execution status text;
this separate result summary records actual completion.

All twelve stages exited0, stayed below 8 GiB and recorded empty owned groups.
Final independent group checks also found all twelve PGIDs absent; process
census found no native/model helper or runner. Maximum sampled export RSS was
5,582,127,104 bytes (kernel child peak5,602,787,328); graph RSS2,224,340,992 bytes.
Exact argv/logs/resource receipts are preserved at
`runs/nine-model-native-cpu-20261004/calibrated-fc16-{family}-a{bits}-{stage}/`.
The heavy CPU slot was explicitly released, with available memory21,895,544,832
of38,654,705,664 bytes measured by CPython3.11/psutil7.2.2. No further model runs
were started.

Source/helper/scripts/runtime-library disk hashes were checked adjacent to the
runs and before each remaining stage. A separate receipt-only reviewer checked
DSpark A8's member/projection counts, protected BF16 hashes, actual CPU node bits
and slot layout, and below-cap/group-empty receipts. This grants CPU operator
coverage only. CUDA execution, convergence, quality/throughput and original
frozen Q4 control admission remain pending; none is inferred from these passes.
