# Calibrated FC16 native CPU preparation

Status on 2026-10-04: **plan only, awaiting native root GO**. The training owner
explicitly released the CPU heavy slot after all four calibrated-init cells
passed and their process groups exited. Post-release available memory was
22,188,785,664 bytes of 38,654,705,664 total. No actual composition, export or
model load has started for these native cells.
This report extends `native-cpu-artifacts.md`; original prototypes and failed
runs remain unchanged.

The four planned cells are DSpark/DFlash W1A8/W1A1 `ffn15_fusion`. Each combines
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

New artifacts will use distinct ignored directories
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

Planned jobs are sequential on the Apple M3 Max CPU, each with an 8 GiB process
group RSS cap. Export timeout is 600 seconds; graph timeout is 180 seconds.
The graph uses three synthetic context rows and seven intact anchor/MASK slots,
actual MASK151669 and all sixteen selected I32/W1 projections. Passing would
prove calibrated weight serialization/load/operator execution with synthetic
activations. It would not prove native TRAIN trajectory quality, convergence,
CUDA execution, latency or throughput. Zero optimizer updates are planned.

Acceptance checks are exact member preservation, protected tensor preservation,
exact sixteen named projections/bits/shapes, CPU buffers, intact slot/mask layout,
finite full-vocabulary outputs, exit0 and empty owned process groups. Actual
resource/result receipts and new GGUF hashes are pending execution approval.

A bounded synthetic composition review passed all 32 exact NPY member byte
matches and sixteen canonical projections. A missing FFN member was rejected
before reserving its output directory. Receipt:
`runs/nine-model-native-cpu-20261004/calibrated-fc16-plan/composition-fixture-review.json`.
Composition source SHA256 is
`5d2c29a46ef544044b0e27ca3d1ef1f0ba4a69e8403e3af486e078712b5a1c0f`.
The fixture used tiny synthetic arrays only and grants no actual-model readiness.
