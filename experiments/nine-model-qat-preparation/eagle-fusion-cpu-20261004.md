# Saved EAGLE TRAIN fusion preparation — October 4, 2026

Actual bounded CPU fitting produced separate fixed A8 and A1 fusion initializers
and scale-only controls from the previously authenticated native TRAIN operands.
No GPU operation, capture, real-model optimizer update or quality evaluation ran.
This preserves the historical fits and the protected Q4 controls.

Source `b8d0ff4`, `block_fusion.py` SHA256
`f7717a71b4fda1afcff62bea517f9e7602e5846e2196543e6bb0992663eed723`.
Historical data-v2 provenance receipt externally pinned to
`75cbf2b83fdcc14b2dc338a5ac2fe27188a41cd6957fa311ea7d0e6bcd857c87`.
The existing authenticated importer rechecked the original BF16 fusion weight,
F16 base, target-capture ancestry, row joins and original TRAIN membership.
Same 256 fitting rows/eight prompts and 128 validation rows/four disjoint
TRAIN prompts. Historical validation has prose and reasoning but **no code**;
this cannot grant all-domain campaign calibration readiness.

Teacher arithmetic is raw native-captured F32 inputs multiplied by original
BF16-promoted F32 fusion weights through CPU BLAS. Candidate uses exact integer
dot, separately rounded F32 row scale and token scale. A8 uses F32 absmax,
reciprocal and nearest-even codes; A1 uses F64 mean-absolute rounded to F32 and
positive sign for either zero. Norm diagnostics use the original
`midlayer.hidden_norm.weight` with epsilon `1e-6`; native RMS reduction validation
remains pending. Frozen F16 target/verifier precision is unchanged.

Both fits enabled whole-row orientation rescue **only**, with zero coordinate
search flips. Scale-only controls preserve original signs and nonnegative LS
scales. Rescue negates negatively correlated zero-scale rows, re-solves scale,
and requires an improving separately rounded exported objective. Initializers
and controls were saved and hashed before validation; no validation update or
selection occurred.

| Precision | Rescued rows | Raw relative SSE control → rescue | Post-norm relative SSE control → rescue | Post-norm cosine control → rescue |
|---|---:|---:|---:|---:|
| A8 | 383 | 0.074499 → 0.050273 | 0.958212 → 1.076628 | 0.549544 → 0.494952 |
| A1 | 989 | 0.749849 → 0.642366 | 1.285218 → 1.685017 | 0.387733 → 0.206992 |

The raw objective improves while post-norm direction worsens. Rescue remains an
explicit off-by-default probe; these artifacts do not select it as the campaign
default. Neither objective is vocabulary accuracy, native acceptance or speed.
Native trajectories and the human's coverage/recipe decision remain necessary.

Apple M3 Max/macOS arm64 CPU, two BLAS/OpenMP threads. Fit-only times were 2.891 s
A8 and 2.879 s A1. Whole-process maximum RSS was 969,490,432 bytes (924.6 MiB),
below the unchanged 1 GiB bound. Capture producer was the historical RTX5080;
this new operation used CPU only and does not certify SM75/SM120 execution.

Artifacts remain outside Git under
`results/nine-model-qat-preparation/eagle-fusion-20261004/`. Each NPZ contains
`fc.latent` F32 `[2560,7680]` and `fc.scale` F32 `[2560]`; it initializes fusion
only and must be merged with the complete declared projection checkpoint before
native model export.

| Artifact | SHA256 |
|---|---|
| `fusion-a8.npz` | `bfbbcc2e5595fe7426a0e683d6215a9349acd2de1dae94d491c4a8f54a5fc67a` |
| `control-a8.npz` | `2642a292a77a5b687896292a06b741d7fb06306fd9d22260a587bf038f2aec44` |
| `fusion-a1.npz` | `b6d616ef6871ad5273016e5cdc7a100760c4c93deb35916a979b462c0ab6d62b` |
| `control-a1.npz` | `72de7c9369e3503bf8e1217fa76129d88e0a21dcf5fc896950b07f89c758fd4a` |

The original executed `run_script.py` and full `report.json` retain all selected
rows, prompt hashes, finite-objective events, data/model/source pins and raw/post-
norm metrics. Reproduction uses `scripts/fit_saved_eagle_fusion.py` with explicit
data/receipt/model pins and a new output directory; it never overwrites history.

Remaining work: expand code validation and campaign-selected calibration coverage;
produce/authenticate five-tap block TRAIN captures; fit block-family fusion at
both deployed precisions; validate actual native operators/graphs/trajectories;
run the narrow fresh SM120 portability gates when RTX5080 is available.
