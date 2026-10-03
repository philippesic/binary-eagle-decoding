# CPU fusion sign-and-scale fitting — October 3, 2026

**Synthetic fallback completed; no real TRAIN fit or EAGLE quality result.**
The local archive has no provenance-checked, prompt-disjoint raw pre-A8 fusion
inputs. The authorized fallback implemented and executed the bounded algorithm,
exported a synthetic candidate, and verified its actual CPU native arithmetic.
A8 QAT remains the only active goal; this team made no QAT, GPU, Metal, remote,
STATUS, goal-file, target/verifier, or sealed-final change.

## Result on separate synthetic prompts

The fixed seed `20261003` creates correlated F32 features with original synthetic
weights rounded through BF16. Six synthetic prompts / 144 rows fit the 8×64
projection; two separate prompts / 48 rows validate after checkpoint freezing.
Correlations intentionally make sign changes useful. This fixture establishes
algorithm behavior, not improvement on the actual 2,560×7,680 EAGLE fusion layer.

| Method | Fit relative squared error | Validation relative squared error | Validation cosine | Coordinate sign agreement | Largest-coordinate agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original signs + mean-absolute row scales | 0.241524 | 0.244041 | 0.868393 | 85.42% | 56.25% |
| Original signs + converged row scales | 0.222780 | 0.214383 | 0.884843 | 85.42% | 56.25% |
| Fitted signs + row scales | 0.006432 | 0.007612 | 0.995259 | 97.92% | 93.75% |

Signs reduced validation error by **96.45% relative to the scale-only control**
on this separate synthetic sample. Direction and coordinate rankings concern
fusion output coordinates, not vocabulary decisions or accepted drafts. Native
acceptance, latency, and throughput against primary baseline Q4_0 remain deferred.

## Algorithm and validation

- Fixed A8 contract: raw F32 absmax, F32 `beta=absmax/127` and reciprocal,
  nearest-even signed codes, exact integer dot, then separately rounded F32
  dot×row-scale×token-scale. Clip is fixed at 1; midpoint/residual/architecture
  and learned-activation changes are absent. Either signed zero initializes
  a positive weight sign; packed tail bits are zero and bit order is little-endian.
- The frozen reconstruction teacher is raw F32 input × BF16-promoted F32 source
  weights using F32 BLAS. Candidate objective uses native exported arithmetic.
  The scale-only control uses the exact nonnegative scalar least-squares optimum
  in F64 plus bounded neighboring-F32-scale checks against that finite objective.
  Maximum normalized continuous KKT residual was `1.64335e-16`; every control
  row converged with zero improving adjacent-scale gain.
- Two alternating passes / four correlation scans / at most 32 flips per output
  row. Stale-ranked proposals are checked sequentially against updated integer
  dots and the actual finite objective. All 43 accepted flips decreased SSE;
  every scale update was non-increasing. Zero-scale rows are skipped by the
  fixed-scale flip search; no escape from every discrete minimum is promised.
- Fit took 0.392 s on Apple M3 Max CPU, with two-thread environment limits,
  NumPy 2.4.6 / PyTorch 2.14.0. Estimated workspace: 856,064 bytes. Defaults cap
  1,024 total examples, 1 GiB estimated workspace and 12 CPU-hours. This tiny
  run neither measures full-layer fit time nor predicts accelerator performance.
- Frozen NPZ and GGUF were byte-identical across three fresh fits. Export/reload
  preserves signs, F32 scales, code/scale operands, all three nonfusion tensor
  payloads and original metadata. The candidate uses existing native v2
  fusion-only row/A8 GGUF representation with truthful nonnegative-LS scale
  metadata. The synthetic base is an operator fixture, not a full loadable EAGLE.
- Actual native CPU replay checked **1,536/1,536 outputs, zero absolute/relative
  error** against its independent scalar reference. Built with CUDA/Metal/BLAS/
  OpenMP disabled, two compile workers, clean llama.cpp
  `9e2c7a90051e738751aab7d7bd7c2d8201fb76e3`. No SM75 or native trajectory claim.

Independent test/review results are recorded in
[independent-validation.md](independent-validation.md). Exact config/source,
input-row/code/token-scale/teacher/candidate hashes and retained artifact paths
are in [summary.json](summary.json); weights/captures/raw runs are outside Git
under the primary checkout's `results/fusion-binary-discrete-a8-20261003/`.
Implementation is `f7fdfab` plus formatting `06ffb02`, integrated as `5c564df` and
`67cb7fe`; native wrapper is `22e46d8`. The reconstruction teacher and finite
candidate arithmetic are deliberately distinct and fully stated.

## Excluded local data and exact missing package

The scale artifact index says complete calibration captures remain remote.
Its local sampled replay manifest uses only one TRAIN prompt,
`qat-revisit-train-prose-urban-waterways-01`, with post-A16 values; it cannot
supply prompt-disjoint validation or raw pre-A8 values. The recurrent A16 bundle
explicitly declares `training_eligible:false`, with native model identity,
feature parity, cache/mask and request ancestry unverified. Exact exclusion
paths/hashes are retained in the summary. These values were not fitted.

The original real BF16 source is already local, SHA256
`58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e`, as is the
frozen F16 base GGUF,
`c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1`.
Neither was changed or evaluated by this fallback.

The smallest missing interface-demonstration package is **two distinct eligible
TRAIN prompts, 16 raw pre-A8 F32 rows of width 7,680 per prompt**, with one prompt
for fitting and one for validation: 983,040 bytes (960 KiB) plus manifests. A more
useful bounded screen is 8 fitting + 4 validation prompts × 32 rows: 11.25 MiB.
Include TRAIN inventory and prompt content hashes, explicit prompt split,
raw capture files/hashes/selected offsets, native revision, fixed target/draft
hashes and precision, fusion tensor/input-stage identity, and prompt/token/
position/invocation/depth/cache/mask joins with a verified eligibility receipt.
Teacher layer outputs can be derived under the stated frozen-layer contract;
logits and final data are unnecessary.

**Real importing currently fails closed.** Once this package is supplied,
a small provenance adapter must verify capture-to-row and TRAIN membership
before enabling actual fitting. Arbitrary NPZ eligibility labels are not enough.
No transfer, new capture, QAT, or GPU evaluation is initiated by this report.

## Reproduce the synthetic fit and CPU export gate

From a populated repository checkout with its existing Python environment:

```sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
export EAGLE_GGUF_PY="$PWD/third_party/llama.cpp/gguf-py"
FUSION_RUN=results/fusion-binary-discrete-a8-rerun
.venv/bin/python scripts/create_fusion_binary_discrete_fixture.py --output-dir "$FUSION_RUN/fixture"
.venv/bin/python scripts/fit_fusion_binary_discrete.py --manifest "$FUSION_RUN/fixture/synthetic_manifest.json" --base-gguf "$FUSION_RUN/fixture/synthetic_base.gguf" --output-dir "$FUSION_RUN/fit"
cmake -S kernels/w1ax-replay -B "$FUSION_RUN/native-build" -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$FUSION_RUN/native-build" -j 2
.venv/bin/python scripts/check_fusion_binary_discrete_native.py --operands "$FUSION_RUN/fixture/synthetic_operands.npz" --gguf "$FUSION_RUN/fit/fusion_candidate.gguf" --binary "$FUSION_RUN/native-build/w1ax_operator_replay" --output-dir "$FUSION_RUN/native"
.venv/bin/python -m unittest discover -s tests -p test_fusion_binary_discrete_independent.py
```

Use new output directories; existing artifacts are never overwritten. Full fit
reports retain per-flip objectives and per-scale transitions outside Git.
