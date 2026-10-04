# Real TRAIN fusion binary fitting — October 3, 2026

**Completed real CPU fitting and full-model export.** On four prompt-disjoint
validation TRAIN groups, relative squared reconstruction error fell from
**0.074499 for converged scale-only to 0.054397 for fitted signs/scales: 26.98%
lower**. Every validation prompt improved. The candidate is stored separately;
QAT models, settings, processes, active goal and verifier remain unchanged.
Native acceptance/throughput against primary baseline Q4_0 remain deferred.

## Matched real-input results

384 original native F32 fusion input rows, 32 per prompt: 8 fitting groups /
256 rows and 4 independent validation groups / 128 rows. Fit only fitting rows,
then persisted and hashed candidate and controls before validation. Teacher is
raw F32 input × original BF16-promoted F32 FC weights using CPU F32 BLAS;
candidate arithmetic is the actual exported fixed-A8 integer-dot/F32 arithmetic.

| Method | Fit relative squared error | Validation relative squared error | Validation cosine | Coordinate signs | Largest-coordinate agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original BF16 signs + mean-absolute row scales | 2.137618 | 2.137201 | 0.667877 | 74.25% | 15.63% |
| Original signs + converged row scales | 0.075354 | 0.074499 | 0.578504 | 71.11% | 3.91% |
| Fitted signs + row scales | 0.049855 | 0.054397 | 0.665599 | 73.61% | 10.16% |

Direction/ranking measures improve over scale-only but remain slightly worse
than the original initializer. These are fusion-coordinate diagnostics, not
vocabulary decisions, accepted drafts or throughput. Lower reconstruction error
therefore does not establish improved native acceptance.

| Validation prompt | Scale-only relative error | Sign/scale relative error |
| --- | ---: | ---: |
| dolly:line-009429 | 0.072795 | 0.054314 |
| dolly:line-012832 | 0.076376 | 0.057681 |
| gsm8k:train-000312 | 0.075769 | 0.053618 |
| gsm8k:train-005241 | 0.073043 | 0.051965 |

Fit contains 3 prose, 3 code and 2 reasoning prompts; validation contains 2 prose
and 2 reasoning prompts, **no code validation**. Deterministic first 12 qualifying
independent TRAIN groups from the first completed 32-prompt shard, evenly spaced
accepted-prefix rows including prefill and later accepted positions. This is a
bounded sample, not a full-corpus or sealed-final result.

## Provenance and fixed contract

Saved artifacts were copied read-only through tmux MCP using the shared registry.
The successful transfer read 319,543,136 remote bytes under a 512 MiB per-attempt
cap and copied only the bounded package/proofs. Both owned transports closed.
No remote writes/source changes, new model capture, GPU/CUDA/Metal computation,
QAT messaging or live-process inspection was used.

The exact completed preparation-ready receipt (`bdfa56f8…`) binds full TRAIN
provider index (`7bc3f956…`), provider (`07e693c0…`), original capture (`b754ff9b…`),
semantic audit and continuous readiness (`cab889b9…`). Original capture remains
`training_eligible:false`, `preparation_only`; the separate readiness/full TRAIN
provider grant authorizes its use. Historical eligibility bytes were preserved.
All 384 selected bytes match original `heads.target_features.f32` offsets, with
prompt/group/content, target/draft/runtime, token-prefix, position, native decoded
and retained disposition, anchor/cache/mask joins checked. Native raw prefix IDs
are not mislabeled as the complete request prompt. No sealed-final values read.

Original capture producer is `b4e366d4f0a30cac07f14d51c54c5b1329b3f485`; native
CPU export checks use `9e2c7a90051e738751aab7d7bd7c2d8201fb76e3`. The initial
receipt misidentified that historical revision and was corrected before fitting.
Both source revisions confirm F32 target taps `[2,18,33]` concatenated before FC;
original base metadata has `norm_before_fc=false`. No cast/normalization precedes
A8 quantization. Target/verifier FP16 identities and captured ancestry are frozen.

A8: F32 absmax and reciprocal, nearest-even codes in `[-127,127]`, exact integer
signed dot, separately rounded F32 dot×row-scale×token-scale. Fixed clip=1,
positive original sign at either signed zero, little-endian packing, nonnegative
F32 row scales. No midpoint, residual, architecture, LR/loss sweep or QAT added.

## Solver, resource and export evidence

- All 2,560 scale-only rows converged: exact nonnegative scalar LS in F64,
  neighboring exported F32-scale objective checks; maximum normalized KKT
  residual `8.06153e-16`, zero improving finite neighbor gain. Every sign proposal
  is tested sequentially on the updated finite objective; conflicting stale
  proposals cannot be jointly accepted solely on their stale estimates.
- Two alternating passes / four scans / 32 accepted flips per row. 69,663 accepted
  improving flips produced **66,803 net sign changes** out of 19,660,800 weights.
  All 2,560 row scales changed versus initializer; 2,176 changed versus scale-only.
  Every scale update was non-increasing. 383 zero-scale rows cannot move through
  this fixed-scale flip search. **2,176 rows hit the flip cap**; no sign optimum
  or exhausted discrete search is claimed.
- Final full fit/export/reload completed exit0 in **5.39 s**, fitting itself
  2.389 s, Apple M3 Max / macOS arm64, NumPy 2.4.6 / PyTorch 2.14.0, two-thread
  environment. Measured maximum RSS **770,818,048 bytes (735.1 MiB)**, no swaps,
  below unchanged 1 GiB cap and 12-hour fit budget. Phase telemetry includes
  ancestry, all scans, freeze, validation and serialization. These are CPU run
  costs, not an accelerator/serving performance benchmark.
- Original fit/export peaked at **2,796,650,496 bytes (2.60 GiB)**, exceeding the
  estimate; an initial full-reader audit also peaked at 3,122,364,416 bytes.
  Both failed memory bounds are preserved. Expanded source/output GGUF metadata
  and temporary arrays caused the overhead. Bounded streaming export copies
  original KV and nonfusion tensor bytes with 1 MiB buffers and no tokenizer
  objects/mmap. Serialization-only check used 248,430,592 bytes; the complete
  corrected reproduction stayed below 1 GiB and produced **byte-identical NPZ,
  controls and GGUF**, with no optimization/data changes or third fit.
- Export is full EAGLE-3: **fusion W1A8**, other matrices remain original F16,
  norms/mapping retain original types. All 13 nonfusion tensor payloads and
  original KV bytes are unchanged. Header counts intentionally change from
  14→15 tensors and 36→48 KV records. Existing native v2 row/A8 representation,
  truthful nonnegative-LS scale metadata; no runtime/kernel/Gitlink change.
- Full native CPU model loading and tensor validation passed, hidden width 2,560,
  one decoder layer, zero GPU layers, no context/inference/target loading.
  Actual A8 native operator replay on eight authenticated validation rows checked
  **20,480 outputs with exactly zero error**, 63,979,520 bytes wrapper peak RSS.
  CPU library uses CUDA/Metal/BLAS/OpenMP OFF. Initial all-target build encountered
  unrelated app includes; the intended `llama` target built without source edits.

28 focused fitter/data tests plus 6 independent real-data tests passed; Ruff and
compilation passed. See [independent-validation.md](independent-validation.md)
and [summary.json](summary.json) for all exact input/source/config/model/export,
producer-chain hashes, commands, per-prompt metrics and retained evidence paths.
No fit process or owned transport remains; unrelated partial work is preserved.

## Candidate and reproduction

Primary raw root: `results/fusion-binary-real-a8-20261003/`.
Final artifacts: `fit-real-02/fusion_candidate.gguf` (387.05 MiB),
`fusion_candidate.npz`, `control_scales.npz`, `fit_report.json`; dataset/receipt
in `data-v2/`. Original over-cap run `fit-real-01/` remains preserved outside Git.

- Receipt: `75cbf2b83fdcc14b2dc338a5ac2fe27188a41cd6957fa311ea7d0e6bcd857c87`.
- Original BF16 source: `58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e`.
- Original F16 base: `c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1`.
- Candidate NPZ: `96ac80053802ae01ad6c7b9a8c328adbe4386da9d5807ebdca215c483373320b`.
- Candidate GGUF: `b9eae46c19b95ca904855c403c67b2b0d0d8134040b97c9c089e96ab50a68574`.

Local reproduction from preserved authenticated data, using a **new** output
folder (no more run is initiated by this report):

```sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
FUSION_DATA=results/fusion-binary-real-a8-20261003/data-v2
.venv/bin/python scripts/fit_fusion_binary_discrete.py \
  --manifest "$FUSION_DATA/fusion-manifest.json" \
  --provenance-receipt "$FUSION_DATA/provenance-receipt.json" \
  --provenance-receipt-sha256 75cbf2b83fdcc14b2dc338a5ac2fe27188a41cd6957fa311ea7d0e6bcd857c87 \
  --source-weights models/hf/Qwen3-4B_eagle3/model.safetensors \
  --base-gguf models/gguf/Qwen3-4B-eagle3-f16.gguf \
  --output-dir results/fusion-binary-real-a8-reproduction
```

Exact acquisition and CPU gate commands, native source/library flags and hashes
are retained alongside the package/command records. Data adapter `efbaeb6`,
real importer `1eda427`, memory fix `382ced5`, independent audit `40ecd45`.
The [earlier synthetic report](../fusion-binary-discrete-a8-2026-10-03/README.md)
is historical. Neither synthetic nor real fitting initializes the current QAT
run. Native acceptance/depth survival and throughput against Q4_0 require later,
separately authorized GPU availability.

## CPU permission resumed — October 3, 23:22 PDT

The human requested “resume cpu work” in this supporting chat. CPU permission is
recorded separately in the shared machine-local `cpu-control.json`; this team
does not acquire a GPU slot or restart an experiment. Calibration and its CPU
checks are complete, so no unfinished CPU fit needs restarting. The separate
A8 goal records an independently authorized “resume gpu usage” at 23:21 PDT;
that owner and its GPU-control flag are left unchanged. Fusion native GPU
acceptance/throughput still require separate authorization and coordination.
