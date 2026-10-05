# Data and fusion implementation checkpoint

Owner `/root/data_fusion`, assigned worktree
`/private/tmp/nine-model-qat-20261004/data-fusion`, branch
`prep/nine-model-data-fusion`. Root alone owns active STATUS/goal/decisions and
integration. No GPU/SSH or real-model optimizer update was performed by this
owner or its short capture orchestration worker.

Actual source commits, in order:

- `b8d0ff4`: five-tap/full-vocabulary block contracts, whole-chain cursor,
  deployed fixed A8/A1 calibration/control/rescue, bounded preparation/fit CLIs.
- `43a8404`: original raw native receipt importer, full initial audit and
  source/manifest/host-file-identity bound reuse, historical EAGLE CPU fit report.
- `4f49ef4`, `38e718c`: actual CUDA execution-buffer/device/numeric portability
  gate, bounded new EAGLE three-tap TRAIN golden receipts. Synthetic injected
  cases are always explicitly fixture evidence, never production admission.
- `361c627`: native prefill/greedy partition and F16 KV history validation/replay;
  cropped prefixes preserve every original partition before their boundary.
- `90877ea`: capture worker `b18bf72` integrated after review; original TRAIN
  corpus/shard/row/content/index/domain/group joins, native tokenizer/template
  generation, one persistent target-only producer, bounded raw storage and
  separate shared block/EAGLE golden captures.
- `7c6b6b1`: calibrated hard bits separated from explicit latent initialization;
  preserve block source magnitudes/EAGLE fixed `0.5`, unit probe off by default.
- `3043462`: capture worker `5011c13` reviewed/integrated; exact runtime/client
  source joins, Linux MemAvailable floor, bounded selected JSONL row streaming.
- `fb67e0b`: capture worker `a1917d9` reviewed/integrated; explicit native prompt
  token cap forwarded before decoding; old API refuses before model construction.

Local checks: 43 data/fusion/portability tests, 19 capture orchestration tests,
20 existing discrete-fitter tests, all owned Ruff checks pass. Tests exercise
success/refusal/corruption/cursor/teacher bounds/partition/STOP/cleanup/device/
source/runtime/tokenizer/latent-policy cases. The independent QA owner reviews
integrated APIs and real saved calibration artifacts separately. The worker's
actual native CPU negative integration correctly refused F32/CPU as CUDA/F16,
then closed/reaped its producer. This is source-contract evidence only.

Actual ignored EAGLE CPU A8/A1 calibration artifacts and controls are recorded in
[the report](eagle-fusion-cpu-20261004.md). Row orientation improves raw SSE but
worsens post-norm diagnostics, so it remains an off-by-default probe. Original
historical unit latents remain intact; new reference-half siblings preserve
all deployed bits/scales without refitting.

Root approved a **development-only** pilot: nine original TRAIN prompts,
three domains × train/calibration-fit/calibration-validation, six bounded new
EAGLE/block golden captures, maximum 512 native prompt tokens and 32 new tokens,
544-row chain bound, 15 native calls, total retained disk cap 8 GiB. This is
not serious long-QAT coverage, quality selection or an optimizer budget.

Actual original source packet and nine explicit row/content selections are at
ignored `results/nine-model-qat-preparation/development-pilot-source-20261004/`.
Copied corpus/shard/index bytes match the independently pinned historical
receipt `75cbf2b8…`: manifest `9fc8caa8…`, prompts `0aed2973…`, index `35762cd6…`.
Selected historical prompt lengths are 25–123 tokens; actual native templating
must enforce the new 512-token cap. Exact-soft upper bound is 3,226,189,824 bytes
for nine block chains plus 137,339,904 bytes for six goldens, before source/
receipt headroom. Hard CE can omit full-logit chains while retaining bounded
full-vocabulary golden rows. A corpus of 3.9m F32 full-vocabulary teacher rows
would cost roughly 2.37 TB; no such coverage is selected or silently allocated.

Remaining actual preparation work is explicit: sole operator pins current
native source/binary/client/runtime/tokenizer/template and libraries, executes
plan-only then root-reviewed bounded capture on the newly opened RTX5080;
validates data, fits both block-family fusion precisions using actual frozen
weights/norms, and runs fresh native portability/model/operator checks. Native
producer must include the companion pre-decode prompt cap API. Keep natural
EOG/STOP/cap outcomes and failures. Cross-family raw files can be shared; each
family manifest has its own initial audit, with per-family process reuse.

Human campaign coverage/objective/recipe/budget choices and real training or
quality/performance evaluation remain pending. Existing Q4 controls and EAGLE
speculative/tree corpora are preserved. New target-only golden prefixes do not
relabel or replace those corpora. No CPU/fixture result grants SM120 memory,
convergence or throughput readiness.

## Mac-only continuation and finite A8 contract

The human paused all RTX5080 usage after the development opening. Root owns the
pause and sole operator cleanup; this package performs no remote operation.
Actual pilot captures remain PENDING. Existing code is integrated through main
`658bead`; local source and saved-artifact checks continue within the same goal.

All four actual fixed-half EAGLE NPZ files passed the integrated
`qat_initialization.apply_binary_initialization` consumer on Mac at main
`658bead`: exact F32 scales and latent values, reference SHA `9b24af26…`, zero
optimizer updates. Original unit magnitudes refused the preserved-reference
policy and passed only the explicit unit probe. This was actual saved projection
artifact integration, with no model forward/backward or CUDA operation. Maximum
RSS was 668,663,808 bytes. Detailed evidence remains outside Git in
`results/nine-model-qat-preparation/calibration-initialization-integration-mac-20261004.json`.

Root approved a narrow fixed-A8 finite reciprocal correction shared with native
and training owners. Prepared fitter arithmetic is now
`fixed_w1a8_finite_reciprocal_v2`: bitwise F32 absolute maximum; finite F32 inverse
and multiply/RNE unchanged; reciprocal overflow uses
`double(x)/double(absmax)*127`, casts normalized values to F32, then rounds
nearest-even and clamps. This matches the specified CPU/native CUDA expression;
the new actual CUDA execution check is still PENDING while paused. Half-tie,
max-subnormal, signed-zero and normal-domain source-oracle tests pass locally.

The independently hash-joined original operand archive (`04ad9777…`) supplied
all 384 saved TRAIN rows/2,949,120 values for a bounded CPU compatibility check:
**zero changed codes and zero changed token-scale bits** versus the historical
fixed-A8 quantizer. No fitting or model execution was repeated, and old unit/
fixed-half NPZ files, reports and their source tags remain unchanged. The new
fitter source tag/hash is separately recorded in ignored
`fixed-a8-v2-saved-normal-domain-mac-20261004.json` (270,499,840-byte maximum RSS).
