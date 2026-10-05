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

Root also approved a narrow model-bound post-norm reference adapter.
`scripts/extract_block_fusion_reference.py` hashes an original dense full-
vocabulary block GGUF once, uses the existing bounded streaming header parser,
and extracts only FC/gamma F32 NPY files plus `block_fusion_reference_v1` metadata.
Tokenizer arrays are skipped rather than expanded. Source model/FC/gamma
payloads and extracted files are SHA-bound; the native norm scalar must be an
actual GGUF FLOAT32 with its exact canonical bits recorded. Existing binary,
pruned, scaled/biased fusion, unsupported tensor formats and insufficient array
bounds refuse explicitly. This is an original-model reference extractor, not a
full model-readiness check.

`fit_block_fusion.py` now requires the pinned `--norm-metadata` descriptor and
checks supplied epsilon by exact F32 bits, plus family/weight/gamma/shape joins.
No broad tolerance is used; different double spellings that round to the same
native F32 value are identical. Per-precision fitting reads the extracted NPY/
descriptor rather than rescanning the original GGUF. Historical EAGLE reports
remain untouched. Eight CPU tests use actual small synthetic GGUF payloads to
exercise round-trip, one-ULP mismatch, wrong source/gamma/family/pruned models,
scalar type/range, extraction caps and overwrite refusal. Actual block-model
reference extraction/fitting remains PENDING local artifacts or later authorized
operator access. No remote operation or model context was used.

## Actual original block references — Mac materialization

The native owner then materialized and audited both original model snapshots
locally. Their FC payload is **BF16**, so `6156200` extends the bounded parser to
GGML kind 30 and decodes BF16 with exact raw U16→U32 bit shift into F32, without a
lossy F16 round trip. A real small BF16 GGUF byte oracle passes alongside the
F16/F32 cases. Gamma is F32 in both actual original models; originals and the
historical EAGLE artifacts remain untouched.

Both actual pinned source models were extracted and revalidated on Apple M3 Max
CPU without a model context, optimizer or remote operation. Ignored outputs:
`results/nine-model-qat-preparation/block-fusion-references-20261004/{dspark,dflash}/`
contains `fc-weight.npy`, `fc-norm.npy`, `reference.json`; the shared
`preparation-report.json` records exact model/payload/file pins and scope.

| Reference | DSpark | DFlash |
|---|---|---|
| Original GGUF | `3685ea7785729b4d3e9de3939264c1c87eb98a2bf7935a8c838eccf36be9fd13` | `61b294dc798c507fdc4a87f397b4015a78b3cdd7edad1e940e683bba10fb5575` |
| Metadata | `455caaf63bff36821233abfde5a88cf9607e6e5a138457666cfce21fd264a3c4` | `ca1bcb35755f394ceea1b30008176a375ae78aa4b38d7698da8ca2462e1fe9ec` |
| F32 FC NPY | `5ef5ce45c065f85d6c730ea8eea198e6d9c0f2fe7e92c83ae8504aecca3e1a86` | `a40c8492ac447ab4a2991bd86f219ad7e1a1778d792d0c09c7faf86b278239be` |
| F32 gamma NPY | `e618c94c0feb35fe13d7c4a6f7edc34da13e17b24e7098822bafa92c9279714c` | `ce40f6da96b3a095ca4066c0a3d06e80d4c1a149739642d79b864cb5e72969af` |

Both FC references are F32 `[2560,12800]`, gamma `[2560]`, 131,082,240 combined
array bytes; source-zero FC magnitudes are zero. Both model norm epsilons are
canonical F32 `9.999999974752427e-7`, bits `897988541`. Extractor maximum RSS was
412,680,192 bytes (DSpark) and 414,826,496 bytes (DFlash), below 396 MiB. Actual
weight/gamma/epsilon provenance is now materialized rather than a missing
dependency. **Fusion fitting still requires authentic five-tap TRAIN captures**,
which remain PENDING while GPU use is paused.

For the approved pilot's three calibration groups × 32 rows, use
`--max-total-rows 96 --max-array-bytes 1073741824`: conservative fitter workspace
estimate is 833,617,920 bytes (795 MiB), not a process-RSS guarantee. Include the
existing source-bound completed admission to avoid full teacher payload audits
per precision. All applicable CPU checks currently pass: 53 package tests,
19 capture tests and 20 historical discrete-fitter tests; actual CUDA remains
unverified under the pause.

## CPU pilot generated-chain protocol checkpoint

The approved Mac-only capture attempts 01 and 02 are preserved as FAIL artifacts
under `results/nine-model-qat-preparation/development-cpu-pilot-capture-20261004-{01,02}`.
Each produced one authentic original TRAIN chain (21 native prompt tokens plus
32 target-only greedy tokens), five native taps and full-vocabulary logits.
Both producers closed and were reaped. Attempt 01 consumed 13.776090708 seconds;
attempt 02 consumed 32.369236750 seconds. The original cumulative 1800-second
capture budget has 1753 whole seconds remaining; another capture requires root
review and GO. No fusion fit or optimizer update ran.

Attempt 01 exposed the native CPU hardware description protocol; the corrected
guard requires exactly the queried CPU brand, CPU result/storage buffers, the
CPU-only build and exact loaded project library closure. Passive transitive
system Metal.framework is recorded separately and does not grant GPU execution.
NativeTeacher's Mac runtime mapping remains explicitly unchecked; actual DYLD
path/hash evidence supplements it without rewriting that field.

Attempt 02 exposed a shared CPU/CUDA metadata bug: generated receipts correctly
declare `native_tokenized_prompt_then_target_only_greedy`, while the controller
and importer expected an exact-prefix replay. Generated chains now have a
separate validation path for original prompt bytes, native tokenizer flags and
pins, rendered/template hashes, greedy termination and the exact prefill/greedy
decode boundary. The importer retains client/tokenizer/template identity from
the pinned runtime. Replay goldens retain their strict original contract, and
generated receipts cannot be relabeled as replay.

The bounded offline audit consumed both original native receipts and raw files,
validated CPU DYLD closure, imported the single chains and read exact seven-slot
full-vocabulary batches. It launched zero producers, changed neither failed
report nor original receipt, and grants no production or CUDA readiness:
`results/nine-model-qat-preparation/development-cpu-generated-protocol-audit-20261004/report.json`.
The source hashes and new report pin are reported with the reviewed commit.
