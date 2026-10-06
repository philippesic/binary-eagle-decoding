# Two unselected production capture proposals

Both **captured-prefix 450** and **captured-prefix 750** remain
`PROPOSAL_ONLY_NOT_SELECTED`. This prepares the human's pending 450/750 TRAIN
exposure choice; it does not choose captured versus live student prefixes or
start capture, calibration, training, or admission. The continuing nine-model
goal and its original TRAIN/held-out ancestry are unchanged.

The CPU-only helper `scripts/prepare_block_production_capture_plans.py` produces
both options together, authenticates their original shard-zero selectors and
content/group joins through the actual production producer's `prepare_plan`, and
writes separate local and remote plans. It has no execution/GPU/SSH option.

| Bound / role | Captured-prefix 450 | Captured-prefix 750 |
|---|---:|---:|
| TRAIN prompts/domain | 150 | 250 |
| Calibration fit prompts/domain | 32 | 32 |
| Calibration validation prompts/domain | 16 | 16 |
| All original TRAIN-derived chains | 594 | 894 |
| Maximum native requests, including six goldens | 600 | 900 |
| Maximum TRAIN blocks/pass | 8,100 | 13,500 |
| Maximum TRAIN labels/pass | 56,700 | 94,500 |
| Raw retained features + logits | 64,950,183,936 B | 97,753,307,136 B |
| Capture retention, both family imports | 65,301,129,075 B | 98,190,412,275 B |
| Conservative capture/storage cap | 66,384,128,883 B (61.825 GiB) | 99,820,381,683 B (92.965 GiB) |
| Incremental capture free space, including 10 GiB reserve | 77,121,547,123 B | 110,557,799,923 B |
| Proposed wall cap | 7,200 s | 10,800 s |

Every chain uses the same proposed native geometry: at most 512 native prompt
tokens plus 128 greedy target tokens, at most 640 total rows. Full-chain F32
features have shape `[native_token_count, 5, 2560]`, with native layer-input taps
`[2,10,18,26,34]`. F32 teacher logits retain **all 151,936 vocabulary values** at
all potentially eligible seven-slot teacher positions. For a full-length chain,
the existing complete-horizon policy yields 18 anchors and 126 retained teacher
rows; native generation temporarily writes at most 129 rows before truncation.
The worst raw request is **111,166,976 B**: 32,768,000 feature bytes plus
78,398,976 temporary logit bytes. Request/shard caps add 262,144 B metadata margin
for **111,429,120 B** each. No prompt feature rows, potential labels, or vocabulary
values are removed to meet a cap. EOG, actual native tokenization and loss masks
can reduce realized blocks/labels; these upper bounds do not establish coverage.

The same raw capture is usable for DSpark's captured-prefix
`full_probability_l1` and DFlash's hard CE. Import creates distinct family
manifests, receipts, token files and admissions; raw F32 feature/logit files stay
shared. These captured teacher distributions cannot stand in for the different
current student prefixes required by the live DSpark path.

## Source ancestry and transport

The helper preserves the exact original selectors from ignored local evidence:

- `results/nine-model-qat-overnight/block-capture-options-20261005/{150,250}-train-per-domain-exact_soft.json`.
- `results/nine-model-qat-overnight/block-exposure-depth-options-20261005/evidence.json`,
  SHA256 `7e50d79353de6714011d0150d5e7d0ada117e0968a30dcab665444db860dbbd6`.

Within each domain, ascending original shard-zero rows with historical
`input_tokens <= 512` supply first 32 fit, next 16 validation, then next 150 or
250 TRAIN prompts. Existing ordinals, IDs, messages, content SHA256 and group
identities are authenticated; no new sample, relabeling, truncation, or held-out
promotion occurs. Historical input lengths are eligibility evidence, not proof
that the actual pinned native chat tokenizer will produce at most 512 tokens.
The native producer must reject an over-cap prompt without silently substituting
or truncating it; realized counts and any failure require review.

The historical corpus receipt stays byte-identical, including its ten original
TRAIN shards and other split metadata. A **new selector-scoped transport corpus
manifest** lists only original shard ordinal zero, keeps its original row
ordinals and file SHA256 values, and explicitly links the untouched original
corpus pin. This is a locator/inventory derivative, not a rewritten source
receipt or a claim that other shards or held-out files were transported.

Verified remote assets preserve local relative paths under
`/home/philip/binary-eagle-decoding/data/block-production-assets-20261006-01`.
The helper verifies the four actually required source files against the original
pins and the actual `verified-transfer.json` locator/hash joins. Transfer receipt
SHA256 is `9921b8815179df7902f865a8e24fc7ae480c637a3982b4cf379a211c5f2e9039`.
Local proposals authenticate original Mac files; remote proposals point to the
verified remote corpus/shard/index/target-source-pins bytes and current remote
client. Each proposal records both locator sets and original ancestry.

Native source is `ecff6d4e74814c631801df2e74f562d4ed6bd0eb`, SM120. Binary:
`/home/philip/binary-eagle-decoding/runs/build/native-indexed-ecff-sm120-20261005-v2/bin/llama-block-teacher`,
SHA256 `63eacbb8e600488c76122dfd29d2da168a8a371db7c244b2b10d90d1d21c478f`.
Target: `/home/philip/binary-eagle-decoding/models/gguf/Qwen3-4B-f16.gguf`,
SHA256 `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`.
Target/verifier/KV stay F16. Original tokenizer/template pins are joined against
byte-authenticated `target-source-pins.json`; the current native client pin is
`865e61948cbcf25c2df17d218842b332015668cf25cf47a69404cff1c35793c3`.

The strict `nine_model_train_capture_runtime_v1` schema has **no library fields**.
It remains unchanged. Separate proposal evidence carries the actual build-file,
CMakeCache, compile_commands and project library locator/hash inventory from
build-provenance SHA256
`b65884e17e8b0f293bb6a92178059d5d6ff4b5055f17c35f2ce7f4e6df2e17f9`.
Before any selected execution, the owner must rehash these actual artifacts and
join the resulting `NativeTeacher.native_runtime_binding` mapped-library hashes
against that inventory. Build PASS and primitive CUDA 272/272 PASS do not admit
this capture or any complete model. The failed oracle bookkeeping wrapper and
independent release reconciliation remain separate, preserved evidence.

## Operational caps and disk limits

Reversible proposed caps are 120 s/request, 12 GiB owner-plus-producer RSS,
4 GiB Linux `MemAvailable` floor, 10 GiB free disk, 8 MiB source inventory,
64 KiB source row, and 640 EAGLE-golden tokens. Wall caps budget 12 s per maximum
native request, with 120 s allowed for an individual request. These are bounded
operational proposals, **not measured rates or completion ETAs**. No capture
throughput measurement exists for this exact selected workload. A slow request
or whole-job expiry preserves failed/partial evidence rather than weakening the
scientific geometry or dropping samples to pass.

Production RSS/disk/time checks occur at operation boundaries, including after
imports; continuous monitoring exists only in the CPU-development execution
profile. These fields are not a hard process/cgroup limit, cannot prove
in-flight native/allocator/KV/importer peaks, and can detect an overshoot after
it happened. Both family imports audit memory-mapped full arrays; their peak
resident memory and open-file limits must be checked for 594/894 chains. A fresh
supervisor/hard-limit strategy is the sole operator's execution decision. Source
planning does not claim an enforceable continuous resource envelope.

Storage charges full features, full-vocabulary indexed logits and every chain's
pre-truncation peak, both token NPY copies, all seven indexed-map metadata copies,
256 KiB per-chain metadata, six golden payloads/metadata, 16 MiB global logs and
metadata, and authenticated source/runtime/client/derived-manifest bytes. The
23,862 B increase over the earlier 128-token report comes from the new runtime,
client and explicit derived locator manifest charge. Goldens alone reserve
160,932,864 B raw payload; feature/logit duplicates are zero for the current
shared-raw importer. Copying importer changes require a new larger envelope.

Parent's fresh UTC **2026-10-06 08:10** disk snapshot reports
**189,381,636,096 B free** (176.375 GiB). Each capture-only requirement fits that
snapshot; proposal disk status is `SNAPSHOT_SUFFICIENT_NOT_ADMITTED`. It must be
refreshed at actual launch. This does not admit the full campaign workspace.
The earlier gross four-block-model illustrations remain 209,845,220,011 B
(450) / 243,281,472,811 B (750), using a proposed three-generation checkpoint
policy. Exact existing target/base/reference reuse must be subtracted from gross
allowances before comparison with free space; currently present files must not
be counted twice. Existing EAGLE checkpoints and other runs remain protected.
Retention policy, future fits/initial artifacts, progressed checkpoints,
atomic-write reserve, export endpoints and logs need a fresh incremental ledger.
The 750 proposal remains explicitly infeasible whenever its actual incremental
requirement exceeds fresh free space; the helper reports `INSUFFICIENT` without
selecting 450 or dropping data.

## Reproduction and remaining gates

The final local ignored packet is
`results/nine-model-qat-overnight/block-production-capture-plans-20261006-05/`
in the primary workspace. Each `captured-prefix-{450,750}/` contains original
option bytes, `local-plan.json`, `remote-plan.json`, the two derived corpus
manifests, strict `runtime.json`, actual producer cost and `proposal.json`.
Build/transfer originals and historical depth evidence accompany it. Exact
plan/source/cost pins are in that packet; raw data and proposals stay out of Git.

Run this local CPU command from the frozen source checkout, replacing the output
suffix with a fresh unused directory for every invocation:

```sh
python3 scripts/prepare_block_production_capture_plans.py \
  --options-dir /Users/pippo/github/binary-eagle-decoding/results/nine-model-qat-overnight/block-capture-options-20261005 \
  --depth-evidence /Users/pippo/github/binary-eagle-decoding/results/nine-model-qat-overnight/block-exposure-depth-options-20261005/evidence.json \
  --source-root /Users/pippo/github/binary-eagle-decoding \
  --checkout-root /home/philip/binary-eagle-decoding/runs/checkouts/block-capture-plans-3142d43 \
  --asset-root /home/philip/binary-eagle-decoding/data/block-production-assets-20261006-01 \
  --build-provenance /Users/pippo/github/binary-eagle-decoding/results/twelve-hour-qa/block-runtime-proof/runs/checkouts/nine-model-qat-overnight-5f53740/runs/native-indexed-sm120-build-20261006-03/build-provenance.json \
  --verified-transfer /Users/pippo/github/binary-eagle-decoding/results/twelve-hour-qa/block-runtime-proof/data/block-production-assets-20261006-01/verified-transfer.json \
  --observed-free-disk-bytes 189381636096 \
  --output-root results/nine-model-qat-overnight/block-production-capture-plans-20261006-NEW
```

For root's remote CPU validation after source freeze, transport the packet
unchanged to the following project-relative location and run the producer's
**default planning command** for each option. The snippet reads the reviewed
plan pin from each original `proposal.json`; it executes neither a teacher nor
a model. All SSH/transport remains the root owner's tmux MCP responsibility.

```sh
python3 - <<'PY'
import json, subprocess, sys
from pathlib import Path
base = Path('results/nine-model-qat-overnight/block-production-capture-plans-20261006-05')
for count in (450, 750):
    root = base / f'captured-prefix-{count}'
    proposal = json.loads((root / 'proposal.json').read_text())
    subprocess.run([
        sys.executable, 'scripts/capture_nine_model_train_data.py',
        '--plan', str(root / 'remote-plan.json'),
        '--plan-sha256', proposal['plans']['remote']['sha256'],
        '--output-root', str(base / f'remote-validation-{count}-NEW'),
    ], check=True)
PY
```

Actual acceptance before capture: human exposure/conditioning choice; frozen
producer/client hashes and actual remote CPU path/schema/ancestry validation;
fresh sole-owner GPU release/lease/device/target/library evidence; fresh
incremental storage, host/VRAM/importer/FD limits and supervised stop procedure.
Actual capture then requires native prompt caps/EOG/realized role coverage,
authentic raw receipts, separate family admissions and fresh portability numeric/
decision evidence. Production fusion fits, initialized models, exports, complete
SM120 model/backward admission, training/checkpoint accounting, authentic Q4
controls and final-held-out authority remain later campaign work.

Checks actually run: **11** focused proposal CPU tests and **22** existing
synthetic producer CPU tests PASS; both real original selector proposals pass
actual `prepare_plan` on the local source snapshot. No remote validation, model
load, GPU operation, native capture, scientific selection or production-readiness
claim was made by this worker. No live process is owned.

Parent validation, October 6, 2026: the old primary remote checkout lacks the
teacher client. Packet04 remains historical and unlaunchable at that client
locator. Packet05 was regenerated with frozen checkout
`/home/philip/binary-eagle-decoding/runs/checkouts/block-capture-plans-3142d43`.
Both unchanged packet05 remote plans pass the actual producer's default CPU
planning command on the host, with `PENDING_NATIVE_CAPTURE` and no failures.
No model or teacher was executed. Parent11checks and independent11checks PASS;
independent original-byte and rehashed duplicate-group tamper probes reject.
The old report's statement that remote validation is unverified is superseded
by this dated parent evidence; actual capture/admission remains pending.
