# Reviewable block capture and exposure alternatives

Owner `/root/overnight_eagle_data`; isolated worktree
`/private/tmp/nine-model-qat-20261004/overnight-block-capture-options`, source
`f6671e3aa8feced085805094f4faa214512dabca`, native gitlink
`ecff6d4e74814c631801df2e74f562d4ed6bd0eb`.
This is local source/JSON/stat inspection only: no remote/2080/GPU/model loading,
capture, fit, training, budget selection or changes to code. Healthy EAGLE A8
continues unchanged. Both alternatives are **PENDING scientific exposure review**;
counts alone do not establish serious coverage, convergence or readiness.

## Genuine source selection

The copied original TRAIN shard has1,000 records. All prompt IDs/domains and
canonical message-content hashes were joined to the original opaque index;
all1,000 group IDs and content hashes are distinct. Filtering the original
`input_tokens <=512` gives985 records:343 prose,310 code,332 reasoning.
These are original tokenizer metadata counts, not fresh native token counts;
actual native512-token admission remains required.

Both alternatives have sufficient real unique groups/content. First32 eligible
records per domain are calibration-fit, next16 validation, then next150 or250
TRAIN. Calibration is identical between alternatives; the150 TRAIN subset is
nested in250. Roles and domains are interleaved for capture. The existing dataset
shuffles/interleaves whole chains per epoch and consumes their blocks in order.
Every explicit selection includes original shard ordinal0, source row, prompt
ID/content SHA/domain and role; additional evidence preserves group/index joins.

| Alternative | TRAIN/domain | Fit/domain | Validation/domain | All chains | TRAIN chains | Maximum TRAIN blocks/pass | Maximum label positions/pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| Smaller |150|32|16|594|450|1,800|12,600|
| Larger |250|32|16|894|750|3,000|21,000|

Original TRAIN input-token totals are63,171/105,829 respectively; ranges18–464
and18–492. Fit totals13,702 tokens (20–392), validation6,591 (20–371).
These are prompt lengths, not supervised exposure. Natural EOG, early capture
failure and conditioning masks can reduce realized block/label counts. Neither
alternative changes the existing seven-slot horizon, anchors or prefix masks.
A32-token continuation gives at most four complete blocks per chain; repeatedly
training these12,600/21,000 positions is not new unique exposure.

Source files in primary main's ignored
`results/nine-model-qat-preparation/development-pilot-source-20261004/`:

| File | SHA256 |
|---|---|
| `corpus-manifest.json` |`9fc8caa80f2c29785eeaeff01f3875c27fee46853350de9ceebe682f8d12b8dc`|
| `train-00000.jsonl` |`0aed2973080abe253e9ca5ca773e791c62c706ae4d0b9b6893524bd3cfba15c8`|
| `train-00000.index.jsonl` |`35762cd62319b648f99e66c453c3aaba540a14d463d2e37283ea20943d7807cf`|
| `target-source-pins.json` |`8f2b1c0d1a2b75165d702120a5b499c77a56a5a2f1e4c9fa3df68e483d8fd315`|

## Full-chain geometry and indexed teachers

Each request retains all native context/generated feature rows: maximum
512 prompt +32 greedy tokens =544 rows ×5 taps `[2,10,18,26,34]` ×2560 F32.
This is27,852,800 feature bytes per maximum-length chain. Target/verifier/KV
stay F16. Private full vocabulary is151,936, MASK151669, seven author slots.

The actual `generated_block_anchors(512,544)` returns511/518/525/532.
`block_teacher_indices` is the full union of all seven potentially supervised
rows per anchor:28 absolute positions511…538. Indexed `exact_soft` retains
all151,936 F32 logits for **every** one of these28 rows (17,016,832 bytes).
It does not filter using future valid/loss masks or omit an anchor. Only rows
outside all existing complete-horizon teacher windows are excluded. Native
writes up to33 candidate rows before truncating the incomplete horizon;
20,055,552 logit bytes are therefore reserved for a request's temporary peak.

DFlash hard-CE may omit per-chain full logits, retaining the same full features,
tokens, anchors and hard labels plus full-logit portability goldens. This
alternative does **not** satisfy DSpark `full_probability_l1`. Indexed exact-soft
can share its same raw payload across DSpark full-L1 and DFlash hard-CE consumers;
a second hard-CE recapture is unnecessary when those source/split/prefix joins
are identical and admitted. Objective/conditioning selection remains explicit.

## Storage includes the real imports and protected artifacts

Current `block_data.import_capture_plan` is zero-copy for raw feature/logit
payloads: family manifests reference the same original `.f32` paths; `_array`
memory-maps them. It writes **two separate token NPYs per chain**, one per family,
and separate family manifest/producer-receipt/stat-bound completed admission.
It no longer creates two full feature/logit NPY copies. This was checked in the
actual current importer, not inferred from historical pilot directories.

Costs include shared raw files, BOTH family token-NPY copies (128-byte headers),
six full-context native block/EAGLE golden raw captures (137,339,904 bytes),
256KiB per chain for raw/client/external receipts plus both imported metadata
sets, all seven persisted indexed maps,6×256KiB golden metadata,16MiB global
logs/metadata allowance, and copied original source files. The conservative
capture envelope charges the33-row temporary logit peak on every chain, although
sequential production normally has only one active request; it is an upper
allowance rather than an observed disk/RSS peak.

| Alternative | Shared retained raw GiB | Complete capture envelope GiB | Common protected preparation GiB | All prelaunch allowance GiB | If two full family NPY copies were used GiB |
|---|---:|---:|---:|---:|---:|
|150/domain indexed exact-soft|24.822|26.801|37.482|64.283|113.928|
|250/domain indexed exact-soft|37.359|40.263|37.482|77.745|152.463|
|150/domain DFlash hard-CE|15.408|15.704|37.482|53.186|84.003|
|250/domain DFlash hard-CE|23.190|23.562|37.482|61.044|107.425|

The final column explicitly budgets legacy two-family full payload/header
copies in addition to the raw files. It must not be silently replaced with the
zero-copy column if another importer/runtime is staged. Common protected
allowance conservatively reserves all four block candidates/controls even for
the DFlash-only data alternative; it does not imply DSpark data is available.
It includes existing F16 target8.051GB, both BF16 bases5.429GB, four complete
initial GGUFs7.812GB and full-projection NPZs6.503GB (actual CPU prototype size
references), four zero-update resume allowances6.639GB, four FC fits **and four
physical control NPZs**1.049GB, both extracted FC/gamma references0.262GB,
three original Q4 control files4.484GB, and16MiB model metadata. Each resume
allowance includes32MiB overhead. New production artifact hashes remain pending.
Private embedding/head/attention/norm/Markov bytes are included in the complete
model files; they are not omitted or borrowed to shrink this budget.

This is a conservative total footprint assuming these dependencies must all be
staged. Already present, exactly verified files can reduce incremental allocation;
no current remote inventory/free-disk claim is made. Indexed CUDA build/cache
storage and later trained endpoints/checkpoints are additional. Reported historic
~199GB free space is not a fresh admission or permission to consume that amount.

**Checkpoint retention is a separate concrete gap.** Current block CLI calls
`save_block_checkpoint` at each cadence with a unique path and has no generation
pruning. A progressed FP32 block resume payload is4,877,680,640 bytes before
metadata (sign/scale state plus full Adam moments). Merely assuming the EAGLE
three-generation policy would be false. Three such checkpoints would require
14,733,705,216 bytes with32MiB/file overhead, but three is neither implemented
nor selected for blocks. Long block QAT needs explicit retention/total-disk
policy and resource admission; this report changes neither trainer nor budget.

## Preserved proposals and next actions

All four exact selections/cost breakdowns, module Git/SHA bindings, shape/exposure
bounds and non-executable capture-plan drafts are preserved in primary main's
ignored directory (so retiring this worktree does not delete them):

`results/nine-model-qat-overnight/block-capture-options-20261005/`

| Proposal | SHA256 |
|---|---|
|`options-index.json`|`37d7ea82d3c404260b01e12b55a713b45fe1005dc7e2fe8c1fa19d15e13bd0b1`|
|`150-train-per-domain-exact_soft.json`|`f27cd74b21c35352a383085dfc8770aa55d7961142c5e5b033358da6f20ded55`|
|`250-train-per-domain-exact_soft.json`|`854cef9c90bd7dacd970240891c77849c7b253aad3ad9c37732876812540b8b5`|
|`150-train-per-domain-hard_ce.json`|`713fcfcde18de81d80d5bb57e85da2278184fbdb714a4bf5863bbce1ce977d89`|
|`250-train-per-domain-hard_ce.json`|`8f6a45e122f3d7562e05be221030315061d1b4cddb0cc92134379da94ee4ca13`|

Draft runtime/binary and time/RAM/disk execution caps are explicitly null/PENDING;
there is no invented throughput/rate/ETA, real capture admission or start command.
Next: human/coordinator reviews unique exposure versus continuation length and
chooses a serious coverage/objective/conditioning option; source owner binds
actual indexed SM120 build/client/runtime/tokenizer pins; sole operator measures
bounded capture rate and sets reviewed caps after an available GPU slot; then
runs actual capture/import/goldens, validates both destination admissions,
performs separate production fits/initial exports and fresh model/backward/memory
admission. Healthy EAGLE A8 must not be interrupted to make that slot available.
Address block checkpoint retention before any long block launch.

Verification here: actual1,000-row source/shape/index/content/group audit,
quota/disjointness and nested-selection assertions, actual anchor/map functions,
exact byte arithmetic and existing file-size references. No model or production
validation is claimed; no new test suite was run for a JSON/report-only task.
All source rates, serious-coverage selection and execution readiness remain
PENDING. This feature has no live jobs or remote/session handles.
