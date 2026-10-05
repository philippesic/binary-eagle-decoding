# Deeper whole-chain block exposure options

Source-only advice at parent `50e6e38`, October 5, 2026. No GPU, model loading,
remote operation, capture, packet change, objective selection or training occurred.
These are proposals for human review, **not selected or admitted production data**.
This extends [block-capture-options.md](block-capture-options.md), retaining its
original source and role assignments. Healthy EAGLE A8 is unaffected.

## Recommendation for review

Consider **150 TRAIN prompts/domain with up to 128 new target tokens** as the
first bounded deeper-exposure candidate. It gives at most 56,700 distinct TRAIN
label positions/pass across 450 prompts, versus 12,600 with 32 new tokens, while
its conservative capture-plus-protected-model footprint is 99.307 GiB. This is a
more informative depth proposal, not evidence of sufficient serious training
coverage or convergence. It remains a subset of the original data; the project's
approximately 3.9M original EAGLE rows are a scale reference, not interchangeable
block labels or proof of equivalent exposure. Do not approve a long block run
solely from these bounds.

The alternative 250/domain × 128 offers 750 distinct TRAIN prompts and up to
94,500 labels for 130.447 GiB before progressed checkpoints, endpoint outputs,
builds and operating headroom. Under the explicit illustrative retention scenario
below it reaches 226.574 GiB. A fresh inventory may show reusable files, but the
historical free-disk figure is insufficient to admit it. At 150/domain × 256,
maximum labels rise to 113,400, but unique TRAIN prompts remain 450 and the same
scenario reaches 242.132 GiB. Prefer more genuinely distinct prompts when breadth
is the question; prefer deeper continuations when later target-trajectory coverage
is the question. Both require a scientific decision and realized exposure audit.
No rate, time-to-capture or ETA is inferred from file sizes.

## Exact ancestry and role counts

The same original shard 0 contains 1,000 unique prompt/content/group records.
Original input-token metadata admits 985 at `<=512`: 343 prose, 310 code,
332 reasoning. These are original tokenizer counts; native chat tokenization must
still admit each complete prompt. A prompt exceeding the native bound is rejected,
not truncated or silently replaced. No held-out EVAL rows are used.

Within each domain, use original ascending row order: first 32 calibration-fit,
next 16 calibration-validation, then next 150 or 250 TRAIN. The selections remain
identical at every continuation depth, permitting a controlled depth comparison.
Fit and validation each stay disjoint from TRAIN in ID, content and group.
Calibration-validation here is TRAIN-derived preparation validation, not final
held-out model evaluation. A final held-out evaluation remains separately required.

| TRAIN/domain | TRAIN | Fit | Validation | All selected | Unselected original shard rows |
|---|---:|---:|---:|---:|---:|
| 150 | 450 | 96 | 48 | 594 | 406 = 15 over original length cap + 391 beyond quotas |
| 250 | 750 | 96 | 48 | 894 | 106 = 15 over original length cap + 91 beyond quotas |

The ignored evidence contains explicit **selected and unselected** shard/row,
prompt ID, message-content SHA256, group, domain and original token count; selected
entries also carry role. Unselected entries carry exclusion reason. Unselected
means omitted from this copied shard proposal; it does not claim to inventory any
other original corpus shards. The original shard/index and corpus manifest pins
are the exact pins in the earlier report, reproduced in the evidence. No source
rows, messages, groups or role identities were invented.

## Existing geometry and full-context retention

Direct execution of `generated_block_anchors(512,512+G)` and
`block_teacher_indices(anchors)` from `src/w1a1_eagle/block_data.py` gives:

| New-token cap G | Complete blocks/chain | Retained teacher rows/chain | Temporary logit rows | Maximum full feature rows |
|---|---:|---:|---:|---:|
| 32 | 4 | 28 | 33 | 544 |
| 128 | 18 | 126 | 129 | 640 |
| 256 | 36 | 252 | 257 | 768 |

Anchors begin at native prompt length minus one and advance seven. Retained rows
are the union of all seven potential teacher positions for each complete anchor;
there is no future loss-mask filtering. At prompt length 512, these unions are
511–538, 511–636 and 511–762. Natural EOG/short completion reduces full horizons;
valid/loss/conditioning masks can further reduce actual supervised positions.
The table counts maximum distinct positions in a captured pass, not realized
labels. Repeated epochs reuse these positions and do not increase unique exposure.
All fit/validation captures are included in storage but excluded from TRAIN counts.

Every native prompt/generated feature row stays available: five taps
`[2,10,18,26,34]`, each 2560-wide F32, or `L*51,200` bytes/chain. Both imported
families retain access to this complete context; this proposal does not crop
context to the seven-slot horizon. Target/verifier and KV remain F16. Indexed
teachers keep **all 151,936 vocabulary logits in F32** at every selected row,
or `rows*607,744` bytes. This changes disk row selection only: full-head logits,
softmax and full-vocabulary L1 computation remain dense. No top-k, low precision
teacher or CE substitution is proposed. DFlash can consume its native hard-CE
objective from the same admitted raw chains without a second capture; DSpark
requires its declared full-probability L1 objective.

`capture_nine_model_train_data.py` production plan permits chains up to 32,768
with explicit prompt/generation/request/resource bounds. Only `development_CPU`
has 544-chain/512-prompt/32-generation/15-request pilot caps. Thus these production
lengths are representable; source permission is not fresh SM120 admission.

## Complete storage accounting and remaining admission

Current `block_data.import_capture_plan` references shared raw feature/logit
payloads and memory maps them, with two separate token NPYs per chain and separate
family manifests/receipts/admissions. Raw payload duplication is zero for this
importer; use a larger estimate if staged source differs. Every envelope below
includes both family token files (128-byte headers), all seven indexed-map copies
at `7*(64+16*teacher_rows)` bytes/chain, 256 KiB/chain metadata allowance, six native
portability goldens at the new full-chain cap, 256 KiB/golden, 16 MiB global
logs/metadata, and copied original source files. It conservatively charges the
native pre-truncation logit peak to **every** chain, although sequential capture
normally has only one live request. No feature rows are excluded from the cost.

| TRAIN/domain | New tokens | Max TRAIN blocks/pass | Max TRAIN labels/pass | Shared retained raw GiB | Capture envelope GiB | Plus protected models GiB | Illustrative total GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 150 | 32 | 1,800 | 12,600 | 24.822 | 26.801 | 64.283 | 160.410 |
| 150 | 128 | 8,100 | 56,700 | 60.490 | 61.825 | 99.307 | 195.434 |
| 150 | 256 | 16,200 | 113,400 | 106.477 | 108.523 | 146.005 | 242.132 |
| 250 | 32 | 3,000 | 21,000 | 37.359 | 40.263 | 77.745 | 173.872 |
| 250 | 128 | 13,500 | 94,500 | 91.040 | 92.965 | 130.447 | 226.574 |
| 250 | 256 | 27,000 | 189,000 | 160.254 | 163.233 | 200.716 | 296.842 |

Protected models retain the earlier **40,246,112,024 bytes (37.482 GiB)**:
F16 target, both original BF16 bases, four complete initial GGUFs, four projection
NPZs, four zero-update resumes, four fits plus four physical control FC NPZs,
both extracted FC/norm references, three original Q4 control size allowances and
metadata. All private embedding/head/attention/norm/Markov bytes remain included.
Original DSpark/DFlash Q4 availability is still unresolved; reserving sizes is
not evidence that authentic controls exist.

The last column adds an explicitly **illustrative, unselected** 96.126 GiB
operating allowance. It assumes four block lanes retain three progressed
checkpoints each, each `4,877,680,640 + 32 MiB = 4,911,235,072` bytes; one active
writer reserves one additional complete checkpoint before atomic publication or
pruning. It also reserves four trained GGUF+NPZ endpoints at current prototype
sizes (`14,315,288,440` bytes), one lane's additional temporary export
(`3,578,822,110` bytes), 8 GiB build/cache, 2 GiB later logs and 10 GiB free-disk
headroom. The latter three are proposed allowances, not measured usage or a
selected policy. This is a sequential-lane illustration, not permission for four
concurrent GPU jobs. A different retention/archival policy changes the number.
The separate retention owner is addressing implementation; this source snapshot
must not be described as already enforcing three-generation retention.

These are gross block-workspace allowances, not a fresh host disk or RAM
measurement. Existing EAGLE training/checkpoints, unrelated protected runs, target
build workspaces and other present bytes must be inventoried and protected in
addition unless exact overlap/reuse is verified. Do not add gross totals directly
to free space while also double-counting existing files: reconcile immutable
present bytes, genuinely incremental bytes and required free reserve. Fresh RAM,
VRAM/backward and teacher coexistence admission at the selected full context are
still required. No capture/fit/training throughput or quality claim follows.

## DSpark conditioning decision remains open

`block_qat.py` supports captured-prefix predecessors with stored full logits and
CE plus full-probability L1. It also supports native-greedy student predecessors:
without a live teacher, its static loss mask admits only the contiguous exact
prefix before first divergence, even if later tokens happen to match again.
With a teacher callback, the frozen native target is queried on each complete
current student prefix and yields fresh full logits. Indexed captured teachers
cannot stand in for those different-prefix distributions.

The production packet adapter is stricter: captured-prefix admits no live teacher;
`native_greedy` requires an explicitly admitted **DSpark** current-prefix teacher
with complete context bounds and numeric/device evidence. Thus offline captured
full-L1 and live current-prefix full-L1 are supported source paths with different
scientific conditioning and runtime costs. This report does not select between
them or imply live-path admission. Longer capture depth alone does not resolve
that choice. Final review needs realized prompts/blocks/labels by domain and role,
EOG lengths, masked fractions and repeated versus unique exposure, followed by
held-out native acceptance/latency/throughput against authentic Q4 controls.

## Evidence and checks

Preserved in primary main's ignored directory before worktree retirement:
`results/nine-model-qat-overnight/block-exposure-depth-options-20261005/`.
`calculate.py` reconstructs costs and extracts the actual two pure geometry
functions from source AST without loading a model. `evidence.json` SHA256:
`7e50d79353de6714011d0150d5e7d0ada117e0968a30dcab665444db860dbbd6`.
It records source module hashes, all six option counts/costs and exact selected/
unselected ancestry. Checks passed: prompt/index/content/group joins, stable
balanced role selections, distinct selected groups/content, selected+unselected
partition of 1,000 rows, actual geometry and arithmetic. The 32-token rows exactly
reproduce the prior report's storage figures. No production plan validator, test
suite, native capture, model admission or new experiment was run for this report.
