# Bounded native capture storage for the frozen W1A 2k train split

CPU-only Phase 1A plan, prepared 2026-09-28. No model inference, accelerator
or remote check was run. No legacy or new final prompt contents were opened.
`scripts/prepare_w1ax_capture_shards.py` reads only the frozen candidate
`train_small` JSONL/index and parent manifest, then writes ignored prompt
shards, pinned metadata and a fail-closed observed-cell storage gate. It
never deletes raw logits.

## Frozen input and generated plan

The parent candidate freeze manifest SHA256 is
`dc37f752bb137054183162dcb7ca96004edabeb752bd611996d9cb289bfdc667`.
Its `train_small.jsonl` SHA256 is
`bfbe292fc8c19f4b4394dbcb6577140cbabf435bd530648adedaba71845ea647`,
and train index SHA256 is
`74f0e920830020146e6546e4e375e759710a37c4346dc6b4e8c0f62ac2fb469b`.
The new ignored output is `data/w1ax-capture-shards/plan-003/`; `plan.json`
SHA256 is `a20a9f8e3a48c65dfc754c0598c1c256f77c9754d874caa04dabcb60b4095c1d`.
It has schema `w1ax_capture_shard_plan_v1`, an ordered list of 65 child
`w1ax_capture_shard_v1` manifests, every parent train prompt ID in frozen
order and its SHA, all child IDs/paths/hashes/counts, and full coverage hash.
The first child is `shard-0000/manifest.json` (SHA256
`17b8c65c47b335449e7573e42ec644f0682a7b5dfb97bb1874f1bc8eb05cd8b6`):
31 prompts, estimated 16,143 verifier-logit rows and 9,810,811,392 raw bytes.
The parent manifest/index hashes and source positions are repeated in each
child. Prompt JSONL bytes are copied exactly from parent train rows; IDs and
message-content hashes are checked against the frozen train index. Conversation
and topic families stay within one shard. These hashes bind the planned
source; they do not certify the prompts as training-ready.

Reproduce into a new ignored output directory:

```sh
python3 scripts/prepare_w1ax_capture_shards.py plan \
  --parent-manifest data/w1a-public-freeze-002/manifest.json \
  --output data/w1ax-capture-shards/plan-003 \
  --split train_small --target-vocab 151936 \
  --max-prompts 96 --max-rows 16384 --max-raw-bytes 12884901888 \
  --base-rows-per-prompt 512 --rows-per-word 1
```

The output directory must be new. ID-list hashes use SHA256 of UTF-8 compact
JSON (`ensure_ascii=False`, sorted keys, separators `(',', ':')`) with no
newline. A training provider can verify the ordered plan and then bind each
ordinal to one separately audited native bundle/teacher. The first 100-step
calibration can use one eligible shard; training across all 2k prompts needs
the QAT provider's planned multi-shard iterator with one optimizer state.

## Row and byte bounds

A full target F32-logit row at vocabulary 151,936 occupies 607,744 bytes.
The inherited 96-prompt capture had 40,815 rows, or 24,805,071,360 bytes
(24.81 GB / 23.10 GiB). At the same rows per prompt, a 2k capture would be
about **516.77 GB** of raw logits. The planner uses at least 512 estimated
rows per prompt, increased for prompts over 512 words. Its current total is
1,041,989 estimated rows / **633.26 GB**; this is a planning number, not a
native observation. Actual acceptance and request lengths can change it.

Every child has at most 96 prompts, 16,384 estimated/observed rows and
12 GiB raw bytes. The generated plan's largest child has 32 prompts and
16,384 estimated rows, or **9.96 GB** raw. The planner fails if one declared
conversation/topic family alone exceeds the prompt, row or byte cap. The
native runner must use `--target-logits-limit 16384`; after capture, the
`observe` command checks actual source-cell request ownership, contiguous
logit ranges, raw SHA/size and both hard caps. A captured shard over cap fails
closed and cannot advance. The estimate alone is never evidence of safety.

The existing bundle builder copies the full raw file into the bundle. At the
maximum planned shard, capture raw plus bundle copy transiently need about
**19.91 GB**, before feature arrays, server files and filesystem overhead.
Keeping both raw copies for all shards would approach **1.27 TB** at the
planner estimate. The current bundle schema therefore does not solve retained
storage by sharding alone. This plan caps peak shard raw storage but leaves
raw retirement disabled pending a verified compact-evidence handoff.

## Native command sequence and current interface gate

The capture, row/feature preparer, bundle builder and audit CLIs now accept a
manifest-aware path (`a18e5a9`) with runner `--shard-manifest` and paired
`--expected-prompt-sha256`/`--expected-prompt-count` flags across those tools.
The commands below require pinned model/map files and variants plus explicit
restored GPU access. The first shard values are shown;
repeat from the ordered plan rather than discovering directories by glob.

```sh
SHARD=data/w1ax-capture-shards/plan-003/shard-0000
PROMPT_SHA=968ffbb21b23f934912862ef6f7add7d03bf0cfbbb3910492f2f0f50075fc18a
PROMPT_COUNT=31
RAW_RUN=runs/w1ax-capture-shard-0000
ROWS=data/w1ax-capture-shards/prepared-0000/rows
FEATURES=data/w1ax-capture-shards/prepared-0000/features
BUNDLE=data/w1ax-capture-shards/bundles/shard-0000
TEACHER=data/w1ax-capture-shards/teachers/shard-0000

python3 scripts/run_binary_head_capture.py --mode recurrent-train \
  --binary "$SERVER_BINARY" --target "$TARGET_GGUF" \
  --prompts "$SHARD/train_prompts.jsonl" --variants "$VARIANTS_JSON" \
  --output "$RAW_RUN" --d2t "$D2T_NPY" --target-vocab-size 151936 \
  --shard-manifest "$SHARD/manifest.json" \
  --expected-prompt-sha256 "$PROMPT_SHA" --expected-prompt-count "$PROMPT_COUNT" \
  --target-logits-limit 16384 --target-features-limit 131072

python3 scripts/prepare_w1ax_capture_shards.py observe \
  --shard-manifest "$SHARD/manifest.json" \
  --cell-manifest "$RAW_RUN/d_d/manifest.json" \
  --output "$RAW_RUN/shard-observation.json"

python3 scripts/prepare_recurrent_native_rows.py \
  --heads "$RAW_RUN/d_d/heads.jsonl" \
  --rounds "$RAW_RUN/d_d/forced-rounds.jsonl" \
  --task-map "$RAW_RUN/task_prompt_ids.json" \
  --cell-manifest "$RAW_RUN/d_d/manifest.json" \
  --prompts "$SHARD/train_prompts.jsonl" --absolute-d2t "$D2T_NPY" \
  --target-vocab-size 151936 --target-logits "$RAW_RUN/d_d/heads.target_logits.f32" \
  --expected-prompt-sha256 "$PROMPT_SHA" --expected-prompt-count "$PROMPT_COUNT" \
  --output "$ROWS"

python3 scripts/prepare_recurrent_native_features.py \
  --native-jsonl "$RAW_RUN/d_d/heads.target_features.jsonl" \
  --native-f32 "$RAW_RUN/d_d/heads.target_features.f32" \
  --anchors "$ROWS/anchors.jsonl" --task-prompts "$RAW_RUN/task_prompt_ids.json" \
  --cell-manifest "$RAW_RUN/d_d/manifest.json" \
  --train-prompts "$SHARD/train_prompts.jsonl" \
  --expected-prompt-sha256 "$PROMPT_SHA" --expected-prompt-count "$PROMPT_COUNT" \
  --output-dir "$FEATURES"

python3 scripts/build_recurrent_capture_bundle.py \
  --rows-dir "$ROWS" --features-dir "$FEATURES" \
  --target-logits "$RAW_RUN/d_d/heads.target_logits.f32" \
  --cell-manifest "$RAW_RUN/d_d/manifest.json" \
  --train-prompts "$SHARD/train_prompts.jsonl" \
  --shard-manifest "$SHARD/manifest.json" \
  --expected-prompt-sha256 "$PROMPT_SHA" --expected-prompt-count "$PROMPT_COUNT" \
  --output-dir "$BUNDLE"

python3 scripts/audit_recurrent_binary_capture.py \
  --manifest "$BUNDLE/manifest.json" --prompts "$BUNDLE/train_prompts.jsonl" \
  --expected-prompt-sha256 "$PROMPT_SHA" --expected-prompt-count "$PROMPT_COUNT" \
  --output "$BUNDLE/recheck-audit.json"
```

The builder's audit and an independent recheck must agree, including exact
prefix, mask/cache, request and target-logit row joins. The current builder
marks bundles preparation-only (`training_eligible:false`); further native
model identity, full drafter parity and request completeness gates remain.
The manifest-aware CPU contract has synthetic tests. The first supervised
native `shard-0000` command sequence completed on RTX 5080 on 2026-09-29;
the [run report](w1ax-shard0000-capture-5080.md) records exact observed
rows/bytes, independent audit, compact teacher and remaining readiness gates.

After a bundle is audited, this CPU adapter emits a row map from its audited
`rows.jsonl` and raw target-logit file. It requires the bundle `audit.json`
to bind source hashes and row count, and rejects missing, repeated or
misaligned prefix/label joins. The compact teacher remains a preparation
artifact until the bundle passes the full training eligibility gates.

```sh
python3 scripts/prepare_w1ax_capture_shards.py teacher-rows \
  --shard-manifest "$SHARD/manifest.json" --bundle-manifest "$BUNDLE/manifest.json" \
  --output "$BUNDLE/teacher_rows.jsonl"

python3 scripts/compact_w1a_teacher.py \
  --logits "$BUNDLE/target_logits.f32" --logits-format raw-f32 \
  --rows "$BUNDLE/teacher_rows.jsonl" --d2t "$D2T_NPY" \
  --output "$TEACHER" --target-vocab 151936 --topk 64 \
  --target-gguf-sha256 "$TARGET_GGUF_SHA256" \
  --prompts-sha256 "$PROMPT_SHA" \
  --capture-manifest-sha256 "$BUNDLE_MANIFEST_SHA256"
```

Before any future raw retirement, independently verify the source-cell raw
SHA/bytes, bundle copy SHA/bytes, bundle audit SHA, adapter row-map hash,
compact teacher manifest/shard hashes, full row coverage and exact prefix
ancestry. The source and bundle raw files must stay until a narrow replacement
schema can re-audit from compact evidence and any required archival copy is
verified. No deletion command is provided or enabled by this tool. The
pre-existing `recurrent_binary_capture_v1` requires its full raw
`target_logits` file, so removing it now would invalidate the bundle.

## CPU checks and limitations

Four synthetic CPU tests cover deterministic full train coverage, sealed
final-path avoidance, topic-family cap rejection, native-cell actual row/byte
guard, and exact prefix/label teacher-row mapping. Ruff passes. The generated
plan and all prompt shards remain ignored local data; only this report and the
code/tests are tracked. The 2k source itself remains a candidate: license,
quality, tokenizer, target-token yield and native training eligibility still
need their stated gates. CPU checks do not establish GPU capture throughput
or SM75 behavior.
