# W1A larger-data preparation and compact teacher contract

CPU Phase 1A implementation. No public dataset was downloaded or admitted to
training by this change. The repository contains reproducible preparers and
synthetic CPU fixtures, not a claim of 2,000 or 10,000 usable prompts. Source
selection, license review, target tokenization and native capture remain gates.
The inherited 96 prompts remain smoke/regression data. The legacy final-set
content was not read.

## Source and split contract

`scripts/prepare_w1a_data.py` reads a local JSON catalog and JSONL source
files. A source entry requires a stable ID, URI, revision, license, SHA256,
domain (`prose`, `code`, `reasoning`) and path relative to the catalog. Optional
`fields` maps `id`, `messages`, `group_id`, `topic_id` to source column names.
Each row has a stable ID and nonempty chat `messages`; `group_id` defaults to
the row ID. `topic_id` should represent a narrow problem or document family,
not a broad domain. Conversation/topic IDs are source-qualified. A source file
hash mismatch fails before output creation. The catalog and source hashes are
recorded in the output manifest.

Example catalog shape (substitute real revision, license and file hash):

```json
{
  "schema": "w1a_source_catalog_v1",
  "sources": [
    {
      "id": "source-revision-prose",
      "uri": "https://example.invalid/dataset",
      "revision": "immutable upstream revision",
      "license": "verified license identifier",
      "domain": "prose",
      "path": "prose.jsonl",
      "sha256": "64 lowercase hex characters",
      "fields": {"id": "id", "messages": "messages",
                 "group_id": "conversation_id", "topic_id": "document_id"}
    }
  ]
}
```

Run from the repository root after putting the catalog and sources under
ignored `data/`:

```sh
python3 scripts/prepare_w1a_data.py \
  --catalog data/w1a-sources/catalog.json \
  --output data/w1a-prepared/freeze-001 \
  --train-small 2000 --train-large 10000 --dev 192 --final 192 --seed 1429
```

The output directory must be new. It contains `train_small`, `train_large`,
`dev` and `final` prompt JSONL files, index JSONL files and `manifest.json`.
The smaller training tier is a group-level subset of the larger. Development
and final groups are independent of both training tiers and of each other.
Each split targets equal domain quotas; multi-row groups can make actual
counts exceed a requested number. The manifest reports actual counts,
per-domain counts, source IDs, groups, words and characters. Word counts are
**not** target token counts. Final prompt contents should remain sealed after
manifest hash review. A frozen manifest is not proof of source diversity or
quality: inspect its coverage summary and approve sources before native
capture. The pipeline does not touch the legacy final set.

Exact duplicates use normalized text. Near duplicate candidates use four
SimHash bands on five-word shingles, followed by exact shingle Jaccard at the
configured threshold (default 0.88). This is deterministic but can miss
similar pairs that share no band. Conversation and topic grouping, filtering
and deduplication precede seeded split assignment. Splits prevent leakage
for declared groups and detected duplicates; source family metadata must be
good enough to catch remaining relations. Length filtering is explicit;
EOS/truncation, answer quality and target-token coverage must be audited after
tokenization/capture. Repeated training presentations and overlapping unrolls
must be counted separately from unique prompts and target tokens.

## Compact teacher contract

`scripts/compact_w1a_teacher.py` reads captured native F32 full-target logits
(`.npy` or raw little-endian F32 rows), a unique draft-to-absolute-target ID
map (`.npy`) and a JSONL row mapping. Each row mapping requires `id`,
`prompt_id`, `capture_id`, `logits_row` and exact `prefix_token_ids` (target
token IDs); optional `next_target_id` records an exact-prefix target label.
The command pins target GGUF, prompt-set, capture manifest, logits, row map and
draft map hashes. It performs CPU softmax on each full-target row and writes
compressed shards. Example:

```sh
python3 scripts/compact_w1a_teacher.py \
  --logits data/native-capture/logits.f32 --logits-format raw-f32 \
  --rows data/native-capture/teacher_rows.jsonl --d2t data/maps/d2t.npy \
  --output data/teachers/freeze-001 --target-vocab 151936 --topk 64 \
  --target-gguf-sha256 "$TARGET_GGUF_SHA256" \
  --prompts-sha256 "$PROMPTS_SHA256" \
  --capture-manifest-sha256 "$CAPTURE_MANIFEST_SHA256"
```

Schema `w1a_compact_teacher_v1` stores `draft_topk_ids` (draft IDs),
`draft_topk_probs` (unconditional full-target softmax mass),
`draft_tail_mass` (remaining probability on mapped draft IDs), and
`outside_draft_mass` (all probability on unmapped target IDs). These four
terms sum to one per row within F32 rounding. It also stores
`target_topk_ids/probs` and `target_tail_mass` for target-wide coverage audit,
plus `next_target_id` and mapped `next_draft_id` (`-1` if absent or unsupported).
The target top-k and draft top-k are ranked separately. Target top-k values
are absolute target IDs; no draft renormalization is implicit. A loss that
spreads draft tail mass uniformly and conditions away outside mass is an
**approximation** and must report that choice. Top-k KL is not exact full-vocab
KL. Report supported mass, unsupported target labels and target top-k coverage
before choosing this objective over hard labels.

`iter_verified_shards` checks manifest identity, shard/index hashes, shapes,
mass normalization and internal prefix hashes. Its `expected_prefixes`
argument rejects a teacher row whose captured prefix differs from the current
sample prefix. A consumer must also match row ID/prompt ID and state/mask
ancestry before using features or labels. A saved D-prefix teacher row cannot
be assigned to a later student-generated prefix merely because positions or
lengths match. Teacher-forced offline rows are the intended first use; fresh
native rollout capture is required for a bounded trajectory refresh.

## Current checks and next data gate

The synthetic CPU fixture checks catalog hash rejection, exact deduplication,
group-disjoint nested splits, reproducible hashes, teacher full-target mass,
shard hashes and exact-prefix rejection. No tokenizer, target model, GPU or
final-set prompt content was loaded. The next gate is to select and legally
review real diverse source revisions, put the catalog/files under ignored
`data/`, run this preparer, review the manifest and freeze its hash. After GPU
access returns, capture native target features/logits for the frozen training
set and audit label alignment, EOS/truncation, draft vocabulary coverage and
target-token yield. A 100-step calibration then informs whether the 10,000
prompt tier is useful and feasible.
