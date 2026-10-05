# Sealed-final collection admission source — October 5, 2026

Bounded source deliverable: an opaque-metadata final admission route for the
independent six-lane endpoint collection. This is not a final-data opening,
model/GPU execution, readiness grant, new scientific selection, or result.
Development admissions and the existing metrics/protocol remain unchanged.

The existing `nine_model_campaign_bundle_v1` evaluation source can now carry
`heldout_selection`. `source_context` exposes that as `final_selection`; absent
selection plus `final_set_authorized=true` fails closed. An EAGLE-only endpoint
plan continues to have no final authorization and cannot authenticate this route.
No existing archived final authority is claimed to satisfy the new contract.

Selection fields are exactly `phase`, `corpus_manifest`, `split`, `shard`,
`prompts`, `protocol`, `target`, `origins`, `authorization`,
`selection_provenance: agent_selected`, `human_selected: false`. `phase` is
`final`; corpus `split` is original `final` or `sealed_test`. The complete
original shard selected by ordinal must equal the source's pinned prompt
locator. Partial row lists, concatenated/rehashed prompts and caller disjointness
booleans are unsupported. This supports one explicitly frozen original prompt
shard per existing evaluator prompt locator. Expanding that scope remains a
source/protocol selection, never an inference made by the helper.

`origins` binds all six candidate names to the exact original frozen lane,
state, supervisor state, training receipt, export receipt, lane config,
frozen training source and selected exported model. It excludes only the new
heldout admission locator, avoiding an impossible receipt hash cycle. Collection
identity and training-lane identities remain separate. Current staged endpoint
and export validators preserve their original producer/serialized source checks.

Standing human authorization is separate from this later concrete agent-frozen
selection: `source.origin.evaluation.authorization` and the selection must carry
the same `record`, `instruction`, and `phase: final`. The unchanged pinned
record must actually contain the declared instruction. A conservative text
consistency gate requires final/sealed/reserved scope plus evaluate/evaluation/
compare/comparison and rejects explicit negative/pause wording. Training/GPU-only
instructions fail closed. The instruction requires no future artifact hashes,
canonical JSON text or new user approval. Ambiguous language remains PENDING;
this parser does not authenticate human authorship or replace review of genuine
standing provenance. The already authenticated original evaluation-source
locator remains the trust root. An arbitrary hashed record is insufficient.

Opaque metadata uses the actual existing `w1a_data_manifest_v1` and
`continuous_w1ax_prompt_manifest_v1` schemas. Their original index SHA and
prompt SHA are preserved. Prompt locators use `Files.opaque`, never `Files.check`
or a prompt read. Index rows require ID, group, content SHA, original source ID
and source-row ID. A strict primitive-field allowlist rejects messages, text,
content, prompt and arbitrary nested payload fields. Source-row identity uses
`(source_id, source_row_id)`; sharing a public source dataset alone is not
contamination. No semantic/near-duplicate disjointness claim is introduced.

EAGLE actual TRAIN membership joins the pinned continuous config's original
corpus/captures/source positions to the source-bound `train-providers.json`
execution manifest and each original provider's prompt hash/count and TRAIN
eligibility. No captured prompts/messages are read. Block actual membership
joins the frozen data-manifest hash, its original `block_train_inventory_v1`,
and every train/calibration TRAIN-derived chain's ID/content to the original
corpus TRAIN index. An original corpus/index lacking these identities cannot
pass. Every selected final row is compared against each lane's actual frozen
TRAIN-derived membership for ID/group/content/original source-row overlap.

`prepare_nine_model_heldout_admission.py --inputs COLLECTION_INPUTS --output
NEW_DIRECTORY` validates the original source and all six endpoint/export
associations plus final evidence, then publishes six deterministic
`nine_model_lane_final_admission_v1` association receipts. Reimport recomputes
all evidence and requires exact receipt equality; a receipt itself grants no
readiness. `--inspect-draft` reports PENDING and writes nothing. The helper
selects no corpus/scope/recipe/protocol/models/authorization itself.

Runtime API is preserved: `evaluation_view(context)` returns the existing
`(view, models)` pair. Existing `context.files/source/protocol/contexts/exports`
remain unchanged; new `context.heldout` records split, selection, sealed status
and prompt count. Final prompt bytes can open only in the separate actual
runtime after its fresh release/admission gates. Source import alone executes
nothing.

## Validation and remaining inputs

Mac arm64, primary `.venv` Python 3.11.15: 15 combined focused collection,
original-control QA, owner final-admission and independent source-join QA test
methods PASS. Owner parameterized cases cover missing/boolean-only authority,
wrong lane/source/config/model/export/target/protocol/prompt/shard, stale receipts,
caller booleans, ID/group/content/source-row overlap, wrong roles, payload fields,
training-only/prohibited/foreign instruction evidence and false human selection.
A `Path.open` spy verifies both TRAIN and final prompt files stay unopened in
preparation/import. Luna's independent tests check EAGLE index/position joins and
block actual trainer/inventory membership. Initial local invocation had a
missing `tests` import path; corrected with `PYTHONPATH=src:scripts:tests`.
Changed-file Ruff check/format and diff check PASS. All fixtures are software
fixtures, never production provenance or scientific evidence.

Real final admission remains PENDING: authentic full-six endpoints/exports,
all three original control joins, original opaque corpus/index files and an
existing evaluation source carrying genuine standing final authority and its
concrete frozen selected scope are required. The current overnight GPU/QAT
instruction alone is not asserted to authorize sealed-final evaluation. No
final prompt/message/pretrained weight, real TRAIN capture, remote/tmux/GPU,
native inference, watcher, sampler, label, budget or running source was accessed
or changed during this source assignment. Root owns durable goal/status updates
and final runtime integration.
