# CPU staged-lane export adapter

Deliverable: `scripts/export_nine_model_lane_candidate.py` validates and exports
one original production `nine_model_training_lane_v1`, independently of other
five candidates and their bundle hashes. EAGLE/DSpark/DFlash direct A8/A1 are
supported. Existing full-bundle exporter, endpoint/trainer/binder/pipeline and
healthy 30a/cc9 runtime are unchanged. Curricula are outside this direct-staged
contract; no allocation, scientific policy or source pin is selected here.

Public interfaces:

- `validate_endpoint(lane_locator, lane_state_locator, supervisor_locator,
  training_locator, *, files=None, checkpoint_loader=None)` returns original
  lane/admission/spec, target/base, source bindings, limits, receipt and serializer
  records. The loader seam is for CPU tests; production uses hash-bound
  `torch.load(map_location="cpu", weights_only=False, mmap=True)`, matching the
  actual trainers' local-pickle contract. No model/target reconstruction, dataset
  read, CUDA RNG restore or GPU query occurs.
- `validate_export(context, receipt_locator, files=None)` imports unchanged
  `nine_model_lane_endpoint_export_v1`, including old EAGLE receipts without
  additional fields. EAGLE audits use complete input locators and
  `checkpoint_manifest`; block audits use hash-only `base_gguf`/`checkpoint`/
  `manifest` plus family/profile. No invented normalized audit is written.
- `export_endpoint(context, new_directory, *, release_check, cpu_admission,
  wall_seconds=3600, run=subprocess.run)` requires trusted actual runtime release
  and CPU admission callbacks. CLI only inspects metadata and reports execution
  false; it cannot promote an artifact or infer kernel release from JSON.

Validation requires original canonical lane/state/supervisor/train locators,
positive committed successful production receipt, natural exit0/no signal,
original GPU UUID/SM120, exact config/target/F16 KV/source pins and complete
cumulative seconds. Zero updates, fixture/synthetic/failed/stopped/partial-budget
receipts fail. The serialized EAGLE payload joins exact manifest schema/source,
optimizer/RNG, original prepared TRAIN metadata/source, integer cursor/step/token counters, original immutable config and
single lane NPZ inventory. Serialized block payload joins actual
`block_qat_checkpoint_v1`, contract/source SHA, original cursor, optimizer/RNG,
base/target/TRAIN dataset cursor (split/seed/source hash) and receipt sidecar (or original latest pointer when the historical
producer did not write sidecars). Every selected latent and effective row scale
is compared directly with the NPZ to prevent unrelated final-export promotion.
The exact nine EAGLE or fifteen FFN plus optional fusion inventories are checked.
Block private heads remain original preserved tensors through the real serializer.

EAGLE continuous accounting finishes after the final serialization; the receipt's
elapsed seconds can legitimately exceed saved payload elapsed seconds. The gate
requires finite nonnegative saved time no later than the full-budget receipt,
with exact serialized optimizer/step/cursor/tokens; it does not require false
bit-identical accounting timestamps or relabel/retrain the original lane.

Export runs the **original frozen lane serializer path**, using original pinned
Python, as a lazy subprocess with CUDA hidden and bounded timeout/two CPU thread
settings. Original lane source pins are checked before and after serialization;
receipt.bundle_sha256 remains exactly the original frozen lane SHA. New receipt
fields add original config/target/base/training source, source-specific serializer
pin, adapter source pin and original state/manifest links. An aggregate collection
hash is never training provenance. Existing output directories are rejected;
failed output and audit/validation files remain, with no committed export receipt.
Native load/graph/dispatch and dense fallback remain explicitly PENDING.

Acceptance: seven focused CPU tests pass, including all six family/precision
serialized state joins and actual tiny GGUF/NPZ family serializers with their
real audit API layouts. Negative checks cover zero/boolean updates, partial and
nonfinite elapsed time, synthetic/fixture/failed/stopped receipts, config/hash
changes, signals/release metadata, missing serialized optimizer/RNG, unrelated
export arrays, old EAGLE import, pinned command/CUDA hiding, callback order,
immutable destinations and retained failures. Source/frozen-lane preparer is
mocked only in the tiny metadata test seam; this does not grant production
readiness. Actual family serializers run on constructed tiny CPU assets, without
pretrained weights or captures. Mac arm64, Python3.11.15, Torch2.14.0; this is
serialization/source proof, never SM120/SM75 performance or acceptance evidence.

Remaining integration: root must supply a live runtime adapter holding the
common GPU lock while verifying original supervisor/controller/trainer identities,
all descendants/groups/device holders and release, current pause/availability,
source producer lifecycle, fresh CPU RAM/disk peaks and bounded supervision.
Callbacks must recheck release before and after serializer and return literal
True only on real success; JSON alone is not kernel proof. The collection owner
reuses both validators and preserves each lane's independently frozen hash.
Actual first trained endpoint files, committed full-budget production receipts,
release/CPU/native admissions and actual pretrained candidate serialization are
not available to this worker and remain missing. No GPU, remote, physical capture,
model-weight access, launch, watcher or new budget operation was performed.

Independent Luna QA also passes the focused adapter plus existing EAGLE/block
CPU serialization suite: 21 tests total, 17 PASS and four expected native-fixture skips. Final
Ruff check/format and diff whitespace checks pass. Ignored QA proof is retained
at `results/nine-model-qat-overnight/staged-export-qa/qa.txt` and copied to primary
before the worktree is retired.
