# Independent endpoint collection — October 5, 2026

Bounded source deliverable: strict immutable import of six independently frozen
production lanes and three exact original Q4 controls, plus a source-only native
evaluator input route. This closes the requirement that every trained lane share
one training bundle SHA. It does not authorize or implement collection GPU
execution. No training is repeated and no historical hash/receipt is rewritten.

`prepare_nine_model_endpoint_collection.py` accepts schema
`nine_model_endpoint_collection_inputs_v1`. Its required fields are `candidates`,
`evaluation_source`, `controls`, `hardware_stage`. Candidate keys are exactly
`eagle_a8/eagle_a1/dspark_a8/dspark_a1/dflash_a8/dflash_a1`. Each contains exactly
`frozen_lane`, `lane_state`, `supervisor_state`, `training_receipt`,
`export_receipt`, `heldout_admission`: canonical path/SHA256 locators to the
original files. The collection stores these original records unchanged. Its own
file SHA is exclusively the collection identity, never a training bundle SHA.

`hardware_stage` contains the original physical `gpu_uuid`,
`compute_capability: [12, 0]`, and
`runtime_gate: PENDING_fresh_lease_load_dispatch_release`. This declaration
cannot establish a real CUDA context, current device admission or release.

Production imports call the sibling staged-export adapter's
`validate_endpoint` and `validate_export` from
`scripts/export_nine_model_lane_candidate.py`. That adapter owns the genuine
full-allocation positive checkpoint/optimizer/RNG/cursor/source/admission/config/
base/target/serializer joins. The collection preserves every independent frozen
lane hash and enforces exact target/F16 verifier/KV/device identity. It accepts
unchanged historical `nine_model_lane_endpoint_export_v1` EAGLE receipts; the
real family audit validator handles EAGLE's full input locators and block's
hash-only audit inputs separately. No synthetic schema is used to normalize or
replace producer receipts. Candidate coverage comes from the real serializer's
selected projection inventory, not an arbitrary collection coverage label.

`evaluation_source` is an existing authenticated
`nine_model_campaign_bundle_v1` or `nine_model_lane_endpoint_plan_v1` locator.
Their existing source validators preserve protocol/prompts/target/runtime/source
and policy bindings. No protocol or allowance is selected by collection code.
Each lane's heldout admission must use the existing
`nine_model_lane_development_admission_v1` shape and match the common prompts
and that lane's exact frozen source bindings with hashed disjointness evidence.
No held-out prompt messages are inspected. Final prompt bytes remain sealed;
this initial import implementation has no authenticated final heldout receipt
route and cannot freeze a final collection. This is an import gap, not a new
quality gate or permission flow.

Each control declares its genuine family, Q4_0 precision, immutable original
model/provenance locators and explicit deployment coverage/exceptions. Its
existing `nine_model_original_q4_reference_v1` must join the same model/family/
precision and hashed historical evidence. Every family must also join the
original control in the authenticated evaluation source. The immutable EAGLE
hash is checked against the existing known original hash. Old EAGLE provenance
need not be rewritten to add coverage; any coverage present in the original
source/provenance must match exactly. Missing authentic controls stay PENDING.

Independent QA caught an initial gap: the EAGLE-only evaluation source had no
trusted DSpark/DFlash original-control bindings, permitting self-asserted block
references. The final validator rejects each missing original source control.
An EAGLE endpoint plan can therefore be inspected as an evaluation source, but
cannot by itself freeze a full-nine collection. A real evaluation source with
all three original controls remains required; this does not require retraining
independent candidate endpoints or changing their hashes.

The evaluator adds `--collection PATH --collection-sha256 SHA
--inspect-collection`. It validates imports and builds all ten argument lists
through the unchanged existing `native_command`, when the existing protocol
declares all family draft lengths. It reports execution/readiness false.
Attempting collection execution fails closed before `LinuxResources` or any
native subprocess. The existing same-bundle training/export/bundle equality
checks and native loop remain intact.

## Acceptance and remaining integration

Model-free collection tests cover six different lane hashes preserved byte for
byte, authentic locator associations, mixed targets/devices, lane/model/control
relabeling, private heldout-source substitutions, missing controls, hash tamper,
fixture/readiness claims, immutable EAGLE control hash, draft pending inspection,
route mixing, native F16-KV/model/target argv, and pre-GPU execution rejection.

Mac arm64, Python 3.11.15, primary checkout `.venv`: focused 8 tests PASS.
Related nine-model suite: 230 tests PASS, five existing device/platform skips
(before the eighth focused regression was added); final focused rerun includes
the regression. Independent Luna QA passed the same seven original focused tests and an additional
original-control-source regression. Final combined targeted run: nine tests PASS
(eight owner cases plus one independent regression, without duplicate discovery).
Changed-file Ruff check/format and diff check PASS. The isolated
worktree needed `PYTHONPATH=src:scripts` and the primary native `gguf-py` path for
the broader suite; initial missing-module runs failed and were corrected without
source changes. No CPU check establishes CUDA, quality, acceptance or throughput.
Collection fixtures inject the staged-checkpoint validator boundary; actual
serialized producer joins remain separately tested by the sibling adapter.

Integration dependency: land the staged exporter adapter before this collection
feature. Root owns the active goal checkpoint and subsequent runtime adapter.
That adapter still needs a fresh collection-bound lease, actual held GPU lock,
live supervised controller/continuation, explicit full-nine evaluation scope,
current selected-model native load/dispatch and actual per-cell process/context
release/resource-return checks. It should reuse the existing native loop and
report ancestry with both collection identity and individual lane hashes; never
manufacture a same-bundle Campaign receipt. Missing original block controls,
production endpoints, disjoint heldout admissions or operational policy remain
PENDING. Healthy frozen 30a/cc9 training, budgets, sampler/head/cache semantics,
existing watcher and all remote jobs are untouched. This worker ran no remote,
GPU, pretrained-model, capture, launcher, watcher or training action.

Root reviewed published1cb225b and dependency8cd5ba9, integrated them in that
order (ebf2c92/8d420c0), verified exact owned-file equality and preserved both
branch histories. Combined16focused integration checks PASS in0.507s. Runtime
dispatch, authenticated sealed-final import and actual endpoint/control inputs
remain pending; this integration does not expand readiness claims.
