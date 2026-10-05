# Staged production QAT launch integration

Owner: `/root/overnight_bundle_owner`; isolated branch
`prep/nine-model-overnight-bundle`. Sole RTX5080 access belongs to
`/root/overnight_5080_operator`. This feature owner made no remote call, model
load, capture, optimizer update or GPU query. Existing preparation packets,
ledgers and all other teams' files remain unchanged.

Deliverable: an additive one-candidate production manifest, source config and
fresh SM120 admission path, followed by the unchanged production trainer under
detached supervision. Acceptance: exactly one known candidate can prepare while
the other five remain PENDING; the unchanged full-campaign schema still requires
six candidates. All seven per-candidate source/resource/kernel/model/backward/
full-F32-moment-memory/capture-portability checks remain mandatory. An explicit
delegated operational budget must not claim the human chose its exact numbers.

## Source changes and actual checks

`prepare_nine_model_bundle.py` recognizes separate `nine_model_lane_inputs_v1`
descriptors for source-config and admission-plan materialization. Staged budgets
use `human_selected=false` plus `authorization.kind` equal to
`human_delegated_operational_settings`, the direct instruction and a SHA-pinned
written authorization record. The original whole-campaign budget rule remains.
No production status is inferred from source config parsing.

`nine_model_lane_sm120_plan_v1` requires exactly one of the six known candidates
and exactly its family's portability producer. Admission execution is unchanged:
positive-count A8/A1 native kernels, native projection dispatch, actual selected
model/backward including later-state/K/V paths, scratch FP32 full-moment
reservations held through backward, three-domain label/numeric portability and
process/context/resource return. The trainer's strict source identity now also
includes its imported resource/runtime leaves and admission verifier.

`prepare_nine_model_lane.py` freezes the selected lane's explicit portable QA,
actual production data admission, calibrated initializer, initial native
model/export, immutable config/budget and actual admission plan. Initial/final
target precision remains F16 and KV F16. Other candidates and evaluation remain
PENDING; original Q4 controls are still required for eventual campaign comparison.
The first admitted lane does not wait for unrelated family artifacts.

`run_nine_model_lane.py` validates all pins, requires a fresh sole-owner lease,
unpaused control and detached Linux tmux `remote_job` supervision with a 90-second
grace, then takes the shared campaign lock. Each attempt performs fresh unchanged
Admission before production training. Training uses the same single-lane
`train_nine_model_qat.py`, exact frozen config/hash and cumulative accounting.
`--resume` verifies the same manifest, old process identities/resource release
and restores a real latest checkpoint when one exists. No healthy run is stopped
to produce a resume demonstration. Positive committed endpoints are preserved;
evaluation stays explicitly PENDING for subsequent integration.

Checks on Apple M3 Max/macOS CPU: 7 new staged-lane tests PASS, 41 affected
builder/plan/admission/staged tests PASS, changed-file Ruff and format PASS.
The first affected run correctly failed because this isolated worktree lacked
the native Python GGUF dependency; an offline local shared clone at unchanged
native commit `624f50e74` supplied it. No native Git revision was changed. CPU
fixtures retain `artifact_kind=fixture` and cannot grant production admission.
Fresh RTX5080 execution is still unproven by these source tests.

## Concrete operator sequence

First candidate is calibrated fixed-A8 EAGLE if its original completed continuous
provider is authentic and fits the fresh SM120 gates. Otherwise select the first
genuinely ready block A8 lane. Root-selected operational setting is 86,400
cumulative trainer-accounted seconds, with no smaller step/token/epoch stop,
initial checkpoint cadence 250 updates and observed target 5–15 minutes. These
are disclosed agent allocations under the direct overnight delegation.
The outer training wall cap is 108,000 seconds (30 hours), allowing checkpoint,
I/O and startup overhead beyond the cumulative trainer allocation. Copy the
authorization text into a new immutable run-input record; do not pin a mutable
STATUS/goal checkpoint as a live operational budget input.

1. Operator stages published parent/native source in a clean isolated checkout
   and supplies the exact Python, GPU UUID, native binary/library, target and
   existing original completed EAGLE provider/config/stages locators. Do not use
   the old source6f launcher; current `eagle_inputs()` authenticates the full
   prepared corpus and invokes `create_current_native_child`.
2. The separate EAGLE data owner prepares real three-domain disjoint TRAIN
   calibration and validation operands from the existing native accepted-prefix
   payload, then a scale-only initializer with fixed latent magnitudes 0.5.
   Historical 384-row rescue artifacts and the ineligible local diagnostic are
   not production replacements. Preserve full initial nine-projection exports
   and source-bound native audits for admission.
   The source-bound initial EAGLE export sequence is:

   ```sh
   python scripts/train_nine_model_qat.py --config EAGLE_A8_CONFIG --run-dir INITIAL_PREPARE_DIR --bundle-sha256 INITIAL_PREPARATION_IDENTITY_SHA --stage-name eagle_a8/initial-prepare --completion-output INITIAL_PREPARE_RECEIPT --allow-cuda --prepare-only
   python scripts/export_recurrent_binary.py --base ORIGINAL_EAGLE_F16_GGUF --checkpoint INITIAL_PREPARE_DIR/checkpoints/step-000000000000-e000000-r000000000000/A8/joint.npz --manifest INITIAL_PREPARE_DIR/checkpoints/step-000000000000-e000000-r000000000000/A8/joint.json --output CALIBRATED_INITIAL_A8_GGUF --audit CALIBRATED_INITIAL_A8_EXPORT_AUDIT
   ```

   Run the prepare-only producer in its own detached `remote_job` under the sole
   operator. `INITIAL_PREPARATION_IDENTITY_SHA` hashes the explicit frozen
   initialization request; it is not a training admission or the later lane
   bundle hash. Zero updates/moments must remain zero. `eagle_inputs()` applies
   the sparse FC initializer inside `build_lanes()` before saving the complete
   nine-projection `joint.npz`/manifest. The serializer checks all nine source
   projection shapes, Q/K row permutation, selected hard bits/scales and protected
   norm/d2t/tensor byte preservation. Bind the serializer's output SHA and audit
   as `initial_model`/`initial_export_audit`; native admission must test this exact
   calibrated GGUF, not the original F16 or an uncalibrated W1 model. Never use
   `export_nine_model_candidate.py` for this initialization: that producer requires
   a genuine positive-step trained endpoint and complete campaign bundle.
3. Root/operator materializes a new `nine_model_lane_inputs_v1` descriptor with
   one candidate, explicit operational budget, config-template/prepared data,
   selected initial export/calibration, native binaries and three-domain TRAIN
   goldens. The portable QA ledger is additive and pins the final source.
   EAGLE's config template must be the authenticated prepared run's original
   resolved config/stages, with only permitted current training controls changed;
   a generic template cannot substitute for its frozen model/data declarations.
4. Source-config preparation (no model/GPU):

   ```sh
   python scripts/prepare_nine_model_bundle.py --inputs lane-inputs.json --materialize-configs lane-configs
   python scripts/prepare_nine_model_bundle.py --inputs lane-configs/resolved-inputs.json --materialize-admission-plan lane-admission.json
   python scripts/prepare_nine_model_lane.py --inputs lane-admission.json-inputs/resolved-inputs.json --output lane.json
   ```

5. After fresh resource/ownership and detached durability admission, freeze the
   manifest SHA and issue an unchanged availability lease bound to that SHA.
   The run must be in its own remote project directory, with its caches and
   outputs outside Git. Launch through the sole operator's tmux MCP transport:

   ```sh
   tmux -L binary-eagle-runtime new-session -d -s RUN_SESSION -c PROJECT 'exec python scripts/remote_job.py RUN_ID --stop-grace-seconds 90 -- python scripts/run_nine_model_lane.py --start --lane LANE_JSON --lane-sha256 LANE_SHA --availability LEASE_JSON --run-dir runs/RUN_ID/lane --supervisor-state runs/RUN_ID/state.json'
   ```

6. Runtime acceptance requires genuine seven-PASS training admission and growing
   positive optimizer updates/cumulative seconds on actual RTX5080/SM120. No
   claimed ready state precedes those receipts. First follow-up checks gradients,
   finite losses, movement, checkpoint publication, host/device margins and
   active process/session handles. Scheduled 30-minute healthchecks monitor and
   repair from the same frozen config/committed checkpoint; leave a successful
   trainer running. Repair must preserve charged elapsed budget and ancestry.

## Remaining integration

Independent QA must review the source changes before root integration/push.
Operator and data owners supply all actual host/model/provider/calibration/
golden/native-model/portable-QA pins; this source feature invents none. Root
preserves all six candidates/three frozen Q4 comparisons in the objective and
records remaining family lanes and automatic evaluation integration separately.
Full trained exports, all-nine final evaluation and actual throughput/acceptance
are later campaign outcomes, not prerequisites for the first safe lane.
