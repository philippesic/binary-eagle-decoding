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

## Concrete EAGLE packet helper

`scripts/prepare_eagle_lane_packet.py` supplies the actual source-bound config,
initial preparation identity and argv lists. It is metadata-only: it does not
load a model, construct NativeTeacher, launch a process or query hardware.
Its `prepare` mode requires the successful production initializer and original
completed receipt `bdfa56f8...`; it verifies their full source/target/base join,
preserves the original resolved config's frozen stages/model/data declarations,
copies immutable authorization, and writes fixed A8/probes-off 24h/30h settings.
It requests the existing fixed-reference cache/head optimizations; the current
typed production admission must show executed cache calls and an effective
batched head for this exact config before any update. Learned quantizers,
midpoints, correction/alternative binary optimizer recipes remain OFF. This is
an explicit current execution request, not reuse of historical optimization
readiness. The packet writes `effective-controls.json` with these selections.
The packet's `commands.json` binds exact producer source pins and complete argv
lists for zero-update prepare, all-nine export, native generation and replay.
Root reviews those commands before the operator executes them under remote_job.

Minimum runtime JSON (`schema=eagle_lane_packet_runtime_v1`) from the operator:
`native_source_revision` (native624 full SHA), actual `gpu_uuid` and `device_name`,
canonical `gpu_control_path`, original `base_model` F16 EAGLE locator, `inputs`
with exact path/SHA pins for `backend_binary`, `teacher_binary`, `binary` server,
F16 `target` and `runtime_library_0...N`, exact `environment` including the UUID
visibility and current library path, and the four established `resource_policy`
floor/return-tolerance fields. Missing build hashes remain unavailable until the
actual build completes; no guessed runtime pin is published.
Include `build_provenance` with pinned native build/compile-flag metadata and
the external `cuda-glibc-compat` include-shim manifest used for CUDA13.1/GCC15.2;
the descriptor adds these to the frozen admission source inventory. This records
the actual header/flags without changing native624 or asserting runtime readiness.

The original prepared directory is:

```
/home/philip/binary-eagle-decoding/runs/qat-optimization-readiness/retained-capture-adoption-20261002-01/checkout/runs/retained-a8-a1-preparation-20261002-01
```

Use these commands with the operator's exact runtime JSON/pin and the successful
initializer at `data/nine-model-overnight/eagle-fixed-a8-initializer-01`:

```sh
python scripts/prepare_eagle_lane_packet.py prepare --runtime RUNTIME_JSON --runtime-sha256 RUNTIME_SHA --prepared-run-dir ORIGINAL_PREPARED_DIR --initializer-dir /home/philip/binary-eagle-decoding/data/nine-model-overnight/eagle-fixed-a8-initializer-01 --authorization IMMUTABLE_AUTHORIZATION_TEXT --output /home/philip/binary-eagle-decoding/data/nine-model-overnight/eagle-fixed-a8-packet-01
```

The packet prepares exactly one original TRAIN prompt per domain for bounded
target-only greedy generation: 512 native prompt-token cap, 32 new tokens,
544 chain cap, five native taps and no stored chain logits. This capture is
portability goldens only; it does not recapture or replace the full training
corpus. After the operator runs the exact `native_generation` argv, use its
three JSON receipt lines to build source-preserving replay requests:

```sh
python scripts/prepare_eagle_lane_packet.py replay --packet PACKET_DIR --receipts GENERATION_JOB_STDOUT_LOG
```

`replay` verifies unchanged native generation/CUDA/source/target/tokenizer/input
ancestry through the existing validator and preserves every native-issued F16
KV decode partition. The operator then runs the exact `native_replay` argv:
three taps 2/18/33 and full-vocabulary **last-row** logits. After replay and the
separate initial model export complete, bind genuine receipts and current QA:

```sh
python scripts/prepare_eagle_lane_packet.py bind --packet PACKET_DIR --generation-link-sha256 REPLAY_PREPARATION_LINK_SHA --receipts REPLAY_JOB_STDOUT_LOG --initial-model PACKET_DIR/initial-calibrated-a8.gguf --export-audit PACKET_DIR/initial-export-audit.json --qa-ledger CURRENT_ADDITIVE_QA_LEDGER
python scripts/prepare_nine_model_bundle.py --inputs PACKET_DIR/production-inputs.json --materialize-admission-plan PACKET_DIR/lane-admission.json
python scripts/prepare_nine_model_lane.py --inputs PACKET_DIR/lane-admission.json-inputs/resolved-inputs.json --output PACKET_DIR/lane.json
```

`bind` validates the exact three-domain replay file bytes with
`NativeCaptureGoldens`; it cannot promote generated receipts to replay goldens.
`replay` prints the generation/replay-link locator; pass that externally recorded
SHA to `bind`, which rejects changed link/request/raw-generation bytes.
All-nine initial model and selected data/config pins then join the ordinary
fresh SM120 plan. Missing actual initialization/native/model/memory/QA evidence
still refuses production launch. These helper steps emit no production PASS
hardware claim and zero optimizer updates. Binding rederives each replay's
ordered prompt/domain/tokens/partitions from the pinned native generated parent;
substituted prefixes, duplicate domains and reordered requests refuse. Initial
binding checks the actual zero-update producer receipt/config/request/source,
step-zero outer checkpoint and all-nine joint NPZ/manifest hashes against the
serializer's base/output/checkpoint audit. The trainer's preparation receipt now
includes these truthful source/config/stage fields; no math or admission is
changed. Five focused metadata fixture tests verify positive joins and reject
corpus/prefix/domain/order/checkpoint/source/export substitutions; the relevant
trainer test module passes 30 tests. Changed-file Ruff and format pass. No full
completed suite was repeated.

The training provider iterates all 320 original shards and sorted native
`(prompt_id, round_index)` anchors, preserving whole prompt chains and exact
resume cursor. Its selected-shard iterator is smoke-only. There is no 13-prompt
or pilot exposure cap: actual unique prompt/row counters must be monitored.
The 24-hour allocation may end before every original TRAIN prompt is consumed;
source eligibility for 10,000 prompts is not a claim of completed exposure.
Zero-update smoke executes cache/head requests without requiring an existing
optimization receipt, so admission is not circular. Production `run()` then
requires current typed admission whose observed execution paths match the
selected fixed recipe. No historical readiness or silent fallback is used.
Periodic standalone development is explicitly deferred with
`development_every=9223372036854775807`; the inherited 1,000-update boundary
otherwise returns from the trainer and stops a healthy lane. Checkpoint cadence
remains 250, the 86,400-second training cap remains exact, and the existing final
standalone evaluation request still publishes at the successful cap. A focused
tiny CPU regression crosses update 1,000 with two actual updates and proves the
trainer reaches its fixture cap and emits the final request; it is control-flow
evidence, not an RTX5080 performance or production-training result.
The frozen admission source inventory retains the externally pinned generation
link, raw parent receipt log, exact replay requests and original prompt joins,
plus the initial request, actual zero-update receipt and outer checkpoint
manifest. These proofs remain inspectable after packet freezing.

## Official author source dependency

Current `w1ax_capture_provider.NativeCaptureProvider.load_models_cpu()` calls
`w1a1_eagle.official_loader.load_official_eagle3`. It requires installed AngelSlim
distribution metadata whose real VCS `commit_id` is
`0358da9c651e6a7d7ccafea26ced4b9c98d11681`, plus these official package modules:

- `angelslim.compressor.speculative.inference.models.eagle3.configuration_eagle3_model`
- `angelslim.compressor.speculative.inference.models.eagle3.draft`
- `angelslim.compressor.speculative.inference.models.eagle3.eagle3_model`

The third supplies `Eagle3Model` and `ModelLoader`; Transformers must provide
`AutoTokenizer` and `ROPE_INIT_FUNCTIONS['default']`. Locate the original working
environment/pinned author Git checkout on RTX5080 through the sole operator.
Reusing an existing exact installation is preferable. If installation into the
locked current environment is needed, use the authentic local Git checkout and
`pip install --no-deps git+file://PINNED_AUTHOR_CHECKOUT@0358da9c651e6a7d7ccafea26ced4b9c98d11681`,
then check revision metadata and official module imports without model loading.
Record loaded module paths/hashes and package versions. Do not replace this
official loader with the private leaf namespace from the Mac diagnostic or
forge distribution metadata; neither establishes production model ancestry.

## Bounded initial all-nine SM120 phase plan

Root approved these bounds in principle; actual command/config binding and a
separate final GO remain required. This is not hardware admission. The
operator's actual runtime/packet supplies the source-bound argv and final pins.
Current published native is `cc9cab3c64f61580cf63e5ef050b075b11cd1fb9`; its bool
metadata loader link repair leaves training arithmetic unchanged. Actual native
build/library/header/flags pins belong to the operator's runtime JSON. Official
AngelSlim0358 public loader imports and Transformers4.57.6 were verified by the
operator without loading a model; actual model execution remains a separate gate.

Run only `commands.json.initial_prepare`, under a distinct detached remote_job,
after the heavy compiler/initializer groups have exited and fresh resources are
verified. Proposed bounds: 1,800 seconds whole phase, cgroup MemoryMax16GiB and
MemorySwapMax2GiB within the observed20GiB WSL cap, no concurrent heavy phase,
and at least8GiB free disk. Before frozen CPU target/drafter construction, source
requires MemAvailable at least14GiB (12GiB additional plus2GiB retained floor).
Subsequent source gates retain2GiB host available,1GiB whole-GPU free and the
selected continuous config's12GiB reserved CUDA ceiling. If actual packet
settings differ, review before GO rather than weakening a gate during execution.
Root-selected return policy is2GiB host/1GiB GPU floors and512MiB host/128MiB GPU
return tolerances. Exact owned process/group/CUDA/DXG release is still mandatory;
these explicit operational tolerances are not historical production evidence.

Nine source projection shapes contain218,234,880 latent elements plus65,280 row
scales:218,300,160 selected trainable F32 parameters. One master's arrays cost
873,200,640 bytes; gradients cost the same; two scratch F32 Adam moment shapes
cost1,746,401,280 bytes. The subtotal is3,492,802,560 bytes (3.253GiB), excluding
frozen embedding/norms, graphs, sign temporaries, cache/head activations, context,
allocator fragmentation and checkpoint host staging. These are source arithmetic
estimates, not measured capacity or GPU performance. `smoke_with_training_memory`
holds both scratch moment shapes through actual model forward/backward, attaches
zero optimizer moments, then releases them; this must actually pass on SM120.

Acceptance evidence before native export is the exact actual zero-update
preparation receipt joined to its request/config/source, `effective-config.json`
with all nine parameter families and fixed-A8/cache/head settings, and
`dual_smoke.json` with finite loss/nonzero selected gradients, later-state/K/V
gradients, actual cache calls/effective batched head and gradient-resident resource
peaks. All counters and optimizer state remain zero; the final checkpoint's
step/epoch/cursor are zero and its all-nine NPZ/manifest includes the calibrated
FC initializer. Native export then validates original F16 base/protected norms/
d2t and packed projections against those exact checkpoint files. Record actual
host/device peaks, wall time, process identities and group/context release; no
successful preparation or full-moment capacity is inferred from startup/source
alone. This phase establishes initialization/serialization evidence; the frozen
lane's fresh native/kernel/portability/current-package admission remains required
before real optimizer updates.

## Actual initializer report schema correction

The first actual metadata packet attempt refused before output publication:
the packet helper expected invented `orientation_rescue` / `coordinate_flips`
keys, while the production fitter serializes canonical `FusionFitConfig`
fields `zero_scale_orientation_rescue` and `max_coordinate_flips_per_row`.
The fixture repeated the same hand-written aliases and therefore missed this
source/API mismatch. The failure and original initializer/report bytes remain
unchanged; it was not a data or eligibility failure and no refit is needed.
The helper now constructs the actual dataclass from its complete canonical
schema and verifies fixed A8, rescue OFF, flips zero and reference-half policy.
Fixtures serialize `dataclasses.asdict(FusionFitConfig(...))`. Three focused
checks pass for the actual schema and refusal of rescue/flips/A1/unit/reference
substitutions or nonempty orientation-event reports before publication;
independent QA reran the seven packet checks before the final event assertion.
Changed-file Ruff/format pass. All existing
source, full TRAIN, zero-state, export and exact replay gates remain intact.

## First bind overwrite-guard correction

Actual bind attempt01 refused before publication because its broad
`golden-*.json` guard matched preparation's own `golden-source-joins.json`.
The guard now checks only exact filenames that bind creates, including the
three golden receipts, their manifest/inventory and native smoke outputs.
Preparation's immutable source joins remain valid input. Actual failed run and
existing packet/model/generation/replay/QA bytes are preserved; retry needs no
hardware run or artifact regeneration. A filesystem regression executes prepare
through first bind with finite synthetic F32 files and the real metadata
validators, then proves a second bind preserves every file while refusing.
Separate preexisting golden/manifest cases refuse without changing source joins.
Three focused checks and independent QA/Ruff/format/diff checks pass. Synthetic
test data provides no production hardware or readiness evidence.

## WSL global device-holder observer correction

The first actual final-controller attempt failed in the initial resource
snapshot before admission/model/kernel/update execution: an unprivileged global
`/proc/*/fd` census could not inspect root-owned PID1. Its failure remains
preserved. Root's read-only probes found `sudo -n` unavailable but the fixed
Ubuntu WSL root bridge usable. The corrected observer retains the ordinary
unprivileged scan and uses the following fixed fallback only for PermissionError
on the real `/proc`:

```
/mnt/c/Windows/System32/wsl.exe -d Ubuntu -u root -- /usr/bin/python3 -B -I FROZEN_CHECKOUT/scripts/read_only_dxg_census.py --expected-sha256 PINNED_HELPER_SHA
```

The helper is stdlib-only and reads only the fixed global `/proc` FD links/stat
identities/boot marker. It accepts no command or proc-path selector, does not
query a GPU API, load a model, write bytecode or run training. Parent invocation
has closed stdin, a minimal environment (existing WSL_INTEROP retained only for
the Windows bridge),15-second timeout and fixed Ubuntu/system interpreter; the
helper has its own10-second deadline. Complete/readonly/UID0/source-SHA/current
boot/PID-namespace proof and exact PID/start-tick/boot identities are validated.
Missing permission/bridge/protocol/identity proof fails closed with release
PENDING; it can never become an empty-holder success. Existing process/group,
CUDA PID, exact DXG holder and memory-return gates remain intact. The entire
controller/trainer stays unprivileged.

Seven new local fixture tests pass for direct/global identity observation,
fixed fallback argv, custom-path/distro refusal, malformed namespace/source/UID/
holder proofs, denied/timeout bridge behavior and unchanged new-holder refusal.
These tests invoke no privileged command or GPU. Actual fixed-helper bridge and
snapshot/release proof on WSL is still required after source integration.
Independent QA must repin changed `nine_model_pipeline.py` and the new
`scripts/read_only_dxg_census.py` in the additive critical-source ledger, then
freeze a new manifest before controller retry. Calibrated initialization,
original training data, exported model and native goldens remain unchanged.

## Native admission logging visibility correction

Actual controller02 passed272/272 CUDA kernel cases and capture portability,
then the native smoke failed its required loader/dispatch evidence despite
loading the calibrated model and generating eight tokens. The preserved server
log exposed only srv/cmn INFO/WARN at default verbosity3. Source confirms
`common_log_default_callback` maps GGML/LLAMA INFO to TRACE4; the all-nine EAGLE
loader message, packed CUDA dispatch marker and typed `W1AX_ADMISSION_TRACE`
are all emitted at that native INFO level and were filtered out.
`check_eagle_binary_native.py` now passes explicit `--log-verbosity 4`, the minimum
threshold that exposes that evidence. This changes only bounded smoke logging;
native arithmetic/model/data and exact all-nine typed CUDA trace validation are
unchanged. The mocked lifecycle regression captures the actual Popen argv and
ties its threshold to the current native log header/callback; scoped typed
dispatch/projection rejection tests remain intact. Actual native admission must
still execute and prove all nine CUDA operators; generated tokens or newly
visible messages alone cannot grant readiness. Controller02/failure artifacts
remain preserved, and the source/QA/manifest must be repinned before retry.

## Torch physical UUID representation correction

Actual controller03 passed kernel, portability and all-nine typed native gates;
its zero-update model/backward execution passed finite later gradients and full
F32 moment reservation but the consumer refused the hardware UUID. Torch2.14
reports the complete bare UUID `44ceb8b5-b67a-a317-fee3-f01c9201994e`; the actual
NVIDIA observer/lease reports `GPU-44ceb8b5-b67a-a317-fee3-f01c9201994e`.
The producer now validates a complete8-4-4-4-12 hexadecimal UUID and emits its
canonical NVIDIA `GPU-` representation. The live typed training admission uses
the same conversion on freshly queried device properties. Strict stored-record
equality is unchanged: another physical UUID, an incomplete/MIG/ordinal value
or an old bare receipt against canonical expected identity still refuses.
No old receipt is edited or promoted; root's in-memory normalization was a
diagnostic only. Fresh source/QA/manifest and actual backward admission are
required before updates. Four focused tests and independent QA/Ruff/format/diff
checks pass for producer/live-query normalization and unchanged mismatch refusal;
no device query or GPU execution occurred in these mocked property tests.
