# Single block lane source packet

New additive adapter `scripts/prepare_block_lane_packet.py` owns one selected
`dspark_a8`, `dspark_a1`, `dflash_a8` or `dflash_a1` source packet. Worktree
`/tmp/binary-eagle-block-lane-packet`, branch `feature/block-lane-packet`, starts
from parent `a16c0d3`; native worktree is read-only at published
`ecff6d4e74814c631801df2e74f562d4ed6bd0eb`. Only this new script, its focused new
test file and this report are changed. Root owns integration and the active goal
checkpoint. Existing common builder/trainer, endpoint scripts, shared Files guard,
other candidate statuses and frozen controls/target/KV precision are untouched.

No real model or capture, GPU query, GPU process, SSH or 2080 work was performed.
The healthy EAGLE_A8 execution remains immutable and owns the 5080. All future
GPU stages must wait for its natural released slot and root coordination. No
complete-campaign, acceptance, throughput or quality claim follows from this
source feature; Q4_0 EAGLE remains the primary eventual baseline.

## Explicit supplied inputs

`inspect --inputs <JSON> --inputs-sha256 <pin>` accepts a partial descriptor with
`schema=block_lane_packet_inputs_v1` and always reports PENDING, no model/GPU use,
no production readiness and the other five candidates still remaining. `prepare`
requires the actual source inputs below. All artifact locators use the existing
canonical absolute path/SHA256 `Files` contract.

| Input | Contract |
| --- | --- |
| `candidate` | Exactly one of the four direct block cells above. |
| `runtime` | Pinned `block_lane_packet_runtime_v1` metadata: original `base_model`; `inputs.target/teacher_binary/backend_binary/block_native_binary`; exact 40-hex `native_source_revision`; `gpu_uuid`, `device_name`, `gpu_control_path`, `resource_policy`, explicit `golden_max_tokens`; optional existing `environment/build_provenance`. |
| `budget` | Existing `nine_model_selected_budget_v1`, exactly the selected candidate, `human_selected=false` with pinned `human_delegated_operational_settings` provenance. Supplied `training_limits` use real block names `max_steps/max_supervised_tokens/max_seconds/max_epochs`; no EAGLE `max_tokens` alias or invented allocation. |
| `data/data_admission` | Actual `native_block_train_v1` manifest and current-host `block_data_completed_admission_v1`. The real `BlockDataset` verifies complete source/stat joins, original TRAIN membership, target F16/KV F16, whole context/tokens, split disjointness and selected-anchor teachers. Native receipts must show CUDA computation. |
| `coverage_policy` | Pinned `block_lane_coverage_policy_v1`, `source_role=original_TRAIN`, supplied `min_unique_train_prompts/min_train_blocks` and pinned delegated authorization. Both thresholds must exceed nine to exclude the preserved development pilot, and actual TRAIN extent must meet them. This software floor does **not** establish that an arbitrary larger subset is serious enough; extent/exposure remains a root/user research decision. All three TRAIN-derived roles require all three domains. |
| `reference` | Actual pinned `weights/norm/metadata` from `extract_block_fusion_reference.py`. The canonical `block_fusion_reference_v1` must bind the original base SHA, BF16 `fc.weight`, post-FC gamma payload/shape and exact original FLOAT32 epsilon bits. |
| `initializer` | Pinned `fit_report/artifact` from the existing `fit_block_fusion.py` producer, not an invented initializer report. |
| `calibration_policy` | Pinned `block_lane_calibration_policy_v1` with supplied `rows_per_chain/max_total_rows/max_array_bytes/max_seconds` and delegated authorization. Actual fit selector positions and estimated workspace must match this policy. No bounds are selected by the adapter. |
| `objective/conditioning` | Explicit `full_probability_l1/captured_prefix` for offline DSpark, or explicit DSpark `full_probability_l1/native_greedy` with the admitted live teacher below. DFlash uses explicit source `hard_ce/captured_prefix`. DSpark never silently falls back to hard CE. |
| `checkpoint_every/resource_floors` | Supplied metadata consumed by the real source trainer. Optional geometry/optimizer-value `qat_overrides` go through its real dataclass. Recipe, precision, profile, objective, prefix, epsilon, latent encoding/policy and optimizer backend cannot be overridden. |

For an explicit live DSpark current-prefix path, `teacher` uses exactly the
existing `train_nine_model_qat.native_teacher` kwargs:
`binary/target/target_sha256/max_tokens/gpu_layers/producer_source_revision`.
`live_teacher_admission` must be an actual production
`nine_model_capture_portability_v1` PASS for the same data/target/native binary,
current teacher client/gate source, five taps, device UUID and SM120, with all
three domains' finite numeric/unchanged target-argmax checks and a closed producer.
Its token cap must cover the complete native chains. Future prepare-only smoke
still exercises the actual callback/backward/resident-memory path; this metadata
gate does not replace that proof. Offline captured-prefix packets cannot attach
an implicit live teacher.

Actual block coverage/storage, serious exposure, live-versus-offline selection,
all source inputs and independently authorized operational policies remain
PENDING. This adapter creates no corpus subset, new capture or training duration.
The indexed-teacher feature's full-context storage limits remain applicable.

## Canonical recipe and source sequence

Each candidate has its own actual fixed-arithmetic scale-only fit. The adapter
consumes `block_fusion_fit_v1`, reconstructs the real `FusionFitConfig`, requires
matching A8/A1 precision, no orientation rescue/coordinate flips/events, original
reference-magnitude latents and exact data/FC/gamma/base/epsilon/artifact pins.
Fit/validation prompt groups and positions must equal the existing source
selector under the supplied caps; no validation-based selection is introduced.
The canonical initializer locator is derived from the actual fit's
`latent_initialization` contract (`block_source_weight_magnitudes`, contiguous
little-endian absolute F32 weight hash). It preserves actual model magnitudes.

`prepare` calls `prepare_nine_model_bundle.materialize_configs` and the actual
trainer/dataclass parser for a single candidate. The chosen contract is
`ffn15_fusion`, fixed A8 or direct A1, reference magnitudes, `policy_latents` and
serial FP32 AdamW; experimental probes and A8-to-A1 warm starts stay off. Each
source request binds config and producer script hashes, input descriptor and
zero optimizer updates. Metadata source-config validation can PASS while actual
preparation/native readiness remains explicitly PENDING.

Generated `commands.json` records exact supported argv for:

1. Existing BF16 reference extraction and separate precision-specific scale-only
   fitting precede `prepare`; the packet records the existing fit argv and never
   overwrites its completed fit directory.
2. Actual `train_nine_model_qat.py --prepare-only` for the selected source config,
   request pin and new `initial-prepare` directory. This is future CUDA execution
   under sole root coordination, not authorization from the commands file.
3. Explicit `prepare_block_lane_packet.py export-initial --allow-cuda` with a
   fresh root-provided exclusive 5080 availability lease bound to the request.
4. Actual `capture_block_qat_teacher.py` with five taps, last full-vocabulary
   logits and exact replay requests selected from one original TRAIN parent per
   domain. Tokens preserve the full prompt and the supplied bounded golden
   prefix; decode history is cropped only at the final intersecting partition.
   Training data/context/anchors are never cropped by golden selection.
5. Metadata `bind` with exact native receipt bytes and independent portable QA.
6. Actual `prepare_nine_model_bundle.py --inputs production-inputs.json
   --materialize-admission-plan admission-plan.json`.
7. Metadata `finalize` with the actual plan SHA; it reuses
   `prepare_nine_model_lane.build_lane`, binds the existing admission-plan locator
   and preserves the other five candidates. The wrapper still reports
   PENDING fresh native admission and `production_ready=false`.

The source commands explicitly set `execution_authorized_by_this_file=false`.
They require real pins for receipts/QA/plan/availability before execution; no
model/GPU command or watcher was launched here.

## Zero-update export bridge and bind

The actual block prepare-only producer returns `nine_model_preparation_v1` with
`checkpoint/smoke/smoke_contract/training_memory/resources`. It does **not** emit
EAGLE's `stage/config/source` aliases or an NPZ. Its checkpoint receipt has
`block_qat_checkpoint_v1`, path/SHA, contract/source digests, committed cursor.
`export_nine_model_candidate.py` correctly refuses zero-update trained endpoints.
The new adapter bridges only initial serialization, using the actual
`block_inputs/block_optimizer/load_block_checkpoint/export_block_checkpoint`
and `export_block_binary.export_model` APIs.

Forcing this restore to CPU would fail the immutable runtime contract:
`block_training._runtime` binds device type and CUDA version/device/capability,
Torch, source files and optimizer backend. The producer therefore keeps the
original CUDA config/runtime, makes no header changes and is explicitly gated:
nonblocking `~/.config/binary-eagle-decoding/rtx5080-campaign.lock` first,
nonblocking private `cuda-0.owner.lock` second, valid same-UUID 5080 lease,
unpaused/unchanged ownership, STOP/source checks and fresh **empty actual CUDA
and DXG censuses** plus supplied resource floors under the global lock before
any model construction. Healthy ownership refuses without disturbing it.

Initial joins reject positive/mixed/charged cursors, wrong initial data-order
cursor, source/config/contract/path/commit changes, fewer than 16 selected
projections, optimizer moments or arithmetic mismatch. Actual restore enforces
empty optimizer state. The real serializer publishes complete NPZ/manifest/
packed GGUF; bind checks exact canonical projection names/shapes and all 16
packed/latent/scale audit hashes, original base/checkpoint/manifest/output pins,
private head/embedding/norm/attention/Markov protected tensor proof and the exact
original gamma payload hash. Native runtime readiness remains PENDING.

The export process itself cannot claim the GPU free while its Torch CUDA context
still exists: its receipt explicitly leaves resource release PENDING until the
parent reaps the process and verifies owned context/memory return. Root's future
stage supervisor must perform that release proof before another GPU stage.

Native bind joins every golden to the same input data/admission/runtime, selected
TRAIN chain and parent receipt, exact tokens, absolute decode history, ancestry,
taps and full vocabulary. It validates through real native receipt and
`NativeCaptureGoldens` contracts. Independent existing QA ledger requirements
apply to only the chosen profile and current source inventory, including this
adapter. The adapter never rewrites other profiles' statuses or creates a full
six-candidate/three-control PASS. Bind and finalize retain native admission as
PENDING; no actual production packet was constructed by this task.

## Acceptance checks and remaining work

On local macOS arm64 CPU, the new suite passed **12/12**. It uses tiny synthetic
BF16 GGUFs and actual source extraction/fit/dataclass/builder/smoke/checkpoint/
serializer APIs; schema-shaped native metadata remains a fixture and certifies
no actual target/GPU result. Checks cover all four source configs; explicit loss/
prefix/probe/precision and pilot thresholds; exact epsilon/source fit positions;
current-host artifact mutation; real zero-state smoke/checkpoint/restore and
complete protected exports; positive/wrong source/contract/optimizer/cursor
refusal; exact parent five-tap golden binding; real admission-plan/single-lane
metadata sequence; explicit current-prefix gate fields; CPU-to-CUDA runtime
relabel refusal; and actual nonblocking global/private lock refusal before any
GPU query/model call.

Independent Luna review reran 12/12 and Ruff/whitespace checks. Existing suites
also passed: staged lane7, nine-model training30, checkpoint contract3, block
export3 and EAGLE packet14; four optional actual-native block fixture cases were
skipped because no native model execution was selected. These are 69 passing
source/CPU cases plus four skips, not CUDA/SM120 or SM75 measurements. Review
caught an informational `--help` argv mistakenly labeled admission-plan creation;
it was replaced with the real supported flags and a regression assertion.

Raw logs are ignored under
`/tmp/binary-eagle-block-lane-packet/results/block-lane-packet-qa/`, including
`final-suite-12.log`; the first existing-suite invocation lacked PYTHONPATH and
was rerun correctly with `src:scripts:tests`. Venv:
`/Users/pippo/github/binary-eagle-decoding/.venv/bin/python`. Temporary tiny
fixtures cleaned themselves. No raw weights, data, captures or runs enter Git.

Next: root review/integration and active-goal checkpoint; preserve QA logs before
retiring this worktree. Real original BF16 reference/precision-specific fits,
serious current-host TRAIN admission/exposure policy, explicit prefix/objective,
operational budget/runtime/source pins and independent actual portable QA are
still needed. After healthy EAGLE_A8 naturally releases, root may coordinate the
real zero-update preparation/export/goldens/fresh native admission. Future
positive QAT and endpoint evaluation remain separate work; existing controls,
target F16 and F16 KV remain immutable.
