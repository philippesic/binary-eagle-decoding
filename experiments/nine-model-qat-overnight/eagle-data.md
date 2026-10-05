# EAGLE production data and scale-only initialization

Owner: `/root/overnight_eagle_data`. Only additive helper/tests and this report
are owned here. The sole RTX5080 operator performs remote execution; no GPU or
remote operation was performed by this feature owner.

## Verified locators and reuse boundary

Operator metadata inspection found the original completed preparation receipt at:

```
/home/philip/binary-eagle-decoding/runs/qat-optimization-readiness/retained-capture-adoption-20261002-01/checkout/runs/retained-a8-a1-preparation-20261002-01/preparation-ready.json
```

SHA256 is `bdfa56f8b10e44e82a6d807a71f32d68c39143af7094e6f8f0da63504d41a498`.
The original provider binding is
`runs/qat-optimization-readiness/prepared-full-source-20261003-01/provider-binding.json`.
Its retained stages are
`runs/qat-optimization-readiness/retained-capture-adoption-20261002-01/inputs/retained-stages.json`,
SHA256 `ccc43a104a3a09e1a910746c319f55df33761bec5616378254a90e9339dbdafe`.
Original resolved configuration is beside the ready receipt. The historical
preparation covers 10,000 TRAIN prompts and 3,899,930 supervised rows across 320
TRAIN providers. This is existing native teacher data, not the ineligible local
diagnostic or the nine-chain block pilot. Metadata inspection is not current
production actor or hardware admission.

Original HF EAGLE drafter is
`/home/philip/binary-eagle-decoding/models/hf/Qwen3-4B_eagle3/model.safetensors`,
SHA256 `58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e`.
Original F16 draft GGUF is `models/gguf/Qwen3-4B-eagle3-f16.gguf`,
SHA256 `c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1`.
Original target is `models/gguf/Qwen3-4B-f16.gguf`,
SHA256 `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`.

Current `train_nine_model_qat.eagle_inputs` reuses these data through
`train_continuous_w1ax.load_config`, `prepared_continuous_provider.authenticate`,
`PreparedProvider` and `create_current_native_child`. The completed checkpoint
is identity evidence only; a new lane initializes its own model and optimizer.
Historical receipt/source declarations remain unchanged. Current source/native,
actual model/backward/full-moment memory, resume and native trajectory gates are
separate and still required. A source rebind alone cannot supply those gates.

## Additive initializer

`prepare_eagle_production_initializer.py` authenticates that complete original
preparation using its ordinary helper once, joins retained source positions to
original TRAIN opaque indices, then loads only native capture children needed
for selected calibration prompts. It does not recapture or rerun a whole-corpus
semantic audit loop; each selected child still uses the native provider's exact
receipt/byte/trace checks.

Default calibration is 32 fit and 16 validation prompts **per domain**, for
prose, code and reasoning, with 16 accepted-prefix raw feature rows per prompt:
1,536 fit rows and 768 validation rows. Prompt IDs, corpus groups, topics and
content hashes are disjoint across the entire selection. This is an operational
calibration allocation, not a training exposure cap or final quality sample.
Validation is drawn from TRAIN for calibration diagnostics; the original
separate development and sealed evaluation remain intact. The ordinary trainer
can expose the original full TRAIN corpus, including these calibration holdouts;
the helper makes no claim of held-out QAT model quality.

The teacher is original BF16 `fc.weight` promoted to F32, evaluated on original
native F32 raw taps 2/18/33 before EAGLE fusion/norm. Each selected row retains
original capture/shard/round/feature index, exact accepted-prefix tokens and raw
input hash, plus the native recurrent trace's CE mask. No HF-derived target
feature/logit replacement is introduced.

Fit uses fixed deployed A8 (or separately requested direct A1) arithmetic,
nonnegative row scales, orientation rescue OFF and coordinate flips zero.
Latents preserve original EAGLE reference magnitude ±0.5. Legal zero scales
remain zero. A8/A1 are separate artifacts, not a curriculum transition.
No training optimizer is created and no current GPU admission is granted.

Outputs: `initializer.npz` with only `fc.latent` and `fc.scale`,
`initializer.json` with exact loader locator/activation/latent policy/reference
SHA, `raw-input.npy`, `row-evidence.jsonl`, and `report.json` with full corpus
identity, source weight/config/module pins and separate per-domain validation.
The selected sparse initializer is consumed by the existing nine-projection
trainer preparation/export path; it is not itself a complete native GGUF.

## Operator command plan

Run from the newly frozen current checkout, using the locked project Python
runtime, under the sole operator's CPU preparation supervisor. Set numeric
thread counts for this bounded CPU fit; no GPU allocation is required.
`QAT_PREPARED...` gate environment is not needed by this helper.

```sh
python scripts/prepare_eagle_production_initializer.py \
  --config /home/philip/binary-eagle-decoding/runs/qat-optimization-readiness/retained-capture-adoption-20261002-01/checkout/runs/retained-a8-a1-preparation-20261002-01/resolved_config.json \
  --prepared-run-dir /home/philip/binary-eagle-decoding/runs/qat-optimization-readiness/retained-capture-adoption-20261002-01/checkout/runs/retained-a8-a1-preparation-20261002-01 \
  --prepared-ready-sha256 bdfa56f8b10e44e82a6d807a71f32d68c39143af7094e6f8f0da63504d41a498 \
  --source-weights /home/philip/binary-eagle-decoding/models/hf/Qwen3-4B_eagle3/model.safetensors \
  --source-weights-sha256 58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e \
  --output-dir /home/philip/binary-eagle-decoding/data/nine-model-overnight/eagle-fixed-a8-initializer-01 \
  --activation-bits 8 \
  --fit-prompts-per-domain 32 --validation-prompts-per-domain 16 \
  --rows-per-prompt 16 --max-fit-seconds 300
```

The original resolved config embeds its stages; if the operator uses its
original source config instead, supply the pinned retained `--stages-manifest`.
No helper/path alteration is permitted to paper over a genuine failed receipt,
source-index, native trace, model snapshot or domain quota check. CPU fit timeout
is a calibration preparation failure, not permission to weaken the data.

Next, the bundle owner binds the output initializer into the current single-lane
EAGLE config, invokes current zero-update preparation to serialize all nine
projections, exports the complete calibrated native candidate, and obtains
fresh admission. The operator then starts long QAT only after those gates pass.

## Acceptance checks and remaining work

Four focused local tests pass: alias-disjoint selection, missing code validation
refusal, missing opaque identity/invalid quota refusal, and negative-correlation
scale-only preservation of ±0.5 with rescue disabled. Changed-file Ruff and
format checks pass. No actual production calibration or model test has yet run.
Independent QA source review passed on receipt ancestry, native accepted-prefix
joins, three-domain split and scale-only/reference policy. It explicitly did
not infer actual initialized production input readiness from the helper tests.

Remaining: operator executes authentic-source calibration once, verifies its
rows and output SHA bindings; bundle owner produces current complete actor and
native export/admission; independent QA checks the resulting packet; sole
operator launches and monitors the admitted lane. Hardware and training
readiness remain PENDING until their authoritative reports exist.

## Full calibrated native actor preparation

The bundle owner owns the exact selected configuration and native admission
interfaces; its [production plan](production-plan.md) specifies this sequence:

```sh
python scripts/train_nine_model_qat.py \
  --config SELECTED_A8_CONFIG --run-dir INITIAL_PREPARE \
  --bundle-sha256 INITIAL_PREPARATION_REQUEST_SHA256 \
  --stage-name eagle_a8/initial-prepare \
  --completion-output INITIAL_PREPARATION_RECEIPT \
  --allow-cuda --prepare-only

python scripts/export_recurrent_binary.py \
  --base /home/philip/binary-eagle-decoding/models/gguf/Qwen3-4B-eagle3-f16.gguf \
  --checkpoint INITIAL_PREPARE/checkpoints/step-000000000000-e000000-r000000000000/A8/joint.npz \
  --manifest INITIAL_PREPARE/checkpoints/step-000000000000-e000000-r000000000000/A8/joint.json \
  --output CALIBRATED_INITIAL_GGUF --audit INITIAL_EXPORT_AUDIT
```

Uppercase paths/digest are explicit generated request/config/artifact locators
from that plan, not literal shell commands ready for execution. Initial request
SHA binds the real preparation request and is not a fabricated production bundle
or admission. The operator runs preparation under its own `remote_job` and
verifies current source, GPU availability and exclusive ownership first.

The existing trainer installs calibrated FC latents/scales before `build_lanes`
and checkpoint publication. Its zero-update actual forward/backward and
full-Adam-moment reservation are model preparation evidence. The saved all-nine
NPZ/manifest must retain zero optimizer updates and actual calibrated FC; the
serializer must pass every selected projection's packed-bit/scale hash and
unchanged protected tensor bytes/metadata. `serialization_audit_passed` alone
is not native loader/graph proof. Native admission must load and exercise this
exact **calibrated initial GGUF** joined to its export audit and sparse
initializer; original F16/unmodified W1 models cannot stand in for it.

The positive-trained-endpoint `export_nine_model_candidate.py` wrapper rejects
step zero and is intentionally not used for this initial actor. Production
training starts only after separate fresh current candidate admission.
