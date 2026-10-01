# Binary optimizer and curriculum controls

Prepared 2026-10-01 in `feature/qat-recipes`, from the active
`qat-optimization-readiness` checkpoint. This is a bounded implementation and
synthetic CPU validation report. No real-data optimizer updates, model loading,
accelerator discovery, remote/GPU use, or sealed-final reads were performed.
The root owns integration, the project goal/status checkpoint, and publication.

## Delivered APIs

`src/w1a1_eagle/qat_optimization.py` provides:

- `BinaryOptimizationConfig`: baseline ±0.5 initialization, AdamW, sign/scale
  LR 1e-3/1e-5, betas .9/.999, epsilon 1e-8, zero decay, `foreach=False`,
  joint norm clipping at 1. Finite bounds reject invalid settings.
- `initialize_latents_`: selectable positive magnitude in (0,1], preserving
  deployed signs, zero-positive convention, and scales. ±0.1 is an explicit
  lower-inertia experiment; initialization is never an exact-resume operation.
- `make_binary_optimizer`: AdamW or SGD/momentum with distinct sign/scale
  groups; optional explicitly rated additional parameters form a third group.
  Bop is deferred: no EAGLE evidence justifies its gradient-scale-sensitive
  flip threshold in this bounded deliverable.
- `transform_binary_gradients_`: baseline or bounded weight-unit sign gradient
  normalization, baseline or inverse-square-root-fan-in scale gradients, and
  joint or per-family clipping. At zero row scale the normalized sign gradient
  stays zero; additive scale gradients can still revive the row. The weight
  rule is an explicit adapted optimizer experiment, not a claim to reproduce
  the full ParetoQ recipe. Existing inclusive clipped sign STE and hard native
  forward remain unchanged. Additional parameters share joint clipping, or
  receive a separate max_grad_norm bound with per-family clipping.
- `project_binary_parameters_`: finite latent/raw-scale gate, inclusive [-1,1]
  latent projection, and nonnegative row-scale projection.
- `optimizer_checkpoint` / `load_optimizer_checkpoint`: stable sorted named
  binary layout, recipe, extra ordered shapes/dtypes/LR, and external model/data
  contract digest. Ownership, option/LR bounds, state inventory, moment shapes,
  finite moments and nonnegative second moments are checked before loading.
  Include additional parameter *names* in the external contract; shape alone
  cannot identify same-shaped learned parameters.
- `SignFlipDiagnostics`: initial/previous/ever-flipped/disagreement packed
  CPU masks; net Hamming distance, unique observed changed bits, cumulative
  observed flips and returns to the initial sign, and disagreement sustained
  across adjacent observations. Four masks cost approximately 0.5 bytes/sign
  (about 109 MB for 218,234,880 signs), plus per-module copy/unpack temporaries.
  Sampling intervals can miss chatter. State preserves sampled history exactly;
  pass `resumed_step` when a checkpoint lies after its latest observation.

`src/w1a1_eagle/qat_curriculum.py` provides:

- `depth_weighted_supported_ce(logits,audit,depths,decay=1)`: exact baseline
  default, normalized supported-label weights (e.g. .8^depth), attached later
  state gradients, unchanged unsupported-label quality denominator.
  `depth_exposure` reports supported weighted and all-valid exposures separately.
  The trainer must supply audited per-row depths in matching order.
- `PrecisionStage(bits,gpu_seconds,max_updates)` and `CurriculumConfig`: direct
  A1, short A8→A1 (default proposed .2/.8 time allocation), or an explicit
  A8→A4→A1 ladder. The ladder is prepared, not a recommended default.
- `CurriculumState`: bounded measured training GPU time, update/row/token/
  supported-label counters by phase, separate capture/audit/export/development
  overhead, active phase, global warmup count, JSON-serializable exact resume,
  and immutable budget/data/model digests. `can_start_update` checks a supplied
  upper duration bound; an over-budget measured update fails closed. Use
  `finish_phase` to preserve unused budget when another bounded update cannot
  fit. First destination update requires recorded transition ancestry.
- `transfer_precision_state_`: exact latent magnitudes and additive scale
  representation transfer to independent row modules, frozen-bias/shape/
  projection checks, fresh optimizer state, global exposure/warmup counters
  retained. Initial-scale/offset pairs are copied to avoid cancellation from
  reconstructing offsets against a different initial scale. `record_transition`
  also supports a runner-validated in-place precision change without keeping a
  second model resident. The runner owns the exact-state and fresh-optimizer
  proof for that in-place path. Learned activation/correction states require
  their own explicit phase transition contract.
- `plan_curriculum_refresh`: active checkpoint/export/train-prompt/native-teacher
  and refresh-round bindings, delegating exact-prefix matching and file hashes
  to existing `trajectory_refresh.build_plan`. Changed prefixes retain new
  native label/feature capture requests; old rows are never relabeled. Plans
  stay training-ineligible until capture auditing/provider admission creates
  an explicit new corpus contract. Changing that corpus is not exact resume.

## Acceptance checks

Python: repository `.venv/bin/python`, `PYTHONPATH=src`. Hardware: local macOS
CPU only. Synthetic model precision: F32 row parameters and gradients, with
hard simulated A1/A8 activations. No CPU result validates SM75 throughput.

- `python -m unittest discover -s tests -p 'test_qat_optimization.py' -v`:
  8 passed. Covers finite guards, same initial hard forward at lower inertia,
  baseline AdamW equivalence with stable clipping order, zero-scale/fan-in
  rules, additional groups, SGD, projections, ownership, byte-exact resumed
  subsequent parameters, corrupt moments/LR/contracts, packed persistent
  flip/flip-back state and resume between sampled observations.
- `python -m unittest discover -s tests -p 'test_qat_curriculum.py' -v`:
  5 passed. Covers normalized weighted CE with later-state gradient, unchanged
  unsupported exposure, strict phase budgets/JSON resume, preserved latent
  magnitudes/scales with fresh moments, frozen/nonfinite transfer rejection,
  and exact-prefix new native capture requests with stale/tampered source guards.
- `python -m unittest discover -s tests -p 'test_qat_*.py' -v`:
  35 passed at the initial integrated local checkpoint. The final targeted
  rerun passed after stricter transition/moment/cache/padding guards.

The baseline comparison uses the API's sorted named parameter ordering.
Different reduction ordering in legacy clipping can produce last-bit numeric
differences; this does not change its optimizer recipe and the existing frozen
driver is not modified by this delivery.

## Integration and remaining work

Root must wire initialization before optimizer creation, apply gradient rules
after backward and before exactly one clip/step, project after step, checkpoint
the recipe/diagnostic state with complete named model/data contracts, and retain
strict independent lane ownership. The curriculum runner must admit budgets,
keep native/held-out gates, account source A8 cost, and record each verified
precision transition before the first destination update. No real-data run is
authorized by these APIs or tests.

Prepared controls are not demonstrated quality improvements. There is no GPU
memory/timing result, acceptance result, or native-ready claim in this report.
The next authorized comparison must keep target/verifier/data/held-out
contracts fixed and measure native acceptance, latency, and total throughput
against Q4_0 EAGLE. FP16 EAGLE remains diagnostic. Depth weighting, optimizer
changes, precision staging, and refreshed trajectories should be separate
experiment options rather than simultaneous unmeasured changes.

## Follow-up: provider, refresh actor, and readiness integration

The isolated follow-up owns `w1ax_continuous_stages.py`,
`w1ax_capture_provider.py`, `check_continuous_w1ax_readiness.py`, and explicit
A4 enumeration in `run_binary_head_capture.py`. Root's integrated optimized
core/exporter files were copied as **uncommitted test dependencies only**;
the delivery does not commit those files or change the llama.cpp gitlink.

- New `w1ax_continuous_readiness_v2` admits only precision entries with their
  own `w1ax_continuous_precision_gate_v2`. A4 cannot inherit A8/A1 or a legacy
  gate. Gate-v2 independently validates the strict exporter manifest/audit,
  recomputed projection/correction tensor hashes, actual hashed packed dispatch
  log, requested recipe, deduplicated trainable-family counts, and finite
  backward tensors. Existing v1 A8/A1 proofs retain their original contract.
- Native A4 capture explicitly requires `CUDA packed W1A4 BITSERIAL dispatch`;
  the native capture helper accepts A4 without changing target precision,
  token/microshard/graph limits, or cancellation-group ownership.
- New `w1ax_exact_prefix_refresh_v3` supports exporter manifests 2/3/4 with
  strict `validate_actor_export`: exact field inventory, six learned shared
  boundaries, correction descriptor/factor hashes, projection NPZ inventory,
  effective numeric values, checkpoint/export/source bindings. Legacy receipt
  v2 remains restricted to baseline manifest2 A8/A1. Fresh captures still have
  `training_eligible=False` and `changed_prefix_labels_reused=False`; existing
  native teacher/prefix/feature audits remain authoritative.
- Providers reject requested learned/correction configurations that differ
  from their own versioned precision proof. Refresh provider adoption builds
  each configuration from its corresponding independently admitted proof.
- `checkpoint_joint_config` builds fixed/learned/fusion modules from the
  strict manifest. `_load_checkpoint` validates all arrays and attachments
  before changing any parameter, restores learned scalar thresholds/clips and
  effective correction tensors, and rejects undeclared extras. It returns
  `checkpoint_recipe(manifest)`.
- `run_gate(..., checkpoint_bundle={checkpoint,checkpoint_manifest})` gates a
  specific declared actor rather than silently using checkpoint-zero. It owns
  every attached parameter exactly once through `joint_optimizer`, requires
  finite extra-family backward gradients and nonzero binary gradients, and
  replays the head using its actual learned quantizer. A provided
  `measurement_binding={source_sha256,training_runtime,recipe,native_commit}`
  produces an actual `qat_native_measurements_v1` native-decisions artifact
  for the independent readiness harness. Its state hash helper is supplied by
  the separately owned `check_qat_optimization_readiness` harness.

CPU synthetic checks passed: 5 new `test_qat_recipe_provider` checks, 6 existing
`test_continuous_readiness`, 4 `test_w1ax_refreshed_provider`, and 19 discovered
`test_w1ax_continuous_stages` checks (including its imported capture fixtures).
Tests use real synthetic NPZ arrays and hashed fake artifacts, rejecting
missing A4 proof, unknown audit options, changed thresholds/correction hashes,
stale actor/prefix reuse, and mismatched loader shapes. Native capture execution
and CUDA cache release are mocked in the dispatch/cancellation fixture. No
CUDA query, native/GPU execution, model data, remote use, or real optimizer
update was performed. Python compilation and `git diff --check` passed.

Remaining integration boundary: export NPZ stores effective scales and F16
correction factors; it does not preserve the original F32 factor masters or
additive initial-scale/offset decomposition. Its zero-update replay state hash
therefore identifies that exact replay state, and must not be relabeled as a
different trainer master state. The independent harness must require matching
live hashes; exact trainer-master native evidence needs a full trainer-state
restore or a separately proved deployment-effective identity. Dedicated
learned packing and raw-input/nonzero fusion artifacts are not manufactured
from trajectory RMS measurements. Actual GPU gates remain unexecuted, and
this follow-up asserts no native-ready status or quality improvement.

Integration fingerprint clarification: root now supplies
`qat_state.deployment_state_sha256(linears)` over the native effective
representation (hard signs, effective F32 scales/scalars, rounded F16 factors,
effective bias). Gate-v2 and decision artifacts use `deployment_state_sha256`,
so equivalent NPZ rehydration can bind the exact same deployed operands while
full F32 master-state hashing remains a separate no-update assertion. This
resolves the representation mismatch above without labeling different master
states equal. Production execution requires root's shared helper; the provider
fixtures do not query devices or execute the GPU producer. Cached gates cannot
silently omit or substitute a newly requested bound decision artifact.

## Follow-up: affine row mean deployment integration

Schema5 affine actors now follow the same strict receipt-v3/gate-v2 path:
`validate_actor_export` recomputes exporter-permuted F32 mean hashes and rejects
missing, wrong-dtype, nonfinite, or wrong-inventory means. Gate configuration
declares enabled fusion/all coverage; hydration copies actual NPZ F32 means
in **original checkpoint row order**, after validating every attachment and
array. Undeclared midpoint attachments cannot be ignored. Provider readiness
compares affine coverage. Legacy gate-v1 explicitly rejects schema5/affine
evidence even with a baseline activation rule or zero current means.

Native head replay accepts the actual midpoint and adds
`(sum(integer_codes)*mu)*beta` after `(integer_dot*alpha)*beta`, with the
corresponding rounded-input sum for A16. Learned head codes/scales use their
actual quantizer. All declared midpoint gradients must be present and finite;
zero gradients are legal when the quantized input sum vanishes. Affine gates
allow finite zero binary gradients at degenerate alpha/input states rather
than assuming every parameter family must move on every bounded root. Existing
non-affine binary nonzero gates remain unchanged. Every family is still owned
exactly once, and current root API's empty `midpoint` family is handled without
changing old v2 proof inventories.

The deployment fingerprint includes F32 means, so a mean change cannot inherit
another actor's native evidence. Midpoint LR, optional bound, and regularization
are training policies rather than native descriptor fields; actor hydration
uses a finite unbounded deployment configuration, while measurement receipts
retain and compare the caller's complete recipe binding on reuse. The full
trainer/curriculum state remains responsible for exact training-policy resume.

Checks: 7 `test_qat_recipe_provider` CPU tests passed, including schema5 fixed
and learned hydration, all-nine mean ownership, Q/K original-row order,
changed mean hashes, missing/F16 mean rejection, alpha=0 with nonzero mean head
replay for A1/A4/A8/A16, legitimate zero mean gradient at code sum zero, and
legacy proof refusal. Existing 6 readiness and 4 refreshed-provider checks
also passed. Python compilation and diff checks passed. Root core/schema5
exporter updates are copied test dependencies and are excluded from this
commit. No GPU/native process, device query, real-data update, or quality/speed
claim was made. The dedicated native affine-unit-sum/timing artifact remains
the independent harness's required gate; trajectory RMS alone does not grant
affine readiness.
