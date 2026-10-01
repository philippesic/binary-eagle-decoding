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
