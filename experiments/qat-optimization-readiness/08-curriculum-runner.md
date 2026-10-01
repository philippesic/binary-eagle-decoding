# Runnable precision curriculum readiness

## Deliverable and acceptance

`src/w1a1_eagle/qat_curriculum_runner.py` and
`scripts/train_qat_curriculum.py` implement one-model recurrent training for
**direct A1**, **A8 → A1**, and **A8 → A4 → A1**. This is an executable trainer,
not only a curriculum configuration helper. Acceptance is CPU fixture proof of
stage transitions, exact optimizer/model/RNG/cursor resume, all-stage
zero-update preparation, and fail-closed source/budget/resource admission.

The 13 dedicated CPU tests pass with integration dependencies from commit
`36e53a8` (binary curriculum, learned activations, computation/cache/head, and
fusion correction). The combined runner/curriculum/optimizer/recurrent QAT suite passes **32 tests**.
Focused Ruff checks and Python compilation also pass.
All checks use synthetic F32 CPU tensors on an **Apple M3 Max**,
**PyTorch 2.14.0**. Activation precisions are simulated A1/A4/A8 as declared. No CUDA, Metal,
remote operation, real capture/model loading, optimizer run on real data,
native execution, or sealed-final access occurred. These checks establish
runner readiness, not Q4_0 acceptance or SM75/SM120 performance.

## Execution contract

The single installed drafter and its nine row projection parameters survive
stage switches. The latent values retain their magnitudes; effective scales,
frozen biases, and the attached FC correction retain their exact values and
parameter identities. A new activation bank uses the destination precision;
A1 thresholds initialize at zero, and A4/A8 clip ratios initialize at one.
Every switch creates a fresh optimizer including activation and correction
parameters. Global warmup and per-phase exposure counters continue across
switches. Each teacher round rebuilds the current-student prefix cache and
owns one attached recurrent autograd graph. Shared hard-sign reuse ends
before backward; no graph/cache survives an optimizer update.

Preparation performs forward/backward at **every declared stage**, with no
optimizer calls. It restores the original model, activation values and RNG,
saves checkpoint zero, and publishes a preparation report. CUDA preparation
also allocates two moment-sized tensors per trainable parameter to expose
Adam-like memory occupancy without an optimizer update. Resume performs a
current-stage backward admission check without changing model or optimizer
state. STOP/SIGINT/SIGTERM/SIGHUP stop at a round boundary and save progress.

Checkpoint payloads contain canonical six-boundary learned scalars exactly
once, core binary operands, FC correction state, optimizer, RNG, curriculum,
model phase and provider cursor. Payload fsync/rename precedes an atomic,
hashed latest pointer. A source-phase boundary checkpoint exists before
contract mutation; a second checkpoint publishes the switched fresh
optimizer. Either boundary or a mid-stage checkpoint resumes exactly. Shared
alias entries, changed frozen operands, changed precision/budgets/config,
source ancestry, math hashes or runtime/hardware identity are refused.
Checkpoint retention is bounded. The latest pointer retains synchronized
residency through payload serialization; reconstructing a model on resume
adds its measured startup time rather than resetting cumulative occupancy.

## GPU budget and source admission

There are two explicit budget gates. CurriculumState binds synchronized
forward/backward/optimizer wall seconds per phase, plus max updates. A declared
upper-bound reservation must fit before another update starts; otherwise a
nonempty phase finishes with unused budget. Measured stage overrun fails
before publishing that update. The runner separately measures conservative
model-resident wall occupancy, including CPU/provider waiting, smoke, switches
and checkpoints while CUDA is initialized. This is reported as `gpu_seconds`
and must fit the sum of declared stage GPU budgets. CPU fixtures use the same
clock policy but are labeled CPU, never accelerator performance. Smoke and
transition durations are separately reported. Capture/audit/development are
not started by this runner and are not silently charged as training updates.

Budget failures publish `budget_failed=true` and refuse exact resume under
the old budget. The user must select an explicit new experiment budget.
A budget declaration cannot eliminate a long individual kernel; an overrun
can occur, but it cannot be silently credited to another phase or resumed as
successful progress. Failed updates recover only their prior atomic model
checkpoint; failed budget runs remain rejected.

CUDA requires explicit `start=True`, `allow_cuda=True`, an enabled accelerator
QAT config, `load_models_cpu`, and a fresh-source revalidator. CPU/host memory
admission precedes model loading. Whole-device free memory, allocator peak,
reserved memory, disk free space, and checkpoint CPU-buffer headroom are
checked. CLI ownership locks share the existing `cuda-0.owner.lock` with the
continuous trainer. Only declared CUDA hardware can start.

The provider must expose train split, `training_eligible=True`,
`full_body_qat_eligible=True`, and `readiness_scope="full_body_qat"`.
Calibration-only and development/final providers fail. The CLI reconstructs
each stage's provider through its audited factory, compares immutable source
metadata, and never loads a model or discovers CUDA by default. Factory
reconstruction is also required at initialization, each epoch/stage, and
resume; cached metadata alone cannot authorize CUDA. Provider implementation
and critical math/runtime hashes are bound in the checkpoint contract.

**New real-source admission remains necessary.** Existing v2 continuous
readiness and refreshed-capture ancestry receipts authorize A1/A8, not a new
A4 or learned/fusion curriculum recipe. Do not relabel that old readiness as
A4. Supply independently audited per-stage readiness through a new provider
manifest/factory while preserving source ancestry, split, masks, labels,
precision and native verifier contracts. A changed corpus is a new experiment,
not exact resume. The runner rejects any stage the factory cannot admit.

## CLI contract and integration

A JSON config uses `schema="qat_curriculum_experiment_v1"`, with:

- `curriculum`: `stages` of `activation_bits`, positive `gpu_seconds`, and
  positive `max_updates`; `optimizer_transition="fresh"`.
- `qat`: the complete JointQATConfig, including a `contract` dictionary,
  `device="cuda:0"`, and `allow_accelerator=true`; the initial activation bits
  must match the first stage. Optional learned/binary/fusion recipes use the
  root's validated config fields.
- `runner`: optional RunnerConfig safety, warmup, reservation and retention
  fields. These fields are frozen for exact resume.
- `provider`: `factory="MODULE:FACTORY"` and an absolute audited train manifest.
- `hardware`: exact `device_name` and `compute_capability`.

```sh
python scripts/train_qat_curriculum.py --config NEW_CURRICULUM.json
python scripts/train_qat_curriculum.py --config NEW_CURRICULUM.json \
  --start --allow-cuda --prepare-only --run-dir NEW_RUN_DIRECTORY
```

The first command validates config and source readiness only. The second
requires a separately authorized manual CUDA launch and stops with zero real
updates. Omitting `--prepare-only` starts optimization and is not authorized
by this readiness work. `--resume` requires the existing resolved config and
complete atomic checkpoint; failed budgets and changed contracts are rejected.

Goal handoff: integrate the four owned files after prerequisite root recipe
commits, run the dedicated tests in the integration checkout, record this
report/checkpoint in `docs/goals/qat-optimization-readiness.md`, and preserve
preparation-only scope. Future work is new per-stage provider/native admission,
manual CUDA zero-update preparation, then a user-owned optimizer budget decision.
