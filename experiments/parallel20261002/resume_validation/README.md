# Joint curriculum resume validation

October 2, 2026. Third CPU slate, packet 2. Deliverable: executable semantic
validator, candidate-object restore prototype, isolated integration patch and
independent adversarial checks. Production `src/` and current QAT jobs are
unchanged. This report addresses the previously confirmed malformed-but-rehashed
optimizer-state gap; ordinary crash recovery was not rerun.

## Result

Five owner tests and four independent tests pass on arm64 CPU, PyTorch 2.14.0.
The owner checks seven fixed/combined checkpoint positions (updates 0, 1, 2, 3,
4 and completed 6, with both fixed and combined update 1), exact next updates
against the unchanged runner, and binary-recipe AdamW/SGD with momentum 0/0.8.
Trainable masters, auxiliary parameters and moments are F32; tiny fixture
frozen embeddings use FP16. No native, GPU, acceptance or throughput result.

The independent validator checks 11 malformed optimizer cases on combined
learned/fusion/affine checkpoints. Every rejection preserves live tensor values,
parameter identity/storage/trainability/gradients, optimizer, curriculum,
cursor, RNG, timing and checkpoint fields. Valid staging also preserves live
state before publication. Its separate midphase/boundary continuation matches
uninterrupted execution; measured wall times and source-checkpoint hashes are
excluded only from comparisons between separate runs. See
[the independent report](independent_validation.md).

## Bound validation contract

`reference/staged_resume.py` uses actual `joint_optimizer` construction and
`joint_parameter_families`. Family order is sign, scale, then present activation,
fusion and midpoint groups. Default core order follows the installed linears;
`BinaryOptimizationConfig` core order follows sorted projection paths. Shared
activation parameters appear once. Serialized IDs must exactly match the
fresh optimizer's IDs, group/order/count and ownership. The combined fixture
has 9 signs, 9 scales, 6 activation scalars, 3 correction parameters and 9
midpoint parameters.

All group options except effective LR are immutable, with type-aware equality:
the actual constructor's betas/eps/momentum/decay/foreach/capturable/fused and
other admitted-version fields are checked, rather than invented defaults.
Only actual AdamW and SGD classes are admitted. AdamW state requires exactly
`step`, `exp_avg`, `exp_avg_sq`; SGD with positive momentum requires only
`momentum_buffer`, and zero-momentum SGD has no state. Moments must be detached,
dense CPU tensors of parameter shape/dtype, finite; second moments must be
nonnegative. Adam steps are scalar, nonnegative, integral and no greater than
updates in the saved model phase. Step dtype matches actual PyTorch
`Adam._init_group` / `_get_scalar_dtype`: F32 with the measured default dtype,
F64 when the default is F64. Per-parameter steps are **not** equated to global
update count.

Effective saved LR is exactly `base * min(1, global_updates/max(1,warmup))`
after at least one update in the model phase. At a fresh phase with zero
updates, the current constructor's base LR is valid even when earlier phases
have global progress; this matches `_bind_phase`. A pending boundary checkpoint
whose model is still in the preceding phase uses its preceding optimizer's
last global warmup rate. Its next update still uses the runner's unchanged
`global_updates+1` schedule.

Empty fresh-phase state is accepted. Missing moments can mean legitimate
`grad=None`; missing state alone cannot establish corruption. One owner fixture
performs an actual tiny zero-gradient Adam step for only one parameter while
all other gradients are `None`, producing and restoring only one state entry.
An optional named participation receipt catches omissions and wrong Adam steps
when exact per-parameter participation counts were recorded at save time.
Current production checkpoints lack that receipt; the prototype neither
fabricates it nor claims to detect all omissions without it. Adding receipt
capture is a separate save-format adoption decision.

## Staging and memory

The prototype deserializes independently, validates candidate curriculum and
core state, copies module containers as metadata, and replaces candidate core
parameters/buffers with checkpoint-backed tensor objects. Frozen non-core model
tensors remain shared. Learned activation is constructed for the saved phase;
fusion and midpoint modules are detached candidate copies. The registered
canonical affine bank is rebound **before** building `NativeStepAdapter`, whose
freezing pass would otherwise affect old live parameters retained through a
shell bank. Existing actual model loaders validate and populate only candidate
objects; optimizer loading targets a fresh candidate optimizer. Python, NumPy
and Torch RNG are checked using local generators, leaving global RNG unchanged.
Cursor, timing, inventory and residency checks finish before `commit()`.

At publication the live runner adopts candidate fields and the prevalidated RNG.
There is no second full-model clone or rollback copy. The caller transfers
exclusive ownership of the deserialized payload and must not mutate it after
staging. This is a single-threaded resume section with no outstanding borrowed
model/optimizer references, and swaps the drafter object along with its adapter.
It guarantees semantic rejection before mutation; process death/asynchronous
failure during publication still relies on the original atomic checkpoint.

[Measured storage](staging_memory.json), deduplicated by storage identity on the
tiny combined one-update fixture: live tensors 9,188 B, loaded checkpoint
16,484 B, candidate additional auxiliary tensors 428 B, resident union
26,100 B. Candidate core parameters and optimizer moments share loaded storage;
208 B of frozen tensors are shared with the old model. Python metadata and
NumPy RNG storage are excluded from these Torch numbers. Existing loader probe
objects are transient, so conservative extra CPU peak is loaded checkpoint
`C + candidate auxiliary A + largest auxiliary probe + tensor-validation
temporaries + metadata/RNG`. Checks create bounded temporary masks/absolute
values for the largest tensor; this is not a zero-allocation validator.
These formulas and measured aliases were sent to the memory-ledger owner.

CUDA is explicitly rejected by this executable prototype. Future CUDA adoption
must stage core masters, moments and auxiliary state on device while the old
model/moments remain resident, admitting that additional peak; frozen non-core
weights can remain shared. No CPU storage measurement establishes CUDA fit.

## Integration and remaining work

`cpu-integration.patch` is an unapplied CPU proof scaffold: move the owned helper
to `src/w1a1_eagle/curriculum_resume.py`, route `CurriculumRunner.resume` through
it, and bind the helper's source hash in `EXTRA_MATH`. Research allowance-control
code is omitted from that production-shaped patch. The patch is checked for
clean application, not applied to current source. It retains a CPU-only guard
and requires QAT-owner review before any adoption. A source-byte change creates
a fresh runtime identity and must not relabel protected current jobs/checkpoints.

Remaining: QAT-owner integration decision, CUDA transfer/admission implementation
and measurement, and optional durable participation receipts. No numerical gate
or optimizer math was changed. No evaluator code is owned by this task.

Exact owner command from the isolated worktree:

```sh
PYTHONPATH=src:tests:research/parallel20261002/resume_validation/reference /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest discover -s research/parallel20261002/resume_validation/reference -p 'test_*.py' -v
```

Source hashes and precision/hardware are recorded in `source_binding.json`.
