# Third CPU slate: actionable readiness gaps

October2, 2026. Astra medium. Root and monitor requests are one slate. Control
observed82% weekly used/18% remaining, research_stop=false, original reset
1791049896 unchanged. Read control before costly steps; stop/checkpoint at
<=1%, stop flag or original-window reset. No GPU/SSH, model allocation, real
training, live source/recipe mutation or reserved evaluation access.

Recommend three Sol high owners, each with one Luna high independent validator.
Root dispatches. These tasks make concrete isolated prototypes/decision ledgers,
not another numerical coverage sweep. Major recipe choices remain user-owned.

## 1. Development evaluator must reconstruct the checkpoint's recipe

**Highest direct risk newly found.**
`scripts/w1ax_continuous_stages.py:_evaluate_development` exports current A8/A1
checkpoints and captures native acceptance, then around1619 constructs default
fixed `JointQATConfig`, loads CPU models, installs ordinary row linears and
calls `_load_checkpoint`. The latter, in
`scripts/check_continuous_w1ax_readiness.py:382–463`, supports schemas2–5 but
requires the corresponding activation bank, correction and affine modules
already attached. It does not install them. LearnedActivationBank.from_attached
and explicit correction/affine checks can therefore reject modern recipe
checkpoints at the scheduled Torch validation-loss stage, after native capture
has already spent time. Source evidence is strong; a focused real-API synthetic
probe must establish exact affected call paths before declaring a defect.

Deliverable: CPU-only producer→evaluation-construction→loader proof for fixed,
learned, fusion, affine and combined named profiles. Use existing tiny export
fixtures and actual loader. Avoid executing the native/GPU evaluator; isolate
its pure recipe-construction portion faithfully. Show whether the saved
deployment manifest carries enough information to reconstruct all required
modules or whether source-bound recipe identity must accompany it. Effective
F16 correction tensors are deployment values, not optimizer masters.

Prototype a narrow recipe-aware preflight/construction adapter in the owned
research directory, plus a source-bound patch proposal. Reject unsupported
recipes before launching native capture; load supported recipe state exactly.
Do not suppress mismatches or silently use fixed activation. A8/A1 paired
continuous evaluation is its current scope; A4 curriculum support is a separate
declared gap, not an invitation to broaden the evaluator.

Gate: a valid modern checkpoint reaches identical deployment state and bounded
synthetic logits to its source; fixed schema2 behavior stays equivalent; missing
recipe identity fails before any expensive action. Independent validator checks
the actual call chain rather than a mocked permissive loader. Report actual
affected configuration names and checkpoint fields. No quality claim.

Ownership: `research/parallel20261002/eval_recipe/{reference,validation}/`,
`experiments/parallel20261002/eval_recipe/`; new unique tests there. No core
files changed. Source dependencies above plus `recurrent_qat.py`,
`configs/qat_optimization_profiles.json`, and recipe integration/export tests.

## 2. Stage and validate curriculum optimizer resume before mutation

Confirmed first-slate gap: malformed but rehashed Adam state can be admitted;
`CurriculumRunner.resume` restores curriculum/model fields before complete
optimizer validation. Ordinary atomic recovery already passed, so do not
repeat crash simulations or claim normal checkpoint corruption.

Deliverable: source-bound isolated validator and transactional restore prototype
for the current joint optimizer layout. Implement the previously reviewed
proposal as executable research code and a minimal integration patch. Validate
owned state IDs, stable family order, tensor shape/dtype/finite values,
nonnegative second moments, scalar integral nonnegative steps, and immutable
optimizer options. Compare effective LR against the declared warmup/counter
contract. Handle both AdamW and admitted SGD configurations only where current
joint optimizer actually permits them; inspect source rather than assuming.

Empty state at fresh phase is valid. Parameters with grad=None may legitimately
lack moments. Do not equate per-parameter step with global update count without
a proved participation contract. Keep stronger missing-state detection as an
explicit optional receipt proposal if existing save data cannot determine it.
Validation on detached state must not allocate a second full real model. Define
a bounded staging strategy and its memory cost with team3's ledger if needed.

Gate: malformed owned fixtures fail before changing any live model/optimizer/
curriculum/RNG/cursor field; valid midphase and boundary resumes yield the same
next update; grad=None/fresh-state cases remain legal. Independent validator
targets the prototype against unchanged valid fixtures and adversarial states.
Source-byte change requires a fresh runtime identity; no patch to current jobs.

Ownership: `research/parallel20261002/resume_validation/{reference,validation}/`
and `experiments/parallel20261002/resume_validation/`. Dependencies:
`qat_curriculum_runner.py`, `qat_optimization.py`, `recurrent_qat.py`, first-slate
curriculum report/proposal/tests read-only. Exclude development evaluator code.

## 3. Full-shape allocation-lifetime ledger for QAT readiness

Current `continuous_qat.memory_estimate` computes binary master/moment storage
and adds a user-assumed graph budget. `continuous_resources.py` separately
estimates export/offload buffers; curriculum save recursively copies payloads
to CPU. Actual configured shapes include the81.92M-weight head. A few gigabytes
of omitted simultaneous copies can matter on the12GiB reserved-memory cap.
The smoke already preallocates Adam buffers: do not rediscover lazy allocation
as a bug without tracing this existing protection.

Deliverable: source-bound allocation/lifetime ledger for initialization,
zero-update smoke, one paired step, continuous save/export, resume,
development offload, and curriculum save/transition. Use exact configured shape
arithmetic and unique-storage alias accounting, including optional learned
scalars, rank1/rank4 correction, all-row midpoint, frozen shared weights,
packed sign diagnostics, optimizer buffers, gradients and largest known
temporary. Distinguish host and device, current-resident from additional peak,
and logical required storage from allocator/workspace uncertainty.

Validate formulas with tiny instrumented CPU objects only: storage identities,
views, shared banks, saved payload CPU copies and optimizer states. No full model
allocation, allocator stress, artificial giant tensors or GPU discovery. Read
`configs/continuous_w1ax.json`, `continuous_qat.py`, `continuous_resources.py`,
`qat_curriculum_runner.py`, save_joint_checkpoint and export code. Do not reread
raw train captures or assume any particular remote available memory.

Gate: each concrete simultaneous allocation is charged exactly once in its
stage; tiny known-storage measurements agree with formulas; unknown graph,
CUDA workspace and allocator residency are explicitly unresolved. Compare
ledger to existing admission estimates and report a concrete undercount or
falsify that concern. Recommend at most one small corrected estimate or one
required phase-specific measurement, never lower a safety floor. If logical
minimum alone exceeds a configured cap, that is a decisive preflight result;
otherwise only real GPU measurement can establish fit.

Ownership: `research/parallel20261002/memory_ledger/{reference,validation}/`
and `experiments/parallel20261002/memory_ledger/`. No resource helper edits.
May read team2's proposed staging design, but can complete independently using
current source; final staging estimate is a separately labeled proposal delta.

## Directions explicitly not repeated

The216 fixed recurrence comparisons,48 all-auxiliary comparisons and81 exported
projection comparisons have completed. Their next evidence is native/current
model GPU validation; further tiny numerical cases do not close that gap.
LSQ correction belongs to the sole QAT owner. No parallel core edit.

The completed W1Ax report already contains draft-span-removal ceilings and
required accepted/round thresholds versus Q4_0, so another break-even team is
not warranted. Prior checkpoint selection guidance already distinguishes online
development acceptance from final evaluation. Repeated-selection statistics
would require a separately chosen protocol; it is lower priority than the
concrete evaluation recipe-construction gap found above. Literature on
[time-uniform confidence sequences](https://arxiv.org/abs/1810.08240) does not
by itself justify retrofitting adaptive checkpoint selection or claiming
validity under reused development prompts.

No unsealed operand payload is needed for this slate. If any team reaches a
question whose only missing answer is native memory/timing/acceptance, report
that boundary and finish rather than inventing further synthetic work.
