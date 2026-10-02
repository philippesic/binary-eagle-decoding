# CPU research slate, October 2

Advisor: `/root/astra_research`, Astra medium. Read-only code observation at
`6c613039d2c6bef3f6f7b5c6e9bf5f8cbccac2e8`; working files can advance, so teams
must pin hashes of every imported source. Supporting the existing QAT
optimization readiness goal; no new goal, GPU action, live source/recipe change,
or held-out/final access. Q4_0 EAGLE remains the acceptance and throughput target.

Control observed: `runs/parallel20261002/control.json`, research_stop=false,
78% weekly used, original reset 1791049896. Read this absolute-path control
before expensive steps and dispatch; stop/checkpoint on stop, <=1% remaining,
or original-window reset. Never spend the refreshed window on this work.

## Recommendation

Launch four independent, bounded audit teams. Each gets one Sol high owner and
one Luna high independent validator, using disjoint `reference/` and
`validation/` subdirectories. Small synthetic CPU fixtures suffice; no training
sweep, model loading or broad architecture study is needed. Each team should
return a reproducer or a falsified concern plus an exact integration proposal.
Changing a live recipe remains the QAT owner's decision. These tasks test the
implemented training machinery, whereas the completed overnight studies tested
representation, objectives, architecture, head compression, and data reuse.

### 1. Activation-gradient invariance across head batching

**Priority: highest; concrete source-level risk.**

`learned_activation.py:129–198` scales the learned parameter gradient by
sqrt(N*qmax) for A4/A8 and sqrt(N) for A1, with N invocation-local by default.
The low-level API permits a larger shared normalization_count across chunks.
A source search finds no other production use of that keyword.
`recurrent_rollout.py:173–185` optionally stacks valid recurrent states for one
head call. Consequently, serial versus batched head invocations can give equal
hard logits while changing the clip/threshold gradient. This is a hypothesis
about the composed production path, not yet a demonstrated defect.

Deliverable: executable synthetic production-API reproducer, source hashes,
table of hard logits, loss, input VJP, sign/scale VJP, activation-parameter VJP,
and one optimizer update for serial, batched, and chunked heads. Cover A1/A4/A8,
depths 1/2/4, invalid terminal padding, ragged valid lengths, and tied shared
QKV consumers. Compare the same scalar loss/reduction and identical tensors.
Include a hand-derived independent VJP oracle: never finite-difference a hard
quantizer and call disagreement a wrong STE.

Acceptance/falsification: either all intended-equivalent paths match every
parameter gradient within explicit CPU F32 tolerance, or isolate the exact
normalization ratio and show a changed optimizer step. For repeated identical
rows with fixed mean loss, a per-call versus whole-batch normalizer predicts a
sqrt(depth) gradient ratio, provided the parameter derivative is nonzero.
Test that algebra rather than requiring a ratio in cancellation cases.
An isolated reference fix may forward one declared normalization domain;
do not assume that domain is global step, round or invocation without showing
the existing intended contract. Return the choice to the QAT owner.

Ownership: `research/parallel20261002/activation_batching/{reference,validation}/`
and `experiments/parallel20261002/activation_batching/`. Dependencies: existing
CPU Torch environment, source only. No dependency on the other teams.

### 2. Binary optimizer movement, scale sensitivity, and budget adequacy

`qat_optimization.py` initializes latent magnitudes at 0.5 by default, supports
AdamW/SGD, sign learning rate 1e-3, positive-scale gradient division, and joint
or family clipping. `continuous_qat.py:1007–1016` only raises the zero-flip
warning after 2,000 pairs. Existing tests check individual transforms and
checkpoint restoration; they do not establish that a finite approved update
budget can cross useful decision boundaries under competing parameter groups.

Deliverable: small exact binary regression/recurrent classification fixtures
with known reachable sign solutions and an infeasible control. Use the existing
optimizer API, fixed synthetic sequence, and a predeclared tiny update budget.
Compare only existing options: baseline, weight_unit, and per_family clipping;
vary initial latent magnitude and row scale as controlled nuisance axes, not a
large hyperparameter search. Include frozen-scale and jointly learned-scale
cases. Report first useful flip, chatter, net sustained flips, latent distance
to zero, task margin/hard prefix length, CE, and clipped family gradients.
Use current flip diagnostics instead of reimplementing its packed counters.

Acceptance/falsification: independently enumerate the small hard-sign space;
distinguish optimizer stall from representational impossibility. Derive a
constant-gradient Adam/SGD crossing-time control and verify its assumptions.
Reject any recommendation based solely on more flips or lower soft loss.
Return a minimal training telemetry recommendation and at most one candidate
existing recipe for later owner-authorized validation. Do not introduce Bop as
a new live optimizer or select hyperparameters on project development data.

Ownership: `research/parallel20261002/sign_movement/{reference,validation}/`
and `experiments/parallel20261002/sign_movement/`. Dependencies: CPU Torch and
source only. Existing overnight objective exact-state fixtures may be reused
read-only; do not repeat their prefix-surrogate comparison.

### 3. Current-student recurrent gradient and cache equivalence

The active goal explicitly seeks faster QAT computation and cache construction.
`recurrent_rollout.py` promises attached current-student states/cache across
the draft unroll; `qat_curriculum_runner.py:_forward` rebuilds cache each round.
Forward equality alone cannot establish that this training graph is preserved.

Deliverable: tiny reduced EAGLE-shaped teacher-forced recurrence routed through
the existing adapter/rollout APIs, comparing current optimized cache/encoding
modes with an independently explicit serial reference. Establish the exact
mode inventory first from tests and implementation; do not invent supported
checkpointing switches. Use fixed activations initially so this is independent
of team 1. Test prefix lengths 0/1/multiple, draft depths 1/3, invalid terminal
rows, shared signs, optimizer boundary rebuild, and a nonzero dependence on
earlier cache values. Deliberately detach a recurrent state/cache in the oracle
negative control and require the checker to detect it.

Acceptance/falsification: hard/logit values and VJPs for every trainable family
match a declared F32 tolerance; one step and resumed second step agree. Label
the intended accepted-context truncation boundary explicitly so a correct
stop-gradient is not mistaken for a defect. Test that an updated weight cannot
reuse a stale persistent graph/cache. A concrete discrepancy gets a minimized
fixture and proposed local patch, never a production edit.

Ownership: `research/parallel20261002/recurrent_vjp/{reference,validation}/`
and `experiments/parallel20261002/recurrent_vjp/`. Dependencies: CPU Torch,
existing reduced model test scaffolding; no real capture. Numeric agreement
here makes no native GPU performance claim.

### 4. Learned-activation curriculum transition contract

`qat_curriculum.py:405–419` deliberately limits transfer_precision_state_ to
binary weights and says learned activation needs a separate contract.
`qat_curriculum_runner.py:594–624` instead changes the live contract, creates a
new activation bank and optimizer, preserves binary parameter objects, and
saves before/after transition. The default new bank initializes A4/A8 clip=1
or A1 threshold=0. This is a declared reset mechanism, not automatically a bug.

Deliverable: a transition ledger for A8→A4→A1 using actual runner primitives,
nondefault learned clip/threshold values, binary latents/scales, auxiliary
fusion/midpoint parameters, optimizer moments, budget counters, source hashes,
and refresh ancestry. Compare uninterrupted transition with resume before and
after the durable boundary. Include stale activation-bank references, missing
new optimizer parameters and stale refresh receipt as negative controls.
Quantify the immediate synthetic hard-logit/margin change caused by reset
separately from the precision change, using a reference-only matched transfer
where mathematically meaningful. A1 threshold and multibit clip are different
parameters; no arbitrary numerical copying is justified.

Acceptance/falsification: each state field is explicitly preserve/reset/rebuild;
uninterrupted/resumed paths reach the same declared state and next update;
invalid ancestry fails closed. Report reset shock as a diagnostic, not proof
that warm transfer is superior. Any alternative transfer or calibration rule
is a user-owned research choice requiring a fresh recipe identity.

Ownership: `research/parallel20261002/curriculum_transition/{reference,validation}/`
and `experiments/parallel20261002/curriculum_transition/`. Dependencies: CPU
Torch and runner fixtures; independent of teams 1–3.

## Primary-source interpretation

[Helwegen et al., 2019](https://arxiv.org/abs/1906.02107) interpret latent
weights as inertia and propose Bop. This motivates measuring crossing time
and flip persistence, not assuming that Adam latent movement means useful
binary learning. Their image-classification evidence does not establish EAGLE
acceptance or justify a new optimizer here.

[LSQ, Esser et al.](https://openreview.net/pdf?id=rkgO66VKDS) explicitly scales
step-size gradients using feature/weight counts and quantization levels.
The code uses relative clipping with detached dynamic token extrema, so it is
not identical to an unconstrained LSQ step size. Preserve the code's declared
surrogate in the audit and test execution-shape invariance separately from
whether that surrogate is a good learning rule.

[PACT, Choi et al.](https://arxiv.org/abs/1805.06085) establishes learned clipping
as a training parameter. Its result does not imply that an A8 clip should be
copied into an A1 threshold; those have different forward meanings here.

## Boundaries and deferred options

Do not repeat overnight group-A1, rotation, cold low-rank/centroid head screens,
prefix-objective algebra, DSpark/DFlash scaffolding or startup-reuse proof.
Those reports already completed their synthetic gates; more synthetic sweeps
would not close their missing real-state/native evidence.

A future fifth task could build a minimal observability packet distinguishing
current-student trajectory drift, unsupported-vocabulary labels and sign stalls.
Defer it until teams 1–4 report, since existing trajectory_refresh audits already
cover substantial ancestry/eligibility machinery. No fresh task is warranted
merely to consume allowance. Actual quality, native A1 decisions, Q4_0-relative
throughput and GPU memory/timing remain unresolved owner-run gates.

## Source-path refinement after packet delivery

`recurrent_provider.py:270–317` supplies the concrete controls optimize_cache,
optimize_head and context_chunk_size, and dispatches NativeStep adapter methods.
`native_step.py:486` implements decode_head by calling lm_head on stacked
normalized states. Its preceding build_context_cache implementation builds
FC/K/V in chunks with F16 cache casts. Teams 1 and 3 should include these files
in the source manifest. Existing `tests/test_qat_computation.py` already checks
several operand/subgraph VJPs; team 3's distinct value is the composed adapter/
provider topology, accepted-context truncation boundary, and state lifetime.

No executable experiment, model/data read, optimizer update or implementation
change was performed by this advisor. The four task packets are ready for root
dispatch. Further literature expansion is deferred until a team returns an
unresolved question that can change a practical recommendation.
