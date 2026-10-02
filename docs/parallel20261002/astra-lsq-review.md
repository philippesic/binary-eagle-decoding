# Focused LSQ batching review

October 2, 2026. Astra medium advisory; CPU/source review only. Control observed
80% weekly used, research_stop=false, original reset1791049896. No tests rerun,
model/data opened, GPU actions, or implementation edits. Inspected production
source and the isolated lsq-batching proposal/independent validation report.

## Finding and interpretation

The reported exact hard outputs with serial:batched learned-head parameter
gradient ratio sqrt(valid depth) are consistent with the declared surrogate.
`learned_activation.py:129–198` divides its scalar-parameter VJP by sqrt(N)
for A1 or sqrt(N*qmax) for A4/A8, where default N belongs to one invocation.
Changing d single-row invocations into one d-row invocation therefore changes
the denominator, while summing the same cotangents. This does not demonstrate
an incorrect backward formula or a violation of the LSQ paper.

It does demonstrate that head batching is not training-update-equivalent under
the current invocation-local recipe. `JointQATConfig` defaults optimize_head to
false, while the optimization work presents batching as a computation control.
The team's 72-case and SGD-update result is evidence supplied by the team,
not independently rerun here. Their separate hand-derived VJP validator checks
the actual local surrogate, tied QKV consumers, masks and chunk normalization.

[LSQ](https://openreview.net/pdf?id=rkgO66VKDS) explicitly uses count-dependent
gradient scaling. It does not settle which count should define this project's
recurrent-head optimization recipe. The project's relative clip and detached
dynamic row scales also differ from an unconstrained LSQ step size. Neither
batching nor serial execution can be called universally correct without the
chosen contract; equivalence to the existing serial baseline is the narrow gate.

The reported SGD step difference proves a possible changed update. Do not infer
the same numerical difference for AdamW: constant positive rescaling can cancel
in ideal Adam normalization. Variable depth across updates, optimizer history,
epsilon and joint gradient clipping prevent a general equivalence guarantee.
Clipping can also propagate the changed activation-gradient norm into updates
of other parameter families even when their raw VJPs originally agree.

## Production reachability

This is not an unsupported synthetic head configuration:

- `recurrent_qat.py:510–576` installs all nine row projections and attaches
  LearnedActivationBank when activation_quantization is learned.
- The bank requires the full nine-projection inventory and includes lm_head.
  A1/A4/A8 learned configurations are accepted; learned A16 is rejected.
- `recurrent_provider.py:285–317` dispatches decode_head when optimize_head and
  adapter support permit it; `native_step.py:486` calls the real lm_head on
  stacked normalized states. NativeStepAdapter enables this in F32 attention.
- `configs/qat_optimization_profiles.json` explicitly combines learned
  activations and head optimization in learned-activations and
  combined-contract-smoke. Fixed-activation profiles are not implicated by
  this specific normalization problem. This review does not establish which
  profile a remote live job currently uses.
- `continuous_qat.py:360–368` ObservedAdapter forwards unknown attributes,
  including linears, so the proposed guard sees its underlying learned head.

## Recommendation

The proposed serial learned-head fallback is the smallest correction that
preserves the existing serial training recipe: when gradients are enabled and
the attached head activation parameter requires gradients, do not use the
batched head callback. Retain batching for fixed activation, frozen learned
parameter and no-grad execution. This sacrifices a speed optimization only in
the combination whose update semantics differ.

The guard should remain bound to supported adapter/quantizer contracts; a future
generic adapter that hides its head cannot be assumed covered. Preserve the
ObservedAdapter public decode_step wrapper so later-state gradient observation
continues to work. Report the actual serial fallback in readiness/timing
metadata: a requested optimize_head flag alone would become misleading. Head
saturation diagnostics also change from valid-chain mean to serial scope;
this is reporting behavior to label, not an acceptance result.

Passing one whole-chain normalization_count to both serial and batched paths
would establish a different shared normalization recipe. It could be a good
future choice, but is not the smallest preservation fix. It changes the
reference learned-activation update and needs explicit source/recipe identity.
Do not multiply by an arbitrary sqrt(depth) after clipping or the optimizer;
the required operation belongs inside the declared parameter VJP contract.

The QAT owner should adopt only a reviewed new source revision and its bound
validation receipt; no protected live source or exact-resume identity should
be silently changed. Native acceptance and Q4_0 throughput remain unmeasured.

## Bounded curriculum cross-check

The curriculum team's malformed-Adam-state proposal is sound as checkpoint
semantic hardening. `qat_curriculum_runner.py:547–563` currently restores state,
binds the phase, copies model tensors, then delegates optimizer loading before
checking base learning rates. Model mutation therefore precedes complete
optimizer validation. Validate finite, shaped, owned moments and immutable
options on staged candidate objects before committing live state.

The demonstrated rehashed malformed fixtures do not imply ordinary atomic
crash corruption: the byte hash already rejects accidental payload changes,
and the team reports valid crash paths preserve state. Missing moments need a
defined per-parameter participation contract; grad=None can legitimately leave
Adam state absent and new phases deliberately start with empty moments. A
blanket requirement for every parameter's step to equal global updates would
reject valid checkpoints. Minimal finite/shape/options checks can be considered
separately from a stronger participation receipt. No new experiment is needed
to restate this source-level distinction.
