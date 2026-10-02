# Narrow proposal: validate joint optimizer state before curriculum resume

The primary patch candidate is checkpoint hardening, not activation transfer.
`CurriculumRunner.resume` currently passes saved Adam moments to PyTorch
without semantic checks. Rehashed synthetic checkpoints with omitted/partial
moments are accepted and lose the exact optimizer trajectory; nonfinite and
wrong-shaped moments are accepted and fail later at the next update. Ordinary
byte corruption already fails the pointer SHA256 check. This finding concerns
semantic validation after the integrity gate and does not imply that normal
atomic crash windows lose state.

Scope: one runner-specific validation helper and corresponding adversarial
resume tests. No change to optimizer math, learning rates, precision stages,
training schedule, activation initialization, data eligibility or live jobs.

1. Build a stable named joint optimizer layout from actual `param_groups`
   and `joint_parameter_families`: group family/order, parameter path or
   canonical boundary name, F32 shape/dtype, and parameter count. Preserve
   six canonical activation owners, three correction tensors and optional
   midpoint owners exactly once.
2. Before mutating model, optimizer, cursor or RNG, check saved group family
   ordering, parameter IDs/order/counts, weight decay, betas/momentum and
   other immutable options against the freshly bound optimizer. Permit
   the already-defined global warmup LR range, using saved global counters;
   reject NaN/Inf or changed options. Validate base rates before loading.
3. Validate every saved moment is finite, dense, correctly shaped and has
   the expected dtype; validate scalar integral nonnegative step and
   nonnegative second moments. Check state keys belong to the declared
   layout and moment field inventory matches AdamW/SGD. Empty state is
   legitimate before any update in a new phase.
4. Define the expected moment-presence/step contract explicitly. Since
   `joint_train_step` currently allows `grad=None`, do not assume every
   declared parameter must receive state on every update. Either prove and
   enforce graph participation for the admitted joint graph, or record
   stable named per-parameter participation counters and state inventory
   when saving. The minimal finite/shape guards can land independently;
   rejecting silent omissions needs a defined participation receipt.
5. Stage validation on candidate state/objects before applying it to live
   objects. A failed resume must not leave a partially restored model or
   RNG. Reuse the strict checks in `qat_optimization.load_optimizer_checkpoint`
   where applicable, but its binary-only family ordering and single extra
   group do not directly describe the joint activation/fusion/midpoint layout.

Acceptance: all unchanged valid direct A1/A8→A1/A8→A4→A1 midphase,
boundary and completed checkpoints retain exact model/optimizer/RNG after
resume; missing/partial/NaN/Inf/wrong-shape/wrong-option/unowned moment cases
fail before any live mutation; fresh phase-start optimizer state remains
valid; fault before target pointer publication recovers/replays source exactly.

Deployment needs a separately reviewed source revision and its new bound
runtime identity. The runner hashes its own source and contract, so a patch
must not silently change the protected experiment's exact-resume identity.
The root/QAT owner owns whether/when to adopt it. This research branch
contains the report and reproduction fixtures only.

A secondary documentation-only proposal is to explicitly name the current
activation reset policy in a future curriculum manifest/transition record.
Reusing A8 clips in A4, transferring Adam moments, or inventing a clip-to-A1
threshold mapping are recipe decisions outside this hardening patch.
