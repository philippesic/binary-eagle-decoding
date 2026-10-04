# Reasoning-first latency and training-throughput refinements

Completed read-only research, October 4, 2026. This report was subsequently saved
as documentation at the orchestrator's request. No implementation, tests,
benchmarks, model execution, downloads, GPU/Metal execution, SSH, or host controls
were performed. Local mechanisms were derived and reported to the orchestrator
before the first web search; primary documentation was then used to validate them.

Recommend **two training bookkeeping refinements, a fused-AdamW probe, and a
narrowly scoped DSpark output-assembly probe**. All are additional to the existing
nine-model plan.

## Cost evidence and scope

Current RTX5080 A8 reference costs about **10.45 ms/request token and 10.05
ms/decode token**; candidate costs **14.36/13.91 ms**. The request-versus-decode
remainder is only roughly 0.40–0.45 ms/token, so eliminating that entire remainder
would still leave both below Q4. These differences include prefill and other
request costs; they are not measurements of removable HTTP overhead. Source:
`experiments/a8-qat-recovery/comparison.md`.

Historical SM75 W1A1 already spent less CPU wall time in `draft()` than Q4—3.815
versus 4.435 ms/round—but target decode plus synchronization cost approximately
18 ms/round. Its recorded draft-span-removal proxy remained only 53.76 tok/s versus
Q4's 89.73. This supports prioritizing quality and training exposure over another
isolated microkernel win; it is a fixed-trajectory diagnostic, not a general speed
bound. Source: `experiments/w1ax-activation-precision-results.md:322`.

The DSpark/DFlash SM75 block study has separate clean request measurements and
intrusive CUDA profiles. Profile totals include startup/setup, warmup and request
stages, with CUDA graphs disabled and synchronized node events. They identify
candidate components for investigation; they do not predict request throughput.
Source: `experiments/dspark-sm75-20261003/results.md`.

All deployment probes must preserve target/verifier precision, sampler semantics,
cache ancestry and held-out evaluation, and apply applicable improvements to Q4
controls. Training improvements do not imply faster inference. Nothing here
establishes SM75 gains from RTX5080, CPU, or analytical evidence.

## 1. Consolidate telemetry reads and remove the redundant sign snapshot clone

**Disposition: incorporate after a small correctness gate.**

Local mechanism:

- `src/w1a1_eagle/recurrent_qat.py:760` evaluates
  `latent_sign.detach().clone() < 0` for every binary matrix before each update.
  The comparison itself produces a separate boolean tensor; the intermediate full
  floating-point clone is unnecessary.
- Lines `800`, `842`, `849`, `858`, and `869` separately convert CUDA reductions
  into Python integers/floats. Post-update sign flips, scale movement, clipping
  counts, losses and saturation can be computed first and transferred together in
  typed groups.
- Retain finite-loss and finite-gradient checks **before optimizer mutation**, and
  post-update validity before committing the step. Do not replace exact per-update
  sign counts with sampled counts.

This saves a floating matrix copy per sign tensor plus some host synchronization
boundaries. It does not change recurrence, head batching, quantization, or optimizer
math. PyTorch explicitly identifies scalar reads as synchronization hazards in its
[performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

**Implementation cost:** small Python change; likely under one engineering day
including checks. The code evidence is EAGLE-specific; future block trainers can
reuse the principle when they have equivalent bookkeeping.

**Conditional projection:** if removable copying/synchronization occupies fraction
`f` of clean step wall time, the maximum rate multiplier is `1/(1-f)`. No measured
`f` exists. The reference's 7,200/43,203 ≈ 167 ms/update is charged campaign time,
including downtime, and must not be treated as a clean step measurement.

**Failure/correctness risk:** casting exact integer flip counters into FP32 during
batching; moving failure detection after a bad update; accidentally retaining
tensors/autograd graphs.

**Decisive on/off test:** replay identical short TRAIN rounds; require identical
losses, gradients, parameters, exact flip counts and checkpoint counters for the
clone/read-only change. Then measure clean step wall time, GPU time, peak memory
and supervised tokens/s separately.

## 2. Bound publication cadence for live status and audit JSON

**Disposition: probe.**

This is more concrete than generic logging reduction:

- `src/w1a1_eagle/continuous_qat.py:200` implements `atomic_json` with **file fsync
  and directory fsync**.
- The enabled recipe-audit path invokes it every update at `:1212`.
- `status()` at `:573`, called every lane/update at `:1284`, invokes it again.
- These publications are separate from checkpoint/latest records and the training
  budget ledger.

Publish live status and unchanged audit summaries on a bounded heartbeat cadence,
retaining immediate admission transitions, failures, stop/completion and checkpoint
events. Keep exact metrics in memory and preserve durable checkpoint/budget
semantics. Do not globally weaken `atomic_json`.

**Implementation cost:** small, roughly one day including failure-path checks.
Training only; potentially shared by all six trained candidates.

**Conditional projection:** if these publications cost `S` ms/update and publish
every `k` updates, the ideal saving is approximately `S(1-1/k)`, excluding mandatory
events. File-system latency has not been measured, so this could be negligible.

**Failure/correctness risk:** stale supervision data, lost admission evidence, or
misleading checkpoint-versus-live counters. The existing crash/recovery history
makes these correctness requirements material.

**Decisive on/off test:** first attribute publication CPU wall cost; then compare
identical bounded runs with unchanged checkpoint frequency, a heartbeat safely
below watchdog tolerance, and simulated interruption/resume. Require matching
committed state and charged budget. Incorporate only if useful wall time falls.

## 3. Native fused AdamW without changing the optimizer recipe

**Disposition: probe.**

Both optimizer builders explicitly select the single-tensor path:

- `src/w1a1_eagle/recurrent_qat.py:673` sets `foreach=False`.
- `src/w1a1_eagle/qat_optimization.py:196` does the same.

Try `fused=True` as an explicit, recorded execution option with unchanged
parameters, state precision, betas, epsilon, learning rates, clipping and projection
order. This targets repeated elementwise optimizer launches and memory passes over
large latent sign matrices. [PyTorch 2.14 AdamW documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)
supports fused CUDA execution and describes its vertical/horizontal fusion
advantage; that is mechanism validation, not project-specific timing evidence.

**Implementation cost:** tiny selector change, moderate admission work—approximately
one or two days. Training benefit only.

**Conditional projection:** if optimizer work occupies fraction `f` and becomes `r`
times faster, total step multiplier is `1/(1-f+f/r)`. For illustration only,
`f=0.15,r=2` means 1.081×, not 2×.

**Failure/correctness risk:** floating arithmetic changes near binary sign
boundaries; state-dictionary or update-probe assumptions; support differences on
actual Torch/device versions. Do not silently replace the frozen reference
optimizer path.

**Decisive on/off test:** common checkpoint/rounds, compare gradients, optimizer
moments, projections, sign crossings and save/resume behavior; then measure
equal-exposure learning and tokens/s. Exact floating parity is unnecessary if the
bounded differences do not materially change learning conclusions. Rank this
behind bookkeeping because it can change the optimization trajectory.

## 4. DSpark output assembly with one allocation instead of growing concatenations

**Disposition: probe.**

The serial Markov dependence itself must remain intact:

- `third_party/llama.cpp/src/models/dflash.cpp:348` performs each position's
  lookup/projection/add.
- `:368` repeatedly concatenates the accumulated full-vocabulary columns.
- `:388` uses that position's argmax for the next Markov lookup.
- `:395` restores block-major ordering with permutation/contiguous conversion.
- The local CUDA concat implementation at
  `third_party/llama.cpp/ggml/src/ggml-cuda/concat.cu:18` copies the complete
  destination; [upstream source](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/concat.cu)
  confirms this materialization mechanism.

For seven anchor-first positions, concatenations materialize
`2+3+...+7=27` vocabulary columns. Writing each completed column into a preallocated
final-layout destination requires seven columns, preserving every projection,
addition and argmax. With vocabulary 151,936 and F32 logits, the difference is
approximately **12.2 MB of writes plus corresponding reads per block**, before
considering the final permutation. Actual graph execution must confirm which
copies survive optimization.

**Implementation cost:** modest native graph change, approximately one to three
days; dependency/alias safety makes this more than a mechanical replacement.
Directly applicable to DSpark inference, including its Q4 control; no equivalent
growing Markov concat chain is asserted for EAGLE or DFlash.

**Conditional projection:** saving `c` clean ms/round saves approximately `c/E`
ms/output token at fixed emitted tokens/round `E`. An illustrative 0.1 ms/round at
`E≈4.25` is only 0.024 ms/token—roughly 0.3% against a 7.7 ms/token request. Launch
cost could matter more than bandwidth, but remains unmeasured.

**Failure/correctness risk:** graph scheduler aliasing, missing write dependencies,
incorrect multisequence ordering, anchor-slot mistakes, or altered Markov
conditioning.

**Decisive on/off test:** exact all-slot logits/argmax/proposal/cache checks for one
and multiple blocks; then clean paired complete-request timing on DSpark Q4 and
each admitted binary format. Keep precision unchanged and apply to Q4 fairly.
The intrusive DSpark Markov/output totals justify investigation but do not isolate
concat cost or predict throughput.

## Rejected priority and unresolved measurements

**Defer confidence-head pruning as a performance project:** its intrusive DSpark
total is only 0.832–1.088 ms against tens of milliseconds in body, full head and
Markov/output categories, and confidence remains unused with `p_min=0`. Likewise,
do not spend substantial effort accelerating EAGLE request transport or old A1
draft kernels to close an acceptance-driven gap.

The unresolved priorities for the orchestrator are the actual clean fractions
spent in telemetry, filesystem publication, optimizer updates and DSpark assembly.
One short attribution pass can reject low-value work before implementation. No
component projection above should be added to another or presented as an observed
throughput improvement.
