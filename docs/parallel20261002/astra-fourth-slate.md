# Fourth slate: two concrete QAT computation prototypes

October2, 2026. Root/monitor requests are one slate. Control observed84% used,
16% original allowance remaining, reset1791049896 unchanged, stop=false. Read
absolute control before each costly chunk and stop/checkpoint on stop/reset or
<=1% remaining. No paid/fresh allowance, GPU/Metal/SSH, real model/data access,
live source edits, or new recipe choices.

Recommend two teams only, each Sol high owner plus Luna high validator. Their
work is isolated production-shaped prototypes and measurable removal of known
redundant work. CPU timing must name Apple/ARM hardware and cannot predict
SM75/CUDA benefit. Root launches and QAT owner decides any later integration.

## 1. Share activation quantization within immediate sibling consumers

**Evidence and distinction from completed work.**
`recurrent_qat.RowBinaryLinear.forward` invokes quantizer(input) separately for
each projection, including affine paths. `shared_round_hard_signs` reuses signs
and opens shared_affine_input_sums, but does not reuse the quantized values,
codes, scales or learned-STE saved support. The real NativeStep graph feeds one
fused tensor to Q/K/V and one post-attention tensor to gate/up. These consumers
already share a quantizer parameter/boundary. Existing affine tests prove only
one code sum/Q sum; they explicitly recompute the quantizer for each consumer.

This is not the LSQ head-batching change: sibling consumers see the same input
shape and normalization count N. Reusing the same attached quantizer result
adds their cotangents at one backward node, mathematically equal to summing
the separate same-N VJPs. Independent validation must test this identity and
F32 reassociation rather than assuming it.

Deliverable: a source-bound isolated immediate-sibling reuse adapter/helper,
with an unapplied integration patch. Prefer a narrow scoped cache or explicit
quantized-input handoff. Do not keep every context chunk/depth quantized tensor
alive until round end: that can increase peak memory enough to negate the
benefit. Invalidate at end of the immediate QKV or gate/up group. Distinguish
tensor identity/version, quantizer identity/version, bits, grad-enabled state
and any mask/normalization options. Do not share across optimizer updates,
no-grad/attached boundaries, different tensors or aliased-but-mutated views.

Scope: begin learned A1/A4/A8 in the actual reduced all-nine NativeStep graph,
serial learned head. Only include fixed activation if the same helper preserves
its exact declared native arithmetic without widening the patch substantially.
Fusion correction receives raw input as before; do not fold its computation
into the quantizer. Keep affine midpoint projection and shared sums intact.

Decisive gate: same QKV group quantizer calls3→1 and gate/up2→1; hard outputs,
all unique parameter/input VJPs, and one clipped update agree at declared F32
tolerance. Independent saved-tensor/storage census shows reduced duplicate
support/derivative allocation without longer-lived context tensors. Include one
mutation and one no-grad-to-grad counterexample to ensure stale values cannot
pass. Existing whole-graph tests may be reused; no broad synthetic sweep.
If actual graph does not share eligible operand identities or the memory
lifetime gate fails, reject this design and finish rather than adding a general
cache framework. Optional tiny warmed CPU timing is descriptive only.

Ownership: `research/parallel20261002/activation_reuse/{reference,validation}/`
and `experiments/parallel20261002/activation_reuse/`, unique tests there.
Read-only source: `native_step.py`, `recurrent_qat.py`, `learned_activation.py`,
`affine_binary.py`; reuse prior auxiliary fixtures read-only. No core edits.

## 2. Eliminate redundant diagnostic snapshots without dropping diagnostics

**Concrete source redundancy.**
`recurrent_qat.joint_train_step` snapshots signs using
`m.latent_sign.detach().clone() < 0`. The comparison itself creates an independent
boolean tensor, so cloning the floating master first is unnecessary under the
current same-stream sequential step. The subsequent optimizer mutation cannot
change the already-produced comparison result. This proposition needs an actual
API regression fixture, not a speculation about asynchronous host behavior.

Configured weights total218,234,880/lane. Removing these clones eliminates
872,939,520 bytes of float copies per lane-step, with the largest81.92M head
clone327,680,000 bytes. Each full copy
requires both a read and a write; nominal aggregate memory traffic is twice its
copied payload. The boolean snapshots still exist and must remain exact.
The preceding informal root message's873MB traffic wording should be read as
copied payload, not total read-plus-write traffic. Team must publish exact sums.

The function also converts numerous post-update scalar metrics separately into
Python values. A bounded secondary proposal can collect those already-computed
scalars once, preserving integer count precision and each float's existing
dtype. Do not pack large integer counters into F32 and silently round them.
Do not move finite/error gates past optimizer mutation, omit safety scans,
change diagnostics frequency, or combine this with a new optimizer/fusion rule.
Keep scope to provably redundant copies and reporting extraction. If scalar
consolidation requires broader control changes, deliver snapshot fix alone.

Deliverable: source-bound isolated joint_train_step variant/unapplied patch,
actual Torch operation/allocation census, exact full-shape byte arithmetic, and
small CPU benchmarks or operation counts. Check no equivalent candidate fix
is already owned by QAT before starting. Coordinate via root; do not edit its
memory/eval source changes. This task does not rework admission formulas.

Gate: before-sign snapshots own distinct bool storage and survive master
mutation; sign-flip/scale/loss/saturation/gradient metrics, parameter updates,
optimizer states and error ordering agree with baseline. Count zero full float
clones solely for sign snapshot versus9 before, with boolean storage unchanged.
Use actual step fixtures with learned/affine optional state once and fixed path
once, not a precision grid. A NaN-gradient control must still fail before step.
Host scalar extraction count can be measured by dispatcher/profiler on CPU;
label it a structural bound on potential CUDA synchronizations, not measured
CUDA latency. Finish after these gates; actual hardware timing is the next gate.

Ownership: `research/parallel20261002/step_bookkeeping/{reference,validation}/`
and `experiments/parallel20261002/step_bookkeeping/`. Read-only source:
`recurrent_qat.py`, current optimizer helpers and configured shapes. Separate
files from team1 although both inspect the same source. No core edits.

## Why no third or fourth team

Current memory/eval fixes and resume adoption belong to the QAT owner. Native
CUDA arithmetic repair, real-model backward/peak memory, current source-bound
trajectories and Q4_0 throughput require the GPU/remote ownership gates; another
CPU arithmetic or ancestry fixture cannot substitute. Representation, objective,
architecture and head compression studies already exhausted their authorized
synthetic gates. No legitimate local train-operand payload has been established,
so no real-data experiment is proposed.

If these two structural optimization opportunities are already handled or fail
their scope check, recommend no further CPU team. Remaining allowance is not a
reason to manufacture another study.
