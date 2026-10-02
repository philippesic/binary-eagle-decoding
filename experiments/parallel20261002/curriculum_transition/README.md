# Learned precision transition and crash-resume audit

The current runner deliberately preserves binary weights and recreates both
activation parameters and optimizer state on each precision switch. A8→A4
does **not** carry the learned clip ratio forward; A4 starts at 1. A4→A1 creates
threshold deltas at 0. The CPU fixtures below reproduce that contract. Neither
activation transfer policy nor the live curriculum was changed.

## Scope and source identity

Base parent: `6c613039d2c6bef3f6f7b5c6e9bf5f8cbccac2e8`.
Hardware: local Apple arm64 CPU; PyTorch `2.14.0`. Trainable values and
quantizer arithmetic are F32; the existing tiny synthetic provider has frozen
FP16 operands. No CUDA, Metal, SSH, real model, real data, native server,
reserved final evaluation, or paid credits were used. Synthetic optimizer
steps are CPU oracles only; these results establish no acceptance, latency,
throughput, or SM75 result.

| Source | SHA256 |
| --- | --- |
| `src/w1a1_eagle/qat_curriculum_runner.py` | `2d2e1bf6edc28a16f75a009d77822f2134127a1711ceab3f19f8152b0adece84` |
| `src/w1a1_eagle/qat_curriculum.py` | `ec133082768e7c167bb67a7a1e4013883e51bc9a721ba78884a7c39904742dce` |
| `src/w1a1_eagle/learned_activation.py` | `65b7b0863bb67806991ee4860ce327bf46bfbdf0e8d4e44226aa1ca268e880db` |
| `src/w1a1_eagle/qat_optimization.py` | `d2da18a4f4db612dd3cb51916af688109593d2c9d4cf119a21f8129567c305f7` |
| `src/w1a1_eagle/recurrent_qat.py` | `f2b6d83b8929807b65b6f21826cc99097a6ddd70f99ff23ca0836aebe4db1224` |

## Exact current contract

`CurriculumConfig` (`qat_curriculum.py:104`) accepts direct A1, A8→A1,
or A8→A4→A1 and requires `optimizer_transition="fresh"`.
`transfer_precision_state_` (`:405`) explicitly owns binary weights only and
requires a separate explicit contract for learned activation transfer.
`CurriculumRunner._bind_phase` (`qat_curriculum_runner.py:594`) changes
the modules' activation contracts, creates a new bank, revalidates immutable
training ancestry, constructs a current adapter, and makes a new optimizer.
`_activation_bank` (`:241`) removes old quantizer attachments and attaches six
new shared boundary quantizers. `LearnedActivationQuantizer.__init__`
(`learned_activation.py:242`) initializes clips to 1 and thresholds to 0.

| State | A8→A4 / A4→A1 behavior |
| --- | --- |
| Drafter model | Same object |
| Nine F32 latent/master sign matrices | Same parameter objects and exact values |
| Nine additive scale offsets, frozen initial scales/biases | Same objects/values; exact additive representation and effective scales |
| Three optional fusion correction tensors | Same objects and exact values |
| Nine optional affine midpoints | Same objects and exact values |
| Six activation boundary scalars | New parameters; Q/K/V share one, gate/up share one |
| A4 clip ratios | All 1, even if A8 clips were learned away from 1 |
| A1 threshold deltas | All 0; a clip-to-threshold mapping is not specified |
| Optimizer moments/counters | Entire optimizer is new, including state for retained parameters |
| Exposure and warmup | Global counters persist; existing runner tests verify global warmup |
| Recurrent cache/graph | Each `_forward` (`:631`) rebuilds current-weight cache; none is checkpointed |

## Measured resets and independent gradient check

[`audit.py`](../../../research/parallel20261002/curriculum_transition/audit.py)
uses the existing eight-wide `TrainProvider` fixture and the actual runner
APIs, including real tiny synthetic joint optimizer updates. After one A8
step, QKV and gate/up clips were `0.9999983311`, and down was
`0.9999983907`; transition made all six exactly 1. After one A4 step, fc,
QKV and head were `0.9999966621`; transition created all six thresholds
exactly 0. Each source optimizer held 36 tensor states at Adam step 1;
the destination optimizer held none. In each switch, 9 sign, 9 scale,
3 correction and 9 midpoint parameter objects/values remained exact.
Both post-transition checkpoints restored exact model, optimizer and RNG.

A separate valid sentinel probe sets the six synthetic clips to
`[.55,.60,.65,.70,.75,.80]` after the actual tiny step. A8→A4 again resets
all to 1. Holding A4 precision and input constant, changing only head clip
`.8→1` changes 4/8 integer codes and produces maximum absolute output
difference `0.1999999881`. This is a sensitivity counterfactual, not a
proposed or measured training recipe. The ordinary one-step head clip
remained 1, so that ordinary isolated head comparison had zero change.
Full small numeric summaries are in [`measurements.json`](measurements.json).

The quantizer backward is a declared STE/LSQ surrogate, not the derivative
of its discontinuous hard forward (`learned_activation.py:127`). An
independent F64 frozen-code local proxy includes the interior LSQ `-x log c`
term, while A1 uses the explicit threshold surrogate. Central finite
differences at epsilon `1e-4` agree with autograd to `2.8e-10` (A8),
`7.3e-9` (A4) and `3.0e-8` (A1). The hard A1 finite difference is zero
away from a sign boundary while the intended surrogate VJP is
`-0.2891428471`. Equating those two derivatives would be an invalid gate.

## Crash-resume and proposal

`_transition` (`qat_curriculum_runner.py:606`) saves a source checkpoint
before binding a new phase, then records ancestry and saves the target.
The pending-boundary source checkpoint has `state.phase_index=model_phase+1`
but old precision, optimizer and activation bank. `resume` (`:524`)
permits that relationship, restores the old stage and replays the
transition on the next run. A committed target restores the new stage
directly. A1 completion similarly permits the final model phase to lag
the completed counter by one. Normal source/target checkpoints retain
exact model and optimizer state; wall occupancy deliberately includes
reconstruction and is not bitwise timing continuity.

Independent fault-injection and adversarial checkpoint results are recorded
in [`independent_validation.md`](independent_validation.md). Its fixtures
use disjoint code and independently verify the source/target crash windows.
Injected failure at target `latest.json` publication leaves the committed
source pointer intact; source resume/replay exactly matches fault-free model
and optimizer state. Missing/partial Adam state passes resume after a
synthetic rehash and produces `0.0023282766` maximum updated-model difference
from an exact-resume control at the same input/LR. NaN and wrong-shape
moments also pass loading and fail during the next update. These are
semantic validation defects in deliberately altered synthetic payloads,
not losses observed in the normal atomic crash window.

There is no warrant to change reset/transfer semantics from this synthetic
audit. The narrow documentation proposal is to make the current activation
reset policy explicit in the curriculum manifest/transition receipt in a
future source-versioned experiment. A8→A4 clip reuse remains a user-owned
recipe choice; A4→A1 needs a defined semantic mapping and cannot be inferred
from the parameter names.

One separate hardening lead is that runner `resume` directly invokes
`optimizer.load_state_dict`, whereas `qat_optimization.py:319` has explicit
finite/inventory/group/moment validation for its binary-only API. Any
patch should validate the runner's named joint families and declared
optional parameters, preserve existing valid checkpoint arithmetic, and
fail before mutating live state. Exact detailed reproduction and scope
follow in the independent report; no core patch is included here.
The concrete scope and acceptance criteria are in
[`PATCH_PROPOSAL.md`](PATCH_PROPOSAL.md).

## Reproduction and acceptance

```sh
PYTHONPATH=src:tests /Users/pippo/github/binary-eagle-decoding/.venv/bin/python research/parallel20261002/curriculum_transition/audit.py
PYTHONPATH=src:tests /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest test_parallel20261002_curriculum_transition test_qat_curriculum_runner
```

Acceptance: both phase switches preserve the retained parameter objects and
exact values; new optimizer state is empty; six new activation scalars
match defaults and canonical aliases; post-transition round-trip state
is exact; finite differences independently verify the intended VJP.
The focused suite passes **22/22** tests. `pytest` is absent in the shared
venv, so validation used the repository's unittest workflow.
The independent validator additionally passes its exact-state assertions
and reproduces all **10** checkpoint/failure scenarios under the same
PyTorch 2.14.0 interpreter; see its separate command and report.
