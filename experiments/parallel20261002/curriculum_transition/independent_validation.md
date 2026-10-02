# Independent curriculum checkpoint validation

Ran synthetic actual-API probes on CPU. The corrected exact midphase comparison confirms the restored live model payload, optimizer state, RNG, and cursor match the checkpoint exactly. A saved phase-boundary checkpoint resumes in the source phase, and a target `latest.json` publication failure safely replays from that source checkpoint: replayed model and optimizer state matched a fault-free transition exactly. Activation-bank damage and adversarial phase ancestry were rejected. The optimizer loader accepted missing, partial, nonfinite, and wrong-shape Adam state.

## Source anchors

`CurriculumRunner.save()` serializes model payload, optimizer state, RNG, cursor, and counters before atomically publishing `latest.json` ([`src/w1a1_eagle/qat_curriculum_runner.py:380`](../../../src/w1a1_eagle/qat_curriculum_runner.py#L380)). Resume verifies pointer hash, immutable contract, and phase ancestry, validates the model payload, then calls `optimizer.load_state_dict()` ([`src/w1a1_eagle/qat_curriculum_runner.py:524`](../../../src/w1a1_eagle/qat_curriculum_runner.py#L524)). `_load_model()` checks exact model inventory and validates activation state before mutation ([`src/w1a1_eagle/qat_curriculum_runner.py:450`](../../../src/w1a1_eagle/qat_curriculum_runner.py#L450)); `LearnedActivationBank.load_checkpoint()` checks exact boundary ownership and validates every scalar before copying ([`src/w1a1_eagle/learned_activation.py:438`](../../../src/w1a1_eagle/learned_activation.py#L438)). At a phase boundary `_transition()` saves the source phase, binds a fresh destination optimizer, records ancestry, then saves the destination ([`src/w1a1_eagle/qat_curriculum_runner.py:606`](../../../src/w1a1_eagle/qat_curriculum_runner.py#L606)).

Optimizer resume delegates directly to PyTorch without project-level inventory or moment validation ([`src/w1a1_eagle/qat_curriculum_runner.py:561`](../../../src/w1a1_eagle/qat_curriculum_runner.py#L561)). `joint_train_step()` checks for nonfinite parameters after `optimizer.step()`, so the check follows mutation ([`src/w1a1_eagle/recurrent_qat.py:824`](../../../src/w1a1_eagle/recurrent_qat.py#L824)).

## Results

| Probe | Result |
| --- | --- |
| Exact midphase round-trip after one actual update | Passed: the saved model payload exactly matched `resumed._model_payload()`; optimizer moments, RNG, cursor, and epoch also matched. There were 27 optimizer state entries. |
| Phase-boundary checkpoint | Passed: resume restored `state.phase_index=1`, `model_phase=0`, A8 contract, and 27 source optimizer states. Transition moved to A1, recorded one transition, and created an empty optimizer. |
| Target `latest.json` publication failure | Injected failure after the source checkpoint committed and destination model/optimizer were bound. `latest.json` still pointed to source phase 0. Resume/replay reached phase 1 and matched fault-free replay exactly for full model payload and optimizer state. |
| Omitted / partial activation bank | Both rejected (`model checkpoint inventory differs`; `learned bank parameter ownership mismatch`). |
| Adversarial phase ancestry | Rejected (`checkpoint model/phase/transition ancestry differs`). |
| Omitted optimizer state | Resume accepted zero moment entries; next update completed. Maximum absolute difference from the exact-resume control model after that update: `0.0023282766342163086`. |
| Partial optimizer state | Resume accepted one of 27 moment entries; next update completed. Maximum absolute difference from the exact-resume control model after that update: `0.0023282766342163086`. |
| Nonfinite optimizer moment | Resume accepted a NaN `exp_avg`; next update raised `joint QAT update produced nonfinite parameters` after `optimizer.step()`. |
| Wrong-shape optimizer moment | Resume accepted `exp_avg` shape `[1]` for an `[8,24]` parameter; next update raised `output with shape [1] doesn't match the broadcast shape [8, 24]`. |

Each damaged optimizer case used a synthetic payload edit and recomputed the pointer SHA-256, testing semantic validation after the integrity check. The missing/partial cases used the same checkpoint-restored parameter values, RNG, and optimizer learning rates for one direct `joint_train_step()` on the same synthetic batch; this isolates lost moments without advancing the runner's warmup or update counters. They had equal pre-update loss (`1.5542328357696533`) but changed the updated model by the amount above. No real data, GPU/Metal, SSH, native runtime, or paid credits were used.

## Reproduction record

- Primary run directory: `runs/parallel20261002/independent-validation-torch214/` (ignored by Git; per-case checkpoints, metrics, status, pointer, and JSON result files retained).
- Environment: macOS arm64; shared project interpreter Python 3.11.15; PyTorch 2.14.0; `CUDA_VISIBLE_DEVICES` empty; runner configured with `device="cpu"`.
- Control: read `/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json` before each scenario; `research_stop=false`, no monitor interrupt.
- Exact command from repository root: `CUDA_VISIBLE_DEVICES= PYTHONPATH=src:tests /Users/pippo/github/binary-eagle-decoding/.venv/bin/python research/parallel20261002/curriculum_transition/independent_validation.py`
- Result: exit code 0; all scenarios emitted JSON into the ignored run directory and stdout.
- Cleanup: synchronous command finished with no background processes or temporary host/GPU resources. Raw per-case artifacts remain in the ignored run directory. A rerun removes only its own named scenario directories before recreating them.

An earlier, pre-correction exploratory pass used system Python 3.11.3 / PyTorch 2.8.0 and wrote to `runs/parallel20261002/independent-validation/`. Its initial rejection/fail-open observations are superseded by the corrected primary run above; do not combine those environments or run directories. The isolated validator is [`research/parallel20261002/curriculum_transition/independent_validation.py`](../../../research/parallel20261002/curriculum_transition/independent_validation.py).
