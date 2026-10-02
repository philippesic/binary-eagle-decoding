# Independent step bookkeeping validation

The source-bound candidate removes the float clone of each 3×3 latent weight
while retaining the boolean result of `latent_sign.detach() < 0`. On both CPU
Torch 2.8.0 and 2.14.0, fixed, learned-activation, and affine-weight runs had
exactly equal metrics, all gradients, parameters, and AdamW state versus the
original step. A dispatcher census found 9→0 latent-weight float clones per
step, 9→9 scale float clones, and 27→27 boolean comparison outputs. The latter
includes comparisons elsewhere in the complete training step; the targeted
snapshot comparison was unchanged. A separate storage check confirmed the
boolean snapshot remains independent after source mutation.

Both implementations also rejected an injected NaN gradient before changing
parameters or optimizer state. Each fixture used all nine real `RowBinaryLinear`
modules with a two-row `TraceAudit`; learned mode attached the canonical six
activation boundaries, and affine mode installed midpoint parameters on all
nine modules.

## Reproduction

Worktree: `/private/tmp/eagle-parallel-20261002/step-bookkeeping`  
Branch: `research/20261002-step-bookkeeping`  
Hardware: Apple ARM CPU; macOS; CUDA unavailable; MPS available but unused.  
Python/Torch: system Python 3.11.3 / Torch 2.8.0 and project `.venv` Python
3.11.15 / Torch 2.14.0.

Commands from the worktree root:

```sh
PYTHONPATH=src:. python3 research/parallel20261002/step_bookkeeping/validation/validate_snapshot.py > runs/parallel20261002/step-bookkeeping-validation/raw-torch28.json
PYTHONPATH=src:. /Users/pippo/github/binary-eagle-decoding/.venv/bin/python research/parallel20261002/step_bookkeeping/validation/validate_snapshot.py > runs/parallel20261002/step-bookkeeping-validation/raw-torch214.json
```

Both commands exited 0. Raw JSON was preserved in the ignored run directory;
temporary `/tmp` outputs from an earlier fixture-shape correction were deleted.
No model/data access, GPU, Metal, SSH, or persistent process was used.

SHA-256:

- `research/parallel20261002/step_bookkeeping/validation/validate_snapshot.py`:
  `40853fe5d6f8b8ec7fa5ce01978328816fef4b7b93169d948d4be7ccbf89c2ac`
- `runs/parallel20261002/step-bookkeeping-validation/raw-torch28.json`:
  `f696f8e52becdb0374a2cbd64870132f49574707c3d3457ffd0b35e9f6262b4e`
- `runs/parallel20261002/step-bookkeeping-validation/raw-torch214.json`:
  `24f46053c4e9a393678158e577755469060f4f30ef4c8933147fab9b7d8514c9`
