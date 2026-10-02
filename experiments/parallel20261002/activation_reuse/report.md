# Immediate sibling activation reuse

The bounded CPU prototype passes learned A1/A4/A8 equivalence in the actual
reduced all-nine `NativeStepAdapter` graph with affine weights, raw FC fusion
correction, shared signs/sums, optimized detached context, and serial learned
head. No live source, native runtime, weights, captures, or training recipe changed.

`reference/prototype.py` binds a source-checked decode method and quantizer
wrapper. The only reuse scopes are the immediately adjacent Q/K/V projections
and gate/SILU/up projections. Each scope releases its cache before attention or
down projection. Cache identity includes the actual operand and parameter
identities/mutation versions, quantizer identity/precision/width, grad state,
inference/autocast state, mask identity/version, and explicit normalization N.
It retains strong references only until scope exit. Unversioned inference tensors
and unsupported options bypass reuse. Exception exit also clears all entries.

| Learned bits | Exact projection outputs | Max VJP delta | Max clipped update delta |
|---|---:|---:|---:|
| A1 | 28/28 | 4.657e-10 | 0 |
| A4 | 28/28 | 1.490e-8 | 0 |
| A8 | 28/28 | 1.490e-8 | 9.095e-13 |

Logits are exact in all three profiles. The VJP gate compares all 36 unique
parameters across sign, row scale, learned activation, correction and affine
families, raw captured-feature inputs, and first attached state/K/V inputs.
Explicit tolerance is F32 `atol=2e-6, rtol=5e-5`. One joint AdamW step/profile
clips at 0.07; resulting norms are 0.069999968–0.069999992. The maximum clipped
gradient difference is 3.638e-11. Frozen bias and raw correction arithmetic
remain the source baseline.

Original quantizer-forward call instrumentation proves QKV 3→1 and gate/up
2→1 at each attached proposal. Over the depth-three round, QKV calls fall
11→5 and gate/up 6→3; all other boundaries agree. The two detached K/V-only
context quantizations stay unchanged. This prototype deliberately does not
scope `build_context_cache`, so no detached context chunk survives longer.
It never adds a round/depth cache. Existing affine shared-sum retention is the
same source behavior in baseline and candidate.

Negative controls pass: raw operand mutation, parameter update, a different
aliased view, mask mutation, explicit N change, no-grad→grad boundary,
new group, inference tensors without mutation counters, and exception cleanup.
An equal-valued invalid floating N still raises the baseline validation error.

Independent validation and storage accounting are recorded in the sibling
validation report. Owner results and exact source hashes are in
`summary.json`; full per-tensor tables are outside Git at
`runs/parallel20261002/activation-reuse/results.json` in the isolated worktree.

Reproduction (from the repository root with its existing Torch environment):

```sh
PYTHONPATH=src:. .venv/bin/python -m unittest research.parallel20261002.activation_reuse.reference.test_reuse -v
PYTHONPATH=src:. .venv/bin/python -m research.parallel20261002.activation_reuse.reference.audit
PYTHONPATH=src:. .venv/bin/python -m research.parallel20261002.activation_reuse.reference.build_patch
git apply --check research/parallel20261002/activation_reuse/reference/integration.patch
```

Checks use PyTorch 2.14.0 on local Apple ARM64 CPU, F32 masters/arithmetic and
F16 K/V/factor casts. There is no CPU timing extrapolation or CUDA/SM75
performance claim. Fixed-activation reuse was not needed for this gate.
The unapplied source-bound patch is for QAT-owner review; its next admission
gates are current source regression and actual hardware memory/throughput.
