# CPU fusion binary sign-and-scale fitting

Supporting experiment, October 3, 2026. A8 QAT remains the sole active project
goal. This team does not own QAT and has no accelerator or remote-host access.

## Frozen scope and checkpoint

Base revision: `be8d160`. Only the fusion projection is eligible for changes.
Use the original BF16 checkpoint's signs, including positive signs at either
signed zero, and nonnegative F32 row scales. A8 uses raw F32 fusion inputs,
per-token absmax/127, nearest-even signed integer codes, an integer signed dot,
then separately rounded F32 row-scale and token-scale products. Other model
operands, target/verifier precision, and model architecture are frozen.

Compare original mean-absolute row scales, converged fixed-sign row scales,
and at most two alternating sign/scale passes with four correlation scans.
Accept a sign move only after checking the actual finite exported objective;
stale individual improvements do not establish a combined improvement.
Split eligible TRAIN captures by prompt before fitting and freeze the candidate
before validation. Output reconstruction and direction diagnostics are local
surrogates; native acceptance and throughput remain deferred.

Implementation owner: bounded Sol feature agent in
`/private/tmp/eagle-fusion-discrete-fit`. Independent validator: bounded Luna
agent in `/private/tmp/eagle-fusion-discrete-validator`. Report/integration:
`/private/tmp/eagle-fusion-discrete-report`. Each has a separate branch and
disjoint new files. Existing partial/unmerged worktrees are preserved.

The local-input eligibility audit and code are in progress. If provenance-checked
TRAIN fusion operands are absent, the authorized fallback is synthetic algorithm
and export validation, with an exact smallest missing-package specification.
No diagnostic samples will be represented as real fitting.
