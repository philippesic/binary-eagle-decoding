# Real TRAIN fusion binary calibration

Supporting assignment authorized October 3, 2026 after the synthetic fitter
validation. A8 QAT remains the sole active project goal. This team will not
change QAT models, recipes, live processes, or project goal records.

Scope: authenticate and extract 8 fitting + 4 separate validation TRAIN prompts,
32 raw pre-A8 fusion rows each, from existing saved captures. Read-only remote
streaming is authorized through tmux MCP using the shared host registry; no
GPU/Metal/CUDA computation, new captures, or remote writes. Raw data, candidates
and logs belong in `results/fusion-binary-real-a8-20261003/`, outside Git.

The fit retains the previous fixed A8 row-scale contract, original BF16 source
signs at zero, two alternating passes / four scans / 32 accepted flips per row,
1 GiB estimated workspace, 1,024 total examples, and 12 CPU-hours. Fit only
fitting prompts; persist/hash candidate before validating separate prompts.
Compare original initializer and converged scale-only control, using the finite
exported objective for sign acceptance. Export a full fusion-only candidate
through existing native GGUF v2 interfaces preserving nonfusion model tensors.
Native acceptance/throughput remain deferred and will not initialize QAT.

Bounded owners and isolated worktrees:

- Data/provenance Sol: `/private/tmp/eagle-fusion-real-data`,
  `/root/real_data`; sole remote read-only transport.
- Fitter Sol: `/private/tmp/eagle-fusion-real-fitter`, `/root/fitter`;
  existing fitter real-import seam and focused tests.
- Independent Luna: `/private/tmp/eagle-fusion-real-validator`,
  `/root/validator`; producer/eligibility audit, tests and CPU experiment.
- Report/integration: `/private/tmp/eagle-fusion-real-report`.

Base revision `bdd6fd4`. Discovery and adapter work are underway. Original
source/base hashes and the already tested synthetic algorithm are recorded in
the [prior report](../fusion-binary-discrete-a8-2026-10-03/README.md). Any missing
actual raw boundary or eligibility field will be reported exactly, without
substituting diagnostic or synthetic inputs.
