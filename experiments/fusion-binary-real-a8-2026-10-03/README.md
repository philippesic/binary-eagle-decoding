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

## Authenticated package and CPU admission checkpoint

Read-only extraction completed and both owned local transports were closed.
Final package is `results/fusion-binary-real-a8-20261003/data-v2/` in the primary
checkout. External receipt SHA256:
`75cbf2b83fdcc14b2dc338a5ac2fe27188a41cd6957fa311ea7d0e6bcd857c87`;
manifest `32db46b48d6f9737aefb2a4be01b87c7376e645bbc71972329da0f68cc338619`;
operands `04ad9777d23df2d42863aadde2a8cf672981bcedee83e601cae60da8e8bb6681`.
All 384 selected F32 rows match original native raw feature bytes at the
recorded offsets. Historical producer is
`b4e366d4f0a30cac07f14d51c54c5b1329b3f485`; this is distinct from the current
`9e2c7a90051e738751aab7d7bd7c2d8201fb76e3` CPU export-gate runtime.
An interim receipt misidentified the producer; it was corrected before fitting.

Source capture remains `training_eligible:false`, `preparation_only`. Separate
authenticated continuous readiness, full TRAIN provider and completed
preparation-ready receipt grant its training use. Original eligibility fields
and bytes are preserved. Independent Luna verified this grant chain, all local
source hashes, prompt/group joins, native decoded/retained events, exact F32
boundary, and BF16-promoted source equality.

Selection uses the first 12 qualifying independent TRAIN prompt groups from
the first completed shard, 32 evenly spaced accepted-prefix rows each. Fitting
contains 3 prose, 3 code and 2 reasoning prompts; validation contains 2 prose
and 2 reasoning prompts, with no code validation. This is a bounded sample,
not a full-corpus generalization estimate.

Data adapter `efbaeb6`, real importer `1eda427` are integrated at `9464a3e`.
All 26 relevant CPU tests and Ruff passed. Conservative estimated workspace is
932,577,280 bytes, below the unchanged 1 GiB cap. Luna has GO for exactly one
`fit-real-01` CPU fit/export with the final receipt pin, two-thread environment,
unchanged A8 contract/config and 12-hour fitting cap. No fit result is claimed
at this checkpoint.

Current CPU library and model-load helper compiled with CUDA/Metal/BLAS/OpenMP
disabled. The original full F16 draft passed native load and tensor validation
with zero GPU layers, no context or inference. Candidate loading/replay remains
pending. Initial all-target build hit unrelated disabled app include paths;
building the intended `llama` library target succeeded without source changes.
