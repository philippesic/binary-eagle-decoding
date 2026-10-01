# QAT optimization readiness runbook

## Current boundary

Preparation only. No real-data optimizer updates or sealed-final reads are
permitted by this goal. The existing frozen preparation job retains RTX5080
ownership until its supervisor/process groups are terminal and released. Read
`docs/STATUS.md` and the shared host registry before any remote action. Every SSH
connection goes through tmux MCP; preserve the existing host tmux job/socket.
RTX2080Ti remains paused. See `docs/AGENT_OPERATIONS.md`.

The CPU suite and native CPU fixtures establish implementation correctness,
not CUDA performance, memory fit, or native GPU readiness. An optimized CUDA
launcher refuses a missing/stale measured readiness receipt. There is currently
no passing real CUDA receipt for this goal.

## Prepared controls

`configs/qat_optimization_profiles.json` defines separate paired experiment
profiles. `baseline` retains reference cache/head/gradient controls; `speed`
selects chunked K/V-only context, stacked heads and single-forward A1 backward.
Further profiles add one mechanism: lower sign inertia, SGD, weight-unit and
scale-gradient rules, learned activation ranges/thresholds, depth weighting,
rank1/4 raw-input fusion correction, or fusion/all row midpoints. The combined
profile is a contract smoke, not a selected quality recipe or measured winner.

Produce a new config without loading a model or querying CUDA:

```sh
.venv/bin/python scripts/prepare_qat_optimization_recipe.py \
  --profile speed --native-commit <full-published-native-commit> \
  --output runs/qat-optimization-readiness/speed.json
```

Each optional recipe is recorded in config and critical source identity. Do not
resume the old frozen experiment with changed source/config. Its data may be
used only through the exact audited ancestry and newly validated provider
contracts. Do not change target/verifier precision, frozen datasets or caps to
make a test pass.

## Native and actual CUDA proof

After ownership release, the sole operator creates a separate project worktree
under the registry workdir and a unique run directory. Fetch published parent
and native commits through Git. Build a new CUDA runtime with the actual RTX5080
architecture and record toolkit, compiler, driver, native/project revision,
hashes, placement, precision and live resources. Keep the old binary/source
intact. Supervise remote work through `scripts/remote_job.py` in a new detached
Linux host tmux session, with the ownership and disconnect procedures from
`docs/AGENT_OPERATIONS.md`.

The native backend fixture supports an explicit CUDA request and rejects CPU
fallback:

```sh
./build/bin/test-eagle3-learned --backend CUDA
./build/bin/test-backend-ops -b CUDA -o W1A1_MUL_MAT
```

The unit fixture checks raw-bit A1 signs, A4/A8 code selection, scales, tails,
shared packing, affine input sums, nonzero correction/midpoint execution,
metadata rejection and actual encoder graph placement. CPU results cannot be
relabeled as CUDA. Integer-code sums are exact; A16's FP16 boundary/F32 sum is
an explicitly normal-precision diagnostic exception.

Actual-model native decisions use the versioned gate with independently pinned
train prompt/source/checkpoint/export/runtime artifacts. A4 requires its own
proof; neither A8 nor A1 grants A4 admission. Old schema/proofs cannot admit new
learned or affine formats. New source/config and current effective packed signs,
scales, activation parameters, rounded factors and midpoints bind the native
replay; original latent magnitudes/offsets are training state, not native data.

The measured paired harness consumes hashed actual native evidence and performs
full-shape resident forward/backward with optimizer.step forbidden:

```sh
.venv/bin/python scripts/check_qat_optimization_readiness.py \
  --allow-cuda --provider w1ax_multishard_provider:create_provider \
  --provider-manifest <new-eligible-train-provider.json> \
  --config <new-recipe.json> --native-evidence <actual-native-evidence.json> \
  --output <new-unique-readiness-directory>
```

It requires all nine binary gradients and later-state/K/V reachability, finite
optional gradients, unchanged masters/moments, actual hardware/runtime, native
choice/RMS gates, and reserved/free memory limits. Zero U intentionally gives a
zero V gradient at initialization; the separate nonzero correction fixture
proves both factors/input paths. A legitimate all-zero binary scale gradient
fails smoke; select another bounded audited training root or investigate the
mechanism, never weaken the gate silently.

Warmups and five order-balanced repetitions compare reference/cache/configured
paths. B2/B4 are independent graph/memory probes at one weight snapshot, not an
implemented larger-Adam-update recipe. A failed larger probe is preserved and
has no fit claim. Report allocator and whole-device memory, synchronized times
and any skipped probes. Training speed is distinct from native acceptance and
end-to-end decoding success; Q4_0 remains the comparison target.

## Curricula and refresh

`train_qat_curriculum.py` implements one model with budgeted direct A1,
A8→A1 or explicit A8→A4→A1 stages. Every precision needs a new independently
eligible provider/readiness proof. The schedule, global warmup, update/time and
row/token exposure budgets are immutable; stage switches preserve signs/scales,
raw factors and midpoints, create a fresh activation bank and optimizer, and do
not reset global budget accounting. Exact midphase/boundary resume is tested.

Default CLI validates without model/CUDA work. Explicit `--start --allow-cuda
--prepare-only` performs every-stage forward/backward and saves a zero-update
checkpoint. The actual optimizer path also requires a measured receipt bound
to the whole schedule, not just an isolated stage. Do not start real optimizer
updates under this goal.

Refresh is native recapture on current-student prefixes, with checkpoint,
export, actor, prompt, feature/label and runtime hashes. Changed-prefix teacher
rows are never relabeled or reused. Old v1 readiness cannot admit new affine
actors; versioned v2 gate/refresh-v3 bindings check the complete schema2–5
contract. Refresh changes a data source and therefore requires a new exact
resume boundary, not a silent live-source replacement.

## Raw fusion calibration

`fit_fusion_correction.py` requires provenance-joined train raw fusion inputs,
quantized-parent outputs and matching reference outputs. The calibration split
must separate prompt content; arrays, layouts, quantizer/weight/operand hashes,
positions, domains and depths are verified. Do not substitute captured head
states, diagnostic smoke rows or a different quantizer's references. No real
fit has occurred, and the existing small local manifests do not prove those
operands are locally present. Synthetic tests establish the fitting API only.

## Completion record

Publish raw logs/receipts in ignored runs/results paths and a compact report in
`experiments/qat-optimization-readiness/`. Record actual hardware and precision,
all source/artifact hashes, tests, memory/timing outcomes, decisions and pending
limits in `docs/goals/qat-optimization-readiness.md`. Complete the goal only after
actual CUDA/native proof exists for enabled launch recipes. Integrate tested
commits into main, push, and retire only fully integrated clean worker worktrees.
