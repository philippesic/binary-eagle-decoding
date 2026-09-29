# Opt-in EAGLE runtime engineering

These CPU contract checks prepare native GPU A/B experiments within the joint
body/head goal. They do not establish CUDA correctness or speed. Q4_0 remains
the primary comparison; run shared runtime selectors fairly for Q4_0, FP16 and
W1Ax, retaining target-only as the control. The original pilot runtime remains
pinned independently.

## Direct mapped vocabulary

Native `51eed7b54` adds `GGML_EAGLE_COMPACT_LOGITS=1` for EAGLE's CPU top-10
sampling path. It keeps raw draft-vocabulary logits on the graph and copies only
those entries to host. The I64 `d2t` map is absolute, valid and unique; candidates
carry target IDs. The existing sampler chain computes probabilities and consumes
RNG. A tie among the first eleven ranked values, nonfinite raw/bias values,
insufficient finite support or backend sampling requests expanded fallback.
This preserves the existing tie implementation without inventing a new tie rule.
The current `p_min`, forced-token and accept behavior remains in the caller.
Full-logit getters expand on demand for capture and fallback.

Native `ba0d05079` fixes compact-logit lifetime across encoder-only calls and
failed decode cleanup. Use both patches. The host allocation retains full-vocab
row stride to preserve existing getters and output reordering; this is not a
host-memory reduction claim. Capture that requests full logits exercises the
expanded fallback and is unsuitable for measuring the compact selection gain.

## K/V-only catch-up

Native `ba0d05079` adds `GGML_EAGLE_KV_ONLY_CATCHUP=1`. Only EAGLE `process()`
uses its dedicated decode entry point. Eligibility requires the one-layer EAGLE
architecture, both token and feature input, every logits flag false, no embedding
or layer-tap output, and disabled or masked nextn extraction with zero outputs.
The caller explicitly consumes no decoder output. Ineligible calls run normal
decode. The graph preserves K/V projection, RoPE, optional cache rotations and
the same cache-copy operations; it exits before Q, attention, FFN and head.
This changes process catch-up, not draft-only latency. It is separate from the
existing unused-head pruning selector.

Native `83a070ec19dbe683203400f2cfbb104a4328c5de` supplies the explicit
`<algorithm>` include required by GCC for its eligibility check. Its ancestry
also retains the final pilot cache-capture boundary fix `0be8d5ba`.

## Shared activation packing

Native `a036e5208` adds `GGML_EAGLE_SHARED_PACK=1`. Explicit graph pack nodes
hold A1 signs/scales or A4/A8 codes/scales and A4 bit planes. Q/K/V and FFN
up/gate reuse a pack keyed by their activation tensor identity. Buffer lifetime
is managed by graph allocation; no CUDA pool pointer is reused across projections.
The A16 row/group kernels keep their existing inside-kernel F16 rounding and do
not create a pack node. This is shared operand preparation, not fused dot launches.
It applies to binary projections; common sampler/cache optimizations apply to
Q4_0 and FP16 too.

CPU scalar-reference operator checks passed 133/133 cases, including 21 shared
variants and non-word-aligned/zero/strided inputs. Three additional one-pack/
three-consumer fanout graphs passed. Actual tiny packed EAGLE graphs with sharing
passed the runtime fixture at A1/A4/A8. Native `aeda099ba` fixes an existing
loader gate that unnecessarily required a target peer despite an owned embedding
and packed head. Models borrowing embeddings still require their target peer.
CUDA build, dispatch, trajectories and timing remain unverified for sharing.

## Backend recurrent state

Native `8fd9b399a` adds `GGML_EAGLE_DEVICE_STATE=1` for one sequence and one
output row. A persistent F32 tensor retains the last prenorm state on its actual
backend, copies it to the next input there, and reads back only when a host
getter is requested. Stage inspection uses the ordinary host path. Strict
position, normalized sequence, cache maximum and mutation-generation checks
reject stale state after removals, position changes and successful or failed
restores. The context owns its lifetime and accounts for its allocated bytes.

The opt-in explicitly assigns its input leaf to the model layer backend. GPU
validation must inspect this assignment and verify recurrence because input
placement can affect the scheduler's compute choices. CPU dense and packed A4
fixtures passed host/resident trajectories, serialized cache equality, queued
recurrence without intermediate getters, wrong-position and nonzero-sequence
fallbacks, earlier-row removal with unchanged maximum, different same-maximum
cache restoration and failed restoration. This is a bounded recurrent path;
multiple output rows and sequences keep ordinary behavior.

## Integer reduction experiment

Native `dfc9c5d2a` adds `GGML_W1AX_WARP_REDUCE=1` for CUDA A1 XOR/POPC and
A4/A8 integer dot reductions. Four complete warp32 groups replace their shared
reduction barriers with shuffle sums; invalid rows contribute zero and take
part, and the A1 tail mask is unchanged. The default shared reduction, packing,
scales and A16 inside-kernel cast remain as before. HIP/MUSA exclude this path
at compile time. Shared storage is still declared, so this patch makes no
occupancy-saving claim.

Archived SM75 profiles measured large-N=37 head XOR/POPC dot at approximately
885–1,187 us versus packing at approximately 5–9 us. These profiles do not
measure current launch gaps, barrier savings or pilot draft/process/request
cost. The shuffle mechanism is a hypothesis awaiting CUDA scalar-reference,
integer assertion and paired timing gates. CPU testing cannot execute it.

## Diagnostic event accounting

Native `c3c548d2e` adds default-off `GGML_CUDA_EAGLE_EVENTS=1`. It emits bounded
nested event spans, fused-node counts, tensor/operation dimensions, backend and
model context IDs, process/draft labels, projection pack/scales and dot/output
stages, ordinary projection spans and buffer transfers. A16 dot spans include
the existing inside-kernel cast. Ordinary Q4/FP16 projections remain combined
when there is no separate operand-preparation marker. Buffer annotations are
hints; they do not establish transfer ownership.

CUDA graph capture skips event creation/recording/synchronization for captured
children, emits null timing, and records top-level CUDA graph node/type
inventory. Graph replay retains its outer event span; captured node durations
remain unassigned. Event limits and graph inventory limits explicitly report
truncation. These synchronized event spans include stream idle and dispatch
and alter overlap; they are intrusive diagnostics, not official throughput
measurements or pure kernel busy time.

`scripts/analyze_eagle_gpu_events.py` checks span consistency and reports
missing parent IDs, partitions each frame by its deepest nested scope without
double counting, keeps overlapping siblings explicit, and preserves unassigned
time. Separate frame/stream clock origins are never summed. Optional same-host
CLOCK_MONOTONIC request joins use inclusive host unions and preserve request
remainder; they do not subtract GPU durations from host time. The benchmark
runner retains original request boundaries and now records their clock values;
it permits CUDA events only in `instrumented_env`, excluding them from timed
server environments. Seven synthetic accounting tests pass; they are not GPU
measurements. CPU compilation cannot validate CUDA event APIs.

## CPU acceptance check

Apple M3 Max CPU, NumPy 1.26.4, explicit Metal/CUDA/BLAS/OpenMP disabled, one inference thread:

```sh
cmake -S third_party/llama.cpp -B results/runtime-opt-cpu-build \
  -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build results/runtime-opt-cpu-build --target test-sampling test-backend-ops -j 8
python3 scripts/make_eagle_runtime_fixture.py \
  --output results/eagle-runtime-fixture.gguf
results/runtime-opt-cpu-build/bin/test-sampling
results/runtime-opt-cpu-build/bin/test-sampling \
  --eagle-fixture results/eagle-runtime-fixture.gguf
python3 scripts/make_eagle_runtime_fixture.py --packed \
  --output results/eagle-runtime-packed-fixture.gguf
GGML_EAGLE_SHARED_PACK=1 GGML_W1AX_ACT_BITS=4 \
  results/runtime-opt-cpu-build/bin/test-sampling \
  --eagle-fixture results/eagle-runtime-packed-fixture.gguf
results/runtime-opt-cpu-build/bin/test-backend-ops -b CPU -o W1A1_MUL_MAT
results/runtime-opt-cpu-build/bin/test-backend-ops -b CPU -o ADD -p shared=1
```

The fixture is synthetic: 16 hidden channels, 128 target tokens, 32 mapped draft
tokens and fixed RNG seed 4242. No real checkpoint, capture or held-out prompt is
used. Its SHA256 is
`f55d2b5f62dc7187d741a3d9abc259839f82a94d7122544bdace0170b8c92495`.
The ordinary sampler suite including mapped-ID/bias/probability/RNG fallbacks
passed. The fixture passed exact serialized used-cache equality, following
logit/prenorm equality, output-consuming fallback, compact multirow mapping,
full-getter expansion, mode toggling and encoder-only cleanup. CPU checks do not
validate SM120/SM75 dispatch or graphs. The resident checks described above also
passed in the dense and packed A4 fixtures.

## Queued GPU acceptance and timing

The sole coordinator owns GPU execution. Use separate fresh server processes,
frozen model/config/prompt hashes and the approved benchmark runner. First run
quality checks for each selector individually with flags `0` and `1`, holding all
other optimization selectors at `0`; compare generated raw IDs, proposal counts,
round/depth tokens, stop probabilities, acceptance and cache/state traces.
Cover greedy, nonzero `p_min`, forced proposals where available, and ties in the
operator fixture. Compact selection needs CPU draft sampling; backend sampling
must explicitly demonstrate fallback. Then run five measured repetitions after
warmup in alternating orders, reporting process/draft/request scopes separately.

The preparation tool writes immutable source/config hashes and changes only the
selected common runtime flag. A compact off/on pair uses CPU draft sampling in
both conditions (`draft_backend_sampling: false`). This differs from the historical
backend-sampling policy; make no cross-policy speed claim. Existing configs retain
backend sampling by default, and the existing diagnostic sampling switch remains
supported. Native selectors are fixed for each fresh server process.

Prepared commands on the coordinated RTX 5080 shell, within tmux MCP, after the
pilot and source/quality gates permit the new runtime:

```sh
python3 scripts/prepare_eagle_runtime_ab.py \
  --source configs/w1_phase1b_prune_off.json --selector compact_logits \
  --output results/runtime-compact-configs
python3 scripts/remote_job.py runtime-compact-quality-off -- \
  python3 scripts/run_binary_rescue_benchmark.py \
  --config results/runtime-compact-configs/off.json --mode quality \
  --output results/runtime-compact-quality-off
python3 scripts/remote_job.py runtime-compact-quality-on -- \
  python3 scripts/run_binary_rescue_benchmark.py \
  --config results/runtime-compact-configs/on.json --mode quality \
  --output results/runtime-compact-quality-on
python3 scripts/compare_eagle_prune_quality.py \
  --selector GGML_EAGLE_COMPACT_LOGITS \
  --off results/runtime-compact-quality-off/manifest.json \
  --on results/runtime-compact-quality-on/manifest.json \
  --output results/runtime-compact-quality-comparison.json
```

For timing, use the same prepared pair with `--mode timed`, unique supervisor/run
output names, and the timing comparator's same `--selector` argument. Quality
and timing comparators require source/model/workload/policy equality plus only
the named runtime flag changing. They retain the historical pruning selector as
their default. Quality comparison also checks stop probability. Prepare K/V-only
with `--selector kv_only` and resident state with `--selector device_state`;
shared packing and warp reduction require a pinned source config with an
explicit packed A1/A4/A8 variant (`--selector shared_pack` or `warp_reduce`), as candidate D/A16 has
no activation-pack stage. These are preparation commands, not GPU jobs started by
the engineering worker. Preserve model hashes, precision, proposal cap, confidence,
sampling, context, batch/KV settings and unassigned timing.

## Coordinator CUDA gates

Stable source `8fd9b399a` includes sampler, cache-only, pilot cache boundaries,
shared packing and resident state. Event source `c3c548d2e` and warp source
`dfc9c5d2a` extend that ancestry and remain separate default-off patches. The
engineering worker ran no CUDA compile, GPU job or SSH session. The coordinator
must run these in its isolated checkout through the approved tmux supervisor.
The fixture generator needs NumPy and the checked-out native `gguf-py` package.

After building CUDA tests, use the explicit CUDA fixture mode (it requires actual
CUDA0, offloads the tiny model with `n_gpu_layers=99`, and enables K/Q/V offload):

```sh
build-cuda/bin/test-sampling --eagle-fixture-cuda results/eagle-runtime-fixture.gguf
GGML_EAGLE_SHARED_PACK=0 GGML_W1AX_ACT_BITS=4 \
  build-cuda/bin/test-sampling --eagle-fixture-cuda results/eagle-runtime-packed-fixture.gguf
GGML_EAGLE_SHARED_PACK=1 GGML_W1AX_ACT_BITS=4 \
  build-cuda/bin/test-sampling --eagle-fixture-cuda results/eagle-runtime-packed-fixture.gguf
build-cuda/bin/test-backend-ops -b CUDA0 -o W1A1_MUL_MAT
build-cuda/bin/test-backend-ops -b CUDA0 -o ADD -p shared=1
GGML_W1AX_WARP_REDUCE=1 GGML_W1AX_ASSERT_INT_DOT=1 \
  build-cuda/bin/test-backend-ops -b CUDA0 -o W1A1_MUL_MAT
```

Repeat the packed fixture pair at bits 1 and 8. Require 133 operator cases and
three fanout graphs, actual packed/warp dispatch markers, following logits/state
and serialized cache equivalence. CUDA allocation, scheduling and precision
must be identified in the report. These gates do not validate SM75 speed.

For event compile/capture safety, use a short direct diagnostic fixture pass
with graph capture/replay enabled, then a separate graph-disabled node pass:

```sh
GGML_CUDA_EAGLE_EVENTS=1 GGML_EAGLE_SHARED_PACK=1 GGML_W1AX_ACT_BITS=4 \
  build-cuda/bin/test-sampling --eagle-fixture-cuda results/eagle-runtime-packed-fixture.gguf \
  > results/eagle-events-graph.log 2>&1
GGML_CUDA_EAGLE_EVENTS=1 GGML_CUDA_DISABLE_GRAPHS=1 \
  GGML_EAGLE_SHARED_PACK=1 GGML_W1AX_ACT_BITS=4 \
  build-cuda/bin/test-sampling --eagle-fixture-cuda results/eagle-runtime-packed-fixture.gguf \
  > results/eagle-events-direct.log 2>&1
python3 scripts/analyze_eagle_gpu_events.py \
  --logs results/eagle-events-graph.log results/eagle-events-direct.log \
  --output results/eagle-events-analysis.json
```

Require valid parent/frame references, null captured node times, nonnegative
graph replay spans, pack/dot markers in direct execution, explicit inventory
counts and no event records in a corresponding event-off pass. Inspect missing
parent IDs and outside-root intervals rather than silently assigning them.
Real common EAGLE server instrumentation must separately exercise both
`process` catch-up and `draft` plus optional same-run request joins. Keep this
intrusive instrumentation outside official timed runs. Individual opt-in
quality A/B gates precede paired throughput measurement; no speed claim is made.

## Stable SM120 CUDA checkpoint (2026-09-29)

Coordinator-built native `8fd9b399ab124eeec3b363c56eacbd690f2e75b8` passed
the supervised RTX 5080/SM120 CUDA build. `build_llama.py --with-tests` builds
backend tests but not `test-sampling`; its explicit target build then passed.
An initial fixture launch stopped with exit 127 before inference because that
executable was absent. The unique retry completed successfully.

Actual CUDA results:

- Dense runtime fixture: exit zero, `EAGLE runtime fixture OK`, actual device
  NVIDIA GeForce RTX 5080. Resident state reports CUDA0 buffer, 64 logical bytes
  and 128 allocated bytes for this tiny geometry. Following logits/prenorm
  and serialized cache comparisons, mutation/restore and implicit-sequence
  guards passed. This tiny size does not describe the real model's geometry.
- `test-backend-ops -b CUDA0 -o W1A1_MUL_MAT`: **133/133** tests passed.
- `test-backend-ops -b CUDA0 -o ADD -p shared=1`: **3/3** fanout graphs passed
  for A1/A4/A8, with actual packed CUDA dispatch recorded.

These are correctness checks, not throughput or SM75 evidence. Preserved runtime
copy and 27-entry executable/library hash inventory are under
`runs/w1-runtime-stable-cuda-build-20260929/runtime/` and `runtime-hashes.json`.
Server SHA256 is
`6d855804aa7973dce803fd55a820080a832da304a71155de1b1dcab3c6bc34b9`;
backend-test SHA256 is
`b18ace58d85bc20da889771b694803987fe56f9394d22e75cae96c29a7fe6901`;
sampling-test SHA256 is
`ca9244747e4cb5b5fb975567b5a8a88b3acb027b834e72020c4d759afe2a881f`.

| Supervised run | Exit | stdout SHA256 |
| --- | ---: | --- |
| `w1-runtime-dense-cuda-fixture-retry-20260929` | 0 | `0e449399a48733628ab2f8aa14d75027fba3d9a91f3c88112b22a6840ff2faea` |
| `w1-runtime-w1ax-cuda-ops-20260929` | 0 | `ae50b5e8212cf9600cc474aed3ed55bc26efeae0b7a850a65ba19495fff9d207` |
| `w1-runtime-shared-fanout-cuda-20260929` | 0 | `868f675403b6ae97ec2824fd8fa7b330c4b2302ad94a97af3a5467866a1935ef` |

The first packed A1/shared-off fixture reached its final encoder-only cleanup
check, then aborted because the fixture's encoder batch had uninitialized
position/sequence/logits metadata. Run
`w1-runtime-packed-a1-shared0-cuda-20260929` records exit `-6`; no other packed
run started after it. This is a concrete fixture setup defect; preserve every
assertion and rerun all packed precision/sharing combinations after explicit
metadata initialization. The owner is testing that fix independently on the
stable ancestry. Real selector A/B, event/warp CUDA checks and performance
remain pending. The original calibration/capture evidence stays pinned.

### Corrected fixture gate

Published fixture-only native `1e7625635149deff85d1e5ff9b2f167b61fec4b0`
(parent integration `35c8b27`) initializes the encoder batch's position,
sequence count, sequence ID and logits flag. All original assertions remain.
The explicit CUDA sampling-test target rebuild passed, followed by dense A16
and **all six packed A1/A4/A8 × shared-off/on fixtures**, each exit zero and
`EAGLE runtime fixture OK` on actual RTX 5080. Every run reported resident
CUDA0 storage and passed following-state/cache and lifetime checks.

Combined per-run results, state/log hashes and placement evidence:
`runs/w1-runtime-cuda-fixture-fix-build-20260929/fixture-results.json`, SHA256
`ea79b224a6853555303727e9a712d17f955ec2026d5dffaec1a82807eea779a5`.
Corrected sampling executable SHA256:
`4ea2ca542290240558d149d4237c4660ed423a1070d79abd428568704a4262ad`.
Dense/packed input GGUF SHA256 respectively:
`f55d2b5f62dc7187d741a3d9abc259839f82a94d7122544bdace0170b8c92495` /
`b287e1040a7521608d7809492fda310b462aa1bf0d49116cd557129e89b429ae`.
Server/library runtime is unchanged from the stable preserved build.
All seven process groups were absent afterward; GPU was idle at 0%/2,843 MiB.

Real three-domain diagnostic quality pairs now run sequentially for compact
sampling, K/V-only catch-up and resident state. Each pair holds all other
opt-ins off, Q4_0 and untrained row-A16, 128-token cap, draft length five and
`p_min=0`. The original fixed four-variant pruning result is a separate
completed experiment. These new quality pairs make no timing, nonzero-p_min
or sealed-final quality claim. Their results are pending.
