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

### Real quality gate result

All six inference supervisors completed exit zero. Compact sampling and K/V-only
catch-up each matched **6/6** measured request pairs, with zero raw-ID, text,
finish, acceptance or checked round-field mismatches. Each pair covers Q4_0
and untrained row-A16 over the same prose/reasoning/code sample. Q4_0 accepted
183 over 138 observed rounds; row-A16 accepted 31 over 288 observed rounds.
These are bounded regression results at p_min=0, not broad quality or speed.

Resident state **failed** this gate: output IDs/text still match all six pairs
and Q4_0's counts/rounds match, but all three row-A16 requests change proposal/
acceptance counts and round count (nine reported field mismatches). Row-A16
accepted counts changed from prose/reasoning/code `6/14/11` to `10/18/10`.
Prose and reasoning first proposals already differ in round zero; code diverges
at the second proposal of round zero. No gain is attributed to resident state
and its timing is withheld. The separate packed A16 tiny fixture passed,
showing the tiny synthetic gate did not cover this real-model divergence.

| Comparison report under `runs/` | SHA256 |
| --- | --- |
| `w1-runtime-compact_logits-quality-compare-20260929/report.json` | `7f09fb8a6f4fea1ac488609326b223821de348abcdfca98cbfc933e4dc04468b` |
| `w1-runtime-kv_only-quality-compare-20260929/report.json` | `741caffe9f85e7925230f54d1a8359ceb5610f27c31f5fdf06d345d34cf2b5eb` |
| `w1-runtime-device_state-quality-compare-20260929/report.json` | `f17cb223ba7ddb67b74d3989d57517f7b3fb83763ef9e85cf7c4360656ea6518` |

The resident report directory also retains `first-divergence.json` with exact
off/on round records. All supervisors are terminal and GPU returned to
0%/2,843 MiB. Native owner is investigating the input-leaf placement/copy
structure under a bounded placement-only diagnostic; no numerical assumption
is a proven cause yet. Keep resident off and preserve the failed evidence.
No exact comparator or numeric threshold is being relaxed. Event/warp source
`82b7379d` is coherently published with the fixture fix but remains unintegrated
behind this investigation.

### Placement isolation

The tested default-off diagnostic native `be09f61c5460a77b79f51fc0bd8af285f0ee6371`
is integrated as `e361f74`. `GGML_EAGLE_DEVICE_HOST_INPUT=1` only skips forced
input-leaf assignment, retaining resident capture/API and synchronized existing
cross-backend fallback. The flag and scheduler debug level were identical
in both fresh quality conditions; stage/head capture remained absent.

This probe restored **6/6 exact request pairs**, including every checked
proposal/acceptance/round field. Counts returned to Q4_0 183 accepted and
row-A16 31 accepted. Report
`runs/w1-resident-host-input-compare-20260929/report.json` SHA256 is
`b0081838b0c66d9eae60b45b74bbbb2212c51cf65a5d9d9a6fd484630d8867ad`.
On logs contain 864 Q4_0 and 1,828 row-A16 transfer-fallback markers across
warmup/measured requests, proving resident recurrence remained active. This
isolates forced input placement as the trigger, without proving which
numerical/scheduling change it introduces. It preserves host traffic and is
**not** a device-only solution or timing candidate. Both jobs stopped and
GPU returned to 0%/2,843 MiB.

Scheduler debug was suppressed by the runner's standard `-lv 3`. A single
bounded follow-up run `w1-resident-scheduler-debug-20260929` raises only the
recorded server command's verbosity to five, retaining the same sample,
models, policy and diagnostic flags. It seeks admission evidence for original
CPU consumers and existing GPU input-copy destinations; no proposal-margin
or exact comparison gate is weakened. Permanent changes must preserve
original graph placement/split semantics, reject unsupported CPU/alias/mixed
destination graphs, and retain explicit host fallback. No bit-exact norm
reimplementation is planned.

## Order-balanced runtime timing on RTX 5080

Eight supervised inference jobs completed exit zero on native `be09f61c5`,
using a frozen executable/library copy and explicit verified LD_LIBRARY_PATH.
For compact sampling and K/V-only catch-up separately, block a ran off then on;
block b ran on then off. Each condition used five repetitions, three frozen
train prompts, Q4_0 and untrained row-A16, 128 tokens, draft length five,
p_min=0 and two warmups per variant/server block. Resident, shared packing
and warp reduction were off. No intrusive event/capture instrumentation ran.

All **120 measured request pairs** across four comparisons matched raw IDs and
speculative counters. All **80 server blocks** recorded verified CUDA-graph
launches. Q4_0 accepted 915 drafts over 680 proposal verification rounds per
condition; row-A16 accepted 155 over 1,430. Timed counters exclude no-proposal
rounds, so these are not complete-round acceptance rates.

| Selector | Variant | Order-balanced on/off decode | Order-balanced on/off client request |
| --- | --- | ---: | ---: |
| Compact mapped sampler | Q4_0 | **1.04028** | **1.03611** |
| Compact mapped sampler | Row-A16 checkpoint zero | 1.02666 | 1.02537 |
| K/V-only catch-up | Q4_0 | **1.00817** | **1.00776** |
| K/V-only catch-up | Row-A16 checkpoint zero | 1.08174 | 1.07959 |

These are geometric means of the two block ratios, not new confidence
intervals. They show gains on this bounded three-prompt workload: approximately
4.0%/0.8% Q4_0 server decode for compact/K/V-only, respectively. Row-A16 still
has much lower acceptance and throughput than Q4_0. This does not demonstrate
a binary drafter beating the primary baseline.

Each variant has 15 matched requests and **three prompt clusters per block**.
The bootstrap resamples those prompt clusters and keeps their repetitions;
it does not establish population uncertainty from three train examples. A
historical hard-coded report label said 24 development prompts. Fresh CPU
comparisons use the corrected scope label (`78b7382`) with unchanged arithmetic;
original comparison files and the initial old-label re-audit remain retained.
Only the `reaudit-final` reports below are authoritative for this summary.

| Fresh comparison under `runs/` | SHA256 |
| --- | --- |
| `w1-runtime-timing-reaudit-final-compact_logits-a-20260929/report.json` | `273efcb440c39002b52d6b1dbf14efc399d4d640067647679a354ef5ad6d463e` |
| `w1-runtime-timing-reaudit-final-compact_logits-b-20260929/report.json` | `07c1fb5cefc0382d13ccf8089e1d74303455efb5a4f09e8f62ce292918bad625` |
| `w1-runtime-timing-reaudit-final-kv_only-a-20260929/report.json` | `580be25aad1621f63404799408b8866ace4d3509a149ed0da675ddfcaa81faa8` |
| `w1-runtime-timing-reaudit-final-kv_only-b-20260929/report.json` | `50901eae3dc706ba0238b46c44766b46ca3543ab95e5ec0fd54caf86badd461a` |

Summary `runs/w1-runtime-timing-summary-final-20260929/summary.json` SHA256:
`e38ea936aa8c94c92f759cce5e2ef46f8d7c8dbe673da1134ddcdeda32949117`.
Condition runs are `w1-runtime-<selector>-timed-<a|b>-<off|on>-20260929`;
frozen binaries/config hashes are in `w1-runtime-timing-freeze-20260929/manifest.json`.
Per-block telemetry includes whole-device startup/warmup/measurement/shutdown
samples, clocks and power; polling is one second and can miss short peaks.
The GPU was idle at 0%/2,843 MiB before and after the queue, and no other
project experiment ran. There is no target-only drift control in this bounded
queue, so order balance and telemetry do not exclude all external interference.

These are concurrency-one request/decode observations on repeatedly used train
prompts, not saturated serving capacity, held-out quality, SM75 results or
process/draft component attribution. Both compact conditions use common CPU
draft sampling; do not compare their rates as a gain over the historical
backend-sampling policy. The earlier fixed four-variant pruning experiment
is separate and unchanged.

## Combined CUDA and resident fix validation

Native `b4e366d4f0a30cac07f14d51c54c5b1329b3f485` completed the SM120 CUDA
build plus explicit sampling/allocator targets. CUDA-linked allocator tests
passed; their dummy backends test routing/lifetime rather than CUDA streams.
Dense and packed-A16 fixtures passed exact following logits/prenorm and
serialized used-cache comparisons, with genuine
`existing scheduler copy backend=CUDA0 (ordinary placement)` markers.

Real resident off/on quality then matched **6/6 request pairs** exactly across
Q4_0 and row-A16 on the frozen three-domain sample. Q4_0 accepted 183 drafts
over 138 observed rounds; row-A16 accepted 31 over 288, restoring the baseline.
Both on logs have the accepted CUDA0-copy marker and **zero transfer-fallback
markers**. Only the server log verbosity was raised to four in both conditions
to observe INFO markers; no stage/head capture, events or warp reduction ran.
Report `runs/w1-resident-fixed-quality-compare-20260929/report.json` SHA256:
`7f727172be9ba56a7c02305ee4dfde5f497f7c4f04f22a87bd00571013b84dba`.
The earlier failed forced-placement evidence remains retained. This closes the
concrete decision divergence for this bounded sample, not every possible
context or a resident speed claim.

Opt-in warp32 reduction passed **133/133** CUDA operator cases plus **3/3**
fanout graphs with `GGML_W1AX_ASSERT_INT_DOT=1`, actual warp dispatch and events
off. It changes A1/A4/A8 integer reduction only; A16 remains unchanged.
The corresponding stdout hashes are
`247614722bb7b12822f933d20f893f146f598796fb9a4be427230d6bde9474d7` and
`cdd25f699fd55ff407cc2778181dc30ce983b63d0191cb09b213fbba7a012d97`.
Kernel/end-to-end speed for warp reduction remains unmeasured.

## CUDA event safety and actual-model tracing

A4/shared-pack fixture passes with events on under graphs enabled and under a
separately labelled graph-disabled direct-node control. A corresponding event-off
fixture passes and emits no event records. Their audit contains 2,132 event
records, 806 frames and 62 graph inventory records: zero missing parents,
outside-root intervals, orphan inventory or truncation. Eighty-four captured
inventory events have null GPU times; 15 graph replay frames are present.
Direct-node execution includes 117 pack and 187 dot markers. Event analysis
SHA256: `a837f6e4c19c34fc12f3d8ea8aca0c9e5ba01a0a3761cc5d7f7f332b2a6eccc1`.

One actual-model intrusive trace then completed with Q4_0, row-A16 and row-A4
checkpoint zero over the frozen sample, five requests per variant including
warmups. It uses unchanged F16 target/KV, graphs enabled, no head/state capture
and no warp/resident/compact/cache-only selectors; row-A4 alone enables shared
packing to expose its operand preparation. A4 export SHA256 was reverified as
`a6081a5120d246435f53a3d3711607e156d3332b3ec75780019c842e85f169b1`.
Config SHA256 is
`ab2d4df637b3070ab6899d4585099af1bd8a2cde95480aca78c58a0fbc3cd190`.
Event cap is one million per process; node inventories retain their independent
limits. This configuration is an intrusive diagnostic, not official timing.

The actual trace audit found 152,710 events, 19,784 graph inventory records
and 86,277 frames, with zero missing parents, outside-root intervals, orphan
inventory or truncation. Frames include `process`, `draft`, `target_or_other`
and explicitly unassigned stages. Fifteen same-run monotonic client intervals
join to event records, preserving their unassigned host remainder.
Manifest SHA256:
`3a30ad7fcc15c928a744d68454eaa4b203f1ddf19604017d348a34d380c68baa`;
analysis SHA256:
`52f6f9f80def35b78df82a5a9e7a7c6b4b810da3531ffaface7e687109622bde`.
Runs are `w1-real-events-instrumented-20260929` and
`w1-real-events-analysis-20260929`. GPU returned to idle at 0%/2,763 MiB.

Event spans synchronize and include stream idle/dispatch and capture setup;
they are not pure kernel busy time and cannot be substituted for throughput.
Captured graph nodes have inventory without individual timing. Nested/overlapping
spans are partitioned per frame; independent origins are not summed. Transfer
annotations remain hints. These results validate the tracing/accounting path
without attributing all serving time or promoting any selector to default.

### Packed selector quality and final timing queue

On the pinned row-A4 checkpoint-zero export, shared packing and warp reduction
quality pairs each matched **6/6** requests exactly against their own off
condition (Q4_0 plus row-A4 over the same three prompts). Each holds all other
selectors off. Q4_0 accepted 183 over 138 observed rounds; row-A4 accepted 16
over 303. The packed model remains far below the primary acceptance baseline.
Comparison SHA256 values: shared packing
`9995bc3f90b092c010e8591265efc3e4c5e771dc076eee9a288603b7e4d3d706`,
warp reduction
`83fa6f269d4076f15cbb3fb785e90bf5fcbe52009b1783a907b88f5fbc7bad23`.
Artifacts are under `runs/w1-packed-<selector>-quality-compare-20260929/`.

Final timing freezes the validated b4 runtime and all loaded libraries under
`runs/w1-final-runtime-timing-freeze-20260929/`; manifest SHA256 is
`a199cfdabd81b5ba7414509e31007eab338c125b881a9276a78696cd9208fd5e`,
server SHA256
`b5093749d67888bc2cafdb6a65c479f4c182f0a904820f1dae4870b6ae66d41c`.
Six configs passed validation; explicit LD_LIBRARY_PATH/ldd prove all seven
llama/ggml linkage entries resolve in the frozen directory. Intrusive event
keys are absent and ambient experimental switches are cleared by the runner.
A premeasurement validation manifest is retained separately.

Resident-state, shared-pack and warp-reduction timing each runs off→on then
on→off with five repetitions and the existing frozen workload. Resident
compares Q4_0/row-A16; packed selectors compare Q4_0/row-A4. Expected pairs are
30 per comparison, 180 across six comparisons. These measurements remain
pending and will not be combined into an unmeasured multi-selector gain.

## Final order-balanced selector results

All 12 final inference jobs and six comparators exited zero. All **180 measured
request pairs** matched raw IDs and speculative counts, and all **120 server
blocks** verified CUDA-graph launches. The immutable b4 runtime/config hashes
above apply throughout. Events and captures were absent. Each variant has
three prompt clusters and 15 requests per comparison; all scope/uncertainty
limitations of the preceding bounded timing apply.

| Selector | Variant | Balanced on/off decode | Balanced on/off client request |
| --- | --- | ---: | ---: |
| Resident state | Q4_0 | 0.98814 | 0.99018 |
| Resident state | Row-A16 | 0.99132 | 0.99243 |
| Shared packing | Q4_0 control | 0.99943 | 0.99377 |
| Shared packing | Row-A4 | 1.00312 | 1.00346 |
| Warp reduction | Q4_0 control | 1.00179 | 0.99843 |
| Warp reduction | Row-A4 | 1.00289 | 1.00126 |

Resident state preserves the tested semantics but is about 1% slower here.
Shared packing and warp reduction change A4 decode by about 0.3%; warp's
unaffected Q4_0 control also moves about 0.18%. These small observations do
not establish a clear general performance benefit. Keep these experimental
selectors off by default. Compact sampling and K/V-only catch-up have the
preceding bounded positive results; no combined-selector gain was measured.
No result repairs the binary drafts' acceptance deficit against Q4_0.

Final summary under
`runs/w1-final-runtime-timing-summary-20260929/summary.json`, SHA256
`fe592af605d8abd119b7b4acf93e0045c8438eefec711c47ce480dd29aa109f6`,
records both block ratios, request/decode metrics, source report hashes,
180 matches, 120 graph blocks and no remaining process-group members.
Comparison hashes (block a / b):

- Resident: `51f881128814ca27bcf4eb4ce3b3c7efafc47d4e6d44b30edf65b32b73d8c1fb` /
  `248c7d01f05e6ff5a4184e9c179362c2c790ac2b4fa903b1b1a074a973fdd169`.
- Shared: `709ba3ca863b0d62678e1792b167c099a942115086c1039793ce19e674e636c8` /
  `ed60db3b2bf422e7b3591be6ae108634d5d3c92b8f0ae313425d2cc40b195136`.
- Warp: `a4f4d0e39afeec599af748eaffcb0af5c30c240d52a379da2efbae62c7247bf2` /
  `633c8f1a74027b4e046237327a0ccbd508f28f8cd4431c1a66ef734c72975ed5`.

The final explicit A1/A8 shared-off/on CUDA fixture checks on b4 also passed,
with accepted CUDA0 source-copy markers and exact state/cache assertions.
Raw runs, runtime/library snapshots, models, data and checkpoints remain
outside Git. Source progress is published; fully integrated runtime worker
worktrees are archived/removed, with verified complete-history bundles and
40 hashed CPU logs under `.git/goal-worktree-archives/`.
