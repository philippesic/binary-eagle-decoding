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
validate SM120/SM75 dispatch or graphs.

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
with `--selector kv_only`; shared packing requires a pinned source config with an
explicit packed A1/A4/A8 variant (`--selector shared_pack`), as candidate D/A16 has
no activation-pack stage. These are preparation commands, not GPU jobs started by
the engineering worker. Preserve model hashes, precision, proposal cap, confidence,
sampling, context, batch/KV settings and unassigned timing.

## Remaining engineering

Shared activation pack/scales is published and awaits GPU validation. A16 retains
the inside-kernel F16 cast. A bounded resident
recurrent state path and fine-grained event/node measurement remain to implement.
Archived SM75 profiles measured large-N=37 head XOR/POPC dot at approximately
885–1,187 us versus pack at approximately 5–9 us. They do not measure launch gaps
or attribute current pilot draft/process/request costs. Reduction/launch cleanup
must retain that distinction and await measured GPU A/B before any benefit claim.
