# Released DSpark native admission preparation

Owner: bounded native-admission worker. Parent worktree:
`/private/tmp/eagle-dspark-native-admission`; native worktree:
`/private/tmp/llama-dspark-native-admission`. Other research worktrees and active
A8 ownership are untouched. Sole GPU operator owns actual SM75 execution.

## Checkpoint and acceptance

Native commit `fcdf5822c5b78f9dbcfd1f5c7106f09c3f0c9b1a` (base
`76847aa877261817026e2f29472f080235b6a829`, predecessors
`cbc50bc736a47d911a095bb6a26ab1544d68d3e0` and
`a4884f5dd15f7dcdd2d9c16a8d754467a776dfb3`) is published at
`https://github.com/philippesic/llama.cpp.git`, branch `feature/dspark-admission`.
Base is `9e2c7a90051e738751aab7d7bd7c2d8201fb76e3`. Release AppleClang 21 CPU
`llama-server` build passes on Darwin arm64; this proves compilation only.
Ten evidence-validator tests pass; no released model or SM75 result is claimed.
Remaining: actual source-copy comparison/export, residency/dispatch, native
trajectories, and same-device measurements under the root's frozen protocol.

Follow-up checkpoint: `probe.py` now performs a supervised five-cell admission
transaction without consuming an existing admission. Twenty-two focused tests
pass, including complete mocked transaction, startup failure, cancellation in
the spawn/receipt window, unsafe borrowing, supervisor grace, and tied-target
source guards. These fixtures do not establish actual model/GPU readiness.

Deliverable is executable export/evidence checking plus opt-in runtime state and
graph diagnostics. Acceptance requires full released-source/target tensor
comparisons, both BF16 exports, actual four DSpark/DFlash length cells, unchanged
target hashes/identities, clean-cache reinjection, author-layout evidence and
target-only exact output IDs. `validate_native.py` refuses missing evidence.

## Source audit

Both pinned releases have architecture `Qwen3DSparkModel`, block7, BF16 weights,
five layers, full vocabulary151936, H2560, query width4096, KV width1024,
FFN9728. The author source is DeepSpec
`005e03b81cec38b7da6399833d609ee89a2587f2`. Native `conversion/qwen.py` converts
author block taps `[1,9,17,25,33]` to input taps `[2,10,18,26,34]`, and writes
`dflash.sample_from_anchor=true` for both releases. Rank-zero DFlash must use
explicit `--spec-type draft-dspark --spec-draft-p-min 0`; automatic DFlash skips
slot zero. The DSpark Markov chain and driver are greedy; temperature alone
cannot reproduce stochastic author conditioning.

`DSPARK_REQUIRE_AUTHOR_LAYOUT=1` gates the fixed author conventions and computes
all seven noise slots for both proposal lengths. Short mode reads/proposes slots
0..2; maximum reads/proposes0..6. Mask rows remain bidirectionally visible; admission records the actual populated
mask for all seven query rows, visible noise key positions and last visible
clean key. The validator rejects missing future noise or clean anchor visibility. The
existing server truncates all disposable draft noise after the proposal, then
injects native target features and removes rejected target/draft suffixes.
The next noise event asserts `draft_kv_max < pending_anchor_position`.
Keep all prompts at least seven tokens short of context2048; no near-boundary
or context-shift behavior is admitted by this screen.

The target/verifier sampler remains unchanged. It can sample a verified tail
past an accepted EOS before the server emits only through EOS. Round records
preserve `n_accepted` and add `n_accepted_verified`, `verified_token_ids` and
`n_accepted_usable_prefix=min(n_accepted,n_emitted)`. Do not count an unconsumed
correction/bonus beyond EOS as output. Admission reports which zero/partial/full
acceptance and EOS cases were actually observed; unobserved branches retain
source/fixture evidence, rather than being described as actual model proof.

## Frozen copies and residency

The default native converter omits the full output head but retains embedding
unless `has_embed_tokens=false`. `compare_frozen.py` compares both original BF16
copies with the fixed F16 target in canonical little-endian row-major FP32,
including full hashes, exact shapes/row order, nonfinite/underflow/overflow,
different elements and maximum absolute difference. It never edits the target.
Only its optional copied conversion config sets borrowing for exactly equivalent
copies; unequal original copies are retained (`retain_lm_head=true` is the small
new converter flag). `check_export.py` verifies retained copies byte-for-byte
against the released source and rejects omitted unequal copies.

Header-derived raw tensor payloads: DSpark2,786,267,138 bytes and
DFlash2,630,679,040 bytes. Each full embedding/head is777,912,320 bytes.
Incremental native tensors if both copies can safely be borrowed are
DSpark1,230,442,498 bytes and DFlash1,074,854,400 bytes, before GGUF alignment,
small F32 norm widening, KV, graph scratch, backend staging and allocator reserve.
If either canonical comparison fails, retain the copy and redo actual residency;
do not silently lower target precision or change the verifier. These are storage
counts, not throughput or a claim of11GB fit. Native graph borrowing uses the
unchanged target tensor pointer; admission-only device hashes before and after
the complete binding lifetime verify target bytes and identities.

## Commands

Use a numpy environment with the selected native `gguf-py`. Preserve immutable
original release directories; put the copied conversion config in a separate
conversion-input directory referencing the pinned weight file.

```sh
python3 scripts/dspark_screen/compare_frozen.py \
  --source RELEASE/model.safetensors --target FIXED_TARGET_F16.gguf \
  --llama NATIVE --output canonical.json --config RELEASE/config.json \
  --target-config FIXED_TARGET_TOKENIZER/config.json \
  --target-config-sha256 PINNED_TARGET_CONFIG_SHA256 \
  --conversion-config-out CONVERSION_INPUT/config.json
python3 NATIVE/convert_hf_to_gguf.py CONVERSION_INPUT \
  --target-model-dir FIXED_TARGET_TOKENIZER --outtype bf16 --outfile DRAFT.gguf
python3 scripts/dspark_screen/check_export.py \
  --source RELEASE/model.safetensors --target FIXED_TARGET_F16.gguf \
  --draft DRAFT.gguf --config CONVERSION_INPUT/config.json --llama NATIVE \
  --comparison canonical.json --output export.json
```

Actual server admission command, dispatched only by the sole operator inside
detached Linux tmux/`remote_job.py`; substitute isolated absolute paths:

```sh
DSPARK_REQUIRE_AUTHOR_LAYOUT=1 DSPARK_ADMISSION_JSONL=state.jsonl \
W1AX_ROUND_TRACE_JSONL=rounds.jsonl \
llama-server -m FIXED_TARGET_F16.gguf -md DRAFT.gguf \
  --n-gpu-layers all --spec-draft-ngl all --ctx-size 2048 --parallel 1 \
  --batch-size 32 --ubatch-size 32 \
  --fit off --cache-type-k f16 --cache-type-v f16 \
  --spec-draft-type-k f16 --spec-draft-type-v f16 \
  --spec-type draft-dspark --spec-draft-n-max 3 --spec-draft-p-min 0 \
  --jinja --metrics --perf -lv 4 --host 127.0.0.1 --port 18290
```

Repeat with7 and both draft files. Use root's existing HTTP helpers with frozen
non-thinking greedy settings and the same prompts as target-only. Preserve
raw request/response/measurement JSON. Stop servers gracefully so binding_end
is emitted; operator must separately verify owned groups/contexts are gone.
Same-prefix first-three native proposal IDs must match3 versus7 for each model.

`validate_native.py MANIFEST.json admission.json` consumes this manifest:

```json
{
  "target": "/absolute/fixed-target.gguf",
  "binary": "/absolute/llama-server", "binary_sha256": "...",
  "environment": "/absolute/environment.json",
  "cells": [{
    "kind": "dspark", "maximum": 3, "draft": "/absolute/draft.gguf",
    "source": "/absolute/model.safetensors", "conversion_config": "/absolute/config.json",
    "export": "/absolute/export.json", "launch": "/absolute/launch.json",
    "server_log": "/absolute/server.log", "state": "/absolute/state.jsonl",
    "rounds": "/absolute/rounds.jsonl",
    "outputs": [{"actual": "/absolute/candidate/measurement.json",
                 "reference": "/absolute/target-only/measurement.json",
                 "prompt": "/absolute/request.json", "prompt_sha256": "..."}]
  }]
}
```

All four `(kind,maximum)` pairs are required. Environment uses
`benchmark_native_eagle.environment_manifest`; launch uses root harness
`command`/`trace_env` fields. Prompt hash binds the exact frozen request file.
The final admission flags are computed from evidence, not manually supplied.

## Timing interpretation

Generic round CPU-wall detail now includes DSpark/DFlash feature copies,
injection/noise decode call spans and sampler costs. These detail spans add no
extra synchronization by default; GPU completion can fall inside sampler wall
time. Set `DSPARK_SYNC_COMPONENT_TIMINGS=1` only in a separate diagnostic to
measure synchronized component wall spans. `draft_seed_decode_us` is the complete block;
`draft_step_decode_us` and `draft_sampler_us` are arrays. These nest inside outer
round spans. Regular round trace adds CPU serialization and must be disclosed as a traced
rate; compare a bounded trace-off repeat if it materially changes rates. No
extra DSpark component synchronization is enabled in throughput runs.

Separately set `DSPARK_GRAPH_PROFILE_JSONL` for graph chunk diagnostics. It
splits execution and synchronizes at named boundaries, copies no tensors, and
reports CPU wall, not CUDA event/kernel latency. Body is `inp_noise_embd` to
`result_norm`, head is `result_norm` to `result_output`, Markov/confidence is
`result_output` to `dspark_markov_output`; injection is feature fusion plus
`inp_g_embeddings` to `dspark_injection_end`. Do not use these profiled requests
as uninstrumented throughput. The BF16 SM75 CUDA dispatch and any scratch
conversion remain actual-hardware checks; source support alone is not proof.

## Executable supervised admission transaction

The fixed target is tied: independently inspected target HF config SHA256
`8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a`
declares `Qwen3ForCausalLM` and `tie_word_embeddings=true`, and its GGUF has only
F16 `token_embd.weight`, with no separate `output.weight`. The first comparison
failed on that missing name and is preserved by the operator. Guarded comparison
now requires the pinned config and selected native Qwen3 loader's actual fallback
before using `token_embd.weight` for the target head role. Released embedding
and head are still compared independently; no equality is inferred from tying.
Canonical/export reports name this source alias. Native `TENSOR_DUPLICATED` can
use separate objects/buffers, so admission retains each original role's identity
and requires equal tied-source hashes/type/size, not equal role pointers.

`probe.py` depends on root's reviewed `stop_owned_server(proc, grace_s=3)` helper.
It refuses an older helper before model launch. The probe requests fifteen
seconds for graceful server exit, then the helper has a bounded three-second
kill/wait fallback, so full device hash teardown can produce `binding_end`.
Launch with caller `remote_job.py --stop-grace-seconds 30`; the probe reads that
actual supervisor receipt from `TMPDIR/../state.json` and rejects a shorter
grace. The throughput harness keeps its original three-second default.

```sh
python3 scripts/remote_job.py UNIQUE_ADMISSION_RUN --stop-grace-seconds 30 -- \
  python3 scripts/dspark_screen/probe.py /absolute/probe-config.json \
  /absolute/results/UNIQUE_ADMISSION_RUN
```

The sole GPU operator dispatches this under detached Linux tmux with current
hardware/ownership proof. No helper connects to a GPU host or resumes ownership.
The probe checks fresh single-device RTX2080Ti/SM75 capability before launch,
then runs target-only, DSpark3/7 and author DFlash3/7. Every cell has two warmups,
the first two frozen development prompts, and a separate one-word YES/EOS case,
all at128 output tokens, context2048, F16KV, seed42, temperature0 and non-thinking.
It preserves exact requests/responses/measurements, immediate owned PID/PGID,
startup/GPU snapshots, complete state/round traces and the computed validator
manifest. It verifies EOS emission on the candidate fixture. Context reserve
must cover the128-token output cap and all seven noise positions. Component
sync and profiling environments are removed. After every owned server is
terminal, it computes `admission.json` from `validate_native`; failures preserve
`failure.json` and never create an admission receipt. Operator must still verify
exact groups/native contexts absent before releasing the GPU.

Probe config adds pinned source/conversion/canonical/export assets to each model;
every path must be absolute. The same schema applies to `dflash`:

```json
{
  "binary": {"path": "/absolute/llama-server", "sha256": "..."},
  "target": {"path": "/absolute/fixed-target.gguf", "sha256": "..."},
  "protocol_sha256": "hash of configs/dspark-screen/protocol.json",
  "port": 18290,
  "dspark": {
    "path": "/absolute/dspark-bf16.gguf", "sha256": "...",
    "source": {"path": "/absolute/dspark/model.safetensors", "sha256": "..."},
    "conversion_config": {"path": "/absolute/dspark-conversion/config.json", "sha256": "..."},
    "canonical": {"path": "/absolute/dspark-canonical.json", "sha256": "..."},
    "export": {"path": "/absolute/dspark-export.json", "sha256": "..."}
  },
  "dflash": {
    "path": "/absolute/dflash-bf16.gguf", "sha256": "...",
    "source": {"path": "/absolute/dflash/model.safetensors", "sha256": "..."},
    "conversion_config": {"path": "/absolute/dflash-conversion/config.json", "sha256": "..."},
    "canonical": {"path": "/absolute/dflash-canonical.json", "sha256": "..."},
    "export": {"path": "/absolute/dflash-export.json", "sha256": "..."}
  }
}
```

An explicit `protocol: {path,sha256}` asset can replace `protocol_sha256` when
the frozen file is staged elsewhere; `--project-root` still binds its relative
prompt file and root benchmark helper imports. No pre-existing `admission` entry
is read. This is an admission transaction, not the five-repeat throughput study.

CUDA-event component timing uses the existing generic runtime facility in a
separate tiny pass: `GGML_CUDA_EAGLE_EVENTS=1`,
`GGML_CUDA_EAGLE_EVENT_LIMIT=100000`, `GGML_CUDA_DISABLE_GRAPHS=1`.
Retain `CUDA_EAGLE_EVENT` JSON from stderr and join `model_arch=dflash`, actual
`llama_context`, graph frame and node order. Stage annotations currently call
DFlash `target_or_other`; use architecture/context rather than that label alone.
Follow-up native commit adds named Markov lookup/projection/add/argmax and
confidence projection nodes, plus feature fusion. Full-head MUL_MAT is named
`result_output`; serial factor projections are `dspark_markov_projection-i`.
Classify intermediate unnamed Markov nodes by the node range after the full-head
boundary through `dspark_markov_output`. Injection has n_outputs0 and feature
fusion/Kcur_injected/Vcur_injected/final injection markers. Report graph intervals
and node intervals separately or union same-stream intervals; never add graph
parents, node children and transfer grandchildren as independent costs. These
synchronized CUDA events include stream idle and change execution, so serving
rates come only from the separately measured reference run.

Reference-runtime recovery now reserves seven draft outputs for author mode3,
retains the original eight-output reservation for mode7, and records actual
context capacities. Root froze common batch/ubatch32 for all study arms before
measurements after the old512-microbatch DFlash scratch failure. The validator
requires nonzero/full-length actual drafting cells; explicit no-noise zero-cap
boundaries remain separate. See the durable
[recovery report](../../experiments/dspark-sm75-20261003/native-runtime-admission.md)
for preserved failures, seven native CPU reservation checks, thirteen validator
checks, and remaining actual SM75 admission. This does not change target
precision/verifier or start W1Ax integration.
