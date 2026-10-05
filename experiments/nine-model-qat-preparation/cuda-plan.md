# Nine model preparation CUDA operator plan

Operator: `/root/cuda_operator` (task `01a10903-1c7a-71b1-abb1-0de3ecc046b8`).
Latest status: the operator used RTX5080 briefly for authorized development
admission and CPU-only dependency setup. The human then paused all RTX5080 use
and directed Mac-only continuation. No CUDA compilation, CUDA test, model load,
TRAIN capture, real-model backward, optimizer update, training, or evaluation
occurred. No further remote action is planned while paused. See the
[authoritative pause checkpoint](../../docs/goals/nine-model-qat-preparation.md#current-hardware-boundary-mac-only).

## Historical startup gate — before direct RTX5080 authorization

The shared registry has complete entries for `rtx2080ti` and `rtx5080` at
`~/.config/binary-eagle-decoding/hosts.toml`. It was inspected locally without
printing host values. `agent_env.py status` reports `rtx2080ti.pause_requested`
false, last updated `2026-10-04T00:14:22Z`; this stale flag is not availability.
At this initial checkpoint RTX5080 was paused (`pause_requested=true`, last
updated `2026-10-04T22:09:29Z`) and had not been queried. This paragraph records
that earlier boundary; the later human authorization and direct pause are
recorded below. RTX2080Ti remains unqueried by this operator.

The project tree at dispatch was `main` / `da097e5cfadbf6c783336787007670284f0cab2d`.
The only worktree owned here is `/private/tmp/nine-model-qat-20261004/cuda-operator`,
branch `prep/nine-model-cuda-operator`. Source and test changes belong to their
feature owners; this worktree owns this plan and subsequent CUDA evidence only.

## Evidence available before this task

The prior source-bound operator record is
[`experiments/dspark-sm75-20261003/operator.md`](../dspark-sm75-20261003/operator.md),
with final outcomes in
[`results.md`](../dspark-sm75-20261003/results.md) and
[`checkpoint.md`](../dspark-sm75-20261003/checkpoint.md). It establishes an
RTX 2080 Ti / SM75 native reference and selective FFN-Q4 study, CUDA 12.8.93
toolchain, target and DSpark/DFlash model artifacts, and clean release after
those runs. Those source pins predate the nine-model W1 implementation; do not
reuse their admission as W1 evidence. The prior suite found that the fixed
2048/512 batch/ubatch pair could exhaust device memory. Root's later 32/32
setting resolved that specific admission problem. Keep context 2048 and batch
and ubatch 32 for initial preparation checks; do not raise them to optimize a
result.

Previously observed capacity was tight: 11,264 MiB total, roughly 10.5 GiB
loaded for released DSpark and 10.4 GiB for DFlash in the fixed native probe,
with substantial unaccounted device use. Those are historical observations,
not a fresh baseline or proof of headroom. Load one model/arm at a time and
record external display/context memory in addition to the process breakdown.

The historical artifact inventory names the fixed target as
`models/gguf/Qwen3-4B-f16.gguf` (SHA256
`05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`) and the
released model safetensors under the old remote checkout's
`runs/dspark-sm75-release-fetch-20261003/{dspark,dflash}_model.safetensors`.
Their pinned upstream revisions and source hashes are in the prior operator
record. On the device, these are revalidation leads only: verify files and
hashes against the selected current project checkout before use. The current
W1 candidates, updated native binary, capture outputs, and five-tap full-vocab
TRAIN teachers are not established by that old inventory.

The older main checkout's block-0 helper
[`scripts/native_target_block0_capture.cpp`](../../scripts/native_target_block0_capture.cpp)
captures a bounded 29-token prefix and selected layers 0/14; by itself it does
not provide five-tap teachers. The current locally integrated source snapshot
is local integration snapshot main `3cacf482c8b55f6b0d8a5979de3ec85098775e50` with llama.cpp
`50ca2676fb6fbc6a8455e7bce66a614f84e7016c`. It includes
`llama-block-teacher`, `scripts/capture_block_qat_teacher.py`, and the
high-level capture planner. The low-level teacher API accepts
the target path/hash, producer source revision, max tokens, GPU layer count,
output root, and a JSONL request file. Each request binds exact caller tokens,
five ordered tap IDs, `logits_mode`=`all` or `last`, and optional chain ancestry.
The helper writes F32 features shaped `[tokens,5,hidden]` and full-vocabulary
F32 logits for all tokens or only the final row. The
`block_native_teacher_request_v1` receipt binds the exact tokens/taps and
records zero optimizer updates. Source conversion fixes the ordered native
target taps as `[2,10,18,26,34]` (the released block IDs `[1,9,17,25,33]`
plus the conversion's input-layer offset); capture requests must use this exact
sequence. The current block checker requires an exact `CUDA<n>` output-buffer
name for every selected operator. `NativeTeacher.generate_capture` now binds
original native tokenization/generation and enforces the 512-token prompt cap
before decode. The higher-level TRAIN capture runner is
[`scripts/capture_nine_model_train_data.py`](../../scripts/capture_nine_model_train_data.py).
No actual CUDA capture has run. The nine-row development selection and capture
caps are recorded below; only a source/config-bound plan-only receipt can admit
execution. Existing EAGLE W1 data and 32k masks are not substitutes.

## Local artifact check and source-derived resource budget

At this task's local checkout, these ignored/model artifacts are absent:
`models/gguf/Qwen3-4B-f16.gguf`, the Q4_0 and F16 EAGLE drafts,
`models/hf/Qwen3-4B/config.json`, `data/continuous-w1ax/launch-inputs.tar`,
`data/continuous-w1ax/freeze-002/`, and the historical DSpark/DFlash release
safetensors under `runs/`. I did not search other hosts or mount points. The
old remote report lists model paths and pinned hashes; it does not establish
that those files still exist on 2080Ti. The checked-in
[`released_inventory.json`](../../scripts/dspark_screen/released_inventory.json)
pins upstream model revisions, expected sizes, and header/metadata hashes; its
own source field says the weights were not locally full-file hashed there.
Therefore fresh remote presence and content hashes are a preflight dependency.

The following are byte-count estimates, not GPU measurements. They use the
source-pinned five-layer dimensions in the prior operator record: gate/up
`[9728,2560]` and down `[2560,9728]` for each of five layers, fifteen matrices
and 373,555,200 parameters total. The exported BF16 tensor inventory and
target file sizes are historical actual artifact measurements.

| Allocation or artifact | Estimate | Basis and limitation |
| --- | ---: | --- |
| Fifteen FFN latent weights in FP32 | 1,494,220,800 B / 1.392 GiB | 4 bytes per selected parameter; assumes one trainable FP32 latent per weight. |
| Corresponding FP32 gradients | 1,494,220,800 B / 1.392 GiB | Same tensor count; does not include backward activations or workspaces. |
| Adam first/second moments, FP32 | 2,988,441,600 B / 2.783 GiB | Two additional arrays; **zero real optimizer updates means these must not be allocated for real-model smoke**. Test moments only on synthetic fixtures. |
| Original BF16 storage of selected FFNs | 747,110,400 B / 0.696 GiB | Exact fifteen-matrix count. Whether training keeps these alongside latent weights is implementation-dependent. |
| Packed W1 signs plus F32 row scales | 47,134,720 B / 0.044 GiB | 46,694,400 sign bytes plus 440,320 scale bytes. This is export size, not training-state memory. |
| Released DSpark draft GGUF tensor inventory | 2,786,331,140 B / 2.595 GiB | Historical export audit; includes private embedding/head and non-FFN tensors. Not a CUDA residency measurement. |
| Released DFlash draft GGUF tensor inventory | 2,630,743,040 B / 2.450 GiB | Same limitation. |
| Private BF16 embedding or full head | 777,912,320 B / 741.7 MiB each | `[151936,2560]`, two bytes/value, from prior source inventory. Both are included in each complete draft inventory above and remain private. |
| Frozen target GGUF file | 8,051,285,280 B / 7.498 GiB | Historical file size. Prior native target-only load reached 8,511 MiB, which includes runtime allocations. |
| Five F32 native tap rows per token | 51,200 B / 50 KiB per token | Pending helper shape `[tokens,5,2560]`; the five semantic tap IDs remain a data-owner contract. |
| Full-vocabulary F32 logits per token | 607,744 B / 0.580 MiB per token | Pending helper's `all` mode and vocabulary 151,936. |
| F32 features plus all logits | 658,944 B / 0.628 MiB per token | Raw helper files, excluding metadata/filesystem overhead. 64 rows: 42,172,416 B / 40.22 MiB. One million rows: 613.7 GiB. |
| Native output arrays for a 256-token batch | 168,689,664 B / 160.88 MiB | Geometry implied by `n_batch=n_ubatch=256`: logits 148.38 MiB and features 12.5 MiB. Array placement and overlap with other buffers require measurement. |

A conservative byte subtotal for a real block draft with all released BF16
tensors, FFN FP32 latent weights and FP32 FFN gradients is about 5.38 GiB
before activations, temporary copies, CUDA context, fragmentation, fusion
parameters, and allocator reserve. Adding the historical 7.50 GiB target file
gives a naive 12.88 GiB file-and-tensor sum, above the 11,264 MiB card; this is
an infeasibility warning, not a residency prediction. The target and
training drafter must therefore be staged: capture target-only teachers,
finish and close the target process/context, then load cached teacher shards
and run real-draft forward/backward without loading target weights. A separate
native integration smoke may need target plus candidate; run it as a distinct,
short single-arm admission with fresh memory inventory and immediate teardown,
not co-loaded with training allocations. The historical native candidate peak
around 10.5 GiB leaves little margin and is not a QAT/backward memory pass.

Full-vocabulary rows are a material storage constraint. The pending helper
currently writes F32 only. `all` emits 658,944 bytes per token row, including
all five taps; `last` still writes features for every input token and one final
full-vocabulary logit row. Its 256-token decode chunk implies about 160.88 MiB
of logits and feature output arrays before runtime scratch, with actual buffer
placement pending GPU measurement. A maximum-size 32,768-token `all` request
would emit about 20.11 GiB of raw F32 output in one request directory; do not
set that limit by default. Precompute bytes from the exact request file, bound
each request/shard to the reviewed storage cap, and split only where exact
prefix semantics remain valid. BF16 conversion could reduce durable size but
is not an admitted format and must not replace raw captures silently. The data
owner must declare required rows, output mode, converted format if any, shard
limit, and label semantics. Capture only required TRAIN rows and measure actual
total bytes before admitting a complete corpus.

## Queued 2080Ti work, in order

All items below remain PENDING until the human opens 2080Ti and root assigns the
serialized slot. Stop before any step if the local pause flag turns true, the
remote evidence identifies another team's job, the measured resource floor
fails, or the exact source/artifact binding is absent. Preserve the refusal and
do not alter another team's processes or pause state.

1. **Fresh admission and ownership.** Use the shared registry only, through
   tmux MCP for SSH. Verify host identity, RTX 2080 Ti UUID/SM75, driver/CUDA,
   memory baseline, active process/context inventory, project supervisor
   sessions, working tree and relevant artifact hashes. Confirm there is no
   concurrent project GPU owner. Record producer GPU details. Do not write the
   address or credentials into this report.
2. **Source and build binding.** Use the reviewed/published feature-owner
   revisions and current parent gitlinks. Confirm clean immutable checkouts,
   pinned target/draft/source hashes, exact W1 schema and no ambient experimental
   flags. Rebuild only when changed native source affects the CUDA binary;
   retain build logs and binary hashes under ignored `runs/`.
3. **Synthetic operator/export/load/graph contracts.** Exercise W1A8 and W1A1
   with synthetic tensors for all declared shapes/tails/scales, serialization,
   corrupt/missing/duplicate tensor rejection, dispatch coverage, and explicit
   no-dense-fallback behavior. Test DSpark and DFlash separately, with Q4
   controls unchanged. Capture graph/operator receipts, not throughput claims.
4. **Synthetic optimizer and state fixtures.** Run bounded optimizer/update,
   save/restore RNG/cursor, checkpoint/export and A8-to-A1 transition fixtures.
   These may update synthetic parameters. Verify finite/nonzero gradients and
   resumed-state identity with production APIs; do not load/update a real draft
   optimizer here.
5. **Actual-model native smoke.** Sequentially load one DSpark then DFlash
   candidate in the reviewed native binary with the fixed target. Exercise
   W1A8/W1A1 pack/dispatch/graph and bounded forward/backward on real model
   tensors with **zero real-model optimizer updates**. Check target/teacher
   storage ownership, shapes, masks, injection, K/V/later-state gradients,
   outputs and peak process plus total device memory. Tear down completely and
   verify process group/context/resource return before the next arm. This is
   not candidate quality or held-out evaluation.
6. **TRAIN target-only teacher capture, only after its artifact contract is
   ready.** Capture just the approved TRAIN prompts with the frozen target and
   exact native prefix/trajectory semantics. Record target GGUF/native binary
   hashes, host GPU/driver/CUDA, tokenizer, prompt/split hash, ordered five tap
   names/shapes/dtypes, full-vocabulary label/distribution representation,
   mask/position/slot ancestry, prefix boundaries, capture software revision,
   row/prompt counts and output hash. Bound shard bytes and free host/GPU
   buffers between shards. Validate disjoint split membership and all ancestry
   before publishing the capture manifest. No dev, reserve or sealed-final
   payloads; no drafter scoring, optimizer updates or model comparison.
7. **Resource closeout.** Run all remote work under the project supervisor in
   a uniquely named detached Linux tmux session, socket
   `binary-eagle-runtime`; preserve exact run directory, command, environment,
   source/artifact hashes, state/stdout and child/supervisor PIDs/PGIDs. After
   each run, verify the supervisor and child groups absent, native contexts
   absent, and device memory returned to the fresh baseline within the observed
   display variation. Record any persistent external context separately. Close
   only this operator's tmux MCP transport. A pause request means terminate,
   verify release, checkpoint, and stop.

No timing, recipe selection, quality benchmark, held-out/final evaluation,
long QAT, or real-model optimizer update is in this queue. Those are reserved
for RTX5080 after human availability and campaign choices.

## Serialized execution/staging checklist for feature APIs

When the native/data/training owners publish and tests cover their interfaces,
resolve their immutable commits before occupying the device. Run this order;
each numbered GPU stage is a separate supervised run and must return to the
fresh baseline before the next one:

1. **Artifact and storage preflight (read-only):** under the registered project
   workdir, verify target, base draft, current model snapshots, native binary,
   tokenizer, prompt manifest, source lock, and configuration hashes. Print
   `nvidia-smi`, GPU UUID/SM version, free/used memory, compute applications,
   project process groups, and pause status to that run's evidence. Abort if
   another team owns the card or model/data hashes do not match. This is a new
   per-use check even when old inventories pass.
2. **Synthetic kernel contract:** for each admitted A8/A1 schema and each
   architecture, launch deterministic synthetic matrices spanning every real
   shape and tail rule. Assert bit packing, scales, output/error contracts,
   CUDA dispatch receipt, finite input/output/gradients, no dense fallback,
   and rejection of absent/duplicate/wrong dtype/shape/tail/coverage metadata.
   Compare forward and backward to the independently authored high-precision
   reference on small tensors. Although the GGML op table looks lowercase, the
   executed test filter is `test-backend-ops -b CUDA0 -o W1A1_MUL_MAT`. A
   lowercase regex returned zero tests despite exit 0; require a positive test
   count and both bit modes.
   The native fork now includes K=9728 cases. Its feature-owner CPU run passed
   260/260 cases over bit modes 1/8, one/seven rows, shared/direct, and K
   values including 2560, 4096, 7680, 9728 and 12800. This is CPU-only evidence;
   review source and retain the corresponding CUDA backend result after the
   2080 slot opens. This may exercise optimizer state only with synthetic
   parameters.
   Keep all outputs in its ignored run directory.
3. **Native EAGLE W1 CUDA smoke:** run each admitted activation profile
   sequentially using a current immutable config bound to binary, model,
   target, export audit, authenticated TRAIN prompt and capture. Set
   `hardware: "rtx5080"`, one TRAIN prompt, context to the reviewed profile,
   max output eight, and zero optimizer updates. Its pending invocation is
   `python3 scripts/check_eagle_binary_native.py --config RUN/eagle-native-smoke.json --receipt RUN/eagle-native-smoke.receipt.json`.
   Require all nine loader groups and the corresponding W1A8 INT8 or W1A1
   XOR/POPCOUNT CUDA marker, no dense fallback, exact artifact hashes, and
   resource release. This is a loader/graph smoke, not quality or throughput.
4. **Native block loader/graph:** serialize a single DSpark arm, then DFlash,
   each with only the candidate GGUF and synthetic anchor/noise inputs in the
   native block test. Verify selected W1 tensors and every declared floating
   exception, graph nodes, dispatch receipts, output shapes, masks/slots,
   target identity and absence of dense compute fallback. Record peak/end
   memory; shut down and verify its process group and compute context are gone.
   The pending checker invocation is
   `python3 scripts/check_block_binary_native.py --binary BUILD/bin/test-block-binary --model CANDIDATE.gguf --export EXPORT.json --gpu-layers 99 --require-cuda --receipt RUN/admission.json`.
   Resolve each placeholder to the immutable built source, model/export, and a
   new ignored run path before launch. Current integrated source requires every
   selected node's observed output buffer to exactly match `CUDA<device-index>`.
5. **Target-only TRAIN capture:** after the native integration run has fully
   released, first run the integrated capture planner in CPU plan-only mode:
   `python3 scripts/capture_nine_model_train_data.py --plan INPUT/plan.json --plan-sha256 PLAN_SHA --output-root RUN/plan-only`.
   Require a passed bounded cost/inventory report and verify original source
   row/content/domain joins, balanced TRAIN-derived roles, tokenizer/template
   pins, `[2,10,18,26,34]` taps, output mode, and storage/RSS/free-disk/wall caps.
   Only then may the sole operator launch its `--execute` mode under
   `remote_job.py`, with target-only CUDA and expected capability `[12,0]`.
   The execute invocation uses the same plan/hash and a fresh output path:
   `python3 scripts/capture_nine_model_train_data.py --plan INPUT/plan.json --plan-sha256 PLAN_SHA --output-root RUN/capture --execute`.
   It owns native target-greedy generation and teacher capture; no drafter or
   optimizer is loaded. The data owner must select `hard_ce` (features, no
   block logits) or `exact_soft` (all F32 full-vocabulary rows) in the plan.
   Preserve every F32 shard/receipt in the ignored run, verify hashes, byte
   caps, and ancestry, then close the target and prove process, CUDA-context,
   and memory release. Do not hold target tensors in the drafter process.
6. **Real drafter forward/backward smoke:** load one architecture/candidate at
   a time with target teachers read from the verified shards. Run forward and
   backward only; assert optimizer step count stays zero, optimizer state is
   empty, and target tensor ownership is absent from the process. Record peak
   GPU and host RAM, activation/checkpoint mode, rows, loss and all gradient
   checks. Do not call this convergence, quality, or optimizer admission.
7. **Resume and transition fixtures:** run exact save/load, RNG/cursor and
   A8-to-A1 transition using synthetic parameters and production checkpoint
   APIs. Verify checksum equality and next-batch identity. This is a distinct
   synthetic stage; never update the real draft to validate resume.
8. **Final closeout:** after every stage, capture supervisor state and all
   process identities, wait for clean exit, verify no owned PIDs/PGIDs/compute
   contexts remain and memory is back to the measured baseline, then make the
   next stage eligible. STOP/pause/error paths preserve checkpoints/logs and
   receive the same cleanup proof. Store raw artifacts in ignored per-run
   directories; reference hashes and exact command/environment here.

The first pending `NativeTeacher` revision used a separate process group, which
would have made supervisor STOP unsafe. Parent commit `44d1db6` makes the C++
teacher inherit the supervised Python PGID and publishes producer
PID/PGID/parent PID. The feature owner's CPU fixture
`test_supervisor_group_stop_reaps_actual_native_teacher` passes: SIGTERM to the
shared process group exits the producer and leaves its PID/PGID absent. This
proves the wrapper's CPU process-group contract, not remote GPU resource
release. Keep remote STOP/closeout pending until a supervised 2080 run confirms
both processes, GPU contexts, and memory release.

**Remaining invocation values are deliberate:** the native checker and teacher
wrapper entrypoints are known on pending source, but immutable model/export
paths, exporter outputs, semantic tap IDs, prompt rows, token bounds, storage
caps, and the training-owner smoke command are unresolved. Build each run from
the integrated commit and exact owner manifests. No guessed CLI values or
unbounded capture are authorized.

## Historical 2080 launch form — inactive pending a separate authorization

This applies only if the human later opens 2080 and root confirms the slot; it
is inactive during the current Mac-only pause. A fresh admission is still
required.
Read the exact host alias and workdir via `agent_env.py`; create the MCP tmux
transport using the tmux MCP tools, then execute this shape on the WSL host:

```sh
tmux -L binary-eagle-runtime new-session -d \
  -s nineprep-2080-<check>-<utc-id> -c "$PWD" \
  'exec python3 scripts/remote_job.py nineprep-2080-<check>-<utc-id> -- <pinned-command> <args>'
```

The literal check name, UTC ID, command and arguments must be frozen in the
run's ignored config before launch; do not paste placeholders. Keep every
artifact in a unique `runs/nineprep-2080-<run-id>/` directory under the
registered project workdir. The remote supervisor, not the MCP transport,
owns the child process group.

## Requirement status at preparation dispatch

| Requirement | State | Evidence or dependency |
| --- | --- | --- |
| Registry lookup and no-address handling | PASS | Local status inspection; no address persisted. |
| RTX2080Ti current availability | PENDING | Human has not announced open; unpaused flag is stale. No connection made. |
| RTX5080 development availability | PAUSED | The human subsequently said “pause all 5080 usage continue mac only”; root set the local pause flag. Do not query/build/stage/run until renewed human authorization. |
| RTX5080 initial resource state | PENDING repeated baseline | 13,186 MiB free; compute-app and DXG-holder lists empty, but GPU showed P0/2% and 2,792 MiB used. Confirm stability and expected SM120 before launch. |
| Local/remote artifact availability | PENDING | Local nine-row selection packet exists; target/Q4/F16 EAGLE files exist remotely by size; DSpark/DFlash draft artifacts are absent. Verify hashes and stage into a distinct checkout/run directory. |
| RTX5080 scope isolation | PASS | Only registered RTX5080 was queried through tmux MCP. No 2080 action, job, or flag change. |
| Source checkout staging | PARTIAL / preserve | Dedicated clone b998/native874 was verified clean. A source-freeze update toward main658 was attempted but produced no supervisor state/receipt and was not rechecked after the pause. Current local integration main is `3cacf482` / native `50ca2676`; the remote clone's post-attempt state is unverified and must not be used until reauthorized. |
| Locked project environment | PASS (CPU setup) | Supervised `uv sync --locked --group w1a1` exit0 using byte-matched lockfiles; CPython3.11.15, CMake3.31.10, Ninja1.13.2, Torch2.14.0. No CUDA context used. |
| NVCC/GCC host-compiler compatibility | PENDING | NVCC13.1.115 with GCC15.2.0 was inventoried; the authorized compiler-only probe was not started before human pause. No `--allow-unsupported-compiler` used. |
| DSpark/DFlash native baseline and prior SM75 discipline | PASS (historical) | Prior report cited above; must bind new source separately. |
| W1A8/W1A1 export/load/graph on SM75 | PENDING / paused | Feature-owner CPU oracle260/260; no SM75 result because 2080 was not opened to this operator. |
| EAGLE W1A8/W1A1 native zero-update graph smoke | PENDING / paused | Checker is integrated in local main; requires frozen authenticated EAGLE TRAIN artifacts/config and fresh `hardware=rtx5080` admission. No model loaded. |
| No dense fallback on CUDA | PENDING | Requires runtime coverage receipt from merged implementation. |
| Synthetic optimizer/resume/transition on CUDA | PENDING | Requires training owner APIs and serialized GPU slot. |
| Real-model fwd/bwd, zero optimizer updates | PENDING | Requires merge, model availability verification, and serialized GPU slot. |
| Five-tap full-vocabulary TRAIN teacher capture | PENDING / paused | High-level planner, target producer and generation API are integrated in local main; taps `[2,10,18,26,34]` fixed. The nine-row plan and authentic source packet are selected; no capture plan/receipt was materialized or executed. CUDA goldens and release proof remain. |
| GPU/process/resource closeout | PENDING | Fresh remote run evidence required. |
| RTX5080 SM120 development admission | PENDING / paused | No compiler probe, build, capability confirmation, model load, CUDA test or backward was started. Resume requires fresh human authorization, exact source freeze clone, resource baseline and one-shot config. |
| Long QAT, recipe selection, quality/held-out/final evaluation | EXCLUDED | Direct human authorization is development-only and selects no campaign recipe, budget, or evaluation. |

## Historical RTX5080 development authorization and first admission — October 4, 2026

The human directly authorized: “5080 is open use that for development work.”
Root resumed the local registry pause flag. The human message did not carry a
separate event timestamp; the operator recorded the authorization before the
first host query, and the first recorded clock checkpoint was
`2026-10-04T23:15:18Z`. This authorization permits CUDA development, native
operator/model graph checks, real-model forward/backward with zero real-model
optimizer updates, and necessary target-only TRAIN capture. Long QAT, recipe
selection, quality benchmarking, held-out evaluation, final evaluation, and
campaign budget selection remain outside scope. No RTX2080Ti query or action is
authorized by this dispatch.

Fresh local registry evidence at `2026-10-04T23:14:25.969647Z`: `rtx5080` is
registered and `pause_requested=false`. All remote access below used local tmux
MCP session `nine-model-5080`, with SSH host/user/port read dynamically from the
shared registry. The registered Windows SSH endpoint enters WSL with distro
`Ubuntu`; no address is written here. The first SSH command accidentally ran a
Linux Python command in the Windows default shell and failed with
`'python3' is not recognized`; it did not enter WSL, start a process, or query
the GPU. Subsequent commands used `wsl.exe -d Ubuntu --exec bash -lc ...`.

Fresh WSL2 inventory at `2026-10-04T23:21:54.581981Z`:

| Item | Observation |
| --- | --- |
| Host identity | `DESKTOP-E78LJSI`, Linux WSL2 kernel `6.18.33`, x86_64, project root `/home/philip/binary-eagle-decoding`. |
| GPU | NVIDIA GeForce RTX 5080, UUID `GPU-44ceb8b5-b67a-a317-fee3-f01c9201994e`, 16,303 MiB total. This is the expected SM120 device; the direct query did not report compute capability, so the frozen admission must confirm it. |
| Driver/runtime | WSL KMD driver `616.92`, NVIDIA-SMI `615.71.08`, CUDA UMD `13.4`. |
| Initial resource snapshot | 2,792 MiB used / 13,186 MiB free, P0, 47 C, 2% utilization, 2,655 MHz SM, 15,372 MHz memory, 45.88 W. This was not an empty-memory baseline. |
| Context/process checks | `nvidia-smi --query-compute-apps` empty; full `nvidia-smi` reports no running processes; `fuser -v /dev/dxg` has no holders; no `binary-eagle-runtime` tmux server. A later PID/PGID check found the prior state-record PIDs absent. |
| Host resources | 19,802,336 kB MemAvailable; 219,476,598,784 bytes free under the registered project root. |
| Existing checkout | HEAD `7547d25`, dirty `scripts/train_continuous_w1ax.py`, plus pre-existing untracked `checkouts/` and `rescue-head-20260927/`. These remain untouched. A stale A8 state file says `running` PID/PGID 676, but that PID and group are absent. A stale W1 pilot build state says `starting` with null PIDs. Both records remain untouched. |
| Available local weight files | Frozen target `Qwen3-4B-f16.gguf` (8,051,285,280 bytes), Q4 EAGLE (128,988,160 bytes), and F16 EAGLE (442,700,800 bytes) exist. DSpark/DFlash BF16 draft GGUFs and old pinned safetensors locations were absent. Hashes must be checked before use. |

Root approved a bounded development capture pilot using nine original TRAIN
prompts, one per split/domain cell across train, calibration-fit and
calibration-validation roles, plus six portability goldens. Its local ignored
source packet is
`results/nine-model-qat-preparation/development-pilot-source-20261004/`:
corpus manifest SHA256
`9fc8caa80f2c29785eeaeff01f3875c27fee46853350de9ceebe682f8d12b8dc`, TRAIN
shard `0aed2973080abe253e9ca5ca773e791c62c706ae4d0b9b6893524bd3cfba15c8`,
index `35762cd62319b648f99e66c453c3aaba540a14d463d2e37283ea20943d7807cf`,
and explicit nine-row selection `5f21e2e50d2e22eaa22beb135327f0814c53530500489960f2b9d18f4ac07441`.
The pending capture plan caps native generation at 32 new tokens, each chain at
544 tokens, 15 total requests, 8 GiB total capture storage, and sets the exact
soft-logit bound from all nine chains at 3.00 GiB plus 131 MiB of goldens. The
request/prefix bound, target binary/client/tokenizer/template hashes and fresh
plan-only receipt still need to bind the integrated native source. These local
source files have not been staged to WSL or captured.

No model was loaded and no GPU job or source mutation occurred during those
initial checks. Later actions and the human pause are recorded below.

## Source/toolchain staging and human pause checkpoint

Under the sole operator's tmux MCP transport, Git cloned published main
`b998f4f7ef193654e231338913be4945c98fa766` into the distinct ignored path
`runs/checkouts/nine-model-qat-5080-b998f4f/`, then initialized its native
submodule at `874cd2b04622cb8e5c61494f7e09e23984598fab`. The clone and
submodule both passed exact-SHA/clean-tree checks. The clone's
`pyproject.toml`, `uv.lock`, and `.python-version` SHA256 values matched the
local main at `6153546`; local Git confirms those three files remain byte-
identical through current local main `3cacf482`:

| File | SHA256 |
| --- | --- |
| `pyproject.toml` | `91131f035dac4bfbf0cba10a45806c47eb60e693f1902688ede40d5d7cfb9cb4` |
| `uv.lock` | `33515d09381d8dba1a5059d0dbac474f9d3a9a84ebc5bce718a97d37253cac82` |
| `.python-version` | `49a506dd32096b010d75205acf3430c9ae6c40351888129499e5a5e487126c93` |

The host toolchain probe, before dependency installation, found UV `0.11.32`,
Ninja `1.13.2`, NVCC `13.1.115` at
`/usr/local/cuda-13.1/bin/nvcc`, GCC/G++ `15.2.0`, Python `3.14.4`, and Git
`2.53.0`. CMake and a GCC 13/14 alternative were not on PATH; no bootstrap
toolchain existed at the previous experiment path. Root then explicitly
authorized locked project dependencies in this owned clone.

Exact supervised dependency command:

```sh
tmux -L binary-eagle-runtime new-session -d \
  -s nineprep-uvsync-20261004 -c /home/philip/binary-eagle-decoding \
  'exec python3 scripts/remote_job.py nineprep-5080-uvsync-20261004 -- env \
  UV_CACHE_DIR=/home/philip/binary-eagle-decoding/runs/nineprep-5080-uvsync-20261004/uv-cache \
  UV_PROJECT_ENVIRONMENT=/home/philip/binary-eagle-decoding/runs/checkouts/nine-model-qat-5080-b998f4f/.venv \
  /home/philip/.local/bin/uv sync --locked --group w1a1 \
  --project /home/philip/binary-eagle-decoding/runs/checkouts/nine-model-qat-5080-b998f4f'
```

Supervisor run `runs/nineprep-5080-uvsync-20261004/` started at
`2026-10-04T23:43:01.169839Z`, ended `2026-10-04T23:45:20.701665Z`, status
`finished`, exit code 0. Supervisor PID/PGID was 40873 and child PID/PGID 40874.
It installed CPython 3.11.15, CMake 3.31.10, Ninja 1.13.2, NumPy 2.4.6,
safetensors 0.8.0, PyTorch 2.14.0, and locked CUDA Python libraries into the
clone's `.venv`. Its UV cache is under the same ignored run; an in-progress
poll measured 4,312,671,795 cache bytes. Final post-run host reading showed
MemAvailable 19,709,684 kB and 199 GB project filesystem free. This was a CPU
dependency setup: it launched no model, compiler, CUDA kernel, or GPU context.

The human then directly paused all RTX5080 use (“pause all 5080 usage continue
mac only”), recorded by the operator at `2026-10-04T23:52:21Z`; root confirmed
the machine-local pause flag was set. The dependency supervisor was already
terminal. The compiler probe, native build, model load, target capture, and
evaluation were still unstarted. The current local project source is main
`3cacf482c8b55f6b0d8a5979de3ec85098775e50` / native
`50ca2676fb6fbc6a8455e7bce66a614f84e7016c`. Earlier, root authorized an
isolated clone of published main `b998f4f7ef193654e231338913be4945c98fa766` /
native `874cd2b04622cb8e5c61494f7e09e23984598fab`. Before the pause, the
operator attempted to update that clone to the newer source freeze through a
detached supervisor, but the last read showed no runtime tmux server and no
supervisor state/receipt. Its post-attempt HEAD/submodule state is therefore
unverified; it has not been built or used. Preserve the clone, helper, venv,
cache and logs. Do not query the paused host; resume only after new human
authorization. The operator's local MCP transport was closed. No remote
process was intentionally left running.

## Run evidence template

For each future operation, add an immutable subsection containing exact remote
command and environment, source revision and binary/hash, artifact hashes,
machine/driver/CUDA/precision, supervised tmux socket/session/run ID, UTC start
and finish, outcome/exit code, run-directory path, result hashes, supervisor
and child identities, and post-run process/context/memory proof. Raw logs and
large artifacts stay under ignored `runs/`; this report only links receipts
and hashes. Record cleanup or preserved failure state explicitly.

## Local-only checks for this revision

Environment: Darwin 27.0.0 arm64, Python 3.11.3, operator branch based on
current local main `3cacf482`; work directory
`/private/tmp/nine-model-qat-20261004/cuda-operator`. No remote/GPU process was
created. The ignored model/data presence inventory reported all paths listed
above absent. Registry status was read via `python3 scripts/agent_env.py
status`; host fields were redacted before inspection, and no connection
followed.

The feature-owner CPU test worktree at the time was
`/private/tmp/nine-model-qat-20261004/native`, parent `44d1db6` with forked
llama.cpp `874cd2b04`. The current locally integrated snapshot is main `3cacf482` /
native `50ca2676`; it contains the strict CUDA buffer check and generated-source
capture APIs. The preserved feature-owner test build may have embedded the
earlier submodule revision, so these records are historical CPU evidence only.
They do not test the current CUDA build. Tests were run on Apple M3 Max /
Darwin arm64, Clang 21, with
CUDA and Metal disabled. Their raw logs were copied into this branch's ignored
`runs/nineprep-local-native-review-20261004/raw/` directory:

| Check | Exact command | Result / evidence |
| --- | --- | --- |
| Focused exporter/teacher tests | `BLOCK_TEST_NATIVE=/private/tmp/nine-model-qat-20261004/native-build/bin/test-block-binary BLOCK_TEACHER_NATIVE=/private/tmp/nine-model-qat-20261004/native-build/bin/llama-block-teacher python3 -m unittest discover -s tests -p 'test_block_*' -v` (cwd native worktree) | 10 methods passed in 3.965 s, including group SIGTERM reaping; raw log SHA256 `dc3321ac1f48a1152eccf5796e730ee0ce4639b8aa147063af36f62c3b213401`. |
| W1 backend oracle, correct filter | `/private/tmp/nine-model-qat-20261004/native-build/bin/test-backend-ops -b CPU -o W1A1_MUL_MAT -j 4` | 260/260 passed, including bits 1/8, K=9728, one/seven rows; CPU only. Raw log SHA256 `e2f742fb2dc0a5a5eb3bf3be722b95afcb8ddf6d51aa8b2013b0ad093420c342`. |
| Filter guard failure retained | `/private/tmp/nine-model-qat-20261004/native-build/bin/test-backend-ops -b CPU -o 'w1a1.*' -j 4` | Exit 0 but 0/0 tests; raw log SHA256 `913ec5dc143dbef24e7305f6128bc093a8716793549b401f88c6646113168224`. A zero selected-case count is a failed gate. |

These are feature-owner CPU results, not independent CUDA evidence. The latter
two raw logs and focused test log are preserved in the ignored run directory;
their parent logs remain at `/private/tmp/nine-model-qat-20261004/`. No local
GPU build, actual model load, CUDA test, target capture, or resource measurement
was run by this operator.

Operator report checks and outcomes:

```sh
git diff --check
```

Passed. The arithmetic/report assertion was run as:

```sh
python3 - <<'PY'
from pathlib import Path
s=Path('experiments/nine-model-qat-preparation/cuda-plan.md').read_text()
p=373_555_200
features=5*2560*4
logits=151_936*4
assert p*4 == 1_494_220_800 and p*8 == 2_988_441_600
assert features == 51_200 and logits == 607_744
assert features+logits == 658_944
assert 64*(features+logits) == 42_172_416
assert 256*(features+logits) == 168_689_664
assert round(256*(features+logits)/2**20, 2) == 160.88
assert round(32768*(features+logits)/2**30, 2) == 20.11
assert round(1_000_000*(features+logits)/2**30, 1) == 613.7
for term in ('block_native_teacher_request_v1', 'PGID', 'K=9728',
             '20.11 GiB', 'test-block-binary', 'optimizer step count stays zero'):
    assert term in s, term
print('native API/resource assertions: PASS')
PY
```

It printed `native API/resource assertions: PASS`. This was a
documentation/source-inventory check, not a CUDA test or device memory claim.
No output files were created, so there was no raw-run cleanup. Remote CUDA
requirements in the ledger remain pending.
