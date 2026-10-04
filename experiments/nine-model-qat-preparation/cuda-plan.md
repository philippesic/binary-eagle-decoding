# Nine model preparation CUDA operator plan

Operator: `/root/cuda_operator` (task `01a10903-1c7a-71b1-abb1-0de3ecc046b8`).
Scope: bounded RTX2080Ti preparation checks and their reports. No remote action
has occurred in this task. RTX5080 is outside this operator's scope.

## Current gate

The shared registry has complete entries for `rtx2080ti` and `rtx5080` at
`~/.config/binary-eagle-decoding/hosts.toml`. It was inspected locally without
printing host values. `agent_env.py status` reports `rtx2080ti.pause_requested`
false, last updated `2026-10-04T00:14:22Z`; this stale flag is not availability.
RTX5080 remains paused (`pause_requested=true`, last updated
`2026-10-04T22:09:29Z`). No tmux MCP session was created, and no host was
queried. Wait for an explicit human announcement that 2080Ti is open and a
root dispatch before connecting. Never query, resume, stage, or run on 5080.

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

The current source at dispatch includes the existing block-0 tap utility
[`scripts/native_target_block0_capture.cpp`](../../scripts/native_target_block0_capture.cpp),
which captures a bounded 29-token prefix and selected layers 0/14. It does not
produce the ordered five-tap, full-vocabulary, prompt-disjoint TRAIN teachers
required for DSpark/DFlash. Existing EAGLE W1 data and 32k masks are not a valid
substitute. No teacher capture was started here. The feature/data owner must
publish the exact capture binary/API, tap map, storage cap, and manifest
contract before the operator can schedule it.

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

Full-vocabulary rows are a material storage constraint. At vocabulary 151,936,
one logits row occupies 607,744 bytes (0.580 MiB) in FP32 or 303,872 bytes
(0.290 MiB) in BF16. A bounded 64-row output chunk alone is 38,895,616 bytes
(37.09 MiB) FP32 or 19,447,808 bytes (18.55 MiB) BF16, before five taps and
framework buffers. One million stored rows would be 566 GiB FP32 or 283 GiB
BF16. These figures do not choose a teacher format or row count. The data owner
must declare the required target rows, dtype, shard limit, and label semantics;
the capture process must stream one bounded chunk to disk and release buffers
before continuing. Five-tap feature bytes remain unestimated until the merged
API declares every tap's shape and dtype. Capture only required TRAIN rows and
record total rows and actual shard bytes before admitting a complete corpus.

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
   reference on small tensors. This may exercise optimizer state only with
   synthetic parameters. Keep all outputs in its ignored run directory.
3. **Native loader/graph:** serialize a single DSpark arm, then DFlash, each
   with the frozen target only if required by the native API. Verify selected
   W1 tensors and every declared floating exception, graph nodes, dispatch
   receipts, output shapes, masks/slots, target identity and absence of dense
   compute fallback. Record peak and end memory; shut down and verify its
   supervisor group, child groups and compute context are gone before proceeding.
4. **Target-only TRAIN capture:** after the native integration run has fully
   released, launch only the frozen target and capture the approved TRAIN
   tokens. Use the feature owner's exact prefix/tap API, spill bounded full-vocab
   rows and taps to immutable shard files, hash each shard, validate row counts
   and ancestry, then close the target and prove context release. Do not hold
   target tensors in the drafter training process.
5. **Real drafter forward/backward smoke:** load one architecture/candidate at
   a time with target teachers read from the verified shards. Run forward and
   backward only; assert optimizer step count stays zero, optimizer state is
   empty, and target tensor ownership is absent from the process. Record peak
   GPU and host RAM, activation/checkpoint mode, rows, loss and all gradient
   checks. Do not call this convergence, quality, or optimizer admission.
6. **Resume and transition fixtures:** run exact save/load, RNG/cursor and
   A8-to-A1 transition using synthetic parameters and production checkpoint
   APIs. Verify checksum equality and next-batch identity. This is a distinct
   synthetic stage; never update the real draft to validate resume.
7. **Final closeout:** after every stage, capture supervisor state and all
   process identities, wait for clean exit, verify no owned PIDs/PGIDs/compute
   contexts remain and memory is back to the measured baseline, then make the
   next stage eligible. STOP/pause/error paths preserve checkpoints/logs and
   receive the same cleanup proof. Store raw artifacts in ignored per-run
   directories; reference hashes and exact command/environment here.

**API placeholders remain deliberate:** exact launch commands cannot be frozen
until feature owners publish the current entrypoints, model/tap schema, model
snapshot locations, and output receipts. No guessed CLI is authorized. The
above order and resource separation are implementation-independent and can be
used to wire the final harness once those APIs land.

## Launch form after availability

Do not run this until root confirms the slot and the fresh admission passes.
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
| Local model/data artifact availability | PENDING | Tracked/source metadata is present; ignored model weights, GGUFs and prepared capture packet are absent from this local worktree. Revalidate on the registered host after availability. |
| RTX5080 isolation | PASS | Local pause observed; no query or remote action. |
| DSpark/DFlash native baseline and prior SM75 discipline | PASS (historical) | Prior report cited above; must bind new source separately. |
| W1A8/W1A1 export/load/graph on SM75 | PENDING | Native feature owner must publish the schema/build and tested contracts. |
| No dense fallback on CUDA | PENDING | Requires runtime coverage receipt from merged implementation. |
| Synthetic optimizer/resume/transition on CUDA | PENDING | Requires training owner APIs and serialized GPU slot. |
| Real-model fwd/bwd, zero optimizer updates | PENDING | Requires merge, model availability verification, and serialized GPU slot. |
| Five-tap full-vocabulary TRAIN teacher capture | PENDING | Capture/data implementation and exact manifest/tap contract do not yet exist. |
| GPU/process/resource closeout | PENDING | Fresh remote run evidence required. |
| 5080 SM120/resource admission, QAT and evaluation | EXCLUDED here | Separate future human availability/recipe gate and operator run. |

## Run evidence template

For each future operation, add an immutable subsection containing exact remote
command and environment, source revision and binary/hash, artifact hashes,
machine/driver/CUDA/precision, supervised tmux socket/session/run ID, UTC start
and finish, outcome/exit code, run-directory path, result hashes, supervisor
and child identities, and post-run process/context/memory proof. Raw logs and
large artifacts stay under ignored `runs/`; this report only links receipts
and hashes. Record cleanup or preserved failure state explicitly.

## Local-only checks for this revision

Environment: Darwin 27.0.0 arm64, Python 3.11.3, branch rebased onto main
`c2767405364194981c5a136fded3056422f89ce2`; work directory
`/private/tmp/nine-model-qat-20261004/cuda-operator`. No remote/GPU process or
run directory was created. The ignored model/data presence inventory reported
all paths listed above absent. Registry status was read via
`python3 scripts/agent_env.py status`; its host fields were redacted before
inspection, and no connection followed.

Checks and outcomes:

```sh
git diff --check
```

Passed. The arithmetic/report assertion was run as:

```sh
python3 - <<'PY'
from pathlib import Path
s=Path('experiments/nine-model-qat-preparation/cuda-plan.md').read_text()
p=373_555_200
assert p*4 == 1_494_220_800
assert p*8+2_786_331_140 == 5_774_772_740
assert (8_051_285_280+2_786_331_140+p*8)/2**30 < 12.9
assert 151_936*4 == 607_744
assert 151_936*4*64 == 38_895_616
assert 151_936*2*64 == 19_447_808
for term in ('Local artifact check and source-derived resource budget',
             'Serialized execution/staging checklist',
             '12.88 GiB file-and-tensor sum',
             'optimizer step count stays zero'):
    assert term in s, term
print('resource arithmetic and report assertions: PASS')
PY
```

It printed `resource arithmetic and report assertions: PASS`. This was a
documentation/source-inventory check, not a CUDA test or device memory claim.
No output files were created, so there was no raw-run cleanup. Remote CUDA
requirements in the ledger remain pending.
