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
