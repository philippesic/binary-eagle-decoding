# Block checkpoint retention capability — October 5, 2026

Status: source implementation and CPU acceptance complete; review/integration
pending. No live source, training packet, remote job, GPU, model or campaign
retention/exposure choice changed. Root owns the active goal and its checkpoint.
This closes the source capability gap described in `block-capture-options.md`;
the historical unbounded saver observations in that report remain accurate.

## Explicit opt-in and limits

`train_nine_model_qat.py` accepts an optional block-only `checkpoint_retention`
object with exactly three required positive integer fields: `keep_recent`,
`max_checkpoints`, `max_bytes`. Unknown/missing fields, booleans, nonpositive
limits and use on the EAGLE adapter are rejected. `max_checkpoints` must admit
`keep_recent`. No default retention object or production packet values were
selected. With the object absent, the block saver retains every generation and
publishes no retention sidecars, as before. Staging files now use exclusive
creation so a failed generation cannot overwrite its own preserved evidence.

`keep_recent` keeps the newest supplied number of owned committed payloads,
plus every protected checkpoint. The hard count and byte limits bound the
**publication peak**, including old payloads and the next writer before pruning.
Existing `*.pt` and `step-*.tmp` artifacts count, including foreign, malformed,
changed, linked and uncommitted files. Checkpoint-shaped directories are refused
because their byte accounting is ambiguous. Symlinks are counted by link size
without following their targets; external targets are outside this directory's
payload budget and are never deleted.

Before creating a next staging file, the saver inventories the existing files
and refuses a count overflow or an exhausted byte budget. The serializer uses
a bounded file wrapper: every write must fit the remaining bytes, and seeks
must stay within the byte bounds; file length is tracked across overwrites.
Thus an oversized or failing writer cannot grow its staging file beyond the
remaining configured payload budget. The current temp is counted once in the
post-write publication check. No old checkpoint is deleted to make space for
staging. A whole-run disk/free-space gate is still necessary: JSON/receipts,
exports, models, teachers, logs, build/cache storage, files outside this directory
and other processes are outside these payload limits. Limits are enforced for
one coordinated writer, not a filesystem quota or concurrent-writer protocol.

## Commit and protection contract

Each opt-in generation retains the existing source/config/runtime/cursor-bound
model, complete stage-local FP32 Adam state and Python/NumPy/Torch RNG payload.
The payload is serialized/fsynced, hashed and admitted before rename or latest
publication. After `latest.json` atomic publication succeeds, the saver reads
it back and verifies the new payload hash. It publishes an immutable generation
receipt sidecar, reads/verifies it, then revalidates each eligible old payload
and its receipt immediately before pruning. Directory deletions are fsynced.

Pruning requires a regular single-link payload and regular single-link receipt,
exact source/runtime/config-family ownership, matching canonical path/filename,
valid cursor/receipt metadata and payload hash. The sole declared A8→A1 change
shares the same ownership family; other scientific config, source/data or
arithmetic changes do not. Foreign, malformed, symlink, hard-link, changed and
uncommitted artifacts are never pruning candidates, and still consume limits.
The saver requires physical directory paths without symlink ancestors; on macOS
use canonical resolved temp paths rather than the `/var` symlink alias.

Initial step-zero checkpoints are automatically protected. The block caller
explicitly protects the committed A8 transition source and publishes/protects
the zero-moment A1 destination before its first update. Natural endpoints and
STOP boundaries are protected. If a boundary already has the exact same cursor,
it pins its verified receipt without rewriting the payload. Protection never
removes model/Adam/RNG/source/cursor evidence. A protected set larger than a
supplied limit causes refusal; the implementation never silently drops anchors
or shortens the requested recent window to fit.

Failure handling is conservative. Failed serialization leaves bounded partial
staging; failed latest/receipt publication leaves its staged or final payload;
failed readback/hash verification causes no pruning. These files consume later
admission and are not automatically cleaned up. An identical cursor generation
cannot be retried over an existing staging file. Resume from the exact committed
latest with a fresh charged elapsed-time generation, within the orchestrator's
repair limit, and ensure its policy admits the preserved failed artifacts.
If latest publication succeeded but its retention sidecar failed, exact resume
can still use the existing hash-bound payload; it stays foreign to pruning until
an explicitly reviewed reconciliation. No automatic adoption is implemented.

## Illustrative storage costs, not selected policy

The published production block estimate is **4,877,680,640 raw tensor bytes per
progressed FP32 resume payload**. Adding the prior 32 MiB/file planning allowance
produces 4,911,235,072 bytes (4.574 GiB) per conservative payload slot. Enforcement
uses actual serialized bytes, not this estimate. Initial and reset-moment payloads
are usually smaller; the table conservatively charges them as progressed slots.

| Illustrative peak slot count | Raw payload GiB | With 32 MiB/slot allowance GiB |
| --- | ---: | ---: |
| 3 | 13.628 | 13.722 |
| 4 | 18.171 | 18.296 |
| 5 | 22.713 | 22.870 |
| 6 | 27.256 | 27.444 |
| 7 | 31.799 | 32.018 |

For illustration, keeping three recent progressed checkpoints in a direct lane
requires space for the protected initial checkpoint **and** next writer: up to
five conservative slots, not three. An A8→A1 lane can retain initial plus both
transition anchors, three recent checkpoints and the next writer: seven slots.
Endpoint/STOP anchors may overlap the recent window; continued runs or repeated
stops can leave more protected anchors outside it. Preserved failed artifacts
need additional slots/bytes. At the conservative estimates, four direct block
lanes with three recent checkpoints could require 91.479 GiB for their checkpoint
peaks alone. This excludes captures, final exports, builds and resource reserves.

The user's scientific exposure, conditioning, objective, rolling window and
whole-campaign disk allocation remain pending. The orchestrator should add this
anchor/writer cost to the published storage options before selecting any packet.
No existing packet or healthy frozen EAGLE W1A8 run should adopt a source change
merely because these CPU checks passed; new block sources require fresh original
ancestry, resource and production SM120 admission as usual.

## Acceptance evidence and handoff

Apple arm64/macOS, Python 3.11.15, Torch 2.14.0, CPU tiny models and local files.
No CUDA/SM120/SM75, acceptance, latency or throughput measurement is claimed.

- 16 focused retention checks: default unbounded behavior; actual caller
  transition/endpoint/STOP protection; exact next model and Adam update plus
  Python/NumPy/Torch RNG and cursor/source after pruning; count/bytes prewrite
  admission; actual oversized serialization and bounded seek/length/failure;
  failed commit/readback/hash paths; foreign/malformed/link/uncommitted retention.
- 76 aggregate CPU checks across retention, block checkpoint contracts,
  nine-model training/contracts and block lane packet tests.
- Ruff check/format passed for the three changed Python files; `git diff --check`
  passed. Luna independently reviewed and reran focused retention tests.

Acceptance command (the existing main submodule supplies the local GGUF Python
package; the feature worktree has no initialized submodule):

```sh
PYTHONPATH=src:scripts:tests:/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py \
/Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest \
  test_block_checkpoint_retention test_block_checkpoint_contract \
  test_nine_model_training test_nine_model_contracts test_block_lane_packet -q
```

Deliverables: saver/caller source, focused tests and this report. Root must review,
integrate/push main, record the source/checks in the assigned active goal, and
retire the merged worktree only after preserving any needed artifacts. Production
policy selection and whole-run admission remain with root/user; no block launch
is authorized by this capability report alone.
