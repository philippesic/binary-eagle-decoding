# Direct A1 historical target-only golden reuse

October 5, 2026, 05:28 PDT. Metadata-only work on the authorized RTX5080 host,
with CUDA_VISIBLE_DEVICES empty and two CPU threads. No target/model actor,
optimizer, GPU inference, training source change or additional GPU job.
EAGLE A8 continues under frozen parent30a8dc7/nativecc9cab3.

## What was verified

The new A1 packet's generation request bytes exactly match the admitted A8
packet. Its runtime and all three original TRAIN case selections also match.
Original three native generation receipts and three physical replay receipts
use historical client SHA
`8a43ee3f6f472184b1b229e1c556ae65d45e8fa76fa26c9dc058209ddbb1c052`.
The actual historical client file was rehashed to that SHA; binary/client files
were checked against the A8 link's original locators. Target locator and native
source revision match the new A1 runtime. The large target file was not newly
byte-rehashed here; its frozen admitted locator is reused.

Historical source validators checked all three generation receipts and all
three replay receipts, including F16 target/KV, actual CUDA producer observations,
native source/binary/client hashes, vocabulary/taps/shapes, prompt/template/tokenizer
bindings, zero updates, ordered tokens and contiguous decode history. Every
generation/replay TRAIN ancestry and exact replay request joins. All nine raw
payload files were byte-rehashed with exact F32 shape/size validation: three
generation feature files and three replay feature/logit pairs. No receipt or
client hash was relabeled.

After those read-only checks passed, the ORIGINAL30a helper published the new
A1 `replay-requests.jsonl` and `generation-replay-link.json`, truthfully pointing
to the original historical producer. The A1 requests are byte-identical to A8;
link.requests and link.native also match A8 exactly. A currentc2544aa consumer
then accepted all three historical replay receipts through the same
`exact_replay_join`/`validate_replay` path used by current bind. This exercises
the historical link contract; full bind remains unavailable until A1's own
zero-update actor/export exists.

## Exact artifact identities

Remote packet root:
`/home/philip/binary-eagle-decoding/data/nine-model-overnight/eagle-direct-a1-packet-20261005-01`.
Original logs are beneath frozen checkout
`/home/philip/binary-eagle-decoding/runs/checkouts/nine-model-qat-overnight-5f53740/runs/`.

| Artifact | SHA256 |
|---|---|
| A1 `generation-replay-link.json` | `65b1d0a4a3a0234f41dfe149a9ecc2601a913b916d309f0dd2ff1463173623f7` |
| A1 `replay-requests.jsonl` | `167c9e188510f5c30f267046837d947d915312a37f7d9dd8118af1709d668965` |
| Original `eagle-native-generations-20261005-01/stdout.log` | `1e27a5923364bd7fe5a268a5f2bf05f08874150c1dda483a0c8a8f19d9c058ab` |
| Original `eagle-native-replays-20261005-01/stdout.log` | `9f7912103135d24948174dc01eea860a91decb1ebd84a15eecda70ba548c3003` |
| Original A8 `generation-replay-link.json` | `c7cafa89bfeb9f446535cfdc9f0095dd4683507c4d5384c45e195fa2cabecb1c` |
| Local ignored execution evidence `results/nine-model-qat-overnight/a1-replay-reuse-20261005.json` | `901786c1a2e49c0d5aa74f87ccba554893e71e8c29f3e96c52b47707d977a4f4` |

Three exact successful command handles are preserved in the local execution
evidence, with validation source and tmux results. An initial read-only ad hoc
check incorrectly expected a native field at the top of golden-source-joins;
it stopped with KeyError before publication. Inspection corrected it to actual
runtime/cases equality. The predecessor's current-client mismatch refusal
also remains preserved; neither failure weakened producer validation.

## What remains pending

A1 has its own completed authentic CPU scale-only initializer. This reuse
establishes target-only physical goldens with exact shared requests; it does
not establish the A1 actor, gradients, serialization, admission or quality.
A1 zero-update model/backward/full-moment memory checks, all-nine export,
full current packet bind/independent QA and fresh actual SM120 native admission
remain mandatory after natural owned GPU release. Healthy A8 is uninterrupted.
No acceptance, throughput, SM75 or full-campaign readiness claim.
