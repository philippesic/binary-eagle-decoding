# Matched A8 QAT comparison — interim

The matched comparison is still running on RTX5080/SM120. Both arms have real
optimizer updates and exact positive-step restore evidence. The reference at
5,000 updates remains below Q4_0; the trained candidate comparison is pending.
This report will be replaced with final measured results after both 7,200-second
cumulative training caps and final native development evaluations finish.

## Frozen comparison

Both arms reuse authenticated 10,000 TRAIN prompts / 3,899,930 supervised rows,
seed8101, hard CE, captured native target features/labels, fixed cache/mask/
vocabulary ancestry, and the same data order. No A1, Bop, curriculum, refresh,
fusion correction or architecture change. Reference: fixed A8/symmetric binary
weights, latent magnitude0.5. Candidate: learned A8, all-nine affine midpoints,
latent magnitude0.1. Both use existing AdamW, sign LR0.001, scale/activation/
midpoint LR0.00001 where present, warmup100, norm clip1.

Both request cache/head optimization. Actual cache1/chunk64 and reference
batched head are observed. Candidate training intentionally uses serial head
execution to preserve invocation-local learned-quantizer gradients; inference
uses batched/native execution. Every candidate tensor had finite/nonzero gradients
and measured sampled movement: sign9/9, scale9/9, quantizer6/6, midpoint9/9.

Execution source3bd4837850915cb7d308290e573e8d8a08eee1a2 is a helper-only fix atop
583480c79f3ca090de0952deca8278dcb7f15c2d. Training runtime hashes are unchanged.
Native runtime9e2c7a90051e738751aab7d7bd7c2d8201fb76e3, CUDA RTX5080/SM120,
Torch2.14.0+cu130. Target weights and target/draft KV are F16; A8 uses native packed
INT8 dispatch. Q4_0 EAGLE is the primary baseline with native Q8_1 activation
conversion. These results do not establish SM75 performance.

## Available interim measurements

All measurements use the same24 unsealed development prompts
(SHA131a3db7958ff6aa818b23019297654507d5b80bed3c298349417b7e3b2ba081).
Timing has5 measured repetitions /120 requests per variant, with warmups excluded,
A8/Q4_0/target-only order alternated. Request rate counts actual output tokens /
complete HTTP wall time including prefill; decode uses server predicted_ms.
TTFT is unavailable for this nonstreaming endpoint. Evaluation wall time is
separate overhead and is never a serving-rate denominator.

| Checkpoint | Development CE | Native acceptance rate | Request tokens/s | Relative to Q4_0 |
|---|---:|---:|---:|---:|
| Candidate step0 | 7.5463 | 2.5012% | 63.524 | 0.4703 |
| Reference step5000 | 5.1176 | 7.7106% | 81.477 | 0.6013 |
| Q4_0 at reference5000 | n/a | 26.5917% | 135.491 | 1.0000 |

Raw accepted/proposed/round counts and accepted drafts/round will be transcribed
from the archived original reports, not inferred from rounded rates. Candidate
step0 is an untrained control; do not present it as a trained-candidate result.
Final trained-candidate results and matched cumulative-budget endpoints remain
unverified. No quality, latency or total-throughput win is claimed.

## Resume and recovery evidence

Reference exact restore859→912 and candidate105→1200 passed with optimizer/RNG/
cursor/recipe/probe/telemetry preserved. Human pause saved candidate2081/cursor2084
(checkpointac8af0f058b7d1d...), budget441.281998629seconds, no active attempt;
human resume restored it and advanced2221+. Pause downtime is excluded.
Reference5000 budget settled764.725seconds. Each arm retains its own7,200-second
cap; standalone evaluations are bounded1,200seconds and arms alternate at natural
5,000-update development boundaries. Intermediate snapshots save every1,000.

Two preserved pretraining execution incidents recovered within the two-retry
limit: unbalanced TRAIN gate selection was replaced with already-declared balanced
TRAIN gate prompts using the authenticated existing bootstrap; a timing-helper
ID-prefix mistake was fixed to accept exact frozen prepared development bytes.
No dataset recapture, recipe/precision/budget change or discarded optimizer work.
Checkpoint and successful raw evaluation archives use verified same-filesystem
hardlinks before pruning. Original historical paired step1000 remains untouched.

## Artifact locations and pending completion

Raw runs/archives live outsideGit beneath
`/home/philip/binary-eagle-decoding/checkouts/a8-qat-run-583480c7/runs/qat-a8-comparison-20261003-01`.
Evaluator/supervisor source is the separate immutable3bd checkout. Exact commands,
birth ticks, hashes, budget ledgers, archive locator maps and incidents are in
ignored local `runs/qat-a8-recovery/operator-ledger.json`. Durable current state:
[goal](../../docs/goals/a8-qat-recovery-and-comparison.md).

Pending: trained candidate scheduled/final reports, both cumulative cap endpoints,
complete counts/coverage/variance and final resource-release proof. Monitor stays
active until completion or a human pause. No sealed-final prompts are accessed.
