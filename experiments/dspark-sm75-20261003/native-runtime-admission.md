# Native reference admission recovery

October3,2026, bounded worker checkpoint. No remote/GPU access by this worker,
target/verifier precision changes, allocator/kernel work, W1 implementation or
training. Root owns protocol/harness and Luna owns the sole RTX2080Ti operator.
The actual failed probe/candidate evidence remains preserved by that operator.

## Separate failures and corrected evidence

Actual DSpark3 requested three proposals but author reference mode computes all
seven noise slots. The draft context reserved `n_max+1=4` backend outputs;
decode therefore refused the fifth output with `backend sampling supports at
most4 outputs per sequence`. Correct target output with zero drafting was an
invalid architecture cell, not a measured negative result.

Operator corrected the initial combined3/7 description: DSpark7 really launched
with n_max7 and per-sequence output capacity8, generated nonzero proposals and
had no capacity warning. Its partial old-source evidence is retained; there is
still no complete study admission or throughput verdict.

Independently, DFlash3 loaded at10,931MiB GPU use, then aborted in
`ggml_cuda_pool_vmm::alloc` / `cuMemSetAccess` with device-not-ready and free0.
Actual startup used batch2048/ubatch512. Target F16 GPU model buffer was
7,672.62MiB plus KV288MiB; draft BF16 GPU buffer1,767.00MiB plus KV40MiB.
Both also reported CPU-mapped embedding buffers741.88MiB. Target compute
reservation grew77.01 ->301.75 ->599.08MiB; draft compute buffer was97.01MiB.
These are operator-supplied measurements, not worker GPU queries. Exact owned
groups10728/10731/11057 closed, with desktop GPU use364MiB, before recovery.

## Published runtime correction

Native `fcdf5822c5b78f9dbcfd1f5c7106f09c3f0c9b1a` is published at
`https://github.com/philippesic/llama.cpp.git`, branch `feature/dspark-admission`,
on base `76847aa877261817026e2f29472f080235b6a829`. Only guarded author draft
reservation changes: max(7,n_max+1), so short mode gets seven outputs and N7
retains eight. Target output limits/verifier remain unchanged. Author mode
rejects draft batch/ubatch below7 or insufficient actual context output capacity.
Binding traces now record actual batch/ubatch, total/per-sequence output
capacities and backend-sampling mode alongside target immutable identities.

AppleClang21/Darwin-arm64 CPU release server build passes. Focused native
`test-dspark-reservation` passes seven cases: default reservation, short trained
block, N7 unchanged, two sequences, CPU raw-output reservation, EAGLE unchanged
and target verifier unchanged. These prove configuration behavior and
compilation, not SM75/model admission.

## Memory retry policy and gate

Root accepted common `--batch-size32 --ubatch-size32` for **all** study arms,
before timing. Context remains2048, F16 target/KV and sampling remain frozen,
and seven noise/eight verifier rows fit intact. This reduces graph reservation
and injection activation scratch without changing weights or borrowing policy.
On SM75, existing BF16 MMVF supports the seven-column noise block without a
full matrix conversion. Larger injection batches fall to cuBLAS F32 operand
conversion; the fusion matrix alone can require131,072,000 temporary bytes.
The32-row retry is a practical routine admission setting, not a claimed memory
solution until actual model/native execution passes.

No allocator change is included. If device-not-ready persists despite observed
room on the bounded retry, the existing isolated-build
`-DGGML_CUDA_NO_VMM=ON` selects the ordinary CUDA allocation pool and is a
focused possible driver/VMM diagnostic. It is not enabled automatically; root
and sole operator must preserve allocation/error evidence and re-admit actual
hardware/precision/dispatch before conclusions. Do not invent a memory floor
or substitute another precision/target.

[validate_native.py](../../scripts/dspark_screen/validate_native.py) now requires
the actual seven-output capacity fields and at least one full requested3/7
proposal per cell, with successful native sampler/decode logs. Actual noise
rounds must have nonempty proposals and join the observed masks/cache events.
Legitimate zero-proposal output-cap boundaries explicitly declaring
`n_draft_max=0` are separate and need no fabricated noise event. Same-prefix
first-three invariance and target-only exact output checks remain required.
Thirteen focused validator tests pass, including old capacity4, all-zero or
permanently truncated cells, and a legitimate no-noise EOS/cap boundary.

## Q4 representation normalization

Actual supported Q4 conversion changed all fifteen intended FFN matrices and
preserved protected types/byte sizes. The first strict checker rejected
`conf_proj.weight` BF16 shape `[2816,1]` becoming `[2816]`, both5,632 bytes.
This is native GGML rank normalization: both mean extents `[2816,1,1,1]`, and
the DFlash loader explicitly expects `{H+rank,1}`. The checker now permits only
trailing singleton collapse, requires identical native extents and raw
bytes/type, and reports both serialized shapes. Logical shape changes,
transposes and added dimensions still reject. The original checker failure is
preserved; it reflected an overly strict worker check, not changed model data.

Five focused precision tests pass, including actual CPU native Q4 quantization
with a singleton confidence projection, eight protected byte identities and
tampered-head rejection. This remains format-policy proof, not GPU throughput.

Remaining: root integrates published native and parent helper commits; sole
operator incrementally rebuilds CUDA, retries actual five-cell admission with
the common32 settings, verifies exact owned group/context cleanup, and only then
starts fixed reference/FFN-Q4 measurements under the cumulative7200-second cap.
The human decision for FFN-only W1A8/W1A1 integration is still pending.

## Completed probe: exact initial setup-mask classification

Operator inspected the already completed25-request/four-cell probe on native
fcdf/B32: every candidate state begins with `binding_begin`, followed immediately
by two mask records for seq0, anchor0, query0/1, visible keys `[0,1]` and last
visible clean key-1. They occur **inside** the binding lifetime, before any
feature injection or actual noise event. The initial out-of-binding hypothesis
was incorrect; no report should describe those rows as outside binding.

The CPU validator now recognizes only this exact adjacent schema-v1 pair at
that initial boundary. It also requires actual target-derived injection of
positions0..1 before the first noise, proving setup positions are replaced by
real target ancestry. Raw state files remain unchanged; reports retain the two
records and count them separately as `initial_unframed_setup_masks`. Unknown,
late, wrong-query/visibility or un-overwritten records reject. All actual noise
blocks still require exactly seven observed query masks and complete native
proposal/verification/cache joins. Eighteen focused tests pass, including setup
overwrites, unexpected extras, late startup-shaped records and corrupted real
future-mask/prefix ancestry. No GPU rerun or native source edit is needed.

The completed probe must be revalidated from preserved files on CPU only.
The output-ID gate remains strict: a separately investigated first-prose
target-only/speculative difference at output89 is not waived by this mask fix.

## Proposed source-bound numeric receipt, not yet activated

If root accepts the bounded raw-logit diagnostic, the manifest can explicitly
pin a numeric-gate receipt by path/SHA256. This cannot be a global output-check
skip. The receipt must bind native binary, frozen target, protocol/requests,
actual/reference measurement files and full output-token-ID hashes, each
covered first-divergence position/token pair, and raw-top5 trace files/row keys.
The diagnostic must reproduce the same shared prefix at the first difference;
finite/no-NaN raw top-two winners must match the actual sampled/emitted tokens
on the respective target paths and meet an explicitly root-approved near-tie
bound. Q4's target-correction/no-accepted-draft control isolates a target-path
rounding issue from a draft acceptance explanation.

The validator should recompute these joins, margins and coverage from the
source-pinned evidence. Any unlisted mismatch, changed input/runtime/weights,
different reached prefix, nonfinite row or large-margin contradiction still
rejects. Full output differences and any downstream prefix cascade remain
recorded as a native floating-point correctness caveat; an admitted exception
must never be described as exact greedy-output parity. Subsequent measurements
may require their own scoped gate evidence rather than inheriting a blanket
waiver from this prompt. At this checkpoint no numeric receipt is active and
the existing exact output-ID check has not changed. Root/Luna/Astra own the
actual diagnostic and acceptance decision; W1 representation remains pending.

## Accepted scoped numeric edges and executable CPU receipt

Root accepted two unchanged-runtime diagnostic edges with the same predeclared
0.05 bounds. At position89, target-only72499 versus primary-Q4 target
correction9920 had signed gaps+0.0082206726/-0.0031223297 and centered common
top5 maximum0.0086177826. The actual Q4 row was reached/nonreplay row0 with
no accepted drafts and empty prior-draft prefix. At position94, primary-Q42331
versus short-DS323035 had gaps+0.0012741089/-0.0055007935 and centered maximum
0.004618454. Both paths reached base92 plus identical prior draft tokens
`[2348,279]`; Q4 rejected the next proposal while DS3 accepted its target-matching
proposal. All selected/emitted raw winners were finite and NaN counts zero.
These are operator-provided actual diagnostic values, not this worker's GPU runs.

Observed complete DS7/DF7 paths equal primaryQ4 for all five cases; observed
DS3/DF3 paths equal each other for all five. A single89-edge receipt leaves the
four short-mode prose outputs uncovered; the accepted94 edge connects their
exact observed native paths. Downstream differences are admitted **only** as
those exact complete sequences. No statement that all future prompts match,
bit parity or confirmed batch-arithmetic causation follows.

[numeric_gate.py](../../scripts/dspark_screen/numeric_gate.py) produces a receipt
from existing files only and recomputes it on consumption. It hashes raw JSON
requests/responses/measurements/traces/protocol, uses already verified model
hashes from run configs, checks unchanged runtime/request/sampler settings,
sequential task mapping, full rendered+generated shared prefixes, finite
sampled raw argmax,0.05 gaps/centered-top5 bounds, and causal retained-prefix
joins against complete emitted round streams. It composes only
target-only→primary-Q4@89 and primary-Q4→short-DS3@94. Original candidate outputs
must equal a connected complete observed path; uncovered cases remain reject.

Minimal producer input JSON:

```json
{
  "manifest": "/absolute/existing-probe/manifest.json",
  "primary": "/absolute/baseline-readiness/results",
  "protocol": "/absolute/frozen/configs/dspark-screen/protocol.json",
  "edges": [
    {"left": "target_only", "right": "eagle_q4_0",
     "results": "/absolute/neartie-logit-diagnostic/results", "gate": "/absolute/numeric-gate.json"},
    {"left": "eagle_q4_0", "right": "dspark_3",
     "results": "/absolute/neartie-shortnmax/results", "gate": "/absolute/numeric-gate.json"}
  ]
}
```

```sh
python3 scripts/dspark_screen/numeric_gate.py numeric-inputs.json numeric-receipt.json
```

Copy the immutable original probe manifest to a new filename and add
`numeric_gate: {path,sha256}` pointing to the produced receipt, then run the
existing CPU validator. Do not overwrite original raw manifests/artifacts.
The receipt may be partially covered; it is a numeric-policy artifact rather
than model admission. Final model admission requires all structural/ownership/
output checks and exposes `binary_sha256`, `protocol_sha256`,
`model_sha256: {dspark,dflash}` plus `target_sha256` for harness freshness checks.

Four focused numeric tests pass: two-edge full coverage, one-edge uncovered
short suffixes, large-margin/unreached-prefix rejection, and tampered receipt
rejection. Eighteen existing structural tests still pass. No GPU re-probe,
native edit, relaxed threshold or W1 implementation occurred in this worker.
