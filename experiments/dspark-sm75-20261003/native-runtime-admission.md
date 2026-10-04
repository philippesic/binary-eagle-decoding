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
