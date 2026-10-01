# Chunked accepted-prefix K/V and captured-chain head batching

2026-10-01. Bounded implementation from goal checkpoint `f502d4a` in
`feature/qat-cache-head`. No model/capture files, held-out data, GPU discovery,
CUDA operations, remote connections, or real-data optimizer updates were used.
Q4_0 EAGLE remains the primary acceptance/latency/throughput comparison.

## Deliverable and controls

`NativeStepAdapter.build_context_cache(shifted_tokens, raw_context_rows,
chunk_size=64)` builds one final `(kv_heads, P, head_dim)` K/V allocation using
bounded FC/norm/K/V chunks. It omits Q, attention, O, all FFN projections and
head. Its `no_grad` boundary also applies to direct calls. The one-layer
architecture is checked (`num_hidden_layers`, when declared, must be integer
one); pinned midlayer/norm/projection geometry remains validated. Diagnostic
attention modes keep the original serial path.

`forward_torch_round(..., optimize_cache=True, optimize_head=True,
context_chunk_size=64)` capability-dispatches the optimized operations. Set both
optimization switches to `False` for the old Torch reference; set just one to
isolate its effect. Generic adapters without the explicit capability flags and
methods keep their old positional-only callback API. No optimizer cadence,
sample order, objective, precision policy, target/verifier or data source
identity is changed by this API. Root owns recording selected controls in the
launch/resume identity before real training.

`rebuild_prefix_cache` gains an optional `build_context_cache` callback.
`rollout_captured_prefix` gains optional `decode_head`; when supplied, its
decoder callback returns an empty logits tensor. The provider still invokes
the public `adapter.decode_step(..., compute_logits=False)` at each valid
proposal, allowing `ObservedAdapter.__getattr__` to delegate capabilities while
its wrapped step retains the first attached state/K/V. All valid states,
including unsupported direct labels, receive logits in one `decode_head`
call. Captured proposals continue to control the recurrent tokens. An invalid
terminal row is zero-filled and receives no decoder/cache/head row.

## Semantics and acceptance evidence

CPU fixtures use random synthetic row W1A1/W1A8 models with frozen F16
embeddings, F32 norms/body/head, 16 hidden channels, 2 query heads, 1 KV head,
8 head channels and 7 draft logits. No native GPU performance is implied.

- Context is paired as token `j+1` with captured raw feature `j`; seed token
  `P+1` and feature `P` are excluded from the detached batch. The provider now
  rejects a seed token/parent mismatch against the rebuilt prefix locally.
- Absolute RoPE positions are `0..P-1` for context and `P+d` for proposals.
  CPU frequencies preserve independently repeated F32 multiplication per
  position; accelerator frequencies retain the preexisting power surrogate.
- A1/A8 caches at P=0/1/3/7 and chunk sizes 1/2/64 are exactly equal to the
  serial reference in the fixed well-separated fixtures. K/V are detached,
  contiguous F32 storage exactly representable as F16. Projection hooks and a
  failing softmax verify that context does no Q/O/FFN/head/attention work;
  FC calls at P=7/chunk2 are exactly rows 2/2/2/1.
- All three valid proposal steps retain their serial state/K/V graph and
  F16-to-F32 cast boundaries; only output norm/head are stacked. Later-only CE
  reaches the first pre-norm state, first K/V and seed raw features. Accepted
  raw context has zero gradient. All nine linears' latent-sign and scale
  gradients are finite and nonzero, including unsupported-first/supported-later.
- Logits/loss and one synthetic AdamW update match serial reference at
  rtol=2e-5, atol=2e-6; all 18 parameter gradients match at rtol=5e-5,
  atol=2e-6. Observed fixture maximum logit absolute difference is 0 for A1
  and 1.1921e-7 for A8 (relative RMS 7.4000e-8); maximum parameter-gradient
  absolute difference is 0 for both. Each optimizer state advances once.
- Independent unequal-prefix round graphs match unchanged same-snapshot
  forwards after perturbing the other round. An explicit equal-round mean
  gradient equals the mean of the independent gradients. This fixture does
  not enable a grouped-round optimizer recipe.
- Unknown activation/fusion attachments fail closed; unrelated drafter
  parameters remain frozen. Optional known `LearnedActivationQuantizer`
  scalar and FC `FusionCorrection` u/v/output_bias parameters are explicitly
  retained (deduplicated); norms/embedding/binary aliases are rejected. Their
  actual modules require post-integration tests with the other owners' commits.
  Target-storage protection remains the modules' installer responsibility.

Head saturation is now a mean over the valid chain. The adapter exposes
`head_saturation_scope="valid_chain_mean"` and
`last_head_chain_saturation_fraction`; serial decode resets the scope to
`last_valid_row`. Root logging must carry this scope, because the existing
`lm_head.last_saturation_fraction` value must not be described as last depth.

## Checks

Python 3.11.15 / Torch 2.14.0, macOS 27 arm64, Apple M3 Max **CPU**. Commands
use `/Users/pippo/github/binary-eagle-decoding/.venv/bin/python` and
`PYTHONPATH=src`. No Metal execution was used. All passed:

| CPU suite | Tests |
|---|---:|
| test_qat_cache_head | 9 |
| test_recurrent_rollout | 6 |
| test_recurrent_provider | 10 |
| test_native_step | 10 |
| test_native_attention_surrogate | 7 |
| test_continuous_qat | 14 |

`test_native_step` initially lacked the worktree submodule's GGUF import; using
the existing local main checkout's `third_party/llama.cpp/gguf-py` as an extra
PYTHONPATH fixed that environment issue. No submodule revision was changed.
`git diff --check` passes.

## Bounded CPU probe and larger graph shapes

Reproduce the intentionally tiny helper with
`PYTHONPATH=src <python> tests/test_qat_cache_head.py --benchmark`. It uses one
CPU thread, hidden32/2Q/1KV/head_dim16, F32 arithmetic with A1 or A8 activation
quantization, prefix64/256, three valid recurrent depths, one warmup and three
warmed repeats. Timing includes forward plus later-only backward; it excludes
optimizer and transfer and provides no full-sized or GPU speed estimate.
Representative warmed medians from the completed helper run:

| Lane | P | Serial (ms) | K/V-only (ms) | K/V + stacked head (ms) |
|---|---:|---:|---:|---:|
| A1 | 64 | 40.819 | 4.047 | 4.037 |
| A1 | 256 | 138.096 | 5.806 | 6.006 |
| A8 | 64 | 26.761 | 3.283 | 3.129 |
| A8 | 256 | 102.670 | 4.987 | 4.812 |

Head timing at this width is noisy (A1/P256 was slower); no head speedup is
claimed. These tiny CPU prefix results establish that removed Python/unused
work is exercised, not GPU savings or acceptance quality.

The helper additionally holds 2/4 independent optimized round graphs at one
snapshot, uses an explicit equal-round mean and performs backward with **zero
optimizer updates**. At P256, A1 medians are 10.948/22.130 ms and A8 are
9.534/18.828 ms. Exact combined final-cache storage is 65,536/131,072 bytes;
shared raw input is 98,688 bytes. Process-lifetime RSS high-water ranges from
221,233,152 to 224,231,424 bytes over the whole helper, including imports and
earlier cases. This is neither per-case peak allocation nor an accelerator
admission measurement. The helper does not implement body batching and does
not replace sequential AdamW updates.

For pinned full dimensions H2560/KV8/head_dim128/P2046, storage arithmetic
(not measured allocator memory) gives ~59.97 MiB of raw F32 features and
15.98 MiB of final F32 K/V per independent example. B=2/4 distinct examples
therefore need ~151.9/~303.8 MiB for these two tensors alone. Chunk64 limits
FC output to 0.625 MiB, embedding rows to 0.625 MiB, fused rows to 1.25 MiB,
and unrounded K+V to 0.5 MiB per active chunk, excluding quantization/weight
temporaries and allocator overlap. Proposal graph, gradients, hard-sign
buffers and optimizer state are additional; these estimates do not prove fit.

## Integration and remaining gates

Root should cherry-pick the accompanying commit, check legitimate learned and
fusion attachments retain all registered optimizer parameters, label head
saturation scope, and record reference controls in runtime/checkpoint identity.
No continuous trainer or root config files were edited here.

Before declaring the optimized path GPU-ready, the coordinated GPU owner must
compare full-sized A1/A8 native-source train-root trajectories against the old
Torch reference and apply the practical export gate: relative RMS <=0.10,
no changed native choice with margin >0.02, with lower-margin flips recorded.
Exact ancestry/masks/attached-later-gradients remain mandatory. If batching
changes useful labels/decisions or gradients, use reference switches and
investigate the concrete failure; benign F32 drift alone is not a blocker.
Measure prefix, proposal/head, backward, full-pair time and allocator peaks
at P64/256/1024/2046, depth1/5 on actual RTX5080 before choosing a chunk or
larger batch. B=2/4 changes optimizer cadence unless explicitly declared as
a separate experiment and still requires allocator floors/native acceptance.
No SM75, quality, native acceptance, or inference-throughput claim follows
from this CPU deliverable.
