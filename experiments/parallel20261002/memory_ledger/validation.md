# Independent storage-calibration validation

The assigned CPU calibration passed against the current source helpers. On a
known fixture, `model_storage_bytes` returned 52 bytes for a shared 3×4 F32
parameter bank, its 2×4 view, and a separate F32 scalar buffer; the alias and
view counted once. After one CPU AdamW step materialized state,
`lane_storage_bytes` returned 152 bytes: those 52 model bytes, a 4-byte step,
and two 48-byte moments. The exact expected and measured values matched.

The seven tensor leaves in a state-dict plus optimizer payload occupied 152
unique storage bytes. The actual curriculum `_cpu_tree` routine made 232 bytes
of unique CPU clone storage because it clones each payload leaf independently,
including three references to the shared bank and the bank view. A CPU
`torch.save`/load round trip retained the shared parameter storage aliases. The
small export fixture also matched `checkpoint_host_buffer_bytes` exactly at
16,777,304 bytes. Raw output and the small saved payload are in the ignored run
directory listed below.

The source order supports the ordered-copy part of the primary ledger:
`CurriculumRunner.save` builds model before optimizer in `raw_payload`, and
the recursive conversion walks that insertion order. `_model_payload` emits
core linears before optional banks. Core `state_dict` keys are parameters
first (`latent_sign`, `scale_offset`), then buffers (`initial_scale`, and
`frozen_bias` where present). `joint_parameter_families` creates sign then
scale lists in linear insertion order; `joint_optimizer` installs those groups
in that order, and curriculum smoke creates Adam state in group/parameter
order. The source-derived hard-sign forward is `torch.where(weight < 0,
-torch.ones_like(weight), torch.ones_like(weight))`; at the `where` call, its
boolean predicate, two F32 operands, and F32 result can be live together. For
the configured 81.92M-weight head those are 81,920,000 B and three times
327,680,000 B, respectively. This is shape arithmetic only; no configured
tensor was allocated.

Independent spot arithmetic from `configs/continuous_w1ax.json` confirms
81,920,000 head weights, 218,234,880 binary weights, 65,280 row scales, and
218,300,160 trainable parameters per lane. Calling the unchanged
`memory_estimate` gives 6,017,638,400 B paired persistent, 873,200,640 B for
one dense gradient, a 3,221,225,472 B assumed graph budget, and
10,112,064,512 B estimated peak against the configured 12,884,901,888 B
(12 GiB) reserved-memory ceiling. These figures confirm the function's
arithmetic; the assumed graph budget, allocator retention, actual CUDA
temporaries, and whether the real smoke fits remain unresolved. The configured
headroom is not a measured fit result.

The observed `_cpu_tree` alias expansion is included in curriculum
`tensor_bytes(raw_payload)`, which sums every tensor occurrence and therefore
budgets its final cloned payload. A source-level peak can still exceed that
final-copy sum when a CUDA `.cpu()` transfer result and its subsequent `.clone()`
temporarily coexist. This CPU run validates clone/alias semantics only; it
cannot measure CUDA transfer overlap, CUDA graph/workspace allocation, or
allocator residency. Optional recipes and present `frozen_bias` buffers must
be inventoried for each applicable checkpoint before treating the
core-and-Adam result as a whole-payload lower bound.

## Reproduction record

- Worktree: `/private/tmp/eagle-parallel-20261002/memory-ledger`
- Branch and base: `research/20261002-memory-ledger`, base `ce5da6bf484346181a1e9d6d916a313d3d346f3b`
- Environment: macOS 27.0 arm64; Python 3.11.3; PyTorch 2.8.0; CUDA not built; CPU F32 fixture and arithmetic only.
- Control checked before each run chunk: `last_weekly_used_percent=83`, original reset `1791049896`, `research_stop=false`, `reset_observed=false`, threshold remaining 1%.
- Exact command: `PYTHONPATH=src python3 research/parallel20261002/memory_ledger/validation/storage_calibration.py > runs/parallel20261002/memory-ledger-validation/result.json`
- Result: exit 0; exact expected/measured agreement for model storage (52 B), lane storage (152 B), and export host buffer (16,777,304 B); state payload aliases preserved by `torch.save`; CPU recursive clone storage 232 B versus 152 B unique input storage.
- Raw files: ignored `runs/parallel20261002/memory-ledger-validation/result.json` and `storage-state.pt` (3,227 B).
- Cleanup: the Python process exited; no persistent processes, accelerator work, remote connection, or GPU allocation was started. Ignored raw artifacts are retained.

Source hashes at validation time:

| File | SHA-256 |
| --- | --- |
| `configs/continuous_w1ax.json` | `4ee4ce05139e0ae4762dfa35b6546b79a3fafc8c2b91be07b6ef76e9f5579c8e` |
| `src/w1a1_eagle/continuous_qat.py` | `8e0ed5930d33c8a717536872ec809d4e4b0bfe0826e778ae61ef434a0ea5da0a` |
| `src/w1a1_eagle/continuous_resources.py` | `27156473a17a13c5f5df25cce92e0f975ed51cdd36625e6f78d30df0085c5a62` |
| `src/w1a1_eagle/qat_curriculum_runner.py` | `2d2e1bf6edc28a16f75a009d77822f2134127a1711ceab3f19f8152b0adece84` |
| `src/w1a1_eagle/recurrent_qat.py` | `f2b6d83b8929807b65b6f21826cc99097a6ddd70f99ff23ca0836aebe4db1224` |
| `research/parallel20261002/memory_ledger/validation/storage_calibration.py` | `7de7350fafa404f7c81197ce2189dfd2c53f68bde6a47bbe990f982877b0cbac` |
