# Native session accounting protocol

This opt-in CPU protocol reuses `w1ax_eagle_round_v1`. It joins the
[complete-session objective](../session_objective/REPORT.md) to the real native
producer rather than adding another round event stream. `native_session_accounting_v1`
is the adapter output schema; `session_context_v1` is an additive, UNAPPLIED native
trace extension. A disabled `W1AX_ROUND_TRACE_JSONL` keeps all added event work behind
existing trace guards. No new timing calls, CUDA events, synchronization, readbacks,
cache mutation, verifier changes, or production runtime adoption are included.

## Input and identity

Map serial task groups using existing `scripts/analyze_w1ax_diagnostics.py`
`map_measured_trace_rows`; then call `adapt_session` separately for each task with:

| Metadata key | Meaning |
| --- | --- |
| `session_id` | Runner request identity, include repetition/variant |
| `generated_token_ids` | Complete response raw IDs; require exact trace concatenation or one leading seed |
| `prompt_token_ids` | Original prompt transcript, if available |
| `generation_cap` | Effective finite generation budget; absent means unresolved, not limitless proof |
| `eos_token_ids` | Actual vocabulary end-of-generation IDs |
| `sampling_mode` | `greedy` or `sample_and_match`; probability-ratio verifier rejected |
| `clock_domain` | Unique native process incarnation and host; never reuse after restart |
| `ancestry` | Exact runtime/model/policy/verifier/processor hashes, target precision, actual device, sampling seed, prompt/capture ancestry |

The adapter output preserves ancestry without certifying it. Missing ancestry is
an unresolved domain. Full transcript root is SHA256 of compact UTF-8 JSON token
IDs `prompt + leading_seed + previous_emitted`. This join cannot identify private
cache, feature, grammar, sampler, or drafter state. `root_generated_position` and
`remaining_cap` use the complete response, including the untraced initial seed.
Missing prompt/response data leaves root identity null rather than manufacturing it.

The native extension records `generated_before`, `remaining_before`,
`decoder_position`, `cached_prompt_token_ids`, and `pending_seed_token_id` before
`handle_last_sampled_token` mutates cache. Cached prompt may be context-shifted.
Keep cached prompt and pending seed separate: this is an actual model-input
identity, not a full transcript or private-state snapshot. At terminal processing,
`slot.prompt` can already contain accepted IDs that were never emitted; the
adapter uses actual `emitted_token_ids` for the next transcript root.

## Counts and termination

| Output count | Source rule |
| --- | --- |
| `proposed_all` | Every trace attempt's `n_proposed`, including checkpoint attempts |
| `proposed_quality` | Attempts except `checkpoint_replay` |
| `accepted_decisions` | Quality `n_accepted` assigned before output processing |
| `accepted_emitted` | Non-replay `min(n_accepted,n_emitted)` after ordered-prefix validation |
| `accepted_not_emitted` | Non-replay verifier accepted suffix truncated by stop processing |
| `correction` | Ordinary mismatch target sample when accepted < proposed and one target token emitted |
| `bonus` | Ordinary all-match final target sample, if actually emitted |
| `target_no_proposal` | `no_proposal` one-target-token events |
| `unresolved_role_emitted` | Actual replay quality emissions, whose reused-token offset prevents ordinary role formula |
| `verified_rounds` | `complete` rows; no-proposal and checkpoint rows are not native verifier-step increments |
| `eos_emitted` / `cap_terminated` | Emitted final EOS / response count reaches cap; preserve both at a tie |

Emission conserves exactly as accepted-emitted + correction + bonus + no-proposal
+ unresolved-role emissions. Checkpoint attempts emit nothing and contribute
cost, but their tentative accepted count does not enter native acceptance totals.
The replay flag and attempt index remain available; a logical-round linkage and
replay role decomposition remain unresolved. Supplied request-native deltas must
match exactly: proposed=all attempts, accepted=quality decisions,
rounds=complete rows. An API delta mismatch raises, rather than silently changing
historical counter semantics. Missing deltas stay unresolved.

Native stop enum is separately recorded as none/eos/limit/word. A limit can be
context capacity, newline/indent, time, or token cap; do not relabel all native
`limit` events as token-cap events. Client-visible completion text, byte stop
truncation, and raw processed token counts are different scopes.

## Timing and lifetime

Every timestamp uses native `ggml_time_us_cpu_wall`, with lifetime one bound process
incarnation. No GPU time is measured. Existing `target_decode_sync` wraps
`queue_tasks.yield_to_queue`, `llama_decode`, and the existing output-dependent
`llama_synchronize`; it includes scheduler/host work. Draft/process/check/repair
spans are host-call wall times, with asynchronous device completion unproven.
They must not be presented as CUDA enqueue-only duration or GPU execution time.

`begin` is outside the round. Draft/target/process may be shared across slots.
Details such as encoder/feature-copy/decode are nested; never sum them with their
parent stage. Legacy target/process spans are first-to-last subbatch envelopes,
not exact lists of executed callback intervals. Within-envelope interleaving is
unresolved. The adapter preserves intervals, computes union/overlap, and calls
outside-envelope gaps `other_inside_round_us`; the producer's clamped residual
is validated but never treated as an exact category measurement.

The extension logs target/process callback vectors with actual batch size,
slot token/output rows (target), feature/draft-decode token counts (process),
decode return and existing sync flag. Vectors preserve individual intervals;
legacy envelopes remain unchanged. Output rows count executed verifier shape,
including rows whose tokens are later un-emitted at EOS/cap. Target rows include
failed/retried calls where traced; detail vectors do not infer device completion.

`pool_sessions` unions replicated intervals once per explicitly bound process
clock. It reports inclusive stage unions per domain and refuses unbound clocks.
It does not sum stage unions into total cost or sum different process clocks into
one critical-path time. Round sum and round union are both retained.
`gpu_us` and `full_session_us` remain null. Prefill, setup, first-seed cost,
inter-round gaps, queue scheduling outside traced callbacks, and trace serialization
must be covered by a separate measured full-session clock before ranking latency.
`session_ranking_identified` therefore remains false for this protocol alone.

## Acceptance and future adoption

Run the two uniquely named CPU test modules and `run_callback_fixture.py`.
The latter verifies pinned source and patch hashes, checks/applies the patch only
to temporary source staging, extracts the actual C++ trace struct/emitter, compiles
it with CPU stubs and vendored JSON, then parses the actual emitted event.
Its disabled second call must append zero bytes and make zero clock calls.
This checks the actual emitter, not full native-server compilation or inference.

Before future adoption: review the source patch with the native owner, compile
full server at the new exact source, preserve frozen runtime inventories, publish
any submodule commit before updating a parent gitlink, then use an already
admitted native trajectory check. Bind response/prompt/model/runtime/processor
ancestry and full-session clocks. No additional model run, capture, QAT optimizer
step, final-set access, GPU request, or acceptance/throughput claim is authorized
by this CPU deliverable.
