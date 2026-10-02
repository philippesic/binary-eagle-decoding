# Native round trace semantics: focused review

October2, independent advisor support for `/root/native_round_trace`.
Parent checkoutbb0d30495571b14b1fe751facb1853c6485304e2;
server-context.cpp SHA256
ebb8b4310280b691ca5cebe37dac18a08401cd0063ca5cbccaa4724a7a5060be.
Source-only inspection; no native execution, captures, GPU or source edits.
Control93% used/7% remaining, stop=false, original reset unchanged.

## Root identity is not simply the cached prompt

At trace start around4197, before handle_last_sampled_token, slot.prompt excludes
slot.sampled. That pending token was already processed/emitted on the preceding
loop. Ordinary current committed context is cached prompt plus that pending
token. Record the two pieces and decoder position separately; context shifting
means the cached prefix need not be the entire request transcript.

handle_last_sampled_token inserts the pending token and speculative candidates
into slot.prompt before target decode. At verification end around5448, the
server replaces those candidates with all accepted IDs except the final sampled
token, before process_token iterates emitted output. EOS/stop-string/cap can stop
that loop early. Thus terminal slot.prompt+sampled can contain tokens that were
verified but never committed as processed output. Derive committed continuation
from start identity and the actual emitted list, not that terminal cache image.
Visible text can differ again because output stopping rules suppress text.

## Acceptance and emission have different boundaries

Raw n_accepted is assigned before the process_token loop. Its value can exceed
the number of committed accepted emissions when the output loop terminates
early. Preserve historical raw verifier acceptance; do not silently redefine
sealed experiment metrics.

For an ordinary, nonreplay standard-match round with ordered IDs consisting of
accepted drafts then one correction/bonus, committed accepted count is
min(raw_n_accepted,n_emitted), and committed target count is
max(n_emitted−raw_n_accepted,0). This formula needs those explicit guards.
It does not generalize blindly to replay or synthetic acceptance.

Checkpoint replay records tentative accepted count but emits no tokens, restores
cached state/sampler, then recomputes. Actual replay adjusts acceptance by an
additional one, because the replay candidate list can include the previous
target correction. A replay can therefore expose more than one target-origin
emission across its returned IDs. Link attempts to a logical round or report
role decomposition unavailable. Charge both attempts' time; never add tentative
and committed acceptance as if they were distinct delivered draft tokens.

## Timing, cap and bootstrap conventions

- Batched decode/process wall spans are shared across slots. Preserve batch or
  execution-domain identity and use their union once for aggregate wall time.
  Summing duplicated per-slot spans or unexplained proportional allocation is
  not a measured decomposition.
- begin_us is outside round_us. Include it once if defining full session decode
  timing; do not bury it in a per-round residual.
- Producer residual is clamped to zero. Recompute raw named-span sums and unions
  outside the producer and preserve a negative residual/overlap diagnostic.
- Existing target/process endpoints are first-to-last envelopes: start is set
  on the first participating subbatch and end overwritten by later subbatches.
  They can contain intervening work. v1 alone cannot reconstruct actual
  subbatch interval unions. An extension should distinguish envelope spans
  from arrays of execution intervals; target timing also encloses
  queue_tasks.yield_to_queue and is not pure GPU target time.
- get_n_draft_max applies context slack minus2 and remaining-generation budget
  minus1. Record configured, effective and actual proposed length separately.
- The target graph may execute all proposed rows plus a bonus row even when
  output emission ends early. Executed row count is not committed token count.
- First target token can precede the first active speculative-round trace.
  Reconcile bootstrap emissions explicitly before claiming complete-session
  conservation. Existing trace timing/IDs alone may leave this unidentified.

These constraints were sent to the native trace owner for its adapter and
unapplied extension proposal. They are not a new trace implementation or a
claim that frozen historical measurements are wrong.

## Concrete adapter defect caught before integration

The initial adapter compared native proposed-token deltas against all attempt
proposal vectors. Source4308 increments `stats.n_draft_tokens` only inside
`iterate(drafting)`; replay skips `drafting.push_back` and reuses existing
proposals (explicit comment around4353), while still logging those proposal
vectors. Thus attempted proposals and newly generated proposals differ.

Sent a correction before final commit: preserve proposed_all as an attempt
workload count, but native n_draft_tokens must use newly generated proposals.
The original failed checkpoint attempt counts; its replay does not. Owner
confirmed and added a focused reconciliation fixture plus an explicit
new_proposal_tokens extension field initialized0 and assigned only inside the
actual drafting callback. This is stronger than assuming all future nonreplay
rows necessarily imply new drafting. Legacy derivation remains source-scoped.
