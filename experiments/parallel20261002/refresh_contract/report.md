# Refresh decision units and experiment handoff

Completed bounded CPU research packet C on October 2, 2026. Base source
`ca02272ae4c5740077c96eb2b3993435c814f444`; worktree
`/private/tmp/eagle-parallel-20261002/refresh-contract`, branch
`research/20261002-refresh-contract`. All counts, losses, prompt IDs, identities,
teacher bytes and handoff manifests in this directory are fabricated synthetic
fixtures. No production source changed. No model, live corpus, GPU, Metal,
SSH, development payload or final payload was used.

## Actual producer and consumer

The protocol already specifies `native_acceptance` as accepted/proposed draft
tokens in [0,1]: [protocol](../../w1ax-trajectory-refresh-protocol.md#learning-curve-and-stop-rules).
This is consistent with [evaluation definitions](../../../docs/EVALUATION.md#definitions-and-timing).
It is not evidence of a metric defect.

Repository searches for `w1ax_refresh_learning_curve_v1` and `native_acceptance`
found no automatic producer of that refresh schema. The only constructed curve
inputs are manual synthetic dictionaries in `tests/test_trajectory_refresh.py`
and `tests/test_qat_curriculum.py`; the planner reads an optional user-supplied
JSON file. Neither the curriculum runner nor its CLI emits refresh curves.

There is a separate actual native development aggregate producer:
`scripts/w1ax_continuous_stages.py:native_acceptance_metrics` (line 1494). It sums
each request's `quality.{accepted,proposed,rounds,emitted,accepted_emitted}` and
emits `native_acceptance_rate=accepted/proposed` and
`native_accepted_per_round=accepted/rounds`. `_evaluate_development` (line 1514)
stores these plus checkpoint/native-cell identity. This output does not become
`w1ax_refresh_learning_curve_v1` anywhere automatically. The same development
report compares accepted/round against Q4_0; it does not change the refresh gate's
rate unit. Native quality aggregation excludes checkpoint-replay diagnostic
rows upstream; accepted counts precede output stopping, while accepted-emitted
counts are the accepted prefix bounded by actual emitted tokens.

`src/w1a1_eagle/trajectory_refresh.py:learning_gate` (line 122) validates a
fraction and evaluates `current.native_acceptance < previous.native_acceptance
- max_acceptance_regression`. It does not consume raw counts, exports or any
auxiliary `accepted_per_round` field. The validator demonstrated this callable
behavior by changing raw auxiliary fields while holding the rate field fixed:
the output is identical. A manual fraction with the wrong meaning cannot be
distinguished by this schema; a rate above 1 is rejected. No automatic wrong-unit
assignment was found. There is no reason here to select a replacement metric.

## Count-based proposal and paired policies

`research/parallel20261002/refresh_contract/reference/contract.py` contains a
typed, side-effect-free proposal. `RoundCounts` retains prompt, proposed,
accepted, emitted, cap and EOS disposition; `CurvePoint` retains checkpoint,
export, completed steps, CE loss sum and supported-label count. It retains every
quality round, including zero-proposal rounds when the pooled proposal count is
positive. The independent validator caught that denominator case and the
proposal was corrected before completion.

`decision_receipt` sums raw counts before division, records all three named
units, and maps **only** accepted/proposed into the existing rate field. It
embeds the gate policy, curve input, identities and operands/operators/results
for every gate comparison, then calls the actual `learning_gate`. Floating-point
comparisons reproduce the callable Python gate, including its strict `<`
regression boundary and CE denominator floor `max(previous_ce, 1e-12)`.
`max_acceptance_regression=.02` is an absolute .02 rate difference; it is not
a 2% relative drop or .02 drafts/round. A cap is an upper bound on proposals,
not a denominator substituted for the actual proposed count.

| Fabricated transition | Accepted/proposed | Accepted/round | Existing gate |
| --- | --- | --- | --- |
| Cap 5 to cap 1 | 5/10=.5 → 2/2=1 | 5/2=2.5 → 2/2=1 | pass |
| Cap 1 to cap 5 | 2/2=1 → 5/10=.5 | 2/2=1 → 5/2=2.5 | stop: rate regression |

Both pairs advance steps 100→200, improve CE 2→1.8, and change-prefix fraction
.5. EOS clips one wide round's emitted count to 1 despite 2 accepted tokens;
wide totals accepted=5, proposed=10, rounds=2, emitted=5, accepted-emitted=4.
The narrower totals are 2, 2, 2, 3, 2 respectively. Accepted-emitted and target
bonus tokens do not replace the accepted-draft numerator. These are unit
counterexamples, not performance measurements or selected draft policies.

A further unequal-prompt test has old prompt rates 1/1 and 0/5, new rates 1/2
and 1/4. Pooled rates improve 1/6→2/6; unweighted prompt means decline
.5→.375. The proposal passes based on the pooled counts, never those prompt
means. Per-prompt/domain breakdowns may be reported without changing weighting.
The typed fixture scope uses normal accepted+one-target emission or explicit EOS
clipping; a future real adapter must also audit other output-stop dispositions.

## Legal handoff and budget limits

The generator reuses the existing fabricated `RefreshTests` fixture and calls
the actual `plan_curriculum_refresh`: first three changed-label requests are
queued; after fabricated new labels, missing requirements fall to zero.
**Both** actual plans retain `training_eligible=False`. Zero missing rows mean
provider admission is required, not granted. `require_provider` separately
rejects a planning-only object; a real provider factory must audit payloads and
bind full-body training eligibility. The synthetic admission hash is explicitly
a simulation, never a real capture/storage audit.

`CurriculumState.load_state_dict` binds budget, data and model contracts. Its
actual tests preserve spent seconds on exact resume and reject a changed corpus
or budget. `CurriculumRunner.resume` also checks immutable source/math/runtime
contract, restores optimizer/cursor/RNG and preserves durable model residency
plus reconstruction time; a budget-overrun run is rejected. New provider data
cannot legitimately be disguised as that exact resume.

The proposal shows `old experiment → refresh plan → separately audited corpus
simulation → distinct new experiment`. A new experiment explicitly imports
parent weights only after separate model gates and starts fresh optimizer/update
counters. It retains old ancestry and the cumulative budget ledger. It is not
an exact-resume promise or an implementation of a parent-weight importer.

The fabricated allowance is 20 GPU seconds: old training residency 12, refresh
capture 2, audit 1, export 1, old development 1; remaining 3. Those charges are
nonoverlapping. Training residency already includes its run smoke/export time
as accounted by the runner; overlapping component timers must not be added
again. A requested 3-second new run passes **proposal** checks only with the
separate audit simulation. A spent 17-second allowance rejects that same
request. A refreshed Codex reset timestamp also rejects it, independently of
GPU budget. Every output keeps `training_authorized=False`.

Important boundary: existing code enforces run-local curriculum/residency
budgets, not a persisted cross-experiment project authorization ledger. Calling
`CurriculumState(...)` for a new experiment starts counters at zero by design;
that call is not budget authorization. The proposal makes cumulative accounting
visible but is not wired into a production launcher and does not authenticate
its synthetic authorization hash. Do not claim that this research established
an existing automatic global budget gate. Any real launcher integration needs
the owner's explicit experiment scope, authoritative ledger, provider factory,
native CUDA readiness, all-stage zero-update smoke and exclusive GPU ownership.

## Checks and next action

macOS arm64 CPU, Python 3.11, Torch 2.8.0 for existing curriculum fixtures;
receipt arithmetic is Python scalar arithmetic. No hardware throughput claim.

```sh
PYTHONPATH=src:scripts:tests:. python3 -m unittest test_trajectory_refresh test_qat_curriculum research.parallel20261002.refresh_contract.reference.test_receipt research.parallel20261002.refresh_contract.validation.test_refresh_contract_validation
PYTHONPATH=src:tests:. python3 -m research.parallel20261002.refresh_contract.reference.demo
python3 -m ruff check research/parallel20261002/refresh_contract
```

Results: **36/36 tests pass** (20 existing refresh, 5 existing curriculum,
3 receipt, 8 independent validator); generator emits reproducible synthetic
JSON; Ruff passes. No new stale-hash tests were manufactured. No persistent
process or GPU resource exists. MAIN control at completion: original reset
1791049896, research-stop false, 82% used, no research credits authorized.

Deliverable gate is closed for this synthetic proposal. Remaining owner choices:
whether to adopt the receipt and cumulative admission ledger; any future metric,
live corpus, recipe or training budget selection. Root integrates these commits,
updates the active QAT goal checkpoint, pushes and cleans the isolated worktree.
