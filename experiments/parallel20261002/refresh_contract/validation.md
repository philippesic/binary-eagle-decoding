# Refresh metric and handoff: independent validation

All measurements below are fabricated CPU fixtures. No model, capture, development/final payload, GPU, Metal, SSH, or live training path was opened.

## Source trace

- `scripts/w1ax_continuous_stages.py:native_acceptance_metrics` is the actual development producer. It pools request `quality` counts for accepted, proposed, rounds, emitted, and accepted-emitted, then emits both `accepted/proposed` (`native_acceptance_rate`) and `accepted/round` (`native_accepted_per_round`), plus emitted/round. `_evaluate_development` retains both aggregate metrics for each student and Q4_0 cell.
- Repository search found no production writer for `w1ax_refresh_learning_curve_v1`. `trajectory_refresh.learning_gate` consumes that schema; the only other occurrences are test fixtures and the refresh protocol. Treat its records as manually supplied/synthetic until a producer is implemented.
- The protocol and gate currently specify `native_acceptance` as an aggregate fraction, corresponding to accepted draft tokens / proposed draft tokens. The gate compares only that field, alongside hard CE, completed steps, changed-prefix fraction, and round/cap checks. The producer also exposes accepted/round, and the two units can move in opposite directions. The receipt reports both with named units and preserves the existing gate call; it does not select a replacement metric.
- `plan_curriculum_refresh` returns `training_eligible=False`; `require_provider` independently requires audited full-body train eligibility. The existing curriculum refresh test exercises a real plan and checks that flag. `CurriculumState` checkpoint state binds the budget and data/model contracts; resume preserves phase counters and rejects a changed budget. The synthetic handoff ledger separately carries old experiment charges and the original research reset timestamp.

## Independent gates

Eight focused validator tests cover: pooled count arithmetic with unequal per-prompt proposal totals; opposite movement of acceptance rate and accepted/round; gate invariance to auxiliary metrics and exact replay from the typed receipt; zero-proposal rounds; planning versus provider admission; curriculum budget resume; and cumulative-ledger/new-experiment proposal behavior under successful, overspend, and reset conditions.

The zero-proposal case exposed a receipt gap: the actual producer accepts an aggregate with two rounds and one proposal even when one round had no proposals, but the first `RoundCounts` version required positive proposals in every round. After this was reported, the proposal was revised to allow a zero-proposal round with zero accepted drafts and bounded emitted/EOS handling, while still requiring positive aggregate proposals. The independent producer/receipt fixture now passes.

The ledger fixture begins with 800 of 1,000 authorized synthetic GPU seconds spent across training, capture, and audit. A 150-second new-experiment request is admissible only with changed-corpus audit admission, a distinct experiment ID, and the same still-open 82%-used research window; a 250-second request and a changed/reset research window fail closed. Even an admissible result has `training_authorized=false` and `research_window_resets=false`. This simulates accounting rules; it does not grant real authorization.

## Execution record

- Worktree: `/private/tmp/eagle-parallel-20261002/refresh-contract`, branch `research/20261002-refresh-contract`, base observed at `ca02272ae4c5740077c96eb2b3993435c814f444`.
- Environment: macOS arm64 (`Darwin 27.0.0`), Python 3.11.3, PyTorch 2.8.0. The active control file was read immediately before the broader CPU checks: `research_stop=false`, `reset_observed=false`, `last_weekly_used_percent=82`, `last_reset_unix=1791049896` (original reset unchanged).
- Focused command: `PYTHONPATH=src:scripts:. python3 -m unittest research.parallel20261002.refresh_contract.validation.test_refresh_contract_validation -v` — 8/8 passed.
- Existing contract suite: `PYTHONPATH=src:scripts:tests python3 -m unittest test_trajectory_refresh test_qat_curriculum test_qat_curriculum_runner -v` — 45/45 passed.
- Lint: `ruff check research/parallel20261002/refresh_contract/validation/test_refresh_contract_validation.py` — passed.
- Raw stdout is retained in ignored `runs/parallel20261002/refresh-contract-validation/validator.log` and `relevant-tests.log`. Commands exited; no subprocess or persistent experiment process remains. No temporary model/data artifacts were produced.

## Source hashes

SHA256 at validation:

```text
6975f3bed6240554be87a3898f56b273359b18d58d5ca6e90266aa3bd774c60c  src/w1a1_eagle/trajectory_refresh.py
ec133082768e7c167bb67a7a1e4013883e51bc9a721ba78884a7c39904742dce  src/w1a1_eagle/qat_curriculum.py
2d2e1bf6edc28a16f75a009d77822f2134127a1711ceab3f19f8152b0adece84  src/w1a1_eagle/qat_curriculum_runner.py
cf50bf98874ffcbe6fe7a970d54cfc60a77d73e23da282a6cc41df84f196ce47  scripts/w1ax_continuous_stages.py
273b7689c7b9787dedd2e0690ac94efea5cb474aa75e0a098ca21a56e7208f89  research/parallel20261002/refresh_contract/reference/contract.py
b61fd9da425e1e007c2173dc5c7407918c070acb4921d0c54f0fd54169818d52  research/parallel20261002/refresh_contract/validation/test_refresh_contract_validation.py
```
