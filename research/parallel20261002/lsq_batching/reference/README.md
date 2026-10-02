# Historical evidence and current QAT regressions

The original LSQ audit and its raw results document the old provider source
SHA256 `81cd3643b9a3060ae00d50d18cdf9ebc39846dbafc3b772b62651fb20d4b31a3`.
That evidence remains in historical commits `e61cf9e` and `349ad51`, the original
report, and the archived ignored run records. `proposal.py` and
`serial_learned_head.patch` remain historical, source-pinned proposal artifacts;
their old source hash must not be bumped to inject a second guard into fixed code.

The project regression suite now calls the current `forward_torch_round` and
requires serial learned-head gradient and optimizer-update equivalence, including
ragged chains and the actual observed native adapter. Its CPU correctness tests
are protected QAT work and do not depend on a local supporting-research stop file.
The standalone research audit still enforces its original stop control.

`audit.py` records hashes and paths of the production modules actually imported.
This binds an isolated corrected checkout honestly when `PYTHONPATH` points at it.
Its current run output describes that imported source; it must not overwrite or
be presented as the historical discrepancy measurement. Fixed/frozen/no-gradient
batching and actual execution metadata remain part of the correction owner's
focused tests and current-source readiness evidence.
