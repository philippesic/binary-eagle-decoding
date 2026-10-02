# Activation helper identity closure

Candidate branch `research/20261002-activation-identity`, based on activation
feature `f027609`. This is a bounded CPU implementation packet for the active
QAT optimization readiness goal; root owns review, integration and its goal
checkpoint. No live source or frozen `bb0d304` GPU source changed.

`learned_activation.py` and `native_step.py` import `activation_reuse.py`
unconditionally. Its disabled path executes even when the adapter option is
false. Before this patch the helper was absent from the base and curriculum
critical inventories, and equal supplied/receipt maps could omit it together.

The helper is now named in `continuous_runtime.MATH_FILES` and curriculum
`EXTRA_MATH`. `validate_optimization_readiness` requires its source-map key
before reading a CUDA receipt. Existing runtime/receipt equality rejects a
changed helper hash. The CPU receipt bypass remains unchanged. No generic
dependency traversal, optional feature enabling, config/API changes or numerical
changes are introduced.

Acceptance checks: copied actual helper bytes change the runtime hash with the
adapter disabled; a missing helper file fails runtime generation; base and
curriculum helper hashes agree; a complete default-recipe synthetic receipt
passes; receipt omission, mutual map omission and changed current hash fail.
Owner checks pass 9/9; the combined reproduction passes 59/59 with existing
continuous/curriculum guards, readiness guards and the independent Luna checks.
An actual copied-helper byte mismatch rejects continuous resume before
`torch.load`. A synthetic CUDA launch through the public readiness adapter
rejects a helper mismatch before optimizer execution, parameter mutation or
checkpoint publication; all hardware queries in this check are mocked.
The legacy readiness fixture has only `stub_math.py`; a scoped complete-source
fixture alignment passes its 11/11 existing guard tests. Its source is preserved;
integration must align historical synthetic fixtures with the stricter contract.

Reproduce from this worktree, using the project Python environment:

```sh
PYTHONPATH=src:tests:.:/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py /Users/pippo/github/binary-eagle-decoding/.venv/bin/python experiments/parallel20261002/activation_identity/reproduce.py
```

The JSON stdout is a source/test proof, never a measured readiness receipt.
Environment: macOS arm64, PyTorch 2.14, CPU, one Torch thread, tiny synthetic
models. No GPU/Metal/SSH, weights, captures, datasets or held-out/final evaluation.
Independent Luna commit `16e78b5` passes 5/5: actual helper hashing, mutually
omitted maps, complete receipt, malformed digest and curriculum resume rejection
before parameters/optimizer/RNG/phase/update restoration. Its report is
`independent-cpu-01.md`; raw log SHA-256 is
`6aac5942de82ec60222fcdb3d0f3237b40026d9553a31df3236daebc13bc2621`.
Implementation commit is `7808e02`. The imported feature files
`activation_reuse.py`, `learned_activation.py` and `native_step.py` are unchanged
byte-for-byte from `f027609`; `source-proof.json` records their hashes. Source
edits comprise seven lines in the three assigned inventory/readiness files.
Ruff and `git diff --check` pass. Integration requires root/QAT review plus
historical synthetic-fixture alignment; any future measured CUDA receipt must
be regenerated for the resulting source identity. This grants no live admission
and does not block the independently frozen simplest-recipe preparation.
No persistent process or allocated device remains after owner checks.
