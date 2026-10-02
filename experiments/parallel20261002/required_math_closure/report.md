# Required math-source closure

Active goal: [QAT optimization readiness](../../../docs/goals/qat-optimization-readiness.md).
Candidate base: `5a8cd778fc8451e411244ea124ac61f33d2991e2`.
Worktree: `/private/tmp/eagle-parallel-20261002/required_math_closure`.
Branch: `research/20261002-required_math_closure`.

Deliverable: fail closed on incomplete declared training-source inventories at
measured CUDA receipt admission, using the actual producer registries. Acceptance:
complete default/enabled/curriculum receipts pass; mutual omissions, missing
sources, hash disagreement and unknown precision stages fail; resume rejects
changed runtime before restoring parameters or optimizer state.

## Actual contract probe

Before the patch, `ActivationIdentityTests.validate` constructed a complete
`training_runtime_identity("cpu")` math inventory under explicit CPU CUDA-context
stubs. Mutually removing `learned_activation.py` or `recurrent_qat.py` from context
and receipt passed the public `validate_optimization_readiness` call. The prior
candidate protected only `activation_reuse.py` presence. Normal producers emit
complete maps, but the callable consumer did not require them.

Only `qat_readiness.py` is changed. It reads the literal existing
`continuous_runtime.MATH_FILES` and `qat_curriculum_runner.EXTRA_MATH` registries
without importing Torch or executing either module. This is a two-registry
literal reader, not a Python import resolver. Each required key must be present
and each supplied digest must remain valid hexadecimal. Existing exact
context/receipt comparison rejects differing hashes. The caller still owns
fresh runtime identity; this is not cryptographic producer attestation or an
independent digest audit of arbitrary caller-supplied content.

Curriculum configuration (`curriculum` or `stages`) requires its separate
`curriculum_math_sha256` map; a supplied curriculum map is also checked even when
no schedule is selected. Declared precision stages must be a nonempty list of
A8/A4/A1 entries. There is no new stage identifier or Git-HEAD requirement.
CPU admission still returns before inventory or receipt access.

## CPU checks and fixture issue

Owner checks: `PYTHONPATH=src:tests python3 -m unittest -q
 test_required_math_inventory test_parallel20261002_activation_identity
 test_qat_curriculum_runner` passes **33/33**. The four new owner tests include
subtests for all 16 continuous and all seven curriculum inventory keys, complete
default/enabled contexts, and a mismatched receipt digest.

The unmodified `test_qat_readiness` fixture is deliberately a stub-only
`{"stub_math.py": ...}` map and already conflicts with the prior activation
helper guard. Running its 11 tests unmodified gives four errors and one failure.
Replacing only its fixture map in memory with actual producer inventories gives
**11/11 passing**, including complete curriculum receipts. Existing files were
not edited and the inventory guard was not weakened. The older independent
activation-identity test likewise calls a helper-only map "complete"; it requires
fixture modernization if included in integration checks.

`git diff --check` passes. `activation_reuse.py`, `learned_activation.py`,
`native_step.py`, `continuous_runtime.py`, and `qat_curriculum_runner.py` remain
byte-for-byte unchanged from the candidate base.

Independent Luna [checks](independent_report.md) pass **4/4**: authentic complete
receipt accepted, mutual omissions rejected, A2 curriculum rejected, and changed
runtime resume rejected before parameters or optimizer state are restored.
Combined owner/independent/candidate regression suite passes **37/37**.
No GPU, Metal, SSH, model weights, captures, finals or reset credits were used.
CPU synthetic receipts establish source/API behavior only.

## Integration handoff

Review and cherry-pick this isolated source change onto the intended candidate.
The live/main/frozen `bb0d304` QAT path was not modified. Link this report and the
commit into the active goal checkpoint when integrating; shared goal files were
left to the orchestrator to avoid conflicting edits. Modernize legacy fixtures
with full inventories as a separate owned integration step if needed. This
change invalidates older incomplete receipts; produce fresh measured evidence
before adopting changed source for CUDA training.

Usage observations before cost chunks: 97% used / 3% remaining in the original
window, reset `1791049896`. Research must stop at <=1% remaining or window reset.
