# QAT checkpoint host-save admission validation

Bounded read-only CPU validation of commit `53c2eaae5bf2d6c55962f6172ed23e2fd359354f`
on branch `feature/qat-host-save-admission`. This validates admission
arithmetic, CPU clone storage, and failure ordering. It does not validate an
actual CUDA transfer, fitted training run, host capacity, or GPU performance.

## Arithmetic and code review

`_cpu_tree` recursively copies every tensor value occurrence using
`detach().cpu().clone()`. The helper mirrors that value traversal: it counts
each tensor occurrence in retained CPU clone bytes, including repeated aliases,
and adds the size of the largest CUDA tensor transfer because that one
intermediate overlaps its retained CPU clone during sequential evaluation.
CPU tensors add no transfer bytes. Dictionary keys and non-tensor metadata are
preserved by `_cpu_tree` and are not counted as tensor buffers.

The configured-shape test reports 2,619,863,112 retained bytes and a largest
transfer of 327,680,000 bytes. With the unchanged 16 MiB workspace allowance,
admission adds 2,964,320,328 bytes above the existing configured host floor.
The test's sequential-copy peak is 2,947,020,836 bytes, so the new bound covers
it by 17,299,492 bytes. The previous retained-plus-workspace boundary would be
310,380,508 bytes below that peak. The floor itself remains the configured
`min_host_available_bytes`, and the prior floor-additional arithmetic in
`require_host_memory` is unchanged.

The `_cpu_tree` implementation is unchanged. The diff changes only the helper
and the `save()` admission estimate; it does not touch resume, training math,
optimizer moments, or the moment-probe path. The old-boundary test supplies
available memory equal to floor plus the former estimate, verifies rejection
before `_cpu_tree`, then verifies no checkpoint pointer or file was published.
The exact-new-boundary test checks the preserved floor and new additional byte
count before deliberately stopping at the copy call.

The configured-size `CUDATensorMetadata` values are backed by PyTorch `meta`
tensors; their overridden device descriptor reports `cuda:0` for arithmetic
only. No CUDA allocation or device discovery is performed. Separately, the
storage-calibration test uses actual small CPU tensors: `_cpu_tree` creates
three independent CPU clone storages for three occurrences (including aliases)
and the counted total equals their real storage bytes. This is not a configured
model fit or transfer measurement.

## Environment, hashes, and command

- Worktree: `/private/tmp/eagle-qat-host-save-admission`.
- Commit: `53c2eaae5bf2d6c55962f6172ed23e2fd359354f`.
- Interpreter: `/Users/pippo/github/binary-eagle-decoding/.venv/bin/python`,
  Python 3.11.15, PyTorch 2.14.0, macOS 27.0 ARM64.
- Thread limits: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, and `torch.set_num_threads(1)`.
- `torch.cuda.is_available` was patched to raise if called during the complete
  run.
- Raw output and hash snapshots:
  `/Users/pippo/github/binary-eagle-decoding/runs/qat-host-save-validation-20261002/`.

Before and after SHA-256 values matched:

| File | SHA-256 |
| --- | --- |
| `src/w1a1_eagle/qat_curriculum_runner.py` | `20d4c195fce2bdf4425e7121b617c50604f578693381be3fec7c33b49109943f` |
| `tests/test_qat_checkpoint_host_admission.py` | `cb4521e90f1f9a9aee1adc72130fc336e4ee4978c081df4bf9334f864ceec45f` |
| `tests/test_qat_curriculum_runner.py` | `50960df8621e207cee1173b153a252a030cee3e1c587c992ddc7e07bf9d3ea53` |
| `tests/test_qat_curriculum_readiness_tool.py` | `4a87778c1988171632be656ade7c861a7334a44e8f59675598477f182264310e` |
| `tests/test_continuous_resources.py` | `d2ca946618e9a5b2b0d103401cbfd13f60937cb0b758a60d73b8971de9b6f63d` |

The exact one-pass command was:

```sh
cd /private/tmp/eagle-qat-host-save-admission
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src:.:tests \
/Users/pippo/github/binary-eagle-decoding/.venv/bin/python -c 'import torch; torch.set_num_threads(1); import unittest; from unittest.mock import patch; patterns=("test_qat_checkpoint_host_admission.py", "test_qat_curriculum_runner.py", "test_qat_curriculum_readiness_tool.py", "test_continuous_resources.py"); loader=unittest.TestLoader(); suite=unittest.TestSuite(); [suite.addTests(loader.discover("tests", pattern=p)) for p in patterns]; guard=patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA discovery forbidden in CPU validation")); guard.start(); result=unittest.TextTestRunner(verbosity=2).run(suite); guard.stop(); raise SystemExit(not result.wasSuccessful())'
```

Result: **42 tests passed in 1.457 seconds, exit code 0.** This includes all six
new admission tests and the requested existing curriculum runner, readiness,
and continuous resource suites. No CUDA discovery assertion fired.

## Cleanup

The test process exited. Hashes remained unchanged and the feature worktree had
no source or test changes afterward. Only this validation document is new in
the worktree. Raw output is `tests.log`; before/after snapshots are
`hashes-before.txt` and `hashes-after.txt` in the ignored run directory. No
GPU, SSH, actual model data, or real training was used. No source edit or commit
was made.
