# Device-aware reporting CPU correction

Deliverable: `_reporting_scalars` now reads CPU scalars directly with `Tensor.item()`.
CPU values skip detach, device/dtype group lists, tensor stacks, host-list conversion,
and the final dictionary reconstruction. Non-CPU values retain the existing detached
stack/host-copy/list conversion per `(device, dtype)` group. No update math, sign
snapshot removal, objective, finite checks, optimizer ownership, or gate ordering changed.

Acceptance checks passed on Apple M3 Max, macOS 27 arm64, PyTorch 2.14.0, CPU only:

- Owner's six tests pass. Actual helper dispatch census records exactly eight
  `aten._local_scalar_dense` reads for eight tensors; zero stack, detach, or tensor
  allocation operations. Pinned `93a2ff1` helper records two stacks and eight detaches.
- Actual fixed A4 and learned A4/affine binary steps match the pinned reporting helper
  exactly for metric values/types/order, parameters, gradients, AdamW state/moments,
  recurrent cache gradients, and state gradients. Before-update operation census is
  identical. Existing snapshot removal remains in place.
- CPU boolean, int64 above 2^53, F16, BF16, F32, F64 and native Python scalars keep
  exact values/types and input order. CPU tensors requiring grad are read without
  creating a gradient graph.
- Device-routing stubs prove CUDA device 0/1 and MPS metadata form distinct groups;
  mixed dtypes remain separate. These use real CPU scalar payloads and a stubbed stack.
  No CUDA/Metal discovery, allocation, synchronization, or device arithmetic occurred.
- Existing reporting safety validation, recurrent QAT and curriculum suites combine
  with owner tests for 23/23 passes. Ruff and `git diff --check` pass.

## Bounded CPU timing

Nine trials of 10,000 calls, rotated ordering after warmup, eight already-computed
CPU scalars (six float32, two int64), one PyTorch thread. Median time per call:

| Conversion | Microseconds |
| --- | ---: |
| Original `float`/`int` scalar reference | 1.292 |
| Pinned grouped helper at `93a2ff1` | 6.592 |
| Actual device-aware helper | 1.101 |

Candidate trials range 1.086–1.121 microseconds. This resolves the observed tiny
CPU helper regression on this hardware. It is neither a complete trainer latency
measurement nor evidence for CUDA, Metal, SM75, or native acceptance/throughput.
Non-CPU grouped reporting remains a proposal without actual hardware validation.

Reproduce from the isolated worktree with the main project's virtual environment:

```sh
PYTHONPATH=src:. /Users/pippo/github/binary-eagle-decoding/.venv/bin/python \
  -m unittest tests.test_parallel20261002_device_reporting \
  tests.test_step_metrics_feature_validation tests.test_recurrent_qat \
  tests.test_qat_curriculum -v
PYTHONPATH=src:. /Users/pippo/github/binary-eagle-decoding/.venv/bin/python \
  -m experiments.parallel20261002.device_aware_reporting.proof
```

`summary.json` records hardware, source SHA256, raw timing samples, exact metrics,
and complete actual-step operation censuses. No weights, datasets, captures or raw
training runs are included.

## Integration checkpoint

Branch `research/20261002-device-aware-reporting`, base `93a2ff1`, worktree
`/private/tmp/eagle-parallel-20261002/device-aware-reporting`. Only helper source,
unique owner tests, proof and this report belong to this owner. Independent Luna
owns a separate validation test/report, with its results reported separately.
The orchestrator should checkpoint this outcome in the existing active goal file;
shared goal/source integration remains its ownership.

Existing `tests/test_step_metrics_feature.py` calls the historical step-metrics
probe whose CPU census assumes grouped `.tolist()` extractions and lower scalar
read count. Those mechanism assertions are obsolete for the CPU correction and
need an integration update by their owner/root. Do not interpret that old census
as a numerical or safety failure. This team leaves those historical files intact
under its assigned file boundaries. No live training adoption occurred.

Research control checked before each chunk: original reset 1791049896, no stop
latch/reset observation, 7% remaining on latest recorded check. No persistent
process, GPU job, Metal resource or SSH session is owned.
