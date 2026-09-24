# Pinned drafter linear coverage before CUDA timing

**Status:** static checkpoint audit; no measured latency or native binary speed.

The local pinned `AngelSlim/Qwen3-4B_eagle3` safetensors header at revision
`fd331e59626c8e95c392381a16ee59d518727fbb` contains nine candidate
BF16 drafter-owned linear weights. Their shapes match
`experiments/eagle3-graph-audit.md`. The target's shared token embedding is
excluded. Counts below are exact products of checkpoint matrix dimensions;
MiB assumes two bytes per BF16 parameter and excludes metadata.

| Group | Linears | Parameters | BF16 MiB | Share of these linear parameters |
| --- | ---: | ---: | ---: | ---: |
| Feature fusion | 1 | 19,660,800 | 37.50 | 9.0% |
| Attention Q/K/V/O | 4 | 41,943,040 | 80.00 | 19.2% |
| FFN gate/up/down | 3 | 74,711,040 | 142.50 | 34.2% |
| Drafter vocabulary head | 1 | 81,920,000 | 156.25 | 37.5% |
| **Total** | **9** | **218,234,880** | **416.25** | **100%** |

These weights are candidate binary operands under the project's selective
W1A1 simulation. The surrounding embedding lookup, normalization, attention
softmax, residual arithmetic, cache operations, tree selection, and scale
application remain ordinary operations. Parameter share does not predict draft
time share: the head is called at different token shapes from the other groups,
and launch, packing, and non-linear work matter. The RTX 5080 profiler must
measure actual calls and input shapes before choosing coverage.
