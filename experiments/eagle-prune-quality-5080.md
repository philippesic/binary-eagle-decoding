# EAGLE unused-head pruning: native quality A/B on RTX 5080

On 2026-09-28/29 UTC, the first Phase 1B supervised native A/B compared
`GGML_EAGLE_PRUNE_UNUSED_HEAD=0` and `1` with the same target, drafts,
24-prompt development workload and server policy. The opt-in branch skips the
decoder output norm, head and vocabulary expansion for eligible zero-logit
cache catch-up calls. It is shared by Q4_0, fitted D and FP16 EAGLE.

## Pinned execution

- Host: NVIDIA GeForce RTX 5080, SM120, WSL, CUDA UMD 13.4; native llama.cpp
  gitlink `14c188e`, parent source at `8ae610d` before the quality run's
  docs-only fast-forwards. Built `llama-server` SHA256
  `a57f9e784eb2528d2de954d12ebcb9ff57c1ee24cd8b11e3df55e526125e52f0`.
- Target F16 SHA256 `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`;
  Q4_0 draft `2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280`;
  fitted D group-128/A16 `10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf`;
  FP16 draft `c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1`.
  Prompt JSONL SHA256 `a3b97d942a99f1bddd5bb97216c32a9920aaa50788baa5bdb92354842547e885`.
- Configs: [pruning off](../configs/w1_phase1b_prune_off.json) SHA256
  `38e44ad17c5a32c1fd7d0d52d13d618d8254a238d94dfdbe46980233d6d842c1`,
  [pruning on](../configs/w1_phase1b_prune_on.json) SHA256
  `ad69c379def326c169d00cbb6750553841fddc0dbf9e066862931b0d9e68e8eb`.
  They differ only in the prune switch. CUDA graphs remained enabled.
- The pinned SM120 rebuild exited zero. A separate CUDA backend operator run
  passed **112/112** W1A1 matrix cases over A1/A4/A8/A16 and group-128/A16.

The off and on runs each used one quality repetition: 24 measured requests
per variant, plus 2 warmups per variant. Each job ran through the tmux MCP
session and `scripts/remote_job.py`, with its own process group and preserved
ignored directory. Both supervisor states finished with exit code zero.

## Paired result

The [offline comparator](../scripts/compare_eagle_prune_quality.py) checked
all **96 measured request pairs**. It required equal generated raw token IDs,
completion text hash, finish reason, speculative counters, aggregate quality
counts and each round's proposed/emitted IDs, accepted count and status.
There were **zero mismatches**.

| Variant | Rounds | Proposed | Accepted | Emitted | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_0 EAGLE | 1,493 | 7,320 | 1,555 | 3,048 | 3,072 |
| D group-128/A16 | 2,139 | 10,449 | 909 | 3,048 | 3,072 |
| FP16 EAGLE | 1,496 | 7,329 | 1,552 | 3,048 | 3,072 |
| Target-only | 0 | 0 | 0 | 0 | 3,072 |

The totals reproduce the earlier archived development result. Q4_0 remains
the primary acceptance comparison; D is substantially behind it on these
untrained histories. The old 24 prompts have been used repeatedly and are a
regression workload, not an untouched promotion set.

Ignored remote evidence under `~/binary-eagle-decoding/runs/`:

- `w1-prune-off-quality-20260928/benchmark/manifest.json` SHA256
  `0dbd6996f0d57a17c96ae5c0a02da1c13994cd13b5d129d2958c6857d462c1c5`.
- `w1-prune-on-quality-20260928/benchmark/manifest.json` SHA256
  `55c2c133809bd353a56240a41c1e1b6e2a01893231e38ea8095aa6d5519d959d`.
- `w1-prune-ab-compare-20260928/report.json` SHA256
  `891e8cfb979752cb49c77f5c3fa130142a2d40f568e94abfe5dc84ab079c5f63`.

The device showed sustained external load while project jobs were stopped:
55–87% utilization, roughly 4.1–4.8 GiB whole-device use and up to 323 W,
with no WSL GPU process listed. Request times from these quality runs are
therefore **not** a fair speed A/B. Exact cache bytes were not captured; the
paired token/round result does not prove cache-state equivalence across all
contexts. Keep pruning opt-in pending a bounded cache/state check and clean
timed run. No SM75 performance conclusion follows from this RTX 5080 result.
