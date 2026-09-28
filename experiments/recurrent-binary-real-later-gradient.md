# Real-size later-only EAGLE gradient check

**2026-09-28 UTC.** Parent commit `ba96343` adds a CPU-only diagnostic on
the audited one-prompt frozen training bundle. It loads pinned candidate D
with all nine hard-binary W1A16 linears, rebuilds the accepted-prefix cache,
and unrolls the first two valid, supported proposal positions. The only
loss is cross-entropy on **depth 1**; depth 0 has no direct loss. No optimizer
step, export, new model inference, GPU run or quality measurement occurred.

On Apple M3 Max (PyTorch 2.14.0, 10 CPU threads), the depth-1 label was draft
ID `7146` under the pinned absolute D vocabulary map. Depth-1-only CE was
`2.106572151184082`. The earlier proposal's pre-norm state and the K/V
cache tensors passed into the later proposal retained their autograd links:

| Depth-0 tensor | Nonzero gradient values | Gradient L2 |
| --- | ---: | ---: |
| Pre-norm state | 2,560 / 2,560 | 0.375924 |
| Appended F16-rounded key row | 1,024 / 1,024 | 0.295697 |
| Appended F16-rounded value row | 1,024 / 1,024 | 0.128654 |

Depth-0 draft logits received **zero** gradient, confirming the first row
did not contribute a direct CE term. All nine binary sign and scale parameter
groups had finite nonzero gradients in the shared two-step graph; their
individual counts are in the report. Those parameter gradients may include
the second step's own use of shared weights, so they do not alone prove a
depth-0 path. The retained depth-0 state and appended K/V gradients provide
that causal evidence. Six synthetic/policy tests passed, including rejection
of a deliberately detached earlier key cache; Ruff and formatting passed.

The ignored [report](../results/recurrent-real-later-gradient-20260928/report.json)
has SHA256
`295c22adfae8b1bd0102769eddd0e5b51a0be6bf3c0807ef7d0ef3166101dc33`.
It records source and bundle hashes, labels, masks, map hash, all nine
gradient counts and norms, hardware and elapsed time. The audited bundle
manifest SHA256 is
`52a6f718922c15c46c7bfaa2a2754795322652f7a7ee0d975a21af19c2f51649`;
the pinned D absolute map SHA256 is
`03d2f0e3420175955c14a43a1c1dda360cde26c8a03318ffb1a013bb06d0fe0a`.

This closes the real-size **later-only causal-gradient path** for one
captured proposal pair. The current student still uses F32 attention and
does not have whole-drafter native numeric parity; this gradient check does
not establish a training tolerance, optimizer budget, live acceptance or
Q4_0 throughput.
