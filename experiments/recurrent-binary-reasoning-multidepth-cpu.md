# Five-depth reasoning CPU forward parity

The sealed first reasoning round contains five native draft proposals at
memory positions 46–50. This diagnostic starts from the audited native
46-position context K/V cache, uses the captured raw target feature for
the seed, and follows the captured input tokens 1654, 1477, 264, 3070,
79. Candidate-D projections run in ordered A16/W1 arithmetic; the pinned
ggml CPU helpers supply attention, vector SiLU and query RoPE. The
corrected pre-norm state is passed into the next depth, so errors would
propagate through the actual recurrent input. No backward or optimizer
step is run.

| Checked boundary across five depths | Bitwise exact |
| --- | ---: |
| New F16 key writes | 5,120/5,120 |
| New F16 value writes | 5,120/5,120 |
| ggml query RoPE F32 | 20,480/20,480 |
| ggml attention output F32 | 20,480/20,480 |
| FFN output F32 | 12,800/12,800 |
| Normalized head state F32 | 12,800/12,800 |
| Captured head-logit probes | 40/40 |

All five mapped argmax IDs match native (1477, 264, 3070, 79, 79), as do
each captured argmax and verifier-label logit bitwise and each
verifier-label rank. The eight-probe check per depth is limited to those
captured logits; full mapped-vocabulary logit parity is not claimed.

The native ggml RoPE helper is the same hashed eight-head operator used
for the [context key replay](recurrent-binary-reasoning-rope-oracle.md).
Four eight-head blocks per query give 32 query heads at the same position.
Each of the five helper outputs matches the captured native
`Qcur_rope-0` in all 4,096 F32 values. With the adapter's Python RoPE
instead, depths 0–3 still have exact attention and head state, but depth
4 differs at 128/4,096 attention outputs and leaves only 11/2,560 head
state values exact. Three F16 query thresholds differ at depth 4; the
change at head 1, channel 52 alone restores the native attention,
downstream state and captured head-logit checks. The ggml operator
restores them by replaying all query values, without a captured-value
patch in the five-depth parity path.

The ignored machine report is
`results/recurrent-rope-oracle-20260928/multidepth-cpu.json` in the main
checkout, SHA256
`3259d01947dc5196b0f5eb46dd6cf02d4154e00d88165f90183cdde908bda455`.
It verifies the frozen capture, candidate D, first-seed control and
hashed operator helpers, and records per-depth exact counts, max/RMS
errors, cache writes, captured logit checks and the depth-4 ablation.
The real-input run and Ruff lint/format passed on Apple M3 Max CPU.

This is one native-token-forced, five-depth CPU trajectory. It starts
from captured native context cache bytes; the preceding cache and RoPE
reports validate their ordered-computation reconstruction on the same
prefix. The diagnostic native attention has an F32 surrogate backward,
and this run did not execute or approve that derivative for training.
The default student forward remains unchanged. No CUDA/SM75, complete
96-prompt trajectory, final-set or Q4_0 result follows from this check.
