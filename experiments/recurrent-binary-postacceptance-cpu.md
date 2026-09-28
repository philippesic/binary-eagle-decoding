# Post-acceptance reasoning CPU parity

The sealed reasoning training capture's third round follows a prior round
that accepted one draft. Its accepted prefix has 50 target tokens and
requires draft context positions 0–48 before the seed at position 49.
The native cache includes speculative writes and rewrites: position 47
was written three times, with token 525 from execution 8 retained;
position 48 was written four times, with accepted token 2661 from the
five-column catch-up execution 13 retained. This probe joins each
position to its **latest pre-seed physical cache write**, checks that
write's token against the accepted prefix, and reconstructs K/V from
the corresponding frozen target-feature row and candidate-D ordered
arithmetic. It does not inject captured native cache bytes.

| Rebuilt context boundary | Bitwise exact |
| --- | ---: |
| Token embeddings and both normalized inputs | 125,440/125,440 each |
| Fused input | 250,880/250,880 |
| Raw K, raw V and ggml-rotated K F32 | 50,176/50,176 each |
| Actual stored F16 context K and V | 50,176/50,176 each |

From that reconstructed cache, the three captured draft inputs 1447,
12 and 3070 feed a computed recurrent pre-norm state into the next
depth. Ordered W1/A16 projections and the pinned ggml CPU query RoPE,
Flash Attention and vector SiLU give bitwise parity at every checked
boundary:

| Proposal boundary | Bitwise exact |
| --- | ---: |
| New F16 key and value writes | 3,072/3,072 each |
| Query RoPE and attention output F32 | 12,288/12,288 each |
| FFN output and normalized head state F32 | 7,680/7,680 each |
| Captured head-logit probes | 24/24 |

Mapped argmax IDs match native (12, 3070, 362), as do the captured
argmax and verifier-label logits bitwise and label ranks (1, 1, 23).
The diagnostic checks the actual stored-cache mask and graph joins,
including the catch-up rewrite at position 48. Full mapped-vocabulary
logits were not captured or checked.

The Apple M3 Max CPU capture manifest SHA256 is
`99b9003698bcdf5505421a1667dac4d21a8545471d91330c04181c9f2a931d40`.
The ignored machine report is
`results/recurrent-rope-oracle-20260928/postacceptance-cpu.json` in the
main checkout, SHA256
`31acfdb2c62fb314bced1d4ed188db1855f38fe53ebdd4d02c5aa2a7bce45a4d`.
It records each latest-write source, rewrite count, file/helper hashes,
and per-stage exact/error metrics. The real-input replay and Ruff
lint/format passed. No GPU, model server, optimizer, target forward,
reserved-final prompt or Q4_0 evaluation ran.

This extends the [reasoning first round](recurrent-binary-reasoning-multidepth-cpu.md)
and [independent prose first round](recurrent-binary-prose-multidepth-cpu.md)
CPU diagnostics through one accepted-draft cache catch-up. It remains a
single frozen training prompt and a diagnostic native forward with an
unselected F32 surrogate derivative. It does not set a safe training
tolerance or establish full 96-prompt, CUDA/SM75 or free-running parity.
