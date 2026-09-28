# Integrated native CPU diagnostic forward

An explicit `native_cpu_diagnostic` mode now runs pinned ggml CPU RoPE,
Flash Attention and vector SiLU inside `NativeStepAdapter.decode_step`.
It requires ordered candidate-D binary projections, the fixed EAGLE
geometry, the hashed helper binaries, Apple arm64 and `torch.no_grad()`.
It refuses a gradient-enabled call. The default F32 student and the
existing attention-only surrogate mode retain their previous behavior;
this diagnostic supplies no selected training derivative.

The integrated adapter used its existing `rebuild_prefix_cache` and
ordinary `decode_step` calls, without captured-cache injection or
per-stage substitutions, on three sealed training cases:

| Case | Context F16 K/V exact, each | Draft depths | Normalized head state exact | Captured logit probes exact |
| --- | ---: | ---: | ---: | ---: |
| Prose first round | 31,744/31,744 | 5 | 12,800/12,800 | 40/40 |
| Reasoning first round | 47,104/47,104 | 5 | 12,800/12,800 | 40/40 |
| Reasoning post-acceptance round | 50,176/50,176 | 3 | 7,680/7,680 | 24/24 |

All 13 depths also match every captured graph tap checked by the runner,
including pre-RoPE Q/K/V, rotated Q/K, attention output, FFN input and
output, and pre-norm state. All 13 new K/V writes are bitwise exact.
Mapped argmax IDs, captured argmax/label logits and verifier-label ranks
match on every depth. These are captured samples; full mapped-vocabulary
logits were not checked. The post-acceptance case reconstructs its
49-position context through the sealed cache rewrite history, including
the accepted-draft catch-up at position 48.

The ignored machine reports in the main checkout are:

| Case | Report SHA256 |
| --- | --- |
| `results/recurrent-rope-oracle-20260928/integrated-prose-first.json` | `9b6744fde999fcd19f901bacd4dd9dee3005c6d66fe7b2118374d7ec230973a0` |
| `results/recurrent-rope-oracle-20260928/integrated-reasoning-first.json` | `8b358c1398b62bff05f5f9723510b1b0aa5d1191f447c6f6e852cc8d7636e1e8` |
| `results/recurrent-rope-oracle-20260928/integrated-reasoning-postaccept.json` | `a5595125ba24e95b13b6b0a565f71b66bc42a31293a315ee420caadc1ddd398a` |

Each report verifies the full capture file ledger, frozen target and
candidate-D identities, stored-cache audit, pinned operator hashes and
per-depth graph/head joins. All three real-input checks exited zero;
the new no-grad/pinned-geometry unit check and 62 native CPU
tests passed. Ruff lint and format checks passed. These results are on
Apple M3 Max CPU with the pinned F16 target model, captured raw target
features and W1/A16 candidate D; no GPU, optimizer, target forward,
final prompt or Q4_0 evaluation ran.

The path is a parity diagnostic, not a chosen QAT forward/backward pair.
It covers two first-round prefixes and one post-acceptance round, not
all 96 training requests, free-running serving trajectories or CUDA/SM75
execution. Exact versus numerical/trajectory acceptance and any
all-body optimizer budget remain user-owned.
