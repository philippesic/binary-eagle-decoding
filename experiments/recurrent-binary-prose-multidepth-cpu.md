# Independent prose-prefix CPU forward parity

The frozen prose training prompt supplies a 31-position draft context and
a five-depth first proposal round. This replay builds the entire context
K/V cache from the captured raw target-feature rows, borrowed F16 token
embeddings, candidate-D ordered A16/W1 projections, native-style RMS
norm, and the pinned ggml CPU key RoPE graph. It does **not** inject
captured native cache bytes into the student. The constructed context
matches the sealed native graph and actual stored cache:

| Context boundary | Bitwise exact |
| --- | ---: |
| Token embedding and both normalized inputs | 79,360/79,360 each |
| Fused input | 158,720/158,720 |
| Raw K, raw V and rotated K F32 | 31,744/31,744 each |
| Stored F16 context K and V | 31,744/31,744 each |

The replay then follows the captured input tokens 32, 3070, 3070, 8926,
334. At every depth it appends its own F16 K/V row, runs ordered binary
projections and the pinned ggml CPU query RoPE, Flash Attention and
vector SiLU, and passes its computed pre-norm state into the next depth.
The following all match the native capture bitwise across five depths:

| Proposal boundary | Bitwise exact |
| --- | ---: |
| New F16 key and value writes | 5,120/5,120 each |
| Query RoPE and attention output F32 | 20,480/20,480 each |
| FFN output and normalized head state F32 | 12,800/12,800 each |
| Captured head-logit probes | 40/40 |

All five native mapped argmax IDs are reproduced (3070, 3070, 8926,
334, 32). Each captured argmax logit and verifier-label logit matches
bitwise; verifier-label ranks match 2, 2, 1, 8 and 38. Full mapped target
logits were not captured, so this check is limited to those sampled
head-logit values and discrete decisions.

The capture is the sealed Apple M3 Max CPU prose training diagnostic,
manifest SHA256
`4ad342145a1b77deb5f48cac4e00671e26b691850f398be56687ef05830b7169`.
The ignored machine report is
`results/recurrent-rope-oracle-20260928/prose-multidepth-cpu.json` in
the main checkout, SHA256
`e1896769591a8c85a9fb8ed418f12ee2624bc801409eced39e84e0e3d9bdae60`.
The probe verifies every manifest file, native stored-cache graph join,
model identity and pinned helper hashes before the replay, and enforces
exact context, cache, state and sampled-logit gates. The real-input run
and Ruff lint/format passed. No GPU, model server, optimizer, target
forward, final prompt or Q4_0 evaluation ran.

This adds an independently constructed second prefix to the earlier
[reasoning five-depth CPU replay](recurrent-binary-reasoning-multidepth-cpu.md).
Both are first-round, native-token-following CPU diagnostics. They do
not establish post-acceptance cache scheduling, all 96 training
trajectories, CUDA/SM75 parity or a safe optimizer tolerance. The native
attention path retains an F32 surrogate backward that was not executed
or selected for training.
