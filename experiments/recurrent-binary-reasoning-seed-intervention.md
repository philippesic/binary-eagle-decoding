# Corrected-cache first-seed CPU intervention

The sealed reasoning first round was replayed with candidate-D ordered
binary arithmetic on Apple M3 Max CPU. Two cache modes used the same
native-order seed input and pinned ggml CPU Flash Attention helper:
the adapter's corrected ordered context K/V with its Python RoPE (three
F16 key bits different), and the actual captured native K/V cache.
The baseline and FFN-stage recapture were checked for bitwise identical
first-seed input, output and head state before comparing operations.

| Boundary | Adapter RoPE cache | Native K/V cache |
| --- | ---: | ---: |
| Seed K/V F16 write | 1,024/1,024 each | 1,024/1,024 each |
| Attention output F32 | 3,968/4,096 exact; max 1.1444e-5 | **4,096/4,096 exact** |
| FFN input F32 | **2,560/2,560 exact** | **2,560/2,560 exact** |
| Torch-SiLU FFN output F32 | 5/2,560 exact; max 6.1035e-5 | 5/2,560 exact; max 6.1035e-5 |
| Torch-SiLU normalized head state F32 | 5/2,560 exact; max 5.9009e-6 | 5/2,560 exact; max 5.9009e-6 |

With actual native K/V, the pre-RoPE Q/K/V projections, FFN input and
post-attention norm are bitwise exact; small Q/K RoPE F32 differences do
not change the ggml attention output when the stored operands are exact.
Replacing only Torch SiLU with the pinned ggml CPU vector function in
that same ordered seed makes gate, up, SiLU, product and down all bitwise
exact (9,728 values at each intermediate and 2,560 at down). The
pre-norm and normalized head states then match **2,560/2,560** native
F32 values. Applying candidate D's ordered output head to that exact
state matches all eight captured draft-logit probes, the native argmax
and verifier-label logits bitwise. The mapped argmax is 1477 in both
paths, and the verifier label has rank 4 in both. Full mapped-vocabulary
logit parity was not captured or claimed.

The [key RoPE oracle](recurrent-binary-reasoning-rope-oracle.md) and
[cache-boundary report](recurrent-binary-reasoning-cache-boundary.md)
establish how the native cache operands relate to ordered student
arithmetic on all 47 visible positions. This intervention uses native
cache bytes at the seed; it does not install a production student cache
path or choose a backward derivative for the native operators. The
ignored machine report is
`results/recurrent-rope-oracle-20260928/seed-corrected-cache.json` in
the main checkout, SHA256
`54c095813d5ea704f5b152b59b25dfb2c2e90a7d790fd0f68f11a2cc4909d9ed`.
It includes manifest, model, helper, library and source hashes and each
stage's exact count and max/RMS error.

The real-input replay and Ruff checks passed. No GPU, optimizer, target
forward, reserved-final prompt or Q4_0 evaluation ran. This is one
first-seed Apple CPU intervention, not a recurrent trajectory or CUDA/SM75
parity result. The next useful check is a multi-depth CPU trajectory
with corrected native cache and activations, followed by a user-owned
choice of exact versus numerical/trajectory training tolerance.
