# Python research code

`recurrent_qat.py` provides the joint nine-linear W1Ax training boundary. Its
row-scale simulation supports A16 (F16 boundary cast), A8/A4 (per-token F32
absmax, nearest-even codes), and A1 (positive zero sign, F64 mean-absolute
scale cast to F32). Forward weight signs are hard at every step. Dense F32
matmul carries those hard values and is not a native speed measurement.

Weight signs use clipped identity gradients for latent values within [-1, 1];
the trainer projects them into that interval after updates. Activations use
identity gradients through hard dequantized values with dynamic scales detached.
AdamW uses separate sign/scale rates, zero weight decay and gradient clipping.
Metrics include actual sign flips, gradient coverage, scale movement, latent
clip violations and activation saturation. Tiny CPU recurrence checks ensure
last-position loss reaches the first state and K/V cache.

`scripts/train_joint_w1ax.py` runs a synthetic three-position CPU fixture and
can save a row checkpoint. Real training passes an audited native
exact-prefix rollout to `joint_train_step`; the fixture is not captured data or
model inference. Its `--device` option requires `--allow-accelerator` outside
CPU, and accelerator use remains disabled for the current Phase 1A work.

The existing `GroupedBinaryLinear` and schema-v1 exporter remain the separate
group-128/A16 candidate-D reference. Row checkpoints use schema v2 with explicit
scale layout, activation bits, quantizer rules and original Q/K row order.
`scripts/export_recurrent_binary.py` accepts both schemas and writes row GGUF
with activation-width metadata pinned. Its CPU serialization audit checks frozen
tensors, packed signs, scales and metadata. Native loader and trajectory
validation remain required before a row GGUF can be used as a result.

Compact teacher distillation reads unconditional full-target top-k mass, mapped
draft tail and outside-draft mass. It spreads tail mass uniformly across other
draft tokens and conditionally normalizes by mapped mass. This is an explicit
approximation, not full target KL. The upstream data loader must verify row
prefix ancestry and manifest hashes before supplying these tensors.
