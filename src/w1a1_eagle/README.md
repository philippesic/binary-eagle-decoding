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

`scripts/train_joint_w1ax.py` runs a synthetic three-position CPU fixture by
default and accepts an audited provider with `--provider`. The fixture is not
captured data or model inference. Its `--device` option requires
`--allow-accelerator` outside CPU; the current provider path explicitly
requires CPU while a separate accelerator rollout is pending.

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

## Captured native-prefix training provider

`recurrent_provider.py` defines the executable CPU training boundary. An
importable `MODULE:FACTORY` passed to `scripts/train_joint_w1ax.py --provider`
must return a `JointTrainingProvider`. The provider loads the pinned official
drafter and frozen target, returns the original dense drafter for row training
or candidate-D packed arrays for group128/A16, and constructs a
`NativeStepAdapter` after the nine linears are installed. It must bind frozen
native norms and borrowed embedding before that adapter is constructed.
The provider must declare `training_eligible=True` from a frozen replacement
training manifest; the inherited 96-prompt smoke capture is ineligible.
The factory should only assemble configuration; `load_models()` is called
after the eligibility gate.

Each yielded `ProviderRound` carries its audited round anchor, exact accepted
prefix, native target-feature rows, proposal trace and capture ID. For compact
probability training, the provider first calls
`scripts.compact_w1a_teacher.iter_verified_shards` with expected prompt,
target-GGUF and d2t hashes plus current row ID-to-prefix mapping. It then
passes valid rows, teacher IDs and four mass arrays in the same order. The
trainer repeats the trace audit, checks teacher prompt/capture/prefix/label
binding, rebuilds the current student context cache and retains proposal
state/K/V gradients. Invalid terminal rows cannot borrow teacher mass.

Once an eligible captured provider module exists, the explicit CPU invocation
is:

```sh
PYTHONPATH=src:scripts:$PROVIDER_DIR python3 scripts/train_joint_w1ax.py \
  --provider w1ax_capture_provider:create_provider \
  --scale-layout row --activation-bits 4 --objective hard_ce \
  --device cpu --seed 0 --steps 100 \
  --output-dir runs/joint-w1a4-cpu-gate
```

`$PROVIDER_DIR` must contain the project-specific factory; this repository
does not yet have a larger eligible native capture or a model-backed factory.
Actual target capture, long QAT and native acceptance/throughput measurements
wait for restored GPU access. The current provider runner executes only CPU
rollout; future accelerator providers need a separately validated cache path.
