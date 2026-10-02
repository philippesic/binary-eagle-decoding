# Opt-in NativeStep activation reuse feature

The deliverable is maintainable, default-disabled source reuse of the *same*
attached learned-activation result for immediate Q/K/V and gate/up consumers.
It is confined to `native_step.py`, `learned_activation.py`, and the new
`activation_reuse.py`; no production method is monkeypatched. This is a source
capability for review, not a change to a selected QAT recipe or live job.

## Explicit capability and effective state

Construct `NativeStepAdapter(drafter, activation_reuse=True)` only when the
coordinating QAT owner explicitly admits this capability. Omission retains
`False`. No `JointQATConfig`, curriculum runner, evaluator, checkpoint schema,
optimizer, initializer, or native model precision changes are included.

`activation_reuse_state()` distinguishes the requested flag, currently eligible
shared learned boundaries, enabled groups, and counts for the last decoder
step. `last_decode_events` counts actual hits/misses and always reports zero
retained entries. Eligibility alone is not evidence of a runtime hit. Fixed
activations or distinct sibling quantizers have no effective reuse. Context
K/V-only chunk construction remains unchanged; only full decoder sibling groups
participate. The head remains on its existing serial learned-training fallback.

The scope requires exact input/quantizer identity, tensor and parameter mutation
versions, quantizer contract, mask identity/version, normalization count N, grad
mode, inference mode, autocast state/dtype, and training state. Result versions
guard an instrumented consumer that mutates returned activation data. Views,
changed masks/counts/parameters and unversioned inference tensors take separate
computations. Unsupported options retain ordinary validation. Cache memory is
bounded to one result per group. A `finally` block clears entries and scope-owned
input/quantizer references, restores context, and records tensor-free counts.
QKV exits before attention; gate/up exits before down projection. The raw FC
fusion-correction input and existing affine sum-sharing path are unchanged.

## Acceptance and source binding

The owner gate uses the actual NativeStep recurrent graph with learned A4,
affine midpoint rows, raw-input FC rank-1 correction, shared hard signs, detached
context chunks, three attached draft steps, and the serial learned head. It
requires exact logits and bounded all-family/input/cache VJPs (`atol=2e-6`,
`rtol=5e-5`). Quantization computations are 9→3 for attached QKV and 6→3 for
gate/up; unique STE support/derivative storages are 18→6 and 12→6. FC/head compute
counts are unchanged. Invalidation/cleanup and default-disabled state-dict
compatibility complete the owner's nine focused checks.

78 owner plus existing NativeStep/learned activation/affine/fusion/learned-head
tests pass on macOS arm64, Torch 2.14.0 CPU, with F32 arithmetic and F16 cache/
fusion factors. The first compatibility command had a `gguf` import error
because this isolated worktree has an empty llama.cpp submodule; the final
command uses the existing main submodule's Python reader as a read-only fixture
dependency. No dependency/source mutation was necessary. This is operation/
storage correctness evidence, not SM75 performance or native throughput evidence.

```sh
PYTHONPATH=src:tests:.:/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py \
  /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -c \
  'import torch,unittest; torch.set_num_threads(1); modules=["test_parallel20261002_activation_reuse_feature","test_native_step","test_learned_activation","test_affine_binary","test_fusion_correction","test_learned_head_batching"]; result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(modules)); raise SystemExit(not result.wasSuccessful())'
PYTHONPATH=src:tests:. /Users/pippo/github/binary-eagle-decoding/.venv/bin/python \
  experiments/parallel20261002/activation_reuse_feature/reproduce.py
```

`source-proof.json` binds the measured graph/count/storage evidence to exact
feature, dependency and fixture source SHA256 values. Independent validation is
recorded separately in `independent-validation.md` and its uniquely named test.

## Integration needs and remaining work

Review and merge only with root/QAT source-owner coordination. Do not enable
this optimization in existing live preparation/training. A future separately
owned config binding must add an explicit default-false capability, pass it to
the adapter constructor, record requested/effective state and source hashes in
the existing execution metadata, and update source-bound readiness receipts.
Observers need to forward that metadata if the runner consumes it. The adapter
flag is intentionally runtime-only and absent from weight state dictionaries;
restoring weights does not silently enable reuse. New runtime adoption needs
fresh native/device readiness and bounded same-device acceptance/throughput
gates, under the existing QAT owner's protocol.

Assigned feature checkpoint: implementation and owner acceptance complete;
independent validation pending at report creation. No persistent experiment
process, GPU/Metal/SSH/model/capture/final-set work, or live adoption. Root owns
the active goal checkpoint and will copy this deliverable into its goal record.
