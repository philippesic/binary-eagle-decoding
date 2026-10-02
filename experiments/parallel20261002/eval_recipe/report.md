# Development evaluator recipe construction proof

The current development evaluator rejects every deployed optional recipe tested,
after native A8/A1 and Q4_0 acceptance captures. The failure is a construction
mismatch: `_evaluate_development` uses default fixed `JointQATConfig`, installs
ordinary row modules, then invokes the actual strict `_load_checkpoint`.
The loader validates already-attached learned activation, fusion correction and
affine modules; it does not create them. This is an evaluation readiness defect,
not evidence about acceptance, latency, throughput or model quality.

## Real producer and consumer evidence

CPU-only synthetic checkpoints use actual `save_joint_checkpoint`, actual
`install_joint_linears`, actual `_load_checkpoint` and the existing
`checkpoint_joint_config`. No loader mock or native evaluator is executed.
Each named profile is taken from `configs/qat_optimization_profiles.json`.
Only deployment-affecting keys construct its modules; optimizer, learning-rate
and scheduling controls do not belong to zero-update inference replay.

| Profile | Saved schema | Default development construction |
| --- | --- | --- |
| baseline | 2 | passes, identical deployed state |
| learned-activations | 3 | unknown or mismatched attached activation quantizer |
| fusion-rank1 | 4 | checkpoint fusion correction differs from attached config |
| fusion-rank4 | 4 | checkpoint fusion correction differs from attached config |
| affine-fusion | 5 | checkpoint affine midpoint coverage differs from attachment |
| affine-all | 5 | checkpoint affine midpoint coverage differs from attachment |
| combined-contract-smoke | 5 | unknown or mismatched attached activation quantizer |

These results hold for both A8 and A1: 14 checkpoint constructions, 12 failures
with the old default and 14 successful manifest-aware reconstructions. A4 is
explicitly rejected by this paired evaluator prototype. Curriculum/A4 development
support remains a separately declared gap.

`results.json` records actual exception text, schema, hashes and deployed
recipe fields for every case. The source and replay deployment fingerprints
match exactly for all 14. All 126 projection outputs and all 14 depth-two recurrent
logit arrays match exactly (maximum absolute error 0). The declared numerical
gate remains atol3e-5/rtol3e-6. Recurrent logits use the actual CPU
`NativeStepAdapter` and `forward_torch_round`, with synthetic export-compatible
Q64/K16, hidden4, intermediate5 geometry, frozen deterministic F16 embeddings,
causal cache, norms, RoPE and all nine modules. This is structural Torch replay;
it does not validate native CUDA/Metal/SM75 execution or real-model trajectories.

## What identity is sufficient

The deployed manifest carries every required inference attachment:

- `activation_quantizers`: six shared boundary identities, bits and effective
  F32 threshold/clip scalars; absence represents fixed activations only in a
  valid schema contract.
- `fusion_correction`: rank 1/4, named U/V arrays, optional bias, bias bound and
  declared arithmetic. NPZ U/V are effective F16 deployment values. Replay
  expands those into F32 parameters, preserving their rounded values; it cannot
  reconstruct original optimizer masters. The fixture makes that difference
  observable. Effective F32 output bias is clamped at export.
- `affine_weights`: coverage (`fusion` or `all`), arithmetic and exact named F32
  midpoint inventory. Live replay preserves original checkpoint Q/K row order.
- All nine projection shapes/names, precision, objective, base GGUF hash,
  checkpoint hash, scale/weight/activation rules and row order.

A profile name is neither necessary nor sufficient for deployed forward
reconstruction. Learning rates, optimizer state, training midpoint bound, latent
magnitude and execution optimizations are absent because this is deployment
replay, not training resume. Frozen embeddings/norms and model/corpus provenance
remain protected by existing provider/source gates; this adapter does not replace
them.

There is an important identity boundary. A learned-only checkpoint stores
quantizer scalars in JSON, not NPZ. Rewriting schema 3 as a self-consistent fixed
schema 2 manifest can pass bare NPZ/manifest validation. The test demonstrates
this, then rejects it with the producer's existing publication identity.
`ContinuousTrainer.save` already publishes `manifest.json.exports`, with SHA256
of each lane's `joint.npz` AND `joint.json`. Pair preflight requires that exact
A8/A1 inventory and verifies every file hash before reading deployment recipes.
Missing publication identity, missing recipe fields, changed JSON/NPZ,
undeclared arrays, unsupported bits and clips below 2^-16 fail before capture.
This uses trusted local producer publication; it does not claim cryptographic
protection against someone rewriting both checkpoint and its entire receipt.

## Narrow proposed integration

`reference/adapter.py` provides the research-only pure API.
`preflight_pair` verifies publication identity and both lane contracts/arrays
before the caller proceeds. `construct_and_load` installs the existing
manifest-derived configuration and invokes the real loader; changed identities
are rejected before installation.

`reference/development-preflight.patch` is an **unapplied** proposal for
`scripts/w1ax_continuous_stages.py`. It reuses `checkpoint_joint_config` rather
than inventing a second recipe constructor. It moves paired publication/array
preflight before export/native capture, replaces default QAT construction with
the prepared recipe plus the existing explicit CUDA device flags, and rechecks
file identities before export and model construction. `production_preflight.py`
is the exact proposed pure helper extracted for independent CPU execution.
`make_proposal.py` regenerates it and the patch; `source-identity.json` binds all
relevant source bytes plus patch hash. `git apply --check` passes at this branch's
source. Core files and live recipes are unchanged.

Preflight uses actual exporter validators. It reads effective arrays and creates
packed sign/scales temporarily, then releases them before full model loading;
it does not allocate a second model or load `resume.pt`. This adds host staging
at an offloaded-training boundary. Full-shape host admission must be reviewed
with the memory ledger before production adoption; the CPU proof makes no
full-model memory-fit claim. Provider/capture/held-out and Q4_0 semantics remain
unchanged. A new runtime/source identity is required if the proposal is adopted.

## Checks and acceptance

Owner: six unittest cases pass on Python3.11.15/Torch2.14.0/NumPy2.4.6 CPU,
including actual source-order contract, seven named paired recipes, fixedschema 2
equivalence, actual recurrent logits, F16 factor restoration, exact extracted
production helper, missing identity, downgrade and post-preflight mutation.
Independent Luna validation is recorded in `validation.md`; its final authoritative
run is the current source check. Ruff, format, patch applicability and whitespace
checks pass. No native acceptance or performance result is claimed.

```sh
PYTHONPATH=src:scripts:tests:research/parallel20261002/eval_recipe/reference:/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py \
  /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest discover \
  -s research/parallel20261002/eval_recipe/reference -p 'test_*.py' -v
```

The bounded research deliverable is complete once independent validation is
committed. Remaining integration work belongs to the orchestrator/QAT owner:
review the unapplied source proposal, bind a fresh runtime, complete resource
admission and validate actual native/model development under authorized ownership.
No live recipe change, optimizer update or final-set access is authorized here.
