# Independent validation: development recipe preflight

The recipe preflight passed six independent CPU checks against the research
adapter, source-bound pair-preflight helper, and actual checkpoint
producer/loader. The checks use a tiny EAGLE fixture and do not run the native
evaluator. The main evaluator source still needs to apply the proposed patch to
call pair preflight before exports or captures; this report validates the
helper contract and demonstrates the present ordering gap.

The source path is concrete: `_evaluate_development` currently invokes native
capture before constructing a default fixed `JointQATConfig`. The checkpoint
producer writes schema 2 for fixed deployment state, schema 3 for learned
activation parameters, schema 4 for fusion correction, and schema 5 for affine
midpoints. `checkpoint_joint_config` reconstructs the inference-affecting
attachments from that manifest. `_load_checkpoint` then loads the serialized
effective tensors into those attachments. Its F16 fusion U/V checkpoint values
are copied into F32 PyTorch parameters after F16 rounding; those are the
effective deployment values, not the higher precision training masters.

The independent checks exercised fixed A8 schema 2 and combined A1 schema 5
producer→preflight→`install_joint_linears`→actual `_load_checkpoint` paths.
Both reconstructed deployment-state SHA256 digests matched their source
modules, and the real `forward_torch_round` path through a tiny depth-two
`NativeStepAdapter` produced matching synthetic logits. The independent test
also ran the exact extracted source-bound pair helper on both modern A8/A1
lanes; both state digests and recurrent graph logits matched within
`rtol=3e-6, atol=3e-5`. The combined fixture verified NPZ U/V arrays are F16,
restored module values equal `source.to(F16).to(F32)`, and the pre-rounding
source tensor differs from that effective value. Adversarial checks rejected a
missing schema-5 affine descriptor, an A8 learned clip ratio of `2**-17`, a
checkpoint byte change after preflight, and a manifest byte change after
preflight. Both identity mutations failed before module installation. The
consumer-order check published the producer identity envelope, corrupted the
A1 recipe, then called pair preflight before a simulated capture action; the
export hash mismatch stopped the action.

A separate downgrade check showed that a learned activation schema-3 manifest
can be relabeled as fixed schema 2 with the same NPZ and pass standalone checks
because its six learned scalars live in JSON. Paired preflight rejects that
downgrade against the parent `manifest.json` emitted by
`ContinuousTrainer.save`, which binds hashes for both lane files. The separate
owner fixture covers all seven named profiles at A8/A1. These checks make no
model-quality claim.

This validates the proposed preflight API contract; it does not claim the
production evaluator is already wired to the adapter. The proposed production
patch is preserved in `research/parallel20261002/eval_recipe/reference/` and
bound to its source dependency hashes there.

Exact authoritative validation command, run directory, and environment:

```sh
PYTHONPATH=src:scripts:tests:research/parallel20261002/eval_recipe/reference:/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py \
  /Users/pippo/github/binary-eagle-decoding/.venv/bin/python \
  research/parallel20261002/eval_recipe/validation/independent.py \
  --run-dir runs/parallel20261002/eval_recipe/validation-independent-09 \
  > runs/parallel20261002/eval_recipe/validation-independent-09.stdout 2>&1
```

- Worktree: `/private/tmp/eagle-parallel-20261002/eval-recipe`
- Branch and source HEAD: `research/20261002-eval-recipe`, `67e259bad99ec953ba9bf1f3d636cbb8ebf71579`
- Runtime: Python 3.11.15, PyTorch 2.14.0, NumPy 2.4.6
- Result: **6/6 passed**, exit status 0, in 0.27 seconds. Ruff check and format also passed on the independent validator.
- Ruff commands: `/Users/pippo/github/binary-eagle-decoding/.venv/bin/ruff format research/parallel20261002/eval_recipe/validation/independent.py` and `/Users/pippo/github/binary-eagle-decoding/.venv/bin/ruff check research/parallel20261002/eval_recipe/validation/independent.py`; both exited 0 after formatting.
- Raw outputs: ignored `runs/parallel20261002/eval_recipe/validation-independent-09/` and sibling `.stdout`; synthetic NPZ/JSON files and the machine-readable result are retained.
- Cleanup: no model, native capture, accelerator API, SSH, or long-lived process was created. Scratch files are limited to tiny CPU fixtures retained in the ignored run directory.
- Superseded runs: ignored directories `validation-independent-01` through `validation-independent-08` preserve earlier exploratory outputs and test-harness errors. Run 01's report used `torch.cuda.is_available()` and is excluded; the authoritative script/run 09 makes no accelerator API calls.

Source identities at validation time:

| File | SHA256 |
| --- | --- |
| `research/parallel20261002/eval_recipe/reference/adapter.py` | `f7f2bad7753f5caccc35dc48cc6264c30b03e3112a26398cee69b84e9871a493` |
| `research/parallel20261002/eval_recipe/reference/test_recipe.py` | `d45470815a8029be04863dd268c584f58da39b912d0765fbad007586fdacc8ac` |
| `research/parallel20261002/eval_recipe/reference/production_preflight.py` | `98550b21d52b3aecef0cdffe9ee377f17fe9ecec3ea758e1696ab93925671ffc` |
| `research/parallel20261002/eval_recipe/reference/development-preflight.patch` | `eb2c01d8fab482df4f65cd6a90262bf367561aa5bd910f310a19aecb85eefbcf` |
| `research/parallel20261002/eval_recipe/validation/independent.py` | `d557868a291f8ff073f9aeaaac55492ef03a297e2deacc8c5746a9654ed14654` |
| `scripts/w1ax_continuous_stages.py` | `cf50bf98874ffcbe6fe7a970d54cfc60a77d73e23da282a6cc41df84f196ce47` |
| `scripts/check_continuous_w1ax_readiness.py` | `eddc45abc962971c0bf16f52ff5d4c74fce80d82bd109451d3daa0b964fe199e` |
| `src/w1a1_eagle/recurrent_qat.py` | `f2b6d83b8929807b65b6f21826cc99097a6ddd70f99ff23ca0836aebe4db1224` |
| `scripts/export_recurrent_binary.py` | `2232bf29d25324ca50e8b72e7308d8bb445485793c387fc2af155d4fbe081a2d` |
| `configs/qat_optimization_profiles.json` | `30b50e1a47751a77042c4828bfbd3f3d226d1776e0e20cf12ab66f1280ec0c72` |

The earliest exploratory runtime probe accidentally called
`torch.cuda.is_available()` under system Python/Torch 2.8 and returned `False`;
the first report draft also made that read-only query. Those runs are excluded.
The authoritative run above makes no accelerator API call; the probe incident
did not allocate a model or start a capture.
