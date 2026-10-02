# Exported composed function audit

The actual joint checkpoint and exporter preserve the declared composed forward
on the bounded all-nine synthetic fixtures: **81/81 projection comparisons
match exactly**, maximum absolute error `0.0`. No remedy patch is needed for
these cases. This closes the CPU serialization/function gate; it does not establish
native backend correctness, acceptance, or Q4_0-relative speed.

Owner: `/root/export_function`; independent Luna: `/root/export_function/decode_validation`.
Isolated worktree `/private/tmp/eagle-parallel-20261002/export-function`, branch
`research/20261002-export-function`. CPU only, Torch `2.14.0`, one CPU thread.
No models, captures, GPU, Metal, SSH, native build, real optimizer updates, or
production source changes were used.

## Pinned contract and predeclared gate

The parent source baseline is `ca02272ae4c5740077c96eb2b3993435c814f444`.
The native graph was read only from main's initialized submodule at
`9e2c7a90051e738751aab7d7bd7c2d8201fb76e3`. Exact source and per-case
checkpoint/export SHA256 values are committed in `result-summary.json`.
The worktree's uninitialized submodule was not populated or mutated.

Before fixture execution, both owner and validator were given `atol=3e-5`,
`rtol=3e-6`, compared elementwise against the actual grad-enabled training hard
outputs. Integer sums/dots are exact in these tiny shapes. The gate allows
bounded F32 reduction drift in the two raw correction matmuls; it does not
allow changed codes, row order, scale placement, or factor precision.

Actual path: `save_joint_checkpoint` writes schema5 NPZ/manifest; unmodified
`export_model` reads those and creates GGUF; `GGUFReader` supplies all values
to the independently written NumPy evaluator. The evaluator receives only a
GGUF path, projection base name, and raw input. It never receives the live
linears, checkpoint arrays, parameter bank, or manifest. The validator uses a
second decoder rather than importing the owner's evaluator.

The native encoder graph at `eagle3.cpp:491` dispatches affine packing;
`eagle3.cpp:515` adds raw fusion correction after the binary/affine core,
then correction bias. The decoder dispatch at `eagle3.cpp:576` shares the
same affine operation. `ggml-cpu.c:1646` and `:1686` apply separately rounded
midpoint products after the separately rounded dot products. Declared rules:

- A1: `beta=F32(mean_F64(abs(x)))`; threshold `F32(delta*beta)`;
  sign of `F32(x-threshold)`, ties and both signed zeros positive. Delta zero
  bypasses subtraction. Negative nonzero subnormals keep their sign.
- A4/A8: limit `F32(absmax(x)*clip)`, beta `F32(limit/qmax)`;
  round-even codes from `F32(x*F32(qmax/limit))`, symmetrically clamped.
  Reciprocal overflow uses the versioned F64 division fallback; zero limit
  yields zero codes. Underflow can give nonzero codes with zero beta.
- Signs: little-endian I32 words, logical width read from GGUF, tail bits zero;
  latent zero, including negative zero, packs positive. Exported Q/K rows are
  inverted using independent index enumeration, with all three row fields
  (signs, scales, midpoint) returned to original checkpoint order.
- Affine core: `(F32(D*alpha))*beta + (F32(S*mu))*beta`, with each
  multiplication/addition kept separate. No combined-scale reassociation.
- FC: `latent=raw_F32 @ V_F16_as_F32.T`,
  `delta=latent_F32 @ U_F16_as_F32.T`; add delta to core, then F32 bias.
  The intermediate and input are F32. No activation quantization occurs on
  the correction branch. F32 masters round to F16 before export and forward.

The pinned EAGLE graph's original nine projections are bias-free. The nonzero
bias exercised here is the explicitly exported fusion correction output bias.
This audit does not extend the declared graph with generic frozen linear biases.

## Fixtures and falsification

Nine actual exports cover A1/A4/A8 at logical widths33/65/130, ranks1/4, and all
nine projection names. Each input has five rows: mixed ordinary F32 values,
positive constant values, signed-zero/tie-like values, an all-zero row, and
minimum positive/negative F32 subnormals. Q128/K32 output rows give head dimension4,
so the Q32/K8 permutation is genuinely nonidentity. Scales include a zero row;
row midpoints and all six learned activation scalars are nondefault; correction
U/V and bias are nonzero. Maximum allocation is a128×130 synthetic latent.

Every case independently mutates one actual serialized `fc.w1ax_midpoint`
F32 element by `+0.125`, using its GGUF byte offset. Decoding the changed bytes
changes the FC output by `0.6832031` through `5.0781245`. This demonstrates the
oracle consumes exported auxiliary contents. It is a synthetic semantic
negative control, not a model-loader admission or content-authenticity claim.

Using the unrounded F32 correction masters instead of the declared rounded
factors changes the correction by `1.14576e-6` through `1.63794e-4` depending on
width/rank. The fixtures therefore detect the precision contract. The unique
unit checks also independently assert nonidentity Q/K index order and signed
zero, positive A1 threshold ties, and subnormal A4/A8 overflow fallback codes.
This work deliberately does not repeat existing malformed metadata/hash inventory
or large model-shaped export tests.

## Reproduction and durable artifacts

Use the existing project executable; no environment was installed or changed:

```sh
PYTHONPATH=src:.:/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest discover -s tests -p test_parallel20261002_export_function.py -v
PYTHONPATH=src:/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/pippo/github/binary-eagle-decoding/.venv/bin/python research/parallel20261002/export_function/reference/audit.py --output /a/new/ignored/output/directory
```

The owner unique suite passed3/3; Ruff and whitespace checks passed. Final raw
NPZ, manifests, base/export/mutated GGUF and `index.json` are retained outside
Git at main `runs/parallel20261002/export-function-final/`, which survives
worktree cleanup. An initial aborted fixture directory and one earlier equivalent
successful run are also retained; final report identities use only the final
index. The initial abort was an owner harness metadata-key typo, corrected to
the actual underscored GGUF field naming before any claimed results.

Independent results and decoder live in `validation-report.md` and
`research/parallel20261002/export_function/validation/` when complete.

Integration: review/cherry-pick the owner and validator evidence commits into
main, record the result in the active QAT goal, push, then remove the isolated
worktree/branch after preserving commits. No production remediation or recipe
change follows from this bounded result. Current-source native validation and
all actual-model training admission gates remain with the QAT owner.
