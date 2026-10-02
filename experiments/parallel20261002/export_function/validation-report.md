# Independent serialized decoder validation

The independently written NumPy decoder reproduced every serialized projection
from the GGUF fields and raw fixture inputs: 9 A1/A4/A8 cases, 81 projection
outputs total, and zero values outside `atol=3e-5, rtol=3e-6`. The maximum
absolute error was exactly `0.0` for every case. This is CPU synthetic
serialization evidence, not a native model or GPU result.

The validator reads packed little-bit-order sign words, row scales, learned
activation metadata, optional affine midpoints, and optional fusion correction
tensors directly from GGUF. It reconstructs the Q32/K8 inverse row mapping for
signs, scales, and midpoints. A1 uses F64 mean-absolute reduction rounded to
F32, the F32 threshold and positive tie rule; A4/A8 use the serialized learned
clip ratio, round-to-even codes, and per-row activation scale. It calculates
integer dots in `int64`, then applies the serialized row scale and activation
scale in separate F32 steps. Affine midpoint sums are separately scaled before
addition. The optional fusion residual uses serialized F16-rounded factors,
cast back to F32 for the two independent F32 reductions, followed by its F32
bias. The script imports neither the exporter nor any training forward code;
it imports only the llama.cpp GGUF reader.

The owner also reports a midpoint-byte mutation negative control: all nine
mutated exports changed their forward outputs. That mutation check is separate
from this decoder run.

Run details:

- Worktree: `/private/tmp/eagle-parallel-20261002/export-function`
- Branch: `research/20261002-export-function`
- Run directory: `/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/export-function-final/`
- Device: local macOS CPU; no GPU, model weights, captures, builds, or SSH used.
- Python: `3.11.15`; NumPy: `2.4.6`; GGUF reader: `/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py`.
- Fixture index SHA-256: `77b2f6cfc2d075217004f23ce75a2a703cbfe4442c12ae1aca1167fdc41b074c`.
- Raw validation JSON: `independent-validation.json` in the run directory; it contains all per-case maxima and violation counts.

Exact command:

```sh
PYTHONPATH=/private/tmp/eagle-parallel-20261002/export-function/src:/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py \
  /Users/pippo/github/binary-eagle-decoding/.venv/bin/python \
  research/parallel20261002/export_function/validation/decode_serialized.py \
  /Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/export-function-final/index.json \
  --report /Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/export-function-final/independent-validation.json
```

Result: exit status `0`; `passed=true`; 9/9 cases; 0 violations. There was no
persistent process or temporary artifact outside the ignored run directory.
Raw fixtures and validation output are retained there for reproducibility.
