# Independent validation: fusion binary A8 fitting

2026-10-03. Validation ran on Apple M3 Max, macOS 27.0 arm64, Python 3.11.15,
NumPy 2.4.6 and PyTorch 2.14.0. CPU thread limits were 2 for OpenBLAS, OpenMP
and Accelerate. No CUDA, Metal, GPU, remote host or SSH work was used.

## Local TRAIN operand eligibility

No local package contains prompt-disjoint raw F32 inputs at the pre-A8 fusion
boundary. I did not treat the following local artifacts as fitting operands:

- `results/binary-scale-fitting-5080/artifact-index.json`
  (SHA256 `ccb64b1eeb32df6b3be69b666aa6299f9e37117ff1abbd8e99a5bf1c4782d744`)
  says complete TRAIN captures and source/target models remain in the remote
  archive. The local copy contains derived scale arrays, development traces and
  sampled replay evidence.
- `results/binary-scale-fitting-5080/results/scale-validation/manifest.json`
  (SHA256 `ea5e3568f206dcf5a0f219ca5eddf9392f9ed7fa268818f07b41c2dd4a2f670f`)
  selects 16 sampled operator binaries from the single TRAIN prompt
  `qat-revisit-train-prose-urban-waterways-01`. Its `fc.weight` input is 32 rows
  from the `cast_only` A16 boundary, so it cannot supply A8 pre-quantization
  inputs or an independent prompt-held validation split.
- `results/recurrent-binary-cpu-smoke-20260928-a16/bundle/manifest.json`
  (SHA256 `52a6f718922c15c46c7bfaa2a2754795322652f7a7ee0d975a21af19c2f51649`)
  is marked `split=train` but `training_eligible:false`; required native model
  identity, target feature parity, mask/cache parity and request completeness
  gates are unverified. Its diagnostic step report
  (SHA256 `76d28ac1ae0137e078121e633a858c9395e888733beb67b880aaf34bc7c2ad63`)
  records one SGD step and two positions, not a prompt-disjoint fusion fit and
  validation package.

The original BF16 draft weights and F16 base GGUF are present locally and
hash-checked: `58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e`
and `c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1`.
The minimum missing real-input package is 32 raw F32 fusion rows, 16 each from
two different TRAIN prompts assigned to fit and validation. For K=7,680 that is
983,040 bytes (960 KiB) of inputs, plus row IDs, prompt content hashes, split,
depth/position, per-row input hashes and a capture manifest binding the rows to
the native source/target/runtime/precision ancestry. The existing frozen source
weights supply the reference projection. A more useful follow-up package is 8
fit prompts plus 4 validation prompts with 32 rows each (11.25 MiB of inputs).
No sealed final data was read.

## Independent checks

The six independent tests passed against feature script SHA256
`6327b78820f172764e1b575808c1993beabd75a229ebfd911a1bfc89d40a649d` and
configuration SHA256
`e921e3afb88e98e4c6e0ef009394ba50b53ee03845961f93bd16d458088cd4cb`.
They cover A8 round-to-nearest-even, clipping and zero rows; ordered F32 output
products; positive signs for both positive and negative source zero; exact
sign packing/reload; nonnegative row-scale solution and continuous/finite
neighbor convergence; a stale-flip conflict; validation freeze; real-input
refusal; and fusion-only GGUF export/reload with unchanged nonfusion tensors.

The conflict fixture has two stale proposals that each lower finite SSE from
112,500 to 92,500, while applying both raises it to 152,500. The fitter examined
both proposals and accepted one after checking each against the current exact
finite objective. In the end-to-end test, changing only held-out prompt inputs
left the frozen sign/scale candidate hash unchanged and changed validation
metrics.

The owner’s separate deterministic synthetic fit used 192 rows, 64 inputs, 8
output rows and 6 fit / 2 validation prompts. On the separate synthetic
validation prompts, relative squared reconstruction error fell from 0.2143827
for converged scale-only fitting to 0.00761152 for sign-and-scale fitting;
coordinate sign agreement rose from 85.42% to 97.92%. This establishes a gain
only on synthetic frozen-layer reconstruction. It is not a real TRAIN result,
native acceptance result or throughput claim. The fusion-only v2 GGUF candidate
SHA256 is `0f8f50340112aed33443fe6b6b0614b8cfb3d641e9644b45fdce26e6478b27f7`;
export/reload retained intended packed signs, F32 scales and all three
nonfusion tensors. The separate CPU native A8 replay checked 1,536 outputs with
zero absolute and relative error on Apple M3 Max; this validates serialization
and arithmetic on CPU, not SM75 performance or acceptance.

## Reproduction and run record

The tests were run from `/private/tmp/eagle-fusion-discrete-validator` against
the isolated fitter worktree. Raw stdout and Ruff output are retained outside
Git in
`results/fusion-binary-discrete-a8-20261003/independent-validation/`.

```sh
EAGLE_FUSION_FIT_SCRIPT=/private/tmp/eagle-fusion-discrete-fit/scripts/fit_fusion_binary_discrete.py \
EAGLE_GGUF_PY=/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py \
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
/Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest -v \
tests.test_fusion_binary_discrete_independent

/Users/pippo/github/binary-eagle-decoding/.venv/bin/ruff format \
tests/test_fusion_binary_discrete_independent.py
/Users/pippo/github/binary-eagle-decoding/.venv/bin/ruff check \
tests/test_fusion_binary_discrete_independent.py
```

Result: 6/6 unittest cases passed in 0.492 seconds; Ruff reported all checks
passed. The test’s temporary synthetic fixtures were removed at test teardown.
The console logs remain in the ignored run directory above. Their SHA256 values
are `9f5fa68244fcba2964edecdb7e22f83d2b4db121489f3765b85bdb6b93faa9ca` for
`unittest.log` and
`82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18` for
`ruff.log`.

The owner’s reproducible synthetic fit was run with the same bounded CPU
environment from `/private/tmp/eagle-fusion-discrete-report`:

```sh
cd /private/tmp/eagle-fusion-discrete-report

EAGLE_GGUF_PY=/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
/Users/pippo/github/binary-eagle-decoding/.venv/bin/python \
scripts/create_fusion_binary_discrete_fixture.py \
--output-dir /Users/pippo/github/binary-eagle-decoding/results/fusion-binary-discrete-a8-20261003/final-fixture-v2

EAGLE_GGUF_PY=/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
/Users/pippo/github/binary-eagle-decoding/.venv/bin/python \
scripts/fit_fusion_binary_discrete.py \
--manifest /Users/pippo/github/binary-eagle-decoding/results/fusion-binary-discrete-a8-20261003/final-fixture-v2/synthetic_manifest.json \
--base-gguf /Users/pippo/github/binary-eagle-decoding/results/fusion-binary-discrete-a8-20261003/final-fixture-v2/synthetic_base.gguf \
--output-dir /Users/pippo/github/binary-eagle-decoding/results/fusion-binary-discrete-a8-20261003/final-fit-v2
```

The resulting report SHA256 is
`2b70b46645f62109c410eff22204f639e0649d3ff8c83268c9f27f41e8b5e376`; the
synthetic operand archive SHA256 is
`fac62645c845ae812d6b74c945da69a8e21712c193295e4fe3cd85940a91f29b`. The
fit preserves only the synthetic inputs and candidate/report artifacts in the
ignored experiment directory. Native acceptance and throughput remain deferred.
