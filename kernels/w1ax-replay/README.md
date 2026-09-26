# Captured-input W1Ax operator replay

The replay reads `GGML_W1AX_CAPTURE_DIR` files from a short diagnostic server
run and the **same all-nine packed GGUF** for every precision. It runs the native
`ggml_w1ax_mul_mat` operation on every selected capture, checks deterministic
sampled rows against independent scalar arithmetic, and writes one JSON object
per capture and precision. The measured span includes ggml graph dispatch,
temporary allocation in the CUDA op, activation quantization/packing, dot,
rescale, and backend synchronization. It excludes loading, transfer, and the
scalar reference. The latency is synchronized host wall time, not isolated CUDA
kernel event time. This distinction is recorded in each result.

Each `operator_replay` JSONL record adds an `activation` object, computed before
timing. `elements` counts all K×N inputs. `source_zero_rate` includes both
signed zeros. `code_zero_rate` is the fraction of zero FP16 values or A8/A4
integer codes; it is `null` for A1 signs. `clip_rate` counts FP16 overflow or
A8/A4 codes outside the signed range *before* clamping; it is `null` for A1.
The per-token absmax quantizer usually has zero clipping by construction, so
`saturation_rate` separately counts A8/A4 codes at either range endpoint; it is
`null` for A16/A1. `mae`, `rmse`, and `max_abs_error` compare each reconstructed
activation to its captured F32 source. A1 reconstruction uses its mean-absolute
token scale. An FP16 overflow is counted as clipping and excluded from the
finite reconstruction error sum; the numerical parity gate will fail on a
nonfinite operator output. Aggregate these records by `name`/`group` and shape
for per-layer diagnostics.

For every captured `output.w1a1_packed` operator, a `head_comparison` record is
emitted for each token and candidate precision (8, 4, 1). It requires all
32,000 rows and compares the **full native output vectors** to W1A16 using the
same binary weights. Fields include `reference_bits`, `candidate_bits`,
`top1_agree`, integer `topk_set_overlap` counts for k=1/5/10, both top-1 token
IDs, each top-1 minus top-2 score margin, both top-1 scores, and cross-scores
at the other mode's top-1 ID. Ties rank by lower token ID. These are draft-head
scores, not target/verifier logits. There is no ordinary FP16 head in the packed
GGUF, so this comparison does **not** estimate agreement with ordinary FP16
EAGLE; that requires a separate matched dense-head replay.

Build against the current checkout (on the 2080 Ti set the CUDA architecture):

```sh
cmake -S kernels/w1ax-replay -B build/w1ax-replay -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_METAL=OFF
cmake --build build/w1ax-replay -j 8
build/w1ax-replay/w1ax_operator_replay --gguf models/gguf/Qwen3-4B-eagle3-all-w1a1.gguf \
  --capture-dir runs/<id>/captures --backend gpu --warmups 2 --samples 5 \
  --require-nine \
  > runs/<id>/operator-replay.jsonl
```

Use `--backend cpu` for local correctness smoke checks. `--check-rows 0` checks
every output row; the default checks eight distributed rows plus the first and
last rows for each token. `--limit N` bounds a smoke run. Capture files must be
complete and are checked against their GGUF weight dimensions. Only one file
per observed invocation is used; the emitted `sequence`, layer/group, `N`, and
source precision allow a shape and invocation histogram. CUDA graph capture
must be disabled while collecting inputs because the capture hook synchronizes
the stream; keep collection separate from timing and unset
`GGML_W1AX_CAPTURE_DIR` before replay. `--require-nine` checks coverage of the
nine selected layers. Preserve raw files and GGUF
hashes in the experiment manifest.

The scalar check validates final F32 values and independently computes integer
dots for A1/A4/A8; it cannot directly inspect the native kernel's hidden
integer accumulator. The existing backend operation tests are the exact-dot
gate. This tool currently measures the complete operator only. Substage CUDA
events and already-packed timing require additional instrumentation inside the
CUDA op; do not infer a breakdown from this total.
