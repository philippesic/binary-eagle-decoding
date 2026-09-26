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
