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
EAGLE. Optional anchor GGUF files provide separate same-input operator records,
not a decoded-trajectory comparison.

Build against the current checkout (on the 2080 Ti set the CUDA architecture):

```sh
cmake -S kernels/w1ax-replay -B build/w1ax-replay -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_METAL=OFF
cmake --build build/w1ax-replay -j 8
build/w1ax-replay/w1ax_operator_replay --gguf models/gguf/Qwen3-4B-eagle3-all-w1a1.gguf \
  --fp16-gguf models/gguf/Qwen3-4B-eagle3-F16.gguf \
  --fp16-cast-control \
  --q8-gguf models/gguf/Qwen3-4B-eagle3-Q8_0.gguf \
  --q4-gguf models/gguf/Qwen3-4B-eagle3-Q4_0.gguf \
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

`--act-bits 1`, `4`, `8`, or `16` replays only that W1Ax mode. Omitting it
preserves the default 16/8/4/1 matrix. Every JSONL record includes
`requested_act_bits` (`[16,8,4,1]` by default, or a one-element array), while
operator records retain their actual `replay_bits`. Single-mode runs omit
W1Ax-to-W1A16 head comparisons because the other outputs were not computed;
`w1ax_head_comparison_available` makes that explicit on operator records.
Optional anchors and the ordinary FP16 cast control still execute normally.

For separate CUDA software traces that disambiguate the shared A8/A4 kernel
symbols, omit anchors from each profiling run. For example, after obtaining GPU
ownership and using the project's remote supervisor:

```sh
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  -o runs/<id>/operator-w1a4 \
  build/w1ax-replay/w1ax_operator_replay \
  --gguf models/gguf/Qwen3-4B-eagle3-all-w1a1.gguf \
  --capture-dir runs/<capture-id>/captures --backend gpu --act-bits 4 \
  --warmups 2 --samples 5 --require-nine \
  > runs/<id>/operator-w1a4.jsonl
```

Run A8 separately with `--act-bits 8` and a distinct report/output name. Unset
`GGML_W1AX_CAPTURE_DIR` first. This traces CUDA API calls and GPU work without
requesting CPU sampling, scheduling traces, or GPU hardware counters (see the
[Nsight Systems CLI guide](https://docs.nvidia.com/nsight-systems/UserGuide/)).
The trace includes correctness calls and warmups as well as timed samples;
keep those invocations separate when analyzing kernel counts. Profiled timings
are diagnostics and do not replace uninstrumented throughput measurements.

The three anchor flags are optional and independent. For each supplied GGUF,
the replay maps `fc.w1a1_packed` to `fc.weight`, and likewise for the other
eight names. It requires the captured K and M, no batch dimensions, and exact
weight type F16, Q8_0, or Q4_0. It invokes native `ggml_mul_mat` with F32
captured activations on the selected backend. It never substitutes a dense
dequantized matmul for timing. Every `anchor_operator_replay` JSONL record
contains the same `capture`, `sequence`, `name`, `group`, K/M/N, `source_bits`,
backend, synchronized full-graph timing label, `min_us`, `median_us`, `p95_us`,
and raw `samples_us` as W1Ax records; `anchor_format` and `weight_type` keep
anchors distinct from `replay_bits`. All output values must be finite.
`finite_outputs` counts the checked M×N values. A sampled reference separately
dequantizes weight rows **outside** timing and dots them with the captured F32
activations; `reference_rows`, `reference_outputs`, and
`reference_max_abs_error`/`reference_max_rel_error` describe that check. This
reference is deliberately **non-gating**: ggml's native quantized CUDA path
can quantize activations (for example Q8_1) before the dot, so its output need
not match an F32-activation scalar dot. The `validation` field says
`finite_output_gate_with_sampled_f32_activation_reference_non_gate`.
The anchors share capture identity and shape with W1Ax rows for later pairing,
but their GGUF weights differ. They establish matched operator input, not
identical decoding trajectory or numerical equivalence of models.

`--fp16-cast-control` requires `--fp16-gguf`. It runs the ordinary FP16-weight
operator twice on each capture: once with the original captured F32 input, and
once after an explicit F32→FP16→F32 activation cast. Both outputs are computed
and compared outside timed samples. The original F32 input is restored before
the ordinary FP16 anchor's warmups and timing. A `fp16_cast_control` record
contains capture identity, layer/group/K/M/N, the two activation contracts,
output count, and mean/max **absolute** output difference. For the full
32,000-row head, `fp16_cast_head_comparison` adds one record per token with
top-1 agreement, top-5 set overlap count, both top-1 token IDs, top-1 minus
top-2 margins, and rank-5 minus rank-6 cutoff margins. These head outputs are
ordinary draft logits; the other layers' values are intermediate activations.
Every cast record has `serving_acceptance_metric: false`: it isolates the
operator-boundary cast and does not measure acceptance or decode trajectory.

The scalar check validates final F32 values and independently computes integer
dots for A1/A4/A8; it cannot directly inspect the native kernel's hidden
integer accumulator. The existing backend operation tests are the exact-dot
gate. This tool currently measures the complete operator only. Substage CUDA
events and already-packed timing require additional instrumentation inside the
CUDA op; do not infer a breakdown from this total.
