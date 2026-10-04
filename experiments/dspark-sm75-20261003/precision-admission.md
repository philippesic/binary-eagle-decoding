# Released DSpark/DFlash precision capability admission

Bounded local worker checkpoint, October3,2026. Parent branch
`feature/dspark-native-admission`; native source remains published
`76847aa877261817026e2f29472f080235b6a829`. No remote/GPU access, native edits,
training, QAT, sealed-final access or other-team changes occurred. Root owns
the screen and cumulative7200-second measurement budget; Luna owns actual GPU
operation. Reference phase runs first. Phase2 is the supported FFN-only Q4_0
comparison, with target-only and Q4_0 EAGLE anchors on the same RTX2080Ti.

## Capability verdict

| Requested condition | Existing DSpark/DFlash capability | Action |
| --- | --- | --- |
| Released BF16 | Existing admitted converter/loader/graph | References continue under root/operator |
| FFN-only Q4_0 | Existing standard quantizer and dense quantized graph path | Runnable now; exact15-tensor policy below |
| FFN-only W1A8 | Generic native INT8 activation packing and binary-weight CUDA dot exist; DSpark export/loader/graph integration is absent | Pending human scope decision for bounded integration |
| FFN-only W1A1 | Generic native sign packing and XOR/POPCOUNT CUDA dot exist; DSpark export/loader/graph integration is absent | Same pending integration decision |

Current DSpark/DFlash cannot become W1A8/W1A1 by setting
`GGML_W1AX_ACT_BITS`. `convert_hf_to_gguf.py:307-316` rejects the EAGLE packing
options for non-EAGLE architectures. `src/models/dflash.cpp:246-248` requires
dense FFN gate/down/up weights, and its noise graph uses the ordinary
`build_ffn` at line776. Packed GGUF data would therefore fail loading; retaining
dense weights would execute the original graph. Neither is an A8/A1 result.

The reusable implementation is real: `src/models/eagle3.cpp` validates packed
I32 signs/F32 row-scale metadata and substitutes W1Ax projections, while
`ggml/src/ggml-cuda/w1a1.cu` implements A8 per-column absmax/127 scaling,
round-to-nearest-even INT8 codes and integer dot accumulation, and A1 sign
packing, mean-absolute activation scale and XOR/POPCOUNT dot. Output scaling
and nonlinear FFN scaffolding remain F32. These source facts do not establish
DSpark SM75 dispatch, acceptance or speed; no floating fallback may be labeled
A8/A1. No new binary kernel project is required for the narrow proposed route.

## Runnable Q4_0 policy

Only these fifteen matrices change, for layers0..4:
`blk.<layer>.ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight`.
Preserve original source bytes/types for token embeddings, private full head,
both Markov factors, fusion, attention projections/norms, other norms,
confidence, target binding and all layout/tokenizer metadata. This is an
FFN-only Q4_0 weight-format variant, not full-model Q4_0 or W4A4.

The standard quantizer uses first matching regex and only applies manual
overrides when its default type is quantized. Consequently `COPY` or a BF16
default is unsuitable. `--include-weights`/`--exclude-weights` select imatrix
use rather than quantization coverage and must not be used for this purpose.

```sh
llama-quantize --pure --leave-output-tensor --token-embedding-type BF16 \
  --tensor-type '^blk\.[0-4]\.ffn_(gate|up|down)\.weight$=Q4_0' \
  --tensor-type '^.*$=BF16' \
  SOURCE_BF16.gguf UNIQUE_FFN_Q4.gguf Q4_0 4
```

Build `llama-quantize` from the same isolated published native checkout if only
the server was built. Sole operator executes CPU quantization via a unique
`remote_job.py` transaction after the actual reference probe begins; the worker
does not start any remote job or compete for GPU ownership. Quantize each
reference file separately; never quantize or overwrite the frozen target.

[precision_q4.py](../../scripts/dspark_screen/precision_q4.py) emits an exact
argv/source/binary hash plan and checks output against the passed reference
export receipt. It requires exactly15 BF16-to-Q4_0 tensors, identical tensor
names/native extents (only trailing singleton collapse allowed), and exact raw bytes/types for every other tensor. It also preserves
all metadata except quantization version/file type. A CPU toy file can test the
format policy, but cannot issue a passed released-source receipt.

```sh
python3 scripts/dspark_screen/precision_q4.py command \
  --quantizer /absolute/llama-quantize --source SOURCE_BF16.gguf \
  --output UNIQUE_FFN_Q4.gguf --receipt quantization-plan.json
python3 scripts/dspark_screen/precision_q4.py check \
  --source SOURCE_BF16.gguf --output UNIQUE_FFN_Q4.gguf \
  --llama /absolute/native-source --source-export reference-export.json \
  --receipt q4-ffn-export.json
```

An all-eligible-matrix `--pure Q4_0` conversion is also supported by the generic
quantizer, including private embedding/head and Markov tensors when eligible.
That changes coverage and the proposal recipe. It is outside root's current
phase2 FFN-only policy and would need explicit separate reporting/admission;
the user-requested target/verifier still remains unchanged. No full-Q4 head or
Markov sweep is started here.

## Storage accounting, not speed evidence

The paired releases have the same fifteen FFN shapes. Total FFN parameters are
373,555,200. BF16 storage is747,110,400 bytes; Q4_0 blocks of32 weights use18
bytes, totaling210,124,800 bytes, saving536,985,600 bytes. Existing W1 packed
signs plus one F32 mean-absolute scale per output row would use47,134,720 bytes,
saving699,975,680 bytes. W1A8/W1A1 share that weight representation; their runtime
activation buffers, packing, scales and compute differ. These counts omit
allocator alignment, staging, KV and graph scratch and predict no GPU rate.

## Minimal W1Ax work if the human approves

1. Add a DSpark/DFlash FFN-only exporter option, reusing audited row-sign packing
   (`W>=0`, little-bit-order32-bit words, F32 mean-absolute row scale). Export all
   fifteen source shapes/hashes under a new versioned `dflash.w1ax` contract;
   omit selected dense shadows and copy every protected tensor unchanged.
   Declare activation bits1 or8, exact coverage, scale/sign/tail rules and source
   ancestry. Do not inherit EAGLE learned quantizers, affine midpoints or QAT.
2. Extend the DFlash loader only for this versioned FFN contract. Existing
   generic layer fields and packed-suffix buffer routing can be reused. Validate
   all fifteen I32/F32 pairs, logical shapes and declaration; reject missing,
   duplicate, partial or dense-shadow coverage. Preserve private full head and
   immutable target teacher/verifier bindings.
3. Replace only the noise graph FFN projections with existing
   `ggml_w1ax_mul_mat` routes. Up/gate read the same original normalized row;
   down reads the original SiLU(gate)*up row. Keep norms, residuals, attention,
   K/V injection, masks, Markov, greedy sampling, compute7/propose3-or7 and cache
   repair semantics unchanged. No confidence scheduling or training is enabled.
4. Run focused actual source-to-packed/graph checks and real SM75 trajectories.
   Require explicit activation packing and CUDA W1A8 INT8 or W1A1 XOR/POPCOUNT
   dispatch for all fifteen selected matrices, with shapes/context identity and
   no dense fallback. Profile pack/dot costs separately using existing CUDA
   events; serving comparisons use unprofiled runs. Check source protected-byte
   and target-identity invariants, zero/partial/full acceptance and EOS.

Both K2560 and K9728 are32-bit-word aligned; existing kernels accept seven
columns. The main risks are poor untrained sign-weight acceptance, A1 loss of
activation magnitude, loader/metadata mistakes, and subtle FFN/cache semantics.
Reusing kernels reduces implementation scope, but does not eliminate native
admission or establish that either candidate can beat Q4_0 EAGLE. The human
scope answer is pending; no dependent native implementation is started.

## Checks and remaining work

Actual local CPU policy test passes5/5 on Darwin arm64, AppleClang21 native
`llama-quantize`, Python3.12/numpy2.3.5/PyYAML6.0.3. A tiny synthetic five-layer
BF16 GGUF becomes exactly15 Q4_0 FFN matrices; eight protected tensors retain
exact bytes/types, including confidence projection singleton normalization.
A tampered private head rejects. This is converter-policy proof, not released model quality,
SM75 performance, GPU memory or architecture acceptance. The source tree is
unchanged. The initial bundled-Python test lacked PyYAML; an isolated test venv
resolved that dependency without changing workspace packages.

Remaining: sole operator quantizes and validates actual released files; root
admits their native model/dispatch/output trajectories and runs the six balanced
phase2 repetitions on the fixed24 development prompts/settings within the
combined7200-second cap. W1Ax integration remains conditional on the human
decision. All actual raw evidence stays outside Git; root links this report
and coherent worker commit into the independent study checkpoint.
