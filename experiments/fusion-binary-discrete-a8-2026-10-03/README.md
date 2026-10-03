# CPU fusion binary sign-and-scale fitting

Supporting experiment, October 3, 2026. A8 QAT remains the sole active project
goal. This team does not own QAT and has no accelerator or remote-host access.

## Frozen scope and checkpoint

Base revision: `be8d160`. Only the fusion projection is eligible for changes.
Use the original BF16 checkpoint's signs, including positive signs at either
signed zero, and nonnegative F32 row scales. A8 uses raw F32 fusion inputs,
per-token absmax/127, nearest-even signed integer codes, an integer signed dot,
then separately rounded F32 row-scale and token-scale products. Other model
operands, target/verifier precision, and model architecture are frozen.

Compare original mean-absolute row scales, converged fixed-sign row scales,
and at most two alternating sign/scale passes with four correlation scans.
Accept a sign move only after checking the actual finite exported objective;
stale individual improvements do not establish a combined improvement.
Split eligible TRAIN captures by prompt before fitting and freeze the candidate
before validation. Output reconstruction and direction diagnostics are local
surrogates; native acceptance and throughput remain deferred.

Implementation owner: bounded Sol feature agent in
`/private/tmp/eagle-fusion-discrete-fit`. Independent validator: bounded Luna
agent in `/private/tmp/eagle-fusion-discrete-validator`. Report/integration:
`/private/tmp/eagle-fusion-discrete-report`. Each has a separate branch and
disjoint new files. Existing partial/unmerged worktrees are preserved.

The independent inventory found no eligible local TRAIN fusion inputs. The
scale archive explicitly retains full calibration captures remotely and only
sampled replay operands locally. The local recurrent diagnostic bundle declares
`training_eligible:false`; its native model identity, feature parity, cache/mask,
and request ancestry gates are unverified. Neither source is used for fitting.
The authorized synthetic algorithm/export fallback is therefore selected. No
real TRAIN fitting or deployable EAGLE quality improvement is claimed.

The original BF16 draft source is available locally, SHA256
`58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e`;
the frozen F16 base GGUF is also available, SHA256
`c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1`.
Neither artifact is changed or evaluated by the synthetic fallback.

## Exact missing package

The smallest demonstrator package is two distinct eligible TRAIN prompts, each
with 16 raw **pre-A8** F32 fusion input rows of width 7,680, split by prompt into
one fitting and one validation prompt. Raw values occupy 983,040 bytes (960 KiB),
plus manifests. This is a minimum interface exercise, not a representative
quality screen. A more useful bounded screen uses 8 fitting and 4 validation
TRAIN prompts with 32 rows each: 11,796,480 bytes (11.25 MiB) of raw inputs.

Include the TRAIN prompt inventory/content hashes and explicit split; capture
file hashes and selected row offsets; native capture revision and frozen
target/draft model hashes/precision; fusion tensor name/dimensions/raw-input
stage; prompt IDs, token positions, invocation/sequence/cache/depth/mask joins;
and an eligibility receipt showing verified ancestry. Pin the original draft
source and base hashes above. The frozen layer's F32-BLAS output can be derived
from these inputs and BF16-promoted weights under the declared reconstruction
teacher contract, so teacher logits and final-set data are unnecessary. Existing
post-A16 sampled replay values cannot replace raw pre-A8 values. No remote
transfer, new capture, or GPU allocation is authorized by this report.

## CPU native arithmetic gate

Built the existing `kernels/w1ax-replay` on Apple M3 Max / arm64 with
`GGML_CUDA=OFF`, `GGML_METAL=OFF`, `GGML_BLAS=OFF`, `GGML_OPENMP=OFF`, Release,
and two compile workers. Native source is clean llama.cpp revision
`9e2c7a90051e738751aab7d7bd7c2d8201fb76e3`. Build succeeded; candidate replay
is pending. This is CPU correctness evidence only, with no SM75 performance
or full-drafter acceptance claim.

| Artifact | SHA256 |
| --- | --- |
| Existing replay source | `967ea82cf997ad177eeb829a5dddf23bb7dde20336218646afa4a10cdbddd248` |
| Native CPU source | `8c07660c792a9ac7475388b08a8b46a2a49c5b6e06b0bc4e3ecef7ead7240dc0` |
| CPU replay executable | `3cddac3099cfc71b4f7ece8c570ce3ee909353fe6eb323f64cca406b54d5f8bb` |

Raw build/run artifacts remain outside Git at
`results/fusion-binary-discrete-a8-20261003/` in the primary checkout.
