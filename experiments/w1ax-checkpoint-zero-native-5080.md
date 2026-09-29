# Row W1Ax checkpoint-zero native gate on RTX 5080

On 2026-09-29 UTC, an untrained row-scale EAGLE checkpoint was built from the
pinned original dense F16 drafter weights and tested on NVIDIA GeForce RTX
5080 (SM120). The native target/verifier remained the frozen F16 GGUF. Q4_0
EAGLE was the matched primary acceptance control. This is a checkpoint-zero
deployment and quality diagnostic, not QAT, a timing benchmark or final-set
evaluation.

## Identity and export

The pinned target/draft Hugging Face snapshot manifest SHA256 is
`2db1c860f059bd8702ca6bb523e64f0c7d064c7a95f59919a001fc88168a91e3`;
every listed model file was rehashed. The official AngelSlim runtime revision
was `0358da9c651e6a7d7ccafea26ced4b9c98d11681`; Transformers 4.57.6
was used. The F16 base draft GGUF SHA256 was
`c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1`.
The remote Python 3.11 environment had AngelSlim 0.5.0 at that exact Git
revision, Torch 2.14.0+cu130, NumPy 2.4.6, Transformers 4.57.6,
Accelerate 1.15.0 and Datasets 5.0.1. Its complete supervised `uv pip freeze`
log is `runs/w1-checkpoint-zero-env-20260929/stdout.log`, SHA256
`05e0ddbda6cb6bc87ac27d147fbbef28f3ccbe4f9ee04f4f29e3588b1a549dd8`.
The [checkpoint-zero preparer](../scripts/prepare_w1ax_checkpoint_zero.py)
verified these inputs, loaded the official target/drafter on CPU, installed
all nine row-scale sign/scale linears and saved zero-step state. A4 and A16
produced **the same 833 MiB checkpoint bytes**, SHA256
`5b371f79831c4c8da0ffc4bc5a0a4b9a6817a1fdf7c2bddfeaec6b001c4f6b78`.
Only the activation-width contract differs between their manifests: A4
SHA256 `78837575a967be199852c051912335f28ba191901b579a2b428ab6d7af7e136a`,
A16 SHA256 `6866b63710fa044cf49ebf4c3e1934380eb5d6078f52cf050220257ea1f26ce1`.

Both GGUF exports passed the exporter’s exact nine-projection and preserved
tensor audit. The A4 model SHA256 is
`a6081a5120d246435f53a3d3711607e156d3332b3ec75780019c842e85f169b1`;
the A16 model SHA256 is
`fa9406b72fb6ef19e2eca0bc0891bc20099dc318283721af3f540e056ebd3f45`.
Each is about 33 MiB. Remote supervised artifacts are under
`runs/w1-row-a{4,16}-checkpoint-zero-20260929/` and the corresponding
`*-export-20260929/` directories. Model weights and artifacts remain outside
Git.

## Native quality check

The fixed 24-prompt old development set was run in native quality mode, with
two warmups and one measured request per prompt/variant. Both runs used the
same Q4_0 GGUF, target, verifier, 128-token cap and server policy. Config
SHA256s are A4
`b215d8bc952730bb1bc2f641d9c9e1a8cdedb50a7fcbd12eb8267af3d97a03af`
and A16
`800903ad9903fe7e5d95da858a15b2e5c7e0be622991601e2028cfedf129a228`.
The complete raw benchmark manifests have SHA256
`691c82738a799ce1db53b7ce3b7b1d7c2d88790adc9a59a124e75990bbb7455d`
(A4 pair) and
`d08ba0901de3a109563af036941f3e88321b6aacc855c941bc39e64559234632`
(A16 pair).

| Draft | Accepted drafts | Verification rounds | Accepted/round | Proposed drafts |
| --- | ---: | ---: | ---: | ---: |
| Q4_0 EAGLE (primary; repeated in each pair) | 1,555 | 1,493 | 1.0415 | 7,320 |
| Row A16 checkpoint zero | 323 | 2,725 | 0.1185 | 13,306 |
| Row A4 checkpoint zero | 131 | 2,917 | 0.0449 | 14,244 |

Both row variants emitted the exact same raw generated IDs as Q4_0 on all
24 paired prompts. The native row-A16 and row-A4 blocks both recorded verified
CUDA graph launches with custom W1Ax calls (20,187 and 21,590 launches over
their full server lifetimes, including warmups); neither model was merely
loaded on CPU. Both supervisors exited zero, their process groups stopped,
and RTX 5080 returned to 0% utilization and about 2,900 MiB whole-device
baseline.

The identical checkpoint bytes make the A16-versus-A4 difference a controlled
activation-width observation: A16 accepted about 2.47 times as many drafts
as A4, yet both are far below Q4_0. Candidate D group-128/A16 had 909
accepted over 2,139 rounds on this same old suite in earlier runs, but its
fitted group scales are a different initialization, so this does not isolate
row versus group layout. The old suite is heavily reused, and quality mode
does not support throughput claims. This negative checkpoint-zero result
motivates a bounded training/readiness choice; it cannot rule out recovery by
QAT or establish that any trained row checkpoint will beat Q4_0. The new
independent development and sealed final sets were not used or opened.
