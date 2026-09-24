# PyTorch INT4 and INT8 drafter acceptance on RTX 5080

**Status:** fixed 12-prompt held-out sweep in progress. This experiment measures
post-training W4A4 and W8A8 numerical simulations; it does not execute native
INT4/INT8 kernels or measure inference speed.

## Matched setup and quantization rule

- Target `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` and
  drafter `AngelSlim/Qwen3-4B_eagle3@fd331e59626c8e95c392381a16ee59d518727fbb`.
  All 18 snapshot files match the SHA256 hashes used in the prior W1A1 CUDA
  experiment. The AngelSlim source commit remains
  `0358da9c651e6a7d7ccafea26ced4b9c98d11681`.
- Remote project code `92c91a4cecdd0123a2e1d7bf7c6d394fa673dabe`, clean
  checkout. Config `configs/pytorch_int4_int8.toml` SHA256
  `7ac30ee431e9b2583598ab68cb28731daca6edb5e453ebd1c300578621752c3a`;
  remote model manifest `results/model-manifest-int4-int8-20260924.json`
  SHA256 `872cc50776ce1e5adb9494e1822cadaec37fbc81338dd1a7b455057907c1f807`.
  The 12-prompt manifest SHA256 remains
  `0d6a698d6816592c6ff435fed2fea4cdafe9f5248393d2a5ac091e1551919476`.
- Greedy Qwen3 non-thinking chat template, target and verifier unchanged,
  `max_new_tokens=128`, 60-token EAGLE tree budget (59 proposed draft nodes
  per round), depth 5, top-k 10. Ordinary BF16 EAGLE is rerun in the same
  process as both low-bit variants.
- W4A4 and W8A8 cover the nine drafter-owned feature-fusion, attention, FFN,
  and vocabulary-head linears. Weights use symmetric signed codes `[-7,7]`
  or `[-127,127]` with one absmax scale per output row. Activations use the
  same code range with one absmax scale per input token vector. Rounding is
  nearest-even; zero vectors produce zero codes. PyTorch performs FP32
  `F.linear` on integer-valued floats, applies FP32 scales/bias, and casts
  output to the drafter dtype (BF16). TF32 is disabled and FP32 matmul precision
  is `highest`. Embeddings, normalization, softmax, residuals, and target
  verification retain their ordinary precision. This simulation is not a
  bit-packed kernel or exact INT32 accumulator.

## Parity limitation and one-prompt smoke

The prior strict CUDA check found ordinary BF16 EAGLE diverges from target-only
greedy output at generated token 4 due a target-selected verifier-logit tie;
the tree-versus-prefix logit shift remains unexplained. The new run records
target-greedy mismatches and continues only as an explicitly exploratory,
same-verifier acceptance comparison. It cannot establish target-equivalent
output or a clean quantization-only quality effect.

The supervised `cuda-int4-int8-smoke-20260924` run finished with exit 0 on
`prose-01`. The disabled quantizer wrapper matched ordinary EAGLE exactly.
All three variants recorded the known first target-greedy mismatch at index 3.

| Variant | Accepted / proposed draft nodes | Rounds | Accepted/round |
| --- | ---: | ---: | ---: |
| Ordinary BF16 EAGLE | 84 / 2,773 | 47 | 1.787 |
| W4A4 simulation | 25 / 6,195 | 105 | 0.238 |
| W8A8 simulation | 80 / 3,068 | 52 | 1.538 |

The one-prompt result only checks the path and gives a preliminary quality
signal. The held-out 12-prompt aggregate, prompt spread, environment, raw
hashes, and final GPU cleanup will follow when the supervised sweep finishes.
