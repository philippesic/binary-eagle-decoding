# PyTorch W1A1 acceptance: Metal development run

**Status:** complete exploratory run; RTX 5080 confirmation pending. This is
post-training fake quantization in PyTorch, not packed one-bit execution or a
throughput result.

## Fixed setup

- Target: `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`.
- Drafter: `AngelSlim/Qwen3-4B_eagle3@fd331e59626c8e95c392381a16ee59d518727fbb`.
- Runtime: AngelSlim source `0358da9c651e6a7d7ccafea26ced4b9c98d11681`,
  project commit `f781efd4f6927651c22db06804a4723b08b0f5ff`.
- Hardware and precision: Apple M3 Max, Metal, PyTorch 2.14.0, Transformers
  4.57.6, BF16 target and BF16 drafter. No CUDA kernel or SM75 claim follows.
- Quantizer: `sign(0)=+1`, mean-absolute scale per output weight row and per
  input token vector. Selected weights and activations are fake-binarized at
  linear operations; other operations stay at ordinary precision.
- Greedy Qwen3 non-thinking chat template, temperature 0, 12 self-authored
  held-out prompts (four prose, four code, four reasoning), SHA256
  `0d6a698d6816592c6ff435fed2fea4cdafe9f5248393d2a5ac091e1551919476`.
  Tree settings: total token budget 60 (59 draft nodes plus target seed), depth
  5, top-k 10. Requested output cap 128; actual outputs were 85–132 tokens
  because stopping occurs at a verification-round boundary.

The statistic below is **accepted draft nodes per verification round**. It
excludes the target-selected root/seed and the target recovery/bonus token.
Each round proposes 59 tree nodes, so accepted/proposed node ratios are small
even for useful branching trees. There is one run per prompt; the ranges show
prompt variation, not confidence intervals. FP16 EAGLE throughput and
target-only throughput remain deferred to the later direct native comparison.

| W1A1 layer group | Accepted draft nodes | Rounds | Accepted/round | Prompt median (range) | Zero-accept rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Feature fusion | 627 | 881 | 0.712 | 0.745 (0.491–0.925) | 55.7% |
| Attention Q/K/V/O | 656 | 849 | 0.773 | 0.804 (0.593–0.925) | 38.6% |
| FFN gate/up/down | 778 | 736 | 1.057 | 1.047 (0.711–1.434) | 29.6% |
| Own vocabulary head | 951 | 565 | 1.683 | 1.788 (1.186–2.171) | 15.4% |
| All listed groups | 254 | 1,256 | 0.202 | 0.216 (0.084–0.358) | 81.8% |

The combined W1A1 setting has substantially lower draft acceptance than any
single-group setting under this initial rule. Head-only retains the most draft
acceptance among the configurations tested. This does not establish a speedup:
draft execution time, native bit arithmetic, and the later same-device FP16
comparison have not been measured. A narrower coverage experiment or bounded
QAT is the next research fork after RTX 5080 confirmation and layer-cost audit.

## Greedy parity limitation

The disabled wrapper exactly matched unwrapped EAGLE on the first prompt.
Ordinary BF16 EAGLE and target-only greedy generation first differed at
generated token 21 on Metal. A follow-up trace located it at zero-based
generated index 20, round 9, position 0: it was a **target-selected seed**
from the preceding verification round, not an accepted draft token. The tree
verifier's BF16 logits were tied at 27.5 for token IDs 11 and 34512, so
`argmax` selected the lower ID, 11. A separate full-prefix target computation
on the same 20-token prefix gave 27.75 for ID 34512 and 27.625 for ID 11.
Thus the immediate token-choice mechanism is established; the cause of the
tree-versus-sequential logit shift is not. BF16 arithmetic and cache-history
differences are plausible, while a mask, position, or cache-index defect has
not been ruled out. An FP32 Metal diagnostic matched the first 33 generated
tokens. The run recorded the mismatch and continued only because it was
flagged as a Metal development check; the CUDA runner remains strict. These
counts remain observed acceptance under the Metal verifier, but are not yet a
clean estimate of quantization-induced acceptance loss.

## Raw artifacts

Raw files are ignored by Git at
`results/metal-heldout-20260923/` on the local Mac. They contain the resolved
config, prompt copy, model-file hash manifest, environment, parity token IDs,
per-prompt accepted counts, and aggregate summary. SHA256:

| File | SHA256 |
| --- | --- |
| `acceptance.jsonl` | `5135d3e771b0bb11efb804c7388716d3c06ffa57b61b48b980bccd885f84fcd4` |
| `summary.json` | `e11d96c3da1b05aa144545de749c54efc8ca86512993f26155fd0a85cba25272` |
| `model-manifest.json` | `397937106d9de6f74d556455ccc3a292983e7b200a814dd69451f9a3705efa6c` |
| `greedy-parity.json` | `c1ef27c7e67455b0b4eeb61c70cc4af26c8e03e0f237b719e5600ec850133a38` |

The follow-up verifier trace is at `results/local-prep/greedy-trace-bf16.json`
on the same Mac, SHA256
`debd26d03da06bc37f95fc97b20b4a5f540494e5853f7fbc9811e10055294c22`.
