# Pinned FP16 GGUF conversion and local load check

**Date:** 2026-09-24. **Device:** Apple M3 Max, Metal. This is a conversion and
functional load check, not an RTX 5080 or SM75 throughput result.

The source snapshots are the pinned `Qwen/Qwen3-4B` revision
`1cfa9a7208912126459214e8b04321603b3df60c` and
`AngelSlim/Qwen3-4B_eagle3` revision
`fd331e59626c8e95c392381a16ee59d518727fbb`. File hashes are in
`results/metal-smoke-20260923/model-manifest.json` (ignored raw artifact).
llama.cpp conversion/runtime revision was
`257e2c670e892bd9ca9c404167b2919020b969f8`; parent revision
`855fea51f00b05678faa8d971093b7a78c797e25`. The conversion virtual
environment used Python 3.11.15 and the pinned llama.cpp conversion
requirements, including torch 2.11.0 and Transformers 4.57.6. Its package
list is `results/conversion-baseline-20260924/packages.txt`, SHA256
`d5269b8b800401897f8c6519b745236f1882663c295b6c50891bf9df1fa8af69`.

Commands, from the repository root:

```sh
results/convert-env/bin/python third_party/llama.cpp/convert_hf_to_gguf.py models/hf/Qwen3-4B --outtype f16 --outfile models/gguf/Qwen3-4B-f16.gguf
results/convert-env/bin/python third_party/llama.cpp/convert_hf_to_gguf.py models/hf/Qwen3-4B_eagle3 --target-model-dir models/hf/Qwen3-4B --outtype f16 --outfile models/gguf/Qwen3-4B-eagle3-f16.gguf
```

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `models/gguf/Qwen3-4B-f16.gguf` | 8,051,285,280 | `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6` |
| `models/gguf/Qwen3-4B-eagle3-f16.gguf` | 442,700,800 | `c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1` |

Both converters exited zero. The target contains 398 tensors; the EAGLE-3
draft contains 14, including an FP16 output head and I64 `d2t`. The draft
converter read target layers `[2, 18, 33]`, target hidden width 2560, and the
target tokenizer. Raw target/draft conversion log SHA256 values are
`4b37c14139e1e8f227616610c66a32ae0679994b1702f363a1d5cf319ef87c77`
and `fb95a226a4251dd91876faf9305c45815533ade8cfc24a4e53f6273c8fa46707`.

CPU and Metal `llama-cli` builds completed from this submodule revision. A
16-token greedy Metal target-only smoke loaded and generated text; an ordinary
EAGLE-3 smoke also loaded both converted files and generated text with
`--spec-type draft-eagle3 --spec-draft-n-max 5`. Their raw logs are under
`results/conversion-baseline-20260924/`. The target-only smoke was manually
interrupted after its first response because the CLI entered another input
turn; it did generate the requested 16 tokens. The EAGLE smoke used
`--single-turn` and exited normally. This check establishes that the converted
pair loads locally; it does not establish verifier parity or benchmark timing.

Next, reproduce conversion on the RTX 5080 host or transfer hash-verified
artifacts there under the host's project directory, then run paired same-device
target-only, ordinary EAGLE, and native W1A1 comparisons after the packed-head
bridge is ready.
