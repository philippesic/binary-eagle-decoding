# Focused related-work check

**Date:** 2026-09-24. This is a bounded review of primary papers and official
project artifacts, not an exhaustive novelty or patent search.

The defensible project claim is specific: evaluate an EAGLE-3 drafter with
selected **W1A1 weight-and-activation** linears executed by packed native
binary kernels, jointly measuring acceptance, packing/scaling overhead, and
end-to-end latency on named hardware. A head-only configuration should be
called **“EAGLE-3 with a W1A1 output head”** rather than a wholly binary
drafter. The review did not identify the exact combination already measured,
but it does not establish a first demonstration.

Prior work sets important boundaries:

- [EAGLE](https://arxiv.org/abs/2401.15077) and
  [EAGLE-3](https://arxiv.org/abs/2503.01840) establish feature-based,
  tree-verified drafting and the EAGLE-3 architecture/training method.
- The official [MiniCPM4 Eagle-FRSpec-QAT model
  card](https://huggingface.co/openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu)
  already provides a quantized EAGLE model trained with QAT and deployed
  through native `cpm.cu`. Broad claims of inventing quantized EAGLE or
  native EAGLE QAT are incorrect.
- [QSpec](https://aclanthology.org/2025.emnlp-main.240.pdf) uses W4A4 drafting
  with a W4A16 verifier and reports a substantial acceptance drop when its
  EAGLE drafter is quantized. Our observed loss has precedent, though our
  target/drafter and W1A1 coverage differ.
- [Speculative Decoding Meets
  Quantization](https://arxiv.org/html/2505.22179v1) implements optimized
  quantization kernels for EAGLE and discusses the verification cost of trees.
  Our comparisons must include verifier time and committed tokens, not only
  draft-kernel latency.
- [XNOR-Net](https://arxiv.org/abs/1603.05279) establishes binary
  weight/activation scaling and bitwise compute. The binary arithmetic itself
  is not new.
- The [BSTC paper](https://www.pnnl.gov/publications/bstc-novel-binarized-soft-tensor-core-design-accelerating-bit-based-approximated)
  motivates testing small token batches and packing/launch overhead rather
  than assuming peak binary throughput predicts inference gain.

The [1bit-MONSTER speculative-decoding
README](https://github.com/1bit-MONSTER/1bit-MONSTER/blob/main/spec-decode/README.md)
is a near-looking project reference, but its inspected speculative draft is
labelled FP16 and the cited benchmark simulated. This review did not audit
its entire codebase, so novelty remains provisional.

Practical implications: preserve target precision within each comparison;
report accepted draft tokens/round separately from total emitted tokens/round;
and label native operand precision explicitly. When hardware access permits,
an actual W4A16 drafter baseline would complement the project's W4A4/W8A8
numerical simulations, which are not native INT4/INT8 timings.
