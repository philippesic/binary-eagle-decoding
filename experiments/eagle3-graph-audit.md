# EAGLE-3 graph audit for the PyTorch W1A1 goal

**Status:** structural audit only; no model forward pass or acceptance result yet.  
**Target snapshot:** `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`  
**Drafter snapshot:** `AngelSlim/Qwen3-4B_eagle3@fd331e59626c8e95c392381a16ee59d518727fbb`  
**llama.cpp:** `6e60f35608ec6918b44a9839c0c433687165f086`

## Data path

The target has 36 layers. The pinned converter derives EAGLE feature taps
`[2, 18, 33]` when the draft config does not provide explicit layer IDs. These
are **inputs to the named target blocks**. For a Hugging Face target forward
with `output_hidden_states=True`, tuple entries `[2, 18, 33]` correspond to
those block inputs. The three 2,560-wide states are concatenated in that order
and passed to the drafter's feature-fusion `fc` projection. The drafter then
uses its own single decoder layer to propose tokens. This indexing and the
numeric forward path still require a model-run parity check.

The draft checkpoint has its own 32,000-row vocabulary head and a `d2t`
mapping into target token IDs. It does **not** contain a token embedding matrix;
the decoder borrows the target's embedding weights. The `d2t` conversion must
match the pinned converter before evaluating acceptance. The config defaults
`norm_before_fc` and `norm_before_residual` to false.

## Candidate linear operations

Shapes are shown as `(output features, input features)`, matching PyTorch
`nn.Linear.weight`. The listed checkpoint tensors are BF16. These are coverage
candidates, not a decision to binarize all of them.

| Group | Tensor | Weight shape | Note |
| --- | --- | --- | --- |
| Feature fusion | `fc.weight` | `(2560, 7680)` | Projects three target states. |
| Attention | `midlayer.self_attn.q_proj.weight` | `(4096, 5120)` | Decoder input concatenates normalized embedding and fused state. |
| Attention | `midlayer.self_attn.k_proj.weight` | `(1024, 5120)` | Grouped-query attention. |
| Attention | `midlayer.self_attn.v_proj.weight` | `(1024, 5120)` | Grouped-query attention. |
| Attention | `midlayer.self_attn.o_proj.weight` | `(2560, 4096)` | After attention. |
| Feed-forward | `midlayer.mlp.gate_proj.weight` | `(9728, 2560)` | Parallel SiLU gate path. |
| Feed-forward | `midlayer.mlp.up_proj.weight` | `(9728, 2560)` | Parallel up path. |
| Feed-forward | `midlayer.mlp.down_proj.weight` | `(2560, 9728)` | Projects back to draft width. |
| Vocabulary head | `lm_head.weight` | `(32000, 2560)` | Own reduced head; treat separately. |

RMSNorm, RoPE, attention softmax, residual additions, and output-ID mapping
remain ordinary operations in the PyTorch W1A1 simulation.

## Source and remaining checks

- Local graph: [`src/models/eagle3.cpp`](../third_party/llama.cpp/src/models/eagle3.cpp).
- Local conversion logic: [`conversion/llama.py`](../third_party/llama.cpp/conversion/llama.py).
- [Pinned drafter config](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3/blob/fd331e59626c8e95c392381a16ee59d518727fbb/config.json)
  and [pinned target config](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/config.json).
- Drafter tensor shapes and presence/absence above came from the safetensors
  header, without downloading tensor payloads. Check complete model files and
  hashes when the RTX 5080 host is available.
- Confirm PyTorch output parity on a short fixed token sequence before using
  the adapter for W1A1 acceptance. In particular, verify feature taps, RoPE,
  cache positions, `d2t`, and draft/target vocabulary alignment.
