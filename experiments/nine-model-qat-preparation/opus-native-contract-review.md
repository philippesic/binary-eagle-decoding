# Focused Opus native contract review

Requested exact model: `claude-opus-5-5-high`; peer-review MCP `ask_claude`,
mode `ask`, October 4, 2026. Source baseline parent `da097e5`, native `fcdf5822`.
First mode `review` call was rejected by CLI (allowed `ask`/`plan`); it produced
no review. The corrected bounded call completed. No GPU/remote action.

Coordinator qualification: nonnegative finite row scales may legitimately be
zero; review's blanket positive-scale requirement is not adopted. Zero-scale
orientation rescue and derivative-one additive-scale behavior remain explicit.
Source-bound advice is not a build, native execution, or quality test.

## Reviewer output

+**Verdict:** the current DFlash/DSpark code can't load or run a packed-only binary GGUF without new W1 code. It needs its own loader and graph path. The EAGLE3 W1 helpers are hardwired to EAGLE3's single layer, and four places would silently diverge from training.

**Blocking: export and load**

1. **Dense tensors are required.** `fc` and `ffn_{gate,up,down}` are created with flag `0` (`dflash.cpp:161,245-247`). A GGUF with packed weights only fails as "missing tensor". It needs a DFlash version of `load_w1a1_linear` (`eagle3.cpp:241-302`), including the check that the dense shadow is absent (`eagle3.cpp:271`).
2. **The EAGLE3 helpers can't be reused.**
   - The manifest groups are hardcoded to `blk.0` (`eagle3.cpp:120-125`).
   - `eagle3_quantizer_boundary` throws on `blk.1+` (`eagle3.cpp:22-31`).
   - The learned delta/clip arrays have exactly 6 slots (`llama-model.h:661-662`).

   Fifteen FFN tensors across 5 layers means 10 activation boundaries (gate/up share one per layer) plus `fc`. Use a `dflash.w1a1.*` namespace. The selected manifest should assert `n_layer==5`, the exact set of 15 or 16 tensor names, and that `dsv4_hc_mult==0`.
3. **The DSV4 DSpark variant is out of scope.** Its FFN is MoE plus shared experts (`dflash.cpp:981-998`). The W1 ops are 2D only, with no `mul_mat_id` equivalent. Reject that variant at load.
4. **Activation bits come from the environment.** `GGML_W1AX_ACT_BITS` is process-global and defaults to 1 (`eagle3.cpp:13-20`), and EAGLE3 throws if the metadata disagrees (`eagle3.cpp:158-164`). An A8 GGUF fails unless the env var is set. Let the metadata choose the bits. If `fc` and FFN use different bit widths, store bits per tensor, because the op takes one value per call.
5. **Encoder/decoder width mismatch.** The model computes `n_embd_inp_enc` from the draft `n_embd` (`dflash.cpp:40`), but the runtime packs features using the target's `n_embd` (`speculative.cpp:1246`). Assert they're equal.

**Blocking: graph**

6. **`fc` is built in three places:** the encoder graph (`:282`), the decoder's injection branch (`:627`), and DSV4 (`:891`). Replace all of them. Graph reservation may build the encoder graph, and that dereferences a null `fc`.
7. **`build_ffn` can't take packed weights** (`:776-781`). Write the parallel FFN out explicitly: pack the shared gate/up input once, apply the activation (SiLU/GELU, `:12-19`) times up, then pack again for down. Reject draft LoRA adapters, as EAGLE3 does (`eagle3.cpp:571`). Name the row scales `.w1a1_scale`, never `.scale`. Otherwise `build_lora_mm` applies them a second time, and the generic `.scale` pass (`llama-model.cpp:1566+`) may pick them up.
8. **`fc` input outliers.** DFlash has no norm before `fc`; RMS norm comes after it (`:285`). The input is the raw concatenated residual stream extracted from the target (`speculative.cpp:1461`). Massive outlier channels dominate the A1 mean-abs scale and the A8 per-token absmax (`ggml.h:1500-1501`), so most A8 codes become 0. Measure the code histogram before training.

**Zero-sign, scale and midpoint representation**

- **Zero sign.** The weight rule is `nonnegative_is_one` (`eagle3.cpp:171`). Activation packing maps both +0 and −0 to 1 (`w1a1.cu:40`). PyTorch's `torch.sign(0)=0` differs: the straight-through estimator during training must also use `x>=0 → +1`. Exact-zero weight rows export as `+alpha` (affine: `mu+alpha`), not zero.
- **Scale.** The loader checks only shape, dtype and the rule string. The exporter should assert every row scale is finite and >0, since a negative scale silently flips the row. Group-128 scales require A16 (`eagle3.cpp:154-157`), so use row scales for A1/A8. Tail bits are masked by the kernel (`w1a1.cu:59,64`), but the export should still zero them.
- **Midpoint.** Output = `alpha*s_a*dot + mu*s_a*code_sum` (`w1a1.cu:86,194`). The midpoint is one F32 per row, named `<base>.w1ax_midpoint` (`eagle3.cpp:295-297`). EAGLE3 allows it only for v2 with row scales (`:230`).

**DSpark slots, Markov head and cache**

- **Silent Markov skip.** If a block is longer than `block_size`, the head returns without biasing the logits (`dflash.cpp:324-326`). That acts like a fallback, so make it an assert.
- **Two sources for block size.** `block_size` is read from the raw `gguf_kv` string (`:310`) and also from hparams (`:30`). If `dflash.sample_from_anchor` is absent it defaults to true (`:317`). Write both keys explicitly.
- **Greedy chain.** The Markov chain conditions on argmax (`:345-346,388`); sampling only changes the final pick. Train and evaluate greedy.
- **Short noise blocks.** Outside the author layout, the noise block length is `n_max`(+1) (`speculative.cpp:1561-1562`), shorter than the trained block. The author layout forces block 7 (`:1302`).
- **Confidence head input.** It reads `t_embd`. That's the post-norm state for dense DFlash (`:806`) but pre-norm for DSV4 (`:1011-1012`). Confirm which one the reference was trained on.
- **Markov weights.** Keep `markov_w1` and `markov_w2` at original precision. `w1` is indexed by target vocabulary, `w2` by draft vocabulary, and the `d2t` scatter uses 0 bias against −inf base logits (`:353-360`). Correct.
- **Equal blocks per ubatch.** Each ubatch must hold whole, equal-size blocks (`:320-321`). Set the draft `n_ubatch` ≥ the total noise tokens.

**DFlash masks**

- Attention inside a block is non-causal through `llama_set_causal_attn` (`speculative.cpp:1362`), and `dflash.attention.causal` defaults to false. With full attention, any stale noise or mask K/V left in the cache gets attended to. That invariant is only asserted when admission tracing is on (`speculative.cpp:1542-1543`); make it unconditional for binary runs.
- Injection writes cache entries only (`dflash.cpp:675-678`). Hadamard rotation of K/V is applied when it is set (`:669-673`).
- `wv==null` means K and V are shared, with V RMS-normed (`:635-645`). Keep attention dense, as planned.
