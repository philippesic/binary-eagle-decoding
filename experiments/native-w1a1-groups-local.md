# Selectable native W1A1 EAGLE groups: local conversion and CPU gate

**State:** conversion, exhaustive source-to-GGUF row audit, and short CPU
generation passed. No RTX 2080 Ti/CUDA result or speed measurement exists.
The llama.cpp implementation commit `8d2b18a9c9b3a42927404a91799a823c758c09b6`
is published to the user's fork on `w1a1-integrated` and pinned by the parent
repository at `08f370a`, based on the 5080-tested
`92bc70602e13214d6db94c007894261b51f5f36c`. The temporary implementation
branch and worktree were removed after integration.

The converter retains the existing head-only flag and adds
`--w1a1-eagle-groups fusion|attention|ffn|all`. The loader requires a
versioned audit record, I32 packed signs, F32 row scales, exact tensor shapes,
and no dense shadow for every declared packed linear. The EAGLE graph routes
the selected linears through `ggml_w1a1_mul_mat`; normalization, RoPE,
attention softmax, residuals, and FFN SiLU remain ordinary operations. The
runtime logs each selected tensor's logical K and row count. The CPU GGML
backend reference suite passed 5/5 cases on Apple M3 Max in the isolated
branch build. This CPU gate cannot establish SM75 performance.

## Real checkpoint conversion and audit

Source: the pinned AngelSlim EAGLE-3 BF16 checkpoint
`models/hf/Qwen3-4B_eagle3/model.safetensors`, SHA256
`58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e`.
Each variant was converted from that source with the pinned target tokenizer,
`--outtype f16`, and its listed `--w1a1-eagle-groups` value. The ordinary
drafter's unselected linears remain F16. Exact command shape:

```sh
results/convert-env/bin/python third_party/llama.cpp/convert_hf_to_gguf.py models/hf/Qwen3-4B_eagle3 --target-model-dir models/hf/Qwen3-4B --outtype f16 --w1a1-eagle-groups all --outfile models/gguf/Qwen3-4B-eagle3-all-w1a1.gguf
```

The other three commands change `all` in the flag and output filename to
`fusion`, `attention`, or `ffn`. The original commands used the temporary
implementation worktree path at the same llama.cpp commit; the command above
uses the integrated submodule path for replay. The GGUF readback showed the expected I32
packed tensor count and companion F32 scales, with no dense shadow for each
selected linear:

| Variant | Packed linears | Audited rows | GGUF bytes | GGUF SHA256 |
| --- | ---: | ---: | ---: | --- |
| Fusion | 1 | 2,560 | 405,847,648 | `a4d55883dd0e18f528bd61992414db37c274ec051b97d0cd19ce6b53b5a17534` |
| Attention Q/K/V/O | 4 | 8,704 | 364,094,112 | `70c68b5a8f4faf683b9c29a16c034bc2712173d57a395edc516adb8bf81cef26` |
| FFN gate/up/down | 3 | 22,016 | 302,707,008 | `52d7cdf51c52b4b372b5eb7f55c7e632ed5a8a1f4df1dabc8540151193b7f4b8` |
| All groups including head | 9 | 65,280 | 33,774,784 | `4318123dc9aa5a73c07a13fa7131d5e10fc0c6d0e66d048a5f5f4ece1a835506` |

The independent `scripts/audit_eagle_w1a1_gguf.py` checked **every packed
weight row** against BF16 source signs and NumPy mean-absolute row scales.
It applied llama.cpp's Q/K RoPE row permutation before comparison; comparing
Q/K against unpermuted source rows initially failed, which established that
the layout transformation is material. Across each converted file the final
audit found **zero packed-word and row mismatches**. The largest absolute
scale difference was `7.45e-9` (fusion `5.59e-9`). All 65,280 unique linears'
rows are covered by the all-group audit; the other three files repeat their
selected subset and also passed.

The ignored JSON audits under `results/quantization-prep-20260924/` have
SHA256 values: fusion `6e5e42cd955d21ea7bbe9bf4d5d14619d716e93f641fb5b9cd595430a8b0950d`,
attention `d9a454b423e70ce5c2440738fafd64a46c6cd857f25690edd13c7bb45d40488d`,
FFN `d7da9cbdb829e90e80f40ed904fda24f9ffa50c18371cb2d8cd5f6cdffe6138a`,
and all `152f821c2ba54aa2efab558b1f37602504a876b42e5cc5d5690e27d4872def1c`.
The conversion logs are in the same directory; their SHA256 values are
respectively `9396da21de99c3d6d8649ffb6a04c43fdec0bb9c50a58dc3cedff0b680772eb6`,
`b8f6e83d13fcc34f09e92599cdc5300ba2d3d6819804ff5a8a91126d650e6c6b`,
`9b808ea2e981e53ba51bfdaccaedc7e1932af18af701020c45a338eec6bbff53`,
and `97d61d11f4a008871f29f9ad6ab18d90ff5c98ab392e3f7fe7ed81684e43997e`.

## CPU model smoke

A branch-local CPU `llama-cli` build loaded the unchanged FP16 target and each
new draft GGUF. With a 24-token generation cap and two-token draft limit, all
four runs exited zero and reported nonzero draft-generation calls: fusion 17,
attention 20, FFN 16, all 22. The logs showed exactly the declared packed
group and tensor load records. These one-prompt counts are a graph execution
smoke, not acceptance estimates or performance measurements.

The command shape for all groups was:

```sh
/private/tmp/llama-build-w1a1-full/bin/llama-cli -m models/gguf/Qwen3-4B-f16.gguf -md models/gguf/Qwen3-4B-eagle3-all-w1a1.gguf --spec-type draft-eagle3 --spec-draft-n-max 2 --ctx-size 512 --n-gpu-layers 0 --spec-draft-ngl 0 --no-warmup --single-turn -p 'Write a haiku about rain.' -n 24 -lv 4
```

The other three runs changed only the draft path. Their ignored CPU logs have
SHA256 values: fusion `120d53a80b48e0ce3007636e3b787cd56a82ec37e6e93173848fd58ff2d8da95`,
attention `e384bd28a71de6a645c68813dbf04579b9f2ed8718d49007ed4034999f9bdf58`,
FFN `93509a8fb35b6407f98012b220d41f2e2937fd76599e9bb24dce8908562e857a`,
and all `aece04d96289d8e42d010681d255ec0fdc60518527d974973fff02a98dfde78c`.

## Next gate

On the actual RTX 2080 Ti, rebuild the published branch for SM75, run the
backend packed-op correctness suite, verify CUDA dispatch for each packed
group and one real model request, and then run at least five matched timed
repetitions with the same target and ordinary FP16 anchor. Preserve all
hardware, driver, compiler, CMake, model, prompt, dispatch, acceptance,
memory, power/clock, and per-request timing metadata. Compare the separate
binary-MMA candidate only after it passes the real SM75 correctness gate.
