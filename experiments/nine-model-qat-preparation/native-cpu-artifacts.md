# Actual native block artifacts — October 4, 2026

Mac-only preparation on **Apple M3 Max, Darwin arm64, 38,654,705,664 bytes RAM**.
Native builds disable CUDA, Metal and BLAS. All real-model optimizer update counts
are zero. Synthetic operands exercised actual released weights; these checks are
not calibration, QAT, acceptance, throughput or SM120 memory/execution evidence.
RTX5080 remained paused during this artifact work.

Source: parent feature through `2cb2206`; native fork through `624f50e74`, including
finite fixed-A8 arithmetic `b8f574ef5` and declared MASK handling `a0be95d61`.
Normal-domain CPU binary tests pass 272/272; twelve amplified fixed-tiny cases
cover direct/shared packing and reciprocal overflow. The contract is
`fixed_w1a8_finite_reciprocal_v2`; CUDA compile/execution remains PENDING.

## Immutable originals and fresh conversions

Files remain outside Git under
`models/nine-model-original-snapshots/<family>/<revision>/` in the main checkout.
Both complete downloads match the published inventory's byte counts and SHA256;
`download-receipt.json` records local verification. Original configs are unchanged.
Copied conversion configs explicitly retain private embedding/head; their hashes
also match the earlier conversion-input records.

| Binding | DSpark | DFlash |
|---|---|---|
| Repository | `deepseek-ai/dspark_qwen3_4b_block7` | `deepseek-ai/dflash_qwen3_4b_block7` |
| Revision | `3457dff1417cb84927f6098a5fcb7cee85c934b7` | `02d530b7962ea1412beaf41a05c0b8e36d5f9b1d` |
| Safetensors SHA256 | `f9e31587608441f235d46410e7201f8cb1647be5cd077065c89cc02c37ae86a7` | `68d4138ec35864c47c4856d9f54004f1299d151818c33eacafd8ba19aad9cb9f` |
| Original config SHA256 | `494e5665481ff8216c4e857a531c401f72b7fb795434a8201f45252c0ad7568a` | `e4d3ad2f79210e91cf8b4d78547d23afda9a8cd2a90c4c6b97c9ea617d8f94f1` |
| Copied config SHA256 | `e403743c27eeb271c505212b9999d0734cd950458b669a95941a612f9e465d89` | `bf7deaec638cb7a0bebc1738b5f35b6d855a6b82ebc764ce7531cfa8afa470e1` |
| Fresh BF16 GGUF SHA256 | `3685ea7785729b4d3e9de3939264c1c87eb98a2bf7935a8c838eccf36be9fd13` | `61b294dc798c507fdc4a87f397b4015a78b3cdd7edad1e940e683bba10fb5575` |
| Canonical report SHA256 | `0db00c7f149d162297354c82d51dd027760cacd52091ec340d20c25ae4681997` | `dec52969e301ac8f562e53cb0be340fa143405297785be4ef5806b00288c76fa` |
| Base export audit SHA256 | `40fdcac639f5b5782a9da26b4b9fa8f3f61b7ee1c1b940b4b27714a5cdaab761` | `9b0fb2d12ccf4477ed5292ca99cccade9cea9d873c32a9ee567d133cc1a4d884` |

Both bases have BF16 matrices and F32 norms, full vocabulary, original private
roles, no borrowing and author anchor-first block7/taps `[2,10,18,26,34]`.
DSpark has 64 tensors; DFlash 60. Canonical private-source F32 hash, shared by both
embedding/head roles and families:
`75599dcbc3345b7aa22c4db6be909c5ae109a710cd3e15d842c5e7bcd00620aa`.
Frozen target role canonical hash:
`33f5c1db6eb5f998ec7f5ecc4ea37f36aab01183c5f2caa55333703ac57dd982`.
Each role again has 77,045 differing values, 451 BF16→F16 underflows, maximum
absolute difference `2.9802322387695312e-08`, and no nonfinites/overflows.
Exact borrowing remains false. Raw BF16 private payloads are unchanged through
conversion, for both roles/families:
`eabe5625fc0575bf517c424041e9701c0fd521889e0f547c8522d2aa20e8c0f8`.

Frozen target GGUF is still exactly
`05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`
(8,051,285,280 bytes). Target HF config SHA256 is
`8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a`;
local tokenizer/config cache metadata pins revision
`1cfa9a7208912126459214e8b04321603b3df60c`.

Fresh BF16 file hashes differ from historical conversion artifacts. They are
separate CPU artifacts. **Original frozen DSpark/DFlash Q4 GGUF files remain
missing locally on the inaccessible RTX2080Ti; these bases/prototypes neither
replace nor redefine those controls.**

## Untrained export and actual CPU graphs

`untrained-binary-smoke-v2/` contains original BF16→F32 latents and per-row F32
mean-absolute scales, exact export manifests and receipts. No fitting or optimizer
was run. Each prototype selects fifteen FFN matrices plus FC; every other tensor
retains base bytes/type, including private embedding/head, attention and Markov.

| Cell | Prototype GGUF SHA256 | CPU graph receipt SHA256 |
|---|---|---|
| DSpark A1 | `405466ebf63b8b6227f9d4af0c68b92de4936b1f169ed1255c6817a0613ad7b3` | `c19ca258b478de6bf4ead778a28e17c5c961ecb0c95b3c8e17f66eed579cc4d0` |
| DSpark A8 | `0089fb0f074404cf86d56a6ee22471b7686c34ac71512ac22b21051ba426c613` | `29fa0f182778876532b49f481d9578212382f030000130fd5d07c8f1055b7638` |
| DFlash A1 | `5b4c0b4adabddc0dcb4017e0e36df1961ac6367a30b861059d44e6688e7e03b4` | `e670b5ad987085d6a71497f82f4f93278294f55070e82c00e2c236c0ac454d52` |
| DFlash A8 | `2eade162c9bd7868a280d8687407c78a8f79957ed5cad293686ebf99e4d7a68d` | `dbb465fd9a1132bd289e36b7a72c6ab32976376f1fb616bfe7d3e8065e71686f` |

All four load, inject three synthetic context rows and execute the intact seven
noise slots. Each receipt observes exactly sixteen selected I32/W1 projections,
the declared A1/A8 bits and CPU output buffers; finite full-vocabulary logits and
no selected dense fallback. Noise IDs are `[2,151669,151669,151669,151669,151669,151669]`,
positions `[3,4,5,6,7,8,9]`. MASK selection matches the deployed driver's
`llama_vocab_mask` call and author-layout `151669` guard. Synthetic anchor2 is not
a captured TRAIN prefix. Graph-helper binary SHA256:
`0d3b64c8003f385b34a1c7544a5df201eb435f45b5070f597cfe0de01d6aac05`.

One representative DSpark A8 + actual frozen F16 target co-load also passed.
Private BF16 embedding/head pointers differ from all target roles, each shape is
`[2560,151936]`; target roles are F16, target has 36 layers and all five ordered
taps agree. This is storage/geometry/operator evidence, not target quality.
Paired receipt SHA256:
`728b323fc1f27375a2e94190119d8904e66aa4992ef18cc1168a06627bfc5816`;
paired helper binary SHA256:
`e0c0acf4c8de3016195b132583926a5f3c31a5b2f496aee9ff0fb1fd1f5b766d`.

## Resource and failure record

One heavy CPU job at a time. Main ignored
`runs/nine-model-native-cpu-20261004/` preserves exact argv, stdout, sampled
process-group RSS, kernel child peak RSS, exit status and group-empty receipts.
Normal-stage cap was 8 GiB; one approved paired stage had 12 GiB.
Maximum observed RSS: conversion `5,053,480,960`, successful prototype export
`7,988,723,712`, standalone graphs `2,226,929,664`, paired stage
`10,346,905,600` bytes. All successful jobs exited0 and owned groups were empty.
Heavy slot was explicitly released to the training owner after the paired run.

The first DSpark prep helper decoded GGUF BF16 U8 bytes without a U16 view,
doubling latent K. The exporter correctly refused source-shape mismatch before
writing any GGUF. Failed `dspark-prototype-export/` receipt and NPZ remain intact:
exit1, peak observed RSS `5,259,476,992`, group empty, zero optimizer updates.
Corrected v2 uses U16 reinterpretation/logical reshape; successful history is
separate. No failed gate was waived or original weight modified.

## Coverage ledger

| Profile/requirement | Evidence and status |
|---|---|
| `ffn15`, A1/A8, both families | PASS CPU toy contracts and four actual full-source fifteen-projection export/load/graphs in the follow-up below; CUDA/hardware **PENDING**. |
| `ffn15_fusion`, A1/A8, both families | PASS actual original source/private-copy/export and CPU sixteen-projection graphs above; CUDA/hardware **PENDING**. |
| Paired target ownership | PASS representative actual DSpark A8 CPU co-load; other actual paired profiles/hardware **PENDING**. |
| Calibrated/trained models, production data, quality/throughput | **PENDING**; these prototypes prove none of those outcomes. |
| Original frozen Q4 controls | **PENDING locally**; preserved remote ancestry, no substitute control admitted. |

Actual BF16 FC/gamma reference extraction is separately owned by the data team:
`results/nine-model-qat-preparation/block-fusion-references-20261004/`.
No all-profile or percentage readiness claim follows from test counts.
Machine-readable checkpoint:
`models/nine-model-original-snapshots/actual-cpu-summary.json`, SHA256
`d484a70ab08e3f9a397725bd6edf175d6a66a8a4616c5b27f55c0023575d2628`.


## FFN15-only follow-up — same Mac, source and invariants

The four previously pending actual `ffn15` cells now pass. This follow-up reused
the verified bases and original BF16-derived FFN latent/meanabs arrays; only a
fifteen-projection NPZ subset was written. The FC16 suites were not repeated.
Source/graph helper stayed native `624f50e74`; binary SHA256 is
`e0c0acf4c8de3016195b132583926a5f3c31a5b2f496aee9ff0fb1fd1f5b766d`.

`untrained-ffn15-smoke/` holds distinct manifests/exports/proofs. Every other
base tensor retains bytes/type. In particular FC remains dense **BF16**, raw
SHA256 DSpark `a58762db0ffb08c8727b4999362610491e5dac51baa64e127a41750f2ffab9e3`,
DFlash `691b5fc198e4b1423bcac97bea90f4388b38a02f7b5d9c2afb50f452c403ade6`.

| Cell | FFN15 prototype GGUF SHA256 | CPU graph receipt SHA256 |
|---|---|---|
| DSpark A1 | `20c2a388eb94d77acb8a7e7490922020693540187b9fd45a75508453d22f6e64` | `756cb57d1d130b0c5e8f4b8406b680d9ea3c840e8e843afc6d3e01e28f6311b7` |
| DSpark A8 | `3ef4d9ce7d17ab450db0631cb2cf7a1f44291c1eb27cdbe925ac079efa974201` | `227f14fa4657da3221423ad8a985440b9b29f5ad5b520df4160a430349647598` |
| DFlash A1 | `73e7dc98e7a2e87d498754d98bd0c7a8a30c446b350a43b6e6e6fa8adf012e20` | `891043807c7362755dae856c6846a41f2e72e343dadd36360f1348e3196b9c0f` |
| DFlash A8 | `e16814bc452b40f98aef9dd67276bea4cecf55e092cc571f8e42ecbbff0987ad` | `715c7c058368381e10db0bab63d36ffe84e1ce3fa1ca8dab98bb96aed835e6bd` |

Each actual load/injection/intact-seven-slot graph observes **exactly fifteen**
selected I32/W1 projections with correct bits, CPU buffers, mask151669, the same
anchor/noise IDs/positions and finite full-vocabulary logits. FC is outside W1
coverage; there is no selected FFN dense fallback. All four remain untrained,
uncalibrated prototypes: zero optimizer updates and no quality/timing evaluation.

Six sequential jobs used the existing 8 GiB RSS/timeout guards. Maximum sampled
process-group RSS was `6,914,867,200` bytes for export and `2,292,187,136` for graph;
maximum kernel child export peak was `6,932,791,296`. Every run exited0, recorded
its owned group empty and retained exact command/resource receipts under the
existing ignored run directory (`*-ffn15-export`, `*-ffn15-w1a{1,8}-graph`).
The CPU heavy slot was explicitly released; no further model runs were started.
No original or failed artifact was changed and no deployment source edit occurred.

Separate machine-readable checkpoint, leaving the FC16 summary intact:
`models/nine-model-original-snapshots/actual-cpu-ffn15-summary.json`, SHA256
`5868db9d37ed55168150c9ef516ee74b530df72ca56d27f47b277643113cdbc6`.
CUDA/runtime/memory, production calibration/data/QAT and quality/throughput stay
PENDING. Locally absent original frozen Q4 files are still not substituted.
