# Continuous W1A8 / W1A1 CPU corpus preparation

This is a **prompt corpus**, not supervised training tensors or an eligibility
claim. The primary prepared tier has 10,000 unique prompts; its adequacy for
binary learning needs learning-curve and native development evidence. No model
weights, accelerator operation, GPU query, remote operation, teacher capture or
training ran for this preparation.

The primary ignored output is
`data/continuous-w1ax/freeze-002/manifest.json` in the integration checkout.
Its byte-identical tracked metadata is
[continuous-w1ax-freeze-002.manifest.json](continuous-w1ax-freeze-002.manifest.json).
The manifest SHA256 is
`9fc8caa80f2c29785eeaeff01f3875c27fee46853350de9ceebe682f8d12b8dc`.
It includes every prompt/index shard hash, counts, source revisions and licenses,
opaque exclusions, transformations, tokenizer hashes and package versions.

| Split | Unique prompts | Prose / code / reasoning | Chat input tokens | p50 / p95 / max input tokens |
| --- | ---: | --- | ---: | --- |
| Train | 10,000 | 3,334 / 3,333 / 3,333 | 1,502,567 | 87 / 393 / 1,301 |
| Development | 1,002 | 334 / 334 / 334 | 161,458 | 97 / 409 / 1,781 |
| Sealed test | 1,002 | 334 / 334 / 334 | 152,759 | 86 / 406 / 1,077 |
| Train reserve | 24,507 | 7,170 / 15,328 / 2,009 | 5,134,237 | 216 / 431 / 1,784 |

Counts of input tokens sum tokenized prompt lengths once per unique prompt.
They are neither unique vocabulary tokens nor supervised teacher predictions.
The training engine must report repeated presentations separately. Reserve is
independent of development/test but is deliberately **not balanced**; promotion
requires an explicit domain sampler or a new balanced tier. It is not silently
included in the default 10,000-prompt tier.

There are eight Dolly task categories and nine Magicoder code languages. Train
has 5,760 inputs below 128 tokens, 4,073 at 128–511, and 167 at 512–1,792.
This tier chiefly covers short and medium prompts; it does not establish broad
long-context coverage. The configured hard cap of 1,792 chat input tokens leaves
256 tokens in a 2,048-token context for the planned 128-token generation and
recurrent/tree margin. Longer inputs are rejected, never truncated: 28 exceeded
that cap. Actual native token-prefix/context checks still gate user-started
capture; local tokenizer counts do not replace native prefix ancestry.

Sources are pinned in `configs/continuous_w1ax_sources.json`:

- [Databricks Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/README.md),
  revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, CC-BY-SA-3.0;
  all 15,011 rows considered, including all eight task categories. Contexts
  group related source prompts. Attribution/ShareAlike obligations are retained
  in the source lock; this is not a license change.
- [OpenAI GSM8K](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE),
  revision `3101c7d5072418e28b9008a6636bde82a006892c`, MIT; 7,473 upstream
  **train** rows considered. Upstream test data was never downloaded. Reasoning
  diversity here is arithmetic word problems, not a general reasoning benchmark.
- [Magicoder OSS-Instruct](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K/blob/5f839b1f368a76b161028bb9edff055db34022b2/README.md),
  revision `5f839b1f368a76b161028bb9edff055db34022b2`, upstream dataset card
  declares MIT. The pinned raw download contains 75,197 synthetic instruction
  rows derived from OSS seeds; the initial tier considers its first 20,000.
  Reused seed hashes group related problems. Its upstream declaration and
  synthetic generation provenance are recorded, rather than asserting an
  independent audit of all originating OSS licensing or model-output rights.

Source answers, reference solutions and code seeds are not teacher labels or
model inputs. Source line/index identities are stable. Magicoder's `index` is
not globally unique; IDs also contain the immutable source line number.
Downloads must match locked SHA256; no rolling revision or unpinned fallback is
accepted. Raw files stay ignored. CPU tokenization uses `tokenizers==0.22.1`,
`jinja2==3.1.6` and `numpy==1.26.4`, with the frozen Qwen3-4B tokenizer revision
`1cfa9a7208912126459214e8b04321603b3df60c` and verified tokenizer/config hashes.
The pinned chat template sets generation prompt true and thinking false. A
CPU-only check matched 26 chat tokens against the frozen local AutoTokenizer.

Deduplication runs globally before splitting: exact normalized word equality,
then 4×4 XOR-minhash candidates verified by five-word shingle Jaccard ≥0.88.
It removed 118 exact and 214 detected near duplicates. This is approximate
candidate search and does not prove semantic decontamination. Whole connected
source groups/topics remain in one split. Source-qualified IDs, group/topic
keys and exact message hashes from the prior 2,000/192/192 public freeze and
intermediate continuous freeze-001 dev/test opaque indexes were excluded,
removing 3,613 rows (including grouped context/seed relatives). Another 2,000
rows failed normalized character filters.

No historical or newly generated sealed prompt file was opened for inspection,
hashing or evaluation. Output hashes are accumulated while files are written;
sealed aggregate counts derive from in-memory preparation before publication.
Old final indexes contain no shingle fingerprints, and older historical finals
have no available opaque decontamination index. **Cross-legacy semantic/near
leakage is unproven.** The intermediate `freeze-001` remains immutable (manifest
SHA256 `8a801509fdab9166ef7a963ccfd199b04578fa85590c163e2e6ddcfe9cbdb2cc`);
it used an 8,192-token limit and is superseded for this 2,048-context experiment.
Its sealed test and development rows stay excluded from the primary tier.

Run from the integration checkout using the existing frozen tokenizer and
opaque indexes. The output must be a new ignored directory:

```sh
python3 scripts/prepare_continuous_w1ax_data.py \
  --raw-dir data/continuous-w1ax/sources/raw \
  --tokenizer-dir models/hf/Qwen3-4B \
  --output data/continuous-w1ax/reproduction-002 \
  --exclude-index data/w1a-public-freeze-002/final.index.jsonl \
  --exclude-index data/w1a-public-freeze-002/dev.index.jsonl \
  --exclude-index data/w1a-public-freeze-002/train_large.index.jsonl \
  --exclude-index data/continuous-w1ax/freeze-001/dev-00000.index.jsonl \
  --exclude-index data/continuous-w1ax/freeze-001/dev-00001.index.jsonl \
  --exclude-index data/continuous-w1ax/freeze-001/sealed_test-00000.index.jsonl \
  --exclude-index data/continuous-w1ax/freeze-001/sealed_test-00001.index.jsonl \
  --train 10000 --dev 1002 --sealed-test 1002 --max-tokens 1792 --offline
```

Omit `--offline` to fetch any missing pinned raw sources. `--fetch-only` verifies
or downloads sources without preparing splits. New machines need the retained
opaque exclusion indexes as well as the source lock; skipping exclusions is
not reproduction. Publication uses a sibling temporary directory and atomic
rename; failed writes remove only that temporary directory. Existing freezes
are never replaced, and repository destinations must be Git-ignored.

The raw iterator and 1,000-row JSONL shards support streaming. `--code-limit 0`
considers the full pinned code corpus; changing tier size/seed requires a new
output and all retained dev/sealed-test opaque indexes, including freeze-002.
The frozen GSM8K reasoning pool bounds the maximum balanced tier. Expanding
reasoning diversity requires another reviewed pinned source/adapter, not
repeating prompts and counting repeats as unique data. Long-run data adequacy
must be measured; no finite tier is promised to be enough.

Measured ignored storage is 220,453,028 bytes for sources (including a 314-byte
Magicoder card), 46,365,760 bytes for primary freeze-002 and 48,746,122 bytes
for retained intermediate freeze-001: about 301 MiB total. Teacher features,
recurrent captures, checkpoints and teacher-generated tokens are additional
future storage. This report makes no estimate from prompt downloads of native
teacher tensor volume or CUDA memory. Checks ran on local Darwin/arm64 CPU, with no accelerator precision result.
An actual opaque index audit confirmed disjoint source IDs, groups, topics and
exact message hashes across all four splits. All index hashes and the train/dev/
reserve prompt hashes passed; sealed prompt bytes were not reopened.
CPU tests: 8/8 new corpus fixtures passed
in 0.129 seconds and 4/4 existing preparation tests in 0.048 seconds; Ruff
passed. Tests cover source/tokenizer integrity, grouped split disjointness,
legacy exclusion, exact/near duplicate fixtures, code source ID reuse, sealed
text reopen rejection, atomic write failure cleanup and ignored destinations.

Remaining user-start gates: native exact-prefix teacher capture/refresh, v2
hard-CE label audit, A8/A1 numeric/cache/backward/export eligibility and actual
CUDA memory/resource smoke. Sealed tests remain unavailable for ongoing
selection. Data preparation used local CPU only and activated no monitor.

## Manual transfer packet

`python3 scripts/pack_continuous_w1ax_inputs.py` accepts `--manifest`,
`--source-lock`, `--output` and repeated `--exclusion-index` arguments. Use
`data/continuous-w1ax/freeze-002/manifest.json` and the same seven opaque indexes
listed in the reproduction command above (spell the packaging flag
`--exclusion-index`). The actual ignored packet is
`data/continuous-w1ax/launch-inputs.tar`: **22,016,000 bytes**, SHA256
`b5ce086e87dbdc57e7a0543607f5834ee201fb0c585572fcb63bc9379cb3dc31`.
Its `.tar.json` receipt records the same hash/counts. Tar member headers have
fixed mode, timestamp, owner and sorted order; three CPU packet tests passed in
0.011 seconds, including byte-identical repetition, malicious path rejection,
wrong hashes/exclusions, sealed filename mislabel and zero sealed payload reads.
Ruff and an actual packet ledger/hash audit passed on local Darwin/arm64 CPU.

The packet includes the original manifest, locked source config, **train/dev
prompt payload only**, all split opaque indexes, and all seven prior opaque
exclusion indexes. It has 61 members and includes neither raw sources, tokenizer/
model weights, sealed test prompt text nor default reserve prompt text. A trusted
user may explicitly add reserve payload with `--include-reserve` into a new
packet. Original manifest hashes of omitted sealed payloads remain preserved;
their presence is not a training-readiness claim. Do not require or open those
payloads when validating a training launch.

Transfer this existing packet by the user's manual approved route; agents must
follow the shared registry/tmux MCP rules for any later remote SSH operation.
No remote action or service was started during preparation. After manually
placing the trusted packet on the chosen host, verify and extract it from that
host's project root:

```sh
printf '%s  %s\n' \
  b5ce086e87dbdc57e7a0543607f5834ee201fb0c585572fcb63bc9379cb3dc31 \
  launch-inputs.tar | sha256sum -c -
tar -xf launch-inputs.tar
```

The primary manifest then resolves as
`data/continuous-w1ax/freeze-002/manifest.json`, retaining its exact
`9fc8caa8…` hash. Packet metadata is at
`data/continuous-w1ax/launch-packet.json`; old exclusions resolve under
`data/continuous-w1ax/exclusions/<index-sha256>/<original-index-name>`.
Extraction only prepares files. The manual launcher must still perform native
teacher capture/audit and A8/A1 readiness before advancing either optimizer.
