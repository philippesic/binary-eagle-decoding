# Public-source candidate freeze for W1A CPU Phase 1A

Prepared on 2026-09-28 with `scripts/fetch_w1a_public_sources.py` and
`scripts/prepare_w1a_data.py`. This is a **candidate** 2,000-prompt training
set plus independent 192-prompt development and 192-prompt new final sets.
It has not been tokenized, natively captured, quality-reviewed or admitted
to training. Raw files, normalized prompts, split prompts and indexes are in
ignored local `data/`; this report contains only provenance and counts. The
legacy final set was not opened. No target/draft model or accelerator ran.

## Pinned sources and declared terms

| Domain | Source and exact revision | Declared data license | Raw SHA256 | Normalized SHA256 |
| --- | --- | --- | --- | --- |
| Prose | [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a), `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | [CC BY-SA 3.0](https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/README.md) | `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec` | `3a66d43984b154fe82302ac4be97448222e15607a2164f8557d211e236b49e3e` |
| Reasoning | [OpenAI GSM8K train](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c), `3101c7d5072418e28b9008a6636bde82a006892c` | [MIT](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE) | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` | `4e2c9a6ca7bdd55920335c85f3f1fdfae0502b9f0a4fb19f87c844d8a6cd5c89` |
| Code | [Google Research MBPP](https://github.com/google-research/google-research/tree/d36068b845da4c2b24927fee2cea1e6ef98dadda/mbpp), `d36068b845da4c2b24927fee2cea1e6ef98dadda` | [CC BY 4.0 for datasets](https://github.com/google-research/google-research/blob/d36068b845da4c2b24927fee2cea1e6ef98dadda/README.md) | `ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f` | `c159269fd7cc24d9c22442b246a973867b1c1c446e606f10a2df76a6b52aa938` |

Exact raw download URLs and expected SHA256 digests are constants in the
fetcher; downloads fail closed on a changed hash. Its catalog records source
URI, revision, license, raw hash, normalized hash, transform name and transform
script hash. Dolly contributes its `brainstorming`, `creative_writing`,
`general_qa`, `open_qa` and `summarization` categories. The user instruction
and optional context become a single user turn; repeated exact contexts share
a topic ID. GSM8K contributes training questions only, and MBPP contributes
task text plus public test assertions; reference answers/solutions are not
included in prompts. The [Dolly card](https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/README.md)
explicitly permits training use, including commercial purposes, under its
declared CC BY-SA 3.0 terms. GSM8K's pinned repository declares MIT, and
Google Research's pinned README declares CC BY 4.0 for its datasets; both
permit research use under their stated terms. Preserve attribution and license
notices for all three; Dolly's ShareAlike condition applies to redistribution
of adapted dataset material, while MBPP's CC BY terms require attribution and
indication of changes. This records source terms, not a legal determination
about a future model release. The transformed prompts and trained model, if
any, need separate release review.

To reproduce into **new** ignored directories:

```sh
python3 scripts/fetch_w1a_public_sources.py \
  --directory data/w1a-public-sources/pinned-v1 \
  --freeze-output data/w1a-public-freeze-002 --seed 1429
```

The captured catalog SHA256 is
`498017fd13a49c97f782ff50035d544dcf98bb70583ffadf68d61236c2415647`.
The candidate freeze manifest SHA256 is
`dc37f752bb137054183162dcb7ca96004edabeb752bd611996d9cb289bfdc667`.
The output directory existed locally after this run, so reproduction must use
new directory names. A second preparation using the same pinned raw files
produced identical split prompt and index hashes. Do not open the candidate
final prompt JSONL for selection or tuning.

## Counts from manifest and index metadata

| Split | Prompts/groups | Prose | Code | Reasoning | Prompt SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Train 2k (`train_small` and `train_large` in this candidate) | 2,000 / 2,000 | 667 | 667 | 666 | `bfbe292fc8c19f4b4394dbcb6577140cbabf435bd530648adedaba71845ea647` |
| Development | 192 / 192 | 64 | 64 | 64 | `89b37b0c22f4807813918c598494415f00f3e5b2e35487a6ef3890aff229eade` |
| New final, sealed | 192 / 192 | 64 | 64 | 64 | `3c2d6dcf665887fd836b0ec097a3ab72de38df65fe102cbe1c3ad9ce24de6b4b` |

Raw source rows: Dolly 15,011; GSM8K train 7,473; MBPP 974. Dolly's selected
categories give 9,596 normalized rows. After the 64–24,000 normalized-character
filter, eligible counts are Dolly 3,065, GSM8K 7,468 and MBPP 974. Across
all three, 28 exact and 19 heuristic near duplicates were dropped, leaving
11,460 unique eligible prompts. Metadata checks found zero overlaps in prompt
IDs, declared conversation groups or exact content hashes between train,
development and new final. The 2k training rows are an exact subset of the
nominal large tier. These checks did not read final prompt contents.

## Coverage limits and next gate

This candidate has one upstream source per domain. Code is Python task
writing, reasoning is grade-school arithmetic, and prose is Dolly instruction
following; these are not broad domain coverage. MBPP is a common code
benchmark, so the code split may overlap target pretraining or external
benchmark material. The heuristic near-duplicate search can miss paraphrases.
The manifest records word counts, not target tokens. It does not validate chat
template rendering, EOS/truncation, output length, label alignment, answer
quality or mapped draft-vocabulary support. Those must be measured before
training, using pinned native tokenizer/capture and no legacy final prompts.

A balanced 10k train tier plus 192/192 held-outs requires roughly 3,462
prompts per domain. This exact catalog has only 974 MBPP rows and 3,065
eligible Dolly rows, so it **cannot** supply that tier. A further source and
freeze design must preserve these candidate held-outs and 2k membership.
[APPS](https://huggingface.co/datasets/codeparrot/apps/blob/21e74ddf8de1a21436da12e3e653065c5213e9d1/README.md)
is a possible next code source: its primary card declares MIT and 5,000 train
problems, but its scraped problem provenance and overlap with MBPP require
review before inclusion. For prose, one option is a pinned slice of
[UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/blob/b7fe606ecdbf71e8537946a8d9de5ccf0f6da48b/README.md),
whose card declares MIT; its synthetic conversation distribution should be
measured separately from Dolly. No APPS or UltraChat records entered this
candidate freeze.
