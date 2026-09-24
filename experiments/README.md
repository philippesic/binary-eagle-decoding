# Experiment record

Use one Markdown report per experiment with: question, project/runtime revisions,
resolved config, data/model hashes, command, result, raw artifact location, and
decision. Include failed experiments. Keep large outputs in `results/` or named
external storage.

Current acceptance evidence: [Metal development sweep](pytorch-w1a1-metal-acceptance.md),
[RTX 5080 CUDA diagnostic](pytorch-w1a1-cuda-acceptance.md), and
[W4A4/W8A8 CUDA acceptance](pytorch-int4-int8-cuda-acceptance.md), plus the
[static drafter coverage](drafter-static-coverage.md). Target-only, ordinary
EAGLE, and native W1A1 throughput anchors belong to the later paired runtime
comparison on each relevant GPU.
