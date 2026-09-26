# Experiment record

Use one Markdown report per experiment with: question, project/runtime revisions,
resolved config, data/model hashes, command, result, raw artifact location, and
decision. Include failed experiments. Keep large outputs in `results/` or named
external storage.

Current acceptance evidence: [Metal development sweep](pytorch-w1a1-metal-acceptance.md),
[RTX 5080 CUDA diagnostic](pytorch-w1a1-cuda-acceptance.md), and
[W4A4/W8A8 CUDA acceptance](pytorch-int4-int8-cuda-acceptance.md), plus the
[static drafter coverage](drafter-static-coverage.md). Target-only, ordinary
EAGLE, and native W1A1 throughput anchors were measured in direct paired
runtime comparisons on each relevant GPU.

The completed [RTX 2080 Ti synthesis](rtx2080ti-synthesis.md) summarizes
the full precision suite; [detailed provenance](rtx2080ti-quantization-suite.md)
preserves paired rates, acceptance, commands, raw hashes, and execution paths.

A separate [seven-category local research review](research-review-2026-09-25.md)
collects advisory analyses of QAT, architectures, throughput policies, graph
optimizations, precision, data and evaluation. It contains proposals and source
audits, not new GPU measurements.

The [primary-source cross-reference](research-cross-reference-2026-09-25.md)
records literature-supported revisions to that review, with source versions,
precision/hardware limits and retained local measurement gates.
