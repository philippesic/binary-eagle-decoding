# Decision log

Record research or infrastructure forks when evidence could lead to different
next steps. Keep the current decision and the reason; revisit when new data
changes the tradeoff. Routine implementation choices belong in commits.

| Date | Decision | Basis | Revisit trigger |
| --- | --- | --- | --- |
| 2026-09-23 | Use Qwen3-4B + AngelSlim EAGLE-3 on a pinned llama.cpp submodule as the starting pair. | Pinned upstream documents this conversion and runtime path. | Baseline incompatibility or memory limit. |
| 2026-09-23 | Use RTX 5080 for available experiment work; reserve RTX 2080 Ti for native SM75 binary timing. | User hardware access and Turing hypothesis. | Hardware access changes. |
| 2026-09-23 | Defer target-only and FP16 EAGLE measurements until direct comparison with native W1A1. | The published EAGLE baseline is the functional starting point; the user wants matched measurements alongside binary execution rather than a separate baseline milestone. | A setup issue blocks W1A1 development, or paired comparisons require an earlier diagnostic. |
| 2026-09-23 | Rotate a long Codex task after its second compaction, using a written checkpoint. | Context drift in previous long runs; hooks can inject post-compact instructions but cannot launch a successor. | Handoff fails in practice or hook behavior changes. |
