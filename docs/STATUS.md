# Current project status

**Active goal:** [complete W1A1 EAGLE research program](goals/full-w1a1-eagle-project.md).
The user delegated project research decisions and granted RTX 5080 access for
an initial roughly 10-hour autonomous work window. The first work units are a
bounded BF16 verifier-parity investigation on the 5080, a CPU binary
reference/packer, and native-path planning. Luna operator
`/root/cuda_acceptance_operator` alone owns the 5080; Sol worker
`/root/binary_reference` delivered the CPU packed reference at `f8f3209`
(38 passing tests; worktree cleaned); `/root/parity_advice` is auditing the
native GGML path. The GPU parity operator has passed preflight and is running
bounded diagnostics. The BF16 fake-binary versus FP32 native rounding
contract must be reconciled before native acceptance claims.
The [native W1A1 design](native-w1a1-path.md) uses a dedicated packed
operation and ordinary I32/F32 GGUF companions, initially for the EAGLE head.
A [bounded head-only QAT pilot](../experiments/qat-head-pilot-plan.md) is
planned after the current parity diagnostic. The BF16-forward-exact trainable
head core is integrated at `5cb3ea4` with 43 passing tests, while a separate
capture path was integrated at `f88eaa9`. A local deterministic generator
created ignored 96/24 train/validation prompt manifests, disjoint from the
held-out set; all 48 checks pass. No QAT GPU job has started.
The completed [RTX 5080 W1A1 draft acceptance goal](goals/rtx5080-draft-acceptance.md) and
`experiments/pytorch-w1a1-cuda-acceptance.md` contain the pinned CUDA setup,
strict BF16 parity diagnostic, full 12-prompt exploratory acceptance sweep,
and drafter layer-cost audit. Ordinary EAGLE accepted 2.317 drafts/round,
head-only W1A1 1.677, and all listed W1A1 groups 0.202 under the same
verifier. Strict target-only/ordinary parity failed at generated token 4 due
a target-selected BF16 verifier tie; the deeper logit shift is unresolved.
Candidate linears occupied 38.32% of one-prompt instrumented draft time. These
are verifier-relative acceptance and diagnostic timing results, not native
binary or end-to-end speed claims. All supervised runs ended, the GPU returned
to idle, and tmux SSH sessions were closed.

**Current research fork:** first investigate BF16 verifier parity while a
packed-binary numerical reference proceeds independently. Then choose narrow
coverage and/or bounded drafter QAT from the measured acceptance and layer
cost evidence. The user delegated these research choices for this work window;
record each in `docs/DECISIONS.md`. No QAT or native binary work is active yet.
The user also requested a follow-on INT4/INT8 accepted-per-round comparison;
its [completed report](../experiments/pytorch-int4-int8-cuda-acceptance.md)
records the same 12-prompt RTX 5080 comparison. Accepted drafts/round were
ordinary BF16 2.3166, W4A4 simulation 0.2882, and W8A8 simulation 2.1816.
Both operands were quantized for the selected drafter linears, with FP32
simulated accumulation and BF16 output. The known target/verifier parity
limitation remains; these are verifier-relative counts, not native INT4/INT8
speed results. All supervised jobs ended, the GPU returned to idle, and SSH
sessions closed. The [measurement plan](../experiments/int4-int8-acceptance-plan.md)
has the implementation and run checkpoint.
The prior [PyTorch W1A1 EAGLE
goal](goals/pytorch-w1a1-eagle.md) and
`experiments/pytorch-w1a1-metal-acceptance.md` contain the Metal development
evidence and its BF16 greedy-parity limitation.

**Repository:** the published target/draft pair, PyTorch acceptance simulation,
RTX 5080 development measurements, and CPU packed-binary reference are in
place. Native GGML execution, local GGUF conversion, RTX 2080 Ti binary
measurements, and paired throughput remain unvalidated. See
`docs/PROJECT_OVERVIEW.md` for the overall research gates.

When a goal is active, link its `docs/goals/<slug>.md` here and summarize the
current stage, owner tasks, live remote jobs, next actions, and user decisions.
