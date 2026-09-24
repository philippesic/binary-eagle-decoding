# Current project status

**Active goal:** [complete W1A1 EAGLE research program](goals/full-w1a1-eagle-project.md).
The user delegated project research decisions and granted RTX 5080 access for
an initial roughly 10-hour autonomous work window. The 5080 BF16 parity
diagnostic finished: the ordinary EAGLE tree verifier ties IDs 11/272/7578 at
21.0 at generated position 40, while both incremental and full-prefix target
paths choose 272. Mutating off-path siblings did not change the selected row,
so BF16 tree arithmetic and tie sensitivity explain the immediate divergence.
Acceptance counts remain verifier-relative. The parity job is stopped and GPU
ownership has passed through native prototype validation to
`/root/cuda_acceptance_operator` for supervised QAT capture/training. The
BF16 fake-binary versus FP32 native rounding
contract must still be reconciled before native acceptance claims.
The [native W1A1 design](native-w1a1-path.md) uses a dedicated packed
operation and ordinary I32/F32 GGUF companions, initially for the EAGLE head.
A [bounded head-only QAT pilot](../experiments/qat-head-pilot-plan.md) is
planned after the current parity diagnostic. The BF16-forward-exact trainable
head core is integrated at `5cb3ea4` with 43 passing tests, while a separate
capture path was integrated at `f88eaa9`. A local deterministic generator
created ignored 96/24 train/validation prompt manifests, disjoint from the
held-out set. Bounded head-only trainer, derived BF16 checkpoint exporter, and
standalone CUDA XOR/popcount prototype have been integrated. All 56 checks
pass. The standalone kernel passed real 5080 correctness and two
packing-inclusive component benchmarks; see
[the report](../experiments/native-w1a1-cuda-prototype.md). QAT capture is
running on the 5080. The llama.cpp submodule combines the dedicated packed
W1A1 CPU op, CUDA backend, and EAGLE packed-head converter/loader in pushed
fork commit `d9ab59c`. Mac build, converter tests, and a local packed-draft
smoke passed; GGML CUDA execution remains untested. Pinned target and ordinary
EAGLE draft FP16 GGUF files were [converted and load-checked](../experiments/gguf-baseline-conversion.md).
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

**Current sequence:** finish bounded head-only QAT capture/training/held-out
gate, validate native GGML CUDA dispatch on the 5080, then measure paired
target-only/ordinary/native
throughput. The RTX 2080 Ti host address is still missing, so SM75 claims
remain pending while 5080 work continues.
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
