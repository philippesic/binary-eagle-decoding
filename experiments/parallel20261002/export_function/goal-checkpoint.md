## Supporting exported composed function CPU audit — October 2

Owner `/root/export_function`, isolated worktree
`/private/tmp/eagle-parallel-20261002/export-function`, branch
`research/20261002-export-function`; independent Luna validator
`/root/export_function/decode_validation`. Bounded deliverable is actual
all-nine joint-checkpoint -> unchanged exporter -> serialized GGUF composed
forward reconstruction. Final report and source-bound summary are in
`experiments/parallel20261002/export_function/`.

All81 projection comparisons across9 synthetic A1/A4/A8 exports matched the
training hard outputs exactly (predeclared atol3e-5/rtol3e-6). Independent
NumPy decoder also matched all81 exactly. Coverage includes Q32/K8 nonidentity
inverse permutation, all-row affine midpoints, six learned quantizers,
nonzero rank1/4 raw fusion correction with F16-rounded factors and F32 bias,
zero signs/scales, subnormal input rows and K33/65/130 tail packing. Nine actual
GGUF midpoint byte mutations changed forward values; F32-master factors differ
from the required F16-rounded factors, so precision control is observable.

No production/native source changed; no fix is proposed from these bounded
cases. This is CPU serialization semantics, not native backend validity,
SM75 performance, held-out acceptance, or Q4_0-relative throughput. Current
QAT/preparation admission gates and GPU ownership remain separate. No models,
captures, optimizer, GPU/Metal/build/SSH or persistent research process was used.

Owner3/3 focused tests, Ruff and whitespace passed. Independent validation
passed9/9 cases/81 projections. Raw synthetic checkpoints/GGUF/results are
already retained in MAIN ignored `runs/parallel20261002/export-function-final/`
and survive worktree retirement. Exact SHA256 values are committed in the
summary; final integration commit IDs are supplied to the orchestrator.

Remaining work: orchestrator reviews/cherry-picks owner and independent
validator commits, appends this checkpoint to the active goal, pushes main,
then removes the merged worktree/branch. No research extension is needed.
