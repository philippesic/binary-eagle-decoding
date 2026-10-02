# Task A checkpoint for the active QAT optimization readiness goal

Bounded deliverable: composed learned-activation + all-row affine midpoint +
nonzero FC correction VJP gate through actual NativeStepAdapter/provider.
Acceptance check: hard values and 36 unique parameter VJPs plus first input
state/cache VJPs match independent local backward algebra at F32
`atol=2e-6, rtol=5e-5`; one joint clipped update; detach/alias controls detect
missing paths. Completed 48 comparisons, three clipped updates, owner 3/3 tests,
Ruff pass. Independent Luna validation completed in disjoint validation files:
3/3 hand-VJP tests plus owner rerun on Torch2.8, evidence commit51df26e with
report/lint polish in f99f03f. Owner
reran all6/6 tests on project Torch2.14. Independent source/API checks cover
two-consumer shared-sum accumulation and F16 correction-factor STE.

Detailed report: `experiments/parallel20261002/auxiliary_vjp/report.md`.
Aggregate tables/source hashes: adjacent `summary.json`. Full ignored output
is preserved in both worker and main `runs/parallel20261002/auxiliary-vjp/`.
At the last check control remained stop=false, original reset1791049896,
82% weekly used. No persistent research process, GPU/Metal/SSH/model/data/final
access. No core or live recipe changes; learned heads stayed serial.

All-row affine source always uses single-forward internally, so the distinct
oracle uses handwritten algebra rather than claiming reference/single-forward
select two affine kernels. Every parameter has a nonzero depth-3 VJP in each
precision. Max VJP error2.3842e-7; max one-update parameter error9.0380e-7.
Detached state/cache preserves logits but fails gradients.

Remaining: report coherent commits to root for integration/push/cleanup.
No CPU research gate remains open for this packet. Root owns the main active goal file;
this checkpoint is its integration-ready durable text within assigned paths.
