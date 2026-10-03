# Copied native CUDA validation — October 3, 2026

The current copied native9e2 runtime passed its CUDA operator/encoder fixture on
RTX5080/SM120 at11:10:47 UTC. This verifies synthetic packing, arithmetic,
nonzero option, loader rejection and encoder placement cases; full-shape model
readiness, optimizer training and Q4_0 acceptance/throughput remain pending.

- Parent/model source: exact6f1444b; compiled native9e2c7a900, build parentb32f7fe.
  Copied runtime retains arithmetic bytes; only verified search-path slots differ.
- Hardware/backend: NVIDIA GeForce RTX5080, SM120, actualCUDA/CUDA0. The fixture
  covers W1A1, W1A4, W1A8 and signed FP16 diagnostic paths. No SM75 claim.
- Passed57pack,114loader,59graph,218arithmetic,36projection cases. Rejected malformed
  loader inputs account for expected stderr error lines. No real model was trained.
- Supervisor4068/group4069 exit0; real exec4092/start1770967/UID1000 observed.
  Complete owned process/group/context return and protected-file equality verified.
- Native report SHAa1ac624a618c0bb5f1ce0d65113467ebb0f7ae8dbfd76531313d9f820023ce22;
  raw transaction SHA50b4aa5f3dd1ca149b79eb8759b4ead2856743792870444e71cd162ffefcb6e0.
  All22collected originals and unchanged Python validator independently checked.

Raw files, copied runtime/toolchain inventory and exact environment/source hashes
remain in ignored runs. See the [active goal checkpoint](../../docs/goals/qat-optimization-readiness.md#actual-copied-native-cuda-fixture-passed--october-3-1110-utc)
for exact paths, hashes, closure correction and next gates. Q4_0 EAGLE remains the
primary comparison; FP16 EAGLE is secondary. No latency, throughput, acceptance,
convergence or training-readiness conclusion follows from this synthetic pass.
