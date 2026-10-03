# First 1,000 paired QAT steps and Q4_0 development comparison

The fixed reference A8/A1 pipeline completed 1,000 optimizer updates per model,
published an intact paired checkpoint, and passed a separate native development
evaluation. The checkpoint is substantially below Q4_0 on the measured acceptance
metric. This completes the bounded readiness/training handoff; it does not establish
an acceptance, latency or throughput win, or admit every optional optimization recipe.

## Actual result

Hardware: **RTX 5080 / SM120, CUDA0**. Target weights and target/draft KV are F16.
The primary comparison draft is frozen Q4_0 with native Q8_1 activation conversion;
binary drafts use row W1A8 and row W1A1. Native evaluation used the original frozen
b4 runtime, distinct from current9e2 native unit/model readiness proof.

| Draft | Accepted drafts / round | Relative to Q4_0 | Emitted acceptance / proposals | Development loss |
|---|---:|---:|---:|---:|
| Q4_0 | 1.306255 | 100% | 26.5917% | Not measured |
| W1A8 | 0.127287 | 9.7444% | 2.6037% | 6.930126 |
| W1A1 | 0.069031 | 5.2846% | 1.4132% | 8.373848 |

All 24 response token-ID sequences matched Q4_0 for each binary arm. Each arm
emitted 2,834 tokens. Native raw accepted counts/rounds are A8 320/2,514,
A1 183/2,651, Q4_0 1,608/1,231. Q4_0 emitted 1,606 accepted drafts; the small
end-cap distinction is retained in the original report and the proposal-rate
column uses emitted accepted counts. The per-round column follows the original
raw accepted-drafts convention.

The native subset contains 24 unsealed development prompts from an authenticated
1,002-prompt pool, with one native repetition. Paired loss uses the same 48 selected
rounds and 229 supported labels per model. No sealed-final data was accessed.
The evaluation took 365.0296 seconds, with a 1,200-second bound. This is acceptance
and loss evidence; elapsed evaluation time is not a serving-throughput measurement.
No matched untrained development control was run in this transaction, so it cannot
establish whether these 1,000 updates improved or harmed acceptance.

## Training and admission evidence

Full preparation authenticated 10,000 TRAIN prompts, 3,899,930 supervised rows,
320 TRAIN shards and 353 provider manifests. The capped training run consumed
13 distinct prompts and 4,846 distinct supervised rows, not the whole corpus.

- Fixed A8/A1 reference computation and activations, seeds 8101/1101, warmup100,
  sign LR0.001, scale LR0.00001, gradient clip norm1. Cache/head optimization,
  learned quantizers, alternative binary rules, fusion/affine controls,
  persistent-sign control and curricula were inactive.
- First caps: 1,000 steps, 3,600 trainer-accounted seconds, one epoch; no token cap.
  Original checkpoint250/diagnostics100/development1000 cadence was preserved.
- Actual original checkpoint-zero copy/resume restored both model states and
  optimizer/RNG/cursor state through unchanged producer APIs, then current paired
  backward smoke and state/native deployment gates passed before optimization.
- Both models reached step1000 with finite gradients in all18 sign/scale tensors,
  no latent value outside the clip range, cumulative sign flips A8=398/A1=54,
  and final scale L1 movement A8=0.095762/A1=0.086235.
- Current native CUDA fixture passed57 pack/114 loader/59 graph/218 arithmetic/
  36 projection cases. Separate full-model proof has30 forward/30 backward calls,
  positive later state/K/V gradients, five timing repetitions and12 native cases
  with zero changed/material choices. Peak paired B1 reserved memory9.108GB;
  minimum free6.483GB. The90 independent B1/B2/B4 graph measurements cover three
  variants and both lanes without changing Adam update cadence.

Implementation and independent CPU reviews are published in reports01–11 in this
folder. The original deliverable audit maps computation, cache/head, optimizer,
learned quantizer, curriculum/refresh, correction and affine implementations to
source/tests and permitted deployment limits. Native synthetic optional operators
are tested; inactive optional full-model receipts remain absent. These limits
retain the fail-closed boundaries and do not authorize a new recipe or calibration.

## Preserved failures and recovery

The training supervisor naturally ended at12:41:37.746UTC with exit1: automatic
in-process development deployment preflight required15,032,385,536 available
host-RAM bytes after model/optimizer CPU offload. Step1000 and the paired checkpoint
had already been committed. The original failed training status is preserved;
this report does not relabel that supervisor as successful.

After verified full release, a distinct supervised process evaluated the same
saved checkpoint using the unchanged evaluator, precision, memory gates and locks.
It naturally ended at13:20:22.088UTC with exit0. The first collection wrapper then
failed its16MiB cap by treating raw native heads.jsonl as metadata. A bounded
read-only collector retrieved8 explicit original metadata files (302,734bytes)
and fresh full group/context/source/resource return without rerunning evaluation.
Raw tensors and both failures remain outside Git.

Actual training progressed to terminal after its monitoring connection closed:
closure12:36:31.813UTC, terminal more than306seconds later, same recorded job/source/
command/boot and intact checkpoint. Since the job ended before reconnect, there
is no same-live-birth post-reconnect claim. The prelaunch same-boot CPU durability
probe independently preserved exact supervisor/child births across163seconds
of disconnected heartbeat progress, with instanceIdleTimeout=-1.

Fresh final release at13:29:21.874UTC found both evaluator groups/births absent,
no project submitters, empty GPU contexts/compute-app list, GPUfree13,533MiB,
host available20,272,418,816bytes and diskfree357,356,077,056bytes. The sole
operator's local244 session and keeper71024 closed13:38:45UTC. RTX2080Ti and
supporting research remain paused. No additional training budget or recipe is selected.

## Reproducibility

- Producer source: `6f1444b86dd01862da878c2d5d2434a1d9165c29`.
- Current native gate: `9e2c7a90051e738751aab7d7bd7c2d8201fb76e3`.
- Prepared helper publication: `d211275fc26a7cb0bfcf6c71f30c7396addcd972`.
- Full captured source: `b1a9f991246f07c3b3b078f2602ace339416904295a303f77c2d72c6c1427509`.
- Stage configuration: `ccc43a104a3a09e1a910746c319f55df33761bec5616378254a90e9339dbdafe`.
- Original ready: `bdfa56f8b10e44e82a6d807a71f32d68c39143af7094e6f8f0da63504d41a498`.
- Current full-model readiness: `3badd9775a848ff5ace09fd96e573f96e19b86e85a88a9bb81aa18d8d3e608c7`.
- Original paired zero: `e9d01984f6323cc9789b110cc3029e43f97f5596b3840fb5911256202adf2f15`.
- Paired step1000 resume: `62f88fdd4c696fbfa4e6d0aeba0b260debab0d0a132886ce0050787e34200cb6`.
- Step1000 manifest: `b48c556510697b6f178e671808a96cc5b3e84692f1d179898269b07d0d6fe63e`.
- A8 exported joint bundle: `d29b30d33700b8fc6e8e723680068a9739d75fab9d156f3014887535255b8b9b`.
- A1 exported joint bundle: `2f84a7b55f49463a411eaf2e95c3d578b7147f7b1f8abb80d4bbfecba7929e21`.
- Frozen evaluator binary: `b5093749d67888bc2cafdb6a65c479f4c182f0a904820f1dae4870b6ae66d41c`.
- Frozen target GGUF: `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`.
- Q4_0 baseline GGUF: `2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280`.
- Fixed development subset: `131a3db7958ff6aa818b23019297654507d5b80bed3c298349417b7e3b2ba081`.
- Original development report: `fc4f6bdf3379c621f6b187a979082d4d6fc97821a5a224f77f74ebd47e3be5b9`.
- Authoritative read-only metadata stdout: `6b0c382b53c37e95f29d2c2a92288a7129017f63b7a72df6f16fd6a1a771f726`.
- Final local closure: `e65706f238bc81e7dbd505aa42059ad94467f5100f08512c1cb50877223a534d`.

Runtime: Python3.11.15, NumPy2.4.6, Torch2.14.0+cu130; highest F32 matmul,
matmul TF32false and cuDNN TF32true. Original raw report/metadata manifests,
weights, captures and runs are retained outside Git under the owned project runs.
The native feature worktree and local branch were clean/published and retired;
its fork publication ref remains to keep the parent gitlink fetchable. Unmerged
and paused unrelated research worktrees remain untouched.

Current audit/acceptance indices are in ignored
`runs/qat-optimization-readiness/actual-final-development-root-verified-20261003-02/`,
`actual-final-development-root-acceptance-20261003-02.json` and
`original-deliverables-completion-audit-20261003-01.json`.

## Remaining limits and research decision

The current bounded handoff is operationally demonstrated only for the admitted
fixed A8/A1 recipe on RTX5080/SM120. It is not global optimized-recipe readiness,
A4/curriculum admission, a real correction fit, or SM75 performance validation.
The original in-process development RAM limitation remains; this run used a
verified separate evaluation. Future training requires an explicit budget and
appropriate resume/evaluation lifecycle rather than replaying this run's permits.

Q4_0 remains the model to beat. Broader fixed-recipe training coverage and a
separately admitted optimization/refresh recipe are different next research
options. The 1,000-step result does not choose between them. The user owns that
fork; see the pending decision in `docs/DECISIONS.md`.
