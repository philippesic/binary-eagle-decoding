# Independent overnight staged-QAT QA

Owner: `prep/nine-model-overnight-qa` in `/private/tmp/nine-model-qat-20261004/overnight-qa`.
Scope: independent CPU/source/packet review. No SSH, remote query, staging, build,
model/capture/training/evaluation, or GPU access. Root assigned sole remote
operation to `overnight_5080_operator`.

## Baseline source audit (`8442d84933eb2336265cbdc534a41b0159a92d8c`)

The per-lane trainer already binds training to an exact config SHA, immutable
bundle SHA, candidate, source inventory, physical GPU UUID and seven PASS gates
(`source`, `resource`, `kernel`, `model`, `backward`, `memory`, and
`capture_portability`). It rechecks admission against the actual visible GPU
UUID immediately before entering the family trainer. The data loaders reject
synthetic training data, and successful receipts require a positive optimizer
step. These are necessary lane-level admission and truthfulness invariants.

Current software couples a first lane to six-lane campaign completion in
several preparation paths: `prepare_nine_model_bundle.materialize_configs`
requires a six-candidate budget marked `human_selected`; `inspect_admission_inputs`
and `nine_model_admission.validate_plan` require all six candidates, all-family
portability inputs, and campaign-wide bindings; `nine_model_pipeline.validate_bundle`
requires six candidate configs/stages, all three Q4 controls, and evaluator
inputs. The proposed staged path should create its own immutable lane manifest
and exact source/config/admission hashes, while leaving the final six-candidate
campaign and cross-model evaluation pending. It must not weaken per-lane gates
or relabel delegated operational budget as human-selected.

## Independent validation

The baseline above predates the staged-lane feature. Final staged implementation
was reviewed and tested at commit
`91d07ef1217c88028369d9b5e724e3a005522b57` in the bundle worktree
`/private/tmp/nine-model-qat-20261004/overnight-bundle`. The added EAGLE
initializer was reviewed and tested at commit
`07d91e113ecfd88e76e9af792f439c1330003059` in
`/private/tmp/nine-model-qat-20261004/overnight-eagle-data`.
The same staged source bytes are integrated in main by `584f337`, and the
initializer source bytes by `d027ea1`. At integration verification, main was
`2925c043046ca7202e5f1a92cf5484a11d3c55c2`; all source/test SHA256 values below
matched the tested worktrees exactly. The integration check used
`git merge-base --is-ancestor 584f337 HEAD` and current-file SHA256 comparison;
the source tests themselves ran at their author commits above.

SHA256 of the exact staged source/test files at the passing run:

| Path | SHA256 |
| --- | --- |
| `scripts/prepare_nine_model_bundle.py` | `b9adebe94e70e2c3ec6b04dcadfde1ac7e928d2bf989be3641eb2ed5ceab4b5a` |
| `src/w1a1_eagle/nine_model_admission.py` | `d451adca8f72aa542e591a4406d8f9daf1f3f9e126c0f2942ca5eb4fec05c749` |
| `scripts/train_nine_model_qat.py` | `658f8bc7b75420818459f20dd89f0b62099e2eb8bc0fbc5c6dafe6b8fe47693f` |
| `scripts/prepare_nine_model_lane.py` | `166f2fa7dfe44c3e052ecbd6cd33c999c72c10c0260526dc04b7acd61dd3a81b` |
| `scripts/run_nine_model_lane.py` | `942f67988962b92e22a0dbd7323422b9a4f10ebdac6c7fc2fe7a36884c205ba3` |
| `tests/test_nine_model_staged_lane.py` | `14cbf8957d39ad21054747cc148130060b8f1f36f1ad82f9e5f8625055d14ca4` |
| `scripts/prepare_eagle_production_initializer.py` | `c1482798bdf7af08890d91df54c9c2edde71918d6f5689a6cdcb18ebd8fe851d` |
| `tests/test_eagle_production_initializer.py` | `4de2ee463e3c5f37fa29d9802ebc9b1b9a76c111f0d62fc410c8f25210e2c3a3` |

The new `nine_model_lane_inputs_v1` and
`nine_model_lane_sm120_plan_v1` paths require exactly one recognized candidate
and only its family's native capture portability input. The original campaign
schema still requires six candidates and three families. The lane runner binds
its exact lane/config/plan/QA/budget/source hashes and preserves pause, lease,
tmux supervisor, GPU UUID, lock, seven candidate checks and release checks;
it leaves evaluation `PENDING` and only accepts a committed positive-update
endpoint. The training identity now includes `continuous_resources.py`,
`continuous_runtime.py`, `nine_model_admission.py`, and the existing
`qat_admission.py` leaf, in addition to `MATH_FILES` and `ADMISSION_FILES`.
The full lane source inventory pins every Python file under `src/w1a1_eagle`
and `scripts`.

For EAGLE, code review confirms the actual fixed-A8 smoke calls
`ContinuousTrainer.smoke`: hard signs are installed during the forward, later
proposal state/key/value gradients must be positive, and all nine binary
projection sign/scale gradients must be finite and nonzero. `smoke_with_training_memory`
reserves two F32 tensors per trainable parameter across backward; the admission
summary requires positive reserved bytes and a CUDA peak-reserved measurement
at least that large, along with existing host/GPU resource gates. Actual
EAGLE data eligibility still depends on the newly bound production payload and
admission; source-only tests do not prove it.

Final focused independent CPU tests passed on macOS ARM64, Python 3.11.3,
Torch 2.8.0. No CUDA availability/query, model load, remote operation, or
training occurred. Commands ran with the tested commit checked out. Their raw
logs are retained under ignored
`/private/tmp/nine-model-qat-20261004/overnight-qa/results/nine-model-qat-overnight/`.
The commands for the staged-lane source were run from
`/private/tmp/nine-model-qat-20261004/overnight-bundle`:

| Command | Result |
| --- | --- |
| `PYTHONPATH=src:scripts:tests python3 -m unittest test_nine_model_bundle_builder test_nine_model_admission_plan_builder test_nine_model_sm120_admission test_nine_model_admission_contracts test_nine_model_staged_lane -v` | PASS, 41 tests including all 7 staged-lane tests |
| `PYTHONPATH=src:scripts:tests python3 -m unittest test_nine_model_training.LauncherContractTests.test_admission_refuses_synthetic_wrong_device_or_stale_source -v` | PASS, admission rejects wrong provenance/device/stale source |
| `PYTHONPATH=src:scripts python3` with the assertion shown below | PASS, 40 identities including all four required runtime/admission leaves |
| `PYTHONPATH=src:scripts:tests python3 -m unittest test_eagle_production_initializer -v` | PASS, 4 EAGLE selection/arithmetic tests at initializer commit `07d91e1` |
| `PYTHONPATH=src:scripts:tests python3 results/nine-model-qat-overnight/check_outer_wall_guard.py` | PASS, refuses outer cap `86399 <= 86400` and accepts `108000 > 86400`; uses temporary source fixtures and mocked trainer/admission dependencies |

The source identity assertion body was:

```python
from train_nine_model_qat import training_source_identity
required = {
    'src/w1a1_eagle/continuous_resources.py',
    'src/w1a1_eagle/continuous_runtime.py',
    'src/w1a1_eagle/nine_model_admission.py',
    'src/w1a1_eagle/qat_admission.py',
}
identity = training_source_identity()
assert required <= identity.keys(), sorted(required - identity.keys())
assert all(len(value) == 64 for value in identity.values())
```

Staged-lane run directory was the bundle checkout above; EAGLE initializer tests
ran from `/private/tmp/nine-model-qat-20261004/overnight-eagle-data`. Raw final
outputs are `final-affected-tests.log`, `final-trainer-admission.log`, and
`final-eagle-initializer.log`, and `final-outer-wall-guard.log`; the ad hoc
guard script hash is `547a8fae023dfc691d44833297092485bc9f13b9ce4aefd4373516372bb46ef6`.
The first two guard-harness attempts failed before reaching the wall-cap check
because the temporary ledger and trainer stubs were incomplete; those harness
fixtures were corrected, and the final guard check passed. See
`outer-wall-fixture-attempt1.log` and `outer-wall-fixture-attempt2.log` for the
setup errors. Earlier six-test/plan-builder snapshots were
before the final staged commit and are historical only. Initial test-discovery
attempts lacked `pytest` or the local `PYTHONPATH`; the corrected final
`unittest` commands passed. Temporary test directories were removed by
`TemporaryDirectory`; no long-lived process or GPU allocation was created.

An additional source-only review of the production EAGLE initializer on
`prep/nine-model-overnight-eagle-data` found that it preserves full prepared
TRAIN authentication and native accepted-prefix ancestry, fits only the
fusion scale against the original BF16 snapshot, and enforces prompt/group/topic/
content-disjoint TRAIN fit and validation splits in prose/code/reasoning. The
writer records per-row raw/native identities and the initializer/reference
hashes, and makes no actor/native/SM120 admission claim. Its four new CPU tests
cover identity rejection, domain quotas, split alias rejection, and a negative
correlation case where signs stay at their original ±0.5 magnitude and scale
goes to zero. This was a source review only; no prepared remote payload or
calibration artifact was consumed by QA.

The final staged tests cover family-specific checkpoint discovery for resume,
one-lane vs full-campaign plan schemas, delegated-budget
truthfulness, selected-family portability, retained seven-gate admission,
pause-before-resource-query, and selected-lane portable QA while other lanes
remain pending. Independent review found no staged-path blocker in these
surfaces. Full actual-model/native/device admission, positive training,
checkpoint continuity, automatic export/evaluation and overall campaign
completion remain outside this CPU QA result.

## Actual initializer attempt 04 metadata

Sole operator transferred only the 6.9 KiB metadata receipt; its raw initializer,
feature rows, row-evidence file, prompt bodies/tokens and source corpus remain
remote. Receipt SHA256:
`15753a4375d5a3cb31a40f6a7c40a558e77708bdf1cb87c86753f8beec729c8c`.
The exact same receipt is preserved at the ignored main-worktree path
`results/nine-model-qat-overnight/eagle-initializer-attempt04.json`.

The receipt says the supervised job exited 0 after 29.507432 seconds (6.967064
seconds fitting) on the RTX5080 host with `CUDA_VISIBLE_DEVICES` empty and
execution device CPU. It created no model/optimizer and made zero updates; GPU
utilization and memory were unchanged, and its supervisor/process groups were
released. The selected source was the frozen 10,000-prompt / 3,899,930-row TRAIN
preparation, READY SHA `bdfa56f8...`, resolved-config SHA
`09e8afa7...`, captured-source identity `b1a9f991...`, original FC weights
SHA `58ac5bbf...`, helper source SHA `49395a80...`, and native source commit
`624f50e74f51b6af93bf6b879f84703e726df172`. The helper SHA matches the current
main-tree file and the source commit recorded by the operator (`0b2ca0a`).
The receipt's CUDA visibility is empty and the selected execution device is CPU;
the RTX5080 is host identity only. It reports utilization 0% and 2714 MiB both
before and after. A systemd 8 GiB cap was configured, but actual cap enforcement
was not measured.

Selection counts join: 32 fit and 16 validation prompts in each of prose, code,
and reasoning; 144 distinct prompts total; 1,536 fit rows, 768 validation rows,
and 2,304 row-evidence records with SHA `acec4f98...`. The receipt records a
TRAIN-only, group/topic/content-disjoint holdout. NPZ member metadata reports
only `fc.latent` ([2560, 7680], F32) and `fc.scale` ([2560], F32), with member
hashes recorded in the local receipt. The whole NPZ SHA is
`ae910cd8...`, initializer JSON SHA is `7d2b89a6...`, and reference magnitude
is 0.5 with reference SHA `9b24af26...`. Per-domain validation RSEs are about
0.0615–0.0635, while the scale-only fit RSE is about 0.0627; these remain
initializer diagnostics, not drafter quality or deployment evidence.

The follow-up receipt adds direct scalar report fields: A8, policy
`preserve_reference_magnitudes`, reference kind `eagle_fixed_reference_0.5`,
orientation rescue false, zero coordinate flips, a 300-second fit cap, and zero
fit events. Its candidate and scale-only fit metrics are equal. It also binds
the original report config hash (`09e8afa...`), prepared directory and full
captured-source identity, matching the initializer preparation checks.

The report and initializer files themselves remain remote, pinned by report
SHA `4c2a75db...` and initializer JSON SHA `7d2b89a6...`; no arrays or prompt
content were transferred. Source/report joins are checked from the bounded
receipt and pinned helper contract, not from report bytes. No member values or
corpus material were copied.

## EAGLE packet helper QA (source commit `4a808b5`)

The metadata-only helper and focused tests were reviewed at final author commit
`4a808b5b8600b74a8201d92272a898e8cd9bb206` in
`/private/tmp/nine-model-qat-20261004/overnight-bundle`, based on `5f53740`.
It is integrated in main at `765d9d1c5d5191950b818178231c7944bc78f8cb`;
file hashes in main match the tested author checkout. SHA256 of the files
tested:

| Path | SHA256 |
| --- | --- |
| `scripts/prepare_eagle_lane_packet.py` | `c7fa42a5a6fb88d83b9f654bef9bcdc6589c5a640b2a682e6a88aeceb582db73` |
| `tests/test_eagle_lane_packet.py` | `48f172f2646d9c528002b2670549d7bbb20d6145aa35cb98416a8cd5fd378e9a` |
| `scripts/prepare_nine_model_bundle.py` | `33c6198daafcb91684279803e570131c639c1b2c14a66b40816f0ba319678bf9` |
| `scripts/train_nine_model_qat.py` | `14964516d085adc1b95125c5d47beb326acac6f02a25dea0e0d98dc06d037d69` |
| `experiments/nine-model-qat-overnight/production-plan.md` | `2ab8d660b7325432099e096a0d1483bc780d6d9f12e33544162608021b111eae` |

Final focused test command, run from
`/private/tmp/nine-model-qat-20261004/overnight-bundle` on macOS ARM64 with
Python 3.11.3 and Torch 2.8.0:

```sh
PYTHONPATH=src:scripts:tests python3 -m unittest test_eagle_lane_packet -v
```

Result: **PASS, 5 tests** at author commit `4a808b5` before the separate
standalone-cadence correction. Raw output is retained in ignored
`/private/tmp/nine-model-qat-20261004/overnight-qa/results/nine-model-qat-overnight/final-packet-tests.log`.

The reviewed source binds exact original `preparation-ready.json` and its
10,000 TRAIN prompt / 3,899,930 supervised-row counts to the initializer report,
original resolved-config SHA and prepared-run directory. It freezes the
24-hour cumulative/30-hour outer budget with human-delegated provenance, the
original config's full data iterator and current cache/head request, with
optimization/readiness claims still pending fresh admission. Binding requires
the actual zero-update preparation receipt, exact config/request/source hashes,
step-zero outer checkpoint and all-nine A8 joint exports, then joins those
bytes to the F16 base, exported model and nine-projection serializer audit.
Generation requests select original TRAIN prompts from each domain. The replay
link pins native generated parents and requests; bind requires an external link
SHA and exact per-domain tokens, chain ancestry, and decode history in native
replay receipts before validating the three goldens.

This proves metadata/source contract behavior only. No actual initializer or
initial model artifact was ingested by QA, and no CUDA/GPU query, remote call,
model load, capture, training, or evaluation occurred. I requested the bounded
attempt-04 report/member/source hashes and selection summary through the sole
operator; raw weights and full corpus are excluded. Actual production evidence
and fresh SM120 admission remain pending independently.

## Overnight continuation cadence correction

The initial packet inherited `development_every=1000`; standalone EAGLE training
returns at that boundary and the staged runner treats the return as interrupted
unless the 24-hour cap was reached. Bundle owner fixed this in commit
`21994b2568d05509986575f5f802568ad873fac1` by setting
`development_every=2**63-1` and reporting the deferred periodic evaluation.
The config remains standalone and time-capped at 86,400 seconds; checkpoint
cadence remains 250 updates with three retained checkpoints. At QA time this
commit had not yet been integrated into main.

The focused packet suite passed **6 tests** on the final cadence-fix commit:
real source config parsing accepts the sentinel and preserves standalone/time-
only controls; a tiny CPU `ContinuousTrainer` runs two actual updates from step
999 through the old boundary to fixture cap 1001 and emits the final development
request; the other packet join tests remain passing. This proves lifecycle
control flow only; it does not prove RTX5080 throughput, actual model readiness,
or a production run.

Test command, run from `/private/tmp/nine-model-qat-20261004/overnight-bundle`:

```sh
PYTHONPATH=src:scripts:tests python3 -m unittest test_eagle_lane_packet -v
```

Tested source hashes:

| Path | SHA256 |
| --- | --- |
| `scripts/prepare_eagle_lane_packet.py` | `6c3946a671fd11990b20a663279c18bda5585c302e33beebf6bfad8cac6533e7` |
| `tests/test_eagle_lane_packet.py` | `b57354160fe577a373fe239d1e5bda5cb18597db5c23424e1caea8fce384c2d4` |

Raw 6-test output is retained under ignored
`/private/tmp/nine-model-qat-20261004/overnight-qa/results/nine-model-qat-overnight/final-packet-lifecycle-tests.log`.
The lifecycle fix is integrated in main at `e00c07395f6f76f2eef3b3f40beb051b60895e50`;
main's helper, test and plan SHA256 values match the tested files above.

## Canonical fusion report schema correction

Operator attempt 01 reached the EAGLE packet helper and exited before packet
publication with a `KeyError` because the helper invented `orientation_rescue`
and `coordinate_flips` report keys. The producer serializes
`dataclasses.asdict(FusionFitConfig)`, whose actual fields are
`zero_scale_orientation_rescue`, `max_coordinate_flips_per_row`, `max_seconds`,
`latent_initialization`, `reference_kind`, and `activation_bits`. Bundle owner
fixed the helper to require the canonical field inventory, instantiate the real
`FusionFitConfig`, verify the actual A8/fixed-reference/no-rescue/zero-flip
contract and initializer policy/reference joins, and reject nonempty fit events.
The test fixture now serializes the actual producer dataclass and has negative
cases for altered bits, policy, reference, rescue, coordinate flips and fit
events. This preserves the calibration math and admission scope.

Independent focused validation at commit
`0daff81f5c34fddbe46f9b9e5a08f4df531d2113` passed **8 tests**:

```sh
PYTHONPATH=src:scripts:tests python3 -m unittest test_eagle_lane_packet -v
```

SHA256: helper `334d7f2742bcb85b3a23fb5ed6b2dd5b4c628deacfbea31e600ef07d5336c06c`;
test `e3dac23139fbaed53a930113ac1ab8dda909326704ca862107bbfdf3f5332d0a`.
Raw output is under ignored
`/private/tmp/nine-model-qat-20261004/overnight-qa/results/nine-model-qat-overnight/final-packet-schema-tests.log`.
This is a source/schema test only; attempt 01 was preserved as failed evidence,
and no model/data/calibration rerun or GPU operation occurred during QA. At this
checkpoint the canonical-field fix was not yet in main. It is subsequently
integrated at `9ed0352` in current main `c690625`; the helper and test file
SHA256 values there match the tested values above.
