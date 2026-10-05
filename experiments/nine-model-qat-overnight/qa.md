# Independent overnight staged-QAT QA

Owner: `prep/nine-model-overnight-qa` in `/private/tmp/nine-model-qat-20261004/overnight-qa`.
Scope: independent CPU/source/packet review. No SSH, remote query, staging, build,
model/capture/training/evaluation, or GPU access. Root assigned sole remote
operation to `overnight_5080_operator`.

## Current source audit (baseline `8442d84933eb2336265cbdc534a41b0159a92d8c`)

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

Review target: `prep/nine-model-overnight-bundle` at base
`8442d84933eb2336265cbdc534a41b0159a92d8c`, with staged source uncommitted at
review time. The new `nine_model_lane_inputs_v1` and
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

Focused independent CPU tests passed on macOS ARM64, Python 3.11.3, Torch 2.8.0.
No CUDA availability/query, model load, remote operation, or training occurred.
Raw logs are retained in ignored
`results/nine-model-qat-overnight/` in the bundle worktree. Commands were run
from `/private/tmp/nine-model-qat-20261004/overnight-bundle`:

| Command | Result |
| --- | --- |
| `PYTHONPATH=src:scripts:tests python3 -m unittest test_nine_model_staged_lane -v` | PASS, 6 staged-lane positive/negative tests |
| `PYTHONPATH=src:scripts:tests python3 -m unittest test_nine_model_admission_plan_builder -v` | PASS, 4 plan/CLI/source-binding tests |
| `PYTHONPATH=src:scripts:tests python3 -m unittest test_nine_model_training.LauncherContractTests.test_admission_refuses_synthetic_wrong_device_or_stale_source -v` | PASS, admission rejects wrong provenance/device/stale source |
| `PYTHONPATH=src:scripts python3` source-identity assertion | PASS, 40 identities including four required runtime/admission leaves |

Raw results: `plan-builder.log` and `trainer-admission.log`. The staged-lane
stdout was captured by the tool and recorded in `staged-independent.log`. An
initial `pytest` attempt was unavailable (`pytest` executable and module absent);
the supported repository `unittest` runner was used instead. A first unittest
import omitted `PYTHONPATH` and failed to locate the local package; the corrected
commands above passed. Temporary test directories were removed by
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

The new tests cover one-lane vs full-campaign plan schemas, delegated-budget
truthfulness, selected-family portability, retained seven-gate admission,
pause-before-resource-query, and selected-lane portable QA while other lanes
remain pending. Independent review found no staged-path blocker in these
surfaces. Full actual-model/native/device admission, positive training,
checkpoint continuity, automatic export/evaluation and overall campaign
completion remain outside this CPU QA result.
