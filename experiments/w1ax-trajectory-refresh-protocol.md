# Bounded student trajectory refresh

This implements backlog item 8 from `docs/W1_RESEARCH_PLAN.md`. The CPU planner
creates an immutable capture queue and can re-audit it from the original files.
It does not load models, start inference, authorize a training budget or change
the eligibility of the existing pilot. The current approved pilot remains the
100-step row-A16 calibration. Refresh capture and substantive training remain
separate user-owned decisions.

## Strategy and acceptance

Freeze a small, three-domain **training** sample and its token-prefix limits
before collecting a selected checkpoint's student trajectories. For every
student row, bind the checkpoint bytes and exported native model bytes. Record
the complete absolute target-token prefix whose next label is requested and
the accepted context root whose teacher features rebuild the recurrent cache.
A feature root must be an ancestor of that label prefix.

Teacher reuse keys are `(prompt_id, kind, complete prefix_token_ids)`. Labels
are native verifier labels on that exact prefix. Feature reuse requires every
prefix of the accepted context root, plus the same target, tokenizer, absolute
d2t map, native revision and native execution policy. This additional execution
scope preserves the known native same-prefix feature variation. The policy
should include backend/precision, feature taps/boundary, batch and cache/mask
settings; changing it requires a new index. Student states and K/V are always
rebuilt from the current checkpoint rather than reused from candidate D.

Changed or missing label prefixes enter `capture_requests`; missing feature
prefixes enter independently. The planner deduplicates requirements within one
sample, retains the student row consumers and bounds new labels, features,
student rows, prefix length and refresh rounds. Overflow never truncates the
request set or yields a ready queue. Even a fully populated queue remains
`training_eligible: false` until a downstream provider re-audits it.

Acceptance checks use a frozen CPU fixture containing prose/code/reasoning,
exact feature roots and diverged D/student prefixes. They check changed-prefix
capture, root-chain coverage, no silent cap truncation, final exclusion,
checkpoint and execution-scope mismatch, ambiguous teacher rejection, curve
hold/stop/pending, source and plan tampering, immutable CLI output and the v1
index adapter's label-ledger references. They establish metadata contracts only.

## Commands

All inputs/outputs below are ignored local data/run artifacts. File records use
absolute paths and actual SHA256 values. No placeholder hash is accepted.

```sh
# Optional CPU adapter for existing raw v1 captures. It runs the existing audit,
# verifies the copied native source cell and preserves the existing raw artifacts.
python3 scripts/plan_w1ax_trajectory_refresh.py index-v1 \
  --capture-manifest "$CAPTURE/manifest.json" \
  --prompts "$TRAIN_PROMPTS" --prompts-sha256 "$TRAIN_SHA256" \
  --prompt-count "$TRAIN_COUNT" --native-contract "$RUN/native-contract.json" \
  --output "$RUN/teacher-index"

# Prepare the schedule after freezing policy and collecting a bounded student
# trace. No teacher/model work occurs in this command.
python3 scripts/plan_w1ax_trajectory_refresh.py plan \
  --request "$RUN/refresh-request.json" --output "$RUN/refresh-plan.json"

# Independently rerun all hash, prefix and gate checks. Output must be new.
python3 scripts/plan_w1ax_trajectory_refresh.py audit \
  --plan "$RUN/refresh-plan.json" --output "$RUN/refresh-plan-audit.json"

PYTHONPATH=src python3 -m unittest discover -s tests -p test_trajectory_refresh.py -v
ruff check src/w1a1_eagle/trajectory_refresh.py \
  scripts/plan_w1ax_trajectory_refresh.py tests/test_trajectory_refresh.py
```

The request schema is `w1ax_refresh_request_v1`, with `refresh_round` counting
completed prior refreshes, `student` containing `checkpoint_sha256` and
`export_sha256`, `native_teacher_contract` as below, and `inputs`:

| Input | Contents |
| --- | --- |
| `policy` | Frozen policy JSON below |
| `train_prompts` | Actual frozen train JSONL with unique `id`; sample IDs must be a subset |
| `checkpoint` | Selected checkpoint bytes; hashed without loading weights |
| `student_export` | Selected native GGUF bytes; hashed without loading weights |
| `student_rows` | Bounded trace JSONL below |
| `teacher_manifest` | Normalized teacher-index manifest below |
| `learning_curve` | Optional frozen development evidence below; absence yields pending |

Every `inputs` entry is `{ "path": "/absolute/file", "sha256": "actual lowercase SHA256" }`.
`file_record(Path(...))` from `w1a1_eagle.trajectory_refresh` produces these records.
Use the same native teacher contract in the request and teacher manifest:
`target_gguf_sha256`, `tokenizer_sha256`, `absolute_d2t_sha256`, `native_revision`,
`execution_policy_sha256`, `target_vocab_size`. Native revision is a full lowercase
40-character commit SHA. The v1 adapter binds target identity
to its source cell. Other contract fields are operator-declared, hashed scopes;
the planner does not certify the executed runtime or numerical readiness.

## Frozen policy and trace contracts

Policy `w1ax_refresh_policy_v1` declares `split: "train"`, the actual
`train_prompts_sha256`, `development_sample_sha256`, and `train_sample` entries
with `prompt_id`, `split: "train"`, and `domain: "prose"|"code"|"reasoning"`.
It must include all three domains. Never derive IDs/content by opening final data.
The development hash identifies a predeclared independent development sample;
use matching evidence for every comparison, and keep it disjoint from training.

`caps` has positive integer `max_rounds`, `max_student_rows`,
`max_new_label_rows`, `max_new_feature_rows`, `max_prefix_tokens`.
`learning_curve` has positive `min_completed_steps` and fractional
`min_relative_ce_improvement`, `max_acceptance_regression`,
`min_changed_prefix_fraction`. These are predeclared policy settings, not
implicit approval for a capture size or training run. The executable fixture
uses two refresh rounds and three new labels/features; those are tests only.

Student JSONL rows carry `id`, `prompt_id`, `split: "train"`,
`checkpoint_sha256`, `export_sha256`, `prefix_token_ids` (the complete label
prefix) and `feature_prefix_token_ids` (the complete accepted feature root).
Rows must cover exactly the frozen training sample. IDs cannot be duplicated;
repeated same-prefix requirements across different student rows are deduplicated.
Native collector output must be mapped to these fields by audited task ownership,
not by row order or reconstructed text. Token IDs avoid tokenizer round trips.
This worker does not modify the existing native collector.

## Teacher storage seam

A normalized teacher manifest uses `w1ax_refresh_teacher_index_v1`,
`split: "train"`, `train_prompts_sha256`, `native_teacher_contract`, `index`
(a file record), and nonempty `artifacts` (named file records). Each JSONL index
entry carries `kind: "label"|"feature"`, `prompt_id`, `split: "train"`, the
complete `prefix_token_ids`, `capture_id`, `artifact` (a name in `artifacts`),
and nonnegative `row`. Labels also carry the absolute `next_target_id`.
For v1 labels, `row` refers to the native `rows.jsonl` ledger position, which
binds `target_logits_row` and the audited argmax label. Feature rows refer to
the hashed feature array row. Identical repeated v1 labels are coalesced with
`equivalent_source_rows`; differing labels on the same prefix stop import.

The compact-v2 worker can emit this same normalized index after its versioned
storage audit, pointing labels to audited label/top-k payloads and features to
audited arrays. Existing v1 raw bytes and reports remain intact. A file hash
and normalized index do not replace the storage auditor: the training provider
must check payload row bounds, exact label/features and runtime readiness.

## Learning curve and stop rules

Evidence `w1ax_refresh_learning_curve_v1` uses `split: "development"`,
`development_sample_sha256`, selected `checkpoint_sha256` and `previous` /
`current` objects. Each object has `checkpoint_sha256`, `completed_steps`,
`hard_ce` and `native_acceptance` (accepted/proposed draft tokens in [0,1]).
Use native rollouts under the same settings and report per-domain acceptance
alongside these aggregate fields. The current object must bind the selected
checkpoint and steps must advance. The planner uses only measured records.

The queue passes only after the minimum completed steps, sufficient relative
CE improvement, no excessive native acceptance regression, sufficient missing
student-label prefix fraction, a remaining refresh round and row caps. A
missing curve yields **pending**; insufficient progress/divergence yields
**hold**; nonfinite/invalid loss or acceptance, native acceptance regression,
or exhausted rounds yields **stop**. A pass means a bounded queue is prepared
for review/scheduling; it does not launch it. Thresholds and aggregate evidence
cannot establish substantive training success or permit opening final data.

After an authorized target capture, re-run its native/storage audit, emit a new
normalized teacher index with the same execution scope, update the request to
the new manifest hash and prepare/re-audit a new plan. Missing ancestry must
fall to zero before handing it to a provider eligibility gate. Do not mutate
a saved plan or promote an old provider with stale D-prefix labels. Capture
requires the sole GPU owner's supervised job and an approved explicit budget.

## Worker checkpoint

Deliverable: executable CPU refresh scheduler, v1 index adapter, independent
re-audit command, meaningful frozen-sample tests and this protocol. No target
inference, GPU measurement, new capture/training approval or final-set opening
occurred. Native collection of selected trained-student trajectories, changed
prefix teacher capture, v2 normalized-index emission and provider eligibility
remain downstream integration/measurement work. A 100-step pilot alone does
not satisfy a substantive learning-curve gate.
