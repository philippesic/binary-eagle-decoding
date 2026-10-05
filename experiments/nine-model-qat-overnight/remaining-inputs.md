# Remaining five candidates and original controls

Read-only local audit at main `5c0a1b7`, October 5, 2026. Owner:
`/root/overnight_eagle_data`, worktree `overnight-remaining-inputs`.
Only this new report was written during the audit. No remote/2080 access,
GPU call, model loading, capture, fitting or completed-test repetition occurred.
The healthy EAGLE A8 lane remains with the sole operator; this report neither
rechecks its admission nor proposes interrupting it.

Current selections are fixed A8, direct A1, block `ffn15_fusion`, original
reference magnitudes and private tensors, probes off. Historical draft records
still say these choices are pending; the active overnight goal supersedes that
text. The first lane's 24-hour allocation is frozen. Other lanes still need
explicit immutable operational allocations/configs; another human coverage or
curriculum answer is not inherently required to prepare them.

## Candidate-specific inputs still missing

| Candidate | Reusable authoritative inputs | Missing before its QAT launch |
|---|---|---|
| EAGLE A1 | Original completed 10,000-prompt/3,899,930-row continuous native TRAIN corpus, ready `bdfa56f8...`; fixed original target/base/drafter snapshot; existing direct-A1 trainer/exporter and native W1A1 path | Authentic all-three-domain **A1-arithmetic** scale-only initializer and report; direct-A1 packet/config/budget/markers; complete calibrated all-nine initial GGUF/export audit; candidate-specific current SM120 kernel/model/backward/full-moment-memory/resume/portability/cleanup admission and QA |
| DSpark A8 | Original released BF16 base, extracted FC/gamma/exact norm scalar; FFN15+FC training/native serialization implementation; authentic nine-chain CPU development pilot | Serious balanced original TRAIN five-tap chains, complete full-vocabulary teachers for source-style probability-L1, current-host data admission and native goldens; production A8 fusion fit; complete calibrated initial FC16 native candidate; frozen DSpark objective/conditioning/budget/config; fresh actual SM120 admission/QA |
| DSpark A1 | Same original base/data contracts and direct-A1 software support | Same production chains/teachers may be shared with A8, but an independent fixed-A1 fusion fit, direct-A1 initialized actor/export/config and current A1 admission are required; an A8 initializer or A8 admission is not an A1 result |
| DFlash A8 | Original released BF16 base/references; block hard-CE implementation; FFN15+FC native implementation and CPU pilot | Serious balanced original TRAIN five-tap chains/full-vocabulary hard labels, current-host completed admission and native goldens; production A8 fusion fit; complete calibrated initial FC16 actor/export; frozen hard-CE/conditioning/budget/config and fresh SM120 admission/QA |
| DFlash A1 | Same original base/data; direct-A1 software support | May share eligible DFlash chains with A8; independent A1-arithmetic fusion fit, initial actor/export/config and A1 admission remain missing |

Every row remains PENDING. Historical four block fits and calibrated GGUFs
prove bounded initialization/serialization, not serious production coverage.
The pilot has nine original TRAIN chains: one per domain in each of `train`,
`calibration_fit`, `calibration_validation`; only three are QAT train chains.
It cannot be repeated overnight and represented as the requested campaign.
The six CPU goldens cannot grant fresh SM120 portability.

EAGLE A1's original provider is already admitted for its historical fixed A1
teacher contract; recapturing the corpus is unnecessary. The original local
EAGLE diagnostic remains ineligible. The new production initializer helper
supports `--activation-bits 1`; no production A1 artifact from it exists in
this audit's local records. The packet helper currently hardcodes `eagle_a8`,
A8 fit/model/audit joins, config names, budget entry and native INT8 markers
(`scripts/prepare_eagle_lane_packet.py`). Generalizing those metadata bindings
is the immediate bounded Mac source task; trainer math does not need changing.

## Exact local artifacts and source entry points

Paths below are relative to `/Users/pippo/github/binary-eagle-decoding`.
Original block GGUF existence/size was checked without opening model contexts
or rehashing large files; SHA pins come from existing producer reports.

- DSpark base:
  `models/nine-model-original-snapshots/dspark/3457dff1417cb84927f6098a5fcb7cee85c934b7/base-bf16.gguf`,
  SHA `3685ea7785729b4d3e9de3939264c1c87eb98a2bf7935a8c838eccf36be9fd13`,
  2,792,267,936 bytes.
- DFlash base:
  `models/nine-model-original-snapshots/dflash/02d530b7962ea1412beaf41a05c0b8e36d5f9b1d/base-bf16.gguf`,
  SHA `61b294dc798c507fdc4a87f397b4015a78b3cdd7edad1e940e683bba10fb5575`,
  2,636,679,584 bytes.
- Original extracted references:
  `results/nine-model-qat-preparation/block-fusion-references-20261004/{dspark,dflash}/{fc-weight.npy,fc-norm.npy,reference.json}`.
  Both FC shapes are `[2560,12800]`, gamma `[2560]`; norm epsilon F32 bits
  `897988541`. Exact pins are in `preparation-report.json` and
  `experiments/nine-model-qat-preparation/data-fusion-checkpoint.md`.
- Original copied TRAIN source:
  `results/nine-model-qat-preparation/development-pilot-source-20261004/{corpus-manifest.json,train-00000.jsonl,train-00000.index.jsonl}`.
  Only shard zero is copied locally. This is an authentic source for larger
  explicit selection planning; it is not five-tap teacher data.
- Pilot family manifests/admissions:
  `results/nine-model-qat-preparation/development-cpu-pilot-capture-20261004-03/{dspark,dflash}/{manifest.json,completed-admission.json}`.
- Four existing **development** fusion fits:
  `results/nine-model-qat-preparation/development-cpu-pilot-fusion-20261004/{dspark,dflash}-a{8,1}-scale-only/{fusion_candidate.npz,fit_report.json}`.
  Full calibrated CPU prototypes are
  `models/nine-model-calibrated-native-03-scale-only/{dspark,dflash}-a{8,1}/`.
  These remain preserved; production siblings must have separate locators.
- EAGLE original dense drafter exists locally at
  `models/hf/Qwen3-4B_eagle3/model.safetensors`; original pin is
  `58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e`.
  Production raw calibration operands/report are currently remote artifacts;
  they are not inferred to be copied here.

Actual APIs:

- `scripts/capture_nine_model_train_data.py:prepare_plan`: explicit
  `nine_model_train_capture_plan_v1`; balanced domain counts in all three
  roles; original shard/index/group/content joins, source/storage/time/resource
  bounds. Plan-only does not load a model. Execution needs the sole operator
  and separate available GPU slot, so it must wait while healthy A8 occupies it.
- `scripts/prepare_block_qat_data.py`: imports original native receipts into
  `native_block_train_v1` manifests. `src/w1a1_eagle/block_data.py:BlockDataset`
  emits source/manifest/exact-host-file bound completed admissions and reads
  seven-slot blocks chronologically. Copied stat-bound admissions need genuine
  validation on the destination host, not edited path/stat fields.
- `scripts/fit_block_fusion.py`: separate A8/A1 sparse FC fits from
  `calibration_fit` versus `calibration_validation`, requires original
  model-bound FC/gamma/scalar descriptor; use rescue off, coordinate flips zero,
  reference magnitudes. Never refit the preserved pilot artifacts in place.
- `scripts/train_nine_model_qat.py:run_block` with `--prepare-only` and
  `scripts/export_block_binary.py`: create the actual initialized complete
  sixteen-projection checkpoint/GGUF and protected-tensor audit before native
  admission. `check_block_binary_native.py` and
  `check_block_capture_portability.py` supply distinct actual runtime gates.
- `scripts/prepare_nine_model_bundle.py:materialize_configs` and
  `scripts/prepare_nine_model_lane.py`: already admit any single known candidate
  with explicit `nine_model_lane_inputs_v1`. A block-specific packet helper is
  absent; its artifact/source/config/golden bindings can be prepared on Mac
  without altering the running EAGLE source/config.

## DSpark full-L1 is a data/prefix prerequisite

`BlockQATConfig` defaults to `hard_ce`; the supported DSpark source-style
objective is explicitly `full_probability_l1`. The builder copies
`qat_overrides` and does not infer this objective. A DSpark packet must set the
intended objective explicitly. Silently inheriting the hard-CE default would
change the requested source-style lane rather than solve its teacher dependency.

`src/w1a1_eagle/block_qat.py:block_loss` requires detached finite logits for
all **151,936** vocabulary entries, joined to the actual supervised conditioning
prefix. Neither EAGLE's pruned labels/top-k mass nor five-tap features alone can
satisfy this. Two existing source paths are explicit:

1. Offline `exact_soft` whole-chain native target captures plus
   `conditioning=captured_prefix` author-training semantics; static
   `native_greedy` replay additionally masks everything after its first
   predecessor divergence. Do not call stale post-divergence logits current.
2. `conditioning=native_greedy` with the `teacher` settings consumed by
   `train_nine_model_qat.native_teacher`: a separate frozen native target
   returns final-prefix full logits for each actual student predecessor.
   This requires fresh joint target/trainer resource admission. No co-residency
   fit or safe CPU-teacher alternative is established by the pilot.

Choosing an existing exact teacher path with truthful prefix semantics is an
implementation/operational preparation task. Replacing full-L1 with CE,
changing the target precision, dropping private heads/Markov, or approximating
the distribution is a research change requiring a separate decision; none is
necessary to continue implementing the supported exact path.

Storage planning is decisive. One F32 full-vocabulary row is 607,744 bytes;
five feature taps add 51,200 bytes, totaling 658,944 bytes per retained row
before token/metadata overhead. Seven-logit block reads are 4,254,208 bytes.
A 3.9-million-row exact-soft corpus would cost roughly 2.57 TB with features,
not fit the reported ~199 GB free disk. Serious block coverage must therefore
have a bounded source-selected count/length/storage plan or an admitted live
teacher route. This is a measurable preparation input, not a reason to reuse
only the nine development chains. DFlash hard-CE captures may omit per-row full
logits while retaining full-logit portability goldens.

## Original frozen controls

| Control | Original location/pin | Current availability and need |
|---|---|---|
| EAGLE Q4_0 | `models/gguf/Qwen3-4B-eagle3-q4_0.gguf`, SHA `2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280`, 128,988,160 bytes | Present locally; preserve original identity and eventual same-device comparison |
| DSpark FFN-only Q4_0 | Historical project-relative `runs/precision-q4-20261004/dspark-ffn-q4.gguf`, SHA `a95358608ce2e1c77f9a8253b692e697e3745dd68ba66b859833d41c480193fe`, 2,255,282,336 bytes | Historical SM75 operator recorded it; absent from local packet. Original bytes and original export/check receipts must be obtained through authorized access later |
| DFlash FFN-only Q4_0 | Historical project-relative `runs/precision-q4-20261004/dflash-ffn-q4.gguf`, SHA `6f23297bb623217e3df6352c8e83521f59f3b307b26f27f1262269775552b7c3`, 2,099,693,984 bytes | Same original-artifact dependency; no new remote availability claim |

Source-bound Q4 check-report SHAs are respectively
`766a810b2f0bd0d5ac9a44bd775804e85708d1eb0be9bbdfbb1b03d51c699130`
and `880d696977fdea0f111615d5dd53fffc8c7488dae4cce35e572815688aa4bd44`.
Authority: `experiments/dspark-sm75-20261003/operator.md` (Q4 staging/check
section), `results.md` and `precision-admission.md`; these are historical
records, not an instruction to access2080. The new local BF16 base serializations
have different file hashes from that old screen's BF16 references. Requantizing
them does not recover the immutable old Q4 file by assertion. The controls
quantize fifteen FFNs only; selected binary candidates also calibrate FC.
This is a deployment comparison with different coverage, not isolated precision
attribution. Preserve those labels and the primary Q4_0 EAGLE success baseline.

Absent block controls do not bar an otherwise valid staged training lane.
They remain required for the eventual original-control nine-model comparison.

## Safe next assignments while A8 trains continuously

1. Generalize direct-A1 EAGLE packet metadata and focused refusal tests on Mac:
   candidate/config/budget/precision/native-marker/export joins; default A8
   behavior retained; A8 initializer/model/audit supplied as A1 must refuse.
2. Prepare a source-selected serious block capture/cost draft from the original
   local TRAIN shard/index. Show balanced train/fit/validation counts, lengths,
   full-L1 versus hard-CE storage, exact source pins and all unresolved CUDA
   runtime locators; do not execute captures or invent rates/receipt status.
3. Prepare a block packet adapter for single-lane source configs with explicit
   DSpark full-L1/conditioning/teacher provenance, actual sixteen-projection
   initialization/export joins and PENDING artifact slots. Freeze no green
   admission until real inputs exist.
4. When the authenticated EAGLE calibration operand package is genuinely copied
   to Mac by the sole authorized operator, fit A1 once on its unchanged split,
   bind source/row/hash ancestry and reference-half policy, then stage its packet.
   This uses CPU and requires no interruption of healthy A8.
5. Prepare exact immutable original-Q4 transfer/reuse inventory from the known
   historical hashes/receipts. Actual remote retrieval and current device checks
   remain with the coordinator/operator after authorized host access and a safe
   slot; do not access2080 or regenerate controls in this local task.

Later sole-operator work: bounded serious block capture, separate production
fusion fits, actual initialized candidates/native goldens, candidate-specific
SM120 admission, then serialized long QAT. Healthy A8 is never stopped merely
to make those GPU preparations possible sooner. Each later lane needs its own
frozen source/budget/config and admission; the current live lane stays immutable.
The all-six legacy builder still requires `human_selected=true` for its full
campaign budget, whereas staged lanes accept truthful delegated provenance.
Final full-campaign packaging needs the same truthful authorization treatment or
an actual directly selected full budget; this is source integration, not grounds
for labeling delegated numbers human-selected.
