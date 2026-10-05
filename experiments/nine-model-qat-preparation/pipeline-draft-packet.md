# Inspectable campaign draft packet

Status: **PENDING**, inspected on Mac at main
`566b954d9b7b3139f798a2d14fca782058023b8a`. The real source builder exited0,
reported102 pending entries, queried no GPU and performed zero optimizer
updates. This is an unselected draft, not a frozen production bundle.

All local packet files are outside Git at
`results/nine-model-qat-preparation/final-draft-packet/`. Run the exact source
inspection from the main checkout:

```sh
/Users/pippo/github/binary-eagle-decoding/.venv/bin/python scripts/prepare_nine_model_bundle.py --inputs /Users/pippo/github/binary-eagle-decoding/results/nine-model-qat-preparation/final-draft-packet/bundle-inputs.draft.json --inspect-draft
```

| Preserved packet file | SHA256 |
| --- | --- |
| `bundle-inputs.draft.json` | `85e7c4ce36398afe908e5aac9068e5923f4306a679a59b92db0d796d743a2015` |
| `inspection.stdout.json` | `a8e86b12f87d1a1cbd84f64a070c30a0f4b164dec3aad6717b9fab3f910c2b5a` |
| `packet-receipt.json` | `e580af47370d4aaa27df891683e6f2fb422f940a3dbda1b4e1fddce85d837c0a` |
| `qa-ledger.snapshot.json` | `3b960264099624169db7d047f9f51d8e53dd11f82ad4668c6bafdf3c364f7c5b` |

The receipt binds the exact command, source revision, assembler/finalizer and
stdout/stderr hashes. The independent ledger has51 current source pins; all
nine aggregate and prelaunch statuses remain PENDING. The earlier descriptor
and inspection are preserved separately under `*.interim.json`. Later source
or artifact changes require a distinct packet rather than overwriting these
published identities. Large target/block model pins reuse verified producer
reports; this inspection did not repeat tensor/corpus audits or load a model.

The genuine available references include the immutable F16 target, original
EAGLE Q4, original family BF16 bases, historical unselected EAGLE fixed-half
initializers, native CPU FFN15/FC16 prototypes, failed capture01/02 reports,
successful authentic CPU capture03, CPU goldens, four scale-only fusion fits,
four actual CPU model initializer checks and calibrated FC16 CPU exports.
Their complete locators, hashes and scope are in the descriptor.

Capture03 contains nine authentic TRAIN chains across prose/code/reasoning and
separate train/calibration-fit/calibration-validation roles. Its datasets and
six goldens are `DEVELOPMENT_CPU_ONLY`; CUDA portability remains PENDING.
The four Torch checks used one prose TRAIN block with hard-CE loss and the
actual released model/calibrated FC. Full teacher logits were materialized but
unused in that loss; three-domain backward, full-L1, QAT and quality are not
claimed. All owned groups were reaped before the native phase.

Four native FC16 compositions copied original untrained FFN initializers and
fitted FC members exactly, then passed protected-tensor serialization and CPU
operator graphs with synthetic activations. No native TRAIN trajectory or
CUDA/SM120/quality/performance proof is inferred.

| Unselected block initializer | FC calibration NPZ SHA256 | Calibrated CPU GGUF SHA256 |
| --- | --- | --- |
| DSpark A8 | `caa9d4337015271d69fbfc56a8ef71f9d7d6883b21058b9340b3a9c20ac79c32` | `f508b201a263ab2e3769f94a1eaf3696db9a1b7aaccbf73ea98307509df920f6` |
| DSpark A1 | `06652d0812173994e6d7b3cc830f7b047fb53e4e9c5e19f2818f2211da7c47e0` | `9a59e2a8f933f50631e1da61ab025fc0ab00333c57b3d2ba4773997f3fdf4c47` |
| DFlash A8 | `effb7421beada9addd90f9b73b6146126f7380d5b800b45ac7315fd603a40085` | `2f76dc6e6e2a031cdcbe3dd2350d8305bfc9205958ce964e20fb6c11126232aa` |
| DFlash A1 | `4c392657542350803afa7a41da47228fec084f829ecf0c2463d4b798efeb51fd` | `c6f87378de90ddfc86a011572680cf01a9223ddc35be5f85dcb592ea8a1c9973` |

Production freeze still requires human-selected recipe/coverage/budget and
resource limits, serious production TRAIN coverage/completed admissions,
original frozen DSpark/DFlash Q4 controls, selected source configs/deployment
profiles, appropriate calibration and production portability inputs. Fresh
SM120 kernel/model/backward/full-moment memory and owned-context release proof
is a separate runtime gate. CPU evidence does not supply SM75/SM120 proof.
Prelaunch gates source/data/initialization/initial export contracts; trained
checkpoints and final measured quality remain later campaign outcomes, avoiding
a circular requirement for the first QAT launch.

The current GPU pause remains active. When the actual inputs, human choices and
authorized device availability are supplied, the existing future launch command
runs in a detached Linux tmux session through the tmux MCP transport:

```sh
python3 scripts/remote_job.py RUN_ID --stop-grace-seconds 90 -- python3 scripts/run_nine_model_campaign.py --start --bundle BUNDLE_JSON --bundle-sha256 BUNDLE_SHA256 --availability LEASE_JSON --run-dir runs/RUN_ID/campaign --supervisor-state runs/RUN_ID/state.json
```

The source owner made no GPU/SSH/model/quality runs while building this packet.
Source fixes separating prelaunch/campaign gates and refusing synthetic scope
variants are integrated as32f514c/bd28e17; the usable draft/launch instructions
are integrated as077e9d8. Focused builder checks passed12/12; root's final joined
checks ran172 nine-model tests successfully with5 identified skips;78 block tests passed.
No additional suites were repeated for this artifact-only checkpoint. All
pipeline-owned work is committed and the worktree is clean; ignored packet
helpers/reports already reside in the primary checkout and need no transfer
before this worktree is retired.
