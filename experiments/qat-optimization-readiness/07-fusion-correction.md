# Optional raw-input fusion correction

2026-10-01. Preparation implementation only. This worker used an isolated
`feature/qat-fusion-correction` worktree and touched only the new correction
module, fitting CLI, CPU tests, and this report. No model/capture loads, remote
access, GPU/Metal work, real-data fitting, or sealed-final reads occurred.
Q4_0 remains the primary acceptance, latency, and throughput comparison.

## Deliverable and arithmetic

`FusionCorrectionConfig` defaults to `enabled=False`; disabled installation
registers an empty `fc.fusion_correction` module, with no parameters or forward
hook, and returns the original binary output object. Enabled ranks are exactly
one or four. Masters are F32 `u[out,rank]`, `v[rank,in]` and optional
`output_bias[out]`. V is deterministic nonzero using a local CPU generator;
U and bias initialize to zero. This preserves the binary output at installation
and gives U a useful first-step gradient. With zero U, V and raw-input gradients
through the correction itself must be zero; they become useful after U changes.
The base's own raw-input surrogate gradient remains attached.

The enabled correction is a **binary-core hybrid**, with this arithmetic:

1. Preserve the original pre-activation-quantization FC input and cast to F32.
2. Round V master to F16, convert to F32 for the simulated F32 dot; its backward
   is identity to the master. Produce F32 latent `V raw`.
3. Round U master to F16 with the same surrogate and produce F32 `U latent`.
4. Add optional F32 bias clamped to the configured finite positive bound.
5. Add the delta to the F32 base result. There is no F16 intermediate/output
   cast, activation quantization, or extra scale on the residual.

Autocast is explicitly disabled for both dots. Factor overflow is rejected;
master dtype must stay F32. This maps to ggml F16 weight × F32 activation
producing F32 outputs. It does not claim bit-exact backend reductions or a
measured accelerator speedup.

At deployed FC dimensions `[out,in]=[2560,7680]`, rank one/four factors cost
20/80 KiB of F16 deployment payload and 10,240/40,960 MACs per row. Optional
F32 output bias adds 10 KiB. Two reductions, input reads, temporary state and
launch cost still require complete-operator native measurements at small N.

## Integration API

```python
correction = install_fusion_correction(
    linears["fc"], target=target,
    config=FusionCorrectionConfig(enabled=True, rank=4),
)
```

The attachment helper accepts only `RowBinaryLinear`; root integration owns
selecting drafter FC and excluding the vocabulary head. It rejects target module
and parameter/buffer storage aliases before mutation, and rejects reinstallation.
Its global forward hook reads `module.fusion_correction`, so deepcopied model
lanes use their own registered parameters. Arbitrary row-local leading dimensions
are supported, including batched context chunks. No state/K/V detach occurs.

`correction_parameter_group(correction, target=..., lr=...)` creates a zero-decay
parameter group or returns `None` for disabled correction.
`validate_correction_optimizer(optimizer, corrections, base_parameters=..., target=...)`
checks exact ownership once and excludes target storage. Root owns integration
with the existing binary and learned-quantizer groups, the continuous freeze
allowlist, curriculum/config identity, and training checkpoint version.

`state_payload()` returns strict `identity`, exact F32 `state`, and per-master
`state_sha256`. `load_payload()` validates all keys, config/dimensions/arithmetic,
master dtypes/shapes/finite values and hashes before changing the module. Save
and export leave masters unchanged. Bias may be outside its forward bound in
an exact optimizer checkpoint; `effective_bias()` clamps it, and `project_()`
offers post-update projection. Caller restores the optimizer state separately.

`identity()` binds config (including seed and bound), FC dimensions, arithmetic,
path, version and hybrid classification. `manifest_payload()` includes the
identity, effective tensor shapes/dtypes/hashes, native descriptor and
`requires_native_validation` status; disabled status has no native descriptor.

`native_payload()` returns `(descriptor, tensor_dict)` with:

| Native name | Torch layout | Deployment dtype |
| --- | --- | --- |
| `fc.correction_u.weight` | `[out,rank]` | F16 |
| `fc.correction_v.weight` | `[rank,in]` | F16 |
| `fc.correction_bias` (optional) | `[out]` | F32 |

The exact descriptor fields are `version=1`, `rank`, `u_name`, `v_name`,
`bias_name` (null if absent), `bias_bound` (null if absent), and
`arithmetic="raw_f32_v_f16_dot_f32_u_f16_dot_f32_add_base_f32_bias_f32"`.
Root owns NPZ key mapping/schema version; native owner owns GGUF/export/load
validation and FC graph placement before raw-input information is discarded.

## Joined calibration fitting contract

`scripts/fit_fusion_correction.py` implements CPU ridge reduced-rank regression
on input covariance. It uses a dual solve over train rows to avoid an
input-width-cubed solve. With bias enabled, it centers input/residual separately
and fits a bounded intercept after hard factor rounding. Validation rows never
enter the fit. It evaluates actual hard F16 factors through F32 dots and reports
base/corrected MSE by train/validation and domain/depth/source quantizer.

Version-one manifests require exact fields:

- `projection="fc"`, `raw_input_stage="pre_activation_quantization"`,
  `source_data_split="train"`, `eligibility="training_allowed"`, boolean
  `synthetic`, and exact source base-weight/reference-weight/quantizer SHA256s.
- `operands` path, archive SHA256 and rows. The NPZ contains F32 matrices
  `raw_input`, `binary_output`, `reference_output`, plus Unicode
  `raw_join_ids`, `binary_join_ids`, `reference_join_ids`.
- Each row binds unique `row_id`, `prompt_id`, prompt-content SHA256, domain,
  nonnegative depth/position, train/validation split, source-quantizer SHA256
  and canonical little-endian F32 hashes of all three row operands.

All three join IDs must match row metadata, all row hashes and the archive hash
must match, coordinates cannot repeat, and prompt content cannot cross splits
even through different aliases. Each fit binds exactly one source quantizer;
mixed quantizers fail and need separate fits/validation. The reported validation
contract is `prompt_content_held_out_within_one_source_quantizer`.
Preparation-only and diagnostic eligibility do not pass this loader. An explicit
`--allow-real-data-fit` is required in addition to train eligibility for any
future real-data invocation; it was not used in this task. Declared eligibility
still needs upstream authoritative dataset/ancestry audit before actual fitting.

CLI writes only into a new output directory, storing exact correction masters
with ancestry in `fusion_correction.pt`, and a hash-bound `fit_report.json`.
Both synthetic regression tests and CLI smoke use generated arrays in temporary
directories. Their MSE recovery verifies fitting behavior only, not realistic
error structure, recurrence quality, native acceptance or throughput.

## Acceptance checks and remaining work

Passed on local CPU: 16 new correction/fitting tests and six existing recurrent
QAT checks (22 total); targeted Ruff checks pass. Coverage includes default-off
and enabled-zero equivalence, hard F16 factors/identity STE/F32 latent, raw-input
attribution against identical A1 representations, batched dimensions, deepcopy
lane independence, bounded bias, overflow/dtype guards, global RNG isolation,
storage alias protection, exact optimizer ownership, exact Adam resume, strict
state rejection before mutation, later-only loss reaching correction factors and
earlier state/K/V, rank1/rank4 synthetic fitting, validation nonleakage, bad joins,
bad hashes, prompt-content split leakage, source quantizer mismatch and CLI resume.

Root and native owner still need to integrate/test the continuous optimizer,
freeze allowlist, training/snapshot/manifest schema, and native export/load/graph
for this option. Native decision/raw-output gates and actual RTX5080 complete
operator timing remain required before native-ready status; no SM75 claim is
made. Actual train-data residual fitting and held-out acceptance are later work
requiring the user's research authorization. This implementation does not read
or promote the existing preparation captures to training eligibility.
