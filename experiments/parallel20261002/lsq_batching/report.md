# Learned-head batching gradient audit

**Confirmed execution-shape defect relative to the declared reference control.** With the same synthetic tensors and scalar mean CE, `optimize_head=True` reduces learned A1 threshold/A4/A8 clip VJPs by `1/sqrt(valid_depth)`. Hard logits and loss were exact; input, binary sign, row-scale, and other activation VJPs matched the CPU gate. An actual `joint_optimizer` SGD update differs. No production source, live recipe, data, or GPU job was changed.

## Executable path and contract

`forward_torch_round` selects `adapter.decode_head` when head batching is enabled; `rollout_captured_prefix` stacks **valid** pre-norm states, leaving terminal invalid padding outside the head invocation. `NativeStepAdapter.decode_head` applies the frozen output RMS norm and one `lm_head` call. `RowBinaryLinear.forward` calls its attached learned quantizer without `normalization_count`.

The low-level documented contract counts valid feature elements **per invocation**, excludes Q/K/V consumer multiplicity, and permits one larger count across chunks for full-batch normalization. `forward_torch_round` calls optimized paths “reference controls” with unchanged optimizer/update cadence. Its serial baseline invokes the head once per valid depth, using N=hidden width H; batching uses N=depth*H. There is no currently declared chain-wide or optimizer-step-wide normalization domain. Therefore the defect is composition of those two contracts, not disagreement with the quantizer's handwritten surrogate.

Let R_d = sum_i(g_di * dQ_di/dc) for depth d, with the identical attached cotangent g including the fixed loss reduction. Serial VJP is sum_d R_d/sqrt(H*q); batched VJP is sum_d R_d/sqrt(D*H*q), where q=qmax for A4/A8 and q=1 for A1. For constant valid width H the ratio is sqrt(D) even for nonidentical rows, unless the summed derivative is zero. Chunks of k valid rows instead divide each chunk's numerator by sqrt(k*H*q); ragged last chunks need separate counts. Finite differences of hard quantizers are not used.

## Results

72 fixture combinations: controlled repeated attached states and a reduced actual native EAGLE-shaped decoder; A1/A4/A8; valid depths 1/2/4; with/without invalid terminal padding; reference/single-forward arithmetic. Seven execution variants per combination compare serial, batched, two-row chunks, serial fallback, and serial/batched/chunked declared chain-domain references. Native fixtures install all nine linears and all six learned boundaries; Q/K/V share the same quantizer/input, and current recurrent state/cache graphs remain attached.

The independent Luna validator derives dX and dc directly, verifies true tied Q/K/V consumers with the same input/mask and distinct cotangents, rejects a pooled 3N denominator, and checks masked, all-invalid, zero-row, and explicit full-N chunk controls.

CPU numeric gate: exact hard logits/loss; VJPs and updates atol=2e-6, rtol=2e-5; nonzero gradient ratios absolute tolerance 1e-5.

| Metric, maximum across 72 combinations | Serial versus batched |
| --- | ---: |
| Hard logits | 0 |
| Mean CE | 0 |
| Raw input VJP | 2.98023224e-08 |
| All binary-sign VJPs | 1.49011612e-08 |
| All row-scale VJPs | 1.1920929e-07 |
| Other learned-boundary VJPs | 7.4505806e-09 |
| Head parameter after one SGD step | 0.00274904072 |
| Other parameters after one SGD step | 1.86264515e-09 |

Actual NativeStepAdapter, single-forward, no padding:

| Precision | Valid depth | Serial head VJP | Batched head VJP | Ratio | SGD parameter difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 1 | 0.100633726 | 0.100633726 | 1 | 0 |
| A1 | 2 | 0.10453102 | 0.0739145875 | 1.4142137 | 0.00153082609 |
| A1 | 4 | 0.099937588 | 0.049968794 | 2 | 0.0024984479 |
| A4 | 1 | -0.034696836 | -0.034696836 | 1 | 0 |
| A4 | 2 | -0.0322064236 | -0.0227733813 | 1.4142135 | 0.000471651554 |
| A4 | 4 | -0.0347885191 | -0.0173942577 | 2.0000002 | 0.000869750977 |
| A8 | 1 | -0.00698791444 | -0.00698791444 | 1 | 0 |
| A8 | 2 | -0.00716173509 | -0.00506411213 | 1.4142134 | 0.000104904175 |
| A8 | 4 | -0.00753714982 | -0.00376857538 | 1.9999998 | 0.000188469887 |

SGD uses the actual `joint_optimizer` with binary sign/scale LR=0.01, activation LR=0.05, zero decay, default zero momentum, and unchanged synthetic initial state. One plain step isolates normalization; it does not model training-wide clipping or claim the first AdamW step must differ substantially. AdamW is nearly scale-invariant at its first step away from epsilon, while its moments, clipping, and accumulated varying-depth updates can still differ. With joint clipping, the changed activation-family norm can also change other-family updates despite identical raw VJPs; this experiment intentionally isolates the unclipped SGD transformation.

Additional ragged control chains have valid lengths [1,2,4], each with invalid padding, and one CE mean over all seven valid rows. Its head VJP equals the weighted per-chain expression. A mistaken global seven-row denominator is distinguished from the current per-chain batching result. Terminal padding does not affect normalization or the measured discrepancy.

## Smallest isolated remedy and owner decision

The focused [Astra review](../../../docs/parallel20261002/astra-lsq-review.md) identifies the named `learned-activations` and `combined-contract-smoke` profiles as exposing this permitted combination.

`reference/serial_learned_head.patch` proposes only a provider dispatch guard: while gradients are enabled and the attached head quantizer is trainable, preserve the serial head path. Fixed-activation training and no-grad learned inference retain batching. This preserves existing serial normalization, shared parameter ownership, masks, recurrent graphs, cache semantics and update cadence. Its diagnostics honestly retain last-valid-row saturation scope. It sacrifices the learned-head batching optimization; no speed claim is made. The exact patch is compiled into a private module for tests, without touching production files. Nine actual-native proposal cases check every VJP and one update. Separate call-count controls prove batched inference/fixed training remain enabled. The actual `ObservedAdapter` wrapper delegates `linears`, so the guard sees the learned head; its first attached state still receives a nonzero later-depth gradient. `optimize_head=True` metadata must explicitly describe this fallback and last-row saturation scope rather than imply batched learned-head execution.

A declared chain domain is another mathematically valid reference: forward N=D*H into every serial/batched/chunked call. Existing low-level API supports it; a research-only head-quantizer shim demonstrated parity. However, this changes the serial training recipe and requires an explicit owner choice, plumbing/identity/checkpoint treatment, and current readiness evidence. It must not be silently introduced as an optimization. N is not the global optimizer-step count; QKV multiplicity stays excluded.

Next action: QAT owner reviews/adopts the serial guard or explicitly selects a new chain-domain recipe. For the conservative guard, migrate the patch and its proposal-preservation tests into the live owner worktree, preserving current preparation/runtime hashes and rebuilding source-bound readiness evidence as required. The version-pinned audit's discrepancy assertions document the old source; retarget them to an immutable source fixture or archive them before applying a live fix. This team has no authority to mutate the live recipe or preparation source.

## Checks

55/55 focused tests passed under the project Torch 2.14.0 environment: existing learned-activation, recurrent-provider, and recipe-integration suites plus the nine new audit/proposal tests (72 primary fixture combinations). Ruff check and `git diff --check` passed for owner files. The independent handwritten verifier passed both system Torch 2.8.0 and project Torch 2.14.0. Full test command uses the shared project executable `/Users/pippo/github/binary-eagle-decoding/.venv/bin/python`; no environment was installed or changed.

## Reproduction and provenance

Environment: macOS-27.0-arm64-arm-64bit; PyTorch 2.14.0; F32 projections/STE, F16 native cache/embedding; one CPU thread. Dense CPU simulations do not validate SM75 performance, native acceptance, real-data gradients, held-out quality or Q4_0-relative latency/throughput. No real model weights, captures, final set, GPU/Metal/SSH or credits were accessed.

From the worktree root, with the existing CPU Torch environment:

```sh
PYTHONPATH=src:. OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m unittest discover -s tests -p test_parallel20261002_lsq_batching.py -v
PYTHONPATH=src OMP_NUM_THREADS=1 python research/parallel20261002/lsq_batching/validation/hand_vjp.py
PYTHONPATH=src OMP_NUM_THREADS=1 python research/parallel20261002/lsq_batching/reference/audit.py --output /path/to/ignored/results.json
```

Owner full numeric output is preserved outside Git at `/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/lsq-batching/owner-results.json`. Committed tables are aggregate results. Independent validator commit `e61cf9e`; its report contains exact commands for both environments. Both ignored raw log directories were also copied into the main workspace `runs/parallel20261002/lsq-batching-vjp-validation{,-torch214}/` before any worker cleanup.

Imported production source SHA256 values:

| Source | SHA256 |
| --- | --- |
| `learned_activation.py` | `65b7b0863bb67806991ee4860ce327bf46bfbdf0e8d4e44226aa1ca268e880db` |
| `recurrent_qat.py` | `f2b6d83b8929807b65b6f21826cc99097a6ddd70f99ff23ca0836aebe4db1224` |
| `recurrent_rollout.py` | `a286b198cb921b66cbf9ff86126be929b9ca452baa7a205c11f5caed093afc2b` |
| `recurrent_provider.py` | `81cd3643b9a3060ae00d50d18cdf9ebc39846dbafc3b772b62651fb20d4b31a3` |
| `native_step.py` | `844e52095ee0518cf507c81a10a895cdab7a61e32a1cb75836f2edb70e91d1c9` |
| `qat_optimization.py` | `d2da18a4f4db612dd3cb51916af688109593d2c9d4cf119a21f8129567c305f7` |

All tests execute synthetic fixture gradients only. The absolute research control is checked before each owner fixture; source-bound proposal loading fails closed after a source hash change. Research stop/reset means checkpoint and no further experiments. No process remains running.
