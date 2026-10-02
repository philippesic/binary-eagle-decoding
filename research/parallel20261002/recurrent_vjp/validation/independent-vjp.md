# Independent recurrent/cache VJP validation

## Scope and sources

This CPU-only check routes a deterministic hidden-size-4 EAGLE-shaped drafter
through the production `NativeStepAdapter`, `rebuild_prefix_cache`, and
`rollout_captured_prefix` APIs. The fixture independently constructs the tiny
drafter; it does not call the owner's reference helper. Accepted context is
reconstructed at prefix length one. The scalar loss uses only the valid row at
draft depth one. The final invalid-row case replaces that later row with a
terminal invalid proposal.

Pinned source SHA-256 values in this checkout:

| File | SHA-256 |
|---|---|
| `src/w1a1_eagle/native_step.py` | `844e52095ee0518cf507c81a10a895cdab7a61e32a1cb75836f2edb70e91d1c9` |
| `src/w1a1_eagle/recurrent_rollout.py` | `a286b198cb921b66cbf9ff86126be929b9ca452baa7a205c11f5caed093afc2b` |
| `src/w1a1_eagle/recurrent_binary.py` | `7624fa4486bcb130440cbc6b84c2463af01f0750e0145236063a125ab8b3de06` |
| `tests/test_native_step.py` | `db0f95b0523b888ea9aa5745e77ddcb8eaa9c0ac18dacffe8bab3848f5c123e2` |
| `tests/test_recurrent_rollout.py` | `f21cceabb38e1439b5952c03674626ecbce1133f4269a0e03e1514c07dd0d93e` |
| `tests/test_parallel20261002_recurrent_vjp_validation.py` | `5ee718c79df85559ecbb7eedf424ac8ec54e476eced72baabdfb9370c91f4d4a` |

## Environment and exact run

- Worktree: `/private/tmp/eagle-parallel-20261002/recurrent-vjp`
- Raw-output run directory: `/private/tmp/eagle-parallel-20261002/recurrent-vjp/runs/parallel20261002/recurrent_vjp_validation/`
- Branch: `research/20261002-recurrent-vjp`
- Host: macOS 27.0, arm64; Apple ARM CPU; shared project Python 3.11.15;
  PyTorch 2.14.0.
- Test process constrained to one OpenMP and one MKL thread. No GPU, Metal,
  SSH, model weights, or data were used.
- Passing command:

  ```sh
  set -o pipefail; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest discover -s tests -p 'test_parallel20261002_recurrent_vjp_validation.py' -v 2>&1 | tee runs/parallel20261002/recurrent_vjp_validation/unittest.log
  ```

- Result: **3/3 tests passed** in 0.260 seconds (unittest reported `OK`). The
  raw output is `unittest.log` in the run directory above.
- Formatting and lint commands (both passed):

  ```sh
  /Users/pippo/github/binary-eagle-decoding/.venv/bin/ruff format tests/test_parallel20261002_recurrent_vjp_validation.py
  /Users/pippo/github/binary-eagle-decoding/.venv/bin/ruff check --fix tests/test_parallel20261002_recurrent_vjp_validation.py
  /Users/pippo/github/binary-eagle-decoding/.venv/bin/ruff check tests/test_parallel20261002_recurrent_vjp_validation.py
  /Users/pippo/github/binary-eagle-decoding/.venv/bin/ruff format --check tests/test_parallel20261002_recurrent_vjp_validation.py
  ```
- Exact metric extraction command (run from the worktree root):

  ```sh
  set -o pipefail; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -c 'import sys,torch; sys.path.insert(0,"tests"); from test_parallel20261002_recurrent_vjp_validation import run_case; b=run_case(); s=run_case(detach_state=True); c=run_case(detach_cache=True); i=run_case(invalid=True); print("baseline_scalar",float(b["scalar"])); print("early_vjp_max",{k:float(v.abs().max()) for k,v in b["obs"].items()}); print("detach_state_max",{k:float(v.abs().max()) for k,v in s["obs"].items()}); print("detach_cache_max",{k:float(v.abs().max()) for k,v in c["obs"].items()}); print("detach_cache_delta",{k:float((b["obs"][k]-c["obs"][k]).abs().max()) for k in ("key","value")}); print("raw_feature_grad_max",b["raw_feature_grad"].abs().max(dim=1).values.tolist()); print("invalid_calls",i["calls"],"invalid_scalar",float(i["scalar"])); print("invalid_grad_max",max(float(g.abs().max()) for g in i["trainable_grads"].values() if g is not None))' 2>&1 | tee runs/parallel20261002/recurrent_vjp_validation/metrics.log
  ```
- An earlier environment probe using `uv run ...` failed because its isolated
  `.venv` lacked PyTorch; that generated environment was removed. The final
  run used the shared project environment requested by the owner.

## Measurements and tolerances

The deferred feature from accepted prefix `[0, 2, 3]` is now the actual rollout
input (seed token 3). Its later-only scalar loss was `0.4031370282173157`. All early VJPs were
nonzero, with these maximum absolute entries:

| Early tensor | Baseline max abs VJP |
|---|---:|
| depth-0 pre-norm state | 4.37476921081543 |
| depth-0 accumulated key cache | 0.0018255707109346986 |
| depth-0 accumulated value cache | 0.7391270399093628 |

Detach controls behave distinctly:

| Control | Early state max abs VJP | Early key max abs VJP | Early value max abs VJP | Max abs delta vs baseline (key/value) |
|---|---:|---:|---:|---:|
| detach returned pre-norm state | 0 | 0.0020796535536646843 | 0.32327306270599365 | — |
| detach returned cache | 4.37476921081543 | 0.0007190489559434354 | 0.41838541626930237 | key 0.0020796535536646843; value 0.32327306270599365 |

The validator uses `1e-8` as the minimum meaningful nonzero/delta threshold
for these F32 VJPs. State detach zeros the early state VJP. Cache detach removes
the later-attention contribution to the early cache, while retaining the first
step's own attention contribution; consequently, its early-cache VJP is
expected to change rather than vanish. This is the exact topology exposed by
the adapter's one-step decode. Both detach controls preserve hard logits
exactly (zero max forward delta, checked with `atol=0, rtol=0`).

For the terminal invalid row, rollout called the decoder only at position 1,
returned exact-zero logits for the invalid row, and gave an exact-zero scalar
and maximum trainable-parameter gradient of 0.0. Accepted-context encoder and
decoder callbacks ran with `torch.is_grad_enabled() == False`; rebuilt context
K/V tensors had `requires_grad == False`. This confirms the declared prefix
truncation boundary while preserving draft-state/cache gradients.
The accepted context raw row's maximum absolute VJP was exactly 0.0; the
deferred seed raw row's maximum absolute VJP was 0.748046875. The report's
`1e-8` threshold is used to identify nonzero/different F32 gradient paths; the
invalid-row and accepted-context checks require exact zero.

## Cleanup

The test and metric processes exited successfully; no test process remains.
The temporary `.venv/` created by the failed `uv` probe was removed. Raw stdout
is retained in the ignored run directory above; Python import caches are
ignored. The only intentional source changes from this validator are this
report and `tests/test_parallel20261002_recurrent_vjp_validation.py`.
