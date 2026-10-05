# Focused checkpoint review

October 4, 2026. Requested exact model `claude-opus-5-5-high`, peer-review
`ask_claude`, mode `ask`. Only the current generic optimizer validator was
provided inline (about70lines); no repository crawl or hardware action.
The earlier three-file training review timed out at180seconds and yielded no
review. This narrowed call completed.

## Source qualification and action

The reviewer correctly identifies that the generic validator does not itself
bind complete optimizer state to stage updates or constrain scalar step dtype.
Root's prompt omitted the caller: integrated8a16 training already applies a
separate `validate_optimizer_resume(...expected_updates=cursor.stage_updates)`
load guard. Therefore the review is not proof that the whole loader accepts
the stated counterexamples. The concrete remaining gaps are save/publication
validation, scalar rank/dtype, and consolidating save/load into one exact
stage-bound validator. Training owner is implementing focused corruption,
reset and next-update tests. Existing historical artifacts are preserved.

## Reviewer output

+**Yes, there is a blocking defect.** The validator never ties the saved `step` counters to `expected_updates`, and it never requires every owned parameter to have state. So a checkpoint whose state is partial, stale, or carried over from another stage passes validation and does not resume exactly.

**Counterexample**

- **Partial state:** `expected_updates = 500`. Saved state has entries for parameters `0..N-2`, each with `step = 500`. Parameter `N-1` has no entry, or has `step = 499`. The validator passes: the missing entry becomes `{}` and is skipped, and 499 is a valid non-negative integer. On the next `step()`, AdamW lazily creates fresh zero moments and `step = 0` for parameter `N-1`. That parameter's bias correction and update are now different, so the trajectory diverges.
- **Stage boundary:** At the A8→A1 transition, `expected_updates = 0` and the moments should be fresh. A checkpoint still holding A8 moments with `step = 3000` passes and loads stale moments into A1.

**Fix**

Pass in `expected_updates` and enforce both ownership and the counter value:

```python
state = saved["state"]
if expected_updates == 0:
    if state:
        raise ValueError("fresh stage must have empty optimizer state")
else:
    trainable = {i for i, p in zip(ids, parameters) if p.requires_grad}
    if set(state) != trainable:
        raise ValueError("optimizer state ownership incomplete/extra")
for key, parameter in zip(ids, parameters):
    values = state.get(key)
    if values is None:
        continue
    step = values["step"]
    if (step.shape != () or step.dtype != torch.float32
            or step.device.type != "cpu" or float(step) != expected_updates):
        raise ValueError("optimizer step mismatch")
```

The device check assumes the default AdamW path (not `capturable` or `fused`), where `load_state_dict` keeps `step` on CPU.

**Smaller issues (not blocking for this config)**

- `numel() == 1` accepts a step of shape `(1,)`, or an int64/float64 step. Arithmetic still works, but the saved state is no longer byte-identical to native AdamW state. The `shape == ()` and `float32` checks above close this.
- Moment device isn't checked, but `load_state_dict` moves moments to the parameter's device anyway, so this doesn't block.
- If `amsgrad=True`, the required key set should include `max_exp_avg_sq`. With your stated config it doesn't apply.
- The `zip(ids, parameters)` pairing is safe, because comparing the group dicts already forces `params` to match the reference index lists exactly.
