# Pinned AngelSlim EAGLE-3 acceptance counters

**Status:** source audit only; no model run yet.  
**Source:** Tencent/AngelSlim commit
`0358da9c651e6a7d7ccafea26ced4b9c98d11681`,
`angelslim/compressor/speculative/utils/util.py` and
`angelslim/compressor/speculative/inference/models/eagle3/draft/base_model.py`.

For a greedy `eagle_generate(log=True)` run, the pinned implementation returns
`accept_length_list`, one value per verification round. Its
`evaluate_posterior` compares `candidates[:, 1:]` with the target's greedy
choices from `logits[:, :-1]`. Thus each value is the contiguous count of
accepted **draft** nodes after the leading target-selected seed. The seed and
the target's next bonus/correction token are not accepted draft nodes.

The drafter stores `total_tokens = total_token - 1` and builds each tree as one
seed plus exactly `total_tokens` selected draft nodes. With the initial
`total_token = 60` configuration, each verified tree proposes 59 draft nodes.
For this tree protocol, record `accepted = sum(accept_length_list)` and
`proposed_tree_nodes = 59 * len(accept_length_list)`, along with the complete
per-round accept lengths and the tree parameters. Mean accepted draft nodes per
round is also reported; it is often easier to interpret than the accepted/node
ratio for a branching tree. Confirm these source-derived counts in an actual
run before treating them as measured results.

The runtime's `new_token` counter advances by `accept_length + 1` because it
includes the leading seed. It is not the accepted-draft numerator. The selected
candidate prefix length is also `accept_length + 1`; the separately sampled
next target token feeds the subsequent draft tree. A deterministic test should
exercise zero, partial, and full acceptance and verify the selected path and
cache rollback before a held-out sweep.

The runtime has configurable tree budget, depth, and branching. A depth-five
chain with `top_k = 1` has not yet been validated against this pinned tree
builder, so this goal starts with the official branching protocol and records
its parameters explicitly. The later native runtime comparison remains a
separate experiment.

- [Pinned posterior and update code](https://github.com/Tencent/AngelSlim/blob/0358da9c651e6a7d7ccafea26ced4b9c98d11681/angelslim/compressor/speculative/utils/util.py)
- [Pinned tree construction](https://github.com/Tencent/AngelSlim/blob/0358da9c651e6a7d7ccafea26ced4b9c98d11681/angelslim/compressor/speculative/inference/models/eagle3/draft/base_model.py)
- [Pinned generation loop](https://github.com/Tencent/AngelSlim/blob/0358da9c651e6a7d7ccafea26ced4b9c98d11681/angelslim/compressor/speculative/inference/models/eagle3/eagle3_model.py)
