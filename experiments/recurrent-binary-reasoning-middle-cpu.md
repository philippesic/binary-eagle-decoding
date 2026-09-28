# Integrated CPU parity across a rejected-draft rewind

The sealed reasoning prompt's second round follows a first round with
zero accepted drafts. Native speculative and catch-up executions had
already written proposal token 1477 at memory position 47; the next
seed writes verifier token 525 there instead. The opt-in
`native_cpu_diagnostic` student rebuilt the accepted 47-position context
from frozen raw target features and candidate-D ordered arithmetic,
then ran its ordinary `decode_step` calls through the five native-input
draft depths. No captured K/V bytes or per-stage tensors were injected.

The reconstructed context matched **48,128/48,128 F16 keys and values
each**. The overwritten seed row and four subsequent rows matched all
**5,120/5,120** new F16 key and value writes each. Every checked native
graph tap is bitwise exact across the five depths, including
**20,480/20,480** query-RoPE and attention F32 values each, and
**12,800/12,800** normalized head-state F32 values. All **40/40** captured
head-logit probes match. The mapped argmax IDs are 2661, 504, 264, 264
and 3070 in both paths; argmax/label logits and verifier-label ranks
also match native.

The ignored machine report is
`results/recurrent-rope-oracle-20260928/integrated-reasoning-middle.json`
in the main checkout, SHA256
`1bda1318f85169f586490e62dd9220692824d7f4242c24aa224f02a7aaf26128`.
It verifies the complete frozen CPU capture file ledger, model and pinned
operator hashes, native cache bytes, graph taps and head-state/logit
joins. The real-input run and Ruff checks passed on Apple M3 Max CPU.

This extends the integrated first-round and accepted-draft checks to a
rejected-draft rewind on one training prompt. The mode is forward-only
and rejects gradient-enabled calls; no optimizer, target forward, GPU,
reserved-final prompt or Q4_0 evaluation ran. It is not a general
free-running, CUDA/SM75 or full-vocabulary logit parity result.
