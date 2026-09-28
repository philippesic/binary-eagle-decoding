# Same-input candidate-D FFN arithmetic on CPU

**2026-09-28 UTC.** This CPU-only diagnostic takes the *same native F32
`post_attn_norm-0` input* from the first proposal's archived EAGLE graph
and runs candidate D's binary FFN gate, up and down projections. It compares
the complete 2,560-wide output with native `ffn_out-0`. No context rebuild,
target forward, optimizer, new model inference, GPU work or quality/timing
measurement is involved.

Parent commit `3e2b73f` adds the strict graph/head-state join and two
arithmetic replays. The first seed is uniquely reconstructed at decoder
execution 2, column 0 from its native output-normalized head state. The
script pins the D GGUF, exact native graph input bytes, FFN source operation
order and three binary weight/scale pairs. It evaluates native sequential
F32 group reduction and the practical grouped-F32-matmul path under the
same A16 input casts, SiLU and gate/up multiplication. Six synthetic checks,
Ruff and formatting passed.

| Frozen training capture | Arithmetic | Native FFN output elements bitwise equal | Max absolute error | RMS error |
| --- | --- | ---: | ---: | ---: |
| Prose: urban waterways 01 | Native order | **2,560 / 2,560** | **0** | **0** |
| Prose: urban waterways 01 | Grouped matmul | 1,688 / 2,560 | 1.4305e-6 | 2.2482e-7 |
| Reasoning: rate and work 01 | Native order | 5 / 2,560 | 6.1035e-5 | 1.0246e-5 |
| Reasoning: rate and work 01 | Grouped matmul | 1 / 2,560 | 6.1035e-5 | 1.0260e-5 |

The two Python arithmetic modes differ from each other by at most
`1.9073e-6` on reasoning, much less than either mode's `6.1035e-5`
native difference. Thus grouped reduction order explains the tiny prose
FFN drift but does not explain the remaining reasoning difference. Candidate
causes include nonlinear evaluation or an intermediate F16 threshold; the
native graph does not capture gate/up or SiLU intermediates, so this check
cannot assign the discrepancy to one suboperation.

Ignored reports are
[prose](../results/recurrent-ffn-same-input-20260928/prose.json), SHA256
`25a84157686595caa79d89d721a5f1b2fab41627e066f393001b7eeae9091c23`,
and [reasoning](../results/recurrent-ffn-same-input-20260928/reasoning.json),
SHA256 `afe5faf6ca28dc55d3acbd267eaf0402bdd58ce0a1a5be9dfc8e921d0c39a887`.
They record Apple M3 Max arm64, PyTorch/NumPy versions, one CPU thread,
the pinned D hash, graph/head/source hashes, exact input identity and
per-mode gate/up summaries. The native source operation order is from
fork `eagle3.cpp` SHA256
`c03713b008db4b81eff49f5b1eb5605d4f3b7deb38276e0c5a91eecf177d729d`.

The result is a **single same-input FFN operator check** per prompt. It
does not certify full-drafter state/logit parity, choose training arithmetic,
set a numeric tolerance or predict Q4_0 acceptance and throughput.
