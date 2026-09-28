# Optional native-forward EAGLE attention diagnostic

**2026-09-28 UTC.** The Python W1A16 EAGLE student now has an explicit
CPU-only diagnostic mode with a ggml Flash Attention forward and the former
F32 attention derivative as a **surrogate backward**. The default student
continues to use F32 attention. This mode is limited to a contiguous
single-sequence cache within 256 physical slots and the pinned `32×128` Q,
`8×128` K/V geometry. It runs the validated Apple CPU helper whose SHA256 is
`f63177876148da8e285afc39c021a9d28b9cc75bd020d92471a3d58bcdc3487f`.
No optimizer step, trained model, accelerator or Q4_0 evaluation ran here.

Parent commits `c960078`, `24f140e`, `adf7166`, `2cb1d75` and `d286bf5`
add the opt-in mode, real-step comparison, post-seed cache export, operand
ablation, native Q/K row conversion and later-only gradient check. Native
ggml expects interleaved Q/K channels; the Python drafter holds half-split
RoPE rows. The forward now converts Q and stored K to native row order at
the helper boundary, while the backward differentiates the unchanged
Python-order F32 attention expression. Tests verify exact helper output,
strict 256-slot F16 mask/geometry, the Q/K conversion, default-mode
stability and two-step causal gradients.

## First-seed operand ablation

The seven-mode ggml ablation used each request's archived physical F16 cache,
native graph Q/output, and the current student's exported Q/K/V. Every mode
kept the captured 256-slot mask. Results below are maximum absolute
differences against native `kqv_out-0` (4,096 F32 elements per seed):

| Query and cache operands | Prose | Reasoning |
| --- | ---: | ---: |
| Native Q + native K/V control | 0 | 0 |
| Student Q in raw Python order + native K/V | 14.7963 | 11.0058 |
| Student Q in native row order + native K/V | **0** | **0** |
| Native Q + student K in raw Python order and student V | 7.7967 | 8.1689 |
| Native Q + student K in native row order and student V | **0** | 0.002172 |
| Student Q/K in raw Python order and student V | 0.007402 | 0.009470 |
| Student Q/K in native row order and student V | **0** | 0.002172 |

The corrected full-student prose attention matched **4,096/4,096 F32 elements bitwise**.
The reasoning residual follows its stored K/V operand differences: after
native row conversion, 70 F16 key values and 66 F16 value values differ
across the 47 visible cache positions; maximum absolute differences are
0.00097656 and 0.00390625. In prose, five key and seven value values differ
across 32 positions, but the attention result still matches bitwise. The
student Q, when converted and paired with native K/V, matches native
attention bitwise on both prompts. These comparisons separate row-order
arithmetic from residual cache projection/rounding effects.

The ignored ablation reports are
`results/recurrent-attention-operands-20260928/prose-ablation.json` (SHA256
`3742bb7f6100e7877aa311d3cda1006599e9cc4257a7604d7d8ca6dc7997037f`)
and `reasoning-ablation.json` (SHA256
`3fe43dfd9b6bd76d49fcc8b92b37588a5215e94b948c2a237413a9a220fe4f42`).
They record graph/cache/adapter/helper hashes, unique execution/column
joins, physical slots, mask coverage and per-head errors.

## Student state and causal-gradient checks

Using grouped-F32-matmul binary weights on the same Apple M3 Max CPU
captures, the corrected optional attention mode changed the first-depth
differences as follows:

| Prompt | F32 student attention max error | Native-forward attention max error | F32 first normalized-state max error | Native-forward state max error |
| --- | ---: | ---: | ---: | ---: |
| Prose | 0.009986 | **0** | 0.002019 | **0.000184** |
| Reasoning | 0.009902 | **0.002172** | 0.002422 | **0.000511** |

The recorded first-round proposal top target IDs still match native under
both modes. Remaining downstream differences include binary reduction and
F16 threshold effects; whole-drafter exactness is unproved. The ignored
corrected step/graph reports have SHA256
`392c691ce8d1dc544d4750b2385b126701a61f26d9a4a7c78a83f2a84216d4db` /
`a5d13fd8afcaabf8eeba8fe4a57f4f234b10bf0c390ea95b458352e6747a8f00`
(prose) and
`07b0cf267f667124e702b725643aca7376e6f10e9463ce4d4f60ffb43545c191` /
`ec14b5eb598af43ae396ed0eaededa51d95ecd1f617e846d86eb447d8daa70c5`
(reasoning).

A no-optimizer, real-size **depth-1-only** CE check also passed with native
attention forward and F32 surrogate backward. Earlier depth-0 pre-norm
state gradients were nonzero in all 2,560 values; appended key and value
rows were nonzero in all 1,024 values each. Depth-0 logits had zero direct
gradient, and all nine binary sign/scale groups had finite nonzero shared
parameter gradients. CE was `2.105465888977051`, versus `2.106572151184082`
under the default F32 mode. Its ignored report SHA256 is
`1d50de7bfc48b6211ebc42272183b412b4cae13f0bfb1dc61375d396a814a892`.

This is a CPU diagnostic option, **not a selected training recipe**. The
forward oracle is verified only for the archived Apple ggml CPU backend;
the backward is the declared F32 surrogate, not the derivative of discrete
F16 native arithmetic. It does not authorize all-body optimization or set
a numeric tolerance. Native CUDA/SM75 behavior and Q4_0 acceptance,
latency and throughput remain separate gates.
