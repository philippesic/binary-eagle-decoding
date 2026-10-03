# Full-source model readiness — October 3, 2026

The unchanged fixed reference A8/A1 recipe passed actual current native/model,
backward, memory and five-repeat timing gates on RTX5080/SM120 at11:55:55 UTC.
No optimizer update occurred. This admits the measured recipe/current state;
actual original checkpoint restore/current smoke/liveownership still precede
updates, and heldout Q4_0 acceptance/throughput remain training supervision.

| Measurement | Actual result |
| --- | --- |
| Scope | Full-shape paired A8/A1, eligible bounded TRAIN cases bound to complete10000prompt/3899930row source |
| Forward/backward | 30/30, finite gradients and positive later state/K/V paths |
| Timing | Five measured repetitions under unchanged producer |
| Paired B1 memory | Peak8.56GB allocated/9.11GB reserved; minimum6.48GB free |
| Current native decisions | 12cases;0changed/0material choices; maxlogitRMS0.00643/state0.00263 |
| Supervision |6044/group6045 naturalexit0, allowned groups/contexts returned |

External source6f/native9e2/helperd211 and actual Python3.11.15/Torch2.14+cu130/
NumPy2.4.6/highestF32/matmulTF32false/cudnnTF32true are bound to receipts. FP32
trainable masters simulate binary paths during training; current native decisions
use the copied CUDA runtime. No SM75 or performance/quality-win claim.

Readiness SHA3badd9775a848ff5ace09fd96e573f96e19b86e85a88a9bb81aa18d8d3e608c7;
untouched sidecar4816152723691eb94df34529a1174af0036a3e8a4622cec4ac609e14d0a68a13.
All164originals are saved/independently hashed outsideGit. The [active goal](../../docs/goals/qat-optimization-readiness.md#actual-full-source-model-readiness-passed--october-3-1155-utc)
records exact data/state/runtime/source/command/artifact pins and finite first-run
caps1000steps/3600trainerseconds/1epoch. Optional controls stay inactive and no
quality winner is selected. Q4_0 remains primary; FP16 EAGLE is diagnostic.
