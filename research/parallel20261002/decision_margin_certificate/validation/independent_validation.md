# Independent decision-margin certificate validation

The independent CPU oracle passed 10 tests against the final reference at
`dd06cc138cfed102712b3e72165889acb74ebb55dc077c138e38e61e5355a8c5` (SHA-256
of `reference/certificate.py`). Probability intervals were checked against
direct softmax evaluations at all vertices of 18 tiny boxes, 168 vertices in
total. The fixtures also checked greedy stability, exact and nonzero-allowance
ties, outward rounding at a half-ULP decision boundary, opposite-sign extreme
logits with a large temperature, ordered token mapping, identity binding,
contract and malformed-input rejection, and a constructive case showing why an
RMS error cannot stand in for a coordinatewise maximum. The RMS control
deliberately supplies an invalid maximum-error claim; its false stable result
illustrates the certificate's dependence on externally justified bounds.

Run command:

```sh
python3 research/parallel20261002/decision_margin_certificate/validation/independent_validation.py > runs/parallel20261002/decision-margin-certificate-validation/independent_validation.log 2>&1
```

Result: 10/10 tests passed in 0.002 seconds. The independent probability oracle
uses Python `math` and exhaustive vertex enumeration; no project dependencies
are required.

Validation environment: Python 3.11.3 on macOS 27.0 arm64, CPU only. Ruff
check, Ruff format check, and `git diff --check` passed. Raw log:
`runs/parallel20261002/decision-margin-certificate-validation/independent_validation.log`
(SHA-256
`87777d2f628cecdc30e9ed6385ae2d072e7c15ffb5d1e2510e86ac4aa7abc3b3`). The test
process exited; no GPU, Metal, SSH, model, data, or paid resource was used, and
no persistent process or device allocation remains.
