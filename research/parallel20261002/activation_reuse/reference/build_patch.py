"""Emit the unapplied, source-bound production integration proposal."""

import difflib
import inspect
from pathlib import Path

from research.parallel20261002.activation_reuse.reference import prototype
from w1a1_eagle.native_step import NativeStepAdapter

ROOT = Path(__file__).resolve().parents[4]


def patch_text():
    helper = Path(prototype.__file__).read_text().split('\nQKV = ')[0]
    for line in ("import inspect\n", "import textwrap\n", "from types import MethodType\n",
                 "from w1a1_eagle.native_step import NativeStepAdapter\n"):
        helper = helper.replace(line, "")
    helper = helper.replace("Source-bound research adapter:", "Scoped learned quantization:")
    recurrent = ROOT / "src/w1a1_eagle/recurrent_qat.py"
    native = ROOT / "src/w1a1_eagle/native_step.py"
    old_recurrent, old_native = recurrent.read_text(), native.read_text()
    assert old_recurrent.count("result = quantizer(input)") == 2
    new_recurrent = old_recurrent.replace(
        "result = quantizer(input)", "result = quantize(quantizer, input)"
    ).replace("import torch\n", "import torch\n\nfrom .immediate_activation import quantize\n", 1)
    original = inspect.getsource(NativeStepAdapter.decode_step)
    transformed = prototype.transformed_decode_source().replace(
        ", self.reuse_events", ""
    )
    transformed = "\n".join("    " + line if line else "" for line in transformed.splitlines())
    transformed += "\n"
    assert old_native.count(original) == 1
    new_native = old_native.replace(original, transformed).replace(
        "import torch\n", "import torch\n\nfrom .immediate_activation import immediate_group\n", 1
    )
    patch = []
    for name, before, after in (
        ("src/w1a1_eagle/immediate_activation.py", "", helper),
        ("src/w1a1_eagle/recurrent_qat.py", old_recurrent, new_recurrent),
        ("src/w1a1_eagle/native_step.py", old_native, new_native),
    ):
        patch.extend(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="/dev/null" if not before else "a/" + name, tofile="b/" + name,
        ))
    return "".join(patch)


if __name__ == "__main__":
    (ROOT / "research/parallel20261002/activation_reuse/reference/integration.patch").write_text(
        patch_text()
    )
