"""Exact source-bound isolated provider patch, loaded into a private module only."""

from __future__ import annotations

import difflib
import hashlib
import sys
import types
from pathlib import Path

from .audit import ROOT

EXPECTED_SOURCE_SHA256 = "81cd3643b9a3060ae00d50d18cdf9ebc39846dbafc3b772b62651fb20d4b31a3"
OLD = """    decode_step = adapter.decode_step
    if head is not None:
"""
NEW = """    if head is not None and torch.is_grad_enabled():
        quantizer = getattr(
            getattr(adapter, "linears", {}).get("lm_head"), "activation_quantizer", None
        )
        if quantizer is not None and quantizer.parameter.requires_grad:
            # The learned parameter uses invocation-local normalization. Serial
            # learned heads preserve the reference recipe until a chain domain
            # is explicitly declared and forwarded through every head call.
            head = None
    decode_step = adapter.decode_step
    if head is not None:
"""


def proposed_source():
    source = (ROOT / "src/w1a1_eagle/recurrent_provider.py").read_text()
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Proposal source hash changed: re-review rather than silently applying")
    if source.count(OLD) != 1:
        raise RuntimeError("Proposal insertion point is not unique")
    return source, source.replace(OLD, NEW)


def patch_text():
    source, proposed = proposed_source()
    return "".join(
        difflib.unified_diff(
            source.splitlines(True),
            proposed.splitlines(True),
            fromfile="a/src/w1a1_eagle/recurrent_provider.py",
            tofile="b/src/w1a1_eagle/recurrent_provider.py",
        )
    )


def load_proposed_forward():
    _, source = proposed_source()
    name = "w1a1_eagle._isolated_lsq_batching_proposal"
    module = types.ModuleType(name)
    module.__package__ = "w1a1_eagle"
    module.__file__ = str(Path(__file__))
    sys.modules[name] = module
    try:
        exec(compile(source, "<isolated-provider-proposal>", "exec"), module.__dict__)
        return module.forward_torch_round
    finally:
        del sys.modules[name]
