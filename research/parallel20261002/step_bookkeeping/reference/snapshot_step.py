"""Source-bound, isolated one-line variant of the production joint step.

The original function is compiled in a separate namespace, never monkeypatched
or installed into the live module. Source drift fails closed pending review.
"""

import hashlib
import inspect

from w1a1_eagle import recurrent_qat

FUNCTION_SHA256 = "b7509b40ea550b9b0104a2f3eef5d89d520160916b4b2ada2a83dfa5dd377f52"
OLD = "m.latent_sign.detach().clone() < 0"
NEW = "m.latent_sign.detach() < 0"


def source_pair():
    source = inspect.getsource(recurrent_qat.joint_train_step)
    if hashlib.sha256(source.encode()).hexdigest() != FUNCTION_SHA256:
        raise RuntimeError("joint_train_step source changed; review snapshot prototype")
    if source.count(OLD) != 1:
        raise RuntimeError("expected exactly one diagnostic snapshot expression")
    return source, source.replace(OLD, NEW)


def make_snapshot_step():
    _, candidate = source_pair()
    namespace = dict(vars(recurrent_qat))
    exec(compile(candidate, "<isolated-snapshot-step>", "exec"), namespace)
    return namespace["joint_train_step"]


joint_train_step_without_snapshot_clone = make_snapshot_step()
