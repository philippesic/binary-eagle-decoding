"""Selective W1A1 wrappers for the pinned AngelSlim EAGLE-3 PyTorch drafter.

Install these wrappers *after* the official drafter has loaded its checkpoint.
Only the named drafter-owned linear modules are touched; target modules and the
borrowed token embedding are outside the eligible paths. The official source is
Tencent/AngelSlim at 0358da9c651e6a7d7ccafea26ced4b9c98d11681.
"""

from collections.abc import Iterable
from types import MappingProxyType

from torch import nn

from .fake_binary import W1A1Config, W1A1Linear
from .fake_uniform import FakeUniformLinear, UniformQuantConfig

GROUP_PATHS = MappingProxyType(
    {
        "feature_fusion": ("fc",),
        "attention": (
            "midlayer.self_attn.q_proj",
            "midlayer.self_attn.k_proj",
            "midlayer.self_attn.v_proj",
            "midlayer.self_attn.o_proj",
        ),
        "ffn": (
            "midlayer.mlp.gate_proj",
            "midlayer.mlp.up_proj",
            "midlayer.mlp.down_proj",
        ),
        "lm_head": ("lm_head",),
    }
)


class DrafterStructureError(ValueError):
    """The supplied model does not match the pinned drafter's eligible graph."""


def _parent_and_name(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        child = getattr(parent, part, None)
        if not isinstance(child, nn.Module):
            raise DrafterStructureError(
                f"missing module {path!r}: {part!r} is absent or not a module"
            )
        parent = child
    return parent, parts[-1]


def _target_owns_linear(target: nn.Module, linear: nn.Linear) -> bool:
    """Detect an aliased target module or an aliased target weight parameter."""
    return any(
        module is linear or (isinstance(module, nn.Linear) and module.weight is linear.weight)
        for module in target.modules()
    )


class DrafterW1A1Adapter:
    """Handle for in-place quantizer wrappers on an already-loaded drafter.

    ``set_enabled(False)`` calls each original ``nn.Linear`` directly. The
    original module objects and parameters are retained, so ``uninstall()`` can
    restore the pre-install graph. This handle is deliberately not an
    ``nn.Module``: it must not register another copy of the drafter.
    """

    def __init__(self, drafter: nn.Module, wrappers: dict[str, W1A1Linear | FakeUniformLinear]):
        self.drafter = drafter
        self.wrappers = MappingProxyType(wrappers)
        self._installed = True

    def set_enabled(self, enabled: bool) -> None:
        if not self._installed:
            raise RuntimeError("W1A1 adapter has been uninstalled")
        for wrapper in self.wrappers.values():
            wrapper.set_enabled(enabled)

    def uninstall(self) -> None:
        if not self._installed:
            raise RuntimeError("W1A1 adapter has already been uninstalled")
        # Check all paths before replacing any module, in case another caller
        # modified the drafter after installation.
        for path, wrapper in self.wrappers.items():
            parent, name = _parent_and_name(self.drafter, path)
            if getattr(parent, name, None) is not wrapper:
                raise DrafterStructureError(f"cannot uninstall: module {path!r} changed")
        for path, wrapper in self.wrappers.items():
            parent, name = _parent_and_name(self.drafter, path)
            setattr(parent, name, wrapper.linear)
        self._installed = False


def _validated_linears(
    drafter: nn.Module,
    groups: Iterable[str],
    target: nn.Module | None,
    kind: str,
) -> dict[str, tuple[nn.Module, str, nn.Linear]]:
    """Reject unsupported or target-owned paths before changing the drafter."""
    if not isinstance(drafter, nn.Module):
        raise TypeError("drafter must be an nn.Module")
    if target is not None and not isinstance(target, nn.Module):
        raise TypeError("target must be an nn.Module")
    if isinstance(groups, str):
        raise TypeError("groups must be an iterable of group names, not one string")
    selected = frozenset(groups)
    unknown = selected - GROUP_PATHS.keys()
    if unknown:
        raise ValueError(f"unknown {kind} group(s): {', '.join(sorted(unknown))}")
    if selected & {"attention", "ffn"}:
        draft_config = getattr(drafter, "config", None)
        tp = getattr(draft_config, "pretraining_tp", None)
        if tp != 1:
            raise DrafterStructureError(
                "attention/ffn wrapping requires drafter.config.pretraining_tp == 1; "
                "the pinned forward bypasses module calls otherwise"
            )

    paths = tuple(
        path
        for group, group_paths in GROUP_PATHS.items()
        if group in selected
        for path in group_paths
    )
    pending: dict[str, tuple[nn.Module, str, nn.Linear]] = {}
    for path in paths:
        parent, name = _parent_and_name(drafter, path)
        linear = getattr(parent, name, None)
        if not isinstance(linear, nn.Linear):
            found = type(linear).__name__ if linear is not None else "missing"
            raise DrafterStructureError(f"expected nn.Linear at {path!r}; found {found}")
        if target is not None and _target_owns_linear(target, linear):
            raise DrafterStructureError(f"refusing to wrap target-owned linear at {path!r}")
        pending[path] = (parent, name, linear)
    return pending


def install_w1a1(
    drafter: nn.Module,
    groups: Iterable[str],
    config: W1A1Config = W1A1Config(),
    *,
    enabled: bool = True,
    target: nn.Module | None = None,
) -> DrafterW1A1Adapter:
    """Wrap selected EAGLE-3 linear groups without changing its forward code.

    Group names are ``feature_fusion``, ``attention``, ``ffn``, and ``lm_head``.
    ``target`` is optional, but when given, any selected linear sharing a module
    or weight parameter with it is rejected. In all cases only exact paths
    under ``drafter`` can be wrapped; ``embed_tokens`` is never selected.

    The pinned attention and MLP code bypasses module calls when
    ``config.pretraining_tp > 1``. Those groups are rejected in that mode.
    Validation precedes mutation, so a bad path cannot leave partial wrappers.
    """
    pending = _validated_linears(drafter, groups, target, "W1A1")

    wrappers: dict[str, W1A1Linear] = {}
    for path, (parent, name, linear) in pending.items():
        wrapper = W1A1Linear(linear, config, enabled=enabled)
        setattr(parent, name, wrapper)
        wrappers[path] = wrapper
    return DrafterW1A1Adapter(drafter, wrappers)


def install_fake_uniform(
    drafter: nn.Module,
    groups: Iterable[str],
    config: UniformQuantConfig,
    *,
    enabled: bool = True,
    target: nn.Module | None = None,
) -> DrafterW1A1Adapter:
    """Wrap only the selected drafter linears with uniform low-bit simulation."""
    pending = _validated_linears(drafter, groups, target, "uniform quantization")
    wrappers: dict[str, W1A1Linear | FakeUniformLinear] = {}
    for path, (parent, name, linear) in pending.items():
        wrapper = FakeUniformLinear(linear, config, enabled=enabled)
        setattr(parent, name, wrapper)
        wrappers[path] = wrapper
    return DrafterW1A1Adapter(drafter, wrappers)
