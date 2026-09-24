"""PyTorch components for W1A1 EAGLE research."""

from .adapter import (
    GROUP_PATHS,
    DrafterStructureError,
    DrafterW1A1Adapter,
    install_fake_uniform,
    install_w1a1,
)
from .fake_binary import W1A1Config, W1A1Linear, fake_binary_linear
from .fake_uniform import FakeUniformLinear, UniformQuantConfig, fake_uniform_linear

__all__ = [
    "DrafterStructureError",
    "DrafterW1A1Adapter",
    "FakeUniformLinear",
    "GROUP_PATHS",
    "UniformQuantConfig",
    "W1A1Config",
    "W1A1Linear",
    "fake_binary_linear",
    "fake_uniform_linear",
    "install_fake_uniform",
    "install_w1a1",
]
