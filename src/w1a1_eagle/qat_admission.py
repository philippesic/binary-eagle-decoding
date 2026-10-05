"""Typed current-package admission for the nine-model launcher only.

Legacy experiment launchers retain their existing gates. This object pins an
actual production JSON locator and the complete current configuration/source;
it cannot be constructed from a boolean or a synthetic fixture receipt.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class VerifiedTrainingAdmission:
    locator: Path
    locator_sha256: str
    config_path: Path
    config_sha256: str
    bundle_sha256: str
    candidate: str
    approved_qat_hashes: frozenset[str]

    @classmethod
    def from_locator(cls, locator, *, config_path, bundle_sha256, candidate):
        api = importlib.import_module("train_nine_model_qat")
        config_path = Path(config_path).resolve()
        locator = Path(locator).resolve()
        api.require_admission(locator, bundle_sha256, api.sha256(config_path), candidate=candidate)
        spec = api.load_spec(config_path)
        configs = api.declared_qat_configs(spec)
        return cls(
            locator,
            api.sha256(locator),
            config_path,
            api.sha256(config_path),
            bundle_sha256,
            candidate,
            frozenset(_digest(asdict(config)) for config in configs),
        )

    def require_for_qat(self, config):
        api = importlib.import_module("train_nine_model_qat")
        if (
            api.sha256(self.locator) != self.locator_sha256
            or api.sha256(self.config_path) != self.config_sha256
            or _digest(asdict(config)) not in self.approved_qat_hashes
        ):
            raise ValueError("current training admission locator/config/QAT contract changed")
        props = torch.cuda.get_device_properties(config.device)
        record = api.require_admission(
            self.locator,
            self.bundle_sha256,
            self.config_sha256,
            candidate=self.candidate,
            gpu_uuid=api.canonical_gpu_uuid(props.uuid),
        )
        if [props.major, props.minor] != [12, 0]:
            raise ValueError("current training admission needs actual SM120")
        execution = record.get("executed_paths", {}).get(str(config.contract.activation_bits), {})
        if config.optimize_cache and execution.get("context_cache_calls", 0) < 1:
            raise ValueError("current admission lacks executed cache path")
        if config.optimize_head and not execution.get("effective_batched"):
            if (
                config.activation_quantization != "learned"
                or execution.get("reason") != "trainable_learned_activation"
            ):
                raise ValueError("current admission lacks declared executed head path")
        return record
