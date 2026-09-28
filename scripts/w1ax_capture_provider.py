"""Project-specific lazy provider for an eligible native EAGLE train bundle.

The factory validates small manifests, native trace ancestry and compact
teacher shards. It does not open model weights. `load_models` later verifies
frozen GGUF and snapshot files before loading the pinned official drafter and
target on CPU. The inherited 96-prompt bundle is rejected as ineligible.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from audit_recurrent_binary_capture import load_audited_capture
from compact_w1a_teacher import iter_verified_shards

from w1a1_eagle.native_step import NativeStepAdapter, bind_frozen_norms
from w1a1_eagle.recurrent_provider import TEACHER_FIELDS, ProviderRound

SCHEMA = "w1ax_native_train_provider_v1"
TARGET_REPO = "Qwen/Qwen3-4B"
TARGET_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DRAFT_REPO = "AngelSlim/Qwen3-4B_eagle3"
DRAFT_REVISION = "fd331e59626c8e95c392381a16ee59d518727fbb"
ANGELSLIM_REVISION = "0358da9c651e6a7d7ccafea26ced4b9c98d11681"
TARGET_GGUF_SHA256 = "05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6"
CANDIDATE_D_SHA256 = "10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return Path(value)


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase SHA256")
    return value


def _capture_keys(capture) -> dict[tuple[str, tuple[int, ...], int], Mapping[str, object]]:
    keys = {}
    for rows in capture.rows.values():
        for row in rows:
            if row["valid"] is not True:
                continue
            key = (row["prompt_id"], tuple(row["prefix_token_ids"]), row["target_logits_row"])
            if key in keys:
                raise ValueError("native capture repeats a valid teacher prefix/logit row")
            keys[key] = row
    return keys


def _bind_compact_teacher(
    directory: Path,
    capture,
    *,
    capture_id: str,
    prompts_hash: str,
    target_hash: str,
    d2t_hash: str,
    capture_hash: str,
) -> dict[tuple[str, tuple[int, ...], int], tuple[dict, dict[str, np.ndarray]]]:
    """Verify all shards and match every valid captured row exactly once."""
    teacher_manifest = json.loads((directory / "manifest.json").read_text())
    if teacher_manifest.get("capture_manifest_sha256") != capture_hash:
        raise ValueError("compact teacher came from a different native capture")
    native = _capture_keys(capture)
    index_path = directory / "rows.jsonl"
    expected_prefixes: dict[str, list[int]] = {}
    with index_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            metadata = json.loads(line)
            key = (
                metadata.get("prompt_id"),
                tuple(metadata.get("prefix_token_ids", ())),
                metadata.get("logits_row"),
            )
            if (
                metadata.get("capture_id") != capture_id
                or key not in native
                or metadata.get("id") in expected_prefixes
            ):
                raise ValueError("compact teacher index does not match native capture")
            expected_prefixes[metadata["id"]] = list(native[key]["prefix_token_ids"])
    if len(expected_prefixes) != len(native):
        raise ValueError("compact teacher omits valid native proposal rows")
    selected = {}
    for metadata_rows, arrays in iter_verified_shards(
        directory,
        expected_prompts_sha256=prompts_hash,
        expected_target_gguf_sha256=target_hash,
        expected_d2t_sha256=d2t_hash,
        expected_prefixes=expected_prefixes,
    ):
        for i, metadata in enumerate(metadata_rows):
            key = (
                metadata["prompt_id"],
                tuple(metadata["prefix_token_ids"]),
                metadata["logits_row"],
            )
            if key not in native or key in selected or metadata["capture_id"] != capture_id:
                raise ValueError("compact teacher duplicates or changes captured row ancestry")
            values = {name: np.array(arrays[name][i], copy=True) for name in TEACHER_FIELDS}
            values["next_target_id"] = np.array(arrays["next_target_id"][i], copy=True)
            if int(values["next_target_id"]) != native[key]["verifier_token_id"]:
                raise ValueError("compact teacher target label differs from native verifier")
            selected[key] = metadata, values
    if set(selected) != set(native):
        raise ValueError("compact teacher and native capture coverage differ")
    return selected


class NativeCaptureProvider:
    """A model-lazy adapter around a fully audited eligible native bundle."""

    def __init__(
        self,
        config,
        manifest_path: Path,
        *,
        capture_loader: Callable[..., Any] = load_audited_capture,
        model_loader: Callable[..., tuple[Any, Any]] | None = None,
        adapter_factory: Callable[..., Any] | None = None,
        candidate_loader: Callable[..., Any] | None = None,
    ) -> None:
        if torch.device(config.device).type not in {"cpu", "cuda"}:
            raise ValueError("native capture provider requires CPU or explicit CUDA")
        manifest_path = Path(manifest_path)
        spec = json.loads(manifest_path.read_text())
        if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
            raise ValueError("unsupported native train-provider manifest")
        if spec.get("training_eligible") is not True:
            raise ValueError("training manifest is not eligible")
        if (
            spec.get("split") != "train"
            or type(spec.get("prompt_count")) is not int
            or spec["prompt_count"] < 1
        ):
            raise ValueError("provider needs an eligible nonempty train split")
        paths = spec.get("paths")
        hashes = spec.get("sha256")
        if not isinstance(paths, dict) or not isinstance(hashes, dict):
            raise ValueError("provider paths and source hashes are required")
        required = {
            "capture_manifest",
            "prompts",
            "target_gguf",
            "candidate_d_gguf",
            "base_draft_gguf",
            "absolute_d2t",
            "model_snapshot_manifest",
            "target_model_dir",
            "draft_model_dir",
        }
        if set(paths) != required or set(hashes) != required - {
            "target_model_dir",
            "draft_model_dir",
        }:
            raise ValueError("provider path/hash inventory must be exact")
        self.paths = {name: _path(value, name) for name, value in paths.items()}
        self.hashes = {name: _hash(value, name) for name, value in hashes.items()}
        if (
            self.hashes["target_gguf"] != TARGET_GGUF_SHA256
            or self.hashes["candidate_d_gguf"] != CANDIDATE_D_SHA256
        ):
            raise ValueError("provider changed frozen target or candidate-D identity")
        for name in ("capture_manifest", "prompts", "absolute_d2t", "model_snapshot_manifest"):
            if sha256(self.paths[name]) != self.hashes[name]:
                raise ValueError(f"provider {name} SHA256 mismatch")
        capture_manifest = json.loads(self.paths["capture_manifest"].read_text())
        if (
            capture_manifest.get("schema") != "recurrent_binary_capture_v1"
            or capture_manifest.get("training_eligible") is not True
            or capture_manifest.get("readiness") != "full_body_qat_eligible"
            or capture_manifest.get("prompts_sha256") != self.hashes["prompts"]
            or capture_manifest.get("pinned_source_artifact_hashes_verified") is not True
            or capture_manifest.get("unverified_gates") != []
        ):
            raise ValueError("native capture bundle remains training-ineligible")
        cell_path = self.paths["capture_manifest"].parent / "source_cell_manifest.json"
        source_hashes = capture_manifest.get("source_report_sha256", {})
        if not cell_path.is_file() or sha256(cell_path) != source_hashes.get("cell_manifest"):
            raise ValueError("native capture cell source manifest is missing or changed")
        cell = json.loads(cell_path.read_text())
        if (
            cell.get("target_sha256") != self.hashes["target_gguf"]
            or cell.get("draft_sha256") != self.hashes["candidate_d_gguf"]
        ):
            raise ValueError("native capture target/draft GGUF identity differs")
        self.capture_id = spec.get("capture_id")
        if not isinstance(self.capture_id, str) or not self.capture_id:
            raise ValueError("provider needs a named native capture ID")
        self.training_eligible = True
        self.split = "train"
        self.base_gguf_sha256 = self.hashes["base_draft_gguf"]
        self.candidate_d = None
        snapshot = json.loads(self.paths["model_snapshot_manifest"].read_text())
        models = snapshot.get("models", {})
        for role, repo, revision in (
            ("target", TARGET_REPO, TARGET_REVISION),
            ("draft", DRAFT_REPO, DRAFT_REVISION),
        ):
            entry = models.get(role, {})
            if (
                entry.get("repo") != repo
                or entry.get("revision") != revision
                or Path(entry.get("directory", "")).resolve()
                != self.paths[f"{role}_model_dir"].resolve()
                or not isinstance(entry.get("files"), list)
                or not entry["files"]
            ):
                raise ValueError(f"{role} model snapshot identity differs from pinned revision")
        if spec.get("angelslim_revision") != ANGELSLIM_REVISION:
            raise ValueError("AngelSlim revision differs from pinned drafter")
        self.snapshot = snapshot
        self._config = config
        self._capture_loader = capture_loader
        self._model_loader = model_loader
        self._adapter_factory = adapter_factory
        self._candidate_loader = candidate_loader
        self.capture = capture_loader(
            self.paths["capture_manifest"],
            self.paths["prompts"],
            self.hashes["prompts"],
            expected_prompt_count=spec["prompt_count"],
        )
        self.allowed_prompt_ids = {key[0] for key in self.capture.anchors}
        if not self.allowed_prompt_ids:
            raise ValueError("audited native capture has no prompt IDs")
        capture_data = json.loads(self.paths["capture_manifest"].read_text())
        self.target_vocab_size = capture_data["target_vocab_size"]
        self.draft_vocab_size = capture_data["draft_vocab_size"]
        self.max_depth = capture_data["max_depth"]
        offsets_path = self.paths["capture_manifest"].parent / capture_data["offsets"]["path"]
        self.d2t_offsets = np.load(offsets_path, allow_pickle=False)
        absolute = np.load(self.paths["absolute_d2t"], allow_pickle=False)
        expected_absolute = np.arange(len(self.d2t_offsets), dtype=np.int64) + self.d2t_offsets
        if (
            absolute.dtype.kind not in "iu"
            or absolute.shape != expected_absolute.shape
            or not np.array_equal(absolute, expected_absolute)
        ):
            raise ValueError("absolute teacher d2t map differs from native offset map")
        self.absolute_d2t_raw_sha256 = hashlib.sha256(
            np.ascontiguousarray(absolute, dtype="<i8").tobytes()
        ).hexdigest()
        self.teacher = None
        if config.objective == "compact_probability":
            teacher = spec.get("teacher")
            if not isinstance(teacher, dict) or set(teacher) != {"directory", "manifest_sha256"}:
                raise ValueError("compact objective needs pinned teacher directory and manifest")
            directory = _path(teacher["directory"], "teacher.directory")
            if sha256(directory / "manifest.json") != _hash(
                teacher["manifest_sha256"], "teacher hash"
            ):
                raise ValueError("compact teacher manifest SHA256 mismatch")
            self.teacher = _bind_compact_teacher(
                directory,
                self.capture,
                capture_id=self.capture_id,
                prompts_hash=self.hashes["prompts"],
                target_hash=self.hashes["target_gguf"],
                d2t_hash=self.hashes["absolute_d2t"],
                capture_hash=self.hashes["capture_manifest"],
            )
        elif spec.get("teacher") is not None:
            raise ValueError("hard CE provider must not silently attach compact teacher")

    def load_models(self):
        """Hash pinned weights, then load official CPU target and drafter once."""
        from evaluate_pytorch_w1a1 import verify_model_snapshot

        from w1a1_eagle.official_loader import load_official_eagle3

        if sha256(self.paths["target_gguf"]) != self.hashes["target_gguf"]:
            raise ValueError("target GGUF SHA256 mismatch")
        if sha256(self.paths["candidate_d_gguf"]) != self.hashes["candidate_d_gguf"]:
            raise ValueError("candidate-D GGUF SHA256 mismatch")
        if sha256(self.paths["base_draft_gguf"]) != self.hashes["base_draft_gguf"]:
            raise ValueError("base draft F16 GGUF SHA256 mismatch")
        for role in ("target", "draft"):
            verify_model_snapshot(self.paths[f"{role}_model_dir"], self.snapshot["models"][role])
        if self._model_loader is None:
            model = load_official_eagle3(
                self.paths["target_model_dir"],
                self.paths["draft_model_dir"],
                angelslim_revision=ANGELSLIM_REVISION,
                total_token=60,
                depth=5,
                top_k=10,
                threshold=1.0,
                # Captured native features/logits replace target forward in
                # this QAT path. Keep the frozen target on CPU; only the
                # drafter and its borrowed embedding need accelerator memory.
                target_load_kwargs={"dtype": torch.float16, "device_map": "cpu"},
            )
            model.eval()
            drafter, target = model.eagle_layer, model.base_model
            if torch.device(self._config.device).type == "cuda":
                drafter.to(self._config.device)
        else:
            drafter, target = self._model_loader(self.paths)
        if self._config.contract.scale_layout == "group128":
            if self._candidate_loader is None:
                from load_recurrent_binary_init import load_candidate_d_arrays

                self._candidate_loader = load_candidate_d_arrays
            self.candidate_d, report = self._candidate_loader(
                self.paths["candidate_d_gguf"], self.hashes["candidate_d_gguf"]
            )
            if (
                report["draft_vocab_size"] != self.draft_vocab_size
                or report["d2t_sha256"] != self.absolute_d2t_raw_sha256
            ):
                raise ValueError("candidate-D vocabulary map differs from native capture")
        return drafter, target

    def make_step_adapter(self, drafter):
        if self._adapter_factory is not None:
            return self._adapter_factory(drafter)
        from w1a1_eagle.frozen_operands import FrozenOperands

        operands = FrozenOperands(
            self.paths["target_gguf"],
            self.paths["candidate_d_gguf"],
            target_sha256=self.hashes["target_gguf"],
            draft_sha256=self.hashes["candidate_d_gguf"],
            vocab_size=self.target_vocab_size,
            hidden_size=drafter.config.hidden_size,
        )
        bind_frozen_norms(drafter, operands.norm_arrays)
        if torch.device(self._config.device).type == "cpu":
            return NativeStepAdapter(drafter, embedding_lookup=operands)
        return NativeStepAdapter(drafter)

    def rounds(self) -> Iterable[ProviderRound]:
        for key in sorted(self.capture.anchors):
            round_data = self.capture.round_inputs(*key)
            rows = round_data.rows
            if self.teacher is None:
                yield ProviderRound(
                    round_data.anchor,
                    rows,
                    round_data.prefix_token_ids,
                    round_data.raw_target_features,
                    round_data.feature_positions,
                    self.capture_id,
                )
                continue
            metadata = []
            values = {field: [] for field in (*TEACHER_FIELDS, "next_target_id")}
            ids = []
            for row in rows:
                if not row["valid"]:
                    ids.append(None)
                    continue
                teacher_key = (
                    row["prompt_id"],
                    tuple(row["prefix_token_ids"]),
                    row["target_logits_row"],
                )
                item, payload = self.teacher[teacher_key]
                ids.append(item["id"])
                metadata.append(item)
                for field in values:
                    values[field].append(payload[field])
            arrays = {name: np.asarray(elements) for name, elements in values.items()}
            yield ProviderRound(
                round_data.anchor,
                rows,
                round_data.prefix_token_ids,
                round_data.raw_target_features,
                round_data.feature_positions,
                self.capture_id,
                tuple(ids),
                tuple(metadata),
                arrays,
            )


def create_provider(config, manifest_path: Path | None = None):
    if manifest_path is None:
        raise ValueError("w1ax_capture_provider requires --provider-manifest")
    return NativeCaptureProvider(config, manifest_path)
