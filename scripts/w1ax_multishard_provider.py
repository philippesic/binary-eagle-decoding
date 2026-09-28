"""One-optimizer training provider for the frozen 2k W1Ax capture-shard plan.

An execution manifest binds every planned shard ordinal to one eligible
`w1ax_native_train_provider_v1` manifest after capture. All child capture and
teacher audits run before loading the drafter. The wrapper loads one model,
yields rounds in frozen shard order, and keeps one optimizer/checkpoint path.
No model or accelerator is touched when this module is imported.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from w1ax_capture_provider import NativeCaptureProvider, sha256

SCHEMA = "w1ax_multishard_train_v1"
PLAN_SCHEMA = "w1ax_capture_shard_plan_v1"
SHARD_SCHEMA = "w1ax_capture_shard_v1"
CHILD_SCHEMA = "w1ax_native_train_provider_v1"
COMMON_HASHES = (
    "target_gguf",
    "candidate_d_gguf",
    "base_draft_gguf",
    "absolute_d2t",
    "model_snapshot_manifest",
)


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(canonical(ids)).hexdigest()


def _absolute(value: object, name: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return Path(value)


def _prompt_ids(path: Path) -> list[str]:
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [row.get("id") for row in rows]
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"{path.name} has invalid prompt IDs")
    return ids


def _plan(plan_path: Path, parent_manifest: Path, expected_sha256: str) -> tuple[dict, list[dict]]:
    if sha256(plan_path) != expected_sha256:
        raise ValueError("capture shard plan SHA256 mismatch")
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("wrong capture shard plan schema")
    parent = plan.get("parent")
    if (
        not isinstance(parent, dict)
        or parent.get("split") != "train_small"
        or sha256(parent_manifest) != parent.get("manifest_sha256")
    ):
        raise ValueError("parent frozen train manifest differs from shard plan")
    for field, filename in (
        ("prompts_sha256", "prompts_filename"),
        ("index_sha256", "index_filename"),
    ):
        path = parent_manifest.parent / parent[filename]
        if sha256(path) != parent[field]:
            raise ValueError(f"parent {filename} SHA256 mismatch")
    parent_ids = _prompt_ids(parent_manifest.parent / parent["prompts_filename"])
    if (
        len(parent_ids) != parent.get("count")
        or parent_ids != plan.get("parent_prompt_ids")
        or ids_sha256(parent_ids) != plan.get("parent_prompt_ids_sha256")
        or len(set(parent_ids)) != len(parent_ids)
    ):
        raise ValueError("parent prompt order or coverage differs from frozen plan")
    records = plan.get("shards")
    if (
        not isinstance(records, list)
        or not records
        or len(records) != plan.get("shard_count")
        or plan.get("total_prompts") != len(parent_ids)
    ):
        raise ValueError("capture shard plan count invalid")
    ids_in_order = []
    positions = []
    for ordinal, record in enumerate(records):
        if record.get("ordinal") != ordinal:
            raise ValueError("capture shard ordinals are not contiguous")
        ids = record.get("prompt_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or len(ids) != record.get("prompt_count")
            or ids_sha256(ids) != record.get("prompt_ids_sha256")
        ):
            raise ValueError("capture shard prompt ID hash or count mismatch")
        child_path = (plan_path.parent / record["manifest"]).resolve()
        if sha256(child_path) != record.get("manifest_sha256"):
            raise ValueError("child shard plan manifest SHA256 mismatch")
        child = json.loads(child_path.read_text())
        if (
            child.get("schema") != SHARD_SCHEMA
            or child.get("parent") != parent
            or child.get("ordinal") != ordinal
            or child.get("prompt_ids") != ids
            or child.get("prompts_sha256") != record.get("prompts_sha256")
            or child.get("prompt_count") != len(ids)
        ):
            raise ValueError("child shard manifest disagrees with parent plan")
        prompt_path = child_path.parent / child["prompts_path"]
        if sha256(prompt_path) != record["prompts_sha256"] or _prompt_ids(prompt_path) != ids:
            raise ValueError("child shard prompt bytes or order changed")
        ids_in_order.extend(ids)
        positions.extend(child["source_positions"])
    if ids_in_order != parent_ids or positions != list(range(len(parent_ids))):
        raise ValueError("capture shards omit, duplicate or reorder frozen prompts")
    return plan, records


class MultiShardNativeProvider:
    """Validated ordered children sharing one frozen model and optimizer."""

    def __init__(self, config, execution_manifest: Path, *, child_factory=NativeCaptureProvider):
        execution_manifest = Path(execution_manifest)
        spec = json.loads(execution_manifest.read_text())
        if spec.get("schema") != SCHEMA:
            raise ValueError("wrong multi-shard execution manifest schema")
        plan_path = _absolute(spec.get("plan"), "plan")
        parent_path = _absolute(spec.get("parent_manifest"), "parent manifest")
        if sha256(parent_path) != spec.get("parent_manifest_sha256"):
            raise ValueError("execution parent manifest SHA256 mismatch")
        plan, records = _plan(plan_path, parent_path, spec.get("plan_sha256"))
        attached = spec.get("shards")
        if not isinstance(attached, list) or len(attached) != len(records):
            raise ValueError("execution manifest must attach every planned shard")
        children = []
        first = None
        sources = []
        capture_ids = set()
        for ordinal, (attachment, record) in enumerate(zip(attached, records)):
            if attachment.get("ordinal") != ordinal:
                raise ValueError("execution shard ordinals are not contiguous")
            path = _absolute(attachment.get("provider_manifest"), "provider manifest")
            digest = attachment.get("provider_manifest_sha256")
            if sha256(path) != digest:
                raise ValueError("child provider manifest SHA256 mismatch")
            child_spec = json.loads(path.read_text())
            if (
                child_spec.get("schema") != CHILD_SCHEMA
                or child_spec.get("training_eligible") is not True
                or child_spec.get("prompt_count") != record["prompt_count"]
                or child_spec.get("sha256", {}).get("prompts") != record["prompts_sha256"]
            ):
                raise ValueError("child provider is ineligible or uses wrong shard prompts")
            prompt_path = _absolute(child_spec["paths"]["prompts"], "child prompts")
            if _prompt_ids(prompt_path) != record["prompt_ids"]:
                raise ValueError("child captured prompt order differs from shard plan")
            child = child_factory(config, path)
            if child.target_vocab_size != plan.get("target_vocab_size"):
                raise ValueError("audited child target vocabulary differs from shard plan")
            if set(child.allowed_prompt_ids) != set(record["prompt_ids"]):
                raise ValueError("audited child capture prompt IDs differ from shard plan")
            if child.capture_id in capture_ids:
                raise ValueError("capture ID is reused across planned shards")
            capture_ids.add(child.capture_id)
            if first is None:
                first = child
            elif (
                any(child.hashes[key] != first.hashes[key] for key in COMMON_HASHES)
                or child.target_vocab_size != first.target_vocab_size
                or child.draft_vocab_size != first.draft_vocab_size
                or child.max_depth != first.max_depth
                or tuple(child.d2t_offsets) != tuple(first.d2t_offsets)
            ):
                raise ValueError("capture shards disagree on frozen model/map contract")
            children.append(child)
            sources.append(
                {
                    "ordinal": ordinal,
                    "provider_manifest_sha256": digest,
                    "capture_manifest_sha256": child.hashes["capture_manifest"],
                    "prompts_sha256": record["prompts_sha256"],
                    "prompt_ids_sha256": record["prompt_ids_sha256"],
                    "capture_id": child.capture_id,
                    "teacher_manifest_sha256": (child_spec.get("teacher") or {}).get(
                        "manifest_sha256"
                    ),
                    "rounds": child.total_rounds,
                }
            )
        self.children = tuple(children)
        self.training_eligible = True
        self.split = "train"
        self.allowed_prompt_ids = set(plan["parent_prompt_ids"])
        self.target_vocab_size = first.target_vocab_size
        self.draft_vocab_size = first.draft_vocab_size
        self.max_depth = first.max_depth
        self.d2t_offsets = first.d2t_offsets
        self.base_gguf_sha256 = first.base_gguf_sha256
        self.candidate_d = None
        self.total_rounds = sum(child.total_rounds for child in children)
        self.source_metadata = {
            "factory": "w1ax_multishard_provider:create_provider",
            "execution_manifest_sha256": sha256(execution_manifest),
            "plan_sha256": spec["plan_sha256"],
            "parent_manifest_sha256": plan["parent"]["manifest_sha256"],
            "parent_prompts_sha256": plan["parent"]["prompts_sha256"],
            "parent_prompt_ids_sha256": plan["parent_prompt_ids_sha256"],
            "prompt_count": len(plan["parent_prompt_ids"]),
            "shard_count": len(children),
            "round_count": self.total_rounds,
            "base_gguf_sha256": self.base_gguf_sha256,
            "common_source_sha256": {key: first.hashes[key] for key in COMMON_HASHES},
            "shards": sources,
        }

    def load_models(self):
        result = self.children[0].load_models()
        self.candidate_d = self.children[0].candidate_d
        return result

    def make_step_adapter(self, drafter):
        return self.children[0].make_step_adapter(drafter)

    def rounds(self):
        for ordinal, child in enumerate(self.children):
            for batch in child.rounds():
                yield replace(batch, shard_ordinal=ordinal)


def create_provider(config, manifest_path: Path | None = None):
    if manifest_path is None:
        raise ValueError("w1ax_multishard_provider requires --provider-manifest")
    return MultiShardNativeProvider(config, manifest_path)
