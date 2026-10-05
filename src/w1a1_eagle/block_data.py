"""Authenticated target-only TRAIN chains for the admitted seven-slot block models.

Native tap IDs refer to layer *inputs*, not author hidden-state output IDs.
Artifacts are uncompressed NPY arrays outside Git; full-vocabulary logits are
memory mapped and only one block's rows are copied. This module never captures
teachers, promotes held-out data, or substitutes top-k teachers.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "native_block_train_v1"
TAPS = (2, 10, 18, 26, 34)
DOMAINS = ("prose", "code", "reasoning")
SPLITS = ("train", "calibration_fit", "calibration_validation")
REPLAY_PREFIX_CONTRACT = "teacher_forced_exact_caller_token_ids"
GENERATED_PREFIX_CONTRACT = "native_tokenized_prompt_then_target_only_greedy"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def token_sha256(tokens) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def _hash(value) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_decode_history(history, length):
    """Preserve exact native prefill/greedy partitions and F16 KV reuse."""
    if not isinstance(history, list) or not history:
        raise ValueError("exact native decode history required")
    offset = 0
    greedy_started = False
    for record in history:
        if (
            not isinstance(record, dict)
            or set(record) != {"offset", "count", "phase", "kv_reused_from_same_chain"}
            or type(record["offset"]) is not int
            or record["offset"] != offset
            or type(record["count"]) is not int
            or not 1 <= record["count"] <= 256
            or record["phase"] not in ("prefill", "target_only_greedy")
            or type(record["kv_reused_from_same_chain"]) is not bool
            or record["kv_reused_from_same_chain"] != (offset > 0)
            or (record["phase"] == "target_only_greedy" and record["count"] != 1)
            or (greedy_started and record["phase"] != "target_only_greedy")
        ):
            raise ValueError("native decode partition/phase/F16 KV ancestry differs")
        greedy_started |= record["phase"] == "target_only_greedy"
        offset += record["count"]
    if offset != length:
        raise ValueError("native decode history does not cover exact token chain")
    return history


def crop_decode_history(history, length):
    """Keep all source partitions; truncate only the final intersecting chunk."""
    result = []
    for record in history:
        if record["offset"] >= length:
            break
        result.append(record | {"count": min(record["count"], length - record["offset"])})
    return validate_decode_history(result, length)


def validate_native_generation(
    native,
    *,
    prompt_sha256,
    prompt_length,
    token_count,
    tokenizer_metadata_sha256,
    chat_template_sha256,
):
    """Validate the producer's generated chain without relabeling it as replay."""
    prompt = native.get("prompt", {})
    generation = native.get("generation", {})
    mode = prompt.get("template_mode")
    if mode == "native_chat":
        messages = prompt.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(m, dict)
                or set(m) != {"role", "content"}
                or any(not isinstance(v, str) for v in m.values())
                for m in messages
            )
        ):
            raise ValueError("native generated original messages missing")
        source = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        wanted_prompt = {"messages", "template_mode", "max_new_tokens", "max_prompt_tokens"}
    elif mode == "raw_text" and isinstance(prompt.get("text"), str) and prompt["text"]:
        source = prompt["text"]
        wanted_prompt = {"text", "template_mode", "max_new_tokens", "max_prompt_tokens"}
    else:
        raise ValueError("native generated prompt/template mode differs")
    maximum = prompt.get("max_new_tokens")
    boundary = native.get("prompt_length")
    rendered, template = native.get("rendered_prompt"), native.get("chat_template")
    if (
        native.get("prefix_contract") != GENERATED_PREFIX_CONTRACT
        or native.get("prefix_freshness") != "native_generated_chain"
        or set(prompt) != wanted_prompt
        or type(boundary) is not int
        or boundary != prompt_length
        or not 1 <= boundary <= token_count
        or type(maximum) is not int
        or maximum < 0
        or type(prompt.get("max_prompt_tokens")) is not int
        or not 1 <= boundary <= prompt["max_prompt_tokens"]
        or hashlib.sha256(source.encode()).hexdigest() != prompt_sha256
        or native.get("prompt_source_sha256") != prompt_sha256
        or not isinstance(rendered, str)
        or not rendered
        or native.get("rendered_prompt_sha256") != hashlib.sha256(rendered.encode()).hexdigest()
        or not isinstance(template, str)
        or native.get("chat_template_sha256") != hashlib.sha256(template.encode()).hexdigest()
        or not _hash(chat_template_sha256)
        or (mode == "native_chat" and native.get("chat_template_sha256") != chat_template_sha256)
        or (mode == "raw_text" and (template != "" or rendered != source))
        or native.get("target_chat_template_sha256") != chat_template_sha256
        or not _hash(tokenizer_metadata_sha256)
        or native.get("tokenizer_metadata_sha256") != tokenizer_metadata_sha256
        or native.get("tokenizer")
        != {"implementation": "llama_tokenize", "add_special": True, "parse_special": True}
        or generation.get("mode") != "native_target_greedy"
        or generation.get("max_new_tokens") != maximum
        or type(generation.get("generated_tokens")) is not int
        or generation["generated_tokens"] != token_count - boundary
        or not 0 <= generation["generated_tokens"] <= maximum
        or generation.get("stop_eog") is not True
        or generation.get("termination") not in ("max_new_tokens", "eog")
        or (
            generation["termination"] == "max_new_tokens"
            and generation["generated_tokens"] != maximum
        )
        or (generation["termination"] == "eog" and generation["generated_tokens"] < 1)
    ):
        raise ValueError(
            "native generated prompt/tokenizer/template/greedy history contract differs"
        )
    history = validate_decode_history(native.get("decode_history"), token_count)
    for chunk in history:
        if chunk["phase"] != (
            "prefill" if chunk["offset"] < boundary else "target_only_greedy"
        ) or (chunk["offset"] < boundary and chunk["offset"] + chunk["count"] > boundary):
            raise ValueError("native generated decode history crosses exact prompt boundary")


def _fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def validate_native_receipt(native, chain, tokens, producer, vocab_size, target_width):
    """Join original target-only producer bytes, exact tokens, and original TRAIN."""
    ancestry = {
        "prompt_id": chain["prompt_id"],
        "prompt_sha256": chain["prompt_sha256"],
        "domain": chain["domain"],
        "source_split": "TRAIN",
        "prompt_length": chain["prompt_length"],
    }
    if (
        native.get("schema") != "block_native_teacher_request_v1"
        or native.get("complete") is not True
        or native.get("optimizer_updates") != 0
        or native.get("teacher_context_reset_between_requests") is not True
        or native.get("kv_type") != "F16"
        or native.get("target_precision") != "F16"
        or native.get("tap_ids") != list(TAPS)
        or native.get("tokens") != tokens.tolist()
        or native.get("target_sha256") != producer["target_sha256"]
        or native.get("producer_binary_sha256") != producer["binary_sha256"]
        or native.get("producer_source_revision") != producer["native_revision"]
        or native.get("chain_ancestry") != ancestry
        or native.get("hardware") != json.loads(producer["hardware"])
        or native.get("features_shape") != [len(tokens), 5, target_width]
    ):
        raise ValueError("native producer receipt differs from exact target/prefix/TRAIN ancestry")
    if not _hash(native.get("client_source_sha256")) or not native.get("producer_host"):
        raise ValueError("native receipt missing client/host provenance")
    if "generation_identity" in producer:
        pins = producer["generation_identity"]
        if native.get("client_source_sha256") != pins["teacher_client_sha256"]:
            raise ValueError("generated native client differs from pinned runtime")
        validate_native_generation(
            native,
            prompt_sha256=chain["prompt_sha256"],
            prompt_length=chain["prompt_length"],
            token_count=len(tokens),
            tokenizer_metadata_sha256=pins["tokenizer_metadata_sha256"],
            chat_template_sha256=pins["chat_template_sha256"],
        )
    elif native.get("prefix_contract") != REPLAY_PREFIX_CONTRACT or any(
        k in native for k in ("generation", "prompt", "prompt_source_sha256")
    ):
        raise ValueError("native replay prefix contract differs")
    validate_decode_history(native.get("decode_history"), len(tokens))
    files = native.get("files", {})
    for name in ("features", "logits"):
        array = chain[name]
        if array is None:
            continue  # hard CE may deliberately omit an otherwise retained logit file
        record = files.get(name, {})
        shape = [len(tokens), 5, target_width] if name == "features" else [len(tokens), vocab_size]
        if (
            record.get("sha256") != array["sha256"]
            or record.get("shape") != shape
            or record.get("dtype") != "float32"
        ):
            raise ValueError("native producer feature/full-vocabulary logit artifact differs")
        if name == "logits" and (
            native.get("logits_mode") != "all" or native.get("logits_shape") != shape
        ):
            raise ValueError("last-only teacher cannot supply a whole chain soft objective")


def import_capture_plan(
    plan_path, *, expected_sha256, output_dir, max_capture_bytes=1024**3, admission_output=None
):
    """Materialize a manifest from pinned original native receipts, without tensor copies.

    The plan is admitted by its external hash and contains an original TRAIN
    inventory and runtime manifest. It selects prompt-disjoint roles explicitly;
    it does not invent prompt membership or capture teachers. Raw producer files
    are retained read-only and only the small token arrays are materialized.
    """
    plan_path = Path(plan_path).resolve()
    output = Path(output_dir).resolve()
    if not _hash(expected_sha256) or file_sha256(plan_path) != expected_sha256:
        raise ValueError("capture plan differs from external admission pin")
    if output.exists():
        raise ValueError("refuse to overwrite capture preparation history")
    _positive(max_capture_bytes, "max_capture_bytes")
    plan = json.loads(plan_path.read_text())
    keys = {
        "schema",
        "family",
        "vocab_size",
        "target_width",
        "mask_token_id",
        "train_inventory",
        "runtime",
        "chains",
    }
    if set(plan) != keys or plan["schema"] != "block_capture_plan_v1":
        raise ValueError("unsupported native capture import plan")

    def artifact(record, directory, *, check_hash=True):
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256"}
            or not _hash(record["sha256"])
        ):
            raise ValueError("externally admitted path/SHA256 required")
        path = (directory / record["path"]).resolve()
        if not path.is_file() or (check_hash and file_sha256(path) != record["sha256"]):
            raise ValueError("capture plan artifact missing or SHA256 differs")
        return path

    inventory_path = artifact(plan["train_inventory"], plan_path.parent)
    runtime_path = artifact(plan["runtime"], plan_path.parent)
    runtime = json.loads(runtime_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    chains, producer, receipt_chains, total_bytes = [], None, {}, 0
    if not isinstance(plan["chains"], list) or not plan["chains"]:
        raise ValueError("empty native capture import plan")
    output.mkdir(parents=True)
    for index, source in enumerate(plan["chains"]):
        expected = {
            "chain_id",
            "prompt_id",
            "prompt_sha256",
            "domain",
            "split",
            "prompt_length",
            "anchors",
            "native_receipt",
            "include_logits",
        }
        if set(source) != expected or type(source["include_logits"]) is not bool:
            raise ValueError("capture chain plan fields differ")
        if inventory.get("prompts", {}).get(source["prompt_id"]) != {
            "sha256": source["prompt_sha256"],
            "domain": source["domain"],
            "split": "TRAIN",
        }:
            raise ValueError("capture chain not in original authenticated TRAIN inventory")
        native_path = artifact(source["native_receipt"], plan_path.parent)
        native = json.loads(native_path.read_text())
        current = {
            "kind": "native_target_only",
            "native_revision": native.get("producer_source_revision"),
            "binary_sha256": native.get("producer_binary_sha256"),
            "target_sha256": native.get("target_sha256"),
            "target_precision": "F16",
            "runtime_sha256": file_sha256(runtime_path),
            "hardware": json.dumps(native.get("hardware"), sort_keys=True),
        }
        if native.get("prefix_contract") == GENERATED_PREFIX_CONTRACT:
            expected_runtime = {
                "schema": "nine_model_train_capture_runtime_v1",
                "binary_sha256": current["binary_sha256"],
                "target_sha256": current["target_sha256"],
                "native_source_revision": current["native_revision"],
                "teacher_client_sha256": native.get("client_source_sha256"),
                "tokenizer_metadata_sha256": native.get("tokenizer_metadata_sha256"),
                "chat_template_sha256": native.get("target_chat_template_sha256"),
            }
            if runtime != expected_runtime:
                raise ValueError("generated receipt differs from pinned runtime identity")
            current["generation_identity"] = {
                key: runtime[key]
                for key in (
                    "teacher_client_sha256",
                    "tokenizer_metadata_sha256",
                    "chat_template_sha256",
                )
            }
        if producer is not None and current != producer:
            raise ValueError("mixed target/source/hardware/runtime producer chains")
        producer = current
        tokens = np.asarray(native.get("tokens"), dtype=np.int64)
        if tokens.ndim != 1 or not len(tokens):
            raise ValueError("native receipt missing exact prefix tokens")
        chain = {k: v for k, v in source.items() if k != "include_logits"}
        for name in ("features", "logits"):
            if name == "logits" and not source["include_logits"]:
                chain[name] = None
                continue
            record = native.get("files", {}).get(name)
            if not isinstance(record, dict) or set(record) != {"path", "sha256", "shape", "dtype"}:
                raise ValueError("native raw feature/logit artifact descriptor missing")
            path = artifact(
                {"path": record["path"], "sha256": record["sha256"]},
                native_path.parent,
                check_hash=False,
            )
            total_bytes += path.stat().st_size
            if total_bytes > max_capture_bytes:
                raise MemoryError("native retained capture bytes exceed import bound")
            chain[name] = record | {"path": str(path)}
        chain["native_receipt"] = {"path": str(native_path), "sha256": file_sha256(native_path)}
        validate_native_receipt(
            native, chain, tokens, producer, plan["vocab_size"], plan["target_width"]
        )
        token_path = output / f"tokens-{index:06d}.npy"
        np.save(token_path, tokens)
        chain["tokens"] = {"path": str(token_path), "sha256": file_sha256(token_path)}
        chains.append(chain)
        cid = chain["chain_id"]
        if cid in receipt_chains:
            raise ValueError("duplicate capture chain")
        receipt_chains[cid] = {
            "tokens_sha256": chain["tokens"]["sha256"],
            "features_sha256": chain["features"]["sha256"],
            "logits_sha256": None if chain["logits"] is None else chain["logits"]["sha256"],
            "prompt_id": chain["prompt_id"],
            "prompt_sha256": chain["prompt_sha256"],
            "prompt_length": chain["prompt_length"],
            "native_receipt_sha256": chain["native_receipt"]["sha256"],
        }
    receipt = {
        "schema": "block_target_capture_receipt_v1",
        "producer": producer,
        "train_inventory_sha256": file_sha256(inventory_path),
        "taps": list(TAPS),
        "tap_semantics": "native_layer_input_f32",
        "vocab_size": plan["vocab_size"],
        "capture_mode": "target_only_autoregressive_train",
        "chains": receipt_chains,
        "capture_plan_sha256": expected_sha256,
        "retained_capture_bytes": total_bytes,
    }
    receipt_path = output / "producer-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    manifest = {
        "schema": SCHEMA,
        "family": plan["family"],
        "vocab_size": plan["vocab_size"],
        "target_width": plan["target_width"],
        "mask_token_id": plan["mask_token_id"],
        "taps": list(TAPS),
        "tap_semantics": "native_layer_input_f32",
        "layout": "author_anchor_first",
        "producer": producer
        | {"receipt": {"path": str(receipt_path), "sha256": file_sha256(receipt_path)}},
        "train_inventory": {"path": str(inventory_path), "sha256": file_sha256(inventory_path)},
        "chains": chains,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Re-read the exact completed bytes and reject all shape/split/prefix errors.
    audited = BlockDataset(manifest_path, expected_sha256=file_sha256(manifest_path))
    audited.write_admission(admission_output or output / "completed-admission.json")
    return manifest_path


@dataclass(frozen=True)
class BlockBatch:
    family: str
    chain_id: str
    block_index: int
    context_features: np.ndarray
    prefix_tokens: tuple[int, ...]
    input_tokens: np.ndarray
    positions: np.ndarray
    labels: np.ndarray
    loss_mask: np.ndarray
    attention_allowed: np.ndarray
    teacher_logits: np.ndarray | None
    teacher_prefix_sha256: tuple[str | None, ...]
    predecessor_ids: np.ndarray
    target_sha256: str
    dataset_sha256: str


@dataclass(frozen=True)
class BlockCursor:
    dataset_sha256: str
    order_sha256: str
    split: str
    seed: int
    epoch: int = 0
    chain_offset: int = 0
    block_offset: int = 0

    def payload(self):
        return asdict(self)


class BlockDataset:
    """Fail-closed manifest/producer joins and bounded per-block teacher reads.

    The external manifest pin is an owner admission, not self-authentication.
    The inventory and independently pinned producer receipt bind every chain's
    original TRAIN membership and actual native target/hardware ancestry.
    """

    def __init__(
        self,
        manifest_path,
        *,
        expected_sha256: str,
        allow_synthetic=False,
        max_teacher_bytes=64 * 1024 * 1024,
        verify_artifacts=True,
        admission_path=None,
        admission_sha256=None,
    ):
        self.path = Path(manifest_path).resolve()
        if not _hash(expected_sha256) or file_sha256(self.path) != expected_sha256:
            raise ValueError("manifest differs from external SHA256 pin")
        self.sha256 = expected_sha256
        self._fingerprints = {}
        self._admitted = {}
        self._fully_audited = admission_path is None
        if admission_path is not None:
            admission_path = Path(admission_path)
            if not _hash(admission_sha256) or file_sha256(admission_path) != admission_sha256:
                raise ValueError("completed data admission differs from external pin")
            admission = json.loads(admission_path.read_text())
            if (
                set(admission) != {"schema", "manifest_sha256", "source_sha256", "artifacts"}
                or admission["schema"] != "block_data_completed_admission_v1"
                or admission["manifest_sha256"] != self.sha256
                or admission["source_sha256"] != file_sha256(Path(__file__))
            ):
                raise ValueError("completed data admission source/manifest schema differs")
            for record in admission["artifacts"]:
                if set(record) != {
                    "path",
                    "bytes",
                    "mtime_ns",
                    "ctime_ns",
                    "device",
                    "inode",
                    "sha256",
                }:
                    raise ValueError("completed admission lacks exact host artifact identity")
                if (
                    not _hash(record["sha256"])
                    or _fingerprint(record["path"])
                    != {k: v for k, v in record.items() if k != "sha256"}
                    or record["path"] in self._admitted
                ):
                    raise ValueError("completed admission artifact path/stat identity differs")
                self._admitted[record["path"]] = record
        if verify_artifacts is not True and not self._admitted:
            raise ValueError(
                "unchecked artifacts cannot admit data; pinned completed admission required"
            )
        self.manifest = m = json.loads(self.path.read_text())
        expected = {
            "schema",
            "family",
            "vocab_size",
            "target_width",
            "mask_token_id",
            "taps",
            "tap_semantics",
            "layout",
            "producer",
            "train_inventory",
            "chains",
        }
        if set(m) != expected or m["schema"] != SCHEMA or m["family"] not in ("dspark", "dflash"):
            raise ValueError("unsupported block manifest schema/family")
        if m["layout"] != "author_anchor_first":
            raise ValueError("only released author anchor-first seven-prediction layout supported")
        if m["taps"] != list(TAPS) or m["tap_semantics"] != "native_layer_input_f32":
            raise ValueError("block data requires ordered native layer-input taps")
        self.vocab_size = _positive(m["vocab_size"], "vocab_size")
        self.target_width = _positive(m["target_width"], "target_width")
        if type(m["mask_token_id"]) is not int or not 0 <= m["mask_token_id"] < self.vocab_size:
            raise ValueError("mask token outside full vocabulary")
        self.max_teacher_bytes = _positive(max_teacher_bytes, "max_teacher_bytes")
        producer = m["producer"]
        producer_keys = {
            "kind",
            "native_revision",
            "binary_sha256",
            "target_sha256",
            "target_precision",
            "runtime_sha256",
            "hardware",
            "receipt",
        }
        if set(producer) not in (producer_keys, producer_keys | {"generation_identity"}):
            raise ValueError("producer ancestry fields differ")
        if "generation_identity" in producer and (
            not isinstance(producer["generation_identity"], dict)
            or set(producer["generation_identity"])
            != {"teacher_client_sha256", "tokenizer_metadata_sha256", "chat_template_sha256"}
            or not all(_hash(v) for v in producer["generation_identity"].values())
        ):
            raise ValueError("generated producer identity differs")
        if producer["kind"] not in ("native_target_only", "synthetic_fixture"):
            raise ValueError("teacher producer must be target-only native capture")
        if producer["kind"] == "synthetic_fixture" and not allow_synthetic:
            raise ValueError("synthetic fixture cannot admit production data")
        if (
            not isinstance(producer["native_revision"], str)
            or len(producer["native_revision"]) != 40
            or any(c not in "0123456789abcdef" for c in producer["native_revision"])
        ):
            raise ValueError("exact native producer revision required")
        for key in ("binary_sha256", "target_sha256", "runtime_sha256"):
            if not _hash(producer[key]):
                raise ValueError(f"producer {key} required")
        if (
            producer["target_precision"] != "F16"
            or not isinstance(producer["hardware"], str)
            or not producer["hardware"]
        ):
            raise ValueError("actual producer hardware and frozen F16 target required")
        receipt_path = self._artifact(producer["receipt"], verify_artifacts=True)
        receipt = json.loads(receipt_path.read_text())
        inventory_path = self._artifact(m["train_inventory"], verify_artifacts=True)
        inventory = json.loads(inventory_path.read_text())
        if (
            set(inventory) != {"schema", "prompts"}
            or inventory["schema"] != "block_train_inventory_v1"
        ):
            raise ValueError("original TRAIN inventory schema differs")
        if (
            receipt.get("schema") != "block_target_capture_receipt_v1"
            or receipt.get("producer") != {k: v for k, v in producer.items() if k != "receipt"}
            or receipt.get("train_inventory_sha256") != m["train_inventory"]["sha256"]
            or receipt.get("taps") != list(TAPS)
            or receipt.get("tap_semantics") != m["tap_semantics"]
            or receipt.get("vocab_size") != self.vocab_size
            or receipt.get("capture_mode") != "target_only_autoregressive_train"
        ):
            raise ValueError("producer receipt does not join target/TRAIN/tap/vocabulary contract")
        self._arrays = {}
        self.chains = {}
        self.native_receipts = {}
        prompt_splits = {}
        seen_prompt_hashes = {}
        for chain in m["chains"]:
            keys = {
                "chain_id",
                "prompt_id",
                "prompt_sha256",
                "domain",
                "split",
                "prompt_length",
                "tokens",
                "features",
                "logits",
                "anchors",
                "native_receipt",
            }
            if (
                set(chain) != keys
                or not isinstance(chain["chain_id"], str)
                or not chain["chain_id"]
            ):
                raise ValueError("chain fields differ")
            cid = chain["chain_id"]
            if cid in self.chains or chain["split"] not in SPLITS or chain["domain"] not in DOMAINS:
                raise ValueError("duplicate chain or unsupported domain/split")
            record = inventory["prompts"].get(chain["prompt_id"])
            if record != {
                "sha256": chain["prompt_sha256"],
                "domain": chain["domain"],
                "split": "TRAIN",
            }:
                raise ValueError("chain differs from authenticated original TRAIN inventory")
            if not _hash(chain["prompt_sha256"]):
                raise ValueError("prompt content hash required")
            for map_, key in (
                (prompt_splits, chain["prompt_id"]),
                (seen_prompt_hashes, chain["prompt_sha256"]),
            ):
                if key in map_ and map_[key] != chain["split"]:
                    raise ValueError("calibration splits must be prompt/content disjoint")
                map_[key] = chain["split"]
            tokens = self._array(chain["tokens"], np.dtype("int64"), verify_artifacts)
            features = self._array(chain["features"], np.dtype("float32"), verify_artifacts)
            logits = (
                None
                if chain["logits"] is None
                else self._array(chain["logits"], np.dtype("float32"), verify_artifacts)
            )
            if (
                tokens.ndim != 1
                or len(tokens) < 2
                or np.any(tokens < 0)
                or np.any(tokens >= self.vocab_size)
            ):
                raise ValueError("target tokens outside full vocabulary")
            if features.shape != (len(tokens), 5, self.target_width):
                raise ValueError("five-tap feature shape differs from captured token sequence")
            if logits is not None and logits.shape != (len(tokens), self.vocab_size):
                raise ValueError("exact full-vocabulary teacher logits required")
            if not 1 <= _positive(chain["prompt_length"], "prompt_length") < len(tokens):
                raise ValueError("prompt boundary outside captured chain")
            anchors = chain["anchors"]
            if (
                not isinstance(anchors, list)
                or not anchors
                or any(type(a) is not int for a in anchors)
                or anchors != sorted(set(anchors))
                or anchors[0] < chain["prompt_length"] - 1
                or anchors[-1] + 7 >= len(tokens)
            ):
                raise ValueError("chronological anchors lack complete teacher horizon")
            if receipt.get("chains", {}).get(cid) != {
                "tokens_sha256": chain["tokens"]["sha256"],
                "features_sha256": chain["features"]["sha256"],
                "logits_sha256": None if chain["logits"] is None else chain["logits"]["sha256"],
                "prompt_id": chain["prompt_id"],
                "prompt_sha256": chain["prompt_sha256"],
                "prompt_length": chain["prompt_length"],
                "native_receipt_sha256": None
                if chain["native_receipt"] is None
                else chain["native_receipt"]["sha256"],
            }:
                raise ValueError("chain artifacts/prompt boundary differ from producer receipt")
            if producer["kind"] == "native_target_only":
                if chain["native_receipt"] is None:
                    raise ValueError("production chain requires original native producer receipt")
                native = json.loads(
                    self._artifact(chain["native_receipt"], verify_artifacts=True).read_text()
                )
                validate_native_receipt(
                    native, chain, tokens, producer, self.vocab_size, self.target_width
                )
                self.native_receipts[cid] = native
            self.chains[cid] = chain
            self._arrays[cid] = (tokens, features, logits)
        if not self.chains:
            raise ValueError("empty block capture")

    def _artifact(self, record, *, verify_artifacts):
        if (
            not isinstance(record, dict)
            or set(record) not in ({"path", "sha256"}, {"path", "sha256", "shape", "dtype"})
            or not _hash(record["sha256"])
        ):
            raise ValueError("artifact path/SHA256 required")
        path = (self.path.parent / record["path"]).resolve()
        fingerprint = _fingerprint(path) if path.is_file() else None
        admitted = self._admitted.get(str(path))
        if self._admitted and (
            fingerprint is None
            or admitted is None
            or admitted != fingerprint | {"sha256": record["sha256"]}
        ):
            raise ValueError("completed admission does not bind this exact artifact")
        if not path.is_file() or (not admitted and file_sha256(path) != record["sha256"]):
            raise ValueError("capture artifact missing or SHA256 differs")
        self._fingerprints[str(path)] = fingerprint | {"sha256": record["sha256"]}
        return path

    def _array(self, record, dtype, verify_artifacts):
        path = self._artifact(record, verify_artifacts=verify_artifacts)
        if "shape" in record:
            shape = record["shape"]
            if (
                not isinstance(shape, list)
                or not shape
                or any(type(n) is not int or n < 1 for n in shape)
                or len(shape) > 3
                or record["dtype"] != "float32"
                or dtype != np.dtype("float32")
            ):
                raise ValueError("raw native array requires positive declared F32 shape")
            size = 4
            for n in shape:
                size *= n
            if path.stat().st_size != size:
                raise ValueError("raw capture file length differs from exact native shape")
            value = np.memmap(path, mode="r", dtype="<f4", shape=tuple(shape))
        else:
            value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.dtype != dtype or not value.flags.c_contiguous:
            raise ValueError("capture requires contiguous exact native dtype")
        # Check in bounded row slices rather than forming a full logits bool tensor.
        if self._fully_audited:
            for first in range(0, len(value), 16):
                if not np.isfinite(value[first : first + 16]).all():
                    raise ValueError("capture array contains nonfinite values")
        return value

    def load_block(self, chain_id: str, block_index: int, *, require_teacher=False) -> BlockBatch:
        chain = self.chains[chain_id]
        if type(block_index) is not int or not 0 <= block_index < len(chain["anchors"]):
            raise ValueError("block index outside chain")
        tokens, features, logits = self._arrays[chain_id]
        anchor = chain["anchors"][block_index]
        for name in ("tokens", "features", "logits"):
            record = chain[name]
            if record is not None:
                path = (self.path.parent / record["path"]).resolve()
                if _fingerprint(path) != {
                    k: v for k, v in self._fingerprints[str(path)].items() if k != "sha256"
                }:
                    raise ValueError("consumed capture artifact changed after admission")
        for first in range(0, anchor, 16):
            if not np.isfinite(features[first : first + 16]).all():
                raise ValueError("consumed native context features nonfinite")
        begin = 0  # both selected releases use the admitted author-layout path
        labels = np.full(7, -1, dtype=np.int64)
        labels[begin:] = tokens[anchor + 1 : anchor + 8 - begin]
        predecessors = np.full(7, -1, dtype=np.int64)
        predecessors[begin] = tokens[anchor]
        predecessors[begin + 1 :] = labels[begin:-1]
        teacher = None
        if require_teacher and logits is None:
            raise ValueError("exact full-vocabulary teacher logits absent")
        if logits is not None:
            if 7 * self.vocab_size * 4 > self.max_teacher_bytes:
                raise MemoryError("per-block full-vocabulary teacher exceeds declared bound")
            teacher = np.zeros((7, self.vocab_size), dtype=np.float32)
            teacher[begin:] = logits[anchor : anchor + 7 - begin]
            if not np.isfinite(teacher).all():
                raise ValueError("consumed full-vocabulary teacher nonfinite")
        prefix = tuple(int(t) for t in tokens[: anchor + 1])
        hashes = (None,) * begin + tuple(
            token_sha256(tokens[: anchor + i + 1]) for i in range(7 - begin)
        )
        noise = np.full(7, self.manifest["mask_token_id"], dtype=np.int64)
        noise[0] = tokens[anchor]
        return BlockBatch(
            self.manifest["family"],
            chain_id,
            block_index,
            features[:anchor],
            prefix,
            noise,
            np.arange(anchor, anchor + 7, dtype=np.int64),
            labels,
            np.arange(7) >= begin,
            np.ones((7, anchor + 7), dtype=bool),
            teacher,
            hashes,
            predecessors,
            self.manifest["producer"]["target_sha256"],
            self.sha256,
        )

    def order(self, split, seed, epoch=0):
        if split not in SPLITS or type(seed) is not int or type(epoch) is not int or epoch < 0:
            raise ValueError("invalid chain-order split/seed/epoch")
        buckets = {domain: [] for domain in DOMAINS}
        for cid, chain in self.chains.items():
            if chain["split"] == split:
                buckets[chain["domain"]].append(cid)
        rng = random.Random(identity({"seed": seed, "epoch": epoch, "dataset": self.sha256}))
        for ids in buckets.values():
            ids.sort()
            rng.shuffle(ids)
        result = []
        while any(buckets.values()):
            for domain in DOMAINS:
                if buckets[domain]:
                    result.append(buckets[domain].pop())
        if not result:
            raise ValueError("requested split has no chains")
        return tuple(result)

    def cursor(self, *, split="train", seed=0, epoch=0):
        order = self.order(split, seed, epoch)
        return BlockCursor(self.sha256, identity(order), split, seed, epoch)

    def next_block(self, cursor: BlockCursor, *, require_teacher=False):
        order = self.order(cursor.split, cursor.seed, cursor.epoch)
        if cursor.dataset_sha256 != self.sha256 or cursor.order_sha256 != identity(order):
            raise ValueError("cursor dataset/whole-chain order identity differs")
        if (
            type(cursor.chain_offset) is not int
            or not 0 <= cursor.chain_offset < len(order)
            or type(cursor.block_offset) is not int
            or cursor.block_offset < 0
        ):
            raise ValueError("cursor position invalid")
        cid = order[cursor.chain_offset]
        batch = self.load_block(cid, cursor.block_offset, require_teacher=require_teacher)
        chain_offset, block_offset = cursor.chain_offset, cursor.block_offset + 1
        if block_offset == len(self.chains[cid]["anchors"]):
            chain_offset, block_offset = chain_offset + 1, 0
        if chain_offset == len(order):
            next_cursor = self.cursor(split=cursor.split, seed=cursor.seed, epoch=cursor.epoch + 1)
        else:
            next_cursor = BlockCursor(
                self.sha256,
                cursor.order_sha256,
                cursor.split,
                cursor.seed,
                cursor.epoch,
                chain_offset,
                block_offset,
            )
        return batch, next_cursor

    def write_admission(self, path):
        """Publish only after an original full integrity/finite/schema audit.

        This receipt is reusable on the same host/filesystem only. Copying data
        to another host needs one new audit, never six repeated process audits.
        Caller records its external SHA pin before a prepared launcher reuses it.
        """
        path = Path(path)
        if path.exists() or not self._fully_audited:
            raise ValueError("admission requires fresh complete audit and a new output path")
        for record in self._fingerprints.values():
            if _fingerprint(record["path"]) != {k: v for k, v in record.items() if k != "sha256"}:
                raise ValueError("capture artifact changed during audit")
        receipt = {
            "schema": "block_data_completed_admission_v1",
            "manifest_sha256": self.sha256,
            "source_sha256": file_sha256(Path(__file__)),
            "artifacts": [self._fingerprints[key] for key in sorted(self._fingerprints)],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        return file_sha256(path)

    def storage_summary(self):
        return {
            "chains": len(self.chains),
            "blocks": sum(len(c["anchors"]) for c in self.chains.values()),
            "feature_bytes": sum(f.nbytes for _, f, _ in self._arrays.values()),
            "teacher_bytes": sum(
                0 if logits is None else logits.nbytes for _, _, logits in self._arrays.values()
            ),
            "max_block_teacher_bytes": 7 * self.vocab_size * 4,
            "producer_hardware": self.manifest["producer"]["hardware"],
            "portability_status": "requires_fresh_target_device_trajectory_check",
        }
