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
    ):
        self.path = Path(manifest_path).resolve()
        if not _hash(expected_sha256) or file_sha256(self.path) != expected_sha256:
            raise ValueError("manifest differs from external SHA256 pin")
        self.sha256 = expected_sha256
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
        if set(producer) != producer_keys:
            raise ValueError("producer ancestry fields differ")
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
            }:
                raise ValueError("chain artifacts/prompt boundary differ from producer receipt")
            self.chains[cid] = chain
            self._arrays[cid] = (tokens, features, logits)
        if not self.chains:
            raise ValueError("empty block capture")

    def _artifact(self, record, *, verify_artifacts):
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256"}
            or not _hash(record["sha256"])
        ):
            raise ValueError("artifact path/SHA256 required")
        path = (self.path.parent / record["path"]).resolve()
        if not path.is_file() or (verify_artifacts and file_sha256(path) != record["sha256"]):
            raise ValueError("capture artifact missing or SHA256 differs")
        return path

    def _array(self, record, dtype, verify_artifacts):
        value = np.load(
            self._artifact(record, verify_artifacts=verify_artifacts),
            mmap_mode="r",
            allow_pickle=False,
        )
        if value.dtype != dtype or not value.flags.c_contiguous:
            raise ValueError("capture requires contiguous exact native dtype")
        # Check in bounded row slices rather than forming a full logits bool tensor.
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
