"""Immutable logical corpus over an explicitly owned, bounded physical shard cache.

Capture/reconstruction is a controller phase: a provider never launches a teacher
while its student owns CUDA. Missing shards raise a resumable boundary before any
optimizer data cursor is committed. Physical admission fingerprints are fresh on
restore; original tensor/source receipt identities are retained in durable seals.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .block_data import DOMAINS, SPLITS, BlockDataset, file_sha256, identity
from .nine_model_pipeline import atomic_json

GIB = 1024**3
PLAN_SCHEMA = "block_logical_shard_plan_v1"


def checked(locator):
    path = Path(locator["path"])
    if file_sha256(path) != locator["sha256"]:
        raise ValueError("shard artifact differs from external SHA256 pin")
    return path


def pin(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(Path(path))}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def tree_bytes(root):
    """Count actual unique inode payloads, including retained failed staging."""
    total, seen = 0, set()
    for path in Path(root).rglob("*"):
        _require(not path.is_symlink(), "owned storage cannot contain symlinks")
        if path.is_file():
            stat = path.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode not in seen:
                total += stat.st_size
                seen.add(inode)
    return total


def deployed_source_pins():
    root = Path(__file__).resolve().parents[2]
    return {
        "module:" + name: file_sha256(root / name)
        for name in (
            "src/w1a1_eagle/block_shard_lifecycle.py",
            "src/w1a1_eagle/nine_model_pipeline.py",
            "src/w1a1_eagle/block_data.py",
        )
    }


def validate_deployed_source(plan):
    for name, digest in deployed_source_pins().items():
        _require(
            plan.value["source"].get(name) == digest,
            "frozen deployed provider/source admission changed: " + name,
        )


def freeze_plan(selector, chains, *, source, geometry, generation_variant, shard_bytes=8 * GIB):
    """Freeze whole-group, domain-balanced membership from exact native metadata.

    `chains` supplies source policy and a conservative publication byte bound.
    Actual token boundaries, anchors and tensor hashes are sealed at first capture
    rather than guessed before capture. No group can be split.
    """
    selection = json.loads(checked(selector).read_text())["selection"]
    selected = {row["prompt_id"]: row for row in selection}
    _require(len(selected) == len(selection) == len(chains), "selector/chain inventory differs")
    records, groups = {}, {}
    for chain in chains:
        row = selected[chain["prompt_id"]]
        cid = chain["chain_id"]
        _require(cid not in records, "duplicate logical chain")
        record = {
            **chain,
            "split": row["split"],
            "domain": row["domain"],
            "group_id": row["group_id"],
            "prompt_sha256": row["content_sha256"],
        }
        records[cid] = record
        groups.setdefault(record["group_id"], []).append(cid)
    rng = random.Random(8101)
    buckets = {domain: [] for domain in DOMAINS}
    for group, ids in sorted(groups.items()):
        # Shared cross-domain groups remain indivisible; count them by their
        # smallest domain solely for deterministic balancing order.
        domain = min(records[cid]["domain"] for cid in ids)
        buckets[domain].append(group)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    ordered = []
    while any(buckets.values()):
        for domain in DOMAINS:
            if buckets[domain]:
                ordered.append(buckets[domain].pop())
    shards, current, size = [], [], 0
    for group in ordered:
        ids = sorted(groups[group])
        group_bytes = sum(records[cid]["capture_bytes_bound"] for cid in ids)
        _require(0 < group_bytes <= shard_bytes, "whole group exceeds shard publication byte bound")
        if current and size + group_bytes > shard_bytes:
            shards.append(current)
            current, size = [], 0
        current.extend(ids)
        size += group_bytes
    if current:
        shards.append(current)
    value = {
        "schema": PLAN_SCHEMA,
        "selector": selector,
        "source": {**source, **deployed_source_pins()},
        "geometry": geometry,
        "generation_variant": generation_variant,
        "seed": 8101,
        "shard_bytes": shard_bytes,
        "cache_slots": 2,
        "staging_slots": 1,
        "chains": records,
        "shards": {f"shard-{i:05d}": ids for i, ids in enumerate(shards)},
    }
    FrozenShardPlan(value)  # Validate full production role/group/source contract.
    return value


class FrozenShardPlan:
    def __init__(self, value, *, require_full_pool=True):
        self.production = require_full_pool
        self.value = copy.deepcopy(value)
        self.sha256 = identity(self.value)
        self.chains = self.value["chains"]
        self.shards = self.value["shards"]
        _require(value.get("schema") == PLAN_SCHEMA, "logical shard plan schema differs")
        _require(value.get("seed") == 8101, "frozen shard seed must be 8101")
        _require(
            value.get("cache_slots") == 2 and value.get("staging_slots") == 1,
            "two cached plus one staging shard required",
        )
        _require(
            type(value["shard_bytes"]) is int and 0 < value["shard_bytes"] <= 8 * GIB,
            "shard byte bound exceeds 8 GiB",
        )
        _require(
            value["generation_variant"]["kind"] in {"verified_committed_tokens", "fresh_greedy"},
            "explicit unmixed generation variant required",
        )
        _require(
            _hash(value["generation_variant"].get("receipt_sha256")),
            "generation ancestry/fresh variant freeze receipt required",
        )
        _require(
            value["source"] and all(_hash(v) for v in value["source"].values()),
            "immutable source SHA pins required",
        )
        roles = {split: 0 for split in SPLITS}
        domains = {split: {domain: 0 for domain in DOMAINS} for split in SPLITS}
        groups, prompts, content_roles = {}, set(), {}
        self.chain_shards = {}
        for shard, ids in self.shards.items():
            _require(ids and len(set(ids)) == len(ids), "empty/duplicate shard membership")
            _require(
                sum(self.chains[c]["capture_bytes_bound"] for c in ids) <= value["shard_bytes"],
                "shard exceeds frozen byte bound",
            )
            for cid in ids:
                _require(cid not in self.chain_shards, "logical chain belongs to multiple shards")
                self.chain_shards[cid] = shard
        _require(set(self.chain_shards) == set(self.chains), "shards omit logical chains")
        for cid, chain in self.chains.items():
            split, domain, group = chain["split"], chain["domain"], chain["group_id"]
            _require(
                cid == chain["chain_id"] and split in SPLITS and domain in DOMAINS and group,
                "logical chain role/domain/group differs",
            )
            _require(
                chain["prompt_id"] not in prompts and _hash(chain["prompt_sha256"]),
                "duplicate prompt or missing original prompt hash",
            )
            prompts.add(chain["prompt_id"])
            roles[split] += 1
            domains[split][domain] += 1
            membership = (split, self.chain_shards[cid])
            _require(
                groups.setdefault(group, membership) == membership,
                "canonical group crosses role or physical shard",
            )
            _require(
                content_roles.setdefault(chain["prompt_sha256"], split) == split,
                "prompt content crosses calibration roles",
            )
            _require(
                type(chain["capture_bytes_bound"]) is int and chain["capture_bytes_bound"] > 0,
                "actual capture publication byte bound required",
            )
        if require_full_pool:
            selector = json.loads(checked(value["selector"]).read_text())["selection"]
            selected = {r["prompt_id"]: r for r in selector}
            _require(
                len(selector) == len(selected) == len(self.chains), "full selector identity differs"
            )
            for chain in self.chains.values():
                row = selected[chain["prompt_id"]]
                _require(
                    all(
                        chain[key] == row[source_key]
                        for key, source_key in (
                            ("split", "split"),
                            ("domain", "domain"),
                            ("group_id", "group_id"),
                            ("prompt_sha256", "content_sha256"),
                        )
                    ),
                    "chain differs from original role selector",
                )
            _require(
                roles == {"train": 9856, "calibration_fit": 96, "calibration_validation": 48},
                "all 10000 source prompt roles must remain frozen",
            )
            _require(
                domains
                == {
                    "train": {"prose": 3286, "code": 3285, "reasoning": 3285},
                    "calibration_fit": dict.fromkeys(DOMAINS, 32),
                    "calibration_validation": dict.fromkeys(DOMAINS, 16),
                },
                "full-pool domain role counts differ",
            )
        self.role_counts = roles
        self.domain_counts = domains

    @classmethod
    def load(cls, locator, **kwargs):
        return cls(json.loads(checked(locator).read_text()), **kwargs)

    def order(self, split, seed=8101, epoch=0):
        _require(
            split in SPLITS and seed == 8101 and type(epoch) is int and epoch >= 0,
            "frozen logical split/seed/epoch differs",
        )
        rng = random.Random(identity({"plan": self.sha256, "seed": seed, "epoch": epoch}))
        shards = sorted(self.shards)
        rng.shuffle(shards)
        result = []
        for shard in shards:
            ids = sorted(cid for cid in self.shards[shard] if self.chains[cid]["split"] == split)
            rng.shuffle(ids)
            result.extend(ids)
        _require(result, "split has no actual loss-bearing chains")
        return tuple(result)


def validate_capture_source(plan, shard, dataset):
    """Join actual producer/variant to the frozen native capture contract."""
    source, producer = plan.value["source"], dataset.manifest["producer"]
    _require(producer["target_sha256"] == source["target_sha256"], "shard target source changed")
    capture_plan = None
    if plan.value.get("capture_plans"):
        capture_plan = json.loads(checked(plan.value["capture_plans"][shard]["remote"]).read_text())
    _require(
        not plan.production or capture_plan is not None,
        "production requires frozen native capture contract",
    )
    binary = source.get("teacher_binary_sha256")
    client = source.get("module:scripts/capture_block_qat_teacher.py")
    revision, runtime = None, source.get("capture_runtime_sha256")
    if capture_plan:
        native = capture_plan["native"]
        binary, client = native["binary"]["sha256"], native["client_source"]["sha256"]
        revision, runtime = native["source_revision"], capture_plan["runtime"]["sha256"]
        _require(
            binary == source["teacher_binary_sha256"]
            and client == source["module:scripts/capture_block_qat_teacher.py"],
            "capture plan differs from frozen binary/client SHA",
        )
    for field, expected in (
        ("binary_sha256", binary),
        ("native_revision", revision),
        ("runtime_sha256", runtime),
    ):
        if expected is not None:
            _require(producer.get(field) == expected, "shard producer source changed: " + field)
    if plan.production:
        _require(
            producer.get("kind") == "native_target_only"
            and producer.get("target_precision") == "F16",
            "original native F16 target producer required",
        )
    receipts = getattr(dataset, "native_receipts", {})
    _require(
        not plan.production or set(receipts) == set(dataset.chains),
        "complete original native receipts required",
    )
    for cid, receipt in receipts.items():
        if client is not None:
            _require(
                receipt.get("client_source_sha256") == client,
                "native teacher client source changed",
            )
        if plan.production:
            buffers = receipt.get("executed_result_buffers", [])
            _require(
                buffers
                and all(b.startswith("CUDA") and b[4:].isdigit() for b in buffers)
                and receipt.get("gpu_layers") == capture_plan["native"]["gpu_layers"],
                "actual frozen CUDA target execution required",
            )
    if plan.value["generation_variant"]["kind"] == "fresh_greedy":
        from .block_data import joined_generation_source

        _require(
            set(receipts) == set(dataset.chains)
            and producer.get("generation_identity") is not None,
            "fresh variant requires actual original native generation ancestry",
        )
        pins = producer["generation_identity"]
        if capture_plan:
            _require(
                pins
                == {
                    "teacher_client_sha256": client,
                    "tokenizer_metadata_sha256": capture_plan["native"][
                        "tokenizer_metadata_sha256"
                    ],
                    "chat_template_sha256": capture_plan["native"]["chat_template_sha256"],
                },
                "frozen tokenizer/template/generation source changed",
            )
        for receipt in receipts.values():
            generation = (
                joined_generation_source(receipt) if "generation_source" in receipt else receipt
            )
            _require(
                generation.get("generation", {}).get("mode") == "native_target_greedy"
                and generation.get("prompt", {}).get("max_new_tokens") == 512
                and generation.get("generation", {}).get("stop_eog") is True,
                "fresh variant generation policy changed",
            )
            if client is not None:
                _require(
                    generation.get("client_source_sha256") == client,
                    "original generation client changed",
                )
    else:
        _require(
            all(
                "generation_source" not in receipt and "generation" not in receipt
                for receipt in receipts.values()
            ),
            "verified committed-token variant cannot silently become fresh generation",
        )


@dataclass(frozen=True)
class ShardCursor:
    dataset_sha256: str
    order_sha256: str
    split: str
    seed: int
    epoch: int = 0
    # Epoch-indexed progress supports an exact odd/group tail paired with a
    # distinct next-epoch group, without dropping or reordering chain anchors.
    progress: dict | None = None
    consumed_seals: dict | None = None

    def payload(self):
        return asdict(self)


class ShardRequired(RuntimeError):
    def __init__(self, shard_ids, reservation):
        self.shard_ids = tuple(shard_ids)
        self.reservation = reservation
        self.reserved_cursor = reservation
        super().__init__(
            "controller must release student and publish shards: " + ",".join(shard_ids)
        )


class ShardCache:
    """Owned files only. Tensor seals outlive eviction; no replay proof is inferred."""

    def __init__(self, plan, directory, *, dataset_factory=BlockDataset, replay_admission=None):
        self.plan, self.root, self.dataset_factory = (
            plan,
            Path(directory).resolve(),
            dataset_factory,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.replay_admission = replay_admission
        self.state_path = self.root / "cache-state.json"
        self.state = {
            "schema": "block_shard_cache_v1",
            "plan_sha256": plan.sha256,
            "resident": {},
            "seals": {},
            "owner": None,
        }
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            _require(
                self.state.get("schema") == "block_shard_cache_v1"
                and self.state["plan_sha256"] == plan.sha256,
                "cache logical identity differs",
            )
        self.datasets = {}

    def _save(self):
        atomic_json(self.state_path, self.state)

    def acquire(self, owner, phase):
        _require(
            phase in {"capture", "student", "restore"} and isinstance(owner, str) and owner,
            "explicit GPU phase owner required",
        )
        _require(
            self.state["owner"] in (None, {"owner": owner, "phase": phase}),
            "teacher/student GPU ownership overlaps",
        )
        self.state["owner"] = {"owner": owner, "phase": phase}
        self._save()

    def release(self, owner, *, released_proof):
        _require(
            self.state["owner"] and self.state["owner"]["owner"] == owner,
            "cannot release another GPU owner",
        )
        _require(
            released_proof.get("owned_process_groups_absent") is True
            and released_proof.get("owned_cuda_pids_absent") is True,
            "actual process/GPU release proof required",
        )
        self.state["owner"] = None
        self._save()

    def disk_bytes(self):
        return tree_bytes(self.root)

    def before_staging(self, shard_ids):
        _require(len(set(shard_ids)) == len(shard_ids) <= 2, "batch exceeds two active shards")
        _require(
            self.state["owner"] and self.state["owner"]["phase"] in {"capture", "restore"},
            "student must exit before teacher/reconstruction staging",
        )
        excess = max(0, len(set(self.state["resident"]) | set(shard_ids)) - 2)
        for shard in [s for s in self.state["resident"] if s not in shard_ids][:excess]:
            self.evict(shard)
        _require(
            self.disk_bytes() + self.plan.value["shard_bytes"]
            <= 3 * self.plan.value["shard_bytes"],
            "cached and failed staging bytes cannot admit another publication",
        )

    def _directory(self, path):
        path = Path(path).resolve()
        _require(
            path != self.root and path.is_relative_to(self.root),
            "shard directory outside owned cache",
        )
        _require(path.is_dir(), "shard directory absent")
        return path

    def publish(self, shard, manifest, admission, directory):
        _require(shard in self.plan.shards, "unknown logical shard")
        _require(
            self.state["owner"] and self.state["owner"]["phase"] in {"capture", "restore"},
            "publication requires serialized capture/restore ownership",
        )
        _require(
            shard not in self.state["resident"] and len(self.state["resident"]) < 2,
            "physical cached shard count exceeds two",
        )
        directory = self._directory(directory)
        _require(
            tree_bytes(directory) <= self.plan.value["shard_bytes"],
            "actual shard exceeds 8 GiB publication bound",
        )
        _require(
            self.disk_bytes() <= 3 * self.plan.value["shard_bytes"],
            "actual cache including failed staging exceeds 24 GiB",
        )
        _require(
            checked(manifest).resolve().is_relative_to(directory)
            and checked(admission).resolve().is_relative_to(directory),
            "physical manifest/admission must reside in owned shard directory",
        )
        dataset = self.dataset_factory(
            manifest["path"],
            expected_sha256=manifest["sha256"],
            admission_path=admission["path"],
            admission_sha256=admission["sha256"],
            allow_synthetic=False,
        )
        try:
            _require(
                set(dataset.chains) == set(self.plan.shards[shard]),
                "physical shard membership differs from frozen whole groups",
            )
            for key, wanted in self.plan.value["geometry"].items():
                _require(dataset.manifest.get(key) == wanted, "physical geometry differs: " + key)
            validate_capture_source(self.plan, shard, dataset)
            seal = {}
            for cid, chain in dataset.chains.items():
                frozen = self.plan.chains[cid]
                for key in ("prompt_id", "prompt_sha256", "domain", "split"):
                    _require(chain[key] == frozen[key], "native chain metadata differs: " + key)
                tensors = {}
                for key in ("tokens", "features", "logits", "native_receipt"):
                    record = chain[key]
                    _require(
                        record is not None, "full F32 original teacher/source receipt required"
                    )
                    _require(
                        checked(record).resolve().is_relative_to(directory),
                        "physical tensor/source receipt lies outside bounded shard directory",
                    )
                    tensors[key] = record["sha256"]
                seal[cid] = {
                    "artifacts": tensors,
                    "anchors": chain["anchors"],
                    "prompt_length": chain["prompt_length"],
                    "logits_indices": chain.get("logits_indices"),
                }
            original = self.state["seals"].get(shard)
            _require(
                original is None or original["identity"] == seal,
                "restored teacher/source receipt/tensor bytes changed",
            )
            self.state["seals"].setdefault(shard, {"identity": seal, "recovery": None})
            self.state["resident"][shard] = {
                "manifest": manifest,
                "admission": admission,
                "directory": str(directory),
            }
            self._save()
        finally:
            dataset.close()

    def set_recovery(self, shard, locator):
        """Admit actual archive bytes or an explicitly pinned deterministic replay test."""
        value = json.loads(checked(locator).read_text())
        seal = self.state["seals"][shard]["identity"]
        _require(
            value.get("schema") == "block_shard_recovery_v1"
            and value.get("status") == "PASS"
            and value.get("plan_sha256") == self.plan.sha256
            and value.get("shard") == shard
            and value.get("identity") == seal,
            "recovery evidence differs from original tensors",
        )
        if value.get("kind") == "verified_archive":
            archived_manifest = json.loads(checked(value["manifest"]).read_text())
            _require(
                {c["chain_id"] for c in archived_manifest["chains"]} == set(seal),
                "archive manifest membership differs",
            )
            for chain in archived_manifest["chains"]:
                cid = chain["chain_id"]
                _require(
                    chain["anchors"] == seal[cid]["anchors"]
                    and chain["prompt_length"] == seal[cid]["prompt_length"]
                    and chain.get("logits_indices") == seal[cid]["logits_indices"],
                    "archive row maps/boundaries differ",
                )
                for key, digest in seal[cid]["artifacts"].items():
                    _require(
                        chain[key] == value["artifacts"][cid][key]
                        and chain[key]["sha256"] == digest,
                        "archive manifest tensor identity differs",
                    )
            for record in (
                archived_manifest.get("train_inventory"),
                archived_manifest["producer"].get("receipt"),
            ):
                if record:
                    checked(record)
            for cid, record in seal.items():
                for key, digest in record["artifacts"].items():
                    artifact = value["artifacts"][cid][key]
                    _require(artifact["sha256"] == digest, "archive original tensor SHA differs")
                    archived = checked(artifact).resolve()
                    _require(
                        not archived.is_relative_to(self.root), "archive lies inside evicted cache"
                    )
        elif value.get("kind") == "admitted_deterministic_replay":
            _require(
                self.replay_admission is not None
                and value.get("admission") == self.replay_admission,
                "deterministic replay is not explicitly admitted in frozen config",
            )
            validate_replay_admission(self.plan, self.replay_admission)
            metadata = json.loads(checked(value["original_metadata"]).read_text())
            _require(
                metadata["identity"] == seal
                and metadata["plan_sha256"] == self.plan.sha256
                and metadata["shard"] == shard,
                "original replay metadata differs",
            )
            for record in metadata["files"].values():
                checked(record)
        else:
            raise ValueError("no verified archive or admitted exact deterministic replay")
        self.state["seals"][shard]["recovery"] = locator
        self._save()

    def evict(self, shard):
        _require(
            self.state["owner"] and self.state["owner"]["phase"] != "student",
            "live student cannot evict capture shard",
        )
        locator = self.state["seals"][shard]["recovery"]
        _require(locator is not None, "eviction refused: original bytes have no admitted recovery")
        # Reverify archive bytes every time; replay test is retained as evidence,
        # while restored publication independently rechecks ALL original hashes.
        self.set_recovery(shard, locator)
        if shard in self.datasets:
            self.datasets.pop(shard).close()
        directory = self._directory(self.state["resident"][shard]["directory"])
        paths = list(directory.rglob("*"))
        _require(all(not p.is_symlink() for p in paths), "eviction path contains symlinks")
        for path in sorted(paths, key=lambda p: len(p.parts), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        directory.rmdir()
        del self.state["resident"][shard]
        self._save()

    def dataset(self, shard):
        _require(shard in self.state["resident"], "shard is not published")
        if shard not in self.datasets:
            record = self.state["resident"][shard]
            checked(record["manifest"])
            checked(record["admission"])
            self.datasets[shard] = self.dataset_factory(
                record["manifest"]["path"],
                expected_sha256=record["manifest"]["sha256"],
                admission_path=record["admission"]["path"],
                admission_sha256=record["admission"]["sha256"],
                allow_synthetic=False,
            )
        return self.datasets[shard]

    def close(self):
        for dataset in self.datasets.values():
            dataset.close()
        self.datasets.clear()


class RotatingBlockProvider:
    def __init__(self, plan, cache):
        self.plan, self.cache, self.sha256, self.chains = plan, cache, plan.sha256, plan.chains
        self.manifest = {**plan.value["geometry"], "producer": plan.value["source"]}
        self.vocab_size = self.manifest["vocab_size"]
        self.target_width = self.manifest["target_width"]

    def order(self, split, seed, epoch=0):
        return self.plan.order(split, seed, epoch)

    def cursor(self, *, split="train", seed=8101, epoch=0):
        return ShardCursor(
            self.sha256, identity(self.order(split, seed, epoch)), split, seed, epoch, {}, {}
        )

    def restore_cursor(self, payload):
        cursor = ShardCursor(**payload)
        order = self.order(cursor.split, cursor.seed, cursor.epoch)
        _require(
            cursor.dataset_sha256 == self.sha256 and cursor.order_sha256 == identity(order),
            "resume logical corpus/order identity differs",
        )
        _require(isinstance(cursor.progress, dict), "exact logical progress required")
        _require(isinstance(cursor.consumed_seals, dict), "consumed shard seals required")
        for shard, digest in cursor.consumed_seals.items():
            _require(
                shard in self.cache.state["seals"]
                and identity(self.cache.state["seals"][shard]["identity"]) == digest,
                "consumed original tensor/anchor seal changed",
            )
        for epoch, progress in cursor.progress.items():
            _require(
                str(int(epoch)) == epoch and int(epoch) >= cursor.epoch,
                "logical lookahead epoch differs",
            )
            ids = self.order(cursor.split, cursor.seed, int(epoch))
            _require(
                isinstance(progress, dict) and set(progress) <= set(ids),
                "logical chain progress differs",
            )
            for cid, count in progress.items():
                length = self._length(cid)
                _require(
                    type(count) is int and length is not None and 0 <= count <= length,
                    "logical chronological anchor progress differs",
                )
        return cursor

    def _length(self, cid):
        shard = self.plan.chain_shards[cid]
        if shard not in self.cache.state["seals"]:
            return None
        return len(self.cache.state["seals"][shard]["identity"][cid]["anchors"])

    def group_id(self, batch):
        return self.chains[batch.chain_id]["group_id"]

    def reserve_batch(self, cursor, count=2, *, require_teacher=True, reservation=None):
        """Return copied blocks + exact uncommitted cursor; never skip tail groups."""
        cursor = self.restore_cursor(cursor.payload())
        _require(type(count) is int and 1 <= count <= 2, "effective batch must be one or two")
        _require(
            len({self.chains[c]["group_id"] for c in self.order(cursor.split, cursor.seed)})
            >= count,
            "effective batch needs distinct loss-bearing source groups",
        )
        if reservation is not None:
            _require(
                reservation["plan_sha256"] == self.sha256
                and reservation["cursor"] == cursor.payload(),
                "reserved logical cursor changed",
            )
        progress = copy.deepcopy(cursor.progress)
        selected, excluded = [], set()
        for _ in range(count):
            epoch = cursor.epoch
            while True:
                epoch_progress = progress.setdefault(str(epoch), {})
                candidates = []
                for cid in self.order(cursor.split, cursor.seed, epoch):
                    if self.chains[cid]["group_id"] in excluded:
                        continue
                    length = self._length(cid)
                    if length is None:
                        shard = self.plan.chain_shards[cid]
                        needed = list(
                            dict.fromkeys(
                                [self.plan.chain_shards[c] for _, c, _ in selected] + [shard]
                            )
                        )
                        raise ShardRequired(
                            needed,
                            {
                                "kind": "metadata",
                                "plan_sha256": self.sha256,
                                "cursor": cursor.payload(),
                                "chain_id": cid,
                                "shard_ids": needed,
                            },
                        )
                    if epoch_progress.get(cid, 0) < length:
                        candidates.append(cid)
                        break
                if candidates:
                    cid = candidates[0]
                    block = epoch_progress.get(cid, 0)
                    selected.append((epoch, cid, block))
                    epoch_progress[cid] = block + 1
                    excluded.add(self.chains[cid]["group_id"])
                    break
                _require(
                    any(
                        self.chains[cid]["group_id"] not in excluded
                        and (self._length(cid) is None or self._length(cid) > 0)
                        for cid in self.order(cursor.split, cursor.seed, epoch)
                    ),
                    "no loss-bearing anchors in requested split",
                )
                epoch += 1
        epoch = cursor.epoch
        while all(
            progress.get(str(epoch), {}).get(cid, 0) == self._length(cid)
            for cid in self.order(cursor.split, cursor.seed, epoch)
        ):
            progress.pop(str(epoch), None)
            epoch += 1
        seals = dict(cursor.consumed_seals)
        for _, cid, _ in selected:
            shard = self.plan.chain_shards[cid]
            seals[shard] = identity(self.cache.state["seals"][shard]["identity"])
        next_cursor = replace(
            cursor,
            epoch=epoch,
            progress=progress,
            consumed_seals=seals,
            order_sha256=identity(self.order(cursor.split, cursor.seed, epoch)),
        )
        shards = tuple(dict.fromkeys(self.plan.chain_shards[cid] for _, cid, _ in selected))
        derived = {
            "kind": "batch",
            "plan_sha256": self.sha256,
            "cursor": cursor.payload(),
            "next_cursor": next_cursor.payload(),
            "selected": [list(r) for r in selected],
            "shard_ids": list(shards),
        }
        if reservation is not None and reservation["kind"] == "batch":
            _require(derived == reservation, "exact reserved batch changed during reconstruction")
        if any(s not in self.cache.state["resident"] for s in shards):
            raise ShardRequired(shards, derived)
        blocks = []
        for _, cid, block in selected:
            batch = self.cache.dataset(self.plan.chain_shards[cid]).load_block(
                cid, block, require_teacher=require_teacher
            )
            blocks.append(replace(batch, dataset_sha256=self.sha256))
        return blocks, next_cursor

    def next_block(self, cursor, *, require_teacher=False):
        blocks, following = self.reserve_batch(cursor, 1, require_teacher=require_teacher)
        return blocks[0], following

    def close(self):
        self.cache.close()

    def storage_summary(self):
        known = [
            chain
            for seal in self.cache.state["seals"].values()
            for chain in seal["identity"].values()
        ]
        return {
            "chains": len(self.chains),
            "known_blocks": sum(len(c["anchors"]) for c in known),
            "eligible_role_counts": self.plan.role_counts,
            "captured_chains": len(known),
            "cached_shards": len(self.cache.state["resident"]),
            "physical_bytes": self.cache.disk_bytes(),
        }


def open_provider(data):
    _require(data.get("provider") == "rotating_block_v1", "unknown rotating provider")
    plan = FrozenShardPlan.load({key: data[key] for key in ("path", "sha256")})
    validate_deployed_source(plan)
    return RotatingBlockProvider(
        plan, ShardCache(plan, data["cache_root"], replay_admission=data.get("replay_admission"))
    )


def validate_controller_config(config, data, source):
    """Validate frozen executable helper pins; argv never passes through a shell."""
    _require(config.get("schema") == "block_shard_controller_v1", "shard controller schema differs")
    _require(
        config.get("plan") == {key: data[key] for key in ("path", "sha256")}
        and config.get("cache_root") == data.get("cache_root"),
        "controller/provider differs",
    )
    plan = FrozenShardPlan.load(config["plan"])
    validate_deployed_source(plan)
    _require(
        type(config.get("phase_wall_seconds")) is int and config["phase_wall_seconds"] > 0,
        "bounded capture/reconstruction wall cap required",
    )
    for name in ("capture", "restore"):
        command = config.get(name)
        _require(isinstance(command, dict), "explicit capture and restore commands required")
        producer = command["producer"]
        checked(producer)
        if command.get("replay_producer"):
            checked(command["replay_producer"])
            _require(
                command["replay_producer"] in source.values(), "replay producer source missing"
            )
        _require(producer in source.values(), "shard producer missing frozen source inventory")
        argv = command.get("argv")
        _require(
            isinstance(argv, list)
            and argv
            and all(isinstance(v, str) and v for v in argv)
            and producer["path"] in argv
            and "{request}" in argv
            and "{output}" in argv,
            "explicit producer argv/request/output required",
        )
        _require(
            set(v for v in argv if "{" in v or "}" in v) <= {"{request}", "{output}"},
            "unknown shard command interpolation",
        )
    if config.get("replay_admission"):
        _require(
            config["replay_admission"] == data.get("replay_admission"),
            "controller/provider replay admission differs",
        )
        validate_replay_admission(
            plan,
            config["replay_admission"],
            producer=config["restore"].get("replay_producer", config["restore"]["producer"]),
        )
    return plan


def restore_requested_shards(
    config, request, *, cache, run_dir, attempt, runner, release, authorization, source
):
    """Execute one serialized publication per missing shard, outside trainer time.

    Phase costs are written separately. An unverified replay or failed current-host
    admission cannot authorize eviction or a student restart. Crashed staging is
    retained and counts against the next 24 GiB peak check.
    """
    import time
    import uuid

    _require(
        request.get("plan_sha256") == cache.plan.sha256, "shard boundary plan identity differs"
    )
    ids = request.get("shard_ids")
    _require(
        isinstance(ids, list)
        and 0 < len(ids) <= 2
        and len(set(ids)) == len(ids)
        and set(ids) <= set(cache.plan.shards),
        "bounded known requested shards required",
    )
    reservation = request.get("reservation")
    _require(
        isinstance(reservation, dict)
        and reservation["plan_sha256"] == cache.plan.sha256
        and reservation["shard_ids"] == ids,
        "exact shard reservation differs",
    )
    provider = RotatingBlockProvider(cache.plan, cache)
    provider.restore_cursor(reservation["cursor"])
    # This must be a real release check supplied by the exclusive controller.
    proof = release()
    if cache.state["owner"] is not None:
        cache.release(cache.state["owner"]["owner"], released_proof=proof)
    owner = "controller-" + uuid.uuid4().hex
    cache.acquire(owner, "restore")
    phases = []
    try:
        excess = max(0, len(set(cache.state["resident"]) | set(ids)) - 2)
        for shard in [s for s in cache.state["resident"] if s not in ids][:excess]:
            if cache.replay_admission and cache.state["seals"][shard]["recovery"] is None:
                enable_replay_recovery(cache, shard)
        if config.get("archive_root"):
            for shard in [s for s in cache.state["resident"] if s not in ids][:excess]:
                if cache.state["seals"][shard]["recovery"] is None:
                    archive_shard(
                        cache,
                        shard,
                        config["archive_root"],
                        max_bytes=config["archive_max_bytes"],
                        min_free_bytes=config.get("archive_min_free_bytes", 10 * GIB),
                    )
        cache.before_staging(ids)
        for shard in ids:
            if shard in cache.state["resident"]:
                continue
            authorization()
            stage = Path(attempt) / "shard-phases" / uuid.uuid4().hex
            stage.mkdir(parents=True, exist_ok=False)
            directory = cache.root / shard
            _require(
                not directory.exists(),
                "canonical shard staging already exists; exact repair required",
            )
            original = cache.state["seals"].get(shard)
            publication_request = {
                "schema": "block_shard_restore_request_v1",
                "plan": config["plan"],
                "plan_sha256": cache.plan.sha256,
                "shard": shard,
                "directory": str(directory),
                "original_identity": None if original is None else original["identity"],
                "generation_variant": cache.plan.value["generation_variant"],
                "capture_bytes_bound": cache.plan.value["shard_bytes"],
                "original_metadata": None
                if original is None
                else original.get("original_metadata"),
                "replay_admission": cache.replay_admission,
            }
            request_path, output = stage / "request.json", stage / "publication.json"
            atomic_json(request_path, publication_request)
            operation = "capture" if original is None else "restore"
            command = config[operation]
            checked(command["producer"])
            _require(command["producer"] in source.values(), "producer source changed")
            argv = [
                str(request_path)
                if arg == "{request}"
                else str(output)
                if arg == "{output}"
                else arg
                for arg in command["argv"]
            ]
            started = time.monotonic()
            recovery = None if original is None else original.get("recovery")
            if (
                recovery
                and json.loads(checked(recovery).read_text()).get("kind") == "verified_archive"
            ):
                restore_archive(cache, shard, directory, output, request_path)
            else:
                runner.run(
                    argv,
                    directory=stage / "producer",
                    stop_path=Path(run_dir) / "STOP",
                    wall_seconds=config["phase_wall_seconds"],
                )
            authorization()
            released = release()
            receipt = json.loads(output.read_text())
            _require(
                receipt.get("schema") == "block_shard_publication_v1"
                and receipt.get("status") == "PASS"
                and receipt.get("plan_sha256") == cache.plan.sha256
                and receipt.get("shard") == shard
                and receipt.get("directory") == str(directory)
                and receipt.get("request") == pin(request_path),
                "actual shard publication receipt differs",
            )
            cache.publish(shard, receipt["manifest"], receipt["admission"], directory)
            if config.get("metadata_root") and original is None:
                preserve_original_metadata(
                    cache, shard, config["metadata_root"], max_bytes=config["metadata_max_bytes"]
                )
            if original is not None and receipt.get("reconstruction_join"):
                verify_reconstruction_join(cache, shard, receipt)
            if receipt.get("recovery"):
                cache.set_recovery(shard, receipt["recovery"])
            phase = {
                "operation": operation,
                "shard": shard,
                "wall_seconds": time.monotonic() - started,
                "trainer_seconds_charged": 0,
                "publication": pin(output),
                "released": released,
                "cache_bytes": cache.disk_bytes(),
            }
            atomic_json(stage / "phase-cost.json", phase)
            phases.append(pin(stage / "phase-cost.json"))
        return phases
    finally:
        provider.close()
        # Preserve ownership on a failed release: an alive capture is never
        # described as free merely because an exception occurred.
        proof = release()
        cache.release(owner, released_proof=proof)


def archive_shard(cache, shard, archive_root, *, max_bytes, min_free_bytes=10 * GIB):
    """Copy an original shard to an explicitly budgeted durable filesystem archive.

    This is a real filesystem archive, not evidence of an off-host/object-store
    upload. It preserves native receipt bytes and authentic tensor bytes exactly.
    """
    import shutil
    import uuid

    root = Path(archive_root).resolve()
    _require(
        not root.is_relative_to(cache.root) and not cache.root.is_relative_to(root),
        "archive and rotating cache must use disjoint directories",
    )
    _require(type(max_bytes) is int and max_bytes > 0, "explicit archive capacity required")
    root.mkdir(parents=True, exist_ok=True)
    source = cache._directory(cache.state["resident"][shard]["directory"])
    source_bytes = tree_bytes(source)
    used = tree_bytes(root)
    _require(used + source_bytes <= max_bytes, "archive including failed staging exceeds capacity")
    _require(
        shutil.disk_usage(root).free >= source_bytes + min_free_bytes,
        "archive copy cannot preserve configured free disk floor",
    )
    directory = root / (shard + "-" + uuid.uuid4().hex)
    import os

    copied_inodes = {}

    def copy_unique(old, new):
        stat = Path(old).stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in copied_inodes:
            os.link(copied_inodes[inode], new)
        else:
            shutil.copyfile(old, new)
            copied_inodes[inode] = new
        return new

    shutil.copytree(source, directory, symlinks=False, copy_function=copy_unique)
    manifest = json.loads(checked(cache.state["resident"][shard]["manifest"]).read_text())

    def relocate(record):
        old = checked(record).resolve()
        if old.is_relative_to(source):
            record["path"] = str(directory / old.relative_to(source))
        checked(record)

    for chain in manifest["chains"]:
        for key in ("tokens", "features", "logits", "native_receipt"):
            if chain[key] is not None:
                relocate(chain[key])
    for record in (manifest.get("train_inventory"), manifest["producer"].get("receipt")):
        if record:
            relocate(record)
    archive_manifest = directory / "archive-manifest.json"
    atomic_json(archive_manifest, manifest)
    artifacts = {
        c["chain_id"]: {
            key: c[key]
            for key in cache.state["seals"][shard]["identity"][c["chain_id"]]["artifacts"]
        }
        for c in manifest["chains"]
    }
    receipt = directory / "recovery.json"
    atomic_json(
        receipt,
        {
            "schema": "block_shard_recovery_v1",
            "status": "PASS",
            "kind": "verified_archive",
            "plan_sha256": cache.plan.sha256,
            "shard": shard,
            "identity": cache.state["seals"][shard]["identity"],
            "artifacts": artifacts,
            "manifest": pin(archive_manifest),
            "original_directory": str(source),
            "original_manifest_relative_path": str(
                Path(cache.state["resident"][shard]["manifest"]["path"]).relative_to(source)
            ),
        },
    )
    _require(
        tree_bytes(root) <= max_bytes,
        "archive metadata publication exceeds capacity; evidence retained",
    )
    cache.set_recovery(shard, pin(receipt))
    return pin(receipt)


def restore_archive(cache, shard, directory, output, request_path):
    """Preserve original tensor/receipt bytes and produce fresh host admission."""
    import shutil

    recovery = cache.state["seals"][shard]["recovery"]
    _require(recovery is not None, "original shard has no verified archive")
    cache.set_recovery(shard, recovery)
    receipt = json.loads(checked(recovery).read_text())
    _require(
        receipt["kind"] == "verified_archive", "filesystem restoration requires verified archive"
    )
    _require(
        Path(directory).resolve() == Path(receipt["original_directory"]).resolve(),
        "archive restore requires original canonical directory",
    )
    source = checked(receipt["manifest"]).parent
    _require(not Path(directory).exists(), "canonical restore directory already exists")
    # Copy the complete original tree; historical receipt bytes contain canonical
    # generation/source paths. Preserve aliases without duplicating large arrays.
    import os

    inodes = {}

    def copy_unique(old, new):
        stat = Path(old).stat()
        key = (stat.st_dev, stat.st_ino)
        if key in inodes:
            os.link(inodes[key], new)
        else:
            shutil.copyfile(old, new)
            inodes[key] = new
        return new

    shutil.copytree(source, directory, copy_function=copy_unique)
    manifest_path = Path(directory) / receipt["original_manifest_relative_path"]
    admission_path = Path(directory) / "restored-completed-admission.json"
    dataset = cache.dataset_factory(
        manifest_path, expected_sha256=file_sha256(manifest_path), allow_synthetic=False
    )
    try:
        dataset.write_admission(admission_path)
    finally:
        dataset.close()
    value = {
        "schema": "block_shard_publication_v1",
        "status": "PASS",
        "plan_sha256": cache.plan.sha256,
        "shard": shard,
        "directory": str(directory),
        "manifest": pin(manifest_path),
        "admission": pin(admission_path),
        "request": pin(request_path),
        "recovery": recovery,
    }
    atomic_json(output, value)


def preserve_original_metadata(cache, shard, metadata_root, *, max_bytes):
    """Retain original receipts/token/history bytes, never compressed teacher surrogates."""
    import shutil

    root = Path(metadata_root).resolve()
    _require(
        not root.is_relative_to(cache.root) and not cache.root.is_relative_to(root),
        "compact original records must outlive cache eviction",
    )
    source = cache._directory(cache.state["resident"][shard]["directory"])
    destination = root / shard
    _require(
        not destination.exists(), "original metadata already exists; cannot replace first capture"
    )
    destination.mkdir(parents=True, exist_ok=False)
    manifest_record = cache.state["resident"][shard]["manifest"]
    manifest = json.loads(checked(manifest_record).read_text())
    paths = {p for p in source.rglob("*") if p.suffix in {".json", ".jsonl"} and p.is_file()}
    paths.update(checked(c["tokens"]) for c in manifest["chains"])
    used = tree_bytes(root)
    _require(
        type(max_bytes) is int and used + sum(p.stat().st_size for p in paths) <= max_bytes,
        "compact metadata cannot fit configured byte budget",
    )
    files = {}
    for path in sorted(paths):
        _require(
            not path.is_symlink() and path.resolve().is_relative_to(source),
            "original metadata outside canonical shard directory",
        )
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        _require(file_sha256(target) == file_sha256(path), "original metadata copy bytes differ")
        files[str(relative)] = pin(target)
    output = destination / "original-metadata.json"
    atomic_json(
        output,
        {
            "schema": "block_shard_original_metadata_v1",
            "plan_sha256": cache.plan.sha256,
            "shard": shard,
            "identity": cache.state["seals"][shard]["identity"],
            "original_directory": str(source),
            "files": files,
            "manifest_relative_path": str(Path(manifest_record["path"]).relative_to(source)),
            "admission_relative_path": str(
                Path(cache.state["resident"][shard]["admission"]["path"]).relative_to(source)
            ),
        },
    )
    _require(
        tree_bytes(root) <= max_bytes,
        "original metadata publication exceeds byte budget; evidence retained",
    )
    cache.state["seals"][shard]["original_metadata"] = pin(output)
    cache._save()
    return pin(output)


def validate_replay_admission(plan, locator, *, producer=None):
    """Revalidate actual recapture artifacts; flags alone never authorize eviction.

    The small three-domain trial must remain available in separately budgeted
    preparation/archive storage. Missing proof artifacts fail closed rather than
    turning an old textual PASS into permission to destroy original teachers.
    """
    from types import SimpleNamespace

    value = json.loads(checked(locator).read_text())
    _require(
        value.get("schema") == "block_shard_replay_admission_v1"
        and value.get("status") == "PASS"
        and value.get("plan_sha256") == plan.sha256
        and value.get("source") == plan.value["source"]
        and value.get("original_identity")
        and value.get("original_identity") == value.get("replay_identity")
        and value.get("shard") in plan.shards
        and set(value.get("domains", [])) == set(DOMAINS),
        "actual nonempty three-domain source-bound tensor replay admission required",
    )
    _require(
        producer is None or value.get("producer") == producer, "replay producer source differs"
    )
    checked(value["producer"])
    if "module:scripts/replay_dspark_shard.py" in plan.value["source"]:
        _require(
            value["producer"]["sha256"]
            == plan.value["source"]["module:scripts/replay_dspark_shard.py"],
            "replay producer differs from frozen module",
        )
    metadata = json.loads(checked(value["original_metadata"]).read_text())
    shard = value["shard"]
    expected = value["original_identity"]
    _require(
        metadata.get("schema") == "block_shard_original_metadata_v1"
        and metadata.get("plan_sha256") == plan.sha256
        and metadata.get("shard") == shard
        and metadata.get("identity") == expected
        and set(expected) == set(plan.shards[shard]),
        "actual original trial metadata/membership differs",
    )
    for record in metadata["files"].values():
        checked(record)
    publication = json.loads(checked(value["trial_publication"]).read_text())
    _require(
        publication.get("schema") == "block_shard_publication_v1"
        and publication.get("status") == "PASS"
        and publication.get("plan_sha256") == plan.sha256
        and publication.get("shard") == shard,
        "actual typed cold trial publication required",
    )
    state = {
        "seals": {shard: {"identity": expected, "original_metadata": value["original_metadata"]}}
    }
    trial_cache = SimpleNamespace(state=state, plan=plan)
    join = verify_reconstruction_join(trial_cache, shard, publication)
    _require(join.get("producer") == value["producer"], "actual fresh trial producer differs")
    dataset = BlockDataset(
        checked(publication["manifest"]),
        expected_sha256=publication["manifest"]["sha256"],
        admission_path=str(checked(publication["admission"])),
        admission_sha256=publication["admission"]["sha256"],
        allow_synthetic=False,
    )
    try:
        validate_capture_source(plan, shard, dataset)
        _require(
            set(dataset.chains) == set(expected)
            and {c["domain"] for c in dataset.chains.values()} == set(DOMAINS),
            "actual trial chain/domain coverage differs",
        )
        for cid, chain in dataset.chains.items():
            actual = {
                "artifacts": {
                    key: file_sha256(checked(chain[key]))
                    for key in ("tokens", "features", "logits", "native_receipt")
                },
                "anchors": chain["anchors"],
                "prompt_length": chain["prompt_length"],
                "logits_indices": chain.get("logits_indices"),
            }
            _require(actual == expected[cid], "actual cold trial original tensor bytes differ")
    finally:
        dataset.close()
    return value


def enable_replay_recovery(cache, shard):
    validate_replay_admission(cache.plan, cache.replay_admission)
    original = cache.state["seals"][shard]
    _require(original.get("original_metadata") is not None, "original receipt bytes not retained")
    path = Path(original["original_metadata"]["path"]).parent / "replay-recovery.json"
    value = {
        "schema": "block_shard_recovery_v1",
        "status": "PASS",
        "kind": "admitted_deterministic_replay",
        "plan_sha256": cache.plan.sha256,
        "shard": shard,
        "identity": original["identity"],
        "admission": cache.replay_admission,
        "original_metadata": original["original_metadata"],
    }
    _require(not path.exists(), "preserve original replay recovery receipt")
    atomic_json(path, value)
    cache.set_recovery(shard, pin(path))


def verify_reconstruction_join(cache, shard, publication):
    join = json.loads(checked(publication["reconstruction_join"]).read_text())
    original = cache.state["seals"][shard]
    _require(
        join.get("schema") == "block_shard_reconstruction_join_v1"
        and join.get("status") == "PASS"
        and join.get("plan_sha256") == cache.plan.sha256
        and join.get("shard") == shard
        and join.get("original_metadata") == original["original_metadata"]
        and join.get("original_identity") == original["identity"]
        and join.get("tensor_identity_equal") is True
        and join.get("historical_receipt_bytes_restored") is True,
        "actual replay/historical receipt join incomplete",
    )
    manifest = json.loads(checked(publication["manifest"]).read_text())
    _require(
        set(join["fresh_receipts"]) == set(original["identity"]),
        "replay fresh receipt inventory differs",
    )
    for chain in manifest["chains"]:
        old = json.loads(checked(chain["native_receipt"]).read_text())
        fresh = json.loads(checked(join["fresh_receipts"][chain["chain_id"]]).read_text())
        for key in (
            "target_sha256",
            "producer_binary_sha256",
            "producer_source_revision",
            "client_source_sha256",
            "target_precision",
            "kv_type",
            "tap_ids",
            "tokens",
            "decode_history",
            "features_shape",
            "logits_indices",
            "logits_shape",
            "logits_mode",
        ):
            _require(
                old.get(key) == fresh.get(key), "actual replay native semantics changed: " + key
            )
        for key in ("features", "logits"):
            _require(
                fresh["files"][key]["sha256"] == chain[key]["sha256"],
                "actual replay raw tensor SHA differs",
            )
    return join


def admit_replay(cache, shard, trial_publication, producer, output):
    """Publish a gate only from actual byte-matching three-domain recapture artifacts."""
    publication = json.loads(checked(trial_publication).read_text())
    _require(
        publication.get("schema") == "block_shard_publication_v1"
        and publication.get("status") == "PASS"
        and publication.get("shard") == shard
        and publication.get("plan_sha256") == cache.plan.sha256,
        "actual cold trial publication required",
    )
    checked(producer)
    join = verify_reconstruction_join(cache, shard, publication)
    _require(join.get("producer") == producer, "actual replay producer provenance differs")
    dataset = cache.dataset_factory(
        checked(publication["manifest"]),
        expected_sha256=publication["manifest"]["sha256"],
        allow_synthetic=False,
        admission_path=str(checked(publication["admission"])),
        admission_sha256=publication["admission"]["sha256"],
    )
    try:
        replay_identity = {}
        for cid, chain in dataset.chains.items():
            replay_identity[cid] = {
                "artifacts": {
                    key: pin(checked(chain[key]))["sha256"]
                    for key in ("tokens", "features", "logits", "native_receipt")
                },
                "anchors": chain["anchors"],
                "prompt_length": chain["prompt_length"],
                "logits_indices": chain.get("logits_indices"),
            }
        _require(
            replay_identity == cache.state["seals"][shard]["identity"],
            "cold replay actual bytes differ",
        )
        domains = sorted({chain["domain"] for chain in dataset.chains.values()})
        _require(
            set(domains) == set(DOMAINS), "small replay gate must exercise three source domains"
        )
        _require(not Path(output).exists(), "preserve replay admission history")
        atomic_json(
            output,
            {
                "schema": "block_shard_replay_admission_v1",
                "status": "PASS",
                "plan_sha256": cache.plan.sha256,
                "shard": shard,
                "source": cache.plan.value["source"],
                "producer": producer,
                "domains": domains,
                "trial_publication": trial_publication,
                "original_metadata": cache.state["seals"][shard]["original_metadata"],
                "original_identity": cache.state["seals"][shard]["identity"],
                "replay_identity": replay_identity,
            },
        )
    finally:
        dataset.close()
    return pin(output)


def validate_logical_admission(data, locator):
    plan = FrozenShardPlan.load({key: data[key] for key in ("path", "sha256")})
    validate_deployed_source(plan)
    value = json.loads(checked(locator).read_text())
    _require(
        value.get("schema") == "block_logical_data_admission_v1"
        and value.get("plan_sha256") == plan.sha256
        and value.get("plan_file_sha256") == data["sha256"]
        and value.get("source_sha256") == file_sha256(Path(__file__))
        and value.get("role_counts") == plan.role_counts,
        "completed logical plan/source/role admission differs",
    )
    checked(value["preparation_data"])
    checked(value["preparation_admission"])
    return plan, value


def write_logical_admission(data, preparation_data, preparation_admission, output):
    """Bind complete 96/48 calibration roles plus admitted first TRAIN tensors.

    The receipt admits frozen eligibility and actual preparation bytes, never
    claims that the not-yet-captured 9856 TRAIN teachers already exist.
    """
    provider = open_provider(data)
    try:
        dataset = BlockDataset(
            checked(preparation_data),
            expected_sha256=preparation_data["sha256"],
            admission_path=str(checked(preparation_admission)),
            admission_sha256=preparation_admission["sha256"],
            allow_synthetic=False,
        )
        try:
            roles = {split: set() for split in SPLITS}
            for cid, chain in dataset.chains.items():
                frozen = provider.chains[cid]
                _require(
                    all(
                        chain[key] == frozen[key]
                        for key in ("prompt_id", "prompt_sha256", "domain", "split")
                    ),
                    "preparation chain differs from global selector",
                )
                roles[chain["split"]].add(cid)
            for split in ("calibration_fit", "calibration_validation"):
                _require(
                    roles[split]
                    == {c for c in provider.chains if provider.chains[c]["split"] == split},
                    "complete distinct 96/48 calibration roles required",
                )
            _require(
                {dataset.chains[c]["domain"] for c in roles["train"]} == set(DOMAINS),
                "actual first TRAIN preparation must exercise three domains",
            )
            provider.reserve_batch(provider.cursor(), 2, require_teacher=True)
            _require(not Path(output).exists(), "preserve completed logical admission")
            atomic_json(
                output,
                {
                    "schema": "block_logical_data_admission_v1",
                    "plan_sha256": provider.sha256,
                    "plan_file_sha256": data["sha256"],
                    "source_sha256": file_sha256(Path(__file__)),
                    "role_counts": provider.plan.role_counts,
                    "preparation_data": preparation_data,
                    "preparation_admission": preparation_admission,
                    "initial_shard_seals": {
                        s: identity(v["identity"]) for s, v in provider.cache.state["seals"].items()
                    },
                },
            )
        finally:
            dataset.close()
    finally:
        provider.close()
    return pin(output)


def rebind_physical_manifest(plan, manifest_record, admission_record, output):
    """Map already captured canonical prompt identities to frozen logical chain IDs.

    Only new small manifests/aggregate receipts are written. Every tensor and
    original native receipt keeps its actual bytes and original SHA. This cannot
    turn a partial/incomplete capture into admission or an original replay seal.
    """
    output = Path(output).resolve()
    _require(not output.exists(), "preserve logical rebind provenance")
    dataset = BlockDataset(
        checked(manifest_record),
        expected_sha256=manifest_record["sha256"],
        admission_path=str(checked(admission_record)),
        admission_sha256=admission_record["sha256"],
        allow_synthetic=False,
    )
    try:
        manifest = json.loads(checked(manifest_record).read_text())
        producer_receipt = json.loads(checked(manifest["producer"]["receipt"]).read_text())
        by_prompt = {chain["prompt_id"]: cid for cid, chain in plan.chains.items()}
        mapping, chains, joins = {}, [], {}
        for old, chain in dataset.chains.items():
            _require(chain["prompt_id"] in by_prompt, "captured prompt absent from global selector")
            cid = by_prompt[chain["prompt_id"]]
            _require(
                cid not in mapping.values()
                and all(
                    chain[key] == plan.chains[cid][key]
                    for key in ("prompt_id", "prompt_sha256", "domain", "split")
                ),
                "captured source/role identity differs from immutable selector",
            )
            mapping[old] = cid
            chains.append({**chain, "chain_id": cid})
            joins[cid] = producer_receipt["chains"][old]
        output.mkdir(parents=True)
        receipt_path = output / "logical-producer-receipt.json"
        atomic_json(
            receipt_path,
            {
                **producer_receipt,
                "chains": joins,
                "logical_rebind_provenance": {
                    "original_manifest": manifest_record,
                    "original_admission": admission_record,
                    "original_producer_receipt": manifest["producer"]["receipt"],
                    "chain_id_map": mapping,
                    "plan_sha256": plan.sha256,
                },
            },
        )
        path = output / "manifest.json"
        atomic_json(
            path,
            {
                **manifest,
                "chains": chains,
                "producer": {**manifest["producer"], "receipt": pin(receipt_path)},
            },
        )
        fresh = BlockDataset(path, expected_sha256=file_sha256(path), allow_synthetic=False)
        fresh_admission = output / "completed-admission.json"
        try:
            fresh.write_admission(fresh_admission)
        finally:
            fresh.close()
        result = {
            "schema": "block_logical_chain_rebind_v1",
            "status": "PASS",
            "plan_sha256": plan.sha256,
            "original_manifest": manifest_record,
            "original_admission": admission_record,
            "chain_id_map": mapping,
            "data": pin(path),
            "admission": pin(fresh_admission),
            "tensor_files_copied": 0,
            "native_receipt_bytes_changed": False,
        }
        atomic_json(output / "rebind-provenance.json", result)
        return result
    finally:
        dataset.close()


def merge_preparation_manifests(plan, records, output):
    """Join authentic complete calibration + first TRAIN manifests without tensor copies.

    Original native receipt bytes and array SHA pins remain untouched. New small
    inventory/aggregate producer metadata has an explicit input-provenance ledger;
    no aggregate is presented as one newly executed native capture.
    """
    output = Path(output).resolve()
    _require(not output.exists(), "preserve complete preparation merge history")
    output.mkdir(parents=True)
    merged, prompts, receipts, producer = {}, {}, [], None
    template = None
    for record in records:
        manifest_path, admission_path = checked(record["manifest"]), checked(record["admission"])
        dataset = BlockDataset(
            manifest_path,
            expected_sha256=record["manifest"]["sha256"],
            admission_path=str(admission_path),
            admission_sha256=record["admission"]["sha256"],
            allow_synthetic=False,
        )
        try:
            manifest = json.loads(manifest_path.read_text())
            actual = {k: v for k, v in manifest["producer"].items() if k != "receipt"}
            _require(
                producer is None or producer == actual,
                "capture source/runtime/precision differs across merge",
            )
            producer = actual
            if template is None:
                template = manifest
            else:
                for key in (
                    "family",
                    "vocab_size",
                    "target_width",
                    "mask_token_id",
                    "taps",
                    "tap_semantics",
                    "layout",
                ):
                    _require(
                        manifest[key] == template[key], "capture geometry differs across merge"
                    )
            inventory = json.loads(checked(manifest["train_inventory"]).read_text())
            for pid, prompt in inventory["prompts"].items():
                _require(
                    prompts.setdefault(pid, prompt) == prompt, "original inventory disagreement"
                )
            for cid, chain in dataset.chains.items():
                _require(
                    cid not in merged and cid in plan.chains,
                    "duplicate/unknown merged logical chain",
                )
                _require(
                    all(
                        chain[k] == plan.chains[cid][k]
                        for k in ("prompt_id", "prompt_sha256", "domain", "split")
                    ),
                    "preparation chain differs from immutable global role/source selector",
                )
                merged[cid] = copy.deepcopy(chain)
            receipts.append(
                {
                    "manifest": record["manifest"],
                    "admission": record["admission"],
                    "original_producer_receipt": manifest["producer"]["receipt"],
                }
            )
        finally:
            dataset.close()
    for split in ("calibration_fit", "calibration_validation"):
        _require(
            {cid for cid, c in merged.items() if c["split"] == split}
            == {cid for cid, c in plan.chains.items() if c["split"] == split},
            "all 96 fit / 48 validation calibration sources must be present",
        )
    _require(
        {c["domain"] for c in merged.values() if c["split"] == "train"} == set(DOMAINS),
        "actual first TRAIN cases require three domains",
    )
    inventory_path = output / "original-train-inventory.json"
    atomic_json(inventory_path, {"schema": "block_train_inventory_v1", "prompts": prompts})
    joins = {}
    for cid, chain in merged.items():
        joins[cid] = {
            "tokens_sha256": chain["tokens"]["sha256"],
            "features_sha256": chain["features"]["sha256"],
            "logits_sha256": None if chain["logits"] is None else chain["logits"]["sha256"],
            "prompt_id": chain["prompt_id"],
            "prompt_sha256": chain["prompt_sha256"],
            "prompt_length": chain["prompt_length"],
            "native_receipt_sha256": chain["native_receipt"]["sha256"],
        }
        if "logits_indices" in chain:
            joins[cid]["logits_indices"] = chain["logits_indices"]
    receipt_path = output / "aggregate-producer-receipt.json"
    atomic_json(
        receipt_path,
        {
            "schema": "block_target_capture_receipt_v1",
            "producer": producer,
            "train_inventory_sha256": file_sha256(inventory_path),
            "taps": template["taps"],
            "tap_semantics": template["tap_semantics"],
            "vocab_size": template["vocab_size"],
            "capture_mode": "target_only_autoregressive_train",
            "chains": joins,
            "aggregation_provenance": receipts,
        },
    )
    manifest_path = output / "manifest.json"
    atomic_json(
        manifest_path,
        {
            **template,
            "producer": {**producer, "receipt": pin(receipt_path)},
            "train_inventory": pin(inventory_path),
            "chains": list(merged.values()),
        },
    )
    dataset = BlockDataset(
        manifest_path, expected_sha256=file_sha256(manifest_path), allow_synthetic=False
    )
    admission_path = output / "completed-admission.json"
    try:
        dataset.write_admission(admission_path)
    finally:
        dataset.close()
    result = {
        "schema": "block_preparation_merge_v1",
        "status": "PASS",
        "plan_sha256": plan.sha256,
        "original_manifests": records,
        "data": pin(manifest_path),
        "admission": pin(admission_path),
        "tensor_files_copied": 0,
        "calibration_fit_prompts": 96,
        "calibration_validation_prompts": 48,
    }
    atomic_json(output / "merge-provenance.json", result)
    return result


def freeze_capture_schedule(schedule_locator, *, plan_root):
    """Adapt the authenticated data worker's whole-group byte-bounded schedule."""
    schedule = json.loads(checked(schedule_locator).read_text())
    root = Path(plan_root).resolve()
    _require(
        schedule.get("schema") == "dspark_full_pool_capture_schedule_v1"
        and schedule["seed"] == 8101,
        "frozen full-pool capture schedule required",
    )
    selector = {
        "path": str(root / schedule["role_selector"]["path"]),
        "sha256": schedule["role_selector"]["sha256"],
    }
    checked(selector)
    first = None
    capture_plans, shards = {}, {}
    for chunk in schedule["chunks"]:
        shard = f"chunk-{chunk['chunk_id']:04d}"
        shards[shard] = chunk["chain_ids"]
        capture_plans[shard] = {}
        for kind, record in chunk["plans"].items():
            locator = {"path": str(root / record["path"]), "sha256": record["sha256"]}
            checked(locator)
            capture_plans[shard][kind] = locator
        native = json.loads(checked(capture_plans[shard]["remote"]).read_text())
        if first is None:
            first = native
        else:
            _require(
                native["native"] == first["native"]
                and native["runtime"] == first["runtime"]
                and native["generation"] == first["generation"],
                "capture runtime/variant changes across shards",
            )
    source = {
        "target_sha256": first["native"]["target"]["sha256"],
        "teacher_binary_sha256": first["native"]["binary"]["sha256"],
        "capture_runtime_sha256": first["runtime"]["sha256"],
        "capture_schedule_sha256": schedule_locator["sha256"],
        "source_contract_sha256": schedule["source_contracts"]["remote"]["sha256"],
        "role_selector_sha256": selector["sha256"],
        "generation_variant_sha256": schedule["generation_variant"]["receipt_sha256"],
    }
    source.update({"module:" + key: digest for key, digest in schedule["source_modules"].items()})
    source.update(
        {f"source:{r['kind']}:{r['shard']}": r["sha256"] for r in schedule["source_files"]}
    )
    value = {
        "schema": PLAN_SCHEMA,
        "selector": selector,
        "source": source,
        "seed": 8101,
        "geometry": {
            "family": "dspark",
            "vocab_size": first["vocab_size"],
            "target_width": first["target_width"],
            "mask_token_id": first["mask_token_id"],
            "taps": [2, 10, 18, 26, 34],
            "tap_semantics": "native_layer_input_f32",
        },
        "generation_variant": schedule["generation_variant"],
        "chains": schedule["chains"],
        "shards": shards,
        "capture_plans": capture_plans,
        "shard_bytes": schedule["chunk_bytes"],
        "cache_slots": 2,
        "staging_slots": 1,
    }
    FrozenShardPlan(value)
    return value


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "request",
            "rebind",
            "freeze-schedule",
            "inspect",
            "publish",
            "metadata",
            "admit-replay",
            "merge",
            "admit-data",
        ),
    )
    parser.add_argument("--data", type=Path, help="JSON rotating data descriptor")
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--schedule-sha256")
    parser.add_argument("--plan-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publication", type=Path)
    parser.add_argument("--publication-sha256")
    parser.add_argument("--shard")
    parser.add_argument("--producer", type=Path)
    parser.add_argument("--producer-sha256")
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--metadata-max-bytes", type=int)
    parser.add_argument(
        "--records", type=Path, help="JSON list of manifest/admission pins to merge"
    )
    parser.add_argument("--preparation-data", type=Path)
    parser.add_argument("--preparation-data-sha256")
    parser.add_argument("--preparation-admission", type=Path)
    parser.add_argument("--preparation-admission-sha256")
    args = parser.parse_args()
    if args.operation == "freeze-schedule":
        _require(not args.output.exists(), "preserve frozen logical plan")
        value = freeze_capture_schedule(
            {"path": str(args.schedule), "sha256": args.schedule_sha256}, plan_root=args.plan_root
        )
        atomic_json(args.output, value)
        print(json.dumps({"plan": pin(args.output), "plan_sha256": identity(value)}))
        return
    _require(args.data is not None, "rotating data descriptor required")
    data = json.loads(args.data.read_text())
    provider = open_provider(data)
    try:
        if args.operation == "inspect":
            order = provider.order("train", 8101)
            result = {
                "plan_sha256": provider.sha256,
                "role_counts": provider.plan.role_counts,
                "initial_chain": order[0],
                "initial_shard": provider.plan.chain_shards[order[0]],
                "shards": len(provider.plan.shards),
                "model_loaded": False,
                "gpu_queried": False,
            }
        elif args.operation == "request":
            try:
                batches, _ = provider.reserve_batch(provider.cursor(), 2, require_teacher=True)
                result = {
                    "status": "READY",
                    "plan_sha256": provider.sha256,
                    "initial_chains": [batch.chain_id for batch in batches],
                    "model_loaded": False,
                    "gpu_queried": False,
                    "optimizer_updates": 0,
                }
            except ShardRequired as missing:
                shard = next(
                    s for s in missing.shard_ids if s not in provider.cache.state["resident"]
                )
                original = provider.cache.state["seals"].get(shard)
                result = {
                    "schema": "block_shard_restore_request_v1",
                    "plan": {k: data[k] for k in ("path", "sha256")},
                    "plan_sha256": provider.sha256,
                    "shard": shard,
                    "directory": str(provider.cache.root / shard),
                    "original_identity": None if original is None else original["identity"],
                    "original_metadata": None
                    if original is None
                    else original.get("original_metadata"),
                    "generation_variant": provider.plan.value["generation_variant"],
                    "capture_bytes_bound": provider.plan.value["shard_bytes"],
                    "replay_admission": data.get("replay_admission"),
                }
            if args.output:
                _require(not args.output.exists(), "preserve initial shard request")
                atomic_json(args.output, result)
        elif args.operation == "rebind":
            result = rebind_physical_manifest(
                provider.plan,
                {"path": str(args.preparation_data), "sha256": args.preparation_data_sha256},
                {
                    "path": str(args.preparation_admission),
                    "sha256": args.preparation_admission_sha256,
                },
                args.output,
            )
        elif args.operation == "merge":
            result = merge_preparation_manifests(
                provider.plan, json.loads(args.records.read_text()), args.output
            )
        elif args.operation == "admit-data":
            result = write_logical_admission(
                data,
                {"path": str(args.preparation_data), "sha256": args.preparation_data_sha256},
                {
                    "path": str(args.preparation_admission),
                    "sha256": args.preparation_admission_sha256,
                },
                args.output,
            )
        elif args.operation == "admit-replay":
            result = admit_replay(
                provider.cache,
                args.shard,
                {"path": str(args.publication), "sha256": args.publication_sha256},
                {"path": str(args.producer), "sha256": args.producer_sha256},
                args.output,
            )
        else:
            _require(
                provider.cache.state["owner"] is None,
                "metadata publisher cannot replace active GPU owner",
            )
            provider.cache.acquire("metadata-publisher", "restore")
            try:
                if args.operation == "publish":
                    publication = json.loads(
                        checked(
                            {"path": str(args.publication), "sha256": args.publication_sha256}
                        ).read_text()
                    )
                    _require(
                        publication.get("schema") == "block_shard_publication_v1"
                        and publication.get("status") == "PASS"
                        and publication.get("plan_sha256") == provider.sha256,
                        "actual global shard publication required",
                    )
                    provider.cache.publish(
                        publication["shard"],
                        publication["manifest"],
                        publication["admission"],
                        publication["directory"],
                    )
                    if args.metadata_root:
                        preserve_original_metadata(
                            provider.cache,
                            publication["shard"],
                            args.metadata_root,
                            max_bytes=args.metadata_max_bytes,
                        )
                    result = {"published": publication["shard"], "plan_sha256": provider.sha256}
                else:
                    result = preserve_original_metadata(
                        provider.cache,
                        args.shard,
                        args.metadata_root,
                        max_bytes=args.metadata_max_bytes,
                    )
            finally:
                # These CLI operations only copy/audit CPU files and launch no child/GPU.
                provider.cache.release(
                    "metadata-publisher",
                    released_proof={
                        "owned_process_groups_absent": True,
                        "owned_cuda_pids_absent": True,
                    },
                )
        print(json.dumps(result, sort_keys=True))
    finally:
        provider.close()


if __name__ == "__main__":
    main()
