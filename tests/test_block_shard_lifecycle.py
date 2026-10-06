"""CPU controller/provider contracts; fake producers establish no native admission."""

import copy
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from w1a1_eagle.block_shard_lifecycle import (  # noqa: E402
    FrozenShardPlan,
    RotatingBlockProvider,
    ShardCache,
    ShardRequired,
    archive_shard,
    identity,
    pin,
    preserve_original_metadata,
    restore_archive,
    restore_requested_shards,
    tree_bytes,
)

PROOF = {"owned_process_groups_absent": True, "owned_cuda_pids_absent": True}


@dataclass(frozen=True)
class Batch:
    chain_id: str
    block_index: int
    dataset_sha256: str


class FakeDataset:
    def __init__(self, path, **kwargs):
        value = json.loads(Path(path).read_text())
        self.manifest = value
        self.chains = {c["chain_id"]: c for c in value["chains"]}
        self.closed = False
        # A real BlockDataset additionally checks native schema, array math,
        # immutable artifact bytes and fresh device/inode/mtime admission.
        if not kwargs.get("admission_path"):
            return
        self.admission = json.loads(Path(kwargs["admission_path"]).read_text())
        for record in self.admission["artifacts"]:
            stat = Path(record["path"]).stat()
            if stat.st_ino != record["inode"] or stat.st_mtime_ns != record["mtime_ns"]:
                raise ValueError("fresh host artifact admission differs")

    def load_block(self, cid, block, **kwargs):
        return Batch(cid, block, "physical")

    def write_admission(self, path):
        artifacts = []
        for chain in self.chains.values():
            for key in ("tokens", "features", "logits", "native_receipt"):
                stat = Path(chain[key]["path"]).stat()
                artifacts.append(
                    {"path": chain[key]["path"], "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns}
                )
        Path(path).write_text(json.dumps({"artifacts": artifacts}))

    def close(self):
        self.closed = True


def fixture():
    chains = {}
    for i in range(6):
        cid = f"c{i}"
        chains[cid] = {
            "chain_id": cid,
            "prompt_id": f"p{i}",
            "prompt_sha256": str(i) * 64,
            "domain": ("prose", "code", "reasoning")[i % 3],
            "split": "train",
            "group_id": "shared" if i < 2 else f"g{i}",
            "capture_bytes_bound": 1024,
        }
    return FrozenShardPlan(
        {
            "schema": "block_logical_shard_plan_v1",
            "selector": {},
            "source": {"target_sha256": "a" * 64},
            "geometry": {
                "family": "dspark",
                "vocab_size": 11,
                "target_width": 4,
                "mask_token_id": 10,
            },
            "generation_variant": {"kind": "verified_committed_tokens", "receipt_sha256": "b" * 64},
            "seed": 8101,
            "shard_bytes": 16 * 1024,
            "cache_slots": 2,
            "staging_slots": 1,
            "chains": chains,
            "shards": {"s0": ["c0", "c1"], "s1": ["c2", "c3"], "s2": ["c4", "c5"]},
        },
        require_full_pool=False,
    )


def publication(cache, shard, directory, *, changed=False):
    directory.mkdir(parents=True, exist_ok=True)
    chains, artifacts = [], []
    for cid in cache.plan.shards[shard]:
        chain = dict(cache.plan.chains[cid])
        chain.update(prompt_length=3, anchors=[2, 9, 16] if cid == "c0" else [2])
        for key in ("tokens", "features", "logits", "native_receipt"):
            path = directory / (cid + "-" + key)
            path.write_bytes(
                (cid + key + ("changed" if changed and key == "logits" else "")).encode()
            )
            chain[key] = pin(path)
            stat = path.stat()
            artifacts.append(
                {"path": str(path), "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns}
            )
        chains.append(chain)
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                **cache.plan.value["geometry"],
                "producer": cache.plan.value["source"],
                "chains": chains,
            }
        )
    )
    admission = directory / "admission.json"
    admission.write_text(json.dumps({"artifacts": artifacts}))
    return pin(manifest), pin(admission)


def archive(cache, shard, root):
    archive_shard(cache, shard, root / "archives", max_bytes=10000000, min_free_bytes=0)


class ShardLifecycleTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.plan = fixture()
        self.cache = ShardCache(self.plan, self.root / "cache", dataset_factory=FakeDataset)
        self.cache.acquire("root", "capture")
        self.provider = RotatingBlockProvider(self.plan, self.cache)
        self.addCleanup(self.cache.close)

    def publish(self, shard, name=None, changed=False):
        directory = self.cache.root / (name or shard)
        manifest, admission = publication(self.cache, shard, directory, changed=changed)
        self.cache.publish(shard, manifest, admission, directory)

    def test_metadata_boundary_preserves_identity_and_cursor(self):
        cursor = self.provider.cursor()
        with self.assertRaises(ShardRequired) as caught:
            self.provider.reserve_batch(cursor)
        request = caught.exception.reservation
        self.assertEqual(request["kind"], "metadata")
        self.assertEqual(request["cursor"], cursor.payload())
        self.assertEqual(self.provider.sha256, self.plan.sha256)
        self.publish(caught.exception.shard_ids[0])
        self.assertEqual(self.provider.sha256, self.plan.sha256)
        self.assertEqual(self.provider.restore_cursor(cursor.payload()), cursor)

    def test_batch2_distinct_groups_chronology_and_epoch_tail(self):
        self.publish("s0")
        self.publish("s1")
        # Retain the third seal metadata without retaining its physical tensor shard.
        archive(self.cache, "s1", self.root)
        self.cache.evict("s1")
        self.publish("s2")
        original = self.provider.cursor()
        progress = {cid: self.provider._length(cid) for cid in self.plan.chains}
        progress["c0"] = 1
        cursor = self.provider.restore_cursor({**original.payload(), "progress": {"0": progress}})
        for expected in (1, 2):
            try:
                blocks, following = self.provider.reserve_batch(cursor)
            except ShardRequired as needed:
                archive(self.cache, "s2", self.root)
                self.cache.evict("s2")
                self.publish("s1", "s1-restored")
                blocks, following = self.provider.reserve_batch(
                    cursor, reservation=needed.reservation
                )
            self.assertEqual(blocks[0].chain_id, "c0")
            self.assertEqual(blocks[0].block_index, expected)
            self.assertNotEqual(
                self.provider.group_id(blocks[0]), self.provider.group_id(blocks[1])
            )
            self.assertTrue(all(b.dataset_sha256 == self.plan.sha256 for b in blocks))
            cursor = following
        self.assertEqual(cursor.epoch, 1)
        self.assertIn("1", cursor.progress)
        self.assertEqual(self.provider.restore_cursor(cursor.payload()), cursor)

    def test_eviction_requires_recovery_and_live_student_cannot_evict(self):
        self.publish("s0")
        with self.assertRaisesRegex(ValueError, "no admitted recovery"):
            self.cache.evict("s0")
        archive(self.cache, "s0", self.root)
        self.cache.release("root", released_proof=PROOF)
        self.cache.acquire("student", "student")
        with self.assertRaisesRegex(ValueError, "live student"):
            self.cache.evict("s0")
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.cache.acquire("teacher", "capture")
        with self.assertRaisesRegex(ValueError, "release proof"):
            self.cache.release("student", released_proof={})

    def test_restore_refreshes_fingerprint_but_byte_change_fails_closed(self):
        self.publish("s0")
        archive(self.cache, "s0", self.root)
        original = copy.deepcopy(self.cache.state["seals"]["s0"]["identity"])
        self.cache.evict("s0")
        self.assertFalse((self.cache.root / "s0").exists())
        self.publish("s0", "new-host-inode")
        self.assertEqual(self.cache.state["seals"]["s0"]["identity"], original)
        self.cache.evict("s0")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            self.publish("s0", "bad-restored", changed=True)
        self.assertNotIn("s0", self.cache.state["resident"])
        self.assertTrue((self.cache.root / "bad-restored").exists(), "failure evidence retained")

    def test_consumed_seal_and_reserved_batch_cannot_change(self):
        self.publish("s0")
        self.publish("s1")
        # Keep all first-order chains known for a deterministic reserved batch.
        self.cache.state["seals"]["s2"] = {
            "identity": {cid: {"anchors": [2]} for cid in self.plan.shards["s2"]}
        }
        cursor = self.provider.cursor()
        try:
            _, consumed = self.provider.reserve_batch(cursor)
        except ShardRequired as needed:
            reservation = needed.reservation
            reservation["selected"][0][2] += 1
            with self.assertRaisesRegex(ValueError, "reserved batch changed"):
                self.provider.reserve_batch(cursor, reservation=reservation)
            # A consumed-seal resume must fail even if physical files remain.
            consumed = self.provider.restore_cursor(
                {
                    **cursor.payload(),
                    "consumed_seals": {"s0": identity(self.cache.state["seals"]["s0"]["identity"])},
                }
            )
        shard = next(iter(consumed.consumed_seals))
        self.cache.state["seals"][shard]["identity"][next(iter(self.cache.plan.shards[shard]))][
            "anchors"
        ] = [99]
        with self.assertRaisesRegex(ValueError, "seal changed"):
            self.provider.restore_cursor(consumed.payload())

    def test_failed_staging_is_charged_and_not_cleaned(self):
        failed = self.cache.root / "failed-staging"
        failed.write_bytes(b"x" * (3 * self.plan.value["shard_bytes"]))
        with self.assertRaisesRegex(ValueError, "cannot admit"):
            self.cache.before_staging(["s0"])
        self.assertTrue(failed.exists())

    def test_original_group_cannot_cross_shards_or_roles(self):
        value = copy.deepcopy(self.plan.value)
        value["shards"]["s0"][1], value["shards"]["s1"][0] = "c2", "c1"
        with self.assertRaisesRegex(ValueError, "canonical group crosses"):
            FrozenShardPlan(value, require_full_pool=False)
        value = copy.deepcopy(self.plan.value)
        value["chains"]["c1"]["split"] = "calibration_fit"
        with self.assertRaisesRegex(ValueError, "canonical group crosses"):
            FrozenShardPlan(value, require_full_pool=False)

    def test_compact_original_metadata_and_actual_archive_restoration(self):
        self.publish("s0")
        metadata = preserve_original_metadata(
            self.cache, "s0", self.root / "metadata", max_bytes=100000
        )
        preserved = json.loads(Path(metadata["path"]).read_text())
        self.assertIn("c0-tokens", preserved["files"])
        self.assertNotIn("c0-features", preserved["files"])
        # Fake receipt names lack a .json suffix; real producers' JSON receipts
        # are all retained. This fixture separately covers exact tensor archive.
        archive_shard(self.cache, "s0", self.root / "archive", max_bytes=100000, min_free_bytes=0)
        original = copy.deepcopy(self.cache.state["seals"]["s0"]["identity"])
        self.cache.evict("s0")
        request = self.root / "restore-request.json"
        request.write_text("{}")
        output = self.root / "restored-publication.json"
        restore_archive(self.cache, "s0", self.cache.root / "s0", output, request)
        receipt = json.loads(output.read_text())
        self.cache.publish("s0", receipt["manifest"], receipt["admission"], self.cache.root / "s0")
        self.assertEqual(self.cache.state["seals"]["s0"]["identity"], original)
        self.assertEqual(json.loads(Path(metadata["path"]).read_text()), preserved)

    def test_hardlink_payload_is_counted_once(self):
        import os

        path = self.cache.root / "payload"
        path.write_bytes(b"x" * 2000)
        before = tree_bytes(self.cache.root)
        os.link(path, self.cache.root / "identical-replay-alias")
        self.assertEqual(tree_bytes(self.cache.root), before)

    def test_unadmitted_replay_cannot_authorize_eviction(self):
        self.publish("s0")
        path = self.root / "pretend-replay.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "block_shard_recovery_v1",
                    "status": "PASS",
                    "kind": "admitted_deterministic_replay",
                    "plan_sha256": self.plan.sha256,
                    "shard": "s0",
                    "identity": self.cache.state["seals"]["s0"]["identity"],
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "not explicitly admitted"):
            self.cache.set_recovery("s0", pin(path))

    def test_serialized_controller_capture_phase_cost_and_restart(self):
        cursor = self.provider.cursor()
        with self.assertRaises(ShardRequired) as caught:
            self.provider.reserve_batch(cursor)
        needed = caught.exception
        events = []
        parent = self
        producer = self.root / "producer.py"
        producer.write_text("# CPU fake producer")
        command = {
            "producer": pin(producer),
            "argv": ["python", str(producer), "{request}", "{output}"],
        }
        config = {"plan": {}, "capture": command, "restore": command, "phase_wall_seconds": 20}

        class Runner:
            def run(self, argv, **kwargs):
                parent.assertEqual(events[-1], "release")
                parent.assertEqual(parent.cache.state["owner"]["phase"], "restore")
                request_path, output = Path(argv[-2]), Path(argv[-1])
                request = json.loads(request_path.read_text())
                directory = Path(request["directory"])
                manifest, admission = publication(parent.cache, request["shard"], directory)
                output.write_text(
                    json.dumps(
                        {
                            "schema": "block_shard_publication_v1",
                            "status": "PASS",
                            "plan_sha256": parent.plan.sha256,
                            "shard": request["shard"],
                            "directory": str(directory),
                            "manifest": manifest,
                            "admission": admission,
                            "request": pin(request_path),
                        }
                    )
                )
                events.append("capture")

        def release():
            events.append("release")
            return PROOF

        phases = restore_requested_shards(
            config,
            {
                "plan_sha256": self.plan.sha256,
                "shard_ids": list(needed.shard_ids),
                "reservation": needed.reservation,
            },
            cache=self.cache,
            run_dir=self.root,
            attempt=self.root,
            runner=Runner(),
            release=release,
            authorization=lambda: None,
            source={"producer": pin(producer)},
        )
        self.assertIsNone(self.cache.state["owner"])
        self.assertEqual(len(phases), 1)
        phase = json.loads(Path(phases[0]["path"]).read_text())
        self.assertEqual(phase["trainer_seconds_charged"], 0)
        self.assertGreaterEqual(phase["wall_seconds"], 0)
        restored = ShardCache(self.plan, self.cache.root, dataset_factory=FakeDataset)
        self.assertEqual(restored.state["seals"], self.cache.state["seals"])
        self.assertEqual(
            RotatingBlockProvider(self.plan, restored).restore_cursor(cursor.payload()), cursor
        )


if __name__ == "__main__":
    unittest.main()
