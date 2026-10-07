"""CPU family publication contracts; injected teachers establish no GPU admission."""

import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
import capture_nine_model_train_data as capture  # noqa: E402
import test_nine_model_train_capture as capture_fixture  # noqa: E402
from test_dspark_full_pool_capture import PartialClient  # noqa: E402

from w1a1_eagle.block_data import PARTIAL_LABEL_POLICY, BlockDataset  # noqa: E402
from w1a1_eagle.block_shard_lifecycle import (  # noqa: E402
    FrozenShardPlan,
    freeze_capture_schedule,
    merge_preparation_manifests,
    pin,
    rebind_physical_manifest,
)


class FamilyScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows, chains = [], {}
        role_counts = {
            "train": {"prose": 3286, "code": 3285, "reasoning": 3285},
            "calibration_fit": dict.fromkeys(("prose", "code", "reasoning"), 32),
            "calibration_validation": dict.fromkeys(("prose", "code", "reasoning"), 16),
        }
        for split, domains in role_counts.items():
            for domain, count in domains.items():
                for _ in range(count):
                    cid = f"c{len(rows)}"
                    digest = hashlib.sha256(cid.encode()).hexdigest()
                    rows.append(
                        {
                            "prompt_id": cid,
                            "split": split,
                            "domain": domain,
                            "group_id": cid,
                            "content_sha256": digest,
                        }
                    )
                    chains[cid] = {
                        "chain_id": cid,
                        "prompt_id": cid,
                        "split": split,
                        "domain": domain,
                        "group_id": cid,
                        "prompt_sha256": digest,
                        "capture_bytes_bound": 1,
                    }
        selector = capture.write_json(self.root / "selector.json", {"selection": rows})
        native = capture.write_json(
            self.root / "capture.json",
            {
                "native": {"target": {"sha256": "a" * 64}, "binary": {"sha256": "b" * 64}},
                "runtime": {"sha256": "c" * 64},
                "generation": {"max_new_tokens": 512},
                "vocab_size": 11,
                "target_width": 4,
                "mask_token_id": 10,
            },
        )
        self.schedule = {
            "schema": "dspark_full_pool_capture_schedule_v1",
            "seed": 8101,
            "role_selector": selector,
            "chunks": [{"chunk_id": 0, "chain_ids": list(chains), "plans": {"remote": native}}],
            "source_contracts": {"remote": {"sha256": "d" * 64}},
            "generation_variant": {"kind": "fresh_greedy", "receipt_sha256": "e" * 64},
            "source_modules": {},
            "source_files": [],
            "chains": chains,
            "chunk_bytes": 8 * 1024**3,
        }

    def freeze(self, **kwargs):
        path = self.root / "schedule.json"
        path.write_text(json.dumps(self.schedule))
        return freeze_capture_schedule(pin(path), plan_root=self.root, **kwargs)

    def test_default_exactly_preserves_dspark_and_dflash_reuses_raw_authority(self):
        old_default = self.freeze()
        dspark = self.freeze(family="dspark")
        dflash = self.freeze(family="dflash")
        self.assertEqual(old_default, dspark)
        expected = copy.deepcopy(dspark)
        expected["geometry"]["family"] = "dflash"
        self.assertEqual(dflash, expected)
        self.assertNotEqual(FrozenShardPlan(dspark).sha256, FrozenShardPlan(dflash).sha256)
        self.assertEqual(
            FrozenShardPlan(dflash).role_counts,
            {"train": 9856, "calibration_fit": 96, "calibration_validation": 48},
        )

    def test_dflash_schedule_requires_explicit_dflash_and_unknown_family_fails(self):
        self.schedule["schema"] = "dflash_full_pool_capture_schedule_v1"
        with self.assertRaisesRegex(ValueError, "frozen full-pool"):
            self.freeze()
        self.assertEqual(self.freeze(family="dflash")["geometry"]["family"], "dflash")
        with self.assertRaisesRegex(ValueError, "family must be"):
            self.freeze(family="eagle")
        value = self.freeze(family="dflash")
        value["geometry"]["family"] = "eagle"
        with self.assertRaisesRegex(ValueError, "family must be"):
            FrozenShardPlan(value)


class FamilyPublicationTests(unittest.TestCase):
    def setUp(self):
        case = capture_fixture.CaptureTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.root = case.root
        case.plan.update(
            objective="exact_soft",
            teacher_logits_layout="indexed",
            label_policy=PARTIAL_LABEL_POLICY,
            capture_goldens=False,
        )
        case.plan["caps"].update(
            max_requests=18,
            max_request_bytes=100000,
            max_shard_bytes=100000,
            max_eagle_golden_tokens=24,
        )
        capture_pin = capture.write_json(self.root / "partial-plan.json", case.plan)
        self.logical = {
            "schema": "block_logical_shard_plan_v1",
            "geometry": {
                "family": "dflash",
                "vocab_size": case.plan["vocab_size"],
                "target_width": case.plan["target_width"],
                "mask_token_id": case.plan["mask_token_id"],
                "taps": [2, 10, 18, 26, 34],
                "tap_semantics": "native_layer_input_f32",
            },
            "generation_variant": {"kind": "fresh_greedy"},
            "chains": {"global-" + str(i): r for i, r in enumerate(case.plan["selection"])},
            "shards": {"s0": ["global-" + str(i) for i in range(9)]},
            "capture_plans": {"s0": {"remote": capture_pin}},
        }
        self.logical_path = self.root / "logical.json"
        self.request_path = self.root / "request.json"
        self.request = {
            "schema": "block_shard_restore_request_v1",
            "shard": "s0",
            "directory": str(self.root / "physical"),
            "original_identity": None,
            "generation_variant": self.logical["generation_variant"],
            "capture_bytes_bound": 8 * 1024**3,
        }
        self.kwargs = {
            "teacher_factory": PartialClient,
            "device_query": lambda: {"name": "fixture CUDA", "compute_capability": [7, 5]},
            "rss_query": lambda: 1,
            "available_query": lambda: 10**9,
        }
        self.save()

    def save(self):
        self.logical_path.write_text(json.dumps(self.logical))
        self.request.update(plan=pin(self.logical_path), plan_sha256=capture.identity(self.logical))
        self.request_path.write_text(json.dumps(self.request))

    def publish(self):
        output = self.root / "publication.json"
        capture.run_shard_publication(self.request_path, output, **self.kwargs)
        return json.loads(output.read_text())

    def test_dflash_admission_and_raw_tensor_identity_shared_with_dspark(self):
        result = self.publish()
        self.assertTrue(result["manifest"]["path"].endswith("dflash/manifest.json"))
        self.assertTrue(result["admission"]["path"].endswith("dflash/completed-admission.json"))
        physical = self.root / "physical"
        dflash = json.loads(Path(result["manifest"]["path"]).read_text())
        dspark = json.loads((physical / "dspark/manifest.json").read_text())
        self.assertEqual(dflash["family"], "dflash")
        self.assertEqual(dspark["family"], "dspark")
        for dc, sc in zip(dflash["chains"], dspark["chains"], strict=True):
            for kind in ("features", "logits", "native_receipt"):
                self.assertEqual(dc[kind], sc[kind])
            self.assertEqual(dc["tokens"]["sha256"], sc["tokens"]["sha256"])
        self.assertNotEqual(capture.identity(dflash), capture.identity(dspark))
        ds = BlockDataset(
            result["manifest"]["path"],
            expected_sha256=result["manifest"]["sha256"],
            admission_path=result["admission"]["path"],
            admission_sha256=result["admission"]["sha256"],
        )
        ds.close()

    def test_rebind_merge_preserve_dflash_and_reject_wrong_family_plan(self):
        result = self.publish()
        value = {
            **self.logical,
            "seed": 8101,
            "selector": {},
            "source": {"target_sha256": "a" * 64},
            "cache_slots": 2,
            "staging_slots": 1,
            "shard_bytes": 8 * 1024**3,
            "generation_variant": {"kind": "fresh_greedy", "receipt_sha256": "b" * 64},
        }
        value["chains"] = {
            cid: {**chain, "chain_id": cid, "group_id": cid, "capture_bytes_bound": 100000}
            for cid, chain in value["chains"].items()
        }
        plan = FrozenShardPlan(value, require_full_pool=False)
        rebound = rebind_physical_manifest(
            plan, result["manifest"], result["admission"], self.root / "rebound"
        )
        records = [{"manifest": rebound["data"], "admission": rebound["admission"]}]
        merged = merge_preparation_manifests(plan, records, self.root / "merged")
        self.assertEqual(json.loads(Path(merged["data"]["path"]).read_text())["family"], "dflash")
        self.assertEqual(merged["tensor_files_copied"], 0)
        value["geometry"]["family"] = "dspark"
        wrong_plan = FrozenShardPlan(value, require_full_pool=False)
        with self.assertRaisesRegex(ValueError, "rebind logical geometry differs: family"):
            rebind_physical_manifest(
                wrong_plan, result["manifest"], result["admission"], self.root / "wrong-rebound"
            )
        self.assertFalse((self.root / "wrong-rebound").exists())
        with self.assertRaisesRegex(ValueError, "merge logical geometry differs: family"):
            merge_preparation_manifests(wrong_plan, records, self.root / "wrong-merged")
        self.assertFalse((self.root / "wrong-merged/manifest.json").exists())

    def test_dflash_replay_restores_identical_manifest_and_fresh_admission(self):
        import replay_dspark_shard

        self.publish()
        physical = self.root / "physical"
        original_manifest = pin(physical / "dflash/manifest.json")
        sealed = json.loads((physical / "original-tensor-seal.json").read_text())["identity"]
        metastore = self.root / "metadata"
        files = {}
        for src in physical.rglob("*"):
            if src.is_file() and src.suffix in (".json", ".jsonl", ".npy"):
                relative = src.relative_to(physical)
                dest = metastore / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                files[str(relative)] = pin(dest)
        metadata = capture.write_json(
            self.root / "metadata.json",
            {
                "schema": "block_shard_original_metadata_v1",
                "plan_sha256": self.request["plan_sha256"],
                "shard": "s0",
                "original_directory": str(physical),
                "identity": sealed,
                "manifest_relative_path": "dflash/manifest.json",
                "admission_relative_path": "dflash/completed-admission.json",
                "files": files,
            },
        )
        shutil.rmtree(physical)  # This test owns its synthetic temporary directory.
        self.request.update(
            original_identity=sealed,
            original_metadata=metadata,
            replay_admission=None,
            replay_probe=True,
        )
        self.save()

        class ReplayFixture(PartialClient):
            def make(self, *args, **kwargs):
                (self.root.parent.parent / "client.py").write_bytes(
                    b"synthetic injected client source fixture"
                )
                return super().make(*args, **kwargs)

        output = self.root / "replay-publication.json"
        kwargs = {**self.kwargs, "teacher_factory": ReplayFixture}
        capture.run_shard_publication(self.request_path, output, **kwargs)
        result = json.loads(output.read_text())
        self.assertEqual(result["manifest"], original_manifest)
        join = json.loads(Path(result["reconstruction_join"]["path"]).read_text())
        self.assertTrue(join["tensor_identity_equal"])
        self.assertTrue(join["historical_receipt_bytes_restored"])
        self.assertEqual(join["original_identity"], sealed)
        ds = BlockDataset(
            result["manifest"]["path"],
            expected_sha256=result["manifest"]["sha256"],
            admission_path=result["admission"]["path"],
            admission_sha256=result["admission"]["sha256"],
        )
        self.assertEqual(ds.manifest["family"], "dflash")
        ds.close()
        self.assertLessEqual(replay_dspark_shard.physical_bytes(physical), 8 * 1024**3)
        # A DSpark logical plan cannot borrow DFlash's original metadata/admission.
        self.logical["geometry"]["family"] = "dspark"
        self.save()
        with patch.object(replay_dspark_shard, "replay") as replay:
            with self.assertRaisesRegex(ValueError, "family/geometry"):
                capture.run_shard_publication(self.request_path, self.root / "wrong-replay.json")
            replay.assert_not_called()

    def test_unknown_or_capture_geometry_mismatch_stops_before_teacher(self):
        for key, value in (("family", "eagle"), ("vocab_size", 123), ("taps", [2, 18, 33])):
            with self.subTest(key=key):
                old = self.logical["geometry"][key]
                self.logical["geometry"][key] = value
                self.save()
                with patch.object(capture, "run_capture") as teacher:
                    with self.assertRaises(ValueError):
                        self.publish()
                    teacher.assert_not_called()
                self.logical["geometry"][key] = old
                self.assertFalse((self.root / "physical").exists())

    def test_wrong_family_pinned_manifest_cannot_be_replayed(self):
        result = self.publish()
        metadata = capture.write_json(
            self.root / "wrong-metadata.json",
            {
                "manifest_relative_path": "dspark/manifest.json",
                "files": {"dspark/manifest.json": pin(self.root / "physical/dspark/manifest.json")},
            },
        )
        self.request.update(original_identity={"sealed": True}, original_metadata=metadata)
        self.save()
        import replay_dspark_shard

        with patch.object(replay_dspark_shard, "replay") as replay:
            with self.assertRaisesRegex(ValueError, "family/geometry"):
                capture.run_shard_publication(
                    self.request_path, self.root / "replay-publication.json"
                )
            replay.assert_not_called()
        self.assertEqual(pin(Path(result["manifest"]["path"])), result["manifest"])


if __name__ == "__main__":
    unittest.main()
