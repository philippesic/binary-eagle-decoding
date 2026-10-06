"""Real CPU array/admission/receipt bridges; no native GPU or quality claim."""

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
import test_block_data as fixtures  # noqa: E402

from w1a1_eagle.block_data import BlockDataset, file_sha256, import_capture_plan  # noqa: E402
from w1a1_eagle.block_shard_lifecycle import (  # noqa: E402
    FrozenShardPlan,
    RotatingBlockProvider,
    ShardCache,
    archive_shard,
    pin,
    rebind_physical_manifest,
    restore_archive,
)


class RealArrayBridgeTests(unittest.TestCase):
    def test_rebind_publish_archive_and_fresh_host_restore_preserve_originals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = fixtures.NativeRawImportTests(
                "test_raw_producer_complete_prefix_full_vocab_memmap"
            )
            raw.setUp()
            self.addCleanup(raw.doCleanups)
            directory = root / "cache" / "s0"
            shutil.copytree(raw.root, directory)
            for path in directory.rglob("*.json"):
                path.write_text(path.read_text().replace(str(raw.root), str(directory)))
            source_plan = json.loads((directory / "plan.json").read_text())
            for chain in source_plan["chains"]:
                chain["native_receipt"]["sha256"] = file_sha256(
                    Path(chain["native_receipt"]["path"])
                )
            (directory / "plan.json").write_text(json.dumps(source_plan))
            manifest_path = import_capture_plan(
                directory / "plan.json",
                expected_sha256=file_sha256(directory / "plan.json"),
                output_dir=directory / "imported",
            )
            original = BlockDataset(manifest_path, expected_sha256=file_sha256(manifest_path))
            admission = directory / "original-admission.json"
            original.write_admission(admission)
            frozen_chains = {}
            original_shas = {}
            for i, (cid, chain) in enumerate(original.chains.items()):
                logical = f"source-{i}"
                frozen_chains[logical] = {
                    key: chain[key] for key in ("prompt_id", "prompt_sha256", "domain", "split")
                }
                frozen_chains[logical].update(
                    chain_id=logical, group_id=f"group-{i}", capture_bytes_bound=10000
                )
                original_shas[logical] = {
                    key: chain[key]["sha256"]
                    for key in ("tokens", "features", "logits", "native_receipt")
                }
            geometry = {
                key: original.manifest[key]
                for key in (
                    "family",
                    "vocab_size",
                    "target_width",
                    "mask_token_id",
                    "taps",
                    "tap_semantics",
                )
            }
            original.close()
            plan = FrozenShardPlan(
                {
                    "schema": "block_logical_shard_plan_v1",
                    "seed": 8101,
                    "selector": {},
                    "source": {"target_sha256": "c" * 64},
                    "geometry": geometry,
                    "generation_variant": {
                        "kind": "verified_committed_tokens",
                        "receipt_sha256": "a" * 64,
                    },
                    "shard_bytes": 1024 * 1024,
                    "cache_slots": 2,
                    "staging_slots": 1,
                    "chains": frozen_chains,
                    "shards": {"s0": list(frozen_chains)},
                },
                require_full_pool=False,
            )
            rebound = rebind_physical_manifest(
                plan, pin(manifest_path), pin(admission), directory / "logical"
            )
            cache = ShardCache(plan, root / "cache")
            cache.acquire("CPU-test", "restore")
            cache.publish("s0", rebound["data"], rebound["admission"], directory)
            seal = copy.deepcopy(cache.state["seals"]["s0"]["identity"])
            for cid, record in seal.items():
                self.assertEqual(record["artifacts"], original_shas[cid])
            provider = RotatingBlockProvider(plan, cache)
            blocks, cursor = provider.reserve_batch(provider.cursor(), 2)
            self.assertNotEqual(provider.group_id(blocks[0]), provider.group_id(blocks[1]))
            self.assertTrue(all(b.dataset_sha256 == plan.sha256 for b in blocks))
            archive_shard(cache, "s0", root / "cold", max_bytes=1024 * 1024, min_free_bytes=0)
            cache.evict("s0")
            request = root / "request.json"
            request.write_text("{}")
            publication = root / "publication.json"
            restore_archive(cache, "s0", directory, publication, request)
            value = json.loads(publication.read_text())
            cache.publish("s0", value["manifest"], value["admission"], directory)
            self.assertEqual(cache.state["seals"]["s0"]["identity"], seal)
            self.assertEqual(provider.restore_cursor(cursor.payload()), cursor)
            # Mutating a restored tensor after its new admission fails closed.
            restored = json.loads(Path(value["manifest"]["path"]).read_text())
            tensor = Path(restored["chains"][0]["features"]["path"])
            with tensor.open("r+b") as stream:
                stream.write(b"corrupt!")
            cache.close()
            with self.assertRaisesRegex(ValueError, "identity differs"):
                cache.dataset("s0")


if __name__ == "__main__":
    unittest.main()
