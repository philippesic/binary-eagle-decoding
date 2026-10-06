import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from w1a1_eagle.block_data import (  # noqa: E402
    TAPS,
    BlockCursor,
    BlockDataset,
    crop_decode_history,
    file_sha256,
    import_capture_plan,
    token_sha256,
    validate_decode_history,
)


def fixture(root, family="dspark", with_logits=True):
    def save(name, value):
        path = root / name
        if isinstance(value, dict):
            path.write_text(json.dumps(value))
        else:
            np.save(path, value)
        return {"path": name, "sha256": file_sha256(path)}

    producer = {
        "kind": "synthetic_fixture",
        "native_revision": "a" * 40,
        "binary_sha256": "b" * 64,
        "target_sha256": "c" * 64,
        "target_precision": "F16",
        "runtime_sha256": "d" * 64,
        "hardware": "CPU fixture",
    }
    chains, prompts, receipt_chains = [], {}, {}
    for index, domain in enumerate(("prose", "code", "reasoning")):
        cid = f"chain{index}"
        prompt_hash = str(index) * 64
        tokens = save(f"tokens{index}.npy", np.arange(18, dtype=np.int64))
        features = save(
            f"features{index}.npy", np.arange(18 * 5 * 2, dtype=np.float32).reshape(18, 5, 2)
        )
        logits = (
            save(f"logits{index}.npy", np.arange(18 * 32, dtype=np.float32).reshape(18, 32))
            if with_logits
            else None
        )
        chain = {
            "chain_id": cid,
            "prompt_id": cid,
            "prompt_sha256": prompt_hash,
            "domain": domain,
            "split": "train",
            "prompt_length": 3,
            "tokens": tokens,
            "features": features,
            "logits": logits,
            "anchors": [2, 9],
            "native_receipt": None,
        }
        chains.append(chain)
        prompts[cid] = {"sha256": prompt_hash, "domain": domain, "split": "TRAIN"}
        receipt_chains[cid] = {
            "tokens_sha256": tokens["sha256"],
            "features_sha256": features["sha256"],
            "logits_sha256": logits["sha256"] if logits else None,
            "prompt_id": cid,
            "prompt_sha256": prompt_hash,
            "prompt_length": 3,
            "native_receipt_sha256": None,
        }
    inventory = save("inventory.json", {"schema": "block_train_inventory_v1", "prompts": prompts})
    receipt = save(
        "receipt.json",
        {
            "schema": "block_target_capture_receipt_v1",
            "producer": producer,
            "train_inventory_sha256": inventory["sha256"],
            "taps": list(TAPS),
            "tap_semantics": "native_layer_input_f32",
            "vocab_size": 32,
            "capture_mode": "target_only_autoregressive_train",
            "chains": receipt_chains,
        },
    )
    manifest = {
        "schema": "native_block_train_v1",
        "family": family,
        "vocab_size": 32,
        "target_width": 2,
        "mask_token_id": 31,
        "taps": list(TAPS),
        "tap_semantics": "native_layer_input_f32",
        "layout": "author_anchor_first",
        "producer": producer | {"receipt": receipt},
        "train_inventory": inventory,
        "chains": chains,
    }
    save("manifest.json", manifest)
    return root / "manifest.json"


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = fixture(self.root)

    def load(self, **kwargs):
        return BlockDataset(
            self.path, expected_sha256=file_sha256(self.path), allow_synthetic=True, **kwargs
        )

    def change(self, callback):
        manifest = json.loads(self.path.read_text())
        callback(manifest)
        self.path.write_text(json.dumps(manifest))

    def test_exact_context_noise_label_prefix_and_teacher_join(self):
        batch = self.load().load_block("chain0", 0, require_teacher=True)
        self.assertEqual(batch.context_features.shape, (2, 5, 2))
        self.assertEqual(batch.prefix_tokens, (0, 1, 2))
        np.testing.assert_array_equal(batch.labels, np.arange(3, 10))
        np.testing.assert_array_equal(batch.predecessor_ids, np.arange(2, 9))
        np.testing.assert_array_equal(batch.input_tokens, [2, 31, 31, 31, 31, 31, 31])
        np.testing.assert_array_equal(batch.positions, np.arange(2, 9))
        self.assertEqual(batch.attention_allowed.shape, (7, 9))
        self.assertTrue(batch.attention_allowed.all())
        self.assertTrue(batch.loss_mask.all())
        self.assertEqual(batch.teacher_prefix_sha256[6], token_sha256(np.arange(9)))
        np.testing.assert_array_equal(
            batch.teacher_logits[6], np.arange(18 * 32).reshape(18, 32)[8]
        )

    def test_released_dflash_also_author_anchor_first(self):
        self.change(lambda m: m.update(family="dflash"))
        self.assertTrue(self.load().load_block("chain0", 0).loss_mask.all())

    def test_whole_chain_cursor_exact_resume_and_epoch(self):
        dataset = self.load()
        cursor = dataset.cursor(seed=7)
        first, cursor = dataset.next_block(cursor)
        restored = BlockCursor(**json.loads(json.dumps(cursor.payload())))
        second, after = dataset.next_block(restored)
        self.assertEqual(first.chain_id, second.chain_id)
        self.assertEqual(second.block_index, 1)
        third, cursor = dataset.next_block(after)
        self.assertNotEqual(second.chain_id, third.chain_id)
        for _ in range(3):
            _, cursor = dataset.next_block(cursor)
        self.assertEqual(cursor.epoch, 1)
        self.assertEqual(cursor.chain_offset, 0)

    def test_cursor_foreign_dataset_refuses(self):
        dataset = self.load()
        cursor = dataset.cursor().payload() | {"dataset_sha256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "identity"):
            dataset.next_block(BlockCursor(**cursor))

    def test_teacher_memory_bound(self):
        with self.assertRaisesRegex(MemoryError, "bound"):
            self.load(max_teacher_bytes=100).load_block("chain0", 0)

    def test_missing_soft_teacher_is_explicit(self):
        self.path = fixture(self.root, with_logits=False)
        dataset = self.load()
        self.assertIsNone(dataset.load_block("chain0", 0).teacher_logits)
        with self.assertRaisesRegex(ValueError, "absent"):
            dataset.load_block("chain0", 0, require_teacher=True)

    def test_synthetic_never_admits_real_preparation(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            BlockDataset(self.path, expected_sha256=file_sha256(self.path))

    def test_artifact_corruption_refuses(self):
        (self.root / "features0.npy").write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.load()

    def test_train_inventory_is_required_not_id_prefix(self):
        self.change(lambda m: m["chains"][0].update(prompt_id="train:made-up"))
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.load()

    def test_pruned_vocab_or_wrong_taps_refuses(self):
        self.change(lambda m: m.update(taps=[2, 18, 33]))
        with self.assertRaisesRegex(ValueError, "taps"):
            self.load()

    def test_incomplete_horizon_refuses(self):
        self.change(lambda m: m["chains"][0].update(anchors=[2, 15]))
        with self.assertRaisesRegex(ValueError, "horizon"):
            self.load()

    def test_duplicate_prompt_content_across_splits_refuses(self):
        self.change(
            lambda m: m["chains"].append(
                m["chains"][0] | {"chain_id": "duplicate", "split": "calibration_fit"}
            )
        )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.load()

    def test_completed_admission_reuses_finite_hash_audit(self):
        dataset = self.load()
        audit_path = self.root / "admission.json"
        pin = dataset.write_admission(audit_path)
        original_hash = file_sha256

        def reject_raw_rehash(path):
            if str(path).endswith(".npy"):
                raise AssertionError("completed array audit was repeated")
            return original_hash(path)

        with patch("w1a1_eagle.block_data.file_sha256", side_effect=reject_raw_rehash):
            restored = BlockDataset(
                self.path,
                expected_sha256=file_sha256(self.path),
                allow_synthetic=True,
                admission_path=audit_path,
                admission_sha256=pin,
            )
            restored.load_block("chain0", 0)

    def test_completed_admission_refuses_stat_change(self):
        dataset = self.load()
        audit = self.root / "admission.json"
        pin = dataset.write_admission(audit)
        with (self.root / "logits0.npy").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "identity"):
            BlockDataset(
                self.path,
                expected_sha256=file_sha256(self.path),
                allow_synthetic=True,
                admission_path=audit,
                admission_sha256=pin,
            )

    def test_no_untrusted_audit_bypass(self):
        with self.assertRaisesRegex(ValueError, "unchecked"):
            self.load(verify_artifacts=False)

    def test_consume_refuses_post_constructor_artifact_mutation(self):
        dataset = self.load()
        with (self.root / "features0.npy").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "changed after"):
            dataset.load_block("chain0", 0)

    def test_decode_history_preserves_prefill_and_greedy_kv_partitions(self):
        history = [
            {"offset": 0, "count": 3, "phase": "prefill", "kv_reused_from_same_chain": False},
            {
                "offset": 3,
                "count": 1,
                "phase": "target_only_greedy",
                "kv_reused_from_same_chain": True,
            },
            {
                "offset": 4,
                "count": 1,
                "phase": "target_only_greedy",
                "kv_reused_from_same_chain": True,
            },
        ]
        self.assertEqual(validate_decode_history(history, 5), history)
        self.assertEqual(crop_decode_history(history, 4), history[:2])
        self.assertEqual(crop_decode_history(history, 2)[0]["count"], 2)
        with self.assertRaisesRegex(ValueError, "KV"):
            validate_decode_history(
                history[:1] + [history[1] | {"kv_reused_from_same_chain": False}], 4
            )


class NativeRawImportTests(unittest.TestCase):
    """Synthetic native-producer format; these tests never certify real capture."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        path = fixture(self.root)
        original = json.loads(path.read_text())
        plan_chains = []
        for chain in original["chains"]:
            records = {}
            for name in ("features", "logits"):
                array = np.load(self.root / chain[name]["path"])
                raw = self.root / f"{chain['chain_id']}-{name}.f32"
                array.tofile(raw)
                records[name] = {
                    "path": str(raw),
                    "sha256": file_sha256(raw),
                    "shape": list(array.shape),
                    "dtype": "float32",
                }
            ancestry = {
                k: chain[k] for k in ("prompt_id", "prompt_sha256", "domain", "prompt_length")
            }
            ancestry["source_split"] = "TRAIN"
            native = {
                "schema": "block_native_teacher_request_v1",
                "complete": True,
                "optimizer_updates": 0,
                "decode_history": [
                    {
                        "offset": 0,
                        "count": 18,
                        "phase": "prefill",
                        "kv_reused_from_same_chain": False,
                    }
                ],
                "teacher_context_reset_between_requests": True,
                "prefix_contract": "teacher_forced_exact_caller_token_ids",
                "prefix_freshness": "caller_current_student_prefix",
                "kv_type": "F16",
                "target_precision": "F16",
                "tap_ids": list(TAPS),
                "tokens": np.load(self.root / chain["tokens"]["path"]).tolist(),
                "target_sha256": "c" * 64,
                "producer_binary_sha256": "b" * 64,
                "producer_source_revision": "a" * 40,
                "client_source_sha256": "e" * 64,
                "producer_host": "synthetic-unit-fixture",
                "hardware": ["CPU fixture"],
                "chain_ancestry": ancestry,
                "features_shape": [18, 5, 2],
                "logits_mode": "all",
                "logits_shape": [18, 32],
                "files": records,
            }
            receipt = self.root / f"{chain['chain_id']}-native.json"
            receipt.write_text(json.dumps(native))
            plan_chains.append(
                {
                    k: chain[k]
                    for k in (
                        "chain_id",
                        "prompt_id",
                        "prompt_sha256",
                        "domain",
                        "split",
                        "prompt_length",
                        "anchors",
                    )
                }
                | {
                    "include_logits": True,
                    "native_receipt": {"path": str(receipt), "sha256": file_sha256(receipt)},
                }
            )
        runtime = self.root / "runtime.json"
        runtime.write_text(json.dumps({"scope": "synthetic unit fixture"}))
        self.plan = {
            "schema": "block_capture_plan_v1",
            "family": "dspark",
            "vocab_size": 32,
            "target_width": 2,
            "mask_token_id": 31,
            "train_inventory": original["train_inventory"],
            "runtime": {"path": str(runtime), "sha256": file_sha256(runtime)},
            "chains": plan_chains,
        }
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan))

    def import_(self, **kwargs):
        return import_capture_plan(
            self.plan_path,
            expected_sha256=file_sha256(self.plan_path),
            output_dir=self.root / "materialized",
            **kwargs,
        )

    def test_raw_producer_complete_prefix_full_vocab_memmap(self):
        path = self.import_()
        dataset = BlockDataset(path, expected_sha256=file_sha256(path))
        batch = dataset.load_block("chain0", 1, require_teacher=True)
        np.testing.assert_array_equal(batch.labels, np.arange(10, 17))
        self.assertEqual(batch.context_features.shape, (9, 5, 2))
        self.assertIsInstance(dataset._arrays["chain0"][1], np.memmap)

    def test_import_streams_audit_and_checks_existing_budget(self):
        checks, audits = [], []
        constructor = BlockDataset

        def audit(*args, **kwargs):
            self.assertTrue(kwargs["audit_only"])
            dataset = constructor(*args, **kwargs)
            self.assertFalse(dataset._arrays)
            audits.append(dataset)
            return dataset

        with patch("w1a1_eagle.block_data.BlockDataset", side_effect=audit):
            self.import_(budget_check=lambda: checks.append(True))
        self.assertEqual(len(audits), 1)
        self.assertGreater(len(checks), len(self.plan["chains"]) * 3)

    def test_raw_capture_import_bound_refuses(self):
        with self.assertRaisesRegex(MemoryError, "bound"):
            self.import_(max_capture_bytes=20)

    def test_changed_native_prefix_refuses(self):
        record = self.plan["chains"][0]["native_receipt"]
        path = Path(record["path"])
        native = json.loads(path.read_text())
        native["chain_ancestry"]["source_split"] = "FINAL"
        path.write_text(json.dumps(native))
        record["sha256"] = file_sha256(path)
        self.plan_path.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError, "ancestry"):
            self.import_()

    def test_last_only_logit_receipt_refuses_soft_chain(self):
        record = self.plan["chains"][0]["native_receipt"]
        path = Path(record["path"])
        native = json.loads(path.read_text())
        native["logits_mode"] = "last"
        path.write_text(json.dumps(native))
        record["sha256"] = file_sha256(path)
        self.plan_path.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError, "last-only"):
            self.import_()


if __name__ == "__main__":
    unittest.main()
