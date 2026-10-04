"""CPU-only receipt importer checks with small frozen BF16 source fixtures.

Production hashes/shapes remain pinned. Tests patch only these identities to
small files and never claim their generated examples are captured TRAIN data.
"""

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, os.environ.get("EAGLE_GGUF_PY", str(ROOT / "third_party/llama.cpp/gguf-py")))
import create_fusion_binary_discrete_fixture as fixture  # noqa: E402
import fit_fusion_binary_discrete as fitter  # noqa: E402


class RealReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fusion-real-receipt-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.fixture = self.directory / "fixture"
        fixture.create_fixture(self.fixture, prompts=8, rows_per_prompt=3, width=16, outputs=3)
        self.manifest_path = self.fixture / "synthetic_manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        self.archive_path = self.fixture / self.manifest["operands"]
        with np.load(self.archive_path, allow_pickle=False) as z:
            self.arrays = {name: z[name].copy() for name in z.files}
        self.source = self.fixture / "original.safetensors"
        save_file(
            {"fc.weight": torch.from_numpy(self.arrays["reference_weight"]).bfloat16()}, self.source
        )
        self.base = self.fixture / "synthetic_base.gguf"
        self.source_hash, self.base_hash = fitter.sha256(self.source), fitter.sha256(self.base)
        self.manifest["synthetic"] = False
        self.manifest["source"]["frozen_weights_sha256"] = self.source_hash
        self.prompts = self.fixture / "train_prompts.jsonl"
        self.index = self.fixture / "train_index.json"
        self.capture = self.fixture / "captured_source.json"
        self.runtime = self.fixture / "native-runtime-manifest.json"
        self.runtime.write_text(json.dumps({"native_commit": "a" * 40}))
        prompts = []
        ownership = {}
        for row in self.manifest["rows"]:
            ownership[row["prompt_id"]] = {
                key: row[key] for key in ("prompt_id", "prompt_sha256", "split")
            }
        for prompt in ownership.values():
            messages = [{"role": "user", "content": "test original " + prompt["prompt_id"]}]
            content = json.dumps(
                messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
            prompt["prompt_sha256"] = hashlib.sha256(content).hexdigest()
            for row in self.manifest["rows"]:
                if row["prompt_id"] == prompt["prompt_id"]:
                    row["prompt_sha256"] = prompt["prompt_sha256"]
            prompt["group_id"] = "group:" + prompt["prompt_id"]
            prompts.append(
                {key: prompt[key] for key in ("prompt_id", "prompt_sha256", "group_id")}
                | {"source_split": "train"}
            )
        self.prompts.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": p["prompt_id"],
                        "messages": [
                            {"role": "user", "content": "test original " + p["prompt_id"]}
                        ],
                    }
                )
                for p in prompts
            )
            + "\n"
        )
        self.index.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": p["prompt_id"],
                        "group": p["group_id"],
                        "content_sha256": p["prompt_sha256"],
                    }
                )
                for p in prompts
            )
            + "\n"
        )
        self.capture.write_text(json.dumps({"synthetic_receipt_test_only": True}))
        capture_hash = fitter.sha256(self.capture)
        self.manifest["source"]["capture_manifest_sha256"] = capture_hash
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.inventory = self.fixture / "inventory.json"
        self.inventory.write_text(
            json.dumps(
                {
                    "schema": "fusion_binary_train_inventory_v1",
                    "source_train_prompts_sha256": fitter.sha256(self.prompts),
                    "source_train_index_sha256": fitter.sha256(self.index),
                    "prompts": prompts,
                }
            )
        )

        def file_record(p):
            return {"path": p.name, "sha256": fitter.sha256(p)}

        self.receipt = {
            "schema": "fusion_binary_train_provenance_v1",
            "manifest_sha256": fitter.sha256(self.manifest_path),
            "operands_sha256": fitter.sha256(self.archive_path),
            "source_weights": file_record(self.source),
            "base_gguf": file_record(self.base),
            "producer": {
                "contract": fitter.REAL_PRODUCER_CONTRACT,
                "source_files": {
                    p.name: fitter.sha256(p)
                    for p in (self.prompts, self.index, self.capture, self.runtime)
                },
                "native_revision": "a" * 40,
                "capture_manifest_sha256": capture_hash,
                "readiness_sha256": capture_hash,
            },
            "train_inventory": file_record(self.inventory),
            "selected_prompts": list(ownership.values()),
            "rows": [self.row_evidence(row, i) for i, row in enumerate(self.manifest["rows"])],
            "checks": dict.fromkeys(fitter.RECEIPT_CHECKS, True),
        }
        self.receipt_path = self.fixture / "receipt.json"
        self.pin_receipt()
        for name, value in {
            "FROZEN_SOURCE_WEIGHTS_SHA256": self.source_hash,
            "FROZEN_BASE_GGUF_SHA256": self.base_hash,
            "REAL_FUSION_SHAPE": (3, 16),
            "REAL_PROMPT_COUNTS": {"train": 6, "validation": 2},
            "REAL_ROWS_PER_PROMPT": 3,
        }.items():
            patcher = patch.object(fitter, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.config = fitter.load_config(ROOT / "configs/fusion_binary_discrete_a8.json")

    @staticmethod
    def row_evidence(row, i):
        prefix = [1, 2, 3][: row["position"] + 1]
        metadata = {
            "feature_row": i,
            "prompt_id": row["prompt_id"],
            "position": row["position"],
            "prefix_token_ids": prefix,
            "tap_ids": [2, 18, 33],
            "boundary": "native_target_block_inputs_concat_before_draft_fc",
            "source": "native_target_features_on_accepted_prefix",
            "accepted_prefix": True,
            "native_feature_row": 100 + i,
            "native_decode_ordinal": i,
            "task_id": i // 3,
        }
        decoded = {
            "schema": "eagle_target_feature_v1",
            "event": "decoded_row",
            "feature_row": 100 + i,
            "task_id": i // 3,
            "decode_ordinal": i,
            "position": row["position"],
            "prefix_token_ids": prefix,
            "target_layer_ids": [2, 18, 33],
            "boundary": "raw_target_layer_input_before_eagle_encoder",
            "source": "target_verifier",
            "feature_dtype": "float32_native_endian",
            "feature_dim": 16,
            "phase": "prefill",
        }
        return {
            "row_id": row["row_id"],
            "raw_input_sha256": row["raw_input_sha256"],
            "feature_row": i,
            "feature_metadata": metadata,
            "prompt_token_ids": prefix,
            "anchor": {
                "prompt_id": row["prompt_id"],
                "split": "train",
                "prefix_token_ids": [1, 2, 3],
                "round_index": 0,
                "seed_token_id": 4,
            },
            "native_events": {
                "decoded_row": decoded,
                "disposition": {
                    "schema": "eagle_target_feature_v1",
                    "event": "disposition",
                    "feature_row": 100 + i,
                    "task_id": i // 3,
                    "retained_input": True,
                    "reason": "accepted_prefix",
                },
            },
        }

    def pin_receipt(self):
        self.receipt_path.write_text(json.dumps(self.receipt))
        self.receipt_hash = fitter.sha256(self.receipt_path)

    def load(self, **extra):
        kwargs = {
            "source_weights_path": self.source,
            "base_gguf_path": self.base,
            "provenance_receipt_path": self.receipt_path,
            "provenance_receipt_sha256": self.receipt_hash,
        } | extra
        return fitter.load_operands(self.manifest_path, self.config, **kwargs)

    def test_receipt_verified_run_freezes_actual_control_scales(self):
        import argparse

        output = self.directory / "real-adapter-unit-result"
        report = fitter.run(
            argparse.Namespace(
                manifest=self.manifest_path,
                config=ROOT / "configs/fusion_binary_discrete_a8.json",
                source_weights=self.source,
                base_gguf=self.base,
                output_dir=output,
                provenance_receipt=self.receipt_path,
                provenance_receipt_sha256=self.receipt_hash,
            )
        )
        self.assertFalse(report["synthetic"])
        self.assertEqual(
            report["control_scales_sha256"], fitter.sha256(output / "control_scales.npz")
        )
        with np.load(output / "control_scales.npz", allow_pickle=False) as controls:
            self.assertEqual(set(controls.files), {"initializer_scale", "scale_only_scale"})
            self.assertEqual(controls["scale_only_scale"].dtype, np.float32)
            np.testing.assert_array_equal(
                controls["scale_only_scale"],
                np.array(
                    [detail["exported_scale_f32"] for detail in report["fit"]["control_solver"]],
                    dtype=np.float32,
                ),
            )
        self.assertEqual(report["export"]["unchanged_nonfusion_tensors"], 3)

    def test_verified_receipt_loads_exact_source_and_split(self):
        x, w, train, manifest, identity = self.load()
        np.testing.assert_array_equal(x, self.arrays["raw_input"])
        np.testing.assert_array_equal(w, self.arrays["reference_weight"])
        self.assertEqual(int(train.sum()), 18)
        self.assertFalse(manifest["synthetic"])
        self.assertEqual(identity["provenance_receipt_sha256"], self.receipt_hash)

    def test_missing_or_modified_external_pin_refuses(self):
        with self.assertRaisesRegex(ValueError, "independently pinned"):
            self.load(provenance_receipt_sha256=None)
        with self.assertRaisesRegex(ValueError, "external pin"):
            self.load(provenance_receipt_sha256="0" * 64)

    def test_false_receipt_check_and_wrong_producer_refuse(self):
        self.receipt["checks"]["capture_eligible"] = False
        self.pin_receipt()
        with self.assertRaisesRegex(ValueError, "every required"):
            self.load()
        self.receipt["checks"]["capture_eligible"] = True
        self.receipt["producer"]["contract"] = "unknown_or_already_quantized"
        self.pin_receipt()
        with self.assertRaisesRegex(ValueError, "producer/capture"):
            self.load()

    def test_source_file_or_inventory_changes_are_detected(self):
        self.capture.write_text("changed source")
        with self.assertRaisesRegex(ValueError, "producer source file"):
            self.load()

    def test_runtime_revision_is_bound_to_captured_binary_manifest(self):
        self.receipt["producer"]["native_revision"] = "b" * 40
        self.pin_receipt()
        with self.assertRaisesRegex(ValueError, "captured runtime manifest"):
            self.load()

    def test_selected_dev_prompt_cannot_enter_train_receipt(self):
        inventory = json.loads(self.inventory.read_text())
        inventory["prompts"][0]["source_split"] = "development"
        self.inventory.write_text(json.dumps(inventory))
        self.receipt["train_inventory"]["sha256"] = fitter.sha256(self.inventory)
        self.pin_receipt()
        with self.assertRaisesRegex(ValueError, "non-TRAIN"):
            self.load()

    def test_group_leakage_and_raw_row_substitution_refuse(self):
        self.receipt["selected_prompts"][-1]["group_id"] = self.receipt["selected_prompts"][0][
            "group_id"
        ]
        self.pin_receipt()
        with self.assertRaisesRegex(ValueError, "TRAIN inventory|prompt groups"):
            self.load()
        self.receipt["selected_prompts"][-1]["group_id"] = (
            "group:" + self.receipt["selected_prompts"][-1]["prompt_id"]
        )
        self.receipt["rows"][0]["raw_input_sha256"] = "0" * 64
        self.pin_receipt()
        with self.assertRaisesRegex(ValueError, "ancestry"):
            self.load()

    def test_native_disposition_and_ordered_feature_taps_are_required(self):
        self.receipt["rows"][0]["native_events"]["disposition"]["retained_input"] = False
        self.pin_receipt()
        with self.assertRaisesRegex(ValueError, "did not retain"):
            self.load()
        self.receipt["rows"][0]["native_events"]["disposition"]["retained_input"] = True
        self.receipt["rows"][0]["feature_metadata"]["tap_ids"] = [33, 18, 2]
        self.pin_receipt()
        with self.assertRaisesRegex(ValueError, "tap ordering"):
            self.load()

    def test_speculative_rejected_suffix_is_ineligible(self):
        events = self.receipt["rows"][0]["native_events"]
        events["decoded_row"].update(phase="speculative", spec_input_row=2, round_index=0)
        events["disposition"].update(accepted_drafts=1, round_index=0)
        self.pin_receipt()
        with self.assertRaisesRegex(ValueError, "accepted prefix"):
            self.load()

    def test_nonsource_weight_archive_cannot_self_authorize(self):
        self.arrays["reference_weight"][1, 1] += np.float32(0.01)
        np.savez(self.archive_path, **self.arrays)
        self.manifest["operands_sha256"] = fitter.sha256(self.archive_path)
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.receipt["operands_sha256"] = self.manifest["operands_sha256"]
        self.receipt["manifest_sha256"] = fitter.sha256(self.manifest_path)
        self.pin_receipt()
        with self.assertRaisesRegex(ValueError, "original frozen BF16"):
            self.load()

    def test_bound_receipt_rejects_extra_or_missing_evidence(self):
        original = copy.deepcopy(self.receipt)
        self.receipt["rows"][0]["anchor"] = {}
        self.pin_receipt()
        with self.assertRaisesRegex(ValueError, "ancestry"):
            self.load()
        self.receipt = original
        self.receipt["unknown"] = "ignored ancestry would be unsafe"
        self.pin_receipt()
        with self.assertRaisesRegex(ValueError, "schema"):
            self.load()


if __name__ == "__main__":
    unittest.main()
