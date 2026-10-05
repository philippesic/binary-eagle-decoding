"""Synthetic opaque identity tests; no real heldout, models, or CUDA evidence."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
import prepare_nine_model_heldout_admission as preparer  # noqa: E402
import test_nine_model_endpoint_collection as development  # noqa: E402
from test_nine_model_endpoint_collection import pin, write  # noqa: E402

from w1a1_eagle import nine_model_endpoint_collection as c  # noqa: E402
from w1a1_eagle.nine_model_pipeline import CANDIDATES  # noqa: E402


class HeldoutTests(unittest.TestCase):
    def fixture(self, root):
        value, source, contexts, exports, validators = development.CollectionTests().fixture(root)
        protocol = json.loads(Path(source["protocol"]["path"]).read_text())
        protocol["split"] = "final"
        source["protocol"] = write(root, "final-protocol", protocol)
        row = {
            "id": "train-id",
            "group": "train-group",
            "source_id": "same-public-source",
            "source_row_id": "train-row",
            "content_sha256": "1" * 64,
            "domain": "prose",
        }
        final = dict(
            row,
            id="final-id",
            group="final-group",
            source_row_id="final-row",
            content_sha256="2" * 64,
        )
        records = {}
        for split, item in (("train", row), ("sealed_test", final)):
            prompts = write(root, split + ".jsonl", {"messages": "SEALED SENTINEL"})
            index = root / (split + ".index.jsonl")
            index.write_text(json.dumps(item) + "\n")
            records[split] = {
                "shards": [
                    {
                        "prompts": Path(prompts["path"]).name,
                        "prompts_sha256": prompts["sha256"],
                        "index": index.name,
                        "index_sha256": pin(index)["sha256"],
                        "prompts_count": 1,
                    }
                ]
            }
        corpus = write(
            root,
            "manifest.json",
            {"schema": "continuous_w1ax_prompt_manifest_v1", "files": records},
        )
        source["prompts"] = {
            "path": str(root / "sealed_test.jsonl"),
            "sha256": records["sealed_test"]["shards"][0]["prompts_sha256"],
        }
        for name in CANDIDATES:
            context = contexts[name]
            context["lane"]["config"] = write(root, name + "/config", {"fixture": name})
            if name.startswith("eagle"):
                capture = {
                    "split": "train",
                    "prompt_count": 1,
                    "source_positions": [0],
                    "source_corpus_manifest_sha256": corpus["sha256"],
                    "source_prompts_sha256": records["train"]["shards"][0]["prompts_sha256"],
                    "source_index_sha256": records["train"]["shards"][0]["index_sha256"],
                    "prompts_sha256": "3" * 64,
                }
                config = write(
                    root,
                    name + "/continuous",
                    {
                        "schema": "continuous_w1ax_experiment_v1",
                        "stages": {"corpus_manifest": corpus, "captures": [capture]},
                    },
                )
                provider = write(
                    root,
                    name + "/provider",
                    {
                        "schema": "w1ax_native_train_provider_v2",
                        "split": "train",
                        "training_eligible": True,
                        "prompt_count": 1,
                        "sha256": {"prompts": "3" * 64},
                    },
                )
                execution = write(
                    root,
                    name + "/stages/train-providers.json",
                    {
                        "schema": "w1ax_streaming_train_v2",
                        "split": "train",
                        "training_eligible": True,
                        "shards": [
                            {
                                "provider_manifest": provider["path"],
                                "provider_manifest_sha256": provider["sha256"],
                            }
                        ],
                    },
                )
                context["source_bindings"] = {"execution_manifest_sha256": execution["sha256"]}
                context["spec"] = {
                    "eagle_config": config,
                    "prepared": {"run_dir": str(root / name)},
                }
            else:
                inventory = write(
                    root,
                    name + "/inventory",
                    {
                        "schema": "block_train_inventory_v1",
                        "prompts": {row["id"]: {"sha256": row["content_sha256"], "split": "TRAIN"}},
                    },
                )
                manifest = write(
                    root,
                    name + "/data",
                    {
                        "train_inventory": inventory,
                        "chains": [
                            {
                                "split": "train",
                                "prompt_id": row["id"],
                                "prompt_sha256": row["content_sha256"],
                            }
                        ],
                    },
                )
                context["spec"] = {"data": manifest}
                context["source_bindings"] = {"data_manifest_sha256": manifest["sha256"]}
        selection = {
            "phase": "final",
            "selection_provenance": "agent_selected",
            "human_selected": False,
            "corpus_manifest": corpus,
            "split": "sealed_test",
            "shard": 0,
            "prompts": source["prompts"],
            "protocol": source["protocol"],
            "target": source["target"],
            "origins": c.final_origins(value, contexts, exports),
        }
        authority_path = root / "human-authority"
        instruction = "Evaluate the frozen final prompts after all training completes."
        authority_path.write_text("Synthetic historical user evidence: " + instruction + "\n")
        authority = {"phase": "final", "record": pin(authority_path), "instruction": instruction}
        selection["authorization"] = authority
        source["final_selection"] = selection
        source["final_set_authorized"] = True
        source["origin"] = {"evaluation": {"authorization": authority}}
        self.receipts(root, value, source, contexts, exports)
        return value, source, contexts, exports, validators

    def receipts(self, root, value, source, contexts, exports):
        for name in CANDIDATES:
            value["candidates"][name]["heldout_admission"] = write(
                root,
                name + "/final-admission",
                c.final_admission(value, source, contexts, exports, name),
            )

    def validate(self, value, source, validators, files=None):
        with patch(
            "prepare_nine_model_lane_endpoint.ORIGINAL_EAGLE_Q4_SHA256",
            value["controls"]["eagle"]["model"]["sha256"],
        ):
            return c.validate_collection(
                value,
                files=files,
                validators=validators,
                source_loader=lambda locator, files: source,
            )

    def test_source_preparation_and_import_leave_all_prompt_payloads_unopened(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            value, source, contexts, exports, validators = self.fixture(root)
            forbidden = {root / "sealed_test.jsonl", root / "train.jsonl"}
            original = Path.open

            def guarded(path, *args, **kwargs):
                self.assertNotIn(path, forbidden, "prompt payload opened")
                return original(path, *args, **kwargs)

            with patch.object(Path, "open", guarded):
                context = self.validate(value, source, validators)
                result = preparer.prepare(
                    dict(value, schema="nine_model_endpoint_collection_inputs_v1"),
                    root / "prepared",
                    validators=validators,
                    source_loader=lambda locator, files: source,
                )
            self.assertEqual(context["heldout"]["prompt_count"], 1)
            self.assertFalse(result["prompt_payload_read"])
            self.assertFalse(result["execution_allowed"])
            for name in CANDIDATES:
                self.assertEqual(
                    json.loads((root / "prepared" / (name + ".json")).read_text()),
                    c.final_admission(value, source, contexts, exports, name),
                )

    def test_boolean_missing_authority_stale_origins_and_caller_disjoint_rejected(self):
        for mutation in (
            "boolean",
            "authority",
            "phase",
            "model",
            "config",
            "export",
            "lane",
            "source",
            "target",
            "protocol",
            "prompts",
            "shard",
            "receipt",
            "caller",
            "training_only",
            "prohibition",
            "foreign_authority",
            "human_selected",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                value, source, contexts, exports, validators = self.fixture(root)
                selected = source["final_selection"]
                if mutation in {"training_only", "prohibition", "foreign_authority"}:
                    authority = selected["authorization"]
                    instruction = {
                        "training_only": "Start overnight QAT and continue healthy GPU runs.",
                        "prohibition": "Do not evaluate the final prompts.",
                        "foreign_authority": "Evaluate final prompts immediately.",
                    }[mutation]
                    authority["instruction"] = instruction
                    if mutation != "foreign_authority":
                        path = Path(authority["record"]["path"])
                        path.write_text(instruction)
                        authority["record"] = pin(path)
                    self.receipts(root, value, source, contexts, exports)
                elif mutation == "human_selected":
                    selected["human_selected"] = True
                elif mutation == "boolean":
                    del source["final_selection"]
                elif mutation == "authority":
                    source["origin"]["evaluation"]["authorization"] = {}
                elif mutation == "phase":
                    selected["phase"] = "development"
                elif mutation in ("model", "config", "export", "lane", "source"):
                    key = {
                        "export": "export_receipt",
                        "lane": "frozen_lane",
                        "source": "frozen_training_source",
                    }.get(mutation, mutation)
                    selected["origins"]["eagle_a8"][key] = {"forged": True}
                elif mutation in ("target", "protocol", "prompts"):
                    selected[mutation] = value["evaluation_source"]
                elif mutation == "shard":
                    selected["shard"] = True
                else:
                    name = "eagle_a8"
                    receipt = c.final_admission(value, source, contexts, exports, name)
                    if mutation == "receipt":
                        receipt["candidate"] = "eagle_a1"
                    else:
                        receipt["training_disjoint"] = True
                    value["candidates"][name]["heldout_admission"] = write(root, "forged", receipt)
                with self.assertRaises(ValueError):
                    self.validate(value, source, validators)

    def test_actual_train_overlap_and_index_roles_rejected(self):
        for field in (
            "id",
            "group",
            "content_sha256",
            "source_row_id",
            "role",
            "messages",
            "content",
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                value, source, contexts, exports, validators = self.fixture(root)
                index = root / "sealed_test.index.jsonl"
                row = json.loads(index.read_text())
                train = json.loads((root / "train.index.jsonl").read_text())
                row[field] = train[field] if field in train else "train"
                index.write_text(json.dumps(row) + "\n")
                corpus = json.loads((root / "manifest.json").read_text())
                corpus["files"]["sealed_test"]["shards"][0]["index_sha256"] = pin(index)["sha256"]
                locator = write(root, "manifest.json", corpus)
                selection = source["final_selection"]
                selection["corpus_manifest"] = locator
                # Preserve the corpus anchor to test overlap/index fields.
                for name in ("eagle_a8", "eagle_a1"):
                    config = json.loads(
                        Path(contexts[name]["spec"]["eagle_config"]["path"]).read_text()
                    )
                    config["stages"]["corpus_manifest"] = locator
                    config["stages"]["captures"][0]["source_corpus_manifest_sha256"] = locator[
                        "sha256"
                    ]
                    contexts[name]["spec"]["eagle_config"] = write(
                        root, name + "/continuous", config
                    )
                self.receipts(root, value, source, contexts, exports)
                with self.assertRaises(ValueError):
                    self.validate(value, source, validators)

    def test_pending_preparation_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            result = preparer.prepare(
                {"schema": "nine_model_endpoint_collection_inputs_v1"}, output, inspect_draft=True
            )
            self.assertEqual(result["status"], "PENDING")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
