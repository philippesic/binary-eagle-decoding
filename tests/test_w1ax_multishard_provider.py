"""Tiny file and CPU double tests for one-optimizer shard streaming."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_w1ax_multishard_manifest import prepare  # noqa: E402
from test_recurrent_provider import TinyStep, dense_drafter, provider_round  # noqa: E402
from w1ax_capture_provider import sha256  # noqa: E402
from w1ax_multishard_provider import (  # noqa: E402
    COMMON_HASHES,
    MultiShardNativeProvider,
    ids_sha256,
)

from w1a1_eagle.recurrent_provider import train_from_provider  # noqa: E402
from w1a1_eagle.recurrent_qat import (  # noqa: E402
    JointQATConfig,
    W1AxContract,
    save_joint_checkpoint,
)
from w1a1_eagle.recurrent_trace import RoundAnchor  # noqa: E402


class ChildDouble:
    def __init__(self, ordinal, *, bad_model=False):
        self.ordinal = ordinal
        self.capture_id = f"capture-{ordinal}"
        self.allowed_prompt_ids = {f"p{ordinal}"}
        self.target_vocab_size = 4
        self.draft_vocab_size = 3
        self.max_depth = 2
        self.d2t_offsets = (0, 1, 1)
        self.base_gguf_sha256 = "a" * 64
        self.candidate_d = None
        self.total_rounds = 1
        self.load_count = 0
        self.hashes = {key: "a" * 64 for key in COMMON_HASHES}
        self.hashes["capture_manifest"] = f"{ordinal + 1:064x}"
        if bad_model:
            self.hashes["target_gguf"] = "b" * 64

    def load_models(self):
        self.load_count += 1
        return dense_drafter(), nn.Linear(4, 4)

    def make_step_adapter(self, drafter):
        return TinyStep(drafter)

    def rounds(self):
        original = provider_round()
        prompt = f"p{self.ordinal}"
        rows = tuple(dict(row, prompt_id=prompt) for row in original.rows)
        yield replace(
            original,
            anchor=RoundAnchor(prompt, "train", 0, (0,), 1),
            rows=rows,
            capture_id=self.capture_id,
            teacher_row_ids=(),
            teacher_metadata=(),
            teacher_arrays=None,
        )


def tiny_multi_factory(config, manifest_path):
    def child_factory(config, path):
        ordinal = int(Path(path).stem.split("-")[-1])
        return ChildDouble(ordinal)

    return MultiShardNativeProvider(config, manifest_path, child_factory=child_factory)


class MultiShardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent_dir = self.root / "parent"
        self.parent_dir.mkdir()
        self.parent_manifest = self.parent_dir / "manifest.json"
        self.parent_manifest.write_text(json.dumps({"schema": "frozen-train-double"}))
        prompts = self.parent_dir / "train_small.jsonl"
        prompts.write_text("".join(json.dumps({"id": f"p{i}"}) + "\n" for i in range(2)))
        index = self.parent_dir / "train_small.index.jsonl"
        index.write_text("".join(json.dumps({"id": f"p{i}"}) + "\n" for i in range(2)))
        self.parent = {
            "manifest_sha256": sha256(self.parent_manifest),
            "split": "train_small",
            "prompts_filename": prompts.name,
            "prompts_sha256": sha256(prompts),
            "index_filename": index.name,
            "index_sha256": sha256(index),
            "count": 2,
        }
        self.plan_dir = self.root / "plan"
        self.plan_dir.mkdir()
        records = []
        child_provider_paths = []
        for i in range(2):
            directory = self.plan_dir / f"shard-{i:04d}"
            directory.mkdir()
            shard_prompts = directory / "train_prompts.jsonl"
            shard_prompts.write_text(json.dumps({"id": f"p{i}"}) + "\n")
            child = {
                "schema": "w1ax_capture_shard_v1",
                "parent": self.parent,
                "ordinal": i,
                "prompt_ids": [f"p{i}"],
                "prompt_count": 1,
                "source_positions": [i],
                "prompts_path": shard_prompts.name,
                "prompts_sha256": sha256(shard_prompts),
            }
            child_path = directory / "manifest.json"
            child_path.write_text(json.dumps(child))
            records.append(
                {
                    "ordinal": i,
                    "manifest": f"shard-{i:04d}/manifest.json",
                    "manifest_sha256": sha256(child_path),
                    "prompts_sha256": sha256(shard_prompts),
                    "prompt_ids": [f"p{i}"],
                    "prompt_ids_sha256": ids_sha256([f"p{i}"]),
                    "prompt_count": 1,
                }
            )
            provider_path = self.root / f"provider-{i}.json"
            provider_path.write_text(
                json.dumps(
                    {
                        "schema": "w1ax_native_train_provider_v1",
                        "training_eligible": True,
                        "prompt_count": 1,
                        "sha256": {"prompts": sha256(shard_prompts)},
                        "paths": {"prompts": str(shard_prompts)},
                    }
                )
            )
            child_provider_paths.append(provider_path)
        self.plan_path = self.plan_dir / "plan.json"
        self.plan_path.write_text(
            json.dumps(
                {
                    "schema": "w1ax_capture_shard_plan_v1",
                    "parent": self.parent,
                    "parent_prompt_ids": ["p0", "p1"],
                    "parent_prompt_ids_sha256": ids_sha256(["p0", "p1"]),
                    "shard_count": 2,
                    "total_prompts": 2,
                    "target_vocab_size": 4,
                    "shards": records,
                }
            )
        )
        self.children_index = self.root / "children.jsonl"
        self.children_index.write_text(
            "".join(
                json.dumps({"ordinal": i, "provider_manifest": str(path)}) + "\n"
                for i, path in enumerate(child_provider_paths)
            )
        )
        self.execution_path = self.root / "execution.json"
        prepare(self.plan_path, self.parent_manifest, self.children_index, self.execution_path)
        self.children = []

    def factory(self, config, path):
        ordinal = int(Path(path).stem.split("-")[-1])
        child = ChildDouble(ordinal)
        self.children.append(child)
        return child

    def test_ordered_full_coverage_one_model_one_optimizer_one_checkpoint(self):
        config = JointQATConfig(W1AxContract(4))
        provider = MultiShardNativeProvider(config, self.execution_path, child_factory=self.factory)
        self.assertEqual(provider.total_rounds, 2)
        self.assertEqual(provider.source_metadata["shard_count"], 2)
        self.assertEqual([x["ordinal"] for x in provider.source_metadata["shards"]], [0, 1])
        linears, metrics = train_from_provider(provider, config, max_rounds=2)
        self.assertEqual([item["shard_ordinal"] for item in metrics], [0, 1])
        self.assertTrue(all(item["gradient_tensors"] == 18 for item in metrics))
        self.assertEqual([child.load_count for child in self.children], [1, 0])
        checkpoint = self.root / "joint.npz"
        manifest = self.root / "joint.json"
        save_joint_checkpoint(linears, config, "a" * 64, checkpoint, manifest)
        self.assertTrue(checkpoint.is_file())
        self.assertTrue(manifest.is_file())

    def test_cli_all_rounds_saves_one_complete_provenance_report(self):
        output = self.root / "run"
        command = [
            sys.executable,
            str(ROOT / "scripts/train_joint_w1ax.py"),
            "--provider",
            "test_w1ax_multishard_provider:tiny_multi_factory",
            "--provider-manifest",
            str(self.execution_path),
            "--scale-layout",
            "row",
            "--activation-bits",
            "4",
            "--objective",
            "hard_ce",
            "--device",
            "cpu",
            "--all-rounds",
            "--require-complete",
            "--output-dir",
            str(output),
        ]
        env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{ROOT / 'scripts'}:{ROOT / 'tests'}")
        result = subprocess.run(
            command, cwd=ROOT, env=env, text=True, capture_output=True, check=True
        )
        report = json.loads((output / "training_run.json").read_text())
        self.assertTrue(report["training_complete"])
        self.assertEqual(report["steps"], 2)
        self.assertEqual([m["shard_ordinal"] for m in report["metrics"]], [0, 1])
        self.assertEqual(report["source"]["shard_count"], 2)
        self.assertIn("execution_manifest_sha256", report["source"])
        self.assertTrue((output / "joint.npz").is_file())
        self.assertEqual(json.loads(result.stdout)["steps"], 2)

    def test_attachment_order_and_plan_coverage_fail_before_model_load(self):
        contents = json.loads(self.execution_path.read_text())
        contents["shards"][0]["ordinal"] = 1
        self.execution_path.write_text(json.dumps(contents))
        with self.assertRaisesRegex(ValueError, "ordinals"):
            MultiShardNativeProvider(
                JointQATConfig(W1AxContract(4)),
                self.execution_path,
                child_factory=self.factory,
            )
        self.assertEqual(self.children, [])

    def test_common_model_hash_mismatch_rejected(self):
        def mismatch_factory(config, path):
            ordinal = int(Path(path).stem.split("-")[-1])
            return ChildDouble(ordinal, bad_model=ordinal == 1)

        with self.assertRaisesRegex(ValueError, "model/map contract"):
            MultiShardNativeProvider(
                JointQATConfig(W1AxContract(4)),
                self.execution_path,
                child_factory=mismatch_factory,
            )

    def test_changed_parent_coverage_rejected_before_child_audit(self):
        plan = json.loads(self.plan_path.read_text())
        plan["parent_prompt_ids"] = ["p0", "p0"]
        plan["parent_prompt_ids_sha256"] = ids_sha256(plan["parent_prompt_ids"])
        self.plan_path.write_text(json.dumps(plan))
        execution = json.loads(self.execution_path.read_text())
        execution["plan_sha256"] = sha256(self.plan_path)
        self.execution_path.write_text(json.dumps(execution))
        with self.assertRaisesRegex(ValueError, "parent prompt order or coverage"):
            MultiShardNativeProvider(
                JointQATConfig(W1AxContract(4)),
                self.execution_path,
                child_factory=self.factory,
            )
        self.assertEqual(self.children, [])


if __name__ == "__main__":
    unittest.main()
