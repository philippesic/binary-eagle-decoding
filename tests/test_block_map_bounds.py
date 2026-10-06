"""CPU mapping bounds and data lifetime evidence; no target-device admission."""

import dataclasses
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_block_data import fixture

from w1a1_eagle.block_data import BlockDataset, block_teacher_indices, file_sha256


def indexed_fixture(root, count=3, vocab_size=32):
    path = fixture(root)
    manifest = json.loads(path.read_text())
    producer = manifest["producer"]
    receipt_path = root / producer["receipt"]["path"]
    receipt = json.loads(receipt_path.read_text())
    inventory_path = root / manifest["train_inventory"]["path"]
    inventory = json.loads(inventory_path.read_text())
    templates = manifest["chains"]
    chains, prompts, records = [], {}, {}
    indices = block_teacher_indices([2, 9])
    for index in range(count):
        template = templates[index % 3]
        cid = f"indexed{index}"
        chain = template | {
            "chain_id": cid,
            "prompt_id": cid,
            "prompt_sha256": hashlib.sha256(cid.encode()).hexdigest(),
            "logits_indices": indices,
        }
        for name in ("tokens", "features", "logits"):
            value = (
                np.full((len(indices), vocab_size), index / 1000, dtype=np.float32)
                if name == "logits"
                else np.load(root / template[name]["path"])
            )
            array_path = root / f"{cid}-{name}.npy"
            np.save(array_path, value)
            chain[name] = {"path": array_path.name, "sha256": file_sha256(array_path)}
        chains.append(chain)
        prompts[cid] = {
            "sha256": chain["prompt_sha256"],
            "domain": chain["domain"],
            "split": "TRAIN",
        }
        records[cid] = receipt["chains"][template["chain_id"]] | {
            "prompt_id": cid,
            "prompt_sha256": chain["prompt_sha256"],
            "tokens_sha256": chain["tokens"]["sha256"],
            "features_sha256": chain["features"]["sha256"],
            "logits_sha256": chain["logits"]["sha256"],
            "logits_indices": indices,
        }
    inventory["prompts"] = prompts
    inventory_path.write_text(json.dumps(inventory))
    manifest["train_inventory"]["sha256"] = file_sha256(inventory_path)
    receipt.update(
        chains=records, vocab_size=vocab_size, train_inventory_sha256=file_sha256(inventory_path)
    )
    receipt_path.write_text(json.dumps(receipt))
    producer["receipt"]["sha256"] = file_sha256(receipt_path)
    manifest.update(chains=chains, vocab_size=vocab_size)
    path.write_text(json.dumps(manifest))
    return path


def load(path, **kwargs):
    return BlockDataset(path, expected_sha256=file_sha256(path), allow_synthetic=True, **kwargs)


def stress_child():
    """Fresh process, real mmap/FD/RSS evidence under a 64-descriptor cap."""
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    fd_dir = "/proc/self/fd" if sys.platform.startswith("linux") else "/dev/fd"
    peak_fds = 0

    def budget():
        nonlocal peak_fds
        peak_fds = max(peak_fds, len(os.listdir(fd_dir)))

    def peak_rss():
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == "darwin" else value * 1024

    with tempfile.TemporaryDirectory() as directory:
        path = indexed_fixture(Path(directory), count=450, vocab_size=4096)
        baseline = peak_rss()
        dataset = load(path, audit_only=True, budget_check=budget)
        assert not dataset._arrays
        admission = Path(directory) / "admission.json"
        pin = dataset.write_admission(admission)
        summary = dataset.storage_summary()
        training = load(path, admission_path=admission, admission_sha256=pin)
        for cid in training.chains:
            batch = training.load_block(cid, 1, require_teacher=True)
            assert batch.context_features.shape == (9, 5, 2)
            assert batch.teacher_logits.shape == (7, 4096)
            assert len(training._arrays) == 1
            budget()
        training.close()
        growth = max(0, peak_rss() - baseline)
        assert peak_fds < 20, peak_fds
        assert growth < 64 * 1024**2, growth
        # Mutate a late chain after eviction: remapping must fail closed.
        last = training.chains["indexed449"]
        with (Path(directory) / last["logits"]["path"]).open("ab") as stream:
            stream.write(b"changed")
        try:
            training.load_block("indexed449", 0)
        except ValueError as error:
            assert "changed after" in str(error)
        else:
            raise AssertionError("late remap mutation admitted")
        # Re-pin a late NaN to prove the finite check also visits chain 450.
        manifest = json.loads(path.read_text())
        late = manifest["chains"][-1]
        logits_path = Path(directory) / late["logits"]["path"]
        logits = np.load(logits_path)
        logits[-1, -1] = np.nan
        np.save(logits_path, logits)
        late["logits"]["sha256"] = file_sha256(logits_path)
        receipt_path = Path(directory) / manifest["producer"]["receipt"]["path"]
        receipt = json.loads(receipt_path.read_text())
        receipt["chains"][late["chain_id"]]["logits_sha256"] = late["logits"]["sha256"]
        receipt_path.write_text(json.dumps(receipt))
        manifest["producer"]["receipt"]["sha256"] = file_sha256(receipt_path)
        path.write_text(json.dumps(manifest))
        try:
            load(path, audit_only=True)
        except ValueError as error:
            assert "nonfinite" in str(error)
        else:
            raise AssertionError("late audit NaN admitted")
        print(
            json.dumps(
                {
                    "chains": 450,
                    "rlimit_nofile": 64,
                    "peak_fds": peak_fds,
                    "peak_rss_growth_bytes": growth,
                    "storage": summary,
                }
            )
        )


class MapBoundsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = indexed_fixture(Path(self.temp.name))

    def test_audit_only_admission_and_use_prohibition(self):
        audited = load(self.path, audit_only=True)
        usable = load(self.path)
        self.assertEqual(
            audited.storage_summary()["teacher_bytes"], usable.storage_summary()["teacher_bytes"]
        )
        a, b = Path(self.temp.name) / "a.json", Path(self.temp.name) / "b.json"
        self.assertEqual(audited.write_admission(a), usable.write_admission(b))
        self.assertEqual(a.read_bytes(), b.read_bytes())
        with self.assertRaisesRegex(ValueError, "audit-only"):
            audited.load_block("indexed0", 0)
        with self.assertRaisesRegex(ValueError, "audit-only"):
            audited.copy_feature_rows("indexed0", [0])

    def test_production_vocabulary_keeps_every_f32_teacher_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = indexed_fixture(Path(directory), vocab_size=151936)
            dataset = load(path)
            self.addCleanup(dataset.close)
            batch = dataset.load_block("indexed2", 1, require_teacher=True)
            self.assertEqual(batch.teacher_logits.shape, (7, 151936))
            self.assertEqual(batch.teacher_logits.dtype, np.dtype("float32"))
            self.assertEqual(batch.teacher_logits.nbytes, 7 * 151936 * 4)
            self.assertTrue(batch.context_features.flags.owndata)
            self.assertTrue(batch.teacher_logits.flags.owndata)
            dataset.load_block("indexed1", 0)
            dataset.close()
            np.testing.assert_array_equal(
                batch.teacher_logits, np.full((7, 151936), 0.002, dtype=np.float32)
            )

    def test_cursors_batches_and_torch_gradients_survive_eviction_and_close(self):
        import torch

        from w1a1_eagle.block_qat import BlockQATConfig, block_loss

        bounded, retained = load(self.path), load(self.path, max_mapped_chains=3)
        self.addCleanup(bounded.close)
        self.addCleanup(retained.close)
        held = bounded.load_block("indexed0", 1, require_teacher=True)
        reference = retained.load_block("indexed0", 1, require_teacher=True)
        context_tensor = torch.from_numpy(held.context_features)
        teacher_tensor = torch.from_numpy(held.teacher_logits)
        before = context_tensor.clone()
        bounded.load_block("indexed1", 0)
        bounded.load_block("indexed2", 0)
        self.assertEqual(list(bounded._arrays), ["indexed2"])
        bounded.close()
        self.assertTrue(torch.equal(context_tensor, before))
        self.assertTrue(torch.equal(teacher_tensor, torch.from_numpy(reference.teacher_logits)))
        for field in dataclasses.fields(held):
            left, right = getattr(held, field.name), getattr(reference, field.name)
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            else:
                self.assertEqual(left, right)
        config = BlockQATConfig(
            "dspark",
            8,
            vocab_size=32,
            mask_token_id=31,
            objective="full_probability_l1",
            depth_decay=0.9,
        )
        prediction = torch.arange(224, dtype=torch.float32).reshape(7, 32) / 100
        for divergence in range(8):
            predecessor = torch.from_numpy(held.predecessor_ids.copy())
            if divergence < 7:
                predecessor[divergence] += 1
            for bits in (127, 85, 42, 0):
                mask = torch.tensor([bool(bits & (1 << i)) for i in range(7)])
                outputs = []
                for batch in (held, reference):
                    logits = prediction.clone().requires_grad_()
                    tb = SimpleNamespace(
                        labels=torch.from_numpy(batch.labels),
                        loss_mask=mask,
                        predecessor_ids=torch.from_numpy(batch.predecessor_ids),
                        teacher_logits=torch.from_numpy(batch.teacher_logits),
                    )
                    out = SimpleNamespace(
                        logits=logits, predecessor_ids=predecessor, conditioning="native_greedy"
                    )
                    if not any(bool(mask[i]) for i in range(min(divergence, 7))):
                        with self.assertRaisesRegex(ValueError, "no exact-prefix"):
                            block_loss(out, tb, config)
                        continue
                    loss, report = block_loss(out, tb, config)
                    loss.backward()
                    outputs.append((loss.detach(), logits.grad, report))
                if outputs:
                    self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
                    self.assertTrue(torch.equal(outputs[0][1], outputs[1][1]))
                    self.assertEqual(outputs[0][2], outputs[1][2])
        cursors = [d.cursor(seed=17) for d in (bounded, retained)]
        for _ in range(12):
            pairs = [
                d.next_block(c, require_teacher=True) for d, c in zip((bounded, retained), cursors)
            ]
            self.assertEqual(pairs[0][1], pairs[1][1])
            self.assertEqual(pairs[0][0].chain_id, pairs[1][0].chain_id)
            self.assertEqual(pairs[0][0].teacher_prefix_sha256, pairs[1][0].teacher_prefix_sha256)
            np.testing.assert_array_equal(
                pairs[0][0].attention_allowed, pairs[1][0].attention_allowed
            )
            cursors = [pair[1] for pair in pairs]

    def test_budget_failure_closes_audit_maps_and_fit_copies_survive_close(self):
        import importlib.util

        script = Path(__file__).resolve().parents[1] / "scripts/fit_block_fusion.py"
        spec = importlib.util.spec_from_file_location("bounded_fit", script)
        fit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fit)
        dataset = load(self.path)
        self.addCleanup(dataset.close)
        rows, selected = fit.select_rows(dataset, "train", 3, 9)
        self.assertEqual(rows.shape, (9, 10))
        self.assertEqual(len(selected), 3)
        expected = rows.copy()
        dataset.close()
        np.testing.assert_array_equal(rows, expected)
        maps = []
        original = BlockDataset._array

        def record(owner, *args, **kwargs):
            value = original(owner, *args, **kwargs)
            maps.append(value._mmap)
            return value

        def budget():
            if len(maps) == 2:
                raise MemoryError("test current budget exhausted")

        with patch.object(BlockDataset, "_array", record):
            with self.assertRaisesRegex(MemoryError, "current budget"):
                load(self.path, audit_only=True, budget_check=budget)
        self.assertEqual(len(maps), 2)
        self.assertTrue(all(mapped.closed for mapped in maps))

    def test_late_nonfinite_and_hash_corruption_rejected(self):
        manifest = json.loads(self.path.read_text())
        chain = manifest["chains"][-1]
        logit_path = self.path.parent / chain["logits"]["path"]
        value = np.load(logit_path)
        value[-1, -1] = np.nan
        np.save(logit_path, value)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load(self.path, audit_only=True)
        chain["logits"]["sha256"] = file_sha256(logit_path)
        receipt_path = self.path.parent / manifest["producer"]["receipt"]["path"]
        receipt = json.loads(receipt_path.read_text())
        receipt["chains"][chain["chain_id"]]["logits_sha256"] = chain["logits"]["sha256"]
        receipt_path.write_text(json.dumps(receipt))
        manifest["producer"]["receipt"]["sha256"] = file_sha256(receipt_path)
        self.path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            load(self.path, audit_only=True)

    @unittest.skipUnless(
        sys.platform == "darwin" or sys.platform.startswith("linux"), "POSIX evidence"
    )
    def test_450_exact_soft_chains_under_low_fd_and_rss_caps(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--stress"],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        evidence = json.loads(result.stdout)
        self.assertEqual(evidence["chains"], 450)
        self.assertLess(evidence["peak_fds"], 20)
        self.assertLess(evidence["peak_rss_growth_bytes"], 64 * 1024**2)


if __name__ == "__main__":
    if "--stress" in sys.argv:
        stress_child()
    else:
        unittest.main()
