"""CPU tiny-model/filesystem retention checks; no native or CUDA claim."""

import copy
import json
import random
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_nine_model_training as fixtures
import torch
import train_nine_model_qat as launcher
from test_block_checkpoint_contract import SOURCE

import w1a1_eagle.block_training as checkpoints
from w1a1_eagle.block_qat import BlockDrafter, block_optimizer, block_train_step
from w1a1_eagle.block_training import (
    BlockCheckpointRetention,
    BlockCursor,
    load_block_checkpoint,
    protect_block_checkpoint,
    save_block_checkpoint,
    transition_a8_to_a1,
)


class BlockRetentionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()
        self.cfg, self.tensors, self.batch = fixtures.block_fixture()
        self.model = BlockDrafter(self.tensors, self.cfg)
        self.optimizer = block_optimizer(self.model)
        self.cursor = BlockCursor(data_cursor={"epoch": 0, "dataset": "exact"})
        self.policy = BlockCheckpointRetention(keep_recent=1, max_checkpoints=8, max_bytes=10**8)

    def save(self, *protect, policy=None):
        return save_block_checkpoint(
            self.model,
            self.optimizer,
            self.cursor,
            SOURCE,
            self.root,
            retention=policy or self.policy,
            protect=protect,
        )

    def step(self):
        block_train_step(self.model, self.optimizer, self.batch)
        self.cursor = replace(
            self.cursor,
            step=self.cursor.step + 1,
            block_index=self.cursor.block_index + 1,
            stage_updates=self.cursor.stage_updates + 1,
            supervised_tokens=self.cursor.supervised_tokens + 7,
            stage_supervised_tokens=self.cursor.stage_supervised_tokens + 7,
            presented_tokens=self.cursor.presented_tokens + 7,
            elapsed_seconds=self.cursor.elapsed_seconds + 1,
            unique_blocks=("chain:0",),
            data_cursor={"epoch": 1, "dataset": "exact"},
        )

    def test_default_retains_all_without_retention_sidecars(self):
        for _ in range(4):
            save_block_checkpoint(self.model, self.optimizer, self.cursor, SOURCE, self.root)
            self.step()
        self.assertEqual(len(list(self.root.glob("*.pt"))), 4)
        self.assertEqual(list(self.root.glob("*.receipt.json")), [])

    def test_pruning_preserves_exact_next_update_rng_cursor_source_and_moments(self):
        initial = self.save()
        self.step()
        obsolete = self.save()
        self.step()
        latest = self.save()
        self.assertTrue(Path(initial["path"]).exists())
        self.assertFalse(Path(obsolete["path"]).exists())
        self.assertFalse(Path(obsolete["path"]).with_suffix(".receipt.json").exists())
        clone = BlockDrafter(self.tensors, self.cfg)
        clone_optimizer = block_optimizer(clone)
        expected_draws = (random.random(), np.random.random(), torch.rand(3))
        actual_cursor = load_block_checkpoint(clone, clone_optimizer, SOURCE, latest)
        actual_draws = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(expected_draws[:2], actual_draws[:2])
        self.assertTrue(torch.equal(expected_draws[2], actual_draws[2]))
        self.assertEqual(actual_cursor, self.cursor)
        self.assertEqual(actual_cursor.data_cursor, {"epoch": 1, "dataset": "exact"})
        with self.assertRaisesRegex(ValueError, "source/config"):
            load_block_checkpoint(
                clone, clone_optimizer, SOURCE | {"bundle_sha256": "f" * 64}, latest
            )
        block_train_step(self.model, self.optimizer, self.batch)
        block_train_step(clone, clone_optimizer, self.batch)
        for a, b in zip(self.model.parameters(), clone.parameters()):
            self.assertTrue(torch.equal(a, b))
        for key, value in self.optimizer.state_dict()["state"].items():
            for name, tensor in value.items():
                self.assertTrue(
                    torch.equal(tensor, clone_optimizer.state_dict()["state"][key][name])
                )

    def test_initial_both_transition_sides_endpoint_and_stop_remain_protected(self):
        self.cursor = replace(self.cursor, stage="a8_warm_start")
        protected = [self.save()]
        self.step()
        protected.append(self.save("transition_source"))
        self.model, self.optimizer, _ = transition_a8_to_a1(
            self.model, source_checkpoint_sha256=protected[-1]["sha256"], in_place=True
        )
        self.cursor = replace(
            self.cursor, stage="a1_final", stage_updates=0, stage_supervised_tokens=0
        )
        protected.append(self.save())
        self.step()
        obsolete = self.save()
        self.step()
        protected.append(self.save("endpoint"))
        self.step()
        protected.append(self.save("stop"))
        self.step()
        self.save()
        self.assertFalse(Path(obsolete["path"]).exists())
        for value in protected:
            self.assertTrue(Path(value["path"]).exists())
        self.assertEqual(len(list(self.root.glob("*.pt"))), 6)
        self.assertEqual(protected[2]["retention"]["protected"], ["transition_destination"])
        clone = copy.deepcopy(self.model)
        restored = load_block_checkpoint(clone, block_optimizer(clone), SOURCE, protected[2])
        self.assertEqual(restored.stage_updates, 0)

    def test_protect_existing_endpoint_without_rewriting_resume_payload(self):
        self.save()
        self.step()
        latest = self.save()
        original = Path(latest["path"]).read_bytes()
        protected = protect_block_checkpoint(latest, "endpoint")
        self.step()
        self.save()
        self.assertEqual(Path(latest["path"]).read_bytes(), original)
        self.assertEqual(protected["retention"]["protected"], ["endpoint"])

    def test_count_and_bytes_admission_fails_without_pruning_or_latest_mutation(self):
        initial = self.save()
        self.step()
        latest = self.save()
        self.step()
        for policy in (
            BlockCheckpointRetention(1, 1, 10**8),
            BlockCheckpointRetention(1, 8, Path(initial["path"]).stat().st_size),
        ):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "cannot admit"):
                self.save(policy=policy)
            self.assertTrue(Path(initial["path"]).exists())
            self.assertTrue(Path(latest["path"]).exists())
            self.assertEqual(json.loads((self.root / "latest.json").read_text()), latest)
            # Failed uncommitted staging is preserved; use a distinct generation
            # to test the next admission without recycling its evidence.
            self.cursor = replace(self.cursor, elapsed_seconds=self.cursor.elapsed_seconds + 0.1)

    def test_publication_peak_must_fit_even_when_post_pruning_would_fit(self):
        initial = self.save()
        self.step()
        latest = self.save()
        self.step()
        old_bytes = sum(p.stat().st_size for p in self.root.glob("*.pt"))
        # Post-pruning retains initial+new and would fit both policies. The
        # old committed latest must remain alongside the new writer first.
        for policy in (
            BlockCheckpointRetention(1, 2, 10**8),
            BlockCheckpointRetention(1, 8, old_bytes + 256),
        ):
            with self.assertRaisesRegex(ValueError, "cannot admit"):
                self.save(policy=policy)
            self.assertTrue(Path(initial["path"]).exists())
            self.assertTrue(Path(latest["path"]).exists())
            self.assertEqual(json.loads((self.root / "latest.json").read_text()), latest)
            self.cursor = replace(self.cursor, elapsed_seconds=self.cursor.elapsed_seconds + 0.1)

    def test_failed_staging_consumes_future_peak_budget_and_is_never_recycled(self):
        self.save()
        self.step()
        staging = self.root / checkpoints._checkpoint_name(self.cursor)
        staging = staging.with_suffix(".tmp")
        committed_bytes = sum(p.stat().st_size for p in self.root.glob("*.pt"))
        with self.assertRaisesRegex(ValueError, "serialized payload bytes"):
            self.save(policy=BlockCheckpointRetention(1, 8, committed_bytes + 1024))
        original = staging.read_bytes()
        self.assertLessEqual(len(original), 1024)
        with self.assertRaises(FileExistsError):
            self.save()
        self.cursor = replace(self.cursor, elapsed_seconds=self.cursor.elapsed_seconds + 0.1)
        # initial + failed staging + next writer is three, even though only
        # two committed checkpoints would exist after successful publication.
        with self.assertRaisesRegex(ValueError, "cannot admit"):
            self.save(policy=BlockCheckpointRetention(1, 2, 10**8))
        self.assertEqual(staging.read_bytes(), original)

    def test_count_overflow_rejects_before_creating_next_staging_file(self):
        initial = self.save()
        self.step()
        with self.assertRaisesRegex(ValueError, "before staging"):
            self.save(policy=BlockCheckpointRetention(1, 1, 10**8))
        self.assertEqual(list(self.root.glob("step-*.tmp")), [])
        self.assertEqual(json.loads((self.root / "latest.json").read_text()), initial)

    def test_serializer_failure_and_seek_length_never_grow_staging_beyond_byte_cap(self):
        initial = self.save()
        self.step()
        initial_bytes = Path(initial["path"]).stat().st_size

        def failing_serializer(payload, writer):
            writer.write(b"a" * 32)
            writer.seek(-16, 2)
            writer.write(b"b" * 8)
            self.assertEqual(writer.tell(), 24)
            writer.seek(32)
            writer.write(b"c" * 32)
            raise RuntimeError("injected serializer failure")

        with patch.object(checkpoints.torch, "save", side_effect=failing_serializer):
            with self.assertRaisesRegex(RuntimeError, "injected serializer failure"):
                self.save(policy=BlockCheckpointRetention(1, 8, initial_bytes + 64))
        staged = list(self.root.glob("step-*.tmp"))
        self.assertEqual(len(staged), 1)
        self.assertEqual(staged[0].stat().st_size, 64)
        self.assertEqual(json.loads((self.root / "latest.json").read_text()), initial)
        self.cursor = replace(self.cursor, elapsed_seconds=self.cursor.elapsed_seconds + 0.1)

        def out_of_bounds(payload, writer):
            writer.write(b"d" * 16)
            writer.seek(65)

        with patch.object(checkpoints.torch, "save", side_effect=out_of_bounds):
            with self.assertRaisesRegex(ValueError, "serialized payload bytes"):
                self.save(policy=BlockCheckpointRetention(1, 8, initial_bytes + 128))
        self.assertEqual(sum(p.stat().st_size for p in self.root.glob("step-*.tmp")), 80)
        self.assertEqual(json.loads((self.root / "latest.json").read_text()), initial)

    def test_failed_latest_or_sidecar_publication_never_prunes(self):
        self.save()
        self.step()
        prior = self.save()
        self.step()
        real = checkpoints.atomic_json
        for failed_name in ("latest.json", ".receipt.json"):

            def fail(path, value):
                if path.name.endswith(failed_name):
                    raise OSError("injected publication failure")
                return real(path, value)

            with patch.object(checkpoints, "atomic_json", side_effect=fail):
                with self.assertRaisesRegex(OSError, "publication failure"):
                    self.save()
            self.assertTrue(Path(prior["path"]).exists())
            self.assertTrue(Path(prior["path"]).with_suffix(".receipt.json").exists())
            self.cursor = replace(self.cursor, elapsed_seconds=self.cursor.elapsed_seconds + 0.1)

    def test_failed_publication_readback_or_payload_verification_never_prunes(self):
        self.save()
        self.step()
        prior = self.save()
        self.step()
        real = checkpoints.atomic_json
        for kind in ("receipt", "payload"):

            def corrupt(path, value):
                real(path, value)
                if path.name == "latest.json":
                    if kind == "receipt":
                        path.write_text("{}")
                    else:
                        Path(value["path"]).write_bytes(b"corrupt new checkpoint")

            with patch.object(checkpoints, "atomic_json", side_effect=corrupt):
                with self.assertRaisesRegex(ValueError, "verification failed"):
                    self.save()
            self.assertTrue(Path(prior["path"]).exists())
            self.cursor = replace(self.cursor, elapsed_seconds=self.cursor.elapsed_seconds + 0.1)

    def test_foreign_malformed_symlink_uncommitted_changed_and_hardlinked_stay(self):
        self.save()
        for kind in ("foreign", "malformed", "symlink", "uncommitted", "changed", "hardlink"):
            with self.subTest(kind=kind):
                self.step()
                old = self.save()
                path = Path(old["path"])
                sidecar = path.with_suffix(".receipt.json")
                if kind == "foreign":
                    value = json.loads(sidecar.read_text())
                    value["retention"]["family_sha256"] = "f" * 64
                    sidecar.write_text(json.dumps(value))
                elif kind == "malformed":
                    sidecar.write_text("[null]")
                elif kind == "symlink":
                    target = self.root / f"preserved-{kind}"
                    path.rename(target)
                    path.symlink_to(target)
                elif kind == "uncommitted":
                    sidecar.unlink()
                elif kind == "changed":
                    path.write_bytes(b"changed evidence")
                else:
                    import os

                    os.link(path, self.root / "preserved-hardlink")
                self.step()
                self.save()
                self.assertTrue(path.exists())
                if kind != "uncommitted":
                    self.assertTrue(sidecar.exists())

    def test_foreign_artifacts_consume_count_and_byte_limits(self):
        initial = self.save()
        foreign = self.root / "foreign.pt"
        foreign.write_bytes(b"unknown" * 100)
        self.step()
        for policy in (
            BlockCheckpointRetention(1, 2, 10**8),
            BlockCheckpointRetention(1, 8, 1),
        ):
            with self.assertRaisesRegex(ValueError, "cannot admit"):
                self.save(policy=policy)
            self.assertTrue(foreign.exists())
            self.assertTrue(Path(initial["path"]).exists())
            self.cursor = replace(self.cursor, elapsed_seconds=self.cursor.elapsed_seconds + 0.1)

    def test_retention_refuses_symlink_directory_or_publication_paths(self):
        linked = self.root / "linked"
        real = self.root / "real"
        real.mkdir()
        linked.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            save_block_checkpoint(
                self.model, self.optimizer, self.cursor, SOURCE, linked, retention=self.policy
            )
        latest = self.root / "latest.json"
        latest.symlink_to(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.save()
        self.assertTrue(latest.is_symlink())


class RetentionLauncherTests(unittest.TestCase):
    def test_spec_rejects_unsupported_missing_unknown_bool_and_nonpositive_limits(self):
        cfg, _, _ = fixtures.block_fixture()
        spec = {
            "schema": launcher.SCHEMA,
            "family": "dspark",
            "candidate": "dspark_a8",
            "device": "cuda:0",
            "qat": asdict(cfg),
            "limits": {"max_steps": 3},
            "checkpoint_every": 1,
        }
        valid = {"keep_recent": 1, "max_checkpoints": 5, "max_bytes": 10**8}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "spec.json"
            for invalid in (
                None,
                {},
                valid | {"extra": 1},
                valid | {"keep_recent": True},
                valid | {"max_bytes": 0},
                valid | {"keep_recent": 6},
            ):
                path.write_text(json.dumps(spec | {"checkpoint_retention": invalid}))
                with self.assertRaises(ValueError):
                    launcher.load_spec(path)
            path.write_text(json.dumps(spec | {"checkpoint_retention": valid}))
            self.assertEqual(launcher.load_spec(path)["checkpoint_retention"], valid)
            path.write_text(json.dumps(spec | {"family": "eagle", "checkpoint_retention": valid}))
            with self.assertRaisesRegex(ValueError, "only by the block saver"):
                launcher.load_spec(path)

    def test_actual_caller_retains_transition_and_endpoint_and_stop_boundaries(self):
        real = launcher.run_block

        def bounded(args, spec, hardware):
            spec = dict(
                spec,
                checkpoint_retention={
                    "keep_recent": 1,
                    "max_checkpoints": 6,
                    "max_bytes": 10**8,
                },
            )
            return real(args, spec, hardware)

        with patch.object(launcher, "run_block", side_effect=bounded):
            for stop in (False, True):
                with tempfile.TemporaryDirectory() as folder:
                    folder = str(Path(folder).resolve())
                    helper = fixtures.LauncherLifecycleTests()
                    if stop:
                        with self.assertRaises(InterruptedError):
                            helper.transaction(
                                folder, max_steps=6, stop_after=4, precision_stage="a8_to_a1"
                            )
                    else:
                        helper.transaction(folder, max_steps=6, precision_stage="a8_to_a1")
                    root = Path(folder) / "checkpoints"
                    values = [json.loads(p.read_text()) for p in root.glob("*.receipt.json")]
                    reasons = {reason for v in values for reason in v["retention"]["protected"]}
                    self.assertEqual(
                        reasons,
                        {
                            "initial",
                            "transition_source",
                            "transition_destination",
                            "stop" if stop else "endpoint",
                        },
                    )
                    self.assertEqual(len(values), 4)
                    self.assertEqual(len(list(root.glob("*.pt"))), 4)


if __name__ == "__main__":
    unittest.main()
