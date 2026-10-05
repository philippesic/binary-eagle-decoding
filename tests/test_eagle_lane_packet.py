"""Packet construction preserves frozen declarations and refuses wrong ancestry."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_eagle_lane_packet as packet  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return packet.builder.pin(path)


class PacketTests(unittest.TestCase):
    def fixture(self, root):
        prepared, initializer = root / "prepared", root / "initializer"
        common = {"target_gguf": "a" * 64, "base_draft_gguf": "b" * 64}
        source = {"common_source_sha256": common}
        providers, selected = [], []
        for ordinal, domain in enumerate(packet.DOMAINS):
            messages = [{"role": "user", "content": domain}]
            prompt = {"id": domain, "domain": domain, "messages": messages}
            content = hashlib.sha256(
                json.dumps(
                    messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            prompt_pin = write(root / f"prompts-{domain}.jsonl", prompt)
            provider = write(
                root / f"provider-{domain}.json",
                {
                    "paths": {"prompts": prompt_pin["path"]},
                    "sha256": {"prompts": prompt_pin["sha256"]},
                },
            )
            providers.append(
                {
                    "provider_manifest": provider["path"],
                    "provider_manifest_sha256": provider["sha256"],
                }
            )
            selected.append(
                {
                    "id": domain,
                    "domain": domain,
                    "shard_ordinal": ordinal,
                    "calibration_split": "fit",
                    "content_sha256": content,
                }
            )
        index = write(prepared / "stages/train-providers.json", {"shards": providers})
        source["execution_manifest_sha256"] = index["sha256"]
        ready = write(
            prepared / "preparation-ready.json",
            {
                "preparation_complete": True,
                "optimization_started": False,
                "teacher_coverage": {
                    "source": source,
                    "unique_train_prompts": 10000,
                    "unique_supervised_rows": 3899930,
                },
            },
        )
        original = json.loads((ROOT / "configs/continuous_w1ax.json").read_text())
        original["stages"] = {"frozen_source_marker": "unchanged"}
        write(prepared / "resolved_config.json", original)
        init = write(initializer / "initializer.npz", {"fixture": "never model-loaded"})
        init.update(
            activation_bits=8,
            encoding="policy_latents",
            latent_initialization={
                "policy": "preserve_reference_magnitudes",
                "reference_kind": "eagle_fixed_reference_0.5",
                "reference_sha256": "c" * 64,
            },
        )
        write(initializer / "initializer.json", init)
        report = {
            "schema": "eagle_production_fusion_initializer_v1",
            "initializer": init,
            "prepared_ready_sha256": ready["sha256"],
            "authenticated_full_source": source,
            "fit_config": {"orientation_rescue": False, "coordinate_flips": 0},
            "selected_prompts": selected,
            "config": packet.builder.pin(prepared / "resolved_config.json"),
            "prepared_run_dir": str(prepared.resolve()),
            "source_sha256": hashlib.sha256(
                json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        write(initializer / "report.json", report)
        runtime = write(
            root / "runtime.json",
            {
                "schema": "eagle_lane_packet_runtime_v1",
                "base_model": {"path": "/fixture/base", "sha256": "b" * 64},
                "inputs": {
                    "target": {"path": "/fixture/target", "sha256": "a" * 64},
                    "teacher_binary": {"path": "/fixture/teacher", "sha256": "e" * 64},
                },
                "gpu_uuid": "fixture",
                "gpu_control_path": "/fixture/control",
                "resource_policy": {},
                "native_source_revision": "d" * 40,
            },
        )
        authorization = root / "authorization.md"
        authorization.write_text("Human delegated overnight QAT.")
        return (
            SimpleNamespace(
                runtime=Path(runtime["path"]),
                runtime_sha256=runtime["sha256"],
                prepared_run_dir=prepared,
                initializer_dir=initializer,
                authorization=authorization,
                output=root / "packet",
            ),
            ready["sha256"],
            original,
        )

    def test_source_only_configuration_and_original_domain_prompt_joins(self):
        with tempfile.TemporaryDirectory() as temp:
            args, ready, original = self.fixture(Path(temp))
            with patch.object(packet, "READY_SHA", ready):
                packet.prepare(args)
            template = json.loads((args.output / "template.json").read_text())
            self.assertEqual(template["stages"], original["stages"])
            self.assertTrue(template["training"]["optimize_cache"])
            self.assertTrue(template["training"]["optimize_head"])
            self.assertEqual(template["training"]["checkpoint_every"], 250)
            budget = json.loads((args.output / "budget.json").read_text())
            self.assertFalse(budget["human_selected"])
            self.assertEqual(
                budget["candidates"]["eagle_a8"]["training_limits"]["max_seconds"], 86400
            )
            requests = [
                json.loads(line)
                for line in (args.output / "generation-requests.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                {row["chain_ancestry"]["domain"] for row in requests}, set(packet.DOMAINS)
            )
            self.assertTrue(
                all(
                    row["max_prompt_tokens"] == 512 and row["logits_mode"] == "none"
                    for row in requests
                )
            )
            commands = json.loads((args.output / "commands.json").read_text())
            self.assertFalse(commands["execution_authorized_by_this_file"])
            request = packet.builder.pin(args.output / "initial-preparation-request.json")
            self.assertIn(request["sha256"], commands["initial_prepare"])
            self.assertIn("--prepare-only", commands["initial_prepare"])
            self.assertIn("544", commands["native_generation"])

    def test_wrong_initializer_corpus_pin_refuses_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            args, ready, _ = self.fixture(Path(temp))
            report = json.loads((args.initializer_dir / "report.json").read_text())
            report["prepared_ready_sha256"] = "e" * 64
            write(args.initializer_dir / "report.json", report)
            with (
                patch.object(packet, "READY_SHA", ready),
                self.assertRaisesRegex(ValueError, "corpus join"),
            ):
                packet.prepare(args)
            self.assertFalse(args.output.exists())

    def test_exact_replay_join_rejects_token_domain_order_and_duplicate_substitutions(self):
        import copy

        requests = [
            {
                "tokens": [ordinal + 1, 7],
                "tap_ids": [2, 18, 33],
                "logits_mode": "last",
                "chain_ancestry": {"prompt_id": domain, "domain": domain},
                "decode_history": [{"offset": 0, "count": 2}],
            }
            for ordinal, domain in enumerate(packet.DOMAINS)
        ]
        packet.exact_replay_join(requests, requests)
        mutations = [
            lambda rows: rows[0].update(tokens=[99, 7]),
            lambda rows: rows[0]["chain_ancestry"].update(domain="code"),
            lambda rows: rows.reverse(),
            lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
        ]
        for mutate in mutations:
            rows = copy.deepcopy(requests)
            mutate(rows)
            with self.assertRaises(ValueError):
                packet.exact_replay_join(rows, requests)

    def test_step_zero_export_source_checkpoint_and_initializer_joins(self):
        import copy

        with tempfile.TemporaryDirectory() as temp:
            args, ready, _ = self.fixture(Path(temp).resolve())
            with patch.object(packet, "READY_SHA", ready):
                packet.prepare(args)
            runtime = json.loads(args.runtime.read_text())
            report = json.loads((args.initializer_dir / "report.json").read_text())
            checkpoint = (
                args.output / "initial-prepare/checkpoints/step-000000000000-e000000-r000000000000"
            )
            resume = write(checkpoint / "resume.pt", {"fixture": "no model"})
            joint = write(checkpoint / "A8/joint.npz", {"fixture": "no tensors"})
            joint_manifest = write(checkpoint / "A8/joint.json", {"fixture": "no tensors"})
            outer = {
                "step": 0,
                "epoch": 0,
                "cursor": 0,
                "sha256": resume["sha256"],
                "optimizer_rng_cursor_exact": True,
                "source_sha256": report["source_sha256"],
                "exports": {
                    "A8": {"joint.npz": joint["sha256"], "joint.json": joint_manifest["sha256"]}
                },
            }
            write(checkpoint / "manifest.json", outer)
            request_path = args.output / "initial-preparation-request.json"
            request = json.loads(request_path.read_text())
            receipt = {
                "schema": "nine_model_preparation_v1",
                "status": "PASS",
                "artifact_kind": "production",
                "stage": "eagle_a8/initial-prepare",
                "optimizer_updates": 0,
                "bundle_sha256": packet.builder.sha256(request_path),
                "config_sha256": request["config"]["sha256"],
                "source": report["authenticated_full_source"],
                "checkpoint": dict(resume, step=0),
            }
            receipt_path = args.output / "initial-prepare-receipt.json"
            write(receipt_path, receipt)
            model = write(args.output / "initial.gguf", {"fixture": "not native"})
            audit = {
                "serialization_audit_passed": True,
                "activation_bits": 8,
                "projections": {str(i): {} for i in range(9)},
                "base_gguf": runtime["base_model"],
                "output": model,
                "checkpoint": joint,
                "checkpoint_manifest": joint_manifest,
            }
            audit_path = args.output / "audit.json"
            write(audit_path, audit)
            bind_args = SimpleNamespace(
                packet=args.output, export_audit=audit_path, initial_model=Path(model["path"])
            )
            with patch.object(packet, "READY_SHA", ready):
                packet.initial_export_join(bind_args, packet.Files(), runtime)
                for item in ("receipt_source", "outer_source", "step", "checkpoint"):
                    changed_receipt, changed_outer, changed_audit = (
                        copy.deepcopy(receipt),
                        copy.deepcopy(outer),
                        copy.deepcopy(audit),
                    )
                    if item == "receipt_source":
                        changed_receipt["source"] = {}
                    elif item == "outer_source":
                        changed_outer["source_sha256"] = "f" * 64
                    elif item == "step":
                        changed_outer["step"] = 1
                    else:
                        changed_audit["checkpoint"] = model
                    write(receipt_path, changed_receipt)
                    write(checkpoint / "manifest.json", changed_outer)
                    write(audit_path, changed_audit)
                    with self.subTest(item=item), self.assertRaises(ValueError):
                        packet.initial_export_join(bind_args, packet.Files(), runtime)

    def test_build_and_initialization_provenance_are_frozen_in_admission_source(self):
        import test_nine_model_admission_plan_builder as fixtures

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            descriptor = fixtures.AdmissionPlanBuilderTests().descriptor(root)
            build = write(root / "compiler-shim-manifest.json", {"fixture": "metadata"})
            preparation = write(root / "generation-replay-link.json", {"fixture": "metadata"})
            descriptor["preflight"]["build_provenance"] = {"compiler_shim": build}
            descriptor["preflight"]["preparation_provenance"] = {"generation_link": preparation}
            path = root / "admission.json"
            packet.builder.materialize_admission_plan(descriptor, path, fixture=True)
            plan = json.loads(path.read_text())
            self.assertEqual(plan["source"]["artifact:build_provenance:compiler_shim"], build)
            self.assertEqual(
                plan["source"]["artifact:preparation_provenance:generation_link"], preparation
            )


if __name__ == "__main__":
    unittest.main()
