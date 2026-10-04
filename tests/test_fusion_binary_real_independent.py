"""Read-only independent audit of the authorized real TRAIN fusion fit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIT_SCRIPT = Path(
    os.environ.get("EAGLE_FUSION_FIT_SCRIPT", REPO / "scripts/fit_fusion_binary_discrete.py")
).resolve()
if not FIT_SCRIPT.is_file():
    raise unittest.SkipTest("fusion fitter source is not available in this checkout")
sys.path.insert(0, str(FIT_SCRIPT.parent))
import fit_fusion_binary_discrete as fitter  # noqa: E402

RUN_ROOT = Path(
    os.environ.get(
        "EAGLE_FUSION_REAL_RUN_ROOT",
        REPO / "results/fusion-binary-real-a8-20261003",
    )
)
RUN = RUN_ROOT / "fit-real-01"
DATA = RUN_ROOT / "data-v2"
REPORT_PATH = RUN / "fit_report.json"
RECEIPT_PIN = "75cbf2b83fdcc14b2dc338a5ac2fe27188a41cd6957fa311ea7d0e6bcd857c87"
MANIFEST_PIN = "32db46b48d6f9737aefb2a4be01b87c7376e645bbc71972329da0f68cc338619"
OPERANDS_PIN = "04ad9777d23df2d42863aadde2a8cf672981bcedee83e601cae60da8e8bb6681"
SOURCE_WEIGHTS_PIN = "58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e"
BASE_GGUF_PIN = "c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1"
CAPTURE_REVISION = "b4e366d4f0a30cac07f14d51c54c5b1329b3f485"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _independent_metrics(prediction: np.ndarray, teacher: np.ndarray) -> dict:
    pred = prediction.astype(np.float64)
    target = teacher.astype(np.float64)
    error = pred - target
    sse = float(np.sum(error * error, dtype=np.float64))
    energy = float(np.sum(target * target, dtype=np.float64))
    norms = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    valid = norms > 0
    cosine = np.sum(pred * target, axis=1)[valid] / norms[valid]
    return {
        "sse": sse,
        "relative_squared_error": sse / energy if energy else None,
        "mean_row_cosine": float(np.mean(cosine)) if cosine.size else None,
        "coordinate_sign_agreement": float(np.mean((pred >= 0) == (target >= 0))),
        "max_coordinate_agreement": float(
            np.mean(np.argmax(pred, axis=1) == np.argmax(target, axis=1))
        ),
    }


class IndependentRealFusionFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not REPORT_PATH.is_file():
            raise unittest.SkipTest("the authorized real-fit artifacts are not present")
        cls.report = json.loads(REPORT_PATH.read_text())
        cls.manifest = json.loads((DATA / "fusion-manifest.json").read_text())
        cls.receipt = json.loads((DATA / "provenance-receipt.json").read_text())
        cls.inventory = json.loads((DATA / "train-inventory.json").read_text())

    def test_authenticated_capture_rows_and_separate_training_grant(self):
        self.assertEqual(_sha(DATA / "provenance-receipt.json"), RECEIPT_PIN)
        self.assertEqual(_sha(DATA / "fusion-manifest.json"), MANIFEST_PIN)
        self.assertEqual(_sha(DATA / "fusion-operands.npz"), OPERANDS_PIN)
        self.assertEqual(self.report["provenance_receipt_sha256"], RECEIPT_PIN)
        self.assertEqual(self.report["manifest_sha256"], MANIFEST_PIN)
        self.assertEqual(self.report["operands_sha256"], OPERANDS_PIN)
        self.assertEqual(self.report["source"]["frozen_weights_sha256"], SOURCE_WEIGHTS_PIN)
        self.assertEqual(self.report["source"]["base_gguf_sha256"], BASE_GGUF_PIN)
        self.assertEqual(self.report["producer"]["native_revision"], CAPTURE_REVISION)
        self.assertEqual(
            self.report["producer"]["contract"],
            "native_target_block_inputs_concat_before_draft_fc_f32_taps_2_18_33_no_upstream_cast_or_norm",
        )
        self.assertIs(self.report["synthetic"], False)
        self.assertEqual(self.manifest["source_data_split"], "train")
        self.assertEqual(self.manifest["raw_input_stage"], "pre_activation_quantization")
        self.assertEqual(self.manifest["eligibility"], "training_allowed")

        selected_counts = Counter(p["split"] for p in self.receipt["selected_prompts"])
        self.assertEqual(selected_counts, {"train": 8, "validation": 4})
        source_inventory = {row["prompt_id"]: row for row in self.inventory["prompts"]}
        groups = {split: set() for split in ("train", "validation")}
        for prompt in self.receipt["selected_prompts"]:
            source = source_inventory[prompt["prompt_id"]]
            self.assertEqual(source["source_split"], "train")
            self.assertEqual(source["prompt_sha256"], prompt["prompt_sha256"])
            self.assertEqual(source["group_id"], prompt["group_id"])
            groups[prompt["split"]].add(prompt["group_id"])
        self.assertTrue(groups["train"].isdisjoint(groups["validation"]))
        row_counts = Counter((row["split"], row["prompt_id"]) for row in self.manifest["rows"])
        self.assertEqual(len(row_counts), 12)
        self.assertEqual(set(row_counts.values()), {32})

        capture_path = DATA / "source/capture-manifest.json"
        readiness_path = DATA / "source/readiness.json"
        provider_path = DATA / "source/provider.json"
        binding_path = DATA / "source/provider-binding.json"
        prepared_path = DATA / "source/preparation-ready.json"
        runtime_path = DATA / "source/native-runtime-manifest.json"
        capture = json.loads(capture_path.read_text())
        readiness = json.loads(readiness_path.read_text())
        provider = json.loads(provider_path.read_text())
        binding = json.loads(binding_path.read_text())
        prepared = json.loads(prepared_path.read_text())
        runtime = json.loads(runtime_path.read_text())

        # Preserve the historical capture's own state. Full-use permission is
        # supplied by separately hashed readiness and provider records.
        self.assertIs(capture["training_eligible"], False)
        self.assertEqual(capture["readiness"], "preparation_only")
        self.assertEqual(capture["split"], "train")
        self.assertEqual(capture["activation_bits"], 16)
        self.assertEqual(readiness["training_eligible"], True)
        self.assertEqual(readiness["objective"], "hard_ce")
        self.assertEqual(readiness["scope"], "joint_body_head_exact_prefix_teacher_forced_training")
        self.assertEqual(readiness["unresolved_gates"], [])
        self.assertEqual(provider["training_eligible"], True)
        self.assertEqual(provider["split"], "train")
        self.assertEqual(
            provider["capture_id"], self.receipt["producer"]["capture_manifest_sha256"]
        )
        self.assertEqual(binding["prepared_ready_sha256"], _sha(prepared_path))
        self.assertIs(prepared["preparation_complete"], True)
        self.assertIs(prepared["optimization_started"], False)
        self.assertEqual(prepared["stop_reason"], "prepare_only")
        self.assertEqual(runtime["native_commit"], CAPTURE_REVISION)
        self.assertIs(runtime["intrusive_events"], False)
        self.assertEqual(self.receipt["producer"]["capture_manifest_sha256"], _sha(capture_path))
        self.assertEqual(self.receipt["producer"]["readiness_sha256"], _sha(readiness_path))

        # Join every exported F32 row back to the authenticated accepted-prefix
        # feature event; do not infer ancestry from tensor shape alone.
        with np.load(DATA / "fusion-operands.npz", allow_pickle=False) as archive:
            raw = archive["raw_input"]
            ids = archive["raw_join_ids"]
            self.assertEqual(raw.shape, (384, 7680))
            self.assertEqual(raw.dtype, np.float32)
            self.assertEqual(ids.shape, (384,))
            self.assertEqual(ids.dtype.kind, "U")
            self.assertEqual(len(self.receipt["rows"]), 384)
            native_rows = set()
            for index, (row, evidence) in enumerate(
                zip(self.manifest["rows"], self.receipt["rows"], strict=True)
            ):
                self.assertEqual(str(ids[index]), row["row_id"])
                self.assertEqual(row["row_id"], evidence["row_id"])
                self.assertEqual(row["raw_input_sha256"], evidence["raw_input_sha256"])
                self.assertEqual(row["raw_input_sha256"], _array_sha(raw[index].astype("<f4")))
                metadata = evidence["feature_metadata"]
                decoded = evidence["native_events"]["decoded_row"]
                disposition = evidence["native_events"]["disposition"]
                self.assertEqual(evidence["feature_row"], metadata["feature_row"])
                self.assertEqual(metadata["prompt_id"], row["prompt_id"])
                self.assertEqual(metadata["position"], row["position"])
                self.assertEqual(metadata["prefix_token_ids"], evidence["prompt_token_ids"])
                self.assertEqual(metadata["tap_ids"], [2, 18, 33])
                self.assertEqual(
                    metadata["boundary"], "native_target_block_inputs_concat_before_draft_fc"
                )
                self.assertEqual(metadata["source"], "native_target_features_on_accepted_prefix")
                self.assertIs(metadata["accepted_prefix"], True)
                self.assertEqual(decoded["feature_row"], metadata["native_feature_row"])
                self.assertEqual(decoded["feature_dtype"], "float32_native_endian")
                self.assertEqual(decoded["feature_dim"], 7680)
                self.assertEqual(decoded["target_layer_ids"], [2, 18, 33])
                self.assertEqual(decoded["boundary"], "raw_target_layer_input_before_eagle_encoder")
                self.assertEqual(decoded["source"], "target_verifier")
                self.assertEqual(decoded["prefix_token_ids"], evidence["prompt_token_ids"])
                self.assertEqual(disposition["feature_row"], decoded["feature_row"])
                self.assertIs(disposition["retained_input"], True)
                self.assertEqual(disposition["reason"], "accepted_prefix")
                native_rows.add(metadata["native_feature_row"])
            self.assertEqual(len(native_rows), 384)

    def test_converged_matched_scale_control_and_finite_sign_moves(self):
        report = self.report
        config = report["config"]
        controls = report["fit"]["control_solver"]
        updates = report["fit"]["scale_updates"]
        events = report["fit"]["flip_events"]
        self.assertEqual(len(controls), 2560)
        self.assertEqual(len(updates), 5120)
        self.assertEqual(len(report["fit"]["flip_counts"]), 2560)
        self.assertEqual(len(events), sum(report["fit"]["flip_counts"]))
        for row in controls:
            self.assertTrue(math.isfinite(row["continuous_kkt_relative"]))
            self.assertLess(row["continuous_kkt_relative"], 1e-12)
            threshold = max(
                config["improvement_absolute_margin"],
                config["improvement_relative_margin"] * row["finite_sse"],
            )
            self.assertLessEqual(row["neighbor_gain"], threshold)
        for update in updates:
            self.assertTrue(math.isfinite(update["before_sse"]))
            self.assertTrue(math.isfinite(update["after_sse"]))
            self.assertGreaterEqual(update["before_sse"] + 1e-9, update["after_sse"])
            self.assertGreaterEqual(update["scale_f32"], 0)
        for event in events:
            self.assertTrue(math.isfinite(event["before_sse"]))
            self.assertTrue(math.isfinite(event["after_sse"]))
            self.assertTrue(math.isfinite(event["scale_f32"]))
            self.assertLess(event["after_sse"], event["before_sse"])
            self.assertLessEqual(report["fit"]["flip_counts"][event["row"]], 32)
        self.assertEqual(sum(row["accepted"] for row in report["fit"]["scans"]), len(events))
        self.assertEqual(len(report["fit"]["scans"]), 4)
        self.assertEqual(Counter(report["fit"]["flip_counts"]), Counter({0: 383, 31: 1, 32: 2176}))

        control_path = RUN / "control_scales.npz"
        self.assertEqual(_sha(control_path), report["control_scales_sha256"])
        with np.load(control_path, allow_pickle=False) as control:
            for name in ("initializer_scale", "scale_only_scale"):
                values = control[name]
                self.assertEqual(values.shape, (2560,))
                self.assertEqual(values.dtype, np.float32)
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue(np.all(values >= 0))

    def test_frozen_candidate_and_per_prompt_validation_metrics(self):
        report = self.report
        candidate_path = RUN / "fusion_candidate.npz"
        self.assertEqual(_sha(candidate_path), report["frozen_candidate_sha256"])

        with np.load(candidate_path, allow_pickle=False) as candidate:
            signs = candidate["fc.latent"].astype(np.int8)
            scales = candidate["fc.scale"]
            packed = candidate["fc.w1a1_packed"]
            self.assertEqual(signs.shape, (2560, 7680))
            self.assertTrue(np.all((signs == -1) | (signs == 1)))
            self.assertEqual(scales.shape, (2560,))
            self.assertEqual(scales.dtype, np.float32)
            self.assertTrue(np.isfinite(scales).all())
            self.assertTrue(np.all(scales >= 0))
            np.testing.assert_array_equal(fitter.unpack_signs(packed, 7680), signs)

        split_rows = [row for row in self.manifest["rows"] if row["split"] == "validation"]
        validation_indices = [
            i for i, row in enumerate(self.manifest["rows"]) if row["split"] == "validation"
        ]
        prompt_to_positions: dict[str, list[int]] = {}
        for position, source_index in enumerate(validation_indices):
            prompt_to_positions.setdefault(
                self.manifest["rows"][source_index]["prompt_id"], []
            ).append(position)
        control_path = RUN / "control_scales.npz"
        with np.load(control_path, allow_pickle=False) as control:
            initial_scales = control["initializer_scale"].copy()
            scale_only = control["scale_only_scale"].copy()
        with np.load(DATA / "fusion-operands.npz", allow_pickle=False) as archive:
            raw = archive["raw_input"]
            weight = archive["reference_weight"]
            x = raw[validation_indices]
            teacher = x @ weight.T
            limit = np.max(np.abs(x), axis=1)
            beta = np.divide(limit, np.float32(127), dtype=np.float32)
            inverse = np.divide(np.float32(127), limit, out=np.zeros_like(limit), where=limit > 0)
            codes = np.clip(
                np.rint(np.multiply(x, inverse[:, None], dtype=np.float32)), -127, 127
            ).astype(np.int16)
            initial_signs = np.where(weight < 0, -1, 1).astype(np.float32)
            candidate_signs = signs.astype(np.float32)
            self.assertEqual(int(np.count_nonzero(signs != initial_signs.astype(np.int8))), 66803)
            self.assertEqual(int(np.count_nonzero(scales != scale_only)), 2176)
            self.assertEqual(int(np.count_nonzero(scales == 0)), 383)
            self.assertEqual(int(np.count_nonzero(initial_scales == 0)), 0)
            dots_control = (codes.astype(np.float32) @ initial_signs.T).astype(np.int32)
            dots_candidate = (codes.astype(np.float32) @ candidate_signs.T).astype(np.int32)
            control_prediction = np.multiply(
                np.multiply(dots_control.astype(np.float32), scale_only, dtype=np.float32),
                beta[:, None],
                dtype=np.float32,
            )
            candidate_prediction = np.multiply(
                np.multiply(dots_candidate.astype(np.float32), scales, dtype=np.float32),
                beta[:, None],
                dtype=np.float32,
            )
            rows = []
            for prompt_id, positions in sorted(prompt_to_positions.items()):
                self.assertEqual(len(positions), 32)
                control_metrics = _independent_metrics(
                    control_prediction[positions], teacher[positions]
                )
                candidate_metrics = _independent_metrics(
                    candidate_prediction[positions], teacher[positions]
                )
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt_sha256": next(
                            r["prompt_sha256"] for r in split_rows if r["prompt_id"] == prompt_id
                        ),
                        "rows": len(positions),
                        "converged_scale_only": control_metrics,
                        "sign_and_scale": candidate_metrics,
                        "relative_sse_improved": candidate_metrics["relative_squared_error"]
                        < control_metrics["relative_squared_error"],
                        "coordinate_direction_improved": candidate_metrics[
                            "coordinate_sign_agreement"
                        ]
                        > control_metrics["coordinate_sign_agreement"],
                        "argmax_agreement_improved": candidate_metrics["max_coordinate_agreement"]
                        > control_metrics["max_coordinate_agreement"],
                    }
                )

        control_global = report["metrics"]["validation"]["converged_scale_only"]
        initializer_global = report["metrics"]["validation"]["original_binary_initializer"]
        candidate_global = report["metrics"]["validation"]["sign_and_scale"]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["relative_sse_improved"] for row in rows))
        self.assertLess(
            candidate_global["relative_squared_error"], control_global["relative_squared_error"]
        )
        self.assertGreater(
            candidate_global["coordinate_sign_agreement"],
            control_global["coordinate_sign_agreement"],
        )
        self.assertGreater(
            candidate_global["max_coordinate_agreement"], control_global["max_coordinate_agreement"]
        )
        self.assertLess(candidate_global["mean_row_cosine"], initializer_global["mean_row_cosine"])
        self.assertLess(
            candidate_global["coordinate_sign_agreement"],
            initializer_global["coordinate_sign_agreement"],
        )
        self.assertLess(
            candidate_global["max_coordinate_agreement"],
            initializer_global["max_coordinate_agreement"],
        )
        self.assertTrue(
            math.isclose(
                sum(row["sign_and_scale"]["sse"] for row in rows),
                candidate_global["sse"],
                rel_tol=2e-5,
            )
        )
        out = RUN_ROOT / "independent-validation"
        out.mkdir(parents=True, exist_ok=True)
        (out / "per-prompt-validation.json").write_text(
            json.dumps(
                {"schema": "fusion_binary_real_per_prompt_validation_v1", "rows": rows}, indent=2
            )
            + "\n"
        )

    def test_bounded_streamed_export_roundtrip_preserves_all_nonfusion_operands(self):
        candidate_path = RUN / "fusion_candidate.gguf"
        self.assertEqual(_sha(candidate_path), self.report["export"]["sha256"])
        self.assertEqual(self.report["export"]["base_gguf_sha256"], BASE_GGUF_PIN)
        self.assertEqual(self.report["export"]["fusion_only"], True)
        self.assertEqual(self.report["export"]["unchanged_nonfusion_tensors"], 13)
        self.assertEqual(self.report["export"]["native_validation"], "deferred")

        # The independently staged stream audit reopens the file and verifies
        # all original KVs and all 13 nonfusion tensor payloads without loading
        # two full models into RAM. Its final bytes must match this candidate.
        stream_path = RUN_ROOT / "serialization-memory-probe-01/report.json"
        streaming = json.loads(stream_path.read_text())
        self.assertEqual(streaming["source_candidate_sha256"], _sha(RUN / "fusion_candidate.npz"))
        self.assertEqual(streaming["legacy_gguf_sha256"], _sha(candidate_path))
        self.assertEqual(streaming["base_gguf_sha256"], BASE_GGUF_PIN)
        self.assertEqual(streaming["unchanged_nonfusion_tensors"], 13)
        self.assertEqual(streaming["serialization"], "streamed_raw_kv_and_dense_payloads")
        self.assertEqual(streaming["sha256"], self.report["export"]["sha256"])
        self.assertEqual(streaming["packed_sha256"], self.report["export"]["packed_sha256"])
        self.assertEqual(streaming["scale_sha256"], self.report["export"]["scale_sha256"])
        self.assertLess(
            max(row["process_peak_rss_bytes"] for row in streaming["process_memory_phases"]),
            1 << 30,
        )

    def test_cpu_run_record_reports_measured_rss_and_no_live_fit_process(self):
        run_record = json.loads((RUN_ROOT / "fit-real-01.execution.json").read_text())
        self.assertEqual(run_record["return_code"], 0)
        self.assertEqual(run_record["max_resident_set_size_bytes"], 2_796_650_496)
        self.assertGreater(
            run_record["max_resident_set_size_bytes"], run_record["configured_workspace_cap_bytes"]
        )
        self.assertEqual(run_record["persistent_session_id"], None)
        self.assertIs(run_record["os_pid_captured"], False)
        self.assertIn("does not attribute peak RSS", run_record["resource_note"])

        reproduced = json.loads((RUN_ROOT / "fit-real-02.execution.json").read_text())
        self.assertEqual(reproduced["return_code"], 0)
        self.assertLessEqual(
            reproduced["max_resident_set_size_bytes"], reproduced["configured_workspace_cap_bytes"]
        )
        self.assertLess(
            reproduced["phase_peak_bytes"], reproduced["configured_workspace_cap_bytes"]
        )
        self.assertIs(reproduced["candidate_exactly_matches_first_fit"], True)
        self.assertIs(reproduced["control_scales_exactly_match_first_fit"], True)
        self.assertIs(reproduced["gguf_exactly_matches_first_fit"], True)
        self.assertEqual(
            reproduced["process_check_after_completion"],
            run_record["process_check_after_completion"],
        )
        run1_command = json.loads((RUN_ROOT / "fit-real-01.command.json").read_text())
        run2_command = json.loads((RUN_ROOT / "fit-real-02.command.json").read_text())
        run2_report = json.loads((RUN_ROOT / "fit-real-02/fit_report.json").read_text())
        self.assertEqual(run1_command["hashes"][str(FIT_SCRIPT)], self.report["script_sha256"])
        self.assertEqual(run2_command["hashes"][str(FIT_SCRIPT)], run2_report["script_sha256"])
        config_path = str(FIT_SCRIPT.parent.parent / "configs/fusion_binary_discrete_a8.json")
        self.assertEqual(run1_command["hashes"][config_path], run2_command["hashes"][config_path])
        self.assertEqual(self.report["metrics"], run2_report["metrics"])
        self.assertEqual(
            self.report["frozen_candidate_sha256"], run2_report["frozen_candidate_sha256"]
        )

    def test_full_model_cpu_loader_and_native_operator_gate(self):
        base_loader = json.loads((RUN_ROOT / "base-model-loader.json").read_text())
        candidate_loader = json.loads((RUN_ROOT / "candidate-model-loader.json").read_text())
        for result in (base_loader, candidate_loader):
            self.assertIs(result["cpu_model_load_passed"], True)
            self.assertEqual(result["decoder_layers"], 1)
            self.assertEqual(result["hidden_width"], 2560)
            self.assertEqual(result["gpu_layers"], 0)
            self.assertIs(result["inference_performed"], False)
        native = json.loads((RUN_ROOT / "native-real-01/report.json").read_text())
        self.assertIs(native["gate_passed"], True)
        self.assertIs(native["synthetic"], False)
        self.assertIs(native["native_acceptance_or_throughput_claim"], False)
        self.assertEqual(native["gguf_sha256"], self.report["export"]["sha256"])
        self.assertEqual(native["receipt_sha256"], RECEIPT_PIN)
        self.assertEqual(native["operator"]["backend"], "Apple M3 Max")
        self.assertEqual(native["operator"]["backend_type"], "cpu")
        self.assertEqual(native["operator"]["source_bits"], 32)
        self.assertEqual(native["operator"]["replay_bits"], 8)
        self.assertEqual(native["operator"]["checked_outputs"], 20480)
        self.assertEqual(native["operator"]["max_abs_error"], 0)
        self.assertEqual(native["operator"]["max_rel_error"], 0)


if __name__ == "__main__":
    unittest.main()
