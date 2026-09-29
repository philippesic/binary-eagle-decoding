"""CPU contracts for the frozen W1Ax calibration readiness assembler."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_w1ax_calibration_readiness import (  # noqa: E402
    CALIBRATION_PINNED_INPUT_SHA256,
    DIAGNOSTIC_SAFE_PROMPTS_SHA256,
    DIAGNOSTIC_SOURCE_PROMPTS_SHA256,
    EXPECTED_DOMAINS,
    IDENTITY_REPORT_SHA256,
    ROOT,
    _alias_map,
    _bridge_roots,
    _contract_outcome_summary,
    _exact_response_pairs,
    _gradient_roots,
    _identity,
    _join_head_state,
    _measured_row_a16_task_ids,
    _ordered_head_cache_joins,
    _task_map,
    _validate_mask_placement,
    _validate_runner_manifest,
    _wrong_accepted_labels,
    assemble,
)
from run_binary_rescue_benchmark import CLEAR_PREFIXES  # noqa: E402
from run_binary_rescue_benchmark import command as runner_command  # noqa: E402
from run_binary_rescue_benchmark import server_env as runner_server_env  # noqa: E402
from run_binary_rescue_benchmark import validate_config as validate_runner_config  # noqa: E402
from w1ax_capture_provider import TARGET_GGUF_SHA256, sha256  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class CalibrationReadinessTests(unittest.TestCase):
    def test_frozen_safe_id_alias_map_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "alias-map.json"
            path.write_text(
                json.dumps(
                    {
                        "source_sha256": DIAGNOSTIC_SOURCE_PROMPTS_SHA256,
                        "safe_sha256": DIAGNOSTIC_SAFE_PROMPTS_SHA256,
                        "mapping": [
                            {
                                "source_id": source,
                                "safe_id": alias,
                                "messages_sha256": "a" * 64,
                            }
                            for alias, source in (
                                ("pilot-prose", "dolly:line-005896"),
                                ("pilot-reasoning", "gsm8k:train-000315"),
                                ("pilot-code", "mbpp:task-496"),
                            )
                        ],
                    }
                )
            )
            self.assertEqual(
                _alias_map(path),
                {
                    "pilot-prose": "dolly:line-005896",
                    "pilot-reasoning": "gsm8k:train-000315",
                    "pilot-code": "mbpp:task-496",
                },
            )

    def test_missing_provider_artifact_emits_no_readiness_or_overlay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "readiness.json"
            overlay = root / "provider-overlay.json"
            args = SimpleNamespace(
                provider_manifest=root / "missing-provider.json",
                report=report,
                provider_overlay=overlay,
            )
            with self.assertRaises(FileNotFoundError):
                assemble(args)
            self.assertFalse(report.exists())
            self.assertFalse(overlay.exists())

    def test_benchmark_response_ids_must_match_each_frozen_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = directory / "manifest.json"
            records = []
            for alias in ("pilot-prose", "pilot-reasoning", "pilot-code"):
                for variant in ("q4_0", "row_a16_checkpoint_zero"):
                    records.append(
                        {
                            "prompt_id": alias,
                            "variant": variant,
                            "warmup": False,
                            "generated_token_ids": [1, 3, 5],
                        }
                    )
            manifest_path.write_text(json.dumps({"records": records}))
            self.assertEqual(len(_exact_response_pairs(json.loads(manifest_path.read_text()))), 3)
            records[-1]["generated_token_ids"] = [1, 3, 6]
            manifest_path.write_text(json.dumps({"records": records}))
            with self.assertRaisesRegex(ValueError, "response IDs differ"):
                _exact_response_pairs(json.loads(manifest_path.read_text()))

    def test_canonical_complete_round_status_counts_accepted_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary)
            heads = [
                {
                    "schema": "eagle_head_state_v1",
                    "task_id": 7,
                    "round_index": 0,
                    "depth": depth,
                    "proposed_token_id": token,
                    "verifier_token_id": token,
                    "verifier_reached": True,
                }
                for depth, token in ((0, 101), (1, 202))
            ]
            write_jsonl(capture / "heads.jsonl", heads)
            write_jsonl(
                capture / "rounds.jsonl",
                [
                    {
                        "schema": "w1ax_eagle_round_v1",
                        "status": "complete",
                        "task_id": 7,
                        "round_index": 0,
                        "n_accepted": 2,
                        "n_proposed": 3,
                    },
                    {
                        "schema": "w1ax_eagle_round_v1",
                        "status": "no_proposal",
                        "task_id": 7,
                        "round_index": 1,
                        "n_accepted": 0,
                        "n_proposed": 0,
                    },
                ],
            )
            self.assertEqual(_wrong_accepted_labels(capture, {7}), (2, 0))
            heads[1]["verifier_reached"] = False
            write_jsonl(capture / "heads.jsonl", heads)
            self.assertEqual(_wrong_accepted_labels(capture, {7}), (2, 1))

    def test_accepted_label_gate_scopes_to_measured_tasks_not_warmups(self):
        aliases = {
            "pilot-prose": "dolly:line-005896",
            "pilot-reasoning": "gsm8k:train-000315",
            "pilot-code": "mbpp:task-496",
        }
        task_ids = (181244361, 181244362, 181244363)
        records = [
            {
                "variant": "row_a16_checkpoint_zero",
                "warmup": False,
                "prompt_id": alias,
                "request_digest": {"task_id": task},
            }
            for task, alias in zip(task_ids, aliases)
        ]
        records.append(
            {
                "variant": "row_a16_checkpoint_zero",
                "warmup": True,
                "prompt_id": "pilot-prose",
                "request_digest": {"task_id": 9001},
            }
        )
        measured_ids = _measured_row_a16_task_ids({"records": records}, aliases)
        self.assertEqual(measured_ids, set(task_ids))

        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary)
            rounds, heads = [], []
            for index in range(31):
                task, round_index = task_ids[index % len(task_ids)], index
                rounds.append(
                    {
                        "status": "complete",
                        "task_id": task,
                        "round_index": round_index,
                        "n_accepted": 1,
                        "n_proposed": 1,
                    }
                )
                heads.append(
                    {
                        "task_id": task,
                        "round_index": round_index,
                        "depth": 0,
                        "proposed_token_id": 42,
                        "verifier_token_id": 42,
                        "verifier_reached": True,
                    }
                )
            for round_index in range(2):
                rounds.append(
                    {
                        "status": "complete",
                        "task_id": 9001,
                        "round_index": round_index,
                        "n_accepted": 1,
                        "n_proposed": 1,
                    }
                )
                heads.append(
                    {
                        "task_id": 9001,
                        "round_index": round_index,
                        "depth": 0,
                        "proposed_token_id": 7,
                        "verifier_token_id": 8,
                        "verifier_reached": True,
                    }
                )
            write_jsonl(capture / "rounds.jsonl", rounds)
            write_jsonl(capture / "heads.jsonl", heads)
            self.assertEqual(_wrong_accepted_labels(capture, measured_ids), (31, 0))
            self.assertEqual(_wrong_accepted_labels(capture, {9001}), (2, 2))

        with self.assertRaisesRegex(ValueError, "duplicated|exactly once"):
            _measured_row_a16_task_ids({"records": records[:-1] + [dict(records[0])]}, aliases)

    def test_canonical_benchmark_records_supply_measured_task_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            records = []
            task_ids = (181244361, 181244362, 181244363)
            for task_id, (alias, source) in zip(
                task_ids,
                (
                    ("pilot-prose", "dolly:line-005896"),
                    ("pilot-reasoning", "gsm8k:train-000315"),
                    ("pilot-code", "mbpp:task-496"),
                ),
            ):
                records.append(
                    {
                        "variant": "row_a16_checkpoint_zero",
                        "warmup": False,
                        "prompt_id": alias,
                        "request_digest": {"task_id": task_id},
                    }
                )
            manifest = {
                "prompt_ids": ["pilot-prose", "pilot-reasoning", "pilot-code"],
                "records": records,
            }
            path.write_text(json.dumps(manifest))
            aliases = {
                "pilot-prose": "dolly:line-005896",
                "pilot-reasoning": "gsm8k:train-000315",
                "pilot-code": "mbpp:task-496",
            }
            self.assertEqual(
                _task_map(manifest, aliases),
                {
                    181244361: "dolly:line-005896",
                    181244362: "gsm8k:train-000315",
                    181244363: "mbpp:task-496",
                },
            )

    def test_identity_report_uses_canonical_student_gguf_key(self):
        identity = {
            "schema": "w1ax_pilot_identity_v1",
            "passed": True,
            "provider_hashes": {f"source_{index}": True for index in range(7)},
            "snapshots_verified": True,
            "export_projection_bits_scales": {f"projection_{index}": True for index in range(9)},
            "checkpoint_sha256": CALIBRATION_PINNED_INPUT_SHA256["checkpoint_zero"],
            "student_gguf_sha256": CALIBRATION_PINNED_INPUT_SHA256["exported_gguf"],
            "server_sha256": "c" * 64,
        }
        manifest = {
            "schema": "binary_rescue_benchmark_v1",
            "status": "complete",
            "hashes": {"binary": "c" * 64},
        }
        result = _identity(identity, IDENTITY_REPORT_SHA256, manifest)
        self.assertEqual(
            result["exported_gguf_sha256"],
            CALIBRATION_PINNED_INPUT_SHA256["exported_gguf"],
        )
        wrong_identity = dict(identity)
        wrong_identity.pop("student_gguf_sha256")
        wrong_identity["exportstudent_gguf"] = CALIBRATION_PINNED_INPUT_SHA256["exported_gguf"]
        with self.assertRaisesRegex(ValueError, "identity report"):
            _identity(wrong_identity, IDENTITY_REPORT_SHA256, manifest)

    def test_gradient_report_uses_source_domains_and_candidate_round_indices(self):
        roots = []
        candidate_rounds = {
            "prose": (0, 5, 1),
            "reasoning": (0, 14, 1),
            "code": (0, 9, 1),
        }
        for domain, prompt in EXPECTED_DOMAINS.items():
            for root_index, round_index in enumerate(candidate_rounds[domain]):
                parent = 4 + root_index
                roots.append(
                    {
                        "domain": prompt,
                        "root_index": root_index,
                        "round_index": round_index,
                        "gradient_tensors": 18,
                        "finite_gradient_tensors": 18,
                        "supported_ce_rows": 2,
                        "context_decoder_positions": list(range(parent)),
                        "proposal_decoder_positions": [parent, parent + 1],
                        "final_cache_length": parent + 2,
                        "f16_cache_writes": parent + 2,
                        "preceding_student_round_outcome": "accepted",
                    }
                )
        report = {
            "schema": "w1ax_pilot_gradient_contract_v1",
            "status": "finite_gradient_and_torch_cache_contract_passed",
            "optimizer_steps": 0,
            "checks": {
                "all_selected_roots_have_supported_hard_ce": True,
                "all_selected_roots_have_18_finite_gradients": True,
                "all_torch_cache_writes_are_finite_f16_exact": True,
                "all_torch_cache_lengths_and_positions_match_trace": True,
                "borrowed_embedding_norm_and_d2t_exact": True,
            },
            "frozen_operand_identity": {
                "embedding_tokens_checked": 12,
                "embedding_dtype": "f16_exact",
                "d2t": "candidate_and_row_absolute_maps_exact_to_native_offsets",
                "norms": {
                    "blk.0.attn_norm.weight": "exact_f32",
                    "blk.0.attn_norm_2.weight": "exact_f32",
                    "blk.0.ffn_norm.weight": "exact_f32",
                    "output_norm.weight": "exact_f32",
                },
            },
            "roots": roots,
        }
        normalized = _gradient_roots(report)
        self.assertEqual(set(normalized), set(EXPECTED_DOMAINS))
        self.assertEqual([row["round_index"] for row in normalized["reasoning"]], [0, 14, 1])
        self.assertEqual(normalized["reasoning"][1]["preceding_student_round_outcome"], "accepted")
        broken = dict(report)
        broken["roots"] = [dict(row) for row in roots]
        broken["roots"][0]["proposal_decoder_positions"] = [9]
        with self.assertRaisesRegex(ValueError, "cache ancestry"):
            _gradient_roots(broken)

    def test_numeric_outcomes_translate_to_readiness_contract_vocabulary(self):
        numeric_roots = {
            domain: [
                {"preceding_student_round_outcome": "unknown"},
                {"preceding_student_round_outcome": "accepted"},
                {"preceding_student_round_outcome": "rejected"},
            ]
            for domain in EXPECTED_DOMAINS
        }
        self.assertEqual(
            _contract_outcome_summary(numeric_roots),
            {
                domain: ["accepted_continuation", "verifier_rejection"]
                for domain in EXPECTED_DOMAINS
            },
        )
        numeric_roots["prose"][2]["preceding_student_round_outcome"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported selected outcome in prose"):
            _contract_outcome_summary(numeric_roots)

    def test_cache_audit_accepts_consistent_cuda_host_causal_mask(self):
        audit = {
            "execution_device": "cuda",
            "mask_device": "cuda_host",
            "mask_buffer_types": ["CUDA_Host"],
        }
        _validate_mask_placement(audit)
        _validate_mask_placement({"mask_device": "cuda"})
        for invalid in (
            {"mask_device": "cuda_host"},
            {"mask_device": "cuda_host", "mask_buffer_types": ["CUDA"]},
            {"mask_device": "cpu", "mask_buffer_types": ["CPU"]},
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "mask placement|placement metadata"),
            ):
                _validate_mask_placement(invalid)

    def test_repeated_head_states_join_by_unique_capture_order(self):
        heads = [
            {
                "task_id": task_id,
                "round_index": 0,
                "state_row": index,
                "input_position": 6,
                "input_token_id": 12,
                "slot_id": 0,
            }
            for index, task_id in enumerate((101, 202))
        ]
        cache_writes = [
            {"position": 5, "slot": 5, "token_id": 12, "execution": execution, "column": 0}
            for execution in (4, 9)
        ]
        joined = _ordered_head_cache_joins(heads, cache_writes)
        self.assertEqual(joined, {(101, 0): (4, 0), (202, 0): (9, 0)})
        with self.assertRaisesRegex(ValueError, "unique chronological"):
            _ordered_head_cache_joins(
                heads[:1] + [{**heads[1], "state_row": 1}],
                cache_writes[:1],
            )

    def test_prenorm_reconstruction_uses_bounded_relative_rms(self):
        prenorm = np.asarray([1.0, -2.0], dtype=np.float32)
        output_norm = np.asarray([0.5, 1.5], dtype=np.float32)
        graph_record = {
            "dtype": "f32",
            "n_tokens": 1,
            "token_width": 2,
            "ne": [2, 1],
            "token_axis": 1,
            "f32_offset": 0,
            "f32_count": 2,
        }
        graph_values = prenorm.astype(np.float64)
        expected = (graph_values / np.sqrt(np.mean(graph_values**2) + 1e-6) * output_norm).astype(
            np.float32
        )
        report = _join_head_state(
            expected,
            {"eagle3_prenorm-0": graph_record},
            0,
            graph_values,
            output_norm,
            "a" * 64,
            1e-6,
        )
        self.assertEqual(report["state_join_method"], "eagle3_prenorm_output_norm_reconstruction")
        self.assertFalse(report["native_result_norm_state_exact"])
        self.assertLessEqual(report["state_relative_rms"], 0.10)
        self.assertEqual(report["output_norm_tensor_sha256"], "a" * 64)
        with self.assertRaisesRegex(ValueError, "0.10 relative-RMS"):
            _join_head_state(
                expected + 1.0,
                {"eagle3_prenorm-0": graph_record},
                0,
                graph_values,
                output_norm,
                "a" * 64,
                1e-6,
            )

    def test_actual_runner_manifest_binds_inline_records_and_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads(
                (ROOT / "configs/w1_phase1b_pilot_cuda_cache_diagnostic.json").read_text()
            )
            binary = root / "llama-server"
            binary.write_bytes(b"native server fixture")
            config["binary"] = str(binary)
            config_path = root / "source-config.json"
            config_path.write_text(json.dumps(config))
            run_dir = root / "benchmark"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(json.dumps(config))
            cell = run_dir / "r00-s01-row_a16_checkpoint_zero"
            cell.mkdir()
            policy = validate_runner_config(config, diagnostic=True)
            spec = config["variants"]["row_a16_checkpoint_zero"]
            command = runner_command(config, spec, policy, diagnostic=True)
            env = runner_server_env(config, spec, "instrumented", cell)
            env = {
                key: value
                for key, value in env.items()
                if key.startswith((*CLEAR_PREFIXES, "CUDA_VISIBLE"))
            }
            block = {
                "variant": "row_a16_checkpoint_zero",
                "directory": str(cell),
                "command": command,
                "env": env,
                "server_exit_code": 0,
                "graph_status": "unverified",
            }
            (cell / "manifest.json").write_text(json.dumps(block))
            manifest = {
                "schema": "binary_rescue_benchmark_v1",
                "status": "complete",
                "prompt_sha256": DIAGNOSTIC_SAFE_PROMPTS_SHA256,
                "q4_variant": "q4_0",
                "policy": policy,
                "hashes": {
                    "binary": sha256(binary),
                    "target": TARGET_GGUF_SHA256,
                    "drafts": {
                        "q4_0": "d" * 64,
                        "row_a16_checkpoint_zero": CALIBRATION_PINNED_INPUT_SHA256["exported_gguf"],
                    },
                },
                "blocks": [block],
                "records": [],
            }
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            self.assertEqual(
                _validate_runner_manifest(manifest_path, manifest, config_path, cell, binary),
                block,
            )

    def test_selected_roots_join_head_seed_torch_cache_and_graph_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture_dir = Path(temporary)
            heads, states, cache_rows, rounds = [], [], [], []
            numeric, gradients, task_map = {}, {}, {}
            graph_rows, graph_values = [], []
            head_states = []
            offset = 0
            output_norm = np.asarray([0.5, 1.5], dtype=np.float32)
            rms_norm_epsilon = 1e-6
            for execution, (domain, prompt) in enumerate(EXPECTED_DOMAINS.items()):
                task_id = 11 + execution
                task_map[task_id] = prompt
                parent = (0, 3, 4)[execution]
                prefix = list(range(parent + 2))
                token = prefix[-1]
                prenorm = np.asarray([execution + 1.0, -(execution + 2.0)], dtype=np.float32)
                state = (
                    prenorm.astype(np.float64)
                    / np.sqrt(np.mean(prenorm.astype(np.float64) ** 2) + rms_norm_epsilon)
                    * output_norm.astype(np.float64)
                ).astype(np.float32)
                head_states.append(state)
                heads.append(
                    {
                        "schema": "eagle_head_state_v1",
                        "task_id": task_id,
                        "slot_id": 0,
                        "round_index": 0,
                        "depth": 0,
                        "parent_position": parent,
                        "input_position": parent + 1,
                        "label_position": parent + 2,
                        "input_token_id": token,
                        "prefix_token_ids": prefix,
                        "state_row": execution,
                        "state_dim": 2,
                        "state_dtype": "float32_native_endian",
                        "state_boundary": "native_output_norm_f32_before_head_operand_conversion",
                        "finite": True,
                        "alignment_valid": True,
                    }
                )
                states.append(
                    {
                        "schema": "eagle_state_v1",
                        "event": "seed",
                        "seq_id": 0,
                        "position": parent,
                        "token": token,
                        "kv_max_before": parent if parent > 0 else -1,
                        "kv_max_after": parent - 1,
                    }
                )
                accepted = int(execution == 2)
                state_hash = f"{execution:016x}"
                states.append(
                    {
                        "schema": "eagle_state_v1",
                        "event": "accept",
                        "seq_id": 0,
                        "accepted": accepted,
                        "verify_rows": 2,
                        "selected_row": accepted,
                        "verify_pos_first": parent,
                        "position": parent + accepted,
                        "selected_hash": state_hash,
                        "pending_hash": state_hash,
                    }
                )
                rounds.append(
                    {
                        "schema": "w1ax_eagle_round_v1",
                        "status": "complete",
                        "task_id": task_id,
                        "slot_id": 0,
                        "round_index": 0,
                        "n_accepted": accepted,
                        "n_proposed": 1,
                    }
                )
                cache_rows.append(
                    {
                        "schema": "eagle_draft_cache_v1",
                        "event": "row",
                        "execution": execution,
                        "column": 0,
                        "position": parent,
                        "slot": parent,
                        "token_id": token,
                    }
                )
                graph_rows.append(
                    {
                        "schema": "eagle_draft_graph_v1",
                        "event": "tensor",
                        "tensor_name": "eagle3_prenorm-0",
                        "dtype": "f32",
                        "group_kind": "decoder",
                        "group_execution": execution,
                        "n_tokens": 1,
                        "token_width": 2,
                        "ne": [2, 1],
                        "token_axis": 1,
                        "f32_offset": offset,
                        "f32_count": 2,
                        "f32_bytes": 8,
                    }
                )
                graph_values.extend(prenorm.tolist())
                offset += 2
                numeric[domain] = [
                    {
                        "round_index": 0,
                        "parent_position": parent,
                        "prefix_token_ids": prefix,
                        "preceding_student_round_outcome": "unknown",
                    }
                ]
                gradients[domain] = [
                    {
                        "root_index": 0,
                        "round_index": 99,
                        "context_decoder_positions": list(range(parent)),
                        "proposal_decoder_positions": [parent],
                        "final_cache_length": parent + 1,
                    }
                ]

            heads_path = capture_dir / "heads.jsonl"
            write_jsonl(heads_path, heads)
            (capture_dir / "heads.f32").write_bytes(
                np.asarray(
                    [value for state in head_states for value in state.tolist()],
                    dtype="<f4",
                ).tobytes()
            )
            write_jsonl(capture_dir / "state.jsonl", states)
            write_jsonl(capture_dir / "rounds.jsonl", rounds)
            write_jsonl(capture_dir / "heads.draft_cache.jsonl", cache_rows)
            graph_path = capture_dir / "heads.draft_graph.jsonl"
            footer = {
                "schema": "eagle_draft_graph_v1",
                "event": "capture_end",
                "status": "complete",
                "reason": "",
                "tensor_rows": len(graph_rows),
                "execution_count": len(graph_rows),
                "bytes_written": len(graph_values) * 4,
                "decoder_groups": len(graph_rows),
                "encoder_groups": 0,
            }
            write_jsonl(graph_path, [*graph_rows, footer])
            graph_values_path = capture_dir / "heads.draft_graph.f32"
            graph_values_path.write_bytes(np.asarray(graph_values, dtype="<f4").tobytes())

            bridge = _bridge_roots(
                numeric,
                gradients,
                task_map,
                capture_dir,
                states,
                cache_rows,
                graph_path,
                graph_values_path,
                output_norm,
                hashlib.sha256(output_norm.astype("<f4").tobytes()).hexdigest(),
                rms_norm_epsilon,
            )
            self.assertEqual(len(bridge), 3)
            self.assertTrue(all(not row["native_result_norm_state_exact"] for row in bridge))
            self.assertTrue(
                all(
                    row["state_join_method"] == "eagle3_prenorm_output_norm_reconstruction"
                    and row["state_relative_rms"] <= 0.10
                    and len(row["output_norm_tensor_sha256"]) == 64
                    for row in bridge
                )
            )
            self.assertEqual([row["state_seed_ordinal"] for row in bridge], [0, 1, 2])
            self.assertEqual([row["logical_cache_length_before_seed"] for row in bridge], [0, 3, 4])
            self.assertEqual(bridge[0]["kv_max_before"], bridge[0]["kv_max_after"])
            self.assertEqual(bridge[1]["kv_max_before"], 3)
            self.assertEqual(bridge[1]["kv_max_after"], 2)

            states[0]["kv_max_after"] = 0
            with self.assertRaisesRegex(ValueError, "reserve-row trim"):
                _bridge_roots(
                    numeric,
                    gradients,
                    task_map,
                    capture_dir,
                    states,
                    cache_rows,
                    graph_path,
                    graph_values_path,
                    output_norm,
                    hashlib.sha256(output_norm.astype("<f4").tobytes()).hexdigest(),
                    rms_norm_epsilon,
                )


if __name__ == "__main__":
    unittest.main()
