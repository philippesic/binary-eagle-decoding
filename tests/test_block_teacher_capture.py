import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from capture_block_qat_teacher import NativeTeacher
from export_block_binary import sha256
from gguf import GGUFWriter


class NativeTeacherTests(unittest.TestCase):
    def target_fixture(self, root, layers=6, full_tokenizer=False):
        target = root / "target.gguf"
        w = GGUFWriter(target, "qwen3")
        w.add_name("synthetic CPU target teacher fixture")
        w.add_chat_template("chatml")
        w.add_block_count(layers)
        w.add_embedding_length(32)
        w.add_feed_forward_length(64)
        w.add_head_count(4)
        w.add_head_count_kv(2)
        w.add_context_length(512)
        w.add_rope_dimension_count(8)
        w.add_layer_norm_rms_eps(1e-6)
        w.add_tokenizer_model("llama")
        tokens = [f"token{i}" for i in range(32)]
        if full_tokenizer:
            tokens[:3] = ["<unk>", "<s>", "</s>"]
            tokens += [f"<0x{i:02X}>" for i in range(256)]
            w.add_token_types([2, 3, 3] + [1] * 29 + [6] * 256)
            w.add_token_scores([0.0] * len(tokens))
        w.add_token_list(tokens)
        rng = np.random.default_rng(714)
        for name in ("token_embd.weight", "output.weight"):
            w.add_tensor(name, rng.normal(0, 0.03, size=(len(tokens), 32)).astype(np.float32))
        w.add_tensor("output_norm.weight", np.ones(32, np.float32))
        for i in range(layers):
            for name in ("attn_norm", "ffn_norm"):
                w.add_tensor(f"blk.{i}.{name}.weight", np.ones(32, np.float32))
            for name in ("attn_q_norm", "attn_k_norm"):
                w.add_tensor(f"blk.{i}.{name}.weight", np.ones(8, np.float32))
            for name, shape in [
                ("attn_q", (32, 32)),
                ("attn_k", (16, 32)),
                ("attn_v", (16, 32)),
                ("attn_output", (32, 32)),
                ("ffn_gate", (64, 32)),
                ("ffn_up", (64, 32)),
                ("ffn_down", (32, 64)),
            ]:
                w.add_tensor(
                    f"blk.{i}.{name}.weight", rng.normal(0, 0.02, size=shape).astype(np.float32)
                )
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        return target

    @unittest.skipUnless(
        os.environ.get("BLOCK_TEACHER_NATIVE"), "actual native target helper not selected"
    )
    def test_persistent_exact_prefix_capture_resets_and_owns_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = self.target_fixture(root)
            with NativeTeacher(
                Path(os.environ["BLOCK_TEACHER_NATIVE"]),
                target,
                root / "raw",
                target_sha256=sha256(target),
                max_tokens=512,
                gpu_layers=0,
                producer_source_revision="0" * 40,
                timeout_seconds=30,
            ) as teacher:
                tokens = [2, 3, 1, 4, 5]
                first = teacher.capture_prefix(
                    tokens,
                    [0, 1, 2, 3, 4],
                    logits_mode="all",
                    chain_ancestry={"split": "synthetic"},
                )
                second = teacher.capture_prefix(tokens, [0, 1, 2, 3, 4], logits_mode="last")
                self.assertEqual(first["target_precision"], "F32")
                self.assertTrue(first["target_storage_buffers"])
                self.assertEqual(first["executed_result_buffers"], ["CPU"])
                self.assertEqual(first["files"]["features"]["shape"], [5, 5, 32])
                self.assertEqual(first["files"]["logits"]["shape"], [5, 32])
                np.testing.assert_array_equal(
                    teacher.array(first, "logits")[-1:], teacher.array(second, "logits")
                )
                np.testing.assert_array_equal(
                    teacher.array(first, "features"), teacher.array(second, "features")
                )
                self.assertTrue(np.isfinite(teacher.array(first, "features")).all())
                self.assertTrue(np.isfinite(teacher.array(first, "logits")).all())
                changed = teacher.capture_prefix(
                    [2, 3, 1, 6, 5], [0, 1, 2, 3, 4], logits_mode="last"
                )
                self.assertFalse(
                    np.array_equal(
                        teacher.array(second, "logits"), teacher.array(changed, "logits")
                    )
                )
                with self.assertRaises(ValueError):
                    teacher.capture_prefix([True], [0, 1, 2, 3, 4])
                with self.assertRaises(ValueError):
                    teacher.capture_prefix(tokens, [0, 1, 2, 3, 3])
                pid = teacher.process.pid
                self.assertEqual(os.getpgid(pid), os.getpgrp())
                self.assertEqual(first["producer_pgid"], os.getpgrp())
            self.assertEqual(teacher.process.returncode, 0)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    @unittest.skipUnless(
        os.environ.get("BLOCK_TEACHER_NATIVE"), "actual native target helper not selected"
    )
    def test_supervisor_group_stop_reaps_actual_native_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = self.target_fixture(root)
            script = """
import json, os, signal, sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from capture_block_qat_teacher import NativeTeacher,supervisor_stop
signal.signal(signal.SIGTERM,supervisor_stop)
with NativeTeacher(Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]),
                   target_sha256=sys.argv[5],max_tokens=32,gpu_layers=0,
                   producer_source_revision='0'*40,timeout_seconds=30) as teacher:
    teacher.capture_prefix([2,3],[0,1,2,3,4])
    print(json.dumps({'pid':teacher.process.pid,'pgid':os.getpgid(teacher.process.pid)}),flush=True)
    signal.pause()
"""
            wrapper = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(Path(__file__).resolve().parents[1] / "scripts"),
                    os.environ["BLOCK_TEACHER_NATIVE"],
                    str(target),
                    str(root / "raw"),
                    sha256(target),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            import selectors

            selector = selectors.DefaultSelector()
            selector.register(wrapper.stdout, selectors.EVENT_READ)
            try:
                self.assertTrue(selector.select(30), "actual teacher startup timed out")
                line = wrapper.stdout.readline()
                self.assertTrue(line, "actual teacher did not emit startup identity")
                identity = json.loads(line)
                self.assertEqual(identity["pgid"], wrapper.pid)
                os.killpg(wrapper.pid, signal.SIGTERM)
                wrapper.wait(timeout=10)
                self.assertEqual(wrapper.returncode, 128 + signal.SIGTERM)
                with self.assertRaises(ProcessLookupError):
                    os.kill(identity["pid"], 0)
                with self.assertRaises(ProcessLookupError):
                    os.killpg(wrapper.pid, 0)
            finally:
                selector.close()
                if wrapper.poll() is None:
                    os.killpg(wrapper.pid, signal.SIGKILL)
                    wrapper.wait(timeout=10)
                wrapper.stdout.close()
                wrapper.stderr.close()

    @unittest.skipUnless(
        os.environ.get("BLOCK_TEACHER_NATIVE"), "actual native target helper not selected"
    )
    def test_native_prompt_generation_and_three_tap_replay(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = self.target_fixture(root, full_tokenizer=True)
            messages = [{"role": "user", "content": "token3 token4"}]
            canonical = json.dumps(
                messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            with NativeTeacher(
                Path(os.environ["BLOCK_TEACHER_NATIVE"]),
                target,
                root / "raw",
                target_sha256=sha256(target),
                max_tokens=128,
                gpu_layers=0,
                producer_source_revision="0" * 40,
                timeout_seconds=30,
            ) as teacher:
                generated = teacher.generate_capture(
                    messages=messages,
                    template_mode="native_chat",
                    max_new_tokens=4,
                    tap_ids=[0, 1, 2, 3, 4],
                    logits_mode="all",
                    chain_ancestry={
                        "prompt_id": "synthetic",
                        "prompt_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                        "domain": "code",
                        "source_split": "TRAIN",
                    },
                )
                boundary = generated["prompt_length"]
                self.assertGreater(boundary, 0)
                self.assertEqual(generated["chain_ancestry"]["prompt_length"], boundary)
                self.assertEqual(
                    generated["prompt_source_sha256"],
                    hashlib.sha256(canonical.encode()).hexdigest(),
                )
                self.assertEqual(generated["generation"]["mode"], "native_target_greedy")
                rows = teacher.array(generated, "logits")
                for i in range(boundary, len(generated["tokens"])):
                    self.assertEqual(generated["tokens"][i], int(rows[i - 1].argmax()))
                repeated = teacher.generate_capture(
                    messages=messages,
                    template_mode="native_chat",
                    max_new_tokens=4,
                    tap_ids=[0, 1, 2, 3, 4],
                    logits_mode="none",
                )
                self.assertEqual(repeated["tokens"], generated["tokens"])
                self.assertEqual(set(repeated["files"]), {"features"})
                self.assertEqual(repeated["logits_shape"], [0, 288])
                replay = teacher.capture_prefix(
                    generated["tokens"],
                    [0, 2, 4],
                    logits_mode="last",
                    decode_history=generated["decode_history"],
                )
                self.assertEqual(replay["features_shape"], [len(generated["tokens"]), 3, 32])
                np.testing.assert_array_equal(
                    teacher.array(generated, "features")[:, [0, 2, 4]],
                    teacher.array(replay, "features"),
                )
                np.testing.assert_array_equal(rows[-1:], teacher.array(replay, "logits"))
                raw = teacher.generate_capture(
                    prompt_text="token3",
                    template_mode="raw_text",
                    max_new_tokens=0,
                    tap_ids=[0, 2, 4],
                    logits_mode="none",
                )
                self.assertEqual(raw["rendered_prompt"], "token3")
                self.assertEqual(raw["prompt_source_sha256"], hashlib.sha256(b"token3").hexdigest())
                self.assertEqual(len(raw["tokens"]), raw["prompt_length"])

    @unittest.skipUnless(
        os.environ.get("BLOCK_TEACHER_NATIVE"), "actual native target helper not selected"
    )
    def test_prompt_cap_refuses_before_decode_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = self.target_fixture(root, full_tokenizer=True)
            teacher = NativeTeacher(
                Path(os.environ["BLOCK_TEACHER_NATIVE"]),
                target,
                root / "raw",
                target_sha256=sha256(target),
                max_tokens=128,
                gpu_layers=0,
                producer_source_revision="0" * 40,
                timeout_seconds=30,
            )
            with self.assertRaisesRegex(RuntimeError, "native teacher exited"):
                teacher.generate_capture(
                    messages=[{"role": "user", "content": "token3 token4"}],
                    template_mode="native_chat",
                    max_new_tokens=4,
                    max_prompt_tokens=8,
                    tap_ids=[0, 1, 2, 3, 4],
                    logits_mode="none",
                )
            self.assertEqual(list((root / "raw").glob("*/features.f32")), [])
            self.assertIn("before decode", Path(teacher.log.name).read_text())
            with self.assertRaises(RuntimeError):
                teacher.close()
            self.assertEqual(teacher.process.returncode, 1)

    @unittest.skipUnless(
        os.environ.get("BLOCK_TEACHER_NATIVE"), "actual native target helper not selected"
    )
    def test_native_out_of_vocab_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = self.target_fixture(root)
            teacher = NativeTeacher(
                Path(os.environ["BLOCK_TEACHER_NATIVE"]),
                target,
                root / "raw",
                target_sha256=sha256(target),
                max_tokens=32,
                gpu_layers=0,
                producer_source_revision="0" * 40,
                timeout_seconds=30,
            )
            with self.assertRaisesRegex(RuntimeError, "exited"):
                teacher.capture_prefix([1000], [0, 1, 2, 3, 4])
            with self.assertRaisesRegex(RuntimeError, "failed"):
                teacher.close()
            self.assertEqual(teacher.process.returncode, 1)


if __name__ == "__main__":
    unittest.main()
