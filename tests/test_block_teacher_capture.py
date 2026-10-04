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
    def target_fixture(self, root, layers=6):
        target = root / "target.gguf"
        w = GGUFWriter(target, "qwen3")
        w.add_name("synthetic CPU target teacher fixture")
        w.add_block_count(layers)
        w.add_embedding_length(32)
        w.add_feed_forward_length(64)
        w.add_head_count(4)
        w.add_head_count_kv(2)
        w.add_context_length(512)
        w.add_rope_dimension_count(8)
        w.add_layer_norm_rms_eps(1e-6)
        w.add_tokenizer_model("llama")
        w.add_token_list([f"token{i}" for i in range(32)])
        rng = np.random.default_rng(714)
        for name in ("token_embd.weight", "output.weight"):
            w.add_tensor(name, rng.normal(0, 0.03, size=(32, 32)).astype(np.float32))
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
