"""Host-only compile/format contract; this never grants CUDA execution admission."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class AdmissionTraceHostTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("clang++") or shutil.which("c++"), "host compiler required")
    def test_actual_source_trace_format_dedup_and_default_off(self):
        source = (ROOT / "third_party/llama.cpp/ggml/src/ggml-cuda/w1a1.cu").read_text()
        start = source.index("static void w1ax_admission_trace(")
        end = source.index("void ggml_cuda_w1a1_mul_mat(", start)
        function = source[start:end]
        stub = r"""
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#define GGML_ASSERT assert
#define GGML_LOG_INFO(...) fprintf(stderr,__VA_ARGS__)
#define GGML_CUDA_NAME "CUDA"
struct ggml_backend_cuda_context {int device=0;};
struct ggml_tensor {const char * name;int64_t ne[4];};
"""
        main = r"""
int main() {
    ggml_backend_cuda_context ctx;
    ggml_tensor weights{"fc.w1a1_packed",{400,2560,1,1}};
    w1ax_admission_trace(ctx,&weights,8,12800,2560,7);
    w1ax_admission_trace(ctx,&weights,8,12800,2560,7);
    weights.name="escaped\"name\\tail";
    w1ax_admission_trace(ctx,&weights,1,12800,2560,7);
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cpp, binary = root / "trace.cpp", root / "trace"
            cpp.write_text(stub + function + main)
            subprocess.run(
                [
                    shutil.which("clang++") or shutil.which("c++"),
                    "-std=c++17",
                    str(cpp),
                    "-o",
                    str(binary),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            env = {
                key: value
                for key, value in os.environ.items()
                if key != "GGML_W1AX_ADMISSION_TRACE"
            }
            off = subprocess.run([str(binary)], env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(off.returncode, 0)
            self.assertEqual(off.stderr, "")
            env["GGML_W1AX_ADMISSION_TRACE"] = "1"
            on = subprocess.run([str(binary)], env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(on.returncode, 0)
            records = [
                json.loads(line.removeprefix("W1AX_ADMISSION_TRACE "))
                for line in on.stderr.splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(
                records[0],
                {
                    "schema": "w1ax_cuda_dispatch_v1",
                    "packed": "fc.w1a1_packed",
                    "backend": "CUDA",
                    "device": 0,
                    "activation_bits": 8,
                    "logical_k": 12800,
                    "rows": 2560,
                    "tokens": 7,
                    "packed_type": "i32",
                    "packed_words": 400,
                },
            )
            self.assertEqual(records[1]["packed"], 'escaped"name\\tail')


if __name__ == "__main__":
    unittest.main()
