#!/usr/bin/env python3
"""Persistent native exact-prefix five-tap/full-vocabulary TRAIN teacher client.

Native helper loads target once, resets F16 KV for each request, and writes raw
F32 row-major files. `NativeTeacher.capture_prefix(tokens,tap_ids,logits_mode)`
serves either final-prefix full logits for live DSpark L1, or all rows for bounded
TRAIN sequence capture. Teacher target/precision/hardware ancestry is recorded;
no optimizer or drafter is present. GPU use belongs to the sole operator.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import uuid
from pathlib import Path

import numpy as np
from export_block_binary import sha256
from gguf import GGMLQuantizationType as Type
from gguf import GGUFReader


class NativeTeacher:
    def __init__(
        self,
        binary: Path,
        target: Path,
        output_root: Path,
        *,
        target_sha256: str,
        max_tokens: int,
        gpu_layers: int,
        producer_source_revision: str,
        timeout_seconds: float = 300,
    ):
        self.binary, self.target, self.root = map(Path, (binary, target, output_root))
        if (
            self.binary.resolve() == self.target.resolve()
            or not self.binary.is_file()
            or not self.target.is_file()
        ):
            raise ValueError("distinct native binary and frozen target files required")
        if (
            type(max_tokens) is not int
            or not 0 < max_tokens <= 32768
            or type(gpu_layers) is not int
            or not 0 <= gpu_layers <= 999
        ):
            raise ValueError("invalid token/GPU bounds")
        if (
            not isinstance(producer_source_revision, str)
            or len(producer_source_revision) != 40
            or any(c not in "0123456789abcdef" for c in producer_source_revision)
        ):
            raise ValueError("exact native producer source revision required")
        actual_target = sha256(self.target)
        if actual_target != target_sha256:
            raise ValueError("frozen target SHA256 mismatch")
        if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("invalid timeout")
        target_reader = GGUFReader(self.target)
        target_inventory = {}
        matrix_types = set()
        for tensor in target_reader.tensors:
            name = tensor.tensor_type.name
            target_inventory[name] = target_inventory.get(name, 0) + 1
            if len(tensor.shape) >= 2:
                matrix_types.add(tensor.tensor_type)
        target_precision = (
            "F16"
            if matrix_types == {Type.F16}
            else "F32"
            if matrix_types == {Type.F32}
            else "mixed"
        )
        del target_reader
        self.max_tokens = max_tokens
        self.timeout = timeout_seconds
        self.root.mkdir(parents=True, exist_ok=True)
        self.ancestry = {
            "target_sha256": actual_target,
            "producer_binary_sha256": sha256(self.binary),
            "producer_source_revision": producer_source_revision,
            "client_source_sha256": sha256(Path(__file__)),
            "gpu_layers": gpu_layers,
            "target_kv_type": "F16",
            "target_precision": target_precision,
            "target_tensor_type_inventory": target_inventory,
            "producer_host": os.uname().sysname + " " + os.uname().machine,
        }
        self.log = open(self.root / f"producer-{uuid.uuid4().hex}.log", "w")
        self.process = subprocess.Popen(
            [str(self.binary), str(self.target), str(self.root), str(max_tokens), str(gpu_layers)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            # Stay inside the remote_job process group: a supervisor STOP must
            # signal the teacher alongside its Python owner. The native child
            # never forks, so local timeout escalation targets only its PID.
            start_new_session=False,
        )
        self.ancestry.update(
            producer_pid=self.process.pid,
            producer_pgid=os.getpgid(self.process.pid),
            producer_parent_pid=os.getpid(),
        )
        self.closed = False

    def capture_prefix(self, tokens, tap_ids, *, logits_mode="last", chain_ancestry=None):
        if self.closed or self.process.poll() is not None:
            raise RuntimeError("native teacher process is not live")
        if (
            not isinstance(tokens, (list, tuple))
            or not tokens
            or len(tokens) > self.max_tokens
            or any(type(n) is not int or n < 0 for n in tokens)
        ):
            raise ValueError("exact nonempty bounded integer token prefix required")
        if (
            not isinstance(tap_ids, (list, tuple))
            or len(tap_ids) != 5
            or len(set(tap_ids)) != 5
            or any(type(n) is not int or n < 0 for n in tap_ids)
        ):
            raise ValueError("exact ordered five distinct native input taps required")
        if logits_mode not in ("all", "last"):
            raise ValueError("unsupported logits mode")
        request = {
            "id": uuid.uuid4().hex,
            "tokens": list(tokens),
            "tap_ids": list(tap_ids),
            "logits_mode": logits_mode,
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        import selectors

        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            if not selector.select(self.timeout):
                self.close()
                raise TimeoutError("native teacher request timed out; owned producer stopped")
            line = self.process.stdout.readline()
        finally:
            selector.close()
        if not line:
            raise RuntimeError(f"native teacher exited {self.process.poll()}; see {self.log.name}")
        receipt = json.loads(line)
        if (
            receipt.get("schema") != "block_native_teacher_request_v1"
            or not receipt.get("complete")
            or any(receipt.get(k) != v for k, v in request.items())
        ):
            raise ValueError("native teacher response differs from exact caller prefix")
        directory = self.root / request["id"]
        features_shape, logits_shape = receipt["features_shape"], receipt["logits_shape"]
        if (
            len(features_shape) != 3
            or features_shape[:2] != [len(tokens), 5]
            or features_shape[2] <= 0
            or logits_shape[0] != (len(tokens) if logits_mode == "all" else 1)
            or logits_shape[1] <= 0
        ):
            raise ValueError("native teacher response shape mismatch")
        files = {}
        for name, shape in [("features", features_shape), ("logits", logits_shape)]:
            path = directory / (name + ".f32")
            if path.stat().st_size != 4 * int(np.prod(shape)):
                raise ValueError("native teacher file byte count mismatch")
            files[name] = {
                "path": str(path),
                "shape": shape,
                "dtype": "float32",
                "sha256": sha256(path),
            }
        receipt = {
            **receipt,
            **self.ancestry,
            "files": files,
            "chain_ancestry": chain_ancestry,
            "teacher_context_reset_between_requests": True,
            "prefix_freshness": "caller_current_student_prefix",
        }
        (directory / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        return receipt

    @staticmethod
    def array(receipt, name):
        item = receipt["files"][name]
        return np.memmap(item["path"], dtype=np.float32, mode="r", shape=tuple(item["shape"]))

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
        finally:
            self.process.stdout.close()
            self.log.close()
        if self.process.returncode not in (0, -signal.SIGINT, -signal.SIGTERM, -signal.SIGKILL):
            raise RuntimeError(f"native teacher failed with exit {self.process.returncode}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def supervisor_stop(signum, _frame):
    """Unwind the CLI context so its exact native child is reaped on STOP."""
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("binary", "target", "output-root", "requests"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--producer-source-revision", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--gpu-layers", type=int, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, supervisor_stop)
    signal.signal(signal.SIGINT, supervisor_stop)
    with NativeTeacher(
        args.binary,
        args.target,
        args.output_root,
        target_sha256=args.target_sha256,
        max_tokens=args.max_tokens,
        gpu_layers=args.gpu_layers,
        producer_source_revision=args.producer_source_revision,
    ) as teacher:
        with args.requests.open() as stream:
            for line in stream:
                request = json.loads(line)
                receipt = teacher.capture_prefix(
                    request["tokens"],
                    request["tap_ids"],
                    logits_mode=request.get("logits_mode", "all"),
                    chain_ancestry=request.get("chain_ancestry"),
                )
                print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
