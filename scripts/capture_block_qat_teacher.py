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
import hashlib
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
        tokenizer_metadata = {
            key: field.contents()
            for key, field in target_reader.fields.items()
            if key.startswith("tokenizer.")
        }
        tokenizer_hash = hashlib.sha256(
            json.dumps(
                tokenizer_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        template = tokenizer_metadata.get("tokenizer.chat_template", "")
        template_hash = hashlib.sha256(
            (
                template if isinstance(template, str) else json.dumps(template, sort_keys=True)
            ).encode()
        ).hexdigest()
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
            "tokenizer_metadata_sha256": tokenizer_hash,
            "target_chat_template_sha256": template_hash,
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
        self.native_runtime_binding = None
        self.closed = False

    def capture_prefix(
        self, tokens, tap_ids, *, logits_mode="last", chain_ancestry=None, decode_history=None
    ):
        if self.closed or self.process.poll() is not None:
            raise RuntimeError("native teacher process is not live")
        if (
            not isinstance(tokens, (list, tuple))
            or not tokens
            or len(tokens) > self.max_tokens
            or any(type(n) is not int or n < 0 for n in tokens)
        ):
            raise ValueError("exact nonempty bounded integer token prefix required")
        self._validate_taps(tap_ids)
        if logits_mode not in ("all", "last", "none"):
            raise ValueError("unsupported logits mode")
        request = {
            "id": uuid.uuid4().hex,
            "tokens": list(tokens),
            "tap_ids": list(tap_ids),
            "logits_mode": logits_mode,
        }
        if decode_history is not None:
            if not isinstance(decode_history, list) or not decode_history:
                raise ValueError("nonempty source decode history required")
            offset = 0
            for chunk in decode_history:
                if (
                    not isinstance(chunk, dict)
                    or set(chunk) != {"offset", "count", "phase", "kv_reused_from_same_chain"}
                    or type(chunk["offset"]) is not int
                    or chunk["offset"] != offset
                    or type(chunk["count"]) is not int
                    or not 0 < chunk["count"] <= 256
                    or chunk["phase"] not in ("prefill", "target_only_greedy")
                    or (chunk["phase"] == "target_only_greedy" and chunk["count"] != 1)
                    or type(chunk["kv_reused_from_same_chain"]) is not bool
                    or chunk["kv_reused_from_same_chain"] != (offset > 0)
                ):
                    raise ValueError("invalid exact source decode partition")
                offset += chunk["count"]
            if offset != len(tokens):
                raise ValueError("source decode partitions do not cover exact prefix")
            request["decode_history"] = decode_history
        return self._capture_request(request, chain_ancestry)

    @staticmethod
    def _validate_taps(tap_ids):
        if (
            not isinstance(tap_ids, (list, tuple))
            or len(tap_ids) not in (3, 5)
            or len(set(tap_ids)) != len(tap_ids)
            or any(type(n) is not int or n < 0 for n in tap_ids)
        ):
            raise ValueError("exact ordered three or five distinct native input taps required")

    def generate_capture(
        self,
        *,
        messages=None,
        prompt_text=None,
        template_mode,
        max_new_tokens,
        max_prompt_tokens=None,
        tap_ids,
        logits_mode="all",
        chain_ancestry=None,
    ):
        self._validate_taps(tap_ids)
        if logits_mode not in ("all", "last", "none"):
            raise ValueError("unsupported logits mode")
        if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= self.max_tokens:
            raise ValueError("invalid bounded native continuation")
        if max_prompt_tokens is None:
            max_prompt_tokens = self.max_tokens - max_new_tokens
        if (
            type(max_prompt_tokens) is not int
            or max_prompt_tokens <= 0
            or max_prompt_tokens + max_new_tokens > self.max_tokens
        ):
            raise ValueError("invalid bounded native prompt token cap")
        prompt = {
            "template_mode": template_mode,
            "max_new_tokens": max_new_tokens,
            "max_prompt_tokens": max_prompt_tokens,
        }
        if template_mode == "native_chat" and prompt_text is None:
            if (
                not isinstance(messages, list)
                or not messages
                or any(
                    not isinstance(item, dict)
                    or set(item) != {"role", "content"}
                    or any(not isinstance(value, str) for value in item.values())
                    for item in messages
                )
            ):
                raise ValueError("original role/content messages required")
            prompt["messages"] = messages
            source_text = json.dumps(
                messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        elif template_mode == "raw_text" and messages is None and isinstance(prompt_text, str):
            if not prompt_text:
                raise ValueError("empty raw native prompt")
            prompt["text"] = prompt_text
            source_text = prompt_text
        else:
            raise ValueError("explicit native_chat or raw_text source required")
        request = {
            "id": uuid.uuid4().hex,
            "prompt": prompt,
            "tap_ids": list(tap_ids),
            "logits_mode": logits_mode,
        }
        receipt = self._capture_request(request, chain_ancestry)
        receipt["prompt_source_sha256"] = hashlib.sha256(source_text.encode()).hexdigest()
        receipt["rendered_prompt_sha256"] = hashlib.sha256(
            receipt["rendered_prompt"].encode()
        ).hexdigest()
        receipt["chat_template_sha256"] = hashlib.sha256(
            receipt["chat_template"].encode()
        ).hexdigest()
        if (
            template_mode == "native_chat"
            and receipt["chat_template_sha256"] != self.ancestry["target_chat_template_sha256"]
        ):
            raise ValueError("native selected template differs from frozen GGUF template")
        if chain_ancestry is not None:
            if (
                "prompt_length" in chain_ancestry
                and chain_ancestry["prompt_length"] != receipt["prompt_length"]
            ):
                raise ValueError("caller prompt boundary differs from native tokenization")
            receipt["chain_ancestry"] = {
                **chain_ancestry,
                "prompt_length": receipt["prompt_length"],
            }
        self._save_receipt(receipt)
        return receipt

    def _bind_native_runtime(self):
        """Hash actual mapped project runtime libraries once, on Linux only."""
        if self.native_runtime_binding is not None:
            return self.native_runtime_binding
        maps = Path(f"/proc/{self.process.pid}/maps")
        if not maps.is_file():
            self.native_runtime_binding = {
                "scope": "unavailable_on_non_linux",
                "libraries": [],
                "actual_mapping_checked": False,
            }
            return self.native_runtime_binding
        libraries = {}
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            name = fields[5]
            basename = Path(name).name
            if not basename.startswith(("libllama", "libggml")) or ".so" not in basename:
                continue
            if name.endswith(" (deleted)"):
                raise ValueError("native mapped runtime library was replaced/deleted")
            path = Path(name)
            if path.stat().st_ino != int(fields[4]):
                raise ValueError("native mapped runtime library inode differs from disk")
            if name not in libraries:
                libraries[name] = {"path": name, "sha256": sha256(path)}
        self.native_runtime_binding = {
            "scope": "actual_linux_mapped_project_libraries",
            "actual_mapping_checked": True,
            "libraries": [libraries[name] for name in sorted(libraries)],
            "binary_sha256": self.ancestry["producer_binary_sha256"],
        }
        return self.native_runtime_binding

    def _save_receipt(self, receipt):
        directory = self.root / receipt["id"]
        (directory / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )

    def _capture_request(self, request, chain_ancestry):
        if self.closed or self.process.poll() is not None:
            raise RuntimeError("native teacher process is not live")
        tokens = request.get("tokens")
        logits_mode = request["logits_mode"]
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
        tokens = receipt["tokens"]
        if (
            not tokens
            or len(tokens) > self.max_tokens
            or any(type(n) is not int or n < 0 for n in tokens)
        ):
            raise ValueError("invalid native generated token chain")
        if "prompt" in request:
            generation = receipt.get("generation", {})
            boundary = receipt.get("prompt_length")
            if (
                type(boundary) is not int
                or not 0 < boundary <= len(tokens)
                or generation.get("mode") != "native_target_greedy"
                or generation.get("max_new_tokens") != request["prompt"]["max_new_tokens"]
                or generation.get("generated_tokens") != len(tokens) - boundary
                or generation.get("stop_eog") is not True
                or receipt.get("tokenizer")
                != {"add_special": True, "parse_special": True, "implementation": "llama_tokenize"}
            ):
                raise ValueError("invalid native generation/tokenizer contract")
        directory = self.root / request["id"]
        features_shape, logits_shape = receipt["features_shape"], receipt["logits_shape"]
        if (
            len(features_shape) != 3
            or features_shape[:2] != [len(tokens), len(request["tap_ids"])]
            or features_shape[2] <= 0
            or logits_shape[0]
            != (len(tokens) if logits_mode == "all" else 1 if logits_mode == "last" else 0)
            or logits_shape[1] <= 0
        ):
            raise ValueError("native teacher response shape mismatch")
        files = {}
        descriptors = [("features", features_shape)]
        if logits_mode != "none":
            descriptors.append(("logits", logits_shape))
        for name, shape in descriptors:
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
            "native_runtime_binding": self._bind_native_runtime(),
            "chain_ancestry": chain_ancestry,
            "teacher_context_reset_between_requests": True,
            "prefix_freshness": "native_generated_chain"
            if "prompt" in request
            else "caller_current_student_prefix",
        }
        self._save_receipt(receipt)
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
                if "template_mode" in request:
                    receipt = teacher.generate_capture(
                        messages=request.get("messages"),
                        prompt_text=request.get("prompt_text"),
                        template_mode=request["template_mode"],
                        max_new_tokens=request["max_new_tokens"],
                        max_prompt_tokens=request.get("max_prompt_tokens"),
                        tap_ids=request["tap_ids"],
                        logits_mode=request.get("logits_mode", "all"),
                        chain_ancestry=request.get("chain_ancestry"),
                    )
                else:
                    receipt = teacher.capture_prefix(
                        request["tokens"],
                        request["tap_ids"],
                        logits_mode=request.get("logits_mode", "all"),
                        chain_ancestry=request.get("chain_ancestry"),
                        decode_history=request.get("decode_history"),
                    )
                print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
