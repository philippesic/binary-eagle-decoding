"""Typed, USER-started native preparation for continuous W1A8/W1A1 hard CE.

Import and audit are CPU only. ``run_stages`` is intentionally an execution API:
only the manual launcher calls it after --start --allow-cuda. It never uses SSH,
activates a monitor, evaluates sealed tests, or executes arbitrary shell hooks.
Fresh native label-only bundles use their own schema; legacy v2 raw-logit
conversion policy remains unchanged.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_recurrent_binary_capture import (  # noqa: E402
    AuditedCapture,
    audit_feature_ledger,
    read_jsonl,
    sha256,
)
from audit_recurrent_response import audit_response  # noqa: E402
from prepare_recurrent_native_features import prepare_native_features  # noqa: E402
from prepare_recurrent_native_rows import prepare, write_prepared  # noqa: E402

from w1a1_eagle.recurrent_trace import RoundAnchor, validate_recurrent_trace  # noqa: E402

LABEL_SCHEMA = "recurrent_native_label_capture_v2"
READINESS_SCHEMA = "w1ax_continuous_readiness_v1"
RECIPE_READINESS_SCHEMA = "w1ax_continuous_readiness_v2"
PROVIDER_SCHEMA = "w1ax_native_train_provider_v2"
STAGES_SCHEMA = "w1ax_continuous_stages_v1"
RETAINED_IMPORT_SCHEMA = "w1ax_retained_native_capture_import_v1"
FROZEN_RUNTIME_MANIFEST_SHA256 = "a199cfdabd81b5ba7414509e31007eab338c125b881a9276a78696cd9208fd5e"
FROZEN_BINARY_SHA256 = "b5093749d67888bc2cafdb6a65c479f4c182f0a904820f1dae4870b6ae66d41c"
Q4_0_GGUF_SHA256 = "2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280"
_VERIFIED_RECORDS = {}
_OBSERVED_RECORDS = {}
PROVIDER_SHARED_WEIGHTS = ("target_gguf", "candidate_d_gguf", "base_draft_gguf")
NATIVE_LABEL_RECEIPT_SCHEMA = "w1ax_native_label_audit_receipt_v1"
HISTORICAL_AUDIT_SCHEMA = "w1ax_historical_native_audit_full_pass_v1"
HISTORICAL_PRODUCER_COMMIT = "7547d253b6bf7d8a04ddb3c868e997afad39f31a"
HISTORICAL_OPERATION_SHA256 = "e77fd345a271dc2f2b549c307ee2f17e04077d5a41d0ef7c850bfac9798a9c37"
HISTORICAL_COMPLETION_SHA256 = "b89e1557743d27c7d343ae510108401e26b7134b4d4e54a6532c38bd8be13ef0"
_HISTORICAL_SOURCE_PROOFS = set()
_HISTORICAL_PASS_PROOFS = {}

LABEL_POLICY = {
    "objective": "hard_ce",
    "labels": "cloned_native_verifier_sampler_at_actual_proposal_prefix",
    "features": "native_target_accepted_prefix_only",
    "prefixes": "teacher_forced_exact_captured_prefix_only",
    "probabilities": "absent",
    "raw_logits_captured": False,
    "existing_raw_retirement_allowed": False,
    "native_feature_storage": "immutable_same_filesystem_raw_hardlink_plus_selected_f32",
}


class NativeCancellationGuard:
    def __init__(self):
        self.previous = {}
        self.cancelling = False
        self.critical = 0
        self.pending = None

    def handle(self, signum, frame):
        if self.cancelling:
            return
        if self.critical:
            self.pending = self.pending or (signum, frame)
            return
        self.cancelling = True
        old = self.previous[signum]
        if callable(old) and old is not signal.default_int_handler:
            old(signum, frame)
        raise InterruptedError(f"user signal {signum}; stopping owned native group")

    @contextmanager
    def defer(self):
        """Defer Python exceptions during spawn/assignment, never block child signals."""
        self.critical += 1
        try:
            yield
        finally:
            self.critical -= 1
            if not self.critical and self.pending is not None:
                signum, frame = self.pending
                self.pending = None
                self.handle(signum, frame)


@contextmanager
def native_cancellation():
    """Chain graceful handlers and protect spawn registration before cleanup."""
    guard = NativeCancellationGuard()
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            guard.previous[sig] = signal.signal(sig, guard.handle)
        yield guard
    finally:
        for sig, old in guard.previous.items():
            signal.signal(sig, old)


def check_stop(run_dir: Path) -> None:
    if (Path(run_dir) / "STOP").exists():
        raise InterruptedError("user STOP requested during manual preparation")


def require_unsealed_prompts(
    path: Path, *, sources: dict | None = None, expected_sha256: str | None = None
) -> None:
    # Inspect names/opaque role metadata before any prompt-byte SHA/read.
    if any(word in str(path).lower() for word in ("sealed", "final")):
        raise ValueError("sealed/final prompt contents are prohibited")
    if sources and sources.get("corpus_manifest"):
        corpus_path = checked_record(sources["corpus_manifest"])
        corpus = json.loads(corpus_path.read_text())
        for shard in corpus["files"].get("sealed_test", {}).get("shards", []):
            reserved = (corpus_path.parent / shard["prompts"]).resolve()
            if path.resolve() == reserved or expected_sha256 == shard["prompts_sha256"]:
                raise ValueError("prompt input belongs to the opaque sealed-test role")


def native_runtime_inventory(binary: Path) -> dict:
    directory = binary.resolve().parent
    manifest = directory.parent / "manifest.json"
    if (
        sha256(binary) != FROZEN_BINARY_SHA256
        or not manifest.is_file()
        or sha256(manifest) != FROZEN_RUNTIME_MANIFEST_SHA256
    ):
        raise ValueError("continuous native execution requires the retained frozen b4 runtime")
    # The published freeze manifest is the immutable ancestry authority; recurse
    # through its SHA fields without inventing its archive-specific key layout.
    pinned = set(re.findall(r"[0-9a-f]{64}", manifest.read_text()))
    libraries = {
        p.resolve()
        for p in directory.glob("lib*.so*")
        if p.name.startswith(("libllama", "libggml")) and p.is_file()
    }
    if not any(p.name.startswith("libllama") for p in libraries) or not any(
        p.name.startswith("libggml-cuda") for p in libraries
    ):
        raise ValueError("frozen native llama/CUDA library inventory incomplete")
    records = []
    for path in sorted(libraries):
        digest = sha256(path)
        if digest not in pinned:
            raise ValueError("native library differs from published immutable runtime manifest")
        records.append({"path": str(path), "sha256": digest})
    return {
        "schema": "w1ax_frozen_native_runtime_v1",
        "directory": str(directory),
        "immutable_manifest": file_record(manifest),
        "libraries": records,
        "ld_library_path": str(directory),
    }


def verify_sources(sources: dict) -> None:
    required = {
        "binary",
        "target_gguf",
        "candidate_d_gguf",
        "base_draft_gguf",
        "absolute_d2t",
        "model_snapshot_manifest",
    }
    if not required <= set(sources.get("sha256", {})):
        raise ValueError("native stage source hash inventory incomplete")
    for name, digest in sources["sha256"].items():
        checked_record({"path": str(Path(sources[name]).resolve()), "sha256": digest})
    if sources.get("native_runtime"):
        runtime = sources["native_runtime"]
        checked_record(runtime["immutable_manifest"])
        for library in runtime["libraries"]:
            checked_record(library)


def audit_q4_file(path: Path) -> dict:
    """CPU-only stored-type check for the primary native comparison baseline."""
    if sha256(path) != Q4_0_GGUF_SHA256:
        raise ValueError("Q4_0 comparison differs from the frozen primary baseline SHA256")
    sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))
    from audit_eagle_w1a1_gguf import SOURCE_NAMES
    from gguf import GGMLQuantizationType, GGUFReader

    reader = GGUFReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if reader.fields["general.architecture"].contents() != "eagle3":
        raise ValueError("Q4_0 baseline must be an EAGLE-3 model")
    if any(
        tensors.get(name + ".weight") is None
        or tensors[name + ".weight"].tensor_type != GGMLQuantizationType.Q4_0
        for name in SOURCE_NAMES
    ):
        raise ValueError("primary Q4_0 baseline must store all nine linears as Q4_0")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "architecture": "eagle3",
        "selected_projections": 9,
        "stored_type": "Q4_0",
    }


def host_admission(stage: str, additional_bytes: int) -> dict:
    from w1a1_eagle.continuous_resources import linux_host_memory, require_host_memory

    return require_host_memory(
        linux_host_memory(), floor_bytes=2 * 1024**3, additional_bytes=additional_bytes, stage=stage
    )


def write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def file_record(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def _stat_identity(stat) -> tuple:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def checked_record(record: dict) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("expected an exact path/SHA256 record")
    if (
        not isinstance(record["path"], str)
        or not isinstance(record["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
    ):
        raise ValueError("expected an absolute path and lowercase SHA256")
    path = Path(record["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("evidence must be an absolute regular file")
    stat = path.stat()
    identity = (str(path), record["sha256"], *_stat_identity(stat))
    if identity not in _VERIFIED_RECORDS:
        if sha256(path) != record["sha256"]:
            raise ValueError("evidence SHA256 mismatch")
        if _stat_identity(path.stat()) != _stat_identity(stat):
            raise ValueError("evidence changed while hashing")
        _VERIFIED_RECORDS[identity] = True
    return path


def _files(manifest: dict, directory: Path) -> dict[str, Path]:
    files = {}
    for name, record in manifest["files"].items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError("invalid label-only file record")
        basename = record["path"]
        if not isinstance(basename, str) or Path(basename).name != basename:
            raise ValueError("label-only files must be owned basenames")
        path = directory / basename
        if path.is_symlink() or not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError("label-only file missing or changed")
        files[name] = path
    return files


def _receipt_files(manifest: dict, directory: Path) -> dict[str, Path]:
    """Receipt integrity uses the process hash cache; the full auditor stays exact."""
    files = {}
    for name, record in manifest["files"].items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError("invalid label-only file record")
        basename = record["path"]
        if not isinstance(basename, str) or Path(basename).name != basename:
            raise ValueError("label-only files must be owned basenames")
        path = directory / basename
        checked_record({"path": str(path.absolute()), "sha256": record["sha256"]})
        files[name] = path
    return files


def native_label_audit_source() -> dict:
    """Conservative identity of the auditor and its local/runtime dependencies."""
    import torch

    names = (
        "scripts/w1ax_continuous_stages.py",
        "scripts/w1ax_capture_provider.py",
        "scripts/audit_recurrent_binary_capture.py",
        "scripts/audit_recurrent_response.py",
        "scripts/audit_recurrent_continuity.py",
        "scripts/prepare_recurrent_native_rows.py",
        "scripts/prepare_recurrent_native_features.py",
        "src/w1a1_eagle/__init__.py",
        "src/w1a1_eagle/adapter.py",
        "src/w1a1_eagle/fake_binary.py",
        "src/w1a1_eagle/fake_uniform.py",
        "src/w1a1_eagle/recurrent_trace.py",
    )
    return {
        "files": {name: _observed_record(ROOT / name)["sha256"] for name in names},
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def _observed_record(path: Path) -> dict:
    """Observe an unpinned small source/sidecar once per process file identity."""
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("native label receipt evidence must be a regular file")
    stat = path.stat()
    identity = (str(path), *_stat_identity(stat))
    if identity not in _OBSERVED_RECORDS:
        record = {"path": str(path), "sha256": sha256(path)}
        if _stat_identity(path.stat()) != _stat_identity(stat):
            raise ValueError("native label receipt evidence changed while hashing")
        _OBSERVED_RECORDS[identity] = record
        _VERIFIED_RECORDS[(str(path), record["sha256"], *identity[1:])] = True
    record = _OBSERVED_RECORDS[identity]
    checked_record(record)
    return record


def _native_label_receipt_binding(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_prompt_sha256: str,
    expected_prompt_count: int,
) -> dict:
    """Verify actual bytes, including hardlinks, and owned inventory on each access.

    Only in-process successful hashes are reused; their keys contain expected
    SHA, path, device/inode, size, mtime and ctime. Nothing persists stat trust.
    The manifest digest binds split, source cell, cache/sampler, teacher ancestry
    and every semantic field; all referenced files are independently verified.
    """
    manifest_path = Path(manifest_path).absolute()
    if manifest_path.is_symlink():
        raise ValueError("native label manifest must be an owned regular file")
    manifest_path = checked_record(
        {
            "path": str(manifest_path.resolve()),
            "sha256": expected_manifest_sha256,
        }
    )
    m = json.loads(manifest_path.read_text())
    if (
        m.get("schema") != LABEL_SCHEMA
        or m.get("storage_policy") != LABEL_POLICY
        or m.get("training_eligible") is not False
        or m.get("readiness") != "preparation_only"
        or m.get("split") not in {"train", "development"}
        or m.get("prompts_sha256") != expected_prompt_sha256
        or m.get("prompt_count") != expected_prompt_count
        or type(expected_prompt_count) is not int
        or not 1 <= expected_prompt_count <= 32
    ):
        raise ValueError("native label receipt prompt/split/policy binding differs")
    files = _receipt_files(m, manifest_path.parent)
    owned = {manifest_path.name: expected_manifest_sha256}
    for name, path in files.items():
        owned[path.name] = m["files"][name]["sha256"]
    checked_record({"path": str(files["prompts"]), "sha256": expected_prompt_sha256})
    requests = m.get("requests")
    if not isinstance(requests, list):
        raise ValueError("native label receipt requests missing")
    for record in requests:
        for kind in ("request", "response", "prompt"):
            rec = record[kind]
            if (
                not isinstance(rec, dict)
                or set(rec) != {"path", "sha256"}
                or not isinstance(rec["path"], str)
                or Path(rec["path"]).name != rec["path"]
            ):
                raise ValueError("native label receipt request ownership differs")
            checked_record(
                {
                    "path": str(manifest_path.parent / rec["path"]),
                    "sha256": rec["sha256"],
                }
            )
            owned[rec["path"]] = rec["sha256"]
    # The ordinary sidecar is bound as input, never accepted as audit provenance.
    sidecar = manifest_path.parent / "audit.json"
    if sidecar.exists():
        if sidecar.is_symlink() or not sidecar.is_file():
            raise ValueError("native label sidecar must be an owned regular file")
        owned[sidecar.name] = _observed_record(sidecar)["sha256"]
    if {p.name for p in manifest_path.parent.iterdir()} != set(owned):
        raise ValueError("native label receipt owned inventory differs")
    return {
        "capture_manifest": {"path": str(manifest_path), "sha256": expected_manifest_sha256},
        "split": m["split"],
        "prompts_sha256": expected_prompt_sha256,
        "prompt_count": expected_prompt_count,
        "owned_files": owned,
        "audit_source": native_label_audit_source(),
    }


def audit_native_labels_with_receipt(
    manifest_path: Path,
    *,
    expected_prompt_sha256: str,
    expected_prompt_count: int,
    expected_manifest_sha256: str,
    receipt_path: Path,
) -> dict:
    """Explicit external per-shard cache; first use always does the full audit.

    Wire API: callers bind the immutable manifest SHA and an absolute receipt
    path outside the label directory. Existing mismatched receipts fail closed.
    An old ``audit.json`` or status ordinal cannot initialize a receipt. Historical
    retained-import full-pass provenance may initialize a historical receipt
    through its separate restricted adoption path. Otherwise missing receipts
    require the current full semantic audit once. No eligibility is conferred.
    """
    receipt_path = Path(receipt_path)
    manifest_path = Path(manifest_path).absolute()
    if manifest_path.is_symlink():
        raise ValueError("native label manifest must be an owned regular file")
    manifest_path = manifest_path.resolve()
    if (
        not receipt_path.is_absolute()
        or receipt_path.is_symlink()
        or receipt_path.resolve().is_relative_to(manifest_path.parent.resolve())
    ):
        raise ValueError("native label audit receipt must be external and absolute")
    binding = _native_label_receipt_binding(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_prompt_count=expected_prompt_count,
    )
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (
            not isinstance(receipt, dict)
            or set(receipt) - {"audit_origin"}
            != {"schema", "binding", "report", "report_sha256", "full_semantic_audit"}
            or receipt["schema"] != NATIVE_LABEL_RECEIPT_SCHEMA
            or receipt["full_semantic_audit"] is not True
            or receipt["binding"] != binding
        ):
            raise ValueError("native label audit receipt source/input binding differs")
        report = receipt["report"]
        if (
            not isinstance(report, dict)
            or receipt["report_sha256"] != _native_label_report_sha256(report)
            or report.get("schema") != "recurrent_native_label_audit_v2"
            or report.get("capture_manifest_sha256") != expected_manifest_sha256
            or report.get("training_prompts_sha256") != expected_prompt_sha256
            or report.get("training_prompt_count") != expected_prompt_count
            or report.get("split") != binding["split"]
            or report.get("training_eligible") is not False
        ):
            raise ValueError("native label audit receipt report binding differs")
        if "audit_origin" in receipt:
            _validate_historical_receipt_origin(receipt["audit_origin"], binding, report)
        return report
    report = audit_native_labels(
        manifest_path,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_prompt_count=expected_prompt_count,
    )
    # Reject a mutation during the semantic audit before publication.
    if binding != _native_label_receipt_binding(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_prompt_count=expected_prompt_count,
    ):
        raise ValueError("native label audit inputs changed during audit")
    _publish_native_label_receipt(receipt_path, binding, report)
    return report


def _native_label_report_sha256(report: dict) -> str:
    return hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _publish_native_label_receipt(
    receipt_path: Path,
    binding: dict,
    report: dict,
    *,
    audit_origin: dict | None = None,
) -> None:
    """Called only after a successful full audit and unchanged input binding."""
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": NATIVE_LABEL_RECEIPT_SCHEMA,
        "binding": binding,
        "report": report,
        "report_sha256": _native_label_report_sha256(report),
        "full_semantic_audit": True,
    }
    if audit_origin is not None:
        receipt["audit_origin"] = audit_origin
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=receipt_path.parent,
            prefix=".native-audit-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, receipt_path)
        directory_fd = os.open(receipt_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def audit_native_labels(
    manifest_path: Path, *, expected_prompt_sha256: str, expected_prompt_count: int
) -> dict:
    """Re-derive labels, accepted features, response and trace joins on CPU."""
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    if (
        m.get("schema") != LABEL_SCHEMA
        or m.get("storage_policy") != LABEL_POLICY
        or m.get("training_eligible") is not False
        or m.get("readiness") != "preparation_only"
        or m.get("split") not in {"train", "development"}
        or m.get("prompts_sha256") != expected_prompt_sha256
        or m.get("prompt_count") != expected_prompt_count
    ):
        raise ValueError("native label-only policy/ownership differs")
    if type(expected_prompt_count) is not int or not 1 <= expected_prompt_count <= 32:
        raise ValueError("native label-only shards must contain 1..32 prompts")
    f = _files(m, manifest_path.parent)
    if (
        f["native_feature_values"].stat().st_size > 131072 * 7680 * 4
        or f["native_heads"].stat().st_size > 256 * 1024**2
        or f["native_feature_events"].stat().st_size > 512 * 1024**2
    ):
        raise ValueError("native label-only shard exceeds CPU memory/file bounds")
    expected = {
        "prompts",
        "source_cell",
        "native_heads",
        "native_rounds",
        "native_trace",
        "native_feature_events",
        "native_feature_values",
        "task_map",
        "absolute_d2t",
        "rows",
        "anchors",
        "offsets",
        "t2d",
        "features",
        "feature_rows",
    }
    if set(f) != expected:
        raise ValueError("native label-only file inventory differs")
    prompts = read_jsonl(f["prompts"])
    ids = [p["id"] for p in prompts]
    if (
        sha256(f["prompts"]) != expected_prompt_sha256
        or len(ids) != expected_prompt_count
        or len(set(ids)) != len(ids)
        or any("final" in p.lower() for p in ids)
    ):
        raise ValueError("native label-only prompt count/hash differs or includes sealed final")
    cell = json.loads(f["source_cell"].read_text())
    if (
        cell.get("complete") is not True
        or cell.get("target_sha256") != m["target_sha256"]
        or cell.get("draft_sha256") != m["draft_sha256"]
        or cell.get("env", {}).get("EAGLE_CAPTURE_TARGET_LOGITS") is not None
        or cell.get("env", {}).get("GGML_W1AX_ACT_BITS") != str(m["activation_bits"])
        or cell.get("binary_sha256") != m["binary_sha256"]
    ):
        raise ValueError("native label-only source contract differs")
    # The generic preparer uses train as its structural trace vocabulary. The
    # immutable external capture split remains development where declared; no
    # train prompts are relabeled or reused as development.
    derived = prepare(
        heads_path=f["native_heads"],
        rounds_path=f["native_rounds"],
        task_map_path=f["task_map"],
        prompts_path=f["prompts"],
        absolute_map_path=f["absolute_d2t"],
        cell_manifest_path=f["source_cell"],
        target_vocab_size=m["target_vocab_size"],
        expected_prompt_hash=expected_prompt_sha256,
        expected_prompt_count=expected_prompt_count,
    )
    rows, anchors, offsets, t2d, _ = derived
    if (
        rows != read_jsonl(f["rows"])
        or anchors != read_jsonl(f["anchors"])
        or not np.array_equal(offsets, np.load(f["offsets"], allow_pickle=False))
        or not np.array_equal(t2d, np.load(f["t2d"], allow_pickle=False))
        or any(r.get("target_logits_row") is not None for r in rows)
    ):
        raise ValueError("native label-only derivation changed labels/prefix/map")
    with tempfile.TemporaryDirectory(prefix="label-audit-") as temp:
        temp = Path(temp)
        prepare_native_features(
            f["native_feature_events"],
            f["native_feature_values"],
            f["anchors"],
            f["task_map"],
            f["prompts"],
            temp / "features",
            cell_manifest_path=f["source_cell"],
            expected_prompt_hash=expected_prompt_sha256,
            expected_prompt_count=expected_prompt_count,
            target_vocab_size=m["target_vocab_size"],
        )
        if read_jsonl(temp / "features/feature_rows.jsonl") != read_jsonl(
            f["feature_rows"]
        ) or sha256(temp / "features/features.npy") != sha256(f["features"]):
            raise ValueError("native accepted-prefix feature derivation differs")
    bound_anchors = [RoundAnchor(**a) for a in anchors]
    trace = validate_recurrent_trace(
        rows,
        bound_anchors,
        offsets=offsets,
        target_vocab_size=m["target_vocab_size"],
        draft_vocab_size=m["draft_vocab_size"],
        max_depth=m["max_depth"],
        allowed_prompt_ids=set(ids),
        split="train",
    )
    ledger = audit_feature_ledger(
        f["features"], f["feature_rows"], bound_anchors, set(ids), m["target_vocab_size"]
    )
    requests = m.get("requests")
    if not isinstance(requests, list) or len(requests) != len(cell["requests"]):
        raise ValueError("native label-only requests incomplete")
    for original, record in zip(cell["requests"], requests, strict=True):
        if original["id"] != record["id"] or original["task_id"] != record["task_id"]:
            raise ValueError("native response request ownership differs")
        paths = {}
        for kind in ("request", "response", "prompt"):
            rec = record[kind]
            path = manifest_path.parent / rec["path"]
            if (
                Path(rec["path"]).name != rec["path"]
                or path.is_symlink()
                or sha256(path) != rec["sha256"]
                or rec["sha256"] != original[kind + "_sha256"]
            ):
                raise ValueError("native response source hash differs")
            paths[kind] = path
        request_options = json.loads(paths["request"].read_text())
        if (
            request_options.get("temperature") != 0
            or request_options.get("seed") != 42
            or request_options.get("cache_prompt") is not False
        ):
            raise ValueError("native label sampler/cache options differ from the frozen contract")
        audit_response(
            f["native_rounds"],
            f["native_trace"],
            f["native_feature_events"],
            f["task_map"],
            paths["request"],
            paths["response"],
            record["task_id"],
        )
    claimed = {manifest_path.name, *(p.name for p in f.values())}
    for record in requests:
        claimed.update(record[k]["path"] for k in ("prompt", "request", "response"))
    if (manifest_path.parent / "audit.json").exists():
        claimed.add("audit.json")
    if {p.name for p in manifest_path.parent.iterdir()} != claimed:
        raise ValueError("label-only directory has unclaimed payloads")
    return {
        "schema": "recurrent_native_label_audit_v2",
        "execution_device": "cpu",
        "capture_manifest_sha256": sha256(manifest_path),
        "split": m["split"],
        "training_prompts_sha256": expected_prompt_sha256,
        "training_prompt_count": expected_prompt_count,
        "counts": dict(trace.counts),
        "feature_ledger": ledger,
        "response_requests": len(requests),
        "training_eligible": False,
        "raw_target_logits_in_bundle": False,
        "probability_recomputation": "unavailable_label_only_storage",
    }


def load_native_labels(
    manifest_path: Path,
    *,
    expected_prompt_sha256: str,
    expected_prompt_count: int,
    audit_receipt: dict | None = None,
    expected_manifest_sha256: str | None = None,
) -> AuditedCapture:
    if audit_receipt is None:
        report = audit_native_labels(
            manifest_path,
            expected_prompt_sha256=expected_prompt_sha256,
            expected_prompt_count=expected_prompt_count,
        )
    else:
        # Provider manifests bind an already-published external receipt by SHA.
        receipt_path = checked_record(audit_receipt)
        if expected_manifest_sha256 is None:
            raise ValueError("native label receipt requires explicit capture manifest SHA")
        report = audit_native_labels_with_receipt(
            manifest_path,
            expected_prompt_sha256=expected_prompt_sha256,
            expected_prompt_count=expected_prompt_count,
            expected_manifest_sha256=expected_manifest_sha256,
            receipt_path=receipt_path,
        )
        manifest_path = Path(manifest_path).resolve()
    f = (_receipt_files if audit_receipt is not None else _files)(
        json.loads(Path(manifest_path).read_text()),
        Path(manifest_path).parent,
    )
    anchors = {
        (a["prompt_id"], a["round_index"]): RoundAnchor(**a) for a in read_jsonl(f["anchors"])
    }
    rows = defaultdict(list)
    for r in read_jsonl(f["rows"]):
        rows[(r["prompt_id"], r["round_index"])].append(r)
    lookup = {
        (r["prompt_id"], tuple(r["prefix_token_ids"])): r["feature_row"]
        for r in read_jsonl(f["feature_rows"])
    }
    return AuditedCapture(
        report,
        anchors,
        {k: tuple(v) for k, v in rows.items()},
        lookup,
        np.load(f["features"], mmap_mode="r", allow_pickle=False),
    )


def build_native_labels(
    capture_root: Path,
    prompts: Path,
    absolute_d2t: Path,
    output: Path,
    *,
    split: str,
    target_vocab_size: int = 151936,
    audit_receipt_path: Path | None = None,
) -> dict:
    """Publish fresh label-only data; never rewrite or retire a prior source."""
    require_unsealed_prompts(prompts)
    if split not in {"train", "development"} or output.exists():
        raise ValueError("new label-only output and train/development ownership required")
    if audit_receipt_path is not None:
        audit_receipt_path = Path(audit_receipt_path)
        if (
            not audit_receipt_path.is_absolute()
            or audit_receipt_path.is_symlink()
            or audit_receipt_path.resolve().is_relative_to(output.resolve())
            or audit_receipt_path.exists()
        ):
            raise ValueError("fresh native label audit requires a new external receipt path")
    cell = capture_root / "d_d"
    source = json.loads((cell / "manifest.json").read_text())
    prompt_hash, count = sha256(prompts), len(read_jsonl(prompts))
    task_map = capture_root / "task_prompt_ids.json"
    if task_map.exists():
        if json.loads(task_map.read_text()) != source["task_prompt_ids"]:
            raise ValueError("existing native task map differs; preserve source bytes")
    else:
        write_json(task_map, source["task_prompt_ids"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-labels-", dir=output.parent) as temp:
        temp = Path(temp)
        rows_dir, feature_dir, publish = temp / "rows", temp / "features", temp / "publish"
        write_prepared(
            rows_dir,
            prepare(
                heads_path=cell / "heads.jsonl",
                rounds_path=cell / "forced-rounds.jsonl",
                task_map_path=task_map,
                prompts_path=prompts,
                absolute_map_path=absolute_d2t,
                target_vocab_size=target_vocab_size,
                cell_manifest_path=cell / "manifest.json",
                expected_prompt_hash=prompt_hash,
                expected_prompt_count=count,
            ),
        )
        prepare_native_features(
            cell / "heads.target_features.jsonl",
            cell / "heads.target_features.f32",
            rows_dir / "anchors.jsonl",
            task_map,
            prompts,
            feature_dir,
            cell_manifest_path=cell / "manifest.json",
            expected_prompt_hash=prompt_hash,
            expected_prompt_count=count,
            target_vocab_size=target_vocab_size,
        )
        publish.mkdir()
        sources = {
            "prompts": prompts,
            "source_cell": cell / "manifest.json",
            "native_heads": cell / "heads.jsonl",
            "native_rounds": cell / "forced-rounds.jsonl",
            "native_trace": cell / "rounds.jsonl",
            "native_feature_events": cell / "heads.target_features.jsonl",
            "native_feature_values": cell / "heads.target_features.f32",
            "task_map": task_map,
            "absolute_d2t": absolute_d2t,
            **{
                name: rows_dir / (name + suffix)
                for name, suffix in (
                    ("rows", ".jsonl"),
                    ("anchors", ".jsonl"),
                    ("offsets", ".npy"),
                    ("t2d", ".npy"),
                )
            },
            "features": feature_dir / "features.npy",
            "feature_rows": feature_dir / "feature_rows.jsonl",
        }
        records = {}
        for name, path in sources.items():
            dest = publish / ("source_cell_manifest.json" if name == "source_cell" else path.name)
            if name == "native_feature_values":
                os.link(path, dest)  # No copy fallback: forecast depends on shared immutable bytes.
            else:
                shutil.copyfile(path, dest)
            records[name] = {"path": dest.name, "sha256": sha256(dest)}
        requests = []
        for index, request in enumerate(source["requests"]):
            record = {"id": request["id"], "task_id": request["task_id"]}
            for kind in ("request", "response", "prompt"):
                dest = publish / f"request-{index:05d}-{kind}.json"
                shutil.copyfile(cell / f"request-{index:03d}" / (kind + ".json"), dest)
                record[kind] = {"path": dest.name, "sha256": sha256(dest)}
            requests.append(record)
        m = {
            "schema": LABEL_SCHEMA,
            "split": split,
            "storage_policy": LABEL_POLICY,
            "training_eligible": False,
            "readiness": "preparation_only",
            "prompts_sha256": prompt_hash,
            "prompt_count": count,
            "files": records,
            "requests": requests,
            "max_depth": 5,
            "target_vocab_size": target_vocab_size,
            "draft_vocab_size": int(len(np.load(absolute_d2t, allow_pickle=False))),
            "activation_bits": int(source["env"]["GGML_W1AX_ACT_BITS"]),
            "target_sha256": source["target_sha256"],
            "draft_sha256": source["draft_sha256"],
            "binary_sha256": source["binary_sha256"],
        }
        # Compatibility keys are aliases to the same pinned owned files.
        m.update(
            {
                name: records[name]
                for name in ("rows", "anchors", "offsets", "t2d", "features", "feature_rows")
            }
        )
        write_json(publish / "manifest.json", m)
        audit_source = native_label_audit_source() if audit_receipt_path is not None else None
        report = audit_native_labels(
            publish / "manifest.json",
            expected_prompt_sha256=prompt_hash,
            expected_prompt_count=count,
        )
        write_json(publish / "audit.json", report)
        os.rename(publish, output)
        if audit_receipt_path is not None:
            binding = _native_label_receipt_binding(
                output / "manifest.json",
                expected_manifest_sha256=report["capture_manifest_sha256"],
                expected_prompt_sha256=prompt_hash,
                expected_prompt_count=count,
            )
            if binding["audit_source"] != audit_source:
                raise ValueError("native label audit source changed during publication")
            _publish_native_label_receipt(audit_receipt_path, binding, report)
    return json.loads((output / "manifest.json").read_text())


def native_capture(
    sources: dict,
    prompts: Path,
    output: Path,
    *,
    activation_bits: int = 16,
    draft: Path | None = None,
    cache_gate: bool = False,
    tokens: int = 128,
    stop_file: Path | None = None,
) -> None:
    """Explicit local CUDA native execution, invoked only by the user's start."""
    from run_binary_head_capture import run_cell

    require_unsealed_prompts(prompts, sources=sources)
    verify_sources(sources)
    from w1ax_capture_provider import CANDIDATE_D_SHA256, TARGET_GGUF_SHA256

    if (
        sources["sha256"]["target_gguf"] != TARGET_GGUF_SHA256
        or sources["sha256"]["candidate_d_gguf"] != CANDIDATE_D_SHA256
    ):
        raise ValueError("native capture changed frozen target or norm/map reference")
    if not sources.get("native_runtime"):
        raise ValueError("native capture requires the frozen executable/library inventory")
    host_admission("native teacher capture host staging", 2 * 1024**3)
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()  # USER-start only; release prior gate/evaluation cache.
    if activation_bits not in {1, 4, 8, 16} or not 1 <= tokens <= 128:
        raise ValueError("unsupported native capture arithmetic or token cap")
    if "final" in str(prompts).lower():
        raise ValueError("sealed final is prohibited")
    rows = read_jsonl(prompts)
    if not 1 <= len(rows) <= 32:
        raise ValueError("native captures must use bounded 1..32-prompt microshards")
    if any("final" in str(p["id"]).lower() for p in rows):
        raise ValueError("sealed final is prohibited")
    if stop_file is None and sources.get("stop_file"):
        stop_file = Path(sources["stop_file"])
    if stop_file is not None and stop_file.exists():
        raise InterruptedError("user STOP requested before native capture")
    output.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(
        mode="recurrent-train",
        output=output,
        prompts=prompts,
        binary=Path(sources["binary"]),
        target=Path(sources["target_gguf"]),
        port=sources.get("port", 18092),
        tokens=tokens,
        target_vocab_size=151936,
        target_logits_limit=1,
        target_features_limit=131072,
        shard_manifest=None,
        activation_bits=activation_bits,
        label_only=True,
        stop_file=stop_file,
        request_timeout=180,
        deadline=sources.get("development_deadline"),
        progress_file=sources.get("progress_file"),
        native_runtime=sources.get("native_runtime"),
    )
    env = {"LD_LIBRARY_PATH": sources["native_runtime"]["ld_library_path"]}
    if cache_gate:
        env.update(
            {
                "EAGLE_CAPTURE_DRAFT_GRAPH": "1",
                "EAGLE_CAPTURE_DRAFT_GRAPH_SCOPE": "cache",
                "EAGLE_CAPTURE_DRAFT_GRAPH_MAX_EXECUTIONS": "1024",
                "EAGLE_CAPTURE_DRAFT_GRAPH_MAX_BYTES": str(1024**3),
                "EAGLE_CAPTURE_DRAFT_CACHE": "1",
            }
        )
    spec = {
        "draft": str(draft or sources["candidate_d_gguf"]),
        "body": "D",
        "head": "D",
        "env": env,
    }
    if draft is not None and str(draft) == sources.get("q4_0_draft"):
        spec.update(
            body="Q4_0",
            head="Q4_0",
            activation_precision="native Q8_1 conversion for Q4_0 weight matrices",
        )
    if activation_bits in {1, 4, 8}:
        spec["required_markers"] = [
            "EAGLE3 W1A1 active groups: fusion,attention,ffn,head (9 tensors)",
            {
                8: "CUDA packed W1A8 INT8 dispatch",
                4: "CUDA packed W1A4 BITSERIAL dispatch",
                1: "CUDA packed W1A1 XOR/POPCOUNT dispatch",
            }[activation_bits],
        ]
    with native_cancellation() as guard:
        args.cancellation_guard = guard
        run_cell(args, "d_d", spec, rows)


def validate_readiness(record: dict, *, activation_bits: int, common_hashes: dict) -> dict:
    path = checked_record(record)
    report = json.loads(path.read_text())
    if (
        report.get("schema") not in {READINESS_SCHEMA, RECIPE_READINESS_SCHEMA}
        or report.get("training_eligible") is not True
        or report.get("objective") != "hard_ce"
        or report.get("scale_layout") != "row"
        or report.get("common_source_sha256") != common_hashes
        or report.get("unresolved_gates") != []
    ):
        raise ValueError("continuous full-body readiness is not bound to frozen inputs")
    precision = report.get("precisions", {}).get(str(activation_bits))
    modern = report["schema"] == RECIPE_READINESS_SCHEMA
    if activation_bits not in ({1, 4, 8} if modern else {1, 8}) or not isinstance(precision, dict):
        raise ValueError("readiness needs its own independent requested-precision result")
    runtime = report.get("native_runtime")
    if modern and (
        not isinstance(report.get("native_binary_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", report["native_binary_sha256"]) is None
        or not isinstance(runtime, dict)
        or not isinstance(runtime.get("immutable_manifest"), dict)
        or not isinstance(runtime.get("libraries"), list)
        or not runtime["libraries"]
    ):
        raise ValueError("recipe readiness needs a bound native binary and full runtime inventory")
    if runtime:
        checked_record(runtime["immutable_manifest"])
        for library in runtime["libraries"]:
            checked_record(library)
    from check_continuous_w1ax_readiness import validate_gate_report

    inventory = report.get("precisions", {})
    if modern and (not inventory or set(inventory) - {"1", "4", "8"}):
        raise ValueError("recipe readiness precision inventory differs")
    for bits in [int(key) for key in sorted(inventory)] if modern else (8, 1):
        gate_path = checked_record(report["precisions"][str(bits)])
        gate = json.loads(gate_path.read_text())
        if modern and gate.get("schema") != "w1ax_continuous_precision_gate_v2":
            raise ValueError("recipe readiness requires independently validated versioned gates")
        if modern and (
            gate.get("native_binary_sha256") != report["native_binary_sha256"]
            or gate.get("native_runtime") != runtime
        ):
            raise ValueError(
                "recipe precision gate and teacher readiness use different native runtimes"
            )
        validate_gate_report(gate, bits, common_hashes)
    return report


def provider_manifest(
    sources: dict,
    capture: Path,
    readiness: Path,
    output: Path,
    *,
    native_label_audit_receipt: dict | None = None,
    common_source_sha256: dict | None = None,
) -> dict:
    from w1ax_capture_provider import ANGELSLIM_REVISION

    m = json.loads(capture.read_text())
    snap = json.loads(Path(sources["model_snapshot_manifest"]).read_text())
    paths = {
        name: str(Path(sources[name]).resolve())
        for name in (
            "target_gguf",
            "candidate_d_gguf",
            "base_draft_gguf",
            "absolute_d2t",
            "model_snapshot_manifest",
        )
    }
    paths.update(
        capture_manifest=str(capture.resolve()),
        prompts=str((capture.parent / m["files"]["prompts"]["path"]).resolve()),
        **{f"{role}_model_dir": snap["models"][role]["directory"] for role in ("target", "draft")},
    )
    shared = {}
    if common_source_sha256 is not None:
        if not isinstance(common_source_sha256, dict) or set(common_source_sha256) != set(
            PROVIDER_SHARED_WEIGHTS
        ):
            raise ValueError("provider shared weight hash inventory differs")
        for name, digest in common_source_sha256.items():
            # Verify pinned bytes once per process/file identity, then recheck
            # inode/size/mtime/ctime on every shard. Keep the logical path so a
            # source replaced by a symlink cannot silently follow the cache.
            checked_record({"path": str(Path(sources[name]).absolute()), "sha256": digest})
            shared[name] = digest
    spec = {
        "schema": PROVIDER_SCHEMA,
        "split": m["split"],
        "training_eligible": m["split"] == "train",
        "prompt_count": m["prompt_count"],
        "capture_id": sha256(capture),
        "angelslim_revision": ANGELSLIM_REVISION,
        "paths": paths,
        "sha256": {
            k: shared[k] if k in shared else sha256(Path(p))
            for k, p in paths.items()
            if not k.endswith("_model_dir")
        },
        "teacher": None,
        "continuous_readiness": file_record(readiness),
    }
    if native_label_audit_receipt is not None:
        checked_record(native_label_audit_receipt)
        spec["native_label_audit_receipt"] = native_label_audit_receipt
    write_json(output, spec)
    return spec


def stage_progress(
    run_dir: Path,
    phase: str,
    *,
    captures_done: int = 0,
    captures_total: int = 0,
    prompts_done: int = 0,
    detail: str = "",
) -> None:
    from w1a1_eagle.continuous_resources import linux_host_memory

    host = linux_host_memory()
    write_json(
        Path(run_dir) / "status.json",
        {
            "schema": "continuous_joint_w1ax_v1",
            "status": "preparing",
            "optimization_started": False,
            "phase": phase,
            "heartbeat_unix": time.time(),
            "pid": os.getpid(),
            "models": {},
            "captures_done": captures_done,
            "captures_total": captures_total,
            "captured_unique_prompts": prompts_done,
            "detail": detail,
            "disk_free_bytes": shutil.disk_usage(run_dir).free,
            **host,
        },
    )


def _retained_path(value: str) -> Path:
    """Retained adoption never follows directory aliases or symlink ancestors."""
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError("retained import paths must be absolute without aliases")
    return path


def _historical_audit_source_proof() -> dict:
    """Prove the unchanged semantic auditor against authenticated commit7547.

    That controlled launch wrote readiness only after every full shard audit.
    Compare ASTs without line/format attributes, never normalize semantic code.
    The receipt API has a separate file helper so the original auditor's helper
    and its six local semantic dependencies remain exactly comparable.
    """
    source = native_label_audit_source()
    identity = _native_label_report_sha256(source)
    if identity in _HISTORICAL_SOURCE_PROOFS:
        return source
    names = (
        "scripts/w1ax_continuous_stages.py",
        "scripts/audit_recurrent_binary_capture.py",
        "scripts/audit_recurrent_response.py",
        "scripts/audit_recurrent_continuity.py",
        "scripts/prepare_recurrent_native_rows.py",
        "scripts/prepare_recurrent_native_features.py",
        "src/w1a1_eagle/recurrent_trace.py",
    )
    original = {}
    for name in names:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{HISTORICAL_PRODUCER_COMMIT}:{name}"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        original[name] = result.stdout
        if name != names[0] and hashlib.sha256(result.stdout).hexdigest() != source["files"][name]:
            raise ValueError("historical audit semantic dependency changed: " + name)

    def semantic_nodes(text):
        selected = {}
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef) and node.name in {"audit_native_labels", "_files"}:
                selected[node.name] = ast.dump(node, include_attributes=False)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {
                        "LABEL_SCHEMA",
                        "LABEL_POLICY",
                    }:
                        selected[target.id] = ast.dump(node, include_attributes=False)
            elif isinstance(node, ast.ImportFrom) and node.module in {
                "audit_recurrent_binary_capture",
                "audit_recurrent_response",
                "prepare_recurrent_native_features",
                "prepare_recurrent_native_rows",
                "w1a1_eagle.recurrent_trace",
            }:
                selected[node.module] = ast.dump(node, include_attributes=False)
        return selected

    if semantic_nodes(original[names[0]]) != semantic_nodes((ROOT / names[0]).read_bytes()):
        raise ValueError("historical audit semantic functions/imports/policy changed")
    _HISTORICAL_SOURCE_PROOFS.add(identity)
    return source


def _historical_retained_pass(import_record: dict) -> tuple[dict, dict]:
    """Authenticate the one retained supervisor08 pass; no model/payload reads.

    Trust roots are the exact archived operation and completion bytes. The
    original ordered readiness and both complete provider indexes must bind
    every current manifest and sidecar. A success cache retains only metadata;
    all its path/SHA identities are checked on access, including after mutation.
    """
    source = _historical_audit_source_proof()
    cache_key = (
        str(checked_record(import_record)),
        import_record["sha256"],
        _native_label_report_sha256(source),
    )
    if cache_key in _HISTORICAL_PASS_PROOFS:
        adoption, original, records = _HISTORICAL_PASS_PROOFS[cache_key]
        for record in records:
            _retained_path(record["path"])
            checked_record(record)
        return adoption, original
    records = [import_record]

    def read(record):
        path = _retained_path(record["path"])
        checked_record(record)
        records.append(record)
        return json.loads(path.read_text())

    adoption = read(import_record)
    if (
        not isinstance(adoption, dict)
        or set(adoption)
        != {
            "schema",
            "original_stages",
            "original_run_dir",
            "captures",
            "precision_gates",
            "audit_receipts_dir",
            "output_run_dir",
            "historical_audit_provenance",
        }
        or adoption["schema"] != RETAINED_IMPORT_SCHEMA
    ):
        raise ValueError("historical audit requires the retained import contract")
    provenance = adoption.get("historical_audit_provenance")
    if (
        not isinstance(provenance, dict)
        or set(provenance)
        != {"schema", "operation", "completion", "readiness", "provider_indexes", "audit_reports"}
        or provenance["schema"] != HISTORICAL_AUDIT_SCHEMA
        or provenance["operation"]["sha256"] != HISTORICAL_OPERATION_SHA256
        or provenance["completion"]["sha256"] != HISTORICAL_COMPLETION_SHA256
    ):
        raise ValueError("historical audit requires authenticated full-pass provenance")
    operation, completion = read(provenance["operation"]), read(provenance["completion"])
    original = read(adoption["original_stages"])
    if original.get("schema") != STAGES_SCHEMA:
        raise ValueError("historical audit original stage schema differs")
    old_run = _retained_path(adoption["original_run_dir"])
    checkout = old_run.parent.parent
    preflight = operation["preflight"]["decoded_query"]
    identity = preflight["identity"]
    launch = operation["launch"]["decoded_query"]
    observed = completion["decoded_query"]
    supervisor = operation["first_health"]["decoded_query"]["supervisor"]
    total = len(original["captures"])
    if (
        preflight.get("all_checks_pass") is not True
        or identity.get("git_head") != HISTORICAL_PRODUCER_COMMIT
        or identity.get("git_diff_name_only") != ["scripts/train_continuous_w1ax.py"]
        or identity.get("git_status_tracked") != [" M scripts/train_continuous_w1ax.py"]
        or identity.get("launcher_sha256")
        != "82f0185ab9ac38bc622749d2ca5e5c5a297a72494dcbf3d16d2b08df249cdade"
        or identity.get("stages_sha256") != adoption["original_stages"]["sha256"]
        or launch.get("returncode") != 0
        or launch.get("cwd") != str(checkout)
        or launch.get("supervisor_id") != "luna-supervisor-a8-a1-native-order-20261002-08"
        or observed["status"].get("phase") != "readiness_complete"
        or observed["status"].get("optimization_started") is not False
        or observed["status"].get("models") != {}
        or observed["status"].get("captures_done") != total
        or observed["status"].get("captures_total") != total
        or observed.get("completed_label_manifest_count") != total
        or any(
            observed["supervisor"].get(key) != supervisor.get(key)
            for key in ("pid", "pgid", "supervisor_pid", "started_at_utc")
        )
    ):
        raise ValueError("historical audit controlled source/launch/full-pass identity differs")
    command = launch["exact_command"][-1]
    stages_relative = _retained_path(adoption["original_stages"]["path"]).relative_to(checkout)
    if (
        f"--stages-manifest {stages_relative} --run-dir {old_run}" not in command
        or "--start --allow-cuda --resume --prepare-only" not in command
        or str(old_run / "status.json") not in observed["checker_command"]
        or launch["supervisor_state_path"] not in observed["checker_command"]
    ):
        raise ValueError("historical audit launch does not bind the original stages/run")
    captures = adoption["captures"]
    if len(captures) != total or any(entry["ordinal"] != i for i, entry in enumerate(captures)):
        raise ValueError("historical audit capture ordering differs")
    ready_record = provenance["readiness"]
    if _retained_path(ready_record["path"]) != old_run / "stages/readiness.json":
        raise ValueError("historical audit readiness path differs")
    ready = read(ready_record)
    sources = original["sources"]
    common = {
        k: sources["sha256"][k]
        for k in (
            "target_gguf",
            "candidate_d_gguf",
            "base_draft_gguf",
            "absolute_d2t",
            "model_snapshot_manifest",
        )
    }
    if (
        ready.get("schema") != READINESS_SCHEMA
        or ready.get("objective") != "hard_ce"
        or ready.get("scale_layout") != "row"
        or ready.get("unresolved_gates") != []
        or ready.get("scope") != "joint_body_head_exact_prefix_teacher_forced_training"
        or ready.get("common_source_sha256") != common
        or ready.get("native_binary_sha256") != sources["sha256"]["binary"]
        or ready.get("native_runtime") != sources["native_runtime"]
        or ready.get("precisions") != adoption["precision_gates"]
        or ready.get("teacher_capture_manifest_sha256")
        != [entry["label_manifest"]["sha256"] for entry in captures]
    ):
        raise ValueError("historical audit full ordered readiness/source binding differs")
    reports = provenance["audit_reports"]
    if not isinstance(reports, list) or len(reports) != total:
        raise ValueError("historical audit reports incomplete")
    manifests = []
    for ordinal, (entry, capture, audit_record) in enumerate(
        zip(captures, original["captures"], reports, strict=True)
    ):
        path = old_run / f"stages/capture-{ordinal:05d}/labels/manifest.json"
        if _retained_path(entry["label_manifest"]["path"]) != path:
            raise ValueError("historical audit manifest path/ordinal differs")
        m = read(entry["label_manifest"])
        manifests.append(m)
        if _retained_path(audit_record["path"]) != path.parent / "audit.json":
            raise ValueError("historical audit report ownership differs")
        report = read(audit_record)
        if (
            report.get("schema") != "recurrent_native_label_audit_v2"
            or report.get("capture_manifest_sha256") != entry["label_manifest"]["sha256"]
            or report.get("training_prompts_sha256") != capture["prompts_sha256"]
            or report.get("training_prompt_count") != capture["prompt_count"]
            or report.get("split") != capture["split"]
            or report.get("training_eligible") is not False
            or report.get("execution_device") != "cpu"
            or report.get("raw_target_logits_in_bundle") is not False
            or report.get("probability_recomputation") != "unavailable_label_only_storage"
            or not isinstance(report.get("counts"), dict)
            or not isinstance(report.get("feature_ledger"), dict)
            or report.get("response_requests") != len(m["requests"])
        ):
            raise ValueError("historical audit report manifest/prompt/split contract differs")
    indexes = provenance["provider_indexes"]
    if not isinstance(indexes, dict) or set(indexes) != {"train", "development"}:
        raise ValueError("historical audit requires both complete provider indexes")
    for split, index_record in indexes.items():
        if _retained_path(index_record["path"]) != old_run / f"stages/{split}-providers.json":
            raise ValueError("historical audit provider index path differs")
        index = read(index_record)
        selected = [(i, c) for i, c in enumerate(original["captures"]) if c["split"] == split]
        if (
            index.get("schema") != "w1ax_streaming_train_v2"
            or index.get("split") != split
            or index.get("continuous_readiness") != ready_record
            or not isinstance(index.get("shards"), list)
            or len(index["shards"]) != len(selected)
        ):
            raise ValueError("historical audit index/readiness/full coverage differs")
        for ordinal, (record, (global_ordinal, capture)) in enumerate(
            zip(index["shards"], selected, strict=True)
        ):
            path = old_run / f"stages/provider-{global_ordinal:05d}.json"
            if (
                record.get("ordinal") != ordinal
                or _retained_path(record["provider_manifest"]) != path
            ):
                raise ValueError("historical audit provider ordering differs")
            spec = read({"path": str(path), "sha256": record["provider_manifest_sha256"]})
            entry = captures[global_ordinal]["label_manifest"]
            prompt_path = (
                Path(entry["path"]).parent / manifests[global_ordinal]["files"]["prompts"]["path"]
            )
            if (
                spec.get("schema") != PROVIDER_SCHEMA
                or spec.get("split") != split
                or spec.get("prompt_count") != capture["prompt_count"]
                or spec.get("capture_id") != entry["sha256"]
                or spec.get("continuous_readiness") != ready_record
                or spec.get("teacher") is not None
                or spec["paths"].get("capture_manifest") != entry["path"]
                or spec["paths"].get("prompts") != str(prompt_path)
                or spec["sha256"].get("capture_manifest") != entry["sha256"]
                or spec["sha256"].get("prompts") != capture["prompts_sha256"]
                or any(spec["sha256"].get(k) != v for k, v in common.items())
            ):
                raise ValueError("historical audit provider capture/readiness/source join differs")
    _HISTORICAL_PASS_PROOFS[cache_key] = adoption, original, records
    return adoption, original


def _validate_historical_receipt_origin(origin: dict, binding: dict, report: dict) -> None:
    if (
        not isinstance(origin, dict)
        or set(origin)
        != {"kind", "producer_commit", "retained_import", "provenance_sha256", "ordinal"}
        or origin["kind"] != "historical_full_semantic_pass"
        or origin["producer_commit"] != HISTORICAL_PRODUCER_COMMIT
        or type(origin["ordinal"]) is not int
    ):
        raise ValueError("historical audit receipt origin differs")
    adoption, _ = _historical_retained_pass(origin["retained_import"])
    ordinal = origin["ordinal"]
    provenance = adoption["historical_audit_provenance"]
    if (
        not 0 <= ordinal < len(adoption["captures"])
        or origin["provenance_sha256"] != _native_label_report_sha256(provenance)
        or binding["capture_manifest"] != adoption["captures"][ordinal]["label_manifest"]
        or report != json.loads(checked_record(provenance["audit_reports"][ordinal]).read_text())
    ):
        raise ValueError("historical audit receipt provenance/report/shard binding differs")


def _adopt_retained_historical_audit(import_record: dict, ordinal: int, receipt_path: Path) -> None:
    adoption, original = _historical_retained_pass(import_record)
    entry, capture = adoption["captures"][ordinal], original["captures"][ordinal]
    report_record = adoption["historical_audit_provenance"]["audit_reports"][ordinal]
    binding = _native_label_receipt_binding(
        Path(entry["label_manifest"]["path"]),
        expected_manifest_sha256=entry["label_manifest"]["sha256"],
        expected_prompt_sha256=capture["prompts_sha256"],
        expected_prompt_count=capture["prompt_count"],
    )
    origin = {
        "kind": "historical_full_semantic_pass",
        "producer_commit": HISTORICAL_PRODUCER_COMMIT,
        "retained_import": import_record,
        "provenance_sha256": _native_label_report_sha256(adoption["historical_audit_provenance"]),
        "ordinal": ordinal,
    }
    report = json.loads(checked_record(report_record).read_text())
    _validate_historical_receipt_origin(origin, binding, report)
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        if existing.get("audit_origin") != origin:
            raise ValueError("historical audit receipt existing origin differs")
    else:
        if (
            not receipt_path.is_absolute()
            or receipt_path.is_symlink()
            or receipt_path.resolve().is_relative_to(Path(entry["label_manifest"]["path"]).parent)
        ):
            raise ValueError("historical audit receipt must be external and absolute")
        _publish_native_label_receipt(receipt_path, binding, report, audit_origin=origin)


def _validate_retained_import(config: dict, run_dir: Path) -> dict:
    """Authenticate explicit old artifacts without granting training readiness.

    Optional historical provenance authenticates only the original full shard
    audit pass. It never replaces final preparation, smoke or training gates.
    """
    record = config["retained_native_capture_import"]
    path = _retained_path(record["path"])
    adoption = json.loads(checked_record(record).read_text())
    if (
        not isinstance(adoption, dict)
        or set(adoption)
        != {
            "schema",
            "original_stages",
            "original_run_dir",
            "captures",
            "precision_gates",
            "audit_receipts_dir",
            "output_run_dir",
            "historical_audit_provenance",
        }
        or adoption["schema"] != RETAINED_IMPORT_SCHEMA
    ):
        raise ValueError("unsupported retained import/provenance contract")
    original_path = _retained_path(adoption["original_stages"]["path"])
    original = json.loads(checked_record(adoption["original_stages"]).read_text())
    expected = {
        **original,
        "retained_native_capture_import": record,
        "native_label_audit_receipts_dir": adoption["audit_receipts_dir"],
    }
    if (
        original.get("schema") != STAGES_SCHEMA
        or "retained_native_capture_import" in original
        or config != expected
    ):
        raise ValueError("retained import changes original stages/source/prompt configuration")
    old_run = _retained_path(adoption["original_run_dir"])
    new_run = _retained_path(adoption["output_run_dir"])
    receipts = _retained_path(adoption["audit_receipts_dir"])
    if (
        new_run != Path(run_dir).absolute()
        or new_run.is_relative_to(old_run)
        or old_run.is_relative_to(new_run)
        or receipts.is_relative_to(old_run)
        or old_run.is_relative_to(receipts)
        or path.is_relative_to(old_run)
        or original_path == path
    ):
        raise ValueError("retained import requires separate new output and external receipts")
    if config["native_label_audit_receipts_dir"] != str(receipts):
        raise ValueError("retained import receipt directory changed")
    _retained_path(str(new_run / "stages"))
    for ordinal in range(len(original["captures"])):
        _retained_path(str(receipts / f"capture-{ordinal:05d}.json"))
    sources = original["sources"]
    verify_sources(sources)
    for name in ("legacy_provider", "corpus_manifest"):
        if name in sources:
            checked_record(sources[name])
    if "corpus_manifest" in original:
        checked_record(original["corpus_manifest"])
        if sources.get("corpus_manifest") != original["corpus_manifest"]:
            raise ValueError("retained import corpus ancestry differs")
    captures = adoption["captures"]
    if not isinstance(captures, list) or len(captures) != len(original["captures"]):
        raise ValueError("retained import requires every original capture in order")
    for ordinal, (entry, capture) in enumerate(zip(captures, original["captures"], strict=True)):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"ordinal", "label_manifest"}
            or type(entry["ordinal"]) is not int
            or entry["ordinal"] != ordinal
        ):
            raise ValueError("retained import capture ordinals differ")
        manifest = _retained_path(entry["label_manifest"]["path"])
        if manifest != old_run / f"stages/capture-{ordinal:05d}/labels/manifest.json":
            raise ValueError("retained import label manifest is not the original capture")
        m = json.loads(checked_record(entry["label_manifest"]).read_text())
        if (
            m.get("split") != capture["split"]
            or m.get("prompt_count") != capture["prompt_count"]
            or m.get("prompts_sha256") != capture["prompts_sha256"]
            or m.get("activation_bits") != 16
            or m.get("target_sha256") != sources["sha256"]["target_gguf"]
            or m.get("draft_sha256") != sources["sha256"]["candidate_d_gguf"]
            or m.get("binary_sha256") != sources["sha256"]["binary"]
            or m.get("files", {}).get("absolute_d2t", {}).get("sha256")
            != sources["sha256"]["absolute_d2t"]
        ):
            raise ValueError("retained import teacher/source/prompt/split ancestry differs")
        require_unsealed_prompts(
            Path(capture["prompts"]), sources=sources, expected_sha256=capture["prompts_sha256"]
        )
        checked_record({"path": capture["prompts"], "sha256": capture["prompts_sha256"]})
        # This checks every file and cache/sampler input; it never trusts audit.json.
        _native_label_receipt_binding(
            manifest,
            expected_manifest_sha256=entry["label_manifest"]["sha256"],
            expected_prompt_sha256=capture["prompts_sha256"],
            expected_prompt_count=capture["prompt_count"],
        )
    from check_continuous_w1ax_readiness import validate_gate_report

    gates = adoption["precision_gates"]
    if not isinstance(gates, dict) or set(gates) != {"8", "1"}:
        raise ValueError("retained import requires both original precision gates")
    common = {
        k: sources["sha256"][k]
        for k in (
            "target_gguf",
            "candidate_d_gguf",
            "base_draft_gguf",
            "absolute_d2t",
            "model_snapshot_manifest",
        )
    }
    for bits in (8, 1):
        gate_path = _retained_path(gates[str(bits)]["path"])
        if gate_path != old_run / f"stages/gate-a{bits}/gate.json":
            raise ValueError("retained import gate is not the original stage gate")
        gate = json.loads(checked_record(gates[str(bits)]).read_text())
        if (
            gate.get("schema") != "w1ax_continuous_precision_gate_v1"
            or gate.get("native_binary_sha256") != sources["sha256"]["binary"]
            or gate.get("native_runtime") != sources["native_runtime"]
        ):
            raise ValueError("retained import gate runtime/recipe differs")
        validate_gate_report(gate, bits, common)
        labels = json.loads(checked_record(gate["evidence"]["native_capture_manifest"]).read_text())
        if (
            labels.get("prompts_sha256") != original["gate_prompts_sha256"]
            or labels.get("split") != "train"
        ):
            raise ValueError("retained import gate prompt/split differs")
    if adoption["historical_audit_provenance"] is not None:
        _historical_retained_pass(record)
    return adoption


def prepare_retained_config(import_manifest: Path, output: Path) -> dict:
    """CPU-only metadata adoption; labels remain preparation-only until launch.

    The caller supplies explicit path/SHA records for the original stage config,
    every capture and both independently validated native gates. No old final
    ready receipt is required and no payload or old run is rewritten.
    """
    import_manifest = _retained_path(str(import_manifest.absolute()))
    adoption = json.loads(import_manifest.read_text())
    original = json.loads(checked_record(adoption["original_stages"]).read_text())
    result = {
        **original,
        "retained_native_capture_import": file_record(import_manifest),
        "native_label_audit_receipts_dir": adoption["audit_receipts_dir"],
    }
    _validate_retained_import(result, Path(adoption["output_run_dir"]))
    output = _retained_path(str(output.absolute()))
    if output.exists() or output.is_relative_to(Path(adoption["original_run_dir"])):
        raise ValueError("retained config requires a new output outside the old run")
    write_json(output, result)
    return result


def _run_stages(config: dict, run_dir: Path) -> Path:
    """Capture/audit/gate explicit immutable train shards before optimization."""
    if config.get("schema") != STAGES_SCHEMA:
        raise ValueError("unsupported continuous stage configuration")
    check_stop(run_dir)
    retained = (
        _validate_retained_import(config, run_dir)
        if "retained_native_capture_import" in config
        else None
    )
    sources = config["sources"]
    sources = {
        **sources,
        "stop_file": str((Path(run_dir) / "STOP").resolve()),
        "progress_file": str((Path(run_dir) / "status.json").resolve()),
    }
    from w1ax_capture_provider import CANDIDATE_D_SHA256, TARGET_GGUF_SHA256

    verify_sources(sources)
    audit_q4_file(Path(sources["q4_0_draft"]))
    if (
        sources["sha256"].get("target_gguf") != TARGET_GGUF_SHA256
        or sources["sha256"].get("candidate_d_gguf") != CANDIDATE_D_SHA256
    ):
        raise ValueError("frozen target/candidate-D changed")
    captures = config["captures"]
    if (
        not captures
        or {c.get("split") for c in captures} != {"train", "development"}
        or any(c["prompt_count"] < 1 or c["prompt_count"] > 32 for c in captures)
    ):
        raise ValueError("explicit train/development captures required")
    all_prompts = {"train": {}, "development": {}}
    for capture in captures:
        prompts = Path(capture["prompts"])
        require_unsealed_prompts(
            prompts, sources=sources, expected_sha256=capture["prompts_sha256"]
        )
        if sha256(prompts) != capture["prompts_sha256"]:
            raise ValueError("frozen stage prompt bytes differ")
        rows = read_jsonl(prompts)
        if len(rows) != capture["prompt_count"]:
            raise ValueError("frozen stage prompt count differs")
        for row in rows:
            if row["id"] in all_prompts["train"] or row["id"] in all_prompts["development"]:
                raise ValueError("stage train/development prompt IDs overlap")
            all_prompts[capture["split"]][row["id"]] = row
    development_path = Path(config["development_prompts"])
    if sha256(development_path) != config["development_prompts_sha256"]:
        raise ValueError("frozen development selection changed")
    development_rows = read_jsonl(development_path)
    if len(development_rows) != 24 or any(
        all_prompts["development"].get(r["id"]) != r for r in development_rows
    ):
        raise ValueError("native development subset must be drawn from independent dev captures")
    gate_path = Path(config["gate_prompts"])
    if sha256(gate_path) != config["gate_prompts_sha256"]:
        raise ValueError("frozen gate selection changed")
    if any(all_prompts["train"].get(r["id"]) != r for r in read_jsonl(gate_path)):
        raise ValueError("precision gates must use declared training prompts")
    estimated = sum(c["storage_forecast"]["upper_bound_bytes"] for c in captures)
    if estimated > config["max_capture_storage_bytes"]:
        raise ValueError("declared conservative capture budget exceeded; lower shard/token caps")
    stage_dir = Path(run_dir) / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    # Explicit opt-in only. Receipts are external immutable per-shard evidence;
    # provider manifests bind their path/SHA after publication. Legacy configs
    # continue to run the full semantic audit on every existing call site.
    receipt_dir = config.get("native_label_audit_receipts_dir")
    if receipt_dir is not None:
        receipt_dir = Path(receipt_dir)
        if not receipt_dir.is_absolute() or receipt_dir.is_symlink():
            raise ValueError("native label audit receipts directory must be absolute")
    host_admission("checkpoint-zero CPU model initialization", 12 * 1024**3)
    remaining = sum(
        c["storage_forecast"]["upper_bound_bytes"]
        for ordinal, c in enumerate(captures)
        if retained is None
        and not (stage_dir / f"capture-{ordinal:05d}/labels/manifest.json").is_file()
    )
    if shutil.disk_usage(stage_dir).free < remaining + config.get(
        "min_free_disk_bytes", 10 * 1024**3
    ):
        raise RuntimeError("insufficient disk for conservative capture forecast")
    from check_continuous_w1ax_readiness import run_gate

    common = {
        k: sources["sha256"][k]
        for k in (
            "target_gguf",
            "candidate_d_gguf",
            "base_draft_gguf",
            "absolute_d2t",
            "model_snapshot_manifest",
        )
    }
    results = {}
    # Decisive bounded gates precede the large corpus capture.
    for bits in (8, 1):
        stage_progress(
            run_dir,
            f"gate_A{bits}",
            captures_total=len(captures),
            detail="bounded native/CUDA math/cache/export/backward gate; no optimization",
        )
        check_stop(run_dir)
        report = (
            checked_record(retained["precision_gates"][str(bits)])
            if retained is not None
            else run_gate(
                sources,
                Path(config["gate_prompts"]),
                stage_dir / f"gate-a{bits}",
                bits,
                expected_prompt_sha256=config["gate_prompts_sha256"],
            )
        )
        results[str(bits)] = file_record(report)
    teacher_manifests = []
    receipt_records = {}
    prompts_done = 0
    for ordinal, capture in enumerate(captures):
        stage_progress(
            run_dir,
            "teacher_capture_audit",
            captures_done=ordinal,
            captures_total=len(captures),
            prompts_done=prompts_done,
            detail=f"{capture['split']} shard {ordinal}: native labels/features only",
        )
        check_stop(run_dir)
        prompts = Path(capture["prompts"])
        if (
            sha256(prompts) != capture["prompts_sha256"]
            or len(read_jsonl(prompts)) != capture["prompt_count"]
        ):
            raise ValueError("stage frozen prompt bytes/count changed")
        folder = stage_dir / f"capture-{ordinal:05d}"
        manifest = (
            checked_record(retained["captures"][ordinal]["label_manifest"])
            if retained is not None
            else folder / "labels/manifest.json"
        )
        receipt_path = (
            receipt_dir / f"capture-{ordinal:05d}.json" if receipt_dir is not None else None
        )
        if retained is None and not manifest.exists():
            native_cell = folder / "native/d_d/manifest.json"
            if (folder / "native").exists():
                if (
                    not native_cell.is_file()
                    or json.loads(native_cell.read_text()).get("complete") is not True
                ):
                    raise RuntimeError(
                        "incomplete native capture preserved; use CPU recover-partial "
                        "then explicit manual --resume"
                    )
            else:
                native_capture(
                    sources, prompts, folder / "native", stop_file=Path(run_dir) / "STOP"
                )
            build_native_labels(
                folder / "native",
                prompts,
                Path(sources["absolute_d2t"]),
                folder / "labels",
                split=capture["split"],
                **({"audit_receipt_path": receipt_path} if receipt_path is not None else {}),
            )
        host_admission("bounded native label/feature CPU audit", 6 * 1024**3)
        if receipt_path is None:
            audit_native_labels(
                manifest,
                expected_prompt_sha256=capture["prompts_sha256"],
                expected_prompt_count=capture["prompt_count"],
            )
        else:
            if retained is not None and retained["historical_audit_provenance"] is not None:
                _adopt_retained_historical_audit(
                    config["retained_native_capture_import"],
                    ordinal,
                    receipt_path,
                )
            audit_native_labels_with_receipt(
                manifest,
                expected_prompt_sha256=capture["prompts_sha256"],
                expected_prompt_count=capture["prompt_count"],
                expected_manifest_sha256=(
                    retained["captures"][ordinal]["label_manifest"]["sha256"]
                    if retained is not None
                    else _observed_record(manifest)["sha256"]
                ),
                receipt_path=receipt_path,
            )
            receipt_records[manifest] = file_record(receipt_path)
        teacher_manifests.append((manifest, capture))
        prompts_done += capture["prompt_count"]
        stage_progress(
            run_dir,
            "teacher_shard_complete",
            captures_done=ordinal + 1,
            captures_total=len(captures),
            prompts_done=prompts_done,
        )
    readiness = stage_dir / "readiness.json"
    write_json(
        readiness,
        {
            "schema": READINESS_SCHEMA,
            "training_eligible": True,
            "objective": "hard_ce",
            "scale_layout": "row",
            "common_source_sha256": common,
            "precisions": results,
            "unresolved_gates": [],
            "native_binary_sha256": sources["sha256"]["binary"],
            "native_runtime": sources["native_runtime"],
            "scope": "joint_body_head_exact_prefix_teacher_forced_training",
            "teacher_capture_manifest_sha256": [sha256(m) for m, _ in teacher_manifests],
        },
    )
    validate_readiness(file_record(readiness), activation_bits=8, common_hashes=common)
    train_records, dev_records = [], []
    for ordinal, (manifest, capture) in enumerate(teacher_manifests):
        path = stage_dir / f"provider-{ordinal:05d}.json"
        provider_manifest(
            sources,
            manifest,
            readiness,
            path,
            common_source_sha256={
                name: sources["sha256"][name] for name in PROVIDER_SHARED_WEIGHTS
            },
            **(
                {"native_label_audit_receipt": receipt_records[manifest]}
                if manifest in receipt_records
                else {}
            ),
        )
        record = {
            "ordinal": ordinal,
            "provider_manifest": str(path.resolve()),
            "provider_manifest_sha256": sha256(path),
        }
        (train_records if capture["split"] == "train" else dev_records).append(record)
    for split, records in (("train", train_records), ("development", dev_records)):
        for i, record in enumerate(records):
            record["ordinal"] = i
        write_json(
            stage_dir / f"{split}-providers.json",
            {
                "schema": "w1ax_streaming_train_v2",
                "split": split,
                "shards": records,
                "training_eligible": split == "train",
                "continuous_readiness": file_record(readiness),
            },
        )
    development_ids = {p["id"] for p in development_rows}
    subset_records = []
    for record in dev_records:
        child_spec = json.loads(Path(record["provider_manifest"]).read_text())
        ids = {p["id"] for p in read_jsonl(Path(child_spec["paths"]["prompts"]))}
        if ids.intersection(development_ids):
            subset_records.append({**record, "ordinal": len(subset_records)})
    write_json(
        stage_dir / "development-subset-providers.json",
        {
            "schema": "w1ax_streaming_train_v2",
            "split": "development",
            "shards": subset_records,
            "training_eligible": False,
            "continuous_readiness": file_record(readiness),
        },
    )
    write_json(
        stage_dir / "development.json",
        {
            "schema": "w1ax_continuous_development_v1",
            "sources": sources,
            "providers_manifest": str((stage_dir / "development-subset-providers.json").resolve()),
            "full_pool_manifest": file_record(stage_dir / "development-providers.json"),
            "full_pool_prompt_count": len(all_prompts["development"]),
            "native_prompts": config["development_prompts"],
            "native_prompts_sha256": config["development_prompts_sha256"],
            "max_loss_rounds": 64,
            "native_output_tokens": 128,
            "max_wall_seconds": 1200,
            "keep_native_evaluations": 3,
            "split": "development",
            "native_q4_comparison": "accepted_drafts_per_round_only",
        },
    )
    stage_progress(
        run_dir,
        "readiness_complete",
        captures_done=len(captures),
        captures_total=len(captures),
        prompts_done=prompts_done,
    )
    return (stage_dir / "train-providers.json").resolve()


def run_stages(config: dict, run_dir: Path) -> Path:
    with native_cancellation():
        return _run_stages(config, run_dir)


def _forecast(tokens: list[int], *, output_tokens: int = 128) -> dict:
    # At most five drafts plus one verifier/seed input per emitted token.
    # Preserve native features, hardlink their immutable raw ledger, and write the accepted
    # feature array. Full-vocabulary target logits are never captured.
    feature_upper = sum((2 * n + 7 * output_tokens) * 7680 * 4 for n in tokens)
    native_head_states = len(tokens) * 5 * output_tokens * 2560 * 4
    metadata_upper = sum((n + 6 * output_tokens) * (n + output_tokens) * 48 for n in tokens)
    accepted_min = sum((n + output_tokens) * 7680 * 4 for n in tokens)
    return {
        "unique_input_tokens": sum(tokens),
        "output_cap_per_prompt": output_tokens,
        "native_target_rows_per_emitted_token_bound": 6,
        "raw_and_accepted_feature_bytes_upper_bound": feature_upper,
        "prefix_metadata_bytes_conservative_allowance": metadata_upper,
        "native_head_state_bytes_upper_bound": native_head_states,
        "upper_bound_bytes": feature_upper + metadata_upper + native_head_states,
        "accepted_feature_bytes_at_full_output_cap": accepted_min,
        "uncertainty": "instrumentation/header/log overhead varies; per-shard disk guards apply",
    }


def prepare_config(
    corpus_path: Path,
    legacy_provider: Path,
    binary: Path,
    q4_draft: Path,
    output: Path,
    *,
    shard_prompts: int = 32,
    max_storage_bytes: int = 1024**4,
) -> dict:
    """CPU-only local configuration, microshards and fixed development subset.

    Run this on the machine that holds the retained native/model artifacts.
    Reads only train/dev files from the corpus inventory. Sealed files and
    reserve prompt contents are never opened.
    """
    from w1ax_capture_provider import CANDIDATE_D_SHA256, TARGET_GGUF_SHA256

    if not 1 <= shard_prompts <= 32 or output.exists():
        raise ValueError("prepare-config needs a new output and 1..32-prompt native shards")
    corpus_path, legacy_provider = corpus_path.resolve(), legacy_provider.resolve()
    corpus = json.loads(corpus_path.read_text())
    if corpus.get("schema") != "continuous_w1ax_prompt_manifest_v1":
        raise ValueError("unsupported CPU-prepared corpus inventory")
    old = json.loads(legacy_provider.read_text())
    source_paths = {
        k: str(Path(old["paths"][k]).resolve())
        for k in (
            "target_gguf",
            "candidate_d_gguf",
            "base_draft_gguf",
            "absolute_d2t",
            "model_snapshot_manifest",
        )
    }
    source_paths.update(binary=str(binary.resolve()), q4_0_draft=str(q4_draft.resolve()))
    hashes = {k: sha256(Path(v)) for k, v in source_paths.items()}
    if (
        hashes["target_gguf"] != TARGET_GGUF_SHA256
        or hashes["candidate_d_gguf"] != CANDIDATE_D_SHA256
        or any(hashes[k] != old["sha256"][k] for k in old["sha256"] if k in source_paths)
    ):
        raise ValueError("retained frozen artifact hashes differ")
    sources = {
        **source_paths,
        "sha256": hashes,
        "legacy_provider": file_record(legacy_provider),
        "q4_0_audit": audit_q4_file(q4_draft),
        "native_runtime": native_runtime_inventory(binary),
        "corpus_manifest": file_record(corpus_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    inputs = output.parent / (output.stem + "-inputs")
    inputs.mkdir(exist_ok=False)
    captures = []
    all_ids = set()
    gates = {}
    gate_criteria = {}
    devs = {d: [] for d in ("prose", "reasoning", "code")}
    split_counts = {}
    for public_split, capture_split in (("train", "train"), ("dev", "development")):
        split_counts[public_split] = 0
        ordinal = 0
        for shard in corpus["files"][public_split]["shards"]:
            prompt_path = corpus_path.parent / shard["prompts"]
            index_path = corpus_path.parent / shard["index"]
            if (
                sha256(prompt_path) != shard["prompts_sha256"]
                or sha256(index_path) != shard["index_sha256"]
            ):
                raise ValueError("frozen CPU prompt/index bytes differ")
            prompts, index = read_jsonl(prompt_path), read_jsonl(index_path)
            if len(prompts) != shard["prompts_count"] or len(index) != len(prompts):
                raise ValueError("frozen CPU shard prompt/index counts differ")
            for prompt, record in zip(prompts, index, strict=True):
                if (
                    prompt["id"] != record["id"]
                    or prompt["id"] in all_ids
                    or not 1 <= record["input_tokens"] <= 1792
                    or "final" in prompt["id"].lower()
                ):
                    raise ValueError("train/dev overlap, token cap or ownership differs")
                all_ids.add(prompt["id"])
                if capture_split == "train" and 128 <= record["input_tokens"] <= 512:
                    category = record.get("category")
                    score = int(
                        category
                        in {"brainstorming", "creative_writing", "summarization", "open_qa"}
                    )
                    domain = prompt["domain"]
                    if domain not in gates or score > gate_criteria[domain]["selection_score"]:
                        gates[domain] = prompt
                        gate_criteria[domain] = {
                            "prompt_id": prompt["id"],
                            "category": category,
                            "input_tokens": record["input_tokens"],
                            "source_id": record.get("source_id"),
                            "selection_score": score,
                        }
                elif (
                    capture_split == "development"
                    and record["input_tokens"] <= 512
                    and len(devs[prompt["domain"]]) < 8
                ):
                    devs[prompt["domain"]].append(prompt)
            for begin in range(0, len(prompts), shard_prompts):
                chunk, chunk_index = (
                    prompts[begin : begin + shard_prompts],
                    index[begin : begin + shard_prompts],
                )
                path = inputs / f"{capture_split}-{ordinal:05d}.jsonl"
                path.write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in chunk))
                tokens = [r["input_tokens"] for r in chunk_index]
                captures.append(
                    {
                        "split": capture_split,
                        "prompts": str(path.resolve()),
                        "prompts_sha256": sha256(path),
                        "prompt_count": len(chunk),
                        "max_prompt_tokens": max(tokens),
                        "storage_forecast": _forecast(tokens),
                        "source_corpus_manifest_sha256": sha256(corpus_path),
                        "source_prompts_sha256": shard["prompts_sha256"],
                        "source_index_sha256": shard["index_sha256"],
                        "source_positions": list(range(begin, begin + len(chunk))),
                    }
                )
                ordinal += 1
                split_counts[public_split] += len(chunk)
    if set(gates) != {"prose", "reasoning", "code"} or any(len(v) != 8 for v in devs.values()):
        raise ValueError("training gate/development selection lacks three-domain coverage")
    gate_path, dev_path = inputs / "gate-prompts.jsonl", inputs / "fixed-development.jsonl"
    gate_path.write_text(
        "".join(json.dumps(gates[d], sort_keys=True) + "\n" for d in sorted(gates))
    )
    # Domain-interleaved fixed development set; independent of training selection.
    selected_devs = [devs[d][i] for i in range(8) for d in sorted(devs)]
    dev_path.write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in selected_devs))
    result = {
        "schema": STAGES_SCHEMA,
        "sources": sources,
        "captures": captures,
        "corpus_manifest": file_record(corpus_path),
        "split_counts": split_counts,
        "gate_prompts": str(gate_path.resolve()),
        "gate_prompts_sha256": sha256(gate_path),
        "gate_selection_criteria": {
            "input_tokens": [128, 512],
            "domains": gate_criteria,
            "prose_preference": ["brainstorming", "creative_writing", "summarization", "open_qa"],
            "output_length": "insufficient roots fail; output length is not guaranteed",
        },
        "development_prompts": str(dev_path.resolve()),
        "development_prompts_sha256": sha256(dev_path),
        "development_selected_prompts": 24,
        "development_full_pool_prompts": split_counts["dev"],
        "development_subset_input_token_cap": 512,
        "max_capture_storage_bytes": max_storage_bytes,
        "min_free_disk_bytes": 10 * 1024**3,
        "storage_forecast": {
            "upper_bound_bytes": sum(c["storage_forecast"]["upper_bound_bytes"] for c in captures),
            "full_vocabulary_target_logits_bytes": 0,
        },
        "manual_boundary": (
            "prepare-config is CPU only; launcher --start --allow-cuda executes stages"
        ),
        "presentation": "frozen corpus order; shared identical inputs for A8 and A1",
        "expansion": (
            "create a new pinned configuration from expanded train/dev inventory; "
            "exact resume refuses changed source hashes"
        ),
    }
    write_json(output, result)
    return result


def validate_refresh_declaration(sources: dict, prompts: Path, prompts_sha256: str) -> dict:
    """Require one declared train microshard before any requested-payload read."""
    if not sources.get("stages_config"):
        raise ValueError(
            "refresh requires the hashed stage configuration and declared train microshard"
        )
    config_path = checked_record(sources["stages_config"])
    config = json.loads(config_path.read_text())
    if config.get("schema") != STAGES_SCHEMA:
        raise ValueError("refresh stage configuration schema differs")
    if sources.get("sha256") != config["sources"].get("sha256"):
        raise ValueError("refresh sources differ from the typed immutable stage configuration")
    matching = [
        record
        for record in config["captures"]
        if record["split"] == "train"
        and Path(record["prompts"]).resolve() == prompts.resolve()
        and record["prompts_sha256"] == prompts_sha256
    ]
    if len(matching) != 1:
        raise ValueError(
            "refresh input is not one exact declared train microshard; custom inputs forbidden"
        )
    require_unsealed_prompts(
        prompts,
        sources={**sources, "corpus_manifest": config["corpus_manifest"]},
        expected_sha256=prompts_sha256,
    )
    return matching[0]


def refresh_checkpoint(
    sources: dict,
    checkpoint_dir: Path,
    prompts: Path,
    output: Path,
    bits: int,
    *,
    prompts_sha256: str,
    split: str = "train",
) -> Path:
    """USER-started exact-prefix native refresh for a selected current checkpoint.

    Stop/offload training before calling. Newly proposed prefixes always receive
    new native sampler labels/features; previous bundles remain untouched. This
    writes an audited, ineligible bundle, requiring a new declared experiment
    source manifest before adoption; exact resume cannot silently change data.
    """
    require_unsealed_prompts(prompts, sources=sources, expected_sha256=prompts_sha256)
    validate_refresh_declaration(sources, prompts, prompts_sha256)
    from export_recurrent_binary import export_model

    verify_sources(sources)
    if bits not in {8, 4, 1} or sha256(prompts) != prompts_sha256:
        raise ValueError("refresh arithmetic/prompt identity differs")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint, manifest = checkpoint_dir / "joint.npz", checkpoint_dir / "joint.json"
    m = json.loads(manifest.read_text())
    if (
        m["activation_bits"] != bits
        or m["base_gguf_sha256"] != sources["sha256"]["base_draft_gguf"]
    ):
        raise ValueError("refresh must export the declared A8/A1 checkpoint")
    exported = output / "student.gguf"
    write_json(
        output / "export-audit.json",
        export_model(Path(sources["base_draft_gguf"]), checkpoint, manifest, exported),
    )
    native_capture(sources, prompts, output / "native", activation_bits=bits, draft=exported)
    build_native_labels(
        output / "native", prompts, Path(sources["absolute_d2t"]), output / "labels", split=split
    )
    write_json(
        output / "refresh.json",
        {
            "schema": "w1ax_exact_prefix_refresh_v3",
            "activation_bits": bits,
            "checkpoint": file_record(checkpoint),
            "checkpoint_manifest": file_record(manifest),
            "export": file_record(exported),
            "capture_manifest": file_record(output / "labels/manifest.json"),
            "prompts": file_record(prompts),
            "native_binary": file_record(Path(sources["binary"])),
            "common_source_sha256": sources["sha256"],
            "native_runtime": sources["native_runtime"],
            "stages_config": sources.get("stages_config"),
            "changed_prefix_labels_reused": False,
            "training_eligible": False,
            "adoption": (
                "CPU make-refresh-provider requires new explicit readiness "
                "and captured-actor binding; "
                "exact resume cannot silently append or replace teacher data"
            ),
        },
    )
    return output / "labels/manifest.json"


def make_refresh_provider(
    stages_config: Path, refresh_receipt: Path, readiness: Path, output: Path
) -> dict:
    """CPU-only explicit new provider; unchanged exact-resume inputs are never extended."""
    from w1ax_capture_provider import NativeCaptureProvider, validate_captured_drafter

    from w1a1_eagle.recurrent_qat import JointQATConfig, W1AxContract

    if output.exists():
        raise ValueError("refresh provider output must be new")
    config = json.loads(stages_config.read_text())
    if config.get("schema") != STAGES_SCHEMA:
        raise ValueError("refresh provider needs typed frozen stage sources")
    sources = config["sources"]
    verify_sources(sources)
    receipt = json.loads(refresh_receipt.read_text())
    capture = checked_record(receipt["capture_manifest"])
    manifest = json.loads(capture.read_text())
    common = {
        k: sources["sha256"][k]
        for k in (
            "target_gguf",
            "candidate_d_gguf",
            "base_draft_gguf",
            "absolute_d2t",
            "model_snapshot_manifest",
        )
    }
    permission = validate_readiness(
        file_record(readiness), activation_bits=manifest["activation_bits"], common_hashes=common
    )
    if receipt.get("stages_config") != file_record(stages_config):
        raise ValueError("refresh receipt came from different typed stage sources")
    if sha256(capture) not in permission["teacher_capture_manifest_sha256"]:
        raise ValueError("new readiness does not explicitly include the refreshed capture")
    binding = {name: receipt[name] for name in ("export", "checkpoint", "checkpoint_manifest")}
    binding.update(
        export_audit=file_record(refresh_receipt.parent / "export-audit.json"),
        refresh_receipt=file_record(refresh_receipt),
    )
    validate_captured_drafter(
        binding,
        capture_manifest_sha256=sha256(capture),
        captured_draft_sha256=manifest["draft_sha256"],
        prompts_sha256=manifest["prompts_sha256"],
        common_hashes=common,
        activation_bits=manifest["activation_bits"],
        native_binary_sha256=manifest["binary_sha256"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".refresh-provider-", dir=output.parent) as scratch:
        staged = Path(scratch) / "provider.json"
        spec = provider_manifest(sources, capture, readiness, staged)
        spec["captured_drafter"] = binding
        write_json(staged, spec)
        for bits in (
            [int(key) for key in sorted(permission["precisions"])]
            if permission["schema"] == RECIPE_READINESS_SCHEMA
            else (8, 1)
        ):
            if permission["schema"] == RECIPE_READINESS_SCHEMA:
                from check_continuous_w1ax_readiness import checkpoint_joint_config

                proof = json.loads(checked_record(permission["precisions"][str(bits)]).read_text())
                qat = checkpoint_joint_config(
                    checked_record(proof["evidence"]["checkpoint_manifest"]),
                    bits,
                    common["base_draft_gguf"],
                )
            else:
                qat = JointQATConfig(W1AxContract(bits, "row"))
            NativeCaptureProvider(qat, staged)
        os.rename(staged, output)
    return spec


def prune_owned_evaluations(parent: Path, keep: int, *, active: Path | None = None) -> None:
    if type(keep) is not int or keep < 1:
        raise ValueError("development retention needs at least one owned attempt")
    if not parent.exists():
        return
    owned = sorted(
        (
            p
            for p in parent.iterdir()
            if p.is_dir() and not p.is_symlink() and (p / "ownership.json").is_file()
        ),
        key=lambda p: p.stat().st_mtime_ns,
    )
    summaries_path = parent / "summaries.json"
    summaries = json.loads(summaries_path.read_text()) if summaries_path.exists() else []
    for expired in owned[:-keep]:
        if expired == active:
            continue
        ownership = json.loads((expired / "ownership.json").read_text())
        if ownership.get("schema") != "continuous_development_owned_v1":
            raise ValueError("refuse to prune foreign development directory")
        report_path = expired / "report.json"
        if report_path.exists():
            summaries.append(
                {
                    "attempt": expired.name,
                    "status": "completed",
                    "report_sha256": sha256(report_path),
                    "metrics": json.loads(report_path.read_text())["metrics"],
                }
            )
        else:
            summaries.append(
                {
                    "attempt": expired.name,
                    "status": "partial_owned_evaluation",
                    "ownership_sha256": sha256(expired / "ownership.json"),
                }
            )
        if expired.parent != parent:
            raise ValueError("unsafe development retention path")
        shutil.rmtree(expired)
    write_json(summaries_path, summaries[-256:])


def native_acceptance_metrics(cell: dict) -> dict:
    totals = {
        key: sum(r["quality"][key] for r in cell["requests"])
        for key in ("rounds", "accepted", "proposed", "emitted", "accepted_emitted")
    }
    rounds, proposed = totals["rounds"], totals["proposed"]
    if rounds < 1 or proposed < 1:
        raise ValueError("native development comparison has no draft proposals/rounds")
    return {
        "native_accepted_drafts": totals["accepted"],
        "native_rounds": rounds,
        "native_proposed_drafts": proposed,
        "native_emitted_tokens": totals["emitted"],
        "native_accepted_emitted": totals["accepted_emitted"],
        "native_accepted_per_round": totals["accepted"] / rounds,
        "native_acceptance_rate": totals["accepted"] / proposed,
        "native_mean_emitted_per_round": totals["emitted"] / rounds,
    }


def _development_checkpoint_preflight(checkpoint_dir, base_hash):
    """Authenticate the paired deployment publication before staging arrays.

    Paired A8/A1 only; A4/curriculum development remains unsupported. This is
    effective deployment replay, not optimizer/master-state restoration.
    """
    from check_continuous_w1ax_readiness import checkpoint_joint_config
    from export_recurrent_binary import (
        check_manifest,
        load_affine_weights,
        load_checkpoint,
        load_fusion_correction,
    )

    checkpoint_dir = Path(checkpoint_dir)
    publication = checkpoint_dir / "manifest.json"
    if publication.is_symlink() or not publication.is_file():
        raise ValueError("published paired checkpoint manifest is missing or symlinked")
    identities = {"publication": file_record(publication), "lanes": {}}
    published = json.loads(publication.read_text())
    if published.get("schema") != "continuous_joint_w1ax_v1" or set(
        published.get("exports", {})
    ) != {"A8", "A1"}:
        raise ValueError(
            "published paired A8/A1 export inventory required; A4/curriculum unsupported"
        )
    for bits in (8, 1):
        inventory = published["exports"][f"A{bits}"]
        if not isinstance(inventory, dict) or set(inventory) != {"joint.npz", "joint.json"}:
            raise ValueError("published checkpoint export file inventory differs")
        if any(
            not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in inventory.values()
        ):
            raise ValueError("published checkpoint export requires full SHA256 identities")
        identities["lanes"][bits] = dict(inventory)
    _development_checkpoint_unchanged(checkpoint_dir, 8, identities)
    configs, manifests, shapes = {}, {}, {}
    peak_bytes = 0
    for bits in (8, 1):
        manifest_path = checkpoint_dir / f"A{bits}/joint.json"
        configs[bits] = checkpoint_joint_config(manifest_path, bits, base_hash)
        manifest = json.loads(manifest_path.read_text())
        expected = check_manifest(manifest, base_hash)
        if manifest["checkpoint_sha256"] != identities["lanes"][bits]["joint.npz"]:
            raise ValueError("continuous precision checkpoint contract differs")
        quantizers = manifest.get("activation_quantizers")
        if (
            quantizers is not None
            and bits != 1
            and any(item["clip_ratio"] < 2**-16 for item in quantizers["boundaries"].values())
        ):
            raise ValueError("deployed clip parameter is outside trainable quantizer contract")
        # Eight copies of all declared F32 masters/scales and effective optional
        # arrays conservatively cover NPZ reads, Q/K reorder, hashing/packing
        # temporaries and retained packs. Keep the existing 12 GiB staging bound
        # as a floor. This reservation is source-derived, not a measured fit.
        array_bytes = sum((shape[0] * shape[1] + shape[0]) * 4 for _, shape in expected.values())
        correction, affine = manifest.get("fusion_correction"), manifest.get("affine_weights")
        if correction is not None:
            n, k = expected["fc"][1]
            array_bytes += 2 * correction["rank"] * (n + k)
            array_bytes += 4 * n if correction["bias_name"] is not None else 0
        if affine is not None:
            array_bytes += sum(expected[base][1][0] * 4 for base in affine["tensors"])
        peak_bytes = max(peak_bytes, 8 * array_bytes)
        manifests[bits], shapes[bits] = manifest, expected
    _development_checkpoint_unchanged(checkpoint_dir, 8, identities)
    host_admission("development deployment preflight array staging", max(12 * 1024**3, peak_bytes))
    for bits in (8, 1):
        _development_checkpoint_unchanged(checkpoint_dir, bits, identities)
        checkpoint = checkpoint_dir / f"A{bits}/joint.npz"
        manifest, expected = manifests[bits], shapes[bits]
        correction, affine = manifest.get("fusion_correction"), manifest.get("affine_weights")
        extras = (
            load_fusion_correction(checkpoint, correction, expected["fc"][1])
            if correction is not None
            else {}
        )
        midpoints = load_affine_weights(checkpoint, affine, expected) if affine is not None else {}
        packs = load_checkpoint(
            checkpoint, expected, row_scale=True, extra_names=set(extras) | set(midpoints)
        )
        del packs, extras, midpoints
        _development_checkpoint_unchanged(checkpoint_dir, bits, identities)
    return configs, identities


def _development_checkpoint_unchanged(checkpoint_dir, bits, identities):
    """Recheck the original publication and BOTH lanes at every costly boundary."""
    if bits not in (8, 1):
        raise ValueError("paired development supports A8/A1 only; A4/curriculum unsupported")
    checkpoint_dir = Path(checkpoint_dir)
    publication = checkpoint_dir / "manifest.json"
    if (
        publication.is_symlink()
        or not publication.is_file()
        or sha256(publication) != identities["publication"]["sha256"]
    ):
        raise ValueError("development publication identity changed after preflight")
    for lane in (8, 1):
        for name, digest in identities["lanes"][lane].items():
            path = checkpoint_dir / f"A{lane}" / name
            if path.is_symlink() or not path.is_file() or sha256(path) != digest:
                raise ValueError(
                    "development checkpoint export hash mismatch "
                    "or identity changed after preflight"
                )


def _evaluate_development(config: dict, checkpoint_dir: Path, run_dir: Path) -> dict:
    """Serialized current-checkpoint loss and native acceptance versus Q4_0.

    The continuous engine offloads both models/optimizer moments before this
    call. Native servers and Torch validation run sequentially in one owner.
    These instrumented native captures measure acceptance, not throughput.
    """
    import gc

    import torch
    from check_continuous_w1ax_readiness import _load_checkpoint
    from export_recurrent_binary import export_model
    from w1ax_multishard_provider import StreamingNativeProvider

    from w1a1_eagle.recurrent_loss import supported_prefix_ce
    from w1a1_eagle.recurrent_provider import audit_provider_round, forward_torch_round
    from w1a1_eagle.recurrent_qat import (
        install_joint_linears,
        shared_round_hard_signs,
    )

    if (
        config.get("schema") != "w1ax_continuous_development_v1"
        or config.get("split") != "development"
    ):
        raise ValueError("development requires explicit independent provenance")
    started = time.monotonic()
    deadline = started + config.get("max_wall_seconds", 1200)
    sources = {
        **config["sources"],
        "stop_file": str((Path(run_dir) / "STOP").resolve()),
        "development_deadline": deadline,
        "progress_file": str((Path(run_dir) / "status.json").resolve()),
    }
    check_stop(run_dir)
    checkpoint_configs, checkpoint_identities = _development_checkpoint_preflight(
        checkpoint_dir, sources["sha256"]["base_draft_gguf"]
    )
    prompts = Path(config["native_prompts"])
    if sha256(prompts) != config["native_prompts_sha256"]:
        raise ValueError("frozen development subset changed")
    prompt_rows = read_jsonl(prompts)
    if len(prompt_rows) != 24 or any("final" in p["id"].lower() for p in prompt_rows):
        raise ValueError("fixed development capture requires 24 unsealed prompts")
    parent = Path(run_dir) / "development"
    attempts = (
        [
            int(p.name.rsplit("-", 1)[-1])
            for p in parent.iterdir()
            if p.name.startswith(checkpoint_dir.name + "-attempt-")
        ]
        if parent.exists()
        else []
    )
    attempt = max(attempts, default=-1) + 1
    output = parent / f"{checkpoint_dir.name}-attempt-{attempt:04d}"
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "ownership.json",
        {"schema": "continuous_development_owned_v1", "checkpoint": str(checkpoint_dir.resolve())},
    )
    prune_owned_evaluations(output.parent, config.get("keep_native_evaluations", 3), active=output)
    results = {}
    for bits in (8, 1):
        check_stop(run_dir)
        if time.monotonic() >= deadline:
            raise TimeoutError("development aggregate deadline exceeded before native capture")
        folder = output / f"A{bits}"
        folder.mkdir()
        checkpoint = checkpoint_dir / f"A{bits}/joint.npz"
        manifest = checkpoint_dir / f"A{bits}/joint.json"
        exported = folder / "student.gguf"
        _development_checkpoint_unchanged(checkpoint_dir, bits, checkpoint_identities)
        export_audit = export_model(
            Path(sources["base_draft_gguf"]), checkpoint, manifest, exported
        )
        write_json(folder / "export-audit.json", export_audit)
        _development_checkpoint_unchanged(checkpoint_dir, bits, checkpoint_identities)
        native_capture(sources, prompts, folder / "native", activation_bits=bits, draft=exported)
        native_cell = json.loads((folder / "native/d_d/manifest.json").read_text())
        accepted = sum(r["quality"]["accepted"] for r in native_cell["requests"])
        rounds = sum(r["quality"]["rounds"] for r in native_cell["requests"])
        results[f"A{bits}"] = {
            **native_acceptance_metrics(native_cell),
            "checkpoint_sha256": sha256(checkpoint),
            "native_accepted_drafts": accepted,
            "native_rounds": rounds,
            "native_accepted_per_round": accepted / rounds,
            "native_cell": file_record(folder / "native/d_d/manifest.json"),
            "native_generated_ids": {
                r["id"]: r["generated_token_ids"] for r in native_cell["requests"]
            },
        }
    q4 = output / "q4_0"
    if sha256(Path(sources["q4_0_draft"])) != sources["sha256"]["q4_0_draft"]:
        raise ValueError("frozen Q4_0 baseline changed")
    _development_checkpoint_unchanged(checkpoint_dir, 8, checkpoint_identities)
    native_capture(sources, prompts, q4, draft=Path(sources["q4_0_draft"]))
    q4_cell = json.loads((q4 / "d_d/manifest.json").read_text())
    q4_accepted = sum(r["quality"]["accepted"] for r in q4_cell["requests"])
    q4_rounds = sum(r["quality"]["rounds"] for r in q4_cell["requests"])
    q4_ids = {r["id"]: r["generated_token_ids"] for r in q4_cell["requests"]}
    results["Q4_0"] = {
        **native_acceptance_metrics(q4_cell),
        "native_accepted_drafts": q4_accepted,
        "native_rounds": q4_rounds,
        "native_accepted_per_round": q4_accepted / q4_rounds,
        "native_cell": file_record(q4 / "d_d/manifest.json"),
    }
    for bits in (8, 1):
        _development_checkpoint_unchanged(checkpoint_dir, bits, checkpoint_identities)
        qat = replace(checkpoint_configs[bits], device="cuda:0", allow_accelerator=True)
        provider = StreamingNativeProvider(qat, Path(config["providers_manifest"]))
        if provider.data_split != "development" or provider.training_eligible:
            raise ValueError("validation loss cannot use train ownership")
        host_admission("development CPU target/draft load after training offload", 12 * 1024**3)
        _development_checkpoint_unchanged(checkpoint_dir, bits, checkpoint_identities)
        drafter, target = provider.load_models_cpu()
        _development_checkpoint_unchanged(checkpoint_dir, bits, checkpoint_identities)
        linears = install_joint_linears(drafter, target, qat)
        drafter.to("cuda:0")
        _development_checkpoint_unchanged(checkpoint_dir, bits, checkpoint_identities)
        _load_checkpoint(
            checkpoint_dir / f"A{bits}/joint.npz",
            checkpoint_dir / f"A{bits}/joint.json",
            linears,
            bits,
            provider.base_gguf_sha256,
        )
        _development_checkpoint_unchanged(checkpoint_dir, bits, checkpoint_identities)
        adapter = provider.make_step_adapter(drafter)
        losses, labels, selected = 0.0, 0, 0
        loss_selection = []
        subset = {p["id"] for p in prompt_rows}
        per_prompt = {}
        per_prompt_cap = max(1, config.get("max_loss_rounds", 64) // len(subset))
        with torch.no_grad():
            for batch in provider.rounds():
                check_stop(run_dir)
                if time.monotonic() >= deadline:
                    raise TimeoutError("development aggregate deadline exceeded during loss")
                prompt_id = batch.anchor.prompt_id
                if prompt_id not in subset or per_prompt.get(prompt_id, 0) >= per_prompt_cap:
                    continue
                audit = audit_provider_round(batch, provider)
                if not any(audit.ce_mask):
                    continue
                batch = replace(batch, raw_target_features=batch.raw_target_features.to("cuda:0"))
                with shared_round_hard_signs(linears):
                    logits = forward_torch_round(batch, adapter, provider.draft_vocab_size)
                loss = supported_prefix_ce(logits, audit)
                if not torch.isfinite(loss):
                    raise FloatingPointError("development loss is nonfinite")
                count = sum(audit.ce_mask)
                losses += float(loss) * count
                labels += count
                selected += 1
                per_prompt[prompt_id] = per_prompt.get(prompt_id, 0) + 1
                loss_selection.append(
                    {
                        "prompt_id": prompt_id,
                        "round_index": batch.anchor.round_index,
                        "prefix_sha256": hashlib.sha256(
                            json.dumps(list(batch.prefix_token_ids), separators=(",", ":")).encode()
                        ).hexdigest(),
                        "supported_labels": count,
                    }
                )
                if len(per_prompt) == len(subset) and all(
                    n >= per_prompt_cap for n in per_prompt.values()
                ):
                    break
        if not labels:
            raise ValueError("development yielded no supported labels")
        results[f"A{bits}"].update(
            validation_loss=losses / labels,
            validation_labels=labels,
            validation_rounds=selected,
            validation_selection=loss_selection,
            validation_prompt_count=len(per_prompt),
            validation_subset_sha256=config["native_prompts_sha256"],
            validation_source_manifest=file_record(Path(config["providers_manifest"])),
            response_id_matches_q4=sum(
                results[f"A{bits}"]["native_generated_ids"][p] == q4_ids[p] for p in q4_ids
            ),
            native_acceptance_ratio_to_q4=results[f"A{bits}"]["native_accepted_per_round"]
            / results["Q4_0"]["native_accepted_per_round"]
            if q4_accepted
            else None,
        )
        del results[f"A{bits}"]["native_generated_ids"]
        del adapter, linears, drafter, target, provider, logits, batch
        gc.collect()
        torch.cuda.empty_cache()
    report = {
        "schema": "w1ax_continuous_development_report_v1",
        "split": "development",
        "elapsed_seconds": time.monotonic() - started,
        "max_wall_seconds": deadline - started,
        "checkpoint": str(checkpoint_dir),
        "prompts": file_record(prompts),
        "selected_native_prompts": 24,
        "frozen_source_sha256": sources["sha256"],
        "precision": {
            "target_weights": "F16",
            "target_kv": "F16",
            "draft_kv": "F16",
            "binary_drafts": "row W1A8 and row W1A1",
            "primary_baseline": "frozen Q4_0 draft with native Q8_1 activation conversion",
        },
        "native_repetitions": 1,
        "hardware": torch.cuda.get_device_name(0),
        "execution_device": "cuda:0",
        "metrics": results,
        "comparison": "Q4_0 primary; acceptance only, no timing claim",
        "sealed_test_accessed": False,
    }
    write_json(output / "report.json", report)
    prune_owned_evaluations(output.parent, config.get("keep_native_evaluations", 3), active=output)
    return report


def evaluate_development(config: dict, checkpoint_dir: Path, run_dir: Path) -> dict:
    # Engine checkpoint callback supplies {path:<.../state.json>,step,...}.
    if isinstance(checkpoint_dir, dict):
        checkpoint_dir = Path(checkpoint_dir["path"]).parent
    with native_cancellation():
        return _evaluate_development(config, Path(checkpoint_dir), run_dir)


def recover_partial(run_dir: Path, stages_config: Path) -> dict:
    """CPU-only quarantine of dead, incomplete owned native captures.

    This never deletes raw data, starts a process, or queries an accelerator.
    Completed native cells remain available for CPU label-bundle rebuilding.
    A subsequent explicit USER --start --resume is the only execution trigger.
    """
    run_dir = run_dir.resolve()
    config = json.loads(stages_config.read_text())
    if config.get("schema") != STAGES_SCHEMA:
        raise ValueError("recovery needs the unchanged typed stages configuration")
    resolved_path = run_dir / "resolved_config.json"
    if not resolved_path.is_file() or json.loads(resolved_path.read_text()).get("stages") != config:
        raise ValueError("recovery stages differ from the saved manual-start configuration")
    status = json.loads((run_dir / "status.json").read_text())
    if any(model.get("step", 0) > 0 for model in status.get("models", {}).values()):
        raise ValueError("preparation recovery cannot change substantive training sources")
    owner = status.get("pid")
    if type(owner) is int:
        try:
            os.kill(owner, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("preparation owner remains alive; stop and verify before recovery")
    verify_sources(config["sources"])
    stage_dir = run_dir / "stages"
    quarantine = run_dir / "recovery" / f"partial-{time.time_ns()}"
    records = []
    for folder in sorted(stage_dir.iterdir()) if stage_dir.exists() else []:
        if not folder.is_dir() or folder.is_symlink():
            continue
        native = folder / "native"
        if not native.is_dir() or native.is_symlink():
            continue
        cell_path = native / "d_d/manifest.json"
        cell = json.loads(cell_path.read_text()) if cell_path.is_file() else {}
        if cell.get("complete") is True:
            continue
        pgid = cell.get("server_pgid")
        if (
            type(pgid) is not int
            and cell.get("server_stop", {}).get("process_group_gone") is not True
        ):
            raise RuntimeError(
                "partial native capture lacks process-group stop proof; "
                "verify its owner before CPU recovery"
            )
        if type(pgid) is int:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(
                    "owned native process group remains alive; recovery cannot stop it"
                )
        if not folder.name.startswith(("capture-", "gate-a")):
            raise ValueError("refuse to quarantine a foreign stage directory")
        quarantine.mkdir(parents=True, exist_ok=True)
        destination = quarantine / (folder.name + "-native")
        manifest_hash = sha256(cell_path) if cell_path.is_file() else None
        os.rename(native, destination)
        records.append(
            {
                "original": str(native),
                "preserved": str(destination),
                "partial_cell_sha256": manifest_hash,
            }
        )
        labels = folder / "labels"
        if labels.exists():
            if labels.is_symlink():
                raise ValueError("refuse symlink label recovery target")
            labels_destination = quarantine / (folder.name + "-labels")
            os.rename(labels, labels_destination)
            records.append({"original": str(labels), "preserved": str(labels_destination)})
    report = {
        "schema": "w1ax_cpu_partial_recovery_v1",
        "execution_device": "cpu",
        "stages_config": file_record(stages_config),
        "records": records,
        "raw_artifacts_deleted": False,
        "processes_started": False,
        "next_action": "explicit manual --start --resume; completed native cells rebuild on CPU",
    }
    write_json(quarantine / "recovery.json" if records else run_dir / "recovery-empty.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--prompts-sha256", required=True)
    audit.add_argument("--prompt-count", type=int, required=True)
    prep = sub.add_parser("prepare-config", help="CPU only; writes manual-start inputs")
    prep.add_argument("--corpus-manifest", type=Path, required=True)
    prep.add_argument(
        "--legacy-provider",
        type=Path,
        default=Path("runs/w1-shard0000-provider-hardce-20260929/provider.json"),
    )
    prep.add_argument(
        "--binary",
        type=Path,
        default=Path("runs/w1-final-runtime-timing-freeze-20260929/runtime/llama-server"),
    )
    prep.add_argument(
        "--q4-draft", type=Path, default=Path("models/gguf/Qwen3-4B-eagle3-q4_0.gguf")
    )
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--shard-prompts", type=int, default=32)
    prep.add_argument("--max-storage-bytes", type=int, default=1024**4)
    retained = sub.add_parser(
        "prepare-retained-config", help="CPU-only explicit retained capture/gate adoption"
    )
    retained.add_argument("--import-manifest", type=Path, required=True)
    retained.add_argument("--output", type=Path, required=True)
    recovery = sub.add_parser("recover-partial", help="CPU-only quarantine; no launch/deletion")
    recovery.add_argument("--run-dir", type=Path, required=True)
    recovery.add_argument("--stages-config", type=Path, required=True)
    refreshed_provider = sub.add_parser(
        "make-refresh-provider", help="CPU new explicit source binding"
    )
    refreshed_provider.add_argument("--stages-config", type=Path, required=True)
    refreshed_provider.add_argument("--refresh-receipt", type=Path, required=True)
    refreshed_provider.add_argument("--readiness", type=Path, required=True)
    refreshed_provider.add_argument("--output", type=Path, required=True)
    refresh = sub.add_parser(
        "refresh", help="USER-start native capture; never runs during preparation"
    )
    refresh.add_argument("--stages-config", type=Path, required=True)
    refresh.add_argument("--checkpoint-dir", type=Path, required=True)
    refresh.add_argument("--prompts", type=Path, required=True)
    refresh.add_argument("--prompts-sha256", required=True)
    refresh.add_argument("--activation-bits", type=int, choices=(1, 4, 8), required=True)
    refresh.add_argument("--output", type=Path, required=True)
    refresh.add_argument("--allow-cuda", action="store_true")
    args = parser.parse_args()
    if args.command == "audit":
        result = audit_native_labels(
            args.manifest,
            expected_prompt_sha256=args.prompts_sha256,
            expected_prompt_count=args.prompt_count,
        )
    elif args.command == "prepare-config":
        result = prepare_config(
            args.corpus_manifest,
            args.legacy_provider,
            args.binary,
            args.q4_draft,
            args.output,
            shard_prompts=args.shard_prompts,
            max_storage_bytes=args.max_storage_bytes,
        )
        result = {
            "config": str(args.output),
            "split_counts": result["split_counts"],
            "storage_forecast": result["storage_forecast"],
            "execution_device": "cpu",
        }
    elif args.command == "prepare-retained-config":
        config = prepare_retained_config(args.import_manifest, args.output)
        result = {
            "config": str(args.output),
            "execution_device": "cpu",
            "training_eligible": False,
            "retained_native_capture_import": config["retained_native_capture_import"],
        }
    elif args.command == "recover-partial":
        result = recover_partial(args.run_dir, args.stages_config)
    elif args.command == "make-refresh-provider":
        result = make_refresh_provider(
            args.stages_config, args.refresh_receipt, args.readiness, args.output
        )
    else:
        if not args.allow_cuda:
            parser.error("refresh requires an explicit USER-start --allow-cuda")
        sources = json.loads(args.stages_config.read_text())["sources"]
        source_config = json.loads(args.stages_config.read_text())
        sources = {
            **sources,
            "stages_config": file_record(args.stages_config),
            "corpus_manifest": source_config["corpus_manifest"],
        }
        require_unsealed_prompts(args.prompts, sources=sources, expected_sha256=args.prompts_sha256)
        import torch
        from train_continuous_w1ax import lock

        gpu_lock = lock(Path.home() / ".cache/binary-eagle-decoding/cuda-0.owner.lock")
        try:
            properties = torch.cuda.get_device_properties("cuda:0")
            if properties.name != "NVIDIA GeForce RTX 5080" or [
                properties.major,
                properties.minor,
            ] != [12, 0]:
                raise RuntimeError("manual refresh requires the declared RTX5080 SM120 device")
            result = {
                "manifest": str(
                    refresh_checkpoint(
                        sources,
                        args.checkpoint_dir,
                        args.prompts,
                        args.output,
                        args.activation_bits,
                        prompts_sha256=args.prompts_sha256,
                    )
                )
            }
        finally:
            gpu_lock.close()

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
