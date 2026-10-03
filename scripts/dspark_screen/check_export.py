"""Inspect released/native exports and gate exact embedding borrowing.

Use the selected llama.cpp gguf-py and numpy environment. This performs no
inference. Target hash and native binding traces remain separate admission gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect(source: Path, target: Path, draft: Path, config: Path, llama: Path, comparison: Path) -> dict:
    sys.path.insert(0, str(llama / "gguf-py"))
    import numpy as np
    from gguf import GGUFReader, GGMLQuantizationType

    cfg = json.loads(config.read_text())
    required = {"architectures": ["Qwen3DSparkModel"], "block_size": 7,
                "target_layer_ids": [1, 9, 17, 25, 33], "hidden_size": 2560,
                "num_hidden_layers": 5, "vocab_size": 151936, "dtype": "bfloat16",
                "mask_token_id": 151669}
    if any(cfg.get(k) != v for k, v in required.items()):
        raise ValueError("selected released config contract changed")
    reader = GGUFReader(str(draft))
    def field(name):
        f = reader.get_field(name)
        return f.contents() if f else None
    meta = {k: field(k) for k in ("general.architecture", "dflash.block_size",
            "dflash.sample_from_anchor", "dflash.target_layers", "dflash.attention.causal",
            "dflash.has_confidence_head", "tokenizer.ggml.mask_token_id")}
    if (meta["general.architecture"] != "dflash" or meta["dflash.block_size"] != 7 or
            meta["dflash.sample_from_anchor"] is not True or
            meta["dflash.target_layers"] != [2, 10, 18, 26, 34] or
            meta["dflash.attention.causal"] not in (None, False)):
        raise ValueError("native export does not implement the author anchor-first contract")
    tensors = {t.name: t for t in reader.tensors}
    if "d2t" in tensors:
        raise ValueError("released export must retain the full vocabulary")
    markov = [name for name in tensors if name.startswith("markov_w")]
    if bool(markov) != (cfg.get("markov_rank", 0) > 0):
        raise ValueError("Markov inventory differs from selected release")
    matrix_types = {t.tensor_type.name for t in reader.tensors if len(t.shape) > 1}
    if matrix_types != {"BF16"}:
        raise ValueError(f"released matrix storage changed: {matrix_types}")
    canonical = json.loads(comparison.read_text())
    source_sha, target_sha = sha256(source), sha256(target)
    if canonical["source_sha256"] != source_sha or canonical["target_sha256"] != target_sha:
        raise ValueError("canonical comparison is not bound to these source/target files")
    borrows_embedding = "token_embd.weight" not in tensors
    borrows_head = "output.weight" not in tensors
    if borrows_embedding and not canonical["safe_to_borrow_embedding"]:
        raise ValueError("embedding omitted despite failed canonical equality")
    if borrows_head and not canonical["safe_to_borrow_head"]:
        raise ValueError("head omitted despite failed canonical equality")
    target_reader = GGUFReader(str(target))
    target_tensors = {t.name: t for t in target_reader.tensors}
    target_head_name = "output.weight"
    if target_head_name not in target_tensors:
        from compare_frozen import tied_target_proof
        proof = tied_target_proof(Path(canonical["target_config_path"]), canonical["target_config_sha256"],
                                  llama, target_reader.get_field("general.architecture").contents())
        if not canonical.get("target_tied_head_fallback") or canonical.get("target_head_source_tensor") != "token_embd.weight" or proof["target_loader_sha256"] != canonical["target_loader_sha256"]:
            raise ValueError("canonical/native target tied-head proof differs")
        target_head_name = "token_embd.weight"
    elif canonical.get("target_tied_head_fallback"):
        raise ValueError("canonical tied-head fallback disagrees with actual target inventory")
    for name in ("token_embd.weight", target_head_name):
        if target_tensors[name].tensor_type != GGMLQuantizationType.F16:
            raise ValueError("fixed target embedding/head must remain F16")
    with source.open("rb") as f:
        n_header = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n_header))
    copies = {}
    for suffix, name in (("embed_tokens.weight", "token_embd.weight"), ("lm_head.weight", "output.weight")):
        if name not in tensors:
            continue
        row = header[next(n for n in header if n.endswith(suffix))]
        native = tensors[name]
        if row["dtype"] != "BF16" or native.tensor_type != GGMLQuantizationType.BF16 or native.shape.tolist() != [2560, 151936]:
            raise ValueError("private released copy changed shape/type")
        values = np.memmap(source, mode="r", dtype="<u2", offset=8 + n_header + row["data_offsets"][0], shape=(native.n_elements,))
        original = hashlib.sha256()
        converted = hashlib.sha256()
        for start in range(0, native.n_elements, 1024 * 1024):
            stop = min(start + 1024 * 1024, native.n_elements)
            original.update(values[start:stop].tobytes())
            converted.update(native.data.view("<u2").reshape(-1)[start:stop].tobytes())
        if original.hexdigest() != converted.hexdigest():
            raise ValueError("retained private released copy differs from source bytes")
        copies[name] = {"source_raw_sha256": original.hexdigest(), "native_raw_sha256": converted.hexdigest()}
    return {"schema": "dspark_export_inspection_v1", "passed": True, "metadata": meta,
            "source_sha256": source_sha, "config_sha256": sha256(config),
            "draft_sha256": sha256(draft), "target_sha256": target_sha,
            "canonical_comparison_path": str(comparison), "canonical_comparison_sha256": sha256(comparison),
            "matrix_storage_types": sorted(matrix_types), "borrows_embedding": borrows_embedding,
            "borrows_head": borrows_head, "private_copy_identity": copies,
            "target_tied_head_fallback": canonical.get("target_tied_head_fallback", False),
            "target_head_source_tensor": target_head_name,
            "target_config_path": canonical.get("target_config_path"),
            "target_config_sha256": canonical.get("target_config_sha256"),
            "target_embedding_dtype": "F16", "target_head_dtype": "F16",
            "native_tensor_bytes": sum(t.n_bytes for t in reader.tensors),
            "tensor_inventory": [{"name": t.name, "type": t.tensor_type.name,
                                  "shape": t.shape.tolist(), "bytes": t.n_bytes} for t in reader.tensors],
            "remaining": "actual SM75 load/dispatch, cache/round evidence, target immutability and target-only output checks"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "target", "draft", "config", "llama", "comparison", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    a = p.parse_args()
    report = inspect(a.source, a.target, a.draft, a.config, a.llama, a.comparison)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
