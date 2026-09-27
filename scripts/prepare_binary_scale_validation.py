"""Freeze small replay input set, verify Q4 control and make legacy row endpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/llama.cpp/gguf-py"))
from audit_eagle_w1a1_gguf import SOURCE_NAMES, sha256  # noqa: E402
from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402
from select_w1ax_captures import parse_capture  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-manifest", type=Path, required=True)
    p.add_argument("--fit-dir", type=Path, required=True)
    p.add_argument("--q4", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected = args.output / "selected"
    selected.mkdir()
    prompt = json.loads(args.capture_manifest.read_text())["prompts"][0]
    chosen = {}
    for name in sorted(prompt["files"]):
        path = Path(prompt["capture_dir"]) / name
        cap = parse_capture(path)
        key = (cap["name"], "single" if cap["N"] == 1 else "multi")
        if key in chosen:
            continue
        shutil.copyfile(path, selected / name)
        chosen[key] = {
            "name": cap["name"],
            "N": cap["N"],
            "K": cap["K"],
            "M": cap["M"],
            "source": str(path),
            "file": name,
            "sha256": sha256(path),
        }
    names = {key[0] for key in chosen}
    if names != {f"{base}.weight" for base in SOURCE_NAMES}:
        raise ValueError("replay input set missing selected layer")
    q4 = GGUFReader(args.q4)
    tensor_types = {t.name: t.tensor_type.name for t in q4.tensors}
    for base in SOURCE_NAMES:
        if tensor_types[f"{base}.weight"] != GGMLQuantizationType.Q4_0.name:
            raise ValueError("Q4 control selected tensor type mismatch")
    src = args.fit_dir / "A.gguf"
    dst = args.fit_dir / "A_legacy.gguf"
    if dst.exists():
        raise ValueError("legacy endpoint already exists")
    shutil.copyfile(src, dst)
    modified = GGUFReader(dst, mode="r+")
    version = modified.fields["eagle3.w1a1.version"]
    assert version.contents() == 2
    version.parts[version.data[0]][0] = 1
    del modified
    original, legacy = GGUFReader(src), GGUFReader(dst)
    assert legacy.fields["eagle3.w1a1.version"].contents() == 1
    for a, b in zip(original.tensors, legacy.tensors, strict=True):
        if a.name != b.name or a.tensor_type != b.tensor_type or not np.array_equal(a.data, b.data):
            raise ValueError("legacy row endpoint tensor changed")
    report = {
        "selection": "first train prompt, first single/multi invocation per layer",
        "prompt_id": prompt["id"],
        "captures": list(chosen.values()),
        "q4": {"sha256": sha256(args.q4), "tensor_types": tensor_types},
        "A_sha256": sha256(src),
        "A_legacy_sha256": sha256(dst),
        "legacy_tensor_payloads_exact": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_captures": len(chosen), "layers": len(names)}))


if __name__ == "__main__":
    main()
