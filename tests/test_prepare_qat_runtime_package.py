"""Tiny ELF files as data only; no compiler/native/server/GPU execution."""

import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "packaging", ROOT / "scripts/prepare_qat_runtime_package.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def tiny_elf(
    runpath, soname=None, needed=(), version_kind=None, version_alias=True, path_prefix=""
):
    labels = ["", ".shstrtab", ".dynstr", ".dynamic", ".dynsym", ".text", ".rodata", ".nv_fatbin"]
    if version_kind:
        labels.append(".gnu.version_d" if version_kind == "definition" else ".gnu.version_r")
    names = b"\0"
    name_offsets = {}
    for name in labels[1:]:
        name_offsets[name] = len(names)
        names += name.encode() + b"\0"
    strings = bytearray(b"\0")
    entries = []

    def put(s):
        offset = len(strings)
        strings.extend(s.encode() + b"\0")
        return offset

    entries.extend((1, put(name)) for name in needed)
    if soname:
        entries.append((14, put(soname)))
    safe_version = put("SAFE_VERSION") if version_kind else 0
    if runpath is not None:
        entries.append((29, put(path_prefix + runpath) + len(path_prefix)))
    entries.extend([(5, 0x402000), (10, len(strings)), (0, 0)])
    bodies = [
        b"",
        names,
        bytes(strings),
        b"".join(struct.pack("<qQ", *entry) for entry in entries),
        bytes(24),
        b"code never executed",
        b"readonly identity",
        b"CUDA bytes never executed",
    ]
    if version_kind:
        alias = next(v for t, v in entries if t == 29)
        name = alias if version_alias else safe_version
        if version_kind == "definition":
            bodies.append(
                struct.pack("<HHHHIII", 1, 0, 2, 1, 0, 20, 0) + struct.pack("<II", name, 0)
            )
        else:
            file_name = alias if version_kind == "need-file" and version_alias else 0
            aux_name = name if version_kind == "need-name" else safe_version
            bodies.append(
                struct.pack("<HHIII", 1, 1, file_name, 16, 0)
                + struct.pack("<IHHII", 0, 0, 2, aux_name, 0)
            )
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<Q", header, 40, 64)
    struct.pack_into("<HHH", header, 58, 64, len(labels), 1)
    offset = 64 + 64 * len(labels)
    rows = []
    for index, (name, body) in enumerate(zip(labels, bodies)):
        kind = {".shstrtab": 3, ".dynstr": 3, ".dynamic": 6, ".dynsym": 11}.get(name, 1)
        addr = 0x402000 if name == ".dynstr" else 0
        link = 2 if name in (".dynamic", ".dynsym", ".gnu.version_r", ".gnu.version_d") else 0
        stride = 16 if name == ".dynamic" else 24 if name == ".dynsym" else 0
        rows.append(
            struct.pack(
                "<IIQQQQIIQQ",
                name_offsets.get(name, 0),
                kind,
                2,
                addr,
                offset,
                len(body),
                link,
                0,
                1,
                stride,
            )
        )
        offset += len(body)
    return bytes(header) + b"".join(rows) + b"".join(bodies)


class FixedSlotPackage(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.checkout = self.root / "checkout"
        self.build = self.checkout / "third_party/llama.cpp/build"
        self.bin = self.build / "bin"
        self.bin.mkdir(parents=True)
        self.output = self.root / "runtime-copy"
        self.old = str(self.bin) + ":"
        self.libnames = [
            "libggml-base.so.0.25.1",
            "libggml-cpu.so.0.25.1",
            "libggml-cuda.so.0.25.1",
            "libggml.so.0.25.1",
            "libllama.so.0.5.0",
            "libllama-common.so.0.5.0",
        ]
        self.aliases = {}
        items = []
        for name in [
            "llama-server",
            "test-eagle3-learned",
            *self.libnames,
            "libllama-server-impl.so",
            "libmtmd.so",
        ]:
            library = ".so" in name
            soname = (
                name.split(".so")[0] + ".so.0"
                if name in self.libnames
                else name
                if library
                else None
            )
            needed = ["libc.so.6"] if name.startswith("libggml-base") else ["libggml-base.so.0"]
            if name == "llama-server":
                needed = ["libllama-server-impl.so"]
            if name == "libllama-server-impl.so":
                needed = ["libmtmd.so"]
            runpath = None if name.startswith("libggml-base") else self.old
            path = self.bin / name
            path.write_bytes(tiny_elf(runpath, soname, needed))
            path.chmod(0o755)
            items.append(
                dict(
                    canonical=str(path),
                    sha256=tool.sha(path.read_bytes()),
                    dynamic_paths=[] if runpath is None else [{"kind": "RUNPATH", "raw": runpath}],
                )
            )
            if name in self.libnames:
                for alias in (soname, name.split(".so")[0] + ".so"):
                    (self.bin / alias).symlink_to(name)
                    self.aliases[alias] = name
        original = self.root / "original-census.json"
        closure = self.root / "closure-census.json"
        original.write_text(
            json.dumps(
                dict(
                    schema="qat_elf_metadata_probe_v1",
                    protected_seven_match=True,
                    all_eight_unchanged=True,
                    artifacts=items[:8],
                )
            )
        )
        closure.write_text(
            json.dumps(
                dict(
                    schema="qat_elf_closure_probe_v1",
                    protected_seven_match=True,
                    all_artifacts_unchanged=True,
                    missing_project_needed=[],
                    aliases=self.aliases,
                    artifacts=items,
                )
            )
        )
        self.census, self.original = tool.record(closure), tool.record(original)
        self.before = {
            p: tool.sha(p.read_bytes()) for p in self.bin.iterdir() if not p.is_symlink()
        }

    def create(self):
        return tool.prepare(
            self.checkout, self.build, "a" * 40, "b" * 40, self.census, self.original, self.output
        )

    def verify(self, proof):
        return tool.verify_package(
            proof["manifest"]["path"],
            proof["manifest"]["sha256"],
            self.checkout,
            self.build,
            "a" * 40,
            "b" * 40,
        )

    def test_package_changes_only_one_slot_and_keeps_complete_closure(self):
        proof = self.create()
        self.verify(proof)
        self.assertEqual(len(proof["files"]), 10)
        self.assertTrue(proof["requires_fresh_native_validation"])
        for item in proof["files"]:
            source = Path(item["source"]["path"])
            copy = Path(item["copy"]["path"])
            self.assertEqual(tool.sha(source.read_bytes()), self.before[source])
            old, new = source.read_bytes(), copy.read_bytes()
            if item["patch"]:
                a = item["patch"]["offset"]
                b = a + item["patch"]["bytes"]
                self.assertEqual(old[:a], new[:a])
                self.assertEqual(old[b:], new[b:])
                self.assertEqual(tool.elf(new)["paths"][0][2], "$ORIGIN")
            else:
                self.assertEqual(old, new)
            self.assertEqual(tool.elf(old)["identities"], tool.elf(new)["identities"])
        self.assertTrue(
            all(os.readlink(self.output / name) == target for name, target in self.aliases.items())
        )

    def test_wrong_known_path_and_multiple_path_tags_fail(self):
        with self.assertRaisesRegex(ValueError, "Known DT_RUNPATH differs"):
            tool.transform(tiny_elf("/other:"), self.old)
        with self.assertRaisesRegex(ValueError, "Unexpected runtime path"):
            tool.transform(tiny_elf(self.old), None)

    def test_other_string_dynamic_tags_cannot_overlap_slot(self):
        for tag in (1, 14, 15, 0x7FFFFFFD, 0x7FFFFFFF, 0x6FFFFEFA, 0x6FFFFEFB, 0x6FFFFEFC):
            with self.subTest(tag=tag):
                data = bytearray(tiny_elf(self.old, needed=["placeholder"]))
                layout = tool.elf(data)
                index = layout["paths"][0][1]
                dynamic = layout["sections"][".dynamic"][1]
                struct.pack_into("<qQ", data, dynamic[4], tag, index)
                with self.assertRaisesRegex(ValueError, "overlaps RUNPATH|Exactly one"):
                    tool.transform(bytes(data), self.old)

    def test_dynamic_reference_starting_before_slot_cannot_span_it(self):
        data = bytearray(tiny_elf(self.old, needed=["unused"], path_prefix="prefix"))
        layout = tool.elf(data)
        index = layout["paths"][0][1]
        dynamic = layout["sections"][".dynamic"][1]
        struct.pack_into("<qQ", data, dynamic[4], 0x6FFFFEFC, index - 6)
        with self.assertRaisesRegex(ValueError, "overlaps RUNPATH"):
            tool.transform(bytes(data), self.old)

    def test_symbol_reference_starting_before_slot_cannot_span_it(self):
        data = bytearray(tiny_elf(self.old, path_prefix="prefix"))
        layout = tool.elf(data)
        index = layout["paths"][0][1]
        symbols = layout["sections"][".dynsym"][1]
        struct.pack_into("<I", data, symbols[4], index - 6)
        with self.assertRaisesRegex(ValueError, "overlaps RUNPATH"):
            tool.transform(bytes(data), self.old)

    def test_gnu_version_file_and_auxiliary_names_cannot_alias_slot(self):
        for kind in ("need-file", "need-name", "definition"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "overlaps RUNPATH"):
                tool.transform(tiny_elf(self.old, version_kind=kind), self.old)

    def test_gnu_nonoverlapping_versions_are_preserved(self):
        for kind in ("need-file", "need-name", "definition"):
            with self.subTest(kind=kind):
                data = tiny_elf(self.old, version_kind=kind, version_alias=False)
                changed, patch, _ = tool.transform(data, self.old)
                self.assertEqual(len(data), len(changed))
                self.assertTrue(patch["every_other_byte_identical"])

    def test_gnu_version_out_of_bounds_chain_is_rejected(self):
        data = bytearray(tiny_elf(self.old, version_kind="need-name", version_alias=False))
        row = tool.elf(data)["sections"][".gnu.version_r"][1]
        struct.pack_into("<I", data, row[4] + 8, 0xFFFFFFF0)
        with self.assertRaisesRegex(ValueError, "Bad version auxiliary"):
            tool.transform(bytes(data), self.old)

    def test_changed_copy_code_fails_even_with_new_manifest_sha(self):
        proof = self.create()
        manifest = Path(proof["manifest"]["path"])
        os.chmod(self.output, 0o755)
        os.chmod(manifest, 0o644)
        m = json.loads(manifest.read_text())
        target = self.output / "llama-server"
        os.chmod(target, 0o755)
        data = bytearray(target.read_bytes())
        row = tool.elf(data)["sections"][".text"][1]
        data[row[4]] ^= 1
        target.write_bytes(data)
        m["files"][0]["copy"] = tool.record(target)
        manifest.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "beyond proven RUNPATH"):
            tool.verify_package(
                manifest,
                tool.record(manifest)["sha256"],
                self.checkout,
                self.build,
                "a" * 40,
                "b" * 40,
            )

    def test_changed_original_and_missing_dependency_fail(self):
        proof = self.create()
        (self.bin / "libmtmd.so").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Original ELF bytes differ"):
            self.verify(proof)

    def test_alias_escape_extra_member_and_reused_output_fail(self):
        proof = self.create()
        os.chmod(self.output, 0o755)
        alias = self.output / "libggml-base.so"
        alias.unlink()
        alias.symlink_to(self.bin / self.libnames[0])
        with self.assertRaisesRegex(ValueError, "Runtime alias differs"):
            self.verify(proof)
        with self.assertRaisesRegex(ValueError, "output reused"):
            self.create()

    def test_wrong_census_sha_and_original_ancestry_fail(self):
        bad = {**self.census, "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "Evidence bytes/path differ"):
            tool.prepare(
                self.checkout, self.build, "a" * 40, "b" * 40, bad, self.original, self.output
            )
        closure = json.loads(Path(self.census["path"]).read_text())
        closure["artifacts"][0]["sha256"] = "0" * 64
        Path(self.census["path"]).write_text(json.dumps(closure))
        with self.assertRaisesRegex(ValueError, "Original eight artifact ancestry"):
            tool.prepare(
                self.checkout,
                self.build,
                "a" * 40,
                "b" * 40,
                tool.record(self.census["path"]),
                self.original,
                self.output,
            )

    def test_missing_transitive_project_copy_rejected(self):
        closure = json.loads(Path(self.census["path"]).read_text())
        closure["artifacts"] = [
            r for r in closure["artifacts"] if not r["canonical"].endswith("/libmtmd.so")
        ]
        Path(self.census["path"]).write_text(json.dumps(closure))
        with self.assertRaisesRegex(ValueError, "project dependency closure missing"):
            tool.prepare(
                self.checkout,
                self.build,
                "a" * 40,
                "b" * 40,
                tool.record(self.census["path"]),
                self.original,
                self.output,
            )


if __name__ == "__main__":
    unittest.main()
