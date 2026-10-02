"""Copy a pinned ELF closure, changing only known RUNPATH string slots.

This is CPU packaging, not native validation. Original files are never written.
"""

import argparse
import hashlib
import json
import os
import re
import struct
from pathlib import Path

SHA = re.compile(r"[0-9a-f]{64}$")
MAX_FILE = 256 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path):
    path = Path(path)
    require(path.stat().st_size <= MAX_FILE, "ELF/package file exceeds bound")
    return path.read_bytes()


def record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(read_bytes(path))}


def load_record(item, limit=512 * 1024):
    require(
        isinstance(item, dict) and SHA.fullmatch(item.get("sha256", "")), "Missing evidence SHA"
    )
    data = read_bytes(item["path"])
    require(len(data) <= limit and record(item["path"]) == item, "Evidence bytes/path differ")
    return json.loads(data)


def elf(data):
    """Parse only bounded ELF64LE sections and dynamic string references."""
    require(len(data) >= 64 and data[:6] == b"\x7fELF\x02\x01", "ELF64LE required")
    table = struct.unpack_from("<Q", data, 40)[0]
    stride, count, names_index = struct.unpack_from("<HHH", data, 58)
    require(stride >= 64 and 0 < count <= 4096 and names_index < count, "Bad ELF section table")
    require(table + stride * count <= len(data), "ELF section table outside file")
    rows = [struct.unpack_from("<IIQQQQIIQQ", data, table + stride * i) for i in range(count)]

    def contents(row):
        require(row[1] != 8 and row[4] + row[5] <= len(data), "ELF section outside file")
        return data[row[4] : row[4] + row[5]]

    names = contents(rows[names_index])
    sections = {}
    for index, row in enumerate(rows):
        require(row[0] < len(names), "ELF section name outside table")
        name = names[row[0] :].split(b"\0", 1)[0].decode("ascii")
        require(name not in sections, "Duplicate ELF section name")
        sections[name] = (index, row)
    require(".dynamic" in sections and ".dynstr" in sections, "Dynamic ELF sections missing")
    string_index, strings_row = sections[".dynstr"]
    dynamic = sections[".dynamic"][1]
    require(
        dynamic[1] == 6 and dynamic[6] == string_index and dynamic[5] % 16 == 0,
        "Dynamic string table link differs",
    )
    strings = contents(strings_row)
    entries = []
    for offset in range(0, dynamic[5], 16):
        tag, value = struct.unpack_from("<qQ", data, dynamic[4] + offset)
        if tag == 0:
            break
        entries.append((tag, value))
    require([v for t, v in entries if t == 5] == [strings_row[3]], "DT_STRTAB differs")
    require([v for t, v in entries if t == 10] == [len(strings)], "DT_STRSZ differs")

    def string(offset):
        require(offset < len(strings), "Dynamic string offset outside table")
        end = strings.find(b"\0", offset)
        require(end >= offset, "Dynamic string lacks terminator")
        return strings[offset:end].decode("ascii"), end - offset + 1

    needed = [string(v)[0] for t, v in entries if t == 1]
    sonames = [string(v)[0] for t, v in entries if t == 14]
    paths = [(t, v, *string(v)) for t, v in entries if t in (15, 29)]
    identities = {}
    for name in (".text", ".rodata", ".nv_fatbin", ".nvFatBinSegment", ".dynsym", ".dynamic"):
        if name in sections:
            row = sections[name][1]
            identities[name] = {"sha256": sha(contents(row)), "bytes": row[5]}
    return dict(
        sections=sections,
        strings=strings_row,
        entries=entries,
        needed=needed,
        sonames=sonames,
        paths=paths,
        identities=identities,
    )


def transform(data, old_runpath):
    layout = elf(data)
    if old_runpath is None:
        require(not layout["paths"], "Unexpected runtime path on unchanged copy")
        return data, None, layout
    require(len(layout["paths"]) == 1, "Exactly one known DT_RUNPATH required")
    tag, index, text, width = layout["paths"][0]
    require(
        tag == 29 and text == old_runpath and old_runpath.endswith(":"), "Known DT_RUNPATH differs"
    )
    start = layout["strings"][4] + index
    end = start + width
    strings_row = layout["strings"]
    strings = data[strings_row[4] : strings_row[4] + strings_row[5]]

    def reject_reference(value):
        require(0 <= value < len(strings), "ELF string reference outside table")
        terminator = strings.find(b"\0", value)
        require(terminator >= value, "ELF string reference lacks terminator")
        require(
            terminator + 1 <= index or value >= index + width,
            "ELF string reference overlaps RUNPATH slot",
        )

    # Reject ELF string-table sharing with NEEDED/SONAME or dynamic symbols.
    string_tags = {1, 14, 15, 0x7FFFFFFD, 0x7FFFFFFF, 0x6FFFFEFA, 0x6FFFFEFB, 0x6FFFFEFC}
    for other_tag, value in layout["entries"]:
        if other_tag in string_tags or (other_tag == 29 and value != index):
            reject_reference(value)
    # GNU version records refer to .dynstr too; preserve their string names.
    for section_name, head_size, aux_size, name_field in (
        (".gnu.version_r", 16, 16, 8),
        (".gnu.version_d", 20, 8, 0),
    ):
        if section_name not in layout["sections"]:
            continue
        row = layout["sections"][section_name][1]
        require(row[6] == layout["sections"][".dynstr"][0], "Version string table link differs")
        blob = data[row[4] : row[4] + row[5]]
        require(row[4] + row[5] <= len(data), "Version section outside ELF")
        offset, visited = 0, set()
        while True:
            require(offset not in visited and offset + head_size <= len(blob), "Bad version record")
            visited.add(offset)
            count = struct.unpack_from("<H", blob, offset + (2 if head_size == 16 else 6))[0]
            require(count <= len(blob) // aux_size, "Version auxiliary count exceeds bound")
            aux, following = struct.unpack_from("<II", blob, offset + head_size - 8)
            if head_size == 16:
                file_name = struct.unpack_from("<I", blob, offset + 4)[0]
                reject_reference(file_name)
            require(not count or aux >= head_size, "Version auxiliary overlaps header")
            cursor, aux_seen = offset + aux, set()
            for position in range(count):
                require(
                    cursor not in aux_seen and cursor + aux_size <= len(blob),
                    "Bad version auxiliary",
                )
                aux_seen.add(cursor)
                name = struct.unpack_from("<I", blob, cursor + name_field)[0]
                reject_reference(name)
                next_aux = struct.unpack_from("<I", blob, cursor + aux_size - 4)[0]
                if position + 1 < count:
                    require(next_aux >= aux_size, "Truncated version auxiliary chain")
                    cursor += next_aux
            if not following:
                break
            require(following >= head_size, "Overlapping version records")
            offset += following
    if ".dynsym" in layout["sections"]:
        row = layout["sections"][".dynsym"][1]
        require(
            row[9] == 24 and row[5] % 24 == 0 and row[6] == layout["sections"][".dynstr"][0],
            "Unexpected dynamic symbol layout",
        )
        for offset in range(0, row[5], 24):
            value = struct.unpack_from("<I", data, row[4] + offset)[0]
            reject_reference(value)
    replacement = b"$ORIGIN\0"
    require(width >= len(replacement), "RUNPATH slot too short")
    result = data[:start] + replacement + b"\0" * (width - len(replacement)) + data[end:]
    updated = elf(result)
    require(
        result[:start] == data[:start] and result[end:] == data[end:],
        "Bytes outside RUNPATH changed",
    )
    require(
        updated["identities"] == layout["identities"]
        and updated["needed"] == layout["needed"]
        and updated["sonames"] == layout["sonames"],
        "ELF code/symbol/dynamic identities changed",
    )
    require(updated["paths"] == [(29, index, "$ORIGIN", 8)], "Patched runtime path differs")
    return (
        result,
        {
            "offset": start,
            "bytes": width,
            "old": text,
            "new": "$ORIGIN",
            "every_other_byte_identical": True,
        },
        layout,
    )


def census_records(census_record, original_record, build):
    census = load_record(census_record)
    original = load_record(original_record)
    require(
        census.get("schema") == "qat_elf_closure_probe_v1"
        and not census["missing_project_needed"]
        and census["protected_seven_match"]
        and census["all_artifacts_unchanged"],
        "Closure census incomplete",
    )
    require(
        original.get("schema") == "qat_elf_metadata_probe_v1"
        and original["protected_seven_match"]
        and original["all_eight_unchanged"],
        "Original eight census incomplete",
    )
    require(
        8 <= len(census["artifacts"]) <= 24 and len(original["artifacts"]) == 8,
        "Census size differs",
    )
    source_bin = (build / "bin").resolve(strict=True)
    records = {}
    for item in census["artifacts"]:
        path = Path(item["canonical"])
        require(
            path.parent == source_bin and path.is_file() and str(path) not in records,
            "Census source escapes or duplicates",
        )
        require(SHA.fullmatch(item["sha256"]), "Census source SHA missing")
        paths = item["dynamic_paths"]
        require(
            not paths
            or (
                len(paths) == 1
                and paths[0]["kind"] == "RUNPATH"
                and paths[0]["raw"] == str(source_bin) + ":"
            ),
            "Census has unknown runtime path",
        )
        records[str(path)] = item
    for item in original["artifacts"]:
        require(
            item["canonical"] in records and records[item["canonical"]]["sha256"] == item["sha256"],
            "Original eight artifact ancestry differs",
        )
    return census, records


def verify_package(manifest_path, expected_sha, checkout, build, parent, native):
    require(
        isinstance(expected_sha, str) and SHA.fullmatch(expected_sha),
        "Runtime package requires SHA",
    )
    manifest_record = record(manifest_path)
    require(manifest_record["sha256"] == expected_sha, "Runtime package manifest SHA differs")
    m = load_record(manifest_record)
    directory = Path(m["directory"]).resolve(strict=True)
    require(not directory.is_relative_to(build), "Runtime package overlaps original build")
    require(
        Path(manifest_path).resolve().parent == directory
        and m["schema"] == "qat_clean_runtime_package_v1",
        "Runtime package location/schema differs",
    )
    require(
        m["source"]
        == {
            "checkout": str(checkout),
            "build_directory": str(build),
            "parent_commit": parent,
            "native_commit": native,
        },
        "Runtime package source differs",
    )
    require(m["packager"] == record(__file__), "Runtime package tool differs")
    census, source_records = census_records(m["closure_census"], m["original_census"], build)
    require(len(m["files"]) == len(source_records), "Runtime package closure file count differs")
    copies, original_files = {}, []
    for item in m["files"]:
        source = Path(item["source"]["path"])
        target = directory / source.name
        require(
            not target.samefile(source) and target.stat().st_nlink == 1,
            "Runtime copy shares original inode",
        )
        require(
            str(source) in source_records
            and item["source"] == record(source)
            and item["source"]["sha256"] == source_records[str(source)]["sha256"],
            "Original ELF bytes differ",
        )
        require(
            not target.is_symlink() and item["copy"] == record(target),
            "Runtime copy bytes/path differ",
        )
        paths = source_records[str(source)]["dynamic_paths"]
        expected, patch, layout = transform(read_bytes(source), paths[0]["raw"] if paths else None)
        require(
            read_bytes(target) == expected
            and item["patch"] == patch
            and item["identities"] == layout["identities"],
            "Runtime copy changed bytes beyond proven RUNPATH slot",
        )
        require(
            item["needed"] == layout["needed"] and item["sonames"] == layout["sonames"],
            "Runtime dynamic closure differs",
        )
        require(source.name not in copies, "Duplicate runtime file")
        copies[source.name] = item
        original_files.append(item["source"])
    require(m["aliases"] == census["aliases"], "Runtime alias ancestry differs")
    for name, destination in m["aliases"].items():
        require(
            Path(name).name == name and destination in copies and name not in copies,
            "Bad runtime alias",
        )
        require(
            (build / "bin" / name).resolve(strict=True) == build / "bin" / destination,
            "Original alias changed",
        )
        require(
            (directory / name).is_symlink() and os.readlink(directory / name) == destination,
            "Runtime alias differs",
        )
    names = set(copies) | set(m["aliases"])
    for item in copies.values():
        for needed in item["needed"]:
            if (build / "bin" / needed).exists() or re.match(r"lib(?:llama|ggml|mtmd)", needed):
                require(needed in names, "Runtime project dependency closure missing")
    require(
        {p.name for p in directory.iterdir()} == names | {Path(manifest_path).name},
        "Unexpected runtime package member",
    )
    return {
        "manifest": manifest_record,
        "directory": str(directory),
        "files": m["files"],
        "aliases": m["aliases"],
        "original_files": original_files,
        "evidence": [m["closure_census"], m["original_census"], m["packager"]],
        "requires_fresh_native_validation": True,
    }


def prepare(checkout, build, parent, native, census_record, original_record, output):
    checkout, build = Path(checkout).resolve(strict=True), Path(build).resolve(strict=True)
    output = Path(output).absolute()
    require(
        not output.exists() and not output.is_relative_to(build),
        "Package output reused or inside original build",
    )
    census, records = census_records(census_record, original_record, build)
    output.mkdir(parents=True, exist_ok=False)
    files = []
    try:
        for spelling, item in records.items():
            source = Path(spelling)
            data = read_bytes(source)
            require(sha(data) == item["sha256"], "Census source changed before copy")
            paths = item["dynamic_paths"]
            copied, patch, layout = transform(data, paths[0]["raw"] if paths else None)
            target = output / source.name
            with target.open("xb") as f:
                f.write(copied)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(target, source.stat().st_mode & 0o555)
            os.utime(target, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns))
            files.append(
                dict(
                    source={"path": spelling, "sha256": item["sha256"]},
                    copy=record(target),
                    patch=patch,
                    identities=layout["identities"],
                    needed=layout["needed"],
                    sonames=layout["sonames"],
                )
            )
        for name, target in census["aliases"].items():
            require(Path(name).name == name and Path(target).name == target, "Bad census alias")
            require(target in {Path(p).name for p in records}, "Census alias outside closure")
            os.symlink(target, output / name)
        m = dict(
            schema="qat_clean_runtime_package_v1",
            directory=str(output),
            source=dict(
                checkout=str(checkout),
                build_directory=str(build),
                parent_commit=parent,
                native_commit=native,
            ),
            closure_census=census_record,
            original_census=original_record,
            packager=record(__file__),
            files=files,
            aliases=census["aliases"],
            scope="CPU copy; RUNPATH slot only; no native validation",
            requires_fresh_native_validation=True,
            readiness_granted=False,
            hardware_measured=False,
            optimizer_updates=0,
        )
        manifest = output / "package-manifest.json"
        with manifest.open("x") as f:
            json.dump(m, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        proof = verify_package(
            manifest, record(manifest)["sha256"], checkout, build, parent, native
        )
        os.chmod(manifest, 0o444)
        os.chmod(output, 0o555)
        return proof
    finally:
        require(
            all(record(p)["sha256"] == item["sha256"] for p, item in records.items()),
            "Original ELF changed during packaging",
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkout", type=Path, required=True)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--expected-parent-commit", required=True)
    p.add_argument("--expected-native-commit", required=True)
    p.add_argument("--closure-census", type=Path, required=True)
    p.add_argument("--closure-census-sha256", required=True)
    p.add_argument("--original-census", type=Path, required=True)
    p.add_argument("--original-census-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    proof = prepare(
        a.checkout,
        a.build_dir,
        a.expected_parent_commit,
        a.expected_native_commit,
        {"path": str(a.closure_census.resolve(strict=True)), "sha256": a.closure_census_sha256},
        {"path": str(a.original_census.resolve(strict=True)), "sha256": a.original_census_sha256},
        a.output,
    )
    print(json.dumps(proof, sort_keys=True))


if __name__ == "__main__":
    main()
