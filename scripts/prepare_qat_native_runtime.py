#!/usr/bin/env python3
"""Inspect an explicit current CUDA server build on CPU; grant no readiness.

Only Git, readelf and ldd are invoked, using argv and a clean loader environment.
No server, compiler, build, accelerator, model or dataset is executed/opened.
The frozen continuous runtime builder remains a separate authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

LIBRARY = re.compile(r"lib(?:llama|ggml)[\w.-]*\.so(?:\.[\w.-]+)?$")
COMMIT = re.compile(r"[0-9a-f]{40}$")
SHA256 = re.compile(r"[0-9a-f]{64}$")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def run_checked(argv):
    """Read-only tools only. Never interpolate a shell or inherit loader overrides."""
    if argv[0] not in {"git", "ldd", "readelf"}:
        raise ValueError("only Git/ldd/readelf inspection is permitted")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LD_", "DYLD_"))}
    env["LC_ALL"] = "C"
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=True, env=env)
    if len(result.stdout) + len(result.stderr) > 4 * 1024**2:
        raise ValueError("inspection output exceeds its bound")
    return result.stdout


def inside(path, directory, name):
    try:
        path = Path(path).resolve(strict=True)
    except OSError as error:
        raise ValueError(name + " is absent or cannot be resolved") from error
    if not path.is_relative_to(directory):
        raise ValueError(name + " resolves outside the intended build")
    return path


def parse_cache(text):
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([^#/:][^:=]*):[^=]+=(.*)", line)
        if match:
            key, value = match.groups()
            if key in result:
                raise ValueError("duplicate CMake cache field: " + key)
            result[key] = value
    return result


def parse_compiler(text, language):
    result = {}
    for match in re.finditer(r"set\((CMAKE_" + language + r"_[A-Z_]+)\s+\"([^\"]*)\"\)", text):
        key, value = match.groups()
        if key in result:
            raise ValueError("duplicate compiler record: " + key)
        result[key] = value
    keys = [
        "CMAKE_" + language + suffix
        for suffix in ("_COMPILER", "_COMPILER_ID", "_COMPILER_VERSION")
    ]
    if any(not result.get(k) for k in keys):
        raise ValueError("CMake " + language + " compiler identity is incomplete")
    if not Path(result[keys[0]]).is_absolute():
        raise ValueError("actual compiler path must be absolute")
    return {"path": result[keys[0]], "id": result[keys[1]], "version": result[keys[2]]}


def validate_facts(facts, expected_parent, expected_native, expected_arch, expected_code_arch):
    """Pure validation of collected CPU facts; no imports, files or commands."""
    if not COMMIT.fullmatch(expected_parent) or not COMMIT.fullmatch(expected_native):
        raise ValueError("caller must pin full lowercase parent/native commits")
    if not re.fullmatch(r"\d+[a-z]?", expected_arch) or not re.fullmatch(
        r"\d+[a-z]?", expected_code_arch
    ):
        raise ValueError("caller must pin exact CUDA cache/code architectures")
    if facts["parent_commit"] != expected_parent or facts["native_commit"] != expected_native:
        raise ValueError("checkout differs from caller-pinned source commits")
    if facts["parent_native_gitlink"] != expected_native:
        raise ValueError("parent gitlink differs from native source")
    if facts["parent_status"] or facts["native_status"]:
        raise ValueError("relevant source files are dirty or untracked")
    build_commit = facts["build_commit"]
    if not re.fullmatch(r"[0-9a-f]{7,40}", build_commit) or not expected_native.startswith(
        build_commit
    ):
        raise ValueError("generated native build-info belongs to stale source")
    cache = facts["cache"]
    if (
        cache.get("GGML_CUDA") not in {"ON", "1", "TRUE"}
        or cache.get("CMAKE_BUILD_TYPE") != "Release"
    ):
        raise ValueError("explicit Release/CUDA build required")
    if cache.get("CMAKE_CUDA_ARCHITECTURES") != expected_arch:
        raise ValueError("CUDA architecture differs from caller-pinned cache")
    if cache.get("CMAKE_HOME_DIRECTORY") != facts["native_directory"]:
        raise ValueError("CMake build points at another native source tree")
    for language in ("C", "CXX", "CUDA"):
        compiler = facts["compilers"][language]
        if compiler["path"] != cache.get("CMAKE_" + language + "_COMPILER"):
            raise ValueError("compiler metadata differs from CMake cache")
    if facts["compilers"]["CUDA"]["id"] != "NVIDIA":
        raise ValueError("actual NVIDIA CUDA compiler record required")
    args = facts["w1a1_arguments"]
    if args[0] != facts["compilers"]["CUDA"]["path"]:
        raise ValueError("w1a1 compile command uses a different compiler")
    forwarded = {
        "-Xcompiler",
        "--compiler-options",
        "-Xptxas",
        "--ptxas-options",
        "-Xlinker",
        "--linker-options",
        "-Xcudafe",
        "--cudafe-options",
    }
    device_args = []
    index = 0
    while index < len(args):
        if args[index] in forwarded:
            index += 2
        else:
            device_args.append(args[index])
            index += 1
    args = device_args
    fast = [i for i, arg in enumerate(args) if arg in ("-use_fast_math", "--use_fast_math")]
    ftz = [(i, arg) for i, arg in enumerate(args) if re.match(r"--?ftz(?:=|$)", arg)]
    if (
        not fast
        or not ftz
        or ftz[-1][1] not in {"--ftz=false", "-ftz=false"}
        or ftz[-1][0] <= max(fast)
    ):
        raise ValueError("w1a1 must end with --ftz=false after fast_math")
    generated = " ".join(args)
    codes = set(re.findall(r"(?:compute|sm)_([0-9]+[a-z]?)", generated))
    if codes != {expected_code_arch} or "sm_" + expected_code_arch not in generated:
        raise ValueError("w1a1 generated CUDA code architecture differs")
    if not facts["libraries"] or not any(
        Path(r["path"]).name.startswith("libggml-cuda.so") for r in facts["libraries"]
    ):
        raise ValueError("current CUDA backend library is required")
    if not any(Path(r["path"]).name.startswith("libllama.so") for r in facts["libraries"]):
        raise ValueError("current llama library is required")
    return True


def _compiler_file(build, language):
    matches = list((build / "CMakeFiles").glob("*/CMake" + language + "Compiler.cmake"))
    if len(matches) != 1:
        raise ValueError("one actual CMake " + language + " compiler record required")
    return matches[0]


def _arguments(entry, build, response_records):
    directory = inside(entry["directory"], build, "compile command working directory")
    args = entry.get("arguments")
    if args is None:
        args = shlex.split(entry["command"])
    if not isinstance(args, list) or not args or any(not isinstance(v, str) for v in args):
        raise ValueError("invalid compile arguments")
    # NVCC options files may override flags: inspect them rather than trusting
    # a visible --ftz=false outside the actual expanded command.
    expanded = []

    def expand(values, depth=0):
        if depth > 4:
            raise ValueError("recursive compile response file exceeds bound")
        index = 0
        while index < len(values):
            token = values[index]
            filename = None
            if token in ("--options-file", "-optf"):
                index += 1
                if index == len(values):
                    raise ValueError("missing compile response filename")
                filename = values[index]
            elif token.startswith(("--options-file=", "-optf=")):
                filename = token.split("=", 1)[1]
            elif token.startswith("@"):
                filename = token[1:]
            if filename is not None:
                # NVCC accepts comma-separated files. Spaces are shlex-quoted.
                for item in filename.split(","):
                    response = inside(directory / item, build, "response file")
                    response_records[str(response)] = record(response)
                    expand(shlex.split(response.read_text()), depth + 1)
            else:
                expanded.append(token)
            index += 1

    expand(args)
    return expanded


def _existing_build_proof(path, expected_sha, checkout, build, parent, native):
    """Caller-approved provenance for a commands-file-only timestamp change.

    No historical object hash is inferred. The exact known CUDA library bytes,
    historical command and passed report provide the existing-build authority.
    Source/flags/response/object/backend freshness remain separate requirements.
    """
    if not isinstance(expected_sha, str) or not SHA256.fullmatch(expected_sha):
        raise ValueError("existing build proof requires an explicit SHA-256")
    proof_record = record(path)
    if proof_record["sha256"] != expected_sha:
        raise ValueError("existing build proof SHA-256 differs")

    def load(item, limit):
        if not isinstance(item, dict) or not SHA256.fullmatch(item.get("sha256", "")):
            raise ValueError("existing build evidence SHA-256 is missing")
        actual = record(item["path"])
        if actual != item or Path(actual["path"]).stat().st_size > limit:
            raise ValueError("existing build evidence differs or exceeds bound")
        return json.loads(Path(actual["path"]).read_text())

    proof = load(proof_record, 64 * 1024)
    if (
        proof.get("schema") != "qat_existing_cuda_build_proof_v1"
        or proof.get("checkout") != str(checkout)
        or proof.get("build_directory") != str(build)
    ):
        raise ValueError("existing build proof refers to another checkout/build")
    provenance = load(proof["provenance"], 2 * 1024**2)
    report = load(proof["native_report"], 16 * 1024**2)
    validation = load(proof["native_validation"], 64 * 1024)
    if provenance.get("source") != {"parent": parent, "native": native}:
        raise ValueError("existing build provenance source differs")
    old_build = provenance["build"]
    expected = {old_build["executable"]: old_build["sha256"]}
    shared = provenance["shared_objects"]
    if len(shared) != 6 or len({r["path"] for r in shared}) != 6:
        raise ValueError("existing build provenance must bind six unique libraries")
    expected.update({r["path"]: r["sha256"] for r in shared})
    if proof.get("runtime_files") != expected or len(expected) != 7:
        raise ValueError("existing build proof must bind all seven provenance artifacts")
    names = {Path(p).name.split(".so")[0] for p in expected if ".so" in Path(p).name}
    if names != {
        "libggml-cuda",
        "libggml-base",
        "libggml",
        "libggml-cpu",
        "libllama",
        "libllama-common",
    }:
        raise ValueError("existing build proof library set differs")
    for artifact, digest in expected.items():
        resolved = inside(artifact, build / "bin", "existing build artifact")
        if str(resolved) != artifact or not SHA256.fullmatch(digest) or sha256(resolved) != digest:
            raise ValueError("existing build runtime artifact differs")
    executable = Path(old_build["executable"])
    if executable.name != "test-eagle3-learned":
        raise ValueError("existing build fixture executable differs")
    if (
        report.get("status") != "passed"
        or report.get("input_scope") != "synthetic_operator"
        or report.get("requested_backend") != "CUDA"
        or report.get("backend", {}).get("registration") != "CUDA"
        or not isinstance(report.get("runtime", {}).get("build_commit"), str)
        or not re.fullmatch(r"[0-9a-f]{7,40}", report["runtime"]["build_commit"])
        or not native.startswith(report["runtime"]["build_commit"])
        or not isinstance(report.get("command"), list)
        or not report["command"]
        or report["command"][0] != str(executable)
        or validation.get("status") != "passed"
        or validation.get("report_sha256") != proof["native_report"]["sha256"]
        or validation.get("build_commit") != report["runtime"]["build_commit"]
    ):
        raise ValueError("existing build passed native report differs")
    command = old_build.get("w1a1_compile_command")
    if not isinstance(command, str) or not command:
        raise ValueError("existing build historical CUDA command missing")
    return {
        "proof": proof_record,
        "provenance": proof["provenance"],
        "native_report": proof["native_report"],
        "native_validation": proof["native_validation"],
        "runtime_files": expected,
        "historical_arguments": shlex.split(command),
    }


def _object(entry, args, build):
    positions = [i for i, value in enumerate(args) if value == "-o"]
    if len(positions) != 1 or positions[0] + 1 == len(args):
        raise ValueError("compile command must bind one actual object output")
    path = inside(Path(entry["directory"]) / args[positions[0] + 1], build, "compiled object")
    if path.stat().st_size < 1:
        raise ValueError("compiled object is empty")
    return path


def _dynamic_paths(text, object_path, build, *, required=False, allowed_roots=()):
    fields = re.findall(r"\((?:RUNPATH|RPATH)\).*?\[([^\]]*)\]", text)
    if not fields and required:
        raise ValueError("missing ELF RUNPATH/RPATH: " + str(object_path))
    result = []
    for field in fields:
        for value in field.split(":"):
            if not value:
                raise ValueError("empty runtime search path is not admissible")
            value = value.replace("${ORIGIN}", str(object_path.parent)).replace(
                "$ORIGIN", str(object_path.parent)
            )
            if "$" in value or not Path(value).is_absolute():
                raise ValueError("unsupported relative runtime search path")
            try:
                resolved = Path(value).resolve(strict=True)
            except OSError as error:
                raise ValueError("RUNPATH/RPATH directory is absent") from error
            if not resolved.is_relative_to(build) and resolved not in allowed_roots:
                raise ValueError("RUNPATH/RPATH resolves outside build and explicit runtime roots")
            result.append(str(resolved))
    return result


def _dependencies(
    text, bin_dir, inventory, *, package_names=(), original_build=None, allowed_roots=()
):
    if "not found" in text:
        raise ValueError("runtime dependency is missing")
    result = []
    for line in text.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(.+?)\s+\(0x[0-9a-fA-F]+\)", line)
        if match:
            name, raw = match.groups()
        else:
            direct = re.match(r"\s*(/.+?)\s+\(0x[0-9a-fA-F]+\)", line)
            if not direct:
                continue
            raw = direct.group(1)
            name = Path(raw).name
        path = Path(raw)
        if not path.is_absolute():
            raise ValueError("ldd dependency has no absolute path")
        resolved = path.resolve(strict=True)
        project = LIBRARY.fullmatch(name) or LIBRARY.fullmatch(resolved.name)
        if package_names:
            project = project or name in package_names or resolved.name in package_names
            project = project or resolved.is_relative_to(bin_dir)
            if original_build is not None and resolved.is_relative_to(original_build):
                raise ValueError("packaged dependency resolves to the original build")
        if project:
            inside(resolved, bin_dir, "resolved llama/ggml dependency")
            if name.split(".so")[0] != resolved.name.split(".so")[0]:
                raise ValueError("resolved project dependency basename differs")
            if str(resolved) not in inventory:
                raise ValueError("resolved project dependency is absent from inventory")
        elif package_names:
            system = (Path("/usr/lib/x86_64-linux-gnu"), Path("/usr/lib/wsl/lib"))
            roots = tuple(Path(p).resolve() for p in (*allowed_roots, *system))
            if not any(resolved.is_relative_to(root) for root in roots):
                raise ValueError(
                    "packaged dependency resolves outside approved system/toolkit roots"
                )
        result.append({"name": name, "path": str(resolved)})
    if not result:
        raise ValueError("ldd returned no dynamic dependency records")
    return result


def _packaging_tool():
    spec = importlib.util.spec_from_file_location(
        "qat_runtime_package", Path(__file__).with_name("prepare_qat_runtime_package.py")
    )
    packaging = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packaging)
    return packaging


def inspect_runtime(
    checkout,
    build_dir,
    expected_parent,
    expected_native,
    *,
    expected_arch="120",
    expected_code_arch="120a",
    runner=run_checked,
    allowed_runtime_roots=(),
    existing_build_proof=None,
    existing_build_proof_sha256=None,
    runtime_package=None,
    runtime_package_sha256=None,
):
    """Read explicit source/build artifacts and inspect ELF; never run a server."""
    if not COMMIT.fullmatch(expected_parent) or not COMMIT.fullmatch(expected_native):
        raise ValueError("caller must pin full lowercase parent/native commits")
    checkout = Path(checkout).resolve(strict=True)
    native = (checkout / "third_party/llama.cpp").resolve(strict=True)
    build = Path(build_dir).resolve(strict=True)
    bin_dir = (build / "bin").resolve(strict=True)
    server = inside(bin_dir / "llama-server", bin_dir, "server")
    original_server = server
    allowed_roots = tuple(sorted({Path(p).resolve(strict=True) for p in allowed_runtime_roots}))
    if any(not p.is_dir() for p in allowed_roots):
        raise ValueError("allowed runtime roots must be explicit existing directories")
    if (existing_build_proof is None) != (existing_build_proof_sha256 is None):
        raise ValueError("existing build proof and SHA-256 must be supplied together")
    known_build = None
    if existing_build_proof is not None:
        known_build = _existing_build_proof(
            existing_build_proof,
            existing_build_proof_sha256,
            checkout,
            build,
            expected_parent,
            expected_native,
        )
    regeneration = None
    if (runtime_package is None) != (runtime_package_sha256 is None):
        raise ValueError("runtime package and SHA-256 must be supplied together")
    package = None
    if runtime_package is not None:
        if known_build is None:
            raise ValueError("runtime package requires existing original build proof")
        packaging = _packaging_tool()
        package = packaging.verify_package(
            runtime_package,
            runtime_package_sha256,
            checkout,
            build,
            expected_parent,
            expected_native,
        )
        bin_dir = Path(package["directory"])
        server = inside(bin_dir / "llama-server", bin_dir, "packaged server")

    def git(directory, *args):
        return runner(["git", "-C", str(directory), *args]).strip()

    link = git(checkout, "ls-tree", "HEAD", "third_party/llama.cpp")
    if not re.fullmatch(r"160000 commit [0-9a-f]{40}\tthird_party/llama\.cpp", link):
        raise ValueError("parent must bind a native submodule gitlink")
    server_record = record(server)
    facts = {
        "parent_commit": git(checkout, "rev-parse", "HEAD"),
        "native_commit": git(native, "rev-parse", "HEAD"),
        "parent_native_gitlink": link.split()[2],
        "parent_status": git(
            checkout,
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            "scripts",
            "src",
            "configs",
            "tests",
            ".gitmodules",
            "third_party/llama.cpp",
        ),
        "native_status": git(native, "status", "--porcelain", "--untracked-files=normal"),
        "native_directory": str(native),
        "cache": parse_cache((build / "CMakeCache.txt").read_text()),
        "compilers": {},
    }
    artifacts = [record(build / "CMakeCache.txt"), record(build / "compile_commands.json")]
    for language in ("C", "CXX", "CUDA"):
        compiler_file = _compiler_file(build, language)
        facts["compilers"][language] = parse_compiler(compiler_file.read_text(), language)
        artifacts.append(record(compiler_file))
        artifacts.append(record(facts["compilers"][language]["path"]))
    info = build / "common/build-info.cpp"
    matches = re.findall(r'LLAMA_COMMIT\s*=\s*"([^"]+)"', info.read_text())
    if len(matches) != 1:
        raise ValueError("generated native build-info commit is missing or ambiguous")
    facts["build_commit"] = matches[0]
    artifacts.append(record(info))
    flags_source = native / "ggml/src/ggml-cuda/CMakeLists.txt"
    artifacts.append(record(flags_source))
    commands_path = build / "compile_commands.json"
    commands = json.loads(commands_path.read_text())
    if not isinstance(commands, list):
        raise ValueError("compile commands must be an array")
    required = {
        native / "ggml/src/ggml-cuda/w1a1.cu": "w1a1",
        native / "tools/server/main.cpp": "server",
        native / "tools/server/server.cpp": "server_impl",
        info: "build_info",
    }
    compiled = {}
    response_records = {}
    for source, label in required.items():
        entries = [
            e
            for e in commands
            if isinstance(e, dict)
            and Path(e.get("directory", "."), e.get("file", "")).resolve() == source
        ]
        if len(entries) != 1:
            raise ValueError("one current-source compile entry required for " + label)
        entry = entries[0]
        args = _arguments(entry, build, response_records)
        if "-c" not in args or args.index("-c") + 1 >= len(args):
            raise ValueError("compile command source is missing: " + label)
        operand = Path(entry["directory"]) / args[args.index("-c") + 1]
        if operand.resolve() != source:
            raise ValueError("compile command source differs: " + label)
        obj = _object(entry, args, build)
        if obj.stat().st_mtime_ns < source.stat().st_mtime_ns:
            raise ValueError("compiled object predates its current source: " + label)
        if label != "w1a1" and args[0] != facts["compilers"]["CXX"]["path"]:
            raise ValueError("server/build-info compile command uses a different compiler")
        if label == "w1a1":
            object_time = obj.stat().st_mtime_ns
            if object_time < flags_source.stat().st_mtime_ns:
                raise ValueError("w1a1 object predates current CMake flags/compile commands")
            if object_time < commands_path.stat().st_mtime_ns:
                raw_args = entry.get("arguments")
                if raw_args is None:
                    raw_args = shlex.split(entry["command"])
                if known_build is None or raw_args != known_build["historical_arguments"]:
                    raise ValueError("w1a1 object predates current CMake flags/compile commands")
                if any(Path(p).stat().st_mtime_ns > object_time for p in response_records):
                    raise ValueError("w1a1 response file is newer than inspected object")
                regeneration = {
                    "scope": "global compile_commands timestamp only",
                    "object_mtime_ns": object_time,
                    "commands_mtime_ns": commands_path.stat().st_mtime_ns,
                    "evidence": known_build,
                    "historical_object_hash_claimed": False,
                }
        compiled[label] = {"source": record(source), "object": record(obj), "arguments": args}
    facts["w1a1_arguments"] = compiled["w1a1"]["arguments"]
    libraries = {}
    aliases = []
    library_names = set()
    if package:
        library_names = {
            Path(item["copy"]["path"]).name
            for item in package["files"]
            if ".so" in Path(item["copy"]["path"]).name
        }
    for path in sorted(bin_dir.iterdir()):
        is_library = LIBRARY.fullmatch(path.name)
        if package:
            is_library = (
                path.name in library_names or package["aliases"].get(path.name) in library_names
            )
        if not is_library:
            continue
        resolved = inside(path, bin_dir, "library symlink")
        if not resolved.is_file():
            raise ValueError("runtime library is not a regular file")
        libraries[str(resolved)] = record(resolved)
        aliases.append({"path": str(path), "resolved_path": str(resolved)})
    facts["libraries"] = list(libraries.values())
    validate_facts(facts, expected_parent, expected_native, expected_arch, expected_code_arch)
    if libraries:
        cuda = next(
            Path(r["path"])
            for r in facts["libraries"]
            if Path(r["path"]).name.startswith("libggml-cuda.so")
        )
        if package:
            cuda = build / "bin" / cuda.name
        if cuda.stat().st_mtime_ns < Path(compiled["w1a1"]["object"]["path"]).stat().st_mtime_ns:
            raise ValueError("CUDA backend predates the inspected w1a1 object")
    if original_server.stat().st_mtime_ns < max(
        Path(compiled[k]["object"]["path"]).stat().st_mtime_ns
        for k in ("server", "server_impl", "build_info")
    ):
        raise ValueError("server predates inspected server/build-info objects")
    observations = []
    compiled_commit_artifacts = []
    build_info_observations = []
    runtime_dependencies = {}
    for path in [server, *(Path(p) for p in libraries)]:
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise ValueError("runtime artifact must be an ELF file")
        dynamic = runner(["readelf", "-d", str(path)])
        search = _dynamic_paths(
            dynamic,
            path,
            bin_dir if package else build,
            required=path == server,
            allowed_roots=allowed_roots,
        )
        ldd = runner(["ldd", str(path)])
        package_names = ()
        if package:
            package_names = tuple(library_names | set(package["aliases"]))
        dependencies = _dependencies(
            ldd,
            bin_dir,
            libraries,
            package_names=package_names,
            original_build=build if package else None,
            allowed_roots=allowed_roots,
        )
        for dependency in dependencies:
            dependency_path = Path(dependency["path"])
            if (package and str(dependency_path) not in libraries) or any(
                dependency_path.is_relative_to(root) for root in allowed_roots
            ):
                runtime_dependencies.setdefault(str(dependency_path), record(dependency_path))
        if path == server or path.name.startswith("libllama-common.so"):
            rodata = runner(["readelf", "--string-dump=.rodata", str(path)])
            if re.search(
                r"\]\s+" + re.escape(facts["build_commit"]) + r"\s*$", rodata, re.MULTILINE
            ):
                compiled_commit_artifacts.append(record(path))
                build_info_observations.append(
                    {
                        "command": ["readelf", "--string-dump=.rodata", str(path)],
                        "output_sha256": hashlib.sha256(rodata.encode()).hexdigest(),
                        "matched_commit": facts["build_commit"],
                    }
                )
        observations.append(
            {
                "artifact": record(path),
                "runtime_search_paths": search,
                "readelf_dynamic": dynamic,
                "ldd": ldd,
                "dependencies": dependencies,
            }
        )
    if not compiled_commit_artifacts:
        raise ValueError("compiled server/common artifact lacks the current build-info commit")
    # Recheck to reject files, source revisions or aliases replaced while inspecting.
    for item in (
        artifacts
        + list(runtime_dependencies.values())
        + list(response_records.values())
        + facts["libraries"]
        + [server_record]
        + [v[k] for v in compiled.values() for k in ("source", "object")]
        + (
            package["original_files"] + package["evidence"] + [package["manifest"]]
            if package
            else []
        )
        + (
            [known_build[k] for k in ("proof", "provenance", "native_report", "native_validation")]
            + [{"path": p, "sha256": h} for p, h in known_build["runtime_files"].items()]
            if known_build
            else []
        )
    ):
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("source/build/runtime file changed during inspection")
    if any(str(Path(a["path"]).resolve(strict=True)) != a["resolved_path"] for a in aliases):
        raise ValueError("runtime symlink changed during inspection")
    if package:
        packaging.verify_package(
            runtime_package,
            runtime_package_sha256,
            checkout,
            build,
            expected_parent,
            expected_native,
        )
    if (
        git(checkout, "rev-parse", "HEAD") != expected_parent
        or git(native, "rev-parse", "HEAD") != expected_native
    ):
        raise ValueError("source revision changed during inspection")
    if git(
        checkout,
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        "scripts",
        "src",
        "configs",
        "tests",
        ".gitmodules",
        "third_party/llama.cpp",
    ) or git(native, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("source files changed during inspection")
    return {
        "schema": "qat_current_native_server_build_v1",
        "inventory_tool": record(Path(__file__)),
        "parent_commit": expected_parent,
        "native_commit": expected_native,
        "checkout": str(checkout),
        "build_directory": str(build),
        "binary": server_record,
        "libraries": facts["libraries"],
        "library_aliases": aliases,
        "build_commit": facts["build_commit"],
        "compiled_commit_artifacts": compiled_commit_artifacts,
        "build_info_observations": build_info_observations,
        "cmake": facts["cache"],
        "compilers": facts["compilers"],
        "cuda_architecture": expected_arch,
        "cuda_code_architecture": expected_code_arch,
        "artifacts": artifacts + list(response_records.values()),
        "compiled": compiled,
        "commands_regeneration_proof": regeneration,
        "runtime_package": package,
        "dynamic_observations": observations,
        "allowed_runtime_roots": [str(p) for p in allowed_roots],
        "allowed_runtime_dependency_artifacts": list(runtime_dependencies.values()),
        "scope": "CPU static source/build/ELF inspection; not measured native or CUDA readiness",
        "training_eligible": False,
        "readiness_granted": False,
        "hardware_measured": False,
        "optimizer_updates": 0,
    }


def write_inventory(manifest, output):
    """Only publish to a unique directory, never replacing prior evidence."""
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError("runtime inventory output already exists")
    package = manifest.get("runtime_package")
    if package:
        _packaging_tool().verify_package(
            package["manifest"]["path"],
            package["manifest"]["sha256"],
            Path(manifest["checkout"]),
            Path(manifest["build_directory"]),
            manifest["parent_commit"],
            manifest["native_commit"],
        )
    for item in [
        manifest["binary"],
        *manifest["libraries"],
        *manifest.get("allowed_runtime_dependency_artifacts", []),
    ]:
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("runtime artifact changed before publication")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    runtime = {
        "schema": "qat_current_native_runtime_v1",
        "directory": str(Path(manifest["binary"]["path"]).parent),
        "immutable_manifest": record(manifest_path),
        "libraries": manifest["libraries"],
        "ld_library_path": str(Path(manifest["binary"]["path"]).parent),
    }
    with (output / "native-runtime.json").open("x") as stream:
        json.dump(runtime, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--expected-parent-commit", required=True)
    parser.add_argument("--expected-native-commit", required=True)
    parser.add_argument("--expected-architecture", default="120")
    parser.add_argument("--expected-code-architecture", default="120a")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-build-proof", type=Path)
    parser.add_argument("--existing-build-proof-sha256")
    parser.add_argument("--runtime-package", type=Path)
    parser.add_argument("--runtime-package-sha256")
    parser.add_argument(
        "--allowed-runtime-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Explicit existing toolkit/system RUNPATH directory (repeatable); "
            "project libraries still require this build/bin"
        ),
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be a new immutable inventory directory")
    manifest = inspect_runtime(
        args.checkout,
        args.build_dir,
        args.expected_parent_commit,
        args.expected_native_commit,
        expected_arch=args.expected_architecture,
        expected_code_arch=args.expected_code_architecture,
        allowed_runtime_roots=args.allowed_runtime_root,
        existing_build_proof=args.existing_build_proof,
        existing_build_proof_sha256=args.existing_build_proof_sha256,
        runtime_package=args.runtime_package,
        runtime_package_sha256=args.runtime_package_sha256,
    )
    runtime = write_inventory(manifest, args.output)
    print(
        json.dumps(
            {
                "native_runtime": runtime,
                "binary": manifest["binary"],
                "training_eligible": False,
                "readiness_granted": False,
                "hardware_measured": False,
                "optimizer_updates": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
