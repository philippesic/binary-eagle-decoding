"""Tiny CPU metadata/ELF fixtures; no compiler/server/model/CUDA execution."""

import contextlib
import copy
import importlib.util
import io
import json
import os
import shlex
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "runtime_inventory", ROOT / "scripts/prepare_qat_native_runtime.py"
)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)
PARENT, NATIVE = "a" * 40, "b" * 40


class Fixture:
    def __init__(self, root):
        self.checkout = root / "checkout"
        self.native = self.checkout / "third_party/llama.cpp"
        self.build = self.native / "build-new"
        self.bin = self.build / "bin"
        self.bin.mkdir(parents=True)
        self.calls = []
        self.git_dirty = ""
        self.parent = PARENT
        self.native_commit = NATIVE
        self.link = NATIVE
        self.runpath = str(self.bin)
        self.ldd_extra = ""
        self.rodata_commit = NATIVE[:9]
        self.compiler = {}
        for lang, name in [("C", "cc"), ("CXX", "c++"), ("CUDA", "nvcc")]:
            compiler = root / "toolchain" / name
            compiler.parent.mkdir(exist_ok=True)
            compiler.write_bytes(b"compiler fixture " + name.encode())
            self.compiler[lang] = str(compiler)
            metadata = self.build / "CMakeFiles/3.31.10" / ("CMake" + lang + "Compiler.cmake")
            metadata.parent.mkdir(parents=True, exist_ok=True)
            version = "13.1.115" if lang == "CUDA" else "15.2.0"
            metadata.write_text(
                f'set(CMAKE_{lang}_COMPILER "{compiler}")\n'
                f'set(CMAKE_{lang}_COMPILER_ID "{"NVIDIA" if lang == "CUDA" else "GNU"}")\n'
                f'set(CMAKE_{lang}_COMPILER_VERSION "{version}")\n'
            )
        self.cache = {
            "GGML_CUDA": "ON",
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_CUDA_ARCHITECTURES": "120",
            "CMAKE_HOME_DIRECTORY": str(self.native),
            **{"CMAKE_" + k + "_COMPILER": v for k, v in self.compiler.items()},
        }
        self.write_cache()
        self.info = self.build / "common/build-info.cpp"
        self.info.parent.mkdir(parents=True)
        self.info.write_text('char const * LLAMA_COMMIT = "' + NATIVE[:9] + '";\n')
        self.flags = self.native / "ggml/src/ggml-cuda/CMakeLists.txt"
        self.flags.parent.mkdir(parents=True)
        self.flags.write_text(
            'set_source_files_properties(w1a1.cu PROPERTIES COMPILE_OPTIONS "--ftz=false")\n'
        )
        self.entries = []
        for label, source in [
            ("w1a1", self.flags.parent / "w1a1.cu"),
            ("server", self.native / "tools/server/main.cpp"),
            ("server_impl", self.native / "tools/server/server.cpp"),
            ("build_info", self.info),
        ]:
            source.parent.mkdir(parents=True, exist_ok=True)
            if label != "build_info":
                source.write_bytes(label.encode())
            obj = self.build / (label + ".o")
            obj.write_bytes(b"object fixture " + label.encode())
            args = [self.compiler["CUDA" if label == "w1a1" else "CXX"]]
            if label == "w1a1":
                args += [
                    "-use_fast_math",
                    "--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]",
                    "--ftz=false",
                ]
            args += ["-c", str(source), "-o", str(obj)]
            self.entries.append(
                {"directory": str(self.build), "file": str(source), "arguments": args}
            )
        self.write_commands()
        for name in [
            "llama-server",
            "libllama.so.1",
            "libllama-common.so.1",
            "libggml-cuda.so.1",
            "libggml-base.so.1",
        ]:
            (self.bin / name).write_bytes(b"\x7fELF" + name.encode())
        (self.bin / "libggml-cuda.so").symlink_to("libggml-cuda.so.1")
        self.system = root / "system.so"
        self.system.write_bytes(b"system fixture")
        self.fresh_times()

    def write_cache(self):
        (self.build / "CMakeCache.txt").write_text(
            "\n".join(k + ":STRING=" + v for k, v in self.cache.items())
        )

    def write_commands(self):
        (self.build / "compile_commands.json").write_text(json.dumps(self.entries))

    def fresh_times(self):
        for path in self.checkout.rglob("*"):
            if path.is_file() and not path.is_symlink():
                tick = 3 if path.parent == self.bin else 2 if path.suffix == ".o" else 1
                os.utime(path, ns=(tick * 10**9, tick * 10**9))

    def runner(self, argv):
        self.calls.append(argv)
        if argv[0] == "git":
            native = argv[2] == str(self.native)
            if "rev-parse" in argv:
                return self.native_commit if native else self.parent
            if "ls-tree" in argv:
                return "160000 commit " + self.link + "\tthird_party/llama.cpp\n"
            if "status" in argv:
                return self.git_dirty
        if argv[0] == "readelf":
            if "--string-dump=.rodata" in argv:
                return "  [   20]  " + self.rodata_commit + "\n"
            return "0x01 (RUNPATH) Library runpath: [" + self.runpath + "]\n"
        if argv[0] == "ldd":
            return (
                "libllama.so.1 => "
                + str(self.bin / "libllama.so.1")
                + " (0x01)\nlibstdc++.so.6 => "
                + str(self.system)
                + " (0x02)\n"
                + self.ldd_extra
            )
        raise AssertionError("unexpected command: " + str(argv))

    def inspect(self, **kwargs):
        return tool.inspect_runtime(
            self.checkout, self.build, PARENT, NATIVE, runner=self.runner, **kwargs
        )


class NativeRuntimeInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.fixture = Fixture(self.root)

    def existing_build_proof(self):
        f = self.fixture
        for name in ("libggml.so.1", "libggml-cpu.so.1", "test-eagle3-learned"):
            (f.bin / name).write_bytes(b"\x7fELF" + name.encode())
        f.fresh_times()
        provenance_path = self.root / "original-build.json"
        provenance = {
            "source": {"parent": PARENT, "native": NATIVE},
            "build": {
                "executable": str(f.bin / "test-eagle3-learned"),
                "sha256": tool.sha256(f.bin / "test-eagle3-learned"),
                "w1a1_compile_command": shlex.join(f.entries[0]["arguments"]),
            },
            "shared_objects": [
                tool.record(p) for p in sorted(f.bin.iterdir()) if p.name.endswith(".so.1")
            ],
        }
        provenance_path.write_text(json.dumps(provenance))
        report_path = self.root / "passed-native.json"
        report_path.write_text(
            json.dumps(
                dict(
                    status="passed",
                    input_scope="synthetic_operator",
                    requested_backend="CUDA",
                    backend={"registration": "CUDA"},
                    runtime={"build_commit": NATIVE[:9]},
                    command=[str(f.bin / "test-eagle3-learned")],
                )
            )
        )
        validation_path = self.root / "native-validation.json"
        validation_path.write_text(
            json.dumps(
                dict(
                    status="passed", report_sha256=tool.sha256(report_path), build_commit=NATIVE[:9]
                )
            )
        )
        proof = dict(
            schema="qat_existing_cuda_build_proof_v1",
            checkout=str(f.checkout),
            build_directory=str(f.build),
            provenance=tool.record(provenance_path),
            native_report=tool.record(report_path),
            native_validation=tool.record(validation_path),
            runtime_files={r["path"]: r["sha256"] for r in provenance["shared_objects"]},
        )
        proof["runtime_files"][provenance["build"]["executable"]] = provenance["build"]["sha256"]
        path = self.root / "approved-proof.json"
        path.write_text(json.dumps(proof))
        os.utime(f.build / "compile_commands.json", ns=(4 * 10**9, 4 * 10**9))
        return dict(existing_build_proof=path, existing_build_proof_sha256=tool.sha256(path))

    def test_verified_package_selects_typed_closure_without_admitting_old_paths(self):
        kwargs = self.existing_build_proof()
        package_dir = self.root / "verified-package"
        package_dir.mkdir()
        for path in self.fixture.bin.iterdir():
            if not path.is_symlink():
                shutil.copyfile(path, package_dir / path.name)
        for name in ("libmtmd.so", "libllama-server-impl.so"):
            (package_dir / name).write_bytes(b"\x7fELF" + name.encode())
        manifest_path = package_dir / "package-manifest.json"
        manifest_path.write_text("{}")
        package = dict(
            directory=str(package_dir),
            manifest=tool.record(manifest_path),
            aliases={},
            files=[
                {"copy": tool.record(p)}
                for p in package_dir.iterdir()
                if p.name != "package-manifest.json"
            ],
            original_files=[],
            evidence=[],
            requires_fresh_native_validation=True,
        )
        fake = SimpleNamespace(verify_package=lambda *args: package)
        kwargs.update(
            runtime_package=manifest_path,
            runtime_package_sha256=tool.sha256(manifest_path),
            allowed_runtime_roots=[self.root],
        )

        def runner(argv):
            if argv[0] == "readelf" and "-d" in argv:
                return "0x01 (RUNPATH) Library runpath: [$ORIGIN]\n"
            if argv[0] == "ldd":
                return (
                    "libllama.so.1 => "
                    + str(package_dir / "libllama.so.1")
                    + " (0x01)\nlibstdc++.so.6 => "
                    + str(self.fixture.system)
                    + " (0x02)\n"
                )
            return self.fixture.runner(argv)

        with patch.object(tool, "_packaging_tool", return_value=fake):
            manifest = tool.inspect_runtime(
                self.fixture.checkout, self.fixture.build, PARENT, NATIVE, runner=runner, **kwargs
            )
        self.assertIn(str(package_dir / "libmtmd.so"), [r["path"] for r in manifest["libraries"]])
        self.assertIn(
            str(package_dir / "libllama-server-impl.so"), [r["path"] for r in manifest["libraries"]]
        )
        self.assertEqual(manifest["binary"]["path"], str(package_dir / "llama-server"))
        self.assertFalse(manifest["readiness_granted"])
        self.assertTrue(manifest["runtime_package"]["requires_fresh_native_validation"])
        old = self.fixture.bin / "libllama.so.1"
        inventory = {str(package_dir / "libllama.so.1"): tool.record(package_dir / "libllama.so.1")}
        with self.assertRaisesRegex(ValueError, "original build"):
            tool._dependencies(
                "libllama.so.1 => " + str(old) + " (0x01)",
                package_dir,
                inventory,
                package_names=("libllama.so.1",),
                original_build=self.fixture.build,
            )

    def test_package_dependencies_reject_unapproved_system_paths(self):
        with self.assertRaisesRegex(ValueError, "approved system/toolkit"):
            tool._dependencies(
                "libstdc++.so.6 => " + str(self.fixture.system) + " (0x01)",
                self.fixture.bin,
                {},
                package_names=("libmtmd.so",),
                original_build=self.fixture.build,
            )

    def test_package_cannot_mask_stale_original_server(self):
        kwargs = self.existing_build_proof()
        package_dir = self.root / "fresh-copy"
        package_dir.mkdir()
        for p in self.fixture.bin.iterdir():
            if not p.is_symlink():
                shutil.copyfile(p, package_dir / p.name)
        manifest_path = package_dir / "package-manifest.json"
        manifest_path.write_text("{}")
        os.utime(self.fixture.bin / "llama-server", ns=(0, 0))
        fake = SimpleNamespace(
            verify_package=lambda *args: dict(
                directory=str(package_dir),
                files=[
                    {"copy": tool.record(p)}
                    for p in package_dir.iterdir()
                    if p.name != "package-manifest.json"
                ],
                aliases={},
                original_files=[],
                evidence=[],
                manifest=tool.record(manifest_path),
            )
        )
        with patch.object(tool, "_packaging_tool", return_value=fake):
            with self.assertRaisesRegex(ValueError, "server predates inspected"):
                self.fixture.inspect(
                    runtime_package=manifest_path, runtime_package_sha256="0" * 64, **kwargs
                )

    def test_existing_proof_only_admits_regenerated_commands_timestamp(self):
        kwargs = self.existing_build_proof()
        with self.assertRaisesRegex(ValueError, "predates current CMake"):
            self.fixture.inspect()
        manifest = self.fixture.inspect(**kwargs)
        evidence = manifest["commands_regeneration_proof"]
        self.assertEqual(evidence["scope"], "global compile_commands timestamp only")
        self.assertFalse(evidence["historical_object_hash_claimed"])
        self.assertFalse(manifest["readiness_granted"])
        self.assertFalse(manifest["hardware_measured"])
        runtime = tool.write_inventory(manifest, self.root / "proved-output")
        self.assertEqual(
            runtime["immutable_manifest"]["sha256"],
            tool.sha256(self.root / "proved-output/manifest.json"),
        )

    def test_existing_proof_requires_sha_and_matching_evidence(self):
        kwargs = self.existing_build_proof()
        with self.assertRaisesRegex(ValueError, "supplied together"):
            self.fixture.inspect(existing_build_proof=kwargs["existing_build_proof"])
        with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
            self.fixture.inspect(**{**kwargs, "existing_build_proof_sha256": "0" * 64})
        (self.root / "original-build.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "evidence differs"):
            self.fixture.inspect(**kwargs)

    def test_existing_proof_rejects_changed_runtime_artifact(self):
        kwargs = self.existing_build_proof()
        (self.fixture.bin / "libggml-cpu.so.1").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "runtime artifact differs"):
            self.fixture.inspect(**kwargs)

    def test_existing_proof_rejects_changed_historical_command(self):
        kwargs = self.existing_build_proof()
        self.fixture.entries[0]["arguments"].insert(1, "-DNEW_SEMANTICS=1")
        self.fixture.write_commands()
        with self.assertRaisesRegex(ValueError, "predates current CMake"):
            self.fixture.inspect(**kwargs)

    def test_existing_proof_keeps_source_flags_and_backend_freshness(self):
        kwargs = self.existing_build_proof()
        for path, error in (
            (self.fixture.flags, "predates current CMake"),
            (Path(self.fixture.entries[0]["file"]), "predates its current source"),
            (self.fixture.bin / "libggml-cuda.so.1", "backend predates"),
        ):
            with self.subTest(path=path):
                old = path.stat().st_mtime_ns
                tick = 0 if path.parent == self.fixture.bin else 5 * 10**9
                os.utime(path, ns=(tick, tick))
                with self.assertRaisesRegex(ValueError, error):
                    self.fixture.inspect(**kwargs)
                os.utime(path, ns=(old, old))

    def test_existing_proof_rejects_new_response_file(self):
        response = self.fixture.build / "includes.rsp"
        response.write_text("-I" + str(self.fixture.native))
        self.fixture.entries[0]["arguments"][1:1] = ["--options-file", str(response)]
        self.fixture.write_commands()
        kwargs = self.existing_build_proof()
        self.fixture.inspect(**kwargs)
        os.utime(response, ns=(5 * 10**9, 5 * 10**9))
        with self.assertRaisesRegex(ValueError, "response file is newer"):
            self.fixture.inspect(**kwargs)

    def test_existing_proof_rechecks_authority_during_inspection(self):
        kwargs = self.existing_build_proof()

        def mutate(argv):
            result = self.fixture.runner(argv)
            if argv[0] == "ldd":
                (self.root / "native-validation.json").write_text("{}")
            return result

        with self.assertRaisesRegex(ValueError, "file changed during inspection"):
            tool.inspect_runtime(
                self.fixture.checkout, self.fixture.build, PARENT, NATIVE, runner=mutate, **kwargs
            )

    def test_existing_proof_rejects_wrong_build_and_report_validation(self):
        kwargs = self.existing_build_proof()
        path = kwargs["existing_build_proof"]
        proof = json.loads(path.read_text())
        proof["build_directory"] = str(self.root / "other")
        path.write_text(json.dumps(proof))
        with self.assertRaisesRegex(ValueError, "another checkout/build"):
            self.fixture.inspect(
                existing_build_proof=path, existing_build_proof_sha256=tool.sha256(path)
            )
        proof["build_directory"] = str(self.fixture.build)
        validation = self.root / "native-validation.json"
        validation.write_text(
            json.dumps(dict(status="passed", report_sha256="0" * 64, build_commit=NATIVE[:9]))
        )
        proof["native_validation"] = tool.record(validation)
        path.write_text(json.dumps(proof))
        with self.assertRaisesRegex(ValueError, "passed native report differs"):
            self.fixture.inspect(
                existing_build_proof=path, existing_build_proof_sha256=tool.sha256(path)
            )

    def test_current_inventory_hashes_contract_and_no_execution(self):
        manifest = self.fixture.inspect()
        runtime = tool.write_inventory(manifest, self.root / "output")
        self.assertEqual(manifest["native_commit"], NATIVE)
        self.assertFalse(manifest["training_eligible"])
        self.assertFalse(manifest["hardware_measured"])
        self.assertFalse(manifest["readiness_granted"])
        self.assertEqual(
            runtime["immutable_manifest"]["sha256"], tool.sha256(self.root / "output/manifest.json")
        )
        self.assertEqual(len(runtime["libraries"]), 4)
        self.assertTrue(all(Path(r["path"]).is_absolute() for r in runtime["libraries"]))
        self.assertTrue(all(call[0] in {"git", "readelf", "ldd"} for call in self.fixture.calls))
        self.assertNotIn("sources.json", [p.name for p in (self.root / "output").iterdir()])
        # Match collector.verify_native_revision's commit/string inventory without
        # importing Torch/provider or treating the fixture as actual readiness.
        saved = json.loads((self.root / "output/manifest.json").read_text())
        self.assertEqual(saved["binary"]["sha256"], tool.sha256(saved["binary"]["path"]))
        self.assertEqual(saved["native_commit"], NATIVE)

    def test_existing_consumers_verify_real_runtime_records_and_substitutions(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        sys.path.insert(0, str(ROOT / "src"))
        import collect_qat_native_evidence as collector
        import run_binary_head_capture as capture
        import w1ax_continuous_stages as stages

        manifest = self.fixture.inspect()
        runtime = tool.write_inventory(manifest, self.root / "output")
        sources = {
            "binary": manifest["binary"]["path"],
            "sha256": {"binary": manifest["binary"]["sha256"]},
            "native_runtime": runtime,
        }
        for name in (
            "target_gguf",
            "candidate_d_gguf",
            "base_draft_gguf",
            "absolute_d2t",
            "model_snapshot_manifest",
        ):
            path = self.root / (name + ".fixture")
            path.write_bytes(b"tiny source fixture " + name.encode())
            sources[name] = str(path)
            sources["sha256"][name] = tool.sha256(path)
        stages.verify_sources(sources)
        collector.verify_native_revision(sources, NATIVE)
        with self.assertRaisesRegex(ValueError, "selected published commit"):
            collector.verify_native_revision(sources, "c" * 40)
        maps = "\n".join("000-fff r-xp 00 00:00 0 " + r["path"] for r in runtime["libraries"])
        with patch.object(Path, "read_text", return_value=maps):
            checked = capture.verify_mapped_runtime(123, runtime)
        self.assertEqual(len(checked["mapped_libraries"]), len(runtime["libraries"]))
        Path(runtime["libraries"][0]["path"]).write_bytes(b"replaced library fixture")
        # Clear the existing verifier's per-process checked-record cache to
        # mimic the independently started actual consumer CLI.
        stages._VERIFIED_RECORDS.clear()
        with self.assertRaisesRegex(ValueError, "SHA256|hash"):
            stages.verify_sources(sources)
        with (
            patch.object(Path, "read_text", return_value=maps),
            self.assertRaisesRegex(ValueError, "changed"),
        ):
            capture.verify_mapped_runtime(123, runtime)

    def test_caller_commit_gitlink_and_dirty_source_fail(self):
        for field in ("parent", "native_commit", "link"):
            with self.subTest(field=field):
                saved = getattr(self.fixture, field)
                setattr(self.fixture, field, "c" * 40)
                with self.assertRaisesRegex(ValueError, "source commits|gitlink"):
                    self.fixture.inspect()
                setattr(self.fixture, field, saved)
        self.fixture.git_dirty = " M ggml/src/ggml-cuda/CMakeLists.txt"
        with self.assertRaisesRegex(ValueError, "dirty"):
            self.fixture.inspect()
        with self.assertRaisesRegex(ValueError, "full lowercase"):
            tool.inspect_runtime(
                self.fixture.checkout,
                self.fixture.build,
                "abcd",
                NATIVE,
                runner=self.fixture.runner,
            )

    def test_cmake_target_subdirectories_and_relative_object_outputs(self):
        for entry in self.fixture.entries:
            directory = self.fixture.build / "target-subdirectory" / Path(entry["file"]).stem
            directory.mkdir(parents=True)
            old_object = Path(entry["arguments"][-1])
            new_object = directory / "current.o"
            old_object.rename(new_object)
            entry["directory"] = str(directory)
            entry["arguments"][-1] = "current.o"
        self.fixture.write_commands()
        self.fixture.fresh_times()
        manifest = self.fixture.inspect()
        self.assertEqual(set(manifest["compiled"]), {"w1a1", "server", "server_impl", "build_info"})
        self.assertTrue(
            all("target-subdirectory" in v["object"]["path"] for v in manifest["compiled"].values())
        )
        self.fixture.entries[0]["directory"] = str(self.root)
        self.fixture.write_commands()
        self.fixture.fresh_times()
        with self.assertRaisesRegex(
            ValueError, "current-source compile|working directory.*outside"
        ):
            self.fixture.inspect()

    def test_generated_and_compiled_build_commit_both_required(self):
        self.fixture.info.write_text('char const * LLAMA_COMMIT = "cccccccc";')
        self.fixture.fresh_times()
        with self.assertRaisesRegex(ValueError, "build-info.*stale"):
            self.fixture.inspect()
        self.fixture.info.write_text('char const * LLAMA_COMMIT = "' + NATIVE[:9] + '";')
        self.fixture.fresh_times()
        self.fixture.rodata_commit = "cccccccc"
        with self.assertRaisesRegex(ValueError, "compiled.*current build-info"):
            self.fixture.inspect()

    def test_cuda_config_toolchain_and_compile_source_substitution(self):
        for field, value in [
            ("GGML_CUDA", "OFF"),
            ("CMAKE_BUILD_TYPE", "Debug"),
            ("CMAKE_CUDA_ARCHITECTURES", "75"),
            ("CMAKE_HOME_DIRECTORY", "/elsewhere"),
            ("CMAKE_CUDA_COMPILER", "/different/nvcc"),
        ]:
            with self.subTest(field=field):
                old = self.fixture.cache[field]
                self.fixture.cache[field] = value
                self.fixture.write_cache()
                self.fixture.fresh_times()
                with self.assertRaises(ValueError):
                    self.fixture.inspect()
                self.fixture.cache[field] = old
        self.fixture.write_cache()
        self.fixture.entries[0]["file"] = str(self.root / "other/w1a1.cu")
        self.fixture.write_commands()
        self.fixture.fresh_times()
        with self.assertRaisesRegex(ValueError, "current-source compile entry"):
            self.fixture.inspect()

    def test_ftz_order_override_and_generated_architecture(self):
        original = copy.deepcopy(self.fixture.entries[0]["arguments"])
        replacements = [
            ["--ftz=false", "-use_fast_math", original[2]],
            ["-use_fast_math", original[2], "--ftz=true"],
            ["-use_fast_math", original[2], "-Xcompiler", "--ftz=false"],
            ["-use_fast_math", "--generate-code=arch=compute_75,code=sm_75", "--ftz=false"],
        ]
        for flags in replacements:
            with self.subTest(flags=flags):
                self.fixture.entries[0]["arguments"] = [original[0], *flags, *original[-4:]]
                self.fixture.write_commands()
                self.fixture.fresh_times()
                with self.assertRaisesRegex(ValueError, "ftz|architecture"):
                    self.fixture.inspect()

    def test_response_file_hidden_override_and_external_response_rejected(self):
        response = self.fixture.build / "flags.rsp"
        response.write_text("--ftz=true")
        self.fixture.entries[0]["arguments"].extend(["--options-file", str(response)])
        self.fixture.write_commands()
        self.fixture.fresh_times()
        with self.assertRaisesRegex(ValueError, "ftz"):
            self.fixture.inspect()
        response = self.root / "outside.rsp"
        response.write_text("--ftz=false")
        self.fixture.entries[0]["arguments"][-1] = str(response)
        self.fixture.write_commands()
        self.fixture.fresh_times()
        with self.assertRaisesRegex(ValueError, "outside"):
            self.fixture.inspect()

    def test_missing_cuda_external_symlink_and_mixed_ldd_rejected(self):
        cuda = self.fixture.bin / "libggml-cuda.so"
        cuda.unlink()
        actual = self.fixture.bin / "libggml-cuda.so.1"
        actual.unlink()
        with self.assertRaisesRegex(ValueError, "CUDA backend"):
            self.fixture.inspect()
        actual.symlink_to(self.fixture.system)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.fixture.inspect()
        actual.unlink()
        actual.write_bytes(b"\x7fELFcuda fixture")
        self.fixture.fresh_times()
        self.fixture.ldd_extra = "libggml-cuda.so => " + str(self.fixture.system) + " (0x03)\n"
        with self.assertRaisesRegex(ValueError, "outside"):
            self.fixture.inspect()
        outside = self.root / "libggml-cuda.so"
        outside.write_bytes(b"project dependency fixture")
        self.fixture.ldd_extra = str(outside) + " (0x03)\n"
        with self.assertRaisesRegex(ValueError, "outside"):
            self.fixture.inspect()

    def test_search_path_missing_dependency_and_non_elf_rejected(self):
        for path in (str(self.root), str(self.fixture.bin) + ":", "relative", "$ORIGIN/../old"):
            self.fixture.runpath = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.fixture.inspect()
        self.fixture.runpath = "$ORIGIN"
        self.fixture.inspect()
        self.fixture.ldd_extra = "libggml.so => not found\n"
        with self.assertRaisesRegex(ValueError, "dependency is missing"):
            self.fixture.inspect()
        self.fixture.ldd_extra = ""
        (self.fixture.bin / "llama-server").write_bytes(b"not ELF")
        self.fixture.fresh_times()
        with self.assertRaisesRegex(ValueError, "ELF"):
            self.fixture.inspect()

    def test_explicit_toolkit_search_root_is_bound_without_admitting_project_libs(self):
        toolkit = self.root / "cuda-toolkit/lib64"
        toolkit.mkdir(parents=True)
        cudart = toolkit / "libcudart.so.13"
        cudart.write_bytes(b"tiny explicit runtime dependency")
        self.fixture.runpath = str(self.fixture.bin) + ":" + str(toolkit)
        self.fixture.ldd_extra = "libcudart.so.13 => " + str(cudart) + " (0x03)\n"
        with self.assertRaisesRegex(ValueError, "explicit runtime roots"):
            self.fixture.inspect()
        manifest = self.fixture.inspect(allowed_runtime_roots=[toolkit])
        self.assertEqual(manifest["allowed_runtime_roots"], [str(toolkit)])
        self.assertEqual(manifest["allowed_runtime_dependency_artifacts"], [tool.record(cudart)])
        # Explicit system roots cannot admit an older project backend.
        old_backend = toolkit / "libggml-cuda.so.1"
        old_backend.write_bytes(b"older project dependency")
        self.fixture.ldd_extra += "libggml-cuda.so.1 => " + str(old_backend) + " (0x04)\n"
        with self.assertRaisesRegex(ValueError, "outside"):
            self.fixture.inspect(allowed_runtime_roots=[toolkit])
        # An unlisted outside directory remains forbidden.
        self.fixture.runpath += ":" + str(self.root)
        with self.assertRaisesRegex(ValueError, "explicit runtime roots"):
            self.fixture.inspect(allowed_runtime_roots=[toolkit])

    def test_stale_objects_or_backend_and_duplicate_compile_entries_rejected(self):
        obj = self.fixture.build / "w1a1.o"
        os.utime(obj, ns=(0, 0))
        with self.assertRaisesRegex(ValueError, "predates"):
            self.fixture.inspect()
        self.fixture.fresh_times()
        os.utime(self.fixture.bin / "libggml-cuda.so.1", ns=(0, 0))
        with self.assertRaisesRegex(ValueError, "predates"):
            self.fixture.inspect()
        self.fixture.fresh_times()
        self.fixture.entries.append(copy.deepcopy(self.fixture.entries[0]))
        self.fixture.write_commands()
        self.fixture.fresh_times()
        with self.assertRaisesRegex(ValueError, "one current-source"):
            self.fixture.inspect()

    def test_changed_library_during_inspection_and_existing_output_fail_closed(self):
        runner = self.fixture.runner

        def mutate(argv):
            result = runner(argv)
            if argv[0] == "ldd":
                (self.fixture.bin / "libggml-base.so.1").write_bytes(b"\x7fELFchanged")
            return result

        with self.assertRaisesRegex(ValueError, "changed during inspection"):
            tool.inspect_runtime(
                self.fixture.checkout, self.fixture.build, PARENT, NATIVE, runner=mutate
            )
        output = self.root / "existing"
        output.mkdir()
        (output / "manifest.json").write_text("preserve")
        with self.assertRaises(FileExistsError):
            tool.write_inventory({}, output)
        with (
            self.assertRaises(SystemExit),
            contextlib.redirect_stderr(io.StringIO()),
            patch.object(tool, "inspect_runtime", side_effect=AssertionError("inspection")),
        ):
            tool.main(
                [
                    "--checkout",
                    str(self.fixture.checkout),
                    "--build-dir",
                    str(self.fixture.build),
                    "--expected-parent-commit",
                    PARENT,
                    "--expected-native-commit",
                    NATIVE,
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual((output / "manifest.json").read_text(), "preserve")

    def test_publication_rechecks_runtime_before_creating_output(self):
        manifest = self.fixture.inspect()
        Path(manifest["binary"]["path"]).write_bytes(b"changed after inspection")
        output = self.root / "publication"
        with self.assertRaisesRegex(ValueError, "before publication"):
            tool.write_inventory(manifest, output)
        self.assertFalse(output.exists())

    def test_commandrunner_uses_argv_and_strips_loader_overrides(self):
        with (
            patch.dict(
                os.environ,
                {
                    "LD_LIBRARY_PATH": "/old",
                    "LD_PRELOAD": "old.so",
                    "DYLD_INSERT_LIBRARIES": "old.dylib",
                },
            ),
            patch.object(tool.subprocess, "run") as run,
        ):
            run.return_value.stdout, run.return_value.stderr = "observed", ""
            self.assertEqual(
                tool.run_checked(["ldd", "/path with spaces/llama-server"]), "observed"
            )
            args, kwargs = run.call_args
            self.assertEqual(args[0], ["ldd", "/path with spaces/llama-server"])
            self.assertFalse(any(k.startswith(("LD_", "DYLD_")) for k in kwargs["env"]))
            self.assertNotIn("shell", kwargs)
        with self.assertRaisesRegex(ValueError, "only Git"):
            tool.run_checked(["llama-server", "--version"])


if __name__ == "__main__":
    unittest.main()
