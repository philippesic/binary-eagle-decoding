"""Tiny CPU metadata and real-byte consumer contract tests; no model execution."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "bounded_sources", SCRIPTS / "prepare_qat_bounded_sources.py"
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
PARENT, NATIVE = "a" * 40, "b" * 40


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_record(path):
    return {"path": str(path), "sha256": sha(path.read_bytes())}


class Fixture:
    def __init__(self, root):
        self.root = root
        self.bin = root / "build/bin"
        self.bin.mkdir(parents=True)
        self.paths = {}
        for name in tool.REQUIRED:
            path = self.bin / "llama-server" if name == "binary" else root / name
            path.write_bytes(b"tiny opaque fixture " + name.encode())
            self.paths[name] = path
        self.models = set(self.paths.values()) - {self.paths["binary"]}
        libraries = []
        for name in ("libllama.so.1", "libggml-cuda.so.1", "libggml-base.so.1"):
            path = self.bin / name
            path.write_bytes(b"tiny library " + name.encode())
            libraries.append(file_record(path))
        self.manifest_path = root / "runtime/manifest.json"
        self.manifest_path.parent.mkdir()
        self.manifest = {
            "schema": "qat_current_native_server_build_v1",
            "parent_commit": PARENT,
            "native_commit": NATIVE,
            "build_commit": NATIVE[:9],
            "binary": file_record(self.paths["binary"]),
            "libraries": libraries,
            "training_eligible": False,
            "readiness_granted": False,
            "allowed_runtime_dependency_artifacts": [],
        }
        self.runtime_path = root / "runtime/native-runtime.json"
        self.runtime = {
            "schema": "qat_current_native_runtime_v1",
            "directory": str(self.bin),
            "ld_library_path": str(self.bin),
            "libraries": libraries,
        }
        self.refresh_runtime()
        corpus = root / "corpus-manifest.json"
        corpus.write_text('{"files":{"sealed_test":{"shards":[]}}}')
        self.registration_path = root / "owner-sources.json"
        self.registration = {
            "schema": "qat_bounded_source_records_v1",
            "parent_commit": PARENT,
            "native_commit": NATIVE,
            **{k: str(v) for k, v in self.paths.items()},
            "sha256": {k: file_record(v)["sha256"] for k, v in self.paths.items()},
            "corpus_manifest": file_record(corpus),
        }
        self.prompts_path = root / "prompts.train.jsonl"
        self.prompt_rows = []
        selected = []
        for ordinal, domain in enumerate(tool.DOMAINS):
            prompt = {
                "id": "source:" + domain,
                "domain": domain,
                "messages": [{"role": "user", "content": "Tiny " + domain}],
            }
            line = json.dumps(prompt).encode() + b"\n"
            self.prompt_rows.append(line)
            selected.append(
                {
                    "id": prompt["id"],
                    "source_id": "source",
                    "source_row_id": domain,
                    "domain": domain,
                    "source_row_ordinal": ordinal,
                    "raw_prompt_line_sha256": sha(line),
                    "content_sha256": sha(
                        json.dumps(
                            prompt["messages"],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ),
                }
            )
        self.prompts_path.write_bytes(b"".join(self.prompt_rows))
        self.selection_path = root / "selection.json"
        self.selection = {
            "schema": "qat_bounded_train_prompt_selection_v1",
            "training_eligible": False,
            "provider_admission": False,
            "sealed_or_reserve_payload_read": False,
            "dev_payload_read": False,
            "packet": {"path": str(root / "unopened-packet.tar"), "sha256": "c" * 64},
            "corpus_manifest": file_record(corpus),
            "source_train_shard": {
                "prompts": "train-00000.jsonl",
                "index": "train-00000.index.jsonl",
                "prompts_sha256": "d" * 64,
                "index_sha256": "e" * 64,
                "prompts_count": 1000,
            },
            "selection": {
                **file_record(self.prompts_path),
                "count": 3,
                "roles": ["train"],
                "rows": selected,
            },
        }
        self.refresh_inputs()

    def refresh_runtime(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.runtime["immutable_manifest"] = file_record(self.manifest_path)
        self.runtime_path.write_text(json.dumps(self.runtime))

    def refresh_inputs(self):
        self.registration_path.write_text(json.dumps(self.registration))
        self.selection_path.write_text(json.dumps(self.selection))

    def build(self, parent=PARENT, native=NATIVE):
        return tool.build_sources(
            file_record(self.runtime_path),
            file_record(self.registration_path),
            file_record(self.selection_path),
            file_record(self.prompts_path),
            parent,
            native,
        )


class BoundedSourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name))

    def test_real_consumer_contract_and_byte_mutation(self):
        import collect_qat_native_evidence as collector
        import w1ax_continuous_stages as stages

        f = self.fixture
        sources = f.build()
        stages.verify_sources(sources)
        collector.verify_native_revision(sources, NATIVE)
        self.assertEqual(set(sources["sha256"]), set(tool.REQUIRED))
        self.assertFalse(sources["training_eligible"])
        self.assertFalse(sources["provider_admission"])
        self.assertFalse(sources["readiness_granted"])
        self.assertFalse(sources["actual_source_bytes_validated"])
        self.assertEqual(sources["native_runtime"], f.runtime)
        self.assertEqual(sources["prompt_selection"]["original_ancestry"], f.selection)
        f.paths["target_gguf"].write_bytes(b"changed model fixture")
        # Builder still joins authenticated metadata. Actual bytes are a consumer gate.
        f.build()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            stages.verify_sources(sources)
        with self.assertRaisesRegex(ValueError, "selected published commit"):
            collector.verify_native_revision(sources, "f" * 40)
        f.manifest_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "identity differs"):
            collector.verify_native_revision(sources, NATIVE)

    def test_only_explicit_metadata_and_prompt_inputs_opened(self):
        f = self.fixture
        for path in f.models:
            path.unlink()
        opened = []
        original = Path.open

        def bounded_open(path, *args, **kwargs):
            opened.append(path)
            if path in f.models or path == Path(f.registration["corpus_manifest"]["path"]):
                raise AssertionError("model/map/corpus bytes must remain opaque")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", bounded_open):
            sources = f.build()
        self.assertEqual(
            set(opened),
            {
                f.runtime_path,
                f.manifest_path,
                f.registration_path,
                f.selection_path,
                f.prompts_path,
            },
        )
        self.assertFalse(sources["actual_source_bytes_validated"])

    def test_relocation_preserves_original_ancestry_and_exact_hashes(self):
        f = self.fixture
        old = copy.deepcopy(f.selection)
        remote_corpus = str(f.root / "remote/unopened-corpus.json")
        f.registration["corpus_manifest"]["path"] = remote_corpus
        f.selection["selection"]["path"] = "/original-host/prompts.train.jsonl"
        f.refresh_inputs()
        sources = f.build()
        self.assertEqual(sources["corpus_manifest"]["path"], remote_corpus)
        self.assertEqual(sources["prompt_selection"]["prompts"], file_record(f.prompts_path))
        self.assertEqual(sources["prompt_selection"]["original_ancestry"], f.selection)
        self.assertEqual(sources["corpus_manifest"]["sha256"], old["corpus_manifest"]["sha256"])

    def test_source_registration_rejections(self):
        for mutation in (
            lambda r: r.update(schema="legacy"),
            lambda r: r.update(parent_commit="a" * 39),
            lambda r: r.update(native_commit="f" * 40),
            lambda r: r["sha256"].pop("absolute_d2t"),
            lambda r: r["sha256"].update(extra="e" * 64),
            lambda r: r.update(binary="relative-server"),
            lambda r: r["sha256"].update(binary="f" * 64),
            lambda r: r.update(target_gguf=r["base_draft_gguf"]),
            lambda r: r["corpus_manifest"].update(sha256="f" * 64),
            lambda r: r.update(training_eligible=True),
            lambda r: r.update(native_runtime={"binary": "conflicting pin"}),
        ):
            with self.subTest(mutation=mutation):
                f = Fixture(Path(tempfile.mkdtemp(dir=self.temp.name)))
                mutation(f.registration)
                f.refresh_inputs()
                with self.assertRaises(ValueError):
                    f.build()

    def test_runtime_rejections(self):
        cases = (
            ("manifest", lambda m: m.update(parent_commit="f" * 40)),
            ("manifest", lambda m: m.update(native_commit="f" * 40)),
            ("manifest", lambda m: m.update(build_commit="f" * 9)),
            ("manifest", lambda m: m["binary"].update(sha256="f" * 64)),
            ("manifest", lambda m: m.update(readiness_granted=True)),
            ("runtime", lambda r: r.update(schema="w1ax_frozen_native_runtime_v1")),
            ("runtime", lambda r: r.update(ld_library_path="/old/build/bin")),
            ("runtime", lambda r: r.update(libraries=r["libraries"][:1])),
            ("runtime", lambda r: r.update(libraries=r["libraries"] * 2)),
        )
        for field, mutate in cases:
            with self.subTest(field=field, mutate=mutate):
                f = Fixture(Path(tempfile.mkdtemp(dir=self.temp.name)))
                # Separate shared list identities as serialized runtime/manifest would be.
                f.runtime = copy.deepcopy(f.runtime)
                mutate(getattr(f, field))
                f.refresh_runtime()
                with self.assertRaises(ValueError):
                    f.build()

    def test_selection_rejections(self):
        cases = (
            lambda s: s.update(schema="legacy"),
            lambda s: s.update(training_eligible=True),
            lambda s: s.update(provider_admission=True),
            lambda s: s.update(dev_payload_read=True),
            lambda s: s["selection"].update(count=2),
            lambda s: s["selection"].update(roles=["dev"]),
            lambda s: s["selection"].update(sha256="f" * 64),
            lambda s: s["selection"]["rows"].reverse(),
            lambda s: s["selection"]["rows"][0].update(id="different:id"),
            lambda s: s["selection"]["rows"][0].update(source_id="other"),
            lambda s: s["selection"]["rows"][0].update(role="dev"),
            lambda s: s["selection"]["rows"][0].update(split="sealed_test"),
            lambda s: s["selection"]["rows"][0].update(raw_prompt_line_sha256="f" * 64),
            lambda s: s["selection"]["rows"][0].update(content_sha256="f" * 64),
            lambda s: s["selection"]["rows"][0].update(source_row_ordinal=1000),
            lambda s: s["selection"]["rows"][1].update(source_row_ordinal=0),
            lambda s: s["source_train_shard"].update(prompts="sealed-test.jsonl"),
        )
        for mutate in cases:
            with self.subTest(mutate=mutate):
                f = Fixture(Path(tempfile.mkdtemp(dir=self.temp.name)))
                mutate(f.selection)
                f.refresh_inputs()
                with self.assertRaises(ValueError):
                    f.build()

    def test_prompt_mismatch_with_repinning_still_rejected(self):
        f = self.fixture
        f.prompts_path.write_bytes(b"".join(reversed(f.prompt_rows)))
        f.selection["selection"]["sha256"] = file_record(f.prompts_path)["sha256"]
        f.refresh_inputs()
        with self.assertRaisesRegex(ValueError, "domain/order"):
            f.build()

    def test_metadata_hashes_bounds_duplicate_keys_and_full_commits(self):
        f = self.fixture
        pin = file_record(f.runtime_path)
        f.runtime_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            tool.pinned_json(pin)
        f.runtime_path.write_text('{"key":1,"key":2}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            tool.pinned_json(file_record(f.runtime_path))
        f.runtime_path.write_bytes(b"x" * (tool.MAX_METADATA_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "bounded size"):
            tool.pinned_bytes(file_record(f.runtime_path))
        with self.assertRaisesRegex(ValueError, "full lowercase"):
            f.build(parent=PARENT[:9])
        with self.assertRaisesRegex(ValueError, "normalized absolute"):
            tool.record({"path": "/a/../b", "sha256": "a" * 64})

    def test_output_cannot_occupy_registered_missing_payload_path(self):
        f = self.fixture
        sources = f.build()
        path = f.paths["target_gguf"]
        path.unlink()
        with self.assertRaisesRegex(ValueError, "overlaps"):
            tool.write_sources(sources, path)
        self.assertFalse(path.exists())

    def test_immutable_output_cli_and_original_inputs_unchanged(self):
        f = self.fixture
        inputs = [
            f.runtime_path,
            f.manifest_path,
            f.registration_path,
            f.selection_path,
            f.prompts_path,
        ]
        before = {p: p.read_bytes() for p in inputs}
        argv = []
        for name, path in zip(
            ("native-runtime", "source-records", "selection", "prompts"),
            (f.runtime_path, f.registration_path, f.selection_path, f.prompts_path),
            strict=True,
        ):
            argv += ["--" + name, str(path), "--" + name + "-sha256", file_record(path)["sha256"]]
        output = f.root / "new-output"
        argv += [
            "--expected-parent-commit",
            PARENT,
            "--expected-native-commit",
            NATIVE,
            "--output",
            str(output),
        ]
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            tool.main(argv)
        result = json.loads(stream.getvalue())
        self.assertEqual(result["sources"], file_record(output / "sources.json"))
        self.assertFalse(result["provider_admission"])
        emitted = (output / "sources.json").read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tool.main(argv)
        with self.assertRaises(FileExistsError):
            tool.write_sources(f.build(), output)
        self.assertEqual((output / "sources.json").read_bytes(), emitted)
        self.assertEqual(before, {p: p.read_bytes() for p in inputs})


if __name__ == "__main__":
    unittest.main()
