"""Extract and compile the actual pinned native emitter with CPU-only stubs."""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from research.parallel20261002.native_round_trace.reference.adapter import adapt_session

MAIN = Path("/Users/pippo/github/binary-eagle-decoding")
BASE = MAIN / "third_party/llama.cpp"
HERE = Path(__file__).parent
binding = json.loads((HERE / "source-binding.json").read_text())
source = (BASE / binding["source_path"]).read_text()
assert hashlib.sha256(source.encode()).hexdigest() == binding["original_sha256"]
with tempfile.TemporaryDirectory(prefix="eagle-round-callback-") as folder:
    folder = Path(folder)
    (folder / "tools/server").mkdir(parents=True)
    staged = folder / binding["source_path"]
    staged.write_text(source)
    patch = (HERE / "native-session-context.unapplied.patch").resolve()
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=folder, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=folder, check=True)
    patched = staged.read_text()
    assert hashlib.sha256(patched.encode()).hexdigest() == binding["patched_sha256"]
    struct = patched[
        patched.index("struct server_round_trace {") : patched.index("\nstruct server_batch {")
    ]
    method = patched[
        patched.index("    void emit_round_trace(") : patched.index(
            "\n    bool add_bos_token = true;"
        )
    ]
    (folder / "round_struct.inc").write_text(struct)
    (folder / "round_emitter.inc").write_text(method)
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O0",
            "-I" + str(BASE / "vendor"),
            "-I" + str(folder),
            str(HERE / "callback_fixture.cpp"),
            "-o",
            str(folder / "fixture"),
        ],
        check=True,
    )
    subprocess.run([str(folder / "fixture"), str(folder / "events.jsonl")], check=True)
    rows = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    result = adapt_session(
        rows,
        dict(
            session_id="compiled-actual-emitter",
            sampling_mode="greedy",
            prompt_token_ids=[1, 2],
            generated_token_ids=[9, 10, 99],
            generation_cap=3,
            eos_token_ids=[99],
            clock_domain="synthetic-process",
        ),
        dict(proposed=3, accepted=3, rounds=1),
    )
    assert result["counts"]["accepted_emitted"] == 2
    assert result["counts"]["accepted_not_emitted"] == 1
    assert result["counts"]["eos_emitted"] == 1
    assert result["costs"]["other_inside_round_us"] == 4
    assert len(rows) == 1
    print(
        json.dumps(
            dict(
                status="pass",
                fixture="compiled actual patched producer",
                binding=binding,
                counts=result["counts"],
                costs=result["costs"],
            ),
            indent=2,
        )
    )
