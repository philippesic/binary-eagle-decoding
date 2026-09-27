"""Run four bounded native/BLAS parity audits under one remote supervisor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-dir", type=Path, default=Path("results/scale-fit"))
    p.add_argument("--validation-dir", type=Path, default=Path("results/scale-validation"))
    p.add_argument("--binary", type=Path, default=Path("build/w1ax_operator_replay"))
    args = p.parse_args()
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("GGML_W1", "GGML_EAGLE", "EAGLE_")):
            env.pop(key)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    summaries = {}
    for variant in "ABCD":
        native = args.validation_dir / f"{variant}-native.jsonl"
        blas = args.validation_dir / f"{variant}-blas.jsonl"
        common = [
            "--gguf",
            str(args.fit_dir / f"{variant}.gguf"),
            "--capture-dir",
            str(args.validation_dir / "selected"),
        ]
        command = [
            str(args.binary.resolve()),
            *common,
            "--backend",
            "gpu",
            "--act-bits",
            "16",
            "--check-rows",
            "10",
            "--max-tokens",
            "4",
            "--emit-reference-values",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--require-nine",
        ]
        with (
            native.open("x") as out,
            (args.validation_dir / f"{variant}-native.log").open("x") as err,
        ):
            subprocess.run(command, stdout=out, stderr=err, check=True, env=env)
        records = [json.loads(line) for line in native.read_text().splitlines() if line.strip()]
        operators = [row for row in records if row.get("record_type") == "operator_replay"]
        if len({row["name"] for row in operators}) != 9:
            raise ValueError("native replay omitted layer")
        command2 = [
            sys.executable,
            "kernels/w1ax-replay/check_scale_blas.py",
            *common,
            "--replay-jsonl",
            str(native),
            "--device",
            "cuda",
        ]
        with blas.open("x") as out, (args.validation_dir / f"{variant}-blas.log").open("x") as err:
            subprocess.run(command2, stdout=out, stderr=err, check=True, env=env)
        comparisons = [json.loads(line) for line in blas.read_text().splitlines() if line.strip()]
        summaries[variant] = {
            "native_command": command,
            "blas_command": command2,
            "captures": len(operators),
            "checked_blas_outputs": sum(r["checked_outputs"] for r in comparisons),
            "blas_max_abs_error": max(r["max_abs_error"] for r in comparisons),
            "blas_max_relative_error_1_plus_abs": max(
                r["max_relative_error_1_plus_abs"] for r in comparisons
            ),
        }
        print(variant, json.dumps(summaries[variant]), flush=True)
    (args.validation_dir / "replay-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
