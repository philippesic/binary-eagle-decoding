import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.acceptance import reduce_rounds
from benchmarks.capture import capture
from benchmarks.native import build, helpers
from benchmarks.runner import compare
from benchmarks.specification import Specification
from w1a1_eagle.config import benchmark_environment, numerical_python
from w1a1_eagle.contracts import Record, sequence
from w1a1_eagle.identity import freeze_source, source_identity
from w1a1_eagle.records import read
from w1a1_eagle.storage import output_path, relocated


class Arguments(argparse.Namespace):
    command: str
    input: Path
    output: Path
    requested_output: Path | None
    frozen: bool


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks")
    parser.add_argument("--frozen", action="store_true", help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "verify-inputs", "reduce-trace", "campaign-report", "campaign-preflight"):
        command = commands.add_parser(name)
        command.add_argument("input", type=Path)
    for name in (
        "compare",
        "curate",
        "native-build",
        "capture",
        "admit-bank",
        "native-helpers",
        "evaluate",
        "live",
        "prepare-panels",
        "suite",
        "campaign",
        "verify-sampler",
        "verify-greedy",
        "select-supplement",
        "profile-native",
        "prepare-project-workflows",
        "prepare-replays",
        "record-reference",
        "prepare-checks",
        "prepare-initialization",
        "calibrate-depth",
        "profile-rounds",
        "round-profile-report",
    ):
        comparison = commands.add_parser(name)
        comparison.add_argument("input", type=Path)
        comparison.add_argument("requested_output", metavar="OUTPUT", type=Path, nargs="?")
    args = parser.parse_args(namespace=Arguments())
    try:
        args.input = relocated(args.input).resolve()
        if hasattr(args, "requested_output"):
            categories = {
                "capture": "datasets",
                "curate": "bank",
                "admit-bank": "bank",
                "select-supplement": "bank",
                "native-build": "native",
                "native-helpers": "native",
                "campaign": "training",
                "record-reference": "benchmarks",
                "prepare-initialization": "models",
            }
            if args.command == "record-reference" and args.requested_output is None:
                args.requested_output = Path("artifacts/benchmarks/references")
            args.output = output_path(
                args.requested_output, categories.get(args.command, "benchmarks"), args.input.stem
            )
        if not args.frozen and args.command == "campaign-preflight":
            root = Path(__file__).resolve().parents[1]
            return subprocess.run(
                [
                    numerical_python(root),
                    "-B",
                    "-m",
                    "benchmarks",
                    "--frozen",
                    args.command,
                    str(args.input.resolve()),
                ],
                cwd=root,
                env=benchmark_environment(root, root),
                check=False,
            ).returncode
        if not args.frozen and args.command not in {
            "plan",
            "verify-inputs",
            "reduce-trace",
            "campaign-report",
            "campaign-preflight",
        }:
            root = Path(__file__).resolve().parents[1]
            code = source_identity(root)
            snapshot = freeze_source(
                root, args.output.resolve().parent / "executed-sources", code, 0
            )
            python = (
                numerical_python(root)
                if args.command
                in {
                    "curate",
                    "admit-bank",
                    "prepare-replays",
                    "capture",
                    "evaluate",
                    "campaign",
                    "prepare-initialization",
                }
                else sys.executable
            )
            return subprocess.run(
                [
                    python,
                    "-B",
                    "-m",
                    "benchmarks",
                    "--frozen",
                    args.command,
                    str(args.input.resolve()),
                    str(args.output.resolve()),
                ],
                cwd=snapshot,
                env=benchmark_environment(root, snapshot),
                check=False,
            ).returncode
        if args.command == "campaign-preflight":
            from benchmarks.campaign import preflight

            result = preflight(args.input)
        elif args.command == "calibrate-depth":
            from benchmarks.calibration import calibrate

            result = calibrate(args.input, args.output)
        elif args.command == "profile-rounds":
            from benchmarks.round_profile import profile_rounds

            result = profile_rounds(args.input, args.output)
        elif args.command == "round-profile-report":
            from benchmarks.round_profile import reanalyse

            result = reanalyse(args.input, args.output)
        elif args.command == "prepare-initialization":
            from w1a1_eagle.contracts import integer, record
            from w1a1_eagle.intent import Intent
            from w1a1_eagle.model import Geometry
            from w1a1_eagle.numerical.weights import prepare_initialization

            intent = Intent.load(args.input)
            result = prepare_initialization(
                intent.source,
                intent.vocabulary,
                intent.memory_bytes,
                Geometry(
                    **{key: integer(value) for key, value in record(intent.raw["geometry"]).items()}
                ),
                args.output,
            )
        elif args.command == "prepare-checks":
            from benchmarks.lean import prepare

            result = prepare(args.input, args.output)
        elif args.command == "record-reference":
            from benchmarks.references import register

            result = register(args.input, args.output)
        elif args.command == "prepare-replays":
            from benchmarks.replays import prepare

            result = prepare(args.input, args.output)
        elif args.command == "campaign-report":
            from benchmarks.reporting import report

            result = report(args.input)
        elif args.command == "prepare-project-workflows":
            from benchmarks.project_workflows import prepare

            result = prepare(args.input, args.output)
        elif args.command == "profile-native":
            from benchmarks.profiling import profile

            result = profile(args.input, args.output)
        elif args.command == "select-supplement":
            from benchmarks.supplement import supplement

            result = supplement(args.input, args.output)
        elif args.command == "verify-sampler":
            from benchmarks.verifier import verify_sampler

            result = verify_sampler(args.input, args.output)
        elif args.command == "verify-greedy":
            from benchmarks.verifier import verify_greedy

            result = verify_greedy(args.input, args.output)
        elif args.command == "suite":
            from benchmarks.suite import suite

            result = suite(args.input, args.output)
        elif args.command == "campaign":
            from benchmarks.campaign import campaign

            result = campaign(args.input, args.output)
        elif args.command == "prepare-panels":
            from benchmarks.panels import panels

            result = panels(args.input, args.output)
        elif args.command == "live":
            from benchmarks.workflows import live

            result = live(args.input, args.output)
        elif args.command == "evaluate":
            from w1a1_eagle.numerical.evaluation import evaluate

            result = evaluate(args.input, args.output)
        elif args.command == "native-helpers":
            result = helpers(args.input, args.output)
        elif args.command == "admit-bank":
            from benchmarks.bank_admission import admit

            result = admit(args.input, args.output)
        elif args.command == "capture":
            result = capture(args.input, args.output)
        elif args.command == "native-build":
            result = build(args.input, args.output)
        elif args.command == "curate":
            from benchmarks.bank import curate

            result = curate(args.input, args.output)
        elif args.command == "compare":
            result = compare(args.input, args.output)
        elif args.command == "reduce-trace":
            result = reduce_rounds(sequence(read(args.input, 16 * 1024 * 1024)["rounds"]))
        elif read(args.input)["schema"] == "native-comparison-v1":
            from benchmarks.comparison import Comparison

            native_spec = Comparison.load(args.input)
            hashed = args.command == "verify-inputs"
            if hashed:
                native_spec.verify()
            factor = native_spec.repeats * len(native_spec.arms) * len(native_spec.sampling.seeds)
            groups = native_spec.block_count()
            result = {
                "tier": native_spec.tier,
                "role": native_spec.role,
                "tasks": len(native_spec.tasks),
                "arms": [arm.name for arm in native_spec.arms],
                "historical_arms": [name for name, _ in native_spec.references],
                "measured_units": len(native_spec.tasks) * factor,
                "unit": "live episodes"
                if native_spec.tier == "live"
                else "native completion requests",
                "warmup_units": native_spec.warmups * groups * factor,
                "model_and_runtime_artifacts_hashed": hashed,
                "scope": "CPU input/coverage plan; hardware support, execution, performance and authority remain separate",
            }
        else:
            spec = Specification.load(args.input)
            result = spec.report()
            if args.command == "verify-inputs":
                if not spec.features:
                    raise ValueError("no feature artifacts declared")
                for artifact in spec.features:
                    artifact.verify()
                result["authenticated_artifacts"] = len(spec.features)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError) as error:
        refusal: Record = {"outcome": "refused", "detail": str(error)}
        print(json.dumps(refusal))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
