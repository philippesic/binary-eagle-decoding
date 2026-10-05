"""Modeled OS/resource seams only; these tests establish no CUDA readiness."""

import copy
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import evaluate_nine_model_native as evaluator  # noqa: E402
import run_nine_model_endpoint_collection as runtime  # noqa: E402
import test_nine_model_endpoint_collection as fixtures  # noqa: E402
from test_nine_model_endpoint_collection import pin, write  # noqa: E402

from w1a1_eagle.nine_model_pipeline import CANDIDATES, Files  # noqa: E402


class Observer:
    def __init__(self):
        self.events = []
        self.current = dict(
            boot_id="original-boot",
            hardware="Modeled RTX5080 CPU fixture",
            gpu_uuid="GPU-fixture",
            compute_capability=[12, 0],
            gpu_free_bytes=1000,
            host_available_bytes=2000,
            dxg_holders=[],
        )
        self.release = dict(
            owned_process_groups_absent=True, owned_cuda_pids_absent=True, other_context_pids=[]
        )

    def snapshot(self):
        self.events.append("snapshot")
        return copy.deepcopy(self.current)

    def require_released(self, groups, identities):
        self.events.append(("release", list(groups), list(identities)))
        if any(i.get("live") for i in identities):
            raise ValueError("owned process live")
        return copy.deepcopy(self.release)


class RuntimeTests(unittest.TestCase):
    def setup_context(self, root):
        fixture = fixtures.CollectionTests()
        value, source, contexts, exports, validators = fixture.fixture(root)
        source["origin"] = {
            "candidates": {n: {"native_markers": ["actual native marker"]} for n in CANDIDATES}
        }
        protocol_path = Path(source["protocol"]["path"])
        protocol = json.loads(protocol_path.read_text())
        protocol["evaluation_wall_seconds"] = 100
        protocol_path.write_text(json.dumps(protocol))
        source["protocol"] = pin(protocol_path)
        controls = root / "live-control"
        controls.write_text(json.dumps({"rtx5080": {"pause_requested": False}}))
        owners = {}
        for i, name in enumerate(CANDIDATES):
            contexts[name]["lane"].update(
                resource_policy=dict(
                    host_floor_bytes=1,
                    gpu_floor_bytes=1,
                    host_return_tolerance_bytes=10,
                    gpu_return_tolerance_bytes=10,
                ),
                environment={},
                gpu_control_path=str(controls),
                source={},
            )
            contexts[name]["checkpoint"] = write(root, name + "/committed-checkpoint", {"step": 1})
            contexts[name]["exported"] = {"manifest": write(root, name + "/manifest", {"step": 1})}
            controller = dict(pid=10 + i * 3, start_ticks=100 + i, boot_id="original-boot")
            supervisor = dict(pid=11 + i * 3, start_ticks=200 + i, boot_id="original-boot")
            trainer = dict(pid=12 + i * 3, start_ticks=300 + i, boot_id="original-boot")
            owners[name] = dict(
                controller=controller,
                supervisor=supervisor,
                identity_evidence=write(
                    root, name + "/original-monitor", {"identities": [controller, supervisor]}
                ),
            )
            value["candidates"][name]["supervisor_state"] = write(
                root,
                name + "/supervisor_state",
                dict(
                    status="finished",
                    exit_code=0,
                    received_signal=None,
                    pid=controller["pid"],
                    supervisor_pid=supervisor["pid"],
                    pgid=controller["pid"],
                    supervisor_pgid=supervisor["pid"],
                    command=[
                        "python",
                        "run_nine_model_lane.py",
                        "--run-dir",
                        str((root / name).resolve()),
                    ],
                ),
            )
            value["candidates"][name]["lane_state"] = write(
                root, name + "/lane_state", {"resource_baseline": {"boot_id": "original-boot"}}
            )
            write(
                root,
                name + "/attempts/train/process.json",
                dict(pid=trainer["pid"], pgid=trainer["pid"], kernel_identity=trainer),
            )
            write(
                root,
                name + "/attempts/train/process-lineage.json",
                {"kernel_identities": [trainer]},
            )
        source["origin"].update(
            resource_policy=contexts[CANDIDATES[0]]["lane"]["resource_policy"], environment={}
        )
        context = dict(
            collection=value,
            source=source,
            contexts=contexts,
            exports=exports,
            protocol=protocol,
            files=Files(),
        )
        value["hardware_stage"]["gpu_uuid"] = "GPU-fixture"
        collection = write(root, "collection", value)

        # Collection software fixtures inject serialized checkpoint validator only.
        def validate(v, *, files):
            for selected in v["candidates"].values():
                for locator in selected.values():
                    files.check(locator)
            context["files"] = files
            return context

        with (
            patch.object(runtime, "validate_collection", side_effect=validate),
            patch.object(
                runtime,
                "source_pins",
                return_value={
                    "new-dispatcher": pin(Path(runtime.__file__)),
                    "new-evaluator": pin(Path(evaluator.__file__)),
                },
            ),
        ):
            jobs = {
                name: {
                    **owner,
                    "run_dir": str((root / name).resolve()),
                    "supervisor_state": value["candidates"][name]["supervisor_state"],
                }
                for name, owner in owners.items()
            }
            plan = runtime.freeze(collection, owners, jobs)
        return plan, context, owners

    def guard(self, root):
        plan, context, _ = self.setup_context(root)
        observer = Observer()
        owner = dict(pid=555, start_ticks=777, boot_id="original-boot")
        lease = write(root, "lease", {"immutable": "startup"})
        guard = runtime.Guard(
            plan,
            context,
            context["files"],
            observer,
            observer.snapshot(),
            root,
            lease,
            {"immutable": "startup"},
            owner=owner,
            lock=root / "lock",
        )
        return guard, observer

    def modeled_owner(self, owner):
        stack = ExitStack()
        stack.enter_context(patch.object(runtime, "identity_active", return_value=True))
        stack.enter_context(patch.object(runtime, "process_identity", return_value=owner))
        stack.enter_context(patch.object(runtime, "require_held_owner_lock"))
        return stack

    def test_freeze_preserves_six_original_censuses_and_new_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, _, _ = self.setup_context(Path(tmp))
            self.assertEqual(set(plan["upstream"]), set(CANDIDATES))
            self.assertEqual(len(plan["scope"]), 10)
            self.assertFalse(plan["execution_allowed"])
            self.assertEqual(set(plan["source"]), {"new-dispatcher", "new-evaluator"})
            groups, identities = runtime.upstream_processes(plan, Files())
            self.assertEqual(len(groups), 36)
            self.assertEqual(len(identities), 48)

    def test_upstream_birth_or_process_census_substitution_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, context, owners = self.setup_context(Path(tmp))
            wrong = copy.deepcopy(owners)
            wrong[CANDIDATES[0]]["controller"]["start_ticks"] += 1
            with self.assertRaisesRegex(ValueError, "birth proof"):
                runtime.upstream_records(context, wrong)
            Path(plan["upstream"][CANDIDATES[0]]["process_records"][0]["path"]).write_text(
                "changed"
            )
            with self.assertRaises(ValueError):
                runtime.upstream_processes(plan, context["files"])

    def test_fresh_continuation_and_actual_per_cell_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard, observer = self.guard(root)
            with (
                self.modeled_owner(guard.owner),
                patch.object(runtime, "cleanup_descendants") as clean,
            ):
                cell = root / "cell"
                cell.mkdir()
                guard.before_cell(cell)
                guard.admit_cell_launch()
                guard.after_cell(cell)
                record = json.loads((cell / "cell-continuation.json").read_text())
                self.assertEqual(record["startup_lease"], guard.lease)
                self.assertEqual(record["expires_unix"] - record["issued_unix"], 300)
                self.assertTrue((cell / "actual-release.json").exists())
                clean.assert_called_once()
                self.assertEqual(len([e for e in observer.events if isinstance(e, tuple)]), 2)

    def test_stop_pause_lease_checkpoint_source_model_control_mutation_rejected(self):
        for mutation in ("STOP", "pause", "lease", "checkpoint", "source", "model", "control"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                guard, _ = self.guard(root)
                with self.modeled_owner(guard.owner):
                    guard.before_cell()
                    if mutation == "STOP":
                        (root / "STOP").touch()
                    elif mutation == "pause":
                        Path(guard.plan["gpu_control_path"]).write_text(
                            json.dumps({"rtx5080": {"pause_requested": True}})
                        )
                    elif mutation == "lease":
                        Path(guard.lease["path"]).write_text("changed")
                    elif mutation == "checkpoint":
                        Path(
                            guard.context["contexts"][CANDIDATES[0]]["checkpoint"]["path"]
                        ).write_text("changed")
                    elif mutation == "source":
                        guard.plan["source"]["new-evaluator"] = write(root, "wrong-source", "wrong")
                    elif mutation == "model":
                        Path(guard.context["exports"][CANDIDATES[0]]["model"]["path"]).write_text(
                            "changed"
                        )
                    else:
                        Path(
                            guard.context["collection"]["controls"]["dspark"]["provenance"]["path"]
                        ).write_text("changed")
                    if mutation == "source":
                        # Plan mutations themselves are prevented by its locator/stat
                        # guard; exercise immutable original pinned producer bytes.
                        guard.plan["source"]["new-evaluator"]["sha256"] = "0" * 64
                    with self.assertRaises(ValueError):
                        guard.authorize()

    def test_wrong_boot_device_foreign_context_and_resource_return_rejected(self):
        for key, value in (
            ("boot_id", "newboot"),
            ("gpu_uuid", "GPU-other"),
            ("compute_capability", [7, 5]),
            ("dxg_holders", [{"pid": 12}]),
            ("gpu_free_bytes", 900),
            ("host_available_bytes", 1900),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                guard, observer = self.guard(Path(tmp))
                observer.current[key] = value
                with self.modeled_owner(guard.owner), self.assertRaises(ValueError):
                    guard.before_cell()
        with tempfile.TemporaryDirectory() as tmp:
            guard, observer = self.guard(Path(tmp))
            observer.release["other_context_pids"] = [99]
            with self.modeled_owner(guard.owner), self.assertRaises(ValueError):
                guard.before_cell()

    def test_owned_process_release_failure_and_expired_cell_forbid_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard, observer = self.guard(root)
            with self.modeled_owner(guard.owner):
                guard.before_cell()
                guard.cell_continuation["expires_unix"] = 0
                with self.assertRaisesRegex(ValueError, "stale"):
                    guard.admit_cell_launch()
                guard.identities.append({"live": True})
                with (
                    patch.object(runtime, "cleanup_descendants"),
                    self.assertRaisesRegex(ValueError, "live"),
                ):
                    guard.after_cell(root)
                self.assertFalse((root / "actual-release.json").exists())

    def test_owner_and_lock_identity_are_required_each_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard, _ = self.guard(Path(tmp))
            with (
                patch.object(runtime, "identity_active", return_value=False),
                self.assertRaisesRegex(ValueError, "not live"),
            ):
                guard.authorize()
            with (
                self.modeled_owner(guard.owner),
                patch.object(
                    runtime, "require_held_owner_lock", side_effect=ValueError("lock not held")
                ),
                self.assertRaisesRegex(ValueError, "lock not held"),
            ):
                guard.before_cell()

    def test_stage_continuation_wrong_collection_source_parent_or_expiry_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard, observer = self.guard(root)
            with (
                self.modeled_owner(guard.owner),
                patch.object(runtime, "LOCK", root / "lock"),
                patch.object(runtime.time, "time", return_value=100),
            ):
                locator = runtime.issue_continuation(
                    guard.plan,
                    "planhash",
                    guard.lease,
                    root,
                    observer,
                    guard.baseline,
                    guard.groups,
                    guard.identities,
                )
                record = json.loads(Path(locator["path"]).read_text())
                runtime.verify_continuation(
                    guard.plan, "planhash", guard.lease, locator, Files(), now=101, parent_pid=555
                )
                for key, value in (
                    ("collection", {}),
                    ("runtime_plan_sha256", "other"),
                    ("expires_unix", 101),
                    ("owner", dict(pid=99, start_ticks=99, boot_id="original-boot")),
                ):
                    wrong = {**record, key: value}
                    bad = write(root, "bad-" + key, wrong)
                    with self.assertRaises(ValueError):
                        runtime.verify_continuation(
                            guard.plan,
                            "planhash",
                            guard.lease,
                            bad,
                            Files(),
                            now=102,
                            parent_pid=555,
                        )

    def test_plan_tamper_source_pin_and_changed_upstream_file_rejected_before_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, context, _ = self.setup_context(root)
            path = Path(write(root, "plan", plan)["path"])

            def validate(v, *, files):
                context["files"] = files
                return context

            with (
                patch.object(runtime, "validate_collection", side_effect=validate),
                patch.object(runtime, "source_pins", return_value=plan["source"]),
                patch.object(runtime, "LinuxResources") as observer,
            ):
                runtime.validate_plan(path, pin(path)["sha256"])
                with self.assertRaises(ValueError):
                    runtime.validate_plan(path, "0" * 64)
                changed = copy.deepcopy(plan["source"])
                changed["new-evaluator"]["sha256"] = "0" * 64
                with (
                    patch.object(runtime, "source_pins", return_value=changed),
                    self.assertRaisesRegex(ValueError, "NEW dispatcher"),
                ):
                    runtime.validate_plan(path, pin(path)["sha256"])
                write(root, CANDIDATES[0] + "/attempts/new-process/process.json", {})
                with self.assertRaisesRegex(ValueError, "census changed"):
                    runtime.validate_plan(path, pin(path)["sha256"])
                observer.assert_not_called()

    def test_collection_stage_passes_real_view_and_original_receipts_to_shared_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard, observer = self.guard(root)
            plan, context, files = guard.plan, guard.context, guard.files
            for name in CANDIDATES:
                context["exports"][name]["audit"] = write(
                    root, name + "/audit", {"projections": {"selected-original": {}}}
                )
            lease = write(
                root,
                "real-shape-lease",
                dict(
                    schema="nine_model_gpu_lease_v1",
                    host="rtx5080",
                    user_announced_available=True,
                    sole_owner=True,
                    pause_requested=False,
                    bundle_sha256="planhash",
                    gpu_uuid="GPU-fixture",
                    granted_unix=100,
                    expires_unix=400,
                ),
            )
            supervisor = write(
                root, "live-supervisor", dict(status="running", pid=555, supervisor_pid=556)
            )
            args = SimpleNamespace(
                plan=Path(write(root, "runtime-plan", plan)["path"]),
                plan_sha256="planhash",
                availability=Path(lease["path"]),
                supervisor_state=Path(supervisor["path"]),
                run_dir=root,
                completion_output=root / "receipt",
            )
            with (
                self.modeled_owner(guard.owner),
                patch.object(
                    runtime,
                    "process_identity",
                    side_effect=lambda pid: (
                        {**guard.owner, "pid": 556} if pid == 556 else guard.owner
                    ),
                ),
                patch.object(runtime, "require_remote_job"),
                patch.object(runtime, "LOCK", root / "lock"),
                patch.object(runtime.time, "time", return_value=200),
            ):
                continuation = runtime.issue_continuation(
                    plan,
                    "planhash",
                    lease,
                    root,
                    observer,
                    guard.baseline,
                    guard.groups,
                    guard.identities,
                )
                args.continuation = Path(continuation["path"])
                with (
                    patch.object(runtime, "LinuxResources", return_value=observer),
                    patch.object(runtime.os, "getppid", return_value=555),
                    patch.object(evaluator, "evaluate") as shared,
                ):
                    runtime.evaluate_stage(args, plan, context, files)
                positional = shared.call_args.args
                view, models = positional[1], positional[6]
                self.assertEqual(set(models), set(evaluator.CELLS))
                self.assertEqual(view["inputs"]["binary"], context["source"]["runtime"]["binary"])
                self.assertEqual(
                    positional[3]["export_endpoints"],
                    {
                        n: context["collection"]["candidates"][n]["export_receipt"]
                        for n in CANDIDATES
                    },
                )
                ancestry = shared.call_args.kwargs["ancestry"]
                self.assertEqual(
                    ancestry["original_candidates"], context["collection"]["candidates"]
                )
                self.assertNotIn("bundle_sha256", ancestry)
                for name in (*evaluator.CELLS, "target_only"):
                    argv = evaluator.native_command(
                        view, context["protocol"], name, models.get(name), 18000
                    )
                    self.assertEqual(argv[argv.index("--cache-type-k") + 1], "f16")

    def test_shared_loop_requires_guard_release_after_every_native_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = dict(
                split="development",
                repetitions=5,
                warmups_per_cell=2,
                max_output_tokens=1,
                seed=1,
                port=18000,
                startup_wall_seconds=10,
                evaluation_wall_seconds=1000,
                draft_lengths={"eagle": 5, "dspark": 7, "dflash": 7},
                context_tokens=2048,
                batch_tokens=32,
                microbatch_tokens=32,
            )
            bundle = dict(
                inputs=dict(
                    prompts=write(root, "prompts", "opaque"),
                    protocol=write(root, "protocol", protocol),
                    target=write(root, "target", "opaque"),
                    binary=write(root, "binary", "opaque"),
                ),
                gpu_uuid="GPU-fixture",
                candidates={n: {"native_markers": ["native marker"]} for n in CANDIDATES},
            )
            exports = {
                n: write(
                    root,
                    n + "/export",
                    {
                        "audit": write(
                            root,
                            n + "/audit",
                            dict(
                                activation_bits=8 if n.endswith("a8") else 1,
                                output={"sha256": "fixture"},
                            ),
                        )
                    },
                )
                for n in CANDIDATES
            }
            inputs = dict(export_endpoints=exports, target=bundle["inputs"]["target"])
            models = {n: {"path": str(root / n), "sha256": "fixture"} for n in evaluator.CELLS}
            args = SimpleNamespace(
                run_dir=root, completion_output=root / "receipt", bundle_sha256=None
            )
            events = []

            class CellGuard:
                def before_cell(self, d):
                    events.append("before")

                def admit_cell_launch(self):
                    events.append("admit")

                def register(self, p):
                    events.append("register")

                def authorize(self):
                    pass

                def discover(self, p):
                    pass

                def after_cell(self, d):
                    events.append("release")

            def launch(*args, **kwargs):
                kwargs["stdout"].write(b"native marker\n")
                return SimpleNamespace(pid=999)

            sample = SimpleNamespace(start=lambda: SimpleNamespace(stop=lambda: {}))
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        evaluator,
                        "load_opaque_prompts",
                        return_value=[dict(id="p", split="development")],
                    )
                )
                stack.enter_context(patch.object(evaluator, "available_port", return_value=True))
                stack.enter_context(patch("subprocess.Popen", side_effect=launch))
                stack.enter_context(patch.object(evaluator, "wait_ready"))
                stack.enter_context(patch.object(evaluator, "stop_owned_server"))
                stack.enter_context(patch.object(evaluator, "request_body", return_value={}))
                stack.enter_context(
                    patch.object(
                        evaluator,
                        "execute_request",
                        return_value=dict(
                            generated_token_ids=[1],
                            completion_tokens=1,
                            request_wall_s=0.1,
                            speculative={},
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(evaluator, "round_summary", return_value={"rounds": 1})
                )
                stack.enter_context(
                    patch.object(
                        evaluator, "validate_cuda_dispatch", return_value={"status": "PASS"}
                    )
                )
                stack.enter_context(
                    patch.object(evaluator, "DiagnosticMemorySampler", return_value=sample)
                )
                stack.enter_context(
                    patch.object(evaluator, "aggregate", return_value={"status": "PASS"})
                )
                evaluator.evaluate(
                    args,
                    bundle,
                    Files(),
                    inputs,
                    protocol,
                    {},
                    models,
                    Observer(),
                    guard=CellGuard(),
                    ancestry={"collection": {"sha256": "original"}},
                )
            self.assertEqual(
                events, [e for _ in range(60) for e in ("before", "admit", "register", "release")]
            )
            receipt = json.loads(args.completion_output.read_text())
            self.assertEqual(receipt["schema"], "nine_model_collection_evaluation_receipt_v1")
            self.assertNotIn("bundle_sha256", receipt)
            report = json.loads(Path(receipt["report"]["path"]).read_text())
            self.assertEqual(report["collection_ancestry"], receipt["collection_ancestry"])

    def test_controller_rejects_stale_wrong_and_nonexclusive_lease_before_gpu(self):
        for key, value in (
            ("expires_unix", 199),
            ("bundle_sha256", "foreign"),
            ("sole_owner", False),
            ("pause_requested", True),
            ("gpu_uuid", "GPU-other"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan, context, _ = self.setup_context(root)
                lease = dict(
                    schema="nine_model_gpu_lease_v1",
                    host="rtx5080",
                    user_announced_available=True,
                    sole_owner=True,
                    pause_requested=False,
                    bundle_sha256="planhash",
                    gpu_uuid="GPU-fixture",
                    granted_unix=100,
                    expires_unix=400,
                )
                lease[key] = value
                args = SimpleNamespace(
                    plan=root / "plan",
                    plan_sha256="planhash",
                    start=True,
                    stage=None,
                    run_dir=root / "new-run",
                    availability=Path(write(root, "invalid-lease", lease)["path"]),
                    supervisor_state=root / "supervisor",
                )
                with (
                    patch.object(runtime, "validate_plan", return_value=(plan, context, Files())),
                    patch.object(runtime.time, "time", return_value=200),
                    patch.object(runtime, "LinuxResources") as gpu,
                    self.assertRaises(ValueError),
                ):
                    runtime.run(args)
                gpu.assert_not_called()

    def test_actual_nonblocking_shared_lock_and_owner_file_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            import fcntl

            root = Path(tmp)
            lock = root / "lock"
            owner = {"pid": 555, "start_ticks": 1, "boot_id": "modeled"}
            lock.write_text(json.dumps(owner))
            with self.assertRaisesRegex(ValueError, "not actually held"):
                runtime.require_held_owner_lock(lock, owner)
            with lock.open("a+") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                runtime.require_held_owner_lock(lock, owner)
                with self.assertRaisesRegex(ValueError, "owner differs"):
                    runtime.require_held_owner_lock(lock, {**owner, "pid": 556})
                with lock.open("a+") as competing:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_separate_export_watcher_census_is_required_and_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, context, _ = self.setup_context(root)
            jobs = {
                name: {
                    key: value
                    for key, value in job.items()
                    if key not in {"process_records", "controller_pgid", "supervisor_pgid"}
                }
                for name, job in plan["upstream_jobs"].items()
            }
            with self.assertRaisesRegex(ValueError, "export/watcher jobs"):
                runtime.additional_upstream_records(context, {})
            wrong = copy.deepcopy(jobs)
            wrong[CANDIDATES[0]]["controller"]["start_ticks"] += 1
            with self.assertRaisesRegex(ValueError, "birth proof"):
                runtime.additional_upstream_records(context, wrong)
            context["collection"]["candidates"][CANDIDATES[0]]["export_receipt"] = write(
                root, "uncovered/export", {}
            )
            with self.assertRaisesRegex(ValueError, "every original export"):
                runtime.additional_upstream_records(context, jobs)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard, _ = self.guard(root)
            write(root, CANDIDATES[0] + "/attempts/new/process.json", {})
            with (
                self.modeled_owner(guard.owner),
                self.assertRaisesRegex(ValueError, "census changed"),
            ):
                guard.before_cell()


if __name__ == "__main__":
    unittest.main()
