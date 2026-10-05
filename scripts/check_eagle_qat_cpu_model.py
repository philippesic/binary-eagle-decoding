#!/usr/bin/env python3
"""Direct EAGLE CPU diagnostics on an explicitly ineligible original TRAIN bundle.

No ContinuousTrainer/CurriculumRunner, production admission or target forward is
used. Optimizer updates are forbidden; warm lifecycle and CUDA remain pending.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import torch  # noqa:E402
from audit_recurrent_binary_capture import load_audited_capture  # noqa:E402
from check_block_qat_cpu_model import memory  # noqa:E402
from train_nine_model_qat import calibration, validate_initializer_reference  # noqa:E402

from w1a1_eagle.continuous_qat import (  # noqa:E402
    ObservedAdapter,
    atomic_json,
    later_gradient,
    sha256,
)
from w1a1_eagle.frozen_operands import FrozenOperands  # noqa:E402
from w1a1_eagle.native_step import NativeStepAdapter, bind_frozen_norms  # noqa:E402
from w1a1_eagle.qat_initialization import apply_binary_initialization  # noqa:E402
from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH  # noqa:E402
from w1a1_eagle.recurrent_provider import (  # noqa:E402
    ProviderRound,
    audit_provider_round,
    forward_torch_round,
)
from w1a1_eagle.recurrent_qat import (  # noqa:E402
    JointQATConfig,
    W1AxContract,
    install_joint_linears,
    joint_optimizer,
    save_joint_checkpoint,
    shared_round_hard_signs,
)

ANGELSLIM_REVISION = "0358da9c651e6a7d7ccafea26ced4b9c98d11681"
UNRESOLVED = (
    "native_model_execution_identity",
    "native_target_feature_numeric_parity",
    "full_drafter_mask_position_and_kv_parity",
    "initial_sample_terminal_emission_and_request_completeness",
)


def checked(locator):
    if not isinstance(locator, dict) or set(locator) != {"path", "sha256"}:
        raise ValueError("diagnostic artifact requires exact path/SHA pin")
    path = Path(locator["path"])
    if sha256(path) != locator["sha256"]:
        raise ValueError("diagnostic artifact SHA differs: " + str(path))
    return path


def require_diagnostic_capture(manifest, *, production=False):
    if production:
        raise ValueError("ineligible diagnostic bundle cannot grant production readiness")
    if (
        manifest.get("schema") != "recurrent_binary_capture_v1"
        or manifest.get("split") != "train"
        or manifest.get("training_eligible") is not False
        or manifest.get("readiness") != "preparation_only"
        or tuple(manifest.get("unverified_gates", ())) != UNRESOLVED
    ):
        raise ValueError("preserve original TRAIN diagnostic ineligibility and all four gates")


def authenticate(plan, bits):
    if (
        plan.get("schema") != "eagle_qat_cpu_diagnostic_plan_v1"
        or plan.get("artifact_kind") != "development_CPU"
        or plan.get("optimizer_updates") != 0
        or type(bits) is not int
        or bits not in (1, 8)
    ):
        raise ValueError("direct CPU development plan required")
    if plan.get("limits") != {
        "max_rss_bytes": 16 * 1024**3,
        "prelaunch_free_bytes": 12 * 1024**3,
        "live_free_bytes": 4 * 1024**3,
        "max_seconds": 600,
    }:
        raise ValueError("bounded CPU diagnostic limits differ from reviewed plan")
    for name, digest in plan["source_files"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("current source differs from diagnostic pin: " + name)
    capture_path = checked(plan["capture"])
    manifest = json.loads(capture_path.read_text())
    require_diagnostic_capture(manifest)
    original_train = checked(plan["original_train"])
    prompts = checked(plan["capture_prompts"])
    original_rows = [json.loads(line) for line in original_train.read_text().split("\n") if line]
    selected = [json.loads(line) for line in prompts.read_text().split("\n") if line]
    if (
        len(original_rows) != 96
        or len(selected) != 1
        or [row for row in original_rows if row["id"] == selected[0]["id"]] != selected
    ):
        raise ValueError("diagnostic prompt differs from exact frozen original TRAIN record")
    if manifest["prompts_sha256"] != plan["capture_prompts"]["sha256"]:
        raise ValueError("diagnostic capture prompt pin differs")
    contract = json.loads(checked(plan["initialization_contract"]).read_text())
    fit = json.loads(checked(plan["original_fit_report"]).read_text())
    checked(plan["calibration_manifest"])
    checked(plan["calibration_operands"])
    if (
        contract.get("schema") != "eagle_fusion_fixed_reference_latents_v1"
        or contract.get("optimizer_updates") != 0
        or contract["original_fit_report_sha256"] != plan["original_fit_report"]["sha256"]
        or fit.get("schema") != "saved_eagle_fusion_preparation_v1"
        or fit["ancestry"]["manifest_sha256"] != plan["calibration_manifest"]["sha256"]
        or fit["ancestry"]["operands_sha256"] != plan["calibration_operands"]["sha256"]
    ):
        raise ValueError("scale-only initializer original TRAIN calibration ancestry differs")
    record = contract["artifacts"][f"control-a{bits}"]
    if record["source_artifact_sha256"] != fit["profiles"][str(bits)]["control_sha256"]:
        raise ValueError("scale-only source artifact differs from original fit control")
    locator = {
        "path": record["path"],
        "sha256": record["sha256"],
        "activation_bits": record["activation_bits"],
        "latent_initialization": {
            key: record["latent_initialization"][key]
            for key in ("policy", "reference_kind", "reference_sha256")
        },
    }
    if locator != plan["initializations"][str(bits)]:
        raise ValueError("selected profile must use pinned scale-only reference-half control")
    if locator["latent_initialization"]["policy"] != "preserve_reference_magnitudes":
        raise ValueError("reference magnitude policy required")
    if locator["latent_initialization"]["reference_kind"] != "eagle_fixed_reference_0.5":
        raise ValueError("selected EAGLE initializer must retain historical reference-half")
    checked({key: locator[key] for key in ("path", "sha256")})
    return capture_path, prompts, selected[0], manifest, locator


def official_classes(source):
    """Execute unmodified pinned leaf modules; skip unrelated compressor imports.

    Normal package initialization imports datasets, absent in the local runtime.
    Private namespace paths keep those unexecuted parent initializers explicit;
    class definitions and relative base-class imports are the original source.
    """
    checkout = Path(source["path"])
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if revision != ANGELSLIM_REVISION or source["revision"] != revision:
        raise ValueError("official drafter source revision differs")
    base = checkout / "angelslim/compressor/speculative/inference/models/eagle3"
    for name, digest in source["files"].items():
        if sha256(base / name) != digest:
            raise ValueError("official drafter leaf source SHA differs")
    package = "_eagle_cpu_pinned_source"
    for name, path in ((package, base), (package + ".draft", base / "draft")):
        if name in sys.modules:
            if sys.modules[name].__path__ != [str(path)]:
                raise ValueError("mixed official source namespace")
        else:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    config = importlib.import_module(package + ".configuration_eagle3_model")
    draft = importlib.import_module(package + ".draft.llama3_eagle3")
    if Path(draft.__file__).resolve() != (base / "draft/llama3_eagle3.py").resolve():
        raise ValueError("official class import escaped pinned leaf source")
    return config.Eagle3Config, draft.Llama3Eagle3Drafter


def smoke(adapter, batch, audit, optimizer, *, resource_observer=None):
    """Existing recurrent forward and gradient gates, without a trainer lifecycle."""
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    expected = {id(p) for module in adapter.linears.values() for p in module.parameters()}
    if {id(p) for p in owned} != expected or len(owned) != len(expected) or optimizer.state:
        raise ValueError("optimizer must own exact selected signs/scales and empty state")
    versions = [p._version for p in owned]
    optimizer.zero_grad(set_to_none=True)
    observer = ObservedAdapter(adapter)
    execution = {}
    with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("update forbidden")):
        with shared_round_hard_signs(adapter.linears):
            logits = forward_torch_round(
                batch,
                observer,
                adapter.linears["lm_head"].out_features,
                optimize_cache=True,
                optimize_head=True,
                context_chunk_size=64,
                execution_metadata=execution,
            )
        later = later_gradient(logits, audit, observer)
        execution.update(
            optimize_cache_requested=True,
            optimize_head_requested=True,
            actual_context_cache_calls=observer.context_cache_calls,
        )
        if any(
            later[key] is None or later[key] <= 0
            for key in (
                "later_state_gradient_norm",
                "later_k_gradient_norm",
                "later_v_gradient_norm",
            )
        ):
            raise ValueError("later state/K/V gradient gate failed")
        mask = torch.tensor(audit.ce_mask, dtype=torch.bool)
        loss = torch.nn.functional.cross_entropy(
            logits[mask], torch.tensor(audit.draft_labels)[mask]
        )
        loss.backward()
    if not torch.isfinite(loss) or any(
        p.grad is None or not torch.isfinite(p.grad).all() or not bool((p.grad != 0).any())
        for p in owned
    ):
        raise ValueError("all-nine finite/nonzero gradient gate failed")
    if optimizer.state or versions != [p._version for p in owned]:
        raise ValueError("post-initialization parameters or optimizer state moved")
    result = {
        "loss_diagnostic_only": float(loss.detach()),
        "hard_forward": True,
        "optimizer_updates": 0,
        "optimizer_moment_tensors": 0,
        "selected_gradient_tensors_finite_nonzero": len(owned),
        "parameter_versions_after_initialization_unchanged": True,
        "supervised_rows": int(mask.sum()),
        "logits_shape": list(logits.shape),
        "execution": execution,
        **later,
    }
    if resource_observer is not None:
        result["resources_gradients_resident"] = resource_observer()
    optimizer.zero_grad(set_to_none=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--activation-bits", type=int, choices=(1, 8), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--import-probe", action="store_true")
    args = parser.parse_args()
    plan = json.loads(checked({"path": str(args.plan), "sha256": args.plan_sha256}).read_text())
    if args.output_dir.exists():
        raise ValueError("preserve prior evidence; output directory must be new")
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "report.json"
    start = time.monotonic()
    observed = memory(plan["memory_python"], os.getpid())
    report = {
        "schema": "eagle_qat_cpu_diagnostic_v1",
        "artifact_kind": "development_CPU",
        "status": "RUNNING",
        "plan_sha256": args.plan_sha256,
        "activation_bits": args.activation_bits,
        "training_eligible": False,
        "unresolved_gates": list(UNRESOLVED),
        "production_readiness": "PENDING",
        "warm_lifecycle": "NOT_EXERCISED",
        "optimizer_updates": 0,
        "optimizer_moment_tensors": 0,
        "source_files": plan["source_files"],
        "before": observed,
        "limits": plan["limits"],
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "hardware": {
            "platform": platform.platform(),
            "device": "CPU",
            "torch": str(torch.__version__),
            "student_masters": "F32",
            "KV": "F16",
            "source_weights": "original BF16 converted to source F16 then promoted to F32",
            "teacher_embedding": "read-only target F16 GGUF mmap row copies",
        },
        "scope": (
            "one original TRAIN diagnostic round with four unresolved historical gates; "
            "source-class + native step adapter development only; "
            "no production/quality/CUDA/warm proof"
        ),
    }
    peak, minimum = observed["rss_bytes"], observed["host_available_bytes"]
    stop = threading.Event()
    lock = threading.Lock()

    def publish(status, **extra):
        with lock:
            report.update(
                status=status,
                elapsed_seconds=time.monotonic() - start,
                peak_rss_bytes=peak,
                minimum_host_available_bytes=minimum,
                **extra,
            )
            atomic_json(report_path, report)

    def watch():
        nonlocal peak, minimum
        while not stop.wait(0.5):
            try:
                sample = memory(plan["memory_python"], os.getpid())
                peak = max(peak, sample["rss_bytes"])
                minimum = min(minimum, sample["host_available_bytes"])
                if (
                    peak > plan["limits"]["max_rss_bytes"]
                    or minimum < plan["limits"]["live_free_bytes"]
                    or time.monotonic() - start > plan["limits"]["max_seconds"]
                ):
                    publish(
                        "FAIL",
                        error="declared RSS/available/time bound violated",
                        failed_resource=sample,
                    )
                    os._exit(77)
            except BaseException as error:
                publish("FAIL", error=f"memory observer: {error}")
                os._exit(78)

    if observed["host_available_bytes"] < plan["limits"]["prelaunch_free_bytes"]:
        publish("FAIL", error="prelaunch host floor violated")
        raise RuntimeError("prelaunch host floor violated")
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        torch.set_num_threads(4)
        capture_path, prompts, selected, manifest, locator = authenticate(
            plan, args.activation_bits
        )
        Config, Drafter = official_classes(plan["angelslim_source"])
        if args.import_probe:
            publish(
                "PASS_IMPORT_ONLY_NO_WEIGHTS",
                official_class_source=str(plan["angelslim_source"]["path"]),
            )
            return
        from safetensors.torch import load_file

        config = Config(**json.loads(checked(plan["draft_config"]).read_text()))
        config.rope_scaling = None
        with torch.device("meta"):
            model = Drafter(
                config, load_emb=False, total_tokens=60, depth=5, top_k=10, threshold=1.0
            )
        state = load_file(str(checked(plan["draft_weights"])), device="cpu")
        state = {
            name: value.to(torch.float16) if value.is_floating_point() else value
            for name, value in state.items()
        }
        incompatible = model.load_state_dict(state, strict=False, assign=True)
        if (
            set(incompatible.missing_keys) != {"embed_tokens.weight"}
            or incompatible.unexpected_keys
        ):
            raise ValueError("original official draft checkpoint inventory differs")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for path in CANDIDATE_D_BASE_TO_PATH.values():
            if not torch.equal(model.get_submodule(path).weight, state[path + ".weight"]):
                raise ValueError("original private dense drafter projection differs")
        del state
        operands = FrozenOperands(
            checked(plan["target_gguf"]),
            checked(plan["base_gguf"]),
            target_sha256=plan["target_gguf"]["sha256"],
            draft_sha256=plan["base_gguf"]["sha256"],
        )
        bind_frozen_norms(model, operands.norm_arrays)
        qat = JointQATConfig(W1AxContract(args.activation_bits), device="cpu", objective="hard_ce")
        # No target parameters are loaded: frozen embedding lookup is file-backed,
        # and labels are the original captured verifier result at exact prefixes.
        frozen_owner = torch.nn.Module()
        linears = install_joint_linears(model, frozen_owner, qat)
        if any(
            not torch.equal(module.latent_sign.abs(), torch.full_like(module.latent_sign, 0.5))
            for module in linears.values()
        ):
            raise ValueError("all-nine historical reference-half initialization differs")
        untouched = {
            name: tuple(p._version for p in module.parameters())
            for name, module in linears.items()
            if name != "fc"
        }
        initialization = calibration(locator, qat.contract)
        initialized = apply_binary_initialization(linears, initialization)
        validate_initializer_reference(initialized, locator, "eagle")
        if (
            not torch.equal(linears["fc"].latent_sign, initialization["fc"][0])
            or not torch.equal(linears["fc"].initial_scale, initialization["fc"][1])
            or any(
                tuple(p._version for p in linears[name].parameters()) != versions
                for name, versions in untouched.items()
            )
        ):
            raise ValueError("sparse calibrated FC/untouched non-FC state differs")
        del initialization
        parameter_versions = {name: p._version for name, p in model.named_parameters()}
        capture = load_audited_capture(
            capture_path, prompts, plan["capture_prompts"]["sha256"], expected_prompt_count=1
        )
        import numpy as np

        offsets = tuple(int(v) for v in np.load(capture_path.parent / manifest["offsets"]["path"]))
        if tuple(int(v) for v in model.d2t.cpu()) != offsets or not np.array_equal(
            model.t2d.cpu().numpy(), np.load(capture_path.parent / manifest["t2d"]["path"])
        ):
            raise ValueError("actual private pruned head maps differ from captured verifier maps")
        round_data = capture.round_inputs(selected["id"], 0)
        batch = ProviderRound(
            round_data.anchor,
            round_data.rows,
            round_data.prefix_token_ids,
            round_data.raw_target_features,
            round_data.feature_positions,
            plan["capture"]["sha256"],
        )
        # Structural audit only; ineligible bundle status is deliberately retained.
        provider = SimpleNamespace(
            d2t_offsets=offsets,
            target_vocab_size=151936,
            draft_vocab_size=32000,
            allowed_prompt_ids={selected["id"]},
            split="train",
            max_depth=5,
        )
        audit = audit_provider_round(batch, provider)
        adapter = NativeStepAdapter(model, embedding_lookup=operands)
        optimizer = joint_optimizer(linears, qat)
        if {id(p) for p in model.parameters() if p.requires_grad} != {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }:
            raise ValueError("optimizer ownership includes frozen or misses selected source paths")
        gradient_report = smoke(
            adapter,
            batch,
            audit,
            optimizer,
            resource_observer=lambda: memory(plan["memory_python"], os.getpid()),
        )
        report.update(
            smoke=gradient_report,
            initializer=locator,
            initializer_application=initialized,
            original_private_projection_values_exact=True,
            pruned_maps_exact=True,
            target_parameters_loaded=0,
            unused_embedding=(
                "meta; never executed; FrozenOperands performs actual borrowed embedding lookup"
            ),
            prompt_id=selected["id"],
            context_features_shape=list(batch.raw_target_features.shape),
            capture_sha256=plan["capture"]["sha256"],
        )
        checkpoint, metadata = (
            args.output_dir / "state.npz",
            args.output_dir / "state-manifest.json",
        )
        saved = save_joint_checkpoint(
            linears, qat, plan["base_gguf"]["sha256"], checkpoint, metadata
        )
        from export_recurrent_binary import export_model

        exported = export_model(
            Path(plan["base_gguf"]["path"]),
            checkpoint,
            metadata,
            args.output_dir / "diagnostic.gguf",
        )
        if optimizer.state or any(
            p._version != parameter_versions[name] for name, p in model.named_parameters()
        ):
            raise ValueError("export moved parameters or allocated optimizer state")
        final = memory(plan["memory_python"], os.getpid())
        rss_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak = max(
            peak, final["rss_bytes"], rss_peak if sys.platform == "darwin" else rss_peak * 1024
        )
        minimum = min(minimum, final["host_available_bytes"])
        if peak > plan["limits"]["max_rss_bytes"] or minimum < plan["limits"]["live_free_bytes"]:
            raise ValueError("final measured resource bound failed")
        publish(
            "PASS_CPU_DIAGNOSTIC",
            after=final,
            checkpoint=saved,
            export=exported,
            native_graph_admitted=False,
            optimizer_updates=0,
            optimizer_moment_tensors=0,
        )
    except BaseException as error:
        publish("FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stop.set()
        watcher.join(timeout=12)


if __name__ == "__main__":
    main()
