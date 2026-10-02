"""Emit bounded real-API evidence, without native/model/corpus execution."""

import json
import tempfile
from pathlib import Path

import check_continuous_w1ax_readiness as gate
import torch
from adapter import construct_and_load, preflight_checkpoint
from test_recipe import NAMES, graph_logits, produce, tiny

from w1a1_eagle.qat_state import deployment_state_sha256
from w1a1_eagle.recurrent_qat import JointQATConfig, W1AxContract, install_joint_linears


def audit():
    rows = []
    torch.manual_seed(1791)
    for bits in (8, 1):
        for name in NAMES:
            with tempfile.TemporaryDirectory() as tmp:
                source, checkpoint, manifest = produce(Path(tmp), bits, name)
                original = install_joint_linears(*tiny(), JointQATConfig(W1AxContract(bits)))
                try:
                    gate._load_checkpoint(checkpoint, manifest, original, bits, "a" * 64)
                    error = None
                except ValueError as exc:
                    error = str(exc)
                prepared = preflight_checkpoint(checkpoint, manifest, bits, "a" * 64)
                loaded = construct_and_load(prepared, *tiny())
                max_error = 0.0
                for path in source:
                    x = torch.linspace(-0.81, 0.91, 3 * source[path].in_features).reshape(3, -1)
                    with torch.no_grad():
                        expected, actual = source[path](x), loaded[path](x)
                    torch.testing.assert_close(expected, actual, atol=3e-5, rtol=3e-6)
                    max_error = max(max_error, float((actual - expected).abs().max()))
                source_logits, replay_logits = graph_logits(source), graph_logits(loaded)
                torch.testing.assert_close(source_logits, replay_logits, atol=3e-5, rtol=3e-6)
                declared = json.loads(manifest.read_text())
                source_digest = deployment_state_sha256(source)
                replay_digest = deployment_state_sha256(loaded)
                assert source_digest == replay_digest
                rows.append(
                    {
                        "profile": name,
                        "bits": bits,
                        "schema_version": declared["schema_version"],
                        "fixed_constructor_error": error,
                        "source_deployment_sha256": source_digest,
                        "replay_deployment_sha256": replay_digest,
                        "checkpoint_sha256": prepared.checkpoint_sha256,
                        "manifest_sha256": prepared.manifest_sha256,
                        "max_projection_output_abs_error": max_error,
                        "max_recurrent_logits_abs_error": float(
                            (source_logits - replay_logits).abs().max()
                        ),
                        "deployed_recipe_fields": {
                            key: declared.get(key)
                            for key in (
                                "activation_quantizers",
                                "fusion_correction",
                                "affine_weights",
                            )
                        },
                    }
                )
    return {
        "schema": "development_recipe_synthetic_proof_v1",
        "hardware": "Apple Silicon CPU; no Metal/CUDA execution or discovery",
        "torch": torch.__version__,
        "rows": rows,
        "gate": {"atol": 3e-5, "rtol": 3e-6, "exact_deployment_digest": True},
        "native_quality_or_performance_claim": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
