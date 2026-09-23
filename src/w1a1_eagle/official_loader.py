"""Load the pinned AngelSlim EAGLE-3 runtime with strict checkpoint checks.

AngelSlim's pinned drafter forward/generation code is reused. This loader fixes
only a config compatibility issue: recent Transformers versions synthesize a
default RoPE dictionary when the pinned raw config has ``rope_scaling: null``;
the pinned drafter expects ``None`` for that case.
"""

import json
from importlib.metadata import distribution
from pathlib import Path
from typing import Any


def require_angelslim_revision(expected_revision: str) -> None:
    """Reject an unpinned or different AngelSlim installation."""
    metadata = distribution("angelslim").read_text("direct_url.json")
    if metadata is None:
        raise RuntimeError("AngelSlim must be installed from the pinned Git revision")
    source = json.loads(metadata)
    actual = source.get("vcs_info", {}).get("commit_id")
    if actual != expected_revision:
        raise RuntimeError(
            f"AngelSlim revision mismatch: expected {expected_revision}, got {actual}"
        )


def load_official_eagle3(
    target_dir: str | Path,
    draft_dir: str | Path,
    *,
    angelslim_revision: str,
    total_token: int,
    depth: int,
    top_k: int,
    threshold: float,
    target_load_kwargs: dict[str, Any] | None = None,
):
    """Return an official AngelSlim ``Eagle3Model`` with validated draft weights.

    The caller downloads the immutable snapshots first. This path is scoped to
    the pinned AngelSlim/Qwen3-4B_eagle3 checkpoint, whose own token embedding
    is absent; the official drafter loads that matrix from the target snapshot.
    """
    require_angelslim_revision(angelslim_revision)

    from angelslim.compressor.speculative.inference.models.eagle3.configuration_eagle3_model import (  # noqa: E501
        Eagle3Config,
    )
    from angelslim.compressor.speculative.inference.models.eagle3.draft import (  # noqa: E501
        Llama3Eagle3Drafter,
    )
    from angelslim.compressor.speculative.inference.models.eagle3.eagle3_model import (  # noqa: E501
        Eagle3Model,
        ModelLoader,
    )
    from transformers import AutoTokenizer

    target_dir = Path(target_dir).resolve()
    draft_dir = Path(draft_dir).resolve()
    for label, directory in (("target", target_dir), ("draft", draft_dir)):
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(f"{label} config is missing from {directory}")
    if not (draft_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"draft model.safetensors is missing from {draft_dir}")

    raw_config = json.loads((draft_dir / "config.json").read_text())
    if raw_config.get("rope_scaling") is not None:
        raise ValueError("this loader only supports the pinned drafter's unscaled RoPE config")

    base_model = ModelLoader.load_base_model(str(target_dir), **(target_load_kwargs or {}))
    tokenizer = AutoTokenizer.from_pretrained(str(target_dir), use_fast=False)
    tokenizer.stop_think_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    tokenizer.step_split_ids = []

    config = Eagle3Config.from_pretrained(str(draft_dir / "config.json"))
    config.rope_scaling = None
    device = next(base_model.parameters()).device
    state_dict = ModelLoader.load_eagle_state_dict(str(draft_dir), device)
    for name in ("d2t", "t2d", "fc.weight", "lm_head.weight"):
        if name not in state_dict:
            raise RuntimeError(f"draft checkpoint is missing {name}")

    eagle_layer = Llama3Eagle3Drafter(
        config,
        total_tokens=total_token,
        depth=depth,
        top_k=top_k,
        threshold=threshold,
        path=str(target_dir),
        load_emb=True,
        early_stop_method=None,
    )
    incompatible = eagle_layer.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != {"embed_tokens.weight"} or unexpected:
        raise RuntimeError(
            f"unexpected draft checkpoint keys: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    eagle_layer.to(device=device, dtype=base_model.dtype)
    eagle_layer.init_tree()
    return Eagle3Model(base_model, tokenizer, eagle_layer)
