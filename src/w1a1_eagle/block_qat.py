"""Source-bound Qwen3 DSpark/DFlash hard-forward QAT foundation.

Supported geometry is the released five-layer, seven-slot, anchor-first profile.
The graph follows llama.cpp dflash.cpp at fcdf5822: FC then encoder RMSNorm,
per-layer injected K/V without attention norm, attached evolving noise states,
Q/K per-head norm + NeoX RoPE, F16 cache boundaries, and bidirectional noise.
PyTorch F32 floating exceptions are a numeric reference, not CUDA latency proof;
current native model trajectories must admit their bounded numeric discrepancy.
No target parameters or approximate borrowed embedding/head enter this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .recurrent_qat import (
    FIXED_A8_ARITHMETIC_REVISION,
    RowBinaryLinear,
    W1AxContract,
    fixed_activation_codes,
    shared_round_hard_signs,
)

NATIVE_SOURCE = "fcdf5822c5b78f9dbcfd1f5c7106f09c3f0c9b1a"
TAPS = (2, 10, 18, 26, 34)


@dataclass(frozen=True)
class BlockQATConfig:
    family: str
    activation_bits: int
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    num_layers: int = 5
    block_size: int = 7
    mask_token_id: int = 151669
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    profile: str = "ffn15_fusion"
    objective: str = "hard_ce"
    conditioning: str = "native_greedy"
    cross_entropy_weight: float = 1.0
    probability_l1_weight: float = 1.0
    depth_decay: float = 1.0
    sign_lr: float = 1e-3
    scale_lr: float = 1e-5
    max_grad_norm: float = 1.0
    seed: int = 8101
    latent_initialization: str = "preserve_reference_magnitudes"
    initialization_encoding: str = "policy_latents"
    optimizer_backend: str = "serial"

    def __post_init__(self):
        if (
            self.family not in {"dspark", "dflash"}
            or type(self.activation_bits) is not int
            or self.activation_bits not in {1, 8}
        ):
            raise ValueError("block family/activation width unsupported")
        if self.optimizer_backend not in {"serial", "fused_fp32_probe"}:
            raise ValueError("unknown block optimizer execution backend")
        if self.initialization_encoding not in {"policy_latents", "hard_signs"}:
            raise ValueError("initializer encoding unsupported")
        if self.latent_initialization not in {"preserve_reference_magnitudes", "unit_probe"}:
            raise ValueError("explicit calibrated latent policy required")
        if self.profile not in {"ffn15", "ffn15_fusion"}:
            raise ValueError("only fifteen FFN projections and optional fusion admitted")
        if self.num_layers != 5 or self.block_size != 7:
            raise ValueError("only released five-layer/seven-slot anchor-first geometry supported")
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "vocab_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("positive integer model geometry required")
        if self.head_dim % 2 or self.num_heads % self.num_kv_heads:
            raise ValueError("even rotary width and integral GQA ratio required")
        if type(self.mask_token_id) is not int or not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("mask token outside private vocabulary")
        for name in ("norm_eps", "rope_theta", "sign_lr", "scale_lr", "max_grad_norm"):
            if (
                isinstance(getattr(self, name), bool)
                or not math.isfinite(getattr(self, name))
                or getattr(self, name) <= 0
            ):
                raise ValueError("finite positive arithmetic/optimizer parameters required")
        if self.conditioning not in {"native_greedy", "captured_prefix"}:
            raise ValueError("explicit native_greedy or captured_prefix conditioning required")
        if self.objective not in {"hard_ce", "full_probability_l1"}:
            raise ValueError("unsupported block objective")
        if self.objective == "full_probability_l1" and self.family != "dspark":
            raise ValueError("probability L1 profile belongs to DSpark")
        if not math.isfinite(self.cross_entropy_weight) or self.cross_entropy_weight <= 0:
            raise ValueError("CE coefficient must be finite and positive")
        if not math.isfinite(self.probability_l1_weight) or self.probability_l1_weight < 0:
            raise ValueError("probability L1 weight invalid")
        if not math.isfinite(self.depth_decay) or not 0 < self.depth_decay <= 1:
            raise ValueError("depth decay must be in (0,1]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed outside NumPy/torch shared range")


def _rms(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


def _rope(x: Tensor, positions: Tensor, theta: float) -> Tensor:
    width = x.shape[-1]
    frequency = torch.tensor(theta, dtype=torch.float32, device=x.device).pow(
        -torch.arange(0, width, 2, dtype=torch.float32, device=x.device) / width
    )
    angle = positions.float()[:, None] * frequency
    angle = torch.cat((angle, angle), -1)[:, None, :]
    half = width // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), -1)
    return x * angle.cos() + rotated * angle.sin()


class _FrozenLinear(nn.Module):
    """Own an exact F32 copy of source storage; no target/teacher aliases."""

    def __init__(self, weight: Tensor):
        super().__init__()
        self.register_buffer("weight", weight.detach().float().clone())

    def forward(self, x):
        return F.linear(x, self.weight)


class BlockLayer(nn.Module):
    def __init__(self, tensors, index, config, binary_initializer):
        super().__init__()
        self.config = config
        prefix = f"blk.{index}."
        h, hd, nq, nk = (config.hidden_size, config.head_dim, config.num_heads, config.num_kv_heads)
        shapes = {
            "attn_norm": (h,),
            "attn_q_norm": (hd,),
            "attn_k_norm": (hd,),
            "ffn_norm": (h,),
            "attn_q": (nq * hd, h),
            "attn_k": (nk * hd, h),
            "attn_v": (nk * hd, h),
            "attn_output": (h, nq * hd),
            "ffn_gate": (config.intermediate_size, h),
            "ffn_up": (config.intermediate_size, h),
            "ffn_down": (h, config.intermediate_size),
        }
        for name, shape in shapes.items():
            weight = _tensor(tensors, prefix + name + ".weight", shape)
            if name.startswith("ffn_") and name != "ffn_norm":
                module = _binary(weight, config, binary_initializer, prefix + name)
                setattr(self, name, module)
            elif len(shape) == 2:
                setattr(self, name, _FrozenLinear(weight))
            else:
                self.register_buffer(name, weight.detach().float().clone())

    def project_context(self, encoded, positions):
        cfg = self.config
        k = self.attn_k(encoded).view(-1, cfg.num_kv_heads, cfg.head_dim)
        v = self.attn_v(encoded).view(-1, cfg.num_kv_heads, cfg.head_dim)
        k = _rope(_rms(k, self.attn_k_norm, cfg.norm_eps), positions, cfg.rope_theta)
        return k.half().float(), v.half().float()

    def forward(self, state, context_kv, positions, allowed):
        cfg = self.config
        norm = _rms(state, self.attn_norm, cfg.norm_eps)
        q = self.attn_q(norm).view(-1, cfg.num_heads, cfg.head_dim)
        k = self.attn_k(norm).view(-1, cfg.num_kv_heads, cfg.head_dim)
        v = self.attn_v(norm).view(-1, cfg.num_kv_heads, cfg.head_dim)
        q = _rope(_rms(q, self.attn_q_norm, cfg.norm_eps), positions, cfg.rope_theta)
        k = _rope(_rms(k, self.attn_k_norm, cfg.norm_eps), positions, cfg.rope_theta)
        ck, cv = context_kv
        noise_kv = (k, v)
        k = torch.cat((ck, k.half().float()), 0)
        v = torch.cat((cv, v.half().float()), 0)
        # Group dimensions preserve the native query/head to KV/head mapping;
        # do not materialize repeated K/V storage or detach later state graphs.
        groups = cfg.num_heads // cfg.num_kv_heads
        q = q.view(cfg.block_size, cfg.num_kv_heads, groups, cfg.head_dim)
        score = torch.einsum("qhgd,khd->hgqk", q, k) / math.sqrt(cfg.head_dim)
        score = score.masked_fill(~allowed[None, None], -torch.inf)
        probs = F.softmax(score, -1)
        attended = torch.einsum("hgqk,khd->qhgd", probs, v)
        residual = state + self.attn_output(attended.reshape(cfg.block_size, -1))
        norm = _rms(residual, self.ffn_norm, cfg.norm_eps)
        return (residual + self.ffn_down(F.silu(self.ffn_gate(norm)) * self.ffn_up(norm)), noise_kv)


def _tensor(tensors: Mapping, name: str, shape: tuple) -> Tensor:
    value = tensors.get(name)
    if (
        not isinstance(value, Tensor)
        or tuple(value.shape) != shape
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("missing/invalid source operand: " + name)
    return value


def _binary(weight, config, initialization, name):
    module = BlockBinaryLinear(
        weight, weight.float().abs().mean(-1), W1AxContract(config.activation_bits)
    )
    if initialization is not None and name in initialization:
        from .qat_initialization import apply_binary_initialization

        module.initialization_report = apply_binary_initialization(
            {name: module},
            {name: initialization[name]},
            policy=config.latent_initialization,
            encoding=config.initialization_encoding,
        )
    return module


class BlockBinaryLinear(RowBinaryLinear):
    """Native integer-dot scaling order; retain reference dequantized STE."""

    def forward(self, input):
        surrogate = super().forward(input)
        if self.contract.activation_bits == 1:
            return surrogate
        with torch.no_grad():
            raw = input.float()
            codes, scale, _ = fixed_activation_codes(raw, 8)
            observer = getattr(self, "_diagnostic_a8", None)
            if observer is not None:
                observer(codes)
            signs = (
                torch.where(self.latent_sign < 0, -1.0, 1.0)
                if self._round_hard_signs is None
                else self._round_hard_signs.detach()
            )
            native = (F.linear(codes, signs) * self.effective_scales()) * scale
        return native.detach() + (surrogate - surrogate.detach())


@dataclass
class BlockOutput:
    logits: Tensor
    predecessor_ids: Tensor
    context_kv: tuple
    layer_states: tuple
    noise_kv: tuple
    conditioning: str


class BlockDrafter(nn.Module):
    def __init__(self, tensors: Mapping, config: BlockQATConfig, *, binary_initializer=None):
        super().__init__()
        self.config = config
        supported = {f"blk.{i}.{n}" for i in range(5) for n in ("ffn_gate", "ffn_up", "ffn_down")}
        if config.profile == "ffn15_fusion":
            supported.add("fc")
        if binary_initializer is not None and set(binary_initializer) - supported:
            raise ValueError("calibrated initializer contains unselected projections")
        h, v = config.hidden_size, config.vocab_size
        self.register_buffer(
            "token_embd", _tensor(tensors, "token_embd.weight", (v, h)).detach().float().clone()
        )
        self.register_buffer(
            "output", _tensor(tensors, "output.weight", (v, h)).detach().float().clone()
        )
        fc = _tensor(tensors, "fc.weight", (h, 5 * h))
        self.fc = (
            _binary(fc, config, binary_initializer, "fc")
            if config.profile == "ffn15_fusion"
            else _FrozenLinear(fc)
        )
        self.register_buffer(
            "output_norm_enc",
            _tensor(tensors, "enc.output_norm.weight", (h,)).detach().float().clone(),
        )
        self.register_buffer(
            "output_norm", _tensor(tensors, "output_norm.weight", (h,)).detach().float().clone()
        )
        self.layers = nn.ModuleList(
            [BlockLayer(tensors, i, config, binary_initializer) for i in range(config.num_layers)]
        )
        if config.family == "dspark":
            w1 = tensors.get("markov_w1.weight")
            if not isinstance(w1, Tensor) or w1.ndim != 2 or w1.shape[0] != v:
                raise ValueError("DSpark requires its own full-vocabulary Markov codebook")
            self.register_buffer(
                "markov_w1",
                _tensor(tensors, "markov_w1.weight", tuple(w1.shape)).detach().float().clone(),
            )
            self.register_buffer(
                "markov_w2",
                _tensor(tensors, "markov_w2.weight", tuple(w1.shape)).detach().float().clone(),
            )
            scale = tensors.get("markov_w2.scale", torch.ones(()))
            if (
                not isinstance(scale, Tensor)
                or scale.numel() != 1
                or not bool(torch.isfinite(scale).all())
            ):
                raise ValueError("Markov projection scale invalid")
            self.register_buffer("markov_scale", scale.detach().float().clone().reshape(()))
        expected = {id(p) for m in self.binary_linears().values() for p in m.parameters()}
        if {id(p) for p in self.parameters() if p.requires_grad} != expected:
            raise ValueError("only selected binary signs/scales may be trained")

    def binary_linears(self):
        result = {}
        for i, layer in enumerate(self.layers):
            for name in ("ffn_gate", "ffn_up", "ffn_down"):
                result[f"blk.{i}.{name}"] = getattr(layer, name)
        if self.config.profile == "ffn15_fusion":
            result["fc"] = self.fc
        return result

    def validate_batch(self, batch):
        cfg = self.config
        context = batch.context_features
        size = len(batch.prefix_tokens) - 1
        if context.shape != (size, 5, cfg.hidden_size) or context.dtype != torch.float32:
            raise ValueError("five ordered taps must cover the accepted prefix excluding anchor")
        if not bool(torch.isfinite(context).all()) or context.requires_grad:
            raise ValueError("native teacher features must be finite detached F32")
        if size < 1 or any(
            type(t) is not int or not 0 <= t < cfg.vocab_size for t in batch.prefix_tokens
        ):
            raise ValueError("accepted native token prefix invalid")
        inputs = batch.input_tokens
        if (
            inputs.dtype != torch.int64
            or inputs.shape != (cfg.block_size,)
            or int(inputs[0]) != batch.prefix_tokens[-1]
            or not bool((inputs[1:] == cfg.mask_token_id).all())
        ):
            raise ValueError("noise slots must be actual anchor then six trained mask tokens")
        if batch.positions.dtype != torch.int64 or not torch.equal(
            batch.positions.cpu(), torch.arange(size, size + cfg.block_size)
        ):
            raise ValueError("absolute anchor-first slot positions differ")
        if (
            batch.attention_allowed.dtype != torch.bool
            or batch.attention_allowed.shape != (cfg.block_size, size + cfg.block_size)
            or not bool(batch.attention_allowed.all())
        ):
            raise ValueError("released full context/bidirectional seven-slot noise mask required")
        if (
            batch.labels.dtype != torch.int64
            or batch.labels.shape != (cfg.block_size,)
            or batch.loss_mask.dtype != torch.bool
            or batch.loss_mask.shape != (cfg.block_size,)
        ):
            raise ValueError("full-vocabulary labels/loss mask layout invalid")
        selected = batch.labels[batch.loss_mask]
        if not selected.numel() or bool(((selected < 0) | (selected >= cfg.vocab_size)).any()):
            raise ValueError("supported full-vocabulary labels required")

    def forward(self, batch, *, return_proposals: int = 7):
        self.validate_batch(batch)
        if return_proposals not in {3, 7}:
            raise ValueError("supported proposal return caps are three and seven")
        cfg = self.config
        device = self.token_embd.device
        features = batch.context_features.to(device)
        positions = batch.positions.to(device)
        context_positions = torch.arange(features.shape[0], device=device)
        state = F.embedding(batch.input_tokens.to(device), self.token_embd)
        contexts, states, noise_kv = [], [], []
        with shared_round_hard_signs(self.binary_linears()):
            encoded = _rms(
                self.fc(features.reshape(features.shape[0], -1)), self.output_norm_enc, cfg.norm_eps
            )
            for layer in self.layers:
                context_kv = layer.project_context(encoded, context_positions)
                contexts.append(context_kv)
                state, projected_noise = layer(
                    state, context_kv, positions, batch.attention_allowed.to(device)
                )
                noise_kv.append(projected_noise)
                states.append(state)
        normalized = _rms(state, self.output_norm, cfg.norm_eps)
        logits = F.linear(normalized, self.output)
        predecessor = int(batch.prefix_tokens[-1])
        predecessors, columns = [], []
        for slot in range(cfg.block_size):
            predecessors.append(predecessor)
            column = logits[slot]
            if cfg.family == "dspark":
                column = (
                    column
                    + F.linear(self.markov_w1[predecessor], self.markov_w2) * self.markov_scale
                )
            columns.append(column)
            if cfg.conditioning == "captured_prefix":
                # Unavailable tail targets must never index the Markov codebook.
                # Their seven noise states remain computed but unsupported.
                label = int(batch.labels[slot])
                if 0 <= label < cfg.vocab_size:
                    predecessor = label
                elif bool(batch.loss_mask[slot]):
                    raise ValueError("captured Markov predecessor label invalid")
                else:
                    predecessor = cfg.mask_token_id
            else:
                predecessor = int(column.detach().argmax())
        # Always compute seven trained states/columns. The caller may return a
        # shorter proposal while the exact deployment graph remains invariant.
        return BlockOutput(
            torch.stack(columns),
            torch.tensor(predecessors, device=device),
            tuple(contexts),
            tuple(states),
            tuple(noise_kv),
            cfg.conditioning,
        )


def block_supported_mask(batch, config):
    mask = batch.loss_mask.clone()
    if config.family == "dspark" and config.conditioning == "captured_prefix":
        captured = getattr(batch, "predecessor_ids", None)
        if captured is not None:
            if captured.shape != (config.block_size,):
                raise ValueError("captured predecessor layout invalid")
            mask &= (captured.to(mask.device) >= 0) & (captured.to(mask.device) < config.vocab_size)
    return mask


def block_loss(output: BlockOutput, batch, config: BlockQATConfig, *, teacher_callback=None):
    """Full-head CE or source-style CE + probability L1; normalized valid tokens.

    Greedy DSpark teachers must describe actual sequential student predecessors.
    Static replay admits only the contiguous prefix before the first divergence.
    A callback receives complete current-student prefixes, returns exact full
    logits, and owns a frozen native target in a separate parameter scope.
    DFlash noise logits have no Markov predecessor dependency; captured targets
    retain their target-only teacher prefixes by the explicit block objective.
    """
    logits = output.logits
    device = logits.device
    labels = batch.labels.to(device).clone()
    mask = block_supported_mask(batch, config).to(device)
    captured = getattr(batch, "predecessor_ids", None)
    teacher = getattr(batch, "teacher_logits", None)
    if teacher_callback is not None:
        if config.conditioning != "native_greedy":
            raise ValueError("current-prefix teacher requires native greedy student conditioning")
        prefix = list(batch.prefix_tokens)
        rows = []
        for slot in range(config.block_size):
            row = teacher_callback(tuple(prefix))
            if (
                not isinstance(row, Tensor)
                or row.shape != (config.vocab_size,)
                or row.requires_grad
                or not bool(torch.isfinite(row).all())
            ):
                raise ValueError("native teacher callback must return detached exact full logits")
            rows.append(row.to(device))
            labels[slot] = row.argmax()
            prefix.append(int(logits[slot].detach().argmax()))
        teacher = torch.stack(rows)
    elif config.conditioning == "native_greedy":
        if not isinstance(captured, Tensor) or captured.shape != (config.block_size,):
            raise ValueError("native greedy static replay requires captured predecessor identity")
        match = output.predecessor_ids == captured.to(device)
        # Even if tokens happen to meet again, intervening prefix history differs.
        match = match.to(torch.int64).cumprod(0).bool()
        mask &= match
    if not bool(mask.any()):
        raise ValueError("no exact-prefix supported teacher labels")
    ce = F.cross_entropy(logits[mask], labels[mask], reduction="none")
    weights = config.depth_decay ** torch.arange(config.block_size, device=device).float()[mask]
    probability_l1 = ce.new_zeros(ce.shape)
    if config.objective == "full_probability_l1":
        if (
            not isinstance(teacher, Tensor)
            or teacher.shape != logits.shape
            or teacher.requires_grad
            or not bool(torch.isfinite(teacher).all())
        ):
            raise ValueError("DSpark probability L1 requires exact full-vocabulary logits")
        # Sum over complete vocabulary then normalize valid positions, not cells.
        probability_l1 = (
            (F.softmax(logits[mask], -1) - F.softmax(teacher.to(device)[mask].float(), -1))
            .abs()
            .sum(-1)
        )
    ce_sum = (ce * weights).sum()
    l1_sum = (probability_l1 * weights).sum()
    denominator = weights.sum()
    loss = (
        config.cross_entropy_weight * ce_sum + config.probability_l1_weight * l1_sum
    ) / denominator
    return loss, {
        "weighted_denominator": float(denominator.detach()),
        "ce": float((ce_sum / denominator).detach()),
        "probability_l1": float((l1_sum / denominator).detach()),
        "supervised_tokens": int(mask.sum()),
        "presented_tokens": int(batch.loss_mask.sum()),
        "prefix_mismatch_tokens": int((batch.loss_mask.to(device) & ~mask).sum()),
        "normalization": "weighted_valid_token_mean",
        "conditioning": output.conditioning,
    }


def block_optimizer(model):
    cfg = model.config
    linears = model.binary_linears()
    return torch.optim.AdamW(
        [
            {
                "params": [m.latent_sign for m in linears.values()],
                "lr": cfg.sign_lr,
                "family": "sign",
            },
            {
                "params": [m.scale_offset for m in linears.values()],
                "lr": cfg.scale_lr,
                "family": "scale",
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        foreach=False,
        **({"fused": True} if cfg.optimizer_backend == "fused_fp32_probe" else {}),
    )


def block_train_step(model, optimizer, batch, *, teacher_callback=None, update: bool = True):
    metrics, outputs = block_train_batch(
        model, optimizer, [batch], teacher_callback=teacher_callback, update=update
    )
    return metrics, outputs[0]


def block_train_batch(
    model,
    optimizer,
    batches,
    *,
    teacher_callback=None,
    update=True,
    return_outputs=True,
    diagnostics=False,
):
    """Microbatch-one accumulation with a single supported-token denominator.

    Captured-prefix support is known before forwards, so each graph is freed
    before the next microbatch. Greedy replay needs its actual forward mask;
    retain those graphs only for legacy multi-block diagnostic callers.
    """
    if not batches:
        raise ValueError("nonempty effective batch required")
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    expected = [p for p in model.parameters() if p.requires_grad]
    if len(owned) != len(expected) or {id(p) for p in owned} != {id(p) for p in expected}:
        raise ValueError("optimizer owns exactly student binary signs/scales")
    optimizer.zero_grad(set_to_none=True)
    weights = (
        model.config.depth_decay
        ** torch.arange(model.config.block_size, device=model.token_embd.device).float()
    )
    denominator = None
    if model.config.conditioning == "captured_prefix":
        denominator = sum(
            float(weights[block_supported_mask(b, model.config).to(weights.device)].sum())
            for b in batches
        )
        if denominator <= 0:
            raise ValueError("effective batch has no supported labels")
    outputs, records, retained = [], [], []
    for batch in batches:
        output = model(batch)
        loss, counters = block_loss(output, batch, model.config, teacher_callback=teacher_callback)
        if not loss.requires_grad or not bool(torch.isfinite(loss)):
            raise ValueError("finite differentiable block loss required")
        diagnostic = {}
        if diagnostics:
            from .block_diagnostics import slot_metrics

            diagnostic = slot_metrics(output, batch, model.config)
        records.append({**counters, **diagnostic, "loss": float(loss.detach())})
        if denominator is not None:
            (loss * (counters["weighted_denominator"] / denominator)).backward()
        else:
            retained.append(loss)
        # Do not keep graph tensors alive across the accumulation transaction.
        if return_outputs:
            outputs.append(
                BlockOutput(
                    output.logits.detach(),
                    output.predecessor_ids.detach(),
                    tuple((k.detach(), v.detach()) for k, v in output.context_kv),
                    tuple(s.detach() for s in output.layer_states),
                    tuple((k.detach(), v.detach()) for k, v in output.noise_kv),
                    output.conditioning,
                )
            )
        del output, loss
    if denominator is None:
        denominator = sum(r["weighted_denominator"] for r in records)
        sum(
            loss * r["weighted_denominator"] / denominator for loss, r in zip(retained, records)
        ).backward()
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in expected):
        raise ValueError("missing/nonfinite student gradient before optimizer update")
    norm = torch.nn.utils.clip_grad_norm_(
        expected, model.config.max_grad_norm, error_if_nonfinite=True
    )
    before = [m.latent_sign.detach() < 0 for m in model.binary_linears().values()]
    if update:
        optimizer.step()
        with torch.no_grad():
            for module in model.binary_linears().values():
                module.latent_sign.clamp_(-1, 1)
                module.project_scales_()
        if not all(bool(torch.isfinite(p).all()) for p in expected):
            raise ValueError("nonfinite student after optimizer update")
    modules = list(model.binary_linears().values())
    flips = sum(int(((m.latent_sign.detach() < 0) != old).sum()) for m, old in zip(modules, before))
    scales = torch.cat([m.effective_scales().detach().reshape(-1) for m in modules])
    signs = sum(m.latent_sign.numel() for m in modules)
    return {
        **{
            k: sum(r[k] for r in records)
            for k in ("supervised_tokens", "presented_tokens", "prefix_mismatch_tokens")
        },
        **{
            k: sum(r[k] * r["weighted_denominator"] for r in records) / denominator
            for k in ("loss", "ce", "probability_l1")
        },
        **(
            {
                field: [sum(r[field][i] for r in records) for i in range(model.config.block_size)]
                for field in (
                    "slot_supported",
                    "teacher_forced_top1_hits",
                    "captured_prefix_survival_supported",
                    "captured_prefix_survival_hits",
                )
            }
            if diagnostics
            else {}
        ),
        **({"diagnostic_semantics": records[0]["diagnostic_semantics"]} if diagnostics else {}),
        "weighted_denominator": denominator,
        "normalization": "weighted_valid_token_mean",
        "conditioning": model.config.conditioning,
        "effective_batch_blocks": len(batches),
        "gradient_norm": float(norm),
        "clipped": bool(norm > model.config.max_grad_norm),
        "sign_flips": flips,
        "sign_flip_fraction": flips / signs,
        "scale_min": float(scales.min()),
        "scale_mean": float(scales.mean()),
        "scale_max": float(scales.max()),
        "optimizer_updated": update,
    }, outputs


def block_contract(config):
    return {
        "schema": "block_qat_v1",
        "config": asdict(config),
        "native_source": NATIVE_SOURCE,
        "target_taps": list(TAPS),
        "first_prediction_slot": 0,
        "driver": "draft-dspark",
        "p_min": 0,
        "compute_slots": 7,
        "kv_precision": "F16",
        "fixed_a8_arithmetic_revision": FIXED_A8_ARITHMETIC_REVISION,
        "floating_exceptions": "private full embedding/head, norms, attention, Markov",
        "teacher": "native full vocabulary; exact conditioning prefix",
    }
