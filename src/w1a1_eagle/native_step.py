"""Differentiable Torch reference for one native EAGLE-3 W1Ax step.

This adapter consumes the pinned AngelSlim drafter after all nine linears have
been replaced by group128/A16 or row-scale W1Ax trainable modules. It reproduces the
one-token structure of ``llama_model_eagle3::graph<false>``: borrowed F16 token
embedding, separate F32 RMS norms, embedding-first concatenation, Q/K/V with
half-rotation RoPE, F16 cache storage, GQA attention, residual and SiLU FFN,
then the F32 output norm and binary draft head. The cache is an immutable,
contiguous one-token causal prefix and retains autograd links during unroll.

This is a structural Torch reference. Native GGML full-drafter numeric parity,
cache scheduling and accelerator throughput remain unverified. Standard F32
attention follows the row linears' device; native oracle modes remain CPU-only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .native_attention_oracle import NativeAttentionForward, native_forward_f32_backward
from .native_cpu_diagnostic import NativeCPUDiagnosticOperators
from .recurrent_binary import CANDIDATE_D_BASE_TO_PATH, GroupedBinaryLinear
from .recurrent_qat import RowBinaryLinear
from .recurrent_rollout import DraftStep

NATIVE_NORM_PATHS = {
    "blk.0.attn_norm.weight": "midlayer.input_layernorm",
    "blk.0.attn_norm_2.weight": "midlayer.hidden_norm",
    "blk.0.ffn_norm.weight": "midlayer.post_attention_layernorm",
    "output_norm.weight": "norm",
}


def bind_frozen_norms(drafter: nn.Module, norm_arrays: Mapping[str, Tensor]) -> None:
    """Copy native F32 draft norms without changing borrowed target tensors."""
    if set(norm_arrays) != set(NATIVE_NORM_PATHS):
        raise ValueError("expected the four named frozen native EAGLE norms")
    hidden = getattr(getattr(drafter, "config", None), "hidden_size", None)
    if type(hidden) is not int or hidden < 1:
        raise ValueError("drafter hidden size is invalid")
    pending = []
    for name, path in NATIVE_NORM_PATHS.items():
        module = drafter.get_submodule(path)
        weight = norm_arrays[name]
        if (
            not isinstance(weight, Tensor)
            or weight.device.type != "cpu"
            or weight.dtype != torch.float32
            or weight.shape != (hidden,)
            or not torch.isfinite(weight).all()
        ):
            raise ValueError(f"{name}: native norm must be one finite CPU F32 row")
        if not isinstance(getattr(module, "weight", None), nn.Parameter):
            raise ValueError(f"{path}: expected an existing norm parameter")
        pending.append((module, weight.detach().clone().to(module.weight.device)))
    for module, weight in pending:
        module.weight = nn.Parameter(weight, requires_grad=False)


@dataclass(frozen=True)
class NativeStepCache:
    """Post-RoPE K and raw V in F32 storage, each exactly representable as F16.

    Shape is ``(kv_heads, decoder_positions, head_dim)``. The current step is
    appended without detach so later draft losses reach earlier K/V.
    """

    key: Tensor
    value: Tensor


def _frozen_rms_norm(x: Tensor, module: nn.Module) -> Tensor:
    # ggml accumulates F32 squares in double, rounds the mean to F32, then
    # evaluates sqrt and reciprocal in F32 before the two F32 multiplies.
    variance = x.square().to(torch.float64).sum(dim=-1, keepdim=True)
    variance = (variance / x.shape[-1]).to(torch.float32)
    scale = torch.reciprocal(torch.sqrt(variance + module.variance_epsilon))
    return x * scale * module.weight


class NativeStepAdapter(nn.Module):
    """One-step binary drafter callable by ``rebuild_prefix_cache`` and rollout.

    ``decode_step`` returns draft-vocabulary logits; the frozen d2t/t2d map
    belongs to the trace/verifier boundary, not the differentiable head. Tree
    masks, holes in a prefix and nondefault RoPE are intentionally rejected.
    """

    def __init__(
        self,
        drafter: nn.Module,
        *,
        embedding_lookup: Callable[[int], Tensor] | None = None,
        attention_mode: str = "f32",
        native_attention_oracle: NativeAttentionForward | None = None,
        native_cpu_operators: NativeCPUDiagnosticOperators | None = None,
    ) -> None:
        super().__init__()
        config = getattr(drafter, "config", None)
        if config is None or getattr(config, "pretraining_tp", None) != 1:
            raise ValueError("one-step adapter requires pretraining_tp=1")
        if getattr(drafter, "early_stop_method", None) is not None:
            raise ValueError("early stop is unsupported")
        if any(
            bool(getattr(source, flag, False))
            for source in (drafter, config)
            for flag in ("norm_before_fc", "norm_before_residual")
        ):
            raise ValueError("norm_before_fc/residual is unsupported")
        if getattr(config, "rope_scaling", None) is not None:
            raise ValueError("scaled RoPE is unsupported")
        if getattr(config, "hidden_act", None) != "silu":
            raise ValueError("the pinned drafter requires SiLU")
        if attention_mode not in ("f32", "native_forward_f32_backward", "native_cpu_diagnostic"):
            raise ValueError("unsupported attention mode")
        if attention_mode == "native_forward_f32_backward":
            if not callable(native_attention_oracle) or native_cpu_operators is not None:
                raise ValueError("native attention mode requires a callable oracle")
        elif attention_mode == "native_cpu_diagnostic":
            if not isinstance(native_cpu_operators, NativeCPUDiagnosticOperators):
                raise ValueError("native CPU diagnostic mode requires pinned operators")
            if native_attention_oracle is not None:
                raise ValueError("native CPU diagnostic owns its attention oracle")
        elif native_attention_oracle is not None or native_cpu_operators is not None:
            raise ValueError("native attention oracle requires native attention mode")

        hidden = getattr(config, "hidden_size", None)
        heads = getattr(config, "num_attention_heads", None)
        kv_heads = getattr(config, "num_key_value_heads", None)
        intermediate = getattr(config, "intermediate_size", None)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None and type(hidden) is int and type(heads) is int and heads > 0:
            head_dim = hidden // heads
        max_positions = getattr(config, "max_position_embeddings", None)
        theta = getattr(config, "rope_theta", 10000.0)
        if not all(
            type(value) is int and value > 0
            for value in (hidden, heads, kv_heads, intermediate, head_dim, max_positions)
        ):
            raise ValueError("drafter dimensions must be positive integers")
        if head_dim % 2 or heads % kv_heads:
            raise ValueError("unsupported attention head geometry")
        if attention_mode == "native_cpu_diagnostic" and (
            hidden != 2560
            or heads != 32
            or kv_heads != 8
            or head_dim != 128
            or intermediate != 9728
            or max_positions < 256
        ):
            raise ValueError("native CPU diagnostic requires pinned EAGLE geometry")
        if not isinstance(theta, (int, float)) or not math.isfinite(theta) or theta <= 0:
            raise ValueError("rope_theta must be positive and finite")

        expected = {
            "fc": (hidden, 3 * hidden),
            "midlayer.self_attn.q_proj": (heads * head_dim, 2 * hidden),
            "midlayer.self_attn.k_proj": (kv_heads * head_dim, 2 * hidden),
            "midlayer.self_attn.v_proj": (kv_heads * head_dim, 2 * hidden),
            "midlayer.self_attn.o_proj": (hidden, heads * head_dim),
            "midlayer.mlp.gate_proj": (intermediate, hidden),
            "midlayer.mlp.up_proj": (intermediate, hidden),
            "midlayer.mlp.down_proj": (hidden, intermediate),
            "lm_head": (None, hidden),
        }
        if set(expected) != set(CANDIDATE_D_BASE_TO_PATH.values()):
            raise RuntimeError("candidate-D projection paths changed")
        linears = {}
        for path, (out_features, in_features) in expected.items():
            try:
                module = drafter.get_submodule(path)
            except AttributeError as exc:
                raise ValueError(f"missing binary projection {path}") from exc
            if not isinstance(module, (GroupedBinaryLinear, RowBinaryLinear)):
                raise ValueError(f"{path} must be group-128 or row-scale W1Ax binary linear")
            if isinstance(module, GroupedBinaryLinear) and module.group_size != 128:
                raise ValueError(f"{path} group binary linear must use group size 128")
            if (out_features is not None and module.out_features != out_features) or (
                in_features is not None and module.in_features != in_features
            ):
                raise ValueError(f"{path} has incompatible dimensions")
            linears[path] = module
        if len({type(module) for module in linears.values()}) != 1:
            raise ValueError("binary projections must share row or group layout")
        if isinstance(next(iter(linears.values())), RowBinaryLinear):
            if len({module.contract for module in linears.values()}) != 1:
                raise ValueError("row projections must share one activation contract")
        elif len({module.arithmetic for module in linears.values()}) != 1:
            raise ValueError("group projections must use one declared arithmetic")
        devices = {module.latent_sign.device for module in linears.values()}
        if len(devices) != 1:
            raise ValueError("binary projections must share one execution device")
        self.device = next(iter(devices))
        if self.device.type != "cpu" and (
            attention_mode != "f32" or isinstance(next(iter(linears.values())), GroupedBinaryLinear)
        ):
            raise ValueError("accelerator reference requires row W1Ax and F32 Torch attention")
        if attention_mode == "native_cpu_diagnostic" and any(
            not isinstance(module, GroupedBinaryLinear) or module.arithmetic != "native_order"
            for module in linears.values()
        ):
            raise ValueError("native CPU diagnostic requires ordered binary projections")

        if embedding_lookup is None:
            embedding = getattr(drafter, "embed_tokens", None)
            if not isinstance(embedding, nn.Embedding) or embedding.weight.device != self.device:
                raise ValueError("borrowed token embedding must share the binary device")
            if embedding.weight.dtype != torch.float16 or embedding.embedding_dim != hidden:
                raise ValueError("borrowed token embedding must be F16 with hidden_size columns")

            def read_embedding(token: int) -> Tensor:
                return embedding.weight[token]

            embedding_lookup = read_embedding
            embedding_vocab_size = embedding.num_embeddings
        else:
            embedding_vocab_size = getattr(embedding_lookup, "vocab_size", None)
            if (
                not callable(embedding_lookup)
                or type(embedding_vocab_size) is not int
                or embedding_vocab_size < 1
                or getattr(embedding_lookup, "hidden_size", None) != hidden
            ):
                raise ValueError("external F16 embedding lookup has incompatible dimensions")
        norm_paths = (
            "midlayer.input_layernorm",
            "midlayer.hidden_norm",
            "midlayer.post_attention_layernorm",
            "norm",
        )
        for path in norm_paths:
            try:
                norm = drafter.get_submodule(path)
            except AttributeError as exc:
                raise ValueError(f"missing frozen norm {path}") from exc
            weight = getattr(norm, "weight", None)
            eps = getattr(norm, "variance_epsilon", None)
            if (
                not isinstance(weight, Tensor)
                or weight.device != self.device
                or weight.dtype != torch.float32
                or weight.shape != (hidden,)
                or (self.device.type == "cpu" and not torch.isfinite(weight).all())
                or not isinstance(eps, (int, float))
                or not math.isfinite(eps)
                or eps <= 0
            ):
                raise ValueError(f"{path} must have a finite F32 weight and positive epsilon")

        binary_parameter_ids = {
            id(parameter)
            for module in linears.values()
            for parameter in (module.latent_sign, module.scale_offset)
        }
        # The borrowed target embedding and every original norm are frozen.
        # This also protects any unused drafter parameters from optimization.
        for parameter in drafter.parameters():
            if id(parameter) not in binary_parameter_ids:
                parameter.requires_grad_(False)

        self.drafter = drafter
        self.linears = MappingProxyType(linears)
        self.embedding_lookup = embedding_lookup
        self.embedding_vocab_size = embedding_vocab_size
        self.hidden_size = hidden
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.max_positions = max_positions
        self.rope_theta = float(theta)
        self.attention_mode = attention_mode
        self.native_attention_oracle = native_attention_oracle
        self.native_cpu_operators = native_cpu_operators

    def new_cache(self) -> NativeStepCache:
        shape = (self.kv_heads, 0, self.head_dim)
        return NativeStepCache(
            torch.empty(shape, dtype=torch.float32, device=self.device),
            torch.empty(shape, dtype=torch.float32, device=self.device),
        )

    def _rms_norm(self, x: Tensor, module: nn.Module) -> Tensor:
        if self.device.type == "cpu":
            return _frozen_rms_norm(x, module)
        # F32 Torch surrogate avoids F64 reductions in accelerator QAT.
        return (
            x
            * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + module.variance_epsilon)
            * module.weight
        )

    def _validate_cache(self, cache: NativeStepCache, position: int) -> None:
        if not isinstance(cache, NativeStepCache):
            raise ValueError("decoder cache must be NativeStepCache")
        shape = (self.kv_heads, position, self.head_dim)
        for name, value in (("key", cache.key), ("value", cache.value)):
            if (
                not isinstance(value, Tensor)
                or value.device != self.device
                or value.dtype != torch.float32
                or value.shape != shape
            ):
                raise ValueError(f"{name} cache must contain exactly decoder_position prior rows")
            if self.device.type == "cpu" and (
                not torch.isfinite(value).all()
                or not torch.equal(value, value.to(torch.float16).to(torch.float32))
            ):
                raise ValueError(f"{name} cache must be finite and F16-exact")

    def encode_feature(
        self, raw: Tensor, *, trace_callback: Callable[[str, Tensor], None] | None = None
    ) -> Tensor:
        if (
            not isinstance(raw, Tensor)
            or raw.device != self.device
            or raw.dtype != torch.float32
            or raw.shape != (self.drafter.fc.in_features,)
            or (self.device.type == "cpu" and not torch.isfinite(raw).all())
        ):
            raise ValueError("raw target feature must be one finite F32 CPU row")
        encoded = self.drafter.fc(raw)
        if trace_callback is not None:
            trace_callback("fc_out", encoded.detach().clone())
        return encoded

    def decode_step(
        self,
        token: int,
        feature: Tensor,
        decoder_position: int,
        cache: NativeStepCache,
        *,
        compute_logits: bool = True,
        trace_callback: Callable[[str, Tensor], None] | None = None,
    ) -> DraftStep:
        def trace(name: str, value: Tensor) -> None:
            if trace_callback is not None:
                trace_callback(name, value.detach().clone())

        if type(token) is not int or token < 0 or token >= self.embedding_vocab_size:
            raise ValueError("token must index the borrowed embedding")
        if type(decoder_position) is not int or not 0 <= decoder_position < self.max_positions:
            raise ValueError("decoder_position is outside the supported RoPE context")
        if (
            self.attention_mode in ("native_forward_f32_backward", "native_cpu_diagnostic")
            and decoder_position >= 256
        ):
            raise ValueError("native attention mode supports decoder positions 0..255")
        if self.attention_mode == "native_cpu_diagnostic" and torch.is_grad_enabled():
            raise ValueError("native CPU diagnostic is forward-only; use torch.no_grad()")
        if getattr(self.drafter, "tree_mask", None) is not None:
            raise ValueError("tree mask cannot be used with contiguous one-step cache")
        if (
            not isinstance(feature, Tensor)
            or feature.device != self.device
            or feature.dtype != torch.float32
            or feature.shape != (self.hidden_size,)
            or (self.device.type == "cpu" and not torch.isfinite(feature).all())
        ):
            raise ValueError("feature must be one finite F32 CPU hidden row")
        self._validate_cache(cache, decoder_position)

        layer = self.drafter.midlayer
        attn = layer.self_attn
        mlp = layer.mlp
        trace("inp_g_embeddings", feature)
        embedding_f16 = self.embedding_lookup(token)
        if (
            not isinstance(embedding_f16, Tensor)
            or embedding_f16.device != self.device
            or embedding_f16.dtype != torch.float16
            or embedding_f16.shape != (self.hidden_size,)
            or embedding_f16.requires_grad
            or (self.device.type == "cpu" and not torch.isfinite(embedding_f16).all())
        ):
            raise ValueError("embedding lookup must return a frozen finite CPU F16 row")
        embedding = embedding_f16.to(torch.float32)
        trace("inp_embd", embedding)
        normalized_embedding = self._rms_norm(embedding, layer.input_layernorm)
        trace("embd_norm-0", normalized_embedding)
        normalized_feature = self._rms_norm(feature, layer.hidden_norm)
        trace("g_norm-0", normalized_feature)
        fused = torch.cat((normalized_embedding, normalized_feature), dim=-1)
        trace("concat_embd-0", fused)
        q = attn.q_proj(fused)
        k = attn.k_proj(fused)
        v = attn.v_proj(fused)
        trace("Qcur-0", q)
        trace("Kcur-0", k)
        trace("Vcur-0", v)
        q = q.reshape(self.heads, self.head_dim)
        k = k.reshape(self.kv_heads, self.head_dim)
        v = v.reshape(self.kv_heads, self.head_dim)

        if self.attention_mode == "native_cpu_diagnostic":
            q, k = self.native_cpu_operators.rope(q, k, decoder_position)
        else:
            # ggml builds RoPE frequencies by repeated F32 multiplication. A
            # direct power at every channel changes some F16-rounded cache keys.
            theta_scale = torch.tensor(
                self.rope_theta, dtype=torch.float32, device=self.device
            ).pow(-2.0 / self.head_dim)
            if self.device.type == "cpu":
                angle = torch.empty(self.head_dim // 2, dtype=torch.float32, device=self.device)
                theta = torch.tensor(
                    float(decoder_position), dtype=torch.float32, device=self.device
                )
                for index in range(angle.numel()):
                    angle[index] = theta
                    theta = theta * theta_scale
            else:
                # Vectorized training surrogate; native proposal trajectories
                # remain the deployment/numeric gate after accelerator access.
                channels = torch.arange(self.head_dim // 2, device=self.device)
                angle = float(decoder_position) * theta_scale.pow(channels)
            full_angle = torch.cat((angle, angle))
            cos, sin = full_angle.cos(), full_angle.sin()
            half = self.head_dim // 2
            q = q * cos + torch.cat((-q[:, half:], q[:, :half]), dim=-1) * sin
            k = k * cos + torch.cat((-k[:, half:], k[:, :half]), dim=-1) * sin
        trace("Qcur_rope-0", q)
        trace("Kcur_rope-0", k)

        # Native KV cache writes F16 and reads F32. Include casts in autograd;
        # current K/V must take this boundary before both attention and append.
        k = k.to(torch.float16).to(torch.float32)
        v = v.to(torch.float16).to(torch.float32)
        next_cache = NativeStepCache(
            torch.cat((cache.key, k[:, None, :]), dim=1),
            torch.cat((cache.value, v[:, None, :]), dim=1),
        )
        if self.attention_mode == "native_cpu_diagnostic":
            attention = self.native_cpu_operators.attention(
                q, next_cache.key, next_cache.value
            ).reshape(-1)
        elif self.attention_mode == "native_forward_f32_backward":
            attention = native_forward_f32_backward(
                q, next_cache.key, next_cache.value, self.native_attention_oracle
            ).reshape(-1)
        else:
            repeat = self.heads // self.kv_heads
            keys = next_cache.key.repeat_interleave(repeat, dim=0)
            values = next_cache.value.repeat_interleave(repeat, dim=0)
            scores = torch.einsum("hd,htd->ht", q, keys) / math.sqrt(self.head_dim)
            probabilities = F.softmax(scores, dim=-1, dtype=torch.float32)
            attention = torch.einsum("ht,htd->hd", probabilities, values).reshape(-1)
        trace("kqv_out-0", attention)
        residual = feature + attn.o_proj(attention)
        trace("ffn_inp-0", residual)
        post_attention = self._rms_norm(residual, layer.post_attention_layernorm)
        trace("post_attn_norm-0", post_attention)
        gate = mlp.gate_proj(post_attention)
        activated = (
            self.native_cpu_operators.silu(gate)
            if self.attention_mode == "native_cpu_diagnostic"
            else F.silu(gate)
        )
        ffn = mlp.down_proj(activated * mlp.up_proj(post_attention))
        trace("ffn_out-0", ffn)
        pre_norm = residual + ffn
        trace("eagle3_prenorm-0", pre_norm)
        normalized_output = (
            self._rms_norm(pre_norm, self.drafter.norm)
            if compute_logits or trace_callback is not None
            else None
        )
        if normalized_output is not None:
            trace("result_norm", normalized_output)
        logits = (
            self.drafter.lm_head(normalized_output)
            if compute_logits
            else torch.empty(0, dtype=torch.float32, device=self.device)
        )
        return DraftStep(logits=logits, pre_norm=pre_norm, cache=next_cache)

    def decode_context(
        self, token: int, feature: Tensor, decoder_position: int, cache: NativeStepCache
    ) -> DraftStep:
        """Rebuild a context cache without computing unused draft-head logits."""
        return self.decode_step(token, feature, decoder_position, cache, compute_logits=False)
