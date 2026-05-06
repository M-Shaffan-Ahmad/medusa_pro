import math
import torch

try:
    from .triton_kernels import polar_decode_range_triton
except Exception:  # pragma: no cover - optional CUDA/Triton acceleration
    polar_decode_range_triton = None

try:
    _TORCH_COMPILE = torch.compile
except Exception:  # pragma: no cover
    _TORCH_COMPILE = None


class KVCache:
    def __init__(self, data, current_length):
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        return (
            self.data.shape[0],
            self.data.shape[1],
            int(self.current_length.item()),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        start = int(self.current_length.item())
        dst = self.data.narrow(dim, start, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.fill_(start + tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, int(self.current_length.item()))


class PolarQuantizedKVCache:
    """
    Polar-inspired cache approximation:
    - represent each 2D pair (x_even, x_odd) as radius + angle
    - quantize radius/angle to uint8
    - dequantize on-demand when attention consumes cache

    Note: this is not the full PolarQuant paper algorithm, which uses shared
    random preconditioning plus a recursive multi-level polar transform.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_length: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        current_length: torch.Tensor,
        radius_bits: int = 8,
        theta_bits: int = 8,
        runtime_dequant_cache: bool = True,
        compile_decode: bool = False,
    ):
        if head_dim % 2 != 0:
            raise ValueError("PolarQuantizedKVCache requires an even head_dim.")
        if radius_bits < 1 or radius_bits > 8:
            raise ValueError("radius_bits must be in [1, 8].")
        if theta_bits < 1 or theta_bits > 8:
            raise ValueError("theta_bits must be in [1, 8].")

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_length = max_length
        self.head_dim = head_dim
        self.pair_dim = head_dim // 2
        self.device = device
        self.dtype = dtype
        self.current_length = current_length
        self.runtime_dequant_cache = runtime_dequant_cache
        self.compile_decode = bool(compile_decode and _TORCH_COMPILE is not None)

        self.radius_bits = radius_bits
        self.theta_bits = theta_bits
        self.radius_levels = (1 << radius_bits) - 1
        self.theta_levels = (1 << theta_bits) - 1
        self._inv_radius_levels = 1.0 / max(self.radius_levels, 1)

        self.radius_q = torch.zeros(
            batch_size,
            num_heads,
            max_length,
            self.pair_dim,
            dtype=torch.uint8,
            device=device,
        )
        self.theta_q = torch.zeros(
            batch_size,
            num_heads,
            max_length,
            self.pair_dim,
            dtype=torch.uint8,
            device=device,
        )
        # Per-token-head dynamic scale for radius quantization.
        self.radius_scale = torch.zeros(
            batch_size,
            num_heads,
            max_length,
            1,
            dtype=torch.float16,
            device=device,
        )
        # Optional runtime cache of dequantized KV values to keep attention fast.
        # When enabled, we only decode appended slices, not the entire history.
        self.dequant_data = None
        if self.runtime_dequant_cache:
            self.dequant_data = torch.zeros(
                batch_size,
                num_heads,
                max_length,
                head_dim,
                dtype=dtype,
                device=device,
            )
        # Lookup tables remove per-step trig from decode hot path.
        theta_grid = torch.arange(
            self.theta_levels + 1,
            device=device,
            dtype=torch.float32,
        )
        theta_vals = (theta_grid / max(self.theta_levels, 1)) * (2 * math.pi) - math.pi
        self.theta_cos_lut = theta_vals.cos()
        self.theta_sin_lut = theta_vals.sin()

        if self.compile_decode:
            try:
                self._decode_impl = _TORCH_COMPILE(
                    self._decode_tensorized_impl,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            except Exception:
                self.compile_decode = False
                self._decode_impl = self._decode_tensorized_impl
        else:
            self._decode_impl = self._decode_tensorized_impl

    @property
    def shape(self):
        return (
            self.batch_size,
            self.num_heads,
            int(self.current_length.item()),
            self.head_dim,
        )

    def _encode(self, tensor: torch.Tensor):
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]

        radius = torch.sqrt((even * even) + (odd * odd) + 1e-12)
        scale = radius.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        radius_norm = (radius / scale).clamp(0, 1)
        radius_q = torch.round(radius_norm * self.radius_levels).to(torch.uint8)

        theta = torch.atan2(odd, even)
        theta_norm = ((theta + math.pi) / (2 * math.pi)).clamp(0, 1)
        theta_q = torch.round(theta_norm * self.theta_levels).to(torch.uint8)
        return radius_q, theta_q, scale.to(torch.float16)

    def _decode_tensorized_impl(self, radius_q, theta_q, scale, cos_lut, sin_lut):
        """
        Tensorized decode core:
        - radius from uint8 with per-token scale
        - theta from LUT gather (no trig in hot path)
        """
        radius = radius_q.to(torch.float32) * (scale.to(torch.float32) * self._inv_radius_levels)
        theta_idx = theta_q.to(torch.long)
        cos = cos_lut[theta_idx]
        sin = sin_lut[theta_idx]
        even = radius * cos
        odd = radius * sin
        return even, odd

    def _decode_range(self, start: int, end: int, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty(
                self.batch_size,
                self.num_heads,
                end - start,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )

        if polar_decode_range_triton is not None:
            decoded = polar_decode_range_triton(
                self.radius_q,
                self.theta_q,
                self.radius_scale,
                self.theta_cos_lut,
                self.theta_sin_lut,
                out,
                start,
                end,
                self._inv_radius_levels,
            )
            if decoded:
                return out

        radius_q = self.radius_q[:, :, start:end]
        theta_q = self.theta_q[:, :, start:end]
        scale = self.radius_scale[:, :, start:end]
        try:
            even, odd = self._decode_impl(
                radius_q,
                theta_q,
                scale,
                self.theta_cos_lut,
                self.theta_sin_lut,
            )
        except RuntimeError:
            # Robust fallback for environments where compiled inference interacts
            # poorly with dynamic shapes / inference tensors.
            self.compile_decode = False
            self._decode_impl = self._decode_tensorized_impl
            even, odd = self._decode_impl(
                radius_q,
                theta_q,
                scale,
                self.theta_cos_lut,
                self.theta_sin_lut,
            )

        out[..., 0::2] = even.to(self.dtype)
        out[..., 1::2] = odd.to(self.dtype)
        return out

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        if dim != 2:
            raise ValueError("PolarQuantizedKVCache.copy currently supports dim=2 only.")
        tgt_radius_q = self.radius_q.index_select(dim, indices)
        tgt_theta_q = self.theta_q.index_select(dim, indices)
        tgt_scale = self.radius_scale.index_select(dim, indices)

        self.radius_q.narrow(dim, prev_length, tgt_radius_q.shape[dim]).copy_(
            tgt_radius_q, non_blocking=True
        )
        self.theta_q.narrow(dim, prev_length, tgt_theta_q.shape[dim]).copy_(
            tgt_theta_q, non_blocking=True
        )
        self.radius_scale.narrow(dim, prev_length, tgt_scale.shape[dim]).copy_(
            tgt_scale, non_blocking=True
        )
        if self.dequant_data is not None:
            n_sel = tgt_radius_q.shape[dim]
            shadow_dst = self.dequant_data.narrow(dim, prev_length, n_sel)
            # Fast path: copy from already materialized dequantized cache slice.
            shadow_src = self.dequant_data.index_select(dim, indices)
            shadow_dst.copy_(shadow_src, non_blocking=True)
        self.current_length.fill_(prev_length + tgt_radius_q.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        if dim != 2:
            raise ValueError("PolarQuantizedKVCache.cat currently supports dim=2 only.")
        start = int(self.current_length.item())
        end = start + tensor.shape[dim]
        radius_q, theta_q, scale = self._encode(tensor)

        self.radius_q.narrow(dim, start, tensor.shape[dim]).copy_(radius_q, non_blocking=True)
        self.theta_q.narrow(dim, start, tensor.shape[dim]).copy_(theta_q, non_blocking=True)
        self.radius_scale.narrow(dim, start, tensor.shape[dim]).copy_(scale, non_blocking=True)
        self.current_length.fill_(end)
        if self.dequant_data is not None:
            shadow_slice = self.dequant_data.narrow(dim, start, tensor.shape[dim])
            # Faster than decode: we already have the exact appended FP values.
            if tensor.dtype != self.dtype:
                shadow_slice.copy_(tensor.to(self.dtype), non_blocking=True)
            else:
                shadow_slice.copy_(tensor, non_blocking=True)
            return torch.narrow(self.dequant_data, 2, 0, int(self.current_length.item()))
        # Strict compressed mode: decode full history on demand (slow but lower transient memory).
        return self._decode_range(0, end)


def initialize_past_key_values(
    model,
    safe_max_length: int = 2048,
    turbo_quant: bool = False,
    turbo_radius_bits: int = 8,
    turbo_theta_bits: int = 8,
    turbo_runtime_dequant_cache: bool = True,
    turbo_compile_decode: bool = False,
):
    config = model.config
    batch_size = 1
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    current_length_data = torch.zeros(num_layers * 2, dtype=torch.long, device="cpu")
    past_key_values = []

    if turbo_quant:
        past_key_values_data = None
        for i in range(num_layers):
            layer_caches = []
            for j in range(2):
                idx = i * 2 + j
                layer_caches.append(
                    PolarQuantizedKVCache(
                        batch_size=batch_size,
                        num_heads=num_kv_heads,
                        max_length=safe_max_length,
                        head_dim=head_dim,
                        device=model.device,
                        dtype=model.dtype,
                        current_length=current_length_data[idx],
                        radius_bits=turbo_radius_bits,
                        theta_bits=turbo_theta_bits,
                        runtime_dequant_cache=turbo_runtime_dequant_cache,
                        compile_decode=turbo_compile_decode,
                    )
                )
            past_key_values.append(layer_caches)
        return past_key_values, past_key_values_data, current_length_data

    past_key_values_data = torch.zeros(
        num_layers * 2,
        batch_size,
        num_kv_heads,
        safe_max_length,
        head_dim,
        device=model.device,
        dtype=model.dtype,
    )
    for i in range(num_layers):
        past_key_values.append(
            [
                KVCache(past_key_values_data[i * 2 + j], current_length_data[i * 2 + j])
                for j in range(2)
            ]
        )
    return past_key_values, past_key_values_data, current_length_data
