import math
import torch

try:
    from .triton_kernels import polar_decode_range_triton, turbo_vq_append_triton
except Exception:  # pragma: no cover - optional CUDA/Triton acceleration
    polar_decode_range_triton = None
    turbo_vq_append_triton = None

try:
    _TORCH_COMPILE = torch.compile
except Exception:  # pragma: no cover
    _TORCH_COMPILE = None


class KVCache:
    def __init__(self, data, current_length, qjl_sketch_dim: int = 0, quant_seed: int = 0):
        self.data = data
        self.current_length = current_length
        self.qjl_sketch_dim = int(qjl_sketch_dim)
        self.quant_seed = int(quant_seed)
        self.qjl_words = 0
        self.qjl_proj = None
        self.qjl_bits = None
        self.qjl_pack_weights = None
        if self.qjl_sketch_dim > 0:
            if self.qjl_sketch_dim % 32 != 0:
                raise ValueError("qjl_sketch_dim must be a multiple of 32.")
            self.qjl_words = self.qjl_sketch_dim // 32
            self.qjl_proj = _get_packed_qjl_projection(
                int(data.shape[-1]),
                self.qjl_sketch_dim,
                data.device,
                seed=self.quant_seed,
            )
            self.qjl_bits = torch.zeros(
                int(data.shape[0]),
                int(data.shape[1]),
                int(data.shape[2]),
                self.qjl_words,
                dtype=torch.int32,
                device=data.device,
            )
            self.qjl_pack_weights = (
                1 << torch.arange(32, device=data.device, dtype=torch.int64)
            )

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
        if self.qjl_bits is not None:
            tgt_bits = self.qjl_bits.index_select(dim, indices)
            self.qjl_bits.narrow(dim, prev_length, tgt_bits.shape[dim]).copy_(
                tgt_bits,
                non_blocking=True,
            )
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        start = int(self.current_length.item())
        dst = self.data.narrow(dim, start, tensor.shape[dim])
        dst.copy_(tensor)
        if self.qjl_bits is not None:
            packed = self.pack_qjl_bits(tensor)
            self.qjl_bits.narrow(dim, start, tensor.shape[dim]).copy_(
                packed,
                non_blocking=True,
            )
        self.current_length.fill_(start + tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, int(self.current_length.item()))

    def reset(self):
        self.current_length.fill_(0)

    def pack_qjl_bits(self, tensor: torch.Tensor):
        if self.qjl_proj is None:
            return None
        projected = tensor.to(torch.float32) @ self.qjl_proj
        bits = (projected >= 0).to(torch.int64)
        weights = self.qjl_pack_weights.view(*([1] * (bits.dim() - 1)), 32)
        packed = (
            bits.view(*bits.shape[:-1], self.qjl_words, 32) * weights
        ).sum(dim=-1)
        return packed.to(torch.int32)


class OutlierCalibrationKVCache(KVCache):
    """
    Exact FP KV cache used only during an untimed calibration prefill.

    It behaves like KVCache for attention correctness while accumulating per
    channel |K|/|V| magnitudes so outlier channels can be frozen before the real
    benchmark run.
    """

    def __init__(self, data, current_length):
        super().__init__(data, current_length)
        self.channel_abs_sum = torch.zeros(
            int(data.shape[-1]),
            dtype=torch.float64,
            device=data.device,
        )
        self.channel_count = 0

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        if dim != 2:
            raise ValueError("OutlierCalibrationKVCache.cat currently supports dim=2 only.")
        values = tensor.detach().to(torch.float32)
        self.channel_abs_sum.add_(values.abs().sum(dim=(0, 1, 2)).to(torch.float64))
        self.channel_count += int(values.numel() // max(1, values.shape[-1]))
        return super().cat(tensor, dim=dim)

    def topk_outlier_indices(self, n_outlier: int):
        n_outlier = int(max(0, min(int(n_outlier), int(self.data.shape[-1]) - 1)))
        if n_outlier <= 0:
            return torch.empty(0, dtype=torch.long, device=self.data.device)
        _, top = torch.topk(self.channel_abs_sum, k=n_outlier, largest=True, sorted=False)
        return top.sort()[0].to(dtype=torch.long, device=self.data.device)


_TURBO_ROTATION_CACHE = {}
_TURBO_CODEBOOK_CACHE = {}
_TURBO_RESIDUAL_PROJ_CACHE = {}
_PACKED_QJL_PROJ_CACHE = {}
_POLAR_ANGLE_CODEBOOK_CACHE = {}


def _cache_key(device, *parts):
    device = torch.device(device)
    return (device.type, device.index, *parts)


def _seeded_manual_seed(base_seed: int, quant_seed: int):
    seed = int(base_seed) + (int(quant_seed) * 1_000_003)
    return seed % (2**63 - 1)


def _get_packed_qjl_projection(head_dim: int, sketch_dim: int, device: torch.device, seed: int = 0):
    sketch_dim = int(sketch_dim)
    if sketch_dim <= 0 or sketch_dim % 32 != 0:
        raise ValueError("Packed QJL sketch_dim must be a positive multiple of 32.")
    seed = int(seed)
    key = _cache_key(device, "packed_qjl", int(head_dim), sketch_dim, seed)
    proj = _PACKED_QJL_PROJ_CACHE.get(key)
    if proj is not None:
        return proj

    gen = torch.Generator(device="cpu")
    gen.manual_seed(_seeded_manual_seed(20260507 + int(head_dim) * 17 + sketch_dim, seed))
    blocks = []
    remaining = sketch_dim
    while remaining > 0:
        mat = torch.randn(int(head_dim), int(head_dim), generator=gen, dtype=torch.float32)
        block, _ = torch.linalg.qr(mat, mode="reduced")
        signs = torch.sign(torch.diagonal(block))
        signs[signs == 0] = 1
        block = block * signs
        take = min(remaining, int(head_dim))
        blocks.append(block[:, :take] * math.sqrt(float(head_dim)))
        remaining -= take
    proj = torch.cat(blocks, dim=1).contiguous().to(device=device)
    _PACKED_QJL_PROJ_CACHE[key] = proj
    return proj


def _get_turbo_rotation(head_dim: int, device: torch.device, seed: int = 0):
    seed = int(seed)
    key = _cache_key(device, "rotation", int(head_dim), seed)
    rotation = _TURBO_ROTATION_CACHE.get(key)
    if rotation is not None:
        return rotation

    gen = torch.Generator(device="cpu")
    gen.manual_seed(_seeded_manual_seed(20260427 + int(head_dim), seed))
    mat = torch.randn(int(head_dim), int(head_dim), generator=gen, dtype=torch.float32)
    rotation, _ = torch.linalg.qr(mat, mode="reduced")
    # Fix QR sign ambiguity so the rotation is reproducible across LAPACK builds.
    signs = torch.sign(torch.diagonal(rotation))
    signs[signs == 0] = 1
    rotation = rotation * signs
    rotation = rotation.to(device=device)
    _TURBO_ROTATION_CACHE[key] = rotation
    return rotation


def _build_lloyd_max_codebook(bits: int, head_dim: int, device: torch.device):
    bits = int(bits)
    head_dim = int(head_dim)
    key = _cache_key(device, "lloyd_max_sphere", bits, head_dim)
    cached = _TURBO_CODEBOOK_CACHE.get(key)
    if cached is not None:
        return cached

    levels = 1 << bits
    # Paper Eq. (4): after a random rotation, each coordinate of a unit vector
    # follows the sphere-coordinate Beta density on [-1, 1]. The runtime cache
    # stores RMS scale separately, so quantization happens on z = sqrt(d) * x_j.
    # This keeps the table numerically close to N(0, 1) for large d while still
    # matching the finite-d distribution used by TurboQuant.
    radius = math.sqrt(float(head_dim))
    eps = min(1e-6, 0.5 / max(radius, 1.0))
    grid = torch.linspace(-radius + eps, radius - eps, 40001, dtype=torch.float64)
    sphere_arg = torch.clamp(1.0 - (grid * grid / float(head_dim)), min=0.0)
    pdf = sphere_arg.pow((float(head_dim) - 3.0) * 0.5)
    probs = torch.linspace(0.5 / levels, 1.0 - (0.5 / levels), levels, dtype=torch.float64)
    centroids = torch.erfinv((2.0 * probs) - 1.0) * math.sqrt(2.0)
    centroids = centroids.clamp(-radius + eps, radius - eps)

    for _ in range(80):
        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        bucket = torch.bucketize(grid, boundaries)
        new_centroids = centroids.clone()
        for idx in range(levels):
            mask = bucket == idx
            weight = pdf[mask].sum()
            if weight > 0:
                new_centroids[idx] = (grid[mask] * pdf[mask]).sum() / weight
        if torch.max(torch.abs(new_centroids - centroids)) < 1e-7:
            centroids = new_centroids
            break
        centroids = new_centroids

    codebook = centroids.to(device=device, dtype=torch.float32)
    boundaries = ((codebook[:-1] + codebook[1:]) * 0.5).contiguous()
    _TURBO_CODEBOOK_CACHE[key] = (codebook.contiguous(), boundaries)
    return _TURBO_CODEBOOK_CACHE[key]


def _get_turbo_residual_projection(head_dim: int, residual_dim: int, device: torch.device, seed: int = 0):
    residual_dim = int(max(0, residual_dim))
    seed = int(seed)
    key = _cache_key(device, "residual_qjl", int(head_dim), residual_dim, seed)
    proj = _TURBO_RESIDUAL_PROJ_CACHE.get(key)
    if proj is not None:
        return proj

    gen = torch.Generator(device="cpu")
    gen.manual_seed(_seeded_manual_seed(20260506 + int(head_dim) * 31 + residual_dim, seed))
    head_dim = int(head_dim)
    if residual_dim > 0:
        # QJL benefits from Gaussian-like directions with controlled norms. Build
        # independent orthogonal blocks so m can be larger than head_dim without
        # falling back to a high-variance unstructured projection.
        blocks = []
        remaining = residual_dim
        while remaining > 0:
            mat = torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float32)
            block, _ = torch.linalg.qr(mat, mode="reduced")
            signs = torch.sign(torch.diagonal(block))
            signs[signs == 0] = 1
            block = block * signs
            take = min(remaining, head_dim)
            blocks.append(block[:, :take] * math.sqrt(float(head_dim)))
            remaining -= take
        proj = torch.cat(blocks, dim=1).contiguous()
    else:
        proj = torch.empty(head_dim, 0, dtype=torch.float32)
    proj = proj.to(device=device)
    _TURBO_RESIDUAL_PROJ_CACHE[key] = proj
    return proj


class TurboQuantizedKVCache:
    """
    TurboQuant MSE-stage KV cache:
    - shared random orthogonal rotation
    - coordinate-wise Lloyd-Max scalar quantization
    - per-token RMS scale for stable reconstruction in practical LLM caches
    - optional 1-bit QJL residual sketches for query-key correction
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
        bits: int = 4,
        residual_dim: int = 128,
        residual_scale: float = 1.0,
        runtime_dequant_cache: bool = True,
        quant_seed: int = 0,
    ):
        if bits < 1 or bits > 8:
            raise ValueError("TurboQuantizedKVCache bits must be in [1, 8].")

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_length = max_length
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.current_length = current_length
        self.bits = int(bits)
        self.levels = 1 << self.bits
        self.q_idx_is_packed = self.bits < 8
        self.q_idx_packed_dim = (
            (self.head_dim * self.bits + 7) // 8
            if self.q_idx_is_packed
            else self.head_dim
        )
        self.residual_dim = int(head_dim if int(residual_dim) < 0 else max(0, residual_dim))
        self.quant_seed = int(quant_seed)
        self.residual_packed_dim = (self.residual_dim + 7) // 8 if self.residual_dim > 0 else 0
        self.residual_scale = float(residual_scale)
        self.residual_coeff = (
            self.residual_scale * math.sqrt(math.pi / 2.0) / float(self.residual_dim)
            if self.residual_dim > 0
            else 0.0
        )
        self.runtime_dequant_cache = bool(runtime_dequant_cache)

        self.rotation = _get_turbo_rotation(head_dim, device, seed=self.quant_seed)
        self.rotation_t = self.rotation.t().contiguous()
        self.codebook, self.boundaries = _build_lloyd_max_codebook(self.bits, head_dim, device)
        self.residual_proj = None
        if self.residual_dim > 0:
            self.residual_proj = _get_turbo_residual_projection(
                head_dim,
                self.residual_dim,
                device,
                seed=self.quant_seed,
            )

        self.q_idx = torch.zeros(
            batch_size,
            num_heads,
            max_length,
            self.q_idx_packed_dim,
            dtype=torch.uint8,
            device=device,
        )
        self.scale = torch.zeros(
            batch_size,
            num_heads,
            max_length,
            1,
            dtype=torch.float16,
            device=device,
        )
        self.residual_sign_packed = None
        self.residual_norm = None
        self.residual_bit_shifts = None
        self.residual_pack_weights = None
        if self.residual_dim > 0:
            self.residual_sign_packed = torch.zeros(
                batch_size,
                num_heads,
                max_length,
                self.residual_packed_dim,
                dtype=torch.uint8,
                device=device,
            )
            self.residual_norm = torch.zeros(
                batch_size,
                num_heads,
                max_length,
                1,
                dtype=torch.float16,
                device=device,
            )
            self.residual_bit_shifts = torch.arange(8, dtype=torch.int16, device=device)
            self.residual_pack_weights = (1 << self.residual_bit_shifts).view(
                *([1] * 4),
                8,
            )
        self.dequant_data = None
        self._attention_out_cache = {}
        if self.runtime_dequant_cache:
            self.dequant_data = torch.zeros(
                batch_size,
                num_heads,
                max_length,
                head_dim,
                dtype=dtype,
                device=device,
            )

    @property
    def shape(self):
        return (
            self.batch_size,
            self.num_heads,
            int(self.current_length.item()),
            self.head_dim,
        )

    def get_attention_output(self, query_states: torch.Tensor) -> torch.Tensor:
        key = (tuple(query_states.shape), query_states.dtype)
        cached = self._attention_out_cache.get(key)
        if cached is None:
            cached = torch.empty(
                tuple(query_states.shape),
                dtype=query_states.dtype,
                device=query_states.device,
            )
            self._attention_out_cache[key] = cached
        return cached

    def _pack_residual_sign(self, residual_positive: torch.Tensor):
        bits = residual_positive.to(torch.uint8)
        pad = (self.residual_packed_dim * 8) - self.residual_dim
        if pad > 0:
            bits = torch.cat(
                [
                    bits,
                    torch.zeros(*bits.shape[:-1], pad, dtype=bits.dtype, device=bits.device),
                ],
                dim=-1,
            )
        packed = bits.view(*bits.shape[:-1], self.residual_packed_dim, 8).to(torch.int16)
        packed = (packed * self.residual_pack_weights).sum(dim=-1).to(torch.uint8)
        return packed

    def _unpack_residual_sign_range(self, start: int, end: int):
        if self.residual_sign_packed is None:
            return None
        packed = self.residual_sign_packed[:, :, start:end].to(torch.int16)
        bits = ((packed.unsqueeze(-1) >> self.residual_bit_shifts) & 1).reshape(
            self.batch_size,
            self.num_heads,
            end - start,
            self.residual_packed_dim * 8,
        )
        bits = bits[..., : self.residual_dim]
        return bits.to(torch.float32).mul_(2.0).sub_(1.0)

    def _pack_q_idx(self, q_idx: torch.Tensor) -> torch.Tensor:
        if not self.q_idx_is_packed:
            return q_idx

        q_idx_i64 = q_idx.to(torch.int64)
        bit_offsets = torch.arange(
            self.head_dim,
            device=q_idx.device,
            dtype=torch.int64,
        ) * int(self.bits)
        byte_offsets = torch.div(bit_offsets, 8, rounding_mode="floor")
        bit_shifts = torch.remainder(bit_offsets, 8)

        packed = torch.zeros(
            *q_idx.shape[:-1],
            self.q_idx_packed_dim,
            dtype=torch.int64,
            device=q_idx.device,
        )
        view_shape = *([1] * (q_idx.dim() - 1)), self.head_dim
        byte_index = byte_offsets.view(view_shape).expand_as(q_idx_i64)
        shift = bit_shifts.view(view_shape)
        low = torch.bitwise_and(torch.bitwise_left_shift(q_idx_i64, shift), 0xFF)
        packed.scatter_add_(-1, byte_index, low)

        spill = bit_shifts + int(self.bits) - 8
        spill_mask = spill > 0
        if bool(spill_mask.any()):
            high_index = (byte_offsets + 1).clamp_max(self.q_idx_packed_dim - 1)
            bits_in_low = (8 - bit_shifts).view(view_shape)
            high = torch.bitwise_right_shift(q_idx_i64, bits_in_low)
            high = torch.where(
                spill_mask.view(view_shape),
                high,
                torch.zeros((), dtype=torch.int64, device=q_idx.device),
            )
            packed.scatter_add_(-1, high_index.view(view_shape).expand_as(q_idx_i64), high)

        return torch.bitwise_and(packed, 0xFF).to(torch.uint8)

    def _unpack_q_idx_range(self, start: int, end: int) -> torch.Tensor:
        if not self.q_idx_is_packed:
            return self.q_idx[:, :, start:end].to(torch.long)

        packed = self.q_idx[:, :, start:end].to(torch.int64)
        length = int(end) - int(start)
        bit_offsets = torch.arange(
            self.head_dim,
            device=self.device,
            dtype=torch.int64,
        ) * int(self.bits)
        byte_offsets = torch.div(bit_offsets, 8, rounding_mode="floor")
        bit_shifts = torch.remainder(bit_offsets, 8)

        gather_shape = (1, 1, 1, self.head_dim)
        expand_shape = (self.batch_size, self.num_heads, length, self.head_dim)
        byte_index = byte_offsets.view(gather_shape).expand(expand_shape)
        low = packed.gather(-1, byte_index)
        values = torch.bitwise_right_shift(low, bit_shifts.view(gather_shape))

        spill = bit_shifts + int(self.bits) - 8
        spill_mask = spill > 0
        if bool(spill_mask.any()):
            high_index = (byte_offsets + 1).clamp_max(self.q_idx_packed_dim - 1)
            high = packed.gather(-1, high_index.view(gather_shape).expand(expand_shape))
            high_shift = (8 - bit_shifts).view(gather_shape)
            values = torch.where(
                spill_mask.view(gather_shape),
                torch.bitwise_or(values, torch.bitwise_left_shift(high, high_shift)),
                values,
            )

        return torch.bitwise_and(values, (1 << int(self.bits)) - 1).to(torch.long)

    def _encode(self, tensor: torch.Tensor):
        x = tensor.to(torch.float32)
        scale = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True)).clamp_min_(1e-6)
        rotated = x @ self.rotation
        rotated.div_(scale)
        q_idx = torch.bucketize(rotated, self.boundaries).to(torch.uint8)
        if self.residual_dim <= 0:
            return q_idx, scale, None, None

        decoded_rotated = self.codebook[q_idx.to(torch.long)]
        decoded_rotated.mul_(scale)
        decoded = decoded_rotated @ self.rotation_t
        residual = x - decoded
        residual_norm = torch.sqrt(torch.sum(residual * residual, dim=-1, keepdim=True)).clamp_min_(1e-6)
        residual_proj = residual @ self.residual_proj
        return (
            q_idx,
            scale,
            self._pack_residual_sign(residual_proj >= 0),
            residual_norm,
        )

    def append_compressed(self, tensor: torch.Tensor, dim: int = 2):
        if dim != 2:
            raise ValueError("TurboQuantizedKVCache.append_compressed currently supports dim=2 only.")
        if tensor.shape[-1] != self.head_dim:
            raise ValueError(
                "TurboQuantizedKVCache head_dim mismatch: "
                f"cache head_dim={self.head_dim}, tensor head_dim={tensor.shape[-1]}"
            )
        start = int(self.current_length.item())
        end = start + tensor.shape[dim]
        if end > self.max_length:
            raise RuntimeError(
                "TurboQuantizedKVCache capacity exceeded: "
                f"append [{start}, {end}) with max_length={self.max_length}. "
                "Increase --kv-max-length or reduce prompt/tree length."
            )
        if turbo_vq_append_triton is not None and turbo_vq_append_triton(self, tensor, start):
            self.current_length.fill_(end)
            return start, end

        q_idx, scale, residual_sign, residual_norm = self._encode(tensor)

        packed_q_idx = self._pack_q_idx(q_idx)
        self.q_idx.narrow(dim, start, tensor.shape[dim]).copy_(
            packed_q_idx,
            non_blocking=True,
        )
        self.scale.narrow(dim, start, tensor.shape[dim]).copy_(scale, non_blocking=True)
        if self.residual_sign_packed is not None:
            self.residual_sign_packed.narrow(dim, start, tensor.shape[dim]).copy_(
                residual_sign,
                non_blocking=True,
            )
            self.residual_norm.narrow(dim, start, tensor.shape[dim]).copy_(
                residual_norm,
                non_blocking=True,
            )
        self.current_length.fill_(end)
        return start, end

    def _decode_range(self, start: int, end: int, out: torch.Tensor = None) -> torch.Tensor:
        q_idx = self._unpack_q_idx_range(start, end)
        scale = self.scale[:, :, start:end].to(torch.float32)
        rotated = self.codebook[q_idx] * scale
        decoded = rotated @ self.rotation_t

        if out is None:
            out = torch.empty(
                self.batch_size,
                self.num_heads,
                end - start,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
        out.copy_(decoded.to(self.dtype), non_blocking=True)
        return out

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        if dim != 2:
            raise ValueError("TurboQuantizedKVCache.copy currently supports dim=2 only.")
        tgt_q_idx = self.q_idx.index_select(dim, indices)
        tgt_scale = self.scale.index_select(dim, indices)

        self.q_idx.narrow(dim, prev_length, tgt_q_idx.shape[dim]).copy_(
            tgt_q_idx, non_blocking=True
        )
        self.scale.narrow(dim, prev_length, tgt_scale.shape[dim]).copy_(
            tgt_scale, non_blocking=True
        )
        if self.residual_sign_packed is not None:
            tgt_sign = self.residual_sign_packed.index_select(dim, indices)
            tgt_norm = self.residual_norm.index_select(dim, indices)
            self.residual_sign_packed.narrow(dim, prev_length, tgt_sign.shape[dim]).copy_(
                tgt_sign,
                non_blocking=True,
            )
            self.residual_norm.narrow(dim, prev_length, tgt_norm.shape[dim]).copy_(
                tgt_norm,
                non_blocking=True,
            )
        if self.dequant_data is not None:
            n_sel = tgt_q_idx.shape[dim]
            shadow_dst = self.dequant_data.narrow(dim, prev_length, n_sel)
            shadow_src = self.dequant_data.index_select(dim, indices)
            shadow_dst.copy_(shadow_src, non_blocking=True)
        self.current_length.fill_(prev_length + tgt_q_idx.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        if dim != 2:
            raise ValueError("TurboQuantizedKVCache.cat currently supports dim=2 only.")
        start, end = self.append_compressed(tensor, dim=dim)
        if self.dequant_data is not None:
            shadow_slice = self.dequant_data.narrow(dim, start, tensor.shape[dim])
            if tensor.dtype != self.dtype:
                shadow_slice.copy_(tensor.to(self.dtype), non_blocking=True)
            else:
                shadow_slice.copy_(tensor, non_blocking=True)
            return torch.narrow(self.dequant_data, 2, 0, int(self.current_length.item()))
        return self._decode_range(0, end)

    def reset(self):
        self.current_length.fill_(0)


class HybridTurboVQKVCache:
    """
    Hybrid TurboVQ KV cache:
    - all logical positions are stored in the compressed TurboVQ cache
    - the most recent `hot_window` positions are also kept exactly in FP16/BF16
    - tree verification can materialize old compressed KV + exact hot KV for SDPA
    - q_len=1 decode can use a fused hybrid Triton attention kernel
    """

    is_hybrid_turbo_vq = True

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_length: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        current_length: torch.Tensor,
        bits: int = 8,
        residual_dim: int = 128,
        residual_scale: float = 1.0,
        hot_window: int = 512,
        quant_seed: int = 0,
    ):
        self.compressed_cache = TurboQuantizedKVCache(
            batch_size=batch_size,
            num_heads=num_heads,
            max_length=max_length,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
            current_length=current_length,
            bits=bits,
            residual_dim=residual_dim,
            residual_scale=residual_scale,
            runtime_dequant_cache=False,
            quant_seed=quant_seed,
        )
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_length = max_length
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.current_length = current_length
        self.hot_capacity = int(max(1, min(int(hot_window), int(max_length))))
        self.hot_window = self.hot_capacity
        self.dequant_data = None

        # Expose compressed tensors/metadata so the existing fused kernels can
        # consume this cache without unpacking the wrapper.
        self.bits = self.compressed_cache.bits
        self.levels = self.compressed_cache.levels
        self.q_idx_is_packed = self.compressed_cache.q_idx_is_packed
        self.q_idx_packed_dim = self.compressed_cache.q_idx_packed_dim
        self.residual_dim = self.compressed_cache.residual_dim
        self.residual_packed_dim = self.compressed_cache.residual_packed_dim
        self.residual_scale = self.compressed_cache.residual_scale
        self.residual_coeff = self.compressed_cache.residual_coeff
        self.rotation = self.compressed_cache.rotation
        self.rotation_t = self.compressed_cache.rotation_t
        self.codebook = self.compressed_cache.codebook
        self.boundaries = self.compressed_cache.boundaries
        self.residual_proj = self.compressed_cache.residual_proj
        self.q_idx = self.compressed_cache.q_idx
        self.scale = self.compressed_cache.scale
        self.residual_sign_packed = self.compressed_cache.residual_sign_packed
        self.residual_norm = self.compressed_cache.residual_norm
        self.residual_bit_shifts = self.compressed_cache.residual_bit_shifts
        self.residual_pack_weights = self.compressed_cache.residual_pack_weights

        self.hot_data = torch.zeros(
            batch_size,
            num_heads,
            self.hot_capacity,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._hot_positions = torch.full((self.hot_capacity,), -1, dtype=torch.long)
        self._attention_out_cache = {}

    @property
    def shape(self):
        return (
            self.batch_size,
            self.num_heads,
            int(self.current_length.item()),
            self.head_dim,
        )

    def get_attention_output(self, query_states: torch.Tensor) -> torch.Tensor:
        key = (tuple(query_states.shape), query_states.dtype)
        cached = self._attention_out_cache.get(key)
        if cached is None:
            cached = torch.empty(
                tuple(query_states.shape),
                dtype=query_states.dtype,
                device=query_states.device,
            )
            self._attention_out_cache[key] = cached
        return cached

    def old_length(self, kv_len: int = None) -> int:
        if kv_len is None:
            kv_len = int(self.current_length.item())
        return max(0, int(kv_len) - self.hot_capacity)

    def _record_hot_slice(self, tensor: torch.Tensor, start: int):
        input_len = int(tensor.shape[2])
        if input_len <= 0:
            return
        if start == 0:
            self._hot_positions.fill_(-1)

        copy_len = min(input_len, self.hot_capacity)
        copy_start = input_len - copy_len
        positions = torch.arange(
            int(start) + copy_start,
            int(start) + input_len,
            device=self.device,
            dtype=torch.long,
        )
        slots = torch.remainder(positions, self.hot_capacity)
        src = tensor[:, :, copy_start:].to(dtype=self.dtype)
        self.hot_data.index_copy_(2, slots, src)
        self._hot_positions[slots.detach().cpu()] = positions.detach().cpu()

    def _range_is_hot_exact(self, start: int, end: int) -> bool:
        length = int(end) - int(start)
        if length < 0 or length > self.hot_capacity:
            return False
        for pos in range(int(start), int(end)):
            slot = pos % self.hot_capacity
            if int(self._hot_positions[slot].item()) != pos:
                return False
        return True

    def _gather_hot_range(self, start: int, end: int, out: torch.Tensor = None):
        slots = torch.arange(
            int(start),
            int(end),
            device=self.device,
            dtype=torch.long,
        ).remainder_(self.hot_capacity)
        hot = self.hot_data.index_select(2, slots)
        if out is None:
            return hot
        out.copy_(hot, non_blocking=True)
        return out

    def _overlay_hot_range(self, out: torch.Tensor, start: int, end: int):
        valid_slots = []
        rel_positions = []
        for slot, pos_tensor in enumerate(self._hot_positions.tolist()):
            pos = int(pos_tensor)
            if int(start) <= pos < int(end):
                valid_slots.append(slot)
                rel_positions.append(pos - int(start))
        if not valid_slots:
            return out

        order = sorted(range(len(rel_positions)), key=rel_positions.__getitem__)
        slots = torch.tensor([valid_slots[i] for i in order], device=self.device, dtype=torch.long)
        rel = torch.tensor([rel_positions[i] for i in order], device=self.device, dtype=torch.long)
        hot = self.hot_data.index_select(2, slots)
        out.index_copy_(2, rel, hot)
        return out

    def append_compressed(self, tensor: torch.Tensor, dim: int = 2):
        if dim != 2:
            raise ValueError("HybridTurboVQKVCache.append_compressed currently supports dim=2 only.")
        start, end = self.compressed_cache.append_compressed(tensor, dim=dim)
        self._record_hot_slice(tensor, start)
        return start, end

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        if dim != 2:
            raise ValueError("HybridTurboVQKVCache.copy currently supports dim=2 only.")
        source_positions = [int(x) for x in indices.detach().cpu().tolist()]
        self.compressed_cache.copy(indices, prev_length, dim=dim)

        if prev_length == 0:
            self._hot_positions.fill_(-1)

        for offset, src_pos in enumerate(source_positions):
            dst_pos = int(prev_length) + int(offset)
            dst_slot = dst_pos % self.hot_capacity
            src_slot = src_pos % self.hot_capacity
            if int(self._hot_positions[src_slot].item()) == src_pos:
                src = self.hot_data[:, :, src_slot : src_slot + 1]
            else:
                # Rare fallback for a copied source outside the exact hot window.
                # The accepted Medusa branch is normally hot, but this keeps the
                # cache structurally valid if a custom tree reaches farther back.
                src = self.compressed_cache._decode_range(src_pos, src_pos + 1)
            self.hot_data[:, :, dst_slot : dst_slot + 1].copy_(src, non_blocking=True)
            self._hot_positions[dst_slot] = dst_pos

    def _decode_range(self, start: int, end: int, out: torch.Tensor = None) -> torch.Tensor:
        if self._range_is_hot_exact(start, end):
            return self._gather_hot_range(start, end, out=out)

        decoded = self.compressed_cache._decode_range(start, end, out=out)
        return self._overlay_hot_range(decoded, start, end)

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        if dim != 2:
            raise ValueError("HybridTurboVQKVCache.cat currently supports dim=2 only.")
        _, end = self.append_compressed(tensor, dim=dim)
        return self._decode_range(0, end)

    def reset(self):
        self.compressed_cache.reset()
        self.current_length.fill_(0)
        self._hot_positions.fill_(-1)


class OutlierAwareTurboVQKVCache:
    """
    Outlier-aware TurboQuant cache from the KV-cache recipe in the paper:
    split high-magnitude channels from the first prefill batch and quantize
    them with a separate, higher-bit TurboQuant instance.
    """

    is_outlier_turbo_vq = True

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_length: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        current_length: torch.Tensor,
        regular_bits: int = 3,
        outlier_bits: int = 4,
        n_outlier: int = 16,
        residual_dim: int = -1,
        residual_scale: float = 1.0,
        runtime_dequant_cache: bool = False,
        outlier_idx=None,
        quant_seed: int = 0,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_length = max_length
        self.head_dim = int(head_dim)
        self.device = device
        self.dtype = dtype
        self.current_length = current_length
        self.regular_bits = int(regular_bits)
        self.outlier_bits = int(outlier_bits)
        self.n_outlier = int(max(0, min(int(n_outlier), self.head_dim - 1)))
        self.n_regular = self.head_dim - self.n_outlier
        self.residual_dim_request = int(residual_dim)
        self.residual_scale = float(residual_scale)
        self.runtime_dequant_cache = bool(runtime_dequant_cache)
        self.quant_seed = int(quant_seed)

        self.outlier_idx = None
        self.regular_idx = None
        self.regular_current_length = torch.zeros((), dtype=torch.long, device="cpu")
        self.outlier_current_length = torch.zeros((), dtype=torch.long, device="cpu")
        self.regular_cache = None
        self.outlier_cache = None
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
        if outlier_idx is not None:
            self._set_outlier_channels(outlier_idx)

    @property
    def shape(self):
        return (
            self.batch_size,
            self.num_heads,
            int(self.current_length.item()),
            self.head_dim,
        )

    def _set_outlier_channels(self, outlier_idx):
        if outlier_idx is None:
            outlier_idx = torch.empty(0, dtype=torch.long, device=self.device)
        if not torch.is_tensor(outlier_idx):
            outlier_idx = torch.tensor(outlier_idx, dtype=torch.long, device=self.device)
        outlier_idx = outlier_idx.to(device=self.device, dtype=torch.long).flatten()
        if outlier_idx.numel() > 0:
            outlier_idx = torch.unique(outlier_idx, sorted=True)
            valid = (outlier_idx >= 0) & (outlier_idx < self.head_dim)
            outlier_idx = outlier_idx[valid]
        if outlier_idx.numel() > self.head_dim - 1:
            outlier_idx = outlier_idx[: self.head_dim - 1]
        self.n_outlier = int(outlier_idx.numel())
        self.n_regular = self.head_dim - self.n_outlier
        self.outlier_idx = outlier_idx
        mask = torch.ones(self.head_dim, dtype=torch.bool, device=self.device)
        if self.n_outlier > 0:
            mask[self.outlier_idx] = False
        self.regular_idx = torch.arange(
            self.head_dim,
            dtype=torch.long,
            device=self.device,
        )[mask]

        self._initialize_child_caches()

    def _initialize_child_caches(self):
        if self.n_outlier <= 0:
            self.outlier_idx = torch.empty(0, dtype=torch.long, device=self.device)
            self.regular_idx = torch.arange(self.head_dim, dtype=torch.long, device=self.device)
            self.n_regular = self.head_dim

        regular_residual_dim = (
            self.n_regular
            if self.residual_dim_request < 0
            else min(max(0, self.residual_dim_request), self.n_regular)
        )
        outlier_residual_dim = (
            self.n_outlier
            if self.residual_dim_request < 0
            else min(max(0, self.residual_dim_request), self.n_outlier)
        )
        self.regular_cache = TurboQuantizedKVCache(
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            max_length=self.max_length,
            head_dim=self.n_regular,
            device=self.device,
            dtype=self.dtype,
            current_length=self.regular_current_length,
            bits=self.regular_bits,
            residual_dim=regular_residual_dim,
            residual_scale=self.residual_scale,
            runtime_dequant_cache=False,
            quant_seed=self.quant_seed,
        )
        self.outlier_cache = TurboQuantizedKVCache(
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            max_length=self.max_length,
            head_dim=max(1, self.n_outlier),
            device=self.device,
            dtype=self.dtype,
            current_length=self.outlier_current_length,
            bits=self.outlier_bits,
            residual_dim=outlier_residual_dim,
            residual_scale=self.residual_scale,
            runtime_dequant_cache=False,
            quant_seed=self.quant_seed + 17,
        )

    def _validate_runtime_layout(self, tensor: torch.Tensor):
        tensor_head_dim = int(tensor.shape[-1])
        if tensor_head_dim != self.head_dim:
            if int(self.current_length.item()) != 0:
                raise ValueError(
                    "OutlierAwareTurboVQKVCache cannot change head_dim after data was appended: "
                    f"cache head_dim={self.head_dim}, tensor head_dim={tensor_head_dim}"
                )
            self.head_dim = tensor_head_dim
            self.n_outlier = int(max(0, min(int(self.n_outlier), self.head_dim - 1)))
            self.n_regular = self.head_dim - self.n_outlier
            self.outlier_idx = None
            self.regular_idx = None
            self.regular_cache = None
            self.outlier_cache = None
            if self.runtime_dequant_cache:
                self.dequant_data = torch.zeros(
                    self.batch_size,
                    self.num_heads,
                    self.max_length,
                    self.head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
        if self.outlier_idx is not None and self.outlier_idx.numel() > 0:
            min_idx = int(self.outlier_idx.min().item())
            max_idx = int(self.outlier_idx.max().item())
            if min_idx < 0 or max_idx >= self.head_dim:
                raise ValueError(
                    "OutlierAwareTurboVQKVCache outlier indices are outside the runtime head_dim: "
                    f"min={min_idx}, max={max_idx}, head_dim={self.head_dim}"
                )

    def _select_outlier_channels(self, tensor: torch.Tensor):
        self._validate_runtime_layout(tensor)
        if self.outlier_idx is not None:
            return
        if self.n_outlier <= 0:
            self._set_outlier_channels(None)
            return
        avg_mag = tensor.to(torch.float32).abs().mean(dim=(0, 1, 2))
        _, top = torch.topk(avg_mag, k=self.n_outlier, largest=True, sorted=False)
        self._set_outlier_channels(top.sort()[0].to(device=self.device))

    def append_compressed(self, tensor: torch.Tensor, dim: int = 2):
        if dim != 2:
            raise ValueError("OutlierAwareTurboVQKVCache.append_compressed currently supports dim=2 only.")
        self._select_outlier_channels(tensor)
        start = int(self.current_length.item())
        end = start + int(tensor.shape[dim])
        if end > self.max_length:
            raise RuntimeError(
                "OutlierAwareTurboVQKVCache capacity exceeded: "
                f"append [{start}, {end}) with max_length={self.max_length}. "
                "Increase --kv-max-length or reduce prompt/tree length."
            )

        regular_tensor = tensor.index_select(-1, self.regular_idx)
        self.regular_cache.append_compressed(regular_tensor, dim=dim)
        if self.n_outlier > 0:
            outlier_tensor = tensor.index_select(-1, self.outlier_idx)
        else:
            outlier_tensor = torch.zeros(
                *tensor.shape[:-1],
                1,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        self.outlier_cache.append_compressed(outlier_tensor, dim=dim)

        self.current_length.fill_(end)
        if self.dequant_data is not None:
            dst = self.dequant_data.narrow(dim, start, tensor.shape[dim])
            dst.copy_(tensor.to(self.dtype), non_blocking=True)
        return start, end

    def _decode_range(self, start: int, end: int, out: torch.Tensor = None) -> torch.Tensor:
        if self.regular_cache is None:
            if out is None:
                out = torch.empty(
                    self.batch_size,
                    self.num_heads,
                    end - start,
                    self.head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
            return out.zero_()

        if self.dequant_data is not None:
            view = self.dequant_data[:, :, start:end]
            if out is None:
                return view
            out.copy_(view, non_blocking=True)
            return out

        regular = self.regular_cache._decode_range(start, end)
        outlier = self.outlier_cache._decode_range(start, end) if self.n_outlier > 0 else None
        if out is None:
            out = torch.empty(
                self.batch_size,
                self.num_heads,
                end - start,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
        out.index_copy_(-1, self.regular_idx, regular)
        if outlier is not None:
            out.index_copy_(-1, self.outlier_idx, outlier)
        return out

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        if dim != 2:
            raise ValueError("OutlierAwareTurboVQKVCache.copy currently supports dim=2 only.")
        if self.regular_cache is None:
            self.current_length.fill_(prev_length)
            return
        self.regular_cache.copy(indices, prev_length, dim=dim)
        self.outlier_cache.copy(indices, prev_length, dim=dim)
        if self.dequant_data is not None:
            tgt = self.dequant_data.index_select(dim, indices)
            dst = self.dequant_data.narrow(dim, prev_length, tgt.shape[dim])
            dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + int(indices.numel()))

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        if dim != 2:
            raise ValueError("OutlierAwareTurboVQKVCache.cat currently supports dim=2 only.")
        _, end = self.append_compressed(tensor, dim=dim)
        return self._decode_range(0, end)

    def reset(self):
        self.current_length.fill_(0)
        self.regular_current_length.fill_(0)
        self.outlier_current_length.fill_(0)
        if self.regular_cache is not None and hasattr(self.regular_cache, "reset"):
            self.regular_cache.reset()
        if self.outlier_cache is not None and hasattr(self.outlier_cache, "reset"):
            self.outlier_cache.reset()


class HybridOutlierTurboVQKVCache:
    """
    Outlier-aware compressed cache plus an exact recent hot window.
    """

    is_hybrid_outlier_turbo_vq = True

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_length: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        current_length: torch.Tensor,
        regular_bits: int = 3,
        outlier_bits: int = 4,
        n_outlier: int = 16,
        residual_dim: int = -1,
        residual_scale: float = 1.0,
        hot_window: int = 512,
        outlier_idx=None,
        quant_seed: int = 0,
    ):
        self.compressed_cache = OutlierAwareTurboVQKVCache(
            batch_size=batch_size,
            num_heads=num_heads,
            max_length=max_length,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
            current_length=current_length,
            regular_bits=regular_bits,
            outlier_bits=outlier_bits,
            n_outlier=n_outlier,
            residual_dim=residual_dim,
            residual_scale=residual_scale,
            runtime_dequant_cache=False,
            outlier_idx=outlier_idx,
            quant_seed=quant_seed,
        )
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_length = max_length
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.current_length = current_length
        self.hot_capacity = int(max(1, min(int(hot_window), int(max_length))))
        self.hot_window = self.hot_capacity
        self.dequant_data = None
        self.hot_data = torch.zeros(
            batch_size,
            num_heads,
            self.hot_capacity,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._hot_positions = torch.full((self.hot_capacity,), -1, dtype=torch.long)

    @property
    def shape(self):
        return (
            self.batch_size,
            self.num_heads,
            int(self.current_length.item()),
            self.head_dim,
        )

    def _record_hot_slice(self, tensor: torch.Tensor, start: int):
        input_len = int(tensor.shape[2])
        if input_len <= 0:
            return
        if start == 0:
            self._hot_positions.fill_(-1)
        copy_len = min(input_len, self.hot_capacity)
        copy_start = input_len - copy_len
        positions = torch.arange(
            int(start) + copy_start,
            int(start) + input_len,
            device=self.device,
            dtype=torch.long,
        )
        slots = torch.remainder(positions, self.hot_capacity)
        src = tensor[:, :, copy_start:].to(dtype=self.dtype)
        self.hot_data.index_copy_(2, slots, src)
        self._hot_positions[slots.detach().cpu()] = positions.detach().cpu()

    def _range_is_hot_exact(self, start: int, end: int) -> bool:
        length = int(end) - int(start)
        if length < 0 or length > self.hot_capacity:
            return False
        for pos in range(int(start), int(end)):
            slot = pos % self.hot_capacity
            if int(self._hot_positions[slot].item()) != pos:
                return False
        return True

    def _gather_hot_range(self, start: int, end: int, out: torch.Tensor = None):
        slots = torch.arange(
            int(start),
            int(end),
            device=self.device,
            dtype=torch.long,
        ).remainder_(self.hot_capacity)
        hot = self.hot_data.index_select(2, slots)
        if out is None:
            return hot
        out.copy_(hot, non_blocking=True)
        return out

    def _overlay_hot_range(self, out: torch.Tensor, start: int, end: int):
        valid_slots = []
        rel_positions = []
        for slot, pos_tensor in enumerate(self._hot_positions.tolist()):
            pos = int(pos_tensor)
            if int(start) <= pos < int(end):
                valid_slots.append(slot)
                rel_positions.append(pos - int(start))
        if not valid_slots:
            return out
        order = sorted(range(len(rel_positions)), key=rel_positions.__getitem__)
        slots = torch.tensor([valid_slots[i] for i in order], device=self.device, dtype=torch.long)
        rel = torch.tensor([rel_positions[i] for i in order], device=self.device, dtype=torch.long)
        hot = self.hot_data.index_select(2, slots)
        out.index_copy_(2, rel, hot)
        return out

    def append_compressed(self, tensor: torch.Tensor, dim: int = 2):
        if dim != 2:
            raise ValueError("HybridOutlierTurboVQKVCache.append_compressed currently supports dim=2 only.")
        start, end = self.compressed_cache.append_compressed(tensor, dim=dim)
        self._record_hot_slice(tensor, start)
        return start, end

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        if dim != 2:
            raise ValueError("HybridOutlierTurboVQKVCache.copy currently supports dim=2 only.")
        source_positions = [int(x) for x in indices.detach().cpu().tolist()]
        self.compressed_cache.copy(indices, prev_length, dim=dim)
        if prev_length == 0:
            self._hot_positions.fill_(-1)
        for offset, src_pos in enumerate(source_positions):
            dst_pos = int(prev_length) + int(offset)
            dst_slot = dst_pos % self.hot_capacity
            src_slot = src_pos % self.hot_capacity
            if int(self._hot_positions[src_slot].item()) == src_pos:
                src = self.hot_data[:, :, src_slot : src_slot + 1]
            else:
                src = self.compressed_cache._decode_range(src_pos, src_pos + 1)
            self.hot_data[:, :, dst_slot : dst_slot + 1].copy_(src, non_blocking=True)
            self._hot_positions[dst_slot] = dst_pos

    def _decode_range(self, start: int, end: int, out: torch.Tensor = None) -> torch.Tensor:
        if self._range_is_hot_exact(start, end):
            return self._gather_hot_range(start, end, out=out)
        decoded = self.compressed_cache._decode_range(start, end, out=out)
        return self._overlay_hot_range(decoded, start, end)

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        if dim != 2:
            raise ValueError("HybridOutlierTurboVQKVCache.cat currently supports dim=2 only.")
        _, end = self.append_compressed(tensor, dim=dim)
        return self._decode_range(0, end)

    def reset(self):
        self.compressed_cache.reset()
        self.current_length.fill_(0)
        self._hot_positions.fill_(-1)


def turbo_vq_attention_with_qjl_residual(
    query_states: torch.Tensor,
    key_cache,
    value_cache,
    attention_mask: torch.Tensor,
    num_key_value_groups: int,
    head_dim: int,
):
    """
    TurboQuant inner-product attention:
    score(q, k) ~= <q, k_quant> + QJL(q, k - k_quant).

    Values are decoded from the MSE-stage quantizer. This is intentionally a
    readable reference path; once validated, the same math can be fused.
    """
    if query_states.dim() != 4:
        return None

    kv_len = int(key_cache.current_length.item())
    if kv_len <= 0:
        return None

    if isinstance(key_cache, (OutlierAwareTurboVQKVCache, HybridOutlierTurboVQKVCache)):
        compressed_key_cache = (
            key_cache.compressed_cache
            if isinstance(key_cache, HybridOutlierTurboVQKVCache)
            else key_cache
        )
        if compressed_key_cache.regular_cache is None:
            return None

        q_len = int(query_states.shape[2])
        key_states = compressed_key_cache._decode_range(0, kv_len)
        value_states = value_cache._decode_range(0, kv_len)
        key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

        def _child_qjl_state(child_cache):
            if (
                child_cache.residual_sign_packed is None
                or child_cache.residual_proj is None
                or int(child_cache.residual_dim) <= 0
            ):
                return None
            residual_sign = child_cache._unpack_residual_sign_range(0, kv_len)
            residual_norm = child_cache.residual_norm[:, :, :kv_len, 0].to(torch.float32)
            return (
                child_cache.residual_proj,
                residual_sign.repeat_interleave(num_key_value_groups, dim=1),
                residual_norm.repeat_interleave(num_key_value_groups, dim=1),
                child_cache.residual_coeff,
            )

        def _apply_child_qjl_correction(scores, query_sub, child_state):
            if child_state is None:
                return scores
            residual_proj, residual_sign, residual_norm, residual_coeff = child_state
            if float(residual_coeff) == 0.0:
                return scores
            q_residual_proj = query_sub.to(torch.float32) @ residual_proj
            residual_inner = torch.einsum("bhqm,bhkm->bhqk", q_residual_proj, residual_sign)
            return scores + (residual_coeff * residual_inner * residual_norm.unsqueeze(2))

        regular_state = _child_qjl_state(compressed_key_cache.regular_cache)
        outlier_state = (
            _child_qjl_state(compressed_key_cache.outlier_cache)
            if compressed_key_cache.n_outlier > 0
            else None
        )
        query_regular = query_states.index_select(-1, compressed_key_cache.regular_idx)
        query_outlier = (
            query_states.index_select(-1, compressed_key_cache.outlier_idx)
            if compressed_key_cache.n_outlier > 0
            else None
        )

        hot_positions = None
        hot_keys = None
        if isinstance(key_cache, HybridOutlierTurboVQKVCache):
            valid_slots = []
            hot_pos_list = []
            for slot, pos_tensor in enumerate(key_cache._hot_positions.tolist()):
                pos = int(pos_tensor)
                if 0 <= pos < kv_len:
                    valid_slots.append(slot)
                    hot_pos_list.append(pos)
            if valid_slots:
                order = sorted(range(len(hot_pos_list)), key=hot_pos_list.__getitem__)
                slots = torch.tensor([valid_slots[i] for i in order], device=key_cache.device, dtype=torch.long)
                hot_positions = torch.tensor([hot_pos_list[i] for i in order], device=key_cache.device, dtype=torch.long)
                hot_keys = key_cache.hot_data.index_select(2, slots)
                hot_keys = hot_keys.repeat_interleave(num_key_value_groups, dim=1)

        scale = 1.0 / math.sqrt(float(head_dim))
        out = torch.empty_like(query_states)
        key_t = key_states.transpose(2, 3).to(torch.float32)
        value_f = value_states.to(torch.float32)
        chunk_size = 64 if q_len > 1 else 1
        kv_offsets = None
        if attention_mask is None and q_len > 1:
            kv_offsets = torch.arange(kv_len, device=query_states.device, dtype=torch.long)
            past_len = max(0, kv_len - q_len)

        for q_start in range(0, q_len, chunk_size):
            q_end = min(q_len, q_start + chunk_size)
            q_chunk = query_states[:, :, q_start:q_end]
            scores = torch.matmul(q_chunk.to(torch.float32), key_t)
            scores = _apply_child_qjl_correction(
                scores,
                query_regular[:, :, q_start:q_end],
                regular_state,
            )
            if query_outlier is not None:
                scores = _apply_child_qjl_correction(
                    scores,
                    query_outlier[:, :, q_start:q_end],
                    outlier_state,
                )

            if hot_positions is not None and hot_keys is not None:
                hot_scores = torch.einsum(
                    "bhqd,bhkd->bhqk",
                    q_chunk.to(torch.float32),
                    hot_keys.to(torch.float32),
                )
                scores.index_copy_(-1, hot_positions, hot_scores)

            scores = scores * scale
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, q_start:q_end, :]
            elif kv_offsets is not None:
                query_positions = torch.arange(
                    past_len + q_start,
                    past_len + q_end,
                    device=query_states.device,
                    dtype=torch.long,
                )
                causal_mask = kv_offsets.view(1, 1, 1, kv_len) > query_positions.view(1, 1, -1, 1)
                scores = scores.masked_fill(causal_mask, -float("inf"))

            attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
            out[:, :, q_start:q_end] = torch.matmul(attn_weights, value_f).to(
                query_states.dtype
            )

        return out

    if (
        not isinstance(key_cache, TurboQuantizedKVCache)
        or not isinstance(value_cache, TurboQuantizedKVCache)
        or key_cache.residual_sign_packed is None
        or key_cache.residual_proj is None
    ):
        return None

    key_states = key_cache._decode_range(0, kv_len)
    value_states = value_cache._decode_range(0, kv_len)
    key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
    value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

    scale = 1.0 / math.sqrt(float(head_dim))
    attn_weights = torch.matmul(
        query_states.to(torch.float32),
        key_states.transpose(2, 3).to(torch.float32),
    )

    q_residual_proj = query_states.to(torch.float32) @ key_cache.residual_proj
    residual_sign = key_cache._unpack_residual_sign_range(0, kv_len)
    residual_norm = key_cache.residual_norm[:, :, :kv_len, 0].to(torch.float32)
    residual_sign = residual_sign.repeat_interleave(num_key_value_groups, dim=1)
    residual_norm = residual_norm.repeat_interleave(num_key_value_groups, dim=1)
    residual_inner = torch.einsum("bhqm,bhkm->bhqk", q_residual_proj, residual_sign)
    attn_weights = attn_weights + (
        key_cache.residual_coeff * residual_inner * residual_norm.unsqueeze(2)
    )
    attn_weights = attn_weights * scale

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return torch.matmul(attn_weights, value_states)


def _build_polar_angle_codebook(level: int, bits: int, device: torch.device):
    level = int(level)
    bits = int(bits)
    key = _cache_key(device, "polar_angle", level, bits)
    cached = _POLAR_ANGLE_CODEBOOK_CACHE.get(key)
    if cached is not None:
        return cached

    levels = 1 << bits
    if level == 1:
        step = (2.0 * math.pi) / float(levels)
        codebook = (torch.arange(levels, dtype=torch.float64) + 0.5) * step
    else:
        grid = torch.linspace(0.0, math.pi / 2.0, 20001, dtype=torch.float64)
        exponent = max(1, (1 << (level - 1)) - 1)
        pdf = torch.sin(2.0 * grid).clamp_min(0.0).pow(exponent)
        centroids = (torch.arange(levels, dtype=torch.float64) + 0.5) * (
            (math.pi / 2.0) / float(levels)
        )
        for _ in range(80):
            boundaries = (centroids[:-1] + centroids[1:]) * 0.5
            bucket = torch.bucketize(grid, boundaries)
            new_centroids = centroids.clone()
            for idx in range(levels):
                mask = bucket == idx
                weight = pdf[mask].sum()
                if weight > 0:
                    new_centroids[idx] = (grid[mask] * pdf[mask]).sum() / weight
            if torch.max(torch.abs(new_centroids - centroids)) < 1e-7:
                centroids = new_centroids
                break
            centroids = new_centroids
        codebook = centroids

    codebook = codebook.to(device=device, dtype=torch.float32).contiguous()
    boundaries = ((codebook[:-1] + codebook[1:]) * 0.5).contiguous()
    _POLAR_ANGLE_CODEBOOK_CACHE[key] = (codebook, boundaries)
    return _POLAR_ANGLE_CODEBOOK_CACHE[key]


class PolarQuantizedKVCache:
    """
    Paper-style PolarQuant cache:
    - shared random orthogonal preconditioning
    - recursive polar transform for L levels
    - level-wise scalar angle codebooks
    - final radii stored in FP16/BF16

    The PolarQuant paper's practical recipe uses L=4, 4 bits for level-1
    angles, and 2 bits for later angle levels.
    """

    is_recursive_polar = True

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_length: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        current_length: torch.Tensor,
        first_level_bits: int = 4,
        other_level_bits: int = 2,
        polar_levels: int = 4,
        runtime_dequant_cache: bool = True,
        hot_window: int = 0,
        quant_seed: int = 0,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_length = max_length
        self.head_dim = int(head_dim)
        self.device = device
        self.dtype = dtype
        self.current_length = current_length
        self.first_level_bits = int(first_level_bits)
        self.other_level_bits = int(other_level_bits)
        self.quant_seed = int(quant_seed)
        self.polar_levels = int(max(1, min(int(polar_levels), int(math.log2(self.head_dim)))))
        if self.head_dim & (self.head_dim - 1):
            raise ValueError("Paper PolarQuant requires power-of-two head_dim.")
        if self.head_dim % (1 << self.polar_levels) != 0:
            raise ValueError("head_dim must be divisible by 2**polar_levels.")
        if not (1 <= self.first_level_bits <= 8 and 1 <= self.other_level_bits <= 8):
            raise ValueError("PolarQuant angle bits must be in [1, 8].")

        self.rotation = _get_turbo_rotation(self.head_dim, device, seed=self.quant_seed)
        self.rotation_t = self.rotation.t().contiguous()
        self.final_dim = self.head_dim // (1 << self.polar_levels)
        self.level_dims = [self.head_dim // (1 << level) for level in range(1, self.polar_levels + 1)]
        self.level_bits = [
            self.first_level_bits if level == 1 else self.other_level_bits
            for level in range(1, self.polar_levels + 1)
        ]
        self.level_packed_dims = [
            (dim * bits + 7) // 8 for dim, bits in zip(self.level_dims, self.level_bits)
        ]
        self.hot_capacity = int(max(0, min(int(hot_window), int(max_length))))
        self.angle_codebooks = [
            _build_polar_angle_codebook(level, bits, device)
            for level, bits in enumerate(self.level_bits, start=1)
        ]
        self.angle_cos_luts = [
            codebook.cos().contiguous() for codebook, _ in self.angle_codebooks
        ]
        self.angle_sin_luts = [
            codebook.sin().contiguous() for codebook, _ in self.angle_codebooks
        ]

        self.final_radius = torch.zeros(
            batch_size,
            num_heads,
            max_length,
            self.final_dim,
            dtype=torch.float16,
            device=device,
        )
        self.angle_q = [
            torch.zeros(
                batch_size,
                num_heads,
                max_length,
                packed_dim,
                dtype=torch.uint8,
                device=device,
            )
            for packed_dim in self.level_packed_dims
        ]
        self.dequant_data = None
        if runtime_dequant_cache:
            self.dequant_data = torch.zeros(
                batch_size,
                num_heads,
                max_length,
                head_dim,
                dtype=dtype,
                device=device,
            )
        self.hot_data = None
        self._hot_positions = None
        if self.hot_capacity > 0:
            self.hot_data = torch.zeros(
                batch_size,
                num_heads,
                self.hot_capacity,
                head_dim,
                dtype=dtype,
                device=device,
            )
            self._hot_positions = torch.full((self.hot_capacity,), -1, dtype=torch.long)

    @property
    def shape(self):
        return (
            self.batch_size,
            self.num_heads,
            int(self.current_length.item()),
            self.head_dim,
        )

    @staticmethod
    def _pack_indices(q_idx: torch.Tensor, bits: int, source_dim: int, packed_dim: int):
        if bits == 8:
            return q_idx.to(torch.uint8)
        q_idx_i64 = q_idx.to(torch.int64)
        bit_offsets = torch.arange(source_dim, device=q_idx.device, dtype=torch.int64) * int(bits)
        byte_offsets = torch.div(bit_offsets, 8, rounding_mode="floor")
        bit_shifts = torch.remainder(bit_offsets, 8)
        packed = torch.zeros(
            *q_idx.shape[:-1],
            packed_dim,
            dtype=torch.int64,
            device=q_idx.device,
        )
        view_shape = *([1] * (q_idx.dim() - 1)), source_dim
        byte_index = byte_offsets.view(view_shape).expand_as(q_idx_i64)
        shift = bit_shifts.view(view_shape)
        low = torch.bitwise_and(torch.bitwise_left_shift(q_idx_i64, shift), 0xFF)
        packed.scatter_add_(-1, byte_index, low)

        spill = bit_shifts + int(bits) - 8
        spill_mask = spill > 0
        if bool(spill_mask.any()):
            high_index = (byte_offsets + 1).clamp_max(packed_dim - 1)
            bits_in_low = (8 - bit_shifts).view(view_shape)
            high = torch.bitwise_right_shift(q_idx_i64, bits_in_low)
            high = torch.where(
                spill_mask.view(view_shape),
                high,
                torch.zeros((), dtype=torch.int64, device=q_idx.device),
            )
            packed.scatter_add_(-1, high_index.view(view_shape).expand_as(q_idx_i64), high)
        return torch.bitwise_and(packed, 0xFF).to(torch.uint8)

    @staticmethod
    def _unpack_indices(packed: torch.Tensor, bits: int, source_dim: int, packed_dim: int):
        if bits == 8:
            return packed[..., :source_dim].to(torch.long)
        packed_i64 = packed.to(torch.int64)
        bit_offsets = torch.arange(source_dim, device=packed.device, dtype=torch.int64) * int(bits)
        byte_offsets = torch.div(bit_offsets, 8, rounding_mode="floor")
        bit_shifts = torch.remainder(bit_offsets, 8)
        gather_shape = *([1] * (packed.dim() - 1)), source_dim
        expand_shape = (*packed.shape[:-1], source_dim)
        low = packed_i64.gather(-1, byte_offsets.view(gather_shape).expand(expand_shape))
        values = torch.bitwise_right_shift(low, bit_shifts.view(gather_shape))
        spill = bit_shifts + int(bits) - 8
        spill_mask = spill > 0
        if bool(spill_mask.any()):
            high_index = (byte_offsets + 1).clamp_max(packed_dim - 1)
            high = packed_i64.gather(-1, high_index.view(gather_shape).expand(expand_shape))
            high_shift = (8 - bit_shifts).view(gather_shape)
            values = torch.where(
                spill_mask.view(gather_shape),
                torch.bitwise_or(values, torch.bitwise_left_shift(high, high_shift)),
                values,
            )
        return torch.bitwise_and(values, (1 << int(bits)) - 1).to(torch.long)

    def _encode(self, tensor: torch.Tensor):
        current = tensor.to(torch.float32) @ self.rotation
        packed_angles = []
        for level_idx, (dim, bits, packed_dim) in enumerate(
            zip(self.level_dims, self.level_bits, self.level_packed_dims),
            start=1,
        ):
            left = current[..., 0::2]
            right = current[..., 1::2]
            radius = torch.sqrt((left * left) + (right * right) + 1e-12)
            theta = torch.atan2(right, left)
            if level_idx == 1:
                theta = torch.remainder(theta, 2.0 * math.pi)
            else:
                theta = theta.clamp(0.0, math.pi / 2.0)
            _, boundaries = self.angle_codebooks[level_idx - 1]
            q_idx = torch.bucketize(theta, boundaries).to(torch.uint8)
            packed_angles.append(self._pack_indices(q_idx, bits, dim, packed_dim))
            current = radius
        return current.to(torch.float16), packed_angles

    def _decode_range(self, start: int, end: int, out: torch.Tensor = None) -> torch.Tensor:
        if self._range_is_hot_exact(start, end):
            return self._gather_hot_range(start, end, out=out)

        current = self.final_radius[:, :, start:end].to(torch.float32)
        for level_idx in range(self.polar_levels - 1, -1, -1):
            bits = self.level_bits[level_idx]
            dim = self.level_dims[level_idx]
            packed_dim = self.level_packed_dims[level_idx]
            packed = self.angle_q[level_idx][:, :, start:end]
            q_idx = self._unpack_indices(packed, bits, dim, packed_dim)
            codebook, _ = self.angle_codebooks[level_idx]
            theta = codebook[q_idx]
            expanded = torch.empty(
                *current.shape[:-1],
                dim * 2,
                dtype=torch.float32,
                device=self.device,
            )
            expanded[..., 0::2] = current * theta.cos()
            expanded[..., 1::2] = current * theta.sin()
            current = expanded
        decoded = current @ self.rotation_t
        if out is None:
            out = torch.empty(
                self.batch_size,
                self.num_heads,
                end - start,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
        out.copy_(decoded.to(self.dtype), non_blocking=True)
        return self._overlay_hot_range(out, start, end)

    def _record_hot_slice(self, tensor: torch.Tensor, start: int):
        if self.hot_capacity <= 0 or self.hot_data is None:
            return
        input_len = int(tensor.shape[2])
        if input_len <= 0:
            return
        if int(start) == 0:
            self._hot_positions.fill_(-1)

        copy_len = min(input_len, self.hot_capacity)
        copy_start = input_len - copy_len
        positions = torch.arange(
            int(start) + copy_start,
            int(start) + input_len,
            device=self.device,
            dtype=torch.long,
        )
        slots = torch.remainder(positions, self.hot_capacity)
        self.hot_data.index_copy_(2, slots, tensor[:, :, copy_start:].to(dtype=self.dtype))
        self._hot_positions[slots.detach().cpu()] = positions.detach().cpu()

    def _range_is_hot_exact(self, start: int, end: int) -> bool:
        if self.hot_capacity <= 0 or self._hot_positions is None:
            return False
        length = int(end) - int(start)
        if length < 0 or length > self.hot_capacity:
            return False
        for pos in range(int(start), int(end)):
            slot = pos % self.hot_capacity
            if int(self._hot_positions[slot].item()) != pos:
                return False
        return True

    def _gather_hot_range(self, start: int, end: int, out: torch.Tensor = None):
        slots = torch.arange(
            int(start),
            int(end),
            device=self.device,
            dtype=torch.long,
        ).remainder_(self.hot_capacity)
        hot = self.hot_data.index_select(2, slots)
        if out is None:
            return hot
        out.copy_(hot, non_blocking=True)
        return out

    def _overlay_hot_range(self, out: torch.Tensor, start: int, end: int):
        if self.hot_capacity <= 0 or self._hot_positions is None:
            return out
        valid_slots = []
        rel_positions = []
        for slot, pos_tensor in enumerate(self._hot_positions.tolist()):
            pos = int(pos_tensor)
            if int(start) <= pos < int(end):
                valid_slots.append(slot)
                rel_positions.append(pos - int(start))
        if not valid_slots:
            return out

        order = sorted(range(len(rel_positions)), key=rel_positions.__getitem__)
        slots = torch.tensor([valid_slots[i] for i in order], device=self.device, dtype=torch.long)
        rel = torch.tensor([rel_positions[i] for i in order], device=self.device, dtype=torch.long)
        hot = self.hot_data.index_select(2, slots)
        out.index_copy_(2, rel, hot)
        return out

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        if dim != 2:
            raise ValueError("PolarQuantizedKVCache.copy currently supports dim=2 only.")
        tgt_radius = self.final_radius.index_select(dim, indices)
        self.final_radius.narrow(dim, prev_length, tgt_radius.shape[dim]).copy_(
            tgt_radius,
            non_blocking=True,
        )
        for level_idx, angle_store in enumerate(self.angle_q):
            tgt_angles = angle_store.index_select(dim, indices)
            angle_store.narrow(dim, prev_length, tgt_angles.shape[dim]).copy_(
                tgt_angles,
                non_blocking=True,
            )
        if self.hot_capacity > 0 and self.hot_data is not None:
            if prev_length == 0:
                self._hot_positions.fill_(-1)
            for offset, src_pos in enumerate([int(x) for x in indices.detach().cpu().tolist()]):
                dst_pos = int(prev_length) + int(offset)
                dst_slot = dst_pos % self.hot_capacity
                src_slot = src_pos % self.hot_capacity
                if int(self._hot_positions[src_slot].item()) == src_pos:
                    src = self.hot_data[:, :, src_slot : src_slot + 1]
                else:
                    src = self._decode_range(src_pos, src_pos + 1)
                self.hot_data[:, :, dst_slot : dst_slot + 1].copy_(src, non_blocking=True)
                self._hot_positions[dst_slot] = dst_pos
        if self.dequant_data is not None:
            shadow_dst = self.dequant_data.narrow(dim, prev_length, tgt_radius.shape[dim])
            shadow_src = self.dequant_data.index_select(dim, indices)
            shadow_dst.copy_(shadow_src, non_blocking=True)
        self.current_length.fill_(prev_length + tgt_radius.shape[dim])

    def append_compressed(self, tensor: torch.Tensor, dim: int = 2):
        if dim != 2:
            raise ValueError("PolarQuantizedKVCache.append_compressed currently supports dim=2 only.")
        start = int(self.current_length.item())
        end = start + tensor.shape[dim]
        final_radius, packed_angles = self._encode(tensor)
        self.final_radius.narrow(dim, start, tensor.shape[dim]).copy_(
            final_radius,
            non_blocking=True,
        )
        for level_idx, packed in enumerate(packed_angles):
            self.angle_q[level_idx].narrow(dim, start, tensor.shape[dim]).copy_(
                packed,
                non_blocking=True,
            )
        self._record_hot_slice(tensor, start)
        self.current_length.fill_(end)
        return start, end

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        if dim != 2:
            raise ValueError("PolarQuantizedKVCache.cat currently supports dim=2 only.")
        start, end = self.append_compressed(tensor, dim=dim)
        if self.dequant_data is not None:
            shadow_slice = self.dequant_data.narrow(dim, start, tensor.shape[dim])
            shadow_slice.copy_(tensor.to(self.dtype), non_blocking=True)
            return torch.narrow(self.dequant_data, 2, 0, int(self.current_length.item()))
        return self._decode_range(0, end)


def initialize_past_key_values(
    model,
    safe_max_length: int = 2048,
    turbo_quant: bool = False,
    turbo_kv_quant_mode: str = "polar",
    turbo_radius_bits: int = 8,
    turbo_theta_bits: int = 8,
    turbo_polar_levels: int = 4,
    turbo_vq_bits: int = 4,
    turbo_vq_key_bits=None,
    turbo_vq_outlier_bits: int = 4,
    turbo_vq_key_outlier_bits=None,
    turbo_vq_outlier_channels: int = 0,
    turbo_vq_outlier_indices=None,
    turbo_vq_residual_dim: int = 128,
    turbo_vq_residual_scale: float = 1.0,
    turbo_hybrid_hot_window: int = 512,
    turbo_runtime_dequant_cache: bool = True,
    turbo_compile_decode: bool = False,
    turbo_quant_seed: int = 0,
    packed_qjl_sketch_dim: int = 0,
    packed_qjl_layer: int = -1,
):
    config = model.config
    batch_size = 1
    layers = getattr(getattr(model, "model", None), "layers", None)
    num_layers = config.num_hidden_layers
    num_kv_heads = int(getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 1)))
    head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    if layers is not None and len(layers) > 0:
        attn = getattr(layers[0], "self_attn", None)
        num_kv_heads = int(getattr(attn, "num_key_value_heads", num_kv_heads))
        head_dim = int(getattr(attn, "head_dim", head_dim))

    def _cache_device_for_layer(layer_idx: int):
        if layers is not None and layer_idx < len(layers):
            try:
                device = next(layers[layer_idx].parameters()).device
                return torch.device("cpu") if device.type == "meta" else device
            except StopIteration:
                pass
        return model.device

    layer_devices = [_cache_device_for_layer(i) for i in range(num_layers)]
    packed_qjl_sketch_dim = int(max(0, packed_qjl_sketch_dim))
    packed_qjl_layer = int(packed_qjl_layer)
    if packed_qjl_layer < 0:
        packed_qjl_layer = num_layers + packed_qjl_layer
    packed_qjl_layer = min(max(0, packed_qjl_layer), num_layers - 1)

    current_length_data = torch.zeros(num_layers * 2, dtype=torch.long, device="cpu")
    past_key_values = []

    if turbo_quant:
        past_key_values_data = None
        quant_mode = str(turbo_kv_quant_mode).lower()
        if quant_mode not in {"polar", "turbo_vq"}:
            raise ValueError(
                "turbo_kv_quant_mode must be 'polar' or 'turbo_vq'."
            )
        effective_vq_key_bits = (
            int(turbo_vq_bits)
            if turbo_vq_key_bits is None
            else int(turbo_vq_key_bits)
        )
        effective_vq_key_outlier_bits = (
            min(8, max(1, effective_vq_key_bits + 1))
            if turbo_vq_key_outlier_bits is None
            else int(turbo_vq_key_outlier_bits)
        )
        effective_vq_key_outlier_bits = int(max(1, min(8, effective_vq_key_outlier_bits)))
        turbo_vq_outlier_bits = int(max(1, min(8, int(turbo_vq_outlier_bits))))
        turbo_vq_outlier_channels = int(max(0, int(turbo_vq_outlier_channels)))
        turbo_quant_seed = int(turbo_quant_seed)

        def _calibrated_outlier_idx(layer_idx: int, cache_idx: int):
            if turbo_vq_outlier_indices is None:
                return None
            try:
                return turbo_vq_outlier_indices[layer_idx][cache_idx]
            except (IndexError, KeyError, TypeError):
                return None

        for i in range(num_layers):
            layer_caches = []
            for j in range(2):
                idx = i * 2 + j
                if quant_mode == "turbo_vq":
                    if turbo_vq_outlier_channels > 0:
                        cache_cls = (
                            HybridOutlierTurboVQKVCache
                            if int(turbo_hybrid_hot_window) > 0
                            else OutlierAwareTurboVQKVCache
                        )
                        cache_kwargs = dict(
                            batch_size=batch_size,
                            num_heads=num_kv_heads,
                            max_length=safe_max_length,
                            head_dim=head_dim,
                            device=layer_devices[i],
                            dtype=model.dtype,
                            current_length=current_length_data[idx],
                            regular_bits=effective_vq_key_bits if j == 0 else turbo_vq_bits,
                            outlier_bits=effective_vq_key_outlier_bits if j == 0 else turbo_vq_outlier_bits,
                            n_outlier=turbo_vq_outlier_channels,
                            residual_dim=turbo_vq_residual_dim if j == 0 else 0,
                            residual_scale=turbo_vq_residual_scale if j == 0 else 1.0,
                            outlier_idx=_calibrated_outlier_idx(i, j),
                            quant_seed=turbo_quant_seed,
                        )
                        if cache_cls is HybridOutlierTurboVQKVCache:
                            cache_kwargs["hot_window"] = turbo_hybrid_hot_window
                        else:
                            cache_kwargs["runtime_dequant_cache"] = turbo_runtime_dequant_cache
                    else:
                        cache_cls = (
                            HybridTurboVQKVCache
                            if int(turbo_hybrid_hot_window) > 0
                            else TurboQuantizedKVCache
                        )
                        cache_kwargs = dict(
                            batch_size=batch_size,
                            num_heads=num_kv_heads,
                            max_length=safe_max_length,
                            head_dim=head_dim,
                            device=layer_devices[i],
                            dtype=model.dtype,
                            current_length=current_length_data[idx],
                            bits=effective_vq_key_bits if j == 0 else turbo_vq_bits,
                            residual_dim=turbo_vq_residual_dim if j == 0 else 0,
                            residual_scale=turbo_vq_residual_scale if j == 0 else 1.0,
                            quant_seed=turbo_quant_seed,
                        )
                        if cache_cls is HybridTurboVQKVCache:
                            cache_kwargs["hot_window"] = turbo_hybrid_hot_window
                        else:
                            cache_kwargs["runtime_dequant_cache"] = turbo_runtime_dequant_cache
                    layer_caches.append(cache_cls(**cache_kwargs))
                else:
                    layer_caches.append(
                        PolarQuantizedKVCache(
                            batch_size=batch_size,
                            num_heads=num_kv_heads,
                            max_length=safe_max_length,
                            head_dim=head_dim,
                            device=layer_devices[i],
                            dtype=model.dtype,
                            current_length=current_length_data[idx],
                            first_level_bits=turbo_theta_bits,
                            other_level_bits=turbo_radius_bits,
                            polar_levels=turbo_polar_levels,
                            runtime_dequant_cache=turbo_runtime_dequant_cache,
                            hot_window=turbo_hybrid_hot_window,
                            quant_seed=turbo_quant_seed,
                        )
                    )
            past_key_values.append(layer_caches)
        return past_key_values, past_key_values_data, current_length_data

    if len({str(device) for device in layer_devices}) > 1:
        past_key_values_data = None
        for i in range(num_layers):
            layer_data = torch.zeros(
                2,
                batch_size,
                num_kv_heads,
                safe_max_length,
                head_dim,
                device=layer_devices[i],
                dtype=model.dtype,
            )
            past_key_values.append(
                [
                    KVCache(
                        layer_data[j],
                        current_length_data[i * 2 + j],
                        qjl_sketch_dim=packed_qjl_sketch_dim
                        if (j == 0 and i == packed_qjl_layer)
                        else 0,
                        quant_seed=turbo_quant_seed,
                    )
                    for j in range(2)
                ]
            )
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
                KVCache(
                    past_key_values_data[i * 2 + j],
                    current_length_data[i * 2 + j],
                    qjl_sketch_dim=packed_qjl_sketch_dim
                    if (j == 0 and i == packed_qjl_layer)
                    else 0,
                    quant_seed=turbo_quant_seed,
                )
                for j in range(2)
            ]
        )
    return past_key_values, past_key_values_data, current_length_data


def initialize_outlier_calibration_past_key_values(model, safe_max_length: int = 2048):
    config = model.config
    batch_size = 1
    layers = getattr(getattr(model, "model", None), "layers", None)
    num_layers = config.num_hidden_layers
    num_kv_heads = int(getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 1)))
    head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    if layers is not None and len(layers) > 0:
        attn = getattr(layers[0], "self_attn", None)
        num_kv_heads = int(getattr(attn, "num_key_value_heads", num_kv_heads))
        head_dim = int(getattr(attn, "head_dim", head_dim))

    def _cache_device_for_layer(layer_idx: int):
        if layers is not None and layer_idx < len(layers):
            try:
                device = next(layers[layer_idx].parameters()).device
                return torch.device("cpu") if device.type == "meta" else device
            except StopIteration:
                pass
        return model.device

    layer_devices = [_cache_device_for_layer(i) for i in range(num_layers)]
    current_length_data = torch.zeros(num_layers * 2, dtype=torch.long, device="cpu")
    past_key_values = []
    for i in range(num_layers):
        layer_data = torch.zeros(
            2,
            batch_size,
            num_kv_heads,
            safe_max_length,
            head_dim,
            device=layer_devices[i],
            dtype=model.dtype,
        )
        past_key_values.append(
            [
                OutlierCalibrationKVCache(
                    layer_data[j],
                    current_length_data[i * 2 + j],
                )
                for j in range(2)
            ]
        )
    return past_key_values, current_length_data


def extract_outlier_calibration_indices(past_key_values, n_outlier: int):
    calibrated = []
    for key_cache, value_cache in past_key_values:
        calibrated.append(
            [
                key_cache.topk_outlier_indices(n_outlier).detach().cpu(),
                value_cache.topk_outlier_indices(n_outlier).detach().cpu(),
            ]
        )
    return calibrated
