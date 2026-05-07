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


_TURBO_ROTATION_CACHE = {}
_TURBO_CODEBOOK_CACHE = {}
_TURBO_RESIDUAL_PROJ_CACHE = {}


def _cache_key(device, *parts):
    device = torch.device(device)
    return (device.type, device.index, *parts)


def _get_turbo_rotation(head_dim: int, device: torch.device):
    key = _cache_key(device, "rotation", int(head_dim))
    rotation = _TURBO_ROTATION_CACHE.get(key)
    if rotation is not None:
        return rotation

    gen = torch.Generator(device="cpu")
    gen.manual_seed(20260427 + int(head_dim))
    mat = torch.randn(int(head_dim), int(head_dim), generator=gen, dtype=torch.float32)
    rotation, _ = torch.linalg.qr(mat, mode="reduced")
    # Fix QR sign ambiguity so the rotation is reproducible across LAPACK builds.
    signs = torch.sign(torch.diagonal(rotation))
    signs[signs == 0] = 1
    rotation = rotation * signs
    rotation = rotation.to(device=device)
    _TURBO_ROTATION_CACHE[key] = rotation
    return rotation


def _normal_pdf(x):
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _build_lloyd_max_codebook(bits: int, device: torch.device):
    bits = int(bits)
    key = _cache_key(device, "lloyd_max_normal", bits)
    cached = _TURBO_CODEBOOK_CACHE.get(key)
    if cached is not None:
        return cached

    levels = 1 << bits
    # A dense deterministic quadrature grid is enough for tiny scalar codebooks.
    grid = torch.linspace(-8.0, 8.0, 20001, dtype=torch.float64)
    pdf = _normal_pdf(grid)
    probs = torch.linspace(0.5 / levels, 1.0 - (0.5 / levels), levels, dtype=torch.float64)
    centroids = torch.erfinv((2.0 * probs) - 1.0) * math.sqrt(2.0)

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


def _get_turbo_residual_projection(head_dim: int, residual_dim: int, device: torch.device):
    residual_dim = int(max(0, residual_dim))
    key = _cache_key(device, "residual_qjl", int(head_dim), residual_dim)
    proj = _TURBO_RESIDUAL_PROJ_CACHE.get(key)
    if proj is not None:
        return proj

    gen = torch.Generator(device="cpu")
    gen.manual_seed(20260506 + int(head_dim) * 31 + residual_dim)
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
        self.residual_dim = int(max(0, residual_dim))
        self.residual_packed_dim = (self.residual_dim + 7) // 8 if self.residual_dim > 0 else 0
        self.residual_scale = float(residual_scale)
        self.residual_coeff = (
            self.residual_scale * math.sqrt(math.pi / 2.0) / float(self.residual_dim)
            if self.residual_dim > 0
            else 0.0
        )
        self.runtime_dequant_cache = bool(runtime_dequant_cache)

        self.rotation = _get_turbo_rotation(head_dim, device)
        self.rotation_t = self.rotation.t().contiguous()
        self.codebook, self.boundaries = _build_lloyd_max_codebook(self.bits, device)
        self.residual_proj = None
        if self.residual_dim > 0:
            self.residual_proj = _get_turbo_residual_projection(
                head_dim,
                self.residual_dim,
                device,
            )

        self.q_idx = torch.zeros(
            batch_size,
            num_heads,
            max_length,
            head_dim,
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
        start = int(self.current_length.item())
        end = start + tensor.shape[dim]
        if turbo_vq_append_triton is not None and turbo_vq_append_triton(self, tensor, start):
            self.current_length.fill_(end)
            return start, end

        q_idx, scale, residual_sign, residual_norm = self._encode(tensor)

        self.q_idx.narrow(dim, start, tensor.shape[dim]).copy_(q_idx, non_blocking=True)
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
        q_idx = self.q_idx[:, :, start:end].to(torch.long)
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


def turbo_vq_attention_with_qjl_residual(
    query_states: torch.Tensor,
    key_cache: TurboQuantizedKVCache,
    value_cache: TurboQuantizedKVCache,
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
    if (
        not isinstance(key_cache, TurboQuantizedKVCache)
        or not isinstance(value_cache, TurboQuantizedKVCache)
        or key_cache.residual_sign_packed is None
        or key_cache.residual_proj is None
    ):
        return None
    if query_states.dim() != 4:
        return None

    kv_len = int(key_cache.current_length.item())
    if kv_len <= 0:
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

    def append_compressed(self, tensor: torch.Tensor, dim: int = 2):
        if dim != 2:
            raise ValueError("PolarQuantizedKVCache.append_compressed currently supports dim=2 only.")
        start = int(self.current_length.item())
        end = start + tensor.shape[dim]
        radius_q, theta_q, scale = self._encode(tensor)

        self.radius_q.narrow(dim, start, tensor.shape[dim]).copy_(radius_q, non_blocking=True)
        self.theta_q.narrow(dim, start, tensor.shape[dim]).copy_(theta_q, non_blocking=True)
        self.radius_scale.narrow(dim, start, tensor.shape[dim]).copy_(scale, non_blocking=True)
        self.current_length.fill_(end)
        return start, end

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        if dim != 2:
            raise ValueError("PolarQuantizedKVCache.cat currently supports dim=2 only.")
        start, end = self.append_compressed(tensor, dim=dim)
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
    turbo_kv_quant_mode: str = "polar",
    turbo_radius_bits: int = 8,
    turbo_theta_bits: int = 8,
    turbo_vq_bits: int = 4,
    turbo_vq_key_bits=None,
    turbo_vq_residual_dim: int = 128,
    turbo_vq_residual_scale: float = 1.0,
    turbo_hybrid_hot_window: int = 512,
    turbo_runtime_dequant_cache: bool = True,
    turbo_compile_decode: bool = False,
):
    config = model.config
    batch_size = 1
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    def _cache_device_for_layer(layer_idx: int):
        layers = getattr(getattr(model, "model", None), "layers", None)
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

    if turbo_quant:
        past_key_values_data = None
        quant_mode = str(turbo_kv_quant_mode).lower()
        if quant_mode not in {"polar", "turbo_vq", "hybrid_turbo_vq"}:
            raise ValueError(
                "turbo_kv_quant_mode must be 'polar', 'turbo_vq', or 'hybrid_turbo_vq'."
            )
        effective_vq_key_bits = (
            int(turbo_vq_bits)
            if turbo_vq_key_bits is None
            else int(turbo_vq_key_bits)
        )
        for i in range(num_layers):
            layer_caches = []
            for j in range(2):
                idx = i * 2 + j
                if quant_mode == "hybrid_turbo_vq":
                    layer_caches.append(
                        HybridTurboVQKVCache(
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
                            hot_window=turbo_hybrid_hot_window,
                        )
                    )
                elif quant_mode == "turbo_vq":
                    layer_caches.append(
                        TurboQuantizedKVCache(
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
                            runtime_dequant_cache=turbo_runtime_dequant_cache,
                        )
                    )
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
                            radius_bits=turbo_radius_bits,
                            theta_bits=turbo_theta_bits,
                            runtime_dequant_cache=turbo_runtime_dequant_cache,
                            compile_decode=turbo_compile_decode,
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
                    KVCache(layer_data[j], current_length_data[i * 2 + j])
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
                KVCache(past_key_values_data[i * 2 + j], current_length_data[i * 2 + j])
                for j in range(2)
            ]
        )
    return past_key_values, past_key_values_data, current_length_data
