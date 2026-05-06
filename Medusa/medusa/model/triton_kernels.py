import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional CUDA fast path
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def _is_cuda_tensor(tensor):
    return isinstance(tensor, torch.Tensor) and tensor.is_cuda


if TRITON_AVAILABLE:

    @triton.jit
    def _qjl_path_scores_kernel(
        q_proj,
        sign_cache,
        norm_cache,
        candidates,
        valid_mask,
        out,
        path_len: tl.constexpr,
        sketch_dim: tl.constexpr,
        coeff: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        path_id = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_M)
        sketch_mask = offsets < sketch_dim
        q_vals = tl.load(q_proj + offsets, mask=sketch_mask, other=0.0)

        acc = tl.full((), 0.0, tl.float32)
        count = tl.full((), 0.0, tl.float32)
        for step in tl.static_range(0, path_len):
            path_offset = path_id * path_len + step
            is_valid = tl.load(valid_mask + path_offset)
            token_id = tl.load(candidates + path_offset)
            signs = tl.load(
                sign_cache + token_id * sketch_dim + offsets,
                mask=sketch_mask,
                other=0,
            ).to(tl.float32)
            inner = tl.sum(signs * q_vals, axis=0)
            norm = tl.load(norm_cache + token_id).to(tl.float32)
            token_score = coeff * norm * inner
            acc += tl.where(is_valid, token_score, 0.0)
            count += tl.where(is_valid, 1.0, 0.0)

        tl.store(out + path_id, acc / tl.maximum(count, 1.0))


    @triton.jit
    def _materialize_pruned_medusa_kernel(
        full_tree_candidates,
        selected_nodes,
        token_indices,
        pruned_tree_candidates,
        pruned_candidates,
        n_nodes: tl.constexpr,
        n_tokens: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

        node_mask = offsets < n_nodes
        source_nodes = tl.load(selected_nodes + offsets, mask=node_mask, other=0)
        tree_vals = tl.load(full_tree_candidates + source_nodes, mask=node_mask, other=0)
        tl.store(pruned_tree_candidates + offsets, tree_vals, mask=node_mask)

        token_mask = offsets < n_tokens
        mapped_nodes = tl.load(token_indices + offsets, mask=token_mask, other=-1)
        valid_tokens = token_mask & (mapped_nodes >= 0)
        source_nodes_for_tokens = tl.load(
            selected_nodes + mapped_nodes,
            mask=valid_tokens,
            other=0,
        )
        candidate_vals = tl.load(
            full_tree_candidates + source_nodes_for_tokens,
            mask=valid_tokens,
            other=0,
        )
        tl.store(pruned_candidates + offsets, candidate_vals, mask=token_mask)


    @triton.jit
    def _polar_decode_range_kernel(
        radius_q,
        theta_q,
        radius_scale,
        theta_cos_lut,
        theta_sin_lut,
        out,
        start,
        length,
        max_length: tl.constexpr,
        num_heads: tl.constexpr,
        pair_dim: tl.constexpr,
        head_dim: tl.constexpr,
        inv_radius_levels,
        total_pairs,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_pairs

        pair = offsets % pair_dim
        token_flat = offsets // pair_dim
        rel_pos = token_flat % length
        head = (token_flat // length) % num_heads
        batch = token_flat // (length * num_heads)

        src_pos = start + rel_pos
        src_base = (batch * num_heads + head) * max_length + src_pos
        src_pair_offset = src_base * pair_dim + pair

        radius = tl.load(radius_q + src_pair_offset, mask=mask, other=0).to(tl.float32)
        scale = tl.load(radius_scale + src_base, mask=mask, other=0.0).to(tl.float32)
        theta_idx = tl.load(theta_q + src_pair_offset, mask=mask, other=0).to(tl.int32)
        cos = tl.load(theta_cos_lut + theta_idx, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(theta_sin_lut + theta_idx, mask=mask, other=0.0).to(tl.float32)

        decoded_radius = radius * scale * inv_radius_levels
        dst_base = ((batch * num_heads + head) * length + rel_pos) * head_dim + (pair * 2)
        tl.store(out + dst_base, decoded_radius * cos, mask=mask)
        tl.store(out + dst_base + 1, decoded_radius * sin, mask=mask)


def qjl_path_scores_triton(
    q_proj,
    sign_cache,
    norm_cache,
    candidates,
    valid_mask,
    coeff,
    sketch_dim,
):
    if not TRITON_AVAILABLE or not _is_cuda_tensor(q_proj):
        return None
    if not (_is_cuda_tensor(sign_cache) and _is_cuda_tensor(norm_cache)):
        return None
    if not (_is_cuda_tensor(candidates) and _is_cuda_tensor(valid_mask)):
        return None
    if candidates.dim() != 2 or valid_mask.shape != candidates.shape:
        return None

    n_paths, path_len = candidates.shape
    if n_paths == 0:
        return torch.empty((0,), device=candidates.device, dtype=torch.float32)

    block_m = triton.next_power_of_2(int(sketch_dim))
    if block_m > 1024:
        return None

    q_proj = q_proj.reshape(-1).contiguous()
    candidates = candidates.contiguous()
    valid_mask = valid_mask.contiguous()
    out = torch.empty((n_paths,), device=candidates.device, dtype=torch.float32)
    _qjl_path_scores_kernel[(n_paths,)](
        q_proj,
        sign_cache.contiguous(),
        norm_cache.contiguous(),
        candidates,
        valid_mask,
        out,
        path_len=int(path_len),
        sketch_dim=int(sketch_dim),
        coeff=float(coeff),
        BLOCK_M=block_m,
        num_warps=4 if block_m >= 128 else 1,
    )
    return out


def materialize_pruned_medusa_triton(full_tree_candidates, selected_nodes, token_indices):
    if not TRITON_AVAILABLE or not _is_cuda_tensor(full_tree_candidates):
        return None
    if not (_is_cuda_tensor(selected_nodes) and _is_cuda_tensor(token_indices)):
        return None
    if full_tree_candidates.dim() != 2 or full_tree_candidates.shape[0] != 1:
        return None

    n_nodes = int(selected_nodes.numel())
    n_tokens = int(token_indices.numel())
    if n_nodes == 0:
        return None

    full_tree_candidates = full_tree_candidates.contiguous()
    selected_nodes = selected_nodes.contiguous()
    token_indices = token_indices.contiguous()
    pruned_tree_candidates = torch.empty(
        (1, n_nodes),
        dtype=full_tree_candidates.dtype,
        device=full_tree_candidates.device,
    )
    pruned_candidates = torch.empty(
        token_indices.shape,
        dtype=full_tree_candidates.dtype,
        device=full_tree_candidates.device,
    )

    block = 128
    grid = (triton.cdiv(max(n_nodes, n_tokens), block),)
    _materialize_pruned_medusa_kernel[grid](
        full_tree_candidates,
        selected_nodes,
        token_indices,
        pruned_tree_candidates,
        pruned_candidates,
        n_nodes=n_nodes,
        n_tokens=n_tokens,
        BLOCK=block,
        num_warps=4,
    )
    return pruned_tree_candidates, pruned_candidates


def polar_decode_range_triton(
    radius_q,
    theta_q,
    radius_scale,
    theta_cos_lut,
    theta_sin_lut,
    out,
    start,
    end,
    inv_radius_levels,
):
    if not TRITON_AVAILABLE or not _is_cuda_tensor(radius_q):
        return False
    if not (
        _is_cuda_tensor(theta_q)
        and _is_cuda_tensor(radius_scale)
        and _is_cuda_tensor(theta_cos_lut)
        and _is_cuda_tensor(theta_sin_lut)
        and _is_cuda_tensor(out)
    ):
        return False
    if radius_q.dim() != 4 or theta_q.shape != radius_q.shape:
        return False

    batch_size, num_heads, max_length, pair_dim = radius_q.shape
    length = int(end) - int(start)
    if length <= 0:
        return True

    total_pairs = int(batch_size) * int(num_heads) * int(length) * int(pair_dim)
    block = 256
    grid = (triton.cdiv(total_pairs, block),)
    _polar_decode_range_kernel[grid](
        radius_q,
        theta_q,
        radius_scale,
        theta_cos_lut,
        theta_sin_lut,
        out,
        int(start),
        int(length),
        max_length=int(max_length),
        num_heads=int(num_heads),
        pair_dim=int(pair_dim),
        head_dim=int(pair_dim) * 2,
        inv_radius_levels=float(inv_radius_levels),
        total_pairs=total_pairs,
        BLOCK=block,
        num_warps=4,
    )
    return True
