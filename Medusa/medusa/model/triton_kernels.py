import os

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


def _debug_triton_failure(name, exc):
    if os.environ.get("MEDUSA_TRITON_DEBUG"):
        print(f"{name} failed: {type(exc).__name__}: {exc}")


if TRITON_AVAILABLE:
    @triton.jit
    def _popcount_u32(x):
        x = x.to(tl.uint32)
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        return ((x * 0x01010101) >> 24).to(tl.int32)


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
    def _packed_kv_qjl_node_scores_kernel(
        query_bits,
        key_bits,
        partial,
        kv_len: tl.constexpr,
        max_length: tl.constexpr,
        num_heads: tl.constexpr,
        words: tl.constexpr,
        num_blocks: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        node = tl.program_id(0)
        head = tl.program_id(1)
        block = tl.program_id(2)
        k_offsets = (block * BLOCK_K) + tl.arange(0, BLOCK_K)
        w_offsets = tl.arange(0, BLOCK_W)

        q = tl.load(
            query_bits + (node * num_heads + head) * words + w_offsets,
            mask=w_offsets < words,
            other=0,
        )
        valid = (k_offsets[:, None] < kv_len) & (w_offsets[None, :] < words)
        k = tl.load(
            key_bits + (head * max_length + k_offsets[:, None]) * words + w_offsets[None, :],
            mask=valid,
            other=0,
        )
        matches = _popcount_u32(~(q[None, :] ^ k))
        matches = tl.where(valid, matches, 0)
        score_by_word = tl.sum(matches, axis=0)
        score = tl.sum(score_by_word, axis=0)
        tl.store(partial + (node * num_heads + head) * num_blocks + block, score)


    @triton.jit
    def _turbo_qjl_select_kernel(
        q_proj,
        sign_cache,
        norm_cache,
        candidates,
        valid_mask,
        medusa_scores,
        mandatory_indices,
        approx_out,
        selected_out,
        selected_count_out,
        verify_full_out,
        path_len: tl.constexpr,
        n_paths: tl.constexpr,
        sketch_dim: tl.constexpr,
        coeff: tl.constexpr,
        base_keep: tl.constexpr,
        max_keep: tl.constexpr,
        mandatory_count: tl.constexpr,
        margin_scale: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        paths = tl.arange(0, BLOCK_P)
        path_mask = paths < n_paths
        sketch_offsets = tl.arange(0, BLOCK_M)
        sketch_mask = sketch_offsets < sketch_dim
        q_vals = tl.load(q_proj + sketch_offsets, mask=sketch_mask, other=0.0)

        qjl_acc = tl.zeros((BLOCK_P,), dtype=tl.float32)
        qjl_count = tl.zeros((BLOCK_P,), dtype=tl.float32)
        for step in tl.static_range(0, path_len):
            offset = paths * path_len + step
            is_valid = tl.load(valid_mask + offset, mask=path_mask, other=0)
            token_id = tl.load(candidates + offset, mask=path_mask, other=0)
            signs = tl.load(
                sign_cache + token_id[:, None] * sketch_dim + sketch_offsets[None, :],
                mask=path_mask[:, None] & sketch_mask[None, :],
                other=0,
            ).to(tl.float32)
            inner = tl.sum(signs * q_vals[None, :], axis=1)
            norm = tl.load(norm_cache + token_id, mask=path_mask, other=0.0).to(tl.float32)
            token_score = coeff * norm * inner
            qjl_acc += tl.where(is_valid & path_mask, token_score, 0.0)
            qjl_count += tl.where(is_valid & path_mask, 1.0, 0.0)

        qjl_scores = qjl_acc / tl.maximum(qjl_count, 1.0)
        medusa = tl.load(medusa_scores + paths, mask=path_mask, other=0.0).to(tl.float32)

        denom = tl.full((), n_paths, tl.float32)
        medusa_mean = tl.sum(tl.where(path_mask, medusa, 0.0), axis=0) / denom
        qjl_mean = tl.sum(tl.where(path_mask, qjl_scores, 0.0), axis=0) / denom
        medusa_centered = tl.where(path_mask, medusa - medusa_mean, 0.0)
        qjl_centered = tl.where(path_mask, qjl_scores - qjl_mean, 0.0)
        medusa_std = tl.sqrt(tl.sum(medusa_centered * medusa_centered, axis=0) / denom)
        qjl_std = tl.sqrt(tl.sum(qjl_centered * qjl_centered, axis=0) / denom)

        medusa_norm = tl.where(medusa_std <= 1.0e-6, 0.0, medusa_centered / (medusa_std + 1.0e-6))
        qjl_norm = tl.where(qjl_std <= 1.0e-6, 0.0, qjl_centered / (qjl_std + 1.0e-6))
        approx = (0.75 * medusa_norm) + (0.25 * qjl_norm)
        approx = tl.where(path_mask, approx, -float("inf"))
        tl.store(approx_out + paths, approx, mask=path_mask)

        approx_mean = tl.sum(tl.where(path_mask, approx, 0.0), axis=0) / denom
        approx_centered = tl.where(path_mask, approx - approx_mean, 0.0)
        approx_std = tl.sqrt(tl.sum(approx_centered * approx_centered, axis=0) / denom)
        top1 = tl.max(approx, axis=0)
        top1_idx = tl.min(tl.where((approx == top1) & path_mask, paths, BLOCK_P), axis=0)
        top2_scores = tl.where((paths == top1_idx) | ~path_mask, -float("inf"), approx)
        top2 = tl.max(top2_scores, axis=0)
        margin = top1 - top2
        uncertain = (approx_std <= 1.0e-6) | (margin < margin_scale * approx_std)
        adaptive_uncertain = (approx_std <= 1.0e-6) | (margin < 0.25 * approx_std)
        keep = tl.minimum(base_keep, n_paths)
        keep = tl.where(adaptive_uncertain, tl.minimum(max_keep, n_paths), keep)

        mandatory0 = tl.load(mandatory_indices + 0, mask=mandatory_count > 0, other=-1)
        mandatory1 = tl.load(mandatory_indices + 1, mask=mandatory_count > 1, other=-1)
        selected_mask = (paths == mandatory0) | (paths == mandatory1)

        for out_idx in tl.static_range(0, max_keep):
            mandatory_pick = tl.load(
                mandatory_indices + out_idx,
                mask=out_idx < mandatory_count,
                other=-1,
            )
            selectable_scores = tl.where(path_mask & ~selected_mask, approx, -float("inf"))
            best_val = tl.max(selectable_scores, axis=0)
            best_idx = tl.min(
                tl.where((selectable_scores == best_val) & path_mask, paths, BLOCK_P),
                axis=0,
            )
            selected_idx = tl.where(out_idx < mandatory_count, mandatory_pick, best_idx)
            selected_mask = selected_mask | (paths == selected_idx)
            tl.store(selected_out + out_idx, selected_idx, mask=out_idx < keep)

        tl.store(selected_count_out, keep)
        tl.store(verify_full_out, tl.where(uncertain, 1, 0))


    @triton.jit
    def _node_budget_select_kernel(
        scores,
        retrieve_indices,
        mandatory_indices,
        selected_out,
        selected_count_out,
        n_paths: tl.constexpr,
        path_len: tl.constexpr,
        full_node_count: tl.constexpr,
        node_budget: tl.constexpr,
        min_keep: tl.constexpr,
        max_keep: tl.constexpr,
        mandatory_count,
        BLOCK_P: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        paths = tl.arange(0, BLOCK_P)
        nodes = tl.arange(0, BLOCK_N)
        path_mask = paths < n_paths
        node_mask = nodes < full_node_count

        selected_path_mask = paths < 0
        selected_node_mask = nodes < 0
        selected_count = tl.full((), 0, tl.int32)

        for out_idx in tl.static_range(0, max_keep):
            mandatory_pick = tl.load(
                mandatory_indices + out_idx,
                mask=out_idx < mandatory_count,
                other=-1,
            )
            use_mandatory = (out_idx < mandatory_count) & (mandatory_pick >= 0) & (mandatory_pick < n_paths)

            selected_node_count = tl.sum(selected_node_mask.to(tl.int32), axis=0)
            path_node_mask = (paths[:, None] < 0) & (nodes[None, :] < 0)
            for step in tl.static_range(0, path_len):
                node_id = tl.load(
                    retrieve_indices + paths * path_len + step,
                    mask=path_mask,
                    other=-1,
                )
                path_node_mask = path_node_mask | (
                    path_mask[:, None]
                    & node_mask[None, :]
                    & (node_id[:, None] == nodes[None, :])
                    & (node_id[:, None] >= 0)
                )

            new_node_mask = path_node_mask & (~selected_node_mask[None, :])
            additional_nodes = tl.sum(new_node_mask.to(tl.int32), axis=1)
            within_budget = (selected_node_count + additional_nodes) <= node_budget
            allow_over_budget = selected_count < min_keep
            selectable = path_mask & (~selected_path_mask) & (within_budget | allow_over_budget)
            score_vals = tl.load(scores + paths, mask=path_mask, other=-float("inf"))
            selectable_scores = tl.where(selectable, score_vals, -float("inf"))
            best_val = tl.max(selectable_scores, axis=0)
            best_idx = tl.min(
                tl.where((selectable_scores == best_val) & path_mask, paths, BLOCK_P),
                axis=0,
            )
            selected_idx = tl.where(use_mandatory, mandatory_pick, best_idx)
            have_selection = use_mandatory | (best_val > -float("inf"))

            selected_node_update = nodes < 0
            for step in tl.static_range(0, path_len):
                node_id = tl.load(
                    retrieve_indices + selected_idx * path_len + step,
                    mask=have_selection,
                    other=-1,
                )
                selected_node_update = selected_node_update | (
                    node_mask & (nodes == node_id) & (node_id >= 0)
                )
            selected_node_mask = selected_node_mask | selected_node_update
            selected_path_mask = selected_path_mask | (paths == selected_idx)

            tl.store(selected_out + out_idx, selected_idx, mask=have_selection)
            selected_count += tl.where(have_selection, 1, 0)

        tl.store(selected_count_out, selected_count)


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
    def _greedy_tree_posterior_kernel(
        tree_logits,
        candidates,
        retrieve_indices,
        path_lengths,
        best_candidate_out,
        accept_length_out,
        n_paths: tl.constexpr,
        path_len: tl.constexpr,
        vocab_size: tl.constexpr,
        has_path_lengths: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        paths = tl.arange(0, BLOCK_P)
        path_mask = paths < n_paths
        vocab_offsets = tl.arange(0, BLOCK_V)

        alive = path_mask
        accept_counts = tl.zeros((BLOCK_P,), dtype=tl.int32)

        for step in tl.static_range(0, path_len - 1):
            if has_path_lengths:
                lengths = tl.load(path_lengths + paths, mask=path_mask, other=0).to(tl.int32)
                valid_step = step < lengths
            else:
                valid_step = path_mask

            active = alive & valid_step & path_mask
            node_ids = tl.load(
                retrieve_indices + paths * path_len + step,
                mask=path_mask,
                other=0,
            ).to(tl.int32)
            node_ids = tl.maximum(node_ids, 0)

            best_vals = tl.full((BLOCK_P,), -float("inf"), dtype=tl.float32)
            best_ids = tl.zeros((BLOCK_P,), dtype=tl.int32)

            for vocab_start in tl.range(0, vocab_size, BLOCK_V):
                token_ids = vocab_start + vocab_offsets
                vocab_mask = token_ids < vocab_size
                vals = tl.load(
                    tree_logits + node_ids[:, None] * vocab_size + token_ids[None, :],
                    mask=active[:, None] & vocab_mask[None, :],
                    other=-float("inf"),
                ).to(tl.float32)
                chunk_best = tl.max(vals, axis=1)
                chunk_ids = tl.min(
                    tl.where(
                        (vals == chunk_best[:, None]) & vocab_mask[None, :],
                        token_ids[None, :],
                        vocab_size,
                    ),
                    axis=1,
                ).to(tl.int32)
                better = chunk_best > best_vals
                best_vals = tl.where(better, chunk_best, best_vals)
                best_ids = tl.where(better, chunk_ids, best_ids)

            candidate_tokens = tl.load(
                candidates + paths * path_len + step + 1,
                mask=path_mask,
                other=-1,
            ).to(tl.int32)
            matched = active & (candidate_tokens == best_ids)
            accept_counts += tl.where(matched, 1, 0)
            alive = alive & matched

        best_len = tl.max(tl.where(path_mask, accept_counts, -1), axis=0)
        best_idx = tl.min(
            tl.where((accept_counts == best_len) & path_mask, paths, BLOCK_P),
            axis=0,
        )
        best_idx = tl.where(best_len <= 0, 0, best_idx)
        best_len = tl.maximum(best_len, 0)

        tl.store(best_candidate_out, best_idx)
        tl.store(accept_length_out, best_len)


    @triton.jit
    def _greedy_tree_posterior_unique_kernel(
        tree_logits,
        candidates,
        retrieve_indices,
        path_lengths,
        best_candidate_out,
        accept_length_out,
        n_nodes: tl.constexpr,
        n_paths: tl.constexpr,
        path_len: tl.constexpr,
        vocab_size: tl.constexpr,
        has_path_lengths: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        nodes = tl.arange(0, BLOCK_N)
        paths = tl.arange(0, BLOCK_P)
        vocab_offsets = tl.arange(0, BLOCK_V)
        node_mask = nodes < n_nodes
        path_mask = paths < n_paths

        node_best_vals = tl.full((BLOCK_N,), -float("inf"), dtype=tl.float32)
        node_best_ids = tl.zeros((BLOCK_N,), dtype=tl.int32)

        for vocab_start in tl.range(0, vocab_size, BLOCK_V):
            token_ids = vocab_start + vocab_offsets
            vocab_mask = token_ids < vocab_size
            vals = tl.load(
                tree_logits + nodes[:, None] * vocab_size + token_ids[None, :],
                mask=node_mask[:, None] & vocab_mask[None, :],
                other=-float("inf"),
            ).to(tl.float32)
            chunk_best = tl.max(vals, axis=1)
            chunk_ids = tl.min(
                tl.where(
                    (vals == chunk_best[:, None]) & vocab_mask[None, :],
                    token_ids[None, :],
                    vocab_size,
                ),
                axis=1,
            ).to(tl.int32)
            better = chunk_best > node_best_vals
            node_best_vals = tl.where(better, chunk_best, node_best_vals)
            node_best_ids = tl.where(better, chunk_ids, node_best_ids)

        alive = path_mask
        accept_counts = tl.zeros((BLOCK_P,), dtype=tl.int32)

        for step in tl.static_range(0, path_len - 1):
            if has_path_lengths:
                lengths = tl.load(path_lengths + paths, mask=path_mask, other=0).to(tl.int32)
                valid_step = step < lengths
            else:
                valid_step = path_mask

            active = alive & valid_step & path_mask
            node_ids = tl.load(
                retrieve_indices + paths * path_len + step,
                mask=path_mask,
                other=0,
            ).to(tl.int32)
            node_ids = tl.maximum(node_ids, 0)

            node_matches = nodes[:, None] == node_ids[None, :]
            expected_tokens = tl.max(
                tl.where(node_matches & node_mask[:, None] & path_mask[None, :], node_best_ids[:, None], 0),
                axis=0,
            )
            candidate_tokens = tl.load(
                candidates + paths * path_len + step + 1,
                mask=path_mask,
                other=-1,
            ).to(tl.int32)
            matched = active & (candidate_tokens == expected_tokens)
            accept_counts += tl.where(matched, 1, 0)
            alive = alive & matched

        best_len = tl.max(tl.where(path_mask, accept_counts, -1), axis=0)
        best_idx = tl.min(
            tl.where((accept_counts == best_len) & path_mask, paths, BLOCK_P),
            axis=0,
        )
        best_idx = tl.where(best_len <= 0, 0, best_idx)
        best_len = tl.maximum(best_len, 0)

        tl.store(best_candidate_out, best_idx)
        tl.store(accept_length_out, best_len)


    @triton.jit
    def _greedy_tree_accept_from_argmax_kernel(
        node_argmax,
        candidates,
        retrieve_indices,
        path_lengths,
        best_candidate_out,
        accept_length_out,
        n_paths: tl.constexpr,
        path_len: tl.constexpr,
        has_path_lengths: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        paths = tl.arange(0, BLOCK_P)
        path_mask = paths < n_paths
        alive = path_mask
        accept_counts = tl.zeros((BLOCK_P,), dtype=tl.int32)

        for step in tl.static_range(0, path_len - 1):
            if has_path_lengths:
                lengths = tl.load(path_lengths + paths, mask=path_mask, other=0).to(tl.int32)
                valid_step = step < lengths
            else:
                valid_step = path_mask

            node_ids = tl.load(
                retrieve_indices + paths * path_len + step,
                mask=path_mask,
                other=0,
            ).to(tl.int32)
            node_ids = tl.maximum(node_ids, 0)
            expected_tokens = tl.load(node_argmax + node_ids, mask=path_mask, other=-1).to(tl.int32)
            candidate_tokens = tl.load(
                candidates + paths * path_len + step + 1,
                mask=path_mask,
                other=-1,
            ).to(tl.int32)
            matched = alive & valid_step & path_mask & (candidate_tokens == expected_tokens)
            accept_counts += tl.where(matched, 1, 0)
            alive = alive & matched

        best_len = tl.max(tl.where(path_mask, accept_counts, -1), axis=0)
        best_idx = tl.min(
            tl.where((accept_counts == best_len) & path_mask, paths, BLOCK_P),
            axis=0,
        )
        best_idx = tl.where(best_len <= 0, 0, best_idx)
        best_len = tl.maximum(best_len, 0)

        tl.store(best_candidate_out, best_idx)
        tl.store(accept_length_out, best_len)


    @triton.jit
    def _lm_head_argmax_partial_kernel(
        hidden_states,
        lm_head_weight,
        partial_vals,
        partial_ids,
        num_vocab_blocks: tl.constexpr,
        vocab_size: tl.constexpr,
        hidden_size: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        node_id = tl.program_id(0)
        vocab_block = tl.program_id(1)
        vocab_offsets = vocab_block * BLOCK_V + tl.arange(0, BLOCK_V)
        hidden_offsets = tl.arange(0, BLOCK_H)
        vocab_mask = vocab_offsets < vocab_size
        acc = tl.zeros((BLOCK_V,), dtype=tl.float32)

        for hidden_start in tl.range(0, hidden_size, BLOCK_H):
            hidden_ids = hidden_start + hidden_offsets
            hidden_mask = hidden_ids < hidden_size
            h_vals = tl.load(
                hidden_states + node_id * hidden_size + hidden_ids,
                mask=hidden_mask,
                other=0.0,
            ).to(tl.float32)
            w_vals = tl.load(
                lm_head_weight + vocab_offsets[:, None] * hidden_size + hidden_ids[None, :],
                mask=vocab_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w_vals * h_vals[None, :], axis=1)

        vals = tl.where(vocab_mask, acc, -float("inf"))
        best_val = tl.max(vals, axis=0)
        best_id = tl.min(
            tl.where((vals == best_val) & vocab_mask, vocab_offsets, vocab_size),
            axis=0,
        ).to(tl.int32)
        out_offset = node_id * num_vocab_blocks + vocab_block
        tl.store(partial_vals + out_offset, best_val)
        tl.store(partial_ids + out_offset, best_id)


    @triton.jit
    def _lm_head_argmax_reduce_kernel(
        partial_vals,
        partial_ids,
        node_argmax,
        num_vocab_blocks: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        node_id = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_B)
        mask = offsets < num_vocab_blocks
        vals = tl.load(
            partial_vals + node_id * num_vocab_blocks + offsets,
            mask=mask,
            other=-float("inf"),
        )
        ids = tl.load(
            partial_ids + node_id * num_vocab_blocks + offsets,
            mask=mask,
            other=2147483647,
        ).to(tl.int32)
        best_val = tl.max(vals, axis=0)
        best_id = tl.min(
            tl.where((vals == best_val) & mask, ids, 2147483647),
            axis=0,
        ).to(tl.int32)
        tl.store(node_argmax + node_id, best_id)


    @triton.jit
    def _lm_head_argmax_matmul_partial_kernel(
        hidden_states,
        lm_head_weight,
        partial_vals,
        partial_ids,
        n_nodes: tl.constexpr,
        num_vocab_blocks: tl.constexpr,
        vocab_size: tl.constexpr,
        hidden_size: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        node_block = tl.program_id(0)
        vocab_block = tl.program_id(1)
        nodes = node_block * BLOCK_M + tl.arange(0, BLOCK_M)
        vocab_offsets = vocab_block * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)
        node_mask = nodes < n_nodes
        vocab_mask = vocab_offsets < vocab_size
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in tl.range(0, hidden_size, BLOCK_K):
            k_ids = k_start + k_offsets
            k_mask = k_ids < hidden_size
            h_vals = tl.load(
                hidden_states + nodes[:, None] * hidden_size + k_ids[None, :],
                mask=node_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w_vals = tl.load(
                lm_head_weight + vocab_offsets[None, :] * hidden_size + k_ids[:, None],
                mask=k_mask[:, None] & vocab_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(h_vals, w_vals)

        vals = tl.where(node_mask[:, None] & vocab_mask[None, :], acc, -float("inf"))
        best_vals = tl.max(vals, axis=1)
        best_ids = tl.min(
            tl.where(
                (vals == best_vals[:, None]) & vocab_mask[None, :],
                vocab_offsets[None, :],
                vocab_size,
            ),
            axis=1,
        ).to(tl.int32)
        tl.store(
            partial_vals + nodes * num_vocab_blocks + vocab_block,
            best_vals,
            mask=node_mask,
        )
        tl.store(
            partial_ids + nodes * num_vocab_blocks + vocab_block,
            best_ids,
            mask=node_mask,
        )


    @triton.jit
    def _copy_selected_kv_cache_kernel(
        kv_data,
        select_indices,
        prev_input_len,
        total_elements: tl.constexpr,
        batch_size: tl.constexpr,
        num_heads: tl.constexpr,
        max_length: tl.constexpr,
        head_dim: tl.constexpr,
        copy_start: tl.constexpr,
        copy_count: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_elements

        dim = offsets % head_dim
        tmp = offsets // head_dim
        rel_copy = tmp % copy_count
        tmp = tmp // copy_count
        head = tmp % num_heads
        tmp = tmp // num_heads
        batch = tmp % batch_size
        layer_kv = tmp // batch_size

        rel = rel_copy + copy_start
        src_pos = tl.load(select_indices + rel, mask=mask, other=0).to(tl.int64)
        dst_pos = prev_input_len + rel

        src = (
            (((layer_kv * batch_size + batch) * num_heads + head) * max_length + src_pos)
            * head_dim
            + dim
        )
        dst = (
            (((layer_kv * batch_size + batch) * num_heads + head) * max_length + dst_pos)
            * head_dim
            + dim
        )
        vals = tl.load(kv_data + src, mask=mask, other=0.0)
        tl.store(kv_data + dst, vals, mask=mask)


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


    @triton.jit
    def _polar_compressed_attention_kernel(
        query,
        key_radius_q,
        key_theta_q,
        key_radius_scale,
        value_radius_q,
        value_theta_q,
        value_radius_scale,
        theta_cos_lut,
        theta_sin_lut,
        attention_mask,
        out,
        kv_len,
        max_length: tl.constexpr,
        q_len: tl.constexpr,
        num_key_value_groups: tl.constexpr,
        pair_dim: tl.constexpr,
        head_dim: tl.constexpr,
        inv_radius_levels: tl.constexpr,
        sm_scale: tl.constexpr,
        has_mask: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        q_pos = tl.program_id(0)
        q_head = tl.program_id(1)
        kv_head = q_head // num_key_value_groups

        pair_offsets = tl.arange(0, BLOCK_P)
        pair_mask = pair_offsets < pair_dim
        even_offsets = pair_offsets * 2

        q_base = (q_head * q_len + q_pos) * head_dim
        q_even = tl.load(query + q_base + even_offsets, mask=pair_mask, other=0.0).to(tl.float32)
        q_odd = tl.load(query + q_base + even_offsets + 1, mask=pair_mask, other=0.0).to(tl.float32)

        acc_even = tl.zeros((BLOCK_P,), dtype=tl.float32)
        acc_odd = tl.zeros((BLOCK_P,), dtype=tl.float32)
        m_i = tl.full((), -float("inf"), dtype=tl.float32)
        l_i = tl.full((), 0.0, dtype=tl.float32)

        kv_offsets_base = tl.arange(0, BLOCK_N)
        for block_start in tl.range(0, kv_len, BLOCK_N):
            kv_offsets = block_start + kv_offsets_base
            kv_mask = kv_offsets < kv_len
            src_base = kv_head * max_length + kv_offsets
            pair_src = src_base[:, None] * pair_dim + pair_offsets[None, :]
            load_mask = kv_mask[:, None] & pair_mask[None, :]

            key_radius = tl.load(key_radius_q + pair_src, mask=load_mask, other=0).to(tl.float32)
            key_theta_idx = tl.load(key_theta_q + pair_src, mask=load_mask, other=0).to(tl.int32)
            key_scale = tl.load(key_radius_scale + src_base, mask=kv_mask, other=0.0).to(tl.float32)
            key_cos = tl.load(theta_cos_lut + key_theta_idx, mask=load_mask, other=0.0).to(tl.float32)
            key_sin = tl.load(theta_sin_lut + key_theta_idx, mask=load_mask, other=0.0).to(tl.float32)
            key_radius_dec = key_radius * key_scale[:, None] * inv_radius_levels
            key_even = key_radius_dec * key_cos
            key_odd = key_radius_dec * key_sin

            scores = tl.sum((key_even * q_even[None, :]) + (key_odd * q_odd[None, :]), axis=1)
            scores = scores * sm_scale
            scores = tl.where(kv_mask, scores, -float("inf"))
            if has_mask:
                mask_vals = tl.load(
                    attention_mask + q_pos * kv_len + kv_offsets,
                    mask=kv_mask,
                    other=-float("inf"),
                ).to(tl.float32)
                scores += mask_vals

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)

            value_radius = tl.load(value_radius_q + pair_src, mask=load_mask, other=0).to(tl.float32)
            value_theta_idx = tl.load(value_theta_q + pair_src, mask=load_mask, other=0).to(tl.int32)
            value_scale = tl.load(value_radius_scale + src_base, mask=kv_mask, other=0.0).to(tl.float32)
            value_cos = tl.load(theta_cos_lut + value_theta_idx, mask=load_mask, other=0.0).to(tl.float32)
            value_sin = tl.load(theta_sin_lut + value_theta_idx, mask=load_mask, other=0.0).to(tl.float32)
            value_radius_dec = value_radius * value_scale[:, None] * inv_radius_levels
            value_even = value_radius_dec * value_cos
            value_odd = value_radius_dec * value_sin

            acc_even = (acc_even * alpha) + tl.sum(p[:, None] * value_even, axis=0)
            acc_odd = (acc_odd * alpha) + tl.sum(p[:, None] * value_odd, axis=0)
            l_i = (l_i * alpha) + tl.sum(p, axis=0)
            m_i = m_new

        inv_l = 1.0 / tl.maximum(l_i, 1.0e-20)
        out_base = (q_head * q_len + q_pos) * head_dim
        tl.store(out + out_base + even_offsets, acc_even * inv_l, mask=pair_mask)
        tl.store(out + out_base + even_offsets + 1, acc_odd * inv_l, mask=pair_mask)


    @triton.jit
    def _polar_compressed_attention_block_kernel(
        query,
        key_radius_q,
        key_theta_q,
        key_radius_scale,
        value_radius_q,
        value_theta_q,
        value_radius_scale,
        theta_cos_lut,
        theta_sin_lut,
        attention_mask,
        out,
        kv_len,
        max_length: tl.constexpr,
        q_len: tl.constexpr,
        num_key_value_groups: tl.constexpr,
        pair_dim: tl.constexpr,
        head_dim: tl.constexpr,
        inv_radius_levels: tl.constexpr,
        sm_scale: tl.constexpr,
        has_mask: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        q_block = tl.program_id(0)
        q_head = tl.program_id(1)
        kv_head = q_head // num_key_value_groups

        q_offsets = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
        kv_offsets_base = tl.arange(0, BLOCK_N)
        pair_offsets = tl.arange(0, BLOCK_P)
        q_mask = q_offsets < q_len
        pair_mask = pair_offsets < pair_dim
        even_offsets = pair_offsets * 2

        q_base = (q_head * q_len + q_offsets[:, None]) * head_dim
        q_even = tl.load(
            query + q_base + even_offsets[None, :],
            mask=q_mask[:, None] & pair_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_odd = tl.load(
            query + q_base + even_offsets[None, :] + 1,
            mask=q_mask[:, None] & pair_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc_even = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)
        acc_odd = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)

        for block_start in tl.range(0, kv_len, BLOCK_N):
            kv_offsets = block_start + kv_offsets_base
            kv_mask = kv_offsets < kv_len
            src_base = kv_head * max_length + kv_offsets
            pair_src = src_base[:, None] * pair_dim + pair_offsets[None, :]
            load_mask = kv_mask[:, None] & pair_mask[None, :]

            key_radius = tl.load(key_radius_q + pair_src, mask=load_mask, other=0).to(tl.float32)
            key_theta_idx = tl.load(key_theta_q + pair_src, mask=load_mask, other=0).to(tl.int32)
            key_scale = tl.load(key_radius_scale + src_base, mask=kv_mask, other=0.0).to(tl.float32)
            key_cos = tl.load(theta_cos_lut + key_theta_idx, mask=load_mask, other=0.0).to(tl.float32)
            key_sin = tl.load(theta_sin_lut + key_theta_idx, mask=load_mask, other=0.0).to(tl.float32)
            key_radius_dec = key_radius * key_scale[:, None] * inv_radius_levels
            key_even = key_radius_dec * key_cos
            key_odd = key_radius_dec * key_sin

            scores = tl.dot(q_even, tl.trans(key_even), input_precision="ieee")
            scores += tl.dot(q_odd, tl.trans(key_odd), input_precision="ieee")
            scores *= sm_scale
            scores = tl.where(q_mask[:, None] & kv_mask[None, :], scores, -float("inf"))
            if has_mask:
                mask_vals = tl.load(
                    attention_mask + q_offsets[:, None] * kv_len + kv_offsets[None, :],
                    mask=q_mask[:, None] & kv_mask[None, :],
                    other=-float("inf"),
                ).to(tl.float32)
                scores += mask_vals

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)

            value_radius = tl.load(value_radius_q + pair_src, mask=load_mask, other=0).to(tl.float32)
            value_theta_idx = tl.load(value_theta_q + pair_src, mask=load_mask, other=0).to(tl.int32)
            value_scale = tl.load(value_radius_scale + src_base, mask=kv_mask, other=0.0).to(tl.float32)
            value_cos = tl.load(theta_cos_lut + value_theta_idx, mask=load_mask, other=0.0).to(tl.float32)
            value_sin = tl.load(theta_sin_lut + value_theta_idx, mask=load_mask, other=0.0).to(tl.float32)
            value_radius_dec = value_radius * value_scale[:, None] * inv_radius_levels
            value_even = value_radius_dec * value_cos
            value_odd = value_radius_dec * value_sin

            acc_even = (acc_even * alpha[:, None]) + tl.dot(p, value_even, input_precision="ieee")
            acc_odd = (acc_odd * alpha[:, None]) + tl.dot(p, value_odd, input_precision="ieee")
            l_i = (l_i * alpha) + tl.sum(p, axis=1)
            m_i = m_new

        inv_l = 1.0 / tl.maximum(l_i, 1.0e-20)
        out_offsets = (q_head * q_len + q_offsets[:, None]) * head_dim
        store_mask = q_mask[:, None] & pair_mask[None, :]
        tl.store(out + out_offsets + even_offsets[None, :], acc_even * inv_l[:, None], mask=store_mask)
        tl.store(out + out_offsets + even_offsets[None, :] + 1, acc_odd * inv_l[:, None], mask=store_mask)


    @triton.jit
    def _turbo_vq_append_value_kernel(
        tensor,
        q_idx_out,
        scale_out,
        rotation,
        boundaries,
        start_pos,
        input_stride_h,
        input_stride_t,
        input_stride_d,
        input_len: tl.constexpr,
        max_length: tl.constexpr,
        head_dim: tl.constexpr,
        num_boundaries: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        head = tl.program_id(0)
        rel_pos = tl.program_id(1)
        dim_offsets = tl.arange(0, BLOCK_D)
        boundary_offsets = tl.arange(0, BLOCK_B)
        dim_mask = dim_offsets < head_dim
        boundary_mask = boundary_offsets < num_boundaries

        src_base = head * input_stride_h + rel_pos * input_stride_t
        x = tl.load(
            tensor + src_base + dim_offsets * input_stride_d,
            mask=dim_mask & (rel_pos < input_len),
            other=0.0,
        ).to(tl.float32)
        scale = tl.sqrt(tl.sum(x * x, axis=0) / head_dim)
        scale = tl.maximum(scale, 1.0e-6)

        rot = tl.load(
            rotation + dim_offsets[None, :] * head_dim + dim_offsets[:, None],
            mask=dim_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        rotated = tl.sum(x[:, None] * rot, axis=0) / scale

        boundary_vals = tl.load(
            boundaries + boundary_offsets,
            mask=boundary_mask,
            other=float("inf"),
        ).to(tl.float32)
        q_idx = tl.sum(
            tl.where(rotated[:, None] > boundary_vals[None, :], 1, 0),
            axis=1,
        )

        dst_pos = start_pos + rel_pos
        dst_base = head * max_length + dst_pos
        tl.store(scale_out + dst_base, scale)
        tl.store(
            q_idx_out + dst_base * head_dim + dim_offsets,
            q_idx.to(tl.uint8),
            mask=dim_mask & (rel_pos < input_len),
        )


    @triton.jit
    def _turbo_vq_append_key_kernel(
        tensor,
        q_idx_out,
        scale_out,
        residual_sign_packed_out,
        residual_norm_out,
        rotation,
        rotation_t,
        boundaries,
        codebook,
        residual_proj,
        start_pos,
        input_stride_h,
        input_stride_t,
        input_stride_d,
        input_len: tl.constexpr,
        max_length: tl.constexpr,
        head_dim: tl.constexpr,
        residual_dim: tl.constexpr,
        residual_packed_dim: tl.constexpr,
        num_boundaries: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        head = tl.program_id(0)
        rel_pos = tl.program_id(1)
        dim_offsets = tl.arange(0, BLOCK_D)
        residual_offsets = tl.arange(0, BLOCK_R)
        pack_offsets = tl.arange(0, BLOCK_P)
        boundary_offsets = tl.arange(0, BLOCK_B)
        dim_mask = dim_offsets < head_dim
        residual_mask = residual_offsets < residual_dim
        pack_mask = pack_offsets < residual_packed_dim
        boundary_mask = boundary_offsets < num_boundaries

        src_base = head * input_stride_h + rel_pos * input_stride_t
        x = tl.load(
            tensor + src_base + dim_offsets * input_stride_d,
            mask=dim_mask & (rel_pos < input_len),
            other=0.0,
        ).to(tl.float32)
        scale = tl.sqrt(tl.sum(x * x, axis=0) / head_dim)
        scale = tl.maximum(scale, 1.0e-6)

        rot = tl.load(
            rotation + dim_offsets[None, :] * head_dim + dim_offsets[:, None],
            mask=dim_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        rotated = tl.sum(x[:, None] * rot, axis=0) / scale

        boundary_vals = tl.load(
            boundaries + boundary_offsets,
            mask=boundary_mask,
            other=float("inf"),
        ).to(tl.float32)
        q_idx = tl.sum(
            tl.where(rotated[:, None] > boundary_vals[None, :], 1, 0),
            axis=1,
        )

        dst_pos = start_pos + rel_pos
        dst_base = head * max_length + dst_pos
        tl.store(scale_out + dst_base, scale)
        tl.store(
            q_idx_out + dst_base * head_dim + dim_offsets,
            q_idx.to(tl.uint8),
            mask=dim_mask & (rel_pos < input_len),
        )

        code = tl.load(codebook + q_idx, mask=dim_mask, other=0.0).to(tl.float32) * scale
        rot_t = tl.load(
            rotation_t + dim_offsets[:, None] * head_dim + dim_offsets[None, :],
            mask=dim_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        decoded = tl.sum(code[:, None] * rot_t, axis=0)
        residual = x - decoded
        residual_norm = tl.sqrt(tl.sum(residual * residual, axis=0))
        residual_norm = tl.maximum(residual_norm, 1.0e-6)
        tl.store(residual_norm_out + dst_base, residual_norm)

        proj = tl.load(
            residual_proj + dim_offsets[:, None] * residual_dim + residual_offsets[None, :],
            mask=dim_mask[:, None] & residual_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        residual_scores = tl.sum(residual[:, None] * proj, axis=0)
        sign_bits = tl.where(residual_scores >= 0.0, 1, 0).to(tl.int32)
        bit_offsets = residual_offsets - (residual_offsets // 8) * 8
        shifted_bits = sign_bits << bit_offsets
        packed = tl.sum(
            tl.where(
                (residual_offsets[None, :] // 8) == pack_offsets[:, None],
                shifted_bits[None, :],
                0,
            ),
            axis=1,
        )
        tl.store(
            residual_sign_packed_out + dst_base * residual_packed_dim + pack_offsets,
            packed.to(tl.uint8),
            mask=pack_mask & (rel_pos < input_len),
        )


    @triton.jit
    def _turbo_vq_compressed_attention_decode_kernel(
        query,
        key_q_idx,
        key_scale,
        key_codebook,
        key_rotation_t,
        key_residual_sign_packed,
        key_residual_norm,
        key_residual_proj,
        value_q_idx,
        value_scale,
        value_codebook,
        value_rotation_t,
        attention_mask,
        out,
        kv_len,
        query_stride_h,
        query_stride_q,
        query_stride_d,
        max_length: tl.constexpr,
        num_key_value_groups: tl.constexpr,
        head_dim: tl.constexpr,
        residual_dim: tl.constexpr,
        residual_packed_dim: tl.constexpr,
        residual_coeff: tl.constexpr,
        sm_scale: tl.constexpr,
        has_mask: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        q_head = tl.program_id(0)
        kv_head = q_head // num_key_value_groups

        kv_offsets_base = tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)
        residual_offsets = tl.arange(0, BLOCK_R)
        dim_mask = dim_offsets < head_dim
        residual_mask = residual_offsets < residual_dim

        q_base = q_head * query_stride_h
        q_vals = tl.load(
            query + q_base + dim_offsets * query_stride_d,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)

        rot_for_q = tl.load(
            key_rotation_t + dim_offsets[None, :] * head_dim + dim_offsets[:, None],
            mask=dim_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_rot = tl.sum(q_vals[:, None] * rot_for_q, axis=0)

        residual_proj = tl.load(
            key_residual_proj + dim_offsets[:, None] * residual_dim + residual_offsets[None, :],
            mask=dim_mask[:, None] & residual_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_residual_proj = tl.sum(q_vals[:, None] * residual_proj, axis=0)

        acc_rot = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m_i = tl.full((), -float("inf"), dtype=tl.float32)
        l_i = tl.full((), 0.0, dtype=tl.float32)

        for block_start in tl.range(0, kv_len, BLOCK_N):
            kv_offsets = block_start + kv_offsets_base
            kv_mask = kv_offsets < kv_len
            cache_base = kv_head * max_length + kv_offsets
            kv_dim_base = cache_base[:, None] * head_dim + dim_offsets[None, :]
            kv_dim_mask = kv_mask[:, None] & dim_mask[None, :]

            key_idx = tl.load(key_q_idx + kv_dim_base, mask=kv_dim_mask, other=0).to(tl.int32)
            key_code = tl.load(key_codebook + key_idx, mask=kv_dim_mask, other=0.0).to(tl.float32)
            key_scales = tl.load(key_scale + cache_base, mask=kv_mask, other=0.0).to(tl.float32)
            key_rot = key_code * key_scales[:, None]
            scores = tl.sum(key_rot * q_rot[None, :], axis=1)

            pack_offsets = residual_offsets // 8
            bit_offsets = residual_offsets - (pack_offsets * 8)
            sign_base = cache_base[:, None] * residual_packed_dim + pack_offsets[None, :]
            packed = tl.load(
                key_residual_sign_packed + sign_base,
                mask=kv_mask[:, None] & residual_mask[None, :],
                other=0,
            ).to(tl.int32)
            sign_bits = (packed >> bit_offsets[None, :]) & 1
            signs = sign_bits.to(tl.float32) * 2.0 - 1.0
            signs = tl.where(residual_mask[None, :], signs, 0.0)
            residual_inner = tl.sum(signs * q_residual_proj[None, :], axis=1)
            residual_norm = tl.load(key_residual_norm + cache_base, mask=kv_mask, other=0.0).to(tl.float32)
            scores += residual_coeff * residual_inner * residual_norm

            scores *= sm_scale
            scores = tl.where(kv_mask, scores, -float("inf"))
            if has_mask:
                mask_vals = tl.load(
                    attention_mask + kv_offsets,
                    mask=kv_mask,
                    other=-float("inf"),
                ).to(tl.float32)
                scores += mask_vals

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)

            value_idx = tl.load(value_q_idx + kv_dim_base, mask=kv_dim_mask, other=0).to(tl.int32)
            value_code = tl.load(value_codebook + value_idx, mask=kv_dim_mask, other=0.0).to(tl.float32)
            value_scales = tl.load(value_scale + cache_base, mask=kv_mask, other=0.0).to(tl.float32)
            value_rot = value_code * value_scales[:, None]

            acc_rot = (acc_rot * alpha) + tl.sum(p[:, None] * value_rot, axis=0)
            l_i = (l_i * alpha) + tl.sum(p, axis=0)
            m_i = m_new

        acc_rot = acc_rot / tl.maximum(l_i, 1.0e-20)
        rot_for_out = tl.load(
            value_rotation_t + dim_offsets[:, None] * head_dim + dim_offsets[None, :],
            mask=dim_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        out_vals = tl.sum(acc_rot[:, None] * rot_for_out, axis=0)

        out_base = q_head * head_dim
        tl.store(out + out_base + dim_offsets, out_vals, mask=dim_mask)


    @triton.jit
    def _turbo_vq_hybrid_attention_decode_kernel(
        query,
        key_q_idx,
        key_scale,
        key_codebook,
        key_rotation_t,
        key_residual_sign_packed,
        key_residual_norm,
        key_residual_proj,
        value_q_idx,
        value_scale,
        value_codebook,
        value_rotation_t,
        key_hot,
        value_hot,
        attention_mask,
        out,
        kv_len,
        old_len,
        query_stride_h,
        query_stride_q,
        query_stride_d,
        max_length: tl.constexpr,
        hot_capacity: tl.constexpr,
        num_key_value_groups: tl.constexpr,
        head_dim: tl.constexpr,
        residual_dim: tl.constexpr,
        residual_packed_dim: tl.constexpr,
        residual_coeff: tl.constexpr,
        sm_scale: tl.constexpr,
        has_mask: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        q_head = tl.program_id(0)
        kv_head = q_head // num_key_value_groups

        kv_offsets_base = tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)
        residual_offsets = tl.arange(0, BLOCK_R)
        dim_mask = dim_offsets < head_dim
        residual_mask = residual_offsets < residual_dim

        q_base = q_head * query_stride_h
        q_vals = tl.load(
            query + q_base + dim_offsets * query_stride_d,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)

        rot_for_q = tl.load(
            key_rotation_t + dim_offsets[None, :] * head_dim + dim_offsets[:, None],
            mask=dim_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_rot = tl.sum(q_vals[:, None] * rot_for_q, axis=0)

        residual_proj = tl.load(
            key_residual_proj + dim_offsets[:, None] * residual_dim + residual_offsets[None, :],
            mask=dim_mask[:, None] & residual_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_residual_proj = tl.sum(q_vals[:, None] * residual_proj, axis=0)

        rot_for_out = tl.load(
            value_rotation_t + dim_offsets[:, None] * head_dim + dim_offsets[None, :],
            mask=dim_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        m_i = tl.full((), -float("inf"), dtype=tl.float32)
        l_i = tl.full((), 0.0, dtype=tl.float32)

        for block_start in tl.range(0, old_len, BLOCK_N):
            kv_offsets = block_start + kv_offsets_base
            kv_mask = kv_offsets < old_len
            cache_base = kv_head * max_length + kv_offsets
            kv_dim_base = cache_base[:, None] * head_dim + dim_offsets[None, :]
            kv_dim_mask = kv_mask[:, None] & dim_mask[None, :]

            key_idx = tl.load(key_q_idx + kv_dim_base, mask=kv_dim_mask, other=0).to(tl.int32)
            key_code = tl.load(key_codebook + key_idx, mask=kv_dim_mask, other=0.0).to(tl.float32)
            key_scales = tl.load(key_scale + cache_base, mask=kv_mask, other=0.0).to(tl.float32)
            key_rot = key_code * key_scales[:, None]
            scores = tl.sum(key_rot * q_rot[None, :], axis=1)

            pack_offsets = residual_offsets // 8
            bit_offsets = residual_offsets - (pack_offsets * 8)
            sign_base = cache_base[:, None] * residual_packed_dim + pack_offsets[None, :]
            packed = tl.load(
                key_residual_sign_packed + sign_base,
                mask=kv_mask[:, None] & residual_mask[None, :],
                other=0,
            ).to(tl.int32)
            sign_bits = (packed >> bit_offsets[None, :]) & 1
            signs = sign_bits.to(tl.float32) * 2.0 - 1.0
            signs = tl.where(residual_mask[None, :], signs, 0.0)
            residual_inner = tl.sum(signs * q_residual_proj[None, :], axis=1)
            residual_norm = tl.load(key_residual_norm + cache_base, mask=kv_mask, other=0.0).to(tl.float32)
            scores += residual_coeff * residual_inner * residual_norm

            scores *= sm_scale
            scores = tl.where(kv_mask, scores, -float("inf"))
            if has_mask:
                mask_vals = tl.load(
                    attention_mask + kv_offsets,
                    mask=kv_mask,
                    other=-float("inf"),
                ).to(tl.float32)
                scores += mask_vals

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)

            value_idx = tl.load(value_q_idx + kv_dim_base, mask=kv_dim_mask, other=0).to(tl.int32)
            value_code = tl.load(value_codebook + value_idx, mask=kv_dim_mask, other=0.0).to(tl.float32)
            value_scales = tl.load(value_scale + cache_base, mask=kv_mask, other=0.0).to(tl.float32)
            value_rot = value_code * value_scales[:, None]
            value_vals = tl.dot(value_rot, rot_for_out, input_precision="ieee")

            acc = (acc * alpha) + tl.sum(p[:, None] * value_vals, axis=0)
            l_i = (l_i * alpha) + tl.sum(p, axis=0)
            m_i = m_new

        hot_len = kv_len - old_len
        for hot_block_start in tl.range(0, hot_len, BLOCK_N):
            hot_offsets = hot_block_start + kv_offsets_base
            kv_offsets = old_len + hot_offsets
            kv_mask = kv_offsets < kv_len
            hot_slots = kv_offsets - (kv_offsets // hot_capacity) * hot_capacity
            hot_base = kv_head * hot_capacity + hot_slots
            hot_dim_base = hot_base[:, None] * head_dim + dim_offsets[None, :]
            hot_dim_mask = kv_mask[:, None] & dim_mask[None, :]

            key_vals = tl.load(key_hot + hot_dim_base, mask=hot_dim_mask, other=0.0).to(tl.float32)
            scores = tl.sum(key_vals * q_vals[None, :], axis=1) * sm_scale
            scores = tl.where(kv_mask, scores, -float("inf"))
            if has_mask:
                mask_vals = tl.load(
                    attention_mask + kv_offsets,
                    mask=kv_mask,
                    other=-float("inf"),
                ).to(tl.float32)
                scores += mask_vals

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)

            value_vals = tl.load(value_hot + hot_dim_base, mask=hot_dim_mask, other=0.0).to(tl.float32)
            acc = (acc * alpha) + tl.sum(p[:, None] * value_vals, axis=0)
            l_i = (l_i * alpha) + tl.sum(p, axis=0)
            m_i = m_new

        out_base = q_head * head_dim
        out_vals = acc / tl.maximum(l_i, 1.0e-20)
        tl.store(out + out_base + dim_offsets, out_vals, mask=dim_mask)


    @triton.jit
    def _turbo_vq_compressed_attention_block_kernel(
        query,
        key_q_idx,
        key_scale,
        key_codebook,
        key_rotation_t,
        key_residual_sign_packed,
        key_residual_norm,
        key_residual_proj,
        value_q_idx,
        value_scale,
        value_codebook,
        value_rotation_t,
        attention_mask,
        out,
        kv_len,
        query_stride_h,
        query_stride_q,
        query_stride_d,
        max_length: tl.constexpr,
        q_len: tl.constexpr,
        num_key_value_groups: tl.constexpr,
        head_dim: tl.constexpr,
        residual_dim: tl.constexpr,
        residual_packed_dim: tl.constexpr,
        residual_coeff: tl.constexpr,
        sm_scale: tl.constexpr,
        has_mask: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        q_block = tl.program_id(0)
        q_head = tl.program_id(1)
        kv_head = q_head // num_key_value_groups

        q_offsets = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
        kv_offsets_base = tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)
        residual_offsets = tl.arange(0, BLOCK_R)

        q_mask = q_offsets < q_len
        dim_mask = dim_offsets < head_dim
        residual_mask = residual_offsets < residual_dim

        q_base = q_head * query_stride_h + q_offsets[:, None] * query_stride_q
        q_vals = tl.load(
            query + q_base + dim_offsets[None, :] * query_stride_d,
            mask=q_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Work in the same rotated basis used by TurboVQ: q_rot = q @ rotation.
        rot_for_q = tl.load(
            key_rotation_t + dim_offsets[None, :] * head_dim + dim_offsets[:, None],
            mask=dim_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_rot = tl.dot(q_vals, rot_for_q, input_precision="ieee")

        residual_proj = tl.load(
            key_residual_proj + dim_offsets[:, None] * residual_dim + residual_offsets[None, :],
            mask=dim_mask[:, None] & residual_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q_residual_proj = tl.dot(q_vals, residual_proj, input_precision="ieee")

        acc_rot = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)

        for block_start in tl.range(0, kv_len, BLOCK_N):
            kv_offsets = block_start + kv_offsets_base
            kv_mask = kv_offsets < kv_len
            cache_base = kv_head * max_length + kv_offsets
            kv_dim_base = cache_base[:, None] * head_dim + dim_offsets[None, :]
            kv_dim_mask = kv_mask[:, None] & dim_mask[None, :]

            key_idx = tl.load(key_q_idx + kv_dim_base, mask=kv_dim_mask, other=0).to(tl.int32)
            key_code = tl.load(key_codebook + key_idx, mask=kv_dim_mask, other=0.0).to(tl.float32)
            key_scales = tl.load(key_scale + cache_base, mask=kv_mask, other=0.0).to(tl.float32)
            key_rot = key_code * key_scales[:, None]

            scores = tl.dot(q_rot, tl.trans(key_rot), input_precision="ieee")

            pack_offsets = residual_offsets // 8
            bit_offsets = residual_offsets - (pack_offsets * 8)
            sign_base = cache_base[:, None] * residual_packed_dim + pack_offsets[None, :]
            packed = tl.load(
                key_residual_sign_packed + sign_base,
                mask=kv_mask[:, None] & residual_mask[None, :],
                other=0,
            ).to(tl.int32)
            sign_bits = (packed >> bit_offsets[None, :]) & 1
            signs = sign_bits.to(tl.float32) * 2.0 - 1.0
            signs = tl.where(residual_mask[None, :], signs, 0.0)
            residual_inner = tl.dot(
                q_residual_proj,
                tl.trans(signs),
                input_precision="ieee",
            )
            residual_norm = tl.load(
                key_residual_norm + cache_base,
                mask=kv_mask,
                other=0.0,
            ).to(tl.float32)
            scores += residual_coeff * residual_inner * residual_norm[None, :]

            scores *= sm_scale
            scores = tl.where(q_mask[:, None] & kv_mask[None, :], scores, -float("inf"))
            if has_mask:
                mask_vals = tl.load(
                    attention_mask + q_offsets[:, None] * kv_len + kv_offsets[None, :],
                    mask=q_mask[:, None] & kv_mask[None, :],
                    other=-float("inf"),
                ).to(tl.float32)
                scores += mask_vals

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)

            value_idx = tl.load(value_q_idx + kv_dim_base, mask=kv_dim_mask, other=0).to(tl.int32)
            value_code = tl.load(value_codebook + value_idx, mask=kv_dim_mask, other=0.0).to(tl.float32)
            value_scales = tl.load(value_scale + cache_base, mask=kv_mask, other=0.0).to(tl.float32)
            value_rot = value_code * value_scales[:, None]

            acc_rot = (acc_rot * alpha[:, None]) + tl.dot(p, value_rot, input_precision="ieee")
            l_i = (l_i * alpha) + tl.sum(p, axis=1)
            m_i = m_new

        inv_l = 1.0 / tl.maximum(l_i, 1.0e-20)
        acc_rot = acc_rot * inv_l[:, None]

        rot_for_out = tl.load(
            value_rotation_t + dim_offsets[:, None] * head_dim + dim_offsets[None, :],
            mask=dim_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        out_vals = tl.dot(acc_rot, rot_for_out, input_precision="ieee")

        out_offsets = (q_head * q_len + q_offsets[:, None]) * head_dim
        tl.store(
            out + out_offsets + dim_offsets[None, :],
            out_vals,
            mask=q_mask[:, None] & dim_mask[None, :],
        )


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


def packed_kv_qjl_node_scores_triton(query_bits, key_bits, kv_len, block_k=1024):
    if not TRITON_AVAILABLE or not _is_cuda_tensor(query_bits):
        return None
    if not _is_cuda_tensor(key_bits):
        return None
    if query_bits.dtype != torch.int32 or key_bits.dtype != torch.int32:
        return None
    if query_bits.dim() != 3 or key_bits.dim() != 3:
        return None
    n_nodes, num_heads, words = query_bits.shape
    key_heads, max_length, key_words = key_bits.shape
    if int(num_heads) != int(key_heads) or int(words) != int(key_words):
        return None
    kv_len = min(int(kv_len), int(max_length))
    if n_nodes <= 0 or num_heads <= 0 or words <= 0 or kv_len <= 0:
        return None
    block_w = triton.next_power_of_2(int(words))
    if block_w > 32:
        return None
    block_k = triton.next_power_of_2(int(max(16, min(int(block_k), kv_len))))
    num_blocks = triton.cdiv(kv_len, block_k)
    partial = torch.empty(
        (int(n_nodes), int(num_heads), int(num_blocks)),
        device=query_bits.device,
        dtype=torch.int32,
    )
    try:
        _packed_kv_qjl_node_scores_kernel[(int(n_nodes), int(num_heads), int(num_blocks))](
            query_bits.contiguous(),
            key_bits.contiguous(),
            partial,
            kv_len=int(kv_len),
            max_length=int(max_length),
            num_heads=int(num_heads),
            words=int(words),
            num_blocks=int(num_blocks),
            BLOCK_K=block_k,
            BLOCK_W=block_w,
            num_warps=4,
        )
    except Exception as exc:
        _debug_triton_failure("packed_kv_qjl_node_scores_triton", exc)
        return None
    return partial.to(torch.float32).sum(dim=(1, 2))


def turbo_qjl_select_paths_triton(
    q_proj,
    sign_cache,
    norm_cache,
    candidates,
    valid_mask,
    medusa_scores,
    mandatory_indices,
    coeff,
    sketch_dim,
    keep_target,
    min_keep,
    max_keep,
    margin_scale,
):
    if not TRITON_AVAILABLE or not _is_cuda_tensor(q_proj):
        return None
    tensors = (sign_cache, norm_cache, candidates, valid_mask, medusa_scores, mandatory_indices)
    if not all(_is_cuda_tensor(tensor) for tensor in tensors):
        return None
    if candidates.dim() != 2 or valid_mask.shape != candidates.shape:
        return None
    if medusa_scores.dim() != 1 or medusa_scores.shape[0] != candidates.shape[0]:
        return None

    n_paths, path_len = candidates.shape
    if n_paths == 0:
        return None
    max_keep = int(max(1, min(max_keep, n_paths)))
    base_keep = max(int(min_keep), min(int(max_keep), int(keep_target)))
    base_keep = min(base_keep, n_paths)
    block_p = triton.next_power_of_2(int(n_paths))
    block_m = triton.next_power_of_2(int(sketch_dim))
    if block_p > 128 or block_m > 1024:
        return None

    q_proj = q_proj.reshape(-1).contiguous()
    candidates = candidates.contiguous()
    valid_mask = valid_mask.contiguous()
    medusa_scores = medusa_scores.contiguous()
    mandatory_indices = mandatory_indices.contiguous()
    mandatory_count = min(int(mandatory_indices.numel()), 2)

    approx = torch.empty((n_paths,), device=candidates.device, dtype=torch.float32)
    selected = torch.empty((max_keep,), device=candidates.device, dtype=torch.long)
    selected_count = torch.empty((), device=candidates.device, dtype=torch.long)
    verify_full = torch.empty((), device=candidates.device, dtype=torch.uint8)
    try:
        _turbo_qjl_select_kernel[(1,)](
            q_proj,
            sign_cache.contiguous(),
            norm_cache.contiguous(),
            candidates,
            valid_mask,
            medusa_scores,
            mandatory_indices,
            approx,
            selected,
            selected_count,
            verify_full,
            path_len=int(path_len),
            n_paths=int(n_paths),
            sketch_dim=int(sketch_dim),
            coeff=float(coeff),
            base_keep=int(base_keep),
            max_keep=int(max_keep),
            mandatory_count=int(mandatory_count),
            margin_scale=float(margin_scale),
            BLOCK_P=block_p,
            BLOCK_M=block_m,
            num_warps=4,
        )
    except RuntimeError:
        return None

    count = int(selected_count.item())
    return approx, selected[:count], bool(int(verify_full.item()))


def node_budget_select_triton(
    scores,
    retrieve_indices,
    mandatory_indices,
    node_budget,
    min_keep,
    max_keep,
    full_node_count,
):
    if not TRITON_AVAILABLE or not _is_cuda_tensor(scores):
        return None
    if not (_is_cuda_tensor(retrieve_indices) and _is_cuda_tensor(mandatory_indices)):
        return None
    if scores.dim() != 1 or retrieve_indices.dim() != 2:
        return None
    n_paths, path_len = retrieve_indices.shape
    if int(n_paths) != int(scores.shape[0]) or int(n_paths) == 0:
        return None
    max_keep = int(max(1, min(max_keep, n_paths)))
    min_keep = int(max(1, min(min_keep, max_keep)))
    full_node_count = int(max(1, full_node_count))
    if int(n_paths) > 128 or int(full_node_count) > 128:
        return None

    block_p = triton.next_power_of_2(int(n_paths))
    block_n = triton.next_power_of_2(int(full_node_count))
    mandatory_indices = mandatory_indices.to(device=scores.device, dtype=torch.long).contiguous()
    mandatory_count = min(int(mandatory_indices.numel()), max_keep)
    selected = torch.empty((max_keep,), device=scores.device, dtype=torch.long)
    selected_count = torch.empty((), device=scores.device, dtype=torch.long)
    try:
        _node_budget_select_kernel[(1,)](
            scores.contiguous(),
            retrieve_indices.contiguous(),
            mandatory_indices,
            selected,
            selected_count,
            n_paths=int(n_paths),
            path_len=int(path_len),
            full_node_count=int(full_node_count),
            node_budget=int(max(1, node_budget)),
            min_keep=int(min_keep),
            max_keep=int(max_keep),
            mandatory_count=int(mandatory_count),
            BLOCK_P=block_p,
            BLOCK_N=block_n,
            num_warps=4,
        )
    except Exception:
        return None
    count = int(selected_count.item())
    if count <= 0:
        return None
    return selected[: min(count, max_keep)]


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


def _launch_greedy_tree_unique_posterior(
    node_logits,
    candidates,
    retrieve_indices,
    path_lengths,
    n_paths,
    path_len,
    vocab_size,
):
    n_nodes = int(node_logits.shape[0])
    if n_nodes <= 0 or n_nodes > 128 or n_paths > 128 or path_len > 8:
        return None
    # This variant keeps the whole argmax + prefix selection inside one Triton
    # program. It is useful for fixed compact trees, but can lose to torch.argmax
    # on large vocab/tree shapes, so callers should keep a fallback.
    block_n = triton.next_power_of_2(n_nodes)
    block_p = triton.next_power_of_2(int(n_paths))
    block_v = 256
    if path_lengths is None:
        path_lengths_ptr = candidates
        has_path_lengths = False
    else:
        path_lengths_ptr = path_lengths.contiguous()
        has_path_lengths = True

    best_candidate = torch.empty((), device=candidates.device, dtype=torch.long)
    accept_length = torch.empty((), device=candidates.device, dtype=torch.long)
    try:
        _greedy_tree_posterior_unique_kernel[(1,)](
            node_logits,
            candidates,
            retrieve_indices,
            path_lengths_ptr,
            best_candidate,
            accept_length,
            n_nodes=n_nodes,
            n_paths=int(n_paths),
            path_len=int(path_len),
            vocab_size=int(vocab_size),
            has_path_lengths=has_path_lengths,
            BLOCK_N=block_n,
            BLOCK_P=block_p,
            BLOCK_V=block_v,
            num_warps=8,
        )
    except Exception:
        return None
    return best_candidate, accept_length


def _launch_greedy_accept_from_argmax(
    node_argmax,
    candidates,
    retrieve_indices,
    path_lengths,
    n_paths,
    path_len,
):
    if n_paths <= 0 or n_paths > 256 or path_len <= 1 or path_len > 16:
        return None
    block_p = triton.next_power_of_2(int(n_paths))
    if path_lengths is None:
        path_lengths_ptr = candidates
        has_path_lengths = False
    else:
        path_lengths_ptr = path_lengths.contiguous()
        has_path_lengths = True
    best_candidate = torch.empty((), device=candidates.device, dtype=torch.long)
    accept_length = torch.empty((), device=candidates.device, dtype=torch.long)
    try:
        _greedy_tree_accept_from_argmax_kernel[(1,)](
            node_argmax,
            candidates,
            retrieve_indices,
            path_lengths_ptr,
            best_candidate,
            accept_length,
            n_paths=int(n_paths),
            path_len=int(path_len),
            has_path_lengths=has_path_lengths,
            BLOCK_P=block_p,
            num_warps=4,
        )
    except Exception:
        return None
    return best_candidate, accept_length


def greedy_accept_from_argmax_triton(node_argmax, candidates, retrieve_indices, path_lengths=None):
    if not TRITON_AVAILABLE or not _is_cuda_tensor(node_argmax):
        return None
    if not (_is_cuda_tensor(candidates) and _is_cuda_tensor(retrieve_indices)):
        return None
    if candidates.dim() != 2 or retrieve_indices.shape != candidates.shape:
        return None
    n_paths, path_len = candidates.shape
    if n_paths <= 0 or path_len <= 1:
        return None
    if path_lengths is not None and not _is_cuda_tensor(path_lengths):
        return None
    return _launch_greedy_accept_from_argmax(
        node_argmax.contiguous(),
        candidates.contiguous(),
        retrieve_indices.contiguous(),
        path_lengths.contiguous() if path_lengths is not None else None,
        int(n_paths),
        int(path_len),
    )


def lm_head_argmax_triton(hidden_states, lm_head_weight, block_v=128, block_h=64):
    if not TRITON_AVAILABLE or not _is_cuda_tensor(hidden_states):
        return None
    if not _is_cuda_tensor(lm_head_weight):
        return None
    if hidden_states.dim() != 2 or lm_head_weight.dim() != 2:
        return None
    n_nodes = int(hidden_states.shape[0])
    hidden_size = int(hidden_states.shape[1])
    vocab_size = int(lm_head_weight.shape[0])
    if n_nodes <= 0 or hidden_size <= 0 or vocab_size <= 0:
        return None
    if int(lm_head_weight.shape[1]) != hidden_size:
        return None
    if n_nodes > 256 or vocab_size > 262144:
        return None

    hidden_states = hidden_states.contiguous()
    if not lm_head_weight.is_contiguous():
        lm_head_weight = lm_head_weight.contiguous()

    block_v = int(block_v)
    block_h = int(block_h)
    if block_v <= 0 or block_h <= 0:
        return None
    num_vocab_blocks = triton.cdiv(vocab_size, block_v)
    partial_vals = torch.empty(
        (n_nodes, num_vocab_blocks),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    partial_ids = torch.empty(
        (n_nodes, num_vocab_blocks),
        device=hidden_states.device,
        dtype=torch.int32,
    )
    node_argmax = torch.empty((n_nodes,), device=hidden_states.device, dtype=torch.int32)

    try:
        block_m = 16
        _lm_head_argmax_matmul_partial_kernel[(triton.cdiv(n_nodes, block_m), num_vocab_blocks)](
            hidden_states,
            lm_head_weight,
            partial_vals,
            partial_ids,
            n_nodes=n_nodes,
            num_vocab_blocks=num_vocab_blocks,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            BLOCK_M=block_m,
            BLOCK_N=block_v,
            BLOCK_K=block_h,
            num_warps=4,
        )
        _lm_head_argmax_reduce_kernel[(n_nodes,)](
            partial_vals,
            partial_ids,
            node_argmax,
            num_vocab_blocks=num_vocab_blocks,
            BLOCK_B=triton.next_power_of_2(num_vocab_blocks),
            num_warps=8,
        )
    except Exception:
        return None
    return node_argmax


def greedy_tree_posterior_triton(tree_logits, candidates, retrieve_indices, path_lengths=None):
    if not TRITON_AVAILABLE or not _is_cuda_tensor(tree_logits):
        return None
    if not (_is_cuda_tensor(candidates) and _is_cuda_tensor(retrieve_indices)):
        return None
    if tree_logits.dim() == 3:
        if tree_logits.shape[0] != 1:
            return None
        node_logits = tree_logits[0]
    elif tree_logits.dim() == 2:
        node_logits = tree_logits
    else:
        return None
    if candidates.dim() != 2 or retrieve_indices.shape != candidates.shape:
        return None
    n_paths, path_len = candidates.shape
    if n_paths <= 0 or path_len <= 1:
        return None
    if n_paths > 128 or path_len > 8:
        return None
    if path_lengths is not None and not _is_cuda_tensor(path_lengths):
        return None

    vocab_size = int(node_logits.shape[-1])
    if vocab_size <= 0 or vocab_size > 131072:
        return None
    # Try the true one-kernel verifier only for very small compact trees. On
    # LLM vocab sizes, torch's row-wise argmax is much faster than scanning all
    # logits from one Triton program, so the default fast path below keeps
    # argmax in torch and fuses the remaining prefix/best-path work.
    argmax_work = int(n_paths) * max(1, int(path_len) - 1)
    node_logits = node_logits.contiguous()
    candidates = candidates.contiguous()
    retrieve_indices = retrieve_indices.contiguous()
    if path_lengths is not None:
        path_lengths = path_lengths.contiguous()

    if vocab_size <= 8192 and int(node_logits.shape[0]) <= 16 and int(n_paths) <= 16:
        fused_unique = _launch_greedy_tree_unique_posterior(
            node_logits,
            candidates,
            retrieve_indices,
            path_lengths,
            n_paths,
            path_len,
            vocab_size,
        )
        if fused_unique is not None:
            return fused_unique

    node_argmax = torch.argmax(node_logits, dim=-1)
    accept_result = _launch_greedy_accept_from_argmax(
        node_argmax,
        candidates,
        retrieve_indices,
        path_lengths,
        n_paths,
        path_len,
    )
    if accept_result is not None:
        return accept_result

    # The path-per-program kernel scans vocab once per path position. It is only
    # kept as a fallback for unusual shapes unsupported by the prefix selector.
    if argmax_work > int(node_logits.shape[0]) * 2:
        return None

    block_p = triton.next_power_of_2(int(n_paths))
    block_v = 512
    if path_lengths is None:
        path_lengths_ptr = candidates
        has_path_lengths = False
    else:
        path_lengths_ptr = path_lengths
        has_path_lengths = True

    best_candidate = torch.empty((), device=candidates.device, dtype=torch.long)
    accept_length = torch.empty((), device=candidates.device, dtype=torch.long)
    try:
        _greedy_tree_posterior_kernel[(1,)](
            node_logits,
            candidates,
            retrieve_indices,
            path_lengths_ptr,
            best_candidate,
            accept_length,
            n_paths=int(n_paths),
            path_len=int(path_len),
            vocab_size=vocab_size,
            has_path_lengths=has_path_lengths,
            BLOCK_P=block_p,
            BLOCK_V=block_v,
            num_warps=8,
        )
    except Exception:
        node_argmax = torch.argmax(node_logits, dim=-1)
        return _launch_greedy_accept_from_argmax(
            node_argmax,
            candidates,
            retrieve_indices,
            path_lengths,
            n_paths,
            path_len,
        )
    return best_candidate, accept_length


def copy_selected_kv_cache_triton(
    past_key_values_data,
    select_indices,
    prev_input_len,
    copy_start=1,
):
    if not TRITON_AVAILABLE or not _is_cuda_tensor(past_key_values_data):
        return False
    if not _is_cuda_tensor(select_indices):
        return False
    if past_key_values_data.dim() != 5 or not past_key_values_data.is_contiguous():
        return False
    if select_indices.dim() != 1 or select_indices.numel() <= int(copy_start):
        return True

    copy_start = int(copy_start)
    copy_count = int(select_indices.numel()) - copy_start
    if copy_count <= 0 or copy_count > 8:
        return False

    layer_kv, batch_size, num_heads, max_length, head_dim = past_key_values_data.shape
    total = int(layer_kv) * int(batch_size) * int(num_heads) * copy_count * int(head_dim)
    if total <= 0:
        return True

    block = 256
    grid = (triton.cdiv(total, block),)
    try:
        _copy_selected_kv_cache_kernel[grid](
            past_key_values_data,
            select_indices.contiguous(),
            int(prev_input_len),
            total_elements=total,
            batch_size=int(batch_size),
            num_heads=int(num_heads),
            max_length=int(max_length),
            head_dim=int(head_dim),
            copy_start=copy_start,
            copy_count=copy_count,
            BLOCK=block,
            num_warps=4,
        )
    except Exception:
        return False
    return True


def turbo_vq_append_triton(cache, tensor, start):
    if not TRITON_AVAILABLE or not _is_cuda_tensor(tensor):
        return False
    if tensor.dim() != 4 or tensor.shape[0] != 1:
        return False
    if tensor.shape[1] != int(getattr(cache, "num_heads", -1)):
        return False

    _, num_heads, input_len, head_dim = tensor.shape
    if head_dim != int(getattr(cache, "head_dim", -1)):
        return False
    if int(getattr(cache, "bits", 0)) != 8:
        return False

    q_idx = getattr(cache, "q_idx", None)
    scale = getattr(cache, "scale", None)
    rotation = getattr(cache, "rotation", None)
    rotation_t = getattr(cache, "rotation_t", None)
    boundaries = getattr(cache, "boundaries", None)
    codebook = getattr(cache, "codebook", None)
    if not all(_is_cuda_tensor(t) for t in (q_idx, scale, rotation, rotation_t, boundaries, codebook)):
        return False

    block_d = triton.next_power_of_2(int(head_dim))
    num_boundaries = int(boundaries.numel())
    block_b = triton.next_power_of_2(max(1, num_boundaries))
    if block_d > 128 or block_b > 256:
        return False

    stride_h = int(tensor.stride(1))
    stride_t = int(tensor.stride(2))
    stride_d = int(tensor.stride(3))
    grid = (int(num_heads), int(input_len))

    residual_dim = int(getattr(cache, "residual_dim", 0))
    if residual_dim <= 0:
        try:
            _turbo_vq_append_value_kernel[grid](
                tensor,
                q_idx,
                scale,
                rotation,
                boundaries,
                int(start),
                stride_h,
                stride_t,
                stride_d,
                input_len=int(input_len),
                max_length=int(cache.max_length),
                head_dim=int(head_dim),
                num_boundaries=num_boundaries,
                BLOCK_D=block_d,
                BLOCK_B=block_b,
                num_warps=4,
            )
        except Exception:
            return False
        return True

    residual_packed_dim = int(getattr(cache, "residual_packed_dim", 0))
    residual_sign_packed = getattr(cache, "residual_sign_packed", None)
    residual_norm = getattr(cache, "residual_norm", None)
    residual_proj = getattr(cache, "residual_proj", None)
    if residual_dim > 128:
        return False
    if not all(_is_cuda_tensor(t) for t in (residual_sign_packed, residual_norm, residual_proj)):
        return False
    block_r = triton.next_power_of_2(residual_dim)
    block_p = triton.next_power_of_2(max(1, residual_packed_dim))
    if block_r > 128 or block_p > 32:
        return False

    try:
        _turbo_vq_append_key_kernel[grid](
            tensor,
            q_idx,
            scale,
            residual_sign_packed,
            residual_norm,
            rotation,
            rotation_t,
            boundaries,
            codebook,
            residual_proj,
            int(start),
            stride_h,
            stride_t,
            stride_d,
            input_len=int(input_len),
            max_length=int(cache.max_length),
            head_dim=int(head_dim),
            residual_dim=residual_dim,
            residual_packed_dim=residual_packed_dim,
            num_boundaries=num_boundaries,
            BLOCK_D=block_d,
            BLOCK_R=block_r,
            BLOCK_P=block_p,
            BLOCK_B=block_b,
            num_warps=4,
        )
    except Exception:
        return False
    return True


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


def compressed_kv_attention_polar_triton(
    query_states,
    key_cache,
    value_cache,
    attention_mask,
    num_key_value_groups,
    sm_scale,
):
    if not TRITON_AVAILABLE or not _is_cuda_tensor(query_states):
        return None
    if query_states.dim() != 4 or query_states.shape[0] != 1:
        return None
    if getattr(key_cache, "dequant_data", None) is not None:
        return None
    if getattr(value_cache, "dequant_data", None) is not None:
        return None

    key_tensors = (
        getattr(key_cache, "radius_q", None),
        getattr(key_cache, "theta_q", None),
        getattr(key_cache, "radius_scale", None),
    )
    value_tensors = (
        getattr(value_cache, "radius_q", None),
        getattr(value_cache, "theta_q", None),
        getattr(value_cache, "radius_scale", None),
    )
    if not all(_is_cuda_tensor(tensor) for tensor in key_tensors + value_tensors):
        return None
    if key_cache.radius_q.shape != value_cache.radius_q.shape:
        return None

    _, num_heads, q_len, head_dim = query_states.shape
    kv_len = int(key_cache.current_length.item())
    if kv_len <= 0:
        return None
    if head_dim != int(key_cache.head_dim):
        return None
    if int(key_cache.num_heads) * int(num_key_value_groups) != int(num_heads):
        return None
    if int(key_cache.pair_dim) * 2 != int(head_dim):
        return None
    if attention_mask is not None:
        if not _is_cuda_tensor(attention_mask) or attention_mask.shape != (1, 1, q_len, kv_len):
            return None
        attention_mask = attention_mask.contiguous()

    pair_dim = int(key_cache.pair_dim)
    block_p = triton.next_power_of_2(pair_dim)
    if block_p > 128:
        return None

    query_states = query_states.contiguous()
    out = torch.empty_like(query_states)
    dummy_mask = query_states
    mask_ptr = attention_mask if attention_mask is not None else dummy_mask
    block_n = 64
    try:
        if int(q_len) == 1:
            _polar_compressed_attention_kernel[(int(q_len), int(num_heads))](
                query_states,
                key_cache.radius_q.contiguous(),
                key_cache.theta_q.contiguous(),
                key_cache.radius_scale.contiguous(),
                value_cache.radius_q.contiguous(),
                value_cache.theta_q.contiguous(),
                value_cache.radius_scale.contiguous(),
                key_cache.theta_cos_lut,
                key_cache.theta_sin_lut,
                mask_ptr,
                out,
                kv_len,
                max_length=int(key_cache.max_length),
                q_len=int(q_len),
                num_key_value_groups=int(num_key_value_groups),
                pair_dim=pair_dim,
                head_dim=int(head_dim),
                inv_radius_levels=float(key_cache._inv_radius_levels),
                sm_scale=float(sm_scale),
                has_mask=attention_mask is not None,
                BLOCK_N=block_n,
                BLOCK_P=block_p,
                num_warps=4,
            )
        else:
            block_m = 16
            _polar_compressed_attention_block_kernel[
                (triton.cdiv(int(q_len), block_m), int(num_heads))
            ](
                query_states,
                key_cache.radius_q.contiguous(),
                key_cache.theta_q.contiguous(),
                key_cache.radius_scale.contiguous(),
                value_cache.radius_q.contiguous(),
                value_cache.theta_q.contiguous(),
                value_cache.radius_scale.contiguous(),
                key_cache.theta_cos_lut,
                key_cache.theta_sin_lut,
                mask_ptr,
                out,
                kv_len,
                max_length=int(key_cache.max_length),
                q_len=int(q_len),
                num_key_value_groups=int(num_key_value_groups),
                pair_dim=pair_dim,
                head_dim=int(head_dim),
                inv_radius_levels=float(key_cache._inv_radius_levels),
                sm_scale=float(sm_scale),
                has_mask=attention_mask is not None,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_P=block_p,
                num_warps=4,
            )
    except RuntimeError:
        return None
    return out


def compressed_kv_attention_turbo_vq_triton(
    query_states,
    key_cache,
    value_cache,
    attention_mask,
    num_key_value_groups,
    sm_scale,
):
    if not TRITON_AVAILABLE or not _is_cuda_tensor(query_states):
        return None
    if query_states.dim() != 4 or query_states.shape[0] != 1:
        return None
    if getattr(key_cache, "is_hybrid_turbo_vq", False) or getattr(value_cache, "is_hybrid_turbo_vq", False):
        return None
    if getattr(key_cache, "dequant_data", None) is not None:
        return None
    if getattr(value_cache, "dequant_data", None) is not None:
        return None

    key_tensors = (
        getattr(key_cache, "q_idx", None),
        getattr(key_cache, "scale", None),
        getattr(key_cache, "codebook", None),
        getattr(key_cache, "rotation_t", None),
        getattr(key_cache, "residual_sign_packed", None),
        getattr(key_cache, "residual_norm", None),
        getattr(key_cache, "residual_proj", None),
    )
    value_tensors = (
        getattr(value_cache, "q_idx", None),
        getattr(value_cache, "scale", None),
        getattr(value_cache, "codebook", None),
        getattr(value_cache, "rotation_t", None),
    )
    if not all(_is_cuda_tensor(tensor) for tensor in key_tensors + value_tensors):
        return None

    _, num_heads, q_len, head_dim = query_states.shape
    kv_len = int(key_cache.current_length.item())
    if kv_len <= 0:
        return None
    if head_dim != int(key_cache.head_dim) or head_dim != int(value_cache.head_dim):
        return None
    if int(key_cache.num_heads) != int(value_cache.num_heads):
        return None
    if int(key_cache.num_heads) * int(num_key_value_groups) != int(num_heads):
        return None

    residual_dim = int(getattr(key_cache, "residual_dim", 0))
    residual_packed_dim = int(getattr(key_cache, "residual_packed_dim", 0))
    if residual_dim <= 0 or residual_packed_dim <= 0:
        return None
    if int(key_cache.residual_sign_packed.shape[-1]) != residual_packed_dim:
        return None
    if attention_mask is not None:
        if not _is_cuda_tensor(attention_mask) or attention_mask.shape != (1, 1, q_len, kv_len):
            return None
        attention_mask = attention_mask.contiguous()

    block_d = triton.next_power_of_2(int(head_dim))
    block_r = triton.next_power_of_2(residual_dim)
    if block_d > 128 or block_r > 256:
        return None

    query_stride_h = int(query_states.stride(1))
    query_stride_q = int(query_states.stride(2))
    query_stride_d = int(query_states.stride(3))
    output_getter = getattr(key_cache, "get_attention_output", None)
    if output_getter is not None:
        out = output_getter(query_states)
    else:
        out = torch.empty(
            tuple(query_states.shape),
            dtype=query_states.dtype,
            device=query_states.device,
        )
    dummy_mask = query_states
    mask_ptr = attention_mask if attention_mask is not None else dummy_mask

    try:
        if int(q_len) == 1:
            # Decode is the dominant autoregressive path. A vector kernel avoids
            # computing 15 masked query rows just to satisfy tl.dot tile sizes.
            block_n = 32 if block_d >= 128 or block_r > 128 else 64
            _turbo_vq_compressed_attention_decode_kernel[(int(num_heads),)](
                query_states,
                key_cache.q_idx.contiguous(),
                key_cache.scale.contiguous(),
                key_cache.codebook.contiguous(),
                key_cache.rotation_t.contiguous(),
                key_cache.residual_sign_packed.contiguous(),
                key_cache.residual_norm.contiguous(),
                key_cache.residual_proj.contiguous(),
                value_cache.q_idx.contiguous(),
                value_cache.scale.contiguous(),
                value_cache.codebook.contiguous(),
                value_cache.rotation_t.contiguous(),
                mask_ptr,
                out,
                kv_len,
                query_stride_h,
                query_stride_q,
                query_stride_d,
                max_length=int(key_cache.max_length),
                num_key_value_groups=int(num_key_value_groups),
                head_dim=int(head_dim),
                residual_dim=residual_dim,
                residual_packed_dim=residual_packed_dim,
                residual_coeff=float(key_cache.residual_coeff),
                sm_scale=float(sm_scale),
                has_mask=attention_mask is not None,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                BLOCK_R=block_r,
                num_warps=4,
            )
        else:
            # BLOCK_N=32 keeps the residual sign tile small enough for 128/256-bit
            # QJL sketches while still streaming KV in useful chunks.
            block_m = 16
            block_n = 16 if block_r > 128 else 32
            _turbo_vq_compressed_attention_block_kernel[
                (triton.cdiv(int(q_len), block_m), int(num_heads))
            ](
                query_states,
                key_cache.q_idx.contiguous(),
                key_cache.scale.contiguous(),
                key_cache.codebook.contiguous(),
                key_cache.rotation_t.contiguous(),
                key_cache.residual_sign_packed.contiguous(),
                key_cache.residual_norm.contiguous(),
                key_cache.residual_proj.contiguous(),
                value_cache.q_idx.contiguous(),
                value_cache.scale.contiguous(),
                value_cache.codebook.contiguous(),
                value_cache.rotation_t.contiguous(),
                mask_ptr,
                out,
                kv_len,
                query_stride_h,
                query_stride_q,
                query_stride_d,
                max_length=int(key_cache.max_length),
                q_len=int(q_len),
                num_key_value_groups=int(num_key_value_groups),
                head_dim=int(head_dim),
                residual_dim=residual_dim,
                residual_packed_dim=residual_packed_dim,
                residual_coeff=float(key_cache.residual_coeff),
                sm_scale=float(sm_scale),
                has_mask=attention_mask is not None,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                BLOCK_R=block_r,
                num_warps=4,
            )
    except Exception as exc:
        _debug_triton_failure("compressed_kv_attention_turbo_vq_triton", exc)
        return None
    return out


def hybrid_kv_attention_turbo_vq_triton(
    query_states,
    key_cache,
    value_cache,
    attention_mask,
    num_key_value_groups,
    sm_scale,
):
    if not TRITON_AVAILABLE or not _is_cuda_tensor(query_states):
        return None
    if not (
        getattr(key_cache, "is_hybrid_turbo_vq", False)
        and getattr(value_cache, "is_hybrid_turbo_vq", False)
    ):
        return None
    if query_states.dim() != 4 or query_states.shape[0] != 1:
        return None

    _, num_heads, q_len, head_dim = query_states.shape
    if int(q_len) != 1:
        return None

    kv_len = int(key_cache.current_length.item())
    old_len = int(key_cache.old_length(kv_len))
    if kv_len <= 0 or old_len <= 0:
        return None
    if int(key_cache.hot_capacity) != int(value_cache.hot_capacity):
        return None
    if head_dim != int(key_cache.head_dim) or head_dim != int(value_cache.head_dim):
        return None
    if int(key_cache.num_heads) != int(value_cache.num_heads):
        return None
    if int(key_cache.num_heads) * int(num_key_value_groups) != int(num_heads):
        return None

    key_tensors = (
        getattr(key_cache, "q_idx", None),
        getattr(key_cache, "scale", None),
        getattr(key_cache, "codebook", None),
        getattr(key_cache, "rotation_t", None),
        getattr(key_cache, "residual_sign_packed", None),
        getattr(key_cache, "residual_norm", None),
        getattr(key_cache, "residual_proj", None),
        getattr(key_cache, "hot_data", None),
    )
    value_tensors = (
        getattr(value_cache, "q_idx", None),
        getattr(value_cache, "scale", None),
        getattr(value_cache, "codebook", None),
        getattr(value_cache, "rotation_t", None),
        getattr(value_cache, "hot_data", None),
    )
    if not all(_is_cuda_tensor(tensor) for tensor in key_tensors + value_tensors):
        return None

    residual_dim = int(getattr(key_cache, "residual_dim", 0))
    residual_packed_dim = int(getattr(key_cache, "residual_packed_dim", 0))
    if residual_dim <= 0 or residual_packed_dim <= 0:
        return None
    if attention_mask is not None:
        if not _is_cuda_tensor(attention_mask) or attention_mask.shape != (1, 1, 1, kv_len):
            return None
        attention_mask = attention_mask.contiguous()

    block_d = triton.next_power_of_2(int(head_dim))
    block_r = triton.next_power_of_2(residual_dim)
    if block_d > 128 or block_r > 256:
        return None

    query_stride_h = int(query_states.stride(1))
    query_stride_q = int(query_states.stride(2))
    query_stride_d = int(query_states.stride(3))
    output_getter = getattr(key_cache, "get_attention_output", None)
    if output_getter is not None:
        out = output_getter(query_states)
    else:
        out = torch.empty(
            tuple(query_states.shape),
            dtype=query_states.dtype,
            device=query_states.device,
        )
    dummy_mask = query_states
    mask_ptr = attention_mask if attention_mask is not None else dummy_mask
    block_n = 16 if block_d >= 128 else (32 if block_r > 128 else 64)

    try:
        _turbo_vq_hybrid_attention_decode_kernel[(int(num_heads),)](
            query_states,
            key_cache.q_idx.contiguous(),
            key_cache.scale.contiguous(),
            key_cache.codebook.contiguous(),
            key_cache.rotation_t.contiguous(),
            key_cache.residual_sign_packed.contiguous(),
            key_cache.residual_norm.contiguous(),
            key_cache.residual_proj.contiguous(),
            value_cache.q_idx.contiguous(),
            value_cache.scale.contiguous(),
            value_cache.codebook.contiguous(),
            value_cache.rotation_t.contiguous(),
            key_cache.hot_data.contiguous(),
            value_cache.hot_data.contiguous(),
            mask_ptr,
            out,
            kv_len,
            old_len,
            query_stride_h,
            query_stride_q,
            query_stride_d,
            max_length=int(key_cache.max_length),
            hot_capacity=int(key_cache.hot_capacity),
            num_key_value_groups=int(num_key_value_groups),
            head_dim=int(head_dim),
            residual_dim=residual_dim,
            residual_packed_dim=residual_packed_dim,
            residual_coeff=float(key_cache.residual_coeff),
            sm_scale=float(sm_scale),
            has_mask=attention_mask is not None,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            BLOCK_R=block_r,
            num_warps=4,
        )
    except Exception as exc:
        _debug_triton_failure("hybrid_kv_attention_turbo_vq_triton", exc)
        return None
    return out
