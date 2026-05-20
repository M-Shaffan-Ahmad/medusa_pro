#!/usr/bin/env python3
"""Smoke-check Medusa Triton kernels.

Without CUDA this verifies Triton imports and wrapper availability. With CUDA
it launches small kernels for the TurboQuant and QJL hot paths and compares
simple outputs against PyTorch/Python references.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from medusa.model import triton_kernels as tk
from medusa.model.kv_cache import (
    PolarQuantizedKVCache,
    TurboQuantizedKVCache,
)


WRAPPER_NAMES = (
    "qjl_path_scores_triton",
    "packed_kv_qjl_node_scores_triton",
    "turbo_qjl_select_paths_triton",
    "node_budget_select_triton",
    "materialize_pruned_medusa_triton",
    "greedy_accept_from_argmax_triton",
    "greedy_tree_posterior_triton",
    "lm_head_argmax_triton",
    "copy_selected_kv_cache_triton",
    "turbo_vq_append_triton",
    "compressed_kv_attention_polar_triton",
    "compressed_kv_attention_turbo_vq_triton",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, name: str, atol=1e-4) -> None:
    require(actual is not None, f"{name} returned None")
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), atol=atol, rtol=1e-4)


def popcount32(value: int) -> int:
    return (int(value) & 0xFFFFFFFF).bit_count()


def signed32(value: int) -> int:
    value = int(value) & 0xFFFFFFFF
    if value >= 0x80000000:
        return value - 0x100000000
    return value


def check_imports() -> None:
    require(tk.TRITON_AVAILABLE, "Triton is not importable in this Python environment.")
    missing = [name for name in WRAPPER_NAMES if not callable(getattr(tk, name, None))]
    require(not missing, f"Missing Triton wrapper(s): {', '.join(missing)}")
    print(f"triton: available ({getattr(tk.triton, '__version__', 'unknown')})")
    print(f"torch: {torch.__version__} cuda_build={torch.version.cuda}")


def check_qjl_path_scores(device: torch.device) -> None:
    sketch_dim = 32
    q_proj = torch.randn(sketch_dim, device=device)
    sign_cache = torch.sign(torch.randn(8, sketch_dim, device=device)).to(torch.int8)
    sign_cache[sign_cache == 0] = 1
    norm_cache = torch.rand(8, device=device, dtype=torch.float16).add_(0.5)
    candidates = torch.tensor([[1, 2, 3], [3, 4, 0], [5, 6, 7]], device=device)
    valid_mask = torch.tensor(
        [[True, True, False], [True, True, True], [True, False, False]],
        device=device,
    )
    coeff = math.sqrt(math.pi / 2.0) / sketch_dim

    actual = tk.qjl_path_scores_triton(
        q_proj,
        sign_cache,
        norm_cache,
        candidates,
        valid_mask,
        coeff,
        sketch_dim,
    )

    q = q_proj.view(1, 1, -1)
    token_scores = coeff * norm_cache[candidates].float() * (sign_cache[candidates].float() * q).sum(dim=-1)
    mask_f = valid_mask.float()
    expected = (token_scores * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
    assert_close(actual, expected, "qjl_path_scores_triton")


def check_packed_kv_qjl(device: torch.device) -> None:
    query_bits = torch.tensor(
        [
            [[signed32(0x00000000), signed32(0xFFFFFFFF)], [signed32(0x55555555), signed32(0xAAAAAAAA)]],
            [[signed32(0xFFFFFFFF), signed32(0x00000000)], [signed32(0x0F0F0F0F), signed32(0xF0F0F0F0)]],
        ],
        device=device,
        dtype=torch.int32,
    )
    key_bits = torch.tensor(
        [
            [
                [signed32(0x00000000), signed32(0xFFFFFFFF)],
                [signed32(0xFFFFFFFF), signed32(0x00000000)],
                [signed32(0x55555555), signed32(0xAAAAAAAA)],
            ],
            [
                [signed32(0x55555555), signed32(0xAAAAAAAA)],
                [signed32(0x0F0F0F0F), signed32(0xF0F0F0F0)],
                [signed32(0xFFFFFFFF), signed32(0x00000000)],
            ],
        ],
        device=device,
        dtype=torch.int32,
    )
    actual = tk.packed_kv_qjl_node_scores_triton(query_bits, key_bits, kv_len=3, block_k=2)

    q_cpu = query_bits.cpu()
    k_cpu = key_bits.cpu()
    expected = []
    for node in range(q_cpu.shape[0]):
        score = 0
        for head in range(q_cpu.shape[1]):
            for pos in range(3):
                for word in range(q_cpu.shape[2]):
                    score += popcount32(~(int(q_cpu[node, head, word]) ^ int(k_cpu[head, pos, word])))
        expected.append(score)
    expected = torch.tensor(expected, dtype=torch.float32, device=device)
    assert_close(actual, expected, "packed_kv_qjl_node_scores_triton")


def check_planner_kernels(device: torch.device) -> None:
    selected = tk.node_budget_select_triton(
        torch.tensor([0.1, 0.9, 0.3], device=device),
        torch.tensor([[0, 1, -1], [0, 2, 3], [0, 4, -1]], device=device),
        torch.tensor([0], device=device),
        node_budget=3,
        min_keep=1,
        max_keep=2,
        full_node_count=5,
    )
    require(selected is not None and selected.numel() > 0, "node_budget_select_triton returned no selection")
    require(int(selected[0].item()) == 0, "node_budget_select_triton did not preserve mandatory path")

    full_tree = torch.tensor([[10, 20, 30, 40]], device=device)
    materialized = tk.materialize_pruned_medusa_triton(
        full_tree,
        torch.tensor([0, 2], device=device),
        torch.tensor([0, 1, -1], device=device),
    )
    require(materialized is not None, "materialize_pruned_medusa_triton returned None")
    pruned_tree, pruned_candidates = materialized
    assert_close(pruned_tree, torch.tensor([[10, 30]], device=device), "materialize_pruned tree", atol=0)
    assert_close(pruned_candidates, torch.tensor([10, 30, 0], device=device), "materialize_pruned candidates", atol=0)

    q_proj = torch.randn(32, device=device)
    sign_cache = torch.sign(torch.randn(8, 32, device=device)).to(torch.int8)
    sign_cache[sign_cache == 0] = 1
    norm_cache = torch.ones(8, device=device, dtype=torch.float16)
    candidates = torch.tensor([[1, 2, -1], [3, 4, 5], [6, -1, -1]], device=device)
    valid_mask = candidates >= 0
    safe_candidates = candidates.clamp_min(0)
    plan = tk.turbo_qjl_select_paths_triton(
        q_proj,
        sign_cache,
        norm_cache,
        safe_candidates,
        valid_mask,
        torch.tensor([0.8, 0.4, 0.2], device=device),
        torch.tensor([0], device=device),
        math.sqrt(math.pi / 2.0) / 32,
        32,
        keep_target=2,
        min_keep=1,
        max_keep=2,
        margin_scale=0.5,
    )
    require(plan is not None, "turbo_qjl_select_paths_triton returned None")
    approx, selected_paths, _ = plan
    require(approx.shape == (3,), "turbo_qjl_select_paths_triton returned bad approx shape")
    require(selected_paths.numel() > 0, "turbo_qjl_select_paths_triton selected no paths")


def check_greedy_verifier_kernels(device: torch.device) -> None:
    node_argmax = torch.tensor([5, 6, 7], device=device)
    candidates = torch.tensor([[0, 5, 6], [0, 5, 8]], device=device)
    retrieve_indices = torch.tensor([[0, 1, 2], [0, 1, 2]], device=device)
    path_lengths = torch.tensor([2, 2], device=device)

    best, accept = tk.greedy_accept_from_argmax_triton(
        node_argmax,
        candidates,
        retrieve_indices,
        path_lengths,
    )
    require(int(best.item()) == 0 and int(accept.item()) == 2, "greedy_accept_from_argmax_triton mismatch")

    logits = torch.zeros(3, 10, device=device)
    logits[0, 5] = 10
    logits[1, 6] = 10
    logits[2, 7] = 10
    best, accept = tk.greedy_tree_posterior_triton(
        logits,
        candidates,
        retrieve_indices,
        path_lengths,
    )
    require(int(best.item()) == 0 and int(accept.item()) == 2, "greedy_tree_posterior_triton mismatch")

    hidden = torch.randn(4, 8, device=device)
    weight = torch.randn(16, 8, device=device)
    actual = tk.lm_head_argmax_triton(hidden, weight)
    expected = torch.argmax(hidden @ weight.t(), dim=-1).to(torch.int32)
    assert_close(actual, expected, "lm_head_argmax_triton", atol=0)


def check_copy_kernel(device: torch.device) -> None:
    data = torch.arange(2 * 1 * 1 * 8 * 2, device=device, dtype=torch.float16).reshape(2, 1, 1, 8, 2)
    original = data.clone()
    copied = tk.copy_selected_kv_cache_triton(
        data,
        torch.tensor([0, 2, 4], device=device),
        prev_input_len=3,
        copy_start=1,
    )
    require(copied, "copy_selected_kv_cache_triton returned False")
    torch.testing.assert_close(data[:, :, :, 4], original[:, :, :, 2])
    torch.testing.assert_close(data[:, :, :, 5], original[:, :, :, 4])


def check_turbo_quant_kernels(device: torch.device) -> None:
    head_dim = 16
    key_cache = TurboQuantizedKVCache(
        batch_size=1,
        num_heads=2,
        max_length=8,
        head_dim=head_dim,
        device=device,
        dtype=torch.float16,
        current_length=torch.zeros((), dtype=torch.long),
        bits=8,
        residual_dim=32,
        runtime_dequant_cache=False,
    )
    value_cache = TurboQuantizedKVCache(
        batch_size=1,
        num_heads=2,
        max_length=8,
        head_dim=head_dim,
        device=device,
        dtype=torch.float16,
        current_length=torch.zeros((), dtype=torch.long),
        bits=8,
        residual_dim=0,
        runtime_dequant_cache=False,
    )
    key_cache.cat(torch.randn(1, 2, 4, head_dim, device=device, dtype=torch.float16))
    value_cache.cat(torch.randn(1, 2, 4, head_dim, device=device, dtype=torch.float16))
    require(int(key_cache.current_length.item()) == 4, "turbo_vq_append_triton did not advance key cache")
    require(int(value_cache.current_length.item()) == 4, "turbo_vq_append_triton did not advance value cache")

    query = torch.randn(1, 2, 1, head_dim, device=device, dtype=torch.float16)
    out = tk.compressed_kv_attention_turbo_vq_triton(
        query,
        key_cache,
        value_cache,
        attention_mask=None,
        num_key_value_groups=1,
        sm_scale=1.0 / math.sqrt(float(head_dim)),
    )
    require(out is not None and out.shape == query.shape and torch.isfinite(out).all(), "TurboVQ attention Triton failed")

    packed_key = TurboQuantizedKVCache(
        batch_size=1,
        num_heads=2,
        max_length=8,
        head_dim=head_dim,
        device=device,
        dtype=torch.float16,
        current_length=torch.zeros((), dtype=torch.long),
        bits=3,
        residual_dim=32,
        runtime_dequant_cache=False,
    )
    packed_value = TurboQuantizedKVCache(
        batch_size=1,
        num_heads=2,
        max_length=8,
        head_dim=head_dim,
        device=device,
        dtype=torch.float16,
        current_length=torch.zeros((), dtype=torch.long),
        bits=4,
        residual_dim=0,
        runtime_dequant_cache=False,
    )
    packed_key.cat(torch.randn(1, 2, 4, head_dim, device=device, dtype=torch.float16))
    packed_value.cat(torch.randn(1, 2, 4, head_dim, device=device, dtype=torch.float16))
    out = tk.compressed_kv_attention_turbo_vq_triton(
        query,
        packed_key,
        packed_value,
        attention_mask=None,
        num_key_value_groups=1,
        sm_scale=1.0 / math.sqrt(float(head_dim)),
    )
    require(out is not None and out.shape == query.shape and torch.isfinite(out).all(), "Packed TurboVQ attention Triton failed")


def check_polar_kernel(device: torch.device) -> None:
    head_dim = 16
    query = torch.randn(1, 2, 1, head_dim, device=device, dtype=torch.float16)
    recursive_key = PolarQuantizedKVCache(
        batch_size=1,
        num_heads=2,
        max_length=8,
        head_dim=head_dim,
        device=device,
        dtype=torch.float16,
        current_length=torch.zeros((), dtype=torch.long),
        first_level_bits=4,
        other_level_bits=2,
        polar_levels=4,
        runtime_dequant_cache=False,
    )
    recursive_value = PolarQuantizedKVCache(
        batch_size=1,
        num_heads=2,
        max_length=8,
        head_dim=head_dim,
        device=device,
        dtype=torch.float16,
        current_length=torch.zeros((), dtype=torch.long),
        first_level_bits=4,
        other_level_bits=2,
        polar_levels=4,
        runtime_dequant_cache=False,
    )
    recursive_key.cat(torch.randn(1, 2, 4, head_dim, device=device, dtype=torch.float16))
    recursive_value.cat(torch.randn(1, 2, 4, head_dim, device=device, dtype=torch.float16))
    decoded = recursive_key._decode_range(0, 4)
    require(
        decoded.shape == (1, 2, 4, head_dim) and torch.isfinite(decoded).all(),
        "Recursive PolarQuant decode failed",
    )
    recursive_out = tk.compressed_kv_attention_polar_triton(
        query,
        recursive_key,
        recursive_value,
        attention_mask=None,
        num_key_value_groups=1,
        sm_scale=1.0 / math.sqrt(float(head_dim)),
    )
    require(
        recursive_out is not None and recursive_out.shape == query.shape and torch.isfinite(recursive_out).all(),
        "Recursive PolarQuant fused attention Triton failed",
    )
    key_decoded = recursive_key._decode_range(0, 4).to(torch.float32)
    value_decoded = recursive_value._decode_range(0, 4).to(torch.float32)
    expected_scores = (query.to(torch.float32) @ key_decoded.transpose(-1, -2)) / math.sqrt(float(head_dim))
    expected = torch.softmax(expected_scores, dim=-1) @ value_decoded
    assert_close(recursive_out.to(torch.float32), expected, "Recursive PolarQuant fused attention", atol=2e-2)


def check_cuda_kernels() -> None:
    device = torch.device("cuda")
    torch.manual_seed(1234)
    check_qjl_path_scores(device)
    print("ok qjl_path_scores_triton")
    check_packed_kv_qjl(device)
    print("ok packed_kv_qjl_node_scores_triton")
    check_planner_kernels(device)
    print("ok QJL planner/materialization kernels")
    check_greedy_verifier_kernels(device)
    print("ok greedy verifier/lm-head kernels")
    check_copy_kernel(device)
    print("ok copy_selected_kv_cache_triton")
    check_turbo_quant_kernels(device)
    print("ok TurboVQ append/attention kernels")
    check_polar_kernel(device)
    print("ok Polar compressed attention kernel")
    torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit non-zero if no CUDA device is visible.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    check_imports()
    if not torch.cuda.is_available():
        message = "CUDA device is not visible; skipped Triton kernel launches."
        if args.require_cuda:
            print(message, file=sys.stderr)
            return 2
        print(message)
        return 0

    print(f"cuda: {torch.cuda.get_device_name(0)}")
    check_cuda_kernels()
    print("all Triton smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
