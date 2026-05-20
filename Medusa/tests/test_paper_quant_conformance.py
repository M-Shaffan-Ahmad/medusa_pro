import math
import unittest

import torch
import torch.nn.functional as F

from medusa.model.kv_cache import (
    HybridOutlierTurboVQKVCache,
    HybridTurboVQKVCache,
    PolarQuantizedKVCache,
    TurboQuantizedKVCache,
    turbo_vq_attention_with_qjl_residual,
)
from medusa.model.triton_kernels import (
    compressed_kv_attention_polar_triton,
    hybrid_kv_attention_turbo_vq_triton,
)


def normalized_randn(*shape, device=torch.device("cpu"), dtype=torch.float32):
    values = torch.randn(*shape, device=device, dtype=dtype)
    return F.normalize(values, dim=-1)


class PaperQuantConformanceTest(unittest.TestCase):
    def test_turboquant_mse_distortion_decreases_with_bits(self):
        torch.manual_seed(7)
        vectors = normalized_randn(1, 1, 128, 32)
        rel_errors = []

        for bits in (2, 3, 4):
            cache = TurboQuantizedKVCache(
                batch_size=1,
                num_heads=1,
                max_length=128,
                head_dim=32,
                device=torch.device("cpu"),
                dtype=torch.float32,
                current_length=torch.zeros((), dtype=torch.long),
                bits=bits,
                residual_dim=0,
                runtime_dequant_cache=False,
            )
            cache.cat(vectors)
            decoded = cache._decode_range(0, vectors.shape[2])
            rel_error = (
                (vectors - decoded).pow(2).sum(dim=-1)
                / vectors.pow(2).sum(dim=-1).clamp_min(1e-8)
            ).mean()
            rel_errors.append(float(rel_error.item()))

        self.assertLess(rel_errors[1], rel_errors[0])
        self.assertLess(rel_errors[2], rel_errors[1])
        self.assertLess(rel_errors[2], 0.02)

    def test_turboquantprod_qjl_residual_improves_inner_products(self):
        torch.manual_seed(11)
        keys = normalized_randn(1, 1, 96, 32)
        queries = normalized_randn(1, 1, 64, 32)
        key_cache = TurboQuantizedKVCache(
            batch_size=1,
            num_heads=1,
            max_length=96,
            head_dim=32,
            device=torch.device("cpu"),
            dtype=torch.float32,
            current_length=torch.zeros((), dtype=torch.long),
            bits=2,
            residual_dim=-1,
            residual_scale=1.0,
            runtime_dequant_cache=False,
        )
        key_cache.cat(keys)

        decoded_keys = key_cache._decode_range(0, keys.shape[2])
        true_scores = torch.matmul(queries, keys.transpose(2, 3))
        mse_scores = torch.matmul(queries, decoded_keys.transpose(2, 3))

        q_residual_proj = queries @ key_cache.residual_proj
        residual_sign = key_cache._unpack_residual_sign_range(0, keys.shape[2])
        residual_norm = key_cache.residual_norm[:, :, : keys.shape[2], 0].to(torch.float32)
        residual_inner = torch.einsum("bhqm,bhkm->bhqk", q_residual_proj, residual_sign)
        qjl_scores = mse_scores + (
            key_cache.residual_coeff * residual_inner * residual_norm.unsqueeze(2)
        )

        mse_error = (mse_scores - true_scores).pow(2).mean()
        qjl_error = (qjl_scores - true_scores).pow(2).mean()
        qjl_bias = (qjl_scores - true_scores).mean().abs()

        self.assertLess(float(qjl_error.item()), float(mse_error.item()) * 0.8)
        self.assertLess(float(qjl_bias.item()), 0.01)

    def test_recursive_polarquant_reconstruction_improves_with_bits(self):
        torch.manual_seed(13)
        vectors = torch.randn(1, 1, 32, 16)
        rel_errors = []

        for first_bits, other_bits in ((4, 2), (8, 8)):
            cache = PolarQuantizedKVCache(
                batch_size=1,
                num_heads=1,
                max_length=32,
                head_dim=16,
                device=torch.device("cpu"),
                dtype=torch.float32,
                current_length=torch.zeros((), dtype=torch.long),
                first_level_bits=first_bits,
                other_level_bits=other_bits,
                polar_levels=4,
                runtime_dequant_cache=False,
            )
            self.assertEqual(cache.level_dims, [8, 4, 2, 1])
            self.assertEqual(cache.final_dim, 1)
            cache.cat(vectors)
            decoded = cache._decode_range(0, vectors.shape[2])
            rel_error = (
                (vectors - decoded).pow(2).sum(dim=-1)
                / vectors.pow(2).sum(dim=-1).clamp_min(1e-8)
            ).mean()
            rel_errors.append(float(rel_error.item()))

        self.assertLess(rel_errors[1], rel_errors[0] * 0.1)
        self.assertLess(rel_errors[1], 1e-3)

    def test_paper_hot_tail_keeps_streamed_tokens_exact(self):
        torch.manual_seed(19)
        prompt = torch.randn(1, 1, 8, 16)
        streamed = torch.randn(1, 1, 3, 16)

        caches = [
            HybridTurboVQKVCache(
                batch_size=1,
                num_heads=1,
                max_length=16,
                head_dim=16,
                device=torch.device("cpu"),
                dtype=torch.float32,
                current_length=torch.zeros((), dtype=torch.long),
                bits=4,
                residual_dim=0,
                hot_window=4,
            ),
            PolarQuantizedKVCache(
                batch_size=1,
                num_heads=1,
                max_length=16,
                head_dim=16,
                device=torch.device("cpu"),
                dtype=torch.float32,
                current_length=torch.zeros((), dtype=torch.long),
                first_level_bits=4,
                other_level_bits=2,
                polar_levels=4,
                runtime_dequant_cache=False,
                hot_window=4,
            ),
        ]

        for cache in caches:
            cache.cat(prompt)
            cache.cat(streamed)
            torch.testing.assert_close(cache._decode_range(8, 11), streamed)

            accepted = torch.tensor([8, 10], dtype=torch.long)
            cache.copy(accepted, prev_length=8)
            expected = streamed[:, :, [0, 2]]
            torch.testing.assert_close(cache._decode_range(8, 10), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton fused attention parity")
    def test_turboquant_hot_tail_block_attention_matches_reference(self):
        torch.manual_seed(23)
        device = torch.device("cuda")
        head_dim = 16
        q_len = 3
        kv_len = 8
        key_cache = HybridTurboVQKVCache(
            batch_size=1,
            num_heads=2,
            max_length=16,
            head_dim=head_dim,
            device=device,
            dtype=torch.float16,
            current_length=torch.zeros((), dtype=torch.long),
            bits=3,
            residual_dim=-1,
            residual_scale=1.0,
            hot_window=4,
        )
        value_cache = HybridTurboVQKVCache(
            batch_size=1,
            num_heads=2,
            max_length=16,
            head_dim=head_dim,
            device=device,
            dtype=torch.float16,
            current_length=torch.zeros((), dtype=torch.long),
            bits=4,
            residual_dim=0,
            hot_window=4,
        )
        keys = torch.randn(1, 2, kv_len, head_dim, device=device, dtype=torch.float16)
        values = torch.randn(1, 2, kv_len, head_dim, device=device, dtype=torch.float16)
        key_cache.cat(keys[:, :, :5])
        value_cache.cat(values[:, :, :5])
        key_cache.cat(keys[:, :, 5:])
        value_cache.cat(values[:, :, 5:])
        query = torch.randn(1, 2, q_len, head_dim, device=device, dtype=torch.float16)

        mask = torch.zeros(1, 1, q_len, kv_len, device=device, dtype=torch.float32)
        for idx in range(q_len):
            mask[:, :, idx, kv_len - q_len + idx + 1 :] = -float("inf")

        fused = hybrid_kv_attention_turbo_vq_triton(
            query,
            key_cache,
            value_cache,
            mask,
            num_key_value_groups=1,
            sm_scale=1.0 / math.sqrt(float(head_dim)),
        )
        self.assertIsNotNone(fused)

        old_len = key_cache.old_length(kv_len)
        decoded_keys = key_cache._decode_range(0, kv_len).to(torch.float32)
        decoded_values = value_cache._decode_range(0, kv_len).to(torch.float32)
        scores = query.to(torch.float32) @ decoded_keys.transpose(-1, -2)
        residual_sign = key_cache.compressed_cache._unpack_residual_sign_range(0, old_len)
        residual_norm = key_cache.residual_norm[:, :, :old_len, 0].to(torch.float32)
        q_residual_proj = query.to(torch.float32) @ key_cache.residual_proj
        residual_inner = torch.einsum(
            "bhqm,bhkm->bhqk",
            q_residual_proj,
            residual_sign,
        )
        scores[:, :, :, :old_len] += (
            key_cache.residual_coeff * residual_inner * residual_norm.unsqueeze(2)
        )
        scores = (scores / math.sqrt(float(head_dim))) + mask
        expected = torch.softmax(scores, dim=-1) @ decoded_values
        torch.testing.assert_close(fused.to(torch.float32), expected, atol=3e-2, rtol=3e-2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton fused attention parity")
    def test_outlier_turboquant_hot_tail_block_attention_matches_reference(self):
        torch.manual_seed(31)
        device = torch.device("cuda")
        head_dim = 16
        q_len = 3
        kv_len = 8
        key_cache = HybridOutlierTurboVQKVCache(
            batch_size=1,
            num_heads=2,
            max_length=16,
            head_dim=head_dim,
            device=device,
            dtype=torch.float16,
            current_length=torch.zeros((), dtype=torch.long),
            regular_bits=2,
            outlier_bits=3,
            n_outlier=4,
            residual_dim=-1,
            residual_scale=1.0,
            hot_window=4,
            outlier_idx=torch.tensor([1, 5, 9, 13], device=device),
        )
        value_cache = HybridOutlierTurboVQKVCache(
            batch_size=1,
            num_heads=2,
            max_length=16,
            head_dim=head_dim,
            device=device,
            dtype=torch.float16,
            current_length=torch.zeros((), dtype=torch.long),
            regular_bits=3,
            outlier_bits=4,
            n_outlier=4,
            residual_dim=0,
            hot_window=4,
            outlier_idx=torch.tensor([1, 5, 9, 13], device=device),
        )
        keys = torch.randn(1, 2, kv_len, head_dim, device=device, dtype=torch.float16)
        values = torch.randn(1, 2, kv_len, head_dim, device=device, dtype=torch.float16)
        key_cache.cat(keys[:, :, :5])
        value_cache.cat(values[:, :, :5])
        key_cache.cat(keys[:, :, 5:])
        value_cache.cat(values[:, :, 5:])
        query = torch.randn(1, 2, q_len, head_dim, device=device, dtype=torch.float16)

        mask = torch.zeros(1, 1, q_len, kv_len, device=device, dtype=torch.float32)
        for idx in range(q_len):
            mask[:, :, idx, kv_len - q_len + idx + 1 :] = -float("inf")

        fused = hybrid_kv_attention_turbo_vq_triton(
            query,
            key_cache,
            value_cache,
            mask,
            num_key_value_groups=1,
            sm_scale=1.0 / math.sqrt(float(head_dim)),
        )
        self.assertIsNotNone(fused)

        expected = turbo_vq_attention_with_qjl_residual(
            query,
            key_cache,
            value_cache,
            mask,
            num_key_value_groups=1,
            head_dim=head_dim,
        )
        torch.testing.assert_close(fused.to(torch.float32), expected.to(torch.float32), atol=4e-2, rtol=4e-2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton fused attention parity")
    def test_polarquant_hot_tail_block_attention_matches_reference(self):
        torch.manual_seed(29)
        device = torch.device("cuda")
        head_dim = 16
        q_len = 3
        kv_len = 8
        key_cache = PolarQuantizedKVCache(
            batch_size=1,
            num_heads=2,
            max_length=16,
            head_dim=head_dim,
            device=device,
            dtype=torch.float16,
            current_length=torch.zeros((), dtype=torch.long),
            first_level_bits=4,
            other_level_bits=2,
            polar_levels=4,
            runtime_dequant_cache=False,
            hot_window=4,
        )
        value_cache = PolarQuantizedKVCache(
            batch_size=1,
            num_heads=2,
            max_length=16,
            head_dim=head_dim,
            device=device,
            dtype=torch.float16,
            current_length=torch.zeros((), dtype=torch.long),
            first_level_bits=4,
            other_level_bits=2,
            polar_levels=4,
            runtime_dequant_cache=False,
            hot_window=4,
        )
        keys = torch.randn(1, 2, kv_len, head_dim, device=device, dtype=torch.float16)
        values = torch.randn(1, 2, kv_len, head_dim, device=device, dtype=torch.float16)
        key_cache.cat(keys[:, :, :5])
        value_cache.cat(values[:, :, :5])
        key_cache.cat(keys[:, :, 5:])
        value_cache.cat(values[:, :, 5:])
        query = torch.randn(1, 2, q_len, head_dim, device=device, dtype=torch.float16)

        mask = torch.zeros(1, 1, q_len, kv_len, device=device, dtype=torch.float32)
        for idx in range(q_len):
            mask[:, :, idx, kv_len - q_len + idx + 1 :] = -float("inf")

        fused = compressed_kv_attention_polar_triton(
            query,
            key_cache,
            value_cache,
            mask,
            num_key_value_groups=1,
            sm_scale=1.0 / math.sqrt(float(head_dim)),
        )
        self.assertIsNotNone(fused)

        decoded_keys = key_cache._decode_range(0, kv_len).to(torch.float32)
        decoded_values = value_cache._decode_range(0, kv_len).to(torch.float32)
        scores = (query.to(torch.float32) @ decoded_keys.transpose(-1, -2)) / math.sqrt(
            float(head_dim)
        )
        expected = torch.softmax(scores + mask, dim=-1) @ decoded_values
        torch.testing.assert_close(fused.to(torch.float32), expected, atol=3e-2, rtol=3e-2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton fused attention parity")
    def test_recursive_polarquant_fused_attention_matches_decoded_reference(self):
        torch.manual_seed(17)
        device = torch.device("cuda")
        head_dim = 16
        key_cache = PolarQuantizedKVCache(
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
        value_cache = PolarQuantizedKVCache(
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
        key_cache.cat(torch.randn(1, 2, 4, head_dim, device=device, dtype=torch.float16))
        value_cache.cat(torch.randn(1, 2, 4, head_dim, device=device, dtype=torch.float16))
        query = torch.randn(1, 2, 1, head_dim, device=device, dtype=torch.float16)

        fused = compressed_kv_attention_polar_triton(
            query,
            key_cache,
            value_cache,
            attention_mask=None,
            num_key_value_groups=1,
            sm_scale=1.0 / math.sqrt(float(head_dim)),
        )
        self.assertIsNotNone(fused)

        key_decoded = key_cache._decode_range(0, 4).to(torch.float32)
        value_decoded = value_cache._decode_range(0, 4).to(torch.float32)
        scores = (query.to(torch.float32) @ key_decoded.transpose(-1, -2)) / math.sqrt(
            float(head_dim)
        )
        expected = torch.softmax(scores, dim=-1) @ value_decoded
        torch.testing.assert_close(fused.to(torch.float32), expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
