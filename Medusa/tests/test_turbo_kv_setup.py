from types import SimpleNamespace
import unittest

import torch

from medusa.model.kv_cache import (
    HybridOutlierTurboVQKVCache,
    HybridTurboVQKVCache,
    KVCache,
    OutlierCalibrationKVCache,
    PolarQuantizedKVCache,
    TurboQuantizedKVCache,
    extract_outlier_calibration_indices,
    initialize_past_key_values,
)
from medusa.model.medusa_model import (
    infer_model_context_window,
    resolve_turbo_kv_cache_plan,
)
from medusa.model.utils import reset_past_key_values, update_inference_inputs_from_tree


class DummyLayer(torch.nn.Module):
    def __init__(self, head_dim=None, num_key_value_heads=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        if head_dim is not None:
            self.self_attn = SimpleNamespace(
                head_dim=int(head_dim),
                num_key_value_heads=int(num_key_value_heads),
            )


class DummyModel:
    def __init__(self, num_layers=3, num_kv_heads=2, hidden_size=16, num_heads=4, actual_head_dim=None):
        self.config = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_key_value_heads=num_kv_heads,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            max_position_embeddings=131072,
        )
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        layer_factory = (
            (lambda: DummyLayer(head_dim=actual_head_dim, num_key_value_heads=num_kv_heads))
            if actual_head_dim is not None
            else DummyLayer
        )
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList(layer_factory() for _ in range(num_layers))
        )


class TurboKVSetupTest(unittest.TestCase):
    def test_infer_model_context_window_ignores_tokenizer_sentinel(self):
        config = SimpleNamespace(max_position_embeddings=131072)
        tokenizer = SimpleNamespace(model_max_length=10**30)

        self.assertEqual(infer_model_context_window(config, tokenizer=tokenizer), 131072)

    def test_infer_model_context_window_prefers_explicit_llama32_limit(self):
        config = SimpleNamespace(
            max_position_embeddings=131072,
            rope_scaling={
                "factor": 32.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        )

        self.assertEqual(infer_model_context_window(config), 131072)

    def test_resolve_turbo_kv_cache_plan_can_use_llama32_context_window(self):
        model = DummyModel()

        plan = resolve_turbo_kv_cache_plan(
            model.config,
            input_length=8192,
            max_steps=8,
            max_path_depth=4,
            turbo_kv_max_length=2048,
            turbo_kv_use_model_context=True,
            packed_kv_qjl_requested=True,
            turbo_kv_qjl_min_kv_len=16384,
        )

        self.assertEqual(plan["model_context_window"], 131072)
        self.assertEqual(plan["effective_kv_max_length"], 131072)
        self.assertEqual(plan["required_kv_len"], 8240)
        self.assertIs(plan["use_packed_kv_qjl"], True)

    def test_resolve_turbo_kv_cache_plan_reserves_tree_verification_nodes(self):
        model = DummyModel()

        plan = resolve_turbo_kv_cache_plan(
            model.config,
            input_length=8120,
            max_steps=8,
            max_path_depth=4,
            tree_node_count=64,
            turbo_kv_max_length=8192,
        )

        self.assertEqual(plan["required_kv_len"], 8232)
        self.assertEqual(plan["effective_kv_max_length"], 8232)
        self.assertEqual(plan["tree_node_count"], 64)

    def test_packed_qjl_sidecar_is_allocated_only_on_requested_key_layer(self):
        model = DummyModel(num_layers=3)

        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
            model,
            safe_max_length=32,
            packed_qjl_sketch_dim=64,
            packed_qjl_layer=-1,
        )

        self.assertEqual(tuple(past_key_values_data.shape), (6, 1, 2, 32, 4))
        self.assertEqual(tuple(current_length_data.shape), (6,))
        for layer_idx, (key_cache, value_cache) in enumerate(past_key_values):
            self.assertIsInstance(key_cache, KVCache)
            self.assertIsInstance(value_cache, KVCache)
            if layer_idx == 2:
                self.assertEqual(tuple(key_cache.qjl_bits.shape), (1, 2, 32, 2))
                self.assertEqual(key_cache.qjl_sketch_dim, 64)
            else:
                self.assertIsNone(key_cache.qjl_bits)
            self.assertIsNone(value_cache.qjl_bits)

        key_cache = past_key_values[2][0]
        key_cache.cat(torch.randn(1, 2, 3, 4))
        self.assertEqual(int(key_cache.current_length.item()), 3)
        self.assertEqual(tuple(key_cache.qjl_bits[:, :, :3].shape), (1, 2, 3, 2))

        key_cache.copy(torch.tensor([0, 2], dtype=torch.long), prev_length=3)
        self.assertEqual(int(key_cache.current_length.item()), 5)

    def test_turbo_vq_subbyte_indices_are_bit_packed(self):
        cache = TurboQuantizedKVCache(
            batch_size=1,
            num_heads=2,
            max_length=5,
            head_dim=7,
            device=torch.device("cpu"),
            dtype=torch.float32,
            current_length=torch.zeros((), dtype=torch.long),
            bits=3,
            residual_dim=0,
            runtime_dequant_cache=False,
        )

        raw = (
            torch.arange(1 * 2 * 3 * 7, dtype=torch.uint8)
            .reshape(1, 2, 3, 7)
            .remainder(cache.levels)
        )
        packed = cache._pack_q_idx(raw)
        self.assertEqual(tuple(packed.shape), (1, 2, 3, 3))

        cache.q_idx[:, :, :3].copy_(packed)
        unpacked = cache._unpack_q_idx_range(0, 3).to(torch.uint8)
        torch.testing.assert_close(unpacked, raw, atol=0, rtol=0)

    def test_turbo_prod_negative_residual_dim_uses_full_head(self):
        cache = TurboQuantizedKVCache(
            batch_size=1,
            num_heads=1,
            max_length=4,
            head_dim=6,
            device=torch.device("cpu"),
            dtype=torch.float32,
            current_length=torch.zeros((), dtype=torch.long),
            bits=3,
            residual_dim=-1,
            runtime_dequant_cache=False,
        )

        self.assertEqual(cache.residual_dim, 6)
        self.assertEqual(cache.residual_packed_dim, 1)

    def test_turboquant_prod_cache_initializes_from_factory(self):
        model = DummyModel(num_layers=1)

        past_key_values, past_key_values_data, _ = initialize_past_key_values(
            model,
            safe_max_length=8,
            turbo_quant=True,
            turbo_kv_quant_mode="turbo_vq",
            turbo_vq_bits=4,
            turbo_vq_key_bits=3,
            turbo_vq_residual_dim=-1,
            turbo_runtime_dequant_cache=False,
        )

        self.assertIsNone(past_key_values_data)
        key_cache, value_cache = past_key_values[0]
        self.assertIsInstance(key_cache, HybridTurboVQKVCache)
        self.assertIsInstance(value_cache, HybridTurboVQKVCache)
        self.assertIsInstance(key_cache.compressed_cache, TurboQuantizedKVCache)
        self.assertIsInstance(value_cache.compressed_cache, TurboQuantizedKVCache)
        self.assertEqual(key_cache.bits, 3)
        self.assertEqual(value_cache.bits, 4)
        self.assertEqual(key_cache.residual_dim, key_cache.head_dim)
        self.assertEqual(value_cache.residual_dim, 0)

    def test_factory_prefers_actual_attention_head_dim_over_sidecar_config(self):
        model = DummyModel(
            num_layers=1,
            num_kv_heads=1,
            hidden_size=128,
            num_heads=1,
            actual_head_dim=16,
        )

        past_key_values, _, _ = initialize_past_key_values(
            model,
            safe_max_length=8,
            turbo_quant=True,
            turbo_kv_quant_mode="turbo_vq",
            turbo_vq_bits=4,
            turbo_vq_key_bits=3,
            turbo_vq_residual_dim=-1,
            turbo_runtime_dequant_cache=False,
        )

        key_cache, value_cache = past_key_values[0]
        self.assertEqual(key_cache.head_dim, 16)
        self.assertEqual(value_cache.head_dim, 16)

    def test_outlier_aware_turboquant_cache_initializes_from_factory(self):
        model = DummyModel(num_layers=1, num_kv_heads=1, hidden_size=64, num_heads=4)

        past_key_values, past_key_values_data, _ = initialize_past_key_values(
            model,
            safe_max_length=8,
            turbo_quant=True,
            turbo_kv_quant_mode="turbo_vq",
            turbo_vq_bits=3,
            turbo_vq_key_bits=2,
            turbo_vq_outlier_bits=4,
            turbo_vq_key_outlier_bits=3,
            turbo_vq_outlier_channels=4,
            turbo_vq_residual_dim=-1,
            turbo_runtime_dequant_cache=False,
        )

        self.assertIsNone(past_key_values_data)
        key_cache, value_cache = past_key_values[0]
        self.assertIsInstance(key_cache, HybridOutlierTurboVQKVCache)
        self.assertIsInstance(value_cache, HybridOutlierTurboVQKVCache)

        key_cache.cat(torch.randn(1, 1, 5, 16))
        value_cache.cat(torch.randn(1, 1, 5, 16))
        self.assertEqual(key_cache.compressed_cache.n_outlier, 4)
        self.assertEqual(key_cache.compressed_cache.regular_cache.bits, 2)
        self.assertEqual(key_cache.compressed_cache.outlier_cache.bits, 3)
        self.assertEqual(value_cache.compressed_cache.regular_cache.bits, 3)
        self.assertEqual(value_cache.compressed_cache.outlier_cache.bits, 4)
        self.assertEqual(key_cache.compressed_cache.regular_cache.residual_dim, 12)
        self.assertEqual(key_cache.compressed_cache.outlier_cache.residual_dim, 4)

    def test_calibrated_outlier_indices_are_used_by_factory(self):
        model = DummyModel(num_layers=1, num_kv_heads=1, hidden_size=64, num_heads=4)
        calibrated = [[torch.tensor([1, 5, 7]), torch.tensor([2, 4, 6])]]

        past_key_values, _, _ = initialize_past_key_values(
            model,
            safe_max_length=8,
            turbo_quant=True,
            turbo_kv_quant_mode="turbo_vq",
            turbo_vq_bits=3,
            turbo_vq_key_bits=2,
            turbo_vq_outlier_bits=4,
            turbo_vq_key_outlier_bits=3,
            turbo_vq_outlier_channels=3,
            turbo_vq_outlier_indices=calibrated,
            turbo_vq_residual_dim=-1,
            turbo_runtime_dequant_cache=False,
        )

        key_cache, value_cache = past_key_values[0]
        torch.testing.assert_close(
            key_cache.compressed_cache.outlier_idx.cpu(),
            calibrated[0][0],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            value_cache.compressed_cache.outlier_idx.cpu(),
            calibrated[0][1],
            atol=0,
            rtol=0,
        )
        self.assertIsNotNone(key_cache.compressed_cache.regular_cache)
        self.assertIsNotNone(value_cache.compressed_cache.regular_cache)

    def test_reset_clears_nested_outlier_child_lengths(self):
        model = DummyModel(num_layers=1, num_kv_heads=1, hidden_size=64, num_heads=4)
        past_key_values, _, current_length_data = initialize_past_key_values(
            model,
            safe_max_length=8,
            turbo_quant=True,
            turbo_kv_quant_mode="turbo_vq",
            turbo_vq_bits=3,
            turbo_vq_key_bits=2,
            turbo_vq_outlier_bits=4,
            turbo_vq_key_outlier_bits=3,
            turbo_vq_outlier_channels=4,
            turbo_vq_residual_dim=-1,
            turbo_runtime_dequant_cache=False,
        )
        key_cache, value_cache = past_key_values[0]
        key_cache.cat(torch.randn(1, 1, 5, 16))
        value_cache.cat(torch.randn(1, 1, 5, 16))
        self.assertEqual(int(key_cache.compressed_cache.regular_current_length.item()), 5)
        self.assertEqual(int(value_cache.compressed_cache.outlier_current_length.item()), 5)

        current_length_data.zero_()
        reset_past_key_values(past_key_values)

        self.assertEqual(int(key_cache.current_length.item()), 0)
        self.assertEqual(int(key_cache.compressed_cache.regular_current_length.item()), 0)
        self.assertEqual(int(key_cache.compressed_cache.outlier_current_length.item()), 0)
        self.assertEqual(int(value_cache.compressed_cache.regular_current_length.item()), 0)
        self.assertEqual(int(value_cache.compressed_cache.outlier_current_length.item()), 0)

    def test_zero_accept_update_trims_nested_outlier_child_lengths(self):
        current_lengths = torch.zeros(2, dtype=torch.long)
        key_cache = HybridOutlierTurboVQKVCache(
            batch_size=1,
            num_heads=1,
            max_length=16,
            head_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            current_length=current_lengths[0],
            regular_bits=2,
            outlier_bits=3,
            n_outlier=2,
            residual_dim=0,
            hot_window=4,
        )
        value_cache = HybridOutlierTurboVQKVCache(
            batch_size=1,
            num_heads=1,
            max_length=16,
            head_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            current_length=current_lengths[1],
            regular_bits=2,
            outlier_bits=3,
            n_outlier=2,
            residual_dim=0,
            hot_window=4,
        )
        key_cache.cat(torch.randn(1, 1, 5, 8))
        value_cache.cat(torch.randn(1, 1, 5, 8))
        key_cache.append_compressed(torch.randn(1, 1, 4, 8), dim=2)
        value_cache.append_compressed(torch.randn(1, 1, 4, 8), dim=2)
        self.assertEqual(int(key_cache.compressed_cache.regular_current_length.item()), 9)

        update_inference_inputs_from_tree(
            input_ids=torch.zeros(1, 5, dtype=torch.long),
            candidates=torch.ones(1, 1, dtype=torch.long),
            best_candidate=0,
            accept_length=torch.tensor(0),
            retrieve_indices=torch.tensor([[0]], dtype=torch.long),
            outputs=None,
            tree_logits=torch.randn(1, 4, 10),
            tree_medusa_logits=None,
            new_token=0,
            past_key_values_data=None,
            current_length_data=current_lengths,
            past_key_values=[[key_cache, value_cache]],
        )

        self.assertEqual(int(key_cache.current_length.item()), 6)
        self.assertEqual(int(key_cache.compressed_cache.regular_current_length.item()), 6)
        self.assertEqual(int(key_cache.compressed_cache.outlier_current_length.item()), 6)
        self.assertEqual(int(value_cache.compressed_cache.regular_current_length.item()), 6)
        self.assertEqual(int(value_cache.compressed_cache.outlier_current_length.item()), 6)

    def test_outlier_calibration_cache_extracts_top_channels(self):
        data = torch.zeros(1, 1, 8, 6)
        current_length = torch.zeros((), dtype=torch.long)
        cache = OutlierCalibrationKVCache(data, current_length)
        sample = torch.zeros(1, 1, 4, 6)
        sample[..., 2] = 10.0
        sample[..., 4] = -5.0
        sample[..., 1] = 1.0
        cache.cat(sample)

        calibrated = extract_outlier_calibration_indices([[cache, cache]], n_outlier=2)
        torch.testing.assert_close(
            calibrated[0][0],
            torch.tensor([2, 4], dtype=torch.long),
            atol=0,
            rtol=0,
        )

    def test_quant_seed_controls_turbo_rotation(self):
        cache_a = TurboQuantizedKVCache(
            batch_size=1,
            num_heads=1,
            max_length=2,
            head_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            current_length=torch.zeros((), dtype=torch.long),
            bits=3,
            residual_dim=0,
            runtime_dequant_cache=False,
            quant_seed=123,
        )
        cache_b = TurboQuantizedKVCache(
            batch_size=1,
            num_heads=1,
            max_length=2,
            head_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            current_length=torch.zeros((), dtype=torch.long),
            bits=3,
            residual_dim=0,
            runtime_dequant_cache=False,
            quant_seed=123,
        )
        cache_c = TurboQuantizedKVCache(
            batch_size=1,
            num_heads=1,
            max_length=2,
            head_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            current_length=torch.zeros((), dtype=torch.long),
            bits=3,
            residual_dim=0,
            runtime_dequant_cache=False,
            quant_seed=456,
        )

        torch.testing.assert_close(cache_a.rotation, cache_b.rotation)
        self.assertGreater((cache_a.rotation - cache_c.rotation).abs().max().item(), 1e-3)

    def test_recursive_polar_quant_cache_uses_paper_layout(self):
        cache = PolarQuantizedKVCache(
            batch_size=1,
            num_heads=1,
            max_length=5,
            head_dim=16,
            device=torch.device("cpu"),
            dtype=torch.float32,
            current_length=torch.zeros((), dtype=torch.long),
            first_level_bits=4,
            other_level_bits=2,
            polar_levels=4,
            runtime_dequant_cache=False,
        )

        self.assertEqual(cache.level_dims, [8, 4, 2, 1])
        self.assertEqual(cache.level_packed_dims, [4, 1, 1, 1])
        visible = cache.cat(torch.randn(1, 1, 3, 16))
        self.assertEqual(tuple(visible.shape), (1, 1, 3, 16))
        self.assertTrue(torch.isfinite(visible).all())


if __name__ == "__main__":
    unittest.main()
