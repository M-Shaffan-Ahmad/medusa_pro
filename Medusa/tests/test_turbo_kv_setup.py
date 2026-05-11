from types import SimpleNamespace
import unittest

import torch

from medusa.model.kv_cache import HybridTurboVQKVCache, KVCache, initialize_past_key_values
from medusa.model.medusa_model import (
    infer_model_context_window,
    resolve_turbo_kv_cache_plan,
)


class DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class DummyModel:
    def __init__(self, num_layers=3, num_kv_heads=2, hidden_size=16, num_heads=4):
        self.config = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_key_value_heads=num_kv_heads,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            max_position_embeddings=131072,
        )
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList(DummyLayer() for _ in range(num_layers))
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

    def test_hybrid_turbo_vq_cache_initializes_without_fp16_backing_store(self):
        model = DummyModel(num_layers=2)

        past_key_values, past_key_values_data, _ = initialize_past_key_values(
            model,
            safe_max_length=16,
            turbo_quant=True,
            turbo_kv_quant_mode="hybrid_turbo_vq",
            turbo_vq_bits=4,
            turbo_vq_key_bits=3,
            turbo_vq_residual_dim=32,
            turbo_hybrid_hot_window=4,
        )

        self.assertIsNone(past_key_values_data)
        key_cache, value_cache = past_key_values[0]
        self.assertIsInstance(key_cache, HybridTurboVQKVCache)
        self.assertIsInstance(value_cache, HybridTurboVQKVCache)
        self.assertEqual(key_cache.bits, 3)
        self.assertEqual(value_cache.bits, 4)
        self.assertEqual(key_cache.hot_window, 4)

        visible = key_cache.cat(torch.randn(1, 2, 2, 4))
        self.assertEqual(tuple(visible.shape), (1, 2, 2, 4))
        self.assertEqual(int(key_cache.current_length.item()), 2)


if __name__ == "__main__":
    unittest.main()
