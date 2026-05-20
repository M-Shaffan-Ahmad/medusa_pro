import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from .medusa_choices import mc_sim_7b_63
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download

try:
    from safetensors.torch import load_file as load_safetensors_file
except Exception:  # pragma: no cover - safetensors is optional for legacy heads
    load_safetensors_file = None


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


def _medusa_uses_base_lm_head(config):
    version = str(getattr(config, "version", "")).lower()
    return bool(getattr(config, "medusa_head_uses_base_lm_head", False)) or version in {
        "2",
        "medusa2",
        "medusa-2",
    }


def _make_medusa_head(hidden_size, vocab_size, num_layers, uses_base_lm_head):
    layers = [ResBlock(hidden_size) for _ in range(num_layers)]
    if not uses_base_lm_head:
        layers.append(nn.Linear(hidden_size, vocab_size, bias=False))
    return nn.Sequential(*layers)


def _load_medusa_head_state_dict(pretrained_model_name_or_path):
    local_safetensors = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.safetensors")
    local_pt = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
    if os.path.exists(local_safetensors):
        filename = local_safetensors
    elif os.path.exists(local_pt):
        filename = local_pt
    else:
        try:
            filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.safetensors")
        except Exception:
            filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt")

    if filename.endswith(".safetensors"):
        if load_safetensors_file is None:
            raise ImportError("Loading medusa_lm_head.safetensors requires safetensors.")
        return load_safetensors_file(filename, device="cpu")
    try:
        return torch.load(filename, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(filename, map_location="cpu")


def _compute_medusa_logits(model, hidden_states):
    if getattr(model, "medusa_head_uses_base_lm_head", False):
        medusa_hidden = torch.stack(
            [model.medusa_head[i](hidden_states) for i in range(model.medusa)],
            dim=0,
        )
        return model.base_model.lm_head(medusa_hidden)
    return torch.stack(
        [model.medusa_head[i](hidden_states) for i in range(model.medusa)],
        dim=0,
    )


class MedusaModel(PreTrainedModel):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """
        super().__init__(config)
        # For compatibility with the old APIs
        medusa_num_heads = config.medusa_num_heads
        medusa_num_layers = config.medusa_num_layers
        base_model_name_or_path = config._name_or_path
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.medusa_head_uses_base_lm_head = _medusa_uses_base_lm_head(config)
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # Create a list of Medusa heads
        self.medusa_head = nn.ModuleList(
            [
                _make_medusa_head(
                    self.hidden_size,
                    self.vocab_size,
                    medusa_num_layers,
                    self.medusa_head_uses_base_lm_head,
                )
                for _ in range(medusa_num_heads)
            ]
        )

    # Add a link named base_model to self
    @property
    def base_model(self):
        return self

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
        except Exception:
            medusa_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(medusa_config.base_model_name_or_path)
            base_model_config.medusa_num_heads = getattr(medusa_config, "medusa_num_heads", 5)
            base_model_config.medusa_num_layers = getattr(medusa_config, "medusa_num_layers", 1)
            base_model_config.version = getattr(medusa_config, "version", None)
            base_model_config.medusa_head_uses_base_lm_head = _medusa_uses_base_lm_head(medusa_config)
            model = super().from_pretrained(
                medusa_config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            medusa_head_state_dict = _load_medusa_head_state_dict(pretrained_model_name_or_path)
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        return_medusa_logits=True,
        last_token_logits=False,
        **kwargs,
    ):
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if not medusa_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            hidden_states = outputs[0]
            logits_hidden_states = hidden_states[:, -1:, :] if last_token_logits else hidden_states
            if output_orig:
                orig = self.base_model.lm_head(logits_hidden_states)
        hidden_states = logits_hidden_states if last_token_logits else outputs[0]
        medusa_logits = _compute_medusa_logits(self, hidden_states) if return_medusa_logits else None
        if output_orig:
            return medusa_logits, outputs, orig
        return medusa_logits



class MedusaLlamaModel(KVLlamaForCausalLM):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """   
        # Load the base model
        super().__init__(config)
        # For compatibility with the old APIs

        medusa_num_heads = config.medusa_num_heads
        medusa_num_layers = config.medusa_num_layers
        base_model_name_or_path = config._name_or_path
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.medusa_head_uses_base_lm_head = _medusa_uses_base_lm_head(config)
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # Create a list of Medusa heads
        self.medusa_head = nn.ModuleList(
            [
                _make_medusa_head(
                    self.hidden_size,
                    self.vocab_size,
                    medusa_num_layers,
                    self.medusa_head_uses_base_lm_head,
                )
                for _ in range(medusa_num_heads)
            ]
        )

    # Add a link named base_model to self
    @property
    def base_model(self):
        return self
        
    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
        except Exception:
            medusa_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(medusa_config.base_model_name_or_path)
            base_model_config.medusa_num_heads = getattr(medusa_config, "medusa_num_heads", 5)
            base_model_config.medusa_num_layers = getattr(medusa_config, "medusa_num_layers", 1)
            base_model_config.version = getattr(medusa_config, "version", None)
            base_model_config.medusa_head_uses_base_lm_head = _medusa_uses_base_lm_head(medusa_config)
            model = super().from_pretrained(
                medusa_config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            medusa_head_state_dict = _load_medusa_head_state_dict(pretrained_model_name_or_path)
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        return_medusa_logits=True,
        last_token_logits=False,
        **kwargs,
    ):
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if not medusa_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            hidden_states = outputs[0]
            logits_hidden_states = hidden_states[:, -1:, :] if last_token_logits else hidden_states
            if output_orig:
                orig = self.base_model.lm_head(logits_hidden_states)
        hidden_states = logits_hidden_states if last_token_logits else outputs[0]
        medusa_logits = _compute_medusa_logits(self, hidden_states) if return_medusa_logits else None
        if output_orig:
            return medusa_logits, outputs, orig
        return medusa_logits

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=mc_sim_7b_63,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        top_p=0.8,
        sampling='typical',
        fast=True,
        turbo_quant=False,
        turbo_kv_compression=False,
        turbo_prune_keep=8,
        turbo_prune_min=6,
        turbo_prune_max=16,
        turbo_fallback_full_tree=True,
        turbo_fallback_accept_threshold=0,
        turbo_prune_confidence_margin=1.0,
        turbo_prune_prescreen_margin=0.75,
        turbo_prune_min_fraction=0.25,
        turbo_prune_min_node_fraction=0.30,
        turbo_prune_node_budget=0,
        turbo_prune_decisive_margin=1.5,
        turbo_prune_decisive_keep=8,
        turbo_prune_use_qjl=True,
        turbo_force_full_tree_fast_verifier=False,
        turbo_lazy_tree_medusa_logits=True,
        turbo_skip_threshold_high=1.1,
        turbo_skip_threshold_low=-0.1,
        turbo_kv_max_length=2048,
        turbo_kv_quant_mode="polar",
        turbo_radius_bits=8,
        turbo_theta_bits=8,
        turbo_vq_bits=4,
        turbo_vq_key_bits=None,
        turbo_vq_residual_dim=128,
        turbo_vq_residual_scale=1.0,
        turbo_hybrid_hot_window=512,
        turbo_runtime_dequant_cache=True,
        turbo_compile_decode=False,
        turbo_qjl_dim=128,
        stream=True,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
            top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
            sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
            fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling.
            turbo_quant (bool, optional): Enable TurboQuant two-pass pruning.
            turbo_kv_compression (bool, optional): Enable compressed KV cache.
            turbo_prune_keep (int, optional): Target number of candidate paths kept for high-accuracy verification.
            turbo_prune_min (int, optional): Minimum number of paths to keep.
            turbo_prune_max (int, optional): Maximum number of paths to keep.
            turbo_fallback_full_tree (bool, optional): If the pruned verifier accepts too few tokens,
                retry the full Medusa tree for that step to preserve acceptance quality.
            turbo_fallback_accept_threshold (int, optional): Retry full-tree verification when the
                pruned tree accepts this many speculative tokens or fewer. Defaults to 0, meaning
                fallback only when pruning finds no extra token beyond the greedy root.
            turbo_prune_confidence_margin (float, optional): If the pass-1 margin is smaller than
                this fraction of score stddev, skip pruning and verify the full tree immediately.
            turbo_prune_prescreen_margin (float, optional): Cheap Medusa-only margin gate run
                before QJL. Flat scores skip QJL and verify the full tree immediately.
                Set negative to disable.
            turbo_prune_min_fraction (float, optional): Minimum fraction of candidate paths
                that pruning must remove before using the pruned verifier. If the selected
                path set is too close to the full tree, the code skips pruning.
            turbo_prune_min_node_fraction (float, optional): Minimum fraction of unique
                Medusa tree nodes that must be removed before launching the pruned verifier.
                This catches cases where path pruning still leaves a nearly full tree block.
            turbo_prune_node_budget (int, optional): Maximum unique Medusa tree nodes to keep
                when selecting paths. `0` disables node-budget selection.
            turbo_prune_decisive_margin (float, optional): If the cheap Medusa-only path
                margin exceeds this multiple of score stddev, skip QJL and use a smaller
                verifier tree immediately. Set negative to disable.
            turbo_prune_decisive_keep (int, optional): Path budget used by the decisive
                Medusa-only fast path.
            turbo_prune_use_qjl (bool, optional): Enable the 1-bit/QJL side signal for path
                scoring after the cheap Medusa prescreen passes.
            turbo_force_full_tree_fast_verifier (bool, optional): Skip Turbo pruning/planning
                entirely and use the greedy full-tree verifier fast path when `temperature=0`.
            turbo_lazy_tree_medusa_logits (bool, optional): In greedy Turbo verification,
                compute Medusa-head logits only for the accepted tree node instead of every
                tree node. This preserves outputs and removes a large full-vocab projection.
            turbo_skip_threshold_high (float, optional): If pass-1 top prob exceeds this, skip pass-2 and accept one token.
            turbo_skip_threshold_low (float, optional): If pass-1 top prob is below this, skip pass-2 and do greedy one token.
            turbo_kv_max_length (int, optional): Maximum cache length for KV pre-allocation.
            turbo_kv_quant_mode (str, optional): KV compression backend: "polar" for
                recursive PolarQuant, or "turbo_vq" for TurboQuantprod keys plus
                TurboQuantmse values.
            turbo_radius_bits (int, optional): Later-level angle bits for recursive PolarQuant.
            turbo_theta_bits (int, optional): First-level angle bits for recursive PolarQuant.
            turbo_vq_bits (int, optional): Scalar Lloyd-Max bits for "turbo_vq" KV compression.
            turbo_vq_key_bits (int, optional): Override key-cache Lloyd-Max bits. Set this
                to `turbo_vq_bits - 1` to test the TurboQuant inner-product bit-budget
                variant while keeping value-cache quantization at `turbo_vq_bits`.
            turbo_vq_residual_dim (int, optional): 1-bit QJL residual sketch dimension for
                TurboQuant key-cache inner-product correction. Set 0 to disable.
            turbo_vq_residual_scale (float, optional): Multiplier for the residual-QJL
                correction. `1.0` is the unbiased estimator; lower values can dampen
                sketch variance during calibration.
            turbo_runtime_dequant_cache (bool, optional): Keep an incremental dequantized shadow cache
                for fast attention when polar compression is enabled. Set False to use the
                direct compressed-KV Triton attention path instead of materializing FP16 K/V.
            turbo_compile_decode (bool, optional): Use torch.compile on tensorized polar decode kernel.
            turbo_qjl_dim (int, optional): Sketch dimension for 1-bit QJL sidecar path scoring.
            stream (bool, optional): If True, yield decoded text after every Medusa step.
                If False, keep generation non-streaming and yield decoded text only once at the end.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache medusa buffers (the fixed patterns for tree attention)
        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices
        medusa_choices_key = (str(self.base_model.device), tuple(tuple(path) for path in medusa_choices))
        if (
            not hasattr(self, "turbo_pruned_layout_cache")
            or getattr(self, "turbo_pruned_layout_cache_key", None) != medusa_choices_key
        ):
            self.turbo_pruned_layout_cache = {}
            self.turbo_pruned_layout_cache_key = medusa_choices_key

        max_path_depth = max((len(path) for path in medusa_choices), default=1)
        worst_case_new_tokens = int(max_steps * (max_path_depth + 1))
        required_kv_len = int(input_ids.shape[1] + worst_case_new_tokens + 8)
        effective_kv_max_length = max(int(turbo_kv_max_length), required_kv_len)
        use_compressed_kv = bool(turbo_quant and turbo_kv_compression)
        requested_kv_quant_mode = str(turbo_kv_quant_mode).lower()
        requested_cache_mode = requested_kv_quant_mode if use_compressed_kv else "fp16"
        effective_turbo_vq_key_bits = (
            int(turbo_vq_bits)
            if turbo_vq_key_bits is None
            else int(turbo_vq_key_bits)
        )
        cache_reusable = (
            hasattr(self, "past_key_values")
            and getattr(self, "kv_cache_mode", None) == requested_cache_mode
            and getattr(self, "kv_cache_max_length", None) == effective_kv_max_length
            and (
                not use_compressed_kv
                or (
                    getattr(self, "kv_cache_radius_bits", None) == turbo_radius_bits
                    and getattr(self, "kv_cache_theta_bits", None) == turbo_theta_bits
                    and getattr(self, "kv_cache_vq_bits", None) == turbo_vq_bits
                    and getattr(self, "kv_cache_vq_key_bits", None) == effective_turbo_vq_key_bits
                    and getattr(self, "kv_cache_vq_residual_dim", None) == turbo_vq_residual_dim
                    and getattr(self, "kv_cache_vq_residual_scale", None) == float(turbo_vq_residual_scale)
                    and getattr(self, "kv_runtime_dequant_cache", None) == turbo_runtime_dequant_cache
                    and getattr(self, "kv_compile_decode", None) == turbo_compile_decode
                )
            )
        )

        # Initialize the past key and value states
        if cache_reusable:
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(
                self.base_model,
                safe_max_length=effective_kv_max_length,
                turbo_quant=use_compressed_kv,
                turbo_kv_quant_mode=requested_kv_quant_mode,
                turbo_radius_bits=turbo_radius_bits,
                turbo_theta_bits=turbo_theta_bits,
                turbo_vq_bits=turbo_vq_bits,
                turbo_vq_key_bits=effective_turbo_vq_key_bits,
                turbo_vq_residual_dim=turbo_vq_residual_dim,
                turbo_vq_residual_scale=turbo_vq_residual_scale,
                turbo_runtime_dequant_cache=turbo_runtime_dequant_cache,
                turbo_compile_decode=turbo_compile_decode,
            )
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data
            self.kv_cache_mode = requested_cache_mode
            self.kv_cache_max_length = effective_kv_max_length
            self.kv_cache_radius_bits = turbo_radius_bits
            self.kv_cache_theta_bits = turbo_theta_bits
            self.kv_cache_vq_bits = turbo_vq_bits
            self.kv_cache_vq_key_bits = effective_turbo_vq_key_bits
            self.kv_cache_vq_residual_dim = turbo_vq_residual_dim
            self.kv_cache_vq_residual_scale = float(turbo_vq_residual_scale)
            self.kv_runtime_dequant_cache = turbo_runtime_dequant_cache
            self.kv_compile_decode = turbo_compile_decode

        input_len = input_ids.shape[1]
        input_ids_buffer = torch.empty(
            input_ids.shape[0],
            effective_kv_max_length,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids_buffer[:, :input_len].copy_(input_ids)
        input_ids = input_ids_buffer[:, :input_len]
        full_path_lengths = (medusa_buffers["retrieve_indices"][:, 1:] >= 0).sum(dim=1)
        turbo_pruned_layout_cache = self.turbo_pruned_layout_cache if turbo_quant else None
        embed_weight = None
        qjl_scorer = None
        if turbo_quant and turbo_prune_use_qjl and not turbo_force_full_tree_fast_verifier:
            lm_head = getattr(self.base_model, "lm_head", None)
            embed_weight = getattr(lm_head, "weight", None)
            if embed_weight is None or not embed_weight.dtype.is_floating_point:
                embed_weight = self.base_model.get_input_embeddings().weight
            scorer_reusable = (
                hasattr(self, "turbo_qjl_scorer")
                and getattr(self, "turbo_qjl_dim", None) == int(turbo_qjl_dim)
                and getattr(self, "turbo_qjl_vocab_size", None) == int(embed_weight.shape[0])
                and getattr(self, "turbo_qjl_hidden_size", None) == int(embed_weight.shape[1])
                and getattr(self, "turbo_qjl_device", None) == str(embed_weight.device)
            )
            if not scorer_reusable:
                self.turbo_qjl_scorer = QJLTokenSketchCache(
                    vocab_size=int(embed_weight.shape[0]),
                    hidden_size=int(embed_weight.shape[1]),
                    sketch_dim=int(turbo_qjl_dim),
                    device=embed_weight.device,
                )
                self.turbo_qjl_dim = int(turbo_qjl_dim)
                self.turbo_qjl_vocab_size = int(embed_weight.shape[0])
                self.turbo_qjl_hidden_size = int(embed_weight.shape[1])
                self.turbo_qjl_device = str(embed_weight.device)
            qjl_scorer = self.turbo_qjl_scorer

        reset_medusa_mode(self)
        # Initialize tree attention mask and process prefill tokens
        need_turbo_query_state = bool(turbo_quant and not turbo_force_full_tree_fast_verifier)
        if need_turbo_query_state:
            medusa_logits, logits, query_state = initialize_medusa(
                input_ids,
                self,
                medusa_buffers["medusa_attn_mask"],
                past_key_values,
                return_query_state=True,
                last_token_logits=turbo_quant,
            )
            query_state = query_state.detach()
        else:
            medusa_logits, logits = initialize_medusa(
                input_ids,
                self,
                medusa_buffers["medusa_attn_mask"],
                past_key_values,
                return_query_state=False,
                last_token_logits=turbo_quant,
            )
            query_state = None

        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            # Generate candidates with topk predictions from Medusa heads
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
                temperature=temperature,
                posterior_alpha=posterior_alpha,
                posterior_threshold=posterior_threshold,
                top_p=top_p,
                sampling=sampling,
                fast=fast,
            )

            if turbo_quant:
                if turbo_force_full_tree_fast_verifier:
                    approx_scores = None
                    selected_paths = None
                    verify_full_tree = True
                    use_skip_gating = False
                    approx_top_prob = None
                else:
                    approx_scores, _, selected_paths, verify_full_tree = plan_turbo_pruning(
                        medusa_logits,
                        logits,
                        medusa_buffers["tree_indices"],
                        medusa_buffers["retrieve_indices"],
                        candidates=candidates,
                        query_state=query_state,
                        qjl_scorer=qjl_scorer,
                        embed_weight=embed_weight,
                        keep_target=turbo_prune_keep,
                        min_keep=turbo_prune_min,
                        max_keep=turbo_prune_max,
                        margin_scale=turbo_prune_confidence_margin,
                        prescreen_margin_scale=turbo_prune_prescreen_margin,
                        min_prune_fraction=turbo_prune_min_fraction,
                        min_node_prune_fraction=turbo_prune_min_node_fraction,
                        node_budget=turbo_prune_node_budget,
                        decisive_margin_scale=turbo_prune_decisive_margin,
                        decisive_keep=turbo_prune_decisive_keep,
                        use_qjl=turbo_prune_use_qjl,
                    )
                    use_skip_gating = (
                        (0.0 <= turbo_skip_threshold_high <= 1.0)
                        or (0.0 <= turbo_skip_threshold_low <= 1.0)
                    )
                    approx_top_prob = None
                    if use_skip_gating:
                        approx_probs = torch.softmax(approx_scores, dim=0)
                        approx_top_prob = float(approx_probs.max().item())

                if use_skip_gating and (
                    approx_top_prob >= turbo_skip_threshold_high
                    or approx_top_prob <= turbo_skip_threshold_low
                ):
                    if approx_top_prob >= turbo_skip_threshold_high:
                        picked_idx = int(torch.argmax(approx_scores).item())
                        next_token = candidates[picked_idx, 0]
                    else:
                        next_token = torch.argmax(logits[:, -1], dim=-1).squeeze(0)

                    next_token = next_token.view(1, 1)
                    self.base_model.model.medusa_mask = None
                    medusa_logits, outputs, logits = self(
                        next_token,
                        past_key_values=past_key_values,
                        output_orig=True,
                        medusa_forward=True,
                    )
                    query_state = outputs[0][:, -1, :].detach()
                    self.base_model.model.medusa_mask = medusa_buffers["medusa_attn_mask"]
                    input_ids = append_input_ids(
                        input_ids,
                        next_token,
                        input_ids_buffer=input_ids_buffer,
                    )
                    new_token += 1
                    should_stop = self.tokenizer.eos_token_id in input_ids[0, input_len:]
                    if stream:
                        yield {
                            "text": self.tokenizer.decode(
                                input_ids[0, input_len:],
                                skip_special_tokens=True,
                                spaces_between_special_tokens=False,
                                clean_up_tokenization_spaces=True,
                            )
                        }
                    if should_stop:
                        break
                    continue

                use_tree_update = False
                lazy_tree_medusa_logits = bool(
                    turbo_lazy_tree_medusa_logits and temperature == 0
                )
                if verify_full_tree:
                    if temperature == 0:
                        medusa_logits, logits, outputs, tree_hidden = tree_decoding(
                            self,
                            tree_candidates,
                            past_key_values,
                            medusa_buffers["medusa_position_ids"],
                            input_ids,
                            medusa_buffers["retrieve_indices"],
                            medusa_attn_mask=medusa_buffers["medusa_attn_mask"],
                            return_hidden=True,
                            gather_paths=False,
                            compute_medusa_logits=not lazy_tree_medusa_logits,
                            fast_attention_mask=True,
                        )
                        best_candidate, accept_length = evaluate_posterior_greedy_from_tree(
                            logits,
                            candidates,
                            medusa_buffers["retrieve_indices"],
                            path_lengths=full_path_lengths,
                        )
                        use_tree_update = True
                    else:
                        medusa_logits, logits, outputs, tree_hidden = tree_decoding(
                            self,
                            tree_candidates,
                            past_key_values,
                            medusa_buffers["medusa_position_ids"],
                            input_ids,
                            medusa_buffers["retrieve_indices"],
                            medusa_attn_mask=medusa_buffers["medusa_attn_mask"],
                            return_hidden=True,
                        )
                        best_candidate, accept_length = evaluate_posterior(
                            logits,
                            candidates,
                            temperature,
                            posterior_threshold,
                            posterior_alpha,
                            top_p=top_p,
                            sampling=sampling,
                            fast=fast,
                            path_lengths=full_path_lengths,
                        )
                    update_candidates = candidates
                    update_retrieve_indices = medusa_buffers["retrieve_indices"]
                else:
                    pruned = build_cached_pruned_medusa_buffers(
                        tree_candidates,
                        medusa_buffers["retrieve_indices"],
                        medusa_buffers["medusa_position_ids"],
                        medusa_buffers["medusa_attn_mask"],
                        selected_paths,
                        layout_cache=turbo_pruned_layout_cache,
                        min_node_prune_fraction=turbo_prune_min_node_fraction,
                    )

                    if pruned is None:
                        # Selected paths still retained most unique tree nodes, so a
                        # pruned forward would save little while adding layout overhead.
                        if temperature == 0:
                            medusa_logits, logits, outputs, tree_hidden = tree_decoding(
                                self,
                                tree_candidates,
                                past_key_values,
                                medusa_buffers["medusa_position_ids"],
                                input_ids,
                                medusa_buffers["retrieve_indices"],
                                medusa_attn_mask=medusa_buffers["medusa_attn_mask"],
                                return_hidden=True,
                                gather_paths=False,
                                compute_medusa_logits=not lazy_tree_medusa_logits,
                                fast_attention_mask=True,
                            )
                            best_candidate, accept_length = evaluate_posterior_greedy_from_tree(
                                logits,
                                candidates,
                                medusa_buffers["retrieve_indices"],
                                path_lengths=full_path_lengths,
                            )
                            use_tree_update = True
                        else:
                            medusa_logits, logits, outputs, tree_hidden = tree_decoding(
                                self,
                                tree_candidates,
                                past_key_values,
                                medusa_buffers["medusa_position_ids"],
                                input_ids,
                                medusa_buffers["retrieve_indices"],
                                medusa_attn_mask=medusa_buffers["medusa_attn_mask"],
                                return_hidden=True,
                            )
                            best_candidate, accept_length = evaluate_posterior(
                                logits,
                                candidates,
                                temperature,
                                posterior_threshold,
                                posterior_alpha,
                                top_p=top_p,
                                sampling=sampling,
                                fast=fast,
                                path_lengths=full_path_lengths,
                            )
                        update_candidates = candidates
                        update_retrieve_indices = medusa_buffers["retrieve_indices"]
                    else:
                        if temperature == 0:
                            medusa_logits, logits, outputs, tree_hidden = tree_decoding(
                                self,
                                pruned["tree_candidates"],
                                past_key_values,
                                pruned["medusa_position_ids"],
                                input_ids,
                                pruned["retrieve_indices"],
                                medusa_attn_mask=pruned["medusa_attn_mask"],
                                return_hidden=True,
                                gather_paths=False,
                                compute_medusa_logits=not lazy_tree_medusa_logits,
                                fast_attention_mask=True,
                            )
                            best_candidate, accept_length = evaluate_posterior_greedy_from_tree(
                                logits,
                                pruned["candidates"],
                                pruned["retrieve_indices"],
                                path_lengths=pruned["path_lengths"],
                            )
                            use_tree_update = True
                        else:
                            medusa_logits, logits, outputs, tree_hidden = tree_decoding(
                                self,
                                pruned["tree_candidates"],
                                past_key_values,
                                pruned["medusa_position_ids"],
                                input_ids,
                                pruned["retrieve_indices"],
                                medusa_attn_mask=pruned["medusa_attn_mask"],
                                return_hidden=True,
                            )
                            best_candidate, accept_length = evaluate_posterior(
                                logits,
                                pruned["candidates"],
                                temperature,
                                posterior_threshold,
                                posterior_alpha,
                                top_p=top_p,
                                sampling=sampling,
                                fast=fast,
                                path_lengths=pruned["path_lengths"],
                            )
                        pruned_accept_length = int(
                            accept_length.item() if torch.is_tensor(accept_length) else accept_length
                        )
                        if (
                            turbo_fallback_full_tree
                            and pruned_accept_length <= int(turbo_fallback_accept_threshold)
                        ):
                            current_length_data.fill_(input_ids.shape[1])
                            if temperature == 0:
                                medusa_logits, logits, outputs, tree_hidden = tree_decoding(
                                    self,
                                    tree_candidates,
                                    past_key_values,
                                    medusa_buffers["medusa_position_ids"],
                                    input_ids,
                                    medusa_buffers["retrieve_indices"],
                                    medusa_attn_mask=medusa_buffers["medusa_attn_mask"],
                                    return_hidden=True,
                                    gather_paths=False,
                                    compute_medusa_logits=not lazy_tree_medusa_logits,
                                    fast_attention_mask=True,
                                )
                                best_candidate, accept_length = evaluate_posterior_greedy_from_tree(
                                    logits,
                                    candidates,
                                    medusa_buffers["retrieve_indices"],
                                    path_lengths=full_path_lengths,
                                )
                                use_tree_update = True
                            else:
                                medusa_logits, logits, outputs, tree_hidden = tree_decoding(
                                    self,
                                    tree_candidates,
                                    past_key_values,
                                    medusa_buffers["medusa_position_ids"],
                                    input_ids,
                                    medusa_buffers["retrieve_indices"],
                                    medusa_attn_mask=medusa_buffers["medusa_attn_mask"],
                                    return_hidden=True,
                                )
                                best_candidate, accept_length = evaluate_posterior(
                                    logits,
                                    candidates,
                                    temperature,
                                    posterior_threshold,
                                    posterior_alpha,
                                    top_p=top_p,
                                    sampling=sampling,
                                    fast=fast,
                                    path_lengths=full_path_lengths,
                                )
                            update_candidates = candidates
                            update_retrieve_indices = medusa_buffers["retrieve_indices"]
                        else:
                            update_candidates = pruned["candidates"]
                            update_retrieve_indices = pruned["retrieve_indices"]

                if use_tree_update:
                    input_ids, logits, medusa_logits, new_token = update_inference_inputs_from_tree(
                        input_ids,
                        update_candidates,
                        best_candidate,
                        accept_length,
                        update_retrieve_indices,
                        outputs,
                        logits,
                        medusa_logits,
                        new_token,
                        past_key_values_data,
                        current_length_data,
                        past_key_values=past_key_values,
                        input_ids_buffer=input_ids_buffer,
                    )
                    node_idx = update_retrieve_indices[best_candidate, accept_length].reshape(1)
                    accepted_hidden = None
                    if medusa_logits is None or need_turbo_query_state:
                        accepted_hidden = tree_hidden.index_select(0, node_idx)
                    if medusa_logits is None:
                        medusa_logits = _compute_medusa_logits(self, accepted_hidden.unsqueeze(0))
                    if need_turbo_query_state:
                        query_state = accepted_hidden.detach()
                else:
                    input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                        input_ids,
                        update_candidates,
                        best_candidate,
                        accept_length,
                        update_retrieve_indices,
                        outputs,
                        logits,
                        medusa_logits,
                        new_token,
                        past_key_values_data,
                        current_length_data,
                        past_key_values=past_key_values,
                        input_ids_buffer=input_ids_buffer,
                    )
                    best_idx = int(best_candidate.item()) if torch.is_tensor(best_candidate) else int(best_candidate)
                    accept_idx = int(accept_length.item()) if torch.is_tensor(accept_length) else int(accept_length)
                    query_state = tree_hidden[best_idx, accept_idx].unsqueeze(0).detach()
            else:
                medusa_logits, logits, outputs, tree_hidden = tree_decoding(
                    self,
                    tree_candidates,
                    past_key_values,
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                    return_hidden=True,
                )

                best_candidate, accept_length = evaluate_posterior(
                    logits,
                    candidates,
                    temperature,
                    posterior_threshold,
                    posterior_alpha,
                    top_p=top_p,
                    sampling=sampling,
                    fast=fast,
                    path_lengths=full_path_lengths,
                )

                input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    medusa_buffers["retrieve_indices"],
                    outputs,
                    logits,
                    medusa_logits,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    past_key_values=past_key_values,
                    input_ids_buffer=input_ids_buffer,
                )
                best_idx = int(best_candidate.item()) if torch.is_tensor(best_candidate) else int(best_candidate)
                accept_idx = int(accept_length.item()) if torch.is_tensor(accept_length) else int(accept_length)
                query_state = tree_hidden[best_idx, accept_idx].unsqueeze(0).detach()

            should_stop = self.tokenizer.eos_token_id in input_ids[0, input_len:]
            if stream:
                yield {
                    "text": self.tokenizer.decode(
                        input_ids[0, input_len:],
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    )
                }

            if should_stop:
                break

        if not stream:
            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

# Currently only support LlamaModel
MedusaModel = MedusaLlamaModel
