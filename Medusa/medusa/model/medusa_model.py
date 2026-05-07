import torch
import torch.nn as nn
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
# import transformers

# # monkey patch
# transformers.models.llama.modeling_llama.LlamaForCausalLM = KVLlamaForCausalLM
# transformers.models.mistral.modeling_mistral.MistralForCausalLM = KVMistralForCausalLM

from transformers import PreTrainedModel, PretrainedConfig
from .utils import *
from .kv_cache import initialize_past_key_values
from .medusa_choices import *
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download
import warnings

try:
    from safetensors.torch import load_file as load_safetensors_file
except Exception:  # pragma: no cover - safetensors is optional for legacy heads
    load_safetensors_file = None

class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        version=None,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.version = version if version is not None else getattr(self, "version", None)
        self.base_model_name_or_path = base_model_name_or_path

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


def _infer_medusa_num_heads_from_state_dict(state_dict, fallback):
    head_ids = set()
    for key in state_dict.keys():
        parts = str(key).split(".")
        if parts and parts[0] == "medusa_head":
            parts = parts[1:]
        if parts and parts[0].isdigit():
            head_ids.add(int(parts[0]))
    if not head_ids:
        return int(fallback)
    return max(head_ids) + 1


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


def _resolve_layer_index(model, layer_idx):
    layers = getattr(getattr(model.base_model, "model", None), "layers", None)
    if layers is None:
        return None, None
    n_layers = len(layers)
    layer_idx = int(layer_idx)
    if layer_idx < 0:
        layer_idx = n_layers + layer_idx
    layer_idx = min(max(0, layer_idx), n_layers - 1)
    return layer_idx, layers[layer_idx]


def _estimate_packed_kv_qjl_scores(
    model,
    tree_candidates,
    retrieve_indices,
    query_state,
    past_key_values,
    layer_idx=-1,
):
    resolved_idx, layer = _resolve_layer_index(model, layer_idx)
    if resolved_idx is None or past_key_values is None:
        return None
    if resolved_idx >= len(past_key_values):
        return None
    key_cache = past_key_values[resolved_idx][0]
    if getattr(key_cache, "qjl_bits", None) is None:
        return None
    if query_state is None:
        return None
    if tree_candidates.dim() != 2 or tree_candidates.shape[1] <= 1:
        return None

    attn = getattr(layer, "self_attn", None)
    q_proj = getattr(attn, "q_proj", None)
    if attn is None or q_proj is None:
        return None

    node_tokens = tree_candidates[0, 1:].clamp_min(0)
    embed = model.base_model.get_input_embeddings()(node_tokens)
    hidden = query_state.reshape(1, -1).to(device=embed.device, dtype=embed.dtype)
    if resolved_idx == 0:
        proxy_hidden = embed
    elif hidden.shape[-1] == embed.shape[-1]:
        proxy_hidden = embed + hidden
    else:
        proxy_hidden = embed
    input_norm = getattr(layer, "input_layernorm", None)
    if input_norm is not None:
        proxy_hidden = input_norm(proxy_hidden)

    q = q_proj(proxy_hidden)
    num_heads = int(getattr(attn, "num_heads", 0))
    num_kv_heads = int(getattr(attn, "num_key_value_heads", 0))
    head_dim = int(getattr(attn, "head_dim", q.shape[-1] // max(1, num_heads)))
    if num_heads <= 0 or num_kv_heads <= 0 or q.shape[-1] != num_heads * head_dim:
        return None
    q = q.view(q.shape[0], num_heads, head_dim)
    groups = max(1, num_heads // num_kv_heads)
    if groups > 1:
        q = q.view(q.shape[0], num_kv_heads, groups, head_dim).mean(dim=2)
    else:
        q = q[:, :num_kv_heads]

    query_bits = key_cache.pack_qjl_bits(q)
    node_scores = packed_kv_qjl_node_scores(query_bits, key_cache)
    path_scores, _ = estimate_packed_kv_qjl_path_scores(node_scores, retrieve_indices)
    return path_scores


class MedusaModelABC(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    # Load the base model
    # base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["LlamaDecoderLayer", "MistralDecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True

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
        except:
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            medusa_head_state_dict = _load_medusa_head_state_dict(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.medusa_num_heads = _infer_medusa_num_heads_from_state_dict(
                medusa_head_state_dict,
                getattr(config, "medusa_num_heads", 5),
            )
            base_model_config.medusa_num_layers = config.medusa_num_layers
            base_model_config.version = getattr(config, "version", None)
            base_model_config.medusa_head_uses_base_lm_head = _medusa_uses_base_lm_head(config)
            model = super().from_pretrained(
                config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            return model
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer


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
    def get_medusa_choice(self, model_name):
        if 'vicuna' in model_name:
            if '7b' in model_name:
                return vicuna_7b_stage2
            elif '13b' in model_name:
                return vicuna_13b_stage2
            elif '33b' in model_name:
                return vicuna_33b_stage2
        elif 'zephyr' in model_name:
            return zephyr_stage2
        warnings.warn('Please specify medusa choice configuration!')
        return mc_sim_7b_63

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=None,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True,
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
        turbo_prune_auto_disable_after=4,
        turbo_prune_use_kv_qjl=False,
        turbo_kv_qjl_dim=128,
        turbo_kv_qjl_layer=-1,
        turbo_kv_qjl_keep_fraction=0.30,
        turbo_kv_qjl_weight=0.5,
        turbo_kv_qjl_min_kv_len=16384,
        turbo_kv_qjl_medusa_pool_fraction=0.70,
        turbo_kv_qjl_medusa_anchor_keep=2,
        turbo_packed_kv_qjl_auto_disable_after=4,
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
        collect_stats=False,
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
            fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
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
            turbo_prune_auto_disable_after (int, optional): If pass-1 planning chooses the
                full tree this many consecutive times, bypass pruning/planning for the rest
                of the request. This preserves outputs and avoids repeated planner overhead.
            turbo_prune_use_kv_qjl (bool, optional): Use packed sign(RK) KV-cache sidecar
                bits plus packed candidate query proxies to prefilter Medusa paths.
            turbo_kv_qjl_dim (int, optional): Packed KV-QJL sketch dimension. Must be a
                multiple of 32; 128 stores each key vector in four int32 words.
            turbo_kv_qjl_layer (int, optional): Transformer layer whose key cache is
                sketched for packed KV-QJL pruning. Defaults to the final layer.
            turbo_kv_qjl_keep_fraction (float, optional): Approximate fraction of paths
                to keep before exact verification when packed KV-QJL is active.
            turbo_kv_qjl_weight (float, optional): Weight of packed KV-QJL scores when
                fusing with Medusa branch-prior scores.
            turbo_kv_qjl_min_kv_len (int, optional): Minimum current KV length before
                packed KV-QJL pruning is allowed. Below this, the full-tree fast verifier
                is used because the packed pre-pass overhead usually dominates.
            turbo_kv_qjl_medusa_pool_fraction (float, optional): Fraction of paths kept
                by the Medusa branch prior before packed KV-QJL reranking.
            turbo_kv_qjl_medusa_anchor_keep (int, optional): Number of Medusa-prior best
                paths forced into the survivor competition so the sketch cannot prune
                obvious high-priority branches too early.
            turbo_packed_kv_qjl_auto_disable_after (int, optional): In fallback-safe
                packed KV-QJL mode, bypass pruning for the rest of the request after this
                many consecutive pruned-tree fallbacks. Set 0 to disable.
            turbo_force_full_tree_fast_verifier (bool, optional): Skip Turbo pruning/planning
                entirely and use the greedy full-tree verifier fast path when `temperature=0`.
            turbo_lazy_tree_medusa_logits (bool, optional): In greedy Turbo verification,
                compute Medusa-head logits only for the accepted tree node instead of every
                tree node. This preserves outputs and removes a large full-vocab projection.
            turbo_skip_threshold_high (float, optional): If pass-1 top prob exceeds this, skip pass-2 and accept one token.
            turbo_skip_threshold_low (float, optional): If pass-1 top prob is below this, skip pass-2 and do greedy one token.
                Defaults disable skip-gating (high>1, low<0) to preserve acceptance quality.
            turbo_kv_max_length (int, optional): Maximum cache length for KV pre-allocation.
            turbo_kv_quant_mode (str, optional): KV compression backend: "polar" for the
                previous polar-inspired cache, "turbo_vq" for strict random-rotation
                Lloyd-Max TurboQuant KV, or "hybrid_turbo_vq" for exact hot-window KV
                plus compressed older KV.
            turbo_radius_bits (int, optional): Radius quantization bits for polar KV.
            turbo_theta_bits (int, optional): Angle quantization bits for polar KV.
            turbo_vq_bits (int, optional): Scalar Lloyd-Max bits for "turbo_vq" KV compression.
            turbo_vq_key_bits (int, optional): Override key-cache Lloyd-Max bits. Set this
                to `turbo_vq_bits - 1` to test the TurboQuant inner-product bit-budget
                variant while keeping value-cache quantization at `turbo_vq_bits`.
            turbo_vq_residual_dim (int, optional): 1-bit QJL residual sketch dimension for
                TurboQuant key-cache inner-product correction. Set 0 to disable.
            turbo_vq_residual_scale (float, optional): Multiplier for the residual-QJL
                correction. `1.0` is the unbiased estimator; lower values can dampen
                sketch variance during calibration.
            turbo_hybrid_hot_window (int, optional): Exact recent-KV window for
                `turbo_kv_quant_mode="hybrid_turbo_vq"`. Older positions remain
                compressed and are decoded only when the SDPA verifier needs them.
            turbo_runtime_dequant_cache (bool, optional): Keep an incremental dequantized shadow cache
                for fast attention when polar compression is enabled. Set False to use the
                direct compressed-KV Triton attention path instead of materializing FP16 K/V.
            turbo_compile_decode (bool, optional): Use torch.compile on tensorized polar decode kernel.
            turbo_qjl_dim (int, optional): Sketch dimension for 1-bit QJL sidecar path scoring.
            stream (bool, optional): If True, yield decoded text after every Medusa step.
                If False, keep generation non-streaming and yield decoded text only once at the end.
            collect_stats (bool, optional): Attach lightweight pruning/verifier counters to
                yielded outputs and store the final snapshot on `last_medusa_generate_stats`.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache medusa buffers (the fixed patterns for tree attention)
        if medusa_choices is None:
            medusa_choices = self.get_medusa_choice(self.base_model_name_or_path)

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

        # Medusa may accept multiple tokens per step, so max_steps alone can
        # underestimate cache length. Use tree depth for a safe upper bound.
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
        packed_kv_qjl_requested = bool(
            turbo_quant
            and turbo_prune_use_kv_qjl
            and not turbo_force_full_tree_fast_verifier
            and not use_compressed_kv
        )
        effective_kv_qjl_min_len = max(0, int(turbo_kv_qjl_min_kv_len))
        use_packed_kv_qjl = bool(
            packed_kv_qjl_requested
            and effective_kv_max_length >= effective_kv_qjl_min_len
        )
        effective_kv_qjl_dim = int(turbo_kv_qjl_dim) if use_packed_kv_qjl else 0
        cache_reusable = (
            hasattr(self, "past_key_values")
            and getattr(self, "kv_cache_mode", None) == requested_cache_mode
            and getattr(self, "kv_cache_max_length", None) == effective_kv_max_length
            and getattr(self, "kv_qjl_dim", 0) == effective_kv_qjl_dim
            and getattr(self, "kv_qjl_layer", int(turbo_kv_qjl_layer)) == int(turbo_kv_qjl_layer)
            and (
                not use_compressed_kv
                or (
                    getattr(self, "kv_cache_radius_bits", None) == turbo_radius_bits
                    and getattr(self, "kv_cache_theta_bits", None) == turbo_theta_bits
                    and getattr(self, "kv_cache_vq_bits", None) == turbo_vq_bits
                    and getattr(self, "kv_cache_vq_key_bits", None) == effective_turbo_vq_key_bits
                    and getattr(self, "kv_cache_vq_residual_dim", None) == turbo_vq_residual_dim
                    and getattr(self, "kv_cache_vq_residual_scale", None) == float(turbo_vq_residual_scale)
                    and getattr(self, "kv_cache_hybrid_hot_window", None) == int(turbo_hybrid_hot_window)
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
                turbo_hybrid_hot_window=turbo_hybrid_hot_window,
                turbo_runtime_dequant_cache=turbo_runtime_dequant_cache,
                turbo_compile_decode=turbo_compile_decode,
                packed_qjl_sketch_dim=effective_kv_qjl_dim,
                packed_qjl_layer=int(turbo_kv_qjl_layer),
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
            self.kv_cache_hybrid_hot_window = int(turbo_hybrid_hot_window)
            self.kv_runtime_dequant_cache = turbo_runtime_dequant_cache
            self.kv_compile_decode = turbo_compile_decode
            self.kv_qjl_dim = effective_kv_qjl_dim
            self.kv_qjl_layer = int(turbo_kv_qjl_layer)

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
        full_tree_node_count = int(medusa_buffers["tree_indices"].numel())
        full_candidate_path_count = int(medusa_buffers["retrieve_indices"].shape[0])
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
        stats = None
        if collect_stats:
            stats = {
                "decode_steps": 0,
                "generated_tokens": 0,
                "candidate_paths_considered": 0,
                "selected_candidate_paths": 0,
                "verified_tree_nodes": 0,
                "full_tree_steps": 0,
                "pruned_tree_steps": 0,
                "fallback_full_tree_steps": 0,
                "skip_gating_steps": 0,
                "planner_steps": 0,
                "planner_full_tree_decisions": 0,
                "planner_bypass_steps": 0,
                "packed_kv_qjl_steps": 0,
                "packed_kv_qjl_fallback_steps": 0,
                "packed_kv_qjl_gated_steps": 0,
                "packed_kv_qjl_auto_disable_events": 0,
            }

        def record_verifier(kind, nodes, paths=0):
            if stats is None:
                return
            stats["verified_tree_nodes"] += int(nodes)
            stats["selected_candidate_paths"] += int(paths)
            if kind == "full":
                stats["full_tree_steps"] += 1
            elif kind == "pruned":
                stats["pruned_tree_steps"] += 1
            elif kind == "fallback_full":
                stats["fallback_full_tree_steps"] += 1

        def build_stream_output():
            if stats is not None:
                stats["generated_tokens"] = int(new_token)
                self.last_medusa_generate_stats = dict(stats)
            result = {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }
            if stats is not None:
                result["stats"] = dict(stats)
            return result

        planner_full_tree_streak = 0
        planner_bypass_pruning = False
        packed_kv_qjl_fallback_streak = 0

        for idx in range(max_steps):
            if stats is not None:
                stats["decode_steps"] += 1
            used_packed_kv_qjl = False
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
            if stats is not None:
                stats["candidate_paths_considered"] += int(candidates.shape[0])

            if turbo_quant:
                packed_kv_qjl_gated = bool(
                    packed_kv_qjl_requested
                    and not turbo_prune_use_qjl
                    and (
                        not use_packed_kv_qjl
                        or int(input_ids.shape[1]) < effective_kv_qjl_min_len
                    )
                )
                if (
                    turbo_force_full_tree_fast_verifier
                    or planner_bypass_pruning
                    or packed_kv_qjl_gated
                ):
                    approx_scores = None
                    selected_paths = None
                    verify_full_tree = True
                    use_skip_gating = False
                    approx_top_prob = None
                    if stats is not None:
                        if planner_bypass_pruning:
                            stats["planner_bypass_steps"] += 1
                        if packed_kv_qjl_gated:
                            stats["packed_kv_qjl_gated_steps"] += 1
                else:
                    used_packed_kv_qjl = False
                    if use_packed_kv_qjl:
                        kv_qjl_path_scores = _estimate_packed_kv_qjl_scores(
                            self,
                            tree_candidates,
                            medusa_buffers["retrieve_indices"],
                            query_state,
                            past_key_values,
                            layer_idx=turbo_kv_qjl_layer,
                        )
                        if kv_qjl_path_scores is not None:
                            approx_scores, _, selected_paths, verify_full_tree = plan_packed_kv_qjl_pruning(
                                medusa_logits,
                                logits,
                                medusa_buffers["tree_indices"],
                                medusa_buffers["retrieve_indices"],
                                kv_qjl_path_scores,
                                candidates=candidates,
                                keep_fraction=turbo_kv_qjl_keep_fraction,
                                keep_target=turbo_prune_keep,
                                min_keep=turbo_prune_min,
                                max_keep=turbo_prune_max,
                                min_prune_fraction=turbo_prune_min_fraction,
                                min_node_prune_fraction=turbo_prune_min_node_fraction,
                                node_budget=turbo_prune_node_budget,
                                kv_qjl_weight=turbo_kv_qjl_weight,
                                medusa_pool_fraction=turbo_kv_qjl_medusa_pool_fraction,
                                medusa_anchor_keep=turbo_kv_qjl_medusa_anchor_keep,
                            )
                            used_packed_kv_qjl = True
                            if stats is not None:
                                stats["packed_kv_qjl_steps"] += 1
                        elif stats is not None:
                            stats["packed_kv_qjl_fallback_steps"] += 1

                    if not used_packed_kv_qjl:
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
                    if stats is not None:
                        stats["planner_steps"] += 1
                        if verify_full_tree:
                            stats["planner_full_tree_decisions"] += 1
                    if verify_full_tree:
                        planner_full_tree_streak += 1
                    else:
                        planner_full_tree_streak = 0
                    if (
                        int(turbo_prune_auto_disable_after) > 0
                        and planner_full_tree_streak >= int(turbo_prune_auto_disable_after)
                    ):
                        planner_bypass_pruning = True
                    use_skip_gating = (
                        (0.0 <= turbo_skip_threshold_high <= 1.0)
                        or (0.0 <= turbo_skip_threshold_low <= 1.0)
                    )
                    approx_top_prob = None
                    if use_skip_gating:
                        approx_probs = torch.softmax(approx_scores, dim=0)
                        approx_top_prob = float(approx_probs.max().item())

                # Confidence gating: skip expensive pass-2 when clearly certain/uncertain.
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
                    # Disable tree mask for single-token path.
                    self.base_model.model.medusa_mask = None
                    medusa_logits, outputs, logits = self(
                        next_token,
                        past_key_values=past_key_values,
                        output_orig=True,
                        medusa_forward=True,
                    )
                    query_state = outputs[0][:, -1, :].detach()
                    # Restore default tree mask for future tree decode calls.
                    self.base_model.model.medusa_mask = medusa_buffers["medusa_attn_mask"]
                    input_ids = append_input_ids(
                        input_ids,
                        next_token,
                        input_ids_buffer=input_ids_buffer,
                    )
                    new_token += 1
                    if stats is not None:
                        stats["skip_gating_steps"] += 1
                    should_stop = self.tokenizer.eos_token_id in input_ids[0, input_len:]
                    if stream:
                        yield build_stream_output()
                    if should_stop:
                        break
                    continue

                use_tree_update = False
                lazy_tree_medusa_logits = bool(
                    turbo_lazy_tree_medusa_logits and temperature == 0
                )
                if verify_full_tree:
                    # Ambiguous pass-1 scores: skip the pruned verifier entirely
                    # to avoid pruned-forward + fallback-forward double work.
                    record_verifier("full", full_tree_node_count, full_candidate_path_count)
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
                        record_verifier("full", full_tree_node_count, full_candidate_path_count)
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
                        # High-accuracy verification on pruned tree first.
                        record_verifier(
                            "pruned",
                            int(pruned["tree_candidates"].shape[1]),
                            int(pruned["retrieve_indices"].shape[0]),
                        )
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
                            if used_packed_kv_qjl:
                                packed_kv_qjl_fallback_streak += 1
                                if (
                                    int(turbo_packed_kv_qjl_auto_disable_after) > 0
                                    and packed_kv_qjl_fallback_streak
                                    >= int(turbo_packed_kv_qjl_auto_disable_after)
                                ):
                                    planner_bypass_pruning = True
                                    if stats is not None:
                                        stats["packed_kv_qjl_auto_disable_events"] += 1
                            # Pruning was too conservative for this step. Rewind the
                            # cache length and verify the full tree so accuracy/acceptance
                            # quality stays equivalent to Medusa base.
                            current_length_data.fill_(input_ids.shape[1])
                            record_verifier("fallback_full", full_tree_node_count, full_candidate_path_count)
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
                            if used_packed_kv_qjl:
                                packed_kv_qjl_fallback_streak = 0
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
                # Use tree attention to verify the candidates and get predictions
                record_verifier("full", full_tree_node_count, full_candidate_path_count)
                medusa_logits, logits, outputs, tree_hidden = tree_decoding(
                    self,
                    tree_candidates,
                    past_key_values,
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                    return_hidden=True,
                )

                # Evaluate the posterior of the candidates to select the accepted candidate prefix
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

                # Update the input_ids and logits
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
                yield build_stream_output()

            if should_stop:
                break

        if not stream:
            yield build_stream_output()


class MedusaModelLlama(MedusaModelABC, KVLlamaForCausalLM):
    pass

class MedusaModelMistral(MedusaModelABC, KVMistralForCausalLM):
    pass


class MedusaModel():
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
        except:
            # MEDUSA-v0.1 load
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_model_config.model_type

        if config.model_type == "llama":
            return MedusaModelLlama.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        elif config.model_type == "mistral":
            return MedusaModelMistral.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        else:
            raise ValueError("Only support llama and mistral for now!!")
