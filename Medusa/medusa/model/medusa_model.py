import torch
import torch.nn as nn
import csv
import json
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
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 5.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        version=None,
        draft_head_type="medusa",
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.version = version if version is not None else getattr(self, "version", None)
        self.draft_head_type = draft_head_type
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


def _normalize_draft_head_type(value):
    value = "medusa" if value is None else str(value).lower().strip()
    if value in {"", "default", "config"}:
        value = "medusa"
    if value not in {"medusa", "hydra"}:
        raise ValueError("draft_head_type must be 'medusa' or 'hydra'.")
    return value


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


def _medusa_head_module_state_dict(state_dict):
    if any(str(key).startswith("medusa_head.") for key in state_dict.keys()):
        prefix_len = len("medusa_head.")
        return {
            str(key)[prefix_len:]: value
            for key, value in state_dict.items()
            if str(key).startswith("medusa_head.")
        }
    return {
        str(key): value
        for key, value in state_dict.items()
        if not str(key).startswith("hydra_")
    }


def _load_medusa_head_into_model(model, state_dict):
    module_state_dict = _medusa_head_module_state_dict(state_dict)
    expected_weight = module_state_dict.get("0.0.linear.weight")
    current_weight = None
    try:
        current_weight = model.medusa_head[0][0].linear.weight
    except Exception:
        current_weight = None
    if (
        expected_weight is not None
        and current_weight is not None
        and tuple(current_weight.shape) != tuple(expected_weight.shape)
    ):
        # Quantized base-model loads can accidentally pack the Medusa heads
        # before the sidecar is restored. Rebuild only the draft heads so the
        # trained fp16/fp32 sidecar can load into ordinary Linear layers.
        model.medusa_head = nn.ModuleList(
            [
                _make_medusa_head(
                    model.hidden_size,
                    model.vocab_size,
                    model.medusa_num_layers,
                    model.medusa_head_uses_base_lm_head,
                )
                for _ in range(model.medusa)
            ]
        )
    incompatible = model.medusa_head.load_state_dict(module_state_dict, strict=False)
    if "hydra_prefix_scale" in state_dict and hasattr(model, "hydra_prefix_scale"):
        with torch.no_grad():
            model.hydra_prefix_scale.copy_(
                state_dict["hydra_prefix_scale"].to(
                    device=model.hydra_prefix_scale.device,
                    dtype=model.hydra_prefix_scale.dtype,
                )
            )
    elif getattr(model, "draft_head_type", "medusa") == "hydra":
        model.draft_head_type = "medusa"
        warnings.warn(
            "Hydra draft mode requested but no hydra_prefix_scale was found in "
            "the head sidecar; falling back to standard Medusa heads.",
            RuntimeWarning,
        )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        warnings.warn(
            "Medusa head sidecar did not load cleanly: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}",
            RuntimeWarning,
        )
    return incompatible


def _compute_one_draft_head(model, head_idx, hidden_states):
    if getattr(model, "medusa_head_uses_base_lm_head", False):
        return model.base_model.lm_head(model.medusa_head[head_idx](hidden_states))
    return model.medusa_head[head_idx](hidden_states)


def _compute_medusa_logits(model, hidden_states, draft_head_type=None):
    draft_head_type = _normalize_draft_head_type(
        draft_head_type
        if draft_head_type is not None
        else getattr(model, "_active_draft_head_type", getattr(model, "draft_head_type", "medusa"))
    )
    if draft_head_type == "medusa":
        return torch.stack(
            [_compute_one_draft_head(model, i, hidden_states) for i in range(model.medusa)],
            dim=0,
        )

    prev_embed = None
    logits = []
    embed_tokens = model.base_model.get_input_embeddings()
    scale = getattr(model, "hydra_prefix_scale", None)
    for i in range(model.medusa):
        conditioned = hidden_states
        if prev_embed is not None and scale is not None:
            prefix_scale = scale[i].to(device=hidden_states.device, dtype=hidden_states.dtype)
            conditioned = conditioned + prev_embed.to(hidden_states.dtype) * prefix_scale.view(1, 1, -1)
        head_logits = _compute_one_draft_head(model, i, conditioned)
        logits.append(head_logits)
        # Hydra scaffolding: later heads can condition on the earlier speculative
        # top-1 token. The zero-initialized scale keeps legacy Medusa behavior.
        prev_token = torch.argmax(head_logits.detach(), dim=-1)
        prev_embed = embed_tokens(prev_token)
    return torch.stack(logits, dim=0)


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


def _safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_tree_calibration(path):
    if not path:
        return []
    path = os.path.expanduser(str(path))
    if not os.path.exists(path):
        warnings.warn(
            f"tree_calibration_path does not exist: {path}; using fixed tree policy.",
            RuntimeWarning,
        )
        return []
    try:
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                payload = payload.get("rows") or payload.get("calibration") or []
            rows = list(payload) if isinstance(payload, list) else []
        else:
            with open(path, "r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
    except Exception as exc:
        warnings.warn(
            f"Could not read tree calibration file {path}: {exc}; using fixed tree policy.",
            RuntimeWarning,
        )
        return []

    normalized = []
    for row in rows:
        prefix = _safe_float(row.get("prefix_match_vs_base"), 1.0)
        if prefix < 0.999:
            continue
        choice_limit = _safe_int(row.get("choice_limit", row.get("medusa_choice_limit")), 0)
        max_depth = _safe_int(row.get("max_depth", row.get("choice_max_depth")), 0)
        accepted = _safe_float(row.get("accepted_tokens_per_step"), 0.0)
        nodes = _safe_float(row.get("verified_nodes_per_step"), 0.0)
        tps = _safe_float(row.get("tps"), 0.0)
        if tps <= 0.0:
            continue
        normalized.append(
            {
                "choice_limit": choice_limit,
                "max_depth": max_depth,
                "accepted_tokens_per_step": accepted,
                "verified_nodes_per_step": nodes,
                "tps": tps,
            }
        )
    return normalized


def _select_calibrated_tree_policy(rows, node_budget=0):
    candidates = [row for row in rows if int(row["choice_limit"]) > 0]
    if not candidates:
        return None
    node_budget = int(node_budget)
    budgeted = [
        row
        for row in candidates
        if node_budget <= 0 or row["verified_nodes_per_step"] <= float(node_budget)
    ]
    pool = budgeted or candidates
    primary = max(
        pool,
        key=lambda row: (
            row["tps"],
            row["accepted_tokens_per_step"],
            -row["verified_nodes_per_step"],
        ),
    )
    larger = [
        row for row in candidates if int(row["choice_limit"]) > int(primary["choice_limit"])
    ]
    balanced = None
    if larger:
        balanced = max(
            larger,
            key=lambda row: (
                row["accepted_tokens_per_step"],
                row["tps"],
                int(row["choice_limit"]),
            ),
        )
    accept_threshold = max(0.0, float(primary["accepted_tokens_per_step"]) * 0.85)
    return {
        "primary_choice_limit": int(primary["choice_limit"]),
        "balanced_choice_limit": int(balanced["choice_limit"]) if balanced else 0,
        "max_depth": int(primary["max_depth"]),
        "accept_threshold": accept_threshold,
    }


_CONTEXT_WINDOW_FIELDS = (
    "max_position_embeddings",
    "n_positions",
    "seq_length",
    "max_seq_len",
    "max_sequence_length",
)
_CONTEXT_WINDOW_SENTINEL = 1_000_000_000


def _coerce_context_length(value):
    try:
        if value is None:
            return 0
        if torch.is_tensor(value):
            value = value.item()
        value = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return 0
    if value <= 0 or value >= _CONTEXT_WINDOW_SENTINEL:
        return 0
    return value


def infer_model_context_window(config, tokenizer=None, default=0):
    """
    Best-effort context-window inference for generation/benchmark planning.

    Tokenizers sometimes expose a huge sentinel for "unknown"; those values are
    ignored so Llama 3.x configs with real `max_position_embeddings` win.
    """
    candidates = []
    for field in _CONTEXT_WINDOW_FIELDS:
        length = _coerce_context_length(getattr(config, field, None))
        if length > 0:
            candidates.append(length)

    rope_scaling = getattr(config, "rope_scaling", None)
    if not candidates and isinstance(rope_scaling, dict):
        factor = _safe_float(rope_scaling.get("factor"), 0.0)
        original = _coerce_context_length(
            rope_scaling.get("original_max_position_embeddings")
            or rope_scaling.get("original_max_position")
        )
        if factor > 1.0 and original > 0:
            candidates.append(int(original * factor))

    if tokenizer is not None:
        length = _coerce_context_length(getattr(tokenizer, "model_max_length", None))
        if length > 0:
            candidates.append(length)

    fallback = _coerce_context_length(default)
    if fallback > 0:
        candidates.append(fallback)
    return max(candidates) if candidates else 0


def resolve_turbo_kv_cache_plan(
    config,
    tokenizer=None,
    input_length=0,
    max_steps=512,
    max_path_depth=1,
    tree_node_count=0,
    turbo_kv_max_length=2048,
    turbo_kv_use_model_context=False,
    packed_kv_qjl_requested=False,
    turbo_kv_qjl_min_kv_len=16384,
):
    """
    Resolve cache preallocation and packed KV-QJL availability.

    This is intentionally pure so Llama 3.2 long-context setup can be tested
    without loading model weights or trained Medusa heads.
    """
    input_length = max(0, int(input_length))
    max_steps = max(0, int(max_steps))
    max_path_depth = max(1, int(max_path_depth))
    tree_node_count = max(0, int(tree_node_count))
    worst_case_new_tokens = int(max_steps * (max_path_depth + 1))
    required_kv_len = int(input_length + worst_case_new_tokens + tree_node_count + 8)
    requested_kv_len = max(0, int(turbo_kv_max_length or 0))
    model_context_window = infer_model_context_window(config, tokenizer=tokenizer)

    effective_kv_max_length = max(requested_kv_len, required_kv_len)
    if turbo_kv_use_model_context and model_context_window > 0:
        effective_kv_max_length = max(effective_kv_max_length, model_context_window)

    effective_kv_qjl_min_len = max(0, int(turbo_kv_qjl_min_kv_len))
    use_packed_kv_qjl = bool(
        packed_kv_qjl_requested
        and effective_kv_max_length >= effective_kv_qjl_min_len
    )
    return {
        "model_context_window": int(model_context_window),
        "required_kv_len": int(required_kv_len),
        "requested_kv_len": int(requested_kv_len),
        "effective_kv_max_length": int(effective_kv_max_length),
        "effective_kv_qjl_min_len": int(effective_kv_qjl_min_len),
        "use_packed_kv_qjl": bool(use_packed_kv_qjl),
        "worst_case_new_tokens": int(worst_case_new_tokens),
        "tree_node_count": int(tree_node_count),
    }


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
        self.draft_head_type = _normalize_draft_head_type(
            getattr(config, "draft_head_type", "medusa")
        )
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
        self.hydra_prefix_scale = nn.Parameter(
            torch.zeros(medusa_num_heads, self.hidden_size)
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
            medusa_head_state_dict = None
            try:
                medusa_head_state_dict = _load_medusa_head_state_dict(pretrained_model_name_or_path)
                config.medusa_num_heads = _infer_medusa_num_heads_from_state_dict(
                    medusa_head_state_dict,
                    getattr(config, "medusa_num_heads", 5),
                )
            except Exception:
                pass
            model = super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
            if medusa_head_state_dict is not None:
                _load_medusa_head_into_model(model, medusa_head_state_dict)
            return model
        except:
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            medusa_head_state_dict = None
            try:
                medusa_head_state_dict = _load_medusa_head_state_dict(pretrained_model_name_or_path)
            except Exception:
                warnings.warn(
                    "No Medusa head sidecar found; initializing Medusa heads from scratch.",
                    RuntimeWarning,
                )
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.medusa_num_heads = (
                _infer_medusa_num_heads_from_state_dict(
                    medusa_head_state_dict,
                    getattr(config, "medusa_num_heads", 5),
                )
                if medusa_head_state_dict is not None
                else int(getattr(config, "medusa_num_heads", 5))
            )
            base_model_config.medusa_num_layers = config.medusa_num_layers
            base_model_config.version = getattr(config, "version", None)
            base_model_config.medusa_head_uses_base_lm_head = _medusa_uses_base_lm_head(config)
            base_model_config.draft_head_type = getattr(config, "draft_head_type", "medusa")
            model = super().from_pretrained(
                config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            if medusa_head_state_dict is not None:
                _load_medusa_head_into_model(model, medusa_head_state_dict)
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
        return_outputs=False,
        last_token_logits=False,
        draft_head_type=None,
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
        medusa_logits = (
            _compute_medusa_logits(self, hidden_states, draft_head_type=draft_head_type)
            if return_medusa_logits
            else None
        )
        if output_orig:
            return medusa_logits, outputs, orig
        if return_outputs:
            return medusa_logits, outputs
        return medusa_logits
    def get_medusa_choice(self, model_name, preset=None):
        if preset:
            preset_key = str(preset).lower()
            if preset_key not in MEDUSA_CHOICE_PRESETS:
                available = ", ".join(sorted(MEDUSA_CHOICE_PRESETS))
                raise ValueError(f"Unknown Medusa choice preset '{preset}'. Available presets: {available}")
            return [tuple(path) for path in MEDUSA_CHOICE_PRESETS[preset_key]]

        model_name = str(model_name).lower()
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

    def resolve_medusa_choices(
        self,
        medusa_choices=None,
        medusa_choice_preset=None,
        medusa_choice_limit=0,
        medusa_choice_max_depth=0,
    ):
        if medusa_choices is None:
            medusa_choices = self.get_medusa_choice(
                self.base_model_name_or_path,
                preset=medusa_choice_preset,
            )

        resolved = [tuple(path) for path in medusa_choices]
        max_depth = int(medusa_choice_max_depth)
        if max_depth > 0:
            resolved = [path for path in resolved if len(path) <= max_depth]

        choice_limit = int(medusa_choice_limit)
        if choice_limit > 0:
            resolved = resolved[:choice_limit]

        if not resolved:
            raise ValueError("Medusa choice resolution produced an empty tree.")
        return resolved

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        max_new_tokens=0,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=None,
        medusa_choice_preset=None,
        medusa_choice_limit=0,
        medusa_choice_max_depth=0,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True,
        turbo_auto=False,
        turbo_fast_preset=False,
        turbo_fused_lm_head_argmax=None,
        turbo_fused_lm_head_chunk_size=4096,
        turbo_adaptive_tree=False,
        turbo_adaptive_tree_balanced_limit=32,
        turbo_adaptive_tree_confidence_threshold=0.60,
        turbo_adaptive_tree_check_interval=4,
        turbo_adaptive_tree_accept_threshold=0.0,
        turbo_adaptive_tree_ema_alpha=0.30,
        turbo_quant=False,
        turbo_kv_compression=False,
        turbo_prune_keep=16,
        turbo_prune_min=12,
        turbo_prune_max=24,
        turbo_fallback_full_tree=True,
        turbo_fallback_accept_threshold=0,
        turbo_prune_confidence_margin=0.50,
        turbo_prune_prescreen_margin=-1.0,
        turbo_prune_min_fraction=0.0,
        turbo_prune_min_node_fraction=0.15,
        turbo_prune_node_budget=40,
        turbo_prune_acceptance_prune_threshold=0.0,
        turbo_prune_acceptance_keep_threshold=0.0,
        turbo_prune_acceptance_dynamic=False,
        turbo_prune_acceptance_dynamic_prune_min=0.10,
        turbo_prune_acceptance_dynamic_prune_max=0.45,
        turbo_prune_acceptance_dynamic_keep_min=0.45,
        turbo_prune_acceptance_dynamic_keep_max=0.70,
        turbo_prune_decisive_margin=1.5,
        turbo_prune_decisive_keep=8,
        turbo_prune_use_qjl=True,
        turbo_prune_auto_disable_after=4,
        turbo_prune_use_kv_qjl=False,
        turbo_kv_qjl_dim=128,
        turbo_kv_qjl_layer=-1,
        turbo_kv_qjl_keep_fraction=0.55,
        turbo_kv_qjl_weight=0.05,
        turbo_kv_qjl_min_kv_len=16384,
        turbo_kv_qjl_medusa_pool_fraction=0.80,
        turbo_kv_qjl_medusa_anchor_keep=2,
        turbo_packed_kv_qjl_auto_disable_after=2,
        turbo_force_full_tree_fast_verifier=False,
        turbo_lazy_tree_medusa_logits=True,
        turbo_skip_threshold_high=1.1,
        turbo_skip_threshold_low=-0.1,
        turbo_kv_max_length=2048,
        turbo_kv_use_model_context=False,
        turbo_kv_quant_mode="polar",
        turbo_radius_bits=2,
        turbo_theta_bits=4,
        turbo_polar_levels=4,
        turbo_vq_bits=4,
        turbo_vq_key_bits=None,
        turbo_vq_outlier_bits=4,
        turbo_vq_key_outlier_bits=None,
        turbo_vq_outlier_channels=0,
        turbo_vq_outlier_indices=None,
        turbo_vq_residual_dim=128,
        turbo_vq_residual_scale=1.0,
        turbo_hybrid_hot_window=512,
        turbo_runtime_dequant_cache=True,
        turbo_compile_decode=False,
        turbo_quant_seed=0,
        turbo_qjl_dim=64,
        draft_head_type=None,
        tree_policy="fixed",
        tree_calibration_path=None,
        stream=True,
        collect_stats=False,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            max_new_tokens (int, optional): Stop after this many generated token
                IDs. `0` keeps the historical max-steps-only behavior. When a
                Medusa step would overrun the target, the accepted prefix is
                shortened so fixed-token benchmarks compare equal token budgets.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            medusa_choice_preset (str, optional): Named choice-tree preset. Useful values
                include "tinyllama_1b_fast_24" and "tinyllama_1b_balanced_32".
            medusa_choice_limit (int, optional): Keep only the first N paths from the
                resolved choice tree. `24` was the fastest TinyLlama 1B local setting
                in the benchmark sweep.
            medusa_choice_max_depth (int, optional): Drop choice paths deeper than this
                before applying `medusa_choice_limit`. `0` keeps the model default depth.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
            top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
            sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
            fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
            turbo_auto (bool, optional): SGLang/EAGLE-inspired production profile:
                fast 24-choice verification by default, no QJL planner, no fused
                LM-head argmax by default. Local fixed-token sweeps showed that
                smaller/adaptive trees lose acceptance and 28+ choices can drift.
            turbo_fast_preset (bool, optional): Use the fastest local profile found so far:
                24-choice tree plus greedy full-tree fast verifier, with pruning disabled.
            turbo_fused_lm_head_argmax (bool, optional): In greedy Turbo verification,
                compute verifier argmax ids directly from hidden states and the LM-head
                weights, then materialize logits only for the accepted node. `None`
                enables it only for `turbo_fast_preset`.
            turbo_fused_lm_head_chunk_size (int, optional): Fallback vocab chunk size
                for the non-Triton streaming LM-head argmax path.
            turbo_adaptive_tree (bool, optional): Use the fast tree by default and
                switch to a larger balanced tree on low Medusa confidence steps.
            turbo_adaptive_tree_balanced_limit (int, optional): Choice count for the
                larger adaptive tree. Defaults to 32.
            turbo_adaptive_tree_confidence_threshold (float, optional): If the
                two-token softmax confidence from Medusa heads falls below this,
                use the balanced tree for the current step.
            turbo_adaptive_tree_check_interval (int, optional): Recompute the
                adaptive tree decision every N steps and reuse it between checks
                to avoid a CPU/GPU sync each step.
            turbo_adaptive_tree_accept_threshold (float, optional): Switch to the
                balanced tree when the recent accepted-token EMA falls below this.
                Set 0 to disable the acceptance-history trigger.
            turbo_adaptive_tree_ema_alpha (float, optional): Update rate for the
                accepted-token EMA used by adaptive tree sizing.
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
            turbo_prune_acceptance_prune_threshold (float, optional): Relative Medusa-prior
                confidence below which paths are treated as low-confidence pruning candidates.
                This is a proxy, not exact verifier acceptance probability.
            turbo_prune_acceptance_keep_threshold (float, optional): Relative Medusa-prior
                confidence above which paths are protected from pruning. For example, `0.5`
                keeps paths that look at least half as plausible as the current best path.
            turbo_prune_acceptance_dynamic (bool, optional): Use per-step dynamic acceptance
                thresholds. Sharp Medusa priors prune more aggressively; flat priors keep
                more branches or fall back.
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
            turbo_kv_use_model_context (bool, optional): If True, preallocate at least the
                model/tokenizer advertised context window. This is useful for Llama 3.2
                long-context runs where packed KV-QJL and TurboVQ should be initialized
                for the larger cache ahead of time.
            turbo_kv_quant_mode (str, optional): KV compression backend: "polar" for the
                recursive PolarQuant cache, or "turbo_vq" for random-rotation Lloyd-Max
                TurboQuantprod keys plus TurboQuantmse values.
            turbo_radius_bits (int, optional): Later-level angle bits for recursive
                PolarQuant.
            turbo_theta_bits (int, optional): First-level angle bits for recursive
                PolarQuant.
            turbo_polar_levels (int, optional): Recursive PolarQuant levels. The paper's
                practical implementation uses 4.
            turbo_vq_bits (int, optional): Scalar Lloyd-Max bits for "turbo_vq" KV compression.
            turbo_vq_key_bits (int, optional): Override key-cache Lloyd-Max index bits.
                For TurboQuantprod, set this to `b - 1` and use a full-head QJL residual
                to obtain a total b-bit key quantizer while keeping value-cache
                TurboQuantmse at `turbo_vq_bits`.
            turbo_vq_residual_dim (int, optional): 1-bit QJL residual sketch dimension for
                TurboQuant key-cache inner-product correction. A negative value means the
                full attention head dimension, matching TurboQuantprod Algorithm 2.
            turbo_vq_residual_scale (float, optional): Multiplier for the residual-QJL
                correction. `1.0` is the unbiased estimator; lower values can dampen
                sketch variance during calibration.
            turbo_vq_outlier_indices (list, optional): Pre-calibrated per-layer
                [key_indices, value_indices] channel lists for the outlier-aware
                TurboQuant KV recipe. If omitted, channels are selected from the
                first appended prefill batch.
            turbo_quant_seed (int, optional): Seed for TurboQuant rotations and QJL
                projections. Defaults to deterministic paper-debuggable matrices;
                change it or pass fresh values for multi-seed runs.
            turbo_runtime_dequant_cache (bool, optional): Keep an incremental dequantized shadow cache
                for fast attention when polar compression is enabled. Set False to use the
                direct compressed-KV Triton attention path instead of materializing FP16 K/V.
            turbo_compile_decode (bool, optional): Use torch.compile on tensorized polar decode kernel.
            turbo_qjl_dim (int, optional): Sketch dimension for 1-bit QJL sidecar path scoring.
            draft_head_type (str, optional): Draft-head family for speculative
                logits. `None` uses the model config. Supported values are
                "medusa" and "hydra"; Hydra mode is exact-safe because the
                base verifier still makes every acceptance decision.
            tree_policy (str, optional): "fixed" keeps the requested tree.
                "adaptive_calibrated" reads `tree_calibration_path` and chooses
                calibrated choice limits, falling back to the fixed full tree
                when calibration is missing or unusable.
            tree_calibration_path (str, optional): CSV/JSON calibration file
                produced by `calibrate_tree_policy.py`.
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
        effective_draft_head_type = _normalize_draft_head_type(
            draft_head_type
            if draft_head_type is not None
            else getattr(self, "draft_head_type", "medusa")
        )
        self._active_draft_head_type = effective_draft_head_type
        effective_tree_policy = str(tree_policy or "fixed").lower().strip()
        if effective_tree_policy not in {"fixed", "adaptive_calibrated"}:
            raise ValueError("tree_policy must be 'fixed' or 'adaptive_calibrated'.")
        explicit_medusa_choices = medusa_choices is not None

        if turbo_auto:
            turbo_fast_preset = True
            turbo_adaptive_tree = False
            turbo_fused_lm_head_argmax = False
            turbo_prune_use_kv_qjl = False
            turbo_prune_use_qjl = False

        if turbo_fast_preset:
            turbo_quant = True
            turbo_kv_compression = False
            turbo_force_full_tree_fast_verifier = True
            turbo_prune_use_kv_qjl = False
            if (
                medusa_choices is None
                and medusa_choice_preset is None
                and int(medusa_choice_limit) <= 0
            ):
                medusa_choice_limit = 24
        if turbo_fused_lm_head_argmax is None:
            turbo_fused_lm_head_argmax = bool(
                temperature == 0 and globals().get("lm_head_argmax_triton", None) is not None
            )

        calibrated_tree_policy = None
        if (
            effective_tree_policy == "adaptive_calibrated"
            and not explicit_medusa_choices
            and int(medusa_choice_limit) <= 0
        ):
            calibrated_rows = _read_tree_calibration(tree_calibration_path)
            calibrated_tree_policy = _select_calibrated_tree_policy(
                calibrated_rows,
                node_budget=turbo_prune_node_budget,
            )
            if calibrated_tree_policy is not None:
                medusa_choice_limit = int(calibrated_tree_policy["primary_choice_limit"])
                if int(medusa_choice_max_depth) <= 0 and int(calibrated_tree_policy["max_depth"]) > 0:
                    medusa_choice_max_depth = int(calibrated_tree_policy["max_depth"])
                turbo_quant = True
                turbo_kv_compression = False
                turbo_force_full_tree_fast_verifier = True
                turbo_prune_use_kv_qjl = False
                turbo_adaptive_tree = True
                turbo_adaptive_tree_balanced_limit = int(
                    calibrated_tree_policy["balanced_choice_limit"]
                )
                if float(turbo_adaptive_tree_accept_threshold) <= 0.0:
                    turbo_adaptive_tree_accept_threshold = float(
                        calibrated_tree_policy["accept_threshold"]
                    )

        # Cache medusa buffers (the fixed patterns for tree attention)
        medusa_choices = self.resolve_medusa_choices(
            medusa_choices=medusa_choices,
            medusa_choice_preset=medusa_choice_preset,
            medusa_choice_limit=medusa_choice_limit,
            medusa_choice_max_depth=medusa_choice_max_depth,
        )

        adaptive_medusa_choices = None
        if turbo_adaptive_tree and not explicit_medusa_choices:
            adaptive_medusa_choices = self.resolve_medusa_choices(
                medusa_choices=None,
                medusa_choice_preset=medusa_choice_preset,
                medusa_choice_limit=int(turbo_adaptive_tree_balanced_limit),
                medusa_choice_max_depth=medusa_choice_max_depth,
            )
            if adaptive_medusa_choices == medusa_choices:
                adaptive_medusa_choices = None

        def get_cached_medusa_buffers(choice_set):
            key = (str(self.base_model.device), tuple(tuple(path) for path in choice_set))
            if not hasattr(self, "medusa_buffer_cache"):
                self.medusa_buffer_cache = {}
            cache = self.medusa_buffer_cache
            if key not in cache:
                cache[key] = generate_medusa_buffers(
                    choice_set, device=self.base_model.device
                )
            return cache[key], key

        medusa_buffers, medusa_choices_key = get_cached_medusa_buffers(medusa_choices)
        adaptive_medusa_buffers = None
        adaptive_medusa_choices_key = None
        if adaptive_medusa_choices is not None:
            adaptive_medusa_buffers, adaptive_medusa_choices_key = get_cached_medusa_buffers(
                adaptive_medusa_choices
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices
        if (
            not hasattr(self, "turbo_pruned_layout_cache")
            or getattr(self, "turbo_pruned_layout_cache_key", None) != medusa_choices_key
        ):
            self.turbo_pruned_layout_cache = {}
            self.turbo_pruned_layout_cache_key = medusa_choices_key

        # Medusa may accept multiple tokens per step, so max_steps alone can
        # underestimate cache length. Use tree depth for a safe upper bound.
        max_path_depth = max((len(path) for path in medusa_choices), default=1)
        if adaptive_medusa_choices is not None:
            max_path_depth = max(
                max_path_depth,
                max((len(path) for path in adaptive_medusa_choices), default=1),
            )
        max_tree_node_count = int(medusa_buffers["tree_indices"].numel())
        if adaptive_medusa_buffers is not None:
            max_tree_node_count = max(
                max_tree_node_count,
                int(adaptive_medusa_buffers["tree_indices"].numel()),
            )
        use_compressed_kv = bool(turbo_quant and turbo_kv_compression)
        requested_kv_quant_mode = str(turbo_kv_quant_mode).lower()
        requested_cache_mode = requested_kv_quant_mode if use_compressed_kv else "fp16"
        effective_turbo_vq_key_bits = (
            int(turbo_vq_bits)
            if turbo_vq_key_bits is None
            else int(turbo_vq_key_bits)
        )
        effective_turbo_vq_key_outlier_bits = (
            min(8, max(1, effective_turbo_vq_key_bits + 1))
            if turbo_vq_key_outlier_bits is None
            else int(turbo_vq_key_outlier_bits)
        )

        def _outlier_indices_signature(indices):
            if indices is None:
                return None
            signature = []
            for layer in indices:
                layer_sig = []
                for item in layer:
                    if item is None:
                        layer_sig.append(())
                    elif torch.is_tensor(item):
                        layer_sig.append(tuple(int(x) for x in item.detach().cpu().flatten().tolist()))
                    else:
                        layer_sig.append(tuple(int(x) for x in item))
                signature.append(tuple(layer_sig))
            return tuple(signature)

        turbo_vq_outlier_signature = _outlier_indices_signature(turbo_vq_outlier_indices)
        packed_kv_qjl_requested = bool(
            turbo_quant
            and turbo_prune_use_kv_qjl
            and not turbo_force_full_tree_fast_verifier
            and not use_compressed_kv
        )
        kv_cache_plan = resolve_turbo_kv_cache_plan(
            self.config,
            tokenizer=getattr(self, "tokenizer", None),
            input_length=int(input_ids.shape[1]),
            max_steps=max_steps,
            max_path_depth=max_path_depth,
            tree_node_count=max_tree_node_count,
            turbo_kv_max_length=turbo_kv_max_length,
            turbo_kv_use_model_context=turbo_kv_use_model_context,
            packed_kv_qjl_requested=packed_kv_qjl_requested,
            turbo_kv_qjl_min_kv_len=turbo_kv_qjl_min_kv_len,
        )
        required_kv_len = int(kv_cache_plan["required_kv_len"])
        effective_kv_max_length = int(kv_cache_plan["effective_kv_max_length"])
        effective_kv_qjl_min_len = int(kv_cache_plan["effective_kv_qjl_min_len"])
        use_packed_kv_qjl = bool(kv_cache_plan["use_packed_kv_qjl"])
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
                    and getattr(self, "kv_cache_polar_levels", None) == int(turbo_polar_levels)
                    and getattr(self, "kv_cache_vq_bits", None) == turbo_vq_bits
                    and getattr(self, "kv_cache_vq_key_bits", None) == effective_turbo_vq_key_bits
                    and getattr(self, "kv_cache_vq_outlier_bits", None) == int(turbo_vq_outlier_bits)
                    and getattr(self, "kv_cache_vq_key_outlier_bits", None) == int(effective_turbo_vq_key_outlier_bits)
                    and getattr(self, "kv_cache_vq_outlier_channels", None) == int(turbo_vq_outlier_channels)
                    and getattr(self, "kv_cache_vq_outlier_signature", None) == turbo_vq_outlier_signature
                    and getattr(self, "kv_cache_vq_residual_dim", None) == turbo_vq_residual_dim
                    and getattr(self, "kv_cache_vq_residual_scale", None) == float(turbo_vq_residual_scale)
                    and getattr(self, "kv_cache_hybrid_hot_window", None) == int(turbo_hybrid_hot_window)
                    and getattr(self, "kv_runtime_dequant_cache", None) == turbo_runtime_dequant_cache
                    and getattr(self, "kv_compile_decode", None) == turbo_compile_decode
                    and getattr(self, "kv_cache_quant_seed", None) == int(turbo_quant_seed)
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
            reset_past_key_values(past_key_values)
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
                turbo_polar_levels=turbo_polar_levels,
                turbo_vq_bits=turbo_vq_bits,
                turbo_vq_key_bits=effective_turbo_vq_key_bits,
                turbo_vq_outlier_bits=turbo_vq_outlier_bits,
                turbo_vq_key_outlier_bits=effective_turbo_vq_key_outlier_bits,
                turbo_vq_outlier_channels=turbo_vq_outlier_channels,
                turbo_vq_outlier_indices=turbo_vq_outlier_indices,
                turbo_vq_residual_dim=turbo_vq_residual_dim,
                turbo_vq_residual_scale=turbo_vq_residual_scale,
                turbo_hybrid_hot_window=turbo_hybrid_hot_window,
                turbo_runtime_dequant_cache=turbo_runtime_dequant_cache,
                turbo_compile_decode=turbo_compile_decode,
                turbo_quant_seed=turbo_quant_seed,
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
            self.kv_cache_polar_levels = int(turbo_polar_levels)
            self.kv_cache_vq_bits = turbo_vq_bits
            self.kv_cache_vq_key_bits = effective_turbo_vq_key_bits
            self.kv_cache_vq_outlier_bits = int(turbo_vq_outlier_bits)
            self.kv_cache_vq_key_outlier_bits = int(effective_turbo_vq_key_outlier_bits)
            self.kv_cache_vq_outlier_channels = int(turbo_vq_outlier_channels)
            self.kv_cache_vq_outlier_signature = turbo_vq_outlier_signature
            self.kv_cache_vq_residual_dim = turbo_vq_residual_dim
            self.kv_cache_vq_residual_scale = float(turbo_vq_residual_scale)
            self.kv_cache_hybrid_hot_window = int(turbo_hybrid_hot_window)
            self.kv_runtime_dequant_cache = turbo_runtime_dequant_cache
            self.kv_compile_decode = turbo_compile_decode
            self.kv_cache_quant_seed = int(turbo_quant_seed)
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
        adaptive_full_path_lengths = None
        adaptive_full_tree_node_count = 0
        adaptive_full_candidate_path_count = 0
        if adaptive_medusa_buffers is not None:
            adaptive_full_path_lengths = (
                adaptive_medusa_buffers["retrieve_indices"][:, 1:] >= 0
            ).sum(dim=1)
            adaptive_full_tree_node_count = int(adaptive_medusa_buffers["tree_indices"].numel())
            adaptive_full_candidate_path_count = int(adaptive_medusa_buffers["retrieve_indices"].shape[0])
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
                and getattr(self, "turbo_qjl_seed", None) == int(turbo_quant_seed)
            )
            if not scorer_reusable:
                self.turbo_qjl_scorer = QJLTokenSketchCache(
                    vocab_size=int(embed_weight.shape[0]),
                    hidden_size=int(embed_weight.shape[1]),
                    sketch_dim=int(turbo_qjl_dim),
                    device=embed_weight.device,
                    seed=int(turbo_quant_seed),
                )
                self.turbo_qjl_dim = int(turbo_qjl_dim)
                self.turbo_qjl_vocab_size = int(embed_weight.shape[0])
                self.turbo_qjl_hidden_size = int(embed_weight.shape[1])
                self.turbo_qjl_device = str(embed_weight.device)
                self.turbo_qjl_seed = int(turbo_quant_seed)
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
                last_token_logits=True,
            )
            query_state = query_state.detach()
        else:
            medusa_logits, logits = initialize_medusa(
                input_ids,
                self,
                medusa_buffers["medusa_attn_mask"],
                past_key_values,
                return_query_state=False,
                last_token_logits=True,
            )
            query_state = None

        new_token = 0
        target_new_tokens = max(0, int(max_new_tokens))
        last_round_token = 0
        stats = None
        if collect_stats:
            stats = {
                "decode_steps": 0,
                "generated_tokens": 0,
                "medusa_choice_count": int(len(medusa_choices)),
                "medusa_choice_max_depth": int(max_path_depth),
                "draft_head_type": effective_draft_head_type,
                "tree_policy": effective_tree_policy,
                "model_context_window": int(kv_cache_plan["model_context_window"]),
                "kv_cache_required_len": int(required_kv_len),
                "kv_cache_max_length": int(effective_kv_max_length),
                "kv_cache_use_model_context": int(bool(turbo_kv_use_model_context)),
                "packed_kv_qjl_enabled": int(bool(use_packed_kv_qjl)),
                "packed_kv_qjl_min_len": int(effective_kv_qjl_min_len),
                "turbo_auto_steps": 0,
                "adaptive_tree_steps": 0,
                "adaptive_tree_balanced_steps": 0,
                "adaptive_tree_primary_choice_count": int(len(medusa_choices)),
                "adaptive_tree_balanced_choice_count": int(
                    len(adaptive_medusa_choices)
                    if adaptive_medusa_choices is not None
                    else 0
                ),
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
                "fused_lm_head_argmax_steps": 0,
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

        eos_token_id = self.tokenizer.eos_token_id

        def new_tokens_have_eos(start_len):
            if eos_token_id is None or input_ids.shape[1] <= start_len:
                return False
            return bool((input_ids[0, start_len:] == eos_token_id).any().item())

        planner_full_tree_streak = 0
        planner_bypass_pruning = False
        packed_kv_qjl_fallback_streak = 0
        fused_lm_head = getattr(self.base_model, "lm_head", None)
        fused_lm_head_weight = getattr(fused_lm_head, "weight", None)
        use_fused_lm_head_argmax = bool(
            turbo_fused_lm_head_argmax
            and temperature == 0
            and fused_lm_head is not None
            and fused_lm_head_weight is not None
        )

        def evaluate_greedy_tree(tree_logits, tree_hidden, eval_candidates, eval_retrieve_indices, eval_path_lengths):
            if tree_logits is not None:
                return evaluate_posterior_greedy_from_tree(
                    tree_logits,
                    eval_candidates,
                    eval_retrieve_indices,
                    path_lengths=eval_path_lengths,
                )
            node_argmax = lm_head_argmax(
                tree_hidden,
                fused_lm_head_weight,
                chunk_size=turbo_fused_lm_head_chunk_size,
                prefer_triton=True,
            )
            if stats is not None:
                stats["fused_lm_head_argmax_steps"] += 1
            return evaluate_posterior_greedy_from_argmax(
                node_argmax,
                eval_candidates,
                eval_retrieve_indices,
                path_lengths=eval_path_lengths,
            )

        primary_medusa_buffers = medusa_buffers
        primary_full_path_lengths = full_path_lengths
        primary_full_tree_node_count = full_tree_node_count
        primary_full_candidate_path_count = full_candidate_path_count
        use_adaptive_tree = bool(
            turbo_adaptive_tree
            and adaptive_medusa_buffers is not None
            and turbo_quant
            and turbo_force_full_tree_fast_verifier
            and temperature == 0
        )
        adaptive_confidence_threshold = float(turbo_adaptive_tree_confidence_threshold)
        adaptive_check_interval = max(1, int(turbo_adaptive_tree_check_interval))
        adaptive_accept_threshold = float(turbo_adaptive_tree_accept_threshold)
        adaptive_ema_alpha = max(0.0, min(1.0, float(turbo_adaptive_tree_ema_alpha)))
        adaptive_use_balanced_tree = False
        adaptive_accept_ema = 1.0

        def record_adaptive_acceptance(accepted_tokens):
            nonlocal adaptive_accept_ema
            accepted_tokens = float(max(1, int(accepted_tokens)))
            adaptive_accept_ema = (
                (1.0 - adaptive_ema_alpha) * adaptive_accept_ema
                + adaptive_ema_alpha * accepted_tokens
            )

        for idx in range(max_steps):
            if target_new_tokens > 0 and new_token >= target_new_tokens:
                break
            step_input_len = input_ids.shape[1]
            if turbo_auto and stats is not None:
                stats["turbo_auto_steps"] += 1
            if use_adaptive_tree:
                medusa_buffers = primary_medusa_buffers
                full_path_lengths = primary_full_path_lengths
                full_tree_node_count = primary_full_tree_node_count
                full_candidate_path_count = primary_full_candidate_path_count
                if stats is not None:
                    stats["adaptive_tree_steps"] += 1
                if idx % adaptive_check_interval == 0:
                    head_top2 = torch.topk(medusa_logits[:, 0, -1], k=2, dim=-1).values
                    head_confidence = torch.sigmoid(
                        (head_top2[:, 0] - head_top2[:, 1]).to(torch.float32)
                    ).mean()
                    adaptive_use_balanced_tree = bool(
                        (head_confidence < adaptive_confidence_threshold).item()
                        or (
                            adaptive_accept_threshold > 0.0
                            and adaptive_accept_ema < adaptive_accept_threshold
                        )
                    )
                if adaptive_use_balanced_tree:
                    medusa_buffers = adaptive_medusa_buffers
                    full_path_lengths = adaptive_full_path_lengths
                    full_tree_node_count = adaptive_full_tree_node_count
                    full_candidate_path_count = adaptive_full_candidate_path_count
                    if stats is not None:
                        stats["adaptive_tree_balanced_steps"] += 1
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
                tree_topk=medusa_buffers.get("tree_topk"),
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
                                acceptance_prune_threshold=turbo_prune_acceptance_prune_threshold,
                                acceptance_keep_threshold=turbo_prune_acceptance_keep_threshold,
                                acceptance_threshold_dynamic=turbo_prune_acceptance_dynamic,
                                acceptance_dynamic_prune_min=turbo_prune_acceptance_dynamic_prune_min,
                                acceptance_dynamic_prune_max=turbo_prune_acceptance_dynamic_prune_max,
                                acceptance_dynamic_keep_min=turbo_prune_acceptance_dynamic_keep_min,
                                acceptance_dynamic_keep_max=turbo_prune_acceptance_dynamic_keep_max,
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
                            acceptance_prune_threshold=turbo_prune_acceptance_prune_threshold,
                            acceptance_keep_threshold=turbo_prune_acceptance_keep_threshold,
                            acceptance_threshold_dynamic=turbo_prune_acceptance_dynamic,
                            acceptance_dynamic_prune_min=turbo_prune_acceptance_dynamic_prune_min,
                            acceptance_dynamic_prune_max=turbo_prune_acceptance_dynamic_prune_max,
                            acceptance_dynamic_keep_min=turbo_prune_acceptance_dynamic_keep_min,
                            acceptance_dynamic_keep_max=turbo_prune_acceptance_dynamic_keep_max,
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
                    if use_adaptive_tree:
                        record_adaptive_acceptance(1)
                    if stats is not None:
                        stats["skip_gating_steps"] += 1
                    should_stop = new_tokens_have_eos(step_input_len)
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
                            compute_orig_logits=not use_fused_lm_head_argmax,
                            fast_attention_mask=True,
                        )
                        best_candidate, accept_length = evaluate_greedy_tree(
                            logits,
                            tree_hidden,
                            candidates,
                            medusa_buffers["retrieve_indices"],
                            full_path_lengths,
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
                                compute_orig_logits=not use_fused_lm_head_argmax,
                                fast_attention_mask=True,
                            )
                            best_candidate, accept_length = evaluate_greedy_tree(
                                logits,
                                tree_hidden,
                                candidates,
                                medusa_buffers["retrieve_indices"],
                                full_path_lengths,
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
                                compute_orig_logits=not use_fused_lm_head_argmax,
                                fast_attention_mask=True,
                            )
                            best_candidate, accept_length = evaluate_greedy_tree(
                                logits,
                                tree_hidden,
                                pruned["candidates"],
                                pruned["retrieve_indices"],
                                pruned["path_lengths"],
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
                                    compute_orig_logits=not use_fused_lm_head_argmax,
                                    fast_attention_mask=True,
                                )
                                best_candidate, accept_length = evaluate_greedy_tree(
                                    logits,
                                    tree_hidden,
                                    candidates,
                                    medusa_buffers["retrieve_indices"],
                                    full_path_lengths,
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
                    if target_new_tokens > 0:
                        max_accept_len = max(0, target_new_tokens - new_token) - 1
                        if torch.is_tensor(accept_length):
                            accept_length = accept_length.clamp_max(max_accept_len)
                        else:
                            accept_length = min(int(accept_length), max_accept_len)
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
                        lm_head=fused_lm_head if logits is None else None,
                        tree_hidden=tree_hidden if logits is None else None,
                    )
                    node_idx = update_retrieve_indices[best_candidate, accept_length].reshape(1)
                    accepted_hidden = None
                    if medusa_logits is None or need_turbo_query_state:
                        accepted_hidden = tree_hidden.index_select(0, node_idx)
                    if medusa_logits is None:
                        medusa_logits = _compute_medusa_logits(
                            self,
                            accepted_hidden.unsqueeze(0),
                            draft_head_type=effective_draft_head_type,
                        )
                    if need_turbo_query_state:
                        query_state = accepted_hidden.detach()
                    accept_idx = int(accept_length.item()) if torch.is_tensor(accept_length) else int(accept_length)
                else:
                    if target_new_tokens > 0:
                        max_accept_len = max(0, target_new_tokens - new_token) - 1
                        if torch.is_tensor(accept_length):
                            accept_length = accept_length.clamp_max(max_accept_len)
                        else:
                            accept_length = min(int(accept_length), max_accept_len)
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
                        compute_medusa_logits=False,
                        compute_orig_logits=not use_fused_lm_head_argmax,
                        fast_attention_mask=True,
                    )
                    best_candidate, accept_length = evaluate_greedy_tree(
                        logits,
                        tree_hidden,
                        candidates,
                        medusa_buffers["retrieve_indices"],
                        full_path_lengths,
                    )
                    if target_new_tokens > 0:
                        max_accept_len = max(0, target_new_tokens - new_token) - 1
                        if torch.is_tensor(accept_length):
                            accept_length = accept_length.clamp_max(max_accept_len)
                        else:
                            accept_length = min(int(accept_length), max_accept_len)
                    input_ids, logits, medusa_logits, new_token = update_inference_inputs_from_tree(
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
                        lm_head=fused_lm_head if logits is None else None,
                        tree_hidden=tree_hidden if logits is None else None,
                    )
                    accept_idx = int(accept_length.item()) if torch.is_tensor(accept_length) else int(accept_length)
                    node_idx = medusa_buffers["retrieve_indices"][best_candidate, accept_idx].reshape(1)
                    accepted_hidden = tree_hidden.index_select(0, node_idx)
                    medusa_logits = _compute_medusa_logits(
                        self,
                        accepted_hidden.unsqueeze(0),
                        draft_head_type=effective_draft_head_type,
                    )
                    query_state = accepted_hidden.detach()
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
                    if target_new_tokens > 0:
                        max_accept_len = max(0, target_new_tokens - new_token) - 1
                        if torch.is_tensor(accept_length):
                            accept_length = accept_length.clamp_max(max_accept_len)
                        else:
                            accept_length = min(int(accept_length), max_accept_len)
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

            if use_adaptive_tree:
                record_adaptive_acceptance(accept_idx + 1)

            should_stop = new_tokens_have_eos(step_input_len)
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
