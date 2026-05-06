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
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
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
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # Create a list of Medusa heads
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
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
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.medusa_num_heads = getattr(config, "medusa_num_heads", 5)
            base_model_config.medusa_num_layers = config.medusa_num_layers
            model = super().from_pretrained(
                config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            medusa_head_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
            if os.path.exists(medusa_head_path):
                filename = medusa_head_path
            else:
                filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt")
            medusa_head_state_dict = torch.load(filename, map_location="cpu")
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
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        medusa_logits = []
        # TODO: Consider parallelizing this loop for efficiency?
        for i in range(self.medusa):
            medusa_logits.append(self.medusa_head[i](hidden_states))
        if output_orig:
            return torch.stack(medusa_logits, dim=0), outputs, orig
        return torch.stack(medusa_logits, dim=0)
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
        turbo_skip_threshold_high=1.1,
        turbo_skip_threshold_low=-0.1,
        turbo_kv_max_length=2048,
        turbo_radius_bits=8,
        turbo_theta_bits=8,
        turbo_runtime_dequant_cache=True,
        turbo_compile_decode=False,
        turbo_qjl_dim=128,
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
            turbo_kv_compression (bool, optional): Enable polar KV compression for attention cache.
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
            turbo_skip_threshold_high (float, optional): If pass-1 top prob exceeds this, skip pass-2 and accept one token.
            turbo_skip_threshold_low (float, optional): If pass-1 top prob is below this, skip pass-2 and do greedy one token.
                Defaults disable skip-gating (high>1, low<0) to preserve acceptance quality.
            turbo_kv_max_length (int, optional): Maximum cache length for KV pre-allocation.
            turbo_radius_bits (int, optional): Radius quantization bits for polar KV.
            turbo_theta_bits (int, optional): Angle quantization bits for polar KV.
            turbo_runtime_dequant_cache (bool, optional): Keep an incremental dequantized shadow cache
                for fast attention when polar compression is enabled.
            turbo_compile_decode (bool, optional): Use torch.compile on tensorized polar decode kernel.
            turbo_qjl_dim (int, optional): Sketch dimension for 1-bit QJL sidecar path scoring.
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

        # Medusa may accept multiple tokens per step, so max_steps alone can
        # underestimate cache length. Use tree depth for a safe upper bound.
        max_path_depth = max((len(path) for path in medusa_choices), default=1)
        worst_case_new_tokens = int(max_steps * (max_path_depth + 1))
        required_kv_len = int(input_ids.shape[1] + worst_case_new_tokens + 8)
        effective_kv_max_length = max(int(turbo_kv_max_length), required_kv_len)
        use_polar_kv = bool(turbo_quant and turbo_kv_compression)
        requested_cache_mode = "polar" if use_polar_kv else "fp16"
        cache_reusable = (
            hasattr(self, "past_key_values")
            and getattr(self, "kv_cache_mode", None) == requested_cache_mode
            and getattr(self, "kv_cache_max_length", None) == effective_kv_max_length
            and (
                not use_polar_kv
                or (
                    getattr(self, "kv_cache_radius_bits", None) == turbo_radius_bits
                    and getattr(self, "kv_cache_theta_bits", None) == turbo_theta_bits
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
                turbo_quant=use_polar_kv,
                turbo_radius_bits=turbo_radius_bits,
                turbo_theta_bits=turbo_theta_bits,
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
            self.kv_runtime_dequant_cache = turbo_runtime_dequant_cache
            self.kv_compile_decode = turbo_compile_decode

        input_len = input_ids.shape[1]
        full_path_lengths = (medusa_buffers["retrieve_indices"][:, 1:] >= 0).sum(dim=1)
        embed_weight = None
        qjl_scorer = None
        if turbo_quant:
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
        medusa_logits, logits, query_state = initialize_medusa(
            input_ids,
            self,
            medusa_buffers["medusa_attn_mask"],
            past_key_values,
            return_query_state=True,
        )
        query_state = query_state.detach()

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
                approx_scores, _ = estimate_tree_candidate_scores_1bit(
                    medusa_logits,
                    logits,
                    medusa_buffers["tree_indices"],
                    medusa_buffers["retrieve_indices"],
                    candidates=candidates,
                    query_state=query_state,
                    qjl_scorer=qjl_scorer,
                    embed_weight=embed_weight,
                )
                selected_paths = select_topk_paths_for_verification(
                    approx_scores,
                    keep_target=turbo_prune_keep,
                    min_keep=turbo_prune_min,
                    max_keep=turbo_prune_max,
                    retrieve_indices=medusa_buffers["retrieve_indices"],
                    tree_indices=medusa_buffers["tree_indices"],
                )
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
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    new_token += 1
                    yield {
                        "text": self.tokenizer.decode(
                            input_ids[0, input_len:],
                            skip_special_tokens=True,
                            spaces_between_special_tokens=False,
                            clean_up_tokenization_spaces=True,
                        )
                    }
                    if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                        break
                    continue

                verify_full_tree = should_verify_full_tree(
                    approx_scores,
                    margin_scale=turbo_prune_confidence_margin,
                )
                if verify_full_tree:
                    # Ambiguous pass-1 scores: skip the pruned verifier entirely
                    # to avoid pruned-forward + fallback-forward double work.
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
                    pruned = build_pruned_medusa_buffers(
                        tree_candidates,
                        medusa_buffers["retrieve_indices"],
                        medusa_buffers["medusa_position_ids"],
                        medusa_buffers["medusa_attn_mask"],
                        selected_paths,
                    )

                    # High-accuracy verification on pruned tree first.
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
                        # Pruning was too conservative for this step. Rewind the
                        # cache length and verify the full tree so accuracy/acceptance
                        # quality stays equivalent to Medusa base.
                        current_length_data.fill_(input_ids.shape[1])
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
                )
                best_idx = int(best_candidate.item()) if torch.is_tensor(best_candidate) else int(best_candidate)
                accept_idx = int(accept_length.item()) if torch.is_tensor(accept_length) else int(accept_length)
                query_state = tree_hidden[best_idx, accept_idx].unsqueeze(0).detach()
            else:
                # Use tree attention to verify the candidates and get predictions
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
                )
                best_idx = int(best_candidate.item()) if torch.is_tensor(best_candidate) else int(best_candidate)
                accept_idx = int(accept_length.item()) if torch.is_tensor(accept_length) else int(accept_length)
                query_state = tree_hidden[best_idx, accept_idx].unsqueeze(0).detach()

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break


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
