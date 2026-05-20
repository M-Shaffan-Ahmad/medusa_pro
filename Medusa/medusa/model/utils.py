import math
import os
import torch
import torch.nn.functional as F

try:
    from .triton_kernels import (
        copy_selected_kv_cache_triton,
        greedy_accept_from_argmax_triton,
        greedy_tree_posterior_triton,
        lm_head_argmax_triton,
        materialize_pruned_medusa_triton,
        node_budget_select_triton,
        packed_kv_qjl_node_scores_triton,
        qjl_path_scores_triton,
        turbo_qjl_select_paths_triton,
    )
except Exception:  # pragma: no cover - optional CUDA/Triton acceleration
    copy_selected_kv_cache_triton = None
    greedy_accept_from_argmax_triton = None
    greedy_tree_posterior_triton = None
    lm_head_argmax_triton = None
    materialize_pruned_medusa_triton = None
    node_budget_select_triton = None
    packed_kv_qjl_node_scores_triton = None
    qjl_path_scores_triton = None
    turbo_qjl_select_paths_triton = None

TOPK=10 # max rank stride for sparse tree buffers


def _effective_tree_topk(tree_topk=None):
    if tree_topk is None:
        return TOPK
    if torch.is_tensor(tree_topk):
        tree_topk = int(tree_topk.item())
    return max(1, min(TOPK, int(tree_topk)))


def append_input_ids(input_ids, append_tokens, input_ids_buffer=None):
    """
    Append tokens while optionally reusing a preallocated decode buffer.

    Streaming generation still receives a growing `input_ids` view, but the
    token storage no longer has to be reallocated/copied every accepted step.
    """
    if append_tokens.dim() == 1:
        append_tokens = append_tokens.unsqueeze(0)
    prev_len = input_ids.shape[1]
    append_len = append_tokens.shape[1]
    next_len = prev_len + append_len
    if (
        input_ids_buffer is not None
        and input_ids_buffer.shape[0] == input_ids.shape[0]
        and input_ids_buffer.shape[1] >= next_len
    ):
        input_ids_buffer[:, prev_len:next_len].copy_(append_tokens)
        return input_ids_buffer[:, :next_len]
    return torch.cat([input_ids, append_tokens], dim=-1)


class QJLTokenSketchCache:
    """
    1-bit QJL sidecar cache for token embeddings.

    This is a token/LM-head branch prior, not the packed KV-cache QJL pre-pass.
    It stores sign(S e_token) and ||e_token|| per token id for approximate
    path scoring before exact Medusa verification. A true communication-focused
    KV-QJL filter should sketch cached attention keys and candidate query states,
    then score them with packed XNOR-popcount before compacting tree nodes.
    """

    def __init__(self, vocab_size, hidden_size, sketch_dim=128, device="cuda", seed: int = 0):
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.sketch_dim = int(max(8, min(sketch_dim, hidden_size)))
        self.device = torch.device(device)
        self.seed = int(seed)

        gen = torch.Generator(device="cpu")
        gen.manual_seed((20260512 + self.hidden_size * 31 + self.sketch_dim * 17 + self.seed * 1_000_003) % (2**63 - 1))
        proj = torch.randn(
            self.hidden_size,
            self.sketch_dim,
            generator=gen,
            dtype=torch.float32,
        ).to(self.device)
        # QJL is defined with Gaussian rows. The paper orthogonalizes those rows
        # in practice; scaling by sqrt(d) keeps the row norm comparable to N(0, I).
        if self.sketch_dim <= self.hidden_size:
            try:
                proj, _ = torch.linalg.qr(proj, mode="reduced")
                proj = proj * math.sqrt(float(self.hidden_size))
            except RuntimeError:
                pass
        self.proj = proj
        self.sign_cache = torch.zeros(
            self.vocab_size,
            self.sketch_dim,
            dtype=torch.int8,
            device=self.device,
        )
        self.norm_cache = torch.zeros(self.vocab_size, dtype=torch.float16, device=self.device)
        self.ready = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        self.coeff = math.sqrt(math.pi / 2.0) / float(self.sketch_dim)

    def _cache_token_sketches(self, token_ids, embed_weight):
        if token_ids.numel() == 0:
            return
        unique_ids = torch.unique(token_ids)
        valid = (unique_ids >= 0) & (unique_ids < self.vocab_size)
        unique_ids = unique_ids[valid]
        if unique_ids.numel() == 0:
            return

        miss_mask = ~self.ready[unique_ids]
        missing_ids = unique_ids[miss_mask]
        if missing_ids.numel() == 0:
            return

        embeds = embed_weight.index_select(0, missing_ids).to(torch.float32)
        proj_vals = embeds @ self.proj
        signs = torch.sign(proj_vals)
        signs[signs == 0] = 1
        norms = embeds.norm(dim=-1).clamp_min(1e-6)

        self.sign_cache.index_copy_(0, missing_ids, signs.to(torch.int8))
        self.norm_cache.index_copy_(0, missing_ids, norms.to(torch.float16))
        self.ready[missing_ids] = True

    def score_paths(self, query_state, candidates, valid_mask, embed_weight):
        """
        Approximate path scores via QJL:
        <q, k> ≈ (sqrt(pi/2) / m) * ||k|| * <S q, sign(S k)>
        where k is token embedding and path score is mean token score.
        """
        if query_state.dim() == 2:
            q = query_state[0]
        else:
            q = query_state
        q = q.to(torch.float32)
        q_proj = q @ self.proj
        q_proj = q_proj.view(1, 1, -1)

        safe_candidates = candidates.clamp(0, self.vocab_size - 1)
        self._cache_token_sketches(safe_candidates[valid_mask], embed_weight)

        if qjl_path_scores_triton is not None:
            triton_scores = qjl_path_scores_triton(
                q_proj.reshape(-1),
                self.sign_cache,
                self.norm_cache,
                safe_candidates,
                valid_mask,
                self.coeff,
                self.sketch_dim,
            )
            if triton_scores is not None:
                return triton_scores

        signs = self.sign_cache[safe_candidates].to(torch.float32)
        norms = self.norm_cache[safe_candidates].to(torch.float32)
        inner = (signs * q_proj).sum(dim=-1)
        token_scores = self.coeff * norms * inner

        mask_f = valid_mask.to(token_scores.dtype)
        path_lens = mask_f.sum(dim=1).clamp_min(1.0)
        return (token_scores * mask_f).sum(dim=1) / path_lens


def _normalize_scores(scores):
    """Z-normalize scores for rank fusion without making scale assumptions."""
    scores = scores.to(torch.float32)
    std = scores.std(unbiased=False)
    if std <= 1e-6:
        return torch.zeros_like(scores)
    return (scores - scores.mean()) / (std + 1e-6)


def estimate_medusa_path_scores(medusa_logits, logits, tree_indices, retrieve_indices, candidates=None, tree_topk=None):
    """
    Score paths using the exact root LM logit and exact Medusa-head top-k logits.

    This is still a low-accuracy prior because Medusa heads are speculative, but it
    is much better aligned with the tree construction than ranking token ids only
    with the current hidden state.
    """
    # Ranking candidate paths does not require full-vocab normalization. Avoiding
    # log_softmax here removes a noticeable per-step overhead from the pruning
    # planner; downstream z-normalization handles scale differences.
    root_logits = logits[0, -1]
    if candidates is not None:
        root_token = candidates[0, 0].clamp(0, root_logits.shape[0] - 1)
        root_score = root_logits[root_token]
    else:
        root_score = torch.max(root_logits)

    if tree_topk is None and tree_indices is not None and tree_indices.numel() > 1:
        positive_tree_indices = tree_indices[tree_indices > 0]
        if positive_tree_indices.numel() > 0:
            tree_topk = int((positive_tree_indices - 1).remainder(TOPK).max().item()) + 1
    effective_topk = _effective_tree_topk(tree_topk)
    medusa_topk = torch.topk(medusa_logits[:, 0, -1], effective_topk, dim=-1).values
    if effective_topk == TOPK:
        medusa_score_bank = medusa_topk
    else:
        medusa_score_bank = torch.empty(
            (medusa_topk.shape[0], TOPK),
            dtype=medusa_topk.dtype,
            device=medusa_topk.device,
        )
        medusa_score_bank[:, :effective_topk] = medusa_topk
        medusa_score_bank[:, effective_topk:] = medusa_topk[:, -1:]
    score_bank = torch.cat([root_score.reshape(1), medusa_score_bank.reshape(-1)], dim=0)

    node_scores = score_bank[tree_indices]
    node_scores_ext = torch.cat(
        [
            node_scores,
            torch.full((1,), -1e4, device=node_scores.device, dtype=node_scores.dtype),
        ],
        dim=0,
    )
    node_indices = retrieve_indices[:, 1:].clone()
    valid_mask = node_indices >= 0
    node_indices[~valid_mask] = -1
    path_scores = node_scores_ext[node_indices]
    valid_mask_f = valid_mask.to(path_scores.dtype)
    path_lens = valid_mask_f.sum(dim=1).clamp_min(1.0)
    return (path_scores * valid_mask_f).sum(dim=1) / path_lens

def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def generate_medusa_buffers(medusa_choices, device="cuda"):
    """
    Generate buffers for the Medusa structure based on the provided choices.
    
    Parameters:
    - medusa_choices (list): A nested list representing tree in the Medusa structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".
    
    Returns:
    - dict: A dictionary containing buffers related to the Medusa structure.
    """

    # Sort the medusa_choices based on their lengths and then their values
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    
    # Create the attention mask for Medusa
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)
    medusa_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # retrieve ancestor position
            if len(cur_medusa_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the Medusa structure
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    medusa_tree_indices[0] = 0
    max_tree_rank = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            max_tree_rank = max(max_tree_rank, int(cur_medusa_choice[-1]))
            medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
        start += depth_counts[i]

    # Generate position IDs for the Medusa structure
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i-1]
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                retrieve_paths.append(cur_medusa_choice[:c+1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)

    # Aggregate the generated buffers into a dictionary
    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
        "tree_topk": torch.tensor(max_tree_rank + 1, dtype=torch.long),
        }
    
    # Move the tensors in the dictionary to the specified device
    medusa_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v,  device=device)
        for k, v in medusa_buffers.items()
    }
    return medusa_buffers


def initialize_medusa(
    input_ids,
    model,
    medusa_attn_mask,
    past_key_values,
    return_query_state=False,
    last_token_logits=False,
):
    """
    Initializes the Medusa structure for a given model.

    This function performs the following operations:
    1. Forward pass through the model to obtain the Medusa logits, original model outputs, and logits.
    2. Sets the Medusa attention mask within the base model.

    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - medusa_attn_mask (torch.Tensor): The attention mask designed specifically for the Medusa structure.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - medusa_logits (torch.Tensor): Logits from the Medusa heads.
    - logits (torch.Tensor): Original logits from the base model.
    """
    medusa_logits, outputs, logits = model(
        input_ids,
        past_key_values=past_key_values,
        output_orig=True,
        medusa_forward=True,
        last_token_logits=last_token_logits,
    )
    model.base_model.model.medusa_mask = medusa_attn_mask
    if return_query_state:
        query_state = outputs[0][:, -1, :]
        return medusa_logits, logits, query_state
    return medusa_logits, logits


def get_cached_tree_attention_mask(model, medusa_attn_mask, past_len, dtype, device):
    """
    Return a reusable 4D attention mask for fixed-shape Medusa tree verification.

    The mask is stored with the tree block pinned at the right edge. Each decode
    step slices a different zero-prefix length in front of that fixed block, so
    we avoid rebuilding the causal + tree mask tensor on every verifier call.
    """
    if medusa_attn_mask is None:
        return None
    if medusa_attn_mask.dim() != 4 or medusa_attn_mask.shape[0] != 1:
        return None

    past_len = int(past_len)
    tree_len = int(medusa_attn_mask.shape[-1])
    if tree_len <= 0 or past_len < 0:
        return None

    max_past = int(getattr(model, "kv_cache_max_length", past_len + tree_len + 1))
    max_past = max(max_past, past_len)
    device = torch.device(device)
    cache_key = (
        str(device),
        str(dtype),
        int(tree_len),
        int(max_past),
        int(medusa_attn_mask.data_ptr()) if medusa_attn_mask.is_cuda else id(medusa_attn_mask),
    )
    cached_key = getattr(model, "_turbo_tree_attention_mask_cache_key", None)
    cached_mask = getattr(model, "_turbo_tree_attention_mask_cache", None)
    if cached_key != cache_key or cached_mask is None:
        mask = torch.zeros(
            (1, 1, tree_len, max_past + tree_len),
            dtype=dtype,
            device=device,
        )
        min_value = torch.finfo(dtype).min
        block = torch.full((tree_len, tree_len), min_value, dtype=dtype, device=device)
        offsets = torch.arange(tree_len, device=device)
        block.masked_fill_(offsets < (offsets + 1).view(tree_len, 1), 0)
        tree_mask = medusa_attn_mask[0, 0].to(device=device, dtype=torch.bool)
        block.masked_fill_(~tree_mask, min_value)
        mask[:, :, :, max_past : max_past + tree_len].copy_(block)
        model._turbo_tree_attention_mask_cache_key = cache_key
        model._turbo_tree_attention_mask_cache = mask
        cached_mask = mask

    start = max_past - past_len
    if start < 0:
        return None
    # SDPA requires well-aligned attention-bias storage on CUDA. The right-edge
    # slice avoids rebuilding values, and the contiguous copy gives SDPA an
    # aligned compact view for the current KV length.
    return cached_mask[:, :, :, start : max_past + tree_len].contiguous()


def reset_medusa_mode(
    model,
):
    """
    Resets the Medusa settings and the past key-values to their initial state.

    This function ensures that after any operations involving Medusa,
    the base model and its settings return to their default state.
    Specifically, it performs the following tasks:
    1. Clears the Medusa attention mask in the base model.
    2. Resets the Medusa mode in the base model.
    3. Resets the current lengths in the past key-values to zero for all layers.

    Args:
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - None
    """
    model.base_model.model.medusa_mask = None
    model.base_model.model.medusa_mode = None


def reset_past_key_values(passed_key_values):
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            kv_cache = passed_key_values[i][j]
            if hasattr(kv_cache, "reset"):
                kv_cache.reset()
            else:
                kv_cache.current_length.fill_(0)
    return passed_key_values

def get_nucleus_one_token(logit, temperature, top_p):
    """
    Performs token sampling based on the nucleus (top-p) sampling method.

    This function selects a token from a given logit distribution using the nucleus sampling strategy.
    It allows for more controlled and diverse generation compared to traditional top-k sampling.

    Args:
        logit (torch.Tensor): The logits from a language model output, expected to be a 2D tensor (BxC).
        temperature (float): A temperature parameter to control the randomness in sampling.
                             Higher values increase diversity, lower values make selections more deterministic.
        top_p (float): The cumulative probability threshold for nucleus sampling.
                       It controls the size of the set of high-probability tokens to consider for sampling.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    if top_p >= 1:
        return torch.multinomial(F.softmax(logit / temperature, dim=-1), 1)
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_logits, dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logit[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
    return sampled_tokens

def get_typical_one_token(logit, temperature, posterior_threshold, posterior_alpha):
    """
    Implements token sampling based on the typical sampling method.

    This function selects a token from a given logit distribution using the typical sampling strategy,
    aiming to balance between diversity and likelihood in a more nuanced way compared to traditional methods.

    Args:
        logit (torch.Tensor): The logits from a language model output, expected to be a 2D tensor.
        temperature (float): A parameter to control the randomness in sampling.
                              Higher values increase diversity, lower values make selections more deterministic.
        posterior_threshold (float): A threshold to decide the lower bound of probabilities to be considered for sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * posterior_threshold,
            torch.exp(-entropy) * posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logit[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
    return sampled_tokens

def generate_candidates(medusa_logits, logits, tree_indices, retrieve_indices, temperature = 0, posterior_threshold=0.3, posterior_alpha = 0.09, top_p=0.8, sampling = 'typical', fast = False, tree_topk=None):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - medusa_logits (torch.Tensor): Logits from a specialized Medusa structure, aiding in candidate selection.
    - logits (torch.Tensor): Standard logits from a language model.
    - tree_indices (list or torch.Tensor): Indices representing a tree structure, used for mapping candidates.
    - retrieve_indices (list or torch.Tensor): Indices for extracting specific candidate tokens.
    - temperature (float, optional): Controls the diversity of the sampling process. Defaults to 0.
    - posterior_threshold (float, optional): Threshold for typical sampling. Defaults to 0.3.
    - posterior_alpha (float, optional): Scaling factor for the entropy-based threshold in typical sampling. Defaults to 0.09.
    - top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
    - sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
    - fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.

    Returns:
    - tuple (torch.Tensor, torch.Tensor): A tuple containing two sets of candidates:
        1. Cartesian candidates derived from the combined original and Medusa logits.
        2. Tree candidates mapped from the Cartesian candidates using tree indices.
    """
    # Greedy decoding: Select the most probable candidate from the original logits.
    if temperature == 0 or fast:
        candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
    else:
        if sampling == 'typical':
            candidates_logit = get_typical_one_token(logits[:, -1], temperature, posterior_threshold, posterior_alpha).squeeze(0)
        elif sampling == 'nucleus':
            candidates_logit = get_nucleus_one_token(logits[:, -1], temperature, top_p).squeeze(0)
        else:
            raise NotImplementedError
    # Extract only the ranks used by the active tree, while keeping the
    # historical TOPK-strided candidate bank layout expected by tree_indices.
    effective_topk = _effective_tree_topk(tree_topk)
    candidates_medusa_topk = torch.topk(medusa_logits[:, 0, -1], effective_topk, dim = -1).indices
    if effective_topk == TOPK:
        candidates_medusa_logits = candidates_medusa_topk
    else:
        candidates_medusa_logits = torch.empty(
            (candidates_medusa_topk.shape[0], TOPK),
            dtype=candidates_medusa_topk.dtype,
            device=candidates_medusa_topk.device,
        )
        candidates_medusa_logits[:, :effective_topk] = candidates_medusa_topk
        candidates_medusa_logits[:, effective_topk:] = candidates_medusa_topk[:, -1:]

    # Combine the selected candidate from the original logits with the topk medusa logits.
    candidates = torch.cat([candidates_logit, candidates_medusa_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[tree_indices]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[retrieve_indices]

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, tree_candidates


def estimate_tree_candidate_scores_1bit(
    medusa_logits,
    logits,
    tree_indices,
    retrieve_indices,
    candidates=None,
    query_state=None,
    qjl_scorer=None,
    embed_weight=None,
):
    """
    Conservative low-accuracy path scorer used for TurboQuant early pruning.

    The QJL component is a 1-bit approximation of LM-head inner products. We
    fuse it with exact Medusa-head path probabilities so pruning follows the
    same distribution that built the tree, then leave exact acceptance to the
    high-accuracy verifier.
    """
    valid_mask = retrieve_indices >= 0
    qjl_valid_mask = valid_mask.clone()
    if qjl_valid_mask.shape[1] > 0:
        # The root token is the same for every path, so it should not influence
        # which speculative branches survive pass-1 pruning.
        qjl_valid_mask[:, 0] = False
    medusa_path_scores = estimate_medusa_path_scores(
        medusa_logits,
        logits,
        tree_indices,
        retrieve_indices,
        candidates=candidates,
    )
    qjl_path_scores = None
    if (
        qjl_scorer is not None
        and query_state is not None
        and candidates is not None
        and embed_weight is not None
    ):
        qjl_path_scores = qjl_scorer.score_paths(
            query_state=query_state,
            candidates=candidates,
            valid_mask=qjl_valid_mask,
            embed_weight=embed_weight,
        )
        # Medusa logits are the branch prior; QJL is only a cheap side signal.
        approx_scores = (
            0.75 * _normalize_scores(medusa_path_scores)
            + 0.25 * _normalize_scores(qjl_path_scores)
        )
    else:
        # Fallback proxy when QJL sidecar is not configured.
        approx_scores = medusa_path_scores

    # Number of valid predicted tokens excluding root position.
    path_lengths = (retrieve_indices[:, 1:] >= 0).sum(dim=1)
    return approx_scores, path_lengths


def packed_kv_qjl_node_scores(query_bits, key_cache, block_k=1024):
    """
    Score candidate tree nodes against a packed sign(RK) key sidecar.

    `query_bits`: [nodes, kv_heads, words] int32 packed sign(Rq)
    `key_cache.qjl_bits`: [batch=1, kv_heads, max_len, words] int32 packed sign(RK)

    The score is an XNOR-popcount proxy. It is intended only for pruning before
    exact verification; it is not an acceptance oracle.
    """
    key_bits = getattr(key_cache, "qjl_bits", None)
    if key_bits is None or packed_kv_qjl_node_scores_triton is None:
        return None
    if query_bits is None or query_bits.dim() != 3:
        return None
    if key_bits.dim() != 4 or key_bits.shape[0] != 1:
        return None
    kv_len = int(key_cache.current_length.item())
    if kv_len <= 0:
        return None
    return packed_kv_qjl_node_scores_triton(
        query_bits.contiguous(),
        key_bits[0].contiguous(),
        kv_len=kv_len,
        block_k=block_k,
    )


def estimate_packed_kv_qjl_path_scores(node_scores, retrieve_indices):
    if node_scores is None or retrieve_indices.numel() == 0:
        return None, (retrieve_indices[:, 1:] >= 0).sum(dim=1)
    node_scores = node_scores.to(torch.float32)
    node_scores_ext = torch.cat(
        [
            torch.zeros((1,), device=node_scores.device, dtype=node_scores.dtype),
            node_scores,
            torch.zeros((1,), device=node_scores.device, dtype=node_scores.dtype),
        ],
        dim=0,
    )
    node_indices = retrieve_indices[:, 1:].clone()
    valid_mask = node_indices >= 0
    # retrieve_indices are 1-based for tree nodes; convert invalid padding to
    # the trailing zero score.
    node_indices[~valid_mask] = node_scores.numel() + 1
    path_scores = node_scores_ext[node_indices]
    valid_mask_f = valid_mask.to(path_scores.dtype)
    path_lens = valid_mask_f.sum(dim=1).clamp_min(1.0)
    return (path_scores * valid_mask_f).sum(dim=1) / path_lens, valid_mask.sum(dim=1)


def _unique_index_cat(parts, device):
    valid_parts = []
    for part in parts:
        if part is None or part.numel() == 0:
            continue
        valid_parts.append(part.to(device=device, dtype=torch.long).reshape(-1))
    if not valid_parts:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.unique(torch.cat(valid_parts))


def _mandatory_top1_path_indices(retrieve_indices, tree_indices, max_paths=2):
    """Find deepest all-top1 Medusa paths that should never be pruned."""
    if retrieve_indices.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=retrieve_indices.device)

    node_indices = retrieve_indices[:, 1:]
    valid_mask = node_indices >= 0
    safe_nodes = node_indices.clamp_min(0)
    node_tree_indices = tree_indices[safe_nodes]
    node_ranks = (node_tree_indices - 1).remainder(TOPK)
    all_top1 = ((node_ranks == 0) | ~valid_mask).all(dim=1)
    top1_rows = torch.where(all_top1)[0]
    if top1_rows.numel() == 0:
        return top1_rows

    path_lengths = valid_mask.sum(dim=1)
    order = torch.argsort(path_lengths[top1_rows], descending=True)
    return top1_rows[order[:max_paths]]


def _long_medusa_anchor_path_indices(
    retrieve_indices,
    medusa_path_scores,
    max_paths=4,
    min_length=2,
):
    """
    Pick high-priority long paths that should survive approximate pruning.

    A compact tree can look cheap while accidentally dropping the only plausible
    deep continuation. These anchors keep the best Medusa-prior path at each
    speculative depth, starting from the deepest paths.
    """
    if retrieve_indices.numel() == 0 or medusa_path_scores is None:
        return torch.empty(0, dtype=torch.long, device=retrieve_indices.device)

    path_lengths = (retrieve_indices[:, 1:] >= 0).sum(dim=1)
    if path_lengths.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=retrieve_indices.device)

    max_paths = max(0, int(max_paths))
    if max_paths == 0:
        return torch.empty(0, dtype=torch.long, device=retrieve_indices.device)

    max_length = int(path_lengths.max().item())
    anchors = []
    for length in range(max_length, max(1, int(min_length)) - 1, -1):
        rows = torch.where(path_lengths == length)[0]
        if rows.numel() == 0:
            continue
        best = rows[torch.argmax(medusa_path_scores.index_select(0, rows))]
        anchors.append(best.reshape(1))
        if len(anchors) >= max_paths:
            break

    if not anchors:
        return torch.empty(0, dtype=torch.long, device=retrieve_indices.device)
    return torch.unique(torch.cat(anchors))


def _path_acceptance_confidence(path_scores):
    """
    Convert Medusa path-prior scores into relative confidence in [0, 1].

    This is not the exact verifier acceptance probability. It is a cheap proxy
    that ranks each candidate against the strongest Medusa-prior path in the
    current tree. A value of 0.5 means "about half as plausible as the best path"
    under this local proxy.
    """
    if path_scores is None or path_scores.numel() == 0:
        return None
    probs = torch.softmax(_normalize_scores(path_scores), dim=0)
    return probs / probs.max().clamp_min(1.0e-6)


def _acceptance_confidence_sharpness(path_scores):
    confidence = _path_acceptance_confidence(path_scores)
    if confidence is None or confidence.numel() <= 1:
        return 0.0
    top2 = torch.topk(confidence, k=2, dim=0).values
    return float((1.0 - top2[1]).clamp(0.0, 1.0).item())


def _resolve_acceptance_thresholds(
    path_scores,
    prune_threshold=0.0,
    keep_threshold=0.0,
    dynamic=False,
    dynamic_prune_min=0.10,
    dynamic_prune_max=0.45,
    dynamic_keep_min=0.45,
    dynamic_keep_max=0.70,
):
    if not dynamic:
        return float(prune_threshold or 0.0), float(keep_threshold or 0.0)

    sharpness = _acceptance_confidence_sharpness(path_scores)
    prune_low = float(dynamic_prune_min)
    prune_high = max(prune_low, float(dynamic_prune_max))
    keep_low = float(dynamic_keep_min)
    keep_high = max(keep_low, float(dynamic_keep_max))
    prune = prune_low + sharpness * (prune_high - prune_low)
    keep = keep_low + sharpness * (keep_high - keep_low)
    return prune, keep


def _confidence_anchor_path_indices(path_scores, keep_threshold=0.0):
    if path_scores is None or path_scores.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=path_scores.device if path_scores is not None else "cpu")
    keep_threshold = float(keep_threshold or 0.0)
    if keep_threshold <= 0.0:
        return torch.empty(0, dtype=torch.long, device=path_scores.device)
    confidence = _path_acceptance_confidence(path_scores)
    return torch.where(confidence >= keep_threshold)[0]


def _apply_acceptance_prune_threshold(selection_scores, medusa_path_scores, prune_threshold=0.0):
    prune_threshold = float(prune_threshold or 0.0)
    if prune_threshold <= 0.0 or medusa_path_scores is None or selection_scores.numel() == 0:
        return selection_scores
    confidence = _path_acceptance_confidence(medusa_path_scores)
    low_confidence = confidence < prune_threshold
    if not bool(low_confidence.any().item()):
        return selection_scores
    penalty = (selection_scores.std(unbiased=False) + 1.0) * 4.0
    return torch.where(low_confidence, selection_scores - penalty, selection_scores)


def _acceptance_anchor_path_indices(
    retrieve_indices,
    tree_indices,
    medusa_path_scores=None,
    max_long_paths=4,
    keep_threshold=0.0,
):
    return _unique_index_cat(
        [
            _mandatory_top1_path_indices(retrieve_indices, tree_indices),
            _long_medusa_anchor_path_indices(
                retrieve_indices,
                medusa_path_scores,
                max_paths=max_long_paths,
            ),
            _confidence_anchor_path_indices(
                medusa_path_scores,
                keep_threshold=keep_threshold,
            ),
        ],
        device=retrieve_indices.device,
    )


def select_topk_paths_for_verification(
    approx_scores,
    keep_target=12,
    min_keep=10,
    max_keep=15,
    retrieve_indices=None,
    tree_indices=None,
    mandatory_indices=None,
):
    """Select a small subset of candidate paths for high-accuracy verification."""
    total_paths = approx_scores.shape[0]
    keep = max(min_keep, min(max_keep, keep_target))
    keep = min(keep, total_paths)

    # Confidence-adaptive keep-size: if pass-1 is uncertain, send more paths
    # to exact verification instead of over-pruning.
    if total_paths > 1:
        top2 = torch.topk(approx_scores, k=min(2, total_paths), dim=0).values
        margin = float((top2[0] - top2[-1]).item())
        score_std = float(approx_scores.std(unbiased=False).item())
        if score_std <= 1e-6 or margin < 0.25 * score_std:
            keep = min(max_keep, total_paths)

    mandatory = torch.empty(0, dtype=torch.long, device=approx_scores.device)
    if mandatory_indices is not None:
        mandatory = mandatory_indices.to(device=approx_scores.device, dtype=torch.long)
    elif retrieve_indices is not None and tree_indices is not None:
        mandatory = _mandatory_top1_path_indices(retrieve_indices, tree_indices)
    if mandatory.numel() > 0:
        mandatory = mandatory[(mandatory >= 0) & (mandatory < total_paths)]
        mandatory = torch.unique(mandatory)

    if mandatory.numel() >= keep:
        return mandatory[:keep]

    ranked = torch.topk(approx_scores, k=total_paths, dim=0).indices
    if mandatory.numel() > 0:
        ranked = ranked[~torch.isin(ranked, mandatory)]

    selected = torch.cat([mandatory, ranked[: keep - mandatory.numel()]])
    return selected


def select_paths_for_node_budget(
    approx_scores,
    retrieve_indices,
    tree_indices,
    node_budget,
    min_keep=1,
    max_keep=None,
    mandatory_indices=None,
):
    """
    Select high-scoring paths while capping unique Medusa tree nodes.

    This maps pruning to the real verifier cost better than a fixed path count:
    shared ancestors are cheap, while new unique nodes increase the tree forward.
    """
    total_paths = int(approx_scores.shape[0])
    if total_paths == 0:
        return torch.empty(0, dtype=torch.long, device=approx_scores.device)
    if node_budget is None or int(node_budget) <= 0:
        return select_topk_paths_for_verification(
            approx_scores,
            keep_target=max_keep or total_paths,
            min_keep=min_keep,
            max_keep=max_keep or total_paths,
            retrieve_indices=retrieve_indices,
            tree_indices=tree_indices,
        )

    max_keep = total_paths if max_keep is None else min(int(max_keep), total_paths)
    min_keep = min(max(1, int(min_keep)), total_paths)
    node_budget = max(1, int(node_budget))

    if mandatory_indices is None:
        mandatory = _mandatory_top1_path_indices(retrieve_indices, tree_indices)
    else:
        mandatory = mandatory_indices.to(device=approx_scores.device, dtype=torch.long)
        mandatory = mandatory[(mandatory >= 0) & (mandatory < total_paths)]
        mandatory = torch.unique(mandatory)

    if (
        os.environ.get("MEDUSA_TRITON_NODE_SELECT", "0") == "1"
        and node_budget_select_triton is not None
        and approx_scores.is_cuda
        and retrieve_indices.is_cuda
    ):
        fast_selected = node_budget_select_triton(
            approx_scores,
            retrieve_indices,
            mandatory,
            node_budget=node_budget,
            min_keep=min_keep,
            max_keep=max_keep,
            full_node_count=int(tree_indices.numel()),
        )
        if fast_selected is not None and fast_selected.numel() > 0:
            return fast_selected

    ranked = torch.argsort(approx_scores, descending=True)
    if mandatory.numel() > 0:
        ranked = ranked[~torch.isin(ranked, mandatory)]
        ordered = torch.cat([mandatory, ranked])
    else:
        ordered = ranked

    retrieve_cpu = retrieve_indices.detach().cpu()
    ordered_ids = [int(x) for x in ordered.detach().cpu().tolist()]
    selected = []
    selected_set = set()
    selected_nodes = set()

    for path_idx in ordered_ids:
        if path_idx in selected_set or path_idx < 0 or path_idx >= total_paths:
            continue
        path_nodes = [int(x) for x in retrieve_cpu[path_idx].tolist() if int(x) >= 0]
        next_node_count = len(selected_nodes.union(path_nodes))
        over_budget = next_node_count > node_budget
        if over_budget and len(selected) >= min_keep:
            continue
        selected.append(path_idx)
        selected_set.add(path_idx)
        selected_nodes.update(path_nodes)
        if len(selected) >= max_keep:
            break

    if len(selected) < min_keep:
        for path_idx in ordered_ids:
            if path_idx in selected_set or path_idx < 0 or path_idx >= total_paths:
                continue
            selected.append(path_idx)
            selected_set.add(path_idx)
            if len(selected) >= min_keep:
                break

    if len(selected) == 0:
        selected = [int(torch.argmax(approx_scores).item())]

    return torch.tensor(selected, dtype=torch.long, device=approx_scores.device)


def select_paths_for_pruning(
    approx_scores,
    keep_target=12,
    min_keep=10,
    max_keep=15,
    retrieve_indices=None,
    tree_indices=None,
    node_budget=0,
    mandatory_indices=None,
):
    if (
        node_budget is not None
        and int(node_budget) > 0
        and retrieve_indices is not None
        and tree_indices is not None
    ):
        return select_paths_for_node_budget(
            approx_scores,
            retrieve_indices,
            tree_indices,
            node_budget=node_budget,
            min_keep=min_keep,
            max_keep=max_keep,
            mandatory_indices=mandatory_indices,
        )
    return select_topk_paths_for_verification(
        approx_scores,
        keep_target=keep_target,
        min_keep=min_keep,
        max_keep=max_keep,
        retrieve_indices=retrieve_indices,
        tree_indices=tree_indices,
        mandatory_indices=mandatory_indices,
    )


def should_verify_full_tree(approx_scores, margin_scale=0.25):
    """
    Return True when pass-1 scores are too flat to safely save work via pruning.

    This avoids the expensive pattern of running a pruned tree first and then
    falling back to a full tree on obviously ambiguous steps.
    """
    if approx_scores.shape[0] <= 1:
        return False
    top2 = torch.topk(approx_scores, k=2, dim=0).values
    margin = float((top2[0] - top2[1]).item())
    score_std = float(approx_scores.std(unbiased=False).item())
    return score_std <= 1e-6 or margin < float(margin_scale) * score_std


def should_skip_pruning_for_low_gain(selected_paths, total_paths, min_prune_fraction=0.0):
    """
    Return True when the pruned verifier would keep too much of the full tree.

    Pruning is only a speed win if it removes enough candidate paths to offset
    pass-1 planning, layout materialization, and potential fallback costs.
    """
    if min_prune_fraction <= 0.0 or total_paths <= 0:
        return False
    kept = int(selected_paths.numel())
    pruned_fraction = float(max(0, total_paths - kept)) / float(total_paths)
    return pruned_fraction < float(min_prune_fraction)


def _score_margin_stats(scores):
    """Return top-2 margin and population stddev for a small score vector."""
    if scores.shape[0] <= 1:
        return 0.0, 0.0
    top2 = torch.topk(scores, k=2, dim=0).values
    margin = float((top2[0] - top2[1]).item())
    score_std = float(scores.std(unbiased=False).item())
    return margin, score_std


def pruned_layout_has_enough_node_gain(layout, full_node_count, min_node_prune_fraction=0.0):
    """
    Return True when the compact verifier tree removes enough actual tree nodes.

    Path-count pruning can look useful while still keeping most unique tree nodes.
    Verifier cost is closer to selected node/query count, so this is the final
    low-overhead guard before launching the pruned forward pass.
    """
    if min_node_prune_fraction <= 0.0:
        return True
    full_node_count = max(1, int(full_node_count))
    selected_node_count = int(layout["selected_nodes"].numel())
    pruned_fraction = float(max(0, full_node_count - selected_node_count)) / float(full_node_count)
    return pruned_fraction >= float(min_node_prune_fraction)


def selected_paths_have_enough_node_gain(
    retrieve_indices,
    selected_path_indices,
    full_node_count,
    min_node_prune_fraction=0.0,
):
    """
    Cheap pre-layout version of the node-gain gate used before optional QJL work.
    """
    if min_node_prune_fraction <= 0.0:
        return True
    full_node_count = max(1, int(full_node_count))
    selected_paths = retrieve_indices.index_select(0, selected_path_indices)
    valid_mask = selected_paths >= 0
    selected_node_count = int(torch.unique(selected_paths[valid_mask]).numel())
    pruned_fraction = float(max(0, full_node_count - selected_node_count)) / float(full_node_count)
    return pruned_fraction >= float(min_node_prune_fraction)


def selected_paths_include_required(selected_path_indices, required_indices):
    if required_indices is None or required_indices.numel() == 0:
        return True
    if selected_path_indices is None or selected_path_indices.numel() == 0:
        return False
    required_indices = required_indices.to(device=selected_path_indices.device, dtype=torch.long)
    selected_path_indices = selected_path_indices.to(dtype=torch.long)
    return bool(torch.isin(required_indices, selected_path_indices).all().item())


def plan_turbo_pruning(
    medusa_logits,
    logits,
    tree_indices,
    retrieve_indices,
    candidates=None,
    query_state=None,
    qjl_scorer=None,
    embed_weight=None,
    keep_target=12,
    min_keep=10,
    max_keep=15,
    margin_scale=0.25,
    mandatory_indices=None,
    prescreen_margin_scale=0.75,
    min_prune_fraction=0.25,
    min_node_prune_fraction=0.0,
    node_budget=0,
    decisive_margin_scale=1.5,
    decisive_keep=8,
    use_qjl=True,
    acceptance_prune_threshold=0.0,
    acceptance_keep_threshold=0.0,
    acceptance_threshold_dynamic=False,
    acceptance_dynamic_prune_min=0.10,
    acceptance_dynamic_prune_max=0.45,
    acceptance_dynamic_keep_min=0.45,
    acceptance_dynamic_keep_max=0.70,
):
    """
    Compute Turbo pass-1 scores, selected paths, and full-tree confidence gate.

    CUDA/Triton fast path fuses QJL scoring, score normalization/fusion, top-k
    selection, and confidence gating. CPU or unsupported shapes fall back to the
    reference PyTorch implementation above.
    """
    path_lengths = (retrieve_indices[:, 1:] >= 0).sum(dim=1)
    medusa_path_scores = estimate_medusa_path_scores(
        medusa_logits,
        logits,
        tree_indices,
        retrieve_indices,
        candidates=candidates,
    )
    acceptance_prune_threshold, acceptance_keep_threshold = _resolve_acceptance_thresholds(
        medusa_path_scores,
        prune_threshold=acceptance_prune_threshold,
        keep_threshold=acceptance_keep_threshold,
        dynamic=acceptance_threshold_dynamic,
        dynamic_prune_min=acceptance_dynamic_prune_min,
        dynamic_prune_max=acceptance_dynamic_prune_max,
        dynamic_keep_min=acceptance_dynamic_keep_min,
        dynamic_keep_max=acceptance_dynamic_keep_max,
    )

    qjl_valid_mask = retrieve_indices >= 0
    if qjl_valid_mask.shape[1] > 0:
        qjl_valid_mask = qjl_valid_mask.clone()
        qjl_valid_mask[:, 0] = False

    if mandatory_indices is None:
        mandatory_indices = _acceptance_anchor_path_indices(
            retrieve_indices,
            tree_indices,
            medusa_path_scores,
            max_long_paths=2,
            keep_threshold=acceptance_keep_threshold,
        )

    medusa_margin, medusa_std = _score_margin_stats(medusa_path_scores)
    if prescreen_margin_scale is not None and float(prescreen_margin_scale) >= 0.0:
        if medusa_std <= 1e-6 or medusa_margin < float(prescreen_margin_scale) * medusa_std:
            return medusa_path_scores, path_lengths, mandatory_indices, True

    prescreen_selected_paths = None
    if min_node_prune_fraction > 0.0:
        prescreen_selected_paths = select_paths_for_pruning(
            medusa_path_scores,
            keep_target=keep_target,
            min_keep=min_keep,
            max_keep=max_keep,
            retrieve_indices=retrieve_indices,
            tree_indices=tree_indices,
            node_budget=node_budget,
            mandatory_indices=mandatory_indices,
        )
        if not selected_paths_have_enough_node_gain(
            retrieve_indices,
            prescreen_selected_paths,
            full_node_count=tree_indices.numel(),
            min_node_prune_fraction=min_node_prune_fraction,
        ) or not selected_paths_include_required(
            prescreen_selected_paths,
            mandatory_indices,
        ):
            return medusa_path_scores, path_lengths, mandatory_indices, True

        medusa_only_verify_full = should_verify_full_tree(
            medusa_path_scores,
            margin_scale=margin_scale,
        ) or should_skip_pruning_for_low_gain(
            prescreen_selected_paths,
            medusa_path_scores.shape[0],
            min_prune_fraction=min_prune_fraction,
        )
        if not medusa_only_verify_full:
            # If the cheap Medusa prior already gives a confident layout with
            # enough real node reduction, QJL cannot pay for itself on this step.
            return medusa_path_scores, path_lengths, prescreen_selected_paths, False

    if (
        decisive_margin_scale is not None
        and float(decisive_margin_scale) >= 0.0
        and medusa_std > 1e-6
        and medusa_margin >= float(decisive_margin_scale) * medusa_std
    ):
        decisive_keep = keep_target if decisive_keep is None else int(decisive_keep)
        selected_paths = select_paths_for_pruning(
            medusa_path_scores,
            keep_target=decisive_keep,
            min_keep=min(min_keep, decisive_keep),
            max_keep=min(max_keep, decisive_keep),
            retrieve_indices=retrieve_indices,
            tree_indices=tree_indices,
            node_budget=node_budget,
            mandatory_indices=mandatory_indices,
        )
        verify_full_tree = should_skip_pruning_for_low_gain(
            selected_paths,
            medusa_path_scores.shape[0],
            min_prune_fraction=min_prune_fraction,
        ) or not selected_paths_have_enough_node_gain(
            retrieve_indices,
            selected_paths,
            full_node_count=tree_indices.numel(),
            min_node_prune_fraction=min_node_prune_fraction,
        ) or not selected_paths_include_required(
            selected_paths,
            mandatory_indices,
        )
        return medusa_path_scores, path_lengths, selected_paths, verify_full_tree

    if not use_qjl:
        if prescreen_selected_paths is None:
            prescreen_selected_paths = select_paths_for_pruning(
                medusa_path_scores,
                keep_target=keep_target,
                min_keep=min_keep,
                max_keep=max_keep,
                retrieve_indices=retrieve_indices,
                tree_indices=tree_indices,
                node_budget=node_budget,
                mandatory_indices=mandatory_indices,
            )
        verify_full_tree = should_verify_full_tree(
            medusa_path_scores,
            margin_scale=margin_scale,
        ) or should_skip_pruning_for_low_gain(
            prescreen_selected_paths,
            medusa_path_scores.shape[0],
            min_prune_fraction=min_prune_fraction,
        ) or not selected_paths_include_required(
            prescreen_selected_paths,
            mandatory_indices,
        )
        return medusa_path_scores, path_lengths, prescreen_selected_paths, verify_full_tree

    if (
        turbo_qjl_select_paths_triton is not None
        and qjl_scorer is not None
        and query_state is not None
        and candidates is not None
        and embed_weight is not None
    ):
        if query_state.dim() == 2:
            q = query_state[0]
        else:
            q = query_state
        q = q.to(torch.float32)
        q_proj = q @ qjl_scorer.proj

        safe_candidates = candidates.clamp(0, qjl_scorer.vocab_size - 1)
        qjl_scorer._cache_token_sketches(safe_candidates[qjl_valid_mask], embed_weight)
        fast_plan = turbo_qjl_select_paths_triton(
            q_proj.reshape(-1),
            qjl_scorer.sign_cache,
            qjl_scorer.norm_cache,
            safe_candidates,
            qjl_valid_mask,
            medusa_path_scores,
            mandatory_indices,
            qjl_scorer.coeff,
            qjl_scorer.sketch_dim,
            keep_target,
            min_keep,
            max_keep,
            margin_scale,
        )
        if fast_plan is not None:
            approx_scores, selected_paths, verify_full_tree = fast_plan
            reselection_scores = _apply_acceptance_prune_threshold(
                approx_scores,
                medusa_path_scores,
                prune_threshold=acceptance_prune_threshold,
            )
            if (
                node_budget is not None
                and int(node_budget) > 0
                or float(acceptance_prune_threshold or 0.0) > 0.0
                or int(mandatory_indices.numel()) > 2
            ):
                selected_paths = select_paths_for_pruning(
                    reselection_scores,
                    keep_target=keep_target,
                    min_keep=min_keep,
                    max_keep=max_keep,
                    retrieve_indices=retrieve_indices,
                    tree_indices=tree_indices,
                    node_budget=node_budget,
                    mandatory_indices=mandatory_indices,
                )
            verify_full_tree = verify_full_tree or should_skip_pruning_for_low_gain(
                selected_paths,
                approx_scores.shape[0],
                min_prune_fraction=min_prune_fraction,
            ) or not selected_paths_have_enough_node_gain(
                retrieve_indices,
                selected_paths,
                full_node_count=tree_indices.numel(),
                min_node_prune_fraction=min_node_prune_fraction,
            ) or not selected_paths_include_required(
                selected_paths,
                mandatory_indices,
            )
            return approx_scores, path_lengths, selected_paths, verify_full_tree

    if (
        qjl_scorer is not None
        and query_state is not None
        and candidates is not None
        and embed_weight is not None
    ):
        qjl_path_scores = qjl_scorer.score_paths(
            query_state=query_state,
            candidates=candidates,
            valid_mask=qjl_valid_mask,
            embed_weight=embed_weight,
        )
        approx_scores = (
            0.75 * _normalize_scores(medusa_path_scores)
            + 0.25 * _normalize_scores(qjl_path_scores)
        )
    else:
        approx_scores = medusa_path_scores

    selection_scores = _apply_acceptance_prune_threshold(
        approx_scores,
        medusa_path_scores,
        prune_threshold=acceptance_prune_threshold,
    )
    selected_paths = select_paths_for_pruning(
        selection_scores,
        keep_target=keep_target,
        min_keep=min_keep,
        max_keep=max_keep,
        retrieve_indices=retrieve_indices,
        tree_indices=tree_indices,
        node_budget=node_budget,
        mandatory_indices=mandatory_indices,
    )
    verify_full_tree = should_verify_full_tree(
        approx_scores,
        margin_scale=margin_scale,
    ) or should_skip_pruning_for_low_gain(
        selected_paths,
        approx_scores.shape[0],
        min_prune_fraction=min_prune_fraction,
    ) or not selected_paths_have_enough_node_gain(
        retrieve_indices,
        selected_paths,
        full_node_count=tree_indices.numel(),
        min_node_prune_fraction=min_node_prune_fraction,
    ) or not selected_paths_include_required(
        selected_paths,
        mandatory_indices,
    )
    return approx_scores, path_lengths, selected_paths, verify_full_tree


def plan_packed_kv_qjl_pruning(
    medusa_logits,
    logits,
    tree_indices,
    retrieve_indices,
    kv_qjl_path_scores,
    candidates=None,
    keep_fraction=0.30,
    keep_target=12,
    min_keep=6,
    max_keep=15,
    min_prune_fraction=0.25,
    min_node_prune_fraction=0.0,
    node_budget=0,
    kv_qjl_weight=0.5,
    medusa_pool_fraction=0.70,
    medusa_anchor_keep=2,
    acceptance_prune_threshold=0.0,
    acceptance_keep_threshold=0.0,
    acceptance_threshold_dynamic=False,
    acceptance_dynamic_prune_min=0.10,
    acceptance_dynamic_prune_max=0.45,
    acceptance_dynamic_keep_min=0.45,
    acceptance_dynamic_keep_max=0.70,
):
    """
    Select survivor paths using Medusa branch priors plus packed KV-QJL scores.

    This implements the aggressive prefilter architecture: packed sign(RK)
    scores are used only to compact the tree before exact verification. The
    final acceptance decision remains the normal high-accuracy verifier.
    """
    path_lengths = (retrieve_indices[:, 1:] >= 0).sum(dim=1)
    if kv_qjl_path_scores is None:
        return None, path_lengths, None, True

    medusa_path_scores = estimate_medusa_path_scores(
        medusa_logits,
        logits,
        tree_indices,
        retrieve_indices,
        candidates=candidates,
    )
    acceptance_prune_threshold, acceptance_keep_threshold = _resolve_acceptance_thresholds(
        medusa_path_scores,
        prune_threshold=acceptance_prune_threshold,
        keep_threshold=acceptance_keep_threshold,
        dynamic=acceptance_threshold_dynamic,
        dynamic_prune_min=acceptance_dynamic_prune_min,
        dynamic_prune_max=acceptance_dynamic_prune_max,
        dynamic_keep_min=acceptance_dynamic_keep_min,
        dynamic_keep_max=acceptance_dynamic_keep_max,
    )
    kv_qjl_weight = float(max(0.0, min(1.0, kv_qjl_weight)))
    total_paths = int(medusa_path_scores.shape[0])
    anchor_keep = min(total_paths, max(0, int(medusa_anchor_keep)))
    mandatory = _acceptance_anchor_path_indices(
        retrieve_indices,
        tree_indices,
        medusa_path_scores,
        max_long_paths=max(2, anchor_keep + 2),
        keep_threshold=acceptance_keep_threshold,
    )
    fused_scores = (
        (1.0 - kv_qjl_weight) * _normalize_scores(medusa_path_scores)
        + kv_qjl_weight * _normalize_scores(kv_qjl_path_scores)
    )
    length_bonus = 0.15 * _normalize_scores(path_lengths.to(torch.float32))
    ranked_scores = _apply_acceptance_prune_threshold(
        fused_scores + length_bonus,
        medusa_path_scores,
        prune_threshold=acceptance_prune_threshold,
    )

    selection_scores = ranked_scores
    pool_fraction = float(max(0.0, min(1.0, medusa_pool_fraction)))

    if 0.0 < pool_fraction < 1.0 and total_paths > 1:
        pool_keep = int(math.ceil(float(total_paths) * pool_fraction))
        pool_keep = min(
            total_paths,
            max(pool_keep, int(min_keep), int(max_keep), anchor_keep, int(mandatory.numel())),
        )
        medusa_pool = torch.topk(medusa_path_scores, k=pool_keep, dim=0).indices
        pool_parts = [medusa_pool]
        if mandatory.numel() > 0:
            pool_parts.append(mandatory)
        if anchor_keep > 0:
            pool_parts.append(torch.topk(medusa_path_scores, k=anchor_keep, dim=0).indices)
        medusa_pool = torch.unique(torch.cat(pool_parts))
        pool_mask = torch.zeros(total_paths, dtype=torch.bool, device=fused_scores.device)
        pool_mask[medusa_pool] = True
        spread = ranked_scores.std(unbiased=False) + 1.0
        floor = ranked_scores.min() - (10.0 * spread)
        selection_scores = torch.where(pool_mask, ranked_scores, floor)
    else:
        selection_scores = ranked_scores.clone()

    if anchor_keep > 0 and total_paths > 1:
        anchors = torch.topk(medusa_path_scores, k=anchor_keep, dim=0).indices
        anchor_boost = selection_scores.max() + torch.arange(
            anchor_keep,
            0,
            -1,
            device=selection_scores.device,
            dtype=selection_scores.dtype,
        )
        selection_scores[anchors] = anchor_boost
    if mandatory.numel() > 0 and total_paths > 1:
        required_boost = selection_scores.max() + torch.arange(
            int(mandatory.numel()),
            0,
            -1,
            device=selection_scores.device,
            dtype=selection_scores.dtype,
        )
        selection_scores[mandatory] = required_boost

    fraction_keep = max(1, int(math.ceil(float(total_paths) * float(keep_fraction))))
    keep_target = min(int(keep_target), fraction_keep)
    max_keep = min(int(max_keep), max(fraction_keep, 1))
    min_keep = min(int(min_keep), max_keep)
    selected_paths = select_paths_for_pruning(
        selection_scores,
        keep_target=keep_target,
        min_keep=min_keep,
        max_keep=max_keep,
        retrieve_indices=retrieve_indices,
        tree_indices=tree_indices,
        node_budget=node_budget,
        mandatory_indices=mandatory,
    )
    verify_full_tree = should_skip_pruning_for_low_gain(
        selected_paths,
        total_paths,
        min_prune_fraction=min_prune_fraction,
    ) or not selected_paths_have_enough_node_gain(
        retrieve_indices,
        selected_paths,
        full_node_count=tree_indices.numel(),
        min_node_prune_fraction=min_node_prune_fraction,
    ) or not selected_paths_include_required(
        selected_paths,
        mandatory,
    )
    return selection_scores, path_lengths, selected_paths, verify_full_tree


def build_pruned_medusa_buffers(
    full_tree_candidates,
    full_retrieve_indices,
    full_medusa_position_ids,
    full_medusa_attn_mask,
    selected_path_indices,
):
    """
    Build a compact Medusa tree containing only nodes needed by selected paths.
    """
    layout = build_pruned_medusa_layout(
        full_retrieve_indices,
        full_medusa_position_ids,
        full_medusa_attn_mask,
        selected_path_indices,
    )
    return materialize_pruned_medusa_buffers(full_tree_candidates, layout)


def build_cached_pruned_medusa_buffers(
    full_tree_candidates,
    full_retrieve_indices,
    full_medusa_position_ids,
    full_medusa_attn_mask,
    selected_path_indices,
    layout_cache=None,
    max_cache_size=128,
    min_node_prune_fraction=0.0,
):
    """
    Build/materialize pruned buffers while reusing selected-path layouts.

    Token values change every step, but the pruned tree layout depends only on
    selected path ids. Caching avoids repeatedly rebuilding masks and remaps.
    """
    if layout_cache is None:
        layout = build_pruned_medusa_layout(
            full_retrieve_indices,
            full_medusa_position_ids,
            full_medusa_attn_mask,
            selected_path_indices,
        )
        if not pruned_layout_has_enough_node_gain(
            layout,
            full_node_count=full_medusa_position_ids.numel(),
            min_node_prune_fraction=min_node_prune_fraction,
        ):
            return None
        return materialize_pruned_medusa_buffers(full_tree_candidates, layout)

    key_weights = torch.arange(
        1,
        selected_path_indices.numel() + 1,
        dtype=torch.long,
        device=selected_path_indices.device,
    )
    key_hash = torch.sum((selected_path_indices.to(torch.long) + 1) * key_weights)
    key = (int(selected_path_indices.numel()), int(key_hash.item()))
    layout = layout_cache.get(key)
    if layout is None:
        layout = build_pruned_medusa_layout(
            full_retrieve_indices,
            full_medusa_position_ids,
            full_medusa_attn_mask,
            selected_path_indices,
        )
        if len(layout_cache) >= int(max_cache_size):
            layout_cache.pop(next(iter(layout_cache)))
        layout_cache[key] = layout
    if not pruned_layout_has_enough_node_gain(
        layout,
        full_node_count=full_medusa_position_ids.numel(),
        min_node_prune_fraction=min_node_prune_fraction,
    ):
        return None
    return materialize_pruned_medusa_buffers(full_tree_candidates, layout)


def build_pruned_medusa_layout(
    full_retrieve_indices,
    full_medusa_position_ids,
    full_medusa_attn_mask,
    selected_path_indices,
):
    """
    Build reusable indexing/mask tensors for a compact selected-path tree.

    The selected layout depends only on path ids, not on token values, so callers
    can cache this when a pruning pattern repeats.
    """
    selected_paths = full_retrieve_indices.index_select(0, selected_path_indices)
    valid_mask = selected_paths >= 0
    selected_nodes = torch.unique(selected_paths[valid_mask])
    selected_nodes, _ = torch.sort(selected_nodes)

    if selected_nodes.numel() == 0:
        selected_nodes = torch.cat(
            [
                torch.zeros((1,), dtype=torch.long, device=full_retrieve_indices.device),
                selected_nodes,
            ],
            dim=0,
        )

    node_map = torch.full(
        (int(full_medusa_position_ids.numel()),),
        0,
        dtype=torch.long,
        device=full_retrieve_indices.device,
    )
    node_map[selected_nodes] = torch.arange(
        selected_nodes.shape[0], device=full_retrieve_indices.device, dtype=torch.long
    )

    mapped_retrieve_indices = selected_paths.clone()
    mapped_retrieve_indices[valid_mask] = node_map[selected_paths[valid_mask]]
    mapped_retrieve_indices[~valid_mask] = 0
    path_lengths = (selected_paths[:, 1:] >= 0).sum(dim=1)

    pruned_position_ids = full_medusa_position_ids.index_select(0, selected_nodes)
    attn2d = full_medusa_attn_mask[0, 0]
    pruned_attn = attn2d.index_select(0, selected_nodes).index_select(1, selected_nodes)
    pruned_attn = pruned_attn.unsqueeze(0).unsqueeze(0)

    token_indices = mapped_retrieve_indices.clone()
    token_indices[~valid_mask] = -1

    return {
        "selected_nodes": selected_nodes,
        "retrieve_indices": mapped_retrieve_indices,
        "medusa_position_ids": pruned_position_ids,
        "medusa_attn_mask": pruned_attn,
        "token_indices": token_indices,
        "path_lengths": path_lengths,
    }


def materialize_pruned_medusa_buffers(full_tree_candidates, layout):
    """
    Gather per-step token tensors using a prebuilt pruned tree layout.
    """
    selected_nodes = layout["selected_nodes"]
    token_indices = layout["token_indices"]

    if materialize_pruned_medusa_triton is not None:
        fast_materialized = materialize_pruned_medusa_triton(
            full_tree_candidates,
            selected_nodes,
            token_indices,
        )
        if fast_materialized is not None:
            pruned_tree_candidates, pruned_candidates = fast_materialized
            return {
                "tree_candidates": pruned_tree_candidates,
                "retrieve_indices": layout["retrieve_indices"],
                "medusa_position_ids": layout["medusa_position_ids"],
                "medusa_attn_mask": layout["medusa_attn_mask"],
                "candidates": pruned_candidates,
                "path_lengths": layout["path_lengths"],
            }

    pruned_tree_candidates = full_tree_candidates.index_select(1, selected_nodes)

    pruned_tree_ext = torch.cat(
        [
            pruned_tree_candidates,
            torch.zeros(
                (1, 1),
                dtype=torch.long,
                device=pruned_tree_candidates.device,
            ),
        ],
        dim=1,
    )
    # tree candidates are batch-1 by construction in this decoder path.
    if pruned_tree_ext.shape[0] != 1:
        raise ValueError("build_pruned_medusa_buffers currently supports batch size 1.")
    # Advanced indexing over dim=1: -1 targets appended zero-padding slot.
    pruned_candidates = pruned_tree_ext[0, token_indices]

    return {
        "tree_candidates": pruned_tree_candidates,
        "retrieve_indices": layout["retrieve_indices"],
        "medusa_position_ids": layout["medusa_position_ids"],
        "medusa_attn_mask": layout["medusa_attn_mask"],
        "candidates": pruned_candidates,
        "path_lengths": layout["path_lengths"],
    }


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    medusa_position_ids,
    input_ids,
    retrieve_indices,
    medusa_attn_mask=None,
    return_hidden=False,
    gather_paths=True,
    compute_medusa_logits=True,
    compute_orig_logits=True,
    fast_attention_mask=False,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    
    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - medusa_position_ids (torch.Tensor): Positional IDs associated with the Medusa structure.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.
    - medusa_attn_mask (torch.Tensor, optional): Per-call override for Medusa tree attention mask.
    
    Returns:
    - tuple: Returns medusa logits, regular logits, and other outputs from the model.
    """

    # Compute new position IDs by adding the Medusa position IDs to the length of the input sequence.
    position_ids = medusa_position_ids + input_ids.shape[1]

    attention_mask = None
    if fast_attention_mask and medusa_attn_mask is not None:
        attention_mask = get_cached_tree_attention_mask(
            model,
            medusa_attn_mask,
            past_len=input_ids.shape[1],
            dtype=model.base_model.dtype,
            device=tree_candidates.device,
        )

    if medusa_attn_mask is not None and attention_mask is None:
        model.base_model.model.medusa_mask = medusa_attn_mask
    elif attention_mask is not None:
        model.base_model.model.medusa_mask = None

    # Use the model to decode the tree candidates.
    # The model is expected to return logits for the Medusa structure, original logits, and possibly other outputs.
    if compute_orig_logits:
        tree_medusa_logits, outputs, tree_logits = model(
            tree_candidates,
            attention_mask=attention_mask,
            output_orig=True,
            past_key_values=past_key_values,
            position_ids=position_ids,
            medusa_forward=True,
            return_medusa_logits=compute_medusa_logits,
        )
    else:
        tree_medusa_logits, outputs = model(
            tree_candidates,
            attention_mask=attention_mask,
            output_orig=False,
            return_outputs=True,
            past_key_values=past_key_values,
            position_ids=position_ids,
            medusa_forward=True,
            return_medusa_logits=compute_medusa_logits,
        )
        tree_logits = None
    
    if not gather_paths:
        if return_hidden:
            return tree_medusa_logits, tree_logits, outputs, outputs[0][0]
        return tree_medusa_logits, tree_logits, outputs

    # Reorder the obtained logits based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = None if tree_logits is None else tree_logits[0, retrieve_indices]
    medusa_logits = None if tree_medusa_logits is None else tree_medusa_logits[:, 0, retrieve_indices]
    if return_hidden:
        hidden_paths = outputs[0][0, retrieve_indices]
        return medusa_logits, logits, outputs, hidden_paths
    return medusa_logits, logits, outputs


def evaluate_posterior_greedy_from_tree(
    tree_logits,
    candidates,
    retrieve_indices,
    path_lengths=None,
):
    """
    Greedy posterior evaluation directly from tree-node logits.

    This avoids materializing duplicated `[paths, depth, vocab]` logits. For
    temperature=0, posterior acceptance only needs each source node's argmax.
    """
    if greedy_tree_posterior_triton is not None:
        fast_result = greedy_tree_posterior_triton(
            tree_logits,
            candidates,
            retrieve_indices,
            path_lengths=path_lengths,
        )
        if fast_result is not None:
            return fast_result

    node_logits = tree_logits[0] if tree_logits.dim() == 3 else tree_logits
    node_argmax = torch.argmax(node_logits, dim=-1)
    source_nodes = retrieve_indices[:, :-1].clamp_min(0)
    posterior_mask = (candidates[:, 1:] == node_argmax[source_nodes]).int()
    if path_lengths is not None:
        valid_pos = (
            torch.arange(posterior_mask.shape[1], device=posterior_mask.device)
            .unsqueeze(0)
            < path_lengths.unsqueeze(1)
        )
        posterior_mask = posterior_mask * valid_pos.int()
    candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
    accept_length = candidates_accept_length.max()
    if accept_length == 0:
        best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
    else:
        best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
    return best_candidate, accept_length


def lm_head_argmax(hidden_states, lm_head_weight, chunk_size=4096, prefer_triton=True):
    """
    Compute argmax(hidden @ lm_head.T) without materializing the full logits.

    The Triton path stores only a small `[nodes, vocab_blocks]` partial-max
    buffer. The fallback streams vocab chunks through regular matmuls.
    """
    if hidden_states.dim() == 3:
        if hidden_states.shape[0] != 1:
            return torch.argmax(torch.matmul(hidden_states, lm_head_weight.t()), dim=-1)
        hidden_states = hidden_states[0]
    if hidden_states.dim() != 2 or lm_head_weight.dim() != 2:
        return torch.argmax(torch.matmul(hidden_states, lm_head_weight.t()), dim=-1)

    if prefer_triton and lm_head_argmax_triton is not None:
        node_argmax = lm_head_argmax_triton(hidden_states, lm_head_weight)
        if node_argmax is not None:
            return node_argmax

    vocab_size = int(lm_head_weight.shape[0])
    chunk_size = int(max(1, chunk_size))
    best_vals = None
    best_ids = None
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        scores = torch.matmul(hidden_states, lm_head_weight[start:end].t())
        vals, ids = torch.max(scores, dim=-1)
        ids = ids + start
        if best_vals is None:
            best_vals = vals
            best_ids = ids
        else:
            better = vals > best_vals
            best_vals = torch.where(better, vals, best_vals)
            best_ids = torch.where(better, ids, best_ids)
    return best_ids


def evaluate_posterior_greedy_from_argmax(
    node_argmax,
    candidates,
    retrieve_indices,
    path_lengths=None,
):
    """
    Greedy posterior evaluation when verifier node argmax ids are already known.
    """
    if greedy_accept_from_argmax_triton is not None:
        fast_result = greedy_accept_from_argmax_triton(
            node_argmax,
            candidates,
            retrieve_indices,
            path_lengths=path_lengths,
        )
        if fast_result is not None:
            return fast_result

    source_nodes = retrieve_indices[:, :-1].clamp_min(0)
    expected_tokens = node_argmax[source_nodes].to(candidates.dtype)
    posterior_mask = (candidates[:, 1:] == expected_tokens).int()
    if path_lengths is not None:
        valid_pos = (
            torch.arange(posterior_mask.shape[1], device=posterior_mask.device)
            .unsqueeze(0)
            < path_lengths.unsqueeze(1)
        )
        posterior_mask = posterior_mask * valid_pos.int()
    candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
    accept_length = candidates_accept_length.max()
    if accept_length == 0:
        best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
    else:
        best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
    return best_candidate, accept_length


def get_nucleus_posterior_mask(logits, candidates, temperature, top_p):
    """
    Generates a posterior mask for token candidates using nucleus (top-p) sampling.

    This function applies nucleus sampling to a set of logits, and then generates a mask indicating 
    which candidate tokens are selected. It adapts the sampling strategy to accommodate for 
    temperature scaling and cumulative probability thresholding.

    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        top_p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    if top_p >= 1:
        sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
        sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
        posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
        return posterior_mask
    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)

    
    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')
    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask

def get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha):
    """
    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        posterior_threshold (float): The minimum threshold for probabilities to be considered in sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * posterior_threshold,
            torch.exp(-entropy) * posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logits[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
    return posterior_mask
    
    

def evaluate_posterior(
    logits,
    candidates,
    temperature,
    posterior_threshold=0.3,
    posterior_alpha = 0.09,
    top_p=0.8,
    sampling='typical',
    fast=True,
    path_lengths=None,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.
    - top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
    - sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
    - fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        if path_lengths is not None:
            valid_pos = (
                torch.arange(posterior_mask.shape[1], device=posterior_mask.device)
                .unsqueeze(0)
                < path_lengths.unsqueeze(1)
            )
            posterior_mask = posterior_mask * valid_pos.int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
        
    if sampling == 'typical':
        if fast:
            posterior_prob = torch.softmax(logits[:, :-1] / temperature, dim=-1)
            candidates_prob = torch.gather(
                posterior_prob, dim=-1, index=candidates[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            posterior_entropy = -torch.sum(
                posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
            )  # torch.sum(torch.log(*)) is faster than torch.prod
            threshold = torch.minimum(
                torch.ones_like(posterior_entropy) * posterior_threshold,
                torch.exp(-posterior_entropy) * posterior_alpha,
            )
            posterior_mask = candidates_prob > threshold
            if path_lengths is not None:
                valid_pos = (
                    torch.arange(posterior_mask.shape[1], device=posterior_mask.device)
                    .unsqueeze(0)
                    < path_lengths.unsqueeze(1)
                )
                posterior_mask = posterior_mask & valid_pos
            candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)

            # Choose the best candidate based on the evaluated posterior probabilities
            accept_length = candidates_accept_length.max()
            if accept_length == 0:
                # If no candidates are accepted, just choose the first one
                best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            else:
                best_candidates = torch.where(candidates_accept_length == accept_length)[0]
                # Accept the best one according to likelihood
                likelihood = torch.sum(
                    torch.log(candidates_prob[best_candidates, :accept_length]), dim=-1
                )
                best_candidate = best_candidates[torch.argmax(likelihood)]
            return best_candidate, accept_length
        # Calculate posterior probabilities and thresholds for candidate selection
        posterior_mask = get_typical_posterior_mask(
            logits, candidates, temperature, posterior_threshold, posterior_alpha
        )
        if path_lengths is not None:
            valid_pos = (
                torch.arange(posterior_mask.shape[1], device=posterior_mask.device)
                .unsqueeze(0)
                < path_lengths.unsqueeze(1)
            )
            posterior_mask = posterior_mask * valid_pos.int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        # Choose the best candidate based on the evaluated posterior probabilities
        accept_length = candidates_accept_length.max()
        
        if accept_length == 0:
            # If no candidates are accepted, just choose the first one
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
            # Accept the best one according to likelihood
        return best_candidate, accept_length
    
    if sampling == 'nucleus':
        assert top_p < 1.0 + 1e-6, "top_p should between 0 and 1"
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        if path_lengths is not None:
            valid_pos = (
                torch.arange(posterior_mask.shape[1], device=posterior_mask.device)
                .unsqueeze(0)
                < path_lengths.unsqueeze(1)
            )
            posterior_mask = posterior_mask * valid_pos.int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    else:
        raise NotImplementedError
def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    outputs,
    logits,
    medusa_logits,
    new_token,
    past_key_values_data,
    current_length_data,
    past_key_values=None,
    input_ids_buffer=None,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - outputs, logits, medusa_logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - medusa_logits (torch.Tensor): Updated medusa logits.
    - new_token (int): Updated counter for the new tokens added.
    """
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    accept_len_int = int(accept_length.item() if torch.is_tensor(accept_length) else accept_length)
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_len_int + 1] + prev_input_len
    )
    input_ids = append_input_ids(
        input_ids,
        candidates[None, best_candidate, : accept_len_int + 1],
        input_ids_buffer=input_ids_buffer,
    )
    # Update the past key values based on the selected tokens.
    if accept_len_int == 0:
        # The root candidate KV is already written at prev_input_len by tree_decoding,
        # so copying that position onto itself only burns bandwidth for flat FP
        # caches. Nested compressed caches still need copy() to trim their child
        # lengths back from the full tree append to the single accepted root.
        if past_key_values_data is None and past_key_values is not None:
            root_index = select_indices[:1]
            for layer_kv in past_key_values:
                for kv_cache in layer_kv:
                    kv_cache.copy(root_index, prev_input_len, dim=2)
        current_length_data.fill_(prev_input_len + 1)
    elif past_key_values_data is not None:
        copied = False
        if copy_selected_kv_cache_triton is not None:
            # rel=0 is the root token and is already at prev_input_len.
            copied = copy_selected_kv_cache_triton(
                past_key_values_data,
                select_indices,
                prev_input_len,
                copy_start=1,
            )
        if not copied:
            # Source tensor that contains relevant past information based on the selected candidate
            tgt = past_key_values_data[..., select_indices, :]
            # Destination tensor where the relevant past information will be stored
            dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
            # Copy relevant past information from the source to the destination
            dst.copy_(tgt, non_blocking=True)
            copied_len = tgt.shape[-2]
        else:
            copied_len = select_indices.shape[0]
        # Update the current length tensor (currently only support batch size is 1)
        current_length_data.fill_(prev_input_len + copied_len)
    else:
        if past_key_values is None:
            raise ValueError(
                "past_key_values must be provided when past_key_values_data is None."
            )
        for layer_kv in past_key_values:
            for kv_cache in layer_kv:
                kv_cache.copy(select_indices, prev_input_len, dim=2)
        current_length_data.fill_(prev_input_len + select_indices.shape[0])

    # Extract logits and medusa logits for the accepted tokens
    logits = logits[None, best_candidate, accept_len_int : accept_len_int + 1]
    medusa_logits = medusa_logits[
        :, None, best_candidate, accept_len_int : accept_len_int + 1
    ]
    # Update the new token counter
    new_token += accept_len_int + 1

    return input_ids, logits, medusa_logits, new_token


def update_inference_inputs_from_tree(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    outputs,
    tree_logits,
    tree_medusa_logits,
    new_token,
    past_key_values_data,
    current_length_data,
    past_key_values=None,
    input_ids_buffer=None,
    lm_head=None,
    tree_hidden=None,
):
    """
    Update inference state using raw tree-node logits instead of gathered path logits.
    """
    prev_input_len = input_ids.shape[1]
    accept_len_int = int(accept_length.item() if torch.is_tensor(accept_length) else accept_length)
    select_indices = (
        retrieve_indices[best_candidate, : accept_len_int + 1] + prev_input_len
    )
    input_ids = append_input_ids(
        input_ids,
        candidates[None, best_candidate, : accept_len_int + 1],
        input_ids_buffer=input_ids_buffer,
    )

    if accept_len_int == 0:
        # The accepted root KV is already in the first tree slot.
        if past_key_values_data is None and past_key_values is not None:
            root_index = select_indices[:1]
            for layer_kv in past_key_values:
                for kv_cache in layer_kv:
                    kv_cache.copy(root_index, prev_input_len, dim=2)
        current_length_data.fill_(prev_input_len + 1)
    elif past_key_values_data is not None:
        copied = False
        if copy_selected_kv_cache_triton is not None:
            copied = copy_selected_kv_cache_triton(
                past_key_values_data,
                select_indices,
                prev_input_len,
                copy_start=1,
            )
        if not copied:
            tgt = past_key_values_data[..., select_indices, :]
            dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
            dst.copy_(tgt, non_blocking=True)
            copied_len = tgt.shape[-2]
        else:
            copied_len = select_indices.shape[0]
        current_length_data.fill_(prev_input_len + copied_len)
    else:
        if past_key_values is None:
            raise ValueError(
                "past_key_values must be provided when past_key_values_data is None."
            )
        for layer_kv in past_key_values:
            for kv_cache in layer_kv:
                kv_cache.copy(select_indices, prev_input_len, dim=2)
        current_length_data.fill_(prev_input_len + select_indices.shape[0])

    node_idx = retrieve_indices[best_candidate, accept_len_int].reshape(1)
    if tree_logits is None:
        if lm_head is None or tree_hidden is None:
            raise ValueError("lm_head and tree_hidden are required when tree_logits is None.")
        accepted_hidden = tree_hidden.index_select(0, node_idx)
        logits = lm_head(accepted_hidden.unsqueeze(0))
    else:
        logits = tree_logits.index_select(1, node_idx)
    medusa_logits = None if tree_medusa_logits is None else tree_medusa_logits.index_select(2, node_idx)
    new_token += accept_len_int + 1

    return input_ids, logits, medusa_logits, new_token
