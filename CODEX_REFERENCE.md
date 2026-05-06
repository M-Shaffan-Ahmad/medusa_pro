# Medusa + TurboQuant: Quick Reference for Code Generation

## TL;DR
**Medusa** adds multiple prediction heads to LLMs to predict multiple tokens in parallel. **TurboQuant** compresses the model to 1/4 size. Together: **3-4x speedup + 75% memory reduction**.

---

## Core Architecture

### Three Main Components

#### 1. Base Model (Frozen or Quantized)
```python
# Standard LLaMA/Mistral architecture
base_model = transformers.AutoModelForCausalLM.from_pretrained("vicuna-7b")
# Can be quantized to INT8 with TurboQuant
```

#### 2. Prediction Heads (New Learnable Layers)
```python
class MedusaHeads(nn.Module):
    """Predict multiple future tokens in parallel"""
    def __init__(self, hidden_size=4096, vocab_size=32000, num_heads=4, depth=4):
        super().__init__()
        # Each head predicts 'depth' tokens ahead
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, vocab_size * depth)
            )
            for _ in range(num_heads)
        ])
        self.depth = depth
        self.num_heads = num_heads
    
    def forward(self, hidden_states):
        # Input: (batch, seq_len, hidden_size)
        # Output: (batch, seq_len, vocab_size, depth, num_heads)
        outputs = []
        for head in self.heads:
            logits = head(hidden_states)  # (batch, seq_len, vocab_size * depth)
            logits = logits.reshape(-1, logits.shape[-2], self.vocab_size, self.depth)
            outputs.append(logits)
        return torch.stack(outputs, dim=-1)
```

#### 3. Tree Verification (Candidate Acceptance)
```python
def verify_and_accept_candidates(base_model, candidates, context, threshold=0.8):
    """Verify which candidate token sequences are valid"""
    # Build tree structure from candidates
    tree = CandidateTree(candidates)
    
    # Create attention mask that prevents context mixing
    attn_mask = create_tree_attention_mask(tree)
    
    # Run base model to verify candidates
    with torch.no_grad():
        outputs = base_model(context, attention_mask=attn_mask)
    
    # Extract valid sequences
    valid_sequences = []
    for path in tree.all_paths():
        prob = compute_sequence_probability(outputs, path)
        if prob >= threshold:
            valid_sequences.append(path)
    
    # Return longest valid sequence
    return longest_sequence(valid_sequences)
```

---

## Implementation Steps

### Step 1: Forward Pass with Speculation
```python
def medusa_forward(model, input_ids, max_new_tokens=50):
    """
    Main inference loop with Medusa
    
    Traditional:
      for i in range(max_new_tokens):
          next_token = model(input_ids).argmax(-1)
          input_ids = cat([input_ids, next_token])
    
    Medusa:
      We predict k tokens ahead with multiple heads
    """
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        while len(generated_ids) < max_new_tokens:
            # Get base model hidden states
            outputs = base_model(generated_ids)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # Get predictions from all heads
            head_logits = medusa_heads(hidden_states)  # (batch, seq, vocab, depth, num_heads)
            
            # Convert to probabilities
            head_probs = torch.softmax(head_logits, dim=2)
            
            # Get top-k candidates for each head and position
            candidates = generate_tree_candidates(head_probs, k=3)
            
            # Verify candidates with base model
            verified_tokens = verify_and_accept_candidates(
                base_model, candidates, generated_ids
            )
            
            # Accept valid tokens and continue
            generated_ids = torch.cat([generated_ids, verified_tokens])
    
    return generated_ids
```

### Step 2: Tree-Based Attention Mask
```python
def create_tree_attention_mask(num_branches, depth, seq_len):
    """
    Prevent context mixing between different candidate paths
    
    Tree structure:
           root (original sequence)
          / | \ (multiple branches)
         b1 b2 b3 (different candidates)
        /  |  \ (continuations)
       ...
    
    Attention should NOT flow between branches
    """
    batch_size = 1
    total_seq_len = seq_len + depth * num_branches
    
    # Start with causal mask
    mask = torch.full((total_seq_len, total_seq_len), float('-inf'))
    mask = torch.triu(mask, diagonal=1)  # Causal
    
    # Allow attention within same branch
    for branch_id in range(num_branches):
        start_pos = seq_len + branch_id * depth
        end_pos = start_pos + depth
        
        # Within branch: allow full attention
        mask[start_pos:end_pos, start_pos:end_pos] = 0
        
        # From branch to original: allow attention
        mask[start_pos:end_pos, :seq_len] = 0
    
    # Prevent cross-branch attention
    for i in range(num_branches):
        for j in range(num_branches):
            if i != j:
                start_i = seq_len + i * depth
                start_j = seq_len + j * depth
                mask[start_i:start_i+depth, start_j:start_j+depth] = float('-inf')
    
    return mask
```

### Step 3: Training Medusa Heads
```python
def train_medusa_heads(base_model, train_dataloader, num_epochs=3, lr=1e-4):
    """Fine-tune only the Medusa heads, keep base model frozen"""
    
    heads = MedusaHeads(hidden_size=4096, vocab_size=32000)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=lr)
    
    base_model.eval()  # Freeze base model
    heads.train()
    
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            # Forward through base model (no gradient)
            with torch.no_grad():
                outputs = base_model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
            
            # Forward through Medusa heads
            head_logits = heads(hidden_states)
            
            # Compute loss for next K tokens
            loss = 0
            for head_idx in range(head_logits.shape[-1]):
                for depth_idx in range(head_logits.shape[-2]):
                    # Shift labels and targets for prediction
                    shift_logits = head_logits[..., depth_idx, head_idx, :-1, :].contiguous()
                    shift_labels = labels[..., depth_idx, :-1].contiguous()
                    
                    loss += F.cross_entropy(
                        shift_logits.view(-1, 32000),
                        shift_labels.view(-1)
                    )
            
            # Backward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Step 4: TurboQuant Integration
```python
import torch.quantization as tq

def apply_turboquant(model):
    """Convert model to INT8 quantization"""
    
    # Prepare model for quantization
    model.qconfig = tq.get_default_qat_qconfig('fbgemm')
    tq.prepare(model, inplace=True)
    
    # Calibration (optional, for better quality)
    # Run model on calibration data
    for batch in calibration_data:
        model(batch['input_ids'])
    
    # Convert to quantized model
    tq.convert(model, inplace=True)
    
    return model

# Usage
quantized_model = apply_turboquant(base_model)
medusa_heads_quantized = apply_turboquant(medusa_heads)

# Result: 4x memory reduction, slightly faster inference
print(f"Model size: {model.get_buffer_size() / 1e9:.2f} GB")  # ~3.5 GB instead of 14 GB
```

---

## Tree Candidate Generation Algorithm

```python
def generate_tree_candidates(head_probs, k=3):
    """
    Generate candidate tree from head predictions
    
    Input: head_probs of shape (batch, seq_len, vocab_size, depth, num_heads)
    Output: Tree structure with all possible paths
    """
    
    class Node:
        def __init__(self, token_id, prob, depth):
            self.token_id = token_id
            self.prob = prob
            self.depth = depth
            self.children = []
            self.path = [token_id]
    
    # For each position, get top-k tokens from each head
    candidates_by_position = []
    for pos in range(head_probs.shape[-2]):  # depth dimension
        head_topk = []
        for head_idx in range(head_probs.shape[-1]):  # num_heads
            topk_probs, topk_ids = torch.topk(head_probs[:, -1, :, pos, head_idx], k=k)
            head_topk.append(list(zip(topk_ids[0].tolist(), topk_probs[0].tolist())))
        candidates_by_position.append(head_topk)
    
    # Build tree
    root = Node(None, 1.0, 0)
    queue = [root]
    
    for pos, candidates in enumerate(candidates_by_position):
        new_queue = []
        for parent in queue:
            for head_idx, token_candidates in enumerate(candidates):
                for token_id, prob in token_candidates[:k]:
                    node = Node(token_id, prob, pos)
                    node.path = parent.path + [token_id]
                    parent.children.append(node)
                    new_queue.append(node)
        queue = new_queue
    
    return root
```

---

## Performance Calculation

```python
def calculate_speedup(baseline_latency, medusa_latency, num_tokens):
    """
    Example:
    - Baseline: 4.2 seconds per token
    - With Medusa (4 heads, depth 4): 5.0s initial + 4 accepted tokens → 5.0s for 4 tokens
    - Speedup: (4 * 4.2) / 5.0 = 3.36x
    
    With Quantization:
    - Quantized model: 2.1s per token
    - With Medusa: 2.5s initial + 4 tokens → 2.5s for 4 tokens
    - Speedup: (4 * 2.1) / 2.5 = 3.36x on top of quantization speedup!
    """
    
    baseline_total = baseline_latency * num_tokens
    medusa_total = medusa_latency
    
    speedup = baseline_total / medusa_total
    return speedup
```

---

## Key Hyperparameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `num_heads` | 4 | 1-8 | More heads = more parallelism, more memory |
| `depth` | 4 | 1-8 | How many tokens ahead to predict |
| `vocab_size_k` | 3 | 1-10 | Top-k candidates per head |
| `acceptance_threshold` | 0.75 | 0.5-0.95 | Higher = more conservative |
| `quantization_bits` | 8 | 4-8 | Lower = smaller model, lower quality |

---

## Common Issues & Solutions

### Issue 1: Low Acceptance Rate (< 50%)
```python
# Problem: Too many rejected tokens
# Solution: Lower acceptance threshold or reduce num_heads

# Try:
acceptance_threshold = 0.5  # More lenient
num_heads = 2  # Fewer predictions = higher confidence
```

### Issue 2: Quality Degradation
```python
# Problem: Generated text is lower quality
# Solution: Use more heads, train longer, or higher acceptance threshold

# Try:
num_heads = 8  # Better diversity
training_epochs = 5  # Train longer
acceptance_threshold = 0.8  # Higher confidence required
```

### Issue 3: Memory OOM with Quantization
```python
# Problem: Still running out of memory
# Solution: Use lower precision, reduce batch size

import bitsandbytes as bnb  # 4-bit quantization
model = bnb.load_quantized_model("vicuna-7b", load_in_4bit=True)
```

### Issue 4: Slow Verification
```python
# Problem: Tree verification is bottleneck
# Solution: Cache attention masks, use smaller trees

# Optimize:
cache_attention_masks = True
max_tree_size = 16  # Limit tree candidates
batch_tree_verification = True  # Batch verify multiple trees
```

---

## Quick Implementation Checklist

- [ ] Load base LLM model
- [ ] Create MedusaHeads architecture
- [ ] Implement tree generation logic
- [ ] Implement attention masking for tree
- [ ] Implement candidate verification
- [ ] Train heads on dataset
- [ ] Apply TurboQuant quantization
- [ ] Benchmark against baseline
- [ ] Evaluate quality metrics
- [ ] Optimize for production
- [ ] Create API/inference server
- [ ] Document usage

---

## Code Snippets for Codex

### For LLaMA/Vicuna compatibility:
```python
# Use compatible base models
compatible_models = [
    "meta-llama/Llama-2-7b",
    "lmsys/vicuna-7b-v1.3",
    "mistralai/Mistral-7B",
]

model = AutoModelForCausalLM.from_pretrained(model_name)
```

### For distributed training:
```python
from torch.nn.parallel import DataParallel, DistributedDataParallel

# Single GPU
model = DataParallel(medusa_heads)

# Multi-GPU
model = DistributedDataParallel(
    medusa_heads,
    device_ids=[0, 1],
    output_device=0
)
```

### For batched inference:
```python
def batch_medusa_inference(batch_prompts, batch_size=4):
    results = []
    for i in range(0, len(batch_prompts), batch_size):
        batch = batch_prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True)
        outputs = medusa_forward(model, inputs['input_ids'])
        results.extend(tokenizer.batch_decode(outputs))
    return results
```

---

## Expected Metrics

After implementation, you should see:

```
Vicuna-7B Baseline:
- Tokens/sec: 25
- Memory: 14 GB
- Latency/token: 40ms

With Medusa:
- Tokens/sec: 85 (3.4x faster)
- Memory: 15.5 GB (+11%)
- Latency for 4 tokens: 200ms (50ms/token)

With Medusa + TurboQuant:
- Tokens/sec: 110 (4.4x faster overall)
- Memory: 3.9 GB (75% reduction)
- Latency for 4 tokens: 160ms (40ms/token)
```

---

## References

- Base Code: `/home/shaffan/Desktop/Uni/PDC/Project/Medusa/`
- Paper: https://arxiv.org/abs/2401.10774
- Transformers: https://huggingface.co/docs/transformers/
- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
