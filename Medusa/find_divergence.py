import time
import torch
import re
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from medusa.model.medusa_model import MedusaModel

prompt_text = "Write a short sci-fi story about a sentient probe exploring the event horizon of a black hole."
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
full_prompt = f"<|user|>\n{prompt_text}\n<|assistant|>\n"
inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
prompt_len = inputs.input_ids.shape[1]

# 1. RUN BASELINE
print("Running Baseline...")
baseline_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16).to("cuda")
with torch.no_grad():
    outputs = baseline_model.generate(**inputs, max_new_tokens=200, do_sample=False)
baseline_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

del baseline_model
torch.cuda.empty_cache()
gc.collect()

# 2. RUN MEDUSA
print("Running Medusa...")
medusa_model = MedusaModel.from_pretrained("./TinyLlama-1.1B-Chat-v1.0-4heads", torch_dtype=torch.float16).to("cuda")
with torch.no_grad():
    output_stream = medusa_model.medusa_generate(inputs.input_ids, temperature=0.0, max_steps=200)
    for current_output in output_stream:
        final_data = current_output
medusa_text = final_data["text"]

# 3. FIND EXACT DIVERGENCE
base_clean = re.sub(r'\s+', '', baseline_text)
med_clean = re.sub(r'\s+', '', medusa_text)

min_len = min(len(base_clean), len(med_clean))
divergence_index = -1

for i in range(min_len):
    if base_clean[i] != med_clean[i]:
        divergence_index = i
        break

print("\n" + "="*50)
if divergence_index == -1:
    print("Wait, they actually matched perfectly! The cutoff just sliced the last word.")
else:
    print(f"DIVERGENCE FOUND AT CHARACTER {divergence_index}!")
    # Print a window of 30 characters before and after the split
    print(f"Baseline path: '...{base_clean[divergence_index-30 : divergence_index+30]}...'")
    print(f"Medusa path:   '...{med_clean[divergence_index-30 : divergence_index+30]}...'")
print("="*50)