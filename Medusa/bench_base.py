import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TARGET_NEW_TOKENS = 200


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

print("Loading Standard Baseline Model...")
# Load the raw base model (No Medusa)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    torch_dtype=torch.float16
).to("cuda")

prompt = "<|user|>\nWrite a C++ program using MPI and OpenMP for parallel matrix multiplication.\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Generating Autoregressively (1 token per pass)...")
sync()
start_time = time.perf_counter()

with torch.inference_mode():
    # Force greedy decoding (do_sample=False) for exact reproducibility
    outputs = model.generate(**inputs, max_new_tokens=TARGET_NEW_TOKENS, do_sample=False)

sync()
end_time = time.perf_counter()

print("\n" + "-"*30)
print(" BASELINE OUTPUT TEXT")
print("-"*30)
# Decode the output tensor back into human-readable text
baseline_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(baseline_text)

# Calculate the metrics
generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
time_taken = end_time - start_time
tps = generated_tokens / time_taken

print("\n" + "="*30)
print("  BASELINE RESULTS (Sequential)")
print("="*30)
print(f"Total Tokens: {generated_tokens}")
print(f"Total Time:   {time_taken:.2f} seconds")
print(f"Speed (TPS):  {tps:.2f} tokens/sec")
