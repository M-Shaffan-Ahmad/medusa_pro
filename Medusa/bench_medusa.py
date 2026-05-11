import time
import torch
from transformers import AutoTokenizer
from medusa.model.medusa_model import MedusaModel

TARGET_NEW_TOKENS = 200


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

print("Loading Medusa Accelerated Model...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

model = MedusaModel.from_pretrained(
    "./TinyLlama-1.1B-Chat-v1.0-4heads", 
    torch_dtype=torch.float16
).to("cuda")

prompt = "<|user|>\nWrite a C++ program using MPI and OpenMP for parallel matrix multiplication.\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Generating with Medusa Tree Decoding (Parallel)...")
sync()
start_time = time.perf_counter()

with torch.inference_mode():
    output_stream = model.medusa_generate(
        inputs.input_ids,
        temperature=0.0,
        max_steps=200,
        max_new_tokens=TARGET_NEW_TOKENS,
        stream=False,
        collect_stats=True,
    )
    
    # Exhaust the stream to get the final dictionary state
    for current_output in output_stream:
        final_data = current_output

sync()
end_time = time.perf_counter()

# Extract the final text from the dictionary
final_text = final_data["text"]

# --- ADD THESE LINES TO PRINT TEXT ---
print("\n" + "-"*30)
print(" MEDUSA OUTPUT TEXT")
print("-"*30)
print(final_text)

generated_tokens = int(final_data.get("stats", {}).get("generated_tokens", 0))
if generated_tokens <= 0:
    generated_tokens = len(tokenizer(final_text, add_special_tokens=False).input_ids)
generated_tokens = max(1, generated_tokens)

time_taken = end_time - start_time
tps = generated_tokens / time_taken

print("\n" + "="*30)
print("  MEDUSA RESULTS (Parallel)")
print("="*30)
print(f"Total Tokens: {generated_tokens}")
print(f"Total Time:   {time_taken:.2f} seconds")
print(f"Speed (TPS):  {tps:.2f} tokens/sec")
