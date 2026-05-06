import time
import torch
from transformers import AutoTokenizer
from medusa.model.medusa_model import MedusaModel

print("Loading Medusa Accelerated Model...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

model = MedusaModel.from_pretrained(
    "./TinyLlama-1.1B-Chat-v1.0-4heads", 
    torch_dtype=torch.float16
).to("cuda")

prompt = "<|user|>\nWrite a C++ program using MPI and OpenMP for parallel matrix multiplication.\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
prompt_length = inputs.input_ids.shape[1]

print("Generating with Medusa Tree Decoding (Parallel)...")
start_time = time.time()

with torch.no_grad():
    output_stream = model.medusa_generate(
        inputs.input_ids, 
        temperature=0.0, 
        max_steps=200
    )
    
    # Exhaust the stream to get the final dictionary state
    for current_output in output_stream:
        final_data = current_output
        
end_time = time.time()

# Extract the final text from the dictionary
final_text = final_data["text"]

# --- ADD THESE LINES TO PRINT TEXT ---
print("\n" + "-"*30)
print(" MEDUSA OUTPUT TEXT")
print("-"*30)
print(final_text)

# Re-tokenize the text so we can accurately count the tokens
total_tokens = len(tokenizer.encode(final_text))
generated_tokens = total_tokens - prompt_length



# Ensure we don't divide by zero if it stops early
if generated_tokens <= 0:
    generated_tokens = 1

time_taken = end_time - start_time
tps = generated_tokens / time_taken

print("\n" + "="*30)
print("  MEDUSA RESULTS (Parallel)")
print("="*30)
print(f"Total Tokens: {generated_tokens}")
print(f"Total Time:   {time_taken:.2f} seconds")
print(f"Speed (TPS):  {tps:.2f} tokens/sec")
