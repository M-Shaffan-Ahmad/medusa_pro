import time
import torch
import csv
import gc
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from medusa.model.medusa_model import MedusaModel

TARGET_NEW_TOKENS = 200


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# The Full 10-Prompt Test Suite
TEST_SUITE = [
    {"category": "HPC/C++ (Hard)", "prompt": "Write a C++ program using MPI and OpenMP for parallel matrix multiplication."},
    {"category": "Python (Easy)", "prompt": "Write a simple Python function to check if a string is a palindrome."},
    {"category": "Sci-Fi (Medium)", "prompt": "Write a short sci-fi story about a sentient probe exploring the event horizon of a black hole."},
    {"category": "History (Easy)", "prompt": "Explain the primary causes of the French Revolution in detail."},
    {"category": "Math/Logic (Hard)", "prompt": "Provide a step-by-step mathematical proof of the Pythagorean theorem."},
    {"category": "Cooking/Recipe (Easy)", "prompt": "Give me a highly detailed recipe for making authentic Italian carbonara from scratch."},
    {"category": "Biology (Medium)", "prompt": "Describe the process of cellular respiration and exactly how ATP is generated."},
    {"category": "Physics (Medium)", "prompt": "Explain the concept of quantum entanglement as if I were a high school physics student."},
    {"category": "Economics (Medium)", "prompt": "What are the macroeconomic effects of raising interest rates to combat high inflation?"},
    {"category": "Philosophy (Hard)", "prompt": "Compare and contrast Immanuel Kant's Categorical Imperative with Utilitarianism."}
]

results = []
baseline_texts_dict = {} 
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# ==========================================
# PHASE 1: BASELINE PROFILING (STREAMING)
# ==========================================
print("\n[PHASE 1] Loading Standard Baseline Model...")
baseline_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    torch_dtype=torch.float16
).to("cuda")

for i, test in enumerate(TEST_SUITE):
    print(f"  -> Profiling Baseline {i+1}/10: {test['category']}")
    full_prompt = f"<|user|>\n{test['prompt']}\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs.input_ids.shape[1]

    sync()
    start_time = time.perf_counter()
    with torch.inference_mode():
        output_ids = baseline_model.generate(
            **inputs,
            max_new_tokens=TARGET_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    sync()
    end_time = time.perf_counter()

    generated_ids = output_ids[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    token_count = int(generated_ids.shape[0])
    
    baseline_texts_dict[i] = generated_text
    
    ttft = ""
    decode_time = end_time - start_time
    tpot = decode_time / (token_count - 1) if token_count > 1 else 0 
    
    tps = token_count / (end_time - start_time)
    
    results.append({
        "Prompt_ID": i+1, "Category": test["category"], "Mode": "Baseline", 
        "Tokens": token_count, "Compute_TTFT_Sec": ttft,
        "Memory_Decode_Sec": round(decode_time, 2), "TPOT_ms": round(tpot * 1000, 2),
        "Overall_TPS": round(tps, 2), "Exact_Match": "N/A"
    })

print("\n[MEMORY FLUSH] Unloading Baseline from VRAM...")
del baseline_model
torch.cuda.empty_cache()
gc.collect()
time.sleep(2)

# ==========================================
# PHASE 2: MEDUSA PROFILING
# ==========================================
print("\n[PHASE 2] Loading Medusa Accelerated Model...")
medusa_model = MedusaModel.from_pretrained(
    "./TinyLlama-1.1B-Chat-v1.0-4heads", 
    torch_dtype=torch.float16
).to("cuda")

for i, test in enumerate(TEST_SUITE):
    print(f"  -> Profiling Medusa {i+1}/10: {test['category']}")
    full_prompt = f"<|user|>\n{test['prompt']}\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    sync()
    start_time = time.perf_counter()

    with torch.inference_mode():
        output_stream = medusa_model.medusa_generate(
            inputs.input_ids,
            temperature=0.0,
            max_steps=200,
            max_new_tokens=TARGET_NEW_TOKENS,
            stream=False,
            collect_stats=True,
        )
        for current_output in output_stream:
            final_data = current_output

    sync()
    end_time = time.perf_counter()
    
    medusa_gen_text = final_data["text"]
    
    base_clean = re.sub(r'\s+', '', baseline_texts_dict[i])
    med_clean = re.sub(r'\s+', '', medusa_gen_text)
    compare_length = min(len(base_clean), len(med_clean))
    is_match = base_clean[:compare_length] == med_clean[:compare_length] if compare_length > 0 else False
    
    gen_tokens = int(final_data.get("stats", {}).get("generated_tokens", 0))
    if gen_tokens <= 0:
        gen_tokens = len(tokenizer(medusa_gen_text, add_special_tokens=False).input_ids)
    if gen_tokens <= 0:
        gen_tokens = 1

    ttft = ""
    decode_time = end_time - start_time
    tpot = decode_time / (gen_tokens - 1) if gen_tokens > 1 else 0
    
    tps = gen_tokens / (end_time - start_time)
    
    results.append({
        "Prompt_ID": i+1, "Category": test["category"], "Mode": "Medusa", 
        "Tokens": gen_tokens, "Compute_TTFT_Sec": ttft,
        "Memory_Decode_Sec": round(decode_time, 2), "TPOT_ms": round(tpot * 1000, 2),
        "Overall_TPS": round(tps, 2), "Exact_Match": is_match
    })

# ==========================================
# PHASE 3: EXPORTING DATA
# ==========================================
csv_file = "hpc_profiling_results_full.csv"
print(f"\n[PHASE 3] Writing results to {csv_file}...")

with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Prompt_ID", "Category", "Mode", "Tokens", "Compute_TTFT_Sec", "Memory_Decode_Sec", "TPOT_ms", "Overall_TPS", "Exact_Match"])
    writer.writeheader()
    writer.writerows(results)

print("\n🎉 Full 10-Prompt Profiling Complete!")
