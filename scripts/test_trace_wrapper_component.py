import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dual_loop.neuromorphic import attach_hadl_v5_spiking_brain

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {DEVICE}...")
tok = AutoTokenizer.from_pretrained("dist/staging_hf_fused")
base_m = AutoModelForCausalLM.from_pretrained(
    "dist/staging_hf_fused",
    torch_dtype=torch.bfloat16,
    device_map=DEVICE
)
lm = PeftModel.from_pretrained(base_m, "dist/frontier_lora_adapter")
model = attach_hadl_v5_spiking_brain(lm)
model.tokenizer = tok

mbpp_prompt = (
    "Problem 2: In Python, write a function to check if all elements in a list are positive.\n"
    "Provide the Python code definition for `all_positive(lst)`:\n"
    "Answer:"
)
inp = tok(mbpp_prompt, return_tensors="pt").to(DEVICE)

# Case 1: Wrapper with enable_answer_first=False
print("\n--- CASE 1: Wrapper with enable_answer_first=False ---")
with torch.no_grad():
    out1 = model.generate(**inp, max_new_tokens=40, do_sample=False, enable_answer_first=False)
print("Case 1 (enable_answer_first=False):", repr(tok.decode(out1[0][inp.input_ids.shape[1]:])))

# Case 2: Wrapper with enable_answer_first=True
print("\n--- CASE 2: Wrapper with enable_answer_first=True ---")
with torch.no_grad():
    out2 = model.generate(**inp, max_new_tokens=40, do_sample=False, enable_answer_first=True, prompt_text=mbpp_prompt, tokenizer=tok)
print("Case 2 (enable_answer_first=True):", repr(tok.decode(out2[0][inp.input_ids.shape[1]:])))
