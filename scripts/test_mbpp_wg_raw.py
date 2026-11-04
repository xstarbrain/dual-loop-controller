import sys
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {DEVICE}...")
tok = AutoTokenizer.from_pretrained("dist/staging_hf_fused")
base_m = AutoModelForCausalLM.from_pretrained(
    "dist/staging_hf_fused",
    torch_dtype=torch.bfloat16,
    device_map=DEVICE
)
m = PeftModel.from_pretrained(base_m, "dist/frontier_lora_adapter")

# Let's test MBPP prompt with Code Scaffolding vs without
mbpp_prompt = (
    "Problem 2: In Python, write a function to check if all elements in a list are positive.\n"
    "Provide the Python code definition for `all_positive(lst)`:\n"
    "Answer:"
)

# Test WinoGrande prompt
wg_prompt = (
    "Sentence: The glass fell onto the carpet and did not break because it was soft. What was soft?\n"
    "Identify the referent clearly.\n"
    "Answer:"
)

print("\n--- TEST 1: MBPP Prompt ---")
inp = tok(mbpp_prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out = m.generate(**inp, max_new_tokens=100, do_sample=False)
print("MBPP Raw Out:\n", repr(tok.decode(out[0][inp.input_ids.shape[1]:])))

print("\n--- TEST 2: WinoGrande Prompt ---")
inp_wg = tok(wg_prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out_wg = m.generate(**inp_wg, max_new_tokens=100, do_sample=False)
print("WG Raw Out:\n", repr(tok.decode(out_wg[0][inp_wg.input_ids.shape[1]:])))
