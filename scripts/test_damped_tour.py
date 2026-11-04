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

# Test MBPP
mbpp_prompt = (
    "Problem 2: In Python, write a function to check if all elements in a list are positive.\n"
    "Provide the Python code definition for `all_positive(lst)`:\n"
    "Answer:"
)

# Test WinoGrande
wg_prompt = (
    "Sentence: The glass fell onto the carpet and did not break because it was soft. What was soft?\n"
    "Identify the referent clearly.\n"
    "Answer:"
)

print("\n--- TEST: MODULATING TOUR LEADER & PLASTIC LEAK ---")
# 1. Gate plastic by viscosity (currently leak)
# In test: let's set tour_leader mu_tour = 0.0 on coding, or check what happens when mu_tour is dynamically damped
model.brain.dual_synthesizer.tour_leader.mu_tour = 0.02 # soft damp instead of aggressive 0.15

inp_mbpp = tok(mbpp_prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out_mbpp = model.generate(**inp_mbpp, max_new_tokens=80, do_sample=False, enable_answer_first=True, prompt_text=mbpp_prompt, tokenizer=tok)
print("MBPP (damped tour):", repr(tok.decode(out_mbpp[0][inp_mbpp.input_ids.shape[1]:])))

inp_wg = tok(wg_prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out_wg = model.generate(**inp_wg, max_new_tokens=40, do_sample=False, enable_answer_first=True, prompt_text=wg_prompt, tokenizer=tok)
print("WG (damped tour):", repr(tok.decode(out_wg[0][inp_wg.input_ids.shape[1]:])))
