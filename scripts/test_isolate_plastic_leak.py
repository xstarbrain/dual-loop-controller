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

# Test with current plastic leak
mbpp_prompt = (
    "Problem 2: In Python, write a function to check if all elements in a list are positive.\n"
    "Provide the Python code definition for `all_positive(lst)`:\n"
    "Answer:"
)

wg_prompt = (
    "Sentence: The glass fell onto the carpet and did not break because it was soft. What was soft?\n"
    "Identify the referent clearly.\n"
    "Answer:"
)

print("\n--- TEST A: ORIGINAL WITH PLASTIC LEAK ---")
inp_mbpp = tok(mbpp_prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out1 = model.generate(**inp_mbpp, max_new_tokens=60, do_sample=False, enable_answer_first=True, prompt_text=mbpp_prompt, tokenizer=tok)
print("MBPP (with leak):", repr(tok.decode(out1[0][inp_mbpp.input_ids.shape[1]:])))

# Now let's temporarily set w_plastic to 0 to test pure fluid gate
print("\n--- TEST B: ZEROING W_PLASTIC (TESTING IF W_PLASTIC IS THE SOLE CAUSE OF LEAK) ---")
with torch.no_grad():
    model.brain.plastic_unit.w_plastic.zero_()
    out2 = model.generate(**inp_mbpp, max_new_tokens=60, do_sample=False, enable_answer_first=True, prompt_text=mbpp_prompt, tokenizer=tok)
print("MBPP (without w_plastic leak):", repr(tok.decode(out2[0][inp_mbpp.input_ids.shape[1]:])))

inp_wg = tok(wg_prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out_wg = model.generate(**inp_wg, max_new_tokens=60, do_sample=False, enable_answer_first=True, prompt_text=wg_prompt, tokenizer=tok)
print("WG (without w_plastic leak):", repr(tok.decode(out_wg[0][inp_wg.input_ids.shape[1]:])))
