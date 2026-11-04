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

print("\n--- TEST: MBPP WITH WRAPPER ---")
inp = tok(mbpp_prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out = model.generate(
        **inp,
        max_new_tokens=100,
        do_sample=False,
        enable_answer_first=True,
        prompt_text=mbpp_prompt,
        tokenizer=tok
    )

gen_tokens = out[0][inp.input_ids.shape[1]:]
print("Output Text:\n", repr(tok.decode(gen_tokens)))
print("First 10 token IDs:\n", gen_tokens[:10].tolist())
for tid in gen_tokens[:10].tolist():
    print(f"  {tid}: {repr(tok.decode([tid]))}")
