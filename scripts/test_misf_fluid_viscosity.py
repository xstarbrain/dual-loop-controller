import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dual_loop.vgni_lbu_cloud_engine import MonolithicIncoherentSensitivityField

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading tokenizer and base model on {DEVICE}...")
tok = AutoTokenizer.from_pretrained("dist/staging_hf_fused")
model = AutoModelForCausalLM.from_pretrained(
    "dist/staging_hf_fused",
    torch_dtype=torch.bfloat16,
    device_map=DEVICE
)

misf = MonolithicIncoherentSensitivityField(d_model=model.config.hidden_size).to(DEVICE, dtype=torch.bfloat16)

test_prompts = [
    # MBPP Code prompt
    ("MBPP_Code", "Problem 2: In Python, write a function to check if all elements in a list are positive.\nProvide the Python code definition for `all_positive(lst)`:\nAnswer:"),
    # HumanEval prompt
    ("HumanEval", "Complete the Python function to find minimum.\n```python\ndef min_three(a, b, c):\n    # Your implementation here\n```\nPython code:"),
    # WinoGrande Physical Referent prompt
    ("WinoGrande", "Sentence: The glass fell onto the carpet and did not break because it was soft. What was soft?\nIdentify the referent clearly.\nAnswer:"),
    # Math prompt
    ("GSM8K_Math", "Ava wants to buy 12 staplers at $10 each with $4 discount. How much does Ava pay in total?\nAnswer:"),
    # DailyChat prompt
    ("DailyChat", "Halo! Bagaimana kabarmu hari ini? Ceritakan pengalaman menarikmu akhir pekan kemarin.")
]

print("\n--- TESTING MISF CONTINUOUS VISCOSITY PROJECTION ---")
for label, p in test_prompts:
    inp = tok(p, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        # Get hidden state from Layer 2
        outputs = model.model(inp.input_ids, output_hidden_states=True)
        h_l2 = outputs.hidden_states[2] # Layer 2
        viscosity, telem = misf(h_l2)
    
    v = viscosity[0].tolist()
    print(f"\n[{label:12s}] Prompt: '{p[:60].replace(chr(10), ' ')}...'")
    print(f"  --> Code: {v[0]:.3f} | Math: {v[1]:.3f} | Logic: {v[2]:.3f} | Discourse: {v[3]:.3f}")
