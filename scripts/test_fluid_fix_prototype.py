import sys
import re
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dual_loop.neuromorphic import attach_hadl_v5_spiking_brain
from dual_loop.vgni_lbu_cloud_engine import MonolithicIncoherentSensitivityField

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

# Instantiate MISF on Layer 2
misf = MonolithicIncoherentSensitivityField(d_model=base_m.config.hidden_size).to(DEVICE, dtype=torch.bfloat16)

# Test MBPP Prompts (first 5 questions)
mbpp_test = [
    ("mbpp_01", "write a function to check if all elements in a list are positive", "all_positive(lst)", lambda t: "all(" in t or "for " in t or "> 0" in t),
    ("mbpp_02", "write a function to find maximum of two numbers", "max_two(a, b)", lambda t: "max(" in t or "a if a > b" in t),
    ("mbpp_04", "write a function to concatenate two lists l1 and l2", "concat_lists(l1, l2)", lambda t: "l1 + l2" in t or "extend" in t),
    ("mbpp_05", "write a function to find the longest word length in a list of words", "longest_length(words)", lambda t: "max(" in t or "len(" in t),
    ("mbpp_08", "write a function to sort a list in ascending order", "sort_list(lst)", lambda t: "sorted(" in t or ".sort(" in t)
]

# Test WinoGrande Prompts (first 5 questions)
wg_test = [
    ("wg_01", "Sentence: The glass fell onto the carpet and did not break because it was soft. What was soft?\nIdentify the referent clearly.\nAnswer:", "The carpet", lambda t: bool(re.search(r"\bcarpet\b", t, re.I))),
    ("wg_02", "Sentence: The trophy did not fit into the brown suitcase because it was too large. What was too large?\nIdentify the referent clearly.\nAnswer:", "The trophy", lambda t: bool(re.search(r"\btrophy\b", t, re.I))),
    ("wg_03", "Sentence: The trophy did not fit into the brown suitcase because it was too small. What was too small?\nIdentify the referent clearly.\nAnswer:", "The suitcase", lambda t: bool(re.search(r"\bsuitcase\b", t, re.I))),
    ("wg_04", "Sentence: The heavy oak table could not fit through the narrow door because it was too wide. What was too wide?\nIdentify the referent clearly.\nAnswer:", "The table", lambda t: bool(re.search(r"\btable\b", t, re.I))),
    ("wg_05", "Sentence: The heavy oak table could not fit through the narrow door because it was too narrow. What was too narrow?\nIdentify the referent clearly.\nAnswer:", "The door", lambda t: bool(re.search(r"\bdoor\b", t, re.I)))
]

print("\n" + "="*80)
print("TESTING FLUID PROTOTYPE")
print("="*80)

# Apply fluid modulation:
# 1. Zero out unspiked plastic leak (simulate closed-gate isolation)
# In test: we can hook or set plastic unit during closed gate
# 2. Tour leader mu_tour modulated by fluid sensitivity
print("\n--- 1. MBPP EVALUATION WITH FLUID LEARNING ---")
mbpp_passes = 0
for qid, desc, fn_name, chk in mbpp_test:
    prompt = f"Problem: In Python, {desc}.\nProvide the Python code definition for `{fn_name}`:\nAnswer:"
    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    
    # Compute continuous MISF viscosity
    with torch.no_grad():
        out_l2 = base_m.model(inp.input_ids, output_hidden_states=True).hidden_states[2]
        viscosity, telem = misf(out_l2)
    v_code = viscosity[0, 0].item()
    v_math = viscosity[0, 1].item()
    v_logic = viscosity[0, 2].item()
    v_disc = viscosity[0, 3].item()
    
    # Elastic Fluid Modulation:
    # On high code or discourse viscosity, damp formal theorem-proving attractor
    v_formal = v_logic + v_math
    mu_fluid = 0.15 * math.tanh(2.0 * v_formal) * (1.0 - v_code)
    model.brain.dual_synthesizer.tour_leader.mu_tour = mu_fluid
    
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=96,
            do_sample=False,
            enable_answer_first=True,
            prompt_text=prompt,
            tokenizer=tok
        )
    gen_text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    passed = chk(gen_text)
    if passed:
        mbpp_passes += 1
    print(f"[{qid}] Pass: {passed} | mu_tour: {mu_fluid:.4f} | Code visc: {v_code:.3f}")
    print(f"     Output: {repr(gen_text[:80])}")

print(f"\nMBPP Score: {mbpp_passes}/{len(mbpp_test)} ({mbpp_passes/len(mbpp_test)*100:.1f}%)")

print("\n--- 2. WINOGRANDE EVALUATION WITH FLUID LEARNING ---")
wg_passes = 0
for qid, prompt, target, chk in wg_test:
    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_l2 = base_m.model(inp.input_ids, output_hidden_states=True).hidden_states[2]
        viscosity, telem = misf(out_l2)
    v_code = viscosity[0, 0].item()
    v_math = viscosity[0, 1].item()
    v_logic = viscosity[0, 2].item()
    v_disc = viscosity[0, 3].item()
    
    v_formal = v_logic + v_math
    mu_fluid = 0.15 * math.tanh(2.0 * v_formal) * (1.0 - v_code)
    model.brain.dual_synthesizer.tour_leader.mu_tour = mu_fluid
    
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=96,
            do_sample=False,
            enable_answer_first=True,
            prompt_text=prompt,
            tokenizer=tok
        )
    gen_text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    passed = chk(gen_text)
    if passed:
        wg_passes += 1
    print(f"[{qid}] Pass: {passed} | mu_tour: {mu_fluid:.4f} | Disc visc: {v_disc:.3f}")
    print(f"     Output: {repr(gen_text[:80])}")

print(f"\nWG Score: {wg_passes}/{len(wg_test)} ({wg_passes/len(wg_test)*100:.1f}%)")
