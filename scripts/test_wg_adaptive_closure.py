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

# Test WinoGrande Prompts (10 questions from benchmark pool)
winogrande_pool = [
    ("wg_01", "The glass fell onto the carpet and did not break because it was soft. What was soft?", "The carpet", lambda t: bool(re.search(r"\bcarpet\b", t, re.I))),
    ("wg_02", "The trophy did not fit into the brown suitcase because it was too large. What was too large?", "The trophy", lambda t: bool(re.search(r"\btrophy\b", t, re.I))),
    ("wg_03", "The trophy did not fit into the brown suitcase because it was too small. What was too small?", "The suitcase", lambda t: bool(re.search(r"\bsuitcase\b", t, re.I))),
    ("wg_04", "The heavy oak table could not fit through the narrow door because it was too wide. What was too wide?", "The table", lambda t: bool(re.search(r"\btable\b", t, re.I))),
    ("wg_05", "The heavy oak table could not fit through the narrow door because it was too narrow. What was too narrow?", "The door", lambda t: bool(re.search(r"\bdoor\b", t, re.I))),
    ("wg_06", "The bowling ball fell onto the foam mat and left a dent because it was heavy. What was heavy?", "The bowling ball", lambda t: bool(re.search(r"\bball\b", t, re.I))),
    ("wg_07", "The metal key could not open the rusty lock because it was stiff. What was stiff?", "The lock", lambda t: bool(re.search(r"\block\b", t, re.I))),
    ("wg_08", "The porcelain cup fell onto the hardwood floor and shattered because it was hard. What was hard?", "The floor", lambda t: bool(re.search(r"\bfloor\b", t, re.I))),
    ("wg_09", "The water filled the ceramic mug quickly because it was narrow. What was narrow?", "The mug", lambda t: bool(re.search(r"\bmug\b", t, re.I))),
    ("wg_10", "The ice cube melted in the hot soup because it was boiling. What was boiling?", "The soup", lambda t: bool(re.search(r"\bsoup\b", t, re.I)))
]

print("\n" + "="*80)
print("TESTING WINOGRANDE WITH FLUID ADAPTIVE THOUGHT BUDGET")
print("="*80)

# Custom dynamic logits processor for adaptive thought closing
class DynamicThoughtLogitsProcessor:
    def __init__(self, tokenizer, max_new_tokens=96):
        self.tokenizer = tokenizer
        self.step = 0
        self.max_tokens = max_new_tokens
        # Adaptive closure at 35 steps
        self.tau_think = 35

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] > 0:
            gen_ids = input_ids[0].tolist()
            if 248068 in gen_ids and 248069 not in gen_ids:
                if self.step >= self.tau_think:
                    scores[..., 248069] += 35.0
        self.step += 1
        return scores

from transformers import LogitsProcessorList

wg_passes = 0
for qid, sent, target, chk in winogrande_pool:
    prompt = f"Sentence: {sent}\nIdentify the referent clearly.\nAnswer:"
    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        out_l2 = base_m.model(inp.input_ids, output_hidden_states=True).hidden_states[2]
        viscosity, _ = misf(out_l2)
    v_code = viscosity[0, 0].item()
    v_math = viscosity[0, 1].item()
    v_logic = viscosity[0, 2].item()
    v_disc = viscosity[0, 3].item()
    
    v_formal = v_logic + v_math
    mu_fluid = 0.15 * math.tanh(2.0 * v_formal) * max(0.0, 1.0 - 2.0 * v_code)
    model.brain.dual_synthesizer.tour_leader.mu_tour = mu_fluid
    
    lp = DynamicThoughtLogitsProcessor(tok, max_new_tokens=96)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=96,
            do_sample=False,
            enable_answer_first=False,
            enable_shadow_thought=False, # see raw generation with adaptive closure
            logits_processor=LogitsProcessorList([lp])
        )
    gen_text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=False).strip()
    clean_text = re.sub(r"<think>.*?</think>", "", gen_text, flags=re.DOTALL).strip()
    passed = chk(clean_text) or chk(gen_text)
    if passed:
        wg_passes += 1
    print(f"[{qid}] Pass: {str(passed):5s} | Target: {target:18s}")
    print(f"     Clean out: {repr(clean_text[:70])}")
    if "<think>" in gen_text and "</think>" in gen_text:
        thought_body = gen_text[gen_text.index("<think>")+7:gen_text.index("</think>")].strip()
        print(f"     Thought ({len(thought_body.split())} words): {repr(thought_body[:70])}...")

print(f"\nWG Score with Adaptive Closure: {wg_passes}/{len(winogrande_pool)} ({wg_passes/len(winogrande_pool)*100:.1f}%)")
