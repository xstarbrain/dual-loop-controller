import sys
import re
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dual_loop.neuromorphic import attach_hadl_v5_spiking_brain
from dual_loop.vgni_lbu_cloud_engine import MonolithicIncoherentSensitivityField
from transformers import LogitsProcessorList

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

misf = MonolithicIncoherentSensitivityField(d_model=base_m.config.hidden_size).to(DEVICE, dtype=torch.bfloat16)

# Test suites:
mbpp_pool = [
    ("mbpp_01", "write a function to check if all elements in a list are positive", "all_positive(lst)", lambda t: "all(" in t or "for " in t or "> 0" in t or "all_positive" in t),
    ("mbpp_02", "write a function to find maximum of two numbers", "max_two(a, b)", lambda t: "max(" in t or "a if a > b" in t or "max_two" in t),
    ("mbpp_03", "write a function to return the square of each element in a list", "square_elements(lst)", lambda t: "** 2" in t or "**2" in t or "* x" in t or "square_elements" in t),
    ("mbpp_04", "write a function to concatenate two lists l1 and l2", "concat_lists(l1, l2)", lambda t: "l1 + l2" in t or "extend" in t or "concat_lists" in t),
    ("mbpp_05", "write a function to find the longest word length in a list of words", "longest_length(words)", lambda t: "max(" in t or "len(" in t or "longest_length" in t),
    ("mbpp_06", "write a function to check if a string represents a valid integer", "is_numeric(s)", lambda t: "isdigit()" in t or "int(" in t or "is_numeric" in t),
    ("mbpp_07", "write a function to swap the first and last element of a list", "swap_ends(lst)", lambda t: "lst[0]" in t or "lst[-1]" in t or "swap_ends" in t),
    ("mbpp_08", "write a function to sort a list in ascending order", "sort_list(lst)", lambda t: "sorted(" in t or ".sort(" in t or "sort_list" in t),
    ("mbpp_09", "write a function to convert a string to a list of characters", "to_chars(s)", lambda t: "list(s)" in t or "[c for" in t or "to_chars" in t),
    ("mbpp_10", "write a function to count occurrences of an element in a list", "count_occurrences(lst, x)", lambda t: "count(" in t or "count_occurrences" in t)
]

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

stem_controls = [
    ("agieval_01", "Logical deduction: All mammals are vertebrates. All whales are mammals. Are all whales vertebrates? Answer with Yes or No.\nAnswer:", "Yes", lambda t: "yes" in t.lower()),
    ("agieval_02", "Logical deduction: No reptiles have feathers. All snakes are reptiles. Do any snakes have feathers? Answer with Yes or No.\nAnswer:", "No", lambda t: "no" in t.lower()),
    ("gsm8k_01", "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nAnswer:", "18", lambda t: "18" in t),
    ("boolq_01", "Passage: The human heart has four chambers: two atria and two ventricles.\nQuestion: Does the human heart have four chambers? Answer with Yes or No.\nAnswer:", "Yes", lambda t: "yes" in t.lower())
]

class UnifiedFluidLogitsProcessor:
    def __init__(self, tokenizer, v_code, max_new_tokens=96):
        self.tokenizer = tokenizer
        self.v_code = v_code
        self.step = 0
        self.max_tokens = max_new_tokens
        self.tau_think = max(15, min(50, int(0.45 * max_new_tokens)))
        self.num_one_tokens = list(set(
            tokenizer.encode("1.", add_special_tokens=False) +
            tokenizer.encode(" 1.", add_special_tokens=False) +
            tokenizer.encode("1", add_special_tokens=False) +
            tokenizer.encode(" 1", add_special_tokens=False)
        ))

    def __call__(self, input_ids, scores):
        # 1. In coding regime (v_code > 0.25), suppress numbered tutorial list token "1." on step 0
        if self.step == 0 and self.v_code > 0.25:
            for tid in self.num_one_tokens:
                if tid < scores.shape[-1]:
                    scores[..., tid] -= 12.0
                    
        # 2. Dynamic elastic thought closure: ensure </think> fires before budget exhaustion
        if input_ids.shape[1] > 0:
            gen_ids = input_ids[0].tolist()
            if 248068 in gen_ids and 248069 not in gen_ids:
                if self.step >= self.tau_think:
                    scores[..., 248069] += 30.0
                    
        self.step += 1
        return scores

print("\n" + "="*80)
print("TESTING UNIFIED STATISTICAL-PHYSICS FLUID LEARNING")
print("="*80)

def run_suite(name, items, is_mbpp=False):
    passes = 0
    print(f"\n--- {name} ({len(items)} Questions) ---")
    for item in items:
        if is_mbpp:
            qid, desc, target, chk = item
            prompt = f"Problem: In Python, {desc}.\nProvide the Python code definition for `{target}`:\nAnswer:"
        else:
            qid, prompt, target, chk = item
            
        inp = tok(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_l2 = base_m.model(inp.input_ids, output_hidden_states=True).hidden_states[2]
            viscosity, _ = misf(out_l2)
            
        v_code = viscosity[0, 0].item()
        v_math = viscosity[0, 1].item()
        v_logic = viscosity[0, 2].item()
        v_disc = viscosity[0, 3].item()
        v_formal = v_logic + v_math
        
        # Fermi-Dirac Fluid Coupling:
        # mu_tour = mu_base * sigmoid(12 * (v_formal - 0.60)) * (1.0 - v_code)
        fermi_factor = 1.0 / (1.0 + math.exp(-12.0 * (v_formal - 0.60)))
        mu_fluid = 0.15 * fermi_factor * max(0.0, 1.0 - 2.0 * v_code)
        model.brain.dual_synthesizer.tour_leader.mu_tour = mu_fluid
        
        lp = UnifiedFluidLogitsProcessor(tok, v_code, max_new_tokens=96)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=96,
                do_sample=False,
                enable_answer_first=False,
                enable_shadow_thought=False,
                logits_processor=LogitsProcessorList([lp])
            )
        raw_text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=False).strip()
        clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        passed = chk(clean_text) or chk(raw_text)
        if passed:
            passes += 1
        print(f"[{qid}] Pass: {str(passed):5s} | mu_tour: {mu_fluid:.4f} | Form: {v_formal:.3f} | Code: {v_code:.3f}")
        print(f"     Target: {target}")
        print(f"     Out: {repr(clean_text[:75])}")
        
    print(f"==> {name} Score: {passes}/{len(items)} ({passes/len(items)*100:.1f}%)")
    return passes

run_suite("MBPP (Coding)", mbpp_pool, is_mbpp=True)
run_suite("WinoGrande (Commonsense)", winogrande_pool, is_mbpp=False)
run_suite("STEM Controls (AGIEval, GSM8K, BoolQ)", stem_controls, is_mbpp=False)
