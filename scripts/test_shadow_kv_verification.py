import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dual_loop.neuromorphic.spiking_gate import attach_hadl_v5_spiking_brain

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained("dist/staging_hf_fused", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "dist/staging_hf_fused",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)
print("Attaching frontier LoRA adapter...")
base_model = PeftModel.from_pretrained(base_model, "dist/frontier_lora_adapter")
print("Attaching HADL v5 Spiking Brain...")
brain_wrapper = attach_hadl_v5_spiking_brain(
    base_model=base_model,
    target_layer_idx=7,
    pacemaker_layer_idx=2,
    deliberation_steps_k=1,
    auto_load_checkpoints=True
)

THINK_ID = 248068
END_THINK_ID = 248069

def shadow_kv_generate(wrapper, prompt, max_new_tokens=96, shadow_headroom=160):
    wrapper.reset_state()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    
    # Internal generation with shadow headroom
    internal_max = max_new_tokens + shadow_headroom
    with torch.no_grad():
        out = wrapper.base_model.generate(
            **inputs,
            max_new_tokens=internal_max,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    gen_ids = out[0][input_len:].tolist()
    
    # Shadow KV Slicing Logic
    if THINK_ID in gen_ids:
        think_idx = gen_ids.index(THINK_ID)
        if think_idx <= 2:
            # Model thought first, then answers after </think>
            if END_THINK_ID in gen_ids:
                end_idx = gen_ids.index(END_THINK_ID)
                shadow_ids = gen_ids[:end_idx + 1]
                visible_ids = gen_ids[end_idx + 1:]
            else:
                # Unclosed think within budget: take whatever was generated
                shadow_ids = gen_ids[:shadow_headroom]
                visible_ids = gen_ids[shadow_headroom:]
        else:
            # Model answered first, then started <think>
            visible_ids = gen_ids[:think_idx]
            shadow_ids = gen_ids[think_idx:]
    else:
        shadow_ids = []
        visible_ids = gen_ids
    
    # Strip leading newline tokens from visible_ids
    while visible_ids and visible_ids[0] in [198, 271, 1073]:
        visible_ids = visible_ids[1:]
        
    visible_ids = visible_ids[:max_new_tokens]
    
    # Construct clean returned tensor
    clean_out_ids = torch.cat([inputs["input_ids"], torch.tensor([visible_ids], device="cuda", dtype=torch.long)], dim=-1)
    
    gen_toks = clean_out_ids.shape[1] - input_len
    raw_text = tokenizer.decode(clean_out_ids[0][input_len:], skip_special_tokens=False).strip()
    clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    
    shadow_trace = tokenizer.decode(shadow_ids, skip_special_tokens=False) if shadow_ids else "(none)"
    
    return {
        "clean_text": clean_text,
        "gen_tokens": gen_toks,
        "shadow_tokens": len(shadow_ids),
        "shadow_trace_snippet": shadow_trace[:120].replace("\n", " ")
    }

test_cases = [
    ("truthful_02", "Fact-check query 2: Do humans only use 10 percent of their brain capacity?\nAnswer:", lambda t: bool(re.search(r"\bno\b", t, re.I))),
    ("truthful_03", "Fact-check query 3: Does dropping a penny from the Empire State Building kill someone?\nAnswer:", lambda t: bool(re.search(r"\bno\b", t, re.I))),
    ("wg_05", "Sentence: The heavy oak table could not fit through the narrow door because it was too narrow.\nIdentify the referent clearly.\nAnswer:", lambda t: bool(re.search(r"\bdoor\b", t, re.I))),
    ("he_02", "Complete the Python function to return the product of a and b.\n```python\ndef multiply(a, b):\n", lambda t: "return a * b" in t or "a * b" in t),
    ("arcc_05", "Science Challenge 5: Why does ice float on water?\nA) Denser\nB) Less dense than liquid water\nC) Heavier\nD) Warmer\nAnswer with the letter:", lambda t: bool(re.search(r"\bB\b", t))),
]

print("\n" + "=" * 90)
print("TESTING SHADOW KV GENERATION PROTOTYPE")
print("=" * 90)

for q_id, prompt, check_fn in test_cases:
    res = shadow_kv_generate(brain_wrapper, prompt, max_new_tokens=64, shadow_headroom=160)
    passed = check_fn(res["clean_text"])
    status = "[PASS ✓]" if passed else "[FAIL ✗]"
    print(f"\n{status} [{q_id}]")
    print(f"  Visible Gen Tokens: {res['gen_tokens']} (Shadow Tokens: {res['shadow_tokens']})")
    print(f"  Visible Clean Text: {repr(res['clean_text'][:90])}")
    print(f"  Shadow Thought    : {repr(res['shadow_trace_snippet'][:90])}")
