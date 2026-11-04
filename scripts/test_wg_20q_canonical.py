import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dual_loop.neuromorphic import attach_hadl_v5_spiking_brain
from scripts.suite_1000q_20bench import build_1000q_suite

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[*] Testing all 20 Canonical WinoGrande questions on {DEVICE}...")

tok = AutoTokenizer.from_pretrained("dist/staging_hf_fused")
base_m = AutoModelForCausalLM.from_pretrained("dist/staging_hf_fused", torch_dtype=torch.bfloat16, device_map=DEVICE)
lm = PeftModel.from_pretrained(base_m, "dist/frontier_lora_adapter")
model = attach_hadl_v5_spiking_brain(lm)
model.tokenizer = tok
model.eval()

suite = [x for x in build_1000q_suite() if x["benchmark"] == "WinoGrande"][:20]

passes = 0
for x in suite:
    inp = tok(x["prompt"], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=x.get("max_tokens", 200),
            do_sample=False,
            enable_answer_first=True,
            enable_shadow_thought=True,
            prompt_text=x["prompt"],
            tokenizer=tok
        )
    raw_text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=False).strip()
    clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    passed = x["check"](clean_text) or x["check"](raw_text)
    if passed:
        passes += 1
    print(f"[{x['id']}] Pass: {str(passed):<5} | Target: {x['target']:<15}")
    print(f"     Clean: {repr(clean_text[:80])}")

print(f"\n--> WinoGrande 20Q Result: {passes}/20 ({passes*5.0:.1f}%)")
