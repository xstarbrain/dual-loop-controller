import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dual_loop.neuromorphic import attach_hadl_v5_spiking_brain
from scripts.suite_1000q_20bench import build_1000q_suite

tok = AutoTokenizer.from_pretrained("dist/staging_hf_fused")
base_m = AutoModelForCausalLM.from_pretrained("dist/staging_hf_fused", torch_dtype=torch.bfloat16, device_map="cuda:0")
lm = PeftModel.from_pretrained(base_m, "dist/frontier_lora_adapter")
model = attach_hadl_v5_spiking_brain(lm)
model.tokenizer = tok
model.eval()

suite = [x for x in build_1000q_suite() if x["benchmark"] == "WinoGrande"]
for qid in ["wg_01", "wg_08"]:
    item = next(x for x in suite if x["id"] == qid)
    inp = tok(item["prompt"], return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=150,
            do_sample=False,
            enable_answer_first=True,
            enable_shadow_thought=True,
            prompt_text=item["prompt"],
            tokenizer=tok
        )
    raw_text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=False).strip()
    clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    print(f"=== {qid} Target: {item['target']} ===")
    print("CLEAN TEXT FULL:")
    print(clean_text)
    print("RAW TEXT FULL:")
    print(raw_text)
    print("PASS:", item["check"](clean_text) or item["check"](raw_text))
    print()
