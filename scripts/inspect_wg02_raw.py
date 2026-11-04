import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dual_loop.neuromorphic import attach_hadl_v5_spiking_brain
from scripts.suite_1000q_20bench import build_1000q_suite

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("dist/staging_hf_fused")
base_m = AutoModelForCausalLM.from_pretrained("dist/staging_hf_fused", torch_dtype=torch.bfloat16, device_map=DEVICE)
lm = PeftModel.from_pretrained(base_m, "dist/frontier_lora_adapter")
model = attach_hadl_v5_spiking_brain(lm)
model.tokenizer = tok
model.eval()

suite = build_1000q_suite()
wg_02 = next(x for x in suite if x["id"] == "wg_02")
inp = tok(wg_02["prompt"], return_tensors="pt").to(DEVICE)

with torch.no_grad():
    out = model.generate(
        **inp,
        max_new_tokens=180,
        do_sample=False,
        enable_answer_first=True,
        enable_shadow_thought=True,
        prompt_text=wg_02["prompt"],
        tokenizer=tok
    )

print("VISIBLE OUT:", repr(tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=False)))
print("SHADOW TOKENS:", model.last_telemetry.get("shadow_thought_tokens"))
print("SHADOW TRACE:")
print(repr(model.last_telemetry.get("shadow_thought_trace")))
