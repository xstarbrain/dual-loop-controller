import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dual_loop.neuromorphic import attach_hadl_v5_spiking_brain
from scripts.suite_1000q_20bench import build_1000q_suite

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[*] Running Comprehensive Fluid Learn Evaluation on {DEVICE}...")

tok = AutoTokenizer.from_pretrained("dist/staging_hf_fused")
base_m = AutoModelForCausalLM.from_pretrained("dist/staging_hf_fused", torch_dtype=torch.bfloat16, device_map=DEVICE)
lm = PeftModel.from_pretrained(base_m, "dist/frontier_lora_adapter")
model = attach_hadl_v5_spiking_brain(lm)
model.tokenizer = tok
model.eval()

suite = build_1000q_suite()
mbpp_items = [x for x in suite if x["benchmark"] == "MBPP"][:20]
wg_items = [x for x in suite if x["benchmark"] == "WinoGrande"][:20]
ctrl_items = [x for x in suite if x["benchmark"] in ["GSM8K", "AGIEval", "BoolQ", "MMLU"]][:10]

eval_batches = [
    ("MBPP (Coding)", mbpp_items),
    ("WinoGrande (Commonsense)", wg_items),
    ("STEM & Logic Controls", ctrl_items)
]

summary_results = {}

for name, items in eval_batches:
    print("\n" + "="*85)
    print(f"EVALUATING SUITE: {name} ({len(items)} questions)")
    print("="*85)
    passes = 0
    t0 = time.perf_counter()
    for item in items:
        inp = tok(item["prompt"], return_tensors="pt").to(DEVICE)
        max_tok = item.get("max_tokens", 180)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=max_tok,
                do_sample=False,
                enable_answer_first=True,
                enable_shadow_thought=True,
                prompt_text=item["prompt"],
                tokenizer=tok
            )
        raw_text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=False).strip()
        clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        passed = item["check"](clean_text) or item["check"](raw_text)
        if passed:
            passes += 1
        print(f"[{item['id']}] Pass: {str(passed):<5} | Target: {str(item['target'])[:20]:<20} | Out: {repr(clean_text[:60])}")

    elapsed = time.perf_counter() - t0
    acc = passes / len(items) * 100.0
    summary_results[name] = {"passes": passes, "total": len(items), "accuracy": acc, "time": elapsed}
    print(f"\n--> {name} Score: {passes}/{len(items)} ({acc:.1f}%) in {elapsed:.1f}s")

print("\n" + "="*85)
print("FLUID LEARN CONVERGENCE SUMMARY")
print("="*85)
for k, v in summary_results.items():
    print(f"{k:<30}: {v['passes']}/{v['total']} ({v['accuracy']:.1f}%) [elapsed: {v['time']:.1f}s]")
print("="*85)
