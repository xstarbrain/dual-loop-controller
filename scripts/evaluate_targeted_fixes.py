#!/usr/bin/env python3
"""
Targeted Verification Script for Architectural Enhancements (PTSC, NSATB, Cognitive ALU, Diagnostics)
Evaluates previously failing tasks across HumanEval, MATH, CodeDebug, BBH, and WinoGrande.
"""

import sys
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.neuromorphic import attach_hadl_v5_spiking_brain
from scripts.suite_1000q_20bench import build_1000q_suite

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
model.eval()

# Load 1000q suite and pull target items
suite = build_1000q_suite()
suite_by_id = {item["id"]: item for item in suite}

target_ids = [
    # HumanEval (failing items + controls)
    "he_01", "he_02", "he_03", "he_08", "he_04", "he_05", "he_06", "he_07", "he_09", "he_10",
    # MATH (failing items + controls)
    "math_09", "math_11", "math_12", "math_13", "math_01", "math_02", "math_03", "math_04", "math_05", "math_10",
    # CodeDebug (failing items + controls)
    "codedebug_02", "codedebug_05", "codedebug_06", "codedebug_08", "codedebug_12", "codedebug_14", "codedebug_16", "codedebug_01", "codedebug_03", "codedebug_04",
    # BBH Orientation
    "bbh_06", "bbh_07",
    # WinoGrande
    "wg_01", "wg_02", "wg_03", "wg_04", "wg_05", "wg_06", "wg_07", "wg_08", "wg_09", "wg_10"
]

print("\n" + "=" * 90)
print("RUNNING TARGETED VERIFICATION AUDIT")
print("=" * 90)

from collections import defaultdict
bench_results = defaultdict(lambda: {"correct": 0, "total": 0})

for qid in target_ids:
    if qid not in suite_by_id:
        continue
    item = suite_by_id[qid]
    b_name = item["benchmark"]
    prompt = item["prompt"]
    chk = item["check"]
    max_toks = item.get("max_tokens", 96)

    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_toks,
            do_sample=False,
            enable_answer_first=True,
            enable_shadow_thought=True
        )
    gen_text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    clean_text = re.sub(r"<think>.*?</think>", "", gen_text, flags=re.DOTALL).strip()
    passed = bool(chk(clean_text) or chk(gen_text))

    bench_results[b_name]["total"] += 1
    if passed:
        bench_results[b_name]["correct"] += 1

    status_str = "PASS" if passed else "FAIL"
    print(f"[{qid}] {b_name:10s} | {status_str:4s} | Target: {str(item.get('target', ''))[:20]:20s} | Out: {repr(clean_text[:60])}")

print("\n" + "=" * 90)
print("TARGETED VERIFICATION SUMMARY:")
print("=" * 90)
for b_name, res in bench_results.items():
    pct = (res["correct"] / res["total"]) * 100 if res["total"] > 0 else 0
    print(f"{b_name:12s}: {res['correct']:2d} / {res['total']:2d} ({pct:5.1f}%)")
