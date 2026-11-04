#!/usr/bin/env python3
"""
Fast Targeted Verification Script with immediate flush
Tests 16 crucial previously failing tasks on GPU RTX 5060.
"""

import sys
import re
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.neuromorphic import attach_hadl_v5_spiking_brain
from scripts.suite_1000q_20bench import build_1000q_suite

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {DEVICE}...", flush=True)

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

suite = build_1000q_suite()
suite_by_id = {item["id"]: item for item in suite}

target_ids = [
    # 4 HumanEval
    "he_01", "he_02", "he_03", "he_08",
    # 4 MATH
    "math_09", "math_11", "math_12", "math_13",
    # 4 CodeDebug
    "codedebug_02", "codedebug_05", "codedebug_06", "codedebug_08",
    # 2 BBH
    "bbh_06", "bbh_07",
    # 2 WinoGrande
    "wg_05", "wg_07"
]

print("\n" + "=" * 95, flush=True)
print(f"{'ID':8s} | {'BENCHMARK':10s} | {'STATUS':6s} | {'LATENCY':7s} | {'OUTPUT':50s}", flush=True)
print("=" * 95, flush=True)

passed_count = 0

for qid in target_ids:
    if qid not in suite_by_id:
        continue
    item = suite_by_id[qid]
    b_name = item["benchmark"]
    prompt = item["prompt"]
    chk = item["check"]
    max_toks = item.get("max_tokens", 96)

    inp = tok(prompt, return_tensors="pt").to(DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_toks,
            do_sample=False,
            enable_answer_first=True,
            enable_shadow_thought=True
        )
    torch.cuda.synchronize()
    lat = time.perf_counter() - t0

    gen_text = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    clean_text = re.sub(r"<think>.*?</think>", "", gen_text, flags=re.DOTALL).strip()
    passed = bool(chk(clean_text) or chk(gen_text))

    if passed:
        passed_count += 1

    status_str = "PASS" if passed else "FAIL"
    out_snippet = repr(clean_text[:45])
    print(f"{qid:8s} | {b_name:10s} | {status_str:6s} | {lat:5.2f}s | {out_snippet}", flush=True)

print("=" * 95, flush=True)
print(f"TOTAL PASSED: {passed_count} / {len(target_ids)} ({passed_count / len(target_ids) * 100:.1f}%)", flush=True)
print("=" * 95, flush=True)
