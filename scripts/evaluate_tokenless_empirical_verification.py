#!/usr/bin/env python3
"""
Empirical Benchmark & Verification: Tokenless Latent Reasoning + IPv6 Numeric Register
=====================================================================================
Head-to-head empirical testing on the 60 most critical questions from the 200Q suite
(DROP, AGIEval, BoolQ, TruthfulQA, BBH, MATH) where token ceiling truncation and
arithmetic drift previously occurred.

Measures:
1. Accuracy: Direct comparison between Dual-Stream Baseline vs Tokenless Latent + IPv6 ALU.
2. Latency & Tokens: Impact on generated token count and time per question.
3. Ceiling Truncation Elimination: Percentage of answers successfully output without <think> truncation.
4. Arithmetic Precision: Verification of exact discrete integer evaluation via IPv6 Register.
"""

import sys
import os
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.neuromorphic import (
    attach_hadl_v5_spiking_brain,
    TokenlessLatentThoughtEngine,
    IPv6NumericRegister,
    AnswerFirstLogitsProcessor,
)
from scripts.suite_200q_20bench import build_200q_suite

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_MODEL_DIR = PROJECT_ROOT / "dist" / "staging_hf_fused"
ADAPTER_DIR = PROJECT_ROOT / "dist" / "frontier_lora_adapter"
REPORT_JSON = PROJECT_ROOT / "eval_results" / "tokenless_latent_empirical_audit.json"
REPORT_TXT = PROJECT_ROOT / "eval_results" / "tokenless_latent_empirical_log.txt"


def run_benchmark():
    print("=" * 105)
    print("EMPIRICAL VALIDATION: TOKENLESS LATENT REASONING + IPv6 NUMERIC REGISTER")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 105)

    all_suite = build_200q_suite()
    # Filter to the 6 most impacted benchmarks (60 questions total)
    target_benchmarks = ["DROP", "AGIEval", "BoolQ", "TruthfulQA", "BBH", "MATH"]
    target_suite = [item for item in all_suite if item["benchmark"] in target_benchmarks]
    print(f"[*] Filtered {len(target_suite)} questions across {len(target_benchmarks)} benchmarks: {target_benchmarks}")

    print("\n[*] Loading Tokenizer and Base Model in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR))
    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        torch_dtype=torch.bfloat16
    ).to(DEVICE)

    if ADAPTER_DIR.exists():
        print(f"[*] Injecting Frontier LoRA adapter from {ADAPTER_DIR.name}...")
        lora_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR), is_trainable=False)
        lora_model.eval()
    else:
        lora_model = base_model

    print("[*] Attaching HADL v5.5 Neuromorphic Spiking Brain Wrapper...")
    model = attach_hadl_v5_spiking_brain(
        base_model=lora_model,
        target_layer_idx=7,
        pacemaker_layer_idx=2,
        reentrant_layers=[11, 15, 19],
        spike_v_th=0.85,
        beta_decay=0.90,
        target_firing_rate=0.20,
        deliberation_steps_k=1,
        auto_load_checkpoints=True,
        enable_directional_safety=True
    )
    model.tokenizer = tokenizer
    model.eval()

    results_baseline = []
    results_tokenless = []

    print("\n" + "=" * 105)
    print("PHASE 1: RUNNING BASELINE (Dual-Stream without Tokenless Logits Clamping)")
    print("=" * 105)

    for idx, item in enumerate(target_suite, 1):
        prompt = item["prompt"]
        max_tok = item.get("max_tokens", 96)
        check_fn = item["check"]
        target = item["target"]
        bench = item["benchmark"]
        q_id = item["id"]

        inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_len = inp.input_ids.shape[1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                **inp,
                max_new_tokens=max_tok,
                do_sample=False,
                enable_answer_first=False,  # Baseline
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        gen_tokens = out_ids.shape[1] - input_len
        raw_text = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=False).strip()
        clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        is_correct = check_fn(clean_text)

        results_baseline.append({
            "id": q_id,
            "benchmark": bench,
            "target": target,
            "raw_text": raw_text[:120] + ("..." if len(raw_text) > 120 else ""),
            "clean_text": clean_text[:80],
            "correct": is_correct,
            "tokens": gen_tokens,
            "time_sec": round(dt, 4),
            "had_think_truncation": ("<think>" in raw_text and "</think>" not in raw_text)
        })

        if idx % 10 == 0 or idx == len(target_suite):
            print(f"  [Baseline] Completed {idx}/{len(target_suite)} items...")

    print("\n" + "=" * 105)
    print("PHASE 2: RUNNING TOKENLESS LATENT REASONING + IPv6 NUMERIC REGISTER ACTIVE")
    print("=" * 105)

    for idx, item in enumerate(target_suite, 1):
        prompt = item["prompt"]
        max_tok = item.get("max_tokens", 96)
        check_fn = item["check"]
        target = item["target"]
        bench = item["benchmark"]
        q_id = item["id"]

        inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_len = inp.input_ids.shape[1]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                **inp,
                max_new_tokens=max_tok,
                do_sample=False,
                enable_answer_first=True,  # Active Tokenless Latent + IPv6 ALU
                prompt_text=prompt,
                tokenizer=tokenizer,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        gen_tokens = out_ids.shape[1] - input_len
        raw_text = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=False).strip()
        clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        is_correct = check_fn(clean_text)

        results_tokenless.append({
            "id": q_id,
            "benchmark": bench,
            "target": target,
            "raw_text": raw_text[:120] + ("..." if len(raw_text) > 120 else ""),
            "clean_text": clean_text[:80],
            "correct": is_correct,
            "tokens": gen_tokens,
            "time_sec": round(dt, 4),
            "had_think_truncation": ("<think>" in raw_text and "</think>" not in raw_text)
        })

        if idx % 10 == 0 or idx == len(target_suite):
            print(f"  [Tokenless Active] Completed {idx}/{len(target_suite)} items...")

    # Summary Statistics
    def compute_stats(res_list):
        bench_stats = {}
        for r in res_list:
            b = r["benchmark"]
            if b not in bench_stats:
                bench_stats[b] = {"correct": 0, "total": 0, "tokens": 0, "time": 0.0, "truncations": 0}
            bench_stats[b]["total"] += 1
            if r["correct"]:
                bench_stats[b]["correct"] += 1
            bench_stats[b]["tokens"] += r["tokens"]
            bench_stats[b]["time"] += r["time_sec"]
            if r["had_think_truncation"]:
                bench_stats[b]["truncations"] += 1
        return bench_stats

    stats_base = compute_stats(results_baseline)
    stats_tokenless = compute_stats(results_tokenless)

    total_correct_base = sum(s["correct"] for s in stats_base.values())
    total_tokens_base = sum(s["tokens"] for s in stats_base.values())
    total_time_base = sum(s["time"] for s in stats_base.values())
    total_trunc_base = sum(s["truncations"] for s in stats_base.values())

    total_correct_tl = sum(s["correct"] for s in stats_tokenless.values())
    total_tokens_tl = sum(s["tokens"] for s in stats_tokenless.values())
    total_time_tl = sum(s["time"] for s in stats_tokenless.values())
    total_trunc_tl = sum(s["truncations"] for s in stats_tokenless.values())

    print("\n" + "=" * 105)
    print("HONEST EMPIRICAL FINDINGS: COMPARISON SUMMARY (60 CRITICAL QUESTIONS)")
    print("=" * 105)
    print(f"{'Benchmark':<15} | {'Baseline Acc':<14} | {'Tokenless Acc':<14} | {'Δ Accuracy':<12} | {'Tokens Base':<12} | {'Tokens Active':<12} | {'Token Savings':<14}")
    print("-" * 105)

    for b in target_benchmarks:
        sb = stats_base.get(b, {"correct": 0, "total": 10, "tokens": 0})
        st = stats_tokenless.get(b, {"correct": 0, "total": 10, "tokens": 0})
        acc_b = f"{sb['correct']}/{sb['total']} ({sb['correct']/sb['total']*100:.1f}%)"
        acc_t = f"{st['correct']}/{st['total']} ({st['correct']/st['total']*100:.1f}%)"
        diff = st['correct'] - sb['correct']
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        tok_sav = f"{(1.0 - st['tokens']/max(1, sb['tokens']))*100:.1f}%"
        print(f"{b:<15} | {acc_b:<14} | {acc_t:<14} | {diff_str:<12} | {sb['tokens']:<12} | {st['tokens']:<12} | {tok_sav:<14}")

    print("-" * 105)
    acc_tot_b = f"{total_correct_base}/60 ({total_correct_base/60*100:.1f}%)"
    acc_tot_t = f"{total_correct_tl}/60 ({total_correct_tl/60*100:.1f}%)"
    net_diff = total_correct_tl - total_correct_base
    net_diff_str = f"+{net_diff}" if net_diff > 0 else str(net_diff)
    net_tok_sav = f"{(1.0 - total_tokens_tl/max(1, total_tokens_base))*100:.1f}%"
    print(f"{'TOTAL (60Q)':<15} | {acc_tot_b:<14} | {acc_tot_t:<14} | {net_diff_str:<12} | {total_tokens_base:<12} | {total_tokens_tl:<12} | {net_tok_sav:<14}")
    print("=" * 105)
    print(f"Total Truncated Questions: Baseline = {total_trunc_base}/60 -> Tokenless = {total_trunc_tl}/60 ({(total_trunc_base - total_trunc_tl)/max(1, total_trunc_base)*100:.1f}% reduction)")
    print(f"Total Execution Time     : Baseline = {total_time_base:.2f}s -> Tokenless = {total_time_tl:.2f}s (Speedup: {total_time_base/max(0.1, total_time_tl):.2f}x)")
    print("=" * 105)

    print("\n" + "=" * 105)
    print("PHASE 3: REAL-WORLD ROBUSTNESS PROBE (CHAT, CODING, AND ADVICE)")
    print("=" * 105)

    conversational_probes = [
        {
            "id": "chat_intro",
            "type": "Conversational & Philosophy",
            "prompt": "Halo! Bagaimana kabarmu hari ini? Ceritakan sedikit tentang bagaimana caramu berpikir.",
            "max_tokens": 128
        },
        {
            "id": "metaphor_calendar",
            "type": "Metaphorical Context",
            "prompt": "Yesterday was a really tough day at work, what should I do tomorrow to bounce back?",
            "max_tokens": 128
        },
        {
            "id": "metaphor_compass",
            "type": "Metaphorical Direction",
            "prompt": "My career turned 180 degrees after 10 years in tech. How should I navigate this change?",
            "max_tokens": 128
        },
        {
            "id": "sports_opinion",
            "type": "Contextual Sports Opinion",
            "prompt": "My basketball team scored 10 points in the first quarter and 15 in the second quarter. Do you think our offensive strategy is working, and how should we adjust?",
            "max_tokens": 128
        },
        {
            "id": "code_fibonacci",
            "type": "Python Code Generation",
            "prompt": "Tuliskan fungsi Python def fibonacci(n) secara rekursif dan jelaskan cara kerjanya.",
            "max_tokens": 160
        },
        {
            "id": "code_function_fx",
            "type": "Python Code with f(x)",
            "prompt": "Buatkan kode Python untuk mendefinisikan fungsi f(x) = 2*x + 3 dan menghitung nilainya dari x=1 sampai 5.",
            "max_tokens": 160
        }
    ]

    phase3_results = []
    for probe in conversational_probes:
        p_id = probe["id"]
        p_type = probe["type"]
        prompt = probe["prompt"]
        max_tok = probe["max_tokens"]

        inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_len = inp.input_ids.shape[1]

        with torch.no_grad():
            out_ids = model.generate(
                **inp,
                max_new_tokens=max_tok,
                do_sample=False,
                enable_answer_first=True,
                prompt_text=prompt,
                tokenizer=tokenizer,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_tokens = out_ids.shape[1] - input_len
        raw_text = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True).strip()

        print(f"\n[PROBE: {p_id}] ({p_type})")
        print(f"  Prompt: {prompt}")
        print(f"  Generated Tokens: {gen_tokens}")
        print(f"  Output Sample:\n{raw_text[:200]}...")
        phase3_results.append({
            "id": p_id,
            "type": p_type,
            "prompt": prompt,
            "tokens": gen_tokens,
            "output": raw_text
        })

    # Save complete audit report
    audit_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "summary": {
            "baseline": {
                "score": f"{total_correct_base}/60",
                "accuracy_pct": round(total_correct_base / 60 * 100, 2),
                "total_tokens": total_tokens_base,
                "total_time_sec": round(total_time_base, 2),
                "truncations": total_trunc_base
            },
            "tokenless_active": {
                "score": f"{total_correct_tl}/60",
                "accuracy_pct": round(total_correct_tl / 60 * 100, 2),
                "total_tokens": total_tokens_tl,
                "total_time_sec": round(total_time_tl, 2),
                "truncations": total_trunc_tl
            },
            "deltas": {
                "accuracy_gain_pct": round((total_correct_tl - total_correct_base) / 60 * 100, 2),
                "token_savings_pct": round((1.0 - total_tokens_tl / max(1, total_tokens_base)) * 100, 2),
                "speedup_factor": round(total_time_base / max(0.1, total_time_tl), 2)
            }
        },
        "per_benchmark": {
            b: {
                "baseline": stats_base[b],
                "tokenless": stats_tokenless[b]
            }
            for b in target_benchmarks
        },
        "item_details": {
            "baseline": results_baseline,
            "tokenless": results_tokenless
        },
        "phase3_robustness_probes": phase3_results
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(str(REPORT_JSON), "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Detailed JSON audit saved to: {REPORT_JSON}")

    with open(str(REPORT_TXT), "w", encoding="utf-8") as f:
        f.write(f"=== EMPIRICAL AUDIT REPORT: TOKENLESS LATENT REASONING + IPv6 REGISTER ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Baseline Score: {total_correct_base}/60 ({total_correct_base/60*100:.1f}%)\n")
        f.write(f"Tokenless Active Score: {total_correct_tl}/60 ({total_correct_tl/60*100:.1f}%)\n")
        f.write(f"Accuracy Gain: +{(total_correct_tl - total_correct_base)/60*100:.1f}%\n")
        f.write(f"Token Savings: {net_tok_sav}\n")
        f.write(f"Speedup Factor: {total_time_base/max(0.1, total_time_tl):.2f}x\n\n")
        f.write("=== ITEM DETAILS ===\n")
        for i in range(len(results_baseline)):
            rb = results_baseline[i]
            rt = results_tokenless[i]
            f.write(f"\n[{rb['benchmark']}] {rb['id']} - Target: {rb['target']}\n")
            f.write(f"  Base Output     : {rb['clean_text']} (Correct: {rb['correct']}, Tokens: {rb['tokens']})\n")
            f.write(f"  Tokenless Output: {rt['clean_text']} (Correct: {rt['correct']}, Tokens: {rt['tokens']})\n")

    print(f"[+] Text audit log saved to: {REPORT_TXT}")


if __name__ == "__main__":
    run_benchmark()
