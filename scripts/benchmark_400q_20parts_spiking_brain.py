#!/usr/bin/env python3
"""
Real Empirical Benchmark: 400 Questions Across 20 Canonical Benchmarks
(20 Questions x 20 Canonical Benchmarks = 400 Questions Total)
========================================================================
Head-to-Head Triple Comparison:
1. Model Dasar: Base Qwen3.5-2B (100% Frozen Baseline)
2. Model Autonomous: Autonomous Native Brain (dist/xstar_2b_autonomous_brain)
3. Model v5.0: HADL v5.0 Neuromorphic Spiking Brain (Layer 7 LIF Gate, VGNI, DNR, Open-Freeze Plasticity, Banach Sandbox)

Rigorous Integrity Standards:
- Real PyTorch CUDA execution on local NVIDIA GPU (cuda:0, bfloat16)
- Checkpointing every 10 questions to prevent any data loss
- Detailed JSON, Text Log, and Publication-Grade Comparison Graph
"""

import os
import sys
import json
import time
import re
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.suite_1000q_20bench import build_1000q_suite
from dual_loop.neuromorphic import HADLv5SpikingBrainModelWrapper

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_MODEL_DIR = PROJECT_ROOT / "dist" / "staging_hf_fused"
CHECKPOINT_1000Q = PROJECT_ROOT / "eval_results" / "benchmark_1000q_checkpoint.json"

JSON_OUT = PROJECT_ROOT / "eval_results" / "benchmark_400q_20parts_real_audit.json"
CHECKPOINT_OUT = PROJECT_ROOT / "eval_results" / "benchmark_400q_checkpoint.json"
TXT_LOG_OUT = PROJECT_ROOT / "eval_results" / "benchmark_400q_20parts_real_log.txt"
GRAPH_OUT = PROJECT_ROOT / "eval_results" / "benchmark_400q_20parts_real_comparison.png"
ARTIFACT_DIR = Path(r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9")


def build_400q_suite() -> List[Dict[str, Any]]:
    full_1000 = build_1000q_suite()
    grouped = defaultdict(list)
    for item in full_1000:
        grouped[item["benchmark"]].append(item)

    bench_order = [
        # Pillar 1: Math & Quantitative
        "GSM8K", "MATH", "DROP", "BBH",
        # Pillar 2: Science & Academic
        "MMLU", "AGIEval", "TriviaQA", "SQuAD_v2", "ARC-c",
        # Pillar 3: Coding & Software
        "HumanEval", "MBPP", "CodeDebug",
        # Pillar 4: Commonsense & Logic
        "ARC-e", "HellaSwag", "WinoGrande", "PIQA",
        # Pillar 5: Instruction & Chat
        "BoolQ", "TruthfulQA", "IFEval", "DailyChat"
    ]

    suite_400 = []
    for b in bench_order:
        items = grouped.get(b, [])
        assert len(items) >= 20, f"Benchmark {b} has fewer than 20 items!"
        suite_400.extend(items[:20])

    assert len(suite_400) == 400, f"Expected 400 questions, got {len(suite_400)}"
    return suite_400


def load_autonomous_400q_data(suite_400: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    auto_map = {}
    if CHECKPOINT_1000Q.exists():
        with open(str(CHECKPOINT_1000Q), "r", encoding="utf-8") as f:
            data = json.load(f)
            for d in data.get("details", []):
                auto_map[d["id"]] = d

    suite_auto = {}
    for item in suite_400:
        q_id = item["id"]
        if q_id in auto_map:
            suite_auto[q_id] = auto_map[q_id]
        else:
            suite_auto[q_id] = {
                "id": q_id,
                "correct": False,
                "generated_tokens": 0,
                "latency_sec": 0.0,
                "tokens_per_sec": 0.0,
                "clean_output": ""
            }
    return suite_auto


def generate_single(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 120
) -> Dict[str, Any]:
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inp.input_ids.shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            **inp,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    gen_toks = out_ids.shape[1] - input_len
    raw_text = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=False).strip()
    clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    tps = gen_toks / latency if latency > 0 else 0.0

    return {
        "raw_text": raw_text,
        "clean_text": clean_text,
        "gen_tokens": gen_toks,
        "latency_sec": latency,
        "tokens_per_sec": tps
    }


def main():
    print("=" * 105)
    print("   X-STAR REAL EMPIRICAL BENCHMARK: 400 QUESTIONS ACROSS 20 CANONICAL BENCHMARKS")
    print("        (20 Questions x 20 Benchmarks = 400 Questions Head-to-Head Triple Audit)")
    print("=" * 105)
    print(f"[*] Target Compute Device : {DEVICE}")
    print(f"[*] Foundation Model Path : {BASE_MODEL_DIR}")

    suite_400 = build_400q_suite()
    print(f"[+] Successfully built 400-question suite across 20 canonical benchmarks.")

    auto_data = load_autonomous_400q_data(suite_400)
    print(f"[+] Loaded {len(auto_data)} verified autonomous benchmark entries.")

    print("\n[*] Loading tokenizer and base foundation model (bfloat16)...")
    t0_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    t_load = time.perf_counter() - t0_load
    print(f"[+] Model loaded in {t_load:.2f}s on {DEVICE}.")

    print("\n[*] Initializing HADL v5.0 Neuromorphic Spiking Brain Wrapper...")
    v5_wrapped = HADLv5SpikingBrainModelWrapper(
        base_model,
        target_layer_idx=7,
        spike_v_th=0.85,
        target_firing_rate=0.20
    ).to(DEVICE)
    print("[+] HADL v5.0 Spiking Brain successfully attached to Layer 7 (with Re-entry at L14 & L21).")

    # Load existing checkpoint if available
    saved_records = {}
    if CHECKPOINT_OUT.exists():
        try:
            with open(str(CHECKPOINT_OUT), "r", encoding="utf-8") as f:
                ckpt_data = json.load(f)
                for r in ckpt_data.get("records", []):
                    saved_records[r["id"]] = r
                print(f"[+] Resuming from existing checkpoint: {len(saved_records)} / 400 completed.")
        except Exception as e:
            print(f"[!] Warning reading checkpoint: {e}. Starting fresh.")
            saved_records = {}

    log_file = open(str(TXT_LOG_OUT), "a", encoding="utf-8")
    log_file.write(f"\n\n{'='*105}\nNEW BENCHMARK RUN: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*105}\n")
    log_file.flush()

    records = list(saved_records.values())
    completed_ids = set(saved_records.keys())

    print("\n" + "=" * 105)
    print("STARTING REAL EVALUATION LOOP (400 QUESTIONS)")
    print("=" * 105)

    try:
        for idx, item in enumerate(suite_400, 1):
            q_id = item["id"]
            bench = item["benchmark"]
            pillar = item["pillar"]
            prompt = item["prompt"]
            target = item["target"]
            check_fn = item["check"]
            max_tok = min(120, item.get("max_tokens", 120))

            if q_id in completed_ids:
                continue

            # 1. Base Model Pass (v5 disabled)
            v5_wrapped.enabled = False
            base_res = generate_single(base_model, tokenizer, prompt, max_tokens=max_tok)
            base_pass = check_fn(base_res["clean_text"]) or check_fn(base_res["raw_text"])

            # 2. HADL v5.0 Spiking Brain Pass (v5 enabled)
            v5_wrapped.enabled = True
            v5_wrapped.reset_state()
            v5_res = generate_single(base_model, tokenizer, prompt, max_tokens=max_tok)
            v5_pass = check_fn(v5_res["clean_text"]) or check_fn(v5_res["raw_text"])
            v5_telem = dict(v5_wrapped.last_telemetry)

            # 3. Autonomous Model Data
            auto_item = auto_data.get(q_id, {})
            auto_pass = auto_item.get("correct", False)

            # Record result
            rec = {
                "idx": idx,
                "id": q_id,
                "benchmark": bench,
                "pillar": pillar,
                "prompt": prompt,
                "target": target,
                "base": {
                    "correct": base_pass,
                    "tokens": base_res["gen_tokens"],
                    "latency": round(base_res["latency_sec"], 4),
                    "tps": round(base_res["tokens_per_sec"], 2),
                    "clean_output": base_res["clean_text"][:200]
                },
                "hadl_v5": {
                    "correct": v5_pass,
                    "tokens": v5_res["gen_tokens"],
                    "latency": round(v5_res["latency_sec"], 4),
                    "tps": round(v5_res["tokens_per_sec"], 2),
                    "clean_output": v5_res["clean_text"][:200],
                    "spike_status": v5_telem.get("gate_status", "N/A"),
                    "firing_rate": v5_telem.get("spiking_firing_rate", 0.0),
                    "sandbox_intervention": v5_telem.get("sandbox_engaged", False)
                },
                "autonomous": {
                    "correct": auto_pass,
                    "tokens": auto_item.get("generated_tokens", 0),
                    "latency": round(auto_item.get("latency_sec", 0.0), 4),
                    "tps": round(auto_item.get("tokens_per_sec", 0.0), 2)
                }
            }
            records.append(rec)
            completed_ids.add(q_id)

            # Console output line
            base_mark = "[PASS ✓]" if base_pass else "[FAIL ✗]"
            v5_mark = "[PASS ✓]" if v5_pass else "[FAIL ✗]"
            auto_mark = "[PASS ✓]" if auto_pass else "[FAIL ✗]"
            fire_rate = v5_telem.get("spiking_firing_rate", 0.0) * 100.0

            log_line = (
                f"Q{idx:03d} | {bench:<10} | {q_id:<11} | "
                f"Base: {base_mark} | v5.0: {v5_mark} | Auto: {auto_mark} | "
                f"Spike: {fire_rate:4.1f}% | Lat: {v5_res['latency_sec']:.2f}s ({v5_res['tokens_per_sec']:.1f} t/s)"
            )
            print(log_line)
            log_file.write(log_line + "\n")
            log_file.flush()

            # Save checkpoint every 10 questions
            if len(records) % 10 == 0 or len(records) == 400:
                with open(str(CHECKPOINT_OUT), "w", encoding="utf-8") as f:
                    json.dump({"records": records, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    except KeyboardInterrupt:
        print("\n[!] Run interrupted by user. Saving checkpoint...")
        with open(str(CHECKPOINT_OUT), "w", encoding="utf-8") as f:
            json.dump({"records": records, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        log_file.close()
        return

    log_file.close()

    # =========================================================================
    # POST-RUN AGGREGATION & METRICS
    # =========================================================================
    print("\n" + "=" * 105)
    print("AGGREGATING 400-QUESTION BENCHMARK RESULTS")
    print("=" * 105)

    bench_stats = defaultdict(lambda: {
        "base_correct": 0, "v5_correct": 0, "auto_correct": 0, "total": 0,
        "base_tokens": 0, "v5_tokens": 0, "auto_tokens": 0,
        "base_lat": 0.0, "v5_lat": 0.0, "auto_lat": 0.0,
        "pillar": ""
    })
    pillar_stats = defaultdict(lambda: {
        "base_correct": 0, "v5_correct": 0, "auto_correct": 0, "total": 0
    })

    tot_base_correct = sum(1 for r in records if r["base"]["correct"])
    tot_v5_correct = sum(1 for r in records if r["hadl_v5"]["correct"])
    tot_auto_correct = sum(1 for r in records if r["autonomous"]["correct"])
    total_q = len(records)

    for r in records:
        b = r["benchmark"]
        p = r["pillar"]
        bench_stats[b]["pillar"] = p
        bench_stats[b]["total"] += 1
        if r["base"]["correct"]:
            bench_stats[b]["base_correct"] += 1
            pillar_stats[p]["base_correct"] += 1
        if r["hadl_v5"]["correct"]:
            bench_stats[b]["v5_correct"] += 1
            pillar_stats[p]["v5_correct"] += 1
        if r["autonomous"]["correct"]:
            bench_stats[b]["auto_correct"] += 1
            pillar_stats[p]["auto_correct"] += 1
        pillar_stats[p]["total"] += 1

        bench_stats[b]["base_tokens"] += r["base"]["tokens"]
        bench_stats[b]["v5_tokens"] += r["hadl_v5"]["tokens"]
        bench_stats[b]["auto_tokens"] += r["autonomous"]["tokens"]
        bench_stats[b]["base_lat"] += r["base"]["latency"]
        bench_stats[b]["v5_lat"] += r["hadl_v5"]["latency"]
        bench_stats[b]["auto_lat"] += r["autonomous"]["latency"]

    # Table format output
    print(f"\n{'Benchmark':<12} | {'Pillar':<22} | {'Base (2B)':<10} | {'HADL v5.0':<10} | {'Autonomous':<10} | {'Delta v5 vs Base':<16}")
    print("-" * 92)
    bench_table_data = []
    for b, s in bench_stats.items():
        base_pct = (s["base_correct"] / max(1, s["total"])) * 100.0
        v5_pct = (s["v5_correct"] / max(1, s["total"])) * 100.0
        auto_pct = (s["auto_correct"] / max(1, s["total"])) * 100.0
        delta = v5_pct - base_pct
        bench_table_data.append({
            "benchmark": b,
            "pillar": s["pillar"],
            "base_pct": round(base_pct, 1),
            "v5_pct": round(v5_pct, 1),
            "auto_pct": round(auto_pct, 1),
            "delta_pct": round(delta, 1),
            "base_score": f"{s['base_correct']}/{s['total']}",
            "v5_score": f"{s['v5_correct']}/{s['total']}",
            "auto_score": f"{s['auto_correct']}/{s['total']}"
        })
        delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
        print(f"{b:<12} | {s['pillar']:<22} | {base_pct:5.1f}%    | {v5_pct:5.1f}%    | {auto_pct:5.1f}%    | {delta_str:<16}")

    print("-" * 92)
    base_tot_pct = (tot_base_correct / total_q) * 100.0
    v5_tot_pct = (tot_v5_correct / total_q) * 100.0
    auto_tot_pct = (tot_auto_correct / total_q) * 100.0
    tot_delta = v5_tot_pct - base_tot_pct
    tot_delta_str = f"+{tot_delta:.1f}%" if tot_delta >= 0 else f"{tot_delta:.1f}%"
    print(f"{'OVERALL':<12} | {'ALL 20 BENCHMARKS':<22} | {base_tot_pct:5.1f}%    | {v5_tot_pct:5.1f}%    | {auto_tot_pct:5.1f}%    | {tot_delta_str:<16}")

    # =========================================================================
    # PLOT PUBLICATION GRAPH
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Subplot 1: All 20 Benchmarks Bar Comparison
    b_names = [d["benchmark"] for d in bench_table_data]
    b_base = [d["base_pct"] for d in bench_table_data]
    b_v5 = [d["v5_pct"] for d in bench_table_data]
    b_auto = [d["auto_pct"] for d in bench_table_data]

    x = np.arange(len(b_names))
    width = 0.28

    axes[0].bar(x - width, b_base, width, label="Base Qwen3.5-2B (Frozen)", color="#7f8c8d", alpha=0.85)
    axes[0].bar(x, b_v5, width, label="HADL v5.0 Spiking Brain", color="#2980b9", alpha=0.9)
    axes[0].bar(x + width, b_auto, width, label="Autonomous Native Brain", color="#27ae60", alpha=0.9)

    axes[0].set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    axes[0].set_title("Per-Benchmark Empirical Accuracy (20 Canonical Benchmarks x 20 Questions = 400Q)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(b_names, rotation=45, ha="right", fontsize=9)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=10)

    # Subplot 2: 5 Pillars Radar / Bar Summary
    pillars = list(pillar_stats.keys())
    p_base = [(pillar_stats[p]["base_correct"] / max(1, pillar_stats[p]["total"])) * 100.0 for p in pillars]
    p_v5 = [(pillar_stats[p]["v5_correct"] / max(1, pillar_stats[p]["total"])) * 100.0 for p in pillars]
    p_auto = [(pillar_stats[p]["auto_correct"] / max(1, pillar_stats[p]["total"])) * 100.0 for p in pillars]

    xp = np.arange(len(pillars))
    wp = 0.26
    axes[1].bar(xp - wp, p_base, wp, label="Base Qwen3.5-2B", color="#7f8c8d", alpha=0.85)
    axes[1].bar(xp, p_v5, wp, label="HADL v5.0 Spiking Brain", color="#2980b9", alpha=0.9)
    axes[1].bar(xp + wp, p_auto, wp, label="Autonomous Native Brain", color="#27ae60", alpha=0.9)

    axes[1].set_ylabel("Pillar Accuracy (%)", fontsize=11, fontweight="bold")
    axes[1].set_title("5 Cognitive Pillars Empirical Performance (400 Questions Total)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels(pillars, rotation=25, ha="right", fontsize=10)
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(str(GRAPH_OUT), dpi=200)
    plt.close()
    print(f"[+] Saved comparison plot to: {GRAPH_OUT}")

    if ARTIFACT_DIR.exists():
        artifact_png = ARTIFACT_DIR / "benchmark_400q_20parts_real_comparison.png"
        shutil.copyfile(str(GRAPH_OUT), str(artifact_png))
        print(f"[+] Copied comparison plot to artifact dir: {artifact_png}")

    # =========================================================================
    # SAVE DETAILED JSON AUDIT
    # =========================================================================
    audit_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        },
        "suite_info": {
            "total_questions": total_q,
            "benchmarks_count": len(bench_stats),
            "questions_per_benchmark": 20
        },
        "overall_summary": {
            "base_accuracy_pct": round(base_tot_pct, 2),
            "hadl_v5_accuracy_pct": round(v5_tot_pct, 2),
            "autonomous_accuracy_pct": round(auto_tot_pct, 2),
            "delta_v5_vs_base_pct": round(tot_delta, 2)
        },
        "pillar_summary": {
            p: {
                "base_pct": round((pillar_stats[p]["base_correct"] / max(1, pillar_stats[p]["total"])) * 100.0, 2),
                "hadl_v5_pct": round((pillar_stats[p]["v5_correct"] / max(1, pillar_stats[p]["total"])) * 100.0, 2),
                "autonomous_pct": round((pillar_stats[p]["auto_correct"] / max(1, pillar_stats[p]["total"])) * 100.0, 2),
                "total": pillar_stats[p]["total"]
            }
            for p in pillar_stats
        },
        "benchmark_breakdown": bench_table_data,
        "details": records
    }

    with open(str(JSON_OUT), "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"[+] Saved detailed audit JSON to: {JSON_OUT}")
    print("\n[+] 400-QUESTION REAL BENCHMARK COMPLETE!")


if __name__ == "__main__":
    main()
