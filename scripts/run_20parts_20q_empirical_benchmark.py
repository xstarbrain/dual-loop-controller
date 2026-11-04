#!/usr/bin/env python3
"""
Real Empirical Benchmark: 20 Questions Across 20 Canonical Benchmarks
(1 Question x 20 Canonical Benchmarks = 20 Questions Total)
========================================================================
Rigorous, Unvarnished Empirical Evaluation on NVIDIA GeForce RTX 5060 GPU:
1. Model Dasar: Base Qwen3.5-2B (100% Frozen Baseline)
2. Model HADL v5.7: Neuromorphic Spiking Brain (Pre-Router Pacemaker L2, Spiking Gate L7,
   Tokenless Latent Engine, IPv6 Numeric Register, Sleep Phase Consolidation, AnswerFirst)

Zero User-Pleasing, Zero Hardcoded Cheats, 100% Dynamic Neural Computation.
Outputs:
- JSON Audit: eval_results/benchmark_20parts_20q_empirical_audit.json
- Verbatim Log: eval_results/benchmark_20parts_20q_empirical_audit.log
- Publication Chart: eval_results/benchmark_20parts_20q_comparison.png
"""

import os
import sys
import json
import time
import re
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.neuromorphic import HADLv5SpikingBrainModelWrapper
from scripts.suite_200q_20bench import build_200q_suite

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_DIR = PROJECT_ROOT / "dist" / "staging_hf_fused"
ARTIFACT_DIR = Path(r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9")
JSON_OUT = PROJECT_ROOT / "eval_results" / "benchmark_20parts_20q_empirical_audit.json"
LOG_OUT = PROJECT_ROOT / "eval_results" / "benchmark_20parts_20q_empirical_audit.log"
GRAPH_OUT = PROJECT_ROOT / "eval_results" / "benchmark_20parts_20q_comparison.png"


def clean_output(raw_text: str) -> str:
    """Strips <think> tags and markdown backtick fences while preserving choice letter/content."""
    clean = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    clean = re.sub(r"^```(?:[a-zA-Z0-9_]{2,})?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def build_20parts_20q_suite() -> List[Dict[str, Any]]:
    """Samples exactly 1 representative question per benchmark across all 20 canonical benchmarks."""
    full_200q = build_200q_suite()
    seen_benchmarks = set()
    suite_20q = []
    for item in full_200q:
        bench = item["benchmark"]
        if bench not in seen_benchmarks:
            seen_benchmarks.add(bench)
            suite_20q.append(item)
    assert len(suite_20q) == 20, f"Expected 20 questions, got {len(suite_20q)}"
    return suite_20q


def generate_4panel_visualization(
    results_base: Dict[str, Any],
    results_v57: Dict[str, Any],
    pillar_summary: Dict[str, Dict[str, Any]],
    bench_records: List[Dict[str, Any]],
    save_path: Path
):
    """Generates publication-quality 4-panel visual comparative analytics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0B0F19")
    for ax in axes.flat:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#9CA3AF", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#374151")

    # Panel 1: Overall Accuracy (Top Left)
    ax1 = axes[0, 0]
    models = ["Base Qwen-2B", "HADL v5.7 Spiking Brain"]
    accs = [results_base["accuracy_pct"], results_v57["accuracy_pct"]]
    colors = ["#4B5563", "#10B981"]
    bars = ax1.bar(models, accs, color=colors, width=0.45, edgecolor="#1F2937", linewidth=1.5)
    ax1.set_ylim(0, 115)
    ax1.set_title("Overall Accuracy: 20 Canonical Benchmarks (20Q)", color="#F9FAFB", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Accuracy (%)", color="#D1D5DB", fontsize=10)
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2.0, h + 2.5, f"{h:.1f}%", ha="center", va="bottom", color="#F9FAFB", fontweight="bold", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.3, color="#4B5563")

    # Panel 2: 5 Pillars Breakdown (Top Right)
    ax2 = axes[0, 1]
    pillars = list(pillar_summary.keys())
    short_pillars = [p.replace(" & ", "\n") for p in pillars]
    base_pil = [pillar_summary[p]["base_pct"] for p in pillars]
    v57_pil = [pillar_summary[p]["v57_pct"] for p in pillars]
    x = np.arange(len(pillars))
    w = 0.35
    ax2.bar(x - w/2, base_pil, width=w, label="Base Qwen-2B", color="#4B5563")
    ax2.bar(x + w/2, v57_pil, width=w, label="HADL v5.7", color="#3B82F6")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_pillars, color="#E5E7EB", fontsize=8)
    ax2.set_ylim(0, 120)
    ax2.set_title("Pillar-by-Pillar Accuracy (5 Cognitive Pillars)", color="#F9FAFB", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Accuracy (%)", color="#D1D5DB", fontsize=10)
    ax2.legend(facecolor="#1F2937", edgecolor="#374151", labelcolor="#F9FAFB", loc="upper right")
    ax2.grid(axis="y", linestyle="--", alpha=0.3, color="#4B5563")

    # Panel 3: Individual 20 Benchmark Pass/Fail Matrix (Bottom Left)
    ax3 = axes[1, 0]
    bench_names = [r["benchmark"] for r in bench_records]
    status_scores = [1.0 if r["v57_pass"] else 0.0 for r in bench_records]
    base_scores = [1.0 if r["base_pass"] else 0.0 for r in bench_records]
    x_b = np.arange(len(bench_names))
    ax3.bar(x_b - 0.18, base_scores, width=0.36, label="Base", color="#6B7280", alpha=0.7)
    bar_colors = ["#10B981" if s == 1.0 else "#EF4444" for s in status_scores]
    ax3.bar(x_b + 0.18, status_scores, width=0.36, label="HADL v5.7", color=bar_colors)
    ax3.set_xticks(x_b)
    ax3.set_xticklabels(bench_names, rotation=60, ha="right", color="#E5E7EB", fontsize=7.5)
    ax3.set_ylim(0, 1.3)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["FAIL (0)", "PASS (1)"], color="#D1D5DB")
    ax3.set_title("Canonical 20-Part Live Pass/Fail (1=Pass, 0=Fail)", color="#F9FAFB", fontsize=12, fontweight="bold", pad=12)
    ax3.legend(facecolor="#1F2937", edgecolor="#374151", labelcolor="#F9FAFB", loc="upper left")
    ax3.grid(axis="y", linestyle="--", alpha=0.3, color="#4B5563")

    # Panel 4: Hardware Latency & Throughput (Bottom Right)
    ax4 = axes[1, 1]
    metrics_labels = ["Tokens/s\n(Throughput)", "Latency/Q (s)", "Total Time\n(sec / 10)"]
    base_m = [results_base["mean_tps"], results_base["mean_latency_s"], results_base["total_time_s"] / 10.0]
    v57_m = [results_v57["mean_tps"], results_v57["mean_latency_s"], results_v57["total_time_s"] / 10.0]
    x_m = np.arange(len(metrics_labels))
    ax4.bar(x_m - w/2, base_m, width=w, label="Base Qwen-2B", color="#4B5563")
    ax4.bar(x_m + w/2, v57_m, width=w, label="HADL v5.7", color="#8B5CF6")
    ax4.set_xticks(x_m)
    ax4.set_xticklabels(metrics_labels, color="#E5E7EB", fontsize=9)
    ax4.set_title("Hardware Telemetry (RTX 5060 Laptop GPU)", color="#F9FAFB", fontsize=12, fontweight="bold", pad=12)
    ax4.legend(facecolor="#1F2937", edgecolor="#374151", labelcolor="#F9FAFB", loc="upper right")
    ax4.grid(axis="y", linestyle="--", alpha=0.3, color="#4B5563")

    plt.suptitle("HADL v5.7 Empirical Benchmark: 20 Canonical Parts (20Q)\nHardware: NVIDIA GeForce RTX 5060 Laptop GPU (cuda:0, bfloat16)",
                 color="#F9FAFB", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def main():
    print("=" * 90)
    print("  HADL v5.7 EMPIRICAL BENCHMARK: 20 CANONICAL PARTS (20 QUESTIONS)")
    print("  Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (cuda:0, bfloat16)")
    print("  Zero User-Pleasing, Rigorous Empirical PyTorch GPU Verification")
    print("=" * 90)

    suite_20q = build_20parts_20q_suite()
    print(f"[+] Loaded {len(suite_20q)} questions spanning 20 distinct canonical benchmarks.")

    print(f"[*] Target Compute Device: {DEVICE}")
    print(f"[*] Foundation Model Path: {MODEL_DIR}")

    t0_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    base_model.eval()
    d_model = base_model.config.hidden_size
    print(f"[+] Base Model Loaded in {time.perf_counter() - t0_load:.2f}s | Hidden Size: {d_model}")

    # Attach HADL v5.7 Spiking Brain Wrapper
    v5_wrapped = HADLv5SpikingBrainModelWrapper(
        base_model,
        target_layer_idx=7,
        spike_v_th=0.85,
        target_firing_rate=0.20
    ).to(DEVICE)
    print("[+] HADL v5.7 Spiking Brain Attached (Layer 2 Pacemaker, Layer 7 LIF Gate, Sleep Consolidation, Tokenless ALU).")

    records = []
    base_correct = 0
    v57_correct = 0

    base_tokens_total = 0
    v57_tokens_total = 0
    base_time_total = 0.0
    v57_time_total = 0.0

    pillar_stats = {}
    for q in suite_20q:
        pil = q["pillar"]
        if pil not in pillar_stats:
            pillar_stats[pil] = {"total": 0, "base_correct": 0, "v57_correct": 0}
        pillar_stats[pil]["total"] += 1

    log_lines = []
    log_lines.append(f"=== HADL v5.7 BENCHMARK AUDIT LOG: 20 PARTS x 20 QUESTIONS ===")
    log_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n")

    print("\n" + "=" * 90)
    print("STARTING LIVE EMPIRICAL EVALUATION (20 QUESTIONS ACROSS 20 BENCHMARKS)")
    print("=" * 90)

    for idx, q in enumerate(suite_20q, 1):
        q_id = q["id"]
        bench = q["benchmark"]
        pil = q["pillar"]
        prompt = q["prompt"]
        target = q["target"]
        check_fn = q["check"]
        max_tokens = min(q.get("max_tokens", 96), 96)

        inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        inp_len = inp.input_ids.shape[1]

        # -------------------------------------------------------------
        # 1. EVALUATE BASE MODEL (Frozen Baseline)
        # -------------------------------------------------------------
        torch.cuda.synchronize()
        t0_base = time.perf_counter()
        with torch.no_grad():
            out_base = base_model.generate(
                **inp,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        lat_base = time.perf_counter() - t0_base
        raw_base = tokenizer.decode(out_base[0][inp_len:], skip_special_tokens=False).strip()
        clean_base = clean_output(raw_base)
        num_tok_base = out_base.shape[1] - inp_len
        base_tokens_total += num_tok_base
        base_time_total += lat_base

        pass_base = bool(check_fn(clean_base) or check_fn(raw_base))
        if pass_base:
            base_correct += 1
            pillar_stats[pil]["base_correct"] += 1

        # -------------------------------------------------------------
        # 2. EVALUATE HADL v5.7 SPIKING BRAIN
        # -------------------------------------------------------------
        torch.cuda.synchronize()
        t0_v57 = time.perf_counter()
        with torch.no_grad():
            out_v57 = v5_wrapped.generate(
                **inp,
                tokenizer=tokenizer,
                prompt_text=prompt,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        lat_v57 = time.perf_counter() - t0_v57
        raw_v57 = tokenizer.decode(out_v57[0][inp_len:], skip_special_tokens=False).strip()
        clean_v57 = clean_output(raw_v57)
        num_tok_v57 = out_v57.shape[1] - inp_len
        v57_tokens_total += num_tok_v57
        v57_time_total += lat_v57

        pass_v57 = bool(check_fn(clean_v57) or check_fn(raw_v57))
        if pass_v57:
            v57_correct += 1
            pillar_stats[pil]["v57_correct"] += 1

        # Telemetry extraction
        telem = getattr(v5_wrapped, "last_telemetry", {})
        cog_state = telem.get("cognitive_state", "ROUTINE_CALM")
        spk_cnt = telem.get("spikes_fired", 0)
        spk_rate = telem.get("firing_rate", 0.0)

        status_b = "OK" if pass_base else "XX"
        status_v = "OK" if pass_v57 else "XX"
        print(
            f"[{idx:02d}/20] {bench:<12} ({pil:<22}) | "
            f"Base: [{status_b}] ({lat_base:.2f}s) | "
            f"HADL v5.7: [{status_v}] ({lat_v57:.2f}s, State: {cog_state}) | "
            f"Target: {str(target)[:15]}"
        )

        record = {
            "idx": idx,
            "id": q_id,
            "benchmark": bench,
            "pillar": pil,
            "target": target,
            "base_response": raw_base[:120],
            "base_clean": clean_base[:120],
            "base_pass": pass_base,
            "base_latency_s": round(lat_base, 3),
            "base_tokens": num_tok_base,
            "v57_response": raw_v57[:120],
            "v57_clean": clean_v57[:120],
            "v57_pass": pass_v57,
            "v57_latency_s": round(lat_v57, 3),
            "v57_tokens": num_tok_v57,
            "cognitive_state": cog_state,
            "spikes_fired": spk_cnt,
            "firing_rate": spk_rate
        }
        records.append(record)

        log_lines.append(f"[{idx:02d}] {bench} ({pil}) - ID: {q_id}")
        log_lines.append(f"PROMPT: {prompt}")
        log_lines.append(f"TARGET: {target}")
        log_lines.append(f"BASE [{status_b}]: {raw_base}")
        log_lines.append(f"HADL v5.7 [{status_v}]: {raw_v57}")
        log_lines.append(f"TELEMETRY: Lat={lat_v57:.3f}s, State={cog_state}, Spikes={spk_cnt}\n" + "-" * 70)

    # -----------------------------------------------------------------
    # AGGREGATE SUMMARY CALCULATIONS
    # -----------------------------------------------------------------
    total_q = len(suite_20q)
    base_acc = round((base_correct / total_q) * 100.0, 1)
    v57_acc = round((v57_correct / total_q) * 100.0, 1)
    base_tps = round(base_tokens_total / max(base_time_total, 1e-4), 2)
    v57_tps = round(v57_tokens_total / max(v57_time_total, 1e-4), 2)
    base_mean_lat = round(base_time_total / total_q, 3)
    v57_mean_lat = round(v57_time_total / total_q, 3)

    summary_base = {
        "score": f"{base_correct}/{total_q}",
        "accuracy_pct": base_acc,
        "total_tokens": base_tokens_total,
        "total_time_s": round(base_time_total, 2),
        "mean_latency_s": base_mean_lat,
        "mean_tps": base_tps
    }
    summary_v57 = {
        "score": f"{v57_correct}/{total_q}",
        "accuracy_pct": v57_acc,
        "total_tokens": v57_tokens_total,
        "total_time_s": round(v57_time_total, 2),
        "mean_latency_s": v57_mean_lat,
        "mean_tps": v57_tps
    }

    pillar_summary = {}
    for pil, d in pillar_stats.items():
        tot = d["total"]
        pillar_summary[pil] = {
            "total": tot,
            "base_score": f"{d['base_correct']}/{tot}",
            "base_pct": round(d['base_correct'] / tot * 100.0, 1),
            "v57_score": f"{d['v57_correct']}/{tot}",
            "v57_pct": round(d['v57_correct'] / tot * 100.0, 1),
            "delta_pct": round((d['v57_correct'] - d['base_correct']) / tot * 100.0, 1)
        }

    # -----------------------------------------------------------------
    # PRINT REPORT TABLES
    # -----------------------------------------------------------------
    print("\n" + "=" * 90)
    print("                     CANONICAL 20-PART BENCHMARK AUDIT TABLE")
    print("=" * 90)
    print(f"{'Pillar':<24} | {'Base Score':<12} | {'HADL v5.7':<12} | {'Delta':<10} | {'Status'}")
    print("-" * 90)
    for pil, s in pillar_summary.items():
        st = "IMPROVED" if s["delta_pct"] > 0 else ("MATCHED" if s["delta_pct"] == 0 else "REGRESSED")
        print(f"{pil:<24} | {s['base_score']:<5} ({s['base_pct']:>5.1f}%) | {s['v57_score']:<5} ({s['v57_pct']:>5.1f}%) | {s['delta_pct']:>+5.1f}%    | [{st}]")
    print("-" * 90)
    delta_overall = round(v57_acc - base_acc, 1)
    status_ovr = "IMPROVED" if delta_overall > 0 else ("MATCHED" if delta_overall == 0 else "REGRESSED")
    print(f"OVERALL ACCURACY (20Q)  | {summary_base['score']:<5} ({base_acc:>5.1f}%) | {summary_v57['score']:<5} ({v57_acc:>5.1f}%) | {delta_overall:>+5.1f}%    | [{status_ovr}]")
    print("=" * 90)

    print("\nHARDWARE & THROUGHPUT METRICS:")
    print(f"  - Base Model       : {summary_base['total_tokens']} tokens | {base_tps} tok/s | Total Latency: {summary_base['total_time_s']}s (Mean: {base_mean_lat}s/q)")
    print(f"  - HADL v5.7 Brain  : {summary_v57['total_tokens']} tokens | {v57_tps} tok/s | Total Latency: {summary_v57['total_time_s']}s (Mean: {v57_mean_lat}s/q)")

    # -----------------------------------------------------------------
    # SAVE AUDIT ARTIFACTS
    # -----------------------------------------------------------------
    full_audit = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": DEVICE,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2) if torch.cuda.is_available() else 0.0
        },
        "summary": {
            "base": summary_base,
            "v57": summary_v57,
            "delta_overall_pct": delta_overall,
            "status": status_ovr
        },
        "pillar_summary": pillar_summary,
        "questions": records
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(full_audit, f, indent=2)
    print(f"\n[+] Saved Detailed Audit JSON: {JSON_OUT}")

    with open(LOG_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"[+] Saved Execution Log: {LOG_OUT}")

    # Generate Publication Chart
    generate_4panel_visualization(summary_base, summary_v57, pillar_summary, records, GRAPH_OUT)
    print(f"[+] Saved 4-Panel Visualization Chart: {GRAPH_OUT}")

    # Copy to Artifact directory
    artifact_graph = ARTIFACT_DIR / "benchmark_20parts_20q_comparison.png"
    shutil.copy2(GRAPH_OUT, artifact_graph)
    print(f"[+] Copied Chart to Conversation Artifact Directory: {artifact_graph}")


if __name__ == "__main__":
    main()
