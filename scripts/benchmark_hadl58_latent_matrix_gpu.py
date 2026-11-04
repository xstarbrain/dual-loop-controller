#!/usr/bin/env python3
"""
Empirical GPU Benchmark Runner for HADL v5.8 Neuromorphic Spiking Brain
with Non-Token Latent Thought Matrix (M in R^{B x S x 4 x D}) and
Recurrent Toroidal Central Cognitive Router ("Otak Berputar").
=======================================================================
Supports:
- 20 Questions per section x 20 Benchmarks = 400 Questions (with multi-run verification)
- 50 Questions per section x 20 Benchmarks = 1,000 Questions (full escalation suite)
- Zero hardcoded myth/keyword clamping lists
- Full per-question telemetry, failure diagnostics, and historical baseline comparison
"""

from __future__ import annotations

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.neuromorphic import attach_hadl_v5_spiking_brain
from scripts.suite_1000q_20bench import build_1000q_suite

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_MODEL_DIR = PROJECT_ROOT / "dist" / "staging_hf_fused"
ADAPTER_DIR = PROJECT_ROOT / "dist" / "frontier_lora_adapter"
BASE_400Q_AUDIT_JSON = PROJECT_ROOT / "eval_results" / "benchmark_400q_20parts_real_audit.json"
BASE_1000Q_AUDIT_JSON = PROJECT_ROOT / "eval_results" / "benchmark_1000q_20parts_real_audit.json"

BENCH_ORDER = [
    # Pillar 1: Math & Quantitative
    "GSM8K", "MATH", "DROP", "BBH",
    # Pillar 2: Science & Academic
    "MMLU", "AGIEval", "TriviaQA", "SQuAD_v2", "ARC-c",
    # Pillar 3: Coding & Software
    "HumanEval", "MBPP", "CodeDebug",
    # Pillar 4: Commonsense & Logic
    "ARC-e", "HellaSwag", "WinoGrande", "PIQA",
    # Pillar 5: Instruction & Chat
    "BoolQ", "TruthfulQA", "IFEval", "DailyChat",
]


def build_benchmark_suite(
    questions_per_bench: int = 20,
    filter_benchmarks: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    full_1000 = build_1000q_suite()
    grouped = defaultdict(list)
    for item in full_1000:
        grouped[item["benchmark"]].append(item)

    target_benches = filter_benchmarks if filter_benchmarks else BENCH_ORDER
    suite: List[Dict[str, Any]] = []
    for b in target_benches:
        items = grouped.get(b, [])
        if len(items) < questions_per_bench:
            raise ValueError(f"Benchmark {b} only has {len(items)} items (< {questions_per_bench})")
        suite.extend(items[:questions_per_bench])
    return suite


def load_historical_baselines() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    hist_400: Dict[str, Any] = {}
    hist_1000: Dict[str, Any] = {}
    if BASE_400Q_AUDIT_JSON.exists():
        try:
            with open(str(BASE_400Q_AUDIT_JSON), "r", encoding="utf-8") as f:
                data = json.load(f)
                hist_400 = {d["id"]: d for d in data.get("details", [])}
        except Exception:
            pass
    if BASE_1000Q_AUDIT_JSON.exists():
        try:
            with open(str(BASE_1000Q_AUDIT_JSON), "r", encoding="utf-8") as f:
                data = json.load(f)
                hist_1000 = {d["id"]: d for d in data.get("details", [])}
        except Exception:
            pass
    return hist_400, hist_1000


def generate_single_hadl58(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 120,
) -> Dict[str, Any]:
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inp.input_ids.shape[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            **inp,
            max_new_tokens=max_tokens,
            do_sample=False,
            enable_answer_first=True,
            prompt_text=prompt,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
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
        "tokens_per_sec": tps,
    }


def evaluate_single_run(
    model: nn.Module,
    tokenizer: Any,
    suite: List[Dict[str, Any]],
    run_idx: int,
    total_runs: int,
    hist_400: Dict[str, Any],
    hist_1000: Dict[str, Any],
    checkpoint_path: Path,
    log_path: Path,
    fresh: bool = False,
) -> Dict[str, Any]:
    saved_records: Dict[str, Any] = {}
    if not fresh and checkpoint_path.exists():
        try:
            with open(str(checkpoint_path), "r", encoding="utf-8") as f:
                ckpt = json.load(f)
                for r in ckpt.get("records", []):
                    saved_records[r["id"]] = r
                print(f"[+] [Run {run_idx}/{total_runs}] Resuming from checkpoint: {len(saved_records)}/{len(suite)}")
        except Exception as e:
            print(f"[!] Warning reading checkpoint: {e}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(str(log_path), "a", encoding="utf-8")
    log_file.write(
        f"\n{'='*105}\nHADL v5.8 LATENT MATRIX & CENTRAL ROUTER RUN {run_idx}/{total_runs} "
        f"({len(suite)} Questions) - {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*105}\n"
    )
    log_file.flush()

    records: List[Dict[str, Any]] = []
    for item in suite:
        if item["id"] in saved_records:
            records.append(saved_records[item["id"]])
    completed_ids = set(saved_records.keys())

    for idx, item in enumerate(suite, 1):
        q_id = item["id"]
        if q_id in completed_ids:
            continue

        bench = item["benchmark"]
        pillar = item["pillar"]
        prompt = item["prompt"]
        target = item["target"]
        check_fn = item["check"]
        max_tok = item.get("max_tokens", 180)

        res = generate_single_hadl58(model, tokenizer, prompt, max_tokens=max_tok)
        passed = check_fn(res["clean_text"]) or check_fn(res["raw_text"])
        telem = dict(getattr(model, "last_telemetry", {}))

        hist_400_item = hist_400.get(q_id, {})
        hist_1000_item = hist_1000.get(q_id, {})
        base_pass = hist_400_item.get("base", {}).get("correct", None)
        auto_pass = hist_400_item.get("autonomous", {}).get(
            "correct", hist_1000_item.get("correct", None)
        )

        rec = {
            "idx": idx,
            "id": q_id,
            "benchmark": bench,
            "pillar": pillar,
            "prompt": prompt,
            "target": target,
            "correct": bool(passed),
            "tokens": res["gen_tokens"],
            "latency": round(res["latency_sec"], 4),
            "tps": round(res["tokens_per_sec"], 2),
            "clean_output": res["clean_text"][:300],
            "raw_output": res["raw_text"][:300],
            "veracity_conflict_score": round(float(telem.get("veracity_conflict_score", 0.0)), 4),
            "central_router_cycles": int(telem.get("central_router_cycles", 2)),
            "spiking_firing_rate": round(float(telem.get("spiking_firing_rate", 0.0)), 4),
            "base_correct": base_pass,
            "autonomous_correct": auto_pass,
        }
        records.append(rec)
        completed_ids.add(q_id)

        mark = "[PASS ✓]" if passed else "[FAIL ✗]"
        v_score = rec["veracity_conflict_score"]
        snippet = res["clean_text"].replace("\n", " ")[:60]
        log_line = (
            f"[R{run_idx}] Q{idx:04d}/{len(suite):04d} | {bench:<10} | {q_id:<12} | {mark} | "
            f"Veracity: {v_score:+6.3f} | Lat: {res['latency_sec']:.2f}s ({res['tokens_per_sec']:4.1f} t/s) | "
            f"Out: {snippet}"
        )
        if not passed or idx % 20 == 0 or idx == len(suite):
            print(log_line, flush=True)
            if not passed:
                print(f"      [FAIL DETAIL] Target: {target} | Clean: {res['clean_text'][:140]!r}", flush=True)
        log_file.write(log_line + "\n")
        if not passed:
            log_file.write(f"      [FAIL DETAIL] Target: {target} | Clean: {res['clean_text'][:180]!r}\n")
        log_file.flush()

        if len(records) % 20 == 0 or len(records) == len(suite):
            with open(str(checkpoint_path), "w", encoding="utf-8") as f:
                json.dump({"records": records, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    log_file.close()

    # Aggregate benchmark breakdown
    bench_stats = defaultdict(lambda: {"correct": 0, "total": 0, "tokens": 0, "lat": 0.0, "pillar": ""})
    pillar_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        b = r["benchmark"]
        p = r["pillar"]
        bench_stats[b]["pillar"] = p
        bench_stats[b]["total"] += 1
        bench_stats[b]["tokens"] += r["tokens"]
        bench_stats[b]["lat"] += r["latency"]
        pillar_stats[p]["total"] += 1
        if r["correct"]:
            bench_stats[b]["correct"] += 1
            pillar_stats[p]["correct"] += 1

    total_q = len(records)
    total_corr = sum(1 for r in records if r["correct"])
    total_tok = sum(r["tokens"] for r in records)
    total_lat = sum(r["latency"] for r in records)
    overall_acc = (total_corr / max(1, total_q)) * 100.0
    overall_tps = total_tok / max(0.001, total_lat)

    print(f"\n{'='*90}")
    print(f"RUN {run_idx}/{total_runs} SUMMARY ({total_corr}/{total_q} = {overall_acc:.2f}% | {overall_tps:.2f} tok/s)")
    print(f"{'='*90}")
    print(f"{'Benchmark':<12} | {'Pillar':<22} | {'Score':<10} | {'Accuracy':<10} | {'Avg Lat (s)':<12}")
    print("-" * 90)
    bench_breakdown = []
    for b in BENCH_ORDER:
        if b not in bench_stats:
            continue
        s = bench_stats[b]
        acc = (s["correct"] / max(1, s["total"])) * 100.0
        avg_lat = s["lat"] / max(1, s["total"])
        bench_breakdown.append({
            "benchmark": b,
            "pillar": s["pillar"],
            "correct": s["correct"],
            "total": s["total"],
            "score": f"{s['correct']}/{s['total']}",
            "accuracy_pct": round(acc, 2),
            "avg_latency_sec": round(avg_lat, 3),
        })
        print(f"{b:<12} | {s['pillar']:<22} | {s['correct']:>2}/{s['total']:<7} | {acc:6.2f}%    | {avg_lat:6.3f}s")
    print("-" * 90)
    print(f"{'OVERALL':<12} | {'All Pillars':<22} | {total_corr:>3}/{total_q:<6} | {overall_acc:6.2f}%    | {total_lat/max(1,total_q):6.3f}s\n")

    pillar_breakdown = {}
    for p, ps in pillar_stats.items():
        pillar_breakdown[p] = {
            "correct": ps["correct"],
            "total": ps["total"],
            "accuracy_pct": round((ps["correct"] / max(1, ps["total"])) * 100.0, 2),
        }

    return {
        "run_idx": run_idx,
        "total_questions": total_q,
        "correct_count": total_corr,
        "accuracy_pct": round(overall_acc, 2),
        "total_tokens": total_tok,
        "total_latency_sec": round(total_lat, 2),
        "tokens_per_sec": round(overall_tps, 2),
        "avg_latency_per_question_sec": round(total_lat / max(1, total_q), 3),
        "benchmark_breakdown": bench_breakdown,
        "pillar_breakdown": pillar_breakdown,
        "details": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HADL v5.8 Latent Matrix & Central Router GPU Benchmark")
    parser.add_argument("--questions-per-bench", type=int, default=20, help="Questions per benchmark (20 or 50)")
    parser.add_argument("--runs", type=int, default=1, help="Number of full evaluation runs")
    parser.add_argument("--benchmarks", type=str, default="", help="Optional comma-separated benchmark filter")
    parser.add_argument("--output", type=str, default="", help="Output JSON path")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoint and start fresh")
    args = parser.parse_args()

    filter_benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()] if args.benchmarks else None
    suite = build_benchmark_suite(questions_per_bench=args.questions_per_bench, filter_benchmarks=filter_benches)
    hist_400, hist_1000 = load_historical_baselines()

    total_q = len(suite)
    default_out = PROJECT_ROOT / "eval_results" / f"benchmark_{total_q}q_hadl58_latent_matrix_audit.json"
    out_path = Path(args.output) if args.output else default_out

    print("=" * 95)
    print(f"HADL v5.8 NON-TOKEN LATENT MATRIX & CENTRAL ROUTER ('OTAK BERPUTAR') GPU BENCHMARK")
    print(f"Questions/Bench: {args.questions_per_bench} | Total Questions/Run: {total_q} | Runs: {args.runs}")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 95)

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), trust_remote_code=True)
    t0_load = time.perf_counter()
    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    if ADAPTER_DIR.exists():
        lora_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR), is_trainable=False)
        lora_model.eval()
    else:
        lora_model = base_model

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
        enable_directional_safety=True,
    )
    model.tokenizer = tokenizer
    model.eval()
    print(f"[+] Loaded HADL v5.8 model in {time.perf_counter() - t0_load:.2f}s.")

    all_runs: List[Dict[str, Any]] = []
    for r_idx in range(1, args.runs + 1):
        ckpt_path = out_path.with_name(f"{out_path.stem}_run{r_idx}_ckpt.json")
        log_path = out_path.with_name(f"{out_path.stem}_run{r_idx}_log.txt")
        run_res = evaluate_single_run(
            model=model,
            tokenizer=tokenizer,
            suite=suite,
            run_idx=r_idx,
            total_runs=args.runs,
            hist_400=hist_400,
            hist_1000=hist_1000,
            checkpoint_path=ckpt_path,
            log_path=log_path,
            fresh=args.fresh,
        )
        all_runs.append(run_res)

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "HADL v5.8 Non-Token Latent Matrix & Recurrent Central Cognitive Router (Otak Berputar)",
        "hardware": {
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "questions_per_benchmark": args.questions_per_bench,
        "total_questions_per_run": total_q,
        "num_runs": args.runs,
        "mean_accuracy_pct": round(sum(r["accuracy_pct"] for r in all_runs) / max(1, len(all_runs)), 2),
        "mean_tokens_per_sec": round(sum(r["tokens_per_sec"] for r in all_runs) / max(1, len(all_runs)), 2),
        "runs": all_runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[+] Saved complete benchmark audit JSON to: {out_path}")


if __name__ == "__main__":
    main()
