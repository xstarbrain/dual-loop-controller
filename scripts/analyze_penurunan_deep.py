#!/usr/bin/env python3
"""
Deep Technical & Mathematical Analysis of Benchmark Decrease
Compares HADL v5.5 (Latest) vs Autonomous Native Brain vs Base Frozen
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

audit_path = Path("eval_results/benchmark_200q_dual_stream_audit.json")
with open(audit_path, "r", encoding="utf-8") as f:
    audit = json.load(f)

latest = audit["models_evaluated"]["model_terbaru"]["details"]
auto = audit["models_evaluated"]["model_autonomous"]["details"]
base = audit["models_evaluated"]["model_dasar"]["details"]

auto_map = {d["id"]: d for d in auto}
base_map = {d["id"]: d for d in base}

print(f"Total questions: {len(latest)}")
latest_failures = [d for d in latest if not d["correct"]]
print(f"Total latest failures: {len(latest_failures)} / 200 (Score: {len(latest) - len(latest_failures)}/200, 71.0%)")

# Compare with autonomous: where did latest fail that auto passed?
regressions = [d for d in latest_failures if auto_map[d["id"]]["correct"]]
print(f"Total regressions vs Autonomous Native: {len(regressions)} questions")

# Categorize regressions by benchmark
reg_by_bench = Counter(d["benchmark"] for d in regressions)
print("\nRegressions by Benchmark (where Autonomous passed, but Latest failed):")
for b, c in reg_by_bench.most_common():
    print(f"  {b:<12}: {c} questions")

print("\n=== REGRESSIONS VS BASE FROZEN MODEL ===")
reg_vs_base = [d for d in latest_failures if base_map[d["id"]]["correct"]]
print(f"Total regressions vs Base: {len(reg_vs_base)}")

# Group regressions vs base into patterns
patterns = defaultdict(list)
for d in reg_vs_base:
    b = base_map[d["id"]]
    q_id = d["id"]
    bench = d["benchmark"]
    target = d["target"]
    raw = d["raw_output"]
    toks = d["generated_tokens"]
    b_toks = b["generated_tokens"]
    b_out = b["clean_output"]

    if "<think>" in raw and "</think>" not in raw:
        pat = "Think_Truncation (Model thought inside <think> and hit max_tokens)"
    elif toks <= 6 and "<|endoftext|>" in raw:
        pat = "Premature_EOS (Emitted EOS in <=6 tokens without finishing)"
    elif "def " in raw and bench in ["HumanEval", "CodeDebug"]:
        pat = "Coding_Format_Mismatch (Generated full function instead of one-line body)"
    else:
        pat = "Logic_Divergence (Chose different token than base)"

    patterns[pat].append({
        "bench": bench,
        "id": q_id,
        "target": target,
        "base_toks": b_toks,
        "base_out": b_out[:70],
        "late_toks": toks,
        "late_out": raw[:70]
    })

print("\n=== CODING & SOFTWARE COMPARISON (HumanEval & CodeDebug) ===")
for b_name in ["HumanEval", "CodeDebug"]:
    items_l = [d for d in latest if d["benchmark"] == b_name]
    items_a = [d for d in auto if d["benchmark"] == b_name]
    items_b = [d for d in base if d["benchmark"] == b_name]
    print(f"\n--- Benchmark: {b_name} ---")
    for dl, da, db in zip(items_l[:5], items_a[:5], items_b[:5]):
        print(f"[{dl['id']}] Target: {dl['target']} | Base: {db['correct']} | Auto: {da['correct']} | Late: {dl['correct']}")
        print(f"   Prompt: {repr(dl['prompt'][:65])}")
        print(f"   Base  : {repr(db['clean_output'][:70])}")
        print(f"   Auto  : {repr(da['clean_output'][:70])}")
        print(f"   Late  : {repr(dl['clean_output'][:70])}")




