import json
import os
import sys

ckpt_path = "eval_results/benchmark_400q_hadl56_checkpoint.json"
if not os.path.exists(ckpt_path):
    print("Checkpoint file not found.")
    sys.exit(0)

try:
    with open(ckpt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    print(f"Error reading checkpoint: {e}")
    sys.exit(0)

recs = data.get("records", [])
n = len(recs)
if n == 0:
    print("Zero records in checkpoint.")
    sys.exit(0)

base_pass = sum(1 for r in recs if r.get("base", {}).get("correct"))
v50_pass = sum(1 for r in recs if r.get("hadl_v5", {}).get("correct"))
auto_pass = sum(1 for r in recs if r.get("autonomous", {}).get("correct"))
v56_pass = sum(1 for r in recs if r.get("hadl_v56", {}).get("correct"))

benchmarks = {}
for r in recs:
    b = r.get("benchmark", "Unknown")
    if b not in benchmarks:
        benchmarks[b] = {"total": 0, "base": 0, "v50": 0, "auto": 0, "v56": 0}
    benchmarks[b]["total"] += 1
    if r.get("base", {}).get("correct"): benchmarks[b]["base"] += 1
    if r.get("hadl_v5", {}).get("correct"): benchmarks[b]["v50"] += 1
    if r.get("autonomous", {}).get("correct"): benchmarks[b]["auto"] += 1
    if r.get("hadl_v56", {}).get("correct"): benchmarks[b]["v56"] += 1

print(f"Progress: {n}/400 ({n/400*100:.1f}%) | Latest: {recs[-1].get('id')} ({recs[-1].get('benchmark')})")
print(f"Accuracies across {n} questions:")
print(f"  - Base (Frozen)  : {base_pass}/{n} ({base_pass/n*100:.1f}%)")
print(f"  - HADL v5.0      : {v50_pass}/{n} ({v50_pass/n*100:.1f}%)")
print(f"  - Autonomous     : {auto_pass}/{n} ({auto_pass/n*100:.1f}%)")
print(f"  - HADL v5.6      : {v56_pass}/{n} ({v56_pass/n*100:.1f}%)")
print("-" * 65)
for b, stat in benchmarks.items():
    print(f"{b:15s} | Total: {stat['total']:2d} | Base: {stat['base']:2d} | v5.0: {stat['v50']:2d} | Auto: {stat['auto']:2d} | v5.6: {stat['v56']:2d}")
