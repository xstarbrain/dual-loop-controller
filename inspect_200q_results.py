import sys
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

with open("eval_results/benchmark_200q_dual_stream_audit.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("=" * 90)
print(f"{'Benchmark':<14} | {'Base Frozen':<12} | {'HADL v4.5':<12} | {'Autonomous':<12} | {'Terbaru (v5.6)':<14} | {'Δ vs Auto':<10}")
print("=" * 90)

for b, comp in d.get("benchmark_comparison", {}).items():
    diff = comp["latest"] - comp["auto"]
    diff_str = f"+{diff}" if diff > 0 else str(diff)
    print(f"{b:<14} | {comp['base']:2d}/10        | {comp['prev']:2d}/10        | {comp['auto']:2d}/10        | {comp['latest']:2d}/10          | {diff_str:<10}")

print("=" * 90)
print("\nPILLAR SUMMARY:")
for p, s in d.get("pillar_summary", {}).items():
    print(f"  * {p:<24}: Base {s['base']} | Auto {s['auto']} | Terbaru {s['latest']}")

# Check failed questions in Terbaru
failed_by_bench = {}
for item in d["models_evaluated"]["model_terbaru"]["details"]:
    if not item["correct"]:
        b = item["benchmark"]
        if b not in failed_by_bench:
            failed_by_bench[b] = []
        failed_by_bench[b].append({
            "id": item["id"],
            "prompt": item["prompt"][:80],
            "target": item["target"],
            "clean_output": item["clean_output"][:100],
            "tokens": item["generated_tokens"]
        })

print("\nFAILURES IN TERBARU (TOTAL = 30):")
for b, fails in failed_by_bench.items():
    print(f"\n--- {b} ({len(fails)} fails) ---")
    for f_item in fails:
        print(f"  [{f_item['id']}] Target: {f_item['target']} | Output: {f_item['clean_output']} (Tokens: {f_item['tokens']})")
