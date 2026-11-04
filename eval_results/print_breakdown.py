import json

with open("eval_results/benchmark_400q_hadl56_audit.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("=" * 115)
print(f"{'Benchmark':<12} | {'Pillar':<22} | {'Base (2B)':<14} | {'HADL v5.0':<14} | {'Autonomous':<14} | {'HADL v5.6':<14} | {'Delta v5.6-Base'}")
print("-" * 115)

for s in d["benchmark_breakdown"]:
    b = s["benchmark"]
    p = s["pillar"]
    base_s = f"{s['base_score']} ({s['base_pct']:.1f}%)"
    v5_s = f"{s['v5_score']} ({s['hadl_v5_pct']:.1f}%)"
    auto_s = f"{s['auto_score']} ({s['autonomous_pct']:.1f}%)"
    v56_s = f"{s['v56_score']} ({s['hadl_v56_pct']:.1f}%)"
    delta = f"{s['delta_pct']:+.1f}%"
    print(f"{b:<12} | {p:<22} | {base_s:<14} | {v5_s:<14} | {auto_s:<14} | {v56_s:<14} | {delta}")

print("=" * 115)
print("PILLAR SUMMARY:")
for p, stats in d["pillar_summary"].items():
    print(f"[{p:<22}] Base: {stats['base_pct']:>5.1f}% | v5.0: {stats['hadl_v5_pct']:>5.1f}% | Auto: {stats['autonomous_pct']:>5.1f}% | v5.6: {stats['hadl_v56_pct']:>5.1f}% ({int(stats['hadl_v56_pct']*stats['total']/100)}/{stats['total']})")

print("=" * 115)
ov = d["overall_summary"]
print(f"OVERALL 400Q: Base: {ov['base_correct_count']}/400 ({ov['base_accuracy_pct']}%) | v5.0: {ov['hadl_v5_correct_count']}/400 ({ov['hadl_v5_accuracy_pct']}%) | Auto: {ov['autonomous_correct_count']}/400 ({ov['autonomous_accuracy_pct']}%) | HADL v5.6: {ov['hadl_v56_correct_count']}/400 ({ov['hadl_v56_accuracy_pct']}%)")
print(f"Speed: Base TPS={d['speed_metrics']['base_tps']}, v5.0 TPS={d['speed_metrics']['hadl_v5_tps']}, Auto TPS={d['speed_metrics']['auto_tps']}, v5.6 TPS={d['speed_metrics']['hadl_v56_tps']}")
print(f"Latency: Base={d['speed_metrics']['base_lat']}s, v5.0={d['speed_metrics']['hadl_v5_lat']}s, Auto={d['speed_metrics']['auto_lat']}s, v5.6={d['speed_metrics']['hadl_v56_lat']}s")


print("\n" + "=" * 115)
print("DEEP DIVE: ALL 32 FAILURES IN HADL v5.6")
print("=" * 115)

for r in d["details"]:
    if not r["hadl_v56"]["correct"]:
        b = r["benchmark"]
        qid = r["id"]
        tgt = str(r["target"])[:40]
        out = r["hadl_v56"]["clean_output"].replace("\n", " ")[:70]
        base_p = "PASS" if r["base"]["correct"] else "FAIL"
        auto_p = "PASS" if r["autonomous"]["correct"] else "FAIL"
        print(f"[{b:<10}] {qid:<12} | Target: {tgt:<25} | Out: {out:<40} | Base: {base_p} | Auto: {auto_p}")
