import json

d = json.load(open('eval_results/benchmark_400q_hadl56_audit.json', 'r', encoding='utf-8'))
print(f"{'No':<3} | {'Benchmark':<12} | {'Pillar':<22} | {'Base (20Q)':<12} | {'HADL v5.0':<12} | {'Autonomous':<12} | {'HADL v5.6':<12} | {'Delta':<8}")
print("-" * 110)
for i, b in enumerate(d.get('benchmark_breakdown', []), 1):
    print(f"{i:<3} | {b['benchmark']:<12} | {b['pillar']:<22} | {b['base_score']:<5} ({b['base_pct']:>5.1f}%) | {b['v5_score']:<5} ({b['hadl_v5_pct']:>5.1f}%) | {b['auto_score']:<5} ({b['autonomous_pct']:>5.1f}%) | {b['v56_score']:<5} ({b['hadl_v56_pct']:>5.1f}%) | {b['delta_pct']:>+5.1f}%")
print("-" * 110)
ov = d['overall_summary']
print(f"TOTAL: 400 QUESTIONS | Base: {ov['base_correct_count']}/400 ({ov['base_accuracy_pct']}%) | v5.0: {ov['hadl_v5_correct_count']}/400 ({ov['hadl_v5_accuracy_pct']}%) | Auto: {ov['autonomous_correct_count']}/400 ({ov['autonomous_accuracy_pct']}%) | v5.6: {ov['hadl_v56_correct_count']}/400 ({ov['hadl_v56_accuracy_pct']}%) | Delta vs Base: {ov['delta_v56_vs_base_pct']:+5.1f}%")
