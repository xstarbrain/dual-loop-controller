import json

with open('eval_results/benchmark_400q_hadl56_audit.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

bench_comp = {}
for r in data['details']:
    b = r['benchmark']
    if b not in bench_comp:
        bench_comp[b] = {'v5_correct': 0, 'v56_correct': 0, 'lost': []}
    if r['hadl_v5']['correct']:
        bench_comp[b]['v5_correct'] += 1
    if r['hadl_v56']['correct']:
        bench_comp[b]['v56_correct'] += 1
    if r['hadl_v5']['correct'] and not r['hadl_v56']['correct']:
        bench_comp[b]['lost'].append((r['id'], r['prompt'], r['target'], r['hadl_v56']['clean_output'], r['hadl_v5']['clean_output']))

print("=" * 80)
print(f"{'Benchmark':<12} | {'HADL v5.0':<10} | {'HADL v5.6':<10} | {'Delta':<8} | Status")
print("=" * 80)
total_v5 = 0
total_v56 = 0
for b, s in bench_comp.items():
    total_v5 += s['v5_correct']
    total_v56 += s['v56_correct']
    delta = s['v56_correct'] - s['v5_correct']
    tag = "UP (+)" if delta > 0 else ("DOWN (-)" if delta < 0 else "EQUAL (=)")
    print(f"{b:<12} | {s['v5_correct']:>2}/20 ({s['v5_correct']*5:3d}%) | {s['v56_correct']:>2}/20 ({s['v56_correct']*5:3d}%) | {delta:>+3}     | {tag}")

print("=" * 80)
print(f"{'TOTAL':<12} | {total_v5}/400 ({total_v5/4:.2f}%) | {total_v56}/400 ({total_v56/4:.2f}%) | {total_v56-total_v5:>+3}")
print("=" * 80)

print("\n\n=== DEEP DIVE: WHY DID V5.6 LOSE QUESTIONS WHERE V5.0 WON? ===")
for b, s in bench_comp.items():
    if s['lost']:
        print(f"\n--- Benchmark: {b} (Lost {len(s['lost'])} questions vs v5.0) ---")
        for q_id, prompt, target, v56_out, v5_out in s['lost'][:3]:
            print(f"[{q_id}] Target: {target}")
            print(f"  Prompt : {prompt[:90]}...")
            print(f"  v5.0   : {repr(v5_out[:80])}")
            print(f"  v5.6   : {repr(v56_out[:80])}")
