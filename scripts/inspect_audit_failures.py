import json

with open('eval_results/benchmark_400q_hadl56_audit.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('=== MBPP FAILURES IN HADL v5.6 ===')
for d in data['details']:
    if d['benchmark'] == 'MBPP' and not d['hadl_v56']['correct']:
        print(f"ID: {d['id']} | Target: {d['target']} | Base: {d['base']['correct']} | v56: {d['hadl_v56']['correct']}")
        print(f"  Base out: {repr(d['base']['clean_output'][:70])}")
        print(f"  v56 out:  {repr(d['hadl_v56']['clean_output'][:70])}")

print('\n=== WINOGRANDE FAILURES IN HADL v5.6 ===')
for d in data['details']:
    if d['benchmark'] == 'WinoGrande' and not d['hadl_v56']['correct']:
        print(f"ID: {d['id']} | Target: {d['target']} | Base: {d['base']['correct']} | v56: {d['hadl_v56']['correct']}")
        print(f"  Base out: {repr(d['base']['clean_output'][:70])}")
        print(f"  v56 out:  {repr(d['hadl_v56']['clean_output'][:70])}")
