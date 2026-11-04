import json

d = json.load(open('benchmarks/benchmark_16q_3runs_multimodal.json', 'r', encoding='utf-8'))
for i, r in enumerate(d['runs_detailed']):
    print(f"=== RUN {i+1} (Passed: {r['overall_passed']}/{r['total_questions']}) ===")
    for item in r['items']:
        if not item['overall_passed']:
            print(f"  FAIL: [{item['id']}] Bench={item['benchmark_passed']}, Popper={item['popperian_verified']}")
            print(f"    Prompt: {item['prompt']}")
            print(f"    Response: {repr(item['response'])}")
            print("-" * 50)
