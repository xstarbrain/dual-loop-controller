import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

with open("eval_results/benchmark_20parts_fresh_and_arc_agi_audit.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("=== IFEVAL ITEMS ===")
for item in d["detailed_results"]:
    if item.get("benchmark") == "IFEval":
        print(f"ID: {item['id']}")
        print(f"Prompt: {item['prompt']}")
        print(f"Target: {item['target']}")
        print(f"Base Score: {item.get('base_score')} | Response:\n{item.get('base_response')}")
        print(f"HADL Score: {item.get('hadl_v52_score')} | Response:\n{item.get('hadl_v52_response')}")
        print("-" * 60)

print("\n=== BBH ITEMS ===")
for item in d["detailed_results"]:
    if item.get("benchmark") == "BBH":
        print(f"ID: {item['id']}")
        print(f"Prompt: {item['prompt']}")
        print(f"Target: {item['target']}")
        print(f"Base Score: {item.get('base_score')} | Response:\n{item.get('base_response')}")
        print(f"HADL Score: {item.get('hadl_v52_score')} | Response:\n{item.get('hadl_v52_response')}")
        print("-" * 60)
