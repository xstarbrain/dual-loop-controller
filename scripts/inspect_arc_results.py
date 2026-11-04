import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

with open("eval_results/benchmark_20parts_fresh_and_arc_agi_audit.json", "r", encoding="utf-8") as f:
    d = json.load(f)

arc_items = [x for x in d["detailed_results"] if x["benchmark"] == "ARC-AGI-2"]
for i, item in enumerate(arc_items[:4]):
    print(f"==================================================")
    print(f"Task {i+1}: {item['task_id']} ({item['split']})")
    print(f"Target Solution: {item['solution_grid']}")
    print(f"Base Output:\n{item.get('base_arc_response')}")
    print(f"HADL v5.2 Output:\n{item.get('hadl_v52_arc_response')}")
    print(f"Boost L4 Output:\n{item.get('boost_arc_response')}")
