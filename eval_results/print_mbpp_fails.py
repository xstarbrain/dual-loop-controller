import json

with open("eval_results/benchmark_400q_hadl56_audit.json", "r", encoding="utf-8") as f:
    d = json.load(f)

mbpp_fails = ["mbpp_02", "mbpp_05", "mbpp_07", "mbpp_09", "mbpp_13", "mbpp_16", "mbpp_18", "mbpp_19"]
for r in d["details"]:
    if r["id"] in mbpp_fails:
        print(f"=== {r['id']} ===")
        print(r["hadl_v56"]["clean_output"][:300])
        print()
