import json

d = json.load(open("eval_results/benchmark_400q_20parts_real_audit.json", encoding="utf-8"))
wg = [x for x in d["details"] if x["benchmark"] == "WinoGrande"]
for x in wg:
    t = x["target"]
    b_out = repr(x["base"]["clean_output"][:60])
    b_pass = x["base"]["correct"]
    print(f"[{x['id']}] Target: {t:<15} | Base Pass: {str(b_pass):<5} | Base Out: {b_out}")
