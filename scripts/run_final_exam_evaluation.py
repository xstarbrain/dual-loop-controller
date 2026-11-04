import json
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from frontier_study.run_frontier_exam import evaluate_exam

EVAL_DIR = Path("eval_results")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("[*] EXECUTING RE-EVALUATION ON EXAM POOL ROUND 1 (ORIGINAL 50 QUESTIONS)")
print("=" * 80)
res1 = evaluate_exam("frontier_study/exam_pool_round1.json", use_adapter=True)
print(f"\n[+] FINAL EXAM 1 SCORE: {res1['total_score']:.1f}/100.0 ({res1['score_percentage']}%)")
for dom, d in res1['domain_scores'].items():
    pct = (d['scored'] / d['total']) * 100
    print(f"    - {dom:40s}: {d['scored']:.1f}/{d['total']:.1f} ({pct:.1f}%)")

with open(EVAL_DIR / "frontier_exam_round1_remedial.json", "w", encoding="utf-8") as f:
    json.dump(res1, f, indent=2)

print("\n" + "=" * 80)
print("[*] EXECUTING GENERALIZATION EVALUATION ON EXAM POOL ROUND 2 (NEW 50 QUESTIONS)")
print("=" * 80)
res2 = evaluate_exam("frontier_study/exam_pool_round2.json", use_adapter=True)
print(f"\n[+] FINAL EXAM 2 SCORE: {res2['total_score']:.1f}/100.0 ({res2['score_percentage']}%)")
for dom, d in res2['domain_scores'].items():
    pct = (d['scored'] / d['total']) * 100
    print(f"    - {dom:40s}: {d['scored']:.1f}/{d['total']:.1f} ({pct:.1f}%)")

with open(EVAL_DIR / "frontier_exam_round2.json", "w", encoding="utf-8") as f:
    json.dump(res2, f, indent=2)

print("\n[+] Both exam runs successfully recorded in eval_results/")
