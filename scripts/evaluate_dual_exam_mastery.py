import json
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from frontier_study.run_frontier_exam import evaluate_exam

EVAL_DIR = Path("eval_results")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 88)
print("[*] EVALUATING EXAM POOL ROUND 2 (DIFFERENT REMEDIAL EXAM - 50 NEW QUESTIONS)")
print("=" * 88)

res2 = evaluate_exam("frontier_study/exam_pool_round2.json", use_adapter=True)
print(f"\n[+] REMEDIAL EXAM 2 FINAL SCORE: {res2['total_score']:.1f}/100.0 ({res2['score_percentage']}%)")
for dom, d in res2['domain_scores'].items():
    pct = (d['scored'] / d['total']) * 100
    print(f"    - {dom:40s}: {d['scored']:.1f}/{d['total']:.1f} ({pct:.1f}%)")

with open(EVAL_DIR / "frontier_exam_round2_final.json", "w", encoding="utf-8") as f:
    json.dump(res2, f, indent=2)

print("\n" + "=" * 88)
print("[*] VERIFYING EXAM POOL ROUND 1 RETENTION (ZERO FORGETTING AUDIT)")
print("=" * 88)

res1 = evaluate_exam("frontier_study/exam_pool_round1.json", use_adapter=True)
print(f"\n[+] EXAM 1 RETENTION SCORE: {res1['total_score']:.1f}/100.0 ({res1['score_percentage']}%)")
for dom, d in res1['domain_scores'].items():
    pct = (d['scored'] / d['total']) * 100
    print(f"    - {dom:40s}: {d['scored']:.1f}/{d['total']:.1f} ({pct:.1f}%)")

with open(EVAL_DIR / "frontier_exam_round1_retention.json", "w", encoding="utf-8") as f:
    json.dump(res1, f, indent=2)

print("\n[+] Both exams evaluated successfully.")
