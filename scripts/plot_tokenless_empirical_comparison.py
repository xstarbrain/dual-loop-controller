import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

audit_path = Path("eval_results/tokenless_latent_empirical_audit.json")
if not audit_path.exists():
    print("Audit path not found.")
    exit(0)

data = json.load(open(audit_path, encoding="utf-8"))
per_bench = data["per_benchmark"]

benchmarks = list(per_bench.keys())
base_acc = [per_bench[b]["baseline"]["correct"] / per_bench[b]["baseline"]["total"] * 100 for b in benchmarks]
tokenless_acc = [per_bench[b]["tokenless"]["correct"] / per_bench[b]["tokenless"]["total"] * 100 for b in benchmarks]

base_tokens = [per_bench[b]["baseline"]["tokens"] for b in benchmarks]
tokenless_tokens = [per_bench[b]["tokenless"]["tokens"] for b in benchmarks]

summary = data["summary"]
base_tot_acc = summary["baseline"]["accuracy_pct"]
tl_tot_acc = summary["tokenless_active"]["accuracy_pct"]
gain_acc = summary["deltas"]["accuracy_gain_pct"]
base_toks = summary["baseline"]["total_tokens"]
tl_toks = summary["tokenless_active"]["total_tokens"]
tok_sav = summary["deltas"]["token_savings_pct"]
spd_factor = summary["deltas"]["speedup_factor"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(benchmarks))
width = 0.35

# Chart 1: Accuracy
rects1 = ax1.bar(x - width/2, base_acc, width, label='Dual-Stream Baseline', color='#6c757d', alpha=0.85)
rects2 = ax1.bar(x + width/2, tokenless_acc, width, label='Tokenless Latent + IPv6 ALU', color='#28a745', alpha=0.95)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title(f'Empirical Accuracy Comparison (60 Critical Questions)\nNet Gain: {base_tot_acc}% -> {tl_tot_acc}% (+{gain_acc}%)', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(benchmarks, fontsize=11, fontweight='bold')
ax1.legend(loc='lower left', fontsize=11)
ax1.set_ylim(0, 115)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.0f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f'{h:.0f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Chart 2: Generated Token Count (Inference Footprint)
rects3 = ax2.bar(x - width/2, base_tokens, width, label='Dual-Stream Baseline', color='#dc3545', alpha=0.75)
rects4 = ax2.bar(x + width/2, tokenless_tokens, width, label='Tokenless Latent + IPv6 ALU', color='#17a2b8', alpha=0.95)
ax2.set_ylabel('Generated Tokens', fontsize=12, fontweight='bold')
ax2.set_title(f'Token Consumption & Bloat Elimination\nTotal Savings: {base_toks} -> {tl_toks} (-{tok_sav}% Tokens, {spd_factor}x Faster)', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(benchmarks, fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.3)

for rect in rects3:
    h = rect.get_height()
    ax2.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for rect in rects4:
    h = rect.get_height()
    ax2.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
out_png = Path("eval_results/tokenless_latent_empirical_comparison.png")
plt.savefig(out_png, dpi=300)
print(f"Saved comparison plot to {out_png}")

# Also copy to artifacts dir
art_dir = Path(r"C:\Users\Matthew Chen\.gemini\antigravity\brain\19bea55e-42a6-476a-af5b-9c25391e2be9")
if art_dir.exists():
    import shutil
    shutil.copy(out_png, art_dir / "tokenless_latent_empirical_comparison.png")
    print(f"Copied to artifact dir: {art_dir}")
