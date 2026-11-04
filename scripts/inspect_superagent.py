import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Matthew Chen\Downloads\gemma-4-superagent.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    ctype = cell['cell_type']
    if any(k in src.lower() for k in ['bundle', 'prompt', 'sampling', 'eval_config', '0.2', '0.1', 'score', 'leaderboard', 'top', 'recipe']):
        print(f"=== Cell {i} ({ctype}) ===")
        lines = src.splitlines()
        for l in lines[:25]:
            print("  ", l[:120])
        if len(lines) > 25:
            print(f"   ... ({len(lines)-25} more lines)")
