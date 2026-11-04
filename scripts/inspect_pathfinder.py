import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Matthew Chen\Downloads\pathfinder-gemma-4-agent-eda-baseline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if any(k in src.lower() for k in ['prompt', 'rule', 'docs_src', 'eval_config', 'top', 'score']):
        print(f"=== Pathfinder Cell {i} ({cell.get('cell_type')}) ===")
        for l in src.splitlines()[:20]:
            print("  ", l[:120])
