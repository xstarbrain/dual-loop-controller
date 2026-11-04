import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Matthew Chen\Downloads\gemma-eda-baseline-for-a-start-lb-top-1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if any(k in src.lower() for k in ['prompt', 'bundle', 'eval_config', 'top-1', 'lb', 'score', 'submission']):
        print(f"=== Cell {i} ({cell.get('cell_type')}) ===")
        for l in src.splitlines()[:15]:
            print("  ", l[:120])
