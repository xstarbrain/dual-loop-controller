import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Matthew Chen\Downloads\gemma-4-measure-before-you-tune.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if any(k in src for k in ['0.24', '0.23', '0.22', '0.21', '0.20']):
        print(f"*** CELL {i} ({cell.get('cell_type')}) ***")
        lines = [l for l in src.splitlines() if any(k in l for k in ['0.24', '0.23', '0.22', '0.21', '0.20', 'score', 'rate', 'top', 'pass'])]
        for l in lines[:10]:
            print("  ", l[:120])
