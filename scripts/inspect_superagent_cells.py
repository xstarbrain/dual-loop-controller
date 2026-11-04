import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Matthew Chen\Downloads\gemma-4-superagent.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in range(35, 52):
    cell = nb['cells'][i]
    src = ''.join(cell['source'])
    print(f"*** CELL {i} ({cell.get('cell_type')}) ***")
    for l in src.splitlines()[:30]:
        print("  ", l[:120])
