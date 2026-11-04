import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Matthew Chen\Downloads\gemma-gemini-the-ultimate-duo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    print(f"*** CELL {i} ({cell.get('cell_type')}) ***")
    for l in src.splitlines()[:25]:
        print("  ", l[:120])
