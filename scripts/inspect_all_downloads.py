import json
import os
import glob

notebooks = [
    r"C:\Users\Matthew Chen\Downloads\gemma-4-superagent.ipynb",
    r"C:\Users\Matthew Chen\Downloads\gemma-eda-baseline-for-a-start-lb-top-1.ipynb",
    r"C:\Users\Matthew Chen\Downloads\black-cat-swe-agent-pack-instinct.ipynb",
    r"C:\Users\Matthew Chen\Downloads\black-cat-swe-agent-second-strike.ipynb",
    r"C:\Users\Matthew Chen\Downloads\pathfinder-gemma-4-agent-eda-baseline.ipynb",
    r"C:\Users\Matthew Chen\Downloads\gemma-4-measure-before-you-tune.ipynb",
    r"C:\Users\Matthew Chen\Downloads\gemma-4-agent-budget-fit-single-agent.ipynb"
]

for path in notebooks:
    if not os.path.exists(path):
        continue
    print(f"\n==================================================")
    print(f"FILE: {os.path.basename(path)} ({os.path.getsize(path)} bytes)")
    print(f"==================================================")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        for i, cell in enumerate(nb.get('cells', [])):
            src = ''.join(cell.get('source', []))
            src_lower = src.lower()
            if any(k in src_lower for k in ['eval_config', 'agent.yaml', 'sampling.yaml', 'system.md', 'submission.zip', '0.2', 'score', 'lb', 'top']):
                ctype = cell.get('cell_type')
                print(f"--- Cell {i} [{ctype}] ---")
                lines = src.splitlines()
                for line in lines[:10]:
                    print("  ", line)
                if len(lines) > 10:
                    print(f"   ... ({len(lines)-10} more lines)")
    except Exception as e:
        print(f"Error reading {path}: {e}")
