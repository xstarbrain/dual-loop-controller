import json
import os

notebooks = [
    r"C:\Users\Matthew Chen\Downloads\gemma-4-superagent.ipynb",
    r"C:\Users\Matthew Chen\Downloads\gemma-eda-baseline-for-a-start-lb-top-1.ipynb",
    r"C:\Users\Matthew Chen\Downloads\black-cat-swe-agent-pack-instinct.ipynb",
    r"C:\Users\Matthew Chen\Downloads\black-cat-swe-agent-second-strike.ipynb",
    r"C:\Users\Matthew Chen\Downloads\pathfinder-gemma-4-agent-eda-baseline.ipynb"
]

out = []

for path in notebooks:
    if not os.path.exists(path):
        continue
    out.append(f"\n==================================================")
    out.append(f"FILE: {os.path.basename(path)} ({os.path.getsize(path)} bytes)")
    out.append(f"==================================================")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        for i, cell in enumerate(nb.get('cells', [])):
            src = ''.join(cell.get('source', []))
            src_lower = src.lower()
            if any(k in src_lower for k in ['eval_config', 'agent.yaml', 'sampling.yaml', 'prompts/', 'system.md', 'submission.zip', 'score', 'benchmark', 'lb', 'top']):
                ctype = cell.get('cell_type')
                out.append(f"--- Cell {i} [{ctype}] ---")
                lines = src.splitlines()
                for line in lines[:25]:
                    out.append(f"  {line}")
                if len(lines) > 25:
                    out.append(f"   ... ({len(lines)-25} more lines)")
    except Exception as e:
        out.append(f"Error reading {path}: {e}")

with open("eval_results/notebook_inspection.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))

print(f"Inspection written to eval_results/notebook_inspection.txt, total chars: {sum(len(x) for x in out)}")
