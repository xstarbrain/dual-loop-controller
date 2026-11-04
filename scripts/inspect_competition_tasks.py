import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('competition/tasks.jsonl', 'r', encoding='utf-8') as f:
    tasks = [json.loads(l) for l in f if l.strip()]

print(f"Total tasks in competition/tasks.jsonl: {len(tasks)}\n")

for i in [0, 5, 15, 30, 50, 70, 90, 110, 125]:
    t = tasks[i]
    iid = t.get('instance_id')
    repo = t.get('repo')
    print(f"==================================================")
    print(f"Task #{i}: {iid} | Repo: {repo} | Base Commit: {t.get('base_commit')[:8]}")
    print(f"==================================================")
    print("Problem Statement Excerpt:")
    print("  " + t.get('problem_statement', '')[:250].replace('\n', ' '))
    print("\nGold Patch Diff Files:")
    for line in t.get('patch', '').splitlines():
        if line.startswith('diff --git'):
            print("  " + line)
    print("\nTest Patch Diff Files:")
    for line in t.get('test_patch', '').splitlines():
        if line.startswith('diff --git'):
            print("  " + line)
    print()
