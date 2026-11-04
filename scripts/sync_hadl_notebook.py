import json
import sys
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts_dir))
import create_trace_synthesis_notebook as master

nb_path = scripts_dir.parent / "getting_started_gemma_4_hadl_dualloop.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell4_code = f"""# Clean and create agent directory
if AGENT_DIR.exists():
    shutil.rmtree(AGENT_DIR, ignore_errors=True)
AGENT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = \"\"\"{master.system_md.strip()}\"\"\"

AGENT_FILES = {{
  "agent.yaml": \"\"\"{master.agent_yaml.strip()}\\n\"\"\",
  "configs/sampling.yaml": \"\"\"{master.sampling_yaml.strip()}\\n\"\"\",
  "eval_config.yaml": \"\"\"{master.eval_config_yaml.strip()}\\n\"\"\",
  "prompts/system.md": SYSTEM_PROMPT.strip() + "\\n",
}}

# Write all agent files
for rel_path, content in sorted(AGENT_FILES.items()):
    target_file = AGENT_DIR / rel_path
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(content.strip() + '\\n', encoding='utf-8')

print(f'[+] Materialized HADL Dual-Loop Unified Agent in: {{AGENT_DIR}}')
print(f'    Total files written: {{len(AGENT_FILES)}}')
for f in sorted(AGENT_FILES.keys()):
    print(f'     - {{f}}')

# Safe tasks loading (handles hidden dataset variations without crashing)
TASKS_PATH = DATA_DIR / 'tasks.jsonl'
tasks = []
if TASKS_PATH.exists():
    try:
        from swegemma.models import load_tasks
        tasks = load_tasks(TASKS_PATH)
    except Exception:
        with open(TASKS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    tasks.append(type('Task', (), d))
print(f'[+] Loaded {{len(tasks)}} benchmark tasks from {{TASKS_PATH.name}}')
"""

nb["cells"][4]["source"] = [line + "\n" for line in cell4_code.splitlines()]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("[+] Successfully synchronized getting_started_gemma_4_hadl_dualloop.ipynb with master configs!")
