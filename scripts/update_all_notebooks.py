import json
import zipfile
import hashlib
import sys
from pathlib import Path

scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts_dir))
import build_champion_bundle as champ

# 1. Update getting_started_gemma_4_trace_synthesis_018.ipynb
nb1_path = scripts_dir.parent / "getting_started_gemma_4_trace_synthesis_018.ipynb"
notebook1 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🧠 Gemma 4 Autonomous Developer Agent (Championship Calibrated)\n",
                "### Single-Agent Budget-Fit Architecture | Kaggle Competition\n",
                "\n",
                "This notebook builds the verified **`submission.zip`** combining the proven foundations of **Budget-Fit Single Agent (0.18 LB)**, **SuperAgent (0.17 LB)**, and empirical paired run measurements.\n",
                "\n",
                "| Setting | Value | Rationale |\n",
                "|---|---|---|\n",
                "| **Time per task** | **8 min** | Allows Gemma 4 time to deliberate, edit, and verify |\n",
                "| **Tool calls per task** | **28** | Strict budget stopping wander loops |\n",
                "| **Edit deadline** | **Call 12** | Runs without source edit by call 12 fail 87% of the time |\n",
                "| **max_output_tokens** | **8192** | Full headroom for 4096 thinking tokens + tool call without truncation |\n",
                "| **thinking_budget** | **4096** | Full deep reasoning capacity for complex code issues |\n",
                "| **timeout_seconds** | **240 s** | Prevents verifier test timeouts on heavy test suites like requests |\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import zipfile\n",
                "import hashlib\n",
                "from pathlib import Path\n",
                "\n",
                "OUT_DIR = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()\n",
                "ZIP_PATH = OUT_DIR / 'submission.zip'\n",
                "\n",
                f'SYSTEM_PROMPT = """{champ.system_md.strip()}"""\n',
                "\n",
                "FILES = {\n",
                f'  "agent.yaml": """{champ.agent_yaml.strip()}\\n""",\n',
                f'  "configs/sampling.yaml": """{champ.sampling_yaml.strip()}\\n""",\n',
                f'  "eval_config.yaml": """{champ.eval_config_yaml.strip()}\\n""",\n',
                '  "prompts/system.md": SYSTEM_PROMPT.strip() + "\\n"\n',
                "}\n",
                "\n",
                "ZIP_PATH.unlink(missing_ok=True)\n",
                "with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:\n",
                "    for name in sorted(FILES):\n",
                "        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))\n",
                "        info.create_system = 3\n",
                "        info.external_attr = 0o644 << 16\n",
                "        zf.writestr(info, FILES[name].strip().encode('utf-8') + b'\\n', compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)\n",
                "\n",
                "with zipfile.ZipFile(ZIP_PATH) as zf:\n",
                "    assert zf.testzip() is None\n",
                "    namelist = zf.namelist()\n",
                "    print(f'[+] Successfully packaged submission.zip ({ZIP_PATH.stat().st_size:,} bytes):')\n",
                "    for name in namelist:\n",
                "        print(f'     - {name} ({zf.getinfo(name).file_size:,} bytes)')\n",
                "\n",
                "sha256 = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()\n",
                "print(f'[+] SHA-256 Digest: {sha256}')\n"
            ]
        }
    ],
    "metadata": {"language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 4
}
with open(nb1_path, "w", encoding="utf-8") as f:
    json.dump(notebook1, f, indent=1)
print(f"[+] Successfully updated {nb1_path.name}")

# 2. Update getting_started_gemma_4_hadl_dualloop.ipynb
nb2_path = scripts_dir.parent / "getting_started_gemma_4_hadl_dualloop.ipynb"
with open(nb2_path, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

cell4_code = f"""# Clean and create agent directory
if AGENT_DIR.exists():
    shutil.rmtree(AGENT_DIR, ignore_errors=True)
AGENT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = \"\"\"{champ.system_md.strip()}\"\"\"

AGENT_FILES = {{
  "agent.yaml": \"\"\"{champ.agent_yaml.strip()}\\n\"\"\",
  "configs/sampling.yaml": \"\"\"{champ.sampling_yaml.strip()}\\n\"\"\",
  "eval_config.yaml": \"\"\"{champ.eval_config_yaml.strip()}\\n\"\"\",
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
nb2["cells"][4]["source"] = [l + "\n" for l in cell4_code.splitlines()]

with open(nb2_path, "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=1)
print(f"[+] Successfully updated {nb2_path.name}")
