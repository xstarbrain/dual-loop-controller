import zipfile
import hashlib
import json
from pathlib import Path

# Load BF_FILES from gemma-4-measure-before-you-tune.ipynb
with open(r'C:\Users\Matthew Chen\Downloads\gemma-4-measure-before-you-tune.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][25]['source'])

import re
bf_match = re.search(r'BF_FILES\s*=\s*\{.*?\"prompts/system\.md\":\s*\"(.*?)\"\s*\}', src, re.DOTALL)
bf_prompt = bf_match.group(1).encode('utf-8').decode('unicode_escape')

# Extract exact configs
agent_yaml = """name: swe_agent
model: gemma-4-31b-it-qat-w4a16-ct
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
generate_content_config: !include configs/sampling.yaml
"""

sampling_yaml = """temperature: 0.2
top_p: 0.95
max_output_tokens: 8192
thinking_config:
  thinking_budget: 4096
  include_thoughts: true
"""

eval_config_yaml = """evaluation:
  timeout_seconds: 240
  max_tool_calls: 28
  max_time_minutes: 8
  max_turns: 80
"""

# 1. Pure 0.18 Bundle
bf18_files = {
    "agent.yaml": agent_yaml,
    "configs/sampling.yaml": sampling_yaml,
    "eval_config.yaml": eval_config_yaml,
    "prompts/system.md": bf_prompt,
}

def build_zip(files, zip_name):
    p = Path(zip_name)
    p.unlink(missing_ok=True)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in sorted(files.items()):
            z.writestr(name, content.strip() + "\n")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    print(f"[+] Built {zip_name}: {p.stat().st_size:,} bytes | SHA-256: {digest}")
    return digest

print("Building pure 0.18 anchor...")
bf_sha = build_zip(bf18_files, "bf18_submission.zip")
