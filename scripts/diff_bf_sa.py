import sys
import json
import difflib

sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Matthew Chen\Downloads\gemma-4-measure-before-you-tune.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][25]['source'])

import re
bf_match = re.search(r'BF_FILES\s*=\s*\{.*?\"prompts/system\.md\":\s*\"(.*?)\"\s*\}', src, re.DOTALL)
sa_match = re.search(r'SA_FILES\s*=\s*\{.*?\"prompts/system\.md\":\s*\"(.*?)\"\s*\}', src, re.DOTALL)

if bf_match and sa_match:
    bf_prompt = bf_match.group(1).encode('utf-8').decode('unicode_escape')
    sa_prompt = sa_match.group(1).encode('utf-8').decode('unicode_escape')
    print(f"BF_PROMPT (0.18): {len(bf_prompt)} chars, {len(bf_prompt.splitlines())} lines")
    print(f"SA_PROMPT (0.17): {len(sa_prompt)} chars, {len(sa_prompt.splitlines())} lines")
    
    with open('eval_results/bf_prompt.md', 'w', encoding='utf-8') as f:
        f.write(bf_prompt)
    with open('eval_results/sa_prompt.md', 'w', encoding='utf-8') as f:
        f.write(sa_prompt)
    print("Written to eval_results/bf_prompt.md and eval_results/sa_prompt.md")
else:
    print("Could not regex match prompts")
