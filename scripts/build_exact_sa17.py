import sys
import json
import zipfile
import hashlib
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Matthew Chen\Downloads\gemma-4-measure-before-you-tune.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

full_text = ''.join(nb['cells'][25]['source'])

start_idx = full_text.find('SA_FILES = {')
end_idx = full_text.find('SA_SHA256 =')
sa_code = full_text[start_idx:end_idx]

loc = {}
exec(sa_code, loc)
SA_FILES = loc['SA_FILES']

print('Extracted SA_FILES:')
for k, v in sorted(SA_FILES.items()):
    print(f'  {k}: {len(v)} chars')

path = Path('sa17_submission.zip')
with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for member in sorted(SA_FILES):
        info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o644 << 16
        zf.writestr(info, SA_FILES[member].encode('utf-8'), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

digest = hashlib.sha256(path.read_bytes()).hexdigest()
expected = "8dfd2bf8ddfbcc84182c024358a55fd3556fbc89345b4688f383d93f5b9c10b1"
print(f'Generated sa17_submission.zip SHA256: {digest}')
print(f'Expected SHA256:                      {expected}')
print(f'PERFECT BYTE-IDENTICAL MATCH:        {digest == expected}')
assert digest == expected, "SHA-256 mismatch!"
