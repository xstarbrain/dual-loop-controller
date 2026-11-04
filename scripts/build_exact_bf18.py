import sys
import json
import zipfile
import hashlib
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Matthew Chen\Downloads\gemma-4-measure-before-you-tune.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

full_text = ''.join(nb['cells'][25]['source'])

start_idx = full_text.find('BF_FILES = {')
end_idx = full_text.find('BF_SHA256 =')
bf_code = full_text[start_idx:end_idx]

loc = {}
exec(bf_code, loc)
BF_FILES = loc['BF_FILES']

print('Extracted BF_FILES:')
for k, v in sorted(BF_FILES.items()):
    print(f'  {k}: {len(v)} chars')

path = Path('bf18_submission.zip')
with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for member in sorted(BF_FILES):
        info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o644 << 16
        zf.writestr(info, BF_FILES[member].encode('utf-8'), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

digest = hashlib.sha256(path.read_bytes()).hexdigest()
expected = "eb3d5ebf21ce39b186d42937de593e3be62bc329024d59e8693e6c5ce1b20c63"
print(f'Generated bf18_submission.zip SHA256: {digest}')
print(f'Expected SHA256:                      {expected}')
print(f'PERFECT BYTE-IDENTICAL MATCH:        {digest == expected}')
assert digest == expected, "SHA-256 mismatch!"
