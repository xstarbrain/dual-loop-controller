import sys
import json
import zipfile
import hashlib
import yaml
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Matthew Chen\Downloads\gemma-4-measure-before-you-tune.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][25]['source'])

bf_part = src[src.find('BF_FILES ='):src.find('# Submitted bundle:')]
print("=== BF PART CODE ===")
print(bf_part[-400:])

def build_archive(files, name, expected):
    path = Path(name)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for member in sorted(files):
            assert not member.startswith(('/', '\\')) and '..' not in Path(member).parts
            info = zipfile.ZipInfo(member, date_time=(1980,1,1,0,0,0))
            info.create_system = 3; info.external_attr = 0o644 << 16
            zf.writestr(info, files[member].encode('utf-8'), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == expected, f"Mismatch: {digest} != {expected}"
    return path, digest

g = {'build_archive': build_archive, 'json': json, 'hashlib': hashlib, 'yaml': yaml, 'Path': Path}
exec(bf_part, g)
print("SUCCESS! bf_digest:", g['bf_digest'])
print("Matches BF_SHA256:", g['bf_digest'] == g['BF_SHA256'])
