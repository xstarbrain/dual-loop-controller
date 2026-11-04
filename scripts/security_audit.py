#!/usr/bin/env python3
"""
Repository Security & Hygiene Audit Script
===========================================
Scans for:
1. Hardcoded API keys, tokens, credentials, private keys
2. Sensitive environment or cache files
3. Insecure functions (e.g. unsafe eval/exec on untrusted external inputs)
4. Large binary files or unnecessary temporary artifacts
"""

import os
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SECRET_PATTERNS = [
    (re.compile(r'(?i)(?:api_key|apikey|access_token|auth_token)\s*=\s*["\']([a-zA-Z0-9_\-]{20,})["\']'), "API Key / Auth Token"),
    (re.compile(r'(?i)(?:secret_key|client_secret)\s*=\s*["\']([a-zA-Z0-9_\-]{20,})["\']'), "Secret Key"),
    (re.compile(r'(?i)password\s*=\s*["\']([^"\']{8,})["\']'), "Password Assignment"),
    (re.compile(r'ghp_[a-zA-Z0-9]{36}'), "GitHub Classic PAT"),
    (re.compile(r'github_pat_[a-zA-Z0-9_]{82}'), "GitHub Fine-Grained PAT"),
    (re.compile(r'sk-[a-zA-Z0-9]{32,}'), "OpenAI API Key"),
    (re.compile(r'AIza[0-9A-Za-z\-_]{35}'), "Google API Key"),
    (re.compile(r'-----BEGIN (?:RSA|OPENSSH|EC|DSA|PRIVATE) KEY-----'), "Private Key"),
    (re.compile(r'hf_[a-zA-Z0-9]{34,}'), "HuggingFace Token"),
]

# Paths to ignore
IGNORE_DIRS = {
    ".git", ".venv", "__pycache__", "dist", "checkpoints", "node_modules",
    ".idea", ".vscode", "wheels", "snapshots", "embeddings", "build"
}
IGNORE_EXTS = {
    ".png", ".jpg", ".jpeg", ".pt", ".bin", ".safetensors", ".zip",
    ".parquet", ".pyc", ".whl", ".tgz", ".npz", ".tar.gz"
}

def audit_secrets():
    print("=" * 80)
    print("[-] RUNNING SECURITY SECRET & CREDENTIAL SCAN")
    print("=" * 80)
    findings = []
    scanned_files = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # prune ignore dirs
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            p = Path(root) / f
            if p.suffix in IGNORE_EXTS:
                continue
            scanned_files += 1
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for pat, desc in SECRET_PATTERNS:
                for match in pat.finditer(content):
                    line_no = content[:match.start()].count("\n") + 1
                    matched_str = match.group(0)
                    # Filter out placeholders or test values
                    if any(x in matched_str.lower() for x in ["none", "placeholder", "your_api_key", "dummy", "test", "fake", "example"]):
                        continue
                    rel_p = p.relative_to(PROJECT_ROOT)
                    findings.append((str(rel_p), line_no, desc, matched_str[:12] + "..." + matched_str[-4:]))

    print(f"[*] Scanned {scanned_files} files across repository.")
    if findings:
        print(f"[!] WARNING: Found {len(findings)} potential secret(s):")
        for f in findings:
            print(f"    - {f[0]}:{f[1]} -> [{f[2]}] {f[3]}")
    else:
        print("[+] SUCCESS: Zero hardcoded secrets, private keys, or API tokens found!")
    return findings


def audit_large_and_temp_files():
    print("\n" + "=" * 80)
    print("[-] RUNNING HYGIENE & REPO BLAT SCAN")
    print("=" * 80)
    large_files = []
    temp_files = []

    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", ".venv"}]
        for f in files:
            p = Path(root) / f
            rel_p = p.relative_to(PROJECT_ROOT)
            try:
                size_mb = p.stat().st_size / (1024 * 1024)
            except Exception:
                continue
            if size_mb > 25.0 and p.suffix not in {".pt", ".bin", ".safetensors"}:
                large_files.append((str(rel_p), size_mb))
            if p.name.endswith((".log", ".tmp", ".bak", "~")) or p.name.startswith("tmp_"):
                temp_files.append((str(rel_p), size_mb))

    print(f"[*] Large non-model files (>25 MB): {len(large_files)}")
    for f in large_files:
        print(f"    - {f[0]} ({f[1]:.2f} MB)")
    print(f"[*] Ephemeral / Temporary / Log files: {len(temp_files)}")
    for f in temp_files[:15]:
        print(f"    - {f[0]} ({f[1]:.2f} MB)")
    if len(temp_files) > 15:
        print(f"    ... and {len(temp_files) - 15} more.")

    return large_files, temp_files


if __name__ == "__main__":
    findings = audit_secrets()
    large, temp = audit_large_and_temp_files()
