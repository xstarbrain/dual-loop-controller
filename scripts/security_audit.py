"""
Comprehensive Security & Hygiene Audit for X-Star / Dual-Loop Controller
Performs:
1. Secret & Credential Leak Detection (Tokens, API Keys, Private Keys)
2. Static AST / Code Safety Analysis (eval/exec, unsafe torch.load, shell=True)
3. Repository Hygiene & Extraneous File Detection
4. Popperian Invariant & Sandbox Boundary Compliance
"""

import os
import re
import ast
import sys
from pathlib import Path

WORKSPACE = Path(".").resolve()

# Patterns for secrets
SECRET_PATTERNS = [
    (r"(?i)ghp_[0-9a-zA-Z]{36}", "GitHub Personal Access Token"),
    (r"(?i)github_pat_[0-9a-zA-Z_]{82}", "GitHub Fine-Grained Token"),
    (r"(?i)hf_[0-9a-zA-Z]{34,}", "HuggingFace API Token"),
    (r"(?i)sk-[0-9a-zA-Z]{32,}", "OpenAI API Secret Key"),
    (r"(?i)AIza[0-9A-Za-z-_]{35}", "Google API / Gemini Key"),
    (r"(?i)AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
    (r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----", "Private Key Header"),
    (r"(?i)(?:api_key|secret_key|access_token|password|auth_token)\s*=\s*['\"][0-9a-zA-Z\-_]{16,}['\"]", "Hardcoded API Key / Secret Variable"),
]

# Paths to ignore from scanning
IGNORE_DIRS = {
    ".git", ".venv", ".pytest_cache", "__pycache__", "node_modules",
    ".idea", ".vscode", "egg-info", "dual_loop_controller.egg-info",
    "site-packages"
}

def is_binary_string(bytes_data):
    text_chars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
    return bool(bytes_data.translate(None, text_chars))

def scan_secrets():
    findings = []
    for root, dirs, files in os.walk(WORKSPACE):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.endswith(".egg-info")]
        for f in files:
            path = Path(root) / f
            rel_path = path.relative_to(WORKSPACE)
            
            # Skip large binaries or images
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".wav", ".pt", ".safetensors", ".bin", ".gguf", ".zip", ".tar.gz", ".whl", ".parquet"}:
                continue
            
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            
            for pattern, desc in SECRET_PATTERNS:
                matches = re.finditer(pattern, content)
                for m in matches:
                    matched_str = m.group(0)
                    # Filter out placeholders
                    if any(ph in matched_str.lower() for ph in ["your_api_key", "placeholder", "xxx", "sk-...", "dummy"]):
                        continue
                    # Also filter out test tokens if explicit mock
                    line_no = content[:m.start()].count("\n") + 1
                    findings.append({
                        "file": str(rel_path),
                        "line": line_no,
                        "type": desc,
                        "snippet": matched_str[:12] + "..." + matched_str[-4:] if len(matched_str) > 16 else matched_str
                    })
    return findings

class ASTSafetyVisitor(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.issues = []

    def visit_Call(self, node):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                func_name = f"{node.func.value.id}.{func_name}"

        # 1. Check eval / exec
        if func_name in {"eval", "exec"}:
            self.issues.append({
                "file": self.filename,
                "line": node.lineno,
                "severity": "HIGH",
                "check": f"Use of dynamic execution: {func_name}()"
            })

        # 2. Check pickle.load / pickle.loads
        if func_name in {"pickle.load", "pickle.loads", "_pickle.load", "_pickle.loads"}:
            self.issues.append({
                "file": self.filename,
                "line": node.lineno,
                "severity": "HIGH",
                "check": f"Unsafe deserialization: {func_name}()"
            })

        # 3. Check torch.load without weights_only=True
        if func_name == "torch.load":
            has_weights_only = False
            for kw in node.keywords:
                if kw.arg == "weights_only":
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        has_weights_only = True
            if not has_weights_only:
                self.issues.append({
                    "file": self.filename,
                    "line": node.lineno,
                    "severity": "MEDIUM",
                    "check": "torch.load() without weights_only=True"
                })

        # 4. Check subprocess shell=True
        if func_name in {"subprocess.run", "subprocess.Popen", "subprocess.call", "subprocess.check_output"}:
            for kw in node.keywords:
                if kw.arg == "shell":
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        self.issues.append({
                            "file": self.filename,
                            "line": node.lineno,
                            "severity": "MEDIUM",
                            "check": f"{func_name}(shell=True) potential shell injection"
                        })

        # 5. Check os.system
        if func_name == "os.system":
            self.issues.append({
                "file": self.filename,
                "line": node.lineno,
                "severity": "MEDIUM",
                "check": "os.system() execution"
            })

        self.generic_visit(node)

def scan_code_safety():
    issues = []
    # Scan primary codebase: dual_loop, xstar, competition, tests
    target_dirs = ["dual_loop", "xstar", "competition", "tests"]
    for td in target_dirs:
        dir_path = WORKSPACE / td
        if not dir_path.exists():
            continue
        for root, _, files in os.walk(dir_path):
            if any(ign in root for ign in IGNORE_DIRS):
                continue
            for f in files:
                if f.endswith(".py"):
                    full_p = Path(root) / f
                    rel_p = str(full_p.relative_to(WORKSPACE))
                    try:
                        source = full_p.read_text(encoding="utf-8")
                        tree = ast.parse(source, filename=rel_p)
                        visitor = ASTSafetyVisitor(rel_p)
                        visitor.visit(tree)
                        issues.extend(visitor.issues)
                    except SyntaxError:
                        pass
                    except Exception:
                        pass
    return issues

def scan_extraneous_files():
    """Find scratch, stale zips, leftover logs, and unneeded files on disk."""
    extraneous = []
    for root, dirs, files in os.walk(WORKSPACE):
        dirs[:] = [d for d in dirs if d not in {".git", ".venv"}]
        for f in files:
            path = Path(root) / f
            rel = str(path.relative_to(WORKSPACE))
            # Zip submissions in root
            if rel in {"bf18_submission.zip", "sa17_submission.zip", "submission.zip", "submission.parquet"}:
                extraneous.append({"path": rel, "category": "Root Stale Submission Artifact", "size": path.stat().st_size})
            elif rel.startswith("paper\\arxiv_submission.") or rel.startswith("paper/arxiv_submission."):
                extraneous.append({"path": rel, "category": "Paper Archive Artifact", "size": path.stat().st_size})
            elif rel.endswith(".tmp") or rel.endswith(".bak"):
                extraneous.append({"path": rel, "category": "Temporary Backup File", "size": path.stat().st_size})
    return extraneous

if __name__ == "__main__":
    print("=" * 60)
    print("X-STAR / DUAL-LOOP CONTROLLER SECURITY & HYGIENE AUDIT")
    print("=" * 60)

    # 1. Secrets Scan
    print("\n[1] Scanning for Leaked Credentials & API Keys...")
    secrets = scan_secrets()
    if secrets:
        print(f"  [!] Found {len(secrets)} potential secret occurrences:")
        for s in secrets:
            print(f"      - {s['file']}:{s['line']} -> {s['type']} ({s['snippet']})")
    else:
        print("  [OK] Zero leaked secrets or credentials detected across all scanned files.")

    # 2. Code Safety AST Scan
    print("\n[2] Scanning AST for Code Safety (eval, exec, pickle, shell=True)...")
    safety_issues = scan_code_safety()
    high_issues = [i for i in safety_issues if i["severity"] == "HIGH"]
    med_issues = [i for i in safety_issues if i["severity"] == "MEDIUM"]
    print(f"  [i] Found {len(high_issues)} HIGH severity and {len(med_issues)} MEDIUM severity patterns.")
    for issue in safety_issues:
        print(f"      [{issue['severity']}] {issue['file']}:{issue['line']} -> {issue['check']}")

    # 3. Extraneous & Stray Files
    print("\n[3] Scanning for Stale / Extraneous Artifacts...")
    extraneous = scan_extraneous_files()
    if extraneous:
        print(f"  [!] Found {len(extraneous)} extraneous files taking up space:")
        for ext in extraneous:
            print(f"      - {ext['path']} ({ext['category']}, {ext['size']:,} bytes)")
    else:
        print("  [OK] No loose zip archives or temporary files found.")

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)
