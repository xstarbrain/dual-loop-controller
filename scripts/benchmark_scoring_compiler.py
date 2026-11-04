#!/usr/bin/env python3
"""SWE-Gemma Local Scoring Compiler & Benchmark Evaluator.

Provides automated, reproducible offline grading of agent patches against official
competition tasks from competition/tasks.jsonl:
  1. Instantiates a clean sandbox directory per task instance.
  2. Applies the task's official test_patch.
  3. Verifies baseline test FAILURE (proving the bug is real and reproducible).
  4. Applies candidate agent patch via `git apply --recount`.
  5. Audits AST syntax & py_compile on all modified source files.
  6. Enforces anti-tampering: ensures zero modifications to tests/ directory.
  7. Executes pytest on the affected test suite.
  8. Computes exact resolution accuracy, duration, and failure diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class SweGemmaScoringCompiler:
    def __init__(
        self,
        tasks_path: Path,
        python_bin: Path,
        verbose: bool = True
    ):
        self.tasks_path = Path(tasks_path)
        self.python_bin = Path(python_bin)
        self.verbose = verbose
        self.tasks = self._load_tasks()

    def _load_tasks(self) -> List[Dict[str, Any]]:
        if not self.tasks_path.exists():
            raise FileNotFoundError(f"Tasks file not found: {self.tasks_path}")
        with open(self.tasks_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def run_cmd(
        self,
        cmd: List[str],
        cwd: Path,
        timeout: int = 60,
        env: Optional[Dict[str, str]] = None
    ) -> subprocess.CompletedProcess:
        e = os.environ.copy()
        if env:
            e.update(env)
        # ensure PYTHONPATH includes sandbox
        e["PYTHONPATH"] = str(cwd) + os.pathsep + e.get("PYTHONPATH", "")
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=e
        )

    def evaluate_patch(
        self,
        task: Dict[str, Any],
        candidate_patch: str,
        timeout_seconds: int = 150
    ) -> Dict[str, Any]:
        """Evaluates a candidate patch against a task instance."""
        instance_id = task.get("instance_id", "unknown")
        repo = task.get("repo", "unknown")
        test_patch = task.get("test_patch", "")
        
        sandbox = Path(tempfile.mkdtemp(prefix=f"swe_compiler_{instance_id}_"))
        t0 = time.time()
        
        try:
            # Step 1: Initialize git in sandbox
            self.run_cmd(["git", "init"], sandbox)
            self.run_cmd(["git", "config", "user.name", "KaggleEvaluator"], sandbox)
            self.run_cmd(["git", "config", "user.email", "eval@kaggle.com"], sandbox)
            
            # Step 2: Seed repository base files if needed
            self._seed_sandbox_repo(sandbox, repo, task)
            
            # Commit baseline
            self.run_cmd(["git", "add", "."], sandbox)
            self.run_cmd(["git", "commit", "-m", "baseline", "--allow-empty"], sandbox)
            
            # Step 3: Apply test_patch to verify baseline failure
            baseline_failed = False
            test_files: List[str] = []
            if test_patch:
                tpatch_file = sandbox / "test_patch.diff"
                tpatch_file.write_text(test_patch + "\n", encoding="utf-8")
                apply_test = self.run_cmd(["git", "apply", "--recount", "--whitespace=nowarn", "test_patch.diff"], sandbox)
                if apply_test.returncode != 0:
                    apply_test = self.run_cmd(["git", "apply", "--recount", "--ignore-whitespace", "test_patch.diff"], sandbox)
                
                # identify modified test files
                for l in test_patch.splitlines():
                    if l.startswith("+++ b/"):
                        test_files.append(l[6:].strip())
                
                # Run pre-test pytest
                if test_files:
                    cmd = [str(self.python_bin), "-m", "pytest"] + test_files + ["-q"]
                    try:
                        pre_res = self.run_cmd(cmd, sandbox, timeout=timeout_seconds)
                        baseline_failed = (pre_res.returncode != 0)
                    except subprocess.TimeoutExpired:
                        baseline_failed = True
            
            # Step 4: Reset sandbox back to baseline (remove test_patch temporarily or keep committed)
            self.run_cmd(["git", "reset", "--hard", "HEAD"], sandbox)
            self.run_cmd(["git", "clean", "-fd"], sandbox)
            
            # Step 5: Apply candidate agent patch
            if not candidate_patch.strip():
                return {
                    "instance_id": instance_id,
                    "repo": repo,
                    "resolved": False,
                    "baseline_failed": baseline_failed,
                    "error": "Empty candidate patch",
                    "duration_s": round(time.time() - t0, 2)
                }
            
            patch_file = sandbox / "agent_patch.diff"
            patch_file.write_text(candidate_patch.strip() + "\n", encoding="utf-8")
            apply_res = self.run_cmd(["git", "apply", "--recount", "--whitespace=nowarn", "agent_patch.diff"], sandbox)
            if apply_res.returncode != 0:
                # Fallback git apply with whitespace ignore
                apply_res = self.run_cmd(["git", "apply", "--recount", "--ignore-whitespace", "agent_patch.diff"], sandbox)
            
            if apply_res.returncode != 0:
                return {
                    "instance_id": instance_id,
                    "repo": repo,
                    "resolved": False,
                    "baseline_failed": baseline_failed,
                    "error": f"Patch application failed: {apply_res.stderr.strip()}",
                    "duration_s": round(time.time() - t0, 2)
                }
            
            # Step 6: Anti-tampering check on candidate patch
            status_res = self.run_cmd(["git", "status", "--porcelain"], sandbox)
            modified_files = [line[3:].strip() for line in status_res.stdout.splitlines() if line.strip()]
            illegal = [f for f in modified_files if f.startswith("tests/") and not ("__pycache__" in f or f.endswith(".pyc"))]
            if illegal:
                return {
                    "instance_id": instance_id,
                    "repo": repo,
                    "resolved": False,
                    "baseline_failed": baseline_failed,
                    "error": f"Anti-tampering violation (tests modified): {illegal}",
                    "duration_s": round(time.time() - t0, 2)
                }
            
            # Step 7: AST compile check on modified python files
            for mf in modified_files:
                if mf.endswith(".py"):
                    c_res = self.run_cmd([str(self.python_bin), "-m", "py_compile", mf], sandbox)
                    if c_res.returncode != 0:
                        return {
                            "instance_id": instance_id,
                            "repo": repo,
                            "resolved": False,
                            "baseline_failed": baseline_failed,
                            "error": f"SyntaxError in {mf}: {c_res.stderr.strip()}",
                            "duration_s": round(time.time() - t0, 2)
                        }
            
            # Step 8: Apply official test_patch on top of the candidate patch
            if test_patch:
                tpatch_file = sandbox / "test_patch.diff"
                tpatch_file.write_text(test_patch + "\n", encoding="utf-8")
                t_res = self.run_cmd(["git", "apply", "--recount", "--whitespace=nowarn", "test_patch.diff"], sandbox)
                if t_res.returncode != 0:
                    self.run_cmd(["git", "apply", "--recount", "--ignore-whitespace", "test_patch.diff"], sandbox)
            
            # Step 9: Execute official pytest verification
            if test_files:
                cmd = [str(self.python_bin), "-m", "pytest"] + test_files + ["-q"]
                try:
                    eval_res = self.run_cmd(cmd, sandbox, timeout=timeout_seconds)
                    resolved = (eval_res.returncode == 0)
                    out_snippet = eval_res.stdout.strip()[-300:] if eval_res.stdout else eval_res.stderr.strip()[-300:]
                except subprocess.TimeoutExpired:
                    resolved = False
                    out_snippet = f"Pytest timed out after {timeout_seconds}s"
            else:
                resolved = True
                out_snippet = "No test patch files specified (vacuous pass)"
            
            duration = round(time.time() - t0, 2)
            return {
                "instance_id": instance_id,
                "repo": repo,
                "resolved": resolved,
                "baseline_failed": baseline_failed,
                "duration_s": duration,
                "snippet": out_snippet,
                "status": "RESOLVED (PASS)" if resolved else "FAILED"
            }
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    def _seed_sandbox_repo(self, sandbox: Path, repo: str, task: Dict[str, Any]):
        """Seeds sandbox structure matching the repo using installed packages when available."""
        site_packages = self.python_bin.parent.parent / "Lib" / "site-packages"
        if not site_packages.exists():
            site_packages = self.python_bin.parent / "site-packages"

        patch = task.get("patch", "")
        test_patch = task.get("test_patch", "")
        combined = patch + "\n" + test_patch

        # Identify files being modified vs newly created
        new_files = set()
        modified_files = set()
        current_file = None
        for line in combined.splitlines():
            if line.startswith("--- a/"):
                current_file = line[6:].strip()
            elif line.startswith("--- /dev/null"):
                current_file = None
            elif line.startswith("+++ b/"):
                b_file = line[6:].strip()
                if current_file is None:
                    new_files.add(b_file)
                else:
                    current_file = b_file
            elif line.startswith("@@ -0,0"):
                if current_file:
                    new_files.add(current_file)
            elif line.startswith("@@ -") and current_file:
                modified_files.add(current_file)

        for orig_file in modified_files:
            if orig_file in new_files:
                continue
            target_path = sandbox / orig_file
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Attempt to copy real baseline file from site-packages
            source_file = None
            if orig_file.startswith("fastapi/") and (site_packages / orig_file).exists():
                source_file = site_packages / orig_file
            elif orig_file.startswith("rich/") and (site_packages / orig_file).exists():
                source_file = site_packages / orig_file
            elif orig_file.startswith("src/requests/"):
                rel = orig_file[4:] # requests/...
                if (site_packages / rel).exists():
                    source_file = site_packages / rel
            elif orig_file.startswith("requests/") and (site_packages / orig_file).exists():
                source_file = site_packages / orig_file

            if source_file and source_file.is_file():
                shutil.copy2(source_file, target_path)

    def run_benchmark_suite(
        self,
        task_indices: List[int],
        patch_generator_fn
    ) -> Dict[str, Any]:
        """Runs evaluation across a suite of task indices and reports aggregate statistics."""
        results = []
        resolved_count = 0
        total_time = 0.0

        print(f"\n{'='*80}")
        print(f"[*] RUNNING SWE-GEMMA LOCAL SCORING COMPILER ON {len(task_indices)} TASKS")
        print(f"{'='*80}\n")

        for i, idx in enumerate(task_indices):
            task = self.tasks[idx]
            iid = task.get("instance_id")
            repo = task.get("repo")
            
            print(f"[{i+1}/{len(task_indices)}] Evaluating Task #{idx}: {iid} ({repo})...", end=" ", flush=True)
            
            # Generate candidate patch
            cand_patch = patch_generator_fn(task)
            
            res = self.evaluate_patch(task, cand_patch)
            results.append(res)
            
            total_time += res.get("duration_s", 0)
            if res.get("resolved", False):
                resolved_count += 1
                print(f"\033[1;32mPASSED\033[0m ({res.get('duration_s')}s)")
            else:
                err = res.get("error") or res.get("snippet", "Failed")
                print(f"\033[1;31mFAILED\033[0m ({res.get('duration_s')}s) - {err[:80]}")

        accuracy = (resolved_count / len(task_indices)) if task_indices else 0.0
        
        print(f"\n{'='*80}")
        print(f"[*] BENCHMARK SUITE RESULTS:")
        print(f"    Resolved Tasks : {resolved_count} / {len(task_indices)} ({accuracy:.1%})")
        print(f"    Total Time     : {total_time:.2f}s (Mean: {total_time/max(1, len(task_indices)):.2f}s/task)")
        print(f"{'='*80}\n")
        
        return {
            "total_tasks": len(task_indices),
            "resolved_tasks": resolved_count,
            "accuracy": accuracy,
            "total_time_s": total_time,
            "results": results
        }


def main():
    parser = argparse.ArgumentParser(description="SWE-Gemma Local Scoring Compiler")
    parser.add_argument("--tasks-file", default="competition/tasks.jsonl", help="Path to tasks.jsonl")
    parser.add_argument("--python-bin", default=str(Path(sys.executable)), help="Python binary path")
    parser.add_argument("--num-tasks", type=int, default=10, help="Number of tasks to evaluate")
    args = parser.parse_args()

    python_bin = Path(args.python_bin)
    venv_python = Path(".venv/Scripts/python.exe")
    if venv_python.exists():
        python_bin = venv_python.resolve()

    compiler = SweGemmaScoringCompiler(
        tasks_path=Path(args.tasks_file),
        python_bin=python_bin
    )

    # Sanity test using gold patches (proves the scoring compiler correctly grades 100% on valid patches)
    indices = list(range(min(args.num_tasks, len(compiler.tasks))))
    print(f"[+] Loaded compiler with {len(compiler.tasks)} tasks. Testing compiler on {len(indices)} tasks with gold patches...")
    
    summary = compiler.run_benchmark_suite(
        task_indices=indices,
        patch_generator_fn=lambda t: t.get("patch", "")
    )
    
    # Save benchmark audit
    out_path = Path("eval_results/local_scoring_compiler_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Audit saved to: {out_path}")


if __name__ == "__main__":
    main()
