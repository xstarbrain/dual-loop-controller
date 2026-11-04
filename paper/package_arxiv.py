"""
Helper script to package the LaTeX paper and figures into an arXiv-compliant archive (.tar.gz and .zip).
"""
import tarfile
import zipfile
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent

FILES_TO_INCLUDE = [
    "main.tex",
    "references.bib",
    "figures/xstar_architecture_diagram.png",
    "figures/xstar_1000q_benchmark_audit_english.png",
    "figures/xstar_peer_models_comparison.png",
    "figures/xstar_champion_domains.png",
    "figures/xstar_evolution_progression.png",
    "figures/arc_prize_2026_comprehensive_report.png",
    "figures/agi_5level_maturity_radar.png"
]

def package_arxiv():
    tar_path = PAPER_DIR / "arxiv_submission.tar.gz"
    zip_path = PAPER_DIR / "arxiv_submission.zip"

    print("[*] Packaging arXiv submission...")
    
    # 1. Create .tar.gz (preferred by arXiv)
    with tarfile.open(tar_path, "w:gz") as tar:
        for rel in FILES_TO_INCLUDE:
            p = PAPER_DIR / rel
            if p.exists():
                tar.add(p, arcname=rel)
                print(f"  + Added: {rel}")
            else:
                print(f"  [!] Missing file: {rel}")

    # 2. Create .zip (backup)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in FILES_TO_INCLUDE:
            p = PAPER_DIR / rel
            if p.exists():
                zf.write(p, arcname=rel)

    print(f"\n[+] Created arXiv submission archives:")
    print(f"    - {tar_path} ({tar_path.stat().st_size / 1024:.1f} KB)")
    print(f"    - {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    package_arxiv()
