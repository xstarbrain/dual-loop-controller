import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.suite_1000q_20bench import build_1000q_suite

suite = [x for x in build_1000q_suite() if x["benchmark"] == "WinoGrande"][:20]
fail_ids = ["wg_01", "wg_03", "wg_07", "wg_08", "wg_11"]
for x in suite:
    if x["id"] in fail_ids:
        print(f"=== {x['id']} ===")
        print("Prompt:\n", x["prompt"])
        print("Target:", x["target"])
        print()
