import json
from pathlib import Path

def sync():
    system_prompt = Path("prompts/system.md").read_text(encoding="utf-8").strip()

    # 1. Update getting_started_gemma_4_trace_synthesis_018.ipynb
    p1 = Path("getting_started_gemma_4_trace_synthesis_018.ipynb")
    with open(p1, "r", encoding="utf-8") as f:
        nb1 = json.load(f)

    code1 = "".join(nb1["cells"][1]["source"])
    prefix1 = code1.split('SYSTEM_PROMPT = """')[0]
    suffix1 = code1.split('"""\n\nFILES = {')[1]
    new_code1 = prefix1 + 'SYSTEM_PROMPT = """' + system_prompt + '"""\n\nFILES = {' + suffix1
    nb1["cells"][1]["source"] = [line + "\n" for line in new_code1.splitlines()]

    with open(p1, "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=1, ensure_ascii=False)
    print("[+] Successfully synced getting_started_gemma_4_trace_synthesis_018.ipynb")

    # 2. Update getting_started_gemma_4_hadl_dualloop.ipynb
    p2 = Path("getting_started_gemma_4_hadl_dualloop.ipynb")
    with open(p2, "r", encoding="utf-8") as f:
        nb2 = json.load(f)

    for i, cell in enumerate(nb2["cells"]):
        src = "".join(cell.get("source", []))
        if 'SYSTEM_PROMPT = """' in src:
            prefix2 = src.split('SYSTEM_PROMPT = """')[0]
            # split on closing triple quotes
            rest = src[len(prefix2) + len('SYSTEM_PROMPT = """'):]
            suffix2 = rest.split('"""')[1]
            new_src = prefix2 + 'SYSTEM_PROMPT = """' + system_prompt + '"""' + suffix2
            cell["source"] = [line + "\n" for line in new_src.splitlines()]
            print(f"[+] Successfully synced getting_started_gemma_4_hadl_dualloop.ipynb (cell {i})")
            break

    with open(p2, "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    sync()
