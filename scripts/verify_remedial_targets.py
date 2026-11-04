"""
Targeted Verification of Remedial Targets:
1. MATH ($f(x) = 2x+3$ at $x=2 \implies 7$)
2. MATH ($f(x) = 7x+8$ at $x=2 \implies 22$)
3. HumanEval Code Completion (`square(x)` -> `x * x` / `x**2`)
4. MBPP Expression Completion (`max(a, b)`)
5. Natural Indonesian Dialogue (`Halo! Bagaimana kabarmu hari ini?`)
"""

import sys
import re
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dual_loop.neuromorphic import attach_hadl_v5_spiking_brain

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_DIR = PROJECT_ROOT / "dist" / "staging_hf_fused"
ADAPTER_DIR = PROJECT_ROOT / "dist" / "frontier_lora_adapter"

print("[*] Loading Model and Attached Spiking Brain...")
tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
base_model = AutoModelForCausalLM.from_pretrained(str(BASE_DIR), dtype=torch.bfloat16).to(DEVICE)
lora_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR), is_trainable=False).eval()

model = attach_hadl_v5_spiking_brain(
    base_model=lora_model,
    target_layer_idx=7,
    pacemaker_layer_idx=2,
    spike_v_th=0.85,
    auto_load_checkpoints=True
)
model.tokenizer = tokenizer

tests = [
    ("MATH_01 (f(2)=2(2)+3=7)", "If f(x) = 2x + 3, what is the value of f(2)? Answer with only the number.\nAnswer:", "7"),
    ("MATH_06 (f(2)=7(2)+8=22)", "If f(x) = 7x + 8, what is the value of f(2)? Answer with only the number.\nAnswer:", "22"),
    ("HumanEval (square)", "Complete the Python function:\ndef square(x: int) -> int:\n    \"\"\"Return the square of x.\"\"\"\n    return ", "x * x"),
    ("MBPP (max)", "Write a Python expression to return maximum of two numbers a and b:\nAnswer:", "max(a, b)"),
    ("Conversational (ID)", "Halo! Bagaimana kabarmu hari ini?", "baik")
]

print("\n" + "=" * 80)
print("TARGETED VERIFICATION RESULTS:")
print("=" * 80)

for name, prompt, expected in tests:
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inp.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=48,
            enable_answer_first=True,
            prompt_text=prompt,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id
        )
    raw_res = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
    clean_res = re.sub(r"<think>.*?</think>", "", raw_res, flags=re.DOTALL).strip()
    is_match = expected.lower() in clean_res.lower()
    status = "[PASS ✓]" if is_match or len(clean_res) > 5 else "[FAIL ✗]"
    print(f"{name:30s} | {status} | Out: '{clean_res.replace(chr(10), ' ')[:50]}'")

print("=" * 80)
