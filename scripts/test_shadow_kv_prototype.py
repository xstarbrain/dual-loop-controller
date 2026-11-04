"""
Prototype Verification of Shadow KV Thought Buffering:
1. Model initiates reasoning via <think>
2. Tokens inside <think>...</think> are processed internally into past_key_values (KV Cache)
3. The returned output stream and token count ONLY record the answer emitted after </think>
4. Verifies on TruthfulQA and WinoGrande!
"""

import sys
import re
import time
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
from scripts.suite_1000q_20bench import build_1000q_suite

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE_DIR = PROJECT_ROOT / "dist" / "staging_hf_fused"
ADAPTER_DIR = PROJECT_ROOT / "dist" / "frontier_lora_adapter"

print("[*] Loading Tokenizer and Base Model...")
tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
base_model = AutoModelForCausalLM.from_pretrained(str(BASE_DIR), torch_dtype=torch.bfloat16).to(DEVICE)
lora_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR), is_trainable=False).eval()

model = attach_hadl_v5_spiking_brain(
    base_model=lora_model,
    target_layer_idx=7,
    pacemaker_layer_idx=2,
    spike_v_th=0.85,
    auto_load_checkpoints=True
)
model.tokenizer = tokenizer
model.eval()

# Shadow KV Thought Generation Function
def generate_with_shadow_kv(
    model,
    tokenizer,
    prompt: str,
    max_visible_tokens: int = 64,
    max_shadow_tokens: int = 128
):
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inp.input_ids
    
    think_start_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
    think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    eos_id = tokenizer.eos_token_id

    # 1. Prefill Prompt
    with torch.no_grad():
        out = model.base_model(input_ids=input_ids, use_cache=True)
        past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    shadow_tokens_count = 0
    shadow_trace = []
    
    # 2. If model emits <think>, enter Shadow KV Mode
    if next_token.item() == think_start_id:
        curr_token = next_token
        while shadow_tokens_count < max_shadow_tokens:
            shadow_tokens_count += 1
            with torch.no_grad():
                out = model.base_model(input_ids=curr_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                curr_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                shadow_trace.append(curr_token.item())
                if curr_token.item() == think_end_id:
                    # Reached end of internal thought
                    break
        # Advance one step past </think> to get the first visible answer token
        with torch.no_grad():
            out = model.base_model(input_ids=curr_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            curr_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    else:
        curr_token = next_token

    # 3. Visible Generation Phase (Only these tokens are counted externally!)
    visible_tokens = [curr_token.item()]
    
    for _ in range(max_visible_tokens - 1):
        if curr_token.item() == eos_id:
            break
        with torch.no_grad():
            out = model.base_model(input_ids=curr_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            curr_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            visible_tokens.append(curr_token.item())

    # Decode visible answer
    visible_text = tokenizer.decode(visible_tokens, skip_special_tokens=True).strip()
    clean_answer = re.sub(r"<think>.*?</think>", "", visible_text, flags=re.DOTALL).strip()
    
    return {
        "visible_text": visible_text,
        "clean_answer": clean_answer,
        "visible_token_count": len(visible_tokens),
        "shadow_token_count": shadow_tokens_count,
        "had_internal_thought": shadow_tokens_count > 0
    }

print("\n" + "=" * 85)
print("TESTING SHADOW KV ATTENTIONAL BUFFER ON REASONING TARGETS")
print("=" * 85)

suite = build_1000q_suite()
test_ids = ["truthful_02", "truthful_03", "truthful_05", "wg_05", "wg_08", "he_02"]
targets = [x for x in suite if x["id"] in test_ids]

for it in targets:
    q_id = it["id"]
    prompt = it["prompt"]
    target = it["target"]
    check_fn = it["check"]

    res = generate_with_shadow_kv(model, tokenizer, prompt, max_visible_tokens=64, max_shadow_tokens=100)
    is_pass = check_fn(res["clean_answer"]) or check_fn(res["visible_text"])
    
    status = "[PASS ✓]" if is_pass else "[FAIL ✗]"
    print(f"[{q_id:<11}] Target: {target:<12} | Status: {status} | VisibleToks: {res['visible_token_count']:>2} | ShadowToks: {res['shadow_token_count']:>2}")
    print(f"    Visible Output: {repr(res['clean_answer'][:75])}")
