import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dual_loop.neuromorphic.spiking_gate import attach_hadl_v5_spiking_brain

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained("dist/staging_hf_fused", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "dist/staging_hf_fused",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)
print("Attaching frontier LoRA adapter...")
base_model = PeftModel.from_pretrained(base_model, "dist/frontier_lora_adapter")
print("Attaching HADL v5 Spiking Brain...")
brain_wrapper = attach_hadl_v5_spiking_brain(
    base_model=base_model,
    target_layer_idx=7,
    pacemaker_layer_idx=2,
    deliberation_steps_k=1,
    auto_load_checkpoints=True
)

test_prompts = [
    ("truthful_02", "Fact-check query 2: Do humans only use 10 percent of their brain capacity?\nAnswer:"),
    ("truthful_03", "Fact-check query 3: Does dropping a penny from the Empire State Building kill someone?\nAnswer:"),
    ("wg_05", "Sentence: The heavy oak table could not fit through the narrow door because it was too narrow. What was narrow?\nAnswer:"),
    ("he_02", "Complete the Python function to return the product of a and b.\n```python\ndef multiply(a, b):\n"),
    ("arcc_05", "Science Challenge 5: Why does ice float on water?\nA) Denser\nB) Less dense than liquid water\nC) Heavier\nD) Warmer\nAnswer with the letter:"),
]

print("\n" + "=" * 80)
print("TESTING NATURAL UNCONSTRAINED GENERATION (to observe <think> pattern)")
print("=" * 80)

for q_id, prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = brain_wrapper.base_model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    has_think = "<think>" in text
    has_end_think = "</think>" in text
    after_think = text.split("</think>")[-1].strip() if has_end_think else "(no </think>)"
    print(f"\n[{q_id}]")
    print(f"  Has <think>: {has_think} | Has </think>: {has_end_think} | Total gen tokens: {len(gen_ids)}")
    if has_end_think:
        think_text = text.split("</think>")[0].replace("<think>", "").strip()
        print(f"  Think excerpt: {repr(think_text[:100])}...")
        print(f"  AFTER </think> (Visible answer): {repr(after_think[:100])}")
    else:
        print(f"  Full output: {repr(text[:120])}")
