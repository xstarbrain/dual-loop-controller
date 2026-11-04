"""
LoRA Fine-tuning engine for Frontier Advanced Reasoning on RTX 5060 GPU.
Trains on frontier_curriculum_master.json (or augmented remedial datasets).
"""

import json
import time
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

MODEL_DIR = Path("dist/staging_hf_fused")
ADAPTER_OUTPUT_DIR = Path("dist/frontier_lora_adapter")
ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class FrontierCurriculumDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer, max_length: int = 768):
        with open(data_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for item in self.items:
            instr = item["instruction"]
            resp = item["response"]
            prompt = (
                f"<|im_start|>user\n{instr}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n</think>\n{resp}<|im_end|>"
            )
            tokens = tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            input_ids = tokens.input_ids[0]
            labels = input_ids.clone()
            
            # Mask user instruction from loss calculation
            user_prefix = f"<|im_start|>user\n{instr}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"
            prefix_len = len(tokenizer(user_prefix, add_special_tokens=False).input_ids)
            labels[:prefix_len] = -100  # PyTorch ignore index
            
            self.samples.append({
                "input_ids": input_ids,
                "labels": labels
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for x in batch:
        l = len(x["input_ids"])
        pad_len = max_len - l
        # Pad with eos token
        inp = torch.cat([x["input_ids"], torch.full((pad_len,), 151643, dtype=torch.long)])
        lbl = torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        mask = torch.cat([torch.ones(l, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        input_ids.append(inp)
        labels.append(lbl)
        attention_mask.append(mask)
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask)
    }

def train_adapter(data_file: str, epochs: int = 4, lr: float = 3e-4, resume_adapter: bool = False):
    print("=" * 80)
    print(f"[*] INITIATING FRONTIER ADAPTER TRAINING (Epochs: {epochs}, LR: {lr})")
    print(f"[*] Training Data: {data_file}")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    dataset = FrontierCurriculumDataset(Path(data_file), tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    print(f"[*] Total Training Examples: {len(dataset)}")
    
    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    base_model.gradient_checkpointing_enable()
    
    if resume_adapter and (ADAPTER_OUTPUT_DIR / "adapter_model.safetensors").exists():
        print(f"[*] Resuming from existing adapter in {ADAPTER_OUTPUT_DIR}")
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_OUTPUT_DIR), is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    grad_accum_steps = 4
    total_steps = (len(dataloader) // grad_accum_steps) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-5)
    
    start_time = time.perf_counter()
    step_count = 0
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            epoch_loss += loss.item() * grad_accum_steps
            
            if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1
                
        avg_loss = epoch_loss / len(dataloader)
        print(f"    Epoch [{epoch}/{epochs}] Complete - Avg Training Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")
        
    duration = time.perf_counter() - start_time
    print(f"[+] Training completed in {duration:.2f}s across {step_count} optimizer steps.")
    
    # Save adapter
    model.save_pretrained(str(ADAPTER_OUTPUT_DIR))
    tokenizer.save_pretrained(str(ADAPTER_OUTPUT_DIR))
    print(f"[+] Successfully saved fine-tuned LoRA adapter to: {ADAPTER_OUTPUT_DIR}")
    
    # Clean up GPU memory
    del model
    del base_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="frontier_study/frontier_curriculum_master.json")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    train_adapter(args.data, epochs=args.epochs, lr=args.lr, resume_adapter=args.resume)
