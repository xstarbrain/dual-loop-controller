import json
import uuid
import pathlib

def create_cell(cell_type, source, execution_count=None, outputs=None):
    if isinstance(source, str):
        lines = [line + '\n' for line in source.split('\n')]
        if lines and lines[-1] == '\n':
            lines[-1] = ''
    else:
        lines = source
    
    cell = {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": lines
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs if outputs is not None else []
    return cell

cells = []

# Cell 1: Markdown Header
c1 = """# 🚀 Name That LLM: Deep Learning Pipeline (DeBERTa-v3)
### Targeting 0.50+ on Kaggle GPU (T4 x2 or P100)

---

### ⚠️ IMPORTANT SETUP FOR KAGGLE NOTEBOOK:
This is a **Deep Learning Transformer (DeBERTa-v3)** model designed to beat the ~0.314 TF-IDF ceiling.
To run this in Kaggle, please verify two quick settings in the right sidebar (**Notebook options**):

1. **Accelerator**: Set to **GPU T4 x2** or **GPU P100** (Essential for fast 3-minute training).
2. **Model Weights (Choose Option A or Option B)**:
   - **Option A (Easiest for interactive testing)**: In the right sidebar, toggle **Internet: ON**.
   - **Option B (Required for final offline competition submission)**:
     - In the top-right toolbar of the notebook, click **`+ Add Input`**.
     - In the search bar, type: **`deberta-v3-base`**.
     - Click the **`+`** button on the public dataset (e.g. by `nbroad` or `debertav3base`).
     - The notebook will automatically find the weights and run 100% offline!
"""
cells.append(create_cell("markdown", c1))

# Cell 2: Imports & Environment Setup
c2 = """import os
import gc
import sys
import pathlib
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, get_linear_schedule_with_warmup

# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: GPU is not detected! In Kaggle, switch Accelerator to 'GPU T4 x2' or 'GPU P100' in the right sidebar!")
"""
cells.append(create_cell("code", c2))

# Cell 3: Data Discovery & Loading
c3 = """# Kaggle input directory discovery
CANDIDATES = [
    pathlib.Path('/kaggle/input/competitions/name-that-llm'),
    pathlib.Path('/kaggle/input/name-that-llm'),
    pathlib.Path('.'),
    pathlib.Path('..'),
]

DATA_DIR = None
for p in CANDIDATES:
    if (p / 'train.csv').exists():
        DATA_DIR = p
        break

if DATA_DIR is None:
    zip_candidates = list(pathlib.Path('.').glob('*.zip')) + list(pathlib.Path('..').glob('*.zip'))
    for zpath in zip_candidates:
        if 'name-that-llm' in zpath.name:
            with zipfile.ZipFile(zpath, 'r') as z:
                z.extractall('data')
            DATA_DIR = pathlib.Path('data')
            break

if DATA_DIR is None:
    raise FileNotFoundError("Could not locate train.csv in candidate paths.")

train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df = pd.read_csv(DATA_DIR / 'test.csv')

# Build Label Mappings (Strictly 18 classes)
UNIQUE_MODELS = sorted(train_df['model'].unique())
NUM_CLASSES = len(UNIQUE_MODELS)
model2idx = {m: i for i, m in enumerate(UNIQUE_MODELS)}
idx2model = {i: m for i, m in enumerate(UNIQUE_MODELS)}
train_df['label'] = train_df['model'].map(model2idx)

print(f"Dataset successfully loaded from: {DATA_DIR}")
print(f"Train Shape: {train_df.shape} | Prompts: {train_df['prompt_id'].nunique()} | Models: {NUM_CLASSES}")
print(f"Test Shape:  {test_df.shape}")
train_df.head(3)
"""
cells.append(create_cell("code", c3))

# Cell 4: Robust Model Weight & Tokenizer Discovery (Offline + Online)
c4 = """def discover_model_source():
    \"\"\"
    Searches recursively across all attached Kaggle inputs for pretrained DeBERTa weights.
    Falls back to online HuggingFace download if internet is enabled.
    \"\"\"
    # 1. Search recursively in /kaggle/input for any transformer config
    input_root = pathlib.Path('/kaggle/input')
    if input_root.exists():
        for config_path in input_root.rglob('config.json'):
            parent_dir = config_path.parent
            # Check if this folder contains tokenizer assets or model weights
            has_tok = any(parent_dir.glob('tokenizer*')) or any(parent_dir.glob('spm*')) or any(parent_dir.glob('vocab*'))
            has_weights = any(parent_dir.glob('*.bin')) or any(parent_dir.glob('*.safetensors')) or (parent_dir / 'pytorch_model.bin').exists()
            if has_tok or has_weights:
                print(f"Found offline Transformer model at: {parent_dir}")
                return str(parent_dir)

    # 2. Check common known Kaggle dataset directory names
    known_paths = [
        '/kaggle/input/deberta-v3-base',
        '/kaggle/input/debertav3base',
        '/kaggle/input/deberta-v3-small',
        '/kaggle/input/debertav3small',
        '/kaggle/input/deberta-v3-large',
    ]
    for p in known_paths:
        if pathlib.Path(p).exists():
            print(f"Found offline model at: {p}")
            return p

    # 3. Check if internet connection is alive
    try:
        urllib.request.urlopen('https://huggingface.co', timeout=3)
        print("Internet connection detected. Using HuggingFace Hub: 'microsoft/deberta-v3-base'")
        return "microsoft/deberta-v3-base"
    except Exception:
        pass

    return None

MODEL_SOURCE = discover_model_source()

if MODEL_SOURCE is None:
    raise RuntimeError(
        "\\n" + "="*70 + "\\n"
        "❌ DEBERTA-V3 WEIGHTS / TOKENIZER TIDAK DAPAT DITEMUKAN!\\n\\n"
        "PENYEBAB: Internet di notebook Kaggle mati dan belum ada dataset DeBERTa yang ditambahkan.\\n\\n"
        "CARA MEMPERBAIKI DALAM 10 DETIK:\\n"
        "1. Di pojok kanan atas notebook Kaggle, klik '+ Add Input'.\\n"
        "2. Di kotak pencarian, ketik: 'deberta-v3-base'\\n"
        "3. Klik tombol '+' pada salah satu dataset (misalnya 'nbroad/deberta-v3-base' atau 'debertav3base').\\n"
        "4. Setelah dataset masuk, jalankan 'Run All' kembali!\\n"
        "ATAU (jika sedang eksperimen interaktif): Ubah 'Internet: ON' di panel kanan (Notebook options).\\n"
        "="*70
    )

print(f"Loading Tokenizer from: {MODEL_SOURCE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
print("Tokenizer successfully initialized!")
"""
cells.append(create_cell("code", c4))

# Cell 5: PyTorch Dataset & Dataloader
c5 = """class LLMTextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=256):
        self.texts = [str(t) if pd.notnull(t) else "" for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
"""
cells.append(create_cell("code", c5))

# Cell 6: Training & Validation Loop Functions
c6 = """def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        if 'label' in batch:
            all_labels.append(batch['label'].numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    if all_labels:
        all_labels = np.concatenate(all_labels, axis=0)
        score = balanced_accuracy_score(all_labels, all_preds)
        return score, all_probs, all_preds
    return all_probs, all_preds
"""
cells.append(create_cell("code", c6))

# Cell 7: Full Dataset Training (Fast ~3-4 Epochs on Kaggle GPU)
c7 = """EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 256

print(f"Preparing Full Training on {len(train_df)} samples...")
train_dataset = LLMTextDataset(
    texts=train_df['text'].values,
    labels=train_df['label'].values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# Initialize DeBERTa model with 18 classification classes
print(f"Initializing DeBERTa model from: {MODEL_SOURCE}")
config = AutoConfig.from_pretrained(MODEL_SOURCE, num_labels=NUM_CLASSES)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_SOURCE,
    config=config,
    ignore_mismatched_sizes=True
)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

print(f"Starting Training: {EPOCHS} epochs ({total_steps} optimizer steps total)...")
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, DEVICE)
    print(f"Epoch {epoch}/{EPOCHS} | CrossEntropy Loss: {loss:.4f}")

print("Training Successfully Completed!")
"""
cells.append(create_cell("code", c7))

# Cell 8: Inference on test.csv & Parquet Generation
c8 = """print(f"Running inference on {len(test_df)} test samples...")
test_dataset = LLMTextDataset(
    texts=test_df['text'].values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

test_probs, test_preds_idx = evaluate(model, test_loader, DEVICE)
predicted_models = [idx2model[idx] for idx in test_preds_idx]

submission = pd.DataFrame({
    'id': test_df['id'],
    'model': predicted_models
})

# Rigorous Quality Assurance Constraints
assert len(submission) == len(test_df), f"Row count mismatch: expected {len(test_df)}, got {len(submission)}"
assert list(submission.columns) == ['id', 'model'], f"Invalid columns: {submission.columns}"
assert not submission['model'].isnull().any(), "Found NaN in predicted models!"
assert all(m in UNIQUE_MODELS for m in submission['model']), "Found invalid model names!"

submission.to_parquet('submission.parquet', index=False)
print(f"Successfully generated and verified submission.parquet ({os.path.getsize('submission.parquet')} bytes)!")
print("\\nFirst 10 submission rows:")
print(submission.head(10))
"""
cells.append(create_cell("code", c8))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Write notebook to target locations
paths = [
    r"c:\Users\Matthew Chen\Documents\X-Star\competition\name_that_llm\deberta_v3_name_that_llm.ipynb",
    r"C:\Users\Matthew Chen\Downloads\deberta_v3_name_that_llm.ipynb"
]

for p in paths:
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"Generated DeBERTa-v3 notebook at: {p}")
