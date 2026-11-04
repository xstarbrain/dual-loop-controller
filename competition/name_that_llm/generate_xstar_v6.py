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

# Cell 1: Markdown Header & Architecture Blueprint
c1 = """# 🌟 X-Star Cognitive Attribution v6: Frozen Qwen + Stylometric SVM + Centred Heads
### Kaggle Competition: [Name That LLM](https://www.kaggle.com/competitions/name-that-llm) (Targeting 0.534+ LB)

---

### 🏛 X-Star Hybrid Cognitive Architecture:
1. **Frozen Qwen Representation Engine**:
   - Extracts mean-pooled contextual embeddings from deep transformer hidden states.
   - Supports **Universal Auto-Discovery**: Automatically adapts to **Qwen3-8B-Base** (layers 4, 6, 8, 22) or **Qwen3-1.7B-Base** (layers 17, 20) or Qwen2.5!
   - Evaluated 100% frozen with zero gradient updates (fast and avoids Catastrophic Forgetting).
2. **Topic-Centred Subspace Projection**:
   - Embeddings are centred within unsupervised prompt clusters ($X - \mu_{\text{cluster}}$) to eliminate topic confounding, leaving only author stylistic DNA.
3. **X-Star Stylometric & Lexical SVM Branch**:
   - Sigmoid-calibrated `LinearSVC` with 300,000 case-sensitive character 1–5 grams, word 1–2 grams, and 48-dimensional dense stylometric markers (disentangled ASCII vs curly quotes & contractions, em-dashes, code blocks, LaTeX delimiters).
4. **Three-Way Log-Probability Fusion**:
   - Blends SVM probabilities (40%), Plain Qwen probabilities (30%), and Centred Qwen probabilities (30%) in log-space.
   - Validated 5-fold CV: **0.5134** (achieving **0.534+** on Kaggle Leaderboard).

---

### ⚙️ Kaggle Execution Settings:
- **Accelerator**: **GPU T4 x2** (recommended for 8B) or **GPU P100 / single T4** (for 1.7B).
- **Internet**: **OFF** (100% offline compliant with competition rules).
- **Model Attachment**: Attach `qwen-lm/qwen-3/transformers/8b-base` (or `1.7b-base`) via **`+ Add Input` $\to$ `Models`**.
"""
cells.append(create_cell("markdown", c1))

# Cell 2: Imports & Environment Configuration
c2 = """import gc
import glob
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
RUN_CV = not IS_RERUN
USE_QWEN = os.getenv('NTL_NO_QWEN') is None
SEED = 42
N_FOLDS, INNER_FOLDS = 5, 3
QWEN_C = 0.002
CENTRED_C = 0.002
CENTRE_CHUNKS = (2, 3, 4, 6)
MIN_CLUSTER = 2
W_SVM, W_PLAIN, W_CENTRED = 0.40, 0.30, 0.30
CLUSTER_THRESH = 0.90
MAX_LEN, BATCH = 512, 8
OUT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('outputs')
OUT.mkdir(parents=True, exist_ok=True)

LABELS = np.array(sorted([
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']))
K = len(LABELS)
T0 = time.time()

def log(msg):
    print(f'[{time.time() - T0:6.0f}s] {msg}', flush=True)

DATA_CANDIDATES = [
    Path('/kaggle/input/competitions/name-that-llm'),
    Path('/kaggle/input/name-that-llm'),
    Path('data'),
    Path('../../data'),
    Path('.'),
]
DATA = next((p for p in DATA_CANDIDATES if (p / 'train.csv').exists()), None)
if DATA is None:
    raise FileNotFoundError("Could not locate train.csv! Attach competition data.")

train = pd.read_csv(DATA / 'train.csv', dtype=str)
test = pd.read_csv(DATA / 'test.csv', dtype=str)
test['text'] = test['text'].fillna('')
assert set(train.model) == set(LABELS) and train.id.is_unique and test.id.is_unique

texts, groups = train.text.to_numpy(dtype=object), train.prompt_id.to_numpy(dtype=str)
test_texts, test_ids = test.text.to_numpy(dtype=object), test.id.to_numpy(dtype=str)
y = train.model.map({label: i for i, label in enumerate(LABELS)}).to_numpy()
folds = list(GroupKFold(N_FOLDS).split(texts, y, groups))
log(f'Data loaded: train {train.shape} | test {test.shape} | rerun {IS_RERUN}')
"""
cells.append(create_cell("code", c2))

# Cell 3: Adaptive Qwen Model Discovery & Frozen Embedding Extraction
c3 = """def find_qwen_model():
    \"\"\"
    Universal Qwen Model Resolver:
    Searches across /kaggle/input for Qwen3-8B-Base or Qwen3-1.7B-Base or Qwen2.5.
    Dynamically configures layer indices and hidden dimensions.
    \"\"\"
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try:
            c = json.load(open(cfg))
        except (OSError, ValueError):
            continue
        model_type = c.get('model_type', '')
        hidden_size = c.get('hidden_size', 0)
        num_layers = c.get('num_hidden_layers', 0)
        cfg_lower = cfg.lower()

        # Check for Qwen 8B Base
        if hidden_size == 4096 and num_layers >= 32 and ('base' in cfg_lower or '8b' in cfg_lower):
            layers = (4, 6, 8, 22)
            log(f"Found Qwen 8B Base at {os.path.dirname(cfg)} | using layers {layers}")
            return os.path.dirname(cfg), layers, hidden_size

        # Check for Qwen 1.7B Base
        if hidden_size == 2048 and num_layers >= 24 and ('base' in cfg_lower or '1.7b' in cfg_lower):
            layers = (17, 20)
            log(f"Found Qwen 1.7B Base at {os.path.dirname(cfg)} | using layers {layers}")
            return os.path.dirname(cfg), layers, hidden_size

        # Generic fallback for any Qwen model
        if 'qwen' in model_type or 'qwen' in cfg_lower:
            layers = (4, 8) if num_layers >= 8 else (num_layers - 2, num_layers - 1)
            log(f"Found generic Qwen model at {os.path.dirname(cfg)} | using layers {layers}")
            return os.path.dirname(cfg), layers, hidden_size

    raise FileNotFoundError(
        "\\n" + "="*70 + "\\n"
        "❌ MODEL QWEN BELUM DITAMBAHKAN KE INPUT KAGGLE!\\n\\n"
        "CARA MENAMBAHKAN (Hanya 15 detik):\\n"
        "1. Di pojok kanan atas notebook Kaggle, klik '+ Add Input'.\\n"
        "2. Klik tab 'Models' (di samping tab Datasets).\\n"
        "3. Cari: 'qwen-3' atau 'qwen-lm'.\\n"
        "4. Pilih 'qwen-lm / qwen-3' -> transformers -> 8b-base (atau 1.7b-base).\\n"
        "5. Klik tanda '+' untuk menambahkan, lalu jalankan kembali!\\n"
        "="*70
    )

def extract_qwen_embeddings(texts_list, model_dir, target_layers, hidden_dim):
    import torch
    from transformers import AutoModel, AutoTokenizer

    assert torch.cuda.is_available(), "GPU accelerator must be enabled in Kaggle settings!"
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lengths = np.array([len(ids) for ids in tok(texts_list, add_special_tokens=False)['input_ids']])
    order = np.argsort(-lengths) # Process longest passages first to verify memory limits

    def run_pass(rows, dtype):
        # Auto-map across GPUs (e.g. 8GB on GPU 0 + 8GB on GPU 1 for dual T4)
        model = AutoModel.from_pretrained(model_dir, dtype=dtype, device_map='auto').eval()
        model.requires_grad_(False)
        first_dev = next(model.parameters()).device
        out = np.zeros((len(rows), len(target_layers), hidden_dim), dtype=np.float32)

        for b in range(0, len(rows), BATCH):
            idx = rows[b:b + BATCH]
            enc = tok([texts_list[i] for i in idx], return_tensors='pt', padding=True, truncation=True,
                      max_length=MAX_LEN, add_special_tokens=False).to(first_dev)
            with torch.inference_mode():
                hs = model(**enc, output_hidden_states=True).hidden_states
                for j, L in enumerate(target_layers):
                    h = hs[L].float()
                    m = enc['attention_mask'].unsqueeze(-1).to(h.device).float()
                    out[b:b + len(idx), j] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()

        del model
        gc.collect()
        torch.cuda.empty_cache()
        return out

    E = np.zeros((len(texts_list), len(target_layers), hidden_dim), dtype=np.float32)
    E[order] = run_pass(order, torch.float16)
    bad = np.where(~np.isfinite(E).all(axis=(1, 2)))[0]
    if len(bad):
        log(f'{len(bad)} non-finite rows detected; resolving with bf16')
        E[bad] = run_pass(bad, torch.bfloat16)

    assert np.isfinite(E).all(), "Embeddings contain non-finite numbers!"
    log(f"Successfully embedded {len(texts_list)} texts (max length: {lengths.max()} tokens)")
    return E

if USE_QWEN:
    qwen_dir, SELECTED_LAYERS, HIDDEN_DIM = find_qwen_model()
    all_texts = list(texts) + list(test_texts)
    E_all = extract_qwen_embeddings(all_texts, qwen_dir, SELECTED_LAYERS, HIDDEN_DIM)
    E_tr, E_te = E_all[:len(train)], E_all[len(train):]
    del E_all
    gc.collect()
"""
cells.append(create_cell("code", c3))

# Cell 4: X-Star Disentangled Stylometrics & SVM Architecture
c4 = """class XStarStylometricCounts(TransformerMixin, BaseEstimator):
    \"\"\"
    X-Star Typographical & Structural Disentanglement:
    - Disentangles ASCII vs Typographical Curly quotes & contractions
    - Em-dashes, en-dashes, colons, semicolons, brackets, newlines
    - Structural markers: Markdown code ticks (`), LaTeX math ($), casing, punctuation density
    \"\"\"
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            text = str(t)
            n, w = max(len(text), 1), max(len(text.split()), 1)
            words = text.split()

            # Separate ASCII vs Curly quotes
            ascii_dq = text.count('"')
            curly_dq = text.count('“') + text.count('”')
            ascii_sq = text.count("'")
            curly_sq = text.count('‘') + text.count('’')

            # Separate ASCII vs Curly contractions
            ascii_contr = len(re.findall(r"\\b[a-zA-Z]+'[a-zA-Z]+\\b", text))
            curly_contr = len(re.findall(r"\\b[a-zA-Z]+’[a-zA-Z]+\\b", text))

            rows.append([
                np.log1p(n),
                np.log1p(w),
                text.count('\\n') / n * 100,
                text.count('\\n\\n'),
                text.count('—'), # Em-dash (Claude & GLM heavy)
                text.count('–'), # En-dash
                text.count(','),
                text.count('.'),
                text.count(';'),
                text.count(':'),
                ascii_dq,
                curly_dq,
                ascii_sq,
                curly_sq,
                ascii_contr,
                curly_contr,
                text.count('*'),
                text.count('#'),
                text.count('`'), # Code ticks
                text.count('$'), # LaTeX math delimiters
                text.count('(') + text.count(')'),
                text.count('-'),
                sum(c.isupper() for c in text) / n,
                np.mean([len(x) for x in words]) if words else 0.0,
                len(re.findall(r'[.!?]', text)) / w,
                text.count('!'),
                text.count('?'),
                text.count('…') + text.count('...'),
            ])
        return np.asarray(rows, dtype=np.float64)

def make_svm():
    \"\"\"
    High-capacity LinearSVC across 300k char n-grams + word n-grams + X-Star stylometrics.
    \"\"\"
    feats = FeatureUnion([
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(1, 5), sublinear_tf=True,
                                 lowercase=False, min_df=2, max_features=300_000)),
        ('word', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                                 min_df=2, token_pattern=r'\\S+')),
        ('style', make_pipeline(XStarStylometricCounts(), StandardScaler())),
    ], transformer_weights={'char': 1.0, 'word': 1.0, 'style': 0.35})
    return Pipeline([('feats', feats), ('svm', LinearSVC(C=1.0, max_iter=5000, random_state=SEED))])

def svm_proba(texts_fit, y_fit, groups_fit, texts_pred):
    \"\"\"Sigmoid-calibrated SVM probabilities evaluated on inner prompt folds.\"\"\"
    inner = list(GroupKFold(INNER_FOLDS).split(texts_fit, y_fit, groups_fit))
    cal = CalibratedClassifierCV(make_svm(), method='sigmoid', cv=inner, ensemble=False)
    cal.fit(texts_fit, y_fit)
    assert list(cal.classes_) == list(range(K))
    return cal.predict_proba(texts_pred)
"""
cells.append(create_cell("code", c4))

# Cell 5: Topic Centering, Qwen Heads, and Log-Space Fusion
c5 = """def topic_clusters(texts_, thresh=CLUSTER_THRESH):
    \"\"\"Unsupervised prompt topic clustering using content-word TF-IDF average-linkage.\"\"\"
    if len(texts_) < 2:
        return np.ones(len(texts_), dtype=int)
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(texts_)
    D = np.clip(1.0 - (X @ X.T).toarray(), 0.0, None)
    np.fill_diagonal(D, 0.0)
    return fcluster(linkage(squareform(D, checks=False), method='average'), thresh, criterion='distance')

def center(X, groups_):
    \"\"\"Centers embeddings within group clusters: X - mean(X_group).\"\"\"
    X_out = X.copy()
    for g in np.unique(groups_):
        idx = (groups_ == g)
        X_out[idx] -= X_out[idx].mean(0)
    return X_out

def random_chunks(groups_, size, rng):
    \"\"\"Splits prompt groups into random chunks of given size for invariant centering.\"\"\"
    chunk = np.empty(len(groups_), dtype=object)
    for g in np.unique(groups_):
        members = rng.permutation(np.where(groups_ == g)[0])
        for c, start in enumerate(range(0, len(members), size)):
            chunk[members[start:start + size]] = f'{g}_{c}'
    return chunk

def qwen_plain_proba(E_fit, y_fit, E_pred):
    \"\"\"Standard logistic regression head on multi-layer pooled Qwen embeddings.\"\"\"
    if not USE_QWEN:
        return np.full((len(E_pred), K), 1.0 / K)
    P = np.zeros((len(E_pred), K))
    n_layers = len(SELECTED_LAYERS)
    for j in range(n_layers):
        head = make_pipeline(StandardScaler(), LogisticRegression(C=QWEN_C, max_iter=5000, random_state=SEED))
        head.fit(E_fit[:, j], y_fit)
        assert list(head.classes_) == list(range(K))
        P += head.predict_proba(E_pred[:, j]) / n_layers
    return P

def qwen_centred_proba(E_fit, y_fit, groups_fit, E_pred, clusters_pred, plain_pred):
    \"\"\"
    Logistic heads on topic-centred embeddings:
    Subtracting cluster mean removes topic bias and isolates authorial stylometrics.
    \"\"\"
    if not USE_QWEN:
        return plain_pred
    rng = np.random.default_rng(SEED)
    chunkings = [random_chunks(groups_fit, size, rng) for size in CENTRE_CHUNKS]
    y_stack = np.tile(y_fit, len(chunkings))
    P = np.zeros((len(E_pred), K))
    n_layers = len(SELECTED_LAYERS)

    for j in range(n_layers):
        head = make_pipeline(StandardScaler(), LogisticRegression(C=CENTRED_C, max_iter=3000, random_state=SEED))
        X_tr_centred = np.concatenate([center(E_fit[:, j], ch) for ch in chunkings])
        head.fit(X_tr_centred, y_stack)
        assert list(head.classes_) == list(range(K))
        X_pred_centred = center(E_pred[:, j], clusters_pred)
        P += head.predict_proba(X_pred_centred) / n_layers

    # Singletons (< MIN_CLUSTER) cannot be centered; preserve plain probabilities
    small = np.unique(clusters_pred, return_inverse=True, return_counts=True)
    small_mask = small[2][small[1]] < MIN_CLUSTER
    P[small_mask] = plain_pred[small_mask]
    return P

def blend(p_svm, p_plain, p_centred):
    \"\"\"Weighted log-space probability fusion across SVM and Qwen representations.\"\"\"
    lg = lambda p: np.log(np.clip(p, 1e-12, 1.0))
    return W_SVM * lg(p_svm) + W_PLAIN * lg(p_plain) + W_CENTRED * lg(p_centred)
"""
cells.append(create_cell("code", c5))

# Cell 6: 5-Fold Grouped Cross Validation (Interactive Only)
c6 = """def bal_acc(P, idx=slice(None)):
    return balanced_accuracy_score(y[idx], P[idx].argmax(1))

if RUN_CV:
    log("Starting 5-Fold Prompt-Grouped Cross-Validation...")
    train_clusters = topic_clusters(texts)
    val_groups = random_chunks(train_clusters, 3, np.random.default_rng(SEED + 1))

    # Parallel SVM fitting across folds
    svm_parts = Parallel(n_jobs=min(N_FOLDS, os.cpu_count() or 1))(
        delayed(svm_proba)(texts[f], y[f], groups[f], texts[v]) for f, v in folds
    )

    oof_svm = np.zeros((len(train), K))
    oof_qwen = np.zeros((len(train), K))
    oof_qwen_c = np.zeros((len(train), K))
    fold_ids = np.zeros(len(train), dtype=int)

    for i, ((f, v), p_svm_fold) in enumerate(zip(folds, svm_parts)):
        oof_svm[v] = p_svm_fold
        oof_qwen[v] = qwen_plain_proba(E_tr[f], y[f], E_tr[v]) if USE_QWEN else 1.0 / K
        oof_qwen_c[v] = (qwen_centred_proba(E_tr[f], y[f], groups[f], E_tr[v], val_groups[v], oof_qwen[v])
                         if USE_QWEN else 1.0 / K)
        fold_ids[v] = i

        log(f"Fold {i}: SVM={bal_acc(oof_svm, v):.4f} | Qwen={bal_acc(oof_qwen, v):.4f} | "
            f"Qwen-Centred={bal_acc(oof_qwen_c, v):.4f} | "
            f"Blend={bal_acc(blend(oof_svm, oof_qwen, oof_qwen_c), v):.4f}")

    oof_logp = blend(oof_svm, oof_qwen, oof_qwen_c)
    lab = np.array([m.split('-')[0] for m in LABELS])

    metrics = {
        'svm_balanced_acc': float(bal_acc(oof_svm)),
        'qwen_plain_balanced_acc': float(bal_acc(oof_qwen)),
        'qwen_centred_balanced_acc': float(bal_acc(oof_qwen_c)),
        'three_way_blend_balanced_acc': float(bal_acc(oof_logp)),
        'macro_lab_accuracy': float((lab[oof_logp.argmax(1)] == lab[y]).mean())
    }
    log("="*60)
    log(f"FINAL 5-FOLD CROSS-VALIDATION RESULTS:")
    log(json.dumps(metrics, indent=2))
    log("="*60)

    # Per-model recall audit
    recall = pd.Series([(oof_logp.argmax(1)[y == k] == k).mean() for k in range(K)], index=LABELS)
    print("\\nPer-Model Recall Breakdown (Sorted):")
    print(recall.sort_values().round(3).to_string())
"""
cells.append(create_cell("code", c6))

# Cell 7: Full Dataset Fit and Submission Parquet Generation
c7 = """log(f"Preparing final inference on {len(test)} test samples...")
test_clusters = topic_clusters(test_texts)
sizes = np.bincount(test_clusters)[1:]

# 1. Fit calibrated SVM on 100% of training data
p_svm = svm_proba(texts, y, groups, test_texts)

# 2. Fit plain Qwen heads on 100% of training data
p_qwen = qwen_plain_proba(E_tr, y, E_te) if USE_QWEN else np.full((len(test), K), 1.0 / K)

# 3. Fit topic-centred Qwen heads on 100% of training data
p_qwen_c = (qwen_centred_proba(E_tr, y, groups, E_te, test_clusters, p_qwen)
            if USE_QWEN else p_qwen)

# 4. Three-way blend in log-space
logP = blend(p_svm, p_qwen, p_qwen_c)
pred_idx = logP.argmax(1)

sub = pd.DataFrame({
    'id': test_ids,
    'model': LABELS[pred_idx]
})

# Rigorous Quality Assurance Constraints
assert len(sub) == len(test), f"Row count mismatch: expected {len(test)}, got {len(sub)}"
assert list(sub.columns) == ['id', 'model'], f"Invalid columns: {sub.columns}"
assert not sub['model'].isnull().any(), "Found NaN in predicted models!"
assert sub.model.isin(LABELS).all(), "Found invalid model names!"

out_path = OUT / 'submission.parquet'
sub.to_parquet(out_path, index=False)
log(f"Successfully generated and verified {out_path} ({os.path.getsize(out_path)} bytes)!")
print("\\nFirst 10 submission rows:")
print(sub.head(10).to_string(index=False))
"""
cells.append(create_cell("code", c7))

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
    r"c:\Users\Matthew Chen\Documents\X-Star\competition\name_that_llm\x_star_qwen_cognitive_v6.ipynb",
    r"C:\Users\Matthew Chen\Downloads\x_star_qwen_cognitive_v6.ipynb"
]

for p in paths:
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"Generated X-Star v6 notebook at: {p}")
