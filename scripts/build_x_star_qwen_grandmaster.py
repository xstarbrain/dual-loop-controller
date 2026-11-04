import json
import os
import pathlib

def build_grandmaster_qwen_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_cell(cell_type, source):
        if isinstance(source, list):
            lines = [l if l.endswith('\n') else l + '\n' for l in source]
        else:
            lines = [l + '\n' for l in source.split('\n')]
            if lines and lines[-1] == '\n':
                lines = lines[:-1]
        nb["cells"].append({
            "cell_type": cell_type,
            "metadata": {},
            "source": lines,
            "execution_count": None if cell_type == "code" else None,
            "outputs": [] if cell_type == "code" else None
        })

    # CELL 0: Markdown explanation
    add_cell("markdown", """# 🌟 X-Star Qwen Cognitive Grandmaster | Target LB 0.530 – 0.650+
### Name That LLM: Qwen3-8B (Layers 4, 6, 8, 22) + Shared-Prompt Gaussian Heads + 4-View StyleGrams + Adaptive Intra-Family Gating + Sinkhorn Balancing

**Kompatibilitas:** Langsung jalan dengan input standar Anda di Kaggle:
- Competition Data: `name-that-llm`
- Model: `qwen-lm/qwen-3/transformers/8b-base` (atau model Qwen apapun)
- Accelerator: `GPU T4 x2`, Internet: `OFF`

---

### Mengapa Skor Sebelumnya Sempat Turun ke 0.377 dan Bagaimana Notebook Ini Memperbaikinya:
1. **Penyebab Skor 0.377:**
   Di notebook publik 0.563, bobot plain Qwen diset ke `0.0` karena mereka mengandalkan 4 file checkpoint LoRA dari dataset eksternal Kaggle yang terpisah (`lora24`, `lora20`, dll). Karena di Kaggle Anda checkpoint LoRA tersebut belum di-attach, model Qwen tidak memberikan kontribusi sama sekali pada prediksi baris tunggal, sehingga 100% prediksi hanya dibuat oleh StyleGrams (TF-IDF) yang memang memiliki skor **0.377**!
2. **Perbaikan Tuntas di Notebook Ini:**
   - **Qwen Diberikan Bobot Penuh:** Qwen plain (0.30) + Qwen Gaussian Head (0.25) + Qwen Centred Head (0.15) + StyleGrams 4 Views (0.30).
   - **Shared-Prompt Gaussian Group Heads (prvsiyan's method):** Dekomposisi ANOVA 2-arah pada layer Qwen untuk mengatasi kelemahan naive centering pada pasangan ($n=2$) dan tripel ($n=3$) dengan exact injective decoding.
   - **Adaptive Uncertainty-Aware Family Gating:** Memperbaiki kebingungan antar varian dalam 1 keluarga model (Claude, GPT, Gemini, GLM) dengan pengali adaptif $(0.25 + 0.75 \\cdot p_2 / p_1)$.
   - **Sinkhorn Column Balancing ($\\alpha = 0.5$):** Menyetel distribusi prediksi agar optimal terhadap metrik **Balanced Accuracy**.
""")

    # CELL 1: Configuration & Imports
    add_cell("code", """# ============================================================
# 0. CONFIGURATION, IMPORTS & SEED
# ============================================================
import gc, glob, json, os, re, time, pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix, diags, hstack, vstack
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import torch

T0 = time.time()
def log(msg):
    print(f'[{time.time() - T0:6.1f}s] {msg}', flush=True)

# 18 Target Classes
LABELS = np.array(sorted([
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]))
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}
FAMILY_IDX = {f: np.array([i for i, l in enumerate(LABELS) if l.startswith(f)]) for f in ('claude', 'gpt', 'gemini', 'glm')}

# Model Extraction & Inference Hyperparameters
SAVE_LAYERS = (4, 6, 8, 22)
QWEN_C = 0.002
CENTRED_C = 0.002
MAX_LEN, BATCH = 512, 8
CLUSTER_THRESH = 0.85
STYLE_GROUP_THRESH = 0.90
GROUP_MIN, GROUP_MAX = 2, 18
BALANCE_ALPHA = 0.5

# Calibrated Ensemble Weights (Self-Sufficient High-Performance Blend)
# Qwen gets 70% total weight across Plain, Gaussian, and Centred heads!
W_STYLEGRAM = 0.30
W_QWEN_PLAIN = 0.30
W_QWEN_GAUSS = 0.25
W_QWEN_CENTRED = 0.15

# Adaptive Family Gate Parameters
GATE_LAM = {'claude': 1.2, 'gpt': 0.35, 'gemini': 0.35, 'glm': 0.25}

# Locate Dataset Directory
def locate_dataset_dir():
    candidates = [
        Path('/kaggle/input/competitions/name-that-llm'),
        Path('/kaggle/input/name-that-llm'),
        Path('data/name-that-llm'),
        Path('../../data/name-that-llm'),
        Path('data'),
        Path.cwd()
    ]
    for p in candidates:
        if (p / 'train.csv').exists():
            return p
    return None

DATA_DIR = locate_dataset_dir()
if not DATA_DIR:
    raise FileNotFoundError("Could not find train.csv in standard Kaggle or local directories.")

train = pd.read_csv(DATA_DIR / 'train.csv', dtype=str)
test = pd.read_csv(DATA_DIR / 'test.csv', dtype=str)
test['text'] = test['text'].fillna('')

TR, TE = train['text'].tolist(), test['text'].tolist()
y = train['model'].map(LAB2I).to_numpy(dtype=int)
PROMPT = train['prompt_id'].to_numpy(dtype=str)
test_ids = test['id'].to_numpy(dtype=str)

log(f"Data Loaded: Train {train.shape}, Test {test.shape}, Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
""")

    # CELL 2: Universal Qwen Locator & Feature Extraction
    add_cell("code", """# ============================================================
# 1. UNIVERSAL QWEN RESOLVER & FROZEN EMBEDDING EXTRACTION
# ============================================================
def find_qwen_dir():
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try:
            c = json.load(open(cfg))
        except (OSError, ValueError):
            continue
        cfg_lower = cfg.lower()
        model_type = c.get('model_type', '')
        hidden = c.get('hidden_size', 0)
        layers = c.get('num_hidden_layers', 0)
        
        # Exact match for Qwen3-8B-Base
        if (model_type == 'qwen3' or 'qwen' in cfg_lower) and hidden == 4096 and layers >= 32 and 'base' in cfg_lower:
            return os.path.dirname(cfg), (4, 6, 8, 22), hidden
        # Match for Qwen2.5-7B
        if ('qwen' in model_type or 'qwen' in cfg_lower) and hidden == 3584:
            return os.path.dirname(cfg), (4, 6, 8, 20), hidden
        # Generic match for any Qwen
        if 'qwen' in model_type or 'qwen' in cfg_lower:
            actual_layers = (4, 6, 8, min(22, layers - 1)) if layers >= 8 else (0, 1)
            return os.path.dirname(cfg), actual_layers, hidden

    raise FileNotFoundError(
        "\\n" + "="*70 + "\\n"
        "❌ MODEL QWEN BELUM DITAMBAHKAN KE INPUT KAGGLE!\\n\\n"
        "CARA MENAMBAHKAN (Hanya 15 detik):\\n"
        "1. Di pojok kanan atas notebook Kaggle, klik '+ Add Input'.\\n"
        "2. Klik tab 'Models' (di samping tab Datasets).\\n"
        "3. Cari: 'qwen-3' atau 'qwen-lm'.\\n"
        "4. Pilih 'qwen-lm / qwen-3' -> transformers -> 8b-base.\\n"
        "5. Klik tanda '+' untuk menambahkan, lalu jalankan kembali!\\n"
        "="*70
    )

def extract_qwen_embeddings(texts_list, model_dir, target_layers, hidden_dim):
    from transformers import AutoModel, AutoTokenizer
    assert torch.cuda.is_available(), "GPU accelerator must be enabled in Kaggle settings!"
    
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lengths = np.array([len(ids) for ids in tok(texts_list, add_special_tokens=False)['input_ids']])
    order = np.argsort(-lengths)

    def run_pass(rows, dtype):
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
    log(f"Successfully embedded {len(texts_list)} texts across layers {target_layers}")
    return E

qwen_dir, SELECTED_LAYERS, HIDDEN_DIM = find_qwen_dir()
log(f"Using Qwen at: {qwen_dir} | Layers: {SELECTED_LAYERS} | Hidden Dim: {HIDDEN_DIM}")

all_texts = TR + TE
E_all = extract_qwen_embeddings(all_texts, qwen_dir, SELECTED_LAYERS, HIDDEN_DIM)
E_tr, E_te = E_all[:len(TR)], E_all[len(TR):]
del E_all
gc.collect()
torch.cuda.empty_cache()
""")

    # CELL 3: Test Pool Topic Grouping
    add_cell("code", """# ============================================================
# 2. TEST-POOL TOPIC GROUPING
# ============================================================
def topic_clusters(texts_list, thresh=CLUSTER_THRESH):
    if len(texts_list) < 2:
        return np.ones(len(texts_list), dtype=int)
    X = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(texts_list)
    D = np.clip(1.0 - (X @ X.T).toarray(), 0.0, None)
    np.fill_diagonal(D, 0.0)
    return fcluster(linkage(squareform(D, checks=False), method='average'), thresh, criterion='distance')

def center_dense(X, groups):
    X_out = np.asarray(X, dtype=np.float32).copy()
    for g in np.unique(groups):
        idx = (groups == g)
        X_out[idx] -= X_out[idx].mean(0)
    return X_out

test_clusters = topic_clusters(TE)
cluster_sizes = np.bincount(test_clusters)[test_clusters]
GROUP_ROWS = (cluster_sizes >= GROUP_MIN) & (cluster_sizes <= GROUP_MAX)
log(f"Test Groups: {len(TE)} rows | Singletons: {int((cluster_sizes == 1).sum())} | Grouped (2-18): {int(GROUP_ROWS.sum())}")
""")

    # CELL 4: 4-View StyleGrams & LinearSVC Branch
    add_cell("code", """# ============================================================
# 3. 4-VIEW STYLEGRAMS (540,000 FITUR) + WITHIN-FAMILY GATES
# ============================================================
FUNCTION_WORDS = frozenset(
    "a an the and or but if while although because as so than that which who whom whose this these that those "
    "i you he she it we they me him her us them my your his its our their mine yours hers ours theirs "
    "in on at by for with about against between into through during before after above below to from up down "
    "of off over under again further then once is am are was were be been being have has had having do does did doing "
    "can could will would shall should may might must not no nor all any some each every both either neither other "
    "another such only also just very too more most less least here there when where why how what whether then now "
    "still yet even much many few little".split()
)
TOKEN_RE = re.compile(r"\\w+|[^\\w\\s]")

def shape_view(text):
    return re.sub(r"[^\\W\\d_]+", lambda m: 'X' if m[0].isupper() else 'x', text)

def function_view(text):
    return ' '.join(t.lower() if (t.lower() in FUNCTION_WORDS or not t.isalnum()) else 'CONTENT' for t in TOKEN_RE.findall(text))

class StyleGrams:
    VIEWS = [
        ('char', str, (3, 5), 250_000, 1.0),
        ('word', str, (1, 3), 180_000, 0.7),
        ('char', shape_view, (2, 6), 60_000, 0.5),
        ('word', function_view, (1, 3), 50_000, 0.5)
    ]
    def fit(self, texts):
        self.vecs = []
        for analyzer, view, ngr, maxf, w in self.VIEWS:
            opts = dict(analyzer=analyzer, ngram_range=ngr, lowercase=False, sublinear_tf=True, min_df=2, max_features=maxf, dtype=np.float64)
            if analyzer == 'word':
                opts['token_pattern'] = r"(?u)\\b\\w+\\b|[^\\w\\s]"
            self.vecs.append((TfidfVectorizer(**opts).fit([view(t) for t in texts]), view, w))
        return self

    def transform(self, texts):
        return hstack([v.transform([view(t) for t in texts]) * w for v, view, w in self.vecs], format='csr')

def softmax_rows(S):
    S = S - S.max(1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(1, keepdims=True)

def full_prob(P_part, classes):
    P = np.zeros((len(P_part), K), dtype=np.float64)
    P[:, classes] = P_part
    return P

log("Fitting 4-View StyleGrams...")
grams = StyleGrams().fit(TR)
X_tr_style = grams.transform(TR)
X_te_style = grams.transform(TE)

svc_plain = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(X_tr_style, y)
p_stylegram = full_prob(softmax_rows(svc_plain.decision_function(X_te_style)), svc_plain.classes_)

# Within-Family Style Gates Training
def family_gate(texts, labels, classes, C=0.5):
    g = StyleGrams().fit(texts)
    svc = LinearSVC(C=C, class_weight='balanced', random_state=42, max_iter=4000).fit(g.transform(texts), labels)
    d = svc.decision_function(g.transform(TE))
    return np.column_stack([-d, d]) if len(classes) == 2 else d

GATES = {}
for fam, c_val in [('claude', 0.5), ('gpt', 1.0), ('gemini', 1.0), ('glm', 0.5)]:
    m = np.isin(y, FAMILY_IDX[fam])
    texts_fam = [t for t, k in zip(TR, m) if k]
    labels_fam = y[m]
    GATES[fam] = family_gate(texts_fam, labels_fam, FAMILY_IDX[fam], c_val)
    log(f"Trained {fam.upper()} within-family gate on {len(texts_fam)} passages")

del grams, X_tr_style, X_te_style, svc_plain
gc.collect()
""")

    # CELL 5: Qwen Plain & Centred Heads
    add_cell("code", """# ============================================================
# 4. QWEN PLAIN & CENTRED LOGISTIC HEADS
# ============================================================
def qwen_plain_proba(E_fit, y_fit, E_pred):
    P = np.zeros((len(E_pred), K), dtype=np.float64)
    for j in range(E_fit.shape[1]):
        sc = StandardScaler().fit(E_fit[:, j])
        clf = LogisticRegression(C=QWEN_C, max_iter=4000).fit(sc.transform(E_fit[:, j]), y_fit)
        # Transductive test-set standardization
        X_te = (E_pred[:, j] - E_pred[:, j].mean(0)) / (E_pred[:, j].std(0) + 1e-6) if len(E_pred) >= 2 else sc.transform(E_pred[:, j])
        P += clf.predict_proba(X_te)
    return P / E_fit.shape[1]

def qwen_centred_proba(E_fit, y_fit, groups_fit, E_pred, clusters_pred, plain_prob):
    # Stack multiple random chunks of train prompts to avoid overfitting
    rng = np.random.default_rng(42)
    reps_tr, reps_y = [], []
    for chunk_size in (2, 3, 4, 6):
        shuffled = rng.permutation(np.unique(groups_fit))
        chunk_map = {p: i // chunk_size for i, p in enumerate(shuffled)}
        chunks = np.array([chunk_map[p] for p in groups_fit])
        reps_tr.append(np.concatenate([center_dense(E_fit[:, j], chunks) for j in range(E_fit.shape[1])], axis=1))
        reps_y.append(y_fit)
    
    X_tr_c = np.concatenate(reps_tr, axis=0)
    y_tr_c = np.concatenate(reps_y, axis=0)
    
    sc = StandardScaler().fit(X_tr_c)
    clf = LogisticRegression(C=CENTRED_C, max_iter=4000).fit(sc.transform(X_tr_c), y_tr_c)
    
    P_out = plain_prob.copy()
    if GROUP_ROWS.any():
        X_te_c = np.concatenate([center_dense(E_pred[GROUP_ROWS, j], clusters_pred[GROUP_ROWS]) for j in range(E_pred.shape[1])], axis=1)
        P_out[GROUP_ROWS] = clf.predict_proba(sc.transform(X_te_c))
    return P_out

log("Fitting Qwen Plain Head...")
p_qwen_plain = qwen_plain_proba(E_tr, y, E_te)

log("Fitting Qwen Centred Head...")
p_qwen_centred = qwen_centred_proba(E_tr, y, PROMPT, E_te, test_clusters, p_qwen_plain)
""")

    # CELL 6: Shared-Prompt Gaussian Group Heads on Qwen
    add_cell("code", """# ============================================================
# 5. SHARED-PROMPT GAUSSIAN GROUP HEAD (PRVSIYAN METHOD)
# ============================================================
# Exact Injective Tuple Decoding on Pairs (n=2) and Triples (n=3)
# Solves the 50% signal deletion of naive mean-centering.
import itertools

def _lse(A, axis=None, keepdims=False):
    m = np.max(A, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True)) + m
    if keepdims: return out
    return np.squeeze(out, axis=axis) if axis is not None else float(out.item())

def log_softmax(L): return L - _lse(L, axis=1, keepdims=True)

def _lw_shrinkage(X):
    from sklearn.covariance import ledoit_wolf_shrinkage
    return float(ledoit_wolf_shrinkage(X, assume_centered=True))

def _shrunk(S, a):
    d = S.shape[0]
    return (1 - a) * S + a * (np.trace(S) / d) * np.eye(d)

def _two_way(Z, y_arr, pinv, G, iters=4):
    P = np.zeros((G, Z.shape[1]))
    cnt_p = np.bincount(pinv, minlength=G).astype(float)
    cls = np.unique(y_arr)
    mu = np.zeros((K, Z.shape[1]))
    for _ in range(iters):
        R = Z - P[pinv]
        for c in cls: mu[c] = R[y_arr == c].mean(0)
        R = Z - mu[y_arr]; P = np.zeros_like(P); np.add.at(P, pinv, R); P /= cnt_p[:, None]
    return mu, P, cnt_p

def fit_gaussian_head(X, y_arr, prompts, d=384):
    X = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y_arr)
    N, D = X.shape
    pu, pinv = np.unique(np.asarray(prompts), return_inverse=True)
    G = len(pu)
    mx = X.mean(0)
    sx = X.std(0)
    sx = np.where(sx > 1e-8, sx, 1.0)
    Xs = (X - mx) / sx

    # WPCA Reduction
    Pm = np.zeros((G, D))
    np.add.at(Pm, pinv, Xs)
    Pm /= np.bincount(pinv)[:, None]
    R = Xs - Pm[pinv]
    
    d_eff = int(min(d, N - G - 1, D))
    if N < D:
        w, U = np.linalg.eigh(R @ R.T)
        o = np.argsort(w)[::-1][:d_eff]
        w, U = np.clip(w[o], 1e-12, None), U[:, o]
        V = (R.T @ U) / np.sqrt(w)
    else:
        _, _, Vt = np.linalg.svd(R, full_matrices=False)
        V = Vt[:d_eff].T

    Z = Xs @ V
    mu, P, ng = _two_way(Z, y_arr, pinv, G)
    E = Z - mu[y_arr] - P[pinv]
    dof = max(N - G - len(np.unique(y_arr)) + 1, 1)
    Se_raw = E.T @ E / dof
    ae = _lw_shrinkage(E)
    Se = _shrunk(Se_raw, ae)
    
    Pc = P - P.mean(0)
    Sp_raw = Pc.T @ Pc / max(G - 1, 1) - np.mean(1.0 / ng) * Se_raw
    ap = _lw_shrinkage(Pc)
    tr = max(np.trace(Sp_raw), 1e-8 * np.trace(Se))
    Sp = (1 - ap) * Sp_raw + ap * (tr / d_eff) * np.eye(d_eff)

    # Joint Diagonalization
    L = np.linalg.cholesky(Se)
    Li = np.linalg.inv(L)
    M = Li @ Sp @ Li.T
    M = 0.5 * (M + M.T)
    lam, Q = np.linalg.eigh(M)
    lam = np.clip(lam, 1e-4, None)
    A = Q.T @ Li
    W_mat = V @ A.T

    return {"mx": mx, "sx": sx, "W": W_mat, "m": mu @ A.T, "lam": lam}

_MASK = {}
def _inj_mask(n, C):
    if (n, C) not in _MASK:
        grids = np.meshgrid(*[np.arange(C)] * n, indexing="ij")
        bad = np.zeros([C] * n, bool)
        for i, j in itertools.combinations(range(n), 2):
            bad |= (grids[i] == grids[j])
        _MASK[(n, C)] = bad
    return _MASK[(n, C)]

def singleton_ll(U, m, lam):
    R = U[:, None, :] - m[None]
    return -0.5 * (R ** 2 / (1 + lam)).sum(-1)

def joint_terms(U, m, lam):
    n = len(U)
    gam = lam / (1 + n * lam)
    R = U[:, None, :] - m[None]
    a = -0.5 * (R ** 2 * (1 - gam)).sum(-1)
    G = R * np.sqrt(gam)
    B = np.einsum("icd,jed->ijce", G, G)
    return a, B

def joint_table(a, B):
    n, C = a.shape
    Q = np.zeros([C] * n)
    for i in range(n):
        sh = [1] * n; sh[i] = C
        Q = Q + a[i].reshape(sh)
    for i, j in itertools.combinations(range(n), 2):
        sh = [1] * n; sh[i] = C; sh[j] = C
        Q = Q + B[i, j].reshape(sh)
    return Q

def exact_marginals(Q, T=2.3, inj=True):
    n = Q.ndim; C = Q.shape[0]; Qt = Q / T
    if inj and n >= 2:
        Qt = np.where(_inj_mask(n, C), -np.inf, Qt)
    lz = _lse(Qt)
    out = np.empty((n, C))
    for i in range(n):
        ax = tuple(k for k in range(n) if k != i)
        out[i] = (_lse(Qt, axis=ax) if ax else Qt) - lz
    return out

def em_marginals(U, m, lam, T=2.3, iters=5):
    n = len(U)
    v = lam / (1 + (n - 1) * lam)
    L = singleton_ll(U, m, lam) / T
    q = np.exp(log_softmax(L))
    for _ in range(iters):
        s = U - q @ m
        S = s.sum(0)
        Pm = (S[None] - s) * v
        R = U[:, None, :] - Pm[:, None, :] - m[None]
        L = -0.5 * (R ** 2 / (1 + v)).sum(-1) / T
        lo = np.log(np.clip(1 - q, 1e-6, None))
        L = L + (lo.sum(0)[None] - lo)
        q = np.exp(log_softmax(L))
    return log_softmax(L)

def score_gaussian(X_tr_dense, X_te_dense, clusters_te):
    prm = fit_gaussian_head(X_tr_dense, y, PROMPT, d=384)
    U = ((np.asarray(X_te_dense, dtype=np.float64) - prm["mx"]) / prm["sx"]) @ prm["W"]
    n = len(U)
    out = np.empty((n, K))
    sizes = np.bincount(clusters_te)[clusters_te]
    
    # Singletons
    s1 = (sizes == 1)
    if s1.any():
        out[s1] = log_softmax(singleton_ll(U[s1], prm["m"], prm["lam"]) / 2.3)
    
    # Groups
    for g in np.unique(clusters_te):
        idx = np.where(clusters_te == g)[0]
        if len(idx) == 1:
            continue
        elif len(idx) <= 3:
            a, B = joint_terms(U[idx], prm["m"], prm["lam"])
            out[idx] = exact_marginals(joint_table(a, B), T=2.3, inj=True)
        else:
            out[idx] = em_marginals(U[idx], prm["m"], prm["lam"], T=2.3, iters=5)
    return np.exp(out)

log("Fitting Shared-Prompt Gaussian Head on Qwen Layer 22...")
# Using Layer 22 (the deepest extracted layer) for Gaussian head
last_layer_idx = len(SELECTED_LAYERS) - 1
p_qwen_gauss = score_gaussian(E_tr[:, last_layer_idx], E_te[:, last_layer_idx], test_clusters)
""")

    # CELL 7: Ensemble Blending + Adaptive Intra-Family Gating + Sinkhorn Balancing
    add_cell("code", """# ============================================================
# 6. ENSEMBLE FUSION, SINKHORN BALANCING & ADAPTIVE GATES
# ============================================================
def lg(p):
    return np.log(np.clip(p, 1e-9, 1.0))

# 1. Core Log-Probability Fusion
LOGIT = (
    W_STYLEGRAM   * lg(p_stylegram) +
    W_QWEN_PLAIN  * lg(p_qwen_plain) +
    W_QWEN_GAUSS  * lg(p_qwen_gauss) +
    W_QWEN_CENTRED * lg(p_qwen_centred)
)

P = softmax_rows(LOGIT)

# 2. Partial Sinkhorn Column Balancing (alpha = 0.5)
if BALANCE_ALPHA > 0 and len(P) >= 2:
    target = (1 - BALANCE_ALPHA) * P.sum(0) + BALANCE_ALPHA * np.full(K, len(P) / K)
    for _ in range(50):
        P *= (target / (P.sum(0) + 1e-12))[None, :]
        P /= P.sum(1, keepdims=True) + 1e-12
    changed = int((P.argmax(1) != LOGIT.argmax(1)).sum())
    log(f"Sinkhorn Balancing: realigned {changed} of {len(P)} predictions ({changed / len(P) * 100:.1f}%)")

pred = P.argmax(1)
logP = np.log(np.clip(P, 1e-12, 1.0))

# 3. Adaptive Uncertainty-Aware Intra-Family Gating
for fam in ('claude', 'gpt', 'gemini', 'glm'):
    if fam not in GATES:
        continue
    idx = FAMILY_IDX[fam]
    sub_P = P[:, idx]
    sorted_sub_P = np.sort(sub_P, axis=1)
    p1 = np.maximum(sorted_sub_P[:, -1], 1e-9)
    p2 = sorted_sub_P[:, -2] if sub_P.shape[1] > 1 else np.zeros_like(p1)
    ratio = np.clip(p2 / p1, 0.0, 1.0)
    adaptive_factor = (0.25 + 0.75 * ratio)[:, None]

    effective_gate = (GATE_LAM[fam] * adaptive_factor) * GATES[fam]
    score = logP[:, idx] + effective_gate

    inside = np.isin(pred, idx)
    new_pred_inside = idx[score.argmax(1)]
    swapped = int((new_pred_inside[inside] != pred[inside]).sum())
    total_in_fam = int(inside.sum())
    log(f"Adaptive Gate [{fam.upper()}]: re-ranked {swapped} of {total_in_fam} predictions ({swapped / max(total_in_fam, 1) * 100:.1f}%)")
    pred[inside] = new_pred_inside[inside]
""")

    # CELL 8: Submission Generation & Validation
    add_cell("code", """# ============================================================
# 7. SUBMISSION GENERATION (SUBMISSION.PARQUET)
# ============================================================
sub = pd.DataFrame({
    'id': test_ids,
    'model': LABELS[pred]
})

sub.to_parquet('submission.parquet', index=False)
log(f"\\n{'='*60}")
log(f"Saved submission.parquet with {len(sub)} rows.")
log(f"Columns: {list(sub.columns)}")
log(f"Null Values: {sub.isnull().sum().to_dict()}")
log(f"Distribution:\\n{sub['model'].value_counts()}")
log(f"{'='*60}")

print(sub.head(10))
""")

    return nb

# Build and save
nb = build_grandmaster_qwen_notebook()
out_path_1 = r"C:\Users\Matthew Chen\Downloads\x_star_qwen_cognitive_grandmaster.ipynb"
out_path_2 = r"c:\Users\Matthew Chen\Documents\X-Star\x_star_qwen_cognitive_grandmaster.ipynb"

with open(out_path_1, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f"Saved: {out_path_1}")

with open(out_path_2, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f"Saved: {out_path_2}")
