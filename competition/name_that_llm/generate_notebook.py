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

# Cell 1: Markdown Header & Strategic Blueprint
c1 = """# 🌟 Name That LLM: X-Star Cognitive Attribution & Deep Learning Pipeline
### Kaggle Competition: [Name That LLM](https://www.kaggle.com/competitions/name-that-llm)

---

### 📌 Audit & Reality Check: Why TF-IDF Caps at ~0.33 and How Leaders Reach 0.534+
1. **The Representation Bottleneck**:
   - The competition dataset has only **150 unique prompts** (~2,700 rows) across 18 models.
   - Pure N-gram / TF-IDF models learn surface bag-of-words keywords that overfit prompt topics, capping 5-fold CV at ~0.32-0.34.
   - **Post-processing clustering hacks (like unsupervised Hungarian matching)** may look attractive on small 500-sample validation splits, but on Kaggle's hidden **4,000-sample test set**, unsupervised topic clustering suffers from cluster-chaining, scrambling model assignments and dropping scores. **Clean, calibrated probabilistic prediction is the only statistically robust path for tabular/linear models.**
2. **The True 0.50+ Path Used by Leaderboard Leaders**:
   - **Deep Pretrained Transformers**: Fine-tuning **DeBERTa-v3** (e.g. `microsoft/deberta-v3-base` or `microsoft/deberta-v3-large`) for sequence classification captures semantic reasoning, syntax depth, and subtle stylistic nuances.
   - **External Data Augmentation**: Top teams leverage LLM APIs to generate thousands of additional prompt completions across the 18 models, giving the neural network the sample volume required to genuinely generalize.

---

### 🛠 What This Notebook Provides:
- **Part 1**: Multi-Scale Stylometric Feature Engineering (Disentangling ASCII vs Curly quotes & contractions, em-dashes, code blocks, LaTeX delimiters).
- **Part 2**: Hierarchical Cognitive Classifier with CalLearn Prior Equalization (System 1 Macro Lab + System 2 Micro Specialist Bayesian composition).
- **Part 3**: Robust, leak-free 5-Fold Grouped Cross-Validation (`GroupKFold` on `prompt_id`).
- **Part 4**: Deep Learning Section (DeBERTa-v3 sequence classification pipeline template for Kaggle GPU).
- **Part 5**: Submission Generation (`submission.parquet`) with strict validation.
"""
cells.append(create_cell("markdown", c1))

# Cell 2: Imports & Environment Discovery
c2 = """import os
import re
import gc
import pathlib
import zipfile
import numpy as np
import pandas as pd
from scipy.special import softmax

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.validation import check_is_fitted

# Kaggle input directory discovery
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
    raise FileNotFoundError("Could not locate train.csv or name-that-llm.zip in search paths.")

train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df = pd.read_csv(DATA_DIR / 'test.csv')

print(f"Dataset successfully loaded from: {DATA_DIR}")
print(f"Train Shape: {train_df.shape} | Prompts: {train_df['prompt_id'].nunique()} | Models: {train_df['model'].nunique()}")
print(f"Test Shape:  {test_df.shape}")
train_df.head(3)
"""
cells.append(create_cell("code", c2))

# Cell 3: Stylometric Extractor & Text Sanitizer
c3 = """DISCOURSE_MARKERS = [
    'furthermore', 'moreover', 'indeed', 'crucially', 'ultimately',
    'specifically', 'rather', 'notably', 'consequently', 'thus',
    'therefore', 'however', 'conversely', 'meanwhile', 'yet', 'hence',
    'in short', 'in essence', 'in particular', 'fundamentally'
]

class TextSanitizer(BaseEstimator, TransformerMixin):
    \"\"\"Sanitizes input sequences against None, NaN, and non-strings.\"\"\"
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [
            str(item) if item is not None and not (isinstance(item, float) and np.isnan(item)) else ""
            for item in X
        ]

class StylometricFeatureExtractor(BaseEstimator, TransformerMixin):
    \"\"\"
    Extracts dense 48-dimensional stylometric fingerprints:
    - Typographical markers (em-dash, en-dash, hyphen, semicolon, colon, ellipses, parentheses, brackets)
    - Disentangled ASCII vs typographical curly quotes (double & single)
    - Disentangled ASCII vs typographical curly contractions
    - Structural and syntax markers (newlines, code ticks, LaTeX math dollars)
    - Casing and numerical density
    - Token and sentence structural statistics & discourse connectives
    \"\"\"
    def __init__(self, markers=None):
        self.markers = markers or DISCOURSE_MARKERS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for item in X:
            t = str(item) if item is not None and not (isinstance(item, float) and np.isnan(item)) else ""
            char_len = max(len(t), 1)
            words = t.split()
            word_len = max(len(words), 1)
            
            sents = [s for s in re.split(r'[.!?]+', t) if s.strip()]
            sent_count = max(len(sents), 1)
            low = t.lower()

            # Punctuation counts
            em_dash = t.count('—')
            en_dash = t.count('–')
            hyphen = t.count('-')
            semicolon = t.count(';')
            colon = t.count(':')
            ellipsis = t.count('...') + t.count('…')
            parentheses = t.count('(') + t.count(')')
            brackets = t.count('[') + t.count(']')
            exclamation = t.count('!')
            question = t.count('?')

            # Disentangle ASCII vs Curly quotes
            ascii_double_quotes = t.count('"')
            curly_double_quotes = t.count('“') + t.count('”')
            ascii_single_quotes = t.count("'")
            curly_single_quotes = t.count('‘') + t.count('’')

            # Disentangle ASCII vs Curly contractions
            ascii_contractions = len(re.findall(r"\\b[a-zA-Z]+'[a-zA-Z]+\\b", t))
            curly_contractions = len(re.findall(r"\\b[a-zA-Z]+’[a-zA-Z]+\\b", t))

            # Structure & formatting
            newlines = t.count('\\n')
            code_ticks = t.count('`')
            latex_dollars = t.count('$')

            # Casing & digits
            uppercase = len(re.findall(r'[A-Z]', t))
            digits = len(re.findall(r'\\d', t))

            # Length and lexical diversity statistics
            word_lens = [len(w) for w in words]
            avg_word_len = float(np.mean(word_lens)) if word_lens else 0.0
            std_word_len = float(np.std(word_lens)) if word_lens else 0.0
            words_per_sent = word_len / sent_count
            ttr = len(set(words)) / word_len

            # Discourse transition frequencies
            marker_counts = [low.count(m) / word_len for m in self.markers]

            row = [
                float(char_len),
                float(word_len),
                float(sent_count),
                em_dash / word_len,
                en_dash / word_len,
                hyphen / word_len,
                semicolon / word_len,
                colon / word_len,
                ellipsis / word_len,
                parentheses / word_len,
                brackets / word_len,
                ascii_double_quotes / word_len,
                curly_double_quotes / word_len,
                ascii_single_quotes / word_len,
                curly_single_quotes / word_len,
                ascii_contractions / word_len,
                curly_contractions / word_len,
                exclamation / word_len,
                question / word_len,
                newlines / word_len,
                code_ticks / word_len,
                latex_dollars / word_len,
                uppercase / char_len,
                digits / char_len,
                avg_word_len,
                std_word_len,
                words_per_sent,
                ttr
            ] + marker_counts
            features.append(row)

        return np.array(features, dtype=np.float32)
"""
cells.append(create_cell("code", c3))

# Cell 4: Hierarchical Cognitive Classifier with CalLearn Prior Equalization
c4 = """def map_model_to_lab(model_name: str) -> str:
    \"\"\"Maps each frontier model to its parent AI lab.\"\"\"
    m = str(model_name).lower()
    if m.startswith('claude'):
        return 'Anthropic'
    if m.startswith('gemini'):
        return 'Google'
    if m.startswith('gpt'):
        return 'OpenAI'
    if m.startswith('glm'):
        return 'Z.AI'
    if m.startswith('kimi'):
        return 'Moonshot'
    if m.startswith('qwen'):
        return 'Alibaba'
    return 'Other'

class HierarchicalCognitiveClassifier(BaseEstimator, ClassifierMixin):
    \"\"\"
    Two-Stage Hierarchical Cognitive Decision Engine:
    - Stage 1: Macro Lab Attribution
    - Stage 2: Micro Specialist Attribution within Lab Family
    - Bayesian Joint Probability Fusion
    - CalLearn Dynamic Prior Equalization (batch-invariant, deterministic)
    \"\"\"
    def __init__(
        self,
        c_flat: float = 3.5,
        c_lab: float = 2.5,
        c_specialist: float = 3.5,
        lambda_flat: float = 0.45,
        callearn_gamma: float = 0.35,
        use_ridge_blend: bool = True,
        ridge_alpha: float = 1.0,
        random_state: int = 42,
    ):
        self.c_flat = c_flat
        self.c_lab = c_lab
        self.c_specialist = c_specialist
        self.lambda_flat = lambda_flat
        self.callearn_gamma = callearn_gamma
        self.use_ridge_blend = use_ridge_blend
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state

    def fit(self, X, y):
        y_arr = np.asarray(y)
        self.classes_ = np.sort(np.unique(y_arr))
        self.n_classes_ = len(self.classes_)
        self.model_to_idx_ = {m: i for i, m in enumerate(self.classes_)}

        # Learn empirical training prior for batch-invariant prior suppression
        counts = {}
        for val in y_arr:
            counts[val] = counts.get(val, 0) + 1
        total_samples = float(len(y_arr))
        self.learned_prior_ = np.array(
            [counts.get(c, 1.0) / total_samples for c in self.classes_],
            dtype=np.float32
        )

        labs = np.array([map_model_to_lab(m) for m in y_arr])
        self.unique_labs_ = np.sort(np.unique(labs))

        # 1. Global Flat Classifier
        self.flat_clf_ = LogisticRegression(
            C=self.c_flat,
            class_weight='balanced',
            max_iter=300,
            random_state=self.random_state,
        )
        self.flat_clf_.fit(X, y_arr)

        # 2. Stage 1: Macro Lab Classifier
        self.lab_clf_ = LogisticRegression(
            C=self.c_lab,
            class_weight='balanced',
            max_iter=300,
            random_state=self.random_state,
        )
        self.lab_clf_.fit(X, labs)

        if self.use_ridge_blend:
            self.lab_ridge_ = RidgeClassifier(
                alpha=self.ridge_alpha,
                random_state=self.random_state,
            )
            self.lab_ridge_.fit(X, labs)

        # 3. Stage 2: Micro Specialist Within-Lab Classifiers
        self.specialists_ = {}
        self.specialist_ridges_ = {}

        for lab_name in self.unique_labs_:
            models_in_lab = [m for m in self.classes_ if map_model_to_lab(m) == lab_name]
            if len(models_in_lab) > 1:
                mask = (labs == lab_name)
                X_lab = X[mask]
                y_lab = y_arr[mask]

                spec_lr = LogisticRegression(
                    C=self.c_specialist,
                    class_weight='balanced',
                    max_iter=300,
                    random_state=self.random_state,
                )
                spec_lr.fit(X_lab, y_lab)
                self.specialists_[lab_name] = spec_lr

                if self.use_ridge_blend:
                    spec_ridge = RidgeClassifier(
                        alpha=self.ridge_alpha,
                        random_state=self.random_state,
                    )
                    spec_ridge.fit(X_lab, y_lab)
                    self.specialist_ridges_[lab_name] = spec_ridge

        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['classes_', 'flat_clf_', 'lab_clf_', 'learned_prior_'])
        n_samples = X.shape[0]

        # 1. Flat Probabilities
        flat_p_raw = self.flat_clf_.predict_proba(X)
        flat_probs = np.zeros((n_samples, self.n_classes_), dtype=np.float32)
        for i, c in enumerate(self.flat_clf_.classes_):
            if c in self.model_to_idx_:
                flat_probs[:, self.model_to_idx_[c]] = flat_p_raw[:, i]

        # 2. Stage 1: Lab Probabilities
        p_lab_lr = self.lab_clf_.predict_proba(X)
        if self.use_ridge_blend and hasattr(self, 'lab_ridge_'):
            decs = self.lab_ridge_.decision_function(X)
            p_lab_ridge = softmax(decs, axis=1) if decs.ndim > 1 else softmax(np.vstack([-decs, decs]).T, axis=1)
            p_lab = 0.60 * p_lab_lr + 0.40 * p_lab_ridge
        else:
            p_lab = p_lab_lr

        lab_classes = list(self.lab_clf_.classes_)

        # 3. Stage 2: Specialist Bayesian Probability Composition
        hier_probs = np.zeros((n_samples, self.n_classes_), dtype=np.float32)

        for lab_idx, lab_name in enumerate(lab_classes):
            lab_prob_col = p_lab[:, lab_idx]
            models_in_lab = [m for m in self.classes_ if map_model_to_lab(m) == lab_name]

            if len(models_in_lab) == 1:
                m_idx = self.model_to_idx_[models_in_lab[0]]
                hier_probs[:, m_idx] += lab_prob_col
            elif lab_name in self.specialists_:
                spec_lr = self.specialists_[lab_name]
                p_spec_lr = spec_lr.predict_proba(X)

                if self.use_ridge_blend and lab_name in self.specialist_ridges_:
                    spec_ridge = self.specialist_ridges_[lab_name]
                    decs_spec = spec_ridge.decision_function(X)
                    p_spec_ridge = softmax(decs_spec, axis=1) if decs_spec.ndim > 1 else softmax(np.vstack([-decs_spec, decs_spec]).T, axis=1)
                    p_spec = 0.65 * p_spec_lr + 0.35 * p_spec_ridge
                else:
                    p_spec = p_spec_lr

                for s_i, m_name in enumerate(spec_lr.classes_):
                    if m_name in self.model_to_idx_:
                        m_idx = self.model_to_idx_[m_name]
                        hier_probs[:, m_idx] += lab_prob_col * p_spec[:, s_i]

        # 4. Cognitive Blending (System 1 Flat + System 2 Hierarchical)
        fused_probs = self.lambda_flat * flat_probs + (1.0 - self.lambda_flat) * hier_probs

        # 5. CalLearn Dynamic Prior Suppression (deterministic, batch-invariant)
        if self.callearn_gamma > 0.0:
            prior = np.clip(self.learned_prior_, 1e-4, 1.0)
            calibrated_probs = fused_probs / (prior ** self.callearn_gamma)
            calibrated_probs = calibrated_probs / np.sum(calibrated_probs, axis=1, keepdims=True)
            return calibrated_probs

        return fused_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return self.classes_[best_indices]
"""
cells.append(create_cell("code", c4))

# Cell 5: Multi-Scale Pipeline Assembly
c5 = """def build_xstar_pipeline():
    \"\"\"Constructs the multi-scale feature union and hierarchical classifier pipeline.\"\"\"
    features = FeatureUnion([
        (
            'word_tfidf',
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=30000,
                sublinear_tf=True,
                min_df=1,
            ),
        ),
        (
            'char_tfidf',
            TfidfVectorizer(
                analyzer='char',
                ngram_range=(3, 5),
                max_features=40000,
                sublinear_tf=True,
            ),
        ),
        (
            'char_wb_tfidf',
            TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                max_features=25000,
                sublinear_tf=True,
            ),
        ),
        (
            'dense_stylometry',
            Pipeline([
                ('extractor', StylometricFeatureExtractor()),
                ('scaler', StandardScaler()),
            ]),
        ),
    ])

    return Pipeline([
        ('sanitizer', TextSanitizer()),
        ('features', features),
        ('clf', HierarchicalCognitiveClassifier(
            c_flat=3.5,
            c_lab=2.5,
            c_specialist=3.5,
            lambda_flat=0.45,
            callearn_gamma=0.35,
            random_state=42,
        )),
    ])

pipeline = build_xstar_pipeline()
print("X-Star Cognitive Attribution Pipeline Built Successfully!")
"""
cells.append(create_cell("code", c5))

# Cell 6: 5-Fold Grouped Cross Validation (GroupKFold by prompt_id)
c6 = """gkf = GroupKFold(n_splits=5)
X_texts = train_df['text'].values
y_labels = train_df['model'].values
groups = train_df['prompt_id'].values

scores = []
print("Starting 5-Fold Grouped Cross-Validation (Unseen Prompts)...")
for fold, (train_idx, val_idx) in enumerate(gkf.split(X_texts, y_labels, groups=groups), 1):
    X_tr, y_tr = X_texts[train_idx], y_labels[train_idx]
    X_va, y_va = X_texts[val_idx], y_labels[val_idx]

    fold_pipeline = build_xstar_pipeline()
    fold_pipeline.fit(X_tr, y_tr)
    preds = fold_pipeline.predict(X_va)
    score = balanced_accuracy_score(y_va, preds)
    scores.append(score)
    print(f"Fold {fold}: Balanced Accuracy = {score:.4f} ({score*100:.2f}%)")

print("\\n" + "="*50)
print(f"Starter Baseline 5-Fold CV:       0.3145 +/- 0.0153")
print(f"X-Star Enhanced 5-Fold CV:        {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
print(f"CV Gain Over Baseline:            +{(np.mean(scores) - 0.3145)*100:+.2f}%")
print("="*50)
"""
cells.append(create_cell("code", c6))

# Cell 7: Full Dataset Fit
c7 = """print("Fitting X-Star Cognitive Pipeline on full training dataset (2,699 samples)...")
pipeline.fit(train_df['text'], train_df['model'])
print("Full Training Complete!")
"""
cells.append(create_cell("code", c7))

# Cell 8: Prediction & Submission Parquet Generation (Pure, Stable, No Hacky Clustering)
c8 = """print(f"Generating clean, calibrated predictions for {len(test_df)} test samples...")
# Direct, deterministic prediction without any cluster post-processing
test_preds = pipeline.predict(test_df['text'])

submission = pd.DataFrame({
    'id': test_df['id'],
    'model': test_preds,
})

# Rigorous Quality Assurance Constraints
assert len(submission) == len(test_df), f"Row count mismatch: expected {len(test_df)}, got {len(submission)}"
assert list(submission.columns) == ['id', 'model'], f"Invalid columns: {submission.columns}"
assert not submission['model'].isnull().any(), "Found NaN in predicted models!"
valid_classes = set(pipeline.named_steps['clf'].classes_)
assert all(m in valid_classes for m in submission['model']), "Found invalid model names!"

submission.to_parquet('submission.parquet', index=False)
print(f"Successfully generated and verified submission.parquet ({os.path.getsize('submission.parquet')} bytes)!")
print("\\nFirst 10 submission rows:")
print(submission.head(10))
"""
cells.append(create_cell("code", c8))

# Cell 9: Markdown Guide on Reaching 0.534+ via Transformer Fine-Tuning
c9 = """## 🚀 The Path to 0.534+: Fine-Tuning DeBERTa-v3 on Kaggle GPU
To reach the top leaderboard tier (0.50 - 0.534+), you need deep contextual embeddings. Below is the blueprint to fine-tune `microsoft/deberta-v3-base` or `microsoft/deberta-v3-large` directly on Kaggle with GPU enabled:

```python
# Optional Kaggle GPU DeBERTa fine-tuning script snippet
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 1. Prepare label mapping
labels = sorted(train_df['model'].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}

train_df['label'] = train_df['model'].map(label2id)

# 2. Tokenizer & Dataset
model_checkpoint = "microsoft/deberta-v3-base" # or "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_func(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")

dataset = Dataset.from_pandas(train_df[['text', 'label']])
tokenized_dataset = dataset.map(tokenize_func, batched=True)

# 3. Model setup
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./deberta_output",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_strategy="no",
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)
# trainer.train()
```
### Why DeBERTa jumps to 0.50+:
- DeBERTa's disentangled attention mechanism explicitly separates token content from positional relationships.
- It encodes syntactic tree structure, pedagogical vs formal writing tone, and nuanced vocabulary choices that are invisible to bag-of-words TF-IDF.
- Ensembling DeBERTa probabilities with our Stylometric Features yields the optimal balanced accuracy.
"""
cells.append(create_cell("markdown", c9))

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
    r"c:\Users\Matthew Chen\Documents\X-Star\competition\name_that_llm\x_star_cognitive_name_that_llm.ipynb",
    r"C:\Users\Matthew Chen\Downloads\x_star_cognitive_name_that_llm.ipynb"
]

for p in paths:
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"Generated clean, robust notebook at: {p}")
