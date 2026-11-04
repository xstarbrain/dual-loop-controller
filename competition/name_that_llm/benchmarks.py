"""
Benchmark Suite for Name That LLM
=================================
Runs 5-Fold Grouped Cross-Validation comparing:
1. Starter Baseline (TF-IDF + Flat Logistic Regression)
2. X-Star Cognitive Pipeline (Tri-Scale TF-IDF + Dense Stylometry + Hierarchical Specialist Bayesian Fusion + CalLearn Prior Suppression)

Reports:
- 5-Fold Grouped CV Balanced Accuracy (mean +/- std)
- Fold-by-fold scores
- Per-class recall breakdown
- Macro vs Micro accuracy
- Confusion matrices
"""

import zipfile
import time
import os
import argparse
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, classification_report

from .pipeline import build_name_that_llm_pipeline

def run_benchmarks(zip_path=r'C:\Users\Matthew Chen\Downloads\name-that-llm.zip'):
    print("=" * 70)
    print("Name That LLM - 5-Fold Grouped Cross-Validation Benchmark")
    print("=" * 70)

    with zipfile.ZipFile(zip_path) as z:
        with z.open('train.csv') as f:
            train_df = pd.read_csv(f)

    X = train_df['text'].fillna('')
    y = train_df['model']
    groups = train_df['prompt_id']

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups))

    print(f"Dataset: {len(train_df)} samples, {y.nunique()} models, {groups.nunique()} prompts.")

    # 1. Starter Baseline Pipeline
    print("\n[1/2] Evaluating Starter Baseline Pipeline (Flat TF-IDF + LR C=2.0)...", flush=True)
    starter_pipeline = Pipeline([
        (
            'features',
            FeatureUnion([
                ('word_tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)),
                ('char_tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=35000, sublinear_tf=True)),
            ])
        ),
        ('clf', LogisticRegression(C=2.0, max_iter=300, random_state=42)),
    ])

    starter_scores = []
    starter_oof = [None] * len(train_df)
    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        starter_pipeline.fit(X_tr, y_tr)
        preds = starter_pipeline.predict(X_va)
        sc = balanced_accuracy_score(y_va, preds)
        starter_scores.append(sc)
        for idx, p in zip(va_idx, preds):
            starter_oof[idx] = p
        print(f"  Fold {fold+1}: {sc:.4f}", flush=True)

    starter_time = time.time() - t0
    starter_mean = np.mean(starter_scores)
    starter_std = np.std(starter_scores)

    # 2. X-Star Cognitive Pipeline
    print("\n[2/2] Evaluating X-Star Cognitive Pipeline (Tri-Scale + Stylometry + Hierarchical + CalLearn)...", flush=True)
    xstar_pipeline = build_name_that_llm_pipeline(
        word_max_features=20000,
        char_max_features=30000,
        char_wb_max_features=20000,
    )

    xstar_scores = []
    xstar_oof = [None] * len(train_df)
    t1 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        xstar_pipeline.fit(X_tr, y_tr)
        preds = xstar_pipeline.predict(X_va)
        sc = balanced_accuracy_score(y_va, preds)
        xstar_scores.append(sc)
        for idx, p in zip(va_idx, preds):
            xstar_oof[idx] = p
        print(f"  Fold {fold+1}: {sc:.4f}", flush=True)

    xstar_time = time.time() - t1
    xstar_mean = np.mean(xstar_scores)
    xstar_std = np.std(xstar_scores)

    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD BENCHMARK SCOREBOARD")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Starter Baseline':<18} | {'X-Star Cognitive':<18} | {'Delta':<10}")
    print("-" * 70)
    print(f"{'Mean Balanced Accuracy':<30} | {starter_mean*100:>6.2f}% +/- {starter_std*100:.2f}% | {xstar_mean*100:>6.2f}% +/- {xstar_std*100:.2f}% | {(xstar_mean - starter_mean)*100:>+6.2f}%")
    print(f"{'Fold 1 Balanced Accuracy':<30} | {starter_scores[0]*100:>6.2f}%           | {xstar_scores[0]*100:>6.2f}%           | {(xstar_scores[0] - starter_scores[0])*100:>+6.2f}%")
    print(f"{'Fold 2 Balanced Accuracy':<30} | {starter_scores[1]*100:>6.2f}%           | {xstar_scores[1]*100:>6.2f}%           | {(xstar_scores[1] - starter_scores[1])*100:>+6.2f}%")
    print(f"{'Fold 3 Balanced Accuracy':<30} | {starter_scores[2]*100:>6.2f}%           | {xstar_scores[2]*100:>6.2f}%           | {(xstar_scores[2] - starter_scores[2])*100:>+6.2f}%")
    print(f"{'Fold 4 Balanced Accuracy':<30} | {starter_scores[3]*100:>6.2f}%           | {xstar_scores[3]*100:>6.2f}%           | {(xstar_scores[3] - starter_scores[3])*100:>+6.2f}%")
    print(f"{'Fold 5 Balanced Accuracy':<30} | {starter_scores[4]*100:>6.2f}%           | {xstar_scores[4]*100:>6.2f}%           | {(xstar_scores[4] - starter_scores[4])*100:>+6.2f}%")
    print(f"{'Total Runtime':<30} | {starter_time:>6.1f}s           | {xstar_time:>6.1f}s           | {xstar_time - starter_time:>+6.1f}s")
    print("=" * 70)

    print("\nX-Star Cognitive Out-Of-Fold Classification Report:")
    print(classification_report(y, xstar_oof, digits=3))

    return {
        'starter_mean': starter_mean,
        'starter_std': starter_std,
        'xstar_mean': xstar_mean,
        'xstar_std': xstar_std,
        'delta': xstar_mean - starter_mean,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Name That LLM benchmarks")
    parser.add_argument('--zip-path', type=str, default=r'C:\Users\Matthew Chen\Downloads\name-that-llm.zip')
    args = parser.parse_args()
    run_benchmarks(zip_path=args.zip_path)
