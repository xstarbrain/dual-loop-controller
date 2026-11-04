"""
Submission Generator for Kaggle Name That LLM Competition
=========================================================
Trains the X-Star Cognitive Attribution Pipeline on full training data,
predicts on test.csv, strictly verifies submission format requirements,
and saves the output to submission.parquet.
"""

import os
import sys
import argparse
import pathlib
import zipfile
import pandas as pd
import numpy as np

from .pipeline import build_name_that_llm_pipeline

ALLOWED_MODELS = {
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5',
    'claude-sonnet-5', 'gemini-3.1-pro-preview', 'gemini-3.6-flash',
    'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash',
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
    'gpt-6-astra', 'kimi-k3', 'qwen3.8-max'
}

CANDIDATE_DATA_DIRS = [
    pathlib.Path('/kaggle/input/competitions/name-that-llm'),
    pathlib.Path('/kaggle/input/name-that-llm'),
    pathlib.Path(r'C:\Users\Matthew Chen\Downloads\name-that-llm.zip'),
    pathlib.Path(r'C:\Users\Matthew Chen\Downloads'),
]

def load_competition_data(data_dir=None, zip_path=None):
    """
    Locates and loads train.csv and test.csv from Kaggle input dirs or zip.
    """
    # 1. Direct zip specified or found
    if zip_path and os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            with z.open('train.csv') as f:
                train_df = pd.read_csv(f)
            with z.open('test.csv') as f:
                test_df = pd.read_csv(f)
        return train_df, test_df

    # 2. Check candidate directories
    search_dirs = [pathlib.Path(data_dir)] if data_dir else CANDIDATE_DATA_DIRS
    for p in search_dirs:
        if p.is_file() and str(p).endswith('.zip') and p.exists():
            with zipfile.ZipFile(p) as z:
                with z.open('train.csv') as f:
                    train_df = pd.read_csv(f)
                with z.open('test.csv') as f:
                    test_df = pd.read_csv(f)
            return train_df, test_df
        elif p.is_dir() and (p / 'train.csv').exists():
            train_df = pd.read_csv(p / 'train.csv')
            test_df = pd.read_csv(p / 'test.csv')
            return train_df, test_df

    raise FileNotFoundError("Could not locate train.csv and test.csv in candidate locations.")

def validate_submission_df(submission_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Enforces all Kaggle competition constraints:
    1. Correct column names ('id', 'model')
    2. Exact row count and matching ids in identical order
    3. No NaN / null values
    4. All predicted models belong strictly to the 18 allowed models
    """
    if list(submission_df.columns) != ['id', 'model']:
        raise ValueError(f"Invalid columns: expected ['id', 'model'], got {submission_df.columns.tolist()}")

    if len(submission_df) != len(test_df):
        raise ValueError(f"Row count mismatch: test has {len(test_df)}, submission has {len(submission_df)}")

    if not (submission_df['id'].values == test_df['id'].values).all():
        raise ValueError("Submission 'id' column does not match test 'id' exactly in order and values!")

    if submission_df['model'].isnull().any():
        raise ValueError("Submission contains null/NaN values in 'model' column!")

    invalid_models = set(submission_df['model'].unique()) - ALLOWED_MODELS
    if invalid_models:
        raise ValueError(f"Found invalid model predictions not in allowed list: {invalid_models}")

    print("[Validation Passed] All submission format and integrity checks succeeded.")

def generate_submission(
    data_dir=None,
    zip_path=None,
    output_path='submission.parquet',
):
    print("=" * 60)
    print("X-Star Cognitive Pipeline - Submission Generator")
    print("=" * 60)

    train_df, test_df = load_competition_data(data_dir=data_dir, zip_path=zip_path)
    print(f"Loaded Train: {train_df.shape} ({train_df['prompt_id'].nunique()} prompts, {train_df['model'].nunique()} models)")
    print(f"Loaded Test:  {test_df.shape}")

    # Build and fit pipeline
    print("\nFitting full X-Star Cognitive Pipeline...")
    pipeline = build_name_that_llm_pipeline()
    pipeline.fit(train_df['text'].fillna(''), train_df['model'])
    print("Pipeline training complete.")

    # Predict on test
    print(f"\nPredicting on {len(test_df)} test samples...")
    test_texts = test_df['text'].fillna('')
    preds = pipeline.predict(test_texts)

    submission = pd.DataFrame({
        'id': test_df['id'],
        'model': preds,
    })

    validate_submission_df(submission, test_df)

    submission.to_parquet(output_path, index=False)
    print(f"Successfully saved {len(submission)} rows to: {os.path.abspath(output_path)}")

    print("\nSubmission preview:")
    print(submission.head(10))

    print("\nPredicted class distribution:")
    print(submission['model'].value_counts())

    return submission

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Kaggle Name That LLM submission.parquet")
    parser.add_argument('--data-dir', type=str, default=None, help="Directory containing train.csv and test.csv")
    parser.add_argument('--zip-path', type=str, default=r'C:\Users\Matthew Chen\Downloads\name-that-llm.zip', help="Path to competition zip file")
    parser.add_argument('--output', type=str, default='submission.parquet', help="Output parquet path")
    args = parser.parse_args()

    generate_submission(data_dir=args.data_dir, zip_path=args.zip_path, output_path=args.output)
