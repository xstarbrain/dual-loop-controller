"""
Unit and Integration Tests for Name That LLM Cognitive Engine
=============================================================
Tests:
1. StylometricFeatureExtractor handling edge cases (empty strings, None, punctuation).
2. HierarchicalCognitiveClassifier fitting, probability simplex preservation, and evidential telemetry.
3. End-to-end Pipeline training and prediction.
4. Submission DataFrame validator (integrity checks, rejection of invalid models, NaNs, ID mismatches).
5. Parquet export and round-trip readability.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from competition.name_that_llm.stylometric_extractor import StylometricFeatureExtractor
from competition.name_that_llm.hierarchical_cognitive_classifier import (
    HierarchicalCognitiveClassifier,
    map_model_to_lab,
    LAB_HIERARCHY,
)
from competition.name_that_llm.pipeline import build_name_that_llm_pipeline
from competition.name_that_llm.generate_submission import (
    ALLOWED_MODELS,
    validate_submission_df,
)

# Sample synthetic passages representing distinct labs
SAMPLE_TEXTS = [
    "When a language dies, a particular astronomy dims—not only names for stars, but whole grammars of attention.",
    "When a language dies, an entire universe folds inward, erasing unmapped constellations of thought.",
    "When a language dies, the cosmos loses more than names: it loses a particular sky. Its vanished verbs tracked the moon.",
    "When a language dies, we do not merely lose a catalog of exotic syntax; we lose an entirely unique architecture.",
    "When the last speaker of a language falls silent, a constellation goes unnamed forever—held in grief.",
    "When a language dies, a whole way of seeing the world disappears: its songs, idioms, and stories.",
]

SAMPLE_LABELS = [
    "qwen3.8-max",
    "gemini-3.7-flash",
    "gpt-5.6-terra",
    "gemini-3.1-pro-preview",
    "claude-opus-5",
    "glm-5.3",
]

def test_map_model_to_lab():
    assert map_model_to_lab("claude-opus-5") == "Anthropic"
    assert map_model_to_lab("gemini-3.8-flash") == "Google"
    assert map_model_to_lab("gpt-6-astra") == "OpenAI"
    assert map_model_to_lab("glm-5.3-flash") == "Z.AI"
    assert map_model_to_lab("kimi-k3") == "Moonshot"
    assert map_model_to_lab("qwen3.8-max") == "Alibaba"
    assert map_model_to_lab("unknown-model") == "Other"

def test_stylometric_extractor_shape_and_features():
    extractor = StylometricFeatureExtractor()
    X = [
        "It's a test—really! Here: 123.",
        "Another sentence with... ellipses and; semicolons.",
        "",
        None,
    ]
    feats = extractor.transform(X)
    assert isinstance(feats, np.ndarray)
    assert feats.shape[0] == 4
    assert feats.shape[1] > 20
    # Check no NaNs
    assert not np.isnan(feats).any()
    # Check em-dash detection in first sample
    assert feats[0, 3] > 0.0  # em_dash / word_len

def test_hierarchical_classifier_fit_predict():
    # Construct small synthetic dataset covering multiple labs
    texts = SAMPLE_TEXTS * 4
    labels = SAMPLE_LABELS * 4

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=100)
    X_vec = vec.fit_transform(texts)

    clf = HierarchicalCognitiveClassifier(c_flat=1.0, c_lab=1.0, c_specialist=1.0, random_state=42)
    clf.fit(X_vec, labels)

    preds = clf.predict(X_vec)
    assert len(preds) == len(labels)
    assert set(preds).issubset(set(SAMPLE_LABELS))

    probs = clf.predict_proba(X_vec)
    assert probs.shape == (len(labels), len(np.unique(labels)))
    # Simplex preservation check (each row sums to ~1.0)
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

def test_evidential_telemetry():
    texts = SAMPLE_TEXTS * 5
    labels = SAMPLE_LABELS * 5

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=100)
    X_vec = vec.fit_transform(texts)

    clf = HierarchicalCognitiveClassifier(c_flat=1.0, c_lab=1.0, c_specialist=1.0, random_state=42)
    clf.fit(X_vec, labels)

    telem = clf.compute_evidential_telemetry(X_vec)
    assert "vacuity_u" in telem
    assert "mean_vacuity" in telem
    assert "min_vacuity" in telem
    assert "max_vacuity" in telem
    assert "entropy" in telem
    # Check that vacuity is NOT constant (guards against the constant 0.5 bug)
    assert telem["min_vacuity"] < telem["max_vacuity"]
    assert 0.0 < telem["mean_vacuity"] < 1.0
    assert telem["entropy"] >= 0.0

def test_batch_invariance_deterministic():
    """
    Guarantees that CalLearn prior suppression produces identical predictions
    regardless of whether samples are evaluated in a batch or individually.
    """
    texts = SAMPLE_TEXTS * 4
    labels = SAMPLE_LABELS * 4

    pipeline = build_name_that_llm_pipeline(
        word_max_features=500,
        char_max_features=500,
        char_wb_max_features=500,
        word_min_df=1,
    )
    pipeline.fit(texts, labels)

    eval_samples = [
        "A language dies—stars dim.",
        "An entire universe folds inward.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    batch_preds = pipeline.predict(eval_samples)
    single_preds = [pipeline.predict([s])[0] for s in eval_samples]
    assert list(batch_preds) == single_preds

    batch_probs = pipeline.predict_proba(eval_samples)
    for i, s in enumerate(eval_samples):
        single_p = pipeline.predict_proba([s])[0]
        np.testing.assert_allclose(batch_probs[i], single_p, atol=1e-5)

def test_stylometric_curly_vs_ascii_distinction():
    """
    Ensures that curly quotes/contractions and ASCII quotes/contractions
    are cleanly separated into distinct feature dimensions.
    """
    extractor = StylometricFeatureExtractor()
    X = [
        'He said "hello" and it\'s fine.',           # Pure ASCII double quotes and contraction
        'He said “hello” and it’s fine.',           # Typographical curly quotes and contraction
    ]
    feats = extractor.transform(X)
    # Check ASCII double quotes (col 11) vs Curly double quotes (col 12)
    assert feats[0, 11] > 0.0 and feats[0, 12] == 0.0
    assert feats[1, 11] == 0.0 and feats[1, 12] > 0.0

    # Check ASCII contractions (col 15) vs Curly contractions (col 16)
    assert feats[0, 15] > 0.0 and feats[0, 16] == 0.0
    assert feats[1, 15] == 0.0 and feats[1, 16] > 0.0

def test_pipeline_fit_small_samples_robustness():
    """
    Verifies that pipeline does not crash with empty vocabulary errors
    when fitted on small sample sets (min_df=1).
    """
    pipeline = build_name_that_llm_pipeline(word_min_df=1)
    tiny_texts = ["Completely unique astronomy vocabulary.", "Vanished grammar syntax."]
    tiny_labels = ["gemini-3.7-flash", "claude-opus-5"]
    pipeline.fit(tiny_texts, tiny_labels)
    preds = pipeline.predict(tiny_texts)
    assert len(preds) == 2

def test_stylometric_and_pipeline_edge_cases():
    """
    Attacks model with markdown code blocks, LaTeX, empty strings, punctuation only, and long inputs.
    """
    edge_cases = [
        "",
        "   \n\t  ",
        None,
        np.nan,
        "---...!???:::;;;(((***)))",
        "```python\ndef solve():\n    return 42\n```",
        "The formula is $\\frac{\\alpha}{\\beta} + \\sum_{i=1}^n x_i = 0$.",
        "Word " * 200,
    ]
    extractor = StylometricFeatureExtractor()
    feats = extractor.transform(edge_cases)
    assert feats.shape[0] == len(edge_cases)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()

    # Feed edge cases into trained pipeline
    texts = SAMPLE_TEXTS * 4
    labels = SAMPLE_LABELS * 4
    pipeline = build_name_that_llm_pipeline(word_max_features=200, char_max_features=200, char_wb_max_features=200)
    pipeline.fit(texts, labels)
    preds = pipeline.predict(edge_cases)
    assert len(preds) == len(edge_cases)
    for p in preds:
        assert p in SAMPLE_LABELS

def test_full_pipeline_fit_predict():
    texts = SAMPLE_TEXTS * 4
    labels = SAMPLE_LABELS * 4

    pipeline = build_name_that_llm_pipeline(
        word_max_features=500,
        char_max_features=500,
        char_wb_max_features=500,
    )
    pipeline.fit(texts, labels)

    test_samples = [
        "A language dies—stars dim.",
        "An entire universe folds inward.",
    ]
    preds = pipeline.predict(test_samples)
    assert len(preds) == 2
    for p in preds:
        assert p in SAMPLE_LABELS

    probs = pipeline.predict_proba(test_samples)
    assert probs.shape == (2, len(np.unique(labels)))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

def test_validate_submission_constraints():
    test_df = pd.DataFrame({
        "id": ["id1", "id2", "id3"],
        "text": ["sample1", "sample2", "sample3"],
    })

    # Valid submission
    valid_sub = pd.DataFrame({
        "id": ["id1", "id2", "id3"],
        "model": ["gpt-5.5", "claude-opus-5", "gemini-3.7-flash"],
    })
    validate_submission_df(valid_sub, test_df)

    # Invalid: Column names wrong
    bad_cols = pd.DataFrame({"identifier": ["id1"], "prediction": ["gpt-5.5"]})
    with pytest.raises(ValueError, match="Invalid columns"):
        validate_submission_df(bad_cols, test_df)

    # Invalid: Row count mismatch
    short_sub = pd.DataFrame({"id": ["id1"], "model": ["gpt-5.5"]})
    with pytest.raises(ValueError, match="Row count mismatch"):
        validate_submission_df(short_sub, test_df)

    # Invalid: ID ordering mismatch
    scrambled_sub = pd.DataFrame({"id": ["id2", "id1", "id3"], "model": ["gpt-5.5", "gpt-5.5", "gpt-5.5"]})
    with pytest.raises(ValueError, match="does not match test 'id' exactly"):
        validate_submission_df(scrambled_sub, test_df)

    # Invalid: NaN in model
    nan_sub = pd.DataFrame({"id": ["id1", "id2", "id3"], "model": ["gpt-5.5", None, "gpt-5.5"]})
    with pytest.raises(ValueError, match="contains null/NaN values"):
        validate_submission_df(nan_sub, test_df)

    # Invalid: Unknown model name
    invalid_model_sub = pd.DataFrame({"id": ["id1", "id2", "id3"], "model": ["gpt-5.5", "llama-4-scam", "gpt-5.5"]})
    with pytest.raises(ValueError, match="Found invalid model predictions"):
        validate_submission_df(invalid_model_sub, test_df)

def test_parquet_roundtrip(tmp_path):
    out_file = tmp_path / "submission.parquet"
    df = pd.DataFrame({
        "id": ["test_1", "test_2"],
        "model": ["qwen3.8-max", "kimi-k3"],
    })
    df.to_parquet(out_file, index=False)
    loaded = pd.read_parquet(out_file)
    pd.testing.assert_frame_equal(df, loaded)

def test_hungarian_assignment_permutation():
    from competition.name_that_llm.pipeline import predict_with_hungarian_assignment
    
    classes = [f"model_{i}" for i in range(18)]
    # Create 18 synthetic samples with mock probability matrices
    probs = np.random.RandomState(42).dirichlet(np.ones(18), size=18)
    texts = [f"This is sample prompt number {i} discussing quantum physics and mechanics." for i in range(18)]

    preds = predict_with_hungarian_assignment(texts, probs, classes, cluster_size=18)
    assert len(preds) == 18
    # When n=18 and cluster_size=18, each of the 18 classes must be uniquely assigned!
    assert len(set(preds)) == 18

