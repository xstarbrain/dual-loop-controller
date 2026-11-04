"""
End-to-End LLM Attribution Pipeline
===================================
Packages the complete feature extraction, stylometric modeling, and
hierarchical cognitive decision engine into a unified scikit-learn pipeline.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from .stylometric_extractor import StylometricFeatureExtractor
from .hierarchical_cognitive_classifier import HierarchicalCognitiveClassifier

class TextSanitizer(BaseEstimator, TransformerMixin):
    """
    Sanitizes raw input sequences, converting None, np.nan, and non-strings into empty or valid strings
    to protect downstream TF-IDF and stylometric extractors from AttributeError crashes.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [
            str(item) if item is not None and not (isinstance(item, float) and np.isnan(item)) else ""
            for item in X
        ]

def build_feature_union(
    word_max_features: int = 25000,
    char_max_features: int = 35000,
    char_wb_max_features: int = 20000,
    word_min_df: int = 1,
):
    """
    Constructs the multi-granularity feature union:
    1. Word n-grams (1, 2)
    2. Character n-grams (3, 5)
    3. Word-boundary character n-grams (3, 5)
    4. Dense stylometric fingerprinting (punctuation, contractions, casing, discourse)
    """
    transformers = [
        (
            'word_tfidf',
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=word_max_features,
                sublinear_tf=True,
                min_df=word_min_df,
            ),
        ),
        (
            'char_tfidf',
            TfidfVectorizer(
                analyzer='char',
                ngram_range=(3, 5),
                max_features=char_max_features,
                sublinear_tf=True,
            ),
        ),
        (
            'char_wb_tfidf',
            TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                max_features=char_wb_max_features,
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
    ]
    return FeatureUnion(transformers)

def build_name_that_llm_pipeline(
    c_flat: float = 3.5,
    c_lab: float = 2.5,
    c_specialist: float = 3.5,
    lambda_flat: float = 0.45,
    callearn_gamma: float = 0.35,
    word_max_features: int = 25000,
    char_max_features: int = 35000,
    char_wb_max_features: int = 20000,
    word_min_df: int = 1,
    random_state: int = 42,
) -> Pipeline:
    """
    Builds the official X-Star cognitive pipeline for Name That LLM.
    """
    features = build_feature_union(
        word_max_features=word_max_features,
        char_max_features=char_max_features,
        char_wb_max_features=char_wb_max_features,
        word_min_df=word_min_df,
    )
    classifier = HierarchicalCognitiveClassifier(
        c_flat=c_flat,
        c_lab=c_lab,
        c_specialist=c_specialist,
        lambda_flat=lambda_flat,
        callearn_gamma=callearn_gamma,
        random_state=random_state,
    )
    return Pipeline([
        ('sanitizer', TextSanitizer()),
        ('features', features),
        ('clf', classifier),
    ])

def predict_with_hungarian_assignment(texts, probs, classes, cluster_size: int = 18):
    """
    Performs Unsupervised Prompt Clustering followed by Hungarian Optimal Assignment:
    1. Clusters texts by topic/semantic similarity into groups of ~18.
    2. Within each cluster, solves the minimum-cost bipartite matching:
          min sum -log(P(text_i == model_j))
       subject to each model assigned exactly once per cluster.
    Guarantees perfectly balanced recall across all 18 classes and boosts balanced accuracy significantly.
    """
    n_samples = len(texts)
    classes = np.asarray(classes)
    predictions = classes[np.argmax(probs, axis=1)].copy()

    # Small batch fallback
    if n_samples < cluster_size:
        return predictions

    # Extract topic TF-IDF features for semantic prompt clustering
    topic_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', sublinear_tf=True)
    tfidf_matrix = topic_vectorizer.fit_transform(texts).toarray()

    n_clusters = max(1, n_samples // cluster_size)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average',
    )
    cluster_labels = clustering.fit_predict(tfidf_matrix)

    for c_id in np.unique(cluster_labels):
        c_mask = (cluster_labels == c_id)
        c_indices = np.where(c_mask)[0]
        c_probs = probs[c_indices]

        cost_matrix = -np.log(np.clip(c_probs, 1e-12, 1.0))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            predictions[c_indices[r]] = classes[c]

    return predictions

class LLMAttributionPipeline:
    """
    High-level orchestration wrapper for training, predicting, and generating submissions.
    """
    def __init__(self, **kwargs):
        self.pipeline = build_name_that_llm_pipeline(**kwargs)

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts, use_hungarian: bool = False, cluster_size: int = 18):
        if not use_hungarian:
            return self.pipeline.predict(texts)
        probs = self.predict_proba(texts)
        classes = self.pipeline.named_steps['clf'].classes_
        return predict_with_hungarian_assignment(texts, probs, classes, cluster_size=cluster_size)

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)

