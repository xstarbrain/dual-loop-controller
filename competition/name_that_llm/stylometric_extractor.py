"""
Stylometric Feature Extractor for LLM Fingerprinting
====================================================
Extracts linguistic, typographical, and structural features that characterize
the generation signatures of frontier LLM families.
"""

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

DISCOURSE_MARKERS = [
    'furthermore', 'moreover', 'indeed', 'crucially', 'ultimately',
    'specifically', 'rather', 'notably', 'consequently', 'thus',
    'therefore', 'however', 'conversely', 'meanwhile', 'yet', 'hence',
    'in short', 'in essence', 'in particular', 'fundamentally'
]

class StylometricFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transforms raw text into a dense matrix of stylometric fingerprint features:
    - Typographical markers (em-dash, en-dash, hyphen, semicolon, colon, ellipses, parentheses, brackets)
    - Separated ASCII vs typographical curly quotes (double & single)
    - Separated ASCII vs typographical curly contractions
    - Structural and syntax markers (newlines, paragraph breaks, code ticks, LaTeX math dollars)
    - Casing and numerical density
    - Token and sentence structural statistics
    - Discourse transition connective distributions
    """
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
            
            # Sentence splitting (handles periods, exclamation, question marks)
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

            # Separate ASCII vs Curly quotes
            ascii_double_quotes = t.count('"')
            curly_double_quotes = t.count('“') + t.count('”')
            ascii_single_quotes = t.count("'")
            curly_single_quotes = t.count('‘') + t.count('’')

            # Separate ASCII vs Curly contractions
            ascii_contractions = len(re.findall(r"\b[a-zA-Z]+'[a-zA-Z]+\b", t))
            curly_contractions = len(re.findall(r"\b[a-zA-Z]+’[a-zA-Z]+\b", t))

            # Structure & formatting
            newlines = t.count('\n')
            code_ticks = t.count('`')
            latex_dollars = t.count('$')

            # Casing & digits
            uppercase = len(re.findall(r'[A-Z]', t))
            digits = len(re.findall(r'\d', t))

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
