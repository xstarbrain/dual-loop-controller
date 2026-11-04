"""
Name That LLM - X-Star Cognitive Stylometry & Hierarchical Attribution Engine
=============================================================================
Kaggle Competition: Name That LLM (AI Authorship Attribution)
Author: X-Star Autonomous AI Team

Integrates core X-Star modules:
- Probabilistic Cognitive Judge (dual_loop/cognitive_judge.py)
- CalLearn Contextual Dynamic Prior Suppression (dual_loop/callearn_oc_engine.py)
- Evidential Epistemic Self-Recognition (dual_loop/evidential.py)
- Parsimony & MDL Stylometric Filtering (dual_loop/mdl_selector.py)
"""

from .stylometric_extractor import StylometricFeatureExtractor
from .hierarchical_cognitive_classifier import HierarchicalCognitiveClassifier
from .pipeline import build_name_that_llm_pipeline, LLMAttributionPipeline

__all__ = [
    "StylometricFeatureExtractor",
    "HierarchicalCognitiveClassifier",
    "build_name_that_llm_pipeline",
    "LLMAttributionPipeline",
]
