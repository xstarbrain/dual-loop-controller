"""
Hierarchical Cognitive Classifier for Name That LLM
===================================================
Incorporates X-Star architectural principles:
1. Probabilistic Cognitive Judge (dual_loop/cognitive_judge.py)
   - Two-tier deliberation: Macro Lab Attribution -> Micro Specialist Attribution
2. CalLearn Dynamic Prior Suppression (dual_loop/callearn_oc_engine.py)
   - Rectifies probability collapse on subtle classes to maximize Balanced Accuracy
3. Evidential Epistemic Telemetry (dual_loop/evidential.py)
   - Dirichlet prior and vacuity of evidence u = M / S to track prediction uncertainty
"""

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.utils.validation import check_is_fitted

LAB_HIERARCHY = {
    'Alibaba': ['qwen3.8-max'],
    'Anthropic': [
        'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8',
        'claude-opus-5', 'claude-sonnet-5'
    ],
    'Google': [
        'gemini-3.1-pro-preview', 'gemini-3.6-flash',
        'gemini-3.7-flash', 'gemini-3.8-flash'
    ],
    'Moonshot': ['kimi-k3'],
    'OpenAI': [
        'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol',
        'gpt-5.6-terra', 'gpt-6-astra'
    ],
    'Z.AI': ['glm-5.3', 'glm-5.3-flash'],
}

def map_model_to_lab(model_name: str) -> str:
    """Maps a model identifier to its parent lab/family."""
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
    """
    Two-Stage Hierarchical Cognitive Classifier with Bayesian Fusion
    and CalLearn Dynamic Prior Suppression.
    """
    def __init__(
        self,
        c_flat: float = 3.5,
        c_lab: float = 2.5,
        c_specialist: float = 3.5,
        lambda_flat: float = 0.50,
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
        """
        Fits global flat model, stage-1 lab model, and stage-2 specialist models.
        Computes learned class prior for batch-invariant CalLearn prior suppression.
        """
        y_arr = np.asarray(y)
        self.classes_ = np.sort(np.unique(y_arr))
        self.n_classes_ = len(self.classes_)
        self.model_to_idx_ = {m: i for i, m in enumerate(self.classes_)}

        # Learn training empirical class prior for batch-invariant CalLearn suppression
        counts = {}
        for val in y_arr:
            counts[val] = counts.get(val, 0) + 1
        total_samples = float(len(y_arr))
        self.learned_prior_ = np.array(
            [counts.get(c, 1.0) / total_samples for c in self.classes_],
            dtype=np.float32
        )

        # Lab labels
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

        # 2. Stage 1: Lab Classifier
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

        # 3. Stage 2: Specialist Within-Lab Classifiers
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
        """
        Computes calibrated cognitive probabilities combining flat,
        hierarchical, and prior-suppressed outputs.
        Prior suppression is strictly batch-invariant and deterministic.
        """
        check_is_fitted(self, ['classes_', 'flat_clf_', 'lab_clf_', 'learned_prior_'])
        n_samples = X.shape[0]

        # 1. Flat Probabilities aligned to self.classes_
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
            lab_prob_col = p_lab[:, lab_idx] # [N]
            models_in_lab = [m for m in self.classes_ if map_model_to_lab(m) == lab_name]

            if len(models_in_lab) == 1:
                # Deterministic single model (e.g. Moonshot -> kimi-k3, Alibaba -> qwen3.8-max)
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

        # 4. Cognitive Blending (Flat System 1 + Hierarchical System 2)
        fused_probs = self.lambda_flat * flat_probs + (1.0 - self.lambda_flat) * hier_probs

        # 5. CalLearn Dynamic Prior Suppression (dual_loop/callearn_oc_engine.py)
        # Using learned training prior: batch-invariant, deterministic, zero transductive leakage
        if self.callearn_gamma > 0.0:
            prior = np.clip(self.learned_prior_, 1e-4, 1.0)
            calibrated_probs = fused_probs / (prior ** self.callearn_gamma)
            # Re-normalize to valid probability simplex
            calibrated_probs = calibrated_probs / np.sum(calibrated_probs, axis=1, keepdims=True)
            return calibrated_probs

        return fused_probs

    def predict(self, X):
        """Predicts model class labels for samples in X."""
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return self.classes_[best_indices]

    def compute_evidential_telemetry(self, X):
        """
        Computes Dirichlet evidential telemetry (vacuity of evidence u = M / S)
        derived from unnormalized classifier logits / decision margins, matching
        the Subjective Logic formulation in dual_loop/evidential.py.
        """
        check_is_fitted(self, ['flat_clf_', 'classes_'])
        flat_logits = self.flat_clf_.decision_function(X)
        if flat_logits.ndim == 1:
            flat_logits = np.vstack([-flat_logits, flat_logits]).T

        # Softplus evidence: e_c = log(1 + exp(z_c)) >= 0
        evidence = np.log1p(np.exp(np.clip(flat_logits, -20.0, 20.0)))
        M = float(self.n_classes_)
        alpha = evidence + 1.0
        S = np.sum(alpha, axis=1) # Dirichlet strength
        vacuity_u = M / S

        probs = self.predict_proba(X)
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)), axis=1)
        return {
            'vacuity_u': vacuity_u,
            'mean_vacuity': float(np.mean(vacuity_u)),
            'min_vacuity': float(np.min(vacuity_u)),
            'max_vacuity': float(np.max(vacuity_u)),
            'entropy': float(np.mean(entropy)),
        }
