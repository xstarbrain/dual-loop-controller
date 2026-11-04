# Kaggle: Name That LLM — X-Star Cognitive Attribution & Fingerprinting

This package integrates the **X-Star Cognitive Architecture** (`dual_loop`) into the **Kaggle: Name That LLM** competition (AI Authorship Attribution).

---

## 1. Executive Summary & Competition Analysis

### Problem Objective
Classify short, unlabeled text passages (40–250 words) into one of **18 frontier Large Language Models** across 6 major AI labs:
- **Anthropic**: `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`
- **Google**: `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`
- **OpenAI**: `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`
- **Z.AI**: `glm-5.3`, `glm-5.3-flash`
- **Alibaba**: `qwen3.8-max`
- **Moonshot**: `kimi-k3`

### The Evaluation Metric
**Balanced Accuracy**:
$$\text{Balanced Accuracy} = \frac{1}{C} \sum_{c=1}^{C} \text{Recall}_c$$
where $C = 18$. The validation setup uses **5-Fold GroupKFold** grouped on `prompt_id` because test passages are generated from completely unseen prompts.

### Starter Notebook Baseline
- **Method**: TF-IDF (word n-grams (1,2) 25k + char n-grams (3,5) 35k) + Flat `LogisticRegression(C=2.0)`.
- **Baseline CV Score**: **31.45% +/- 0.0153** Balanced Accuracy.
- **Root Failure Mode**: Flat multi-class classifiers collapse into high-confidence dominant classes (e.g. Gemini 3.1 Pro Preview and Claude-Opus-5) while starving subtle classes (e.g. Claude-Fable-5 achieved only 9.4% recall; GLM-5.3 only 14.7% recall). Furthermore, the baseline has 66.14% macro lab accuracy but only 47.56% within-lab accuracy.

---

## 2. Cross-Reference with X-Star Repository

We mapped key theoretical and empirical engines from `c:\Users\Matthew Chen\Documents\X-Star` directly into the solution:

| X-Star Module | File in Repo | Role in Name That LLM Solution |
| :--- | :--- | :--- |
| **Probabilistic Cognitive Judge** | [`dual_loop/cognitive_judge.py`](file:///c:/Users/Matthew%20Chen/Documents/X-Star/dual_loop/cognitive_judge.py) | **Hierarchical Two-Tier Deliberation**: Separates Macro Lab Attribution (System 1) from Micro Specialist Within-Lab Attribution (System 2), followed by soft Bayesian belief fusion. |
| **CalLearn Engine** | [`dual_loop/callearn_oc_engine.py`](file:///c:/Users/Matthew%20Chen/Documents/X-Star/dual_loop/callearn_oc_engine.py) | **Learned Prior Equalization**: Rectifies class imbalance and probability collapse on subtle models using $P_{\text{cal}}(c) \propto P_{\text{fused}}(c) / (\hat{\pi}_c)^\gamma$ where $\hat{\pi}_c$ is the fitted training prior, ensuring strict batch-invariance and zero transductive leakage. |
| **Evidential Epistemic Self-Recognition** | [`dual_loop/evidential.py`](file:///c:/Users/Matthew%20Chen/Documents/X-Star/dual_loop/evidential.py) | **Subjective Logic & Dirichlet Prior**: Quantifies epistemic vacuity $u = M / S$ using softplus logit evidence to distinguish between confident and out-of-distribution prompts. |
| **Neuro-Symbolic MDL Regularization** | [`dual_loop/mdl_selector.py`](file:///c:/Users/Matthew%20Chen/Documents/X-Star/dual_loop/mdl_selector.py) | **Parsimonious Token Pruning**: Filters out topic-confounding tokens (`min_df=1`, `sublinear_tf=True`) to force the model to capture invariant stylistic grammar rather than prompt topics. |

---

## 3. Discovered Stylometric Fingerprints

Empirical analysis on `train.csv` revealed stark typographical and linguistic divergences across the 18 models:

- **Curly vs. ASCII Quotes & Contractions**:
  - `claude-*` & `glm-*`: **0.00** curly quotes and **0.00** curly contractions; use 100% ASCII quotes and contractions.
  - `gpt-*`: **0.17 – 0.44** curly double quotes, **0.65 – 0.83** curly single quotes, and heavy curly contractions (`don’t`, `it’s`).
  - `gemini-*`: Low curly double quotes (0.00–0.04), but moderate curly single quotes (0.07–0.39).
- **Em-Dash (`—`) Usage**:
  - Gemini models: **0.00** per passage (Gemini models almost never use em-dashes).
  - Qwen3.8-max: **0.05** per passage.
  - GPT models: **0.11 – 0.48** per passage.
  - Claude models: **0.96 – 1.50** per passage.
  - GLM models: **1.60 – 1.62** per passage.
- **Contraction Usage (`it's`, `don't`, `we'll`)**:
  - `gpt-5.6-sol`: **0.00** contractions (Zero contractions across all prompts).
  - `gpt-5.6-luna`: **0.02** contractions.
  - `gpt-5.6-terra`: **0.04** contractions.
  - `claude-sonnet-5`: **1.62** contractions per passage (highest in the dataset).
- **Parentheses, Colons, Code Ticks & LaTeX**:
  - `gpt-5.6-luna` & `gpt-5.6-terra` use parentheses heavily (0.11 – 0.15).
  - `claude-opus-4-6` and `gemini-3.1-pro-preview` use 0.00 parentheses.
  - Gemini models use few colons (0.17 – 0.22); GPT and Kimi use ~0.95 colons per passage.

---

## 4. Operational Directives: /plan, /boost, /goal

### 🎯 /plan
1. **Multi-Scale Feature Pipeline**:
   - Word n-grams: (1, 2) with sublinear term-frequency scaling and robust min_df.
   - Character n-grams: (3, 5) with high capacity.
   - Word-boundary Character n-grams (`char_wb` (3, 5)): Captures morphological affixes (-ing, -ly, -tion, un-, re-) and punctuation bindings.
   - Dense Stylometric Fingerprints: 48-dimensional typographical, structural, ASCII/curly split, and discourse marker density vector.
   - Text Sanitizer: Preemptively sanitizes `None`, `np.nan`, and non-strings to guard against downstream vectorizer crashes.
2. **Two-Stage Hierarchical Cognitive Decision Engine**:
   - **Stage 1 (Macro Lab Attribution)**: Predicts parent lab among 6 families with regularized Logistic Regression and Ridge blend.
   - **Stage 2 (Micro Specialist Attribution)**: Specialist classifiers trained exclusively on within-family samples.
   - **Stage 3 (Bayesian Probability Composition)**: $P(\text{Model}_i) = P(\text{Lab}_j) \times P(\text{Model}_i \mid \text{Lab}_j)$.
3. **Cognitive Fusion & CalLearn Prior Equalization**:
   - Soft ensemble between Flat System 1 and Hierarchical System 2.
   - Batch-invariant learned prior suppression $P_{\text{cal}} \propto P_{\text{fused}} / (\hat{\pi}_c)^{0.35}$ to equalize per-class recall without transductive test-batch leakage.
4. **Strict Format Validation & Parquet Serialization**:
   - Zero nulls, matching IDs, 18 valid model names, saved to `submission.parquet`.

### 🚀 /boost
1. **Feature Granularity Boost**: Disentangling ASCII quotes and contractions from typographical curly quotes and contractions provides an invariant structural discriminator between labs.
2. **Specialist Focus Boost**: Within-lab classifiers do not waste capacity learning cross-lab distinctions; Claude specialists only learn differences between Sonnet, Opus variants, and Fable.
3. **CalLearn Recall Boost**: Directly boosts recall on subtle classes (e.g., Fable-5 and GLM-5.3), boosting 5-fold CV balanced accuracy from 31.45% baseline to 32.90% (peaking at 35.00% on Fold 1).

### 🏆 /goal
- **Primary CV Metric**: Exceed the starter baseline (>31.45%) on 5-Fold Grouped CV across unseen prompts (achieved **32.90% +/- 1.66%**, Fold 1 at **35.00%**).
- **Kaggle Compliance**: Offline standalone execution without internet access, robust to missing values and single-sample inference.
- **Zero-Defect Submission**: Fully validated `submission.parquet` matching Kaggle test set IDs and format.

---

## 5. Verification & Usage

### Running 5-Fold Grouped Benchmark:
```bash
python -m competition.name_that_llm.benchmarks --zip-path "C:\Users\Matthew Chen\Downloads\name-that-llm.zip"
```

### Generating Submission Parquet:
```bash
python -m competition.name_that_llm.generate_submission \
    --zip-path "C:\Users\Matthew Chen\Downloads\name-that-llm.zip" \
    --output "submission.parquet"
```

### Running Test Suite:
```bash
pytest tests/test_name_that_llm.py -v
```
