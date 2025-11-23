# XAI Evaluation Framework - Final Scripts

This directory contains the **4 final evaluation scripts** for the hierarchical XAI evaluation framework.

## Framework Overview

The framework evaluates XAI methods across 4 hierarchical tiers:

| Tier | Evaluation | Script | Output Directory | Purpose |
|------|-----------|--------|------------------|---------|
| **Tier 1** | Scaffold Recognition (SR) | `evaluate_SR.py` | `../sr/` | Essential drug core identification |
| **Tier 2** | Model Independence (MI) | `evaluate_MI.py` | `../model_independence/` | Cross-instance consistency |
| **Tier 3** | Context Sensitivity (CS) | `evaluate_CS.py` | `../cs_final/` | Activity cliff discrimination |
| **Tier 4** | Internal Consistency (IC) | `evaluate_IC.py` | `../prediction_attribution_alignment/` | Prediction-explanation alignment |

---

## 1️⃣ Scaffold Recognition (SR) - Tier 1 Essential

**Script:** `evaluate_SR.py`

**Purpose:** Evaluates whether XAI methods detect and assign meaningful importance to core drug scaffolds.

**Methodology:**
- **RGCN/CNN:** Two metrics required
  1. Top-K recognition (K=3): Scaffold in top-3 substructures
  2. Attribution magnitude >0.1: Meaningful importance assignment
  3. **Complete recognition:** BOTH criteria satisfied

- **RF:** One metric required
  1. Top-K recognition (K=7): Scaffold in top-7 features
  2. **No magnitude threshold:** RF attributions (0.001-0.03) are 10-100× smaller than RGCN/CNN
  3. **Complete recognition:** Top-K satisfied

**Dataset:** 100 active compounds per class (Fluoroquinolones, Beta-lactam, Oxazolidinone)

**Key Outputs:**
- `sr/sr_manuscript_table.csv` - Publication-ready results
- `sr/sr_detailed_results.csv` - Detailed per-compound analysis

**Run:**
```bash
python evaluate_SR.py
```

**Decision Rule:** Methods MUST achieve ≥90% complete recognition to pass Tier 1.

---

## 2️⃣ Model Independence (MI) - Tier 2 Deployment

**Script:** `evaluate_MI.py`

**Purpose:** Evaluates explanation stability across different model training instances.

**Methodology:**
- Compares attributions from 5 different model instances (10 pairwise comparisons)
- **Metrics:**
  - Jaccard similarity (Top-K overlap): Agreement on WHICH features are important
  - Spearman correlation: Consistency of importance ranking
- **Model-specific K:** CNN/RGCN use Top-3, RF uses Top-7
- **RF filtering:** Only present features included (avg 42 vs 85 total)

**Dataset:** 100 sampled compounds per model

**Key Outputs:**
- `model_independence/model_independence_manuscript_table.csv` - Publication results
- `model_independence/model_independence_summary.csv` - Summary statistics
- `model_independence/model_independence_detailed_results.csv` - All pairwise comparisons

**Run:**
```bash
python evaluate_MI.py
```

**Decision Rule:** Methods MUST achieve ≥0.95 MI for reliable deployment (≥0.70 acceptable for research).

---

## 3️⃣ Context Sensitivity (CS) - Tier 3 Validation

**Script:** `evaluate_CS.py`

**Purpose:** Evaluates ability to recognize that identical scaffolds contribute differently based on structural context.

**Methodology:**
- **SR-Aligned:** Uses SAME scaffold detection as SR evaluation
  - RGCN: Strict SMARTS patterns (complete scaffold structures)
  - CNN: Strict SMARTS (official, expected to fail) + Relaxed SMARTS (partial, comparison only)
  - RF: Feature-based functional groups

- **Three Statistical Components:**
  1. **Directionality (35%):** Paired t-test - Scaffold more important in actives?
  2. **Context Awareness (35%):** Levene's test - Variance heterogeneity across contexts?
     - Normalized score: `min(context_responsiveness / 3.0, 1.0)`
  3. **Discrimination (30%):** Binomial test - Detection rate > 50%?

- **Combined CS Score:**
  ```
  CS = 0.35×Directionality + 0.35×Context_Awareness + 0.30×Discrimination
  ```

**Dataset:** 100 activity cliff pairs per class (300 total)

**Key Outputs:**
- `cs_final/cs_official_SR_aligned.csv` - Official results (strict SMARTS/feature-based)
- `cs_final/cs_partial_CNN_comparison.csv` - CNN partial results (relaxed SMARTS, comparison only)

**Run:**
```bash
python evaluate_CS.py
```

**Results:**
- RGCN: 0.844 (1st) - Superior context awareness
- RF: 0.794 (2nd) - Perfect discrimination
- CNN: 0.000 (official, FAILED) | 0.462 (partial, comparison only)

---

## 4️⃣ Internal Consistency (IC) - Tier 4 Confidence

**Script:** `evaluate_IC.py`

**Purpose:** Validates explanation-prediction agreement as a confidence indicator.

**Methodology (Simple Criterion):**
- **All models use the SAME criterion:** Net attribution sign matches prediction direction
  - **Active prediction** (prob ≥ 0.5): mean(attributions) > 0 → aligned
  - **Inactive prediction** (prob < 0.5): mean(attributions) < 0 → aligned

**Why this criterion?**
- Architecture-agnostic: Works for localized (RGCN/CNN) and distributed (RF) attributions
- Interpretable: Direct test of whether explanation supports prediction
- Simple: No arbitrary thresholds or complex rules
- Fair: Doesn't penalize architectural differences

**Characterization (for richer understanding):**
- **Strong alignment**: Large magnitude in correct direction
- **Weak alignment**: Small magnitude in correct direction
- **Neutral**: Near-zero magnitude (weak signal, not wrong)
- **Misalignment**: Wrong direction (explanation contradicts prediction)

**Dataset:** All 450 unique compounds from ensemble predictions

**Key Outputs:**
- `prediction_attribution_alignment/alignment_summary_corrected.csv` - Summary statistics and publication results
- `prediction_attribution_alignment/alignment_detailed_results_corrected.csv` - Per-compound analysis

**Run:**
```bash
python evaluate_IC.py
```

**Results:**
- RGCN: 95.8% alignment (highest - excellent consistency)
- CNN: 91.8% alignment (good consistency)
- RF: 82.4% alignment (acceptable consistency)

**Application:** Filter low-confidence predictions in production systems.

---

## Complete Evaluation Pipeline

Run all evaluations in hierarchical order:

```bash
# Navigate to scripts directory
cd xai_eval_output/modular_evaluation/scripts

# Tier 1 (Essential): Scaffold Recognition
python evaluate_SR.py

# Tier 2 (Deployment): Model Independence
python evaluate_MI.py

# Tier 3 (Validation): Context Sensitivity
python evaluate_CS.py

# Tier 4 (Confidence): Internal Consistency
python evaluate_IC.py
```

---

## Quick Reference - Manuscript Files

| Tier | Evaluation | Key Result File | Location |
|------|-----------|----------------|----------|
| 1 | SR | `sr_manuscript_table.csv` | `../sr/` |
| 2 | MI | `model_independence_manuscript_table.csv` | `../model_independence/` |
| 3 | CS | `cs_official_SR_aligned.csv` | `../cs_final/` |
| 4 | IC | `alignment_summary_corrected.csv` | `../prediction_attribution_alignment/` |

---

## Results Summary

### Scaffold Recognition (SR) - Tier 1

| Model | Complete Recognition | Status |
|-------|---------------------|--------|
| **RGCN** | 86-99% | ✅ PASS (≥90%) |
| **CNN** | 53-89% | ❌ FAIL (<90%) |
| **RF** | 94-100% | ✅ PASS (≥90%) |

### Model Independence (MI) - Tier 2

| Model | MI Score | Status |
|-------|----------|--------|
| **RGCN** | 0.709 | ⚠️ Moderate (≥0.70, <0.95) |
| **CNN** | 0.828 | ⚠️ Good (<0.95) |
| **RF** | 0.706 | ⚠️ Moderate (≥0.70, <0.95) |

### Context Sensitivity (CS) - Tier 3

| Model | Combined CS | Rank |
|-------|-------------|------|
| **RGCN** | 0.844 | 🥇 1st - Superior context awareness |
| **RF** | 0.794 | 🥈 2nd - Perfect discrimination |
| **CNN** | 0.000 (official) | 🥉 3rd - Cannot detect complete scaffolds |
| **CNN** | 0.462 (partial) | - Fragment detection only |

### Internal Consistency (IC) - Tier 4

| Model | Alignment Rate | Strong | Misaligned | Interpretation |
|-------|---------------|--------|------------|----------------|
| **RGCN** | 95.8% | 73.6% | 3.3% | Excellent consistency |
| **CNN** | 91.8% | 58.9% | 6.2% | Good consistency |
| **RF** | 82.4% | 73.1% | 11.1% | Acceptable consistency |

---

## Model-Specific Evaluation Parameters

### Top-K Values (SR and MI)

| Model | K | Coverage | Rationale |
|-------|---|----------|-----------|
| CNN | 3 | 73% of avg 4.1 substructures | High coverage |
| RGCN | 3 | 50% of avg 6.0 substructures | Moderate coverage |
| RF | 7 | 24% of avg 29.4 present features | Comparable selectivity |

### Attribution Magnitude Thresholds

| Model | SR Magnitude | IC Criterion | Rationale |
|-------|-------------|--------------|-----------|
| RGCN | >0.1 ✅ | Sign only | Localized attributions |
| CNN | >0.1 ✅ | Sign only | Localized attributions |
| RF | N/A ❌ | Sign only | Distributed attributions (0.001-0.03) |

**IC uses simple sign matching (architecture-agnostic):**
- Active prediction: mean(attributions) > 0
- Inactive prediction: mean(attributions) < 0
- No fixed magnitude thresholds (works for all architectures)

### RF Feature Filtering

**Problem:** TreeSHAP assigns attributions to ALL 85 features, regardless of presence.

**Solution:** Filter to only PRESENT features using `pos_features` column.

**Impact:**
- SR: Fluoroquinolones 94% → 98%
- IC: Achieves 82.4% alignment (using only present features)
- MI: More accurate stability measurement (0.706)

**Scripts using RF filtering:** All 4 scripts (evaluate_SR.py, evaluate_MI.py, evaluate_CS.py, evaluate_IC.py)

---

## Key Methodological Innovations

### 1. SR-Aligned CS Evaluation

Context Sensitivity now uses THE SAME scaffold detection as Scaffold Recognition:
- **RGCN:** Strict SMARTS (`c1ccc2ncccc2c1`, `N1C(=O)CC1`, `O1C(=O)NCC1`)
- **CNN:** Strict SMARTS (official, fails) + Relaxed SMARTS (partial, comparison)
- **RF:** Feature-based (`fr_pyridine`, `fr_lactam`, `fr_oxazole`)

This ensures methodological consistency: only measure context sensitivity for scaffolds models can actually recognize.

### 2. Dual CNN Reporting

- **Official:** Strict SMARTS (consistent with SR) → 0.000 (FAILED)
- **Partial:** Relaxed SMARTS (fragments only) → 0.462 (comparison)

Acknowledges CNN limitations while showing partial detection capability.

### 3. Normalized Combined Scoring

**Context Awareness normalization:**
```
Context_Awareness_Score = min(context_responsiveness / 3.0, 1.0)
```

**Combined CS Score:**
```
CS = 0.35×Directionality_Score + 0.35×Context_Awareness_Score + 0.30×Discrimination_Score
```

Where:
- Directionality_Score = 1.0 if significant, 0.0 if not
- Discrimination_Score = 1.0 if significant, 0.0 if not

### 4. Model-Specific Scaffold Recognition

**Architectural Fairness:**
- RGCN/CNN: Two metrics (Top-K + Magnitude >0.1) for localized attributions
- RF: One metric (Top-K only) for distributed attributions

RF attributions (0.001-0.03) are 10-100× smaller than RGCN/CNN due to TreeSHAP global distribution. Using >0.1 threshold for RF would unfairly penalize architectural differences.

---

## Dependencies

All scripts require:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scipy` - Statistical tests
- `rdkit` - Chemical structure handling (SR and CS only)
- `json` - Attribution parsing
- `os` - File handling

**Data Requirements:**
- Activity cliff pairs: `*_ensemble_cliffs.parquet` files
- Unique compounds: `unique_compounds_ensemble_*.parquet` files
- Individual models: `*_model{1-5}_cliffs.parquet` files (MI only)
- Located in: `xai_eval_output/{MODEL}_output/`

---

## Critical Notes

### Context Sensitivity (CS)

- **evaluate_CS.py** is the FINAL version (SR-aligned with dual CNN reporting)
- Previous versions (evaluate_cs_*.py) have been removed
- Uses strict SMARTS for RGCN, relaxed for CNN comparison, feature-based for RF

### Model Independence (MI)

- **evaluate_MI.py** generates model_independence_manuscript_table.csv
- Uses Jaccard/Spearman (not CV-based stability)
- Focuses on feature rankings (what matters for medicinal chemistry)

### Internal Consistency (IC)

- **evaluate_IC.py** measures prediction-attribution alignment using SIMPLE criterion
- **All models use same criterion:** Net attribution sign matches prediction direction
  - Active prediction: mean(attributions) > 0
  - Inactive prediction: mean(attributions) < 0
- Architecture-agnostic: Works for localized (RGCN/CNN) and distributed (RF) attributions
- Results: RGCN 95.8%, CNN 91.8%, RF 82.4% alignment
- Simple and interpretable: Direct test of whether explanations support predictions

---

## Output Directory Structure

```
modular_evaluation/
├── scripts/                                    # This directory (4 final scripts)
│   ├── evaluate_SR.py                         # Tier 1: Scaffold Recognition
│   ├── evaluate_MI.py                         # Tier 2: Model Independence
│   ├── evaluate_CS.py                         # Tier 3: Context Sensitivity
│   └── evaluate_IC.py                         # Tier 4: Internal Consistency
│
├── sr/                                        # SR outputs
│   ├── sr_manuscript_table.csv               # Publication-ready
│   └── sr_detailed_results.csv
│
├── model_independence/                        # MI outputs
│   ├── model_independence_manuscript_table.csv
│   ├── model_independence_summary.csv
│   └── model_independence_detailed_results.csv
│
├── cs_final/                                  # CS outputs (SR-aligned)
│   ├── cs_official_SR_aligned.csv            # Official results
│   └── cs_partial_CNN_comparison.csv         # CNN partial (comparison)
│
└── prediction_attribution_alignment/          # IC outputs
    ├── alignment_summary_corrected.csv
    └── alignment_detailed_results_corrected.csv
```

---

## For More Information

See parent directory `README_XAI_Framework.md` for:
- Complete framework methodology
- Statistical interpretation guidelines
- Architectural bias analysis
- Best practices for XAI evaluation in drug discovery

---

**Last Updated:** 2025-01-18
**Status:** Final 4-script structure with SR-aligned CS evaluation and dual CNN reporting
**Scripts:** `evaluate_SR.py`, `evaluate_MI.py`, `evaluate_CS.py`, `evaluate_IC.py`
