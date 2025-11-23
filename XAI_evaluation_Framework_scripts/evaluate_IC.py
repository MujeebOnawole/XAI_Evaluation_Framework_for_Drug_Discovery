"""
INTERNAL CONSISTENCY (IC) EVALUATION - CORRECTED VERSION

CRITICAL FIX: For RF, uses BOTH pos_features AND neg_features from CSV files
to properly calculate mean attribution across all present features.

Previous version only used pos_features, missing features with negative attributions,
leading to incorrectly inflated mean attributions and false misalignment reports.

SIMPLE CRITERION:
- Active prediction (prob >= 0.5): Net attribution should be POSITIVE
- Inactive prediction (prob < 0.5): Net attribution should be NEGATIVE

Input: unique_compounds_ensemble CSV files (not parquet)
Output: Alignment rates by model and class
"""

import pandas as pd
import json
import os
import numpy as np

# Configuration
BASE_DIR = r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\automatic_XAI"
EVAL_OUTPUT_DIR = os.path.join(BASE_DIR, "notebooks", "comparative_output", "xai_eval_output")
OUTPUT_DIR = os.path.join(EVAL_OUTPUT_DIR, "modular_evaluation", "prediction_attribution_alignment")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 100)
print("INTERNAL CONSISTENCY (IC) EVALUATION - CORRECTED VERSION")
print("=" * 100)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nCriterion: Net attribution sign should match prediction direction")
print("  - Active prediction: mean(attributions) > 0")
print("  - Inactive prediction: mean(attributions) < 0")
print("\nCRITICAL FIX: RF uses BOTH pos_features AND neg_features")

def calculate_alignment(attrs, pred_prob, pred_class, is_rf=False, pos_features=None, neg_features=None):
    """
    Calculate prediction-attribution alignment using SIMPLE criterion.

    CORRECTED: For RF, combines pos_features AND neg_features to get all present features.

    Args:
        attrs: Attribution data (dict for RF, list for RGCN/CNN)
        pred_prob: Prediction probability
        pred_class: Predicted class
        is_rf: Whether this is RF model
        pos_features: List of features with positive attributions (RF only)
        neg_features: List of features with negative attributions (RF only)

    Returns:
        dict with alignment status and statistics
    """
    if isinstance(attrs, str):
        attrs = json.loads(attrs)

    # Extract attribution values
    if is_rf:
        # RF: Get ALL present features (pos + neg)
        all_present_features = []

        if pos_features is not None:
            if isinstance(pos_features, str):
                pos_features = json.loads(pos_features)
            if isinstance(pos_features, list):
                all_present_features.extend(pos_features)

        if neg_features is not None:
            if isinstance(neg_features, str):
                neg_features = json.loads(neg_features)
            if isinstance(neg_features, list):
                all_present_features.extend(neg_features)

        # Get attributions for all present features
        if len(all_present_features) > 0:
            attr_values = [
                attrs[feat]
                for feat in all_present_features
                if feat in attrs
            ]
        else:
            # Fallback: use all attributions (shouldn't happen with CSV)
            attr_values = list(attrs.values()) if isinstance(attrs, dict) else []
    else:
        # RGCN/CNN: List of dicts with 'attribution' key
        if isinstance(attrs, list):
            attr_values = [attr.get('attribution', 0) for attr in attrs if isinstance(attr, dict)]
        else:
            attr_values = []

    if len(attr_values) == 0:
        return {
            'n_attributions': 0,
            'mean_attribution': 0,
            'aligned': False,
            'alignment_category': 'no_attributions'
        }

    # Calculate mean attribution
    mean_attr = np.mean(attr_values)

    # Simple alignment criterion: sign matches prediction
    if pred_class == 'active' or pred_prob >= 0.5:
        # Active prediction: need positive mean
        aligned = (mean_attr > 0)
        expected_sign = 'positive'
    else:
        # Inactive prediction: need negative mean
        aligned = (mean_attr < 0)
        expected_sign = 'negative'

    # Categorize alignment strength (for characterization)
    abs_mean = abs(mean_attr)

    # Model-specific magnitude thresholds (for characterization only)
    if is_rf:
        strong_threshold = 0.005  # RF has small attributions
        weak_threshold = 0.001
    else:
        strong_threshold = 0.05   # RGCN/CNN have larger attributions
        weak_threshold = 0.01

    if aligned:
        if abs_mean >= strong_threshold:
            category = 'strong_alignment'
        elif abs_mean >= weak_threshold:
            category = 'weak_alignment'
        else:
            category = 'very_weak_alignment'
    else:
        if abs_mean >= weak_threshold:
            category = 'misalignment'
        else:
            category = 'neutral'  # Near zero, neither aligned nor misaligned

    return {
        'n_attributions': len(attr_values),
        'mean_attribution': mean_attr,
        'aligned': aligned,
        'alignment_category': category,
        'expected_sign': expected_sign,
        'actual_sign': 'positive' if mean_attr > 0 else 'negative' if mean_attr < 0 else 'neutral'
    }

def evaluate_model(model_name, model_file):
    """Evaluate prediction-attribution alignment for a model."""

    print(f"\n{'=' * 100}")
    print(f"{model_name.upper()} - INTERNAL CONSISTENCY")
    print('=' * 100)

    # Read CSV file
    df = pd.read_csv(model_file)
    is_rf = 'RF' in model_name

    print(f"\nTotal compounds: {len(df)}")

    # Verify required columns
    required = ['pred_prob', 'pred_class', 'attributions', 'antibiotic_class']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return []

    # Check for RF feature lists
    has_pos_features = 'pos_features' in df.columns
    has_neg_features = 'neg_features' in df.columns

    if is_rf:
        if has_pos_features and has_neg_features:
            print(f"[INFO] RF: Using BOTH pos_features AND neg_features (CORRECTED)")
        elif has_pos_features:
            print("[WARNING] RF: Only pos_features found (may be incomplete)")
        else:
            print("[WARNING] RF: No feature lists (will use all 85 features)")

    results = []

    for idx, row in df.iterrows():
        # Get feature lists for RF
        pos_features = None
        neg_features = None

        if is_rf:
            if has_pos_features:
                pos_features = row['pos_features']
            if has_neg_features:
                neg_features = row['neg_features']

        # Calculate alignment
        alignment = calculate_alignment(
            row['attributions'],
            row['pred_prob'],
            row['pred_class'],
            is_rf,
            pos_features,
            neg_features
        )

        results.append({
            'model': model_name,
            'compound_id': row['compound_id'],
            'smiles': row['smiles'],
            'antibiotic_class': row['antibiotic_class'],
            'pred_prob': row['pred_prob'],
            'pred_class': row['pred_class'],
            **alignment
        })

    return results

# Main execution
print("\n" + "=" * 100)
print("EVALUATING ALL MODELS")
print("=" * 100)

# Use CSV files instead of parquet
model_files = {
    'RF': os.path.join(EVAL_OUTPUT_DIR, 'eval_result', 'RF_output', 'unique_compounds_ensemble_RF.csv'),
    'CNN': os.path.join(EVAL_OUTPUT_DIR, 'eval_result', 'CNN_output', 'unique_compounds_ensemble_CNN.csv'),
    'RGCN': os.path.join(EVAL_OUTPUT_DIR, 'eval_result', 'RGCN_output', 'unique_compounds_ensemble_RGCN.csv')
}

all_results = []

for model_name, model_file in model_files.items():
    if not os.path.exists(model_file):
        print(f"\n[WARNING] File not found: {model_file}")
        continue

    results = evaluate_model(model_name, model_file)
    all_results.extend(results)

# Create DataFrame
results_df = pd.DataFrame(all_results)

# Save detailed results
detailed_file = os.path.join(OUTPUT_DIR, "alignment_detailed_results_corrected.csv")
results_df.to_csv(detailed_file, index=False)
print(f"\n[SUCCESS] Detailed results: {detailed_file}")

# Calculate summary statistics
print(f"\n{'=' * 100}")
print("SUMMARY STATISTICS")
print('=' * 100)

summary_data = []

# Get all unique classes from data (dynamically)
all_classes = sorted(results_df['antibiotic_class'].unique())
print(f"Found antibiotic classes: {all_classes}")

for model in ['RF', 'CNN', 'RGCN']:  # Fixed order
    if model not in results_df['model'].unique():
        continue

    model_df = results_df[results_df['model'] == model]

    for ab_class in all_classes:  # Use actual classes from data
        class_df = model_df[model_df['antibiotic_class'] == ab_class]

        if len(class_df) == 0:
            continue

        n_compounds = len(class_df)
        n_aligned = class_df['aligned'].sum()
        alignment_rate = (n_aligned / n_compounds) * 100

        # Mean attribution statistics
        mean_attribution = class_df['mean_attribution'].mean()
        std_attribution = class_df['mean_attribution'].std()

        # Category breakdown
        category_counts = class_df['alignment_category'].value_counts()

        summary_data.append({
            'model': model,
            'antibiotic_class': ab_class,
            'n_compounds': n_compounds,
            'n_aligned': n_aligned,
            'alignment_rate': alignment_rate,
            'mean_attribution': mean_attribution,
            'std_attribution': std_attribution,
            'n_strong': category_counts.get('strong_alignment', 0),
            'n_weak': category_counts.get('weak_alignment', 0),
            'n_very_weak': category_counts.get('very_weak_alignment', 0),
            'n_neutral': category_counts.get('neutral', 0),
            'n_misaligned': category_counts.get('misalignment', 0)
        })

summary_df = pd.DataFrame(summary_data)

# Save summary
summary_file = os.path.join(OUTPUT_DIR, "alignment_summary_corrected.csv")
summary_df.to_csv(summary_file, index=False)
print(f"[SUCCESS] Summary: {summary_file}")

# Display summary in requested format
print(f"\n{'=' * 100}")
print("Internal Consistency: Prediction-Attribution Alignment")
print("Evaluation of explanation-prediction coherence across 450 unique compounds from activity cliff pairs")
print('=' * 100)

print("\nModel    Antibiotic Class    N    Alignment Rate (%)")

for model in ['RF', 'CNN', 'RGCN']:
    model_summary = summary_df[summary_df['model'] == model]

    if len(model_summary) == 0:
        continue

    print(f"\n{model}")
    for _, row in model_summary.iterrows():
        # Format class name (normalize for display)
        class_name = row['antibiotic_class']
        # Standardize names for display
        if 'beta' in class_name.lower() or 'lactam' in class_name.lower():
            display_name = 'beta-lactam'
        elif 'fluoroquinolone' in class_name.lower():
            display_name = 'Fluoroquinolone'
        elif 'oxazolidinone' in class_name.lower():
            display_name = 'Oxazolidinone'
        else:
            display_name = class_name

        print(f"    {display_name:<19} {row['n_compounds']:<4} {row['alignment_rate']:.1f}")

# Calculate overall performance
print("\n" + "=" * 100)
overall_stats = []
for model in ['RF', 'CNN', 'RGCN']:
    model_data = summary_df[summary_df['model'] == model]
    if len(model_data) == 0:
        continue

    total_compounds = model_data['n_compounds'].sum()
    total_aligned = model_data['n_aligned'].sum()
    alignment_rate = (total_aligned / total_compounds) * 100

    overall_stats.append(f"{model} = {alignment_rate:.1f}%")

print(f"Overall Performance: {', '.join(overall_stats)}")

print("\n" + "=" * 100)
print("INTERNAL CONSISTENCY EVALUATION COMPLETE (CORRECTED)")
print("=" * 100)
