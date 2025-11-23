"""
MODULAR EVALUATION: MODEL INDEPENDENCE

Evaluates consistency of XAI attributions across individual model instances
within each architecture (5 models per architecture).

Purpose: Determine if different model instances within the same architecture
provide similar explanations for the same compounds (inter-model reliability).

Methodology:
- Compares attributions for the same compound across 5 individual model instances
- Measures consistency using:
  1. Top-K overlap (Jaccard similarity)
     * CNN/RGCN: Top-3 substructures (50-73% coverage)
     * RF: Top-7 features (24% coverage of ~29 present features)
  2. Rank correlation (Spearman correlation of attribution ranks)

Interpretation:
- Excellent stability (≥0.90): Suitable for pharmaceutical applications
- Moderate stability (0.70-0.89): Acceptable for exploratory analysis
- Poor stability (<0.70): Systematic inconsistencies compromising reliability
- Perfect values (1.00): May indicate overfitting or lack of diversity

Model-specific K values account for different granularities:
- CNN averages 4.1 substructures → Top-3 captures 73%
- RGCN averages 6.0 substructures → Top-3 captures 50%
- RF averages 29.4 present features → Top-7 captures 24%

Uses activity cliff pairs from individual model files:
- model1_cliffs.parquet through model5_cliffs.parquet for each architecture

Output Directory: xai_eval_output/modular_evaluation/model_independence/
"""

import pandas as pd
import json
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
from itertools import combinations

# Configuration
BASE_DIR = r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\automatic_XAI"
EVAL_OUTPUT_DIR = os.path.join(BASE_DIR, "notebooks", "comparative_output", "xai_eval_output")
OUTPUT_DIR = os.path.join(EVAL_OUTPUT_DIR, "modular_evaluation", "model_independence")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 100)
print("MODULAR EVALUATION: MODEL INDEPENDENCE")
print("=" * 100)
print(f"\nOutput directory: {OUTPUT_DIR}")

def get_top_k_items(attrs, k=None, is_rf=False, pos_features=None):
    """
    Extract top-K items from attributions with model-specific defaults.

    Args:
        attrs: Attribution data (dict for RF, list for RGCN/CNN)
        k: Number of top items to return (None = use model-specific default)
        is_rf: Whether this is RF model
        pos_features: List of present features (for RF filtering)

    Returns:
        List of top-K feature names or substructure SMILES

    Model-specific defaults:
        - RF: K=7 (24% of avg 29.4 present features)
        - CNN/RGCN: K=3 (50-73% of avg 4-6 substructures)
    """
    # Set model-specific defaults
    if k is None:
        k = 7 if is_rf else 3
    if is_rf:
        # RF: Get top-K feature names (filtered to present features if available)
        if isinstance(attrs, dict):
            # IMPROVED: Filter to only PRESENT features if pos_features provided
            if pos_features is not None:
                if isinstance(pos_features, str):
                    pos_features = json.loads(pos_features)
                # Only rank present features
                filtered_attrs = {feat: val for feat, val in attrs.items() if feat in pos_features}
                sorted_items = sorted(filtered_attrs.items(), key=lambda x: abs(x[1]), reverse=True)
            else:
                # Fallback: use all features (old behavior)
                sorted_items = sorted(attrs.items(), key=lambda x: abs(x[1]), reverse=True)
            return [item[0] for item in sorted_items[:k]]
    else:
        # CNN/RGCN: Get top-K substructures
        if isinstance(attrs, list):
            return [attr.get('substructure', '') for attr in attrs[:k]]
    return []

def jaccard_similarity(list1, list2):
    """Calculate Jaccard similarity between two lists."""
    set1 = set(list1)
    set2 = set(list2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def rank_correlation(attrs1, attrs2, is_rf=False, pos_features1=None, pos_features2=None):
    """
    Calculate Spearman rank correlation.

    Args:
        attrs1: Attribution data from model 1
        attrs2: Attribution data from model 2
        is_rf: Whether this is RF model
        pos_features1: Present features in compound from model 1 (for RF filtering)
        pos_features2: Present features in compound from model 2 (for RF filtering)

    Returns:
        Spearman correlation coefficient or None
    """
    if is_rf:
        # RF: Use common features (filtered to present if available)
        if not isinstance(attrs1, dict) or not isinstance(attrs2, dict):
            return None

        # IMPROVED: Filter to present features if available
        if pos_features1 is not None and pos_features2 is not None:
            if isinstance(pos_features1, str):
                pos_features1 = json.loads(pos_features1)
            if isinstance(pos_features2, str):
                pos_features2 = json.loads(pos_features2)
            # Get intersection of present features from both models
            common_features = set(pos_features1) & set(pos_features2)
        else:
            # Fallback: use all common features
            common_features = set(attrs1.keys()) & set(attrs2.keys())

        if len(common_features) < 3:
            return None
        values1 = [attrs1[f] for f in common_features]
        values2 = [attrs2[f] for f in common_features]
    else:
        # CNN/RGCN: Use attribution values from top-3 (model-specific K)
        if not isinstance(attrs1, list) or not isinstance(attrs2, list):
            return None
        k = 3  # Model-specific: top-3 for CNN/RGCN (50-73% coverage)
        values1 = [abs(a.get('attribution', 0)) for a in attrs1[:k]]
        values2 = [abs(a.get('attribution', 0)) for a in attrs2[:k]]
        if len(values1) < 2 or len(values2) < 2:  # Need at least 2 values
            return None

    try:
        corr, _ = spearmanr(values1, values2)
        return corr if not np.isnan(corr) else None
    except:
        return None

def evaluate_model_independence(model_name, model_files):
    """Evaluate model independence (inter-model consistency) across individual model instances."""

    print(f"\n{'=' * 100}")
    print(f"{model_name.upper()} - MODEL INDEPENDENCE EVALUATION")
    print('=' * 100)

    is_rf = 'RF' in model_name

    # Load all individual model files
    dfs = {}
    for i, file_path in enumerate(model_files, 1):
        df = pd.read_parquet(file_path)
        dfs[f'model{i}'] = df
        print(f"  Loaded model{i}: {len(df)} pairs")

    # Check for RF feature presence columns
    if is_rf:
        has_pos_features = 'pos_features_active' in dfs['model1'].columns
        if has_pos_features:
            print(f"\n[RF INFO] Using 'pos_features_active' to filter RF to PRESENT features only")
            print(f"[RF INFO] This improves stability measurements by excluding noisy absent features")
        else:
            print("\n[WARNING] RF model missing 'pos_features_active' column")
            print("[WARNING] Will evaluate all 85 features (may show artificially low stability)")

    # Get compounds that appear in all models
    compound_ids = set(dfs['model1']['compound_active_id'].unique())
    for model_key in dfs:
        compound_ids &= set(dfs[model_key]['compound_active_id'].unique())

    print(f"\n  Common compounds across all models: {len(compound_ids)}")

    # RF uses 'feature_attr_active', CNN/RGCN use 'substruct_attr_active'
    attr_col = 'feature_attr_active' if is_rf else 'substruct_attr_active'

    results = []

    # Compare each pair of models
    model_pairs = list(combinations(dfs.keys(), 2))

    for model1_key, model2_key in model_pairs:
        df1 = dfs[model1_key]
        df2 = dfs[model2_key]

        jaccard_scores = []
        rank_corrs = []

        for compound_id in list(compound_ids)[:100]:  # Sample 100 compounds for efficiency
            # Get attributions from both models
            row1 = df1[df1['compound_active_id'] == compound_id]
            row2 = df2[df2['compound_active_id'] == compound_id]

            if len(row1) == 0 or len(row2) == 0:
                continue

            attrs1 = row1.iloc[0][attr_col]
            attrs2 = row2.iloc[0][attr_col]

            if isinstance(attrs1, str):
                attrs1 = json.loads(attrs1)
            if isinstance(attrs2, str):
                attrs2 = json.loads(attrs2)

            # Get pos_features for RF filtering
            pos_features1 = None
            pos_features2 = None
            if is_rf and 'pos_features_active' in df1.columns:
                pos_features1 = row1.iloc[0]['pos_features_active']
            if is_rf and 'pos_features_active' in df2.columns:
                pos_features2 = row2.iloc[0]['pos_features_active']

            # Calculate Jaccard similarity (uses model-specific K: RF=7, CNN/RGCN=3)
            top_k1 = get_top_k_items(attrs1, is_rf=is_rf, pos_features=pos_features1)
            top_k2 = get_top_k_items(attrs2, is_rf=is_rf, pos_features=pos_features2)
            jaccard = jaccard_similarity(top_k1, top_k2)
            jaccard_scores.append(jaccard)

            # Calculate rank correlation
            rank_corr = rank_correlation(attrs1, attrs2, is_rf=is_rf, pos_features1=pos_features1, pos_features2=pos_features2)
            if rank_corr is not None:
                rank_corrs.append(rank_corr)

        mean_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
        mean_rank_corr = np.mean(rank_corrs) if rank_corrs else 0

        print(f"  {model1_key} vs {model2_key}: Jaccard={mean_jaccard:.3f}, Rank Corr={mean_rank_corr:.3f}")

        results.append({
            'model': model_name,
            'comparison': f'{model1_key}_vs_{model2_key}',
            'n_compounds': len(jaccard_scores),
            'mean_jaccard_similarity': mean_jaccard,
            'mean_rank_correlation': mean_rank_corr,
            'std_jaccard': np.std(jaccard_scores) if jaccard_scores else 0,
            'std_rank_corr': np.std(rank_corrs) if rank_corrs else 0
        })

    return results

# Main execution
print("\n" + "=" * 100)
print("EVALUATING MODEL INDEPENDENCE FOR ALL MODELS")
print("=" * 100)

all_results = []

# RGCN
print("\n" + "=" * 100)
print("RGCN")
print("=" * 100)
rgcn_files = [
    os.path.join(EVAL_OUTPUT_DIR, 'RGCN_output', f'rgcn_model{i}_cliffs.parquet')
    for i in range(1, 6)
]
rgcn_results = evaluate_model_independence('RGCN', rgcn_files)
all_results.extend(rgcn_results)

# CNN
print("\n" + "=" * 100)
print("CNN")
print("=" * 100)
cnn_files = [
    os.path.join(EVAL_OUTPUT_DIR, 'CNN_output', f'cnn_model{i}_cliffs.parquet')
    for i in range(1, 6)
]
cnn_results = evaluate_model_independence('CNN', cnn_files)
all_results.extend(cnn_results)

# RF
print("\n" + "=" * 100)
print("RF")
print("=" * 100)
rf_files = [
    os.path.join(EVAL_OUTPUT_DIR, 'RF_output', f'rf_model{i}_cliffs.parquet')
    for i in range(1, 6)
]
rf_results = evaluate_model_independence('RF', rf_files)
all_results.extend(rf_results)

# Create DataFrame
results_df = pd.DataFrame(all_results)

# Save detailed results
detailed_file = os.path.join(OUTPUT_DIR, "model_independence_detailed_results.csv")
results_df.to_csv(detailed_file, index=False)

print(f"\n{'=' * 100}")
print("CALCULATING SUMMARY STATISTICS")
print('=' * 100)

# Calculate summary by model
summary_data = []
for model in results_df['model'].unique():
    model_df = results_df[results_df['model'] == model]

    summary_data.append({
        'model': model,
        'n_comparisons': len(model_df),
        'mean_jaccard_similarity': model_df['mean_jaccard_similarity'].mean(),
        'std_jaccard_similarity': model_df['mean_jaccard_similarity'].std(),
        'mean_rank_correlation': model_df['mean_rank_correlation'].mean(),
        'std_rank_correlation': model_df['mean_rank_correlation'].std()
    })

summary_df = pd.DataFrame(summary_data)

# Save summary
summary_file = os.path.join(OUTPUT_DIR, "model_independence_summary.csv")
summary_df.to_csv(summary_file, index=False)

print(f"\n[SUCCESS] Detailed results saved to: {detailed_file}")
print(f"[SUCCESS] Summary saved to: {summary_file}")

# Display summary
print(f"\n{'=' * 100}")
print("MODEL INDEPENDENCE SUMMARY")
print('=' * 100)

print("\n| Model | Comparisons | Mean Jaccard | Std Jaccard | Mean Rank Corr | Std Rank Corr |")
print("|-------|-------------|--------------|-------------|----------------|---------------|")
for _, row in summary_df.iterrows():
    print(f"| {row['model']} | {row['n_comparisons']} | "
          f"{row['mean_jaccard_similarity']:.3f} | {row['std_jaccard_similarity']:.3f} | "
          f"{row['mean_rank_correlation']:.3f} | {row['std_rank_correlation']:.3f} |")

# Create manuscript table
manuscript_df = summary_df[['model', 'mean_jaccard_similarity', 'mean_rank_correlation']].copy()
manuscript_df = manuscript_df.rename(columns={
    'mean_jaccard_similarity': 'Top-K Overlap (Jaccard)',
    'mean_rank_correlation': 'Rank Correlation (Spearman)'
})

manuscript_file = os.path.join(OUTPUT_DIR, "model_independence_manuscript_table.csv")
manuscript_df.to_csv(manuscript_file, index=False)

print(f"\n[SUCCESS] Manuscript table saved to: {manuscript_file}")

print("\nManuscript Table:")
print(manuscript_df.to_string(index=False))

print("\n" + "=" * 100)
print("KEY FINDINGS - MODEL INDEPENDENCE")
print("=" * 100)
print("\n1. Model Independence measured across 5 individual model instances per architecture")
print("2. Model-specific Top-K:")
print("   - CNN/RGCN: Top-3 substructures (50-73% coverage)")
print("   - RF: Top-7 features (24% coverage of present features)")
print("3. Jaccard similarity: Overlap of top-K features/substructures between model pairs")
print("4. Rank correlation: Consistency of attribution ranking between model pairs")
print("\nINTERPRETATION:")
print("  - Excellent stability (>=0.90): Suitable for pharmaceutical applications")
print("  - Moderate stability (0.70-0.89): Acceptable for exploratory analysis")
print("  - Poor stability (<0.70): Systematic inconsistencies compromising reliability")
print("  - Perfect values (1.00): May indicate overfitting or lack of ensemble diversity")
print("\nPURPOSE: Determines if architecture provides reliable explanations regardless")
print("         of which specific model instance is used for pharmaceutical deployment.")
print("\n" + "=" * 100)
print("MODEL INDEPENDENCE EVALUATION COMPLETE")
print("=" * 100)
