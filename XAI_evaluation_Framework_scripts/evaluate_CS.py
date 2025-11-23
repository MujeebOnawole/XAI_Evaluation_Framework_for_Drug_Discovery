"""
FINAL CONTEXT SENSITIVITY EVALUATION - SR-ALIGNED WITH DUAL CNN REPORTING

METHODOLOGY:
- RGCN: Strict SMARTS (official - same as SR complete recognition)
- CNN: BOTH strict (official, expected to fail) AND relaxed (partial, for comparison)
- RF: Feature-based (official - same as SR)

OUTPUT:
- Official CS scores use strict/feature-based only
- CNN partial results reported separately for transparency
- Acknowledges CNN limitations while showing what it CAN detect
"""

import pandas as pd
import json
import os
import numpy as np
from scipy.stats import ttest_1samp, t as t_dist, levene
from rdkit import Chem

try:
    from scipy.stats import binomtest
except ImportError:
    from scipy.stats import binom_test as binomtest

# Configuration
BASE_DIR = r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\automatic_XAI"
EVAL_OUTPUT_DIR = os.path.join(BASE_DIR, "notebooks", "comparative_output", "xai_eval_output")
OUTPUT_DIR = os.path.join(EVAL_OUTPUT_DIR, "modular_evaluation", "cs_final")
CS_DIR = os.path.join(EVAL_OUTPUT_DIR, "modular_evaluation", "cs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 100)
print("CONTEXT SENSITIVITY EVALUATION - FINAL SR-ALIGNED")
print("=" * 100)
print("\nMETHODOLOGY ALIGNMENT WITH SCAFFOLD RECOGNITION:")
print("  RGCN: Strict SMARTS patterns (official)")
print("  CNN:  BOTH strict (official) AND relaxed (partial comparison)")
print("  RF:   Feature-based (official)")

# Scaffold components (FROM SR EVALUATION)
SCAFFOLD_COMPONENTS = {
    'Fluoroquinolones': {
        'components': [
            {
                'name': 'Bicyclic Quinolone Core',
                'smarts': 'c1ccc2ncccc2c1',
                'smarts_relaxed': ['c1ccncc1', 'n1ccccc1', 'c1cnccc1'],
                'rf_features': ['fr_pyridine', 'fr_bicyclic', 'fr_Ar_N']
            },
            {
                'name': 'Carboxylic Acid',
                'smarts': 'C(=O)O',
                'smarts_relaxed': ['C(=O)O', 'C(O)=O'],
                'rf_features': ['fr_COO', 'fr_COO2', 'fr_Ar_COO']
            }
        ]
    },
    'Beta-lactam': {
        'components': [
            {
                'name': 'Beta-lactam Ring',
                'smarts': 'N1C(=O)CC1',
                'smarts_relaxed': ['NC(=O)C', 'C(=O)N', 'N1CCC1', 'N1CC1'],
                'rf_features': ['fr_lactam', 'fr_amide']
            }
        ]
    },
    'Oxazolidinone': {
        'components': [
            {
                'name': 'Oxazolidinone Ring',
                'smarts': 'O1C(=O)NCC1',
                'smarts_relaxed': ['O1CCNC1', 'OC(=O)N', 'O1CCC1', 'NC(=O)O'],
                'rf_features': ['fr_oxazole', 'fr_lactone', 'fr_urea']
            }
        ]
    }
}

def check_smarts_in_substructure(substructure_smiles, smarts_pattern):
    """Check if substructure contains SMARTS pattern (FROM SR)."""
    try:
        mol = Chem.MolFromSmiles(substructure_smiles)
        if mol is None:
            return False
        pattern = Chem.MolFromSmarts(smarts_pattern)
        if pattern is None:
            return False
        return mol.HasSubstructMatch(pattern)
    except:
        return False

def get_scaffold_attribution_atom_based(attrs, scaffold_components, use_relaxed=False):
    """Get scaffold attribution for RGCN/CNN using SMARTS (FROM SR)."""
    if not isinstance(attrs, list):
        return 0.0

    total_scaffold_attr = 0.0
    for component in scaffold_components:
        if use_relaxed and 'smarts_relaxed' in component:
            smarts_list = component['smarts_relaxed']
        else:
            smarts_list = [component['smarts']]

        for attr in attrs:
            substructure = attr.get('substructure', '')
            attribution = attr.get('attribution', 0)
            for smarts in smarts_list:
                if check_smarts_in_substructure(substructure, smarts):
                    total_scaffold_attr += attribution  # SIGNED
                    break

    return total_scaffold_attr

def get_scaffold_attribution_rf(attrs, scaffold_components):
    """Get scaffold attribution for RF (FROM SR)."""
    if not isinstance(attrs, dict):
        return 0.0

    total_scaffold_attr = 0.0
    for component in scaffold_components:
        rf_features = component['rf_features']
        for feature in rf_features:
            if feature in attrs:
                total_scaffold_attr += attrs[feature]

    return total_scaffold_attr

def calculate_context_metrics(deltas):
    """Calculate context awareness metrics."""
    deltas = np.array(deltas)
    if len(deltas) == 0:
        return {'context_responsiveness': 0, 'bidirectional_ratio': 0, 'dynamic_range': 0, 'iqr': 0}

    iqr = np.percentile(deltas, 75) - np.percentile(deltas, 25)
    mean_abs = abs(np.mean(deltas))
    context_responsiveness = iqr / mean_abs if mean_abs > 0 else 0
    negative_ratio = (deltas < 0).sum() / len(deltas)
    dynamic_range = np.max(deltas) - np.min(deltas)

    return {
        'context_responsiveness': context_responsiveness,
        'bidirectional_ratio': negative_ratio,
        'dynamic_range': dynamic_range,
        'iqr': iqr
    }

def evaluate_class(class_df, scaffold_components, is_rf=False, use_relaxed=False):
    """Evaluate single antibiotic class."""
    attr_col_active = 'feature_attr_active' if is_rf else 'substruct_attr_active'
    attr_col_inactive = 'feature_attr_inactive' if is_rf else 'substruct_attr_inactive'

    deltas = []

    for idx, row in class_df.iterrows():
        attrs_active = row[attr_col_active]
        attrs_inactive = row[attr_col_inactive]

        if isinstance(attrs_active, str):
            attrs_active = json.loads(attrs_active)
        if isinstance(attrs_inactive, str):
            attrs_inactive = json.loads(attrs_inactive)

        if is_rf:
            A_scaffold_active = get_scaffold_attribution_rf(attrs_active, scaffold_components)
            A_scaffold_inactive = get_scaffold_attribution_rf(attrs_inactive, scaffold_components)
        else:
            A_scaffold_active = get_scaffold_attribution_atom_based(attrs_active, scaffold_components, use_relaxed)
            A_scaffold_inactive = get_scaffold_attribution_atom_based(attrs_inactive, scaffold_components, use_relaxed)

        delta = A_scaffold_active - A_scaffold_inactive
        deltas.append(delta)

    deltas = np.array(deltas)
    n = len(deltas)

    if n == 0 or np.all(deltas == 0):
        return None

    # Statistics
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas, ddof=1)

    # Directionality test (Student 1908)
    t_stat, p_value = ttest_1samp(deltas, popmean=0, alternative='greater')
    ci_95 = t_dist.interval(0.95, df=n-1, loc=mean_delta, scale=std_delta/np.sqrt(n))

    # Context metrics
    context_metrics = calculate_context_metrics(deltas)
    positive_ratio = (deltas > 0).sum() / n
    negative_ratio = (deltas < 0).sum() / n

    # Significance determination
    directionality_significant = p_value < 0.05 and mean_delta > 0
    context_responsiveness = context_metrics['context_responsiveness']

    # Normalize context awareness to [0, 1] using 3.0 scaling factor
    context_awareness_score = min(context_responsiveness / 3.0, 1.0) if context_responsiveness > 0 else 0.0

    # Directionality score: proportion of positive deltas when significant (per methodology)
    directionality_score = positive_ratio if directionality_significant else 0.0

    return {
        'n_pairs': n,
        'mean_delta': mean_delta,
        'std_delta': std_delta,
        't_statistic': t_stat,
        'p_value_directionality': p_value,
        'directionality_significant': directionality_significant,
        'ci_95_lower': ci_95[0],
        'ci_95_upper': ci_95[1],
        'positive_ratio': positive_ratio,
        'negative_ratio': negative_ratio,
        'directionality_score': directionality_score,
        'context_responsiveness': context_responsiveness,
        'context_awareness_score': context_awareness_score,
        'dynamic_range': context_metrics['dynamic_range'],
        'bidirectional_ratio': context_metrics['bidirectional_ratio']
    }

def evaluate_model(model_name, model_file):
    """Evaluate model with dual CNN reporting."""

    print(f"\n{'=' * 100}")
    print(f"{model_name.upper()}")
    print('=' * 100)

    df = pd.read_parquet(model_file)
    is_rf = 'RF' in model_name
    is_cnn = 'CNN' in model_name

    results_official = []
    results_partial = [] if is_cnn else None
    all_deltas_official = []
    all_deltas_partial = [] if is_cnn else None

    for ab_class in ['Fluoroquinolones', 'Beta-lactam', 'Oxazolidinone']:
        class_df = df[df['antibiotic_class'] == ab_class]
        if len(class_df) == 0:
            continue

        scaffold_components = SCAFFOLD_COMPONENTS[ab_class]['components']

        # OFFICIAL evaluation (strict for RGCN/CNN, feature-based for RF)
        official_result = evaluate_class(class_df, scaffold_components, is_rf=is_rf, use_relaxed=False)

        if official_result:
            official_result['model'] = model_name
            official_result['antibiotic_class'] = ab_class
            official_result['scaffold_detection'] = 'Strict SMARTS' if not is_rf else 'Feature-based'
            official_result['evaluation_type'] = 'Official'
            results_official.append(official_result)

            # Collect deltas for Levene test
            attr_col_a = 'feature_attr_active' if is_rf else 'substruct_attr_active'
            attr_col_i = 'feature_attr_inactive' if is_rf else 'substruct_attr_inactive'

            for idx, row in class_df.iterrows():
                attrs_a = row[attr_col_a]
                attrs_i = row[attr_col_i]

                if isinstance(attrs_a, str):
                    attrs_a = json.loads(attrs_a)
                if isinstance(attrs_i, str):
                    attrs_i = json.loads(attrs_i)

                if is_rf:
                    delta = get_scaffold_attribution_rf(attrs_a, scaffold_components) - get_scaffold_attribution_rf(attrs_i, scaffold_components)
                else:
                    delta = get_scaffold_attribution_atom_based(attrs_a, scaffold_components, False) - get_scaffold_attribution_atom_based(attrs_i, scaffold_components, False)
                all_deltas_official.append(delta)

            print(f"\n{ab_class} (OFFICIAL - {official_result['scaffold_detection']}):")
            print(f"  Mean Delta: {official_result['mean_delta']:.4f}, p={official_result['p_value_directionality']:.4e} ({'SIG' if official_result['directionality_significant'] else 'NS'})")
            print(f"  Context Responsiveness: {official_result['context_responsiveness']:.3f}")
        else:
            print(f"\n{ab_class} (OFFICIAL): FAILED - Cannot detect scaffold")
            results_official.append({
                'model': model_name,
                'antibiotic_class': ab_class,
                'scaffold_detection': 'Strict SMARTS (FAILED)' if is_cnn else 'Strict SMARTS',
                'evaluation_type': 'Official',
                'n_pairs': len(class_df),
                'mean_delta': 0.0,
                'directionality_significant': False,
                'directionality_score': 0.0,
                'context_responsiveness': 0.0,
                'context_awareness_score': 0.0,
                'p_value_directionality': 1.0
            })

        # CNN PARTIAL evaluation (for comparison only)
        if is_cnn:
            partial_result = evaluate_class(class_df, scaffold_components, is_rf=False, use_relaxed=True)

            if partial_result:
                partial_result['model'] = model_name
                partial_result['antibiotic_class'] = ab_class
                partial_result['scaffold_detection'] = 'Relaxed SMARTS (partial)'
                partial_result['evaluation_type'] = 'Partial (comparison only)'
                results_partial.append(partial_result)

                # Collect deltas
                for idx, row in class_df.iterrows():
                    attrs_active = json.loads(row['substruct_attr_active']) if isinstance(row['substruct_attr_active'], str) else row['substruct_attr_active']
                    attrs_inactive = json.loads(row['substruct_attr_inactive']) if isinstance(row['substruct_attr_inactive'], str) else row['substruct_attr_inactive']
                    delta = get_scaffold_attribution_atom_based(attrs_active, scaffold_components, True) - get_scaffold_attribution_atom_based(attrs_inactive, scaffold_components, True)
                    all_deltas_partial.append(delta)

                print(f"  {ab_class} (PARTIAL - for comparison):")
                print(f"    Mean Delta: {partial_result['mean_delta']:.4f}, p={partial_result['p_value_directionality']:.4e} ({'SIG' if partial_result['directionality_significant'] else 'NS'})")
                print(f"    Context Responsiveness: {partial_result['context_responsiveness']:.3f}")

    # Levene test for official results
    if len(all_deltas_official) >= 300:
        deltas_fluoro = all_deltas_official[0:100]
        deltas_beta = all_deltas_official[100:200]
        deltas_oxaz = all_deltas_official[200:300]
        W_stat, p_value_levene = levene(deltas_fluoro, deltas_beta, deltas_oxaz, center='median')
        context_aware_sig = p_value_levene < 0.05

        print(f"\n  Levene's Test (OFFICIAL): W={W_stat:.3f}, p={p_value_levene:.4e} ({'SIG' if context_aware_sig else 'NS'})")

        for r in results_official:
            r['levene_W'] = W_stat
            r['p_value_context_awareness'] = p_value_levene
            r['context_awareness_significant'] = context_aware_sig

    # Levene for CNN partial
    if is_cnn and results_partial and len(all_deltas_partial) >= 300:
        deltas_fluoro_p = all_deltas_partial[0:100]
        deltas_beta_p = all_deltas_partial[100:200]
        deltas_oxaz_p = all_deltas_partial[200:300]
        W_stat_p, p_value_levene_p = levene(deltas_fluoro_p, deltas_beta_p, deltas_oxaz_p, center='median')

        print(f"  Levene's Test (PARTIAL): W={W_stat_p:.3f}, p={p_value_levene_p:.4e}")

        for r in results_partial:
            r['levene_W'] = W_stat_p
            r['p_value_context_awareness'] = p_value_levene_p
            r['context_awareness_significant'] = p_value_levene_p < 0.05

    return results_official, results_partial

# Main execution
model_files = {
    'RGCN': os.path.join(EVAL_OUTPUT_DIR, 'RGCN_output', 'rgcn_ensemble_cliffs.parquet'),
    'CNN': os.path.join(EVAL_OUTPUT_DIR, 'CNN_output', 'cnn_ensemble_cliffs.parquet'),
    'RF': os.path.join(EVAL_OUTPUT_DIR, 'RF_output', 'rf_ensemble_cliffs.parquet')
}

all_results_official = []
all_results_partial = []

for model_name, model_file in model_files.items():
    official, partial = evaluate_model(model_name, model_file)
    all_results_official.extend(official)
    if partial:
        all_results_partial.extend(partial)

# Add discrimination
discrimination_df = pd.read_csv(os.path.join(CS_DIR, "cs_structural_change_discrimination.csv"))

for i, row in enumerate(all_results_official):
    disc_row = discrimination_df[(discrimination_df['model'] == row['model']) &
                                 (discrimination_df['antibiotic_class'] == row['antibiotic_class'])]
    if len(disc_row) > 0:
        disc_row = disc_row.iloc[0]
        try:
            result = binomtest(disc_row['n_discriminative'], disc_row['n_pairs'], p=0.5, alternative='greater')
            p_value_binom = result.pvalue
        except:
            p_value_binom = binomtest(disc_row['n_discriminative'], disc_row['n_pairs'], p=0.5, alternative='greater')

        all_results_official[i]['discrimination_rate'] = disc_row['discrimination_rate']
        all_results_official[i]['p_value_discrimination'] = p_value_binom
        all_results_official[i]['discrimination_significant'] = p_value_binom < 0.05

# Calculate combined CS scores (per methodology: proportions, not binary)
for i, row in enumerate(all_results_official):
    # Component scores: directionality already calculated in evaluate_class
    directionality_score = row.get('directionality_score', 0.0)
    context_awareness_score = row.get('context_awareness_score', 0.0)

    # Discrimination score: discrimination rate when significant, 0 otherwise (per methodology)
    is_disc_sig = row.get('discrimination_significant', False)
    discrimination_rate = row.get('discrimination_rate', 0.0)
    discrimination_score = discrimination_rate if is_disc_sig else 0.0

    # Combined CS score (weighted: 35% dir, 35% ctx, 30% disc)
    combined_cs_score = (0.35 * directionality_score +
                        0.35 * context_awareness_score +
                        0.30 * discrimination_score)

    all_results_official[i]['discrimination_score'] = discrimination_score
    all_results_official[i]['combined_cs_score'] = combined_cs_score

# Add discrimination and combined scores to CNN partial results
if all_results_partial:
    for i, row in enumerate(all_results_partial):
        # Add discrimination data
        disc_row = discrimination_df[(discrimination_df['model'] == 'CNN') &
                                     (discrimination_df['antibiotic_class'] == row['antibiotic_class'])]
        if len(disc_row) > 0:
            disc_row = disc_row.iloc[0]
            try:
                result = binomtest(disc_row['n_discriminative'], disc_row['n_pairs'], p=0.5, alternative='greater')
                p_value_binom = result.pvalue
            except:
                p_value_binom = binomtest(disc_row['n_discriminative'], disc_row['n_pairs'], p=0.5, alternative='greater')

            all_results_partial[i]['discrimination_rate'] = disc_row['discrimination_rate']
            all_results_partial[i]['p_value_discrimination'] = p_value_binom
            all_results_partial[i]['discrimination_significant'] = p_value_binom < 0.05

        # Calculate combined CS score for partial results (proportions, not binary)
        directionality_score = row.get('directionality_score', 0.0)
        context_awareness_score = row.get('context_awareness_score', 0.0)

        # Discrimination score: rate when significant, 0 otherwise
        is_disc_sig = all_results_partial[i].get('discrimination_significant', False)
        discrimination_rate = all_results_partial[i].get('discrimination_rate', 0.0)
        discrimination_score = discrimination_rate if is_disc_sig else 0.0

        combined_cs_score = (0.35 * directionality_score +
                            0.35 * context_awareness_score +
                            0.30 * discrimination_score)

        all_results_partial[i]['discrimination_score'] = discrimination_score
        all_results_partial[i]['combined_cs_score_partial'] = combined_cs_score

# Save results
results_official_df = pd.DataFrame(all_results_official)
results_official_df.to_csv(os.path.join(OUTPUT_DIR, "cs_official_SR_aligned.csv"), index=False)

if all_results_partial:
    results_partial_df = pd.DataFrame(all_results_partial)
    results_partial_df.to_csv(os.path.join(OUTPUT_DIR, "cs_partial_CNN_comparison.csv"), index=False)

# Display
print("\n" + "=" * 100)
print("OFFICIAL RESULTS (SR-ALIGNED)")
print("=" * 100)

print("\n| Model | Class | Detection | Mean Delta | Responsiveness | Dir | Ctx | Disc |")
print("|-------|-------|-----------|------------|----------------|-----|-----|------|")
for _, row in results_official_df.iterrows():
    dir_sig = 'Y' if row.get('directionality_significant', False) else 'N'
    ctx_sig = 'Y' if row.get('context_awareness_significant', False) else 'N'
    disc_sig = 'Y' if row.get('discrimination_significant', False) else 'N'
    print(f"| {row['model']:5s} | {row['antibiotic_class']:15s} | {row['scaffold_detection']:20s} | "
          f"{row.get('mean_delta', 0):.4f} | {row.get('context_responsiveness', 0):14.3f} | "
          f"{dir_sig:^3s} | {ctx_sig:^3s} | {disc_sig:^4s} |")

if all_results_partial:
    print("\n" + "=" * 100)
    print("CNN PARTIAL RESULTS (Relaxed SMARTS - For Comparison Only)")
    print("=" * 100)
    print("\n| Class | Mean Delta | Ctx.Aware | Dir | Disc | Combined (Partial) |")
    print("|-------|------------|-----------|-----|------|--------------------|")
    for _, row in results_partial_df.iterrows():
        ctx = row.get('context_awareness_score', 0)
        dir_s = row.get('directionality_score', 0)
        disc = row.get('discrimination_score', 0)
        comb = row.get('combined_cs_score_partial', 0)
        print(f"| {row['antibiotic_class']:15s} | {row['mean_delta']:10.4f} | {ctx:9.3f} | "
              f"{dir_s:3.0f} | {disc:4.0f} | {comb:18.3f} |")

    avg_partial = results_partial_df['combined_cs_score_partial'].mean()
    print(f"| AVG (Partial)   |            | {results_partial_df['context_awareness_score'].mean():9.3f} | "
          f"    |      | {avg_partial:18.3f} |")
    print("\nNote: Partial scores based on relaxed SMARTS (fragments only).")
    print("      Official CNN CS score remains 0.000 (FAILED on strict SMARTS).")

print(f"\n[SUCCESS] Official results: cs_official_SR_aligned.csv")
if all_results_partial:
    print(f"[SUCCESS] CNN partial: cs_partial_CNN_comparison.csv")

print("\n" + "=" * 100)
print("CONTEXT SENSITIVITY EVALUATION COMPLETE")
print("=" * 100)
