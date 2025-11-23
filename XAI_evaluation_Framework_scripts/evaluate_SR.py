"""
MODULAR EVALUATION: SCAFFOLD RECOGNITION (SR) - PHARMACOPHORE-BASED

Evaluates XAI scaffold recognition using proper pharmacophore definitions.
Scaffolds can be split across multiple substructures - all components must be
recognized for complete scaffold coverage.

APPROACH:
- RGCN/CNN: Component-based SMARTS matching
  1. Top-K Recognition: Are ALL scaffold components in top-K?
  2. Attribution >0.1: Do ALL components have |attribution| >0.1?
  3. Complete: BOTH metrics satisfied

- RF: Feature-based scaffold mapping
  1. Top-K Recognition: Are required fr_* features in top-K?
  2. NO magnitude threshold (RF attributions 0.001-0.03)
  3. Complete: Top-K satisfied

SCAFFOLD COMPONENTS (from pharmacophores.json):
- Fluoroquinolones: 2 components (Bicyclic quinolone core + Carboxylic acid)
- Beta-lactam: 1 component (Beta-lactam ring)
- Oxazolidinone: 1 component (Oxazolidinone ring)

Output Directory: xai_eval_output/modular_evaluation/sr/
"""

import pandas as pd
import json
import os
import numpy as np
from pathlib import Path
from rdkit import Chem

# Configuration
BASE_DIR = r"C:\Users\uqaonawo\OneDrive - The University of Queensland\Desktop\automatic_XAI\notebooks\comparative_output"
EVAL_OUTPUT_DIR = os.path.join(BASE_DIR, "xai_eval_output")
OUTPUT_DIR = os.path.join(EVAL_OUTPUT_DIR, "modular_evaluation", "sr")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 100)
print("MODULAR EVALUATION: SCAFFOLD RECOGNITION (Pharmacophore-Based)")
print("=" * 100)
print(f"\nOutput directory: {OUTPUT_DIR}")

# Define scaffold components with SMARTS patterns (from pharmacophores.json)
SCAFFOLD_COMPONENTS = {
    'Fluoroquinolones': {
        'components': [
            {
                'name': 'Bicyclic Quinolone Core',
                'smarts': 'c1ccc2ncccc2c1',  # quinoline bicyclic from loose_required_any
                'smarts_relaxed': ['c1ccncc1', 'n1ccccc1', 'c1cnccc1'],  # CNN: any 6-membered aromatic N ring
                'rf_features': ['fr_pyridine', 'fr_bicyclic', 'fr_Ar_N'],
                'required': True
            },
            {
                'name': 'Carboxylic Acid',
                'smarts': 'C(=O)O',  # from important_any
                'smarts_relaxed': ['C(=O)O', 'C(O)=O'],  # CNN: carboxylic acid
                'rf_features': ['fr_COO', 'fr_COO2', 'fr_Ar_COO'],
                'required': True
            }
        ],
        'scaffold_name': 'Quinolone Core'
    },
    'Beta-lactam': {
        'components': [
            {
                'name': 'Beta-lactam Ring',
                'smarts': 'N1C(=O)CC1',  # from required_any
                'smarts_relaxed': ['NC(=O)C', 'C(=O)N', 'N1CCC1', 'N1CC1'],  # CNN: any small lactam/amide
                'rf_features': ['fr_lactam', 'fr_amide'],
                'required': True
            }
        ],
        'scaffold_name': 'Beta-lactam Ring'
    },
    'Oxazolidinone': {
        'components': [
            {
                'name': 'Oxazolidinone Ring',
                'smarts': 'O1C(=O)NCC1',  # from required_any
                'smarts_relaxed': ['O1CCNC1', 'OC(=O)N', 'O1CCC1', 'NC(=O)O'],  # CNN: any oxazolidinone-like
                'rf_features': ['fr_oxazole', 'fr_lactone', 'fr_urea'],
                'required': True
            }
        ],
        'scaffold_name': 'Oxazolidinone Ring'
    }
}

def check_smarts_in_substructure(substructure_smiles, smarts_pattern):
    """
    Check if a substructure contains a SMARTS pattern using RDKit.

    Args:
        substructure_smiles: SMILES string of the substructure
        smarts_pattern: SMARTS pattern to match

    Returns:
        Boolean indicating if pattern matches
    """
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

def check_rf_features_present(feature_dict, rf_feature_list, pos_features):
    """
    Check if required RF features are present in the molecule.

    Args:
        feature_dict: Attribution dictionary
        rf_feature_list: List of RF features to check (OR logic)
        pos_features: List of present features

    Returns:
        Boolean indicating if ANY of the required features is present
    """
    if isinstance(pos_features, str):
        pos_features = json.loads(pos_features)

    # Check if ANY of the required features is present in the molecule
    for feature in rf_feature_list:
        if feature in pos_features:
            return True
    return False

def get_top_k_features(attrs, k=7, pos_features=None):
    """
    Get top-K features for RF attributions (filtered to present features).

    Args:
        attrs: Attribution dictionary
        k: Number of top features to return
        pos_features: List of present features

    Returns:
        List of top-K feature names
    """
    if isinstance(attrs, dict):
        if pos_features is not None:
            if isinstance(pos_features, str):
                pos_features = json.loads(pos_features)
            # Only rank present features
            filtered_attrs = {feat: val for feat, val in attrs.items() if feat in pos_features}
            sorted_features = sorted(filtered_attrs.items(), key=lambda x: abs(x[1]), reverse=True)
        else:
            sorted_features = sorted(attrs.items(), key=lambda x: abs(x[1]), reverse=True)
        return [feat for feat, val in sorted_features[:k]]
    return []

def evaluate_scaffold_components_rgcn_cnn(attrs, components, topk=3, use_relaxed=False):
    """
    Evaluate scaffold recognition for RGCN/CNN using SMARTS patterns.
    Checks if ALL required components are covered by top-K substructures.

    Args:
        attrs: List of substructure attributions
        components: List of scaffold component definitions
        topk: Number of top substructures to consider
        use_relaxed: If True, use relaxed SMARTS patterns (for CNN partial matching)

    Returns:
        dict with component_coverage, topk_recognized, magnitude_recognized, complete
    """
    if not isinstance(attrs, list) or len(attrs) == 0:
        return {
            'component_coverage': {},
            'topk_recognized': False,
            'magnitude_recognized': False,
            'complete': False
        }

    top_k_attrs = attrs[:topk]
    component_coverage = {}

    # Check each required component
    for component in components:
        if not component.get('required', True):
            continue

        comp_name = component['name']

        # Choose patterns based on mode
        if use_relaxed and 'smarts_relaxed' in component:
            smarts_list = component['smarts_relaxed']  # List of relaxed patterns
        else:
            smarts_list = [component['smarts']]  # Single strict pattern

        # Find if this component appears in top-K
        found_in_topk = False
        max_attribution = 0
        has_high_attribution = False

        for attr in top_k_attrs:
            substructure = attr.get('substructure', '')
            attribution = attr.get('attribution', 0)

            # Check if ANY of the SMARTS patterns match
            for smarts in smarts_list:
                if check_smarts_in_substructure(substructure, smarts):
                    found_in_topk = True
                    if abs(attribution) > abs(max_attribution):
                        max_attribution = attribution
                    if abs(attribution) > 0.1:
                        has_high_attribution = True
                    break  # Found a match, no need to check other patterns

        component_coverage[comp_name] = {
            'in_topk': found_in_topk,
            'has_high_attr': has_high_attribution,
            'max_attr': max_attribution
        }

    # Scaffold is recognized if ALL required components are found
    all_in_topk = all(cov['in_topk'] for cov in component_coverage.values())
    all_high_attr = all(cov['has_high_attr'] for cov in component_coverage.values())

    return {
        'component_coverage': component_coverage,
        'topk_recognized': all_in_topk,
        'magnitude_recognized': all_high_attr,
        'complete': all_in_topk and all_high_attr
    }

def evaluate_scaffold_components_rf(attrs, components, pos_features, topk=7):
    """
    Evaluate scaffold recognition for RF using feature mapping.
    Checks if required fr_* features are present and in top-K.

    Args:
        attrs: Attribution dictionary
        components: List of scaffold component definitions
        pos_features: List of present features
        topk: Number of top features to consider

    Returns:
        dict with component_coverage, topk_recognized, complete
    """
    if not isinstance(attrs, dict):
        return {
            'component_coverage': {},
            'topk_recognized': False,
            'complete': False
        }

    if isinstance(pos_features, str):
        pos_features = json.loads(pos_features)

    top_k_features = get_top_k_features(attrs, k=topk, pos_features=pos_features)
    component_coverage = {}

    # Check each required component
    for component in components:
        if not component.get('required', True):
            continue

        comp_name = component['name']
        rf_features = component['rf_features']

        # Check if ANY of the component's features is present in molecule
        feature_present = False
        feature_in_topk = False

        for feature in rf_features:
            if feature in pos_features:
                feature_present = True
                if feature in top_k_features:
                    feature_in_topk = True
                    break

        component_coverage[comp_name] = {
            'feature_present': feature_present,
            'in_topk': feature_in_topk
        }

    # Scaffold is recognized if ALL required components have features in top-K
    all_present = all(cov['feature_present'] for cov in component_coverage.values())
    all_in_topk = all(cov['in_topk'] for cov in component_coverage.values() if cov['feature_present'])

    return {
        'component_coverage': component_coverage,
        'topk_recognized': all_in_topk if all_present else False,
        'complete': all_in_topk if all_present else False
    }

def load_and_extract_ground_truth_actives():
    """Load ensemble cliff parquets and extract ground truth active compound IDs (100 per class)."""

    print("\n" + "=" * 100)
    print("STEP 1: EXTRACTING GROUND TRUTH ACTIVE COMPOUNDS (100 PER CLASS)")
    print("=" * 100)

    # Load all ensemble cliff files
    model_files = {
        'RGCN': os.path.join(EVAL_OUTPUT_DIR, 'RGCN_output', 'rgcn_ensemble_cliffs.parquet'),
        'CNN': os.path.join(EVAL_OUTPUT_DIR, 'CNN_output', 'cnn_ensemble_cliffs.parquet'),
        'RF': os.path.join(EVAL_OUTPUT_DIR, 'RF_output', 'rf_ensemble_cliffs.parquet')
    }

    # Extract ALL active compound IDs from first model
    first_model_file = model_files['RGCN']
    df = pd.read_parquet(first_model_file)

    ground_truth_actives = {}
    for ab_class in ['Fluoroquinolones', 'Beta-lactam', 'Oxazolidinone']:
        class_df = df[df['antibiotic_class'] == ab_class]
        active_ids = class_df['compound_active_id'].tolist()
        ground_truth_actives[ab_class] = active_ids

        scaffold_name = SCAFFOLD_COMPONENTS[ab_class]['scaffold_name']
        n_components = len(SCAFFOLD_COMPONENTS[ab_class]['components'])
        print(f"\n{ab_class}:")
        print(f"  Scaffold: {scaffold_name} ({n_components} component(s))")
        print(f"  Ground truth actives: {len(active_ids)} (includes duplicates from pairs)")

    total_actives = sum(len(ids) for ids in ground_truth_actives.values())
    print(f"\nTotal ground truth active evaluations: {total_actives}")

    return model_files, ground_truth_actives

def evaluate_model_scaffold_recognition(model_name, model_file, ground_truth_actives):
    """Evaluate scaffold recognition for a model on ground truth actives."""

    print(f"\n{'=' * 100}")
    print(f"{model_name.upper()} - SCAFFOLD RECOGNITION EVALUATION")
    print('=' * 100)

    df = pd.read_parquet(model_file)
    is_rf = 'RF' in model_name

    if is_rf:
        print(f"[RF INFO] Using TOP-K ONLY (no magnitude threshold)")
        print(f"[RF INFO] RF attributions too small for >0.1 threshold")
        has_pos_features = 'pos_features_active' in df.columns
        if has_pos_features:
            print(f"[RF INFO] Using 'pos_features_active' to filter to PRESENT features")

    results = []

    for ab_class in ['Fluoroquinolones', 'Beta-lactam', 'Oxazolidinone']:
        active_ids = ground_truth_actives[ab_class]
        class_df = df[df['antibiotic_class'] == ab_class]
        components = SCAFFOLD_COMPONENTS[ab_class]['components']
        scaffold_name = SCAFFOLD_COMPONENTS[ab_class]['scaffold_name']

        print(f"\n{'-' * 100}")
        print(f"{ab_class} - {scaffold_name}")
        print(f"Components required: {', '.join([c['name'] for c in components if c.get('required', True)])}")
        print('-' * 100)

        # Collect data for each active compound
        active_data = []
        for compound_id in active_ids:
            compound_rows = class_df[class_df['compound_active_id'] == compound_id]
            if len(compound_rows) == 0:
                continue

            row = compound_rows.iloc[0]
            pred_active = row.get('compound_active_pred_class', 'inactive') == 'active'

            # Get attributions
            if is_rf:
                attrs = row.get('feature_attr_active', '{}')  # RF uses feature_attr_active
                pos_features = row.get('pos_features_active', '[]')
            else:
                attrs = row.get('substruct_attr_active', '[]')  # RGCN/CNN use substruct_attr_active
                pos_features = None

            # Parse attributions
            if isinstance(attrs, str):
                attrs = json.loads(attrs)

            active_data.append({
                'compound_id': compound_id,
                'pred_active': pred_active,
                'attrs': attrs,
                'pos_features': pos_features
            })

        # Evaluate scaffold recognition
        is_cnn = 'CNN' in model_name
        topk_recognized_list = []
        magnitude_recognized_list = [] if not is_rf else None
        complete_list = []

        # For CNN: also track relaxed (partial) recognition
        if is_cnn:
            topk_recognized_relaxed_list = []
            magnitude_recognized_relaxed_list = []
            complete_relaxed_list = []

        for compound in active_data:
            if is_rf:
                result = evaluate_scaffold_components_rf(
                    compound['attrs'],
                    components,
                    compound['pos_features'],
                    topk=7
                )
                topk_recognized_list.append(result['topk_recognized'])
                complete_list.append(result['complete'])
            else:
                # Strict evaluation (complete scaffold)
                result_strict = evaluate_scaffold_components_rgcn_cnn(
                    compound['attrs'],
                    components,
                    topk=3,
                    use_relaxed=False
                )
                topk_recognized_list.append(result_strict['topk_recognized'])
                magnitude_recognized_list.append(result_strict['magnitude_recognized'])
                complete_list.append(result_strict['complete'])

                # For CNN: also do relaxed evaluation (partial scaffold)
                if is_cnn:
                    result_relaxed = evaluate_scaffold_components_rgcn_cnn(
                        compound['attrs'],
                        components,
                        topk=3,
                        use_relaxed=True
                    )
                    topk_recognized_relaxed_list.append(result_relaxed['topk_recognized'])
                    magnitude_recognized_relaxed_list.append(result_relaxed['magnitude_recognized'])
                    complete_relaxed_list.append(result_relaxed['complete'])

        # Calculate rates
        n_total = len(active_data)
        n_pred_active = sum(1 for c in active_data if c['pred_active'])
        topk_rate = sum(topk_recognized_list) / n_total if n_total > 0 else 0
        complete_rate = sum(complete_list) / n_total if n_total > 0 else 0

        if is_rf:
            magnitude_rate = None
            topk_rate_relaxed = None
            magnitude_rate_relaxed = None
            complete_rate_relaxed = None
            print(f"  Total actives evaluated: {n_total}")
            print(f"  Predicted active: {n_pred_active}")
            print(f"  Top-K Recognition ({scaffold_name}): {topk_rate:.1%}")
            print(f"  Complete (Top-K only): {complete_rate:.1%}")
        elif is_cnn:
            # CNN: Show both strict (in parentheses) and relaxed (main)
            magnitude_rate = sum(magnitude_recognized_list) / n_total if n_total > 0 else 0
            topk_rate_relaxed = sum(topk_recognized_relaxed_list) / n_total if n_total > 0 else 0
            magnitude_rate_relaxed = sum(magnitude_recognized_relaxed_list) / n_total if n_total > 0 else 0
            complete_rate_relaxed = sum(complete_relaxed_list) / n_total if n_total > 0 else 0
            print(f"  Total actives evaluated: {n_total}")
            print(f"  Predicted active: {n_pred_active}")
            print(f"  Top-K Recognition - Partial: {topk_rate_relaxed:.1%} (Complete: {topk_rate:.1%})")
            print(f"  Attribution >0.1 - Partial: {magnitude_rate_relaxed:.1%} (Complete: {magnitude_rate:.1%})")
            print(f"  Complete Recognition - Partial: {complete_rate_relaxed:.1%} (Complete: {complete_rate:.1%})")
        else:
            # RGCN: strict only
            magnitude_rate = sum(magnitude_recognized_list) / n_total if n_total > 0 else 0
            topk_rate_relaxed = None
            magnitude_rate_relaxed = None
            complete_rate_relaxed = None
            print(f"  Total actives evaluated: {n_total}")
            print(f"  Predicted active: {n_pred_active}")
            print(f"  Metric 1 - Top-K Recognition ({scaffold_name}): {topk_rate:.1%}")
            print(f"  Metric 2 - Attribution >0.1 ({scaffold_name}): {magnitude_rate:.1%}")
            print(f"  Complete (Both metrics): {complete_rate:.1%}")

        results.append({
            'model': model_name,
            'antibiotic_class': ab_class,
            'scaffold_name': scaffold_name,
            'n_total': n_total,
            'n_evaluated': n_total,
            'n_pred_active': n_pred_active,
            'pred_accuracy': n_pred_active / n_total if n_total > 0 else 0,
            'topk_recognition_rate': topk_rate,
            'magnitude_recognition_rate': magnitude_rate,
            'complete_rate': complete_rate,
            'topk_recognition_rate_relaxed': topk_rate_relaxed,
            'magnitude_recognition_rate_relaxed': magnitude_rate_relaxed,
            'complete_rate_relaxed': complete_rate_relaxed
        })

    return pd.DataFrame(results)

def main():
    """Main evaluation function."""

    # Step 1: Load ground truth actives
    model_files, ground_truth_actives = load_and_extract_ground_truth_actives()

    # Step 2: Evaluate each model
    print("\n" + "=" * 100)
    print("STEP 2: EVALUATING SCAFFOLD RECOGNITION FOR EACH MODEL")
    print("=" * 100)

    all_results = []

    for model_name, model_file in model_files.items():
        results_df = evaluate_model_scaffold_recognition(model_name, model_file, ground_truth_actives)
        all_results.append(results_df)

    # Combine results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Step 3: Save detailed results
    print("\n" + "=" * 100)
    print("STEP 3: SAVING RESULTS")
    print("=" * 100)

    detailed_output = os.path.join(OUTPUT_DIR, 'sr_detailed_results.csv')
    combined_results.to_csv(detailed_output, index=False)
    print(f"\nDetailed results saved to: {detailed_output}")

    # Step 4: Create manuscript table
    manuscript_df = combined_results.copy()

    # Format with CNN dual metrics
    def format_metric(row, col_name, col_relaxed_name):
        """Format metric with CNN showing relaxed (strict) format."""
        val = row[col_name]
        val_relaxed = row[col_relaxed_name]

        is_cnn = 'CNN' in row['model']
        is_rf = 'RF' in row['model']

        if is_rf and pd.isna(val):
            return "N/A"
        elif is_cnn and val_relaxed is not None and not pd.isna(val_relaxed):
            # CNN: show relaxed (strict)
            return f"{val_relaxed:.1%} ({val:.1%})"
        elif val is not None and not pd.isna(val):
            return f"{val:.1%}"
        else:
            return "N/A"

    # Apply formatting
    manuscript_df['Top-K Recognition'] = manuscript_df.apply(
        lambda row: format_metric(row, 'topk_recognition_rate', 'topk_recognition_rate_relaxed'),
        axis=1
    )
    manuscript_df['Attribution >0.1'] = manuscript_df.apply(
        lambda row: format_metric(row, 'magnitude_recognition_rate', 'magnitude_recognition_rate_relaxed'),
        axis=1
    )
    manuscript_df['Complete Recognition'] = manuscript_df.apply(
        lambda row: format_metric(row, 'complete_rate', 'complete_rate_relaxed'),
        axis=1
    )

    # Select and rename columns
    manuscript_df = manuscript_df[['model', 'antibiotic_class', 'scaffold_name', 'n_total',
                                    'n_pred_active', 'Top-K Recognition',
                                    'Attribution >0.1', 'Complete Recognition']].copy()

    manuscript_df.columns = [
        'model', 'antibiotic_class', 'Scaffold Component', 'N (Ground Truth)',
        'N (Predicted Active)', 'Top-K Recognition', 'Attribution >0.1 (RGCN/CNN only)',
        'Complete Recognition'
    ]

    manuscript_output = os.path.join(OUTPUT_DIR, 'sr_manuscript_table.csv')
    manuscript_df.to_csv(manuscript_output, index=False)
    print(f"Manuscript table saved to: {manuscript_output}")

    # Step 5: Display summary
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)

    print("\nComplete Recognition Rates by Model and Class:")
    for _, row in manuscript_df.iterrows():
        model = row['model']
        ab_class = row['antibiotic_class']
        scaffold = row['Scaffold Component']
        topk = row['Top-K Recognition']
        mag = row['Attribution >0.1 (RGCN/CNN only)']
        complete = row['Complete Recognition']
        print(f"{model:6} | {ab_class:18} | {scaffold:25} | Top-K: {topk:6} | Magnitude: {mag:6} | Complete: {complete:6}")

    print("\n" + "=" * 100)
    print("SCAFFOLD RECOGNITION EVALUATION COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()
