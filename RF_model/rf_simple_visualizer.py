"""
Simple RF Visualizer - Uses exact implementation from RF_XAI_activity_pairs.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from typing import Dict, Any
import io
from PIL import Image

# Import from the existing RF_XAI_activity_pairs.py script
try:
    from RF_XAI_activity_pairs import (
        get_ensemble,
        shap_explain_single_for_model,
        prepare_visualization_data_rf,
        FG_SMARTS
    )
    print("Successfully imported from RF_XAI_activity_pairs.py")
except ImportError as e:
    print(f"Error importing from RF_XAI_activity_pairs.py: {e}")
    print("\nMake sure RF_XAI_activity_pairs.py is in the same directory!")
    raise


def _map_shap_to_atoms_simple(smiles: str, shap_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Map SHAP functional group scores to atom-level attributions.
    Uses EXACT implementation from RF_XAI_activity_pairs.py:prepare_visualization_data_rf

    Colors: blue (positive), orange (negative), neutral: no color (None).
    Uses normalization and threshold (0.05) to match reference implementation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'error': 'invalid_smiles'}

    n_atoms = mol.GetNumAtoms()
    atom_scores = np.zeros(n_atoms, dtype=float)

    # Map SHAP values to atoms using FG_SMARTS (same as reference)
    for fg, val in shap_dict.items():
        if not val:  # Skip zero values
            continue

        smt = FG_SMARTS.get(fg)
        if not smt:
            continue

        try:
            patt = Chem.MolFromSmarts(smt)
            if patt is None:
                continue

            matches = mol.GetSubstructMatches(patt)
            if not matches:
                continue

            # Get unique atoms from all matches
            atoms = sorted({a for m in matches for a in m})
            if atoms:
                # Distribute SHAP value equally across atoms
                w = val / float(len(atoms))
                for a in atoms:
                    atom_scores[a] += w
        except:
            continue

    # Robust normalization using p95 percentile (like CNN/RGCN in xai_viz_helper.py)
    # This is MORE ROBUST than max normalization!
    abs_vals = np.abs(atom_scores)
    max_abs = float(abs_vals.max()) if abs_vals.size else 1e-9
    p95 = float(np.percentile(abs_vals, 95)) if abs_vals.size else 0.0
    denom = max(1e-9, p95 if p95 > 1e-12 else max_abs)

    # Color-blind friendly palette (matching CNN/RGCN style)
    BLUE = (31/255.0, 119/255.0, 180/255.0)    # matplotlib default blue
    ORANGE = (1.0, 127/255.0, 14/255.0)        # matplotlib default orange

    colors = []
    pos_idx = []
    neg_idx = []
    neu_idx = []

    MIN_INTEN = 0.25  # 🔸 minimum color intensity for any non-zero SHAP

    for i, s in enumerate(atom_scores):
        # treat truly zero as neutral, everything else gets some color
        if abs(s) < 1e-12:
            colors.append(None)
            neu_idx.append(i)
            continue

        # Normalize and apply gamma correction (boosts mid-range intensities)
        inten = min(abs(s) / denom, 1.0)
        inten = pow(inten, 0.6)      # gamma < 1 boosts mid-range
        inten = max(inten, MIN_INTEN)  # 🔸 enforce minimum visibility

        base = BLUE if s > 0 else ORANGE
        c = (1.0*(1-inten) + base[0]*inten,
             1.0*(1-inten) + base[1]*inten,
             1.0*(1-inten) + base[2]*inten)
        colors.append(c)

        if s > 0:
            pos_idx.append(i)
        else:
            neg_idx.append(i)

    # Compute normalized scores for output
    norm = atom_scores / denom

    return {
        'atom_attributions': atom_scores.tolist(),
        'atom_attributions_normalized': norm.tolist(),
        'atom_colors': colors,
        'positive_atoms': pos_idx,
        'negative_atoms': neg_idx,
        'neutral_atoms': neu_idx,
        'n_atoms': n_atoms
    }


def visualize_rf_xai(smiles: str, top_k: int = 10, figsize=(14, 8)):
    """
    Visualize RF TreeSHAP explanation for a single SMILES.
    Uses the EXACT implementation from RF_XAI_activity_pairs.py.

    Args:
        smiles: SMILES string
        top_k: Number of top features to show
        figsize: Figure size

    Returns:
        matplotlib figure
    """
    # Get the ensemble
    ensemble = get_ensemble()

    # Calculate descriptors
    try:
        descriptors = ensemble.calculate_descriptors(smiles)
    except Exception as e:
        print(f"Error calculating descriptors: {e}")
        return None

    # Get prediction (ensemble average)
    X_raw = np.array([descriptors[name] for name in ensemble.feature_names])
    pred_prob = ensemble.ensemble_predict_proba(X_raw)
    threshold = 0.5  # Use 0.5 threshold
    prediction = 'Active' if pred_prob >= threshold else 'Inactive'

    # Print prediction clearly
    print("="*70)
    print(f"PREDICTION: {prediction}")
    print(f"Probability: {pred_prob:.4f} (threshold: {threshold})")
    print(f"Ensemble models: {len(ensemble.models)}")
    print("="*70 + "\n")

    # Get SHAP explanations from ALL ensemble models and average
    print(f"Computing TreeSHAP across {len(ensemble.models)} ensemble models...")
    all_shap_dicts = []

    for i, (model, scaler) in enumerate(zip(ensemble.models, ensemble.scalers)):
        shap_dict = shap_explain_single_for_model(
            smiles, model, scaler,
            ensemble.training_data,
            ensemble.feature_names
        )
        all_shap_dicts.append(shap_dict)

    # Average SHAP values across ensemble
    averaged_shap = {}
    for feature in ensemble.feature_names:
        values = [d.get(feature, 0.0) for d in all_shap_dicts]
        averaged_shap[feature] = float(np.mean(values))

    # ROBUST FILTERING: Keep features with non-zero SHAP values
    # (Don't rely on descriptor values - RDKit's descriptors may use different SMARTS!)
    filtered_shap = {k: v for k, v in averaged_shap.items() if abs(v) > 1e-9}
    print(f"Features with non-zero SHAP values: {len(filtered_shap)}/{len(ensemble.feature_names)}")

    # Count how many have matching descriptors (for info only)
    present_features = {k: v for k, v in descriptors.items() if v > 0}
    matching = len([k for k in filtered_shap if k in present_features])
    print(f"  - Detected by RDKit descriptors: {matching}")
    print(f"  - SHAP-only (may use different patterns): {len(filtered_shap) - matching}")

    # Map to atom attributions with SIMPLE logic (only present features)
    viz_data = _map_shap_to_atoms_simple(smiles, filtered_shap)

    if 'error' in viz_data:
        print(f"Error: {viz_data['error']}")
        return None

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1.2])

    # 1. Molecule with atom attributions
    ax_mol = fig.add_subplot(gs[0, 0])
    _draw_molecule_with_colors(ax_mol, smiles, viz_data)

    # 2. Prediction summary
    ax_pred = fig.add_subplot(gs[0, 1])
    _draw_prediction_box(ax_pred, smiles, pred_prob, prediction, len(ensemble.models))

    # 3. Feature importance (only present features)
    ax_feat = fig.add_subplot(gs[1, :])
    _draw_feature_bars(ax_feat, filtered_shap, top_k)

    plt.tight_layout()
    return fig


def _draw_molecule_with_colors(ax, smiles: str, viz_data: Dict[str, Any]):
    """Draw molecule with colored atom halos."""
    from IPython.display import SVG, display
    import base64

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        ax.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center')
        ax.axis('off')
        return

    # Extract data
    atom_colors_list = viz_data['atom_colors']
    pos_idx = viz_data['positive_atoms']
    neg_idx = viz_data['negative_atoms']
    atom_scores = np.array(viz_data['atom_attributions'])

    # Convert to dict format needed by RDKit
    atom_cols = {}
    atom_rads = {}
    bond_cols = {}

    # Scale halos based on attribution magnitude
    max_abs_score = np.max(np.abs(atom_scores)) if len(atom_scores) > 0 else 1e-9
    max_abs_score = max(max_abs_score, 1e-9)  # Avoid division by zero

    MIN_RAD = 0.3   # existing minimum
    EXTRA_MIN = 0.2 # 🔸 extra floor for tiny scores

    for i, color in enumerate(atom_colors_list):
        if color is not None:
            atom_cols[i] = color
            normalized_magnitude = abs(atom_scores[i]) / max_abs_score
            # 🔸 enforce a minimum halo size for any colored atom
            normalized_magnitude = max(normalized_magnitude, EXTRA_MIN)
            atom_rads[i] = MIN_RAD + 0.5 * normalized_magnitude

    # Color bonds
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in atom_cols or end_idx in atom_cols:
            bond_cols[bond.GetIdx()] = atom_cols.get(begin_idx, atom_cols.get(end_idx))

    # Draw with RDKit
    drawer = rdMolDraw2D.MolDraw2DSVG(600, 400)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.fillHighlights = True
    opts.highlightBondWidthMult = 25

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(atom_cols.keys()),
        highlightBonds=list(bond_cols.keys()),
        highlightAtomColors=atom_cols,
        highlightBondColors=bond_cols,
        highlightAtomRadii=atom_rads
    )
    drawer.FinishDrawing()
    svg_str = drawer.GetDrawingText()

    # Display SVG directly in Jupyter (this always works!)
    print("\n" + "="*70)
    print("Molecule Visualization (Atom Attributions - TreeSHAP)")
    print("="*70)
    display(SVG(svg_str))
    print("🔵 Blue = Supports activity | 🟠 Orange = Opposes activity")
    print("="*70 + "\n")

    # Also try to show in matplotlib axis
    try:
        from cairosvg import svg2png
        png_bytes = svg2png(bytestring=svg_str.encode('utf-8'), dpi=150)
        img = Image.open(io.BytesIO(png_bytes))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Atom Attributions (TreeSHAP)', fontsize=12, fontweight='bold')
    except:
        # Fallback: show text in axis
        ax.text(0.5, 0.5, 'See molecule above\n(SVG display)',
               ha='center', va='center', fontsize=12)
        ax.axis('off')
        ax.set_title('Atom Attributions (TreeSHAP)', fontsize=12, fontweight='bold')


def _draw_prediction_box(ax, smiles, prob, prediction, n_models):
    """Draw prediction summary."""
    ax.axis('off')

    summary = f"""
    SMILES:
    {smiles[:60]}{'...' if len(smiles) > 60 else ''}

    Prediction: {prediction}
    Probability: {prob:.3f}
    Confidence: {abs(prob - 0.5) * 2:.3f}

    Model: Random Forest Ensemble
    Models averaged: {n_models}
    Method: TreeSHAP (averaged)
    """

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Probability bar
    bar_y, bar_h, bar_w = 0.3, 0.1, 0.9
    ax.add_patch(plt.Rectangle((0.05, bar_y), bar_w, bar_h,
                               facecolor='lightgray', edgecolor='black',
                               transform=ax.transAxes))

    color = 'green' if prediction == 'Active' else 'red'
    ax.add_patch(plt.Rectangle((0.05, bar_y), bar_w * prob, bar_h,
                               facecolor=color, alpha=0.6,
                               transform=ax.transAxes))

    ax.text(0.5, bar_y + bar_h + 0.02, f'Activity Probability: {prob:.1%}',
            ha='center', transform=ax.transAxes, fontsize=10, fontweight='bold')


def _draw_feature_bars(ax, shap_dict, top_k):
    """Draw top feature importance bars."""
    # Filter and sort
    non_zero = {k: v for k, v in shap_dict.items() if abs(v) > 1e-6}
    sorted_features = sorted(non_zero.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]

    if not sorted_features:
        ax.text(0.5, 0.5, 'No significant features', ha='center', va='center')
        ax.axis('off')
        return

    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    colors = ['#1f77b4' if v > 0 else '#ff7f0e' for v in values]

    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('SHAP Value (Probability contribution)', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {len(features)} Feature Attributions', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.7, label='Supports Activity'),
        Patch(facecolor='#ff7f0e', alpha=0.7, label='Opposes Activity')
    ]
    ax.legend(handles=legend_elements, loc='lower right')


if __name__ == "__main__":
    # Test
    test_smiles = "O=C(O)C1=CN(C2CC2)c2cc(N3CCNCC3)c(F)cc2C1=O"  # Fluoroquinolone
    fig = visualize_rf_xai(test_smiles)
    if fig:
        plt.show()
