"""
CNN XAI Visualization Helper
============================

Minimal visualization utilities for CNN occlusion-based XAI analysis.
Provides functions for molecular visualization with attribution highlighting.

Key Features:
- Occlusion-based attribution calculation
- Ensemble model prediction averaging
- Molecular structure visualization with attribution coloring
- Blue = positive attribution (increases activity prediction)
- Red/Orange = negative attribution (decreases activity prediction)
"""

import numpy as np
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Visualization will be limited.")


def occlusion_token_attr(model: nn.Module, x: torch.Tensor, pad_index: int = 0) -> torch.Tensor:
    """
    Compute token-level occlusion attribution for CNN.

    Args:
        model: CNN model
        x: input tensor (1, L, V) - one-hot encoded SMILES
        pad_index: vocabulary index for padding token (default 0)

    Returns:
        Tensor of shape (L, V) containing occlusion attribution scores

    Note:
        Attribution = f(original) - f(masked)
        Positive attribution means token increases prediction
        Negative attribution means token decreases prediction
    """
    model.eval()

    # Base prediction with full sequence
    with torch.no_grad():
        base_logit = model(x).item()

    L = x.shape[1]  # sequence length
    V = x.shape[2]  # vocabulary size

    # Initialize attribution tensor
    occlusion_scores = torch.zeros(L, V, device=x.device)

    # Occlude each token position
    for i in range(L):
        x_occluded = x.clone()
        # Replace token i with padding (zero out and set pad_index to 1)
        x_occluded[:, i, :] = 0.0
        x_occluded[:, i, pad_index] = 1.0

        with torch.no_grad():
            occluded_logit = model(x_occluded).item()

        # Causal contribution: how much does this token contribute to prediction?
        delta = base_logit - occluded_logit

        # Assign attribution to the original token at position i
        original_token_idx = x[:, i, :].argmax().item()
        occlusion_scores[i, original_token_idx] = delta

    return occlusion_scores


def reduce_to_positions(attr: torch.Tensor, x: torch.Tensor) -> np.ndarray:
    """
    Reduce attribution tensor to position-level scores.

    Args:
        attr: Attribution tensor (L, V)
        x: One-hot input tensor (1, L, V)

    Returns:
        Position-level attribution scores (L,)
    """
    pos_attr = (attr * x.squeeze(0)).sum(dim=-1)  # (L,)
    return pos_attr.detach().cpu().numpy()


def interpolate_to_atoms(smiles: str, seq_scores: np.ndarray) -> np.ndarray:
    """
    Map sequence-position scores (length L) to atom-level scores (length n_atoms)
    via linear interpolation. Matches approach used in GradCAM notebook.

    Args:
        smiles: SMILES string
        seq_scores: Sequence position scores (length L)

    Returns:
        Atom-level scores (length n_atoms)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Invalid SMILES in interpolate_to_atoms: {smiles}")
            return np.array([])
        n_atoms = mol.GetNumAtoms()
        if n_atoms <= 0:
            print(f"Warning: No atoms found in molecule")
            return np.array([])
        x_old = np.linspace(0.0, 1.0, num=len(seq_scores))
        x_new = np.linspace(0.0, 1.0, num=n_atoms)
        atom_scores = np.interp(x_new, x_old, seq_scores)
        return atom_scores
    except Exception as e:
        print(f"Error in interpolate_to_atoms: {e}")
        import traceback
        traceback.print_exc()
        return np.array([])


def ensemble_occlusion_attribution(models: List[nn.Module], x: torch.Tensor,
                                   pad_index: int = 0) -> np.ndarray:
    """
    Compute ensemble-averaged occlusion attributions.

    Args:
        models: List of CNN models
        x: Input tensor (1, L, V)
        pad_index: Padding token index

    Returns:
        Averaged position-level attribution scores
    """
    all_pos_scores = []

    for model in models:
        model.eval()
        attr = occlusion_token_attr(model, x, pad_index=pad_index)
        pos_scores = reduce_to_positions(attr, x)
        all_pos_scores.append(pos_scores)

    # Average across models
    avg_scores = np.mean(np.stack(all_pos_scores), axis=0)
    return avg_scores


def ensemble_predict_prob(models: List[nn.Module], x: torch.Tensor) -> float:
    """
    Compute ensemble-averaged prediction probability.

    Args:
        models: List of CNN models
        x: Input tensor (1, L, V)

    Returns:
        Average probability across ensemble
    """
    probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits.squeeze())[()].item()
            probs.append(prob)

    return float(np.mean(probs))


def draw_molecule_with_attributions(smiles: str, atom_scores: np.ndarray,
                                    pred_prob: float, threshold: float = 0.5,
                                    width: int = 800, height: int = 800) -> Optional[Any]:
    """
    Draw molecule with attribution-based highlighting.

    Args:
        smiles: SMILES string
        atom_scores: Per-atom attribution scores
        pred_prob: Prediction probability
        threshold: Classification threshold
        width: Image width
        height: Image height

    Returns:
        PIL Image or None if drawing fails
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available for visualization")
        return None

    try:
        from matplotlib.colors import LinearSegmentedColormap
        from PIL import Image
        import io

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return None

        # Prepare molecule for drawing
        AllChem.Compute2DCoords(mol)

        # Normalize scores
        if atom_scores.size and np.max(np.abs(atom_scores)) > 0:
            atom_scores = atom_scores / np.max(np.abs(atom_scores))

        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        opts = drawer.drawOptions()
        opts.addAtomIndices = True
        opts.padding = 0.1
        opts.bondLineWidth = 2.0
        opts.fixedScale = 45.0
        opts.clearBackground = True
        opts.backgroundColour = (1.0, 1.0, 1.0, 1.0)

        # Color map: Blue (positive) -> White (neutral) -> Red (negative)
        cmap = LinearSegmentedColormap.from_list(
            'attribution_cmap',
            ['blue', 'white', 'red'],
            N=256
        )

        # Create weights dictionary
        weights = {i: float(v) for i, v in enumerate(atom_scores)}

        # Draw with similarity map
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol, weights, colorMap=cmap, contourLines=10, alpha=0.7,
            draw2d=drawer, size=(width, height),
            minWeight=-1.0, maxWeight=1.0
        )

        drawer.FinishDrawing()

        # Convert to PIL Image
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))

        return img

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def print_attribution_statistics(atom_scores: np.ndarray, smiles: str,
                                 pred_prob: float, threshold: float = 0.5):
    """
    Print detailed attribution statistics.

    Args:
        atom_scores: Per-atom attribution scores
        smiles: SMILES string
        pred_prob: Prediction probability
        threshold: Classification threshold
    """
    print(f"\n{'='*70}")
    print(f"XAI ATTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    print(f"SMILES: {smiles}")
    print(f"Prediction Probability: {pred_prob:.4f}")
    print(f"Predicted Class: {'Active' if pred_prob >= threshold else 'Inactive'}")
    print(f"Confidence: {max(pred_prob, 1-pred_prob):.4f}")
    print(f"\n{'='*70}")
    print(f"ATTRIBUTION STATISTICS:")
    print(f"{'='*70}")
    print(f"  Number of atoms: {len(atom_scores)}")
    print(f"  Min attribution: {np.min(atom_scores):.4f}")
    print(f"  Max attribution: {np.max(atom_scores):.4f}")
    print(f"  Mean attribution: {np.mean(atom_scores):.4f}")
    print(f"  Std attribution: {np.std(atom_scores):.4f}")

    pos_attrs = atom_scores[atom_scores > 0.1]
    neg_attrs = atom_scores[atom_scores < -0.1]
    neu_attrs = atom_scores[np.abs(atom_scores) <= 0.1]

    print(f"\n  Positive attributions (>0.1): {len(pos_attrs)} atoms")
    print(f"  Negative attributions (<-0.1): {len(neg_attrs)} atoms")
    print(f"  Neutral attributions (±0.1): {len(neu_attrs)} atoms")

    if len(pos_attrs) > 0:
        print(f"  Mean positive: {np.mean(pos_attrs):.4f}")
    if len(neg_attrs) > 0:
        print(f"  Mean negative: {np.mean(neg_attrs):.4f}")

    print(f"\n{'='*70}")
    print(f"COLOR INTERPRETATION:")
    print(f"{'='*70}")
    print(f"  🔵 BLUE regions: Increase activity prediction")
    print(f"  🔴 RED regions: Decrease activity prediction")
    print(f"  ⚪ WHITE regions: Neutral (minimal impact)")
    print(f"{'='*70}\n")


def get_top_attributions(smiles: str, atom_scores: np.ndarray,
                        top_k: int = 5) -> Dict[str, Any]:
    """
    Get top positive and negative attributions with atom indices.

    Args:
        smiles: SMILES string
        atom_scores: Per-atom attribution scores
        top_k: Number of top attributions to return

    Returns:
        Dictionary with top positive and negative atoms
    """
    if not RDKIT_AVAILABLE:
        return {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    # Get indices sorted by attribution
    pos_indices = np.argsort(atom_scores)[-top_k:][::-1]
    neg_indices = np.argsort(atom_scores)[:top_k]

    result = {
        'top_positive': [],
        'top_negative': []
    }

    for idx in pos_indices:
        if atom_scores[idx] > 0:
            atom = mol.GetAtomWithIdx(int(idx))
            result['top_positive'].append({
                'atom_idx': int(idx),
                'symbol': atom.GetSymbol(),
                'attribution': float(atom_scores[idx])
            })

    for idx in neg_indices:
        if atom_scores[idx] < 0:
            atom = mol.GetAtomWithIdx(int(idx))
            result['top_negative'].append({
                'atom_idx': int(idx),
                'symbol': atom.GetSymbol(),
                'attribution': float(atom_scores[idx])
            })

    return result


def identify_ring_substructures(smiles: str, atom_scores: np.ndarray) -> Dict[str, Any]:
    """
    Identify ring systems and aggregate their attributions.
    Simpler approach: find all rings and treat each as a substructure.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'substructures': [], 'error': 'invalid_smiles'}

        # Get ring info
        ri = mol.GetRingInfo()
        atom_rings = ri.AtomRings()

        substructures = []
        assigned_atoms = set()

        # Process each ring
        for ring_atoms in atom_rings:
            ring_atoms = list(ring_atoms)
            # Get attribution for this ring
            ring_scores = [atom_scores[i] for i in ring_atoms if i < len(atom_scores)]
            if ring_scores:
                avg_score = float(np.mean(ring_scores))
                total_score = float(np.sum(ring_scores))

                # Get substructure SMILES
                try:
                    # Create a substructure from these atoms
                    atom_set = set(ring_atoms)
                    bonds_to_include = []
                    for bond in mol.GetBonds():
                        if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set:
                            bonds_to_include.append(bond.GetIdx())

                    # Get the ring SMILES
                    substruct_smiles = Chem.MolFragmentToSmiles(mol, ring_atoms, bondsToUse=bonds_to_include)

                    substructures.append({
                        'type': 'ring',
                        'smiles': substruct_smiles,
                        'atoms': ring_atoms,
                        'attribution': total_score,
                        'attribution_avg': avg_score,
                        'size': len(ring_atoms)
                    })
                    assigned_atoms.update(ring_atoms)
                except:
                    pass

        # Group unassigned atoms into chains
        unassigned = [i for i in range(mol.GetNumAtoms()) if i not in assigned_atoms]
        if unassigned:
            chain_scores = [atom_scores[i] for i in unassigned if i < len(atom_scores)]
            if chain_scores:
                substructures.append({
                    'type': 'chain',
                    'smiles': 'Other atoms',
                    'atoms': unassigned,
                    'attribution': float(np.sum(chain_scores)),
                    'attribution_avg': float(np.mean(chain_scores)),
                    'size': len(unassigned)
                })

        # Sort by absolute attribution
        substructures.sort(key=lambda x: abs(x['attribution']), reverse=True)

        return {
            'substructures': substructures,
            'total_atoms': mol.GetNumAtoms(),
            'atom_scores': atom_scores.tolist()
        }

    except Exception as e:
        print(f"Error in identify_ring_substructures: {e}")
        import traceback
        traceback.print_exc()
        return {'substructures': [], 'error': str(e)}


def murcko_substructure_aggregation(smiles: str, atom_scores: np.ndarray,
                                     threshold: float = 0.1) -> Dict[str, Any]:
    """
    Aggregate atom attributions into positive/negative contributors.

    Simplified version that uses threshold-based grouping.
    (Full Murcko scaffolds require build_data.py which may not be available)

    Args:
        smiles: SMILES string
        atom_scores: Per-atom attribution scores
        threshold: Absolute threshold for considering atoms as contributors

    Returns:
        Dictionary with:
            - atom_scores: List of atom scores
            - positive_contributors: List of atom indices with positive attribution
            - negative_contributors: List of atom indices with negative attribution
            - num_substructures: Number of substructures (0 for simplified version)
            - total_atoms: Total number of atoms
    """
    try:
        print(f"[DEBUG] murcko_substructure_aggregation called")
        print(f"[DEBUG] atom_scores shape: {atom_scores.shape}, type: {type(atom_scores)}")
        print(f"[DEBUG] threshold: {threshold}")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("[DEBUG] Invalid SMILES, returning error")
            return {
                'atom_scores': atom_scores.tolist() if hasattr(atom_scores, 'tolist') else [],
                'positive_contributors': [],
                'negative_contributors': [],
                'num_substructures': 0,
                'total_atoms': 0,
                'method': 'error',
                'error': 'invalid_smiles'
            }

        num_atoms = mol.GetNumAtoms()
        print(f"[DEBUG] Molecule has {num_atoms} atoms")
        print(f"[DEBUG] atom_scores has {len(atom_scores)} elements")

        # Ensure atom_scores matches molecule size
        if len(atom_scores) != num_atoms:
            print(f"[DEBUG] Size mismatch! Interpolating from {len(atom_scores)} to {num_atoms}")
            if len(atom_scores) > 0 and num_atoms > 0:
                x_old = np.linspace(0, 1, len(atom_scores))
                x_new = np.linspace(0, 1, num_atoms)
                atom_scores = np.interp(x_new, x_old, atom_scores)
                print(f"[DEBUG] After interpolation: {len(atom_scores)} elements")
            else:
                print(f"[DEBUG] Creating zero array of size {num_atoms}")
                atom_scores = np.zeros(num_atoms)

        # Threshold-based grouping
        positive_contributors = [int(i) for i, score in enumerate(atom_scores) if score > threshold]
        negative_contributors = [int(i) for i, score in enumerate(atom_scores) if score < -threshold]

        print(f"[DEBUG] Found {len(positive_contributors)} positive contributors")
        print(f"[DEBUG] Found {len(negative_contributors)} negative contributors")
        print(f"[DEBUG] Score range: [{np.min(atom_scores):.4f}, {np.max(atom_scores):.4f}]")

        return {
            'atom_scores': atom_scores.tolist(),
            'positive_contributors': positive_contributors,
            'negative_contributors': negative_contributors,
            'num_substructures': 0,  # Simplified version doesn't use Murcko
            'avg_substructure_size': 0.0,
            'total_atoms': int(num_atoms),
            'method': 'threshold_based'
        }
    except Exception as e:
        print(f"[ERROR] Exception in murcko_substructure_aggregation: {e}")
        import traceback
        traceback.print_exc()
        return {
            'atom_scores': [],
            'positive_contributors': [],
            'negative_contributors': [],
            'num_substructures': 0,
            'total_atoms': 0,
            'method': 'error',
            'error': str(e)
        }


def visualize_with_substructures(smiles: str, atom_scores: np.ndarray,
                                  pred_prob: float, prediction: str,
                                  width: int = 800, height: int = 600) -> Optional[str]:
    """
    Visualize molecule with substructure-based coloring.

    Uses the aligned style matching comprehensive comparison:
    - p95 percentile normalization (robust to outliers)
    - Gamma correction (0.6) to boost mid-range intensities
    - Blue/Orange color scheme (color-blind friendly)
    - Halo-based highlighting

    Args:
        smiles: SMILES string
        atom_scores: Per-atom attribution scores
        pred_prob: Prediction probability
        prediction: Prediction label ('Active' or 'Inactive')
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        SVG string or None if visualization fails
    """
    try:
        print(f"[VIZ DEBUG] Starting visualization")
        print(f"[VIZ DEBUG] atom_scores type: {type(atom_scores)}, shape: {atom_scores.shape if hasattr(atom_scores, 'shape') else len(atom_scores)}")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("[VIZ DEBUG] Invalid SMILES")
            return None

        AllChem.Compute2DCoords(mol)
        num_atoms = mol.GetNumAtoms()
        print(f"[VIZ DEBUG] Molecule has {num_atoms} atoms")

        # Convert to numpy array if needed
        if not isinstance(atom_scores, np.ndarray):
            atom_scores = np.array(atom_scores)

        print(f"[VIZ DEBUG] atom_scores length: {len(atom_scores)}")

        # Ensure scores match molecule size
        if len(atom_scores) != num_atoms:
            print(f"[VIZ DEBUG] Size mismatch: {len(atom_scores)} scores vs {num_atoms} atoms, interpolating...")
            if len(atom_scores) > 0 and num_atoms > 0:
                x_old = np.linspace(0, 1, len(atom_scores))
                x_new = np.linspace(0, 1, num_atoms)
                atom_scores = np.interp(x_new, x_old, atom_scores)
            else:
                atom_scores = np.zeros(num_atoms)

        # p95 normalization (robust, matching xai_viz_helper.py style)
        abs_vals = np.abs(atom_scores)
        max_abs = float(abs_vals.max()) if abs_vals.size else 1e-9
        p95 = float(np.percentile(abs_vals, 95)) if abs_vals.size else 0.0
        denom = max(1e-9, p95 if p95 > 1e-12 else max_abs)
        print(f"[VIZ DEBUG] Normalization: max={max_abs:.4f}, p95={p95:.4f}, denom={denom:.4f}")

        # Color-blind friendly palette (matplotlib defaults)
        BLUE = (31/255.0, 119/255.0, 180/255.0)
        ORANGE = (1.0, 127/255.0, 14/255.0)

        atom_cols = {}
        atom_rads = {}
        bond_cols = {}

        # Color atoms (no threshold - color all non-zero)
        for i, s in enumerate(atom_scores):
            if s == 0:
                continue

            # Normalize and gamma correct
            inten = min(abs(s) / denom, 1.0)
            inten = pow(inten, 0.6)  # gamma < 1 boosts mid-range

            base = BLUE if s > 0 else ORANGE
            c = (1.0*(1-inten) + base[0]*inten,
                 1.0*(1-inten) + base[1]*inten,
                 1.0*(1-inten) + base[2]*inten)
            atom_cols[i] = c
            atom_rads[i] = 0.3 + 0.5*inten

        print(f"[VIZ DEBUG] Colored {len(atom_cols)} atoms")

        # Color bonds
        for bond in mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            e = bond.GetEndAtomIdx()
            if a in atom_cols or e in atom_cols:
                bond_cols[bond.GetIdx()] = atom_cols.get(a, atom_cols.get(e))

        print(f"[VIZ DEBUG] Colored {len(bond_cols)} bonds")

        # Draw with RDKit
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
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
            highlightAtomRadii=atom_rads,
        )
        drawer.FinishDrawing()

        print(f"[VIZ DEBUG] Successfully created SVG")
        return drawer.GetDrawingText()

    except Exception as e:
        print(f"[VIZ ERROR] Exception in visualization: {e}")
        import traceback
        traceback.print_exc()
        return None
