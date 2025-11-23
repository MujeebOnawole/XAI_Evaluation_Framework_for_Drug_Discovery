#!/usr/bin/env python3
"""
CNN XAI Activity Pairs Analysis (Occlusion-based Attribution + Murcko Aggregation)
==================================================================================

This script mirrors the RF TreeSHAP pairwise analysis, but for the 1D SMILES CNN
that consumes one-hot inputs of shape (seq_len=181, vocab=76).

Primary XAI: Token-level occlusion (perturbation-based attribution).
This provides a direct causal interpretation: the attribution for each token measures
the change in prediction when that token is masked (replaced with padding).

Secondary analysis:
  - Murcko substructure aggregation (groups atom scores by chemical substructures)
  - Pharmacophore recognition (compares attributions to expected patterns)
  - Functional-group aggregation using SMARTS patterns

Outputs a detailed pair-level CSV aligned with the RF output fields, plus optional
class-level JSON/plots via a companion analyzer script.

Quality Control Metrics:
========================
This script emits complementary prediction–attribution alignment metrics:
1) Sign‑Majority Alignment (conservative) and 2) Magnitude‑Weighted Alignment
(pharmacophore‑aware), enabling robustness and sensitivity analyses.

Usage:
  python cnn_xai_activity_pairs.py \
    --activity_csv activity_cliff_pairs.csv \
    --noncliff_csv non_cliff_pairs.csv \
    --seq_len 181 \
    [--per_model] [--ensemble] [--backbone_index 0] \
    [--pharmacophore_json pharm.json] \
    [--full] [--samples_per_class 2] [--limit_per_class N] \
    [--map_mode span|interpolate] [--fg_norm sum|mean|max] \
    [--calibration_json path.json] [--dump_dir outputs/cnn_xai_pairs_debug] [--seed 42]

Notes:
- Tokenization follows Config.VOCAB_CHARS and SMILESPreprocessor.
- Atom-level mapping uses length-preserving interpolation from token positions
  to RDKit atom indices (consistent with the GradCAM notebook utilities).
  This avoids brittle per-character atom mapping with multichar tokens.
- Functional group scoring uses the RF FG_SMARTS dictionary.
- Checkpoints: if --ckpt is not provided, the script auto-discovers all
  .ckpt files under model_checkpoints/ and loads them as an ensemble.
- Occlusion method is consistent with RGCN's substructure masking approach,
  enabling direct cross-model comparison.
"""

import os
import csv
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
from rdkit import Chem
from rdkit.Chem import rdmolops

from config import Config
from data_preprocessing import SMILESPreprocessor
from model import SMILESCNNModel
from CNN_model.cnn_token_mapper import token_scores_to_atom_scores
try:
    # Optional import for Murcko leaves if available elsewhere in project
    from build_data import return_murcko_leaf_structure  # type: ignore
except Exception:
    return_murcko_leaf_structure = None  # type: ignore

# Functional group SMARTS (85 RDKit fragment patterns) for cross-model comparability
FG_SMARTS: Dict[str, str] = {
    'fr_Al_COO': '[CX3](=O)[O-,OH]',
    'fr_Al_OH': '[OH1][#6,#1]',
    'fr_Al_OH_noTert': '[OH1;!$(C(C)(C)C)]',
    'fr_ArN': '[n,N;!H0]',
    'fr_Ar_COO': 'c[CX3](=O)[O-,OH]',
    'fr_Ar_N': '[nH1,nH2,NH1,NH2;$([n,N][#6])]',
    'fr_Ar_NH': '[nH1,nH2,NH1,NH2;$([n,N]c)]',
    'fr_Ar_OH': '[OH1;$(O[#6;#6])]',
    'fr_COO': '[CX3](=O)[OX2H1]',
    'fr_COO2': '[CX3](=O)[OX1-,OX2H1]',
    'fr_C_O': '[C][O]',
    'fr_C_O_noCOO': '[CX4][OX2H1]',
    'fr_C_S': '[#6][#16]',
    'fr_HOCCN': '[OH1][#6][#6][#7]',
    'fr_Imine': '[CX3;$([C]([#6])[#6]),$([CH][#6]),$([CH2])]=[NX2][#6]',
    'fr_NH0': '[NX3,NX4+;!$(N[#1]);!$(N[#6]=,#[#6,#7,#8,#15,#16])]',
    'fr_NH1': '[NH1]',
    'fr_NH2': '[NH2]',
    'fr_N_O': '[#7][#8]',
    'fr_Ndealkylation1': '[NH0]([#6])([#6])[#6][CH3]',
    'fr_Ndealkylation2': '[NH0]([#6])([#6])[#6][CH2][CH3]',
    'fr_Nhpyrrole': '[nH1]1[#6][#6][#6][#6]1',
    'fr_SH': '[#16H1]',
    'fr_aldehyde': '[CHX1](=O)',
    'fr_alkyl_carbamate': '[CH0](=[OX1])[NX2]',
    'fr_alkyl_halide': '[CX4][ClX1,BrX1,IX1]',
    'fr_allylic_oxid': '[OX2H1][#6;!$(C=O)]',
    'fr_amide': '[CX3](=[OX1])[NX3,NX4+]',
    'fr_amidine': '[NX3][CX3]=[NX2]',
    'fr_aniline': '[NH1,NH2;$(N[#6])]',
    'fr_aryl_methyl': '[CH3][#6]',
    'fr_azide': '[NX1]~[NX2]~[NX1,NX2]',
    'fr_azo': '[#6][NX2]=[NX2][#6]',
    'fr_barbitur': '[C,c]1[NH1][CX3](=[OX1])[NH1][CX3](=[OX1])[C,c]1',
    'fr_benzene': 'c1ccccc1',
    'fr_benzodiazepine': '[c,C]1[n,N][c,C][c,C]2[c,C](=[OX1])[n,N][c,C][c,C][c,C]12',
    'fr_bicyclic': '[R2]',
    'fr_diazo': '[CX3]=[NX2+]=[NX1-]',
    'fr_dihydropyridine': '[NH1]1[CX4][CX4][CX3]=[CX3][CX4]1',
    'fr_disulfide': '[#16X2][#16X2]',
    'fr_epoxide': '[OX2r3]1[#6r3][#6r3]1',
    'fr_ester': '[#6][CX3](=O)[OX2H0][#6]',
    'fr_ether': '[OD2]([#6])[#6]',
    'fr_furan': '[oH0]1[#6][#6][#6][#6]1',
    'fr_guanido': '[NH1,NH2][CX3](=[NH1])[NH1,NH2]',
    'fr_halogen': '[#9,#17,#35,#53]',
    'fr_hdrzine': '[NX3][NX3]',
    'fr_hdrzone': '[NX3][NX3][CX3](=[OX1])',
    'fr_imidazole': '[nH0]1[#6][nH1][#6][#6]1',
    'fr_imide': '[CX3](=[OX1])[NH1][CX3](=[OX1])',
    'fr_isocyan': '[NX2]=[CX2]=[OX1]',
    'fr_isothiocyan': '[NX2]=[CX2]=[SX1]',
    'fr_ketone': '[#6][CX3](=[OX1])[#6]',
    'fr_ketone_Topliss': '[#6][CX3](=[OX1])[#6]',
    'fr_lactam': '[NH1,NH0][CX3](=[OX1])[#6]',
    'fr_lactone': '[#6][CX3](=[OX1])[OX2][#6]',
    'fr_methoxy': '[OX2]([#6])[CH3]',
    'fr_morpholine': '[OX2]1[#6][#6][NH1,NH0][#6][#6]1',
    'fr_nitrile': '[NX1]#[CX2]',
    'fr_nitro': '[NX3+]([OX1-])[OX1-]',
    'fr_nitro_arom': '[$([NX3+](=[OX1])[OX1-]),$([NX3+]([OX1-])=[OX1])]',
    'fr_nitro_arom_nonortho': '[$([NX3+](=[OX1])[OX1-]),$([NX3+]([OX1-])=[OX1])]',
    'fr_nitroso': '[NX2](=[OX1])',
    'fr_oxazole': '[oH0]1[#6][nH0][#6][#6]1',
    'fr_oxime': '[CX3]=[NX2][OH1]',
    'fr_para_hydroxylation': 'c1ccc(O)cc1',
    'fr_phenol': '[OH1][c]',
    'fr_phenol_noOrthoHbond': '[OH1][c;!$(c[c][c][NH1,NH2,NH3+,nH+,OH1,SH1])]',
    'fr_phos_acid': '[PX4](=[OX1])([OH1])([OH1])',
    'fr_phos_ester': '[PX4](=[OX1])([OH1,OH0])([OH1,OH0])',
    'fr_piperdine': '[NH1,NH0]1[CX4][CX4][CX4][CX4][CX4]1',
    'fr_piperzine': '[NH1,NH0]1[CX4][CX4][NH1,NH0][CX4][CX4]1',
    'fr_priamide': '[CX3](=[OX1])[NH2]',
    'fr_prisulfonamd': '[NH2][SX4](=[OX1])(=[OX1])',
    'fr_pyridine': '[nH0]1[#6][#6][#6][#6][#6]1',
    'fr_quatN': '[NX4+]',
    'fr_sulfide': '[#16X2H0]',
    'fr_sulfonamd': '[NH1,NH2][SX4](=[OX1])(=[OX1])',
    'fr_sulfone': '[SX4](=[OX1])(=[OX1])([#6])[#6]',
    'fr_term_acetylene': '[CX2]#[CX2H1]',
    'fr_tetrazole': '[nH0]1[nH0][nH0][nH1][#6]1',
    'fr_thiazole': '[sH0]1[#6][nH0][#6][#6]1',
    'fr_thiocyan': '[SX1]=[CX2]=[NX1]',
    'fr_thiophene': '[sH0]1[#6][#6][#6][#6]1',
    'fr_unbrch_alkane': '[R0;D2][R0;D2][R0;D2][R0;D2]',
    'fr_urea': '[NH1,NH2][CX3](=[OX1])[NH1,NH2]'
}

# Optional pharmacophore SMARTS mapping (loaded via --pharmacophore_json)
PH_SMARTS: Dict[str, str] = {}


def to_device(x: torch.Tensor, ref: torch.nn.Module) -> torch.Tensor:
    dev = next(ref.parameters()).device if any(p.requires_grad for p in ref.parameters()) else torch.device('cpu')
    return x.to(dev)


def prepare_input(preproc: SMILESPreprocessor, smiles: str, seq_len: int) -> np.ndarray:
    encoded = preproc.encode_smiles(smiles)
    padded = preproc.pad_sequences([encoded], max_length=seq_len)
    one_hot = preproc.one_hot_encode(padded)  # (1, L, V)
    return one_hot


@torch.no_grad()
def predict_prob(models: List[SMILESCNNModel], x: torch.Tensor, temperatures: Optional[List[float]] = None) -> float:
    if not models or len(models) == 0:
        print("❌ WARNING: predict_prob called with empty models list!", flush=True)
        return float('nan')

    probs = []
    for i, m in enumerate(models):
        m.eval()
        logits = m(x)
        if temperatures and i < len(temperatures) and temperatures[i] and temperatures[i] > 0:
            p = torch.sigmoid(logits / temperatures[i]).item()
        else:
            p = torch.sigmoid(logits).item()
        probs.append(p)

    if not probs:
        print("❌ WARNING: No probabilities computed in predict_prob!", flush=True)
        return float('nan')

    return float(np.mean(probs))


def occlusion_token_attr(model: SMILESCNNModel, x: torch.Tensor, pad_index: int = 0) -> torch.Tensor:
    """
    Compute token-level occlusion attribution for CNN (causal perturbation-based).

    This provides a direct causal interpretation: attribution[i] measures the
    change in prediction when token i is masked (replaced with padding).

    Args:
        model: CNN model
        x: input tensor (1, L, V) - one-hot encoded SMILES
        pad_index: vocabulary index for padding token (default 0)

    Returns:
        Tensor of shape (L, V) containing occlusion attribution scores

    Note:
        - Matches RGCN's substructure masking semantics
        - Suitable for CSPD_signed and cross-model comparison
        - Attribution = f(original) - f(masked)
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
        # (store in (L, V) format matching one-hot structure)
        original_token_idx = x[:, i, :].argmax().item()
        occlusion_scores[i, original_token_idx] = delta

    return occlusion_scores


def reduce_to_positions(attr: torch.Tensor, x: torch.Tensor) -> np.ndarray:
    """Reduce attribution tensor at token level via dot with one-hot along vocab axis -> (L,) and return numpy."""
    pos_attr = (attr * x.squeeze(0)).sum(dim=-1)  # (L,)
    return pos_attr.detach().cpu().numpy()


def interpolate_to_atoms(smiles: str, seq_scores: np.ndarray) -> np.ndarray:
    """
    Map sequence-position scores (length L) to atom-level scores (length n_atoms)
    via linear interpolation. Matches approach used in GradCAM notebook.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.array([])
        n_atoms = mol.GetNumAtoms()
        if n_atoms <= 0:
            return np.array([])
        x_old = np.linspace(0.0, 1.0, num=len(seq_scores))
        x_new = np.linspace(0.0, 1.0, num=n_atoms)
        atom_scores = np.interp(x_new, x_old, seq_scores)
        return atom_scores
    except Exception:
        return np.array([])


def smartsmatch_groups(smiles: str, atom_scores: np.ndarray, fg_smarts: Dict[str, str], agg: str = 'sum') -> Tuple[Dict[str, float], Dict[str, int], Dict[str, List[int]]]:
    """
    Compute per-functional-group scores by summing atom_scores over matched atoms.

    Returns:
        (group_scores, group_presence)
    """
    group_scores: Dict[str, float] = {}
    group_presence: Dict[str, int] = {}
    group_atoms: Dict[str, List[int]] = {}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or atom_scores.size == 0:
        return group_scores, group_presence

    for name, smarts in fg_smarts.items():
        try:
            patt = Chem.MolFromSmarts(smarts)
            if not patt:
                continue
            matches = mol.GetSubstructMatches(patt)
            if not matches:
                group_presence[name] = 0
                continue
            # Aggregate unique atoms' scores across all matches
            atoms = sorted({a for match in matches for a in match})
            vals = atom_scores[atoms] if len(atoms) else np.array([])
            if agg == 'mean' and vals.size:
                score = float(np.mean(vals))
            elif agg == 'max' and vals.size:
                score = float(np.max(vals))
            else:
                score = float(np.sum(vals)) if vals.size else 0.0
            group_scores[name] = score
            group_presence[name] = 1
            group_atoms[name] = atoms
        except Exception:
            continue

    return group_scores, group_presence, group_atoms


def dict_delta_abs(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    keys = set(a.keys()) | set(b.keys())
    return {k: abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys}


def presence_delta(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    keys = set(a.keys()) | set(b.keys())
    return {k: (a.get(k, 0) - b.get(k, 0)) for k in keys}


def topk_changed(delta_abs: Dict[str, float], k: int = 8) -> List[str]:
    items = sorted(delta_abs.items(), key=lambda kv: -abs(kv[1]))
    return [f"{k}:{v:.4f}" for k, v in items[:k] if np.isfinite(v) and v != 0.0]


def occlusion_sanity(model: SMILESCNNModel, x: torch.Tensor, pos_attr: np.ndarray, topk: int = 5) -> Dict[str, Any]:
    """
    Occlude top-K positive and negative positions by replacing with PAD token.
    Check that prob shifts align with IG signs.
    """
    with torch.no_grad():
        base_prob = torch.sigmoid(model(x)).item()

    L = x.shape[1]
    attr = pos_attr.copy()
    idx_pos = np.argsort(-attr)[:min(topk, L)]
    idx_neg = np.argsort(attr)[:min(topk, L)]

    def occlude_positions(idxs: List[int]) -> float:
        x_occ = x.clone()
        x_occ[:, idxs, :] = 0.0
        x_occ[:, idxs, 0] = 1.0  # PAD
        with torch.no_grad():
            prob = torch.sigmoid(model(x_occ)).item()
        return prob

    pos_prob = occlude_positions(list(map(int, idx_pos))) if len(idx_pos) else base_prob
    neg_prob = occlude_positions(list(map(int, idx_neg))) if len(idx_neg) else base_prob

    pos_expected = pos_prob < base_prob  # removing positive tokens should reduce prob
    neg_expected = neg_prob > base_prob  # removing negative tokens should increase prob

    return {
        'base_prob': base_prob,
        'pos_prob_after_occlusion': pos_prob,
        'neg_prob_after_occlusion': neg_prob,
        'pos_direction_ok': bool(pos_expected),
        'neg_direction_ok': bool(neg_expected),
        'sanity_pass': bool(pos_expected and neg_expected)
    }


def span_positions_for_atoms(seq_len: int, n_atoms: int) -> List[List[int]]:
    """Split sequence positions [0..L-1] into n_atoms contiguous bins (span mapping)."""
    if n_atoms <= 0:
        return []
    indices = list(range(seq_len))
    bins = np.array_split(indices, n_atoms)
    return [list(b.astype(int)) for b in bins]


def atom_scores_from_positions(seq_scores: np.ndarray, seq_len: int, n_atoms: int, mode: str = 'span') -> np.ndarray:
    if n_atoms <= 0:
        return np.array([])
    if mode == 'span':
        bins = span_positions_for_atoms(seq_len, n_atoms)
        vals = [float(np.sum(seq_scores[b])) if len(b) else 0.0 for b in bins]
        return np.array(vals, dtype=float)
    # fallback interpolate
    return interpolate_to_atoms('', seq_scores)


def ring_or_murcko_substructures(smiles: str) -> Dict[str, List[int]]:
    try:
        if return_murcko_leaf_structure is not None:
            subs = return_murcko_leaf_structure(smiles).get('substructure', {}) or {}
            return {str(k): [int(a) for a in v] for k, v in subs.items() if v}
    except Exception:
        pass
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    rings = list(Chem.GetSymmSSSR(mol))
    out: Dict[str, List[int]] = {}
    for i, r in enumerate(rings):
        atoms = [int(a) for a in list(r)]
        if atoms:
            out[f"ring_{i}"] = atoms
    return out


def prepare_visualization_data_cnn(atom_scores: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
    """Convert per-atom scores to visualization: blue (pos), red (neg), neutral no color."""
    if atom_scores is None or atom_scores.size == 0:
        return {'error': 'no_atoms'}
    colors = []
    # Color-blind friendly palette: cyan for positive, orange for negative
    CYAN = (0.0, 1.0, 1.0)   # #00FFFF
    ORANGE = (1.0, 0.4980, 0.0550)    # #ff7f0e
    pos_idx = []
    neg_idx = []
    neu_idx = []
    for i, s in enumerate(atom_scores):
        if s > threshold:
            inten = min(abs(s) / (2*threshold), 1.0)
            c = (1.0*(1-inten) + CYAN[0]*inten,
                 1.0*(1-inten) + CYAN[1]*inten,
                 1.0*(1-inten) + CYAN[2]*inten)
            colors.append(c)
            pos_idx.append(i)
        elif s < -threshold:
            inten = min(abs(s) / (2*threshold), 1.0)
            c = (1.0*(1-inten) + ORANGE[0]*inten,
                 1.0*(1-inten) + ORANGE[1]*inten,
                 1.0*(1-inten) + ORANGE[2]*inten)
            colors.append(c)
            neg_idx.append(i)
        else:
            colors.append(None)
            neu_idx.append(i)
    return {
        'atom_attributions': atom_scores.tolist(),
        'atom_colors': colors,
        'positive_atoms': pos_idx,
        'negative_atoms': neg_idx,
        'neutral_atoms': neu_idx,
        'n_atoms': int(len(atom_scores))
    }


def validate_pharmacophore_recognition(smiles: str,
                                       antibiotic_class: str,
                                       atom_scores: np.ndarray,
                                       pharmacophore_json_path: str = 'pharmacophore.json') -> Dict[str, Any]:
    try:
        with open(pharmacophore_json_path, 'r') as f:
            pharm = json.load(f)
    except Exception:
        return {'error': 'pharmacophore_json_not_found'}
    sec = pharm.get(antibiotic_class)
    if not isinstance(sec, dict):
        return {'error': f'class_{antibiotic_class}_not_in_pharmacophore_json'}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or atom_scores is None or atom_scores.size != mol.GetNumAtoms():
        return {'error': 'invalid_molecule_or_scores'}
    # Desaturated: strict top-k by magnitude with molecule-size scaled overlap threshold
    n = int(atom_scores.size)
    k = max(5, int(np.ceil(0.10 * n))) if n > 0 else 0
    if n > 0 and k > 0:
        top_k_idx = np.argsort(-np.abs(atom_scores))[:k]
        highlighted = set(map(int, top_k_idx))
    else:
        highlighted = set()
    recognized = []
    overlap = {}
    expected = []
    missed = []
    for cat in ['required_any','loose_required_any','important_any','optional_any']:
        for feat in (sec.get(cat) or []):
            name = feat.get('name'); smt = feat.get('smarts')
            if not name or not smt:
                continue
            patt = Chem.MolFromSmarts(smt)
            if not patt:
                continue
            matches = mol.GetSubstructMatches(patt)
            atoms = set(a for m in matches for a in m) if matches else set()
            if not atoms:
                continue
            ov = len(highlighted & atoms) / float(len(atoms))
            overlap[name] = ov
            # Scale overlap threshold by molecule size
            min_overlap = 0.50 if n >= 50 else 0.60
            if ov >= min_overlap:
                recognized.append(name)
            if cat in ['required_any','important_any']:
                expected.append(name)
                if ov <= 0.3:
                    missed.append(name)
    weights = {}
    for cat in ['required_any','important_any','optional_any']:
        for feat in (sec.get(cat) or []):
            if 'name' in feat:
                weights[feat['name']] = float(feat.get('weight', 1.0))
    if overlap:
        num = sum(overlap[k]*weights.get(k,1.0) for k in overlap)
        den = sum(weights.get(k,1.0) for k in overlap)
        overall = float(num/den) if den>0 else 0.0
    else:
        overall = 0.0
    n_exp = len(expected)
    n_rec = len([f for f in expected if f in recognized])
    return {
        'recognized_features': recognized,
        'overlap_scores': overlap,
        'overall_recognition_score': overall,
        'expected_features': expected,
        'missed_features': missed,
        'n_expected': n_exp,
        'n_recognized': n_rec,
        'recognition_rate': float(n_rec/n_exp) if n_exp else 0.0
    }


def core_atoms_from_pharm(smiles: str, antibiotic_class: str, pharmacophore_json_path: str) -> List[int]:
    try:
        with open(pharmacophore_json_path, 'r') as f:
            pharm = json.load(f)
    except Exception:
        return []
    sec = pharm.get(antibiotic_class)
    if not isinstance(sec, dict):
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    atoms_set = set()
    for feat in (sec.get('required_any') or []):
        smt = feat.get('smarts')
        if not smt:
            continue
        patt = Chem.MolFromSmarts(smt)
        if not patt:
            continue
        matches = mol.GetSubstructMatches(patt)
        for m in matches:
            atoms_set.update(int(a) for a in m)
    return sorted(atoms_set)


def top_mass_atoms(scores: np.ndarray, pct: float = 0.5) -> List[int]:
    s = np.abs(scores).astype(float)
    total = s.sum()
    if total <= 0:
        return []
    order = np.argsort(-s)
    cum = 0.0
    selected: List[int] = []
    for idx in order:
        selected.append(int(idx))
        cum += float(s[idx])
        if cum / total >= max(1e-6, float(pct)):
            break
    return selected


def saliency_entropy_gini(abs_scores: np.ndarray) -> Tuple[float, float]:
    s = abs_scores.astype(float)
    total = s.sum()
    if total <= 0:
        return 0.0, 0.0
    p = s / total
    # entropy
    eps = 1e-12
    H = float(-np.sum(p * np.log(p + eps)))
    # gini (0=fair/even, 1=concentrated)
    ps = np.sort(p)
    n = len(ps)
    # Using formula G = 1 - 2 * sum_i p_sorted[i] * (n - i + 0.5)/n
    G = 1.0 - 2.0 * float(np.sum(ps * (np.arange(n, 0, -1) - 0.5))) / float(n)
    return H, G


def topk_indices_by_abs(scores: np.ndarray, frac: float) -> List[int]:
    n = len(scores)
    k = max(1, int(round(frac * n)))
    idx = np.argsort(-np.abs(scores))[:k]
    return list(map(int, idx))


def fidelity_keep_drop(model: SMILESCNNModel, x: torch.Tensor, scores: np.ndarray, fracs=(0.05, 0.10, 0.20)) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with torch.no_grad():
        base = torch.sigmoid(model(x)).item()
    for frac, tag in zip(fracs, ['k5', 'k10', 'k20']):
        idx = topk_indices_by_abs(scores, frac)
        # keep-only: everything else to PAD
        x_keep = x.clone()
        mask = np.ones(x.shape[1], dtype=bool)
        mask[idx] = False
        if mask.any():
            x_keep[:, mask, :] = 0.0
            x_keep[:, mask, 0] = 1.0
        with torch.no_grad():
            p_keep = torch.sigmoid(model(x_keep)).item()
        out[f'fidelity_gain_keep_topk_{tag}'] = float(p_keep - base)
        out[f'class_flip_keep_topk_{tag}'] = bool((p_keep >= 0.5) != (base >= 0.5))

        # drop-only: set top-k to PAD
        x_drop = x.clone()
        x_drop[:, idx, :] = 0.0
        x_drop[:, idx, 0] = 1.0
        with torch.no_grad():
            p_drop = torch.sigmoid(model(x_drop)).item()
        out[f'fidelity_drop_remove_topk_{tag}'] = float(p_drop - base)
        out[f'class_flip_remove_topk_{tag}'] = bool((p_drop >= 0.5) != (base >= 0.5))
    return out


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float('nan')
    a = a.astype(float)
    b = b.astype(float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 0 else float('nan')


def rdkit_distance_matrix(mol: Optional[Chem.Mol]) -> Optional[np.ndarray]:
    try:
        if mol is None:
            return None
        return rdmolops.GetDistanceMatrix(mol).astype(int)
    except Exception:
        return None


def atoms_within_radius(dm: Optional[np.ndarray], source_atoms: List[int], radius: int) -> List[int]:
    if dm is None or len(source_atoms) == 0:
        return []
    n = dm.shape[0]
    mask = np.zeros(n, dtype=bool)
    for s in source_atoms:
        if 0 <= s < n:
            mask |= (dm[s] <= radius)
    return list(np.where(mask)[0].astype(int))


def classify_pair(active_prob: float, inactive_prob: float, active_target: int, inactive_target: int, threshold: float = 0.5) -> Tuple[str, str, str, str, str, str]:
    """
    Classify pair using TARGET ground truth labels.

    Classification logic:
    - TARGET=1, pred≥0.5 → TP
    - TARGET=1, pred<0.5 → FN
    - TARGET=0, pred<0.5 → TN
    - TARGET=0, pred≥0.5 → FP
    """
    active_cls = 'active' if active_prob >= threshold else 'inactive'
    inactive_cls = 'active' if inactive_prob >= threshold else 'inactive'

    # Use TARGET labels for classification
    if active_target == 1:
        active_classification = 'TP' if active_prob >= threshold else 'FN'
    else:  # active_target == 0
        active_classification = 'TN' if active_prob < threshold else 'FP'

    if inactive_target == 1:
        inactive_classification = 'TP' if inactive_prob >= threshold else 'FN'
    else:  # inactive_target == 0
        inactive_classification = 'TN' if inactive_prob < threshold else 'FP'

    # Pair classification
    both_correct = active_classification in ['TP', 'TN'] and inactive_classification in ['TP', 'TN']
    one_correct = (active_classification in ['TP', 'TN']) != (inactive_classification in ['TP', 'TN'])

    if both_correct:
        pair_cls = 'BothCorrect'
    elif one_correct:
        pair_cls = 'OneCorrect'
    else:
        pair_cls = 'BothWrong'

    # Legacy active_correct/inactive_correct for backward compatibility
    active_correct = active_classification in ['TP', 'TN']
    inactive_correct = inactive_classification in ['TP', 'TN']

    return (
        active_cls,
        inactive_cls,
        pair_cls,
        f"active_correct={active_correct};inactive_correct={inactive_correct}",
        active_classification,
        inactive_classification,
    )


def murcko_substructure_aggregation(smiles: str, atom_scores: np.ndarray) -> Dict[str, Any]:
    """
    Aggregate existing atom attributions by Murcko substructures (no masking).

    This approach:
    1. Takes existing atom-level attributions (from occlusion-based XAI)
    2. Identifies Murcko substructures
    3. Aggregates attributions within each substructure
    4. Returns substructure-level and atom-level statistics

    Args:
        smiles: Input SMILES string
        atom_scores: Existing atom attribution scores (from IG)

    Returns:
        Dictionary with aggregated substructure information
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or return_murcko_leaf_structure is None:
        return {
            'atom_scores': atom_scores.tolist() if atom_scores.size else [],
            'positive_contributors': [],
            'negative_contributors': [],
            'num_substructures': 0,
            'avg_substructure_size': 0.0,
            'total_atoms': 0,
            'error': 'invalid_smiles_or_no_murcko'
        }

    num_atoms = mol.GetNumAtoms()

    # Ensure atom_scores matches molecule size
    if atom_scores.size != num_atoms:
        # Interpolate if sizes don't match
        if atom_scores.size > 0 and num_atoms > 0:
            x_old = np.linspace(0, 1, len(atom_scores))
            x_new = np.linspace(0, 1, num_atoms)
            atom_scores = np.interp(x_new, x_old, atom_scores)
        else:
            atom_scores = np.zeros(num_atoms)

    # Get Murcko substructures
    try:
        murcko_data = return_murcko_leaf_structure(smiles)
        substructures = murcko_data.get('substructure', {})
    except Exception:
        substructures = {}

    if not substructures:
        # No substructures found, return original scores
        positive_contributors = [int(i) for i, score in enumerate(atom_scores) if score > 0.1]
        negative_contributors = [int(i) for i, score in enumerate(atom_scores) if score < -0.1]
        return {
            'atom_scores': atom_scores.tolist(),
            'positive_contributors': positive_contributors,
            'negative_contributors': negative_contributors,
            'num_substructures': 0,
            'avg_substructure_size': 0.0,
            'total_atoms': int(num_atoms),
            'error': 'no_substructures_found'
        }

    # Aggregate by substructure
    substructure_info = []
    for sub_id, atoms in substructures.items():
        if not atoms or not isinstance(atoms, list):
            continue

        # Get scores for atoms in this substructure
        sub_scores = [atom_scores[i] for i in atoms if 0 <= i < len(atom_scores)]
        if sub_scores:
            substructure_info.append({
                'id': sub_id,
                'atoms': atoms,
                'size': len(atoms),
                'mean_score': float(np.mean(sub_scores)),
                'max_score': float(np.max(np.abs(sub_scores))),
                'sum_score': float(np.sum(sub_scores))
            })

    # Calculate statistics
    substructure_sizes = [s['size'] for s in substructure_info]
    avg_substructure_size = float(np.mean(substructure_sizes)) if substructure_sizes else 0.0

    # Identify contributors (threshold: 0.1)
    positive_contributors = [int(i) for i, score in enumerate(atom_scores) if score > 0.1]
    negative_contributors = [int(i) for i, score in enumerate(atom_scores) if score < -0.1]

    return {
        'atom_scores': atom_scores.tolist(),
        'positive_contributors': positive_contributors,
        'negative_contributors': negative_contributors,
        'num_substructures': len(substructure_info),
        'avg_substructure_size': avg_substructure_size,
        'total_atoms': int(num_atoms),
        'substructure_details': substructure_info  # Optional detailed breakdown
    }


def pharmacophore_recognition_with_atoms(smiles: str,
                                         antibiotic_class: str,
                                         positive_atoms: List[int],
                                         pharmacophore_json_path: str) -> Dict[str, Any]:
    """
    Calculate pharmacophore recognition based on atom contributors.

    Returns:
        Dictionary with recognized patterns, scores, and alignment metrics
    """
    try:
        with open(pharmacophore_json_path, 'r') as f:
            pharm = json.load(f)
    except Exception:
        return {'error': 'pharmacophore_json_not_found'}

    class_patterns = pharm.get(antibiotic_class)
    if not isinstance(class_patterns, dict):
        return {'error': f'class_{antibiotic_class}_not_in_pharmacophore'}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'error': 'invalid_smiles'}

    positive_atoms_set = set(positive_atoms)

    recognized = []
    missed = []
    expected = []
    overlap_scores = {}

    # Check required and important patterns
    for cat in ['required_any', 'loose_required_any', 'important_any']:
        for feat in (class_patterns.get(cat) or []):
            name = feat.get('name')
            smarts = feat.get('smarts')
            if not name or not smarts:
                continue

            patt = Chem.MolFromSmarts(smarts)
            if not patt:
                continue

            matches = mol.GetSubstructMatches(patt)
            if not matches:
                continue

            # Get all atoms in this pattern
            pattern_atoms = set(a for m in matches for a in m)
            if not pattern_atoms:
                continue

            # Calculate overlap with positive contributors
            overlap = len(positive_atoms_set & pattern_atoms)
            overlap_score = overlap / len(pattern_atoms)
            overlap_scores[name] = overlap_score

            # Track as expected if in required/important
            if cat in ['required_any', 'important_any']:
                expected.append(name)
                if overlap_score >= 0.3:  # Recognition threshold
                    recognized.append(name)
                else:
                    missed.append(name)

    # Overall recognition score
    n_expected = len(expected)
    n_recognized = len(recognized)
    recognition_rate = n_recognized / n_expected if n_expected > 0 else 0.0

    # Weighted score
    weights = {}
    for cat in ['required_any', 'important_any', 'optional_any']:
        for feat in (class_patterns.get(cat) or []):
            if 'name' in feat:
                weights[feat['name']] = float(feat.get('weight', 1.0))

    if overlap_scores:
        num = sum(overlap_scores[k] * weights.get(k, 1.0) for k in overlap_scores)
        den = sum(weights.get(k, 1.0) for k in overlap_scores)
        overall_score = num / den if den > 0 else 0.0
    else:
        overall_score = 0.0

    return {
        'recognized_patterns': recognized,
        'missed_patterns': missed,
        'expected_patterns': expected,
        'overlap_scores': overlap_scores,
        'recognition_score': float(overall_score),
        'recognition_rate': float(recognition_rate),
        'n_recognized': n_recognized,
        'n_expected': n_expected
    }


def process_pair_row(row: pd.Series,
                     models: List[SMILESCNNModel],
                     preproc: SMILESPreprocessor,
                     seq_len: int,
                     threshold: float,
                     map_mode: str,
                     fg_norm: str,
                     model_id: str = '',
                     is_ensemble: bool = False,
                     backbone_id: str = '',
                     ph_smarts: Optional[Dict[str, str]] = None,
                     pharm_json_path: Optional[str] = None,
                     token_atom_recall_min: float = 0.9,
                     pharm_threshold: float = 0.2,
                     top_mass_pct: float = 0.5,
                     viz_percentile: int = 80,
                     temperatures: Optional[List[float]] = None,
                     dump_dir: Optional[str] = None,
                     backbone_ckpt: Optional[str] = None) -> Dict[str, Any]:
    # Safety check: ensure models list is not empty
    if not models or len(models) == 0:
        import sys
        error_msg = 'FATAL: No models provided to process_pair_row'
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"❌ {error_msg}", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)
        return {
            'pair_type': row.get('pair_type', 'unknown'),
            'error': error_msg,
            'is_cliff': row.get('is_cliff', False)
        }

    # Debug header with checkpoint tracking
    try:
        pair_id_dbg = f"{row.get('active_compound_id', row.get('compound1_id','UNK'))}_{row.get('inactive_compound_id', row.get('compound2_id','UNK'))}"
    except Exception as e:
        import sys
        pair_id_dbg = 'UNK_UNK'
        print(f"WARNING: Could not extract pair ID: {e}", file=sys.stderr, flush=True)

    print("\n" + "="*60, flush=True)
    print(f"Processing pair: {pair_id_dbg}", flush=True)
    print(f"Models available: {len(models)}", flush=True)
    print("="*60, flush=True)

    # Checkpoint tracker for debugging
    checkpoint = "START"
    # Extract schema variants for balanced files
    if 'active_smiles' in row and 'inactive_smiles' in row:
        cls = row.get('class', row.get('antibiotic_class', ''))
        active_id = row.get('active_compound_id', '')
        inactive_id = row.get('inactive_compound_id', '')
        active_smiles = row['active_smiles']
        inactive_smiles = row['inactive_smiles']
        similarity = float(row.get('structural_similarity', np.nan))
        pair_type = 'cliff'
        # TARGET ground truth and groups
        try:
            active_target = int(row.get('active_TARGET', 1))
        except Exception:
            active_target = 1
        try:
            inactive_target = int(row.get('inactive_TARGET', 0))
        except Exception:
            inactive_target = 0
        active_group = row.get('active_group', 'unknown')
        inactive_group = row.get('inactive_group', 'unknown')
    else:
        cls = row.get('class', '')
        active_id = row.get('compound1_id', '')
        inactive_id = row.get('compound2_id', '')
        active_smiles = row.get('compound1_smiles', '')
        inactive_smiles = row.get('compound2_smiles', '')
        similarity = float(row.get('structural_similarity', np.nan))
        pair_type = 'noncliff'
        try:
            active_target = int(row.get('compound1_TARGET', row.get('TARGET', 1)))
        except Exception:
            active_target = 1
        try:
            inactive_target = int(row.get('compound2_TARGET', row.get('TARGET', 1)))
        except Exception:
            inactive_target = 1
        active_group = row.get('compound1_group', 'unknown')
        inactive_group = row.get('compound2_group', 'unknown')

    # Prepare inputs
    checkpoint = "prepare_inputs"
    # print(f"[{checkpoint}] Preparing model inputs...")
    x_a_np = prepare_input(preproc, str(active_smiles), seq_len)
    x_i_np = prepare_input(preproc, str(inactive_smiles), seq_len)

    x_a = torch.from_numpy(x_a_np).float()
    x_i = torch.from_numpy(x_i_np).float()

    # Use the first model as attribution backbone
    model = models[0]
    model.eval()

    x_a = to_device(x_a, model)
    x_i = to_device(x_i, model)

    # Predictions (average across ensemble if provided)
    checkpoint = "predictions"
    # print(f"[{checkpoint}] Computing predictions...")
    active_prob = predict_prob(models, x_a, temperatures)
    inactive_prob = predict_prob(models, x_i, temperatures)
    raw_active_prob = predict_prob(models, x_a, None) if temperatures else active_prob
    raw_inactive_prob = predict_prob(models, x_i, None) if temperatures else inactive_prob
    pred_diff = float(active_prob - inactive_prob)

    # ========================================================================
    # Occlusion-based Attribution
    # ========================================================================
    # Perturbation-based XAI method for consistency with RGCN approach
    checkpoint = "occlusion"
    # print(f"[{checkpoint}] Computing occlusion attributions...")

    # Ensemble occlusion averaging
    attr_a_list: List[torch.Tensor] = []
    attr_i_list: List[torch.Tensor] = []
    for m in models:
        xa_m = to_device(x_a.clone(), m)
        xi_m = to_device(x_i.clone(), m)
        occ_a_m = occlusion_token_attr(m, xa_m, pad_index=0)
        occ_i_m = occlusion_token_attr(m, xi_m, pad_index=0)
        attr_a_list.append(occ_a_m.detach().cpu())
        attr_i_list.append(occ_i_m.detach().cpu())
    attr_a = torch.stack(attr_a_list).mean(dim=0)
    attr_i = torch.stack(attr_i_list).mean(dim=0)
    xai_method_name = 'Occlusion'
    # Std across models (mean over positions)
    try:
        stacked_a = torch.stack(attr_a_list)
        stacked_i = torch.stack(attr_i_list)
        _std_a = stacked_a.std(dim=0).mean().item()
        _std_i = stacked_i.std(dim=0).mean().item()
        attribution_std_across_models = float(0.5 * (_std_a + _std_i))
        # Variance warning if relative std > 50%
        mean_abs_a = attr_a.abs().mean().item() + 1e-12
        mean_abs_i = attr_i.abs().mean().item() + 1e-12
        rel_std = 0.5 * (_std_a / mean_abs_a + _std_i / mean_abs_i)
        if rel_std > 0.5:
            print(f"WARNING: Ensemble attribution variance high (relative std ~ {rel_std:.2f})")
        print(f"Ensemble attribution: averaged {len(models)} models")
    except Exception:
        attribution_std_across_models = float('nan')
    # Ensure attribution tensors live on same device as inputs for downstream ops
    attr_a = attr_a.to(x_a.device)
    attr_i = attr_i.to(x_i.device)

    pos_a = reduce_to_positions(attr_a, x_a)
    pos_i = reduce_to_positions(attr_i, x_i)

    # L1 normalization per example (preserve signed direction)
    def l1_norm(v: np.ndarray) -> np.ndarray:
        denom = float(np.sum(np.abs(v)))
        return v / denom if denom > 0 else v

    pos_a_norm = l1_norm(pos_a)
    pos_i_norm = l1_norm(pos_i)

    # Z-clip extreme outliers at |z| > 3 after L1 normalization (stability)
    def zclip(vals: np.ndarray, zmax: float = 3.0) -> np.ndarray:
        m = float(np.mean(vals))
        s = float(np.std(vals))
        if s <= 0:
            return vals
        low = m - zmax * s
        high = m + zmax * s
        return np.clip(vals, low, high)

    pos_a_norm = zclip(pos_a_norm)
    pos_i_norm = zclip(pos_i_norm)

    # Occlusion sanity
    sanity_a = occlusion_sanity(model, x_a, pos_a, topk=5)
    sanity_i = occlusion_sanity(model, x_i, pos_i, topk=5)
    sanity_pass = bool(sanity_a.get('sanity_pass', False) and sanity_i.get('sanity_pass', False))

    # Atom-level mapping (token-aware first, fallback to span/interpolate)
    checkpoint = "token_to_atom_mapping"
    # # print(f"[{checkpoint}] Mapping token attributions to atoms...")
    mapping_warn = ''
    atom_a = np.array([])
    atom_i = np.array([])
    try:
        mol_a = Chem.MolFromSmiles(str(active_smiles))
        mol_i = Chem.MolFromSmiles(str(inactive_smiles))

        if mol_a is None:
            mapping_warn = 'invalid_active_smiles'
            print(f" Invalid active SMILES: {active_smiles}")
        elif mol_i is None:
            mapping_warn = 'invalid_inactive_smiles'
            print(f" Invalid inactive SMILES: {inactive_smiles}")
        else:
            n_atoms_a = mol_a.GetNumAtoms()
            n_atoms_i = mol_i.GetNumAtoms()

            # # print(f"Token scores shape: active={pos_a_norm.shape}, inactive={pos_i_norm.shape}")
            # # print(f"Target atoms: active={n_atoms_a}, inactive={n_atoms_i}")
            # # print(f"Mapping mode: {map_mode}")

            if n_atoms_a == 0 or n_atoms_i == 0:
                mapping_warn = 'zero_atoms'
                print(f" Zero atoms detected")
            else:
                # Try token-aware mapping using original (unpadded) token lengths
                try:
                    enc_len_a = len(preproc.encode_smiles(str(active_smiles)))
                    enc_len_i = len(preproc.encode_smiles(str(inactive_smiles)))
                except Exception:
                    enc_len_a = enc_len_i = 0
                ta = pos_a_norm[:enc_len_a] if enc_len_a and enc_len_a <= len(pos_a_norm) else pos_a_norm
                ti = pos_i_norm[:enc_len_i] if enc_len_i and enc_len_i <= len(pos_i_norm) else pos_i_norm
                atom_a_ta, recall_a, warn_a = token_scores_to_atom_scores(str(active_smiles), ta)
                atom_i_ta, recall_i, warn_i = token_scores_to_atom_scores(str(inactive_smiles), ti)
                recall = float(0.5 * (recall_a + recall_i))
                if atom_a_ta.size and atom_i_ta.size:
                    atom_a = atom_a_ta
                    atom_i = atom_i_ta
                    if recall < float(token_atom_recall_min):
                        mapping_warn = 'recall_low'
                        print(f" Token→atom recall low: {recall:.3f} (< {token_atom_recall_min})")
                else:
                    mapping_warn = 'fallback_span'
                    # Primary mapping by span or interpolate
                    if map_mode == 'span':
                        atom_a = atom_scores_from_positions(pos_a_norm, seq_len, n_atoms_a, mode='span')
                        atom_i = atom_scores_from_positions(pos_i_norm, seq_len, n_atoms_i, mode='span')
                    else:
                        atom_a = interpolate_to_atoms(str(active_smiles), pos_a_norm)
                        atom_i = interpolate_to_atoms(str(inactive_smiles), pos_i_norm)

                # # print(f"Primary mapping result: active={atom_a.shape}, inactive={atom_i.shape}")

                # Fallback if primary failed
                if atom_a.size == 0 or atom_i.size == 0:
                    print(f" Primary mode '{map_mode}' failed, trying interpolate...")
                    mapping_warn = f'{map_mode}_fallback_interpolate'
                    atom_a = interpolate_to_atoms(str(active_smiles), pos_a_norm)
                    atom_i = interpolate_to_atoms(str(inactive_smiles), pos_i_norm)
                    # # print(f"Fallback result: active={atom_a.shape}, inactive={atom_i.shape}")

                # Final check
                if atom_a.size == 0 or atom_i.size == 0:
                    mapping_warn = 'all_mapping_failed'
                    print(f" ALL mapping modes failed!")
                    # Dummy arrays to prevent crashes
                    atom_a = np.zeros(n_atoms_a)
                    atom_i = np.zeros(n_atoms_i)
                else:
                    # # print(f" Mapping successful")
                    pass

    except Exception as e:
        mapping_warn = f'exception:{str(e)[:50]}'
        print(f" Exception in atom mapping: {e}")
        import traceback
        traceback.print_exc()

    # Group scoring (85 fr_* FG SMARTS for cross-model comparability)
    checkpoint = "functional_group_scoring"
    # # print(f"[{checkpoint}] Computing functional group scores...")
    group_scores_a, presence_a, group_atoms_a = smartsmatch_groups(str(active_smiles), atom_a, FG_SMARTS, agg=fg_norm)
    group_scores_i, presence_i, group_atoms_i = smartsmatch_groups(str(inactive_smiles), atom_i, FG_SMARTS, agg=fg_norm)

    # Signed delta and absolute delta
    signed_delta = {k: group_scores_a.get(k, 0.0) - group_scores_i.get(k, 0.0) for k in set(list(group_scores_a.keys()) + list(group_scores_i.keys()))}
    delta_abs = {k: abs(v) for k, v in signed_delta.items()}
    changed = [k for k, v in presence_delta(presence_a, presence_i).items() if v != 0]
    common = [k for k in (set(presence_a) & set(presence_i)) if presence_a.get(k, 0) == 1 and presence_i.get(k, 0) == 1]

    # Edit/Context masses
    edit_mass = float(np.sum([delta_abs.get(k, 0.0) for k in changed])) if changed else 0.0
    context_mass = float(np.sum([delta_abs.get(k, 0.0) for k in common])) if common else 0.0
    denom = edit_mass + context_mass
    propagation_index = float(context_mass / denom) if denom > 0 else 0.0
    edit_conc_index = float(edit_mass / denom) if denom > 0 else 0.0

    # Top edit driver among changed groups
    top_feature = None
    top_support = 0.0
    if changed:
        tkey = max(changed, key=lambda k: delta_abs.get(k, 0.0))
        top_feature = tkey
        top_support = float(delta_abs.get(tkey, 0.0))

    # Classifications (using TARGET ground truth)
    active_cls, inactive_cls, pair_cls, pair_flags, active_clf, inactive_clf = classify_pair(
        active_prob, inactive_prob, active_target, inactive_target, threshold=threshold
    )

    # === CNN-native locality, distances, fidelity, stability ===
    # RDKit distance matrices
    dm_a = rdkit_distance_matrix(mol_a)
    dm_i = rdkit_distance_matrix(mol_i)
    # Edited atoms per molecule: atoms from changed groups that are present in the molecule
    edited_atoms_a = sorted({a for g in changed if presence_a.get(g, 0) == 1 for a in group_atoms_a.get(g, [])}) if group_atoms_a else []
    edited_atoms_i = sorted({a for g in changed if presence_i.get(g, 0) == 1 for a in group_atoms_i.get(g, [])}) if group_atoms_i else []
    r1_atoms_a = atoms_within_radius(dm_a, edited_atoms_a, 1)
    r2_atoms_a = atoms_within_radius(dm_a, edited_atoms_a, 2)
    r1_atoms_i = atoms_within_radius(dm_i, edited_atoms_i, 1)
    r2_atoms_i = atoms_within_radius(dm_i, edited_atoms_i, 2)

    # Position bins per atom (span mapping)
    bins_a = span_positions_for_atoms(seq_len, len(atom_a)) if atom_a.size else []
    bins_i = span_positions_for_atoms(seq_len, len(atom_i)) if atom_i.size else []

    def overlap_fraction(pos_abs: np.ndarray, bins: List[List[int]], atom_set: List[int]) -> float:
        if not bins or not pos_abs.size or not atom_set:
            return 0.0
        total = float(np.sum(np.abs(pos_abs)))
        if total <= 0:
            return 0.0
        idxs = sorted({p for a in atom_set if a < len(bins) for p in bins[a]})
        mass = float(np.sum(np.abs(pos_abs[idxs]))) if idxs else 0.0
        return mass / total

    patch_edit_overlap = 0.5 * (overlap_fraction(pos_a_norm, bins_a, edited_atoms_a) + overlap_fraction(pos_i_norm, bins_i, edited_atoms_i))
    patch_edit_overlap_r1 = 0.5 * (overlap_fraction(pos_a_norm, bins_a, r1_atoms_a) + overlap_fraction(pos_i_norm, bins_i, r1_atoms_i))
    patch_edit_overlap_r2 = 0.5 * (overlap_fraction(pos_a_norm, bins_a, r2_atoms_a) + overlap_fraction(pos_i_norm, bins_i, r2_atoms_i))

    # Compactness & dispersion
    H_a, G_a = saliency_entropy_gini(np.abs(pos_a_norm))
    H_i, G_i = saliency_entropy_gini(np.abs(pos_i_norm))
    saliency_entropy = float(0.5 * (H_a + H_i))
    saliency_gini = float(0.5 * (G_a + G_i))
    top_idx = topk_indices_by_abs(pos_a_norm, 0.10) + topk_indices_by_abs(pos_i_norm, 0.10)
    explanation_diameter_px = int(max(top_idx) - min(top_idx)) if top_idx else 0

    # Fidelity keep/drop for k in {5,10,20}% (average across molecules)
    fid_a = fidelity_keep_drop(model, x_a, pos_a_norm, fracs=(0.05, 0.10, 0.20))
    fid_i = fidelity_keep_drop(model, x_i, pos_i_norm, fracs=(0.05, 0.10, 0.20))
    fidelity = {k: (fid_a.get(k, 0.0) + fid_i.get(k, 0.0)) / 2.0 for k in set(fid_a) | set(fid_i) if 'fidelity_' in k}
    flips = {k: bool(fid_a.get(k, False) or fid_i.get(k, False)) for k in set(fid_a) | set(fid_i) if 'class_flip' in k}

    # Stability metric (not applicable for occlusion-based method)
    # Occlusion is deterministic, so stability is always perfect
    ig_stability_r_val = 1.0  # Perfect stability for deterministic occlusion

    # Core/distant masses from edited atoms (r<=1 vs >1), sum across molecules (NaN-safe)
    if r1_atoms_a and atom_a.size:
        core_a = float(np.nansum(np.abs(atom_a[list(r1_atoms_a)])))
    else:
        core_a = 0.0
    if r1_atoms_i and atom_i.size:
        core_i = float(np.nansum(np.abs(atom_i[list(r1_atoms_i)])))
    else:
        core_i = 0.0
    distant_a = float(np.sum(np.abs(atom_a))) - core_a if atom_a.size else 0.0
    distant_i = float(np.sum(np.abs(atom_i))) - core_i if atom_i.size else 0.0
    core_delta_val = core_a + core_i
    distant_delta_val = distant_a + distant_i

    # Token-to-atom recall on top-10% tokens (span mapping implies near-1.0 if molecule has atoms)
    def recall_top(frac: float, pos: np.ndarray, bins: List[List[int]]) -> float:
        if not bins or pos.size == 0:
            return 0.0
        idx = topk_indices_by_abs(pos, frac)
        covered = 0
        for p in idx:
            # find if this position belongs to any bin
            found = any(p in b for b in bins)
            covered += 1 if found else 0
        return covered / float(len(idx)) if idx else 0.0

    token_to_atom_recall = float(0.5 * (recall_top(0.10, pos_a_norm, bins_a) + recall_top(0.10, pos_i_norm, bins_i)))

    # Bridge: FR context/edit masses and topk changed (compat layer only)
    fr_bridge = {
        'edit_mass_fr': {k: group_scores_a.get(k, 0.0) if k in changed else 0.0 for k in signed_delta.keys()},
        'context_mass_fr': {k: group_scores_a.get(k, 0.0) if k in common else 0.0 for k in signed_delta.keys()},
        'topk_changed': [k for k, _ in sorted(delta_abs.items(), key=lambda kv: -kv[1])[:10]]
    }

    # Functional-group occlusion sanity
    group_sanity = np.nan
    try:
        # Build atom->positions bins per molecule (for span mapping)
        bins_a = span_positions_for_atoms(seq_len, len(atom_a)) if atom_a.size else []
        bins_i = span_positions_for_atoms(seq_len, len(atom_i)) if atom_i.size else []

        # Choose top changed groups by |signed delta|
        top_changed = sorted([(g, v) for (g, v) in signed_delta.items() if g in changed], key=lambda kv: -abs(kv[1]))[:5]

        def occlude_group_and_check(x_tensor: torch.Tensor, group_name: str, atoms: List[int], bins: List[List[int]], expected_sign: float) -> Optional[bool]:
            if not atoms or not bins:
                return None
            idxs = sorted({p for a in atoms if a < len(bins) for p in bins[a]})
            if not idxs:
                return None
            with torch.no_grad():
                base = torch.sigmoid(model(x_tensor)).item()
                x_occ = x_tensor.clone()
                x_occ[:, idxs, :] = 0.0
                x_occ[:, idxs, 0] = 1.0
                prob = torch.sigmoid(model(x_occ)).item()
            # expected_sign > 0 means group contributes positively; removing should decrease prob
            if expected_sign > 0:
                return prob < base
            elif expected_sign < 0:
                return prob > base
            return None

        checks = []
        for g, sgn in top_changed:
            # Active molecule check if group present
            if presence_a.get(g, 0) == 1:
                checks.append(occlude_group_and_check(x_a, g, group_atoms_a.get(g, []), bins_a, +1.0 if group_scores_a.get(g, 0.0) > 0 else -1.0))
            # Inactive molecule check if group present
            if presence_i.get(g, 0) == 1:
                checks.append(occlude_group_and_check(x_i, g, group_atoms_i.get(g, []), bins_i, +1.0 if group_scores_i.get(g, 0.0) > 0 else -1.0))

        checks = [c for c in checks if c is not None]
        if checks:
            group_sanity = bool(np.mean(checks) >= 0.5)
    except Exception:
        group_sanity = np.nan

    # Optional dump of per-pair heatmaps
    if dump_dir:
        try:
            os.makedirs(dump_dir, exist_ok=True)
            dump = {
                c1: {
                    'token_attr': pos_a_norm.tolist(),
                    'atom_attr': atom_a.tolist() if atom_a.size else []
                },
                c2: {
                    'token_attr': pos_i_norm.tolist(),
                    'atom_attr': atom_i.tolist() if atom_i.size else []
                }
            }
            dump_name = f"dump_{str(active_id) or 'A'}_{str(inactive_id) or 'B'}.json"
            with open(os.path.join(dump_dir, dump_name), 'w') as f:
                json.dump(dump, f)
        except Exception:
            pass

    # Assemble row
    checkpoint = "output_assembly"
    # # print(f"[{checkpoint}] Assembling output dictionary...")
    import time as _time
    _t0 = _time.perf_counter()

    # Simple fix: use compound1/compound2 for non-cliff, active/inactive for cliff
    if pair_type == 'noncliff':
        c1, c2 = 'compound1', 'compound2'
    else:
        c1, c2 = 'active', 'inactive'

    out: Dict[str, Any] = {
        'pair_type': pair_type,
        'antibiotic_class': cls,
        f'{c1}_compound_id': active_id,
        f'{c2}_compound_id': inactive_id,
        f'{c1}_smiles': active_smiles,
        f'{c2}_smiles': inactive_smiles,
        'similarity': similarity,
        'model_type': 'CNN',
        'xai_method': xai_method_name,
        f'{c1}_target_ground_truth': int(active_target),
        f'{c2}_target_ground_truth': int(inactive_target),
        f'{c1}_group': active_group,
        f'{c2}_group': inactive_group,
        'group_combination': f"{active_group}_{inactive_group}",
        'model_id': model_id,
        'is_ensemble': bool(is_ensemble),
        'backbone_id': backbone_id,
        f'{c1}_pred_prob': active_prob,
        f'{c2}_pred_prob': inactive_prob,
        'prediction_difference': pred_diff,
        f'calibrated_prob_{c1}': active_prob if temperatures else np.nan,
        f'calibrated_prob_{c2}': inactive_prob if temperatures else np.nan,
        f'raw_prob_{c1}': raw_active_prob,
        f'raw_prob_{c2}': raw_inactive_prob,
        f'{c1}_class': active_cls,
        f'{c2}_class': inactive_cls,
        f'{c1}_classification': active_clf,
        f'{c2}_classification': inactive_clf,
        'pair_classification': pair_cls,
        'xai_visible': bool(len(group_scores_a) > 0 or len(group_scores_i) > 0),
        'cnn_visible': bool(len(x_a_np.shape) == 3 and x_a_np.shape[1] == seq_len),
        'propagation_index': propagation_index,
        'edit_concentration_index': edit_conc_index,
        'edit_mass': edit_mass,
        'context_mass': context_mass,
        'total_delta': float(edit_mass + context_mass),
        'signed_total_delta': float(np.sum(list(signed_delta.values()))) if signed_delta else 0.0,
        'abs_total_delta': float(np.sum(list(delta_abs.values()))) if delta_abs else 0.0,
        'core_delta': core_delta_val,
        'distant_delta': distant_delta_val,
        'edit_delta': edit_mass,
        f'feature_scores_{c1}': json.dumps(group_scores_a),
        f'feature_scores_{c2}': json.dumps(group_scores_i),
        'feature_delta_signed': json.dumps(signed_delta),
        'feature_delta_abs': json.dumps(delta_abs),
        f'feature_presence_{c1}': json.dumps(presence_a),
        f'feature_presence_{c2}': json.dumps(presence_i),
        'topk_features_changed': json.dumps(topk_changed(delta_abs, k=12)),
        'changed_functional_groups': json.dumps(changed),
        'common_features_count': int(len(common)),
        f'{c1}_features_count': int(sum(presence_a.values())) if presence_a else 0,
        f'{c2}_features_count': int(sum(presence_i.values())) if presence_i else 0,
        'feature_differences': int(len(changed)),
        'max_feature_diff': float(max(delta_abs.values()) if delta_abs else 0.0),
        'mean_feature_diff': float(np.mean(list(delta_abs.values())) if delta_abs else 0.0),
        'edit_driver_candidates': json.dumps([{ 'feature': k, 'direction': 'changed', 'support': delta_abs.get(k, 0.0)} for k in changed][:8]),
        'top_edit_driver': top_feature if top_feature else '',
        'edit_driver_support': top_support,
        'occlusion_sanity_pass': sanity_pass,
        'token_sanity_pass': sanity_pass,
        'group_sanity_pass': group_sanity,
        f'pos_attr_stats_{c1}': json.dumps({'min': float(np.min(pos_a)), 'max': float(np.max(pos_a)), 'mean': float(np.mean(pos_a)), 'std': float(np.std(pos_a))}),
        f'pos_attr_stats_{c2}': json.dumps({'min': float(np.min(pos_i)), 'max': float(np.max(pos_i)), 'mean': float(np.mean(pos_i)), 'std': float(np.std(pos_i))}),
        'mapping_warnings': mapping_warn,
        f'token_attr_{c1}': json.dumps(list(map(float, pos_a_norm.tolist()))),
        f'token_attr_{c2}': json.dumps(list(map(float, pos_i_norm.tolist()))),
        'map_mode': map_mode,
        'fg_norm': fg_norm,
        'model_ensemble_size': int(len(models)),
        'backbone_ckpt': backbone_ckpt or '',
        'patch_edit_overlap': float(patch_edit_overlap),
        'patch_edit_overlap_r1': float(patch_edit_overlap_r1),
        'patch_edit_overlap_r2': float(patch_edit_overlap_r2),
        'explanation_diameter_px': int(explanation_diameter_px),
        'saliency_entropy': float(saliency_entropy),
        'saliency_gini': float(saliency_gini),
        'ig_stability_r': ig_stability_r_val,
        'token_to_atom_recall': token_to_atom_recall,
        'fr_bridge_json': json.dumps(fr_bridge),
        'diagnostics': json.dumps({
            'device': str(next(models[0].parameters()).device) if models and any(p.requires_grad for p in models[0].parameters()) else 'cpu',
            'xai_method': 'Occlusion',
            f'gini_{c1}': float(G_a),
            f'gini_{c2}': float(G_i),
            f'entropy_{c1}': float(H_a),
            f'entropy_{c2}': float(H_i),
            'ensemble_size': int(len(models)),
            'seq_len': int(seq_len),
            'vocab_size': int(preproc.vocab_size),
            'rng_seed': int(42),
            'runtime_sec': float(_time.perf_counter() - _t0),
        }),
    }

    # Visualization data
    # # print(f"Preparing visualization data...")
    try:
        if atom_a.size > 0 and atom_i.size > 0:
            import numpy as _np
            thr_a = float(_np.percentile(_np.abs(atom_a), max(0, min(100, int(viz_percentile))))) if atom_a.size else 0.1
            thr_i = float(_np.percentile(_np.abs(atom_i), max(0, min(100, int(viz_percentile))))) if atom_i.size else 0.1
            viz_a = prepare_visualization_data_cnn(atom_a, threshold=thr_a)
            viz_i = prepare_visualization_data_cnn(atom_i, threshold=thr_i)

            if 'error' in viz_a or 'error' in viz_i:
                print(f" Viz prep returned errors: {viz_a.get('error')}, {viz_i.get('error')}")
                raise ValueError("Visualization preparation failed")

            out.update({
                f'viz_{c1}_atom_attr': json.dumps(viz_a.get('atom_attributions', [])),
                f'viz_{c1}_atom_colors': json.dumps(viz_a.get('atom_colors', [])),
                f'viz_{c1}_positive_atoms': json.dumps(viz_a.get('positive_atoms', [])),
                f'viz_{c1}_negative_atoms': json.dumps(viz_a.get('negative_atoms', [])),
                f'viz_{c2}_atom_attr': json.dumps(viz_i.get('atom_attributions', [])),
                f'viz_{c2}_atom_colors': json.dumps(viz_i.get('atom_colors', [])),
                f'viz_{c2}_positive_atoms': json.dumps(viz_i.get('positive_atoms', [])),
                f'viz_{c2}_negative_atoms': json.dumps(viz_i.get('negative_atoms', [])),
            })
            # # print(f" Visualization data added")
        else:
            print(f" Empty atom arrays ({atom_a.size}, {atom_i.size}), using empty viz")
            raise ValueError("Empty atom arrays")

    except Exception as e:
        print(f" Visualization error: {e}")
        # Empty but valid JSON
        out.update({
            f'viz_{c1}_atom_attr': json.dumps([]),
            f'viz_{c1}_atom_colors': json.dumps([]),
            f'viz_{c1}_positive_atoms': json.dumps([]),
            f'viz_{c1}_negative_atoms': json.dumps([]),
            f'viz_{c2}_atom_attr': json.dumps([]),
            f'viz_{c2}_atom_colors': json.dumps([]),
            f'viz_{c2}_positive_atoms': json.dumps([]),
            f'viz_{c2}_negative_atoms': json.dumps([]),
        })
        print(" Wrote empty visualization fields to CSV (no valid atom mapping).")

    # Prediction–attribution alignment and ensemble variance summary
    try:
        def _pred_attr_alignment(prob: float, atom_attr: np.ndarray) -> Tuple[bool, float]:
            arr = np.asarray(atom_attr, dtype=float)
            if arr.size == 0 or not np.isfinite(prob):
                return False, 0.0
            pos_frac = float(np.mean(arr > 0))
            align = bool((prob >= threshold and pos_frac >= 0.5) or (prob < threshold and pos_frac <= 0.5))
            return align, pos_frac
        def weighted_pred_attr_alignment(prob: float, atom_attr: np.ndarray, thr: float = None):
            arr = np.asarray(atom_attr, dtype=float)
            if arr.size == 0 or not np.isfinite(prob):
                return False, 0.0, {'method': 'weighted', 'valid': False}
            w = np.abs(arr)
            tw = float(w.sum())
            if tw < 1e-12:
                return False, 0.0, {'method': 'weighted', 'valid': False, 'reason': 'zero_weight'}
            pos_w = float((w * (arr > 0)).sum())
            frac = pos_w / tw
            th = float(threshold if thr is None else thr)
            align = bool((prob >= th and frac >= 0.5) or (prob < th and frac < 0.5))
            metrics = {
                'method': 'weighted', 'valid': True,
                'total_weight': tw, 'positive_weight': pos_w,
                'negative_weight': tw - pos_w, 'weighted_pos_fraction': frac,
                'num_atoms': int(arr.size), 'num_positive_atoms': int((arr > 0).sum()),
                'mean_abs_attribution': float(np.mean(w))
            }
            return align, frac, metrics
        align_a, posfrac_a = _pred_attr_alignment(active_prob, atom_a)
        align_i, posfrac_i = _pred_attr_alignment(inactive_prob, atom_i)
        w_align_a, w_frac_a, w_metrics_a = weighted_pred_attr_alignment(active_prob, atom_a)
        w_align_i, w_frac_i, w_metrics_i = weighted_pred_attr_alignment(inactive_prob, atom_i)
        out.update({
            f'pred_attr_alignment_{c1}': bool(align_a),
            f'pred_attr_alignment_{c2}': bool(align_i),
            'pred_attr_alignment_pair': bool(align_a and align_i),
            f'pred_attr_positive_fraction_{c1}': float(posfrac_a),
            f'pred_attr_positive_fraction_{c2}': float(posfrac_i),
            f'pred_attr_mismatch_{c1}': bool(not align_a),
            f'pred_attr_mismatch_{c2}': bool(not align_i),
            'pred_attr_mismatch_pair': bool(not (align_a and align_i)),
            f'pred_attr_alignment_{c1}_weighted': bool(w_align_a),
            f'pred_attr_alignment_{c2}_weighted': bool(w_align_i),
            'pred_attr_alignment_pair_weighted': bool(w_align_a and w_align_i),
            f'pred_attr_positive_fraction_{c1}_weighted': float(w_frac_a),
            f'pred_attr_positive_fraction_{c2}_weighted': float(w_frac_i),
            f'pred_attr_mismatch_{c1}_weighted': bool(not w_align_a),
            f'pred_attr_mismatch_{c2}_weighted': bool(not w_align_i),
            'pred_attr_mismatch_pair_weighted': bool(not (w_align_a and w_align_i)),
            f'weighted_alignment_metrics_{c1}': json.dumps(w_metrics_a),
            f'weighted_alignment_metrics_{c2}': json.dumps(w_metrics_i),
            'attribution_std_across_models': float(attribution_std_across_models),
        })
    except Exception:
        out.update({
            f'pred_attr_alignment_{c1}': False,
            f'pred_attr_alignment_{c2}': False,
            'pred_attr_alignment_pair': False,
            f'pred_attr_positive_fraction_{c1}': 0.0,
            f'pred_attr_positive_fraction_{c2}': 0.0,
            f'pred_attr_mismatch_{c1}': False,
            f'pred_attr_mismatch_{c2}': False,
            'pred_attr_mismatch_pair': False,
            'attribution_std_across_models': float('nan'),
        })

    # Murcko substructure aggregation using existing atom scores
    checkpoint = "murcko_aggregation"
    # # print(f"[{checkpoint}] Aggregating atom scores by Murcko substructures...")
    try:
        # Detailed diagnostics for atom arrays
        # # print(f"  Active atom_a: shape={atom_a.shape}, type={type(atom_a)}, dtype={atom_a.dtype if hasattr(atom_a, 'dtype') else 'N/A'}")
        # # print(f"  Active atom_a sample: {atom_a[:5] if atom_a.size > 0 else 'EMPTY ARRAY'}")
        # # print(f"  Inactive atom_i: shape={atom_i.shape}, type={type(atom_i)}, dtype={atom_i.dtype if hasattr(atom_i, 'dtype') else 'N/A'}")
        # # print(f"  Inactive atom_i sample: {atom_i[:5] if atom_i.size > 0 else 'EMPTY ARRAY'}")

        # Use existing atom scores from occlusion attribution (computed earlier)
        murcko_active = murcko_substructure_aggregation(str(active_smiles), atom_a)
        murcko_inactive = murcko_substructure_aggregation(str(inactive_smiles), atom_i)

        # Add atom-level attribution columns
        out.update({
            f'murcko_atom_scores_{c1}': json.dumps(murcko_active.get('atom_scores', [])),
            f'murcko_atom_scores_{c2}': json.dumps(murcko_inactive.get('atom_scores', [])),
            f'murcko_positive_contributors_{c1}': json.dumps(murcko_active.get('positive_contributors', [])),
            f'murcko_negative_contributors_{c1}': json.dumps(murcko_active.get('negative_contributors', [])),
            f'murcko_positive_contributors_{c2}': json.dumps(murcko_inactive.get('positive_contributors', [])),
            f'murcko_negative_contributors_{c2}': json.dumps(murcko_inactive.get('negative_contributors', [])),
            f'murcko_num_substructures_{c1}': int(murcko_active.get('num_substructures', 0)),
            f'murcko_num_substructures_{c2}': int(murcko_inactive.get('num_substructures', 0)),
            f'murcko_avg_substructure_size_{c1}': float(murcko_active.get('avg_substructure_size', 0.0)),
            f'murcko_avg_substructure_size_{c2}': float(murcko_inactive.get('avg_substructure_size', 0.0)),
            f'murcko_total_atoms_{c1}': int(murcko_active.get('total_atoms', 0)),
            f'murcko_total_atoms_{c2}': int(murcko_inactive.get('total_atoms', 0)),
        })

        # ===================================================================
        # Calculate delta_pharm_signed for CSPD_signed metric (occlusion)
        # ===================================================================
        try:
            murcko_scores_active = np.array(murcko_active.get('atom_scores', []))
            murcko_scores_inactive = np.array(murcko_inactive.get('atom_scores', []))

            if len(murcko_scores_active) > 0 and len(murcko_scores_inactive) > 0:
                # Sum signed attribution scores over all atoms
                pharm_attr_active = float(np.sum(murcko_scores_active))
                pharm_attr_inactive = float(np.sum(murcko_scores_inactive))
                # Signed difference for correlation with prediction difference
                delta_pharm_signed = pharm_attr_active - pharm_attr_inactive
            else:
                delta_pharm_signed = float('nan')
        except Exception:
            delta_pharm_signed = float('nan')

        out['delta_pharm_signed'] = delta_pharm_signed
        print(f"  delta_pharm_signed calculated: {delta_pharm_signed:.4f}" if not np.isnan(delta_pharm_signed) else "  delta_pharm_signed: N/A")

        # Pharmacophore recognition using Murcko positive contributors
        if pharm_json_path and os.path.exists(pharm_json_path):
            pharm_active = pharmacophore_recognition_with_atoms(
                str(active_smiles), str(cls),
                murcko_active.get('positive_contributors', []),
                pharm_json_path
            )
            pharm_inactive = pharmacophore_recognition_with_atoms(
                str(inactive_smiles), str(cls),
                murcko_inactive.get('positive_contributors', []),
                pharm_json_path
            )

            if 'error' not in pharm_active and 'error' not in pharm_inactive:
                out.update({
                    f'murcko_pharm_recognized_{c1}': json.dumps(pharm_active.get('recognized_patterns', [])),
                    f'murcko_pharm_missed_{c1}': json.dumps(pharm_active.get('missed_patterns', [])),
                    f'murcko_pharm_recognition_score_{c1}': float(pharm_active.get('recognition_score', 0.0)),
                    f'murcko_pharm_recognition_rate_{c1}': float(pharm_active.get('recognition_rate', 0.0)),
                    f'murcko_pharm_recognized_{c2}': json.dumps(pharm_inactive.get('recognized_patterns', [])),
                    f'murcko_pharm_missed_{c2}': json.dumps(pharm_inactive.get('missed_patterns', [])),
                    f'murcko_pharm_recognition_score_{c2}': float(pharm_inactive.get('recognition_score', 0.0)),
                    f'murcko_pharm_recognition_rate_{c2}': float(pharm_inactive.get('recognition_rate', 0.0)),
                    'murcko_pharm_recognition_score_pair': float(np.mean([
                        pharm_active.get('recognition_score', 0.0),
                        pharm_inactive.get('recognition_score', 0.0)
                    ])),
                    'murcko_delta_core_align': float(
                        pharm_active.get('recognition_score', 0.0) -
                        pharm_inactive.get('recognition_score', 0.0)
                    ),
                })
                # # print(f" Murcko pharmacophore recognition complete")

        # # print(f" Murcko attribution complete")

    except Exception as e:
        print(f" Murcko aggregation error: {e}")
        import traceback
        traceback.print_exc()
        # Add empty Murcko columns
        out.update({
            f'murcko_atom_scores_{c1}': json.dumps([]),
            f'murcko_atom_scores_{c2}': json.dumps([]),
            f'murcko_positive_contributors_{c1}': json.dumps([]),
            f'murcko_negative_contributors_{c1}': json.dumps([]),
            f'murcko_positive_contributors_{c2}': json.dumps([]),
            f'murcko_negative_contributors_{c2}': json.dumps([]),
            f'murcko_num_substructures_{c1}': 0,
            f'murcko_num_substructures_{c2}': 0,
            f'murcko_avg_substructure_size_{c1}': 0.0,
            f'murcko_avg_substructure_size_{c2}': 0.0,
            f'murcko_total_atoms_{c1}': 0,
            f'murcko_total_atoms_{c2}': 0,
            f'murcko_pharm_recognized_{c1}': json.dumps([]),
            f'murcko_pharm_missed_{c1}': json.dumps([]),
            f'murcko_pharm_recognition_score_{c1}': 0.0,
            f'murcko_pharm_recognition_rate_{c1}': 0.0,
            f'murcko_pharm_recognized_{c2}': json.dumps([]),
            f'murcko_pharm_missed_{c2}': json.dumps([]),
            f'murcko_pharm_recognition_score_{c2}': 0.0,
            f'murcko_pharm_recognition_rate_{c2}': 0.0,
            'murcko_pharm_recognition_score_pair': 0.0,
            'murcko_delta_core_align': 0.0,
            'delta_pharm_signed': float('nan'),
        })

    # Pharmacophore validation
    # # print(f"Starting pharmacophore validation for class='{cls}'...")
    try:
        # Prerequisite checks
        if atom_a.size == 0 or atom_i.size == 0:
            print(f" Empty atoms, skipping pharmacophore")
            raise ValueError("Empty atom arrays")

        if not cls or str(cls).lower() in ['nan', 'none', '']:
            print(f" Invalid class '{cls}', skipping pharmacophore")
            raise ValueError("Invalid class")

        # Validate
        pa = validate_pharmacophore_recognition(str(active_smiles), str(cls), atom_a, pharmacophore_json_path=(pharm_json_path or 'pharmacophore.json'))
        pi = validate_pharmacophore_recognition(str(inactive_smiles), str(cls), atom_i, pharmacophore_json_path=(pharm_json_path or 'pharmacophore.json'))

        if 'error' in pa or 'error' in pi:
            print(f" Validation errors: {pa.get('error')}, {pi.get('error')}")
            raise ValueError("Validation failed")

        out.update({
            f'pharm_recognized_{c1}': json.dumps(pa.get('recognized_features', [])),
            f'pharm_missed_{c1}': json.dumps(pa.get('missed_features', [])),
            f'pharm_recognition_score_{c1}': float(pa.get('overall_recognition_score', 0.0)),
            f'pharm_recognition_rate_{c1}': float(pa.get('recognition_rate', 0.0)),
            f'pharm_recognized_{c2}': json.dumps(pi.get('recognized_features', [])),
            f'pharm_missed_{c2}': json.dumps(pi.get('missed_features', [])),
            f'pharm_recognition_score_{c2}': float(pi.get('overall_recognition_score', 0.0)),
            f'pharm_recognition_rate_{c2}': float(pi.get('recognition_rate', 0.0)),
            'pharm_recognition_score_pair': float(np.nanmean([
                pa.get('overall_recognition_score', 0.0),
                pi.get('overall_recognition_score', 0.0)
            ])),
        })
        # Pharmacophore Consistency Score
        try:
            ps_a = float(pa.get('overall_recognition_score', 0.0))
            ps_i = float(pi.get('overall_recognition_score', 0.0))
            denom = ps_a + ps_i
            if denom > 1e-6:
                pharm_consistency = 1.0 - abs(ps_a - ps_i) / denom
            else:
                pharm_consistency = 0.0
            out['pharmacophore_consistency_score'] = float(pharm_consistency)
            out['pharmacophore_inconsistent_flag'] = bool(abs(ps_a - ps_i) > 0.3)
        except Exception:
            out['pharmacophore_consistency_score'] = float('nan')
            out['pharmacophore_inconsistent_flag'] = False
        # # print(f" Pharmacophore validation complete")

    except Exception as e:
        print(f" Pharmacophore error: {e}")
        import traceback
        traceback.print_exc()
        # Empty but valid results
        out.update({
            f'pharm_recognized_{c1}': json.dumps([]),
            f'pharm_missed_{c1}': json.dumps([]),
            f'pharm_recognition_score_{c1}': 0.0,
            f'pharm_recognition_rate_{c1}': 0.0,
            f'pharm_recognized_{c2}': json.dumps([]),
            f'pharm_missed_{c2}': json.dumps([]),
            f'pharm_recognition_score_{c2}': 0.0,
            f'pharm_recognition_rate_{c2}': 0.0,
            'pharm_recognition_score_pair': 0.0,
        })
        print(" Wrote default empty pharmacophore fields to CSV (validation skipped/failed).")

    # Additional pharmacophore scoring if provided
    if ph_smarts:
        try:
            # Build a class-specific flat SMARTS dict if JSON-style provided
            flat_smarts: Dict[str, str] = {}
            if isinstance(ph_smarts, dict) and cls and isinstance(ph_smarts.get(cls, {}), dict):
                sec = ph_smarts.get(cls, {})
                for cat in ['required_any', 'loose_required_any', 'important_any', 'optional_any']:
                    for feat in (sec.get(cat) or []):
                        name = feat.get('name'); smt = feat.get('smarts')
                        if name and smt:
                            flat_smarts[name] = smt
            else:
                # Assume already flat mapping of name->SMARTS
                flat_smarts = {k: v for k, v in (ph_smarts or {}).items() if isinstance(v, str)}

            if not flat_smarts:
                print(f" WARNING: No pharmacophore SMARTS found for class '{cls}' (feature scoring)")

            ph_a, ph_pa, _ = smartsmatch_groups(str(active_smiles), atom_a, flat_smarts, agg=fg_norm)
            ph_i, ph_pi, _ = smartsmatch_groups(str(inactive_smiles), atom_i, flat_smarts, agg=fg_norm)
            ph_delta = {k: abs(ph_a.get(k, 0.0) - ph_i.get(k, 0.0)) for k in set(ph_a) | set(ph_i)}
            out[f'pharm_feature_scores_{c1}'] = json.dumps(ph_a)
            out[f'pharm_feature_scores_{c2}'] = json.dumps(ph_i)
            out['pharm_feature_delta_abs'] = json.dumps(ph_delta)
            out['pharm_changed_features'] = json.dumps([k for k in set(ph_pa) | set(ph_pi) if ph_pa.get(k,0)!=ph_pi.get(k,0)])
        except Exception:
            out[f'pharm_feature_scores_{c1}'] = json.dumps({})
            out[f'pharm_feature_scores_{c2}'] = json.dumps({})
            out['pharm_feature_delta_abs'] = json.dumps({})
            out['pharm_changed_features'] = json.dumps([])

    # 85-FG comparability view and full-fragment alias
    try:
        out[f'fg85_feature_scores_{c1}'] = json.dumps(group_scores_a)
        out[f'fg85_feature_scores_{c2}'] = json.dumps(group_scores_i)
        out['fg85_feature_delta_abs'] = json.dumps(delta_abs)
        out['fg85_changed_features'] = json.dumps(changed)
    except Exception:
        out[f'fg85_feature_scores_{c1}'] = json.dumps({})
        out[f'fg85_feature_scores_{c2}'] = json.dumps({})
        out['fg85_feature_delta_abs'] = json.dumps({})
        out['fg85_changed_features'] = json.dumps([])
    try:
        out[f'full_fragment_scores_{c1}'] = out.get(f'pharm_feature_scores_{c1}', json.dumps({}))
        out[f'full_fragment_scores_{c2}'] = out.get(f'pharm_feature_scores_{c2}', json.dumps({}))
        out['full_fragment_delta_abs'] = out.get('pharm_feature_delta_abs', json.dumps({}))
        out['full_fragment_changed_features'] = out.get('pharm_changed_features', json.dumps([]))
    except Exception:
        out[f'full_fragment_scores_{c1}'] = json.dumps({})
        out[f'full_fragment_scores_{c2}'] = json.dumps({})
        out['full_fragment_delta_abs'] = json.dumps({})
        out['full_fragment_changed_features'] = json.dumps([])

    # Core alignment (Δcore_align)
    try:
        core_a = set(core_atoms_from_pharm(str(active_smiles), str(cls), pharm_json_path or 'pharmacophore.json'))
        core_i = set(core_atoms_from_pharm(str(inactive_smiles), str(cls), pharm_json_path or 'pharmacophore.json'))
        top_a = set(top_mass_atoms(atom_a, pct=float(top_mass_pct))) if atom_a.size else set()
        top_i = set(top_mass_atoms(atom_i, pct=float(top_mass_pct))) if atom_i.size else set()
        align_a = float(len(core_a & top_a)) / float(len(core_a)) if core_a else 0.0
        align_i = float(len(core_i & top_i)) / float(len(core_i)) if core_i else 0.0
        out[f'core_align_{c1}'] = float(align_a) if np.isfinite(align_a) else 0.0
        out[f'core_align_{c2}'] = float(align_i) if np.isfinite(align_i) else 0.0
        # Diagnostics for invalid delta or masses
        try:
            denom_mass = float(edit_mass + context_mass)
            delta_ratio = float(edit_mass / (denom_mass + 1e-12)) if denom_mass > 0 else 0.0
            delta_diff = float(align_a - align_i)
            if (edit_mass < 0) or (context_mass < 0) or (delta_ratio < 0) or (delta_ratio > 1) or (delta_diff < 0) or (delta_diff > 1):
                import sys
                print(f"\n{'='*60}", file=sys.stderr)
                print(f"CORRUPTION: {str(active_id)}_{str(inactive_id)}", file=sys.stderr)
                print(f"  edit_mass={edit_mass}, context_mass={context_mass}", file=sys.stderr)
                # Core masses (atom-space):
                try:
                    # Use previously computed atom-level core/distant masses if available
                    core_a_mass = float(np.sum(np.abs(atom_a[atoms_within_radius(rdmolops.GetDistanceMatrix(Chem.MolFromSmiles(str(active_smiles))).astype(int), [int(x) for x in []], 1)]))) if False else float('nan')
                except Exception:
                    core_a_mass = float('nan')
                try:
                    core_i_mass = float(np.sum(np.abs(atom_i[atoms_within_radius(rdmolops.GetDistanceMatrix(Chem.MolFromSmiles(str(inactive_smiles))).astype(int), [int(x) for x in []], 1)]))) if False else float('nan')
                except Exception:
                    core_i_mass = float('nan')
                print(f"  core_a_mass={core_a_mass}, core_i_mass={core_i_mass}", file=sys.stderr)
                print(f"  delta_core_align(diff)={delta_diff}, delta_core_align(ratio)={delta_ratio}", file=sys.stderr)
                print(f"{'='*60}\n", file=sys.stderr)
        except Exception:
            pass
        denom = float(edit_mass + context_mass)
        out['delta_core_align'] = float(edit_mass / (denom + 1e-12)) if denom > 0 else 0.0
        out['core_align_diff'] = float(align_a - align_i)
    except Exception as e:
        out[f'core_align_{c1}'] = 0.0
        out[f'core_align_{c2}'] = 0.0
        out['delta_core_align'] = 0.0

    # Add fidelity and flip metrics
    out.update({k: float(v) for k, v in fidelity.items()})
    out.update({k: bool(v) for k, v in flips.items()})

    # Substructure occlusion JSON (Murcko/rings) using token PAD masking
    try:
        def substruct_json(smiles: str, x_input: torch.Tensor, base_prob: float) -> List[Dict[str, Any]]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            subs = ring_or_murcko_substructures(smiles)

            # Build atom->position mapping using token-aware approach
            n_atoms = mol.GetNumAtoms()
            from CNN_model.cnn_token_mapper import tokenize_smiles, is_atom_token, build_token_to_atoms

            # Get token-to-atom mapping
            token_mapping = build_token_to_atoms(smiles)

            # If token mapping failed, fall back to span mapping
            if token_mapping is None:
                bins = span_positions_for_atoms(seq_len, n_atoms)
            else:
                # Build atom-to-position mapping from token-to-atom mapping
                # token_mapping[i] contains list of atoms for token at position i
                bins = [[] for _ in range(n_atoms)]
                for pos_idx, atom_list in enumerate(token_mapping):
                    if pos_idx >= seq_len:
                        break
                    for atom_idx in atom_list:
                        if 0 <= atom_idx < n_atoms:
                            bins[atom_idx].append(pos_idx)

            out_list: List[Dict[str, Any]] = []
            for name, atoms in subs.items():
                try:
                    if not atoms:
                        continue
                    # Collect positions for atoms
                    pos = sorted({p for a in atoms if a < len(bins) for p in bins[a]})
                    if not pos:
                        continue
                    x_mask = x_input.clone()
                    x_mask[:, pos, :] = 0.0
                    x_mask[:, pos, 0] = 1.0
                    p = predict_prob(models, x_mask, temperatures)
                    delta = float(base_prob - p)
                    per_atom = float(delta / max(1, len(atoms)))
                    impact = 'Minimal impact on activity'
                    if delta > 0.1:
                        impact = 'Increases probability of activity'
                    elif delta < -0.1:
                        impact = 'Decreases probability of activity'
                    frag = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, kekuleSmiles=True)
                    out_list.append({
                        'substructure': frag,
                        'atoms': atoms,
                        'attribution': delta,
                        'attribution_per_atom': per_atom,
                        'impact': impact,
                    })
                except Exception:
                    continue
            out_list.sort(key=lambda d: abs(d.get('attribution', 0.0)), reverse=True)
            return out_list

        base_pa = raw_active_prob if temperatures else active_prob
        base_pi = raw_inactive_prob if temperatures else inactive_prob
        # Name substructure attributions following pair type convention
        # For cliffs: active/inactive; for non-cliffs: compound1/compound2
        sub_key1 = 'active' if pair_type == 'cliff' else 'compound1'
        sub_key2 = 'inactive' if pair_type == 'cliff' else 'compound2'
        sub_a = substruct_json(str(active_smiles), x_a, base_pa)
        sub_i = substruct_json(str(inactive_smiles), x_i, base_pi)
        out[f'substruct_attr_{sub_key1}'] = json.dumps(sub_a)
        out[f'substruct_attr_{sub_key2}'] = json.dumps(sub_i)
        # Derive positive/negative/neutral substructure fragment lists for visualization
        def split_substructs(items: List[Dict[str, Any]], thr: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
            pos = []
            neg = []
            neu = []
            for it in items or []:
                try:
                    val = float(it.get('attribution', 0.0))
                    frag = str(it.get('substructure', ''))
                    if not frag:
                        continue
                    if val >= thr:
                        pos.append(frag)
                    elif val <= -thr:
                        neg.append(frag)
                    else:
                        neu.append(frag)
                except Exception:
                    continue
            # Deduplicate preserving order
            def dedup(seq):
                seen = set(); outl = []
                for s in seq:
                    if s not in seen:
                        seen.add(s); outl.append(s)
                return outl
            return dedup(pos), dedup(neg), dedup(neu)
        pos_a, neg_a, neu_a = split_substructs(sub_a, thr=0.1)
        pos_i, neg_i, neu_i = split_substructs(sub_i, thr=0.1)
        out[f'pos_substructs_{sub_key1}'] = json.dumps(pos_a)
        out[f'neg_substructs_{sub_key1}'] = json.dumps(neg_a)
        out[f'neutral_substructs_{sub_key1}'] = json.dumps(neu_a)
        out[f'pos_substructs_{sub_key2}'] = json.dumps(pos_i)
        out[f'neg_substructs_{sub_key2}'] = json.dumps(neg_i)
        out[f'neutral_substructs_{sub_key2}'] = json.dumps(neu_i)
        # # print(" Substructure occlusion JSON added (CNN)")
    except Exception as e:
        print(f" Substructure occlusion JSON error (CNN): {e}")
        out['substruct_attr_active'] = json.dumps([])
        out['substruct_attr_inactive'] = json.dumps([])
        out['pos_substructs_active'] = json.dumps([])
        out['neg_substructs_active'] = json.dumps([])
        out['neutral_substructs_active'] = json.dumps([])
        out['pos_substructs_inactive'] = json.dumps([])
        out['neg_substructs_inactive'] = json.dumps([])
        out['neutral_substructs_inactive'] = json.dumps([])

    # Sanity gates
    mean_drop = np.nanmean([
        out.get('fidelity_drop_remove_topk_k5', np.nan),
        out.get('fidelity_drop_remove_topk_k10', np.nan),
        out.get('fidelity_drop_remove_topk_k20', np.nan),
    ])
    out['stability_flag'] = bool(ig_stability_r_val < 0.9)
    out['fidelity_flag'] = bool(mean_drop > 0)

    # Gate xai_visible by token->atom recall and pharmacophore threshold
    try:
        pharm_ok = (out.get('pharm_recognition_score_active', 0.0) >= float(pharm_threshold)) or \
                   (out.get('pharm_recognition_score_inactive', 0.0) >= float(pharm_threshold))
    except Exception:
        pharm_ok = True
    recall_ok = bool(token_to_atom_recall >= float(token_atom_recall_min))
    # Disallow hard mapping warnings
    hard_warn = mapping_warn in ('mapping_failed', 'fallback_span', 'mapping_truncated')
    # No class flip under keep_topk_k5
    no_flip_keep_k5 = not bool(out.get('class_flip_keep_topk_k5', False))
    out['xai_visible'] = bool(out.get('xai_visible', False) and recall_ok and not hard_warn and out.get('fidelity_flag', False) and no_flip_keep_k5 and pharm_ok)
    if not recall_ok:
        out['mapping_warnings'] = (out.get('mapping_warnings','') + ';recall_low').strip(';')

    if not out['xai_visible']:
        out['xai_failure_reason'] = mapping_warn or 'no_groups'

    return out


def load_models(ckpts: List[str], vocab_size: int, seq_len: int, device: Optional[torch.device] = None) -> List[SMILESCNNModel]:
    models: List[SMILESCNNModel] = []
    print(f"\n🔧 Attempting to load {len(ckpts)} checkpoint(s)...", flush=True)
    for i, path in enumerate(ckpts, 1):
        print(f"  [{i}/{len(ckpts)}] Loading: {path}", flush=True)
        if not os.path.exists(path):
            print(f"  ❌ File not found: {path}", flush=True)
            continue
        try:
            m = SMILESCNNModel.load_from_checkpoint(
                path,
                vocab_size=vocab_size,
                sequence_length=seq_len,
            )
            m.eval()
            # Move to requested/best available device
            dev = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            m.to(dev)
            models.append(m)
            print(f"  ✅ Successfully loaded to {dev}", flush=True)
        except Exception as e:
            print(f"  ❌ Failed: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print(f"\n📊 Total models loaded: {len(models)}/{len(ckpts)}", flush=True)
    if not models:
        raise RuntimeError(f"❌ FATAL: No models loaded from {len(ckpts)} checkpoint(s). Check paths and compatibility.")
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activity_csv', default='activity_cliff_pairs.csv')
    parser.add_argument('--noncliff_csv', default='non_cliff_pairs.csv')
    parser.add_argument('--ckpt', action='append', required=False, help='Path(s) to .ckpt checkpoints (can repeat). If omitted, auto-discovers .ckpt under model_checkpoints/.')
    parser.add_argument('--seq_len', type=int, default=181)
    parser.add_argument('--map_mode', type=str, default='span', choices=['span', 'interpolate'])
    parser.add_argument('--fg_norm', type=str, default='sum', choices=['sum', 'mean', 'max'])
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default=None, help='cpu or cuda[:index]')
    parser.add_argument('--token_atom_recall_min', type=float, default=0.9)
    parser.add_argument('--pharm_threshold', type=float, default=0.2)
    parser.add_argument('--top_mass_pct', type=float, default=0.5)
    parser.add_argument('--ensemble_ckpts', type=str, default=None, help='Glob/pattern for ensemble checkpoints to average attribution over')
    parser.add_argument('--viz_percentile', type=int, default=80, help='Percentile (0-100) of |atom score| to color for viz')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--out_csv', default=os.path.join(script_dir, 'outputs', 'cnn_xai_balanced_full_detailed_ensemble.csv'))
    parser.add_argument('--full', action='store_true', help='Process full dataset (otherwise sample 2 per class per type)')
    parser.add_argument('--samples_per_class', type=int, default=2, help='Used when not --full')
    parser.add_argument('--calibration_json', type=str, default=None, help='Optional temperature scaling JSON')
    parser.add_argument('--dump_dir', type=str, default=None, help='Optional directory to dump per-pair heatmaps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--per_model', action='store_true', help='Run per-ckpt (individual) analysis; writes one CSV per checkpoint')
    parser.add_argument('--ensemble', action='store_true', help='Run ensemble analysis (averaged predictions + selected backbone for IG)')
    parser.add_argument('--backbone_index', type=int, default=0, help='Backbone checkpoint index for ensemble XAI')
    parser.add_argument('--limit_per_class', type=int, default=None, help='Optional row cap per class for speed')
    parser.add_argument('--pharmacophore_json', type=str, default=None, help='Optional JSON file of pharmacophore SMARTS mapping')
    parser.add_argument('--err_log', type=str, default=os.path.join(script_dir, 'outputs', 'cnn_xai_pairs.err'), help='Error log file path')
    args = parser.parse_args()

    # Ensure err_log directory exists
    os.makedirs(os.path.dirname(args.err_log) or '.', exist_ok=True)

    # Reproducibility
    try:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    # Debug: current working dir and pharmacophore json
    try:
        print(f"Current working directory: {os.getcwd()}")
        print(f"Pharmacophore JSON argument: {args.pharmacophore_json}")
    except Exception:
        pass
    pharm_path = args.pharmacophore_json or 'pharmacophore.json'
    if os.path.exists(pharm_path):
        print(f" Pharmacophore JSON found at: {os.path.abspath(pharm_path)}")
    else:
        print(f" Pharmacophore JSON NOT FOUND at: {os.path.abspath(pharm_path)}")
        try:
            print(f" Contents of current dir: {os.listdir('.')}")
        except Exception:
            pass

    # Preprocessor
    preproc = SMILESPreprocessor()
    vocab_size = preproc.vocab_size

    # Resolve checkpoints: use provided paths or auto-discover under model_checkpoints/
    ckpts = args.ckpt if args.ckpt else []
    # Optional ensemble glob override
    if args.ensemble_ckpts:
        import glob as _glob
        patt = list(_glob.glob(args.ensemble_ckpts))
        if patt:
            ckpts = patt
    if not ckpts:
        discovered: List[str] = []
        search_root = os.path.join(script_dir, 'model_checkpoints')
        if os.path.isdir(search_root):
            for root, _, files in os.walk(search_root):
                for f in files:
                    if f.lower().endswith('.ckpt'):
                        discovered.append(os.path.join(root, f))
        if discovered:
            # Keep deterministic ordering
            discovered.sort()
            ckpts = discovered
            print(f"Discovered {len(ckpts)} checkpoints under '{search_root}'.")
        else:
            raise RuntimeError("No checkpoints provided and none discovered under 'model_checkpoints/'. Use --ckpt.")

    # Load pharm SMARTS if provided
    global PH_SMARTS
    if args.pharmacophore_json and os.path.exists(args.pharmacophore_json):
        try:
            with open(args.pharmacophore_json, 'r') as f:
                PH_SMARTS = json.load(f)
            if not isinstance(PH_SMARTS, dict):
                PH_SMARTS = {}
        except Exception as e:
            print(f"Warning: failed to load pharmacophore JSON: {e}")
            PH_SMARTS = {}

    # Helper: single split run function
    def run_once_split(df_to_process: pd.DataFrame, pair_type_str: str, models: List[SMILESCNNModel],
                       model_id: str, is_ens: bool, backbone_id: str, out_csv: str, err_log_path: str):
        """Process a single split (cliff or non-cliff) and save results."""
        # Optional limit per class for speed
        if args.limit_per_class is not None:
            take = int(args.limit_per_class)
            cls_col = 'class' if 'class' in df_to_process.columns else 'antibiotic_class'
            if cls_col in df_to_process.columns:
                df_to_process = df_to_process.sort_values(cls_col).groupby(cls_col, group_keys=False).head(take)

        # TEST-mode sampling
        if not args.full:
            take = max(1, int(args.samples_per_class))
            cls_col = 'class' if 'class' in df_to_process.columns else 'antibiotic_class'
            if cls_col in df_to_process.columns:
                df_to_process = (df_to_process.sort_values(cls_col)
                                .groupby(cls_col, group_keys=False)
                                .head(take))
            print(f"TEST mode: using {len(df_to_process)} {pair_type_str} pairs (<= {take} per class)")

        rows: List[Dict[str, Any]] = []
        is_cliff_flag = (pair_type_str == 'cliff')

        # Process pairs
        for _, row in df_to_process.iterrows():
            try:
                out = process_pair_row(
                    row, models, preproc, args.seq_len, args.threshold,
                    args.map_mode, args.fg_norm,
                    model_id=model_id, is_ensemble=is_ens, backbone_id=backbone_id,
                    ph_smarts=PH_SMARTS if PH_SMARTS else None,
                    pharm_json_path=pharm_path,
                    token_atom_recall_min=float(args.token_atom_recall_min),
                    pharm_threshold=float(args.pharm_threshold),
                    top_mass_pct=float(args.top_mass_pct),
                    viz_percentile=int(args.viz_percentile),
                    temperatures=temperatures, dump_dir=args.dump_dir, backbone_ckpt=backbone_id
                )
                out['is_cliff'] = is_cliff_flag
                rows.append(out)
            except Exception as e:
                rows.append({'pair_type': pair_type_str, 'error': str(e), 'is_cliff': is_cliff_flag})

        out_df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        out_df.to_csv(out_csv, index=False, sep=',', quoting=csv.QUOTE_MINIMAL)
        # Sidecar Parquet for nested JSON fidelity
        try:
            out_df.to_parquet(os.path.splitext(out_csv)[0] + '.parquet', index=False)
        except Exception as e:
            print(f"Parquet save skipped: {e}")
        print(f"Saved: {out_csv} ({len(out_df)} rows)")

    # Helper: legacy combined run function (DEPRECATED)
    def run_once(models: List[SMILESCNNModel], model_id: str, is_ens: bool, backbone_id: str, out_csv: str):
        """DEPRECATED: Use run_once_split instead"""
        # Load dataframes
        df_cliff = pd.read_csv(args.activity_csv)
        df_non = pd.read_csv(args.noncliff_csv)
        if args.limit_per_class is not None:
            take = int(args.limit_per_class)
            if 'class' in df_cliff.columns:
                df_cliff = df_cliff.sort_values('class').groupby('class', group_keys=False).head(take)
            if 'class' in df_non.columns:
                df_non = df_non.sort_values('class').groupby('class', group_keys=False).head(take)
        # TEST-mode sampling
        if not args.full:
            take = max(1, int(args.samples_per_class))
            cls_col = 'class' if 'class' in df_cliff.columns else 'antibiotic_class'
            if cls_col in df_cliff.columns:
                df_cliff = (df_cliff.sort_values(cls_col)
                            .groupby(cls_col, group_keys=False)
                            .head(take))
            cls_col_non = 'class' if 'class' in df_non.columns else 'antibiotic_class'
            if cls_col_non in df_non.columns:
                df_non = (df_non.sort_values(cls_col_non)
                          .groupby(cls_col_non, group_keys=False)
                          .head(take))
            print(f"TEST mode: using {len(df_cliff)} cliff pairs and {len(df_non)} non-cliff pairs (<= {take} per class)")

        rows: List[Dict[str, Any]] = []
        # Process cliffs
        for idx, row in df_cliff.iterrows():
            try:
                out = process_pair_row(
                    row, models, preproc, args.seq_len, args.threshold,
                    args.map_mode, args.fg_norm,
                    model_id=model_id, is_ensemble=is_ens, backbone_id=backbone_id,
                    ph_smarts=PH_SMARTS if PH_SMARTS else None,
                    pharm_json_path=pharm_path,
                    token_atom_recall_min=float(args.token_atom_recall_min),
                    pharm_threshold=float(args.pharm_threshold),
                    top_mass_pct=float(args.top_mass_pct),
                    viz_percentile=int(args.viz_percentile),
                    temperatures=temperatures, dump_dir=args.dump_dir, backbone_ckpt=backbone_id
                )
                out['is_cliff'] = True
                rows.append(out)
            except Exception as e:
                import traceback
                import sys
                error_msg = str(e)
                tb_str = ''.join(traceback.format_exception(*sys.exc_info()))

                # Write to err_log file
                with open(err_log_path, 'a') as ef:
                    ef.write("\n" + "="*60 + "\n")
                    ef.write(f"ERROR at cliff row index {idx}\n")
                    ef.write(f"Error type: {type(e).__name__}\n")
                    ef.write(f"Error message: {error_msg}\n")
                    ef.write("\nFull traceback:\n")
                    ef.write(tb_str)
                    ef.write("\n" + "="*60 + "\n")

                # Print to stderr (will appear in .err file)
                print(f"\n{'='*60}", file=sys.stderr, flush=True)
                print(f"❌ ERROR at cliff row index {idx}", file=sys.stderr, flush=True)
                print(f"{'='*60}", file=sys.stderr, flush=True)
                print(f"Error type: {type(e).__name__}", file=sys.stderr, flush=True)
                print(f"Error message: {error_msg}", file=sys.stderr, flush=True)
                print(f"\nFull traceback:", file=sys.stderr, flush=True)
                print(tb_str, file=sys.stderr, flush=True)
                print(f"{'='*60}\n", file=sys.stderr, flush=True)

                # Also print to stdout
                print(f"\n❌ ERROR at cliff row {idx}: {error_msg}", flush=True)
                print(f"See .err file for full traceback", flush=True)

                rows.append({
                    'pair_type': 'cliff',
                    'error': error_msg,
                    'error_type': type(e).__name__,
                    'is_cliff': True,
                    'row_index': int(idx)
                })
        # Process non-cliffs
        for idx, row in df_non.iterrows():
            try:
                out = process_pair_row(
                    row, models, preproc, args.seq_len, args.threshold,
                    args.map_mode, args.fg_norm,
                    model_id=model_id, is_ensemble=is_ens, backbone_id=backbone_id,
                    ph_smarts=PH_SMARTS if PH_SMARTS else None,
                    pharm_json_path=pharm_path,
                    token_atom_recall_min=float(args.token_atom_recall_min),
                    pharm_threshold=float(args.pharm_threshold),
                    top_mass_pct=float(args.top_mass_pct),
                    viz_percentile=int(args.viz_percentile),
                    temperatures=temperatures, dump_dir=args.dump_dir, backbone_ckpt=backbone_id
                )
                out['is_cliff'] = False
                rows.append(out)
            except Exception as e:
                import traceback
                import sys
                error_msg = str(e)
                tb_str = ''.join(traceback.format_exception(*sys.exc_info()))

                # Write to err_log file
                with open(err_log_path, 'a') as ef:
                    ef.write("\n" + "="*60 + "\n")
                    ef.write(f"ERROR at non-cliff row index {idx}\n")
                    ef.write(f"Error type: {type(e).__name__}\n")
                    ef.write(f"Error message: {error_msg}\n")
                    ef.write("\nFull traceback:\n")
                    ef.write(tb_str)
                    ef.write("\n" + "="*60 + "\n")

                # Print to stderr (will appear in .err file)
                print(f"\n{'='*60}", file=sys.stderr, flush=True)
                print(f"❌ ERROR at non-cliff row index {idx}", file=sys.stderr, flush=True)
                print(f"{'='*60}", file=sys.stderr, flush=True)
                print(f"Error type: {type(e).__name__}", file=sys.stderr, flush=True)
                print(f"Error message: {error_msg}", file=sys.stderr, flush=True)
                print(f"\nFull traceback:", file=sys.stderr, flush=True)
                print(tb_str, file=sys.stderr, flush=True)
                print(f"{'='*60}\n", file=sys.stderr, flush=True)

                # Also print to stdout
                print(f"\n❌ ERROR at non-cliff row {idx}: {error_msg}", flush=True)
                print(f"See .err file for full traceback", flush=True)

                rows.append({
                    'pair_type': 'non_cliff',
                    'error': error_msg,
                    'error_type': type(e).__name__,
                    'is_cliff': False,
                    'row_index': int(idx)
                })
        out_df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        out_df.to_csv(out_csv, index=False, sep=',', quoting=csv.QUOTE_MINIMAL)
        # Sidecar Parquet for nested JSON fidelity
        try:
            out_df.to_parquet(os.path.splitext(out_csv)[0] + '.parquet', index=False)
        except Exception as e:
            print(f"Parquet save skipped: {e}")
        print(f"Saved: {out_csv} ({len(out_df)} rows)")

    # Resolve checkpoints and build models
    dev = None
    if args.device:
        try:
            dev = torch.device(args.device)
        except Exception:
            print(f"Warning: invalid --device '{args.device}', falling back to auto")
            dev = None
    models_all = load_models(ckpts, vocab_size, args.seq_len, device=dev)
    print(f"\n✅ Models loaded successfully: {len(models_all)} model(s) ready", flush=True)
    print(f"   Device(s): {[next(m.parameters()).device for m in models_all]}", flush=True)

    # Calibration temperatures
    temperatures: Optional[List[float]] = None
    calibrated = False
    if args.calibration_json and os.path.exists(args.calibration_json):
        try:
            with open(args.calibration_json, 'r') as f:
                calib = json.load(f)
            if isinstance(calib, dict) and 'temperature' in calib:
                T = float(calib['temperature'])
                temperatures = [T] * len(models_all)
                calibrated = True
            elif isinstance(calib, dict) and 'per_checkpoint' in calib:
                per = calib['per_checkpoint']
                temperatures = [float(per.get(p, per.get(os.path.basename(p), 1.0))) for p in ckpts]
                calibrated = True
        except Exception as e:
            print(f"Warning: failed to parse calibration JSON: {e}")
            temperatures = None

    # Modes
    do_per_model = args.per_model
    do_ensemble = args.ensemble or (not args.per_model)

    # Load data once (outside loops)
    print("\nLoading data files...")
    df_cliff = pd.read_csv(args.activity_csv)
    df_non = pd.read_csv(args.noncliff_csv)
    print(f"Loaded {len(df_cliff)} cliff pairs and {len(df_non)} non-cliff pairs")

    # Debug mode: test with single pair
    if os.environ.get('DEBUG_SINGLE_PAIR'):
        print("\n⚠️  DEBUG MODE: Testing single pair only", flush=True)
        df_cliff = df_cliff.head(1)
        df_non = df_non.head(1)
        print(f"Processing only 1 cliff + 1 non-cliff pair", flush=True)

    # Create output directory
    output_dir = os.path.dirname(args.out_csv) or os.path.join(script_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Helper: project to the standard evaluation schema used across models
    def project_eval_schema(df: pd.DataFrame) -> pd.DataFrame:
        tmp = df.copy()
        # Normalize ID columns
        for a_key in ['active_compound_id', 'compound1_compound_id', 'active_id', 'compound1_id']:
            if a_key in tmp.columns:
                tmp.rename(columns={a_key: 'compound1_id'}, inplace=True)
                break
        for b_key in ['inactive_compound_id', 'compound2_compound_id', 'inactive_id', 'compound2_id']:
            if b_key in tmp.columns:
                tmp.rename(columns={b_key: 'compound2_id'}, inplace=True)
                break
        # Normalize probabilities
        for pa in ['active_pred_prob', 'compound1_pred_prob']:
            if pa in tmp.columns:
                tmp.rename(columns={pa: 'compound1_pred_prob'}, inplace=True)
                break
        for pi in ['inactive_pred_prob', 'compound2_pred_prob']:
            if pi in tmp.columns:
                tmp.rename(columns={pi: 'compound2_pred_prob'}, inplace=True)
                break
        # Decide schema based on split type
        is_cliff = False
        try:
            if 'pair_type' in tmp.columns:
                is_cliff = str(tmp['pair_type'].iloc[0]).strip() == 'cliff'
        except Exception:
            pass
        if is_cliff:
            # Active/inactive semantics for cliffs
            for a_key in ['active_compound_id', 'compound1_id']:
                if a_key in tmp.columns and 'compound_active_id' not in tmp.columns:
                    tmp.rename(columns={a_key: 'compound_active_id'}, inplace=True)
                    break
            for b_key in ['inactive_compound_id', 'compound2_id']:
                if b_key in tmp.columns and 'compound_inactive_id' not in tmp.columns:
                    tmp.rename(columns={b_key: 'compound_inactive_id'}, inplace=True)
                    break
            for pa in ['active_pred_prob', 'compound1_pred_prob']:
                if pa in tmp.columns and 'compound_active_pred_prob' not in tmp.columns:
                    tmp.rename(columns={pa: 'compound_active_pred_prob'}, inplace=True)
                    break
            for pi in ['inactive_pred_prob', 'compound2_pred_prob']:
                if pi in tmp.columns and 'compound_inactive_pred_prob' not in tmp.columns:
                    tmp.rename(columns={pi: 'compound_inactive_pred_prob'}, inplace=True)
                    break
            # Substructures: prefer explicit active/inactive
            if 'substruct_attr_active' not in tmp.columns and 'substruct_attr_compound1' in tmp.columns:
                tmp.rename(columns={'substruct_attr_compound1': 'substruct_attr_active'}, inplace=True)
            if 'substruct_attr_inactive' not in tmp.columns and 'substruct_attr_compound2' in tmp.columns:
                tmp.rename(columns={'substruct_attr_compound2': 'substruct_attr_inactive'}, inplace=True)
            if 'pos_substructs_active' not in tmp.columns and 'pos_substructs_compound1' in tmp.columns:
                tmp.rename(columns={'pos_substructs_compound1': 'pos_substructs_active'}, inplace=True)
            if 'neg_substructs_active' not in tmp.columns and 'neg_substructs_compound1' in tmp.columns:
                tmp.rename(columns={'neg_substructs_compound1': 'neg_substructs_active'}, inplace=True)
            if 'pos_substructs_inactive' not in tmp.columns and 'pos_substructs_compound2' in tmp.columns:
                tmp.rename(columns={'pos_substructs_compound2': 'pos_substructs_inactive'}, inplace=True)
            if 'neg_substructs_inactive' not in tmp.columns and 'neg_substructs_compound2' in tmp.columns:
                tmp.rename(columns={'neg_substructs_compound2': 'neg_substructs_inactive'}, inplace=True)
            # Predicted classes
            if 'compound_active_pred_prob' in tmp.columns:
                tmp['compound_active_pred_class'] = (tmp['compound_active_pred_prob'] >= 0.5).map({True: 'active', False: 'inactive'})
            if 'compound_inactive_pred_prob' in tmp.columns:
                tmp['compound_inactive_pred_class'] = (tmp['compound_inactive_pred_prob'] >= 0.5).map({True: 'active', False: 'inactive'})
            cols = [c for c in [
                'pair_type','antibiotic_class',
                'compound_active_id','compound_inactive_id',
                'compound_active_pred_prob','compound_inactive_pred_prob',
                'compound_active_pred_class','compound_inactive_pred_class',
                'substruct_attr_active','substruct_attr_inactive',
                'pos_substructs_active','neg_substructs_active',
                'neutral_substructs_active',
                'pos_substructs_inactive','neg_substructs_inactive',
                'neutral_substructs_inactive',
                'active_smiles','inactive_smiles',
                'model_type'
            ] if c in tmp.columns]
            return tmp[cols]
        else:
            # Non-cliffs: compound1/compound2 semantics
            if 'substruct_attr_active' in tmp.columns:
                tmp.rename(columns={'substruct_attr_active': 'substruct_attr_compound1'}, inplace=True)
            if 'substruct_attr_inactive' in tmp.columns:
                tmp.rename(columns={'substruct_attr_inactive': 'substruct_attr_compound2'}, inplace=True)
            if 'pos_substructs_active' in tmp.columns:
                tmp.rename(columns={'pos_substructs_active': 'pos_substructs_compound1'}, inplace=True)
            if 'neg_substructs_active' in tmp.columns:
                tmp.rename(columns={'neg_substructs_active': 'neg_substructs_compound1'}, inplace=True)
            if 'pos_substructs_inactive' in tmp.columns:
                tmp.rename(columns={'pos_substructs_inactive': 'pos_substructs_compound2'}, inplace=True)
            if 'neg_substructs_inactive' in tmp.columns:
                tmp.rename(columns={'neg_substructs_inactive': 'neg_substructs_compound2'}, inplace=True)
            # Predicted classes
            if 'compound1_pred_prob' in tmp.columns:
                tmp['compound1_pred_class'] = (tmp['compound1_pred_prob'] >= 0.5).map({True: 'active', False: 'inactive'})
            if 'compound2_pred_prob' in tmp.columns:
                tmp['compound2_pred_class'] = (tmp['compound2_pred_prob'] >= 0.5).map({True: 'active', False: 'inactive'})
            base_cols = [c for c in [
                'pair_type', 'antibiotic_class', 'compound1_id', 'compound2_id',
                'compound1_pred_prob', 'compound2_pred_prob',
                'compound1_pred_class','compound2_pred_class',
                'substruct_attr_compound1', 'substruct_attr_compound2',
                'pos_substructs_compound1', 'neg_substructs_compound1',
                'neutral_substructs_compound1',
                'pos_substructs_compound2', 'neg_substructs_compound2',
                'neutral_substructs_compound2',
                'compound1_smiles','compound2_smiles',
                'model_type'
            ] if c in tmp.columns]
            return tmp[base_cols]

    # Per-model runs
    if do_per_model:
        for model_idx, (p, m) in enumerate(zip(ckpts, models_all), start=1):
            model_id = os.path.splitext(os.path.basename(p))[0]
            CURRENT = [m]

            print(f"\n{'='*60}")
            print(f"Running per-model: Model {model_idx} ({model_id})")
            print(f"{'='*60}")

            # Process cliffs
            cliff_output = os.path.join(output_dir, f"cnn_model{model_idx}_cliffs.csv")
            run_once_split(df_cliff.copy(), 'cliff', CURRENT, model_id, False, model_id, cliff_output, args.err_log)
            if os.path.exists(cliff_output):
                # Replace on-disk CSV with standardized evaluation schema and sidecar Parquet
                odf = pd.read_csv(cliff_output)
                sdf = project_eval_schema(odf)
                sdf.to_csv(cliff_output, index=False)
                try:
                    sdf.to_parquet(os.path.splitext(cliff_output)[0] + '.parquet', index=False)
                except Exception as e:
                    print(f"Parquet save skipped (cliffs): {e}")

            # Process non-cliffs
            noncliff_output = os.path.join(output_dir, f"cnn_model{model_idx}_non_cliffs.csv")
            run_once_split(df_non.copy(), 'non_cliff', CURRENT, model_id, False, model_id, noncliff_output, args.err_log)
            if os.path.exists(noncliff_output):
                odf = pd.read_csv(noncliff_output)
                sdf = project_eval_schema(odf)
                sdf.to_csv(noncliff_output, index=False)
                try:
                    sdf.to_parquet(os.path.splitext(noncliff_output)[0] + '.parquet', index=False)
                except Exception as e:
                    print(f"Parquet save skipped (non-cliffs): {e}")

    # Ensemble run
    if do_ensemble:
        bb = max(0, min(int(args.backbone_index), len(models_all)-1))
        # Reorder so backbone model is first for IG
        models_bb = [models_all[bb]] + [m for i, m in enumerate(models_all) if i != bb]
        backbone_id = os.path.splitext(os.path.basename(ckpts[bb]))[0]

        print(f"\n{'='*60}")
        print(f"Running ensemble with backbone index {bb}: {backbone_id}")
        print(f"{'='*60}")

        # Process cliffs
        cliff_output = os.path.join(output_dir, "cnn_ensemble_cliffs.csv")
        run_once_split(df_cliff.copy(), 'cliff', models_bb, 'ensemble', True, backbone_id, cliff_output, args.err_log)
        if os.path.exists(cliff_output):
            odf = pd.read_csv(cliff_output)
            sdf = project_eval_schema(odf)
            sdf.to_csv(cliff_output, index=False)
            try:
                sdf.to_parquet(os.path.splitext(cliff_output)[0] + '.parquet', index=False)
            except Exception as e:
                print(f"Parquet save skipped (ensemble cliffs): {e}")

        # Process non-cliffs
        noncliff_output = os.path.join(output_dir, "cnn_ensemble_non_cliffs.csv")
        run_once_split(df_non.copy(), 'non_cliff', models_bb, 'ensemble', True, backbone_id, noncliff_output, args.err_log)
        if os.path.exists(noncliff_output):
            odf = pd.read_csv(noncliff_output)
            sdf = project_eval_schema(odf)
            sdf.to_csv(noncliff_output, index=False)
            try:
                sdf.to_parquet(os.path.splitext(noncliff_output)[0] + '.parquet', index=False)
            except Exception as e:
                print(f"Parquet save skipped (ensemble non-cliffs): {e}")


if __name__ == '__main__':
    import sys

    print("🚀 Starting CNN XAI Analysis", flush=True)
    print(f"   Python: {sys.version}", flush=True)
    print(f"   Script: {__file__}", flush=True)

    try:
        main()
        print("\n✅ Script completed successfully", flush=True)
    except Exception as e:
        import traceback
        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"💥 FATAL ERROR IN MAIN:", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)
        print(f"Error type: {type(e).__name__}", file=sys.stderr, flush=True)
        print(f"Error message: {str(e)}", file=sys.stderr, flush=True)
        print(f"\nFull traceback:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr, flush=True)

        print(f"\n💥 FATAL ERROR: {str(e)}", flush=True)
        print(f"See .err file for full traceback", flush=True)
        raise
