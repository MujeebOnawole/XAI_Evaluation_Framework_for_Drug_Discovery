#!/usr/bin/env python3
"""
RGCN XAI Activity Pairs Analysis (Integrated Gradients + Occlusion)
===================================================================

This script mirrors the CNN XAI pairwise analysis, adapted for a relational GCN
model operating on molecular graphs. It produces a per-pair CSV with the same
schema as the CNN version, plus a few RGCN-specific metrics.

Primary XAI: Integrated Gradients (IG) over node features (atoms).
Sanity checks: node/edge/functional-group occlusion and graph-level fidelity.

Usage:
  python rgcn_xai_activity_pairs.py \
    --activity_csv balanced_activity_cliff_pairs.csv \
    --noncliff_csv balanced_non_cliff_pairs.csv \
    --ig_steps 64 --out_csv outputs/rgcn_xai_balanced_full_detailed.csv \
    [--full] [--samples_per_class 2] \
    [--calibration_json path.json] [--dump_dir outputs/rgcn_xai_pairs_debug] [--seed 42]

Notes:
- Graph construction reuses build_data.construct_mol_graph_from_smiles to match
  the node/edge features used during model training, padding/truncating to 40 dims.
- Ensemble: auto-discovers .ckpt files under model_checkpoints/ if --ckpt omitted.
- Functional group scoring reuses RF FG_SMARTS for parity.
"""

import os
import csv
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any
import re

import numpy as np
import pandas as pd

import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import rdmolops

from config import Configuration
from build_data import return_murcko_leaf_structure
from model import BaseGNN

# Graph utilities
try:
    from build_data import construct_mol_graph_from_smiles
except Exception:
    construct_mol_graph_from_smiles = None  # Will fallback

from torch_geometric.data import Data, Batch

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

# Optional pharmacophore SMARTS mapping (loaded at runtime via --pharmacophore_json)
PH_SMARTS: Dict[str, str] = {}


def to_device_graph(g: Data, ref: nn.Module) -> Data:
    """Move a PyG Data to the same device as the model, preserving all attributes."""
    dev = next(ref.parameters()).device if any(p.requires_grad for p in ref.parameters()) else torch.device('cpu')
    return g.to(dev)


def ensure_edge_type(g: Data) -> Data:
    """Ensure edge_type exists and is long tensor of length E."""
    if not hasattr(g, 'edge_index') or g.edge_index is None:
        return g
    E = int(g.edge_index.shape[1]) if g.edge_index.dim() == 2 else 0
    if not hasattr(g, 'edge_type') or g.edge_type is None:
        g.edge_type = torch.zeros((E,), dtype=torch.long, device=g.edge_index.device)
    else:
        # coerce dtype and shape
        try:
            et = g.edge_type
            if et.dtype != torch.long:
                et = et.long()
            if et.dim() != 1:
                et = et.view(-1)
            if int(et.shape[0]) != E:
                et = torch.zeros((E,), dtype=torch.long, device=g.edge_index.device)
            g.edge_type = et
        except Exception:
            g.edge_type = torch.zeros((E,), dtype=torch.long, device=g.edge_index.device)
    return g


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Fallback SMILES->graph if build_data.construct_mol_graph_from_smiles is unavailable."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Atom features (coarse); will be padded to 40 if needed
        feats = []
        for a in mol.GetAtoms():
            feats.append([
                a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(),
                int(a.GetHybridization()), int(a.GetIsAromatic()), a.GetTotalNumHs()
            ])
        if not feats:
            return None
        x = torch.tensor(np.array(feats), dtype=torch.float)
        if x.shape[1] < 40:
            pad = torch.zeros(x.shape[0], 40 - x.shape[1])
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > 40:
            x = x[:, :40]
        # Edges
        ei = []
        et = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            ei.extend([[i, j], [j, i]])
            bt = int(b.GetBondTypeAsDouble())
            et.extend([bt, bt])
        if not ei:
            # Isolated atoms: create empty edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
        else:
            edge_index = torch.tensor(np.array(ei).T, dtype=torch.long)
            edge_type = torch.tensor(np.array(et), dtype=torch.long)
        g = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        # Default batch of zeros
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
        return g
    except Exception:
        return None


@torch.no_grad()
def predict_prob(models: List[BaseGNN], g: Data, temperatures: Optional[List[float]] = None) -> float:
    probs = []
    for i, m in enumerate(models):
        m.eval()
        gg = ensure_edge_type(g.clone())
        batch = Batch.from_data_list([gg])
        out, _ = m(batch)  # logits shape (B=1, 1)
        logit = out.squeeze()
        if temperatures and i < len(temperatures) and temperatures[i] and temperatures[i] > 0:
            p = torch.sigmoid(logit / temperatures[i]).item()
        else:
            p = torch.sigmoid(logit).item()
        probs.append(p)
    return float(np.mean(probs)) if probs else float('nan')


def integrated_gradients_rgcn(model: BaseGNN, g: Data, steps: int = 64, paths: int = 1, noise: float = 0.0, seed: int = 42) -> np.ndarray:
    """
    Integrated Gradients on node features for a single graph.
    Returns per-node attribution vector (signed), L1-normalized.
    """
    model.eval()
    # Prepare a base graph copy and original features
    g_base = ensure_edge_type(g.clone())
    x_orig = g_base.x.detach().clone()
    # Small constant baseline to avoid degenerate gradients
    baseline = torch.full_like(x_orig, 0.10)

    alphas = torch.linspace(0.0, 1.0, steps=steps, device=x_orig.device)

    # Accumulate gradients over steps
    agg = torch.zeros_like(x_orig)

    rng = torch.Generator(device=x_orig.device)
    rng.manual_seed(int(seed))
    n_paths = max(1, int(paths))
    for _ in range(n_paths):
        for alpha in alphas:
            # Fresh graph each step
            g_step = ensure_edge_type(g_base.clone())
            xi = (baseline + alpha * (x_orig - baseline))
            if float(noise) > 0.0:
                xi = xi + torch.normal(mean=0.0, std=float(noise), size=xi.shape, generator=rng, device=xi.device)
            xi = xi.requires_grad_(True)
            g_step.x = xi

            # Forward on a Batch to match model expectation
            b = Batch.from_data_list([g_step])
            out, _ = model(b)
            # Target single pre-sigmoid logit for attribution
            logit = out.view(-1)[0]

            # Grad
            grad = torch.autograd.grad(logit, xi, retain_graph=False, create_graph=False, allow_unused=True)[0]
            if grad is None:
                grad = torch.zeros_like(xi)
            agg = agg + grad.detach()

    # Average gradients and compute IG
    avg_grad = agg / float(steps * n_paths)
    ig = (x_orig - baseline) * avg_grad

    # Reduce features -> per-node score
    node_scores = ig.sum(dim=1).detach().cpu().numpy()

    # L1 normalization with epsilon
    denom = float(np.sum(np.abs(node_scores))) + 1e-12
    node_scores = node_scores / denom

    # Retry with SmoothIG if nearly zero mass (robustness)
    if np.sum(np.abs(node_scores)) < 1e-12:
        try:
            return integrated_gradients_rgcn(model, g, steps=max(256, int(steps)), paths=max(16, int(paths)), noise=max(0.002, float(noise)), seed=seed)
        except Exception:
            pass
    return node_scores


def _prepare_graph_masked(smiles: str, feature_dim: int, smask: Optional[List[int]], device: torch.device) -> Optional[Data]:
    # Build full graph, then apply feature-zero masking safely (no topology change)
    try:
        g = construct_mol_graph_from_smiles(smiles, smask=[]) if construct_mol_graph_from_smiles else smiles_to_graph(smiles)
        if g is None:
            return None
        # Apply masking by zeroing node features; also attach smask attribute for downstream layers
        if smask:
            idx = torch.tensor(sorted(set(int(i) for i in smask if i is not None)), dtype=torch.long)
            idx = idx[(idx >= 0) & (idx < g.x.shape[0])]
            if idx.numel() > 0:
                x = g.x.clone()
                x[idx] = 0.0
                g.x = x
                sm = torch.ones(g.x.shape[0], dtype=torch.float)
                sm[idx] = 0.0
                g.smask = sm
        # Coerce feature dimension if needed
        if g.x.shape[1] != feature_dim:
            x = g.x
            if x.shape[1] > feature_dim:
                x = x[:, :feature_dim].clone()
            else:
                new_x = torch.zeros((x.shape[0], feature_dim), device=x.device, dtype=x.dtype)
                new_x[:, :x.shape[1]] = x
                x = new_x
            g.x = x
        g = ensure_edge_type(g)
        g = _ensure_batch_attr(g)
        return g.to(device)
    except Exception:
        return None


def _ensure_batch_attr(g: Data) -> Data:
    if not hasattr(g, 'batch') or g.batch is None:
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
    return g


def occlusion_atom_attr_ensemble(models: List[BaseGNN],
                                 smiles: str,
                                 per_atom: bool = False,
                                 batch_size: int = 64) -> np.ndarray:
    """
    Forward-only occlusion attribution using either per-atom masks or Murcko substructures.
    Aggregates across ensemble; returns per-atom vector (signed, L1-normalized).
    """
    # Build a base graph to get atom count and feature dim/device
    device = next(models[0].parameters()).device
    # Use the model's configured input feature dimension (defaults to 40)
    try:
        base_fdim = int(getattr(models[0], 'config', None).num_node_features)  # type: ignore
    except Exception:
        base_fdim = 40
    g0 = _prepare_graph_masked(smiles, feature_dim=base_fdim, smask=[], device=device)
    # The above feature_dim probe is approximated; use actual feature dim from g0
    if g0 is None:
        return np.array([])
    n_atoms = int(g0.x.shape[0])
    fdim = int(g0.x.shape[1])  # expected to be input dim (e.g., 40)
    # Build mask list
    masks: List[List[int]] = []
    if per_atom:
        masks = [[i] for i in range(n_atoms)]
    else:
        try:
            subs = return_murcko_leaf_structure(smiles).get('substructure', {})
            masks = [list(atoms) for _, atoms in subs.items() if atoms]
        except Exception:
            masks = []
        if not masks:
            masks = [[i] for i in range(n_atoms)]

    # Aggregate attribution over ensemble
    atom_attr = np.zeros((n_atoms,), dtype=float)
    for m in models:
        m.eval()
        # Base prob
        with torch.no_grad():
            base_p = torch.sigmoid(m(Batch.from_data_list([g0]))[0].view(-1)).item()
        # Process masks in batches
        deltas: List[float] = []
        for i in range(0, len(masks), max(1, int(batch_size))):
            chunk = masks[i:i+batch_size]
            gs: List[Data] = []
            for sm in chunk:
                gmask = _prepare_graph_masked(smiles, feature_dim=fdim, smask=sm, device=device)
                if gmask is not None:
                    gs.append(gmask)
            if not gs:
                deltas.extend([0.0] * len(chunk))
                continue
            b = Batch.from_data_list(gs)
            with torch.no_grad():
                out, _ = m(b)
                probs = torch.sigmoid(out.view(-1)).detach().cpu().numpy()
            # base - masked
            for j, p in enumerate(probs):
                deltas.append(base_p - float(p))
        # Map mask deltas to atom-level
        if per_atom:
            for idx, d in enumerate(deltas):
                a = masks[idx][0]
                if 0 <= a < n_atoms:
                    atom_attr[a] += float(d)
        else:
            for idx, d in enumerate(deltas):
                atoms = masks[idx]
                if atoms:
                    w = float(d) / float(len(atoms))
                    for a in atoms:
                        if 0 <= a < n_atoms:
                            atom_attr[a] += w
    # Normalize L1
    s = float(np.sum(np.abs(atom_attr)))
    return atom_attr / s if s > 0 else atom_attr


def rdkit_distance_matrix(smiles: str) -> Optional[np.ndarray]:
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        return rdmolops.GetDistanceMatrix(m).astype(int)
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


def smartsmatch_groups(smiles: str, atom_scores: np.ndarray, fg_smarts: Dict[str, str], agg: str = 'sum') -> Tuple[Dict[str, float], Dict[str, int], Dict[str, List[int]]]:
    """Compute per-functional-group scores by summing atom_scores over matched atoms."""
    group_scores: Dict[str, float] = {}
    group_presence: Dict[str, int] = {}
    group_atoms: Dict[str, List[int]] = {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or atom_scores.size == 0:
        return group_scores, group_presence, group_atoms
    for name, smarts in fg_smarts.items():
        try:
            patt = Chem.MolFromSmarts(smarts)
            if not patt:
                continue
            matches = mol.GetSubstructMatches(patt)
            if not matches:
                group_presence[name] = 0
                continue
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


def prepare_visualization_data_rgcn(atom_scores: np.ndarray, threshold: float = 0.1, retry_count: int = 0) -> Dict[str, Any]:
    if atom_scores is None or atom_scores.size == 0:
        return {'error': 'no_atoms'}
    colors = []
    # Color-blind friendly palette: blue for positive, orange for negative
    BLUE = (0.1216, 0.4667, 0.7059)   # #1f77b4
    ORANGE = (1.0, 0.4980, 0.0550)    # #ff7f0e
    pos_idx = []
    neg_idx = []
    neu_idx = []
    for i, s in enumerate(atom_scores):
        if s > threshold:
            inten = min(abs(s) / (2*threshold), 1.0)
            c = (1.0*(1-inten) + BLUE[0]*inten,
                 1.0*(1-inten) + BLUE[1]*inten,
                 1.0*(1-inten) + BLUE[2]*inten)
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
    if not pos_idx and not neg_idx and threshold > 0.001 and retry_count < 3:
        return prepare_visualization_data_rgcn(atom_scores, threshold=threshold/2.0, retry_count=retry_count+1)
    return {
        'atom_attributions': atom_scores.tolist(),
        'atom_colors': colors,
        'positive_atoms': pos_idx,
        'negative_atoms': neg_idx,
        'neutral_atoms': neu_idx,
        'n_atoms': int(len(atom_scores))
    }


def murcko_substructures(smiles: str) -> Dict[str, List[int]]:
    try:
        from build_data import return_murcko_leaf_structure  # prefer project util if available
        subs = return_murcko_leaf_structure(smiles).get('substructure', {}) or {}
        # Ensure keys are strings and values are lists of ints
        return {str(k): [int(a) for a in v] for k, v in subs.items() if v}
    except Exception:
        # Fallback to ring systems using RDKit
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


def presence_delta(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    keys = set(a.keys()) | set(b.keys())
    return {k: (a.get(k, 0) - b.get(k, 0)) for k in keys}


def saliency_entropy_gini(abs_scores: np.ndarray) -> Tuple[float, float]:
    s = abs_scores.astype(float)
    tot = s.sum()
    if tot <= 0:
        return 0.0, 0.0
    p = s / tot
    eps = 1e-12
    H = float(-np.sum(p * np.log(p + eps)))
    ps = np.sort(p)
    n = len(ps)
    G = 1.0 - 2.0 * float(np.sum(ps * (np.arange(n, 0, -1) - 0.5))) / float(n)
    return H, G


def topk_indices_by_abs(scores: np.ndarray, frac: float) -> List[int]:
    n = len(scores)
    k = max(1, int(round(frac * n)))
    idx = np.argsort(-np.abs(scores))[:k]
    return list(map(int, idx))


@torch.no_grad()
def fidelity_keep_drop_nodes(model: BaseGNN, g: Data, scores: np.ndarray, fracs=(0.05, 0.10, 0.20)) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    base = torch.sigmoid(model(Batch.from_data_list([ensure_edge_type(g.clone())]))[0].view(-1)).item()
    for frac, tag in zip(fracs, ['k5', 'k10', 'k20']):
        idx = topk_indices_by_abs(scores, frac)
        # keep-only: zero other nodes' features
        g_keep = ensure_edge_type(g.clone())
        mask = np.ones(g_keep.x.shape[0], dtype=bool)
        mask[idx] = False
        if mask.any():
            xk = g_keep.x.clone()
            xk[mask] = 0.0
            g_keep.x = xk
        p_keep = torch.sigmoid(model(Batch.from_data_list([g_keep]))[0].view(-1)).item()
        out[f'graph_fidelity_gain_keep_topk_{tag}'] = float(p_keep - base)
        # drop-only: zero top-k
        g_drop = ensure_edge_type(g.clone())
        xd = g_drop.x.clone()
        xd[idx] = 0.0
        g_drop.x = xd
        p_drop = torch.sigmoid(model(Batch.from_data_list([g_drop]))[0].view(-1)).item()
        out[f'graph_fidelity_drop_topk_{tag}'] = float(p_drop - base)
    return out


def node_occlusion_sanity(model: BaseGNN, g: Data, scores: np.ndarray, topk: int = 5) -> Dict[str, Any]:
    with torch.no_grad():
        base = torch.sigmoid(model(Batch.from_data_list([ensure_edge_type(g.clone())]))[0].view(-1)).item()
    idx_pos = np.argsort(-scores)[:min(topk, len(scores))]
    idx_neg = np.argsort(scores)[:min(topk, len(scores))]
    def occlude(idxs: List[int]) -> float:
        gg = ensure_edge_type(g.clone())
        xx = gg.x.clone()
        xx[idxs] = 0.0
        gg.x = xx
        with torch.no_grad():
            return torch.sigmoid(model(Batch.from_data_list([gg]))[0].view(-1)).item()
    pos_prob = occlude(list(map(int, idx_pos))) if len(idx_pos) else base
    neg_prob = occlude(list(map(int, idx_neg))) if len(idx_neg) else base
    pos_ok = pos_prob < base
    neg_ok = neg_prob > base
    return {
        'base_prob': base,
        'pos_prob_after_occlusion': pos_prob,
        'neg_prob_after_occlusion': neg_prob,
        'pos_direction_ok': bool(pos_ok),
        'neg_direction_ok': bool(neg_ok),
        'sanity_pass': bool(pos_ok and neg_ok)
    }


def edge_mask_score(model: BaseGNN, g: Data, scores: np.ndarray, frac: float = 0.10) -> float:
    """Edge importance proxy: remove edges incident to top-k nodes and measure prob drop."""
    with torch.no_grad():
        base = torch.sigmoid(model(Batch.from_data_list([ensure_edge_type(g.clone())]))[0].view(-1)).item()
    idx = set(topk_indices_by_abs(scores, frac))
    if g.edge_index is None or g.edge_index.numel() == 0:
        return 0.0
    gg0 = ensure_edge_type(g)
    ei = gg0.edge_index.detach().cpu().numpy()
    keep = [k for k in range(ei.shape[1]) if (ei[0, k] not in idx and ei[1, k] not in idx)]
    if not keep:
        return float(-base)
    gg = ensure_edge_type(g.clone())
    device = gg.x.device
    gg.edge_index = torch.tensor(ei[:, keep], dtype=torch.long, device=device)
    if hasattr(gg, 'edge_type') and gg.edge_type is not None:
        et = gg0.edge_type.detach().cpu()
        gg.edge_type = torch.tensor(et[keep].numpy(), dtype=torch.long, device=device)
    with torch.no_grad():
        p = torch.sigmoid(model(Batch.from_data_list([gg]))[0].view(-1)).item()
    return float(p - base)


def subgraph_coherence(smiles: str, scores: np.ndarray, frac: float = 0.20) -> float:
    """
    Coherence of high-attribution nodes: 1 - cut_ratio(S), where S = top-20% nodes.
    cut_ratio = edges(S, ~S) / max(1, edges(S, all)).
    """
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return float('nan')
        n = len(scores)
        if n == 0:
            return float('nan')
        k = max(1, int(round(frac * n)))
        S = set(map(int, np.argsort(-np.abs(scores))[:k]))
        e_S_all = 0
        e_S_cut = 0
        for b in m.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in S or j in S:
                e_S_all += 1
                if (i in S) ^ (j in S):
                    e_S_cut += 1
        if e_S_all == 0:
            return 0.0
        cut_ratio = e_S_cut / float(e_S_all)
        return float(1.0 - cut_ratio)
    except Exception:
        return float('nan')


def classify_pair(active_prob: float, inactive_prob: float,
                  active_target: int, inactive_target: int,
                  threshold: float = 0.5) -> Tuple[str, str, str, str, str, str]:
    # Predicted labels from probabilities
    active_pred = 'active' if active_prob >= threshold else 'inactive'
    inactive_pred = 'active' if inactive_prob >= threshold else 'inactive'

    # True labels from TARGET ground truth
    active_true = 'active' if int(active_target) == 1 else 'inactive'
    inactive_true = 'active' if int(inactive_target) == 1 else 'inactive'

    # Correctness
    active_correct = (active_pred == active_true)
    inactive_correct = (inactive_pred == inactive_true)

    # Confusion matrix labels per compound based on true class
    if active_true == 'active':
        active_clf = 'TP' if active_pred == 'active' else 'FN'
    else:
        active_clf = 'TN' if active_pred == 'inactive' else 'FP'

    if inactive_true == 'active':
        inactive_clf = 'TP' if inactive_pred == 'active' else 'FN'
    else:
        inactive_clf = 'TN' if inactive_pred == 'inactive' else 'FP'

    # Pair-level classification
    if active_correct and inactive_correct:
        pair_cls = 'BothCorrect'
    elif active_correct or inactive_correct:
        pair_cls = 'OneCorrect'
    else:
        pair_cls = 'BothWrong'

    return (
        active_pred,
        inactive_pred,
        pair_cls,
        f"active_correct={active_correct};inactive_correct={inactive_correct}",
        active_clf,
        inactive_clf,
    )


def prepare_graph(smiles: str) -> Optional[Data]:
    if construct_mol_graph_from_smiles is not None:
        try:
            g = construct_mol_graph_from_smiles(smiles, smask=[])
            return g
        except Exception:
            pass
    return smiles_to_graph(smiles)


def mask_atoms_in_graph(graph: Data, atoms_to_mask: List[int]) -> Data:
    """
    Mask specific atoms in graph by zeroing their features.

    Args:
        graph: PyTorch Geometric Data graph
        atoms_to_mask: List of atom indices to mask

    Returns:
        Masked graph
    """
    if not atoms_to_mask or not hasattr(graph, 'x'):
        return graph

    masked_graph = graph.clone()
    for atom_idx in atoms_to_mask:
        if 0 <= atom_idx < masked_graph.x.size(0):
            masked_graph.x[atom_idx] = 0.0

    return masked_graph


def murcko_attribution_rgcn(smiles: str,
                            models: List,
                            cfg,
                            temperatures: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Calculate atom-level attribution using Murcko substructure masking for RGCN.

    Returns:
        Dictionary with atom_scores, positive_contributors, negative_contributors
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or return_murcko_leaf_structure is None:
        return {
            'atom_scores': [],
            'positive_contributors': [],
            'negative_contributors': [],
            'error': 'invalid_smiles_or_no_murcko'
        }

    num_atoms = mol.GetNumAtoms()

    # Get base graph and prediction
    base_graph = prepare_graph(smiles)
    if base_graph is None:
        return {
            'atom_scores': [0.0] * num_atoms,
            'positive_contributors': [],
            'negative_contributors': [],
            'error': 'graph_construction_failed'
        }

    # Ensure batch attribute
    if not hasattr(base_graph, 'batch') or base_graph.batch is None:
        base_graph.batch = torch.zeros(base_graph.x.size(0), dtype=torch.long, device=base_graph.x.device)

    # Base prediction
    base_preds = []
    for model in models:
        base_graph_copy = base_graph.clone().to(next(model.parameters()).device)
        with torch.no_grad():
            out, _ = model(Batch.from_data_list([base_graph_copy]))
            p = torch.sigmoid(out).item()
        base_preds.append(p)
    base_pred = np.mean(base_preds)

    # Get Murcko substructures
    try:
        murcko_data = return_murcko_leaf_structure(smiles)
        substructures = murcko_data.get('substructure', {})
    except Exception:
        substructures = {}

    if not substructures:
        return {
            'atom_scores': [0.0] * num_atoms,
            'positive_contributors': [],
            'negative_contributors': [],
            'error': 'no_substructures_found'
        }

    # Calculate attribution per substructure
    atom_scores = np.zeros(num_atoms, dtype=float)

    for sub_id, atoms in substructures.items():
        if not atoms or not isinstance(atoms, list):
            continue

        try:
            # Mask this substructure
            masked_graph = mask_atoms_in_graph(base_graph, atoms)

            # Get masked prediction
            masked_preds = []
            for model in models:
                masked_graph_copy = masked_graph.clone().to(next(model.parameters()).device)
                with torch.no_grad():
                    out, _ = model(Batch.from_data_list([masked_graph_copy]))
                    p = torch.sigmoid(out).item()
                masked_preds.append(p)
            masked_pred = np.mean(masked_preds)

            # Attribution = base - masked
            attribution = base_pred - masked_pred

            # Distribute to atoms in this substructure
            for atom_idx in atoms:
                if 0 <= atom_idx < num_atoms:
                    atom_scores[atom_idx] += attribution / len(atoms)
        except Exception:
            continue

    # Identify contributors (threshold: 0.1)
    positive_contributors = [int(i) for i, score in enumerate(atom_scores) if score > 0.1]
    negative_contributors = [int(i) for i, score in enumerate(atom_scores) if score < -0.1]

    return {
        'atom_scores': atom_scores.tolist(),
        'positive_contributors': positive_contributors,
        'negative_contributors': negative_contributors,
        'base_prediction': float(base_pred),
        'num_substructures': len(substructures)
    }


def pharmacophore_recognition_with_atoms_rgcn(smiles: str,
                                              antibiotic_class: str,
                                              positive_atoms: List[int],
                                              pharmacophore_json_path: str) -> Dict[str, Any]:
    """
    Calculate pharmacophore recognition based on atom contributors for RGCN.

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


def rename_for_noncliff(out_dict: Dict[str, Any], pair_type: str) -> Dict[str, Any]:
    """
    Rename active/inactive columns to compound1/compound2 for non-cliff pairs.

    For activity cliff pairs: keep active/inactive naming (they have different activities)
    For non-cliff pairs: use compound1/compound2 naming (they have similar activities)

    Args:
        out_dict: Output dictionary with active/inactive keys
        pair_type: Type of pair ('cliff' or 'non_cliff')

    Returns:
        Dictionary with appropriately named keys
    """
    # Only rename for non-cliff pairs
    if pair_type and 'non' in str(pair_type).lower():
        rename_map = {}
        for key in list(out_dict.keys()):
            new_key = key
            # Replace active_ prefix with compound1_
            if key.startswith('active_'):
                new_key = 'compound1_' + key[7:]  # Remove 'active_'
            # Replace inactive_ prefix with compound2_
            elif key.startswith('inactive_'):
                new_key = 'compound2_' + key[9:]  # Remove 'inactive_'
            # Handle middle occurrences (e.g., calibrated_prob_active)
            elif '_active' in key and not key.startswith('active'):
                new_key = key.replace('_active', '_compound1')
            elif '_inactive' in key and not key.startswith('inactive'):
                new_key = key.replace('_inactive', '_compound2')

            if new_key != key:
                rename_map[key] = new_key

        # Apply renaming
        for old_key, new_key in rename_map.items():
            out_dict[new_key] = out_dict.pop(old_key)

    return out_dict


def process_pair_row(row: pd.Series,
                     models: List[BaseGNN],
                     cfg: Configuration,
                     ig_steps: int,
                     threshold: float,
                     fg_norm: str,
                     xai_method: str,
                     per_atom_occ: bool,
                     model_id: str = '',
                     is_ensemble: bool = False,
                     backbone_id: str = '',
                     ph_smarts: Optional[Dict[str, str]] = None,
                     pharm_json_path: Optional[str] = None,
                     pharm_threshold: float = 0.2,
                     top_mass_pct: float = 0.5,
                     viz_percentile: int = 80,
                     ig_paths: int = 1,
                     ig_noise: float = 0.0,
                     allow_edge_clamp: bool = False,
                     temperatures: Optional[List[float]] = None,
                     dump_dir: Optional[str] = None,
                     backbone_ckpt: Optional[str] = None) -> Dict[str, Any]:

    # Extract schema variants for balanced files
    if 'active_smiles' in row and 'inactive_smiles' in row:
        cls = row.get('class', row.get('antibiotic_class', ''))
        active_id = row.get('active_compound_id', '')
        inactive_id = row.get('inactive_compound_id', '')
        active_smiles = str(row['active_smiles'])
        inactive_smiles = str(row['inactive_smiles'])
        similarity = float(row.get('structural_similarity', np.nan))
        pair_type = 'cliff'
    else:
        cls = row.get('class', '')
        active_id = row.get('compound1_id', '')
        inactive_id = row.get('compound2_id', '')
        active_smiles = str(row.get('compound1_smiles', ''))
        inactive_smiles = str(row.get('compound2_smiles', ''))
        similarity = float(row.get('structural_similarity', np.nan))
        pair_type = 'noncliff'

    # TARGET ground truth and groups (aligns with CNN extraction)
    if pair_type == 'cliff':
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

    # Debug header - add after extracting compound IDs and SMILES
    pair_id = f"{active_id}_{inactive_id}"
    print("\n" + "="*60)
    print(f"RGCN Processing: {pair_id}")
    print(f"Class: {cls}, Pair type: {pair_type}")
    print("="*60)

    # Build graphs
    print(f"Building molecular graphs...")
    g_a = prepare_graph(active_smiles)
    g_i = prepare_graph(inactive_smiles)

    if g_a is None:
        print(f"ERROR: Failed to build graph for active SMILES: {active_smiles}")
        raise RuntimeError('active_graph_construction_failed')
    if g_i is None:
        print(f"ERROR: Failed to build graph for inactive SMILES: {inactive_smiles}")
        raise RuntimeError('inactive_graph_construction_failed')

    print(f" Graphs built successfully:")
    try:
        print(f"  Active: {g_a.x.shape[0]} nodes, {g_a.edge_index.shape[1] if getattr(g_a, 'edge_index', None) is not None else 0} edges")
    except Exception:
        pass
    try:
        print(f"  Inactive: {g_i.x.shape[0]} nodes, {g_i.edge_index.shape[1] if getattr(g_i, 'edge_index', None) is not None else 0} edges")
    except Exception:
        pass

    # Use the first model as attribution backbone
    backbone = models[0]
    backbone.eval()
    g_a = ensure_edge_type(to_device_graph(g_a, backbone))
    g_i = ensure_edge_type(to_device_graph(g_i, backbone))
    # Feature-dim alignment to backbone config if available
    try:
        fdim = int(getattr(getattr(backbone, 'config', None), 'num_node_features', g_a.x.shape[1]))
        if g_a.x.shape[1] != fdim:
            xa = torch.zeros((g_a.x.shape[0], fdim), device=g_a.x.device, dtype=g_a.x.dtype)
            xa[:, :min(fdim, g_a.x.shape[1])] = g_a.x[:, :min(fdim, g_a.x.shape[1])]
            g_a.x = xa
        if g_i.x.shape[1] != fdim:
            xi = torch.zeros((g_i.x.shape[0], fdim), device=g_i.x.device, dtype=g_i.x.dtype)
            xi[:, :min(fdim, g_i.x.shape[1])] = g_i.x[:, :min(fdim, g_i.x.shape[1])]
            g_i.x = xi
    except Exception:
        pass
    # Edge type validation or clamp
    edge_clamped_flag = False
    try:
        n_rel = int(getattr(getattr(backbone, 'config', None), 'num_edge_types', 0))
        def _validate(gt):
            return n_rel and hasattr(gt, 'edge_type') and gt.edge_type is not None and int(gt.edge_type.max().item()) >= n_rel
        if _validate(g_a) or _validate(g_i):
            if not allow_edge_clamp:
                raise RuntimeError(f"edge_type contains relation >= num_edge_types ({n_rel}); use --allow_edge_clamp to clamp.")
            else:
                print(f"WARNING: edge_type out of range; clamping to [0,{n_rel-1}] due to --allow_edge_clamp")
                if hasattr(g_a, 'edge_type') and g_a.edge_type is not None:
                    g_a.edge_type = torch.clamp(g_a.edge_type, 0, n_rel-1)
                if hasattr(g_i, 'edge_type') and g_i.edge_type is not None:
                    g_i.edge_type = torch.clamp(g_i.edge_type, 0, n_rel-1)
                edge_clamped_flag = True
    except Exception:
        pass

    # Predictions (average across ensemble)
    active_prob = predict_prob(models, g_a, temperatures)
    inactive_prob = predict_prob(models, g_i, temperatures)
    raw_active_prob = predict_prob(models, g_a, None) if temperatures else active_prob
    raw_inactive_prob = predict_prob(models, g_i, None) if temperatures else inactive_prob
    pred_diff = float(active_prob - inactive_prob)

    # Attributions per molecule (node-level)
    attr_mode = 'ig'
    if xai_method.lower() == 'ig':
        # Ensemble IG averaging
        a_list = []
        i_list = []
        for m in models:
            a_list.append(integrated_gradients_rgcn(m, g_a, steps=ig_steps, paths=max(1, int(ig_paths)), noise=float(ig_noise)))
            i_list.append(integrated_gradients_rgcn(m, g_i, steps=ig_steps, paths=max(1, int(ig_paths)), noise=float(ig_noise)))
        atom_a = np.mean(np.stack(a_list, axis=0), axis=0)
        atom_i = np.mean(np.stack(i_list, axis=0), axis=0)
        try:
            SA = np.stack(a_list, axis=0)
            SI = np.stack(i_list, axis=0)
            _std_a = np.std(SA, axis=0).mean()
            _std_i = np.std(SI, axis=0).mean()
            attribution_std_across_models = float(0.5 * (_std_a + _std_i))
            # Variance warning if relative std > 50%
            mean_abs_a = float(np.mean(np.abs(np.mean(SA, axis=0)))) + 1e-12
            mean_abs_i = float(np.mean(np.abs(np.mean(SI, axis=0)))) + 1e-12
            rel_std = 0.5 * (_std_a / mean_abs_a + _std_i / mean_abs_i)
            if rel_std > 0.5:
                print(f"WARNING: Ensemble attribution variance high (relative std ~ {rel_std:.2f})")
            print(f"Ensemble attribution: averaged {len(models)} models")
        except Exception:
            attribution_std_across_models = float('nan')
        # Degeneracy guard: if IG mass is nearly zero, force SmoothIG retry
        try:
            total_mass = float(np.sum(np.abs(atom_a))) + float(np.sum(np.abs(atom_i)))
        except Exception:
            total_mass = 0.0
        if not np.isfinite(total_mass) or total_mass < 1e-8:
            a_list = []
            i_list = []
            for m in models:
                a_list.append(integrated_gradients_rgcn(m, g_a, steps=max(256, int(ig_steps)), paths=max(8, int(ig_paths)), noise=max(0.01, float(ig_noise))))
                i_list.append(integrated_gradients_rgcn(m, g_i, steps=max(256, int(ig_steps)), paths=max(8, int(ig_paths)), noise=max(0.01, float(ig_noise))))
            atom_a = np.mean(np.stack(a_list, axis=0), axis=0)
            atom_i = np.mean(np.stack(i_list, axis=0), axis=0)
            attr_mode = 'smoothig'
            try:
                total_mass = float(np.sum(np.abs(atom_a))) + float(np.sum(np.abs(atom_i)))
            except Exception:
                total_mass = 0.0
            # If still degenerate, fallback to per-atom occlusion
            if not np.isfinite(total_mass) or total_mass < 1e-8:
                atom_a = occlusion_atom_attr_ensemble(models, active_smiles, per_atom=True, batch_size=64)
                atom_i = occlusion_atom_attr_ensemble(models, inactive_smiles, per_atom=True, batch_size=64)
                attr_mode = 'occlusion_fallback'
    else:
        atom_a = occlusion_atom_attr_ensemble(models, active_smiles, per_atom=per_atom_occ, batch_size=64)
        atom_i = occlusion_atom_attr_ensemble(models, inactive_smiles, per_atom=per_atom_occ, batch_size=64)
        attribution_std_across_models = float('nan')

    # SmoothIG auto if concentrated
    if xai_method.lower() == 'ig':
        Ha, Ga = saliency_entropy_gini(np.abs(atom_a))
        Hi, Gi = saliency_entropy_gini(np.abs(atom_i))
        if max(Ga, Gi) > 0.90:
            print(f" High Gini ({Ga:.3f},{Gi:.3f}); re-running IG with SmoothIG on ensemble...")
            a_list = []
            i_list = []
            for m in models:
                a_list.append(integrated_gradients_rgcn(m, g_a, steps=max(128, int(ig_steps)), paths=8, noise=0.001))
                i_list.append(integrated_gradients_rgcn(m, g_i, steps=max(128, int(ig_steps)), paths=8, noise=0.001))
            atom_a = np.mean(np.stack(a_list, axis=0), axis=0)
            atom_i = np.mean(np.stack(i_list, axis=0), axis=0)

    # Validate attributions
    print(f"Node attributions computed:")
    try:
        print(f"  Active: shape={getattr(atom_a, 'shape', None)}, sum={np.sum(np.abs(atom_a)):.6f}")
    except Exception:
        print(f"  Active: shape=?, sum=?")
    try:
        print(f"  Inactive: shape={getattr(atom_i, 'shape', None)}, sum={np.sum(np.abs(atom_i)):.6f}")
    except Exception:
        print(f"  Inactive: shape=?, sum=?")
    if (hasattr(atom_a, 'size') and atom_a.size == 0) or (hasattr(atom_i, 'size') and atom_i.size == 0):
        print(f"WARNING: Empty node attributions detected!")

    # Sanity: node occlusion
    sanity_a = node_occlusion_sanity(backbone, g_a, atom_a, topk=5) if xai_method.lower() == 'ig' else {'sanity_pass': True}
    sanity_i = node_occlusion_sanity(backbone, g_i, atom_i, topk=5) if xai_method.lower() == 'ig' else {'sanity_pass': True}
    sanity_pass = bool(sanity_a.get('sanity_pass', False) and sanity_i.get('sanity_pass', False))

    # Group scoring
    group_scores_a, presence_a, group_atoms_a = smartsmatch_groups(active_smiles, atom_a, FG_SMARTS, agg=fg_norm)
    group_scores_i, presence_i, group_atoms_i = smartsmatch_groups(inactive_smiles, atom_i, FG_SMARTS, agg=fg_norm)

    # Deltas
    signed_delta = {k: group_scores_a.get(k, 0.0) - group_scores_i.get(k, 0.0) for k in set(list(group_scores_a.keys()) + list(group_scores_i.keys()))}
    delta_abs = {k: abs(v) for k, v in signed_delta.items()}
    changed = [k for k, v in presence_delta(presence_a, presence_i).items() if v != 0]
    common = [k for k in (set(presence_a) & set(presence_i)) if presence_a.get(k, 0) == 1 and presence_i.get(k, 0) == 1]
    edit_mass = float(np.sum([delta_abs.get(k, 0.0) for k in changed])) if changed else 0.0
    context_mass = float(np.sum([delta_abs.get(k, 0.0) for k in common])) if common else 0.0
    denom = edit_mass + context_mass
    propagation_index = (context_mass / denom) if denom > 0 else 0.0
    edit_conc_index = float(edit_mass / (edit_mass + 1e-12)) if edit_mass > 0 else 0.0
    top_feature = ''
    top_support = 0.0
    if delta_abs:
        tkey = max(delta_abs.keys(), key=lambda k: delta_abs.get(k, 0.0))
        top_feature = tkey
        top_support = float(delta_abs.get(tkey, 0.0))

    # Classifications (use ground truth per compound)
    active_cls, inactive_cls, pair_cls, pair_flags, active_clf, inactive_clf = classify_pair(
        active_prob, inactive_prob, active_target, inactive_target, threshold=threshold
    )

    # RGCN-native: locality, edge importance, subgraph coherence, fidelity
    dm_a = rdkit_distance_matrix(active_smiles)
    dm_i = rdkit_distance_matrix(inactive_smiles)
    edited_atoms_a = sorted({a for g in changed if presence_a.get(g, 0) == 1 for a in group_atoms_a.get(g, [])}) if group_atoms_a else []
    edited_atoms_i = sorted({a for g in changed if presence_i.get(g, 0) == 1 for a in group_atoms_i.get(g, [])}) if group_atoms_i else []
    # === Three-case edit detection ===
    edit_detection_mode = 'fg_delta'
    try:
        mol_a = Chem.MolFromSmiles(active_smiles)
        mol_i = Chem.MolFromSmiles(inactive_smiles)
    except Exception:
        mol_a, mol_i = None, None
    # Case 2: MCS-based edit detection (only if FG deltas empty AND structures differ)
    if (not edited_atoms_a or not edited_atoms_i) and (mol_a is not None and mol_i is not None) and (str(active_smiles) != str(inactive_smiles)):
        try:
            from rdkit.Chem import rdFMCS
            mcs = rdFMCS.FindMCS([mol_a, mol_i], timeout=1)
            if mcs and getattr(mcs, 'numAtoms', 0) > 0:
                mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
                if mcs_mol is not None:
                    match_a = mol_a.GetSubstructMatch(mcs_mol)
                    match_i = mol_i.GetSubstructMatch(mcs_mol)
                    if match_a and match_i:
                        all_a = set(range(mol_a.GetNumAtoms()))
                        all_i = set(range(mol_i.GetNumAtoms()))
                        ea = sorted(all_a - set(match_a))
                        ei = sorted(all_i - set(match_i))
                        if ea:
                            edited_atoms_a = ea
                        if ei:
                            edited_atoms_i = ei
                        if edited_atoms_a and edited_atoms_i:
                            edit_detection_mode = 'mcs'
        except Exception:
            pass
    # Case 3: Ring fallback (only if still empty after MCS)
    if (not edited_atoms_a or not edited_atoms_i) and (mol_a is not None and mol_i is not None):
        if edit_detection_mode != 'mcs':
            edit_detection_mode = 'ring_fallback'
        if not edited_atoms_a:
            edited_atoms_a = [a.GetIdx() for a in mol_a.GetAtoms() if a.IsInRing()]
            if not edited_atoms_a:
                edited_atoms_a = [a.GetIdx() for a in mol_a.GetAtoms() if a.GetAtomicNum() > 1]
        if not edited_atoms_i:
            edited_atoms_i = [a.GetIdx() for a in mol_i.GetAtoms() if a.IsInRing()]
            if not edited_atoms_i:
                edited_atoms_i = [a.GetIdx() for a in mol_i.GetAtoms() if a.GetAtomicNum() > 1]
    r1_a = atoms_within_radius(dm_a, edited_atoms_a, 1)
    r2_a = atoms_within_radius(dm_a, edited_atoms_a, 2)
    r1_i = atoms_within_radius(dm_i, edited_atoms_i, 1)
    r2_i = atoms_within_radius(dm_i, edited_atoms_i, 2)
    def frac_mass(scores: np.ndarray, atoms: List[int]) -> float:
        total = float(np.sum(np.abs(scores)))
        return float(np.sum(np.abs(scores[atoms]))) / total if (total > 0 and atoms) else 0.0
    node_locality_k1 = 0.5 * (frac_mass(atom_a, r1_a) + frac_mass(atom_i, r1_i))
    node_locality_k2 = 0.5 * (frac_mass(atom_a, r2_a) + frac_mass(atom_i, r2_i))
    edge_importance_score = 0.5 * (edge_mask_score(backbone, g_a, atom_a, 0.10) + edge_mask_score(backbone, g_i, atom_i, 0.10))
    subgraph_coh = 0.5 * (subgraph_coherence(active_smiles, atom_a) + subgraph_coherence(inactive_smiles, atom_i))
    fid_a = fidelity_keep_drop_nodes(backbone, g_a, atom_a, fracs=(0.05, 0.10, 0.20))
    fid_i = fidelity_keep_drop_nodes(backbone, g_i, atom_i, fracs=(0.05, 0.10, 0.20))
    graph_fid = {k: (fid_a.get(k, 0.0) + fid_i.get(k, 0.0)) / 2.0 for k in set(fid_a) | set(fid_i)}

    # Always compute masses and core/context split for delta_core_align export
    atom_mass_active = float(np.sum(np.abs(atom_a))) if getattr(atom_a, 'size', 0) else 0.0
    atom_mass_inactive = float(np.sum(np.abs(atom_i))) if getattr(atom_i, 'size', 0) else 0.0
    core_a_mass = float(np.sum(np.abs(atom_a[r1_a]))) if getattr(atom_a, 'size', 0) and r1_a else 0.0
    core_i_mass = float(np.sum(np.abs(atom_i[r1_i]))) if getattr(atom_i, 'size', 0) and r1_i else 0.0
    distant_a_mass = atom_mass_active - core_a_mass
    distant_i_mass = atom_mass_inactive - core_i_mass
    edit_mass = core_a_mass + core_i_mass
    context_mass = distant_a_mass + distant_i_mass
    delta_core_align = float(edit_mass / (edit_mass + context_mass + 1e-12))

    # Core alignment metrics
    try:
        core_a_atoms = set(core_atoms_from_pharm(active_smiles, str(cls), pharm_json_path or 'pharmacophore.json'))
        core_i_atoms = set(core_atoms_from_pharm(inactive_smiles, str(cls), pharm_json_path or 'pharmacophore.json'))
        top_a_atoms = set(top_mass_atoms(atom_a, pct=float(top_mass_pct))) if atom_a.size else set()
        top_i_atoms = set(top_mass_atoms(atom_i, pct=float(top_mass_pct))) if atom_i.size else set()
        core_align_active = float(len(core_a_atoms & top_a_atoms)) / float(len(core_a_atoms)) if core_a_atoms else 0.0
        core_align_inactive = float(len(core_i_atoms & top_i_atoms)) / float(len(core_i_atoms)) if core_i_atoms else 0.0
    except Exception:
        core_align_active = 0.0
        core_align_inactive = 0.0

    # Core alignment (Δcore_align)
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
        # Include both required_any and loose_required_any for broader coverage
        for category in ['required_any', 'loose_required_any']:
            for feat in (sec.get(category) or []):
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
        tot = s.sum()
        if tot <= 0:
            return []
        order = np.argsort(-s)
        acc = 0.0
        sel: List[int] = []
        for idx in order:
            sel.append(int(idx))
            acc += float(s[idx])
            if acc / tot >= max(1e-6, float(pct)):
                break
        return sel

    # Functional group occlusion sanity (mask atoms of groups and observe drop in |pred_diff|)
    fg_occ_pass = np.nan
    try:
        base_diff = abs(pred_diff)
        drops = []
        for gname in changed[:6]:  # cap for speed
            aa = group_atoms_a.get(gname, [])
            ii = group_atoms_i.get(gname, [])
            # Mask atoms for active
            ga = g_a.clone(); xa = ga.x.clone();
            if aa:
                xa[aa] = 0.0
            ga.x = xa
            # Mask atoms for inactive
            gi = g_i.clone(); xi = gi.x.clone();
            if ii:
                xi[ii] = 0.0
            gi.x = xi
            p_a = torch.sigmoid(backbone(ga)[0].view(-1)).item()
            p_i = torch.sigmoid(backbone(gi)[0].view(-1)).item()
            drops.append(base_diff - abs(p_a - p_i))
        if drops:
            fg_occ_pass = float(np.mean([d for d in drops if np.isfinite(d)])) > 0
    except Exception:
        fg_occ_pass = np.nan

    # Dump per-pair details if requested
    if dump_dir:
        try:
            os.makedirs(dump_dir, exist_ok=True)
            dump = {
                'active': {
                    'smiles': active_smiles,
                    'atom_attr': atom_a.tolist()
                },
                'inactive': {
                    'smiles': inactive_smiles,
                    'atom_attr': atom_i.tolist()
                }
            }
            dump_name = f"dump_{str(active_id) or 'A'}_{str(inactive_id) or 'B'}.json"
            with open(os.path.join(dump_dir, dump_name), 'w') as f:
                json.dump(dump, f)
        except Exception:
            pass

    # Assemble row (schema parity with CNN + RGCN extras)
    out: Dict[str, Any] = {
        'pair_type': pair_type,
        'antibiotic_class': cls,
        'active_compound_id': active_id,
        'inactive_compound_id': inactive_id,
        'active_smiles': active_smiles,
        'inactive_smiles': inactive_smiles,
        'similarity': similarity,
        'model_type': 'RGCN_3edge_original',
        'xai_method': 'IntegratedGradients' if xai_method.lower() == 'ig' else 'Occlusion',
        'active_target_ground_truth': int(active_target),
        'inactive_target_ground_truth': int(inactive_target),
        'active_group': active_group,
        'inactive_group': inactive_group,
        'group_combination': f"{active_group}_{inactive_group}",
        'model_id': model_id,
        'is_ensemble': bool(is_ensemble),
        'backbone_id': backbone_id,
        'active_pred_prob': active_prob,
        'inactive_pred_prob': inactive_prob,
        'prediction_difference': pred_diff,
        'calibrated_prob_active': active_prob if temperatures else np.nan,
        'calibrated_prob_inactive': inactive_prob if temperatures else np.nan,
        'raw_prob_active': raw_active_prob,
        'raw_prob_inactive': raw_inactive_prob,
        'active_class': active_cls,
        'inactive_class': inactive_cls,
        'active_classification': active_clf,
        'inactive_classification': inactive_clf,
        'pair_classification': pair_cls,
        'xai_visible': bool(len(group_scores_a) > 0 or len(group_scores_i) > 0),
        'rgcn_visible': True,
        'propagation_index': propagation_index,
        'edit_concentration_index': edit_conc_index,
        'edit_mass': edit_mass,
        'context_mass': context_mass,
        'total_delta': float(edit_mass + context_mass),
        'core_delta': float(np.sum(np.abs(atom_a[atoms_within_radius(dm_a, edited_atoms_a, 1)])) if dm_a is not None else 0.0) +
                      float(np.sum(np.abs(atom_i[atoms_within_radius(dm_i, edited_atoms_i, 1)])) if dm_i is not None else 0.0),
        'distant_delta': float(np.sum(np.abs(atom_a))) + float(np.sum(np.abs(atom_i))),
        'edit_delta': edit_mass,
        'feature_scores_active': json.dumps(group_scores_a),
        'feature_scores_inactive': json.dumps(group_scores_i),
        'feature_delta_signed': json.dumps(signed_delta),
        'feature_delta_abs': json.dumps(delta_abs),
        'feature_presence_active': json.dumps(presence_a),
        'feature_presence_inactive': json.dumps(presence_i),
        'topk_features_changed': json.dumps([k for k, _ in sorted(delta_abs.items(), key=lambda kv: -kv[1])[:12]]),
        'changed_functional_groups': json.dumps(changed),
        'common_features_count': int(len(common)),
        'active_features_count': int(sum(presence_a.values())) if presence_a else 0,
        'inactive_features_count': int(sum(presence_i.values())) if presence_i else 0,
        'feature_differences': int(len(changed)),
        'max_feature_diff': float(max(delta_abs.values()) if delta_abs else 0.0),
        'mean_feature_diff': float(np.mean(list(delta_abs.values())) if delta_abs else 0.0),
        'edit_driver_candidates': json.dumps([{ 'feature': k, 'direction': 'changed', 'support': delta_abs.get(k, 0.0)} for k in changed][:8]),
        'top_edit_driver': top_feature if top_feature else '',
        'edit_driver_support': top_support,
        'occlusion_sanity_pass': sanity_pass,
        'node_sanity_pass': sanity_pass,
        'edge_sanity_pass': bool(np.isfinite(edge_importance_score)),
        'group_sanity_pass': fg_occ_pass,
        'core_align_active': float(core_align_active),
        'core_align_inactive': float(core_align_inactive),
        'delta_core_align': float(core_align_active - core_align_inactive),
        'atom_attr_active': json.dumps([float(v) for v in atom_a.tolist()]),
        'atom_attr_inactive': json.dumps([float(v) for v in atom_i.tolist()]),
        # RGCN-specific
        'node_locality_k1': float(node_locality_k1),
        'node_locality_k2': float(node_locality_k2),
        'edge_importance_score': float(edge_importance_score),
        'graph_fidelity_drop_topk_k5': float(graph_fid.get('graph_fidelity_drop_topk_k5', np.nan)),
        'graph_fidelity_drop_topk_k10': float(graph_fid.get('graph_fidelity_drop_topk_k10', np.nan)),
        'graph_fidelity_drop_topk_k20': float(graph_fid.get('graph_fidelity_drop_topk_k20', np.nan)),
        'subgraph_coherence': float(subgraph_coh),
        # Export masses and delta_core_align
        'atom_mass_active': atom_mass_active,
        'atom_mass_inactive': atom_mass_inactive,
        'core_align_active': core_a_mass,
        'core_align_inactive': core_i_mass,
        'edit_mass': edit_mass,
        'context_mass': context_mass,
        'delta_core_align': delta_core_align,
        'attr_mode': attr_mode,
        'edit_detection_mode': edit_detection_mode,
        # bookkeeping
        'model_ensemble_size': int(len(models)),
        'backbone_ckpt': backbone_ckpt or '',
        'diagnostics': json.dumps({
            'device': str(next(backbone.parameters()).device),
            'ig_steps': int(ig_steps),
            'ig_paths': int(ig_paths),
            'ig_noise': float(ig_noise),
            'edge_clamped': bool(edge_clamped_flag),
            'feature_dim': int(g_a.x.shape[1]) if hasattr(g_a, 'x') else None,
            'num_relations': int(getattr(getattr(backbone, 'config', None), 'num_edge_types', 0)),
        }),
    }

    # Visualization data
    print(f"Preparing visualization data...")
    try:
        if getattr(atom_a, 'size', 0) > 0 and getattr(atom_i, 'size', 0) > 0:
            import numpy as _np
            thr_a = float(_np.percentile(_np.abs(atom_a), max(0, min(100, int(viz_percentile))))) if getattr(atom_a, 'size', 0) else 0.1
            thr_i = float(_np.percentile(_np.abs(atom_i), max(0, min(100, int(viz_percentile))))) if getattr(atom_i, 'size', 0) else 0.1
            viz_a = prepare_visualization_data_rgcn(atom_a, threshold=thr_a)
            viz_i = prepare_visualization_data_rgcn(atom_i, threshold=thr_i)

            # Check for errors in visualization prep
            if 'error' in viz_a:
                print(f"WARNING: Visualization prep failed for active: {viz_a['error']}")
                raise ValueError(f"Active viz error: {viz_a['error']}")
            if 'error' in viz_i:
                print(f"WARNING: Visualization prep failed for inactive: {viz_i['error']}")
                raise ValueError(f"Inactive viz error: {viz_i['error']}")

            # Success - update output dict
            out.update({
                'viz_active_atom_attr': json.dumps(viz_a.get('atom_attributions', [])),
                'viz_active_atom_colors': json.dumps(viz_a.get('atom_colors', [])),
                'viz_active_positive_atoms': json.dumps(viz_a.get('positive_atoms', [])),
                'viz_active_negative_atoms': json.dumps(viz_a.get('negative_atoms', [])),
                'viz_inactive_atom_attr': json.dumps(viz_i.get('atom_attributions', [])),
                'viz_inactive_atom_colors': json.dumps(viz_i.get('atom_colors', [])),
                'viz_inactive_positive_atoms': json.dumps(viz_i.get('positive_atoms', [])),
                'viz_inactive_negative_atoms': json.dumps(viz_i.get('negative_atoms', [])),
                # Prediction–attribution alignment heuristic
                'pred_attr_alignment_active': bool(((active_prob >= threshold) and (float(np.mean(np.asarray(viz_a.get('atom_attributions', [])) > 0)) >= 0.5)) or ((active_prob < threshold) and (float(np.mean(np.asarray(viz_a.get('atom_attributions', [])) > 0)) <= 0.5))),
                'pred_attr_alignment_inactive': bool(((inactive_prob >= threshold) and (float(np.mean(np.asarray(viz_i.get('atom_attributions', [])) > 0)) >= 0.5)) or ((inactive_prob < threshold) and (float(np.mean(np.asarray(viz_i.get('atom_attributions', [])) > 0)) <= 0.5))),
                'pred_attr_alignment_pair': False,  # set below after computing fractions
                'pred_attr_positive_fraction_active': float(np.mean(np.asarray(viz_a.get('atom_attributions', [])) > 0)) if viz_a.get('atom_attributions', []) else 0.0,
                'pred_attr_positive_fraction_inactive': float(np.mean(np.asarray(viz_i.get('atom_attributions', [])) > 0)) if viz_i.get('atom_attributions', []) else 0.0,
                'attribution_std_across_models': float(attribution_std_across_models),
            })
            # Derive pair alignment and mismatch booleans
            pa = bool(out.get('pred_attr_alignment_active', False))
            pi = bool(out.get('pred_attr_alignment_inactive', False))
            out['pred_attr_alignment_pair'] = bool(pa and pi)
            out['pred_attr_mismatch_active'] = bool(not pa)
            out['pred_attr_mismatch_inactive'] = bool(not pi)
            out['pred_attr_mismatch_pair'] = bool(not (pa and pi))
            # Magnitude‑weighted alignment
            try:
                arr_a = np.asarray(viz_a.get('atom_attributions', []), dtype=float)
                arr_i = np.asarray(viz_i.get('atom_attributions', []), dtype=float)
                def _wfrac(a: np.ndarray) -> float:
                    w = np.abs(a); tw = float(w.sum())
                    return float(((w*(a>0)).sum())/(tw+1e-12)) if a.size else 0.0
                wfa = _wfrac(arr_a); wfi = _wfrac(arr_i)
                paw = bool(((active_prob >= threshold) and (wfa >= 0.5)) or ((active_prob < threshold) and (wfa < 0.5)))
                piw = bool(((inactive_prob >= threshold) and (wfi >= 0.5)) or ((inactive_prob < threshold) and (wfi < 0.5)))
                out['pred_attr_alignment_active_weighted'] = bool(paw)
                out['pred_attr_alignment_inactive_weighted'] = bool(piw)
                out['pred_attr_alignment_pair_weighted'] = bool(paw and piw)
                out['pred_attr_positive_fraction_active_weighted'] = float(wfa)
                out['pred_attr_positive_fraction_inactive_weighted'] = float(wfi)
                out['pred_attr_mismatch_active_weighted'] = bool(not paw)
                out['pred_attr_mismatch_inactive_weighted'] = bool(not piw)
                out['pred_attr_mismatch_pair_weighted'] = bool(not (paw and piw))
                out['weighted_alignment_metrics_active'] = json.dumps({'method':'weighted','valid':True,'weighted_pos_fraction':float(wfa)})
                out['weighted_alignment_metrics_inactive'] = json.dumps({'method':'weighted','valid':True,'weighted_pos_fraction':float(wfi)})
            except Exception:
                out['pred_attr_alignment_active_weighted'] = False
                out['pred_attr_alignment_inactive_weighted'] = False
                out['pred_attr_alignment_pair_weighted'] = False
                out['pred_attr_positive_fraction_active_weighted'] = 0.0
                out['pred_attr_positive_fraction_inactive_weighted'] = 0.0
                out['pred_attr_mismatch_active_weighted'] = False
                out['pred_attr_mismatch_inactive_weighted'] = False
                out['pred_attr_mismatch_pair_weighted'] = False
                out['weighted_alignment_metrics_active'] = json.dumps({'method':'weighted','valid':False})
                out['weighted_alignment_metrics_inactive'] = json.dumps({'method':'weighted','valid':False})
            print(f" Visualization data added successfully")
        else:
            print(f"WARNING: Empty node attributions (active={getattr(atom_a, 'size', 0)}, inactive={getattr(atom_i, 'size', 0)})")
            raise ValueError("Empty node attributions")

    except Exception as e:
        print(f"ERROR in visualization preparation: {e}")
        import traceback
        traceback.print_exc()
        # Add empty but valid JSON to prevent CSV parsing errors
        out.update({
            'viz_active_atom_attr': json.dumps([]),
            'viz_active_atom_colors': json.dumps([]),
            'viz_active_positive_atoms': json.dumps([]),
            'viz_active_negative_atoms': json.dumps([]),
            'viz_inactive_atom_attr': json.dumps([]),
            'viz_inactive_atom_colors': json.dumps([]),
            'viz_inactive_positive_atoms': json.dumps([]),
            'viz_inactive_negative_atoms': json.dumps([]),
        })

    # Pharmacophore scoring (optional)
    if ph_smarts:
        print(f"Starting pharmacophore validation...")
        try:
            # Prerequisite checks
            if getattr(atom_a, 'size', 0) == 0 or getattr(atom_i, 'size', 0) == 0:
                print(f"WARNING: Empty node attributions, skipping pharmacophore validation")
                raise ValueError("Empty node attributions")

            if not cls or str(cls).lower() in ['nan', 'none', '']:
                print(f"WARNING: Invalid antibiotic class '{cls}', skipping pharmacophore validation")
                raise ValueError(f"Invalid class: {cls}")

            # Compute pharmacophore scores: build class-specific flat SMARTS
            flat_smarts: Dict[str, str] = {}
            if isinstance(ph_smarts, dict) and cls and isinstance(ph_smarts.get(cls, {}), dict):
                sec = ph_smarts.get(cls, {})
                for cat in ['required_any', 'loose_required_any', 'important_any', 'optional_any']:
                    for feat in (sec.get(cat) or []):
                        name = feat.get('name'); smt = feat.get('smarts')
                        if name and smt:
                            flat_smarts[name] = smt
            else:
                flat_smarts = {k: v for k, v in (ph_smarts or {}).items() if isinstance(v, str)}
            if not flat_smarts:
                print(f" WARNING: No pharmacophore SMARTS found for class '{cls}' (feature scoring)")

            ph_a, ph_pa, _ = smartsmatch_groups(str(active_smiles), atom_a, flat_smarts, agg=fg_norm)
            ph_i, ph_pi, _ = smartsmatch_groups(str(inactive_smiles), atom_i, flat_smarts, agg=fg_norm)

            # Check if any features matched
            if not ph_a and not ph_i:
                print(f"WARNING: No pharmacophore features matched for either compound")

            ph_delta = {k: abs(ph_a.get(k, 0.0) - ph_i.get(k, 0.0)) for k in set(ph_a) | set(ph_i)}

            out['pharm_feature_scores_active'] = json.dumps(ph_a)
            out['pharm_feature_scores_inactive'] = json.dumps(ph_i)
            out['pharm_feature_delta_abs'] = json.dumps(ph_delta)
            out['pharm_changed_features'] = json.dumps([k for k in set(ph_pa) | set(ph_pi) if ph_pa.get(k,0)!=ph_pi.get(k,0)])

            print(f" Pharmacophore validation complete: {len(ph_a)} active features, {len(ph_i)} inactive features")

        except Exception as e:
            print(f"ERROR in pharmacophore validation: {e}")
            import traceback
            traceback.print_exc()
            # Empty but valid results
            out['pharm_feature_scores_active'] = json.dumps({})
            out['pharm_feature_scores_inactive'] = json.dumps({})
            out['pharm_feature_delta_abs'] = json.dumps({})
            out['pharm_changed_features'] = json.dumps([])
    else:
        print(f"INFO: No pharmacophore SMARTS provided (--pharmacophore_json not specified), skipping validation")
        out['pharm_feature_scores_active'] = json.dumps({})
        out['pharm_feature_scores_inactive'] = json.dumps({})
        out['pharm_feature_delta_abs'] = json.dumps({})
        out['pharm_changed_features'] = json.dumps([])

    # Murcko substructure masking attribution
    print(f"Starting Murcko attribution for active compound...")
    try:
        murcko_active = murcko_attribution_rgcn(
            str(active_smiles), models, cfg, temperatures
        )
        murcko_inactive = murcko_attribution_rgcn(
            str(inactive_smiles), models, cfg, temperatures
        )

        # Add atom-level attribution columns
        out.update({
            'murcko_atom_scores_active': json.dumps(murcko_active.get('atom_scores', [])),
            'murcko_atom_scores_inactive': json.dumps(murcko_inactive.get('atom_scores', [])),
            'murcko_positive_contributors_active': json.dumps(murcko_active.get('positive_contributors', [])),
            'murcko_negative_contributors_active': json.dumps(murcko_active.get('negative_contributors', [])),
            'murcko_positive_contributors_inactive': json.dumps(murcko_inactive.get('positive_contributors', [])),
            'murcko_negative_contributors_inactive': json.dumps(murcko_inactive.get('negative_contributors', [])),
            'murcko_num_substructures_active': int(murcko_active.get('num_substructures', 0)),
            'murcko_num_substructures_inactive': int(murcko_inactive.get('num_substructures', 0)),
            'masking_mode': 'node_feature_zeroing',
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
            pharm_active = pharmacophore_recognition_with_atoms_rgcn(
                str(active_smiles), str(cls),
                murcko_active.get('positive_contributors', []),
                pharm_json_path
            )
            pharm_inactive = pharmacophore_recognition_with_atoms_rgcn(
                str(inactive_smiles), str(cls),
                murcko_inactive.get('positive_contributors', []),
                pharm_json_path
            )

            if 'error' not in pharm_active and 'error' not in pharm_inactive:
                out.update({
                    'murcko_pharm_recognized_active': json.dumps(pharm_active.get('recognized_patterns', [])),
                    'murcko_pharm_missed_active': json.dumps(pharm_active.get('missed_patterns', [])),
                    'murcko_pharm_recognition_score_active': float(pharm_active.get('recognition_score', 0.0)),
                    'murcko_pharm_recognition_rate_active': float(pharm_active.get('recognition_rate', 0.0)),
                    'murcko_pharm_recognized_inactive': json.dumps(pharm_inactive.get('recognized_patterns', [])),
                    'murcko_pharm_missed_inactive': json.dumps(pharm_inactive.get('missed_patterns', [])),
                    'murcko_pharm_recognition_score_inactive': float(pharm_inactive.get('recognition_score', 0.0)),
                    'murcko_pharm_recognition_rate_inactive': float(pharm_inactive.get('recognition_rate', 0.0)),
                    'murcko_pharm_recognition_score_pair': float(np.mean([
                        pharm_active.get('recognition_score', 0.0),
                        pharm_inactive.get('recognition_score', 0.0)
                    ])),
                    'murcko_delta_core_align': float(
                        pharm_active.get('recognition_score', 0.0) -
                        pharm_inactive.get('recognition_score', 0.0)
                    ),
                })
                print(f" Murcko pharmacophore recognition complete")

        # Build substructure-level attribution JSON (per Murcko substructure masking)
        try:
            from rdkit import Chem as _Chem
            def _substruct_json_rgcn(smiles: str) -> List[Dict[str, Any]]:
                mol = _Chem.MolFromSmiles(smiles)
                if mol is None:
                    return []
                # Base graph and prediction
                base_graph = prepare_graph(smiles)
                if base_graph is None:
                    return []
                if not hasattr(base_graph, 'batch') or base_graph.batch is None:
                    base_graph.batch = torch.zeros(base_graph.x.size(0), dtype=torch.long, device=base_graph.x.device)
                base_preds = []
                for m in models:
                    g = base_graph.clone().to(next(m.parameters()).device)
                    with torch.no_grad():
                        out, _ = m(Batch.from_data_list([g]))
                        base_preds.append(torch.sigmoid(out).item())
                base_prob = float(np.mean(base_preds)) if base_preds else float('nan')
                # Substructures
                try:
                    subs = return_murcko_leaf_structure(smiles).get('substructure', {})
                except Exception:
                    subs = {}
                out_list: List[Dict[str, Any]] = []
                for _, atoms in (subs or {}).items():
                    if not atoms:
                        continue
                    try:
                        mg = mask_atoms_in_graph(base_graph, atoms)
                        masked_preds = []
                        for m in models:
                            g2 = mg.clone().to(next(m.parameters()).device)
                            with torch.no_grad():
                                out, _ = m(Batch.from_data_list([g2]))
                                masked_preds.append(torch.sigmoid(out).item())
                        masked_prob = float(np.mean(masked_preds)) if masked_preds else float('nan')
                        delta = float(base_prob - masked_prob)
                        per_atom = float(delta / max(1, len(atoms)))
                        frag = _Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, kekuleSmiles=True)
                        out_list.append({
                            'substructure': frag,
                            'atoms': atoms,
                            'attribution': delta,
                            'attribution_per_atom': per_atom
                        })
                    except Exception:
                        continue
                out_list.sort(key=lambda d: abs(d.get('attribution', 0.0)), reverse=True)
                return out_list

            key1 = 'active' if pair_type == 'cliff' else 'compound1'
            key2 = 'inactive' if pair_type == 'cliff' else 'compound2'
            sub_a = _substruct_json_rgcn(str(active_smiles))
            sub_i = _substruct_json_rgcn(str(inactive_smiles))
            out[f'substruct_attr_{key1}'] = json.dumps(sub_a)
            out[f'substruct_attr_{key2}'] = json.dumps(sub_i)
            # Build positive/negative/neutral fragment lists using 0.1 threshold
            def _split(items: List[Dict[str, Any]], thr: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
                pos, neg, neu = [], [], []
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
                # Dedup preserve order
                seen=set(); pos_d=[]
                for s in pos:
                    if s not in seen: seen.add(s); pos_d.append(s)
                seen=set(); neg_d=[]
                for s in neg:
                    if s not in seen: seen.add(s); neg_d.append(s)
                seen=set(); neu_d=[]
                for s in neu:
                    if s not in seen: seen.add(s); neu_d.append(s)
                return pos_d, neg_d, neu_d
            p1, n1, u1 = _split(sub_a, 0.1); p2, n2, u2 = _split(sub_i, 0.1)
            out[f'pos_substructs_{key1}'] = json.dumps(p1)
            out[f'neg_substructs_{key1}'] = json.dumps(n1)
            out[f'neutral_substructs_{key1}'] = json.dumps(u1)
            out[f'pos_substructs_{key2}'] = json.dumps(p2)
            out[f'neg_substructs_{key2}'] = json.dumps(n2)
            out[f'neutral_substructs_{key2}'] = json.dumps(u2)
        except Exception as e:
            print(f"Substructure JSON (RGCN) error: {e}")

        print(f" Murcko attribution complete")

    except Exception as e:
        print(f" Murcko attribution error: {e}")
        import traceback
        traceback.print_exc()
        # Add empty Murcko columns
        out.update({
            'murcko_atom_scores_active': json.dumps([]),
            'murcko_atom_scores_inactive': json.dumps([]),
            'murcko_positive_contributors_active': json.dumps([]),
            'murcko_negative_contributors_active': json.dumps([]),
            'murcko_positive_contributors_inactive': json.dumps([]),
            'murcko_negative_contributors_inactive': json.dumps([]),
            'murcko_num_substructures_active': 0,
            'murcko_num_substructures_inactive': 0,
            'murcko_pharm_recognized_active': json.dumps([]),
            'murcko_pharm_missed_active': json.dumps([]),
            'murcko_pharm_recognition_score_active': 0.0,
            'murcko_pharm_recognition_rate_active': 0.0,
            'murcko_pharm_recognized_inactive': json.dumps([]),
            'murcko_pharm_missed_inactive': json.dumps([]),
            'murcko_pharm_recognition_score_inactive': 0.0,
            'murcko_pharm_recognition_rate_inactive': 0.0,
            'murcko_pharm_recognition_score_pair': 0.0,
            'murcko_delta_core_align': 0.0,
            'delta_pharm_signed': float('nan'),
        })

    # Pharmacophore recognition parity with CNN (uses JSON definitions)
    print(f"Starting pharmacophore recognition (JSON) for class='{cls}'...")
    try:
        # Prerequisite checks
        if getattr(atom_a, 'size', 0) == 0 or getattr(atom_i, 'size', 0) == 0:
            print(f" WARNING: Empty node attributions, skipping JSON pharmacophore recognition")
            raise ValueError("Empty node attributions")

        if not cls or str(cls).lower() in ['nan', 'none', '']:
            print(f" WARNING: Invalid class '{cls}', skipping JSON pharmacophore recognition")
            raise ValueError("Invalid class")

        pa = validate_pharmacophore_recognition(str(active_smiles), str(cls), atom_a, pharmacophore_json_path=(pharm_json_path or 'pharmacophore.json'))
        pi = validate_pharmacophore_recognition(str(inactive_smiles), str(cls), atom_i, pharmacophore_json_path=(pharm_json_path or 'pharmacophore.json'))

        if 'error' in pa or 'error' in pi:
            print(f" WARNING: Recognition errors: {pa.get('error')}, {pi.get('error')}")
            raise ValueError("Recognition failed")

        out.update({
            'pharm_recognized_active': json.dumps(pa.get('recognized_features', [])),
            'pharm_missed_active': json.dumps(pa.get('missed_features', [])),
            'pharm_recognition_score_active': float(pa.get('overall_recognition_score', 0.0)),
            'pharm_recognition_rate_active': float(pa.get('recognition_rate', 0.0)),
            'pharm_recognized_inactive': json.dumps(pi.get('recognized_features', [])),
            'pharm_missed_inactive': json.dumps(pi.get('missed_features', [])),
            'pharm_recognition_score_inactive': float(pi.get('overall_recognition_score', 0.0)),
            'pharm_recognition_rate_inactive': float(pi.get('recognition_rate', 0.0)),
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
                pharmacophore_consistency = 1.0 - abs(ps_a - ps_i) / denom
            else:
                pharmacophore_consistency = 0.0
            out['pharmacophore_consistency_score'] = float(pharmacophore_consistency)
            out['pharmacophore_inconsistent_flag'] = bool(abs(ps_a - ps_i) > 0.3)
        except Exception:
            out['pharmacophore_consistency_score'] = float('nan')
            out['pharmacophore_inconsistent_flag'] = False

        # Gate xai_visible by pharmacophore threshold and fidelity
        pharm_ok = (out.get('pharm_recognition_score_active', 0.0) >= float(pharm_threshold)) or \
                   (out.get('pharm_recognition_score_inactive', 0.0) >= float(pharm_threshold))
        out['xai_visible'] = bool(out.get('xai_visible', False) and bool(pharm_ok) and bool(sanity_pass))
        print(f" Pharmacophore recognition complete")

    except Exception as e:
        print(f" ERROR in JSON pharmacophore recognition: {e}")
        import traceback
        traceback.print_exc()
        out.update({
            'pharm_recognized_active': json.dumps([]),
            'pharm_missed_active': json.dumps([]),
            'pharm_recognition_score_active': 0.0,
            'pharm_recognition_rate_active': 0.0,
            'pharm_recognized_inactive': json.dumps([]),
            'pharm_missed_inactive': json.dumps([]),
            'pharm_recognition_score_inactive': 0.0,
            'pharm_recognition_rate_inactive': 0.0,
            'pharm_recognition_score_pair': 0.0,
        })

    # Substructure occlusion JSON (Murcko/rings)
    try:
        def substruct_json(smiles: str, g: Data, base_prob: float) -> List[Dict[str, Any]]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            subs = murcko_substructures(smiles)
            out_list: List[Dict[str, Any]] = []
            for name, atoms in subs.items():
                try:
                    if not atoms:
                        continue
                    gg = ensure_edge_type(g.clone())
                    xx = gg.x.clone()
                    xx[atoms] = 0.0
                    gg.x = xx
                    prob = torch.sigmoid(backbone(Batch.from_data_list([gg]))[0].view(-1)).item()
                    delta = float(base_prob - prob)
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

        # Use backbone model predictions for deltas
        with torch.no_grad():
            base_pa = torch.sigmoid(backbone(Batch.from_data_list([g_a]))[0].view(-1)).item()
            base_pi = torch.sigmoid(backbone(Batch.from_data_list([g_i]))[0].view(-1)).item()
        out['substruct_attr_active'] = json.dumps(substruct_json(str(active_smiles), g_a, base_pa))
        out['substruct_attr_inactive'] = json.dumps(substruct_json(str(inactive_smiles), g_i, base_pi))
        print(" Substructure occlusion JSON added")
    except Exception as e:
        print(f" Substructure occlusion JSON error: {e}")
        out['substruct_attr_active'] = json.dumps([])
        out['substruct_attr_inactive'] = json.dumps([])

    # 85-FG comparability view and full-fragment alias
    try:
        out['fg85_feature_scores_active'] = json.dumps(group_scores_a)
        out['fg85_feature_scores_inactive'] = json.dumps(group_scores_i)
        out['fg85_feature_delta_abs'] = json.dumps(delta_abs)
        out['fg85_changed_features'] = json.dumps(changed)
    except Exception:
        out['fg85_feature_scores_active'] = json.dumps({})
        out['fg85_feature_scores_inactive'] = json.dumps({})
        out['fg85_feature_delta_abs'] = json.dumps({})
        out['fg85_changed_features'] = json.dumps([])
    try:
        out['full_fragment_scores_active'] = out.get('pharm_feature_scores_active', json.dumps({}))
        out['full_fragment_scores_inactive'] = out.get('pharm_feature_scores_inactive', json.dumps({}))
        out['full_fragment_delta_abs'] = out.get('pharm_feature_delta_abs', json.dumps({}))
        out['full_fragment_changed_features'] = out.get('pharm_changed_features', json.dumps([]))
    except Exception:
        out['full_fragment_scores_active'] = json.dumps({})
        out['full_fragment_scores_inactive'] = json.dumps({})
        out['full_fragment_delta_abs'] = json.dumps({})
        out['full_fragment_changed_features'] = json.dumps([])

    # Rename columns for non-cliff pairs (compound1/compound2 instead of active/inactive)
    out = rename_for_noncliff(out, pair_type)

    return out


def _build_model_from_hparams(cfg: Configuration, hparams: Dict[str, Any]) -> BaseGNN:
    rgcn_hidden = hparams.get('rgcn_hidden_feats', [128, 128])
    if isinstance(rgcn_hidden, str):
        rgcn_hidden = [int(x) for x in rgcn_hidden.replace(',', '-').split('-') if str(x).strip()]
    ffn_hidden = int(hparams.get('ffn_hidden_feats', 256))
    ffn_dropout = float(hparams.get('ffn_dropout', 0.5))
    rgcn_dropout = float(hparams.get('rgcn_dropout', 0.5))
    classification = bool(hparams.get('classification', True))
    num_classes = int(hparams.get('num_classes', 2)) if classification else None
    model = BaseGNN(
        config=cfg,
        rgcn_hidden_feats=rgcn_hidden,
        ffn_hidden_feats=ffn_hidden,
        ffn_dropout=ffn_dropout,
        rgcn_dropout=rgcn_dropout,
        classification=classification,
        num_classes=num_classes,
    )
    return model


def _infer_hparams_from_state_dict(sd: Dict[str, Any]) -> Dict[str, Any]:
    """Infer ffn_hidden_feats and rgcn_hidden_feats from parameter shapes.
    Falls back to reasonable defaults if not recoverable."""
    inferred = {}
    # Normalize keys for probing
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    # Infer FFN hidden from predict layer
    ffn_hidden = None
    for k in keys:
        if k.endswith('predict.0.weight') or k.endswith('predict.weight'):
            try:
                ffn_hidden = int(sd[k].shape[1])
                break
            except Exception:
                pass
    if ffn_hidden is None:
        # Try to find the last BN or Linear in fc_layers3
        for alt in ['fc_layers3.3.weight', 'fc_layers3.0.weight', 'fc_layers2.3.weight']:
            kk = [x for x in keys if x.endswith(alt)]
            if kk:
                try:
                    shp = sd[kk[0]].shape
                    # BN running_mean is [H]; Linear weight is [H, H]
                    ffn_hidden = int(shp[0])
                    break
                except Exception:
                    pass
    if ffn_hidden is not None:
        inferred['ffn_hidden_feats'] = ffn_hidden
    # Infer RGCN hidden sizes from residual connections
    layers = {}
    pat = re.compile(r'.*rgcn_gnn_layers\.(\d+)\.res_connection\.weight$')
    for k in keys:
        m = pat.match(k)
        if m:
            idx = int(m.group(1))
            try:
                out_dim = int(sd[k].shape[0])
                layers[idx] = out_dim
            except Exception:
                pass
    if layers:
        inferred['rgcn_hidden_feats'] = [layers[i] for i in sorted(layers.keys())]
    return inferred


def _normalize_state_dict_keys(sd: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(sd, dict):
        return {}
    # try direct
    options = [sd]
    # stripped prefixes
    for pref in ['model.', 'net.', 'module.']:
        stripped = {}
        for k, v in sd.items():
            nk = k[len(pref):] if k.startswith(pref) else k
            stripped[nk] = v
        options.append(stripped)
    # Choose the variant that yields most matching keys post-filter
    return options[0]  # will be refined during filtering


def _filtered_load_state_dict(model: nn.Module, sd: Dict[str, Any]) -> None:
    """Load only parameters that match in name and shape; ignore the rest."""
    if not isinstance(sd, dict):
        return
    model_sd = model.state_dict()
    # Try original and a few prefix-stripped variants
    variants = [sd]
    for pref in ['model.', 'net.', 'module.']:
        stripped = {}
        for k, v in sd.items():
            nk = k[len(pref):] if k.startswith(pref) else k
            stripped[nk] = v
        variants.append(stripped)
    best = None
    best_count = -1
    for var in variants:
        filtered = {k: v for k, v in var.items() if k in model_sd and hasattr(v, 'shape') and model_sd[k].shape == v.shape}
        if len(filtered) > best_count:
            best_count = len(filtered)
            best = filtered
    if best:
        model.load_state_dict(best, strict=False)


def _parse_rgcn_hidden(val: Any) -> List[int]:
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    if isinstance(val, str):
        s = val.strip().replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(' ', '')
        sep = '-' if '-' in s else ','
        parts = [p for p in s.split(sep) if p]
        return [int(p) for p in parts] if parts else [128, 128]
    if isinstance(val, (int, float)):
        return [int(val)]
    return [128, 128]


def load_models(ckpts: List[str], cfg: Configuration, device: Optional[torch.device] = None) -> List[BaseGNN]:
    models: List[BaseGNN] = []
    dev = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    base_model_for_arch: Optional[BaseGNN] = None
    for path in ckpts:
        if not os.path.exists(path):
            continue
        try:
            ck = torch.load(path, map_location=dev)
            # Extract hyperparameters
            hparams = {}
            if isinstance(ck, dict):
                for key in ['hyperparameters', 'hyper_parameters', 'hparams', 'hyperparams']:
                    if key in ck and isinstance(ck[key], dict):
                        hparams = ck[key]
                        break
            sd = ck.get('state_dict', ck)
            if isinstance(sd, dict):
                inferred = _infer_hparams_from_state_dict(sd)
                # Merge: checkpoint hparams take precedence, then inferred
                hp_build = {**inferred, **hparams}
            else:
                hp_build = hparams
            # Normalize hidden sizes
            if 'rgcn_hidden_feats' in hp_build:
                hp_build['rgcn_hidden_feats'] = _parse_rgcn_hidden(hp_build['rgcn_hidden_feats'])
            if 'ffn_hidden_feats' in hp_build and isinstance(hp_build['ffn_hidden_feats'], list):
                hp_build['ffn_hidden_feats'] = int(hp_build['ffn_hidden_feats'][0]) if hp_build['ffn_hidden_feats'] else 256
            # Build (or reuse) model for this architecture
            if base_model_for_arch is None:
                m = _build_model_from_hparams(cfg, hp_build)
                base_model_for_arch = m
            else:
                # Assume same architecture across ensemble (fallback to base if missing)
                m = _build_model_from_hparams(cfg, {
                    'rgcn_hidden_feats': hp_build.get('rgcn_hidden_feats', base_model_for_arch._rgcn_hidden_feats),
                    'ffn_hidden_feats': hp_build.get('ffn_hidden_feats', base_model_for_arch._ffn_hidden_feats),
                    'ffn_dropout': hp_build.get('ffn_dropout', base_model_for_arch._ffn_dropout),
                    'rgcn_dropout': hp_build.get('rgcn_dropout', base_model_for_arch._rgcn_dropout),
                    'classification': hp_build.get('classification', base_model_for_arch._classification),
                    'num_classes': hp_build.get('num_classes', getattr(base_model_for_arch, '_num_classes', 2)),
                })
            # Load state dict (filter by matching shapes/names)
            if isinstance(sd, dict):
                _filtered_load_state_dict(m, sd)
            m.eval()
            m.to(dev)
            models.append(m)
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}")
    if not models:
        raise RuntimeError("No models loaded from provided checkpoints")
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activity_csv', default='activity_cliff_pairs.csv')
    parser.add_argument('--noncliff_csv', default='non_cliff_pairs.csv')
    parser.add_argument('--ckpt', action='append', required=False, help='Path(s) to .ckpt checkpoints (can repeat). If omitted, auto-discovers .ckpt under model_checkpoints/.')
    parser.add_argument('--ig_steps', type=int, default=64)
    parser.add_argument('--ig_paths', type=int, default=8)
    parser.add_argument('--ig_noise', type=float, default=0.001)
    parser.add_argument('--xai_method', type=str, default='occlusion', choices=['occlusion', 'ig'])
    parser.add_argument('--per_atom_occlusion', action='store_true', help='Use per-atom occlusion (slower) instead of Murcko substructures')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--fg_norm', type=str, default='sum', choices=['sum', 'mean', 'max'])
    parser.add_argument('--device', type=str, default=None, help='cpu or cuda[:index]')
    parser.add_argument('--pharm_threshold', type=float, default=0.2)
    parser.add_argument('--top_mass_pct', type=float, default=0.5)
    parser.add_argument('--allow_edge_clamp', action='store_true', help='Legacy: clamp edge_type to [0,num_rel-1] instead of failing')
    parser.add_argument('--viz_percentile', type=int, default=80, help='Percentile (0-100) of |atom score| to color for viz')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--out_csv', default=os.path.join(script_dir, 'outputs', 'rgcn_xai_balanced_full_detailed_ensemble.csv'))
    parser.add_argument('--full', action='store_true', help='Process full dataset (otherwise sample 2 per class per type)')
    parser.add_argument('--samples_per_class', type=int, default=2, help='Used when not --full')
    parser.add_argument('--calibration_json', type=str, default=None, help='Optional temperature scaling JSON')
    parser.add_argument('--dump_dir', type=str, default=None, help='Optional directory to dump per-pair details')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--per_model', action='store_true', help='Run per-ckpt (individual) analysis; writes one CSV per checkpoint')
    parser.add_argument('--ensemble', action='store_true', help='Run ensemble analysis (averaged predictions + selected backbone for XAI)')
    parser.add_argument('--backbone_index', type=int, default=0, help='Backbone checkpoint index for ensemble XAI')
    parser.add_argument('--limit_per_class', type=int, default=None, help='Optional row cap per class for speed')
    parser.add_argument('--pharmacophore_json', type=str, default=None, help='Optional JSON file of pharmacophore SMARTS mapping')
    parser.add_argument('--ensemble_ckpts', type=str, default=None, help='Glob/pattern for ensemble checkpoints to average attribution over')
    args = parser.parse_args()

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

    # Resolve checkpoints: provided or discover
    ckpts = args.ckpt if args.ckpt else []
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

    # Config and models
    cfg = Configuration()
    dev = None
    if args.device:
        try:
            dev = torch.device(args.device)
        except Exception:
            print(f"Warning: invalid --device '{args.device}', falling back to auto")
            dev = None
    models_all = load_models(ckpts, cfg, device=dev)

    # Calibration temperatures
    temperatures: Optional[List[float]] = None
    calibrated = False
    if args.calibration_json and os.path.exists(args.calibration_json):
        try:
            with open(args.calibration_json, 'r') as f:
                calib = json.load(f)
            if isinstance(calib, dict) and 'temperature' in calib:
                T = float(calib['temperature'])
                temperatures = [T] * len(models)
                calibrated = True
            elif isinstance(calib, dict) and 'per_checkpoint' in calib:
                per = calib['per_checkpoint']
                temps = []
                for p in ckpts:
                    temps.append(float(per.get(p, per.get(os.path.basename(p), 1.0))))
                temperatures = temps
                calibrated = True
        except Exception as e:
            print(f"Warning: failed to parse calibration JSON: {e}")
            temperatures = None

    # Helper: single split run function
    def run_once_split(df_to_process: pd.DataFrame, pair_type_str: str, models: List[BaseGNN],
                       model_id: str, is_ens: bool, backbone_id: str, out_csv: str):
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
                    row, models, cfg, args.ig_steps, args.threshold, args.fg_norm,
                    args.xai_method, args.per_atom_occlusion,
                    model_id=model_id, is_ensemble=is_ens, backbone_id=backbone_id,
                    ph_smarts=PH_SMARTS if PH_SMARTS else None,
                    pharm_json_path=args.pharmacophore_json,
                    pharm_threshold=float(args.pharm_threshold),
                    top_mass_pct=float(args.top_mass_pct),
                    viz_percentile=int(args.viz_percentile),
                    ig_paths=int(args.ig_paths), ig_noise=float(args.ig_noise),
                    allow_edge_clamp=bool(args.allow_edge_clamp),
                    temperatures=temperatures, dump_dir=args.dump_dir, backbone_ckpt=backbone_id
                )
                out['is_cliff'] = is_cliff_flag
                rows.append(out)
            except Exception as e:
                print(f"ERROR processing {pair_type_str} pair at index {_}: {e}")
                import traceback
                traceback.print_exc()
                rows.append({
                    'pair_type': pair_type_str,
                    'error': str(e),
                    'is_cliff': is_cliff_flag,
                    'active_smiles': row.get('active_smiles', row.get('compound1_smiles', '')),
                    'inactive_smiles': row.get('inactive_smiles', row.get('compound2_smiles', '')),
                    'antibiotic_class': row.get('class', '')
                })

        out_df = pd.DataFrame(rows)
        # Diagnostic: edit detection mode distribution
        try:
            if 'edit_detection_mode' in out_df.columns and len(out_df) > 0:
                print("\nEdit Detection Mode Distribution:")
                vc = out_df['edit_detection_mode'].value_counts()
                total = len(out_df)
                for mode, count in vc.items():
                    pct = 100.0 * float(count) / float(total)
                    print(f"  {mode}: {count} pairs ({pct:.1f}%)")
        except Exception:
            pass
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        out_df.to_csv(out_csv, index=False, sep=',', quoting=csv.QUOTE_MINIMAL)
        try:
            out_df.to_parquet(os.path.splitext(out_csv)[0] + '.parquet', index=False)
        except Exception as e:
            print(f"Parquet save skipped: {e}")
        print(f"Saved: {out_csv} ({len(out_df)} rows)")
        # Replace on-disk CSV with standardized evaluation schema including substructure-level attributions
        def project_eval_schema(df: pd.DataFrame) -> pd.DataFrame:
            tmp = df.copy()
            # Decide schema based on split type (cliff vs non_cliff)
            is_cliff = False
            try:
                if 'pair_type' in tmp.columns:
                    is_cliff = str(tmp['pair_type'].iloc[0]).strip() == 'cliff'
            except Exception:
                pass

            if is_cliff:
                # IDs
                for a_key in ['active_compound_id', 'compound1_id', 'active_id']:
                    if a_key in tmp.columns and 'compound_active_id' not in tmp.columns:
                        tmp.rename(columns={a_key: 'compound_active_id'}, inplace=True)
                        break
                for b_key in ['inactive_compound_id', 'compound2_id', 'inactive_id']:
                    if b_key in tmp.columns and 'compound_inactive_id' not in tmp.columns:
                        tmp.rename(columns={b_key: 'compound_inactive_id'}, inplace=True)
                        break
                # Probabilities
                for pa in ['active_pred_prob', 'compound1_pred_prob']:
                    if pa in tmp.columns and 'compound_active_pred_prob' not in tmp.columns:
                        tmp.rename(columns={pa: 'compound_active_pred_prob'}, inplace=True)
                        break
                for pi in ['inactive_pred_prob', 'compound2_pred_prob']:
                    if pi in tmp.columns and 'compound_inactive_pred_prob' not in tmp.columns:
                        tmp.rename(columns={pi: 'compound_inactive_pred_prob'}, inplace=True)
                        break
                # Substructures
                # Prefer explicit active/inactive keys if present
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
                for a_key in ['active_compound_id', 'compound1_compound_id', 'active_id', 'compound1_id']:
                    if a_key in tmp.columns:
                        tmp.rename(columns={a_key: 'compound1_id'}, inplace=True)
                        break
                for b_key in ['inactive_compound_id', 'compound2_compound_id', 'inactive_id', 'compound2_id']:
                    if b_key in tmp.columns:
                        tmp.rename(columns={b_key: 'compound2_id'}, inplace=True)
                        break
                for pa in ['active_pred_prob', 'compound1_pred_prob']:
                    if pa in tmp.columns:
                        tmp.rename(columns={pa: 'compound1_pred_prob'}, inplace=True)
                        break
                for pi in ['inactive_pred_prob', 'compound2_pred_prob']:
                    if pi in tmp.columns:
                        tmp.rename(columns={pi: 'compound2_pred_prob'}, inplace=True)
                        break
                if 'substruct_attr_active' in tmp.columns:
                    tmp.rename(columns={'substruct_attr_active': 'substruct_attr_compound1'}, inplace=True)
                if 'substruct_attr_inactive' in tmp.columns:
                    tmp.rename(columns={'substruct_attr_inactive': 'substruct_attr_compound2'}, inplace=True)
                if 'compound1_pred_prob' in tmp.columns:
                    tmp['compound1_pred_class'] = (tmp['compound1_pred_prob'] >= 0.5).map({True: 'active', False: 'inactive'})
                if 'compound2_pred_prob' in tmp.columns:
                    tmp['compound2_pred_class'] = (tmp['compound2_pred_prob'] >= 0.5).map({True: 'active', False: 'inactive'})
                base_cols = [c for c in [
                    'pair_type','antibiotic_class','compound1_id','compound2_id',
                    'compound1_pred_prob','compound2_pred_prob',
                    'compound1_pred_class','compound2_pred_class',
                    'substruct_attr_compound1','substruct_attr_compound2',
                    'pos_substructs_compound1','neg_substructs_compound1',
                    'neutral_substructs_compound1',
                    'pos_substructs_compound2','neg_substructs_compound2',
                    'neutral_substructs_compound2',
                    'compound1_smiles','compound2_smiles',
                    'model_type'
                ] if c in tmp.columns]
                return tmp[base_cols]
        try:
            sdf = project_eval_schema(out_df)
            sdf.to_csv(out_csv, index=False)
            try:
                sdf.to_parquet(os.path.splitext(out_csv)[0] + '.parquet', index=False)
            except Exception:
                pass
        except Exception as e:
            print(f"Standard schema rewrite skipped: {e}")

    # Helper: legacy combined run function (DEPRECATED)
    def run_once(models: List[BaseGNN], model_id: str, is_ens: bool, backbone_id: str, out_csv: str):
        """DEPRECATED: Use run_once_split instead"""
        df_cliff = pd.read_csv(args.activity_csv)
        df_non = pd.read_csv(args.noncliff_csv)
        if args.limit_per_class is not None:
            take = int(args.limit_per_class)
            if 'class' in df_cliff.columns:
                df_cliff = df_cliff.sort_values('class').groupby('class', group_keys=False).head(take)
            if 'class' in df_non.columns:
                df_non = df_non.sort_values('class').groupby('class', group_keys=False).head(take)
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
        for _, row in df_cliff.iterrows():
            try:
                out = process_pair_row(
                    row, models, cfg, args.ig_steps, args.threshold, args.fg_norm,
                    args.xai_method, args.per_atom_occlusion,
                    model_id=model_id, is_ensemble=is_ens, backbone_id=backbone_id,
                    ph_smarts=PH_SMARTS if PH_SMARTS else None,
                    pharm_json_path=args.pharmacophore_json,
                    pharm_threshold=float(args.pharm_threshold),
                    top_mass_pct=float(args.top_mass_pct),
                    viz_percentile=int(args.viz_percentile),
                    ig_paths=int(args.ig_paths), ig_noise=float(args.ig_noise),
                    allow_edge_clamp=bool(args.allow_edge_clamp),
                    temperatures=temperatures, dump_dir=args.dump_dir, backbone_ckpt=backbone_id
                )
                out['is_cliff'] = True
                rows.append(out)
            except Exception as e:
                print(f"ERROR processing cliff pair at index {_}: {e}")
                import traceback
                traceback.print_exc()
                rows.append({
                    'pair_type': 'cliff',
                    'error': str(e),
                    'is_cliff': True,
                    'active_smiles': row.get('active_smiles', ''),
                    'inactive_smiles': row.get('inactive_smiles', ''),
                    'antibiotic_class': row.get('class', '')
                })
        for _, row in df_non.iterrows():
            try:
                out = process_pair_row(
                    row, models, cfg, args.ig_steps, args.threshold, args.fg_norm,
                    args.xai_method, args.per_atom_occlusion,
                    model_id=model_id, is_ensemble=is_ens, backbone_id=backbone_id,
                    ph_smarts=PH_SMARTS if PH_SMARTS else None,
                    pharm_json_path=args.pharmacophore_json,
                    pharm_threshold=float(args.pharm_threshold),
                    top_mass_pct=float(args.top_mass_pct),
                    viz_percentile=int(args.viz_percentile),
                    ig_paths=int(args.ig_paths), ig_noise=float(args.ig_noise),
                    allow_edge_clamp=bool(args.allow_edge_clamp),
                    temperatures=temperatures, dump_dir=args.dump_dir, backbone_ckpt=backbone_id
                )
                out['is_cliff'] = False
                rows.append(out)
            except Exception as e:
                print(f"ERROR processing non-cliff pair at index {_}: {e}")
                import traceback
                traceback.print_exc()
                rows.append({
                    'pair_type': 'noncliff',
                    'error': str(e),
                    'is_cliff': False,
                    'active_smiles': row.get('compound1_smiles', row.get('active_smiles', '')),
                    'inactive_smiles': row.get('compound2_smiles', row.get('inactive_smiles', '')),
                    'antibiotic_class': row.get('class', '')
                })
        out_df = pd.DataFrame(rows)
        # Diagnostic: edit detection mode distribution
        try:
            if 'edit_detection_mode' in out_df.columns and len(out_df) > 0:
                print("\nEdit Detection Mode Distribution:")
                vc = out_df['edit_detection_mode'].value_counts()
                total = len(out_df)
                for mode, count in vc.items():
                    pct = 100.0 * float(count) / float(total)
                    print(f"  {mode}: {count} pairs ({pct:.1f}%)")
        except Exception:
            pass
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        out_df.to_csv(out_csv, index=False, sep=',', quoting=csv.QUOTE_MINIMAL)
        try:
            out_df.to_parquet(os.path.splitext(out_csv)[0] + '.parquet', index=False)
        except Exception as e:
            print(f"Parquet save skipped: {e}")
        print(f"Saved: {out_csv} ({len(out_df)} rows)")

    # Decide modes and run
    do_per_model = args.per_model
    do_ensemble = args.ensemble or (not args.per_model)

    # Load data once (outside loops)
    print("\nLoading data files...")
    df_cliff = pd.read_csv(args.activity_csv)
    df_non = pd.read_csv(args.noncliff_csv)
    print(f"Loaded {len(df_cliff)} cliff pairs and {len(df_non)} non-cliff pairs")

    # Create output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    if do_per_model:
        for model_idx, (p, m) in enumerate(zip(ckpts, models_all), start=1):
            model_id = os.path.splitext(os.path.basename(p))[0]

            print(f"\n{'='*60}")
            print(f"Running per-model: Model {model_idx} ({model_id})")
            print(f"{'='*60}")

            # Process cliffs
            cliff_output = os.path.join(output_dir, f"rgcn_model{model_idx}_cliffs.csv")
            run_once_split(df_cliff.copy(), 'cliff', [m], model_id, False, model_id, cliff_output)

            # Process non-cliffs
            noncliff_output = os.path.join(output_dir, f"rgcn_model{model_idx}_non_cliffs.csv")
            run_once_split(df_non.copy(), 'non_cliff', [m], model_id, False, model_id, noncliff_output)

    if do_ensemble:
        bb = max(0, min(int(args.backbone_index), len(models_all)-1))
        models_bb = [models_all[bb]] + [m for i, m in enumerate(models_all) if i != bb]
        backbone_id = os.path.splitext(os.path.basename(ckpts[bb]))[0]

        print(f"\n{'='*60}")
        print(f"Running ensemble with backbone index {bb}: {backbone_id}")
        print(f"{'='*60}")

        # Process cliffs
        cliff_output = os.path.join(output_dir, "rgcn_ensemble_cliffs.csv")
        run_once_split(df_cliff.copy(), 'cliff', models_bb, 'ensemble', True, backbone_id, cliff_output)

        # Process non-cliffs
        noncliff_output = os.path.join(output_dir, "rgcn_ensemble_non_cliffs.csv")
        run_once_split(df_non.copy(), 'non_cliff', models_bb, 'ensemble', True, backbone_id, noncliff_output)


if __name__ == '__main__':
    main()


