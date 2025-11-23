from typing import List, Dict, Tuple, Optional

# Lightweight, SMILES-aware tokenizer + token→atom mapper
# Strategy:
# - Tokenize original SMILES deterministically handling bracket atoms, two-letter elements (Cl/Br), ring digits (including %nn), bonds, branches, stereo chars.
# - Count atom tokens in original text; map the nth atom token to RDKit atom index n (input-order heuristic).
# - Build token→atoms mapping: each atom token maps to a single atom index; non-atom tokens map to []
# - Provide utilities to convert token scores → atom scores and compute top-k token-to-atom recall.

import re
from rdkit import Chem
import numpy as np

_TWO_LETTER = {"Cl", "Br", "Si", "Se", "Na", "Li", "Al", "Ca", "Zn", "Mg", "Fe", "Cu", "Mn", "Ag", "Sn", "Pt", "Hg", "Pb", "Ni", "Co"}
_BOND = set("-=#:/.\\")

def tokenize_smiles(smiles: str) -> List[str]:
    s = smiles.strip()
    i = 0
    tokens: List[str] = []
    L = len(s)
    while i < L:
        c = s[i]
        # Bracket atom: [ ... ] possibly with map numbers/stereo
        if c == '[':
            j = i + 1
            depth = 1
            while j < L and depth > 0:
                if s[j] == '[':
                    depth += 1
                elif s[j] == ']':
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            tokens.append(s[i:j])
            i = j
            continue
        # Two-digit ring numbers like %12
        if c == '%':
            j = i + 1
            while j < L and s[j].isdigit():
                j += 1
            tokens.append(s[i:j])
            i = j
            continue
        # Two-letter elements
        if i + 1 < L and s[i:i+2] in _TWO_LETTER:
            tokens.append(s[i:i+2])
            i += 2
            continue
        # Branches
        if c in '()':
            tokens.append(c)
            i += 1
            continue
        # Bonds or ring digits
        if c in _BOND or c.isdigit() or c in '+-':
            tokens.append(c)
            i += 1
            continue
        # Single-letter elements/aromatics
        tokens.append(c)
        i += 1
    return tokens


def is_atom_token(tok: str) -> bool:
    if not tok:
        return False
    if tok[0] == '[':
        return True
    if tok in _TWO_LETTER:
        return True
    # Atomic symbols or aromatic lower-case letters
    if re.fullmatch(r"[A-Z][a-z]?", tok):
        return True
    if re.fullmatch(r"[cnops]", tok):
        return True
    return False


def build_token_to_atoms(smiles: str) -> Optional[List[List[int]]]:
    """
    Returns list of atom indices per token (empty list for non-atom tokens).
    Heuristic: nth atom token in text → RDKit atom index n (input-order assumption).
    If counts mismatch with RDKit atoms, return None.
    """
    toks = tokenize_smiles(smiles)
    atom_pos = [i for i, t in enumerate(toks) if is_atom_token(t)]
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        return None
    n_atoms = mol.GetNumAtoms()
    if len(atom_pos) != n_atoms:
        return None
    mapping: List[List[int]] = [[] for _ in toks]
    for rank, ti in enumerate(atom_pos):
        mapping[ti] = [rank]
    return mapping


def token_scores_to_atom_scores(smiles: str, token_scores: np.ndarray) -> Tuple[np.ndarray, float, str]:
    """
    Map token scores to atom scores using token-aware mapping. Returns (atom_scores, recall, warning).
    recall is computed over top-10% absolute tokens: fraction that map to atoms.
    warning: '' | 'fallback_span' | 'recall_low' | 'mapping_failed'
    """
    mapping = build_token_to_atoms(smiles)
    if mapping is None:
        # Graceful fail
        return np.array([]), 0.0, 'mapping_failed'
    toks = tokenize_smiles(smiles)
    atom_count = sum(1 for t in toks if is_atom_token(t))
    if atom_count <= 0:
        return np.array([]), 0.0, 'mapping_failed'
    # Align mapping length to token_scores length (pad/truncate)
    warn = ''
    L = len(token_scores)
    if len(mapping) < L:
        mapping = mapping + ([[]] * (L - len(mapping)))
    elif len(mapping) > L:
        mapping = mapping[:L]
        warn = 'mapping_truncated'

    atom_scores = np.zeros((atom_count,), dtype=float)
    # top-10% recall over non-PAD region (use full length here; PAD bins are [])
    k = max(1, int(round(0.10 * L)))
    ord_idx = np.argsort(-np.abs(token_scores))
    top_idx = ord_idx[:k]
    mapped_top = 0

    upto = min(len(mapping), L)
    for i in range(upto):
        # Extra bounds check for safety
        if i >= len(mapping):
            break
        atoms = mapping[i]
        if not atoms:
            continue
        # Bounds check for token_scores
        if i >= len(token_scores):
            break
        s = float(token_scores[i])
        w = s / float(len(atoms))
        for a in atoms:
            if 0 <= a < atom_count:
                atom_scores[a] += w

    for i in top_idx:
        if i < len(mapping) and i < len(token_scores) and mapping[i]:
            mapped_top += 1
    recall = float(mapped_top) / float(len(top_idx) if len(top_idx) else 1)
    return atom_scores, recall, warn
