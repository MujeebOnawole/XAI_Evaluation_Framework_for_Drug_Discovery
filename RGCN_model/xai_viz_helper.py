
"""
xai_viz_helper.py
Minimal helper to power the visualization notebook. It loads model outputs
(CSV/Parquet), normalizes minimal metadata, and provides simple filtering and
top-K substructure tables.
"""

import json, random, math, re
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

# Optional imports from a compare module; otherwise provide shims
try:
    from xai_unified_compare_v5 import try_parse_json, vectorize_attr, canon_key
except Exception:
    import json as _json
    def try_parse_json(s):
        if s is None:
            return {}
        if isinstance(s, (dict, list)):
            return s
        try:
            return _json.loads(s)
        except Exception:
            out = {}
            for part in str(s).split(','):
                if ':' in part:
                    k, v = part.split(':', 1)
                    try:
                        out[str(k).strip()] = float(v)
                    except Exception:
                        continue
            return out
    def vectorize_attr(obj, topk=None, use_abs=False):
        d = obj
        if not isinstance(d, (dict, list)):
            d = try_parse_json(obj)
        if isinstance(d, list):
            m = {}
            for it in d:
                if isinstance(it, dict):
                    k = str(it.get('substructure') or it.get('feature') or '')
                    v = float(it.get('attribution', it.get('support', 0.0)))
                    if k:
                        m[k] = m.get(k, 0.0) + (abs(v) if use_abs else v)
            d = m
        if not isinstance(d, dict):
            return {}
        vec = {str(k): float(abs(v) if use_abs else v) for k, v in d.items() if k is not None}
        if topk is not None and topk > 0:
            keys = sorted(vec.keys(), key=lambda k: abs(vec[k]), reverse=True)[:topk]
            return {k: vec[k] for k in keys}
        return vec
    def canon_key(s: str) -> str:
        return str(s or '').strip().lower()

try:
    from xai_unified_compare_v5 import infer_model, infer_pair_type, unify_frame
except Exception:
    import os as _os
    def infer_model(pathlike) -> str:
        p = str(pathlike).lower()
        if 'cnn_model' in p or 'cnn' in p:
            return 'CNN'
        if 'rgcn_model' in p or 'rgcn' in p:
            return 'RGCN'
        if 'rf_model' in p or 'randomforest' in p or 'rf_' in _os.path.basename(p).lower():
            return 'RandomForest'
        return 'Unknown'
    def infer_pair_type(pathlike) -> str:
        b = _os.path.basename(str(pathlike)).lower()
        if 'non_cliff' in b or 'noncliff' in b:
            return 'non_cliff'
        if 'cliff' in b:
            return 'cliff'
        return 'unknown'
    def unify_frame(df, model: str, pair_type: str, model_id: str):
        out = df.copy()
        if 'model' not in out.columns:
            out['model'] = model
        if 'pair_type' not in out.columns:
            out['pair_type'] = pair_type
        if 'model_id' not in out.columns:
            out['model_id'] = model_id
        out['pair_type'] = out['pair_type'].astype(str).str.strip().str.lower().map({'cliff':'cliff','non_cliff':'non_cliff','noncliff':'non_cliff'}).fillna('unknown')
        return out


def _extract_variant(model_id: str) -> str:
    s = str(model_id or '').lower()
    if 'ensemble' in s:
        return 'ensemble'
    m = re.search(r'model\s*([1-5])', s)
    if m:
        return f"model{m.group(1)}"
    m2 = re.search(r'cv\s*(\d+)\s*[_-]?\s*fold\s*(\d+)', s)
    if m2:
        return f"cv{m2.group(1)}_fold{m2.group(2)}"
    return 'unknown'


def load_pairs(files_or_glob: List[str]) -> pd.DataFrame:
    paths: List[Path] = []
    for x in files_or_glob:
        p = Path(x)
        if p.is_file():
            paths.append(p)
        else:
            paths.extend(list(Path('.').glob(x)))
    frames = []
    for p in paths:
        if p.suffix.lower() == '.parquet':
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        df = unify_frame(df, infer_model(p), infer_pair_type(p), p.name)
        # variant column
        if 'variant' not in df.columns:
            try:
                df['variant'] = df.get('model_id', p.name).apply(_extract_variant)
            except Exception:
                df['variant'] = _extract_variant(p.name)
        # stable pair_key for cross-model matching
        def _norm_id(x):
            s = str(x or '').strip().upper()
            return s
        def _pair_key_row(r) -> str:
            # Common schemas
            a = r.get('active_compound_id', None)
            b = r.get('inactive_compound_id', None)
            if pd.notna(a) and pd.notna(b):
                ids = sorted([_norm_id(a), _norm_id(b)])
                return f"{ids[0]}|{ids[1]}"
            c1 = r.get('compound1_id', None)
            c2 = r.get('compound2_id', None)
            if pd.notna(c1) and pd.notna(c2):
                ids = sorted([_norm_id(c1), _norm_id(c2)])
                return f"{ids[0]}|{ids[1]}"
            # Fallback: try generic active_id/inactive_id
            a2 = r.get('active_id', None)
            b2 = r.get('inactive_id', None)
            if pd.notna(a2) and pd.notna(b2):
                ids = sorted([_norm_id(a2), _norm_id(b2)])
                return f"{ids[0]}|{ids[1]}"
            return ''
        try:
            if 'pair_key' not in df.columns:
                df['pair_key'] = df.apply(_pair_key_row, axis=1)
        except Exception:
            pass
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def filter_pairs(df: pd.DataFrame,
                 model: Optional[str] = None,
                 pair_type: Optional[str] = None,
                 antibiotic_class: Optional[str] = None,
                 pair_key: Optional[str] = None,
                 variant: Optional[str] = None,
                 only_wrong: bool = False,
                 thr: Optional[float] = None) -> pd.DataFrame:
    sub = df.copy()
    if pair_key:
        sub = sub[sub.get('pair_key','') == pair_key]
    if model:
        sub = sub[sub['model'] == model]
    if pair_type:
        sub = sub[sub['pair_type'] == pair_type]
    if antibiotic_class:
        sub = sub[sub['antibiotic_class'] == antibiotic_class]
    if variant and 'variant' in sub.columns:
        sub = sub[sub['variant'] == variant]
    if thr is not None and 'prob_gap' in sub.columns:
        sub = sub[sub['prob_gap'] >= thr]
    if only_wrong and 'pair_classification' in sub.columns:
        wrong = ~sub['pair_classification'].isin(['BothCorrect', 'AA', 'II', 'OneCorrect'])
        sub = sub[wrong]
    return sub


def topk_substruct_table(row: pd.Series, k: int = 8) -> pd.DataFrame:
    a = row.get('substruct_attr_active') or row.get('pharm_feature_scores_active') or row.get('feature_scores_active')
    i = row.get('substruct_attr_inactive') or row.get('pharm_feature_scores_inactive') or row.get('feature_scores_inactive')
    a_map = vectorize_attr(a, topk=None, use_abs=True)
    i_map = vectorize_attr(i, topk=None, use_abs=True)
    keys = sorted(set(a_map) | set(i_map), key=lambda k: max(a_map.get(k, 0.0), i_map.get(k, 0.0)), reverse=True)[:k]
    data = []
    for key in keys:
        data.append({'substructure': key,
                     'active_attr': a_map.get(key, 0.0),
                     'inactive_attr': i_map.get(key, 0.0),
                     'delta': a_map.get(key, 0.0) - i_map.get(key, 0.0)})
    return pd.DataFrame(data)


def pick_example(df: pd.DataFrame,
                 model: str,
                 pair_type: str,
                 antibiotic_class: Optional[str] = None,
                 pair_key: Optional[str] = None,
                 variant: Optional[str] = None,
                 correctness: str = 'BothCorrect',
                 rng_seed: int = 0) -> Optional[pd.Series]:
    random.seed(rng_seed)
    sub = filter_pairs(df, model=model, pair_type=pair_type, antibiotic_class=antibiotic_class, pair_key=pair_key, variant=variant)
    if 'pair_classification' in sub.columns and correctness:
        sub = sub[sub['pair_classification'] == correctness]
    if len(sub) == 0:
        return None
    return sub.sample(1, random_state=rng_seed).iloc[0]


def make_pair_key_from_ids(ids: List[str]) -> Optional[str]:
    if not ids or len(ids) < 2:
        return None
    a, b = ids[0], ids[1]
    s = sorted([str(a).strip().upper(), str(b).strip().upper()])
    return f"{s[0]}|{s[1]}"


def extract_pair_ids(row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    """Return (id1, id2) for the row using common schema columns.
    Order is canonical (sorted uppercase) for stable reuse.
    """
    def _norm(x):
        return str(x).strip().upper() if x is not None and str(x).strip() else None
    a = row.get('active_compound_id', None)
    b = row.get('inactive_compound_id', None)
    if pd.notna(a) and pd.notna(b):
        ids = sorted([_norm(a), _norm(b)])
        return ids[0], ids[1]
    c1 = row.get('compound1_id', None)
    c2 = row.get('compound2_id', None)
    if pd.notna(c1) and pd.notna(c2):
        ids = sorted([_norm(c1), _norm(c2)])
        return ids[0], ids[1]
    a2 = row.get('active_id', None)
    b2 = row.get('inactive_id', None)
    if pd.notna(a2) and pd.notna(b2):
        ids = sorted([_norm(a2), _norm(b2)])
        return ids[0], ids[1]
    return None, None


def substructure_attr_to_atom_scores(substruct_json_data, num_atoms: int):
    """
    Convert CNN/RGCN substructure attributions to per-atom scores for visualization.

    Args:
        substruct_json_data: JSON string or parsed list of substructure attributions
        num_atoms: Number of atoms in the molecule

    Returns:
        numpy array of per-atom attribution scores

    Handles overlapping substructures by summing contributions.
    Each substructure's attribution is distributed equally among its atoms.
    """
    import numpy as np

    atom_scores = np.zeros(num_atoms, dtype=float)

    try:
        # Parse JSON if string
        if isinstance(substruct_json_data, str):
            data = try_parse_json(substruct_json_data)
        else:
            data = substruct_json_data

        if not isinstance(data, list):
            return atom_scores

        # Process each substructure
        for item in data:
            if not isinstance(item, dict):
                continue

            atoms = item.get('atoms', [])
            attribution = item.get('attribution', 0.0)

            # Distribute attribution equally to all atoms in substructure
            if atoms and attribution != 0:
                per_atom_value = float(attribution) / len(atoms)
                for atom_idx in atoms:
                    if 0 <= atom_idx < num_atoms:
                        atom_scores[atom_idx] += per_atom_value

        return atom_scores.tolist()

    except Exception as e:
        # Silently fail and return zeros
        return atom_scores.tolist()


def draw_pair(row: pd.Series, width: int = 650, height: int = 240):
    """Return an HTML (SVG) string rendering Active and Inactive molecules side-by-side.
    Uses RDKit MolDraw2D with strong, filled highlights (blue=positive, orange=negative).

    Prioritizes substructure attributions (substruct_attr_*) over token-level attributions
    (viz_*_atom_attr) for CNN/RGCN models to ensure visual consistency with reported explanations.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except Exception:
        return None

    def _parse_scores(v) -> List[float]:
        if isinstance(v, list):
            return [float(x) for x in v]
        try:
            o = try_parse_json(v)
            if isinstance(o, list):
                return [float(x) for x in o]
        except Exception:
            pass
        return []

    def _svg(smiles: str, scores: List[float], heading: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None or not scores or len(scores) != mol.GetNumAtoms():
            return None
        import numpy as _np
        arr = [float(x) for x in scores]
        # Robust scaling to emphasize useful dynamic range
        abs_vals = _np.abs(_np.asarray(arr, dtype=float))
        max_abs = float(abs_vals.max()) if abs_vals.size else 1e-9
        p95 = float(_np.percentile(abs_vals, 95)) if abs_vals.size else 0.0
        denom = max(1e-9, p95 if p95 > 1e-12 else max_abs)
        BLUE = (31/255.0, 119/255.0, 180/255.0)
        ORANGE = (1.0, 127/255.0, 14/255.0)
        atom_cols = {}
        atom_rads = {}
        bond_cols = {}
        # Strong fill with intensity map
        for i, s in enumerate(arr):
            if s == 0:
                continue
            # Normalize by robust denominator and apply gamma for contrast
            inten = min(abs(s)/denom, 1.0)
            inten = pow(inten, 0.6)  # gamma < 1 boosts mid-range intensities
            base = BLUE if s > 0 else ORANGE
            c = (1.0*(1-inten) + base[0]*inten,
                 1.0*(1-inten) + base[1]*inten,
                 1.0*(1-inten) + base[2]*inten)
            atom_cols[i] = c
            atom_rads[i] = 0.3 + 0.5*inten  # scale halo radius with intensity (0.3–0.8)
        # Color bonds if either end highlighted
        for b in mol.GetBonds():
            a = b.GetBeginAtomIdx(); e = b.GetEndAtomIdx()
            if a in atom_cols or e in atom_cols:
                bond_cols[b.GetIdx()] = atom_cols.get(a, atom_cols.get(e))

        d = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = d.drawOptions()
        opts.useBWAtomPalette()  # keep atomic colors subtle; highlights dominate
        opts.fillHighlights = True
        opts.highLightColour = (0.95, 0.95, 0.95)
        # Wider highlighted bonds to better convey importance
        opts.highlightBondWidthMult = 25
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol,
            highlightAtoms=list(atom_cols.keys()),
            highlightBonds=list(bond_cols.keys()),
            highlightAtomColors=atom_cols,
            highlightBondColors=bond_cols,
            highlightAtomRadii=atom_rads,
        )
        d.FinishDrawing()
        svg = d.GetDrawingText()
        # add heading above via simple HTML wrapper
        return f"<div style='text-align:center;font-family:Arial; font-size:13px; font-weight:600; margin-bottom:2px'>{heading}</div>" + svg

    # Get SMILES first to determine molecule size
    a_smiles = row.get('active_smiles', '')
    i_smiles = row.get('inactive_smiles', '')

    # Determine model type
    model = str(row.get('model', '')).upper()

    # Priority 1: Use substructure attributions for CNN/RGCN (more coherent)
    # Priority 2: Fall back to viz_atom_attr (token-mapped or other methods)
    a_scores = []
    i_scores = []

    if model in ['CNN', 'RGCN'] and 'substruct_attr_active' in row and row.get('substruct_attr_active'):
        # Try to use substructure attributions for CNN/RGCN
        try:
            mol_a = Chem.MolFromSmiles(str(a_smiles))
            if mol_a:
                a_scores = substructure_attr_to_atom_scores(
                    row.get('substruct_attr_active'),
                    mol_a.GetNumAtoms()
                )
        except Exception:
            pass

    if model in ['CNN', 'RGCN'] and 'substruct_attr_inactive' in row and row.get('substruct_attr_inactive'):
        try:
            mol_i = Chem.MolFromSmiles(str(i_smiles))
            if mol_i:
                i_scores = substructure_attr_to_atom_scores(
                    row.get('substruct_attr_inactive'),
                    mol_i.GetNumAtoms()
                )
        except Exception:
            pass

    # Fallback to viz_atom_attr if substructure conversion failed or not available
    if not a_scores:
        a_scores = _parse_scores(row.get('viz_active_atom_attr'))
    if not i_scores:
        i_scores = _parse_scores(row.get('viz_inactive_atom_attr'))
    # Legend with counts of positive/negative atoms
    def _counts(scores: List[float]):
        pos = sum(1 for s in scores if s > 0)
        neg = sum(1 for s in scores if s < 0)
        return pos, neg
    pa = float(row.get('active_pred_prob', 'nan')) if str(row.get('active_pred_prob'))!='nan' else float('nan')
    pi = float(row.get('inactive_pred_prob', 'nan')) if str(row.get('inactive_pred_prob'))!='nan' else float('nan')
    ac = str(row.get('active_classification', '') or '').strip().upper()
    ic = str(row.get('inactive_classification', '') or '').strip().upper()
    ac_tag = f" [{ac}]" if ac else ''
    ic_tag = f" [{ic}]" if ic else ''
    cpa = _counts(a_scores); cpi = _counts(i_scores)
    a_head = f"ACTIVE {row.get('active_compound_id','')}: P={pa:.3f}{ac_tag} ({cpa[0]}+/{cpa[1]}-)"
    i_head = f"INACTIVE {row.get('inactive_compound_id','')}: P={pi:.3f}{ic_tag} ({cpi[0]}+/{cpi[1]}-)"
    svg_a = _svg(a_smiles, a_scores, a_head)
    svg_i = _svg(i_smiles, i_scores, i_head)
    if svg_a is None and svg_i is None:
        return None
    # Combine side-by-side in HTML
    left = svg_a or ""
    right = svg_i or ""
    html = f"""
    <div style='display:flex; gap:16px; align-items:flex-start;'>
      <div style='width:{width}px'>{left}</div>
      <div style='width:{width}px'>{right}</div>
    </div>
    """
    return html


# === High-level utilities appended by ChatGPT on request ===

import pandas as _pd
import matplotlib.pyplot as _plt

def _closest_cnn_variant(summary_by_variant_path: str, summary_all_path: str) -> str:
    """Return model_id of the CNN variant closest to the aggregate CNN metrics
    (by |CSI_AUROC diff| + |CSPD diff|). If file missing or no CNN rows, return ''.
    """
    try:
        sv = _pd.read_csv(summary_by_variant_path)
        sa = _pd.read_csv(summary_all_path)
    except Exception:
        return ''
    if 'model' not in sa.columns or 'model' not in sv.columns:
        return ''
    if not (sa['model'] == 'CNN').any():
        return ''
    target = sa[sa['model']=='CNN'].iloc[0]
    target_csi  = float(target.get('CSI_AUROC', float('nan')))
    target_cspd = float(target.get('CSPD', float('nan')))
    sv_cnn = sv[sv['model']=='CNN'].copy()
    if 'model_id' not in sv_cnn.columns:
        return ''
    # Some tables have an explicit 'ensemble' row; prefer that if present.
    if (sv_cnn.get('model_id','')=='ensemble').any():
        return 'ensemble'
    sv_cnn['dist'] = (sv_cnn['CSI_AUROC']-target_csi).abs() + (sv_cnn['CSPD']-target_cspd).abs()
    pick = sv_cnn.sort_values('dist').head(1)
    return str(pick['model_id'].iloc[0]) if len(pick) else ''


def build_frames(
    rf_cliffs: str, rf_noncliffs: str,
    rgcn_cliffs: str, rgcn_noncliffs: str,
    cnn_cliffs: str, cnn_noncliffs: str,
    summary_by_variant: str = "/mnt/data/xai_unified_summary_by_variant.csv",
    summary_all: str = "/mnt/data/xai_unified_summary.csv",
    require_visible: bool = True,
):
    """Load and pre-filter RF/RGCN ensembles and a CNN set (ensemble if available,
    else closest representative). Returns dict with keys 'RF','RGCN','CNN'."""
    rf_df   = load_pairs(rf_cliffs, rf_noncliffs)
    rgcn_df = load_pairs(rgcn_cliffs, rgcn_noncliffs)
    cnn_df  = load_pairs(cnn_cliffs, cnn_noncliffs)

    # Prefer CNN ensemble if present; else pick closest representative model_id
    cnn_model_pick = None
    if 'model_id' in cnn_df.columns and (cnn_df['model_id']=='ensemble').any():
        cnn_model_pick = 'ensemble'
    else:
        cnn_model_pick = _closest_cnn_variant(summary_by_variant, summary_all) or None
    if cnn_model_pick:
        cnn_df = cnn_df[cnn_df.get('model_id','') == cnn_model_pick]

    # Require rows we can visualize
    if require_visible and 'xai_visible' in rf_df.columns:
        rf_df = rf_df.query('xai_visible == True')
    if require_visible and 'xai_visible' in rgcn_df.columns:
        rgcn_df = rgcn_df.query('xai_visible == True')
    if require_visible and 'xai_visible' in cnn_df.columns:
        cnn_df = cnn_df.query('xai_visible == True')

    return {'RF': rf_df, 'RGCN': rgcn_df, 'CNN': cnn_df, 'cnn_model_pick': cnn_model_pick}


def pick_examples(df: _pd.DataFrame, n_per_type: int = 1):
    """Pick up to n_per_type examples for cliff and non_cliff with highest prediction gap."""
    if df is None or df.empty:
        return []
    col_gap = 'prediction_difference' if 'prediction_difference' in df.columns else \
              ('prob_gap' if 'prob_gap' in df.columns else None)
    rows = []
    for t in ['cliff', 'non_cliff']:
        sub = filter_pairs(df, pair_type=t)
        if sub.empty:
            continue
        if col_gap and col_gap in sub.columns:
            sub = sub.sort_values(col_gap, ascending=False)
        rows.extend(sub.head(n_per_type).to_dict('records'))
    return rows


BADGE_COLS_PUBLIC = [
    # Recognition / consistency
    'pharm_recognition_score_pair',
    'pharmacophore_consistency_score',
    'pharm_recognition_rate_active',
    'pharm_recognition_rate_inactive',
    'pharmacophore_inconsistent_flag',
    # meta
    'xai_method'
]


def render_badges(row: dict, cols = BADGE_COLS_PUBLIC) -> str:
    """Return an HTML string with compact badges for selected row fields."""
    items = []
    for k in cols:
        if k in row:
            v = row.get(k)
            items.append(f"<span style='border:1px solid #ddd;padding:4px 8px;border-radius:8px'>{k}: <b>{v}</b></span>")
    return "<div style='display:flex;gap:12px;flex-wrap:wrap'>" + "".join(items) + "</div>"


def plot_similarity_vs_pharm(df: _pd.DataFrame, sim_col='pair_similarity', pharm_col='pharm_recognition_score_pair'):
    """Matplotlib scatter: x=similarity, y=pharm score, marker by pair_type (no explicit colors)."""
    if df is None or df.empty:
        return
    _plt.figure()
    for t, m in [('cliff', 'o'), ('non_cliff', '^')]:
        sub = filter_pairs(df, pair_type=t)
        if sim_col in sub.columns and pharm_col in sub.columns:
            _plt.scatter(sub[sim_col], sub[pharm_col], marker=m, label=t)  # no color set
    _plt.xlabel(sim_col)
    _plt.ylabel(pharm_col)
    _plt.legend()
    _plt.title('Similarity vs Pharmacophore Recognition')
    _plt.show()


def show_dashboard(frames: dict, k_top: int = 8):
    """Display a compact dashboard: one cliff + one non-cliff per arch,
    with badges, SVG, and top-K table. Also draw one scatter per arch to avoid subplotting."""
    import pandas as _pd
    from IPython.display import HTML, display
    for arch in ['RF','RGCN','CNN']:
        df = frames.get(arch)
        if df is None or df.empty:
            continue
        rows = pick_examples(df, n_per_type=1)
        for row in rows:
            display(HTML(f"<h3>{arch} — {row.get('pair_type')} — {row.get('antibiotic_class')}</h3>"))
            display(HTML(render_badges(row)))
            svg = draw_pair(row)  # uses existing strong highlight renderer
            if svg: display(HTML(svg))
            tbl = topk_substruct_table(row, k=k_top)
            if isinstance(tbl, _pd.DataFrame):
                display(tbl)

        # one scatter per arch
        plot_similarity_vs_pharm(df)

# ------------------- XAI CSV EXPORTS FOR UNRELIABLE CASES -------------------
import pandas as _pd
import numpy as _np
from pathlib import Path as _Path

def _vx(x):
    """Coerce the helper's JSON/string/None field to a dict[str,float] (signed)."""
    # Prefer the existing vectorize_attr() if available.
    try:
        return vectorize_attr(x, topk=None, use_abs=False)  # keep sign
    except Exception:
        return x if isinstance(x, dict) else {}

def _extract_substruct_maps(row):
    """
    Return (active_map, inactive_map) from whatever columns we store.
    Update the candidate lists if your helper uses different names.
    """
    cand_active   = ["substruct_attr_active", "pharm_feature_scores_active", "feature_scores_active"]
    cand_inactive = ["substruct_attr_inactive", "pharm_feature_scores_inactive", "feature_scores_inactive"]

    A = None
    I = None
    for k in cand_active:
        if k in row and row[k] is not None:
            A = _vx(row[k]); break
    for k in cand_inactive:
        if k in row and row[k] is not None:
            I = _vx(row[k]); break
    return (A or {}), (I or {})

def export_topk_substructures_csv(row, out_path, model_name, topk=12):
    """
    Write a compact CSV for one MODEL on the currently displayed PAIR.
    Columns: substructure, active_attr, inactive_attr, delta, abs_max,
             polarity_active (+1/-1), polarity_inactive (+1/-1), rank,
             plus metadata (pair_type, antibiotic_class, pair_key, variant, active_compound_id, inactive_compound_id, model)
    """
    A, I = _extract_substruct_maps(row)
    if not A and not I:
        return False

    keys = sorted(
        set(A) | set(I),
        key=lambda k: max(abs(float(A.get(k, 0.0))), abs(float(I.get(k, 0.0)))),
        reverse=True
    )
    if topk is not None:
        keys = keys[:int(topk)]

    data = []
    for k in keys:
        a = float(A.get(k, 0.0))
        i = float(I.get(k, 0.0))
        data.append({
            "substructure": k,
            "active_attr": a,
            "inactive_attr": i,
            "delta": a - i,
            "abs_max": max(abs(a), abs(i)),
            "polarity_active":   1 if a >= 0 else -1,
            "polarity_inactive": 1 if i >= 0 else -1,
        })

    df = _pd.DataFrame(data).sort_values("abs_max", ascending=False)
    df["rank"] = _np.arange(1, len(df) + 1)

    for col in ["pair_type","antibiotic_class","pair_key","variant",
                "active_compound_id","inactive_compound_id"]:
        df[col] = row.get(col, None)
    df["model"] = model_name

    _Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return True


# === ENHANCED VISUALIZATION FUNCTIONS ===

def _compute_shared_normalization(rows: List[pd.Series]) -> float:
    """Compute a shared normalization factor across multiple rows for consistent coloring."""
    import numpy as np
    all_scores = []
    for row in rows:
        if row is None:
            continue
        for col in ['viz_active_atom_attr', 'viz_inactive_atom_attr']:
            scores = _parse_atom_scores(row.get(col))
            if scores:
                all_scores.extend([abs(x) for x in scores])
    
    if not all_scores:
        return 1.0
    
    arr = np.array(all_scores)
    p95 = float(np.percentile(arr, 95)) if len(arr) > 0 else 0.0
    max_abs = float(arr.max()) if len(arr) > 0 else 1e-9
    return max(1e-9, p95 if p95 > 1e-12 else max_abs)


def _parse_atom_scores(v) -> List[float]:
    """Parse atom attribution scores from various formats."""
    if isinstance(v, list):
        return [float(x) for x in v]
    try:
        o = try_parse_json(v)
        if isinstance(o, list):
            return [float(x) for x in o]
    except Exception:
        pass
    return []


def _detect_edit_mask(row: pd.Series) -> Tuple[List[int], List[int]]:
    """Detect edit regions for active and inactive molecules based on SMILES differences."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except Exception:
        return [], []
    
    active_smiles = row.get('active_smiles', '')
    inactive_smiles = row.get('inactive_smiles', '')
    
    if not active_smiles or not inactive_smiles:
        return [], []
    
    try:
        mol_a = Chem.MolFromSmiles(str(active_smiles))
        mol_i = Chem.MolFromSmiles(str(inactive_smiles))
        
        if mol_a is None or mol_i is None:
            return [], []
        
        # Simple heuristic: atoms with different environments are likely in edit regions
        # This is a placeholder - you might have better edit detection logic
        fp_a = rdMolDescriptors.GetMorganFingerprint(mol_a, 2)
        fp_i = rdMolDescriptors.GetMorganFingerprint(mol_i, 2)
        
        edit_a, edit_i = [], []
        
        # Mark atoms that have different local environments
        for i in range(mol_a.GetNumAtoms()):
            env_a = rdMolDescriptors.GetMorganFingerprint(mol_a, 1, fromAtoms=[i])
            # Simplified - in practice you'd want better alignment
            if i < mol_i.GetNumAtoms():
                env_i = rdMolDescriptors.GetMorganFingerprint(mol_i, 1, fromAtoms=[i])
                if env_a != env_i:
                    edit_a.append(i)
        
        for i in range(mol_i.GetNumAtoms()):
            env_i = rdMolDescriptors.GetMorganFingerprint(mol_i, 1, fromAtoms=[i])
            if i < mol_a.GetNumAtoms():
                env_a = rdMolDescriptors.GetMorganFingerprint(mol_a, 1, fromAtoms=[i])
                if env_a != env_i:
                    edit_i.append(i)
        
        return edit_a, edit_i
    except Exception:
        return [], []


def _compute_delta_bars(row: pd.Series, edit_mask_a: List[int], edit_mask_i: List[int]) -> Tuple[float, float]:
    """Compute edit vs context attribution masses for delta bar visualization."""
    a_scores = _parse_atom_scores(row.get('viz_active_atom_attr'))
    i_scores = _parse_atom_scores(row.get('viz_inactive_atom_attr'))
    
    def _split_mass(scores: List[float], edit_indices: List[int]) -> Tuple[float, float]:
        if not scores:
            return 0.0, 0.0
        edit_mass = sum(abs(scores[i]) for i in edit_indices if i < len(scores))
        total_mass = sum(abs(x) for x in scores)
        context_mass = total_mass - edit_mass
        return edit_mass, context_mass
    
    edit_a, context_a = _split_mass(a_scores, edit_mask_a)
    edit_i, context_i = _split_mass(i_scores, edit_mask_i)
    
    # Compute normalized delta (edit fraction difference)
    total_a = edit_a + context_a
    total_i = edit_i + context_i
    
    frac_a = edit_a / max(total_a, 1e-9)
    frac_i = edit_i / max(total_i, 1e-9)
    
    delta_a = frac_a - 0.5  # deviation from balanced (0.5)
    delta_i = frac_i - 0.5
    
    return delta_a, delta_i


def draw_pair_enhanced(row: pd.Series, width: int = 650, height: int = 240, 
                      shared_norm: Optional[float] = None, show_delta_bar: bool = True):
    """Enhanced version of draw_pair with cross-model normalization, edit masks, and delta bars."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        import numpy as np
    except Exception:
        return None

    def _svg_enhanced(smiles: str, scores: List[float], heading: str, edit_mask: List[int], 
                     delta_val: float) -> Optional[str]:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None or not scores or len(scores) != mol.GetNumAtoms():
            return None
        
        arr = np.array([float(x) for x in scores])
        
        # Use shared normalization if provided
        denom = shared_norm if shared_norm else max(1e-9, float(np.percentile(np.abs(arr), 95)))
        
        BLUE = (31/255.0, 119/255.0, 180/255.0)
        ORANGE = (1.0, 127/255.0, 14/255.0)
        EDIT_OUTLINE = (0.8, 0.0, 0.8)  # Purple outline for edit regions
        
        atom_cols = {}
        atom_rads = {}
        bond_cols = {}
        
        # Attribution coloring
        for i, s in enumerate(arr):
            if s == 0:
                continue
            inten = min(abs(s)/denom, 1.0)
            inten = pow(inten, 0.6)  # gamma correction
            base = BLUE if s > 0 else ORANGE
            c = (1.0*(1-inten) + base[0]*inten,
                 1.0*(1-inten) + base[1]*inten,
                 1.0*(1-inten) + base[2]*inten)
            atom_cols[i] = c
            atom_rads[i] = 0.3 + 0.5*inten
        
        # Bond coloring
        for b in mol.GetBonds():
            a = b.GetBeginAtomIdx(); e = b.GetEndAtomIdx()
            if a in atom_cols or e in atom_cols:
                bond_cols[b.GetIdx()] = atom_cols.get(a, atom_cols.get(e))

        d = rdMolDraw2D.MolDraw2DSVG(width, height - (40 if show_delta_bar else 0))
        opts = d.drawOptions()
        opts.useBWAtomPalette()
        opts.fillHighlights = True
        opts.highlightBondWidthMult = 25
        
        # Add edit mask highlighting (dashed outline)
        edit_highlight_colors = {i: EDIT_OUTLINE for i in edit_mask}
        
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol,
            highlightAtoms=list(atom_cols.keys()),
            highlightBonds=list(bond_cols.keys()),
            highlightAtomColors=atom_cols,
            highlightBondColors=bond_cols,
            highlightAtomRadii=atom_rads,
        )
        
        # TODO: Add dashed outline for edit regions (would need custom SVG manipulation)
        
        d.FinishDrawing()
        svg = d.GetDrawingText()
        
        result = f"<div style='text-align:center;font-family:Arial; font-size:13px; font-weight:600; margin-bottom:2px'>{heading}</div>" + svg
        
        # Add delta bar if requested
        if show_delta_bar:
            bar_color = '#ff6b35' if delta_val > 0 else '#4a90e2'
            bar_width = min(abs(delta_val) * 200, 100)  # scale to pixels
            result += f"""
            <div style='text-align:center; margin-top:4px'>
                <div style='display:inline-block; width:200px; height:8px; background:#eee; border-radius:4px; position:relative'>
                    <div style='position:absolute; left:100px; width:{bar_width}px; height:8px; background:{bar_color}; 
                               border-radius:4px; transform:translateX({'-100%' if delta_val < 0 else '0'})'>
                    </div>
                </div>
                <div style='font-size:10px; margin-top:2px'>Δ={delta_val:.3f}</div>
            </div>
            """
        
        return result

    # Get SMILES first
    a_smiles = row.get('active_smiles', '')
    i_smiles = row.get('inactive_smiles', '')

    # Determine model type
    model = str(row.get('model', '')).upper()

    # Priority 1: Use substructure attributions for CNN/RGCN (more coherent)
    # Priority 2: Fall back to viz_atom_attr (token-mapped or other methods)
    a_scores = []
    i_scores = []

    if model in ['CNN', 'RGCN'] and 'substruct_attr_active' in row and row.get('substruct_attr_active'):
        # Try to use substructure attributions for CNN/RGCN
        try:
            mol_a = Chem.MolFromSmiles(str(a_smiles))
            if mol_a:
                a_scores = substructure_attr_to_atom_scores(
                    row.get('substruct_attr_active'),
                    mol_a.GetNumAtoms()
                )
        except Exception:
            pass

    if model in ['CNN', 'RGCN'] and 'substruct_attr_inactive' in row and row.get('substruct_attr_inactive'):
        try:
            mol_i = Chem.MolFromSmiles(str(i_smiles))
            if mol_i:
                i_scores = substructure_attr_to_atom_scores(
                    row.get('substruct_attr_inactive'),
                    mol_i.GetNumAtoms()
                )
        except Exception:
            pass

    # Fallback to viz_atom_attr if substructure conversion failed or not available
    if not a_scores:
        a_scores = _parse_atom_scores(row.get('viz_active_atom_attr'))
    if not i_scores:
        i_scores = _parse_atom_scores(row.get('viz_inactive_atom_attr'))
    
    # Detect edit regions
    edit_mask_a, edit_mask_i = _detect_edit_mask(row)
    
    # Compute delta bars
    delta_a, delta_i = _compute_delta_bars(row, edit_mask_a, edit_mask_i) if show_delta_bar else (0, 0)
    
    # Build headings
    def _counts(scores: List[float]):
        pos = sum(1 for s in scores if s > 0)
        neg = sum(1 for s in scores if s < 0)
        return pos, neg
    
    pa = float(row.get('active_pred_prob', 'nan')) if str(row.get('active_pred_prob'))!='nan' else float('nan')
    pi = float(row.get('inactive_pred_prob', 'nan')) if str(row.get('inactive_pred_prob'))!='nan' else float('nan')
    ac = str(row.get('active_classification', '') or '').strip().upper()
    ic = str(row.get('inactive_classification', '') or '').strip().upper()
    ac_tag = f" [{ac}]" if ac else ''
    ic_tag = f" [{ic}]" if ic else ''
    
    cpa = _counts(a_scores); cpi = _counts(i_scores)
    a_head = f"ACTIVE {row.get('active_compound_id','')}: P={pa:.3f}{ac_tag} ({cpa[0]}+/{cpa[1]}-)"
    i_head = f"INACTIVE {row.get('inactive_compound_id','')}: P={pi:.3f}{ic_tag} ({cpi[0]}+/{cpi[1]}-)"
    
    svg_a = _svg_enhanced(a_smiles, a_scores, a_head, edit_mask_a, delta_a)
    svg_i = _svg_enhanced(i_smiles, i_scores, i_head, edit_mask_i, delta_i)
    
    if svg_a is None and svg_i is None:
        return None
    
    # Combine side-by-side
    left = svg_a or ""
    right = svg_i or ""
    html = f"""
    <div style='display:flex; gap:16px; align-items:flex-start;'>
      <div style='width:{width}px'>{left}</div>
      <div style='width:{width}px'>{right}</div>
    </div>
    """
    return html


def create_thumbnail_scatter(df: pd.DataFrame, current_row: pd.Series, 
                           sim_col: str = 'pair_similarity', 
                           pharm_col: str = 'pharm_recognition_score_pair') -> Optional[str]:
    """Create a small thumbnail scatter plot highlighting the current example."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import io
        import base64
        
        if df is None or df.empty or sim_col not in df.columns or pharm_col not in df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(2.2, 1.5))
        
        # Plot all points
        for pair_type, marker in [('cliff', 'o'), ('non_cliff', '^')]:
            sub = df[df['pair_type'] == pair_type]
            if not sub.empty:
                ax.scatter(sub[sim_col], sub[pharm_col], 
                          marker=marker, alpha=0.6, s=20, label=pair_type)
        
        # Highlight current point
        if sim_col in current_row and pharm_col in current_row:
            curr_sim = current_row[sim_col]
            curr_pharm = current_row[pharm_col]
            curr_type = current_row.get('pair_type', 'unknown')
            marker = 'o' if curr_type == 'cliff' else '^'
            ax.scatter([curr_sim], [curr_pharm], marker=marker, 
                      s=80, color='red', edgecolor='black', linewidth=2, zorder=10)
        
        ax.set_xlabel(sim_col, fontsize=8)
        ax.set_ylabel(pharm_col, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
        
        # Convert to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return f'<img src="data:image/png;base64,{img_b64}" style="width:220px;height:150px;border:1px solid #ddd">'
        
    except Exception:
        return None


def show_enhanced_dashboard(
    input_patterns: List[str],
    pair_type: str = 'cliff',
    antibiotic_class: Optional[str] = None,
    models: List[str] = ['RandomForest', 'CNN', 'RGCN'],
    prefer_ensemble: bool = True,
    selection_strategy: str = 'highest_prediction_difference',
    similarity_col: Optional[str] = None,
    compound_ids: Optional[List[str]] = None,
    rng_seed: int = 0
):
    """Enhanced dashboard with all improvements: shared normalization, edit masks, delta bars, badges, thumbnails."""
    from IPython.display import HTML, display
    
    # Load data
    df = load_pairs(input_patterns)
    if df.empty:
        display(HTML('<div style="color:red">No data loaded from input patterns</div>'))
        return
    
    # Auto-detect similarity column if not specified
    if similarity_col is None:
        for col in ['pair_similarity', 'tanimoto_similarity', 'molecular_similarity']:
            if col in df.columns:
                similarity_col = col
                break
    
    # Select example strategy
    pair_key = make_pair_key_from_ids(compound_ids) if compound_ids else None
    
    # Find the target pair (use first available model as anchor)
    anchor_row = None
    for model in models:
        if not prefer_ensemble:
            # Use first available variant
            variant = None
        else:
            variant = 'ensemble'
        
        anchor_row = pick_example(df, model=model, pair_type=pair_type, 
                                antibiotic_class=antibiotic_class, pair_key=pair_key,
                                variant=variant, correctness=None, rng_seed=rng_seed)
        if anchor_row is not None:
            break
    
    if anchor_row is None:
        display(HTML('<div style="color:orange">No matching examples found</div>'))
        return
    
    # Get the pair key from anchor
    if pair_key is None:
        pair_key = anchor_row.get('pair_key', '')
        if not pair_key:
            # Compute from anchor row
            id1, id2 = extract_pair_ids(anchor_row)
            if id1 and id2:
                pair_key = f"{id1}|{id2}"
    
    # Collect rows for all models
    model_rows = []
    for model in models:
        variant = 'ensemble' if prefer_ensemble else None
        row = pick_example(df, model=model, pair_type=pair_type,
                          antibiotic_class=antibiotic_class, pair_key=pair_key,
                          variant=variant, correctness=None, rng_seed=rng_seed)
        model_rows.append((model, row))
    
    # Compute shared normalization
    valid_rows = [row for _, row in model_rows if row is not None]
    shared_norm = _compute_shared_normalization(valid_rows)
    
    # Render each model
    for model, row in model_rows:
        if row is None:
            display(HTML(f'<h3>{model} — <span style="color:#999">Not Available</span></h3>'))
            continue
        
        variant = row.get('variant', 'unknown')
        display(HTML(f'<h3>{model} / {variant} — {pair_type} — {antibiotic_class or "All Classes"}</h3>'))
        
        # Badges
        badges_html = render_badges(row.to_dict())
        
        # Enhanced visualization
        viz_html = draw_pair_enhanced(row, shared_norm=shared_norm, show_delta_bar=True)
        
        # Thumbnail scatter
        model_df = df[df['model'] == model]
        thumbnail_html = create_thumbnail_scatter(model_df, row, sim_col=similarity_col or 'pair_similarity')
        
        # Combine in row layout
        content_parts = []
        if badges_html:
            content_parts.append(f'<div style="margin-bottom:8px">{badges_html}</div>')
        
        if viz_html and thumbnail_html:
            content_parts.append(f'''
            <div style="display:flex; gap:20px; align-items:flex-start">
                <div style="flex:1">{viz_html}</div>
                <div style="flex:0 0 220px">{thumbnail_html}</div>
            </div>
            ''')
        elif viz_html:
            content_parts.append(viz_html)
        
        if content_parts:
            display(HTML(''.join(content_parts)))
        
        # Top-K table
        tbl = topk_substruct_table(row, k=8)
        if isinstance(tbl, pd.DataFrame) and not tbl.empty:
            display(tbl)
        
        display(HTML('<hr style="margin:20px 0; border:none; border-top:1px solid #eee">'))


# === SIMPLIFIED NOTEBOOK INTERFACE ===

def visualize_xai_comparison(
    rf_patterns: List[str],
    cnn_patterns: List[str],
    rgcn_patterns: List[str],
    pair_type: str = 'cliff',
    antibiotic_class: Optional[str] = None,
    prefer_ensemble: bool = True,
    compound_ids: Optional[List[str]] = None,
    rng_seed: int = 0
):
    """Single function call for the notebook - shows enhanced 3-model comparison."""
    input_patterns = rf_patterns + cnn_patterns + rgcn_patterns
    
    show_enhanced_dashboard(
        input_patterns=input_patterns,
        pair_type=pair_type,
        antibiotic_class=antibiotic_class,
        models=['RandomForest', 'CNN', 'RGCN'],
        prefer_ensemble=prefer_ensemble,
        compound_ids=compound_ids,
        rng_seed=rng_seed
    )
