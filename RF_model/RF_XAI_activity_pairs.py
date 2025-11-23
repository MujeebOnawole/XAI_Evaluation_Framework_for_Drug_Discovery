#!/usr/bin/env python3
"""
This script analyzes Random Forest explainability on activity cliff and non-cliff matched pairs
using SHAP TreeExplainer on 85 RDKit fragment descriptors (fr_*) with feature-space aggregation.

Key improvements:
- SHAP TreeExplainer for deterministic, fast explanations
- Feature-space edit vs context aggregation (no atom distances)
- Similarity-binned analysis and functional group enrichment
- Comprehensive plots and PDF reports
- Scalable for full dataset analysis

Usage: python RF_XAI_activity_pairs.py [--full] [--pdf]

Files required:
- activity_cliff_mmps.csv
- non_cliff_pairs.csv
- CV fold model files (*.pkl)

Outputs:
- rf_xai_pairs_detailed.csv
- rf_xai_summary.txt/.json
- rf_xai_plots.png/pdf (if --pdf specified)
"""

import os
import sys
import json
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import traceback

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import bootstrap
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from tqdm import tqdm

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Fragments import fr_COO, fr_amide, fr_ether, fr_benzene, fr_Al_OH, fr_NH1, fr_NH2, fr_halogen, fr_ketone, fr_methoxy
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

# SHAP imports
import shap

# Plotting imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import json

# Suppress warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# === Visualization and pharmacophore helpers ===
def prepare_visualization_data_rf(smiles: str, shap_weights: Dict[str, float], fg_smarts: Dict[str, str]) -> Dict[str, Any]:
    """Map SHAP functional group scores to atom-level attributions for visualization.
    Colors: blue (positive), red (negative), neutral: no color (None).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        return {'error': 'invalid_smiles'}
    n_atoms = mol.GetNumAtoms()
    atom_scores = np.zeros(n_atoms, dtype=float)
    for fg, val in (shap_weights or {}).items():
        if not val:
            continue
        smt = fg_smarts.get(fg)
        if not smt:
            continue
        patt = Chem.MolFromSmarts(smt)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue
        atoms = sorted({a for m in matches for a in m})
        if atoms:
            w = val / float(len(atoms))
            for a in atoms:
                atom_scores[a] += w
    max_abs = float(np.max(np.abs(atom_scores))) if np.max(np.abs(atom_scores)) > 0 else 1.0
    norm = atom_scores / max_abs
    colors = []
    # Color-blind friendly palette: blue for positive, orange for negative
    BLUE = (0.1216, 0.4667, 0.7059)   # matplotlib default blue #1f77b4
    ORANGE = (1.0, 0.4980, 0.0550)    # matplotlib default orange #ff7f0e
    pos_idx = []
    neg_idx = []
    neu_idx = []
    for i, s in enumerate(norm):
        if s > 0.05:
            inten = min(abs(s), 1.0)
            # Blend from white to BLUE by intensity
            c = (1.0*(1-inten) + BLUE[0]*inten,
                 1.0*(1-inten) + BLUE[1]*inten,
                 1.0*(1-inten) + BLUE[2]*inten)
            colors.append(c)
            pos_idx.append(i)
        elif s < -0.05:
            inten = min(abs(s), 1.0)
            # Blend from white to ORANGE by intensity
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
        'atom_attributions_normalized': norm.tolist(),
        'atom_colors': colors,
        'positive_atoms': pos_idx,
        'negative_atoms': neg_idx,
        'neutral_atoms': neu_idx,
        'n_atoms': n_atoms
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
    if mol is None or atom_scores is None or len(atom_scores) != mol.GetNumAtoms():
        return {'error': 'invalid_molecule_or_scores'}
    arr = np.asarray(atom_scores, dtype=float)
    thr = np.percentile(np.abs(arr), 80) if arr.size else 0.0
    highlighted = set(np.where(np.abs(arr) >= thr)[0].tolist())
    recognized = []
    overlap = {}
    expected = []
    missed = []
    for cat in ['required_any','loose_required_any','important_any','optional_any']:
        for feat in (sec.get(cat) or []):
            name = feat.get('name'); smt = feat.get('smarts');
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
            if ov > 0.3:
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

# Global Configuration
RUN_MODE = "TEST"  # "TEST" or "FULL" (changed from "SMOKE")
GENERATE_PLOTS = False  # Generate plots and PDF report
USE_SHAP = True  # Always use SHAP TreeExplainer (not LIME)
RANDOM_SEED = 42

# Balanced dataset configuration
STRATEGIC_CLASSES = ['Fluoroquinolones', 'Cephalosporins', 'Oxazolidinones', 'Tetracyclines']
TEST_SAMPLES_PER_CLASS = 2  # For TEST mode: 2 pairs per class = 8 total each type
FULL_SAMPLES_PER_CLASS = 125  # For FULL mode: all 125 pairs per class

# Update existing test parameters
TEST_N_CLIFF = 8  # 4 classes × 2 samples
TEST_N_NON = 8    # 4 classes × 2 samples

# Legacy parameters for compatibility
SMOKE_N_CLIFF = TEST_N_CLIFF  # For backward compatibility
SMOKE_N_NON = TEST_N_NON      # For backward compatibility

# Model paths - ensemble configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
BEST_MODELS_PATH = os.path.join(script_dir, "best_models.json")
MODEL_CHECKPOINTS_DIR = os.path.join(script_dir, "model_checkpoints")
TRAINING_DATA_PATH = os.path.join(script_dir, "SA_FG_fragments.csv")
FG_MAPPING_PATH = os.path.join(script_dir, "Functional_Group_mapping.txt")
THRESHOLD = 0.5

# Dataset paths (overridable via CLI)
ACTIVITY_CSV = "activity_cliff_pairs.csv"
NONCLIFF_CSV = "non_cliff_pairs.csv"

# Model-run context annotations
CURRENT_MODEL_ID = "ensemble"
CURRENT_IS_ENSEMBLE = True

# Frozen SMARTS dictionary for common functional groups
FG_SMARTS = {
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

# Global variables for caching
_ensemble = None
_shap_explainer = None
_shap_background = None

class EnsembleRandomForestExplainer:
    """
    Ensemble Random Forest SHAP Explainer for molecular activity prediction.

    Uses the best models selected from cross-validation results to provide
    local explanations using SHAP TreeExplainer method.
    """

    def __init__(self,
                 best_models_path: str = BEST_MODELS_PATH,
                 model_checkpoints_dir: str = MODEL_CHECKPOINTS_DIR,
                 training_data_path: str = TRAINING_DATA_PATH,
                 fg_mapping_path: str = FG_MAPPING_PATH,
                 threshold: float = THRESHOLD):

        self.threshold = threshold
        self.model_checkpoints_dir = model_checkpoints_dir
        self.fg_mapping_path = fg_mapping_path

        print("Initializing Enhanced RF Ensemble...")

        # Load best models information
        self.best_models_info = self._load_best_models(best_models_path)
        print(f"Loaded {len(self.best_models_info)} best models")

        # Load training data for feature names
        self.training_data = pd.read_csv(training_data_path)
        self.feature_names = [col for col in self.training_data.columns if col.startswith('fr_')]
        print(f"Loaded {len(self.feature_names)} molecular features")

        # Load models and scalers
        self.models, self.scalers = self._load_ensemble_models()
        print(f"Loaded {len(self.models)} RF models")

        # Prepare training data for SHAP background
        self.shap_background = self._prepare_shap_background()
        print(f"Prepared SHAP background data: {self.shap_background.shape}")

        # Load functional group mappings
        self.fg_mapping = self._load_fg_mapping()

        # Create SMARTS mapping (use the comprehensive FG_SMARTS dictionary)
        self.smarts_mapping = FG_SMARTS

        print("Enhanced RF Ensemble ready!")

    def _load_best_models(self, path: str) -> List[Dict]:
        """Load best models information from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return data['models']

    def _load_ensemble_models(self) -> Tuple[List, List]:
        """Load all ensemble models and scalers"""
        models, scalers = [], []
        for model_info in self.best_models_info:
            cv, fold = model_info['cv'], model_info['fold']
            model_dir = os.path.join(self.model_checkpoints_dir, f'cv{cv}_fold{fold}')
            model_path = os.path.join(model_dir, 'model.joblib')
            scaler_path = os.path.join(model_dir, 'scaler.joblib')

            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                models.append(model)
                scalers.append(scaler)
                print(f"  Loaded CV{cv} Fold{fold}")
            except Exception as e:
                print(f"  Failed to load CV{cv} Fold{fold}: {e}")

        if not models:
            raise ValueError("No models were successfully loaded!")
        return models, scalers

    def _prepare_shap_background(self) -> np.ndarray:
        """Prepare background data for SHAP TreeExplainer"""
        # Get training features in correct order
        X_train = self.training_data[self.training_data['group'] == 'training'][self.feature_names]

        # Sample a subset for SHAP background (1000 samples for speed)
        n_background = min(1000, len(X_train))
        background_indices = np.random.choice(len(X_train), size=n_background, replace=False)
        X_bg_raw = X_train.iloc[background_indices].values

        # Scale with first scaler (they should be similar across CV folds)
        X_bg_scaled = self.scalers[0].transform(X_bg_raw)

        return X_bg_scaled

    def _load_fg_mapping(self) -> Dict:
        """Load functional group mapping"""
        try:
            if os.path.exists(self.fg_mapping_path):
                with open(self.fg_mapping_path, 'r') as f:
                    return json.load(f)
            else:
                print(f"Functional group mapping file not found: {self.fg_mapping_path}")
                return {}
        except Exception as e:
            print(f"Error loading functional group mapping: {e}")
            return {}

    def restrict_to_cv_fold(self, cv: int, fold: int) -> None:
        """Restrict the ensemble to a single CV/fold model and recompute SHAP background.

        After calling, self.models/scalers contain only the chosen model.
        """
        # Find index
        idx = None
        for i, mi in enumerate(self.best_models_info):
            if int(mi.get('cv')) == int(cv) and int(mi.get('fold')) == int(fold):
                idx = i
                break
        if idx is None:
            raise ValueError(f"Requested model cv{cv}_fold{fold} not found in best_models.json")
        # Restrict lists
        self.models = [self.models[idx]]
        self.scalers = [self.scalers[idx]]
        # Recompute SHAP background with this scaler
        self.shap_background = self._prepare_shap_background()

    def calculate_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculate molecular descriptors from SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        descriptors = {}
        for feature in self.feature_names:
            try:
                descriptor_func = getattr(Descriptors, feature)
                descriptors[feature] = int(descriptor_func(mol))
            except AttributeError:
                descriptors[feature] = 0

        return descriptors

    def ensemble_predict_proba(self, X: np.ndarray) -> float:
        """Ensemble prediction using all models"""
        predictions = []

        for model, scaler in zip(self.models, self.scalers):
            try:
                # Ensure X has the right shape
                if len(X.shape) == 1:
                    X_reshaped = X.reshape(1, -1)
                else:
                    X_reshaped = X

                # Scale and predict
                X_scaled = scaler.transform(X_reshaped)
                pred_proba = model.predict_proba(X_scaled)
                if pred_proba.shape[1] > 1:
                    predictions.append(pred_proba[0, 1])  # Probability of positive class
                else:
                    predictions.append(pred_proba[0, 0])

            except Exception as e:
                print(f"Model prediction failed: {e}")
                continue

        if not predictions:
            raise ValueError("All model predictions failed!")

        # Average predictions
        return np.mean(predictions)

    def prepare_input_features(self, smiles: str) -> np.ndarray:
        """Prepare molecular input features for prediction and explanation"""
        descriptors = self.calculate_descriptors(smiles)
        X = pd.DataFrame([descriptors])

        # Ensure all features are present in correct order
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0

        X = X[self.feature_names]  # Ensure correct order
        return X.values

    def predict(self, smiles: str) -> float:
        """Predict probability for a single molecule"""
        X = self.prepare_input_features(smiles)
        return self.ensemble_predict_proba(X)

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load balanced activity cliff and non-cliff CSV files."""
    print("Loading balanced data files...")

    if not os.path.exists(ACTIVITY_CSV):
        raise FileNotFoundError(f"{ACTIVITY_CSV} not found")
    if not os.path.exists(NONCLIFF_CSV):
        raise FileNotFoundError(f"{NONCLIFF_CSV} not found")

    df_cliff = pd.read_csv(ACTIVITY_CSV)
    df_non = pd.read_csv(NONCLIFF_CSV)

    print(f"Loaded {len(df_cliff)} balanced activity cliff pairs")
    print(f"Loaded {len(df_non)} balanced non-cliff pairs")

    # Verify balanced structure
    print("Cliff pairs class distribution:")
    print(df_cliff['class'].value_counts().sort_index())
    print("Non-cliff pairs class distribution:")
    print(df_non['class'].value_counts().sort_index())

    return df_cliff, df_non

def sample_balanced_data(df_cliff: pd.DataFrame, df_non: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample data based on RUN_MODE while maintaining class balance."""

    if RUN_MODE == "TEST":
        print(f"TEST MODE: Sampling {TEST_SAMPLES_PER_CLASS} pairs per class")

        # Sample from each strategic class
        cliff_samples = []
        non_cliff_samples = []

        for cls in STRATEGIC_CLASSES:
            # Sample cliff pairs for this class
            cliff_class = df_cliff[df_cliff['class'] == cls]
            if len(cliff_class) >= TEST_SAMPLES_PER_CLASS:
                cliff_sample = cliff_class.sample(n=TEST_SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
                cliff_samples.append(cliff_sample)
            else:
                print(f"Warning: Only {len(cliff_class)} cliff pairs available for {cls}")
                cliff_samples.append(cliff_class)

            # Sample non-cliff pairs for this class
            non_cliff_class = df_non[df_non['class'] == cls]
            if len(non_cliff_class) >= TEST_SAMPLES_PER_CLASS:
                non_cliff_sample = non_cliff_class.sample(n=TEST_SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
                non_cliff_samples.append(non_cliff_sample)
            else:
                print(f"Warning: Only {len(non_cliff_class)} non-cliff pairs available for {cls}")
                non_cliff_samples.append(non_cliff_class)

        df_cliff_sampled = pd.concat(cliff_samples, ignore_index=True)
        df_non_sampled = pd.concat(non_cliff_samples, ignore_index=True)

        print(f"TEST sampling result: {len(df_cliff_sampled)} cliffs, {len(df_non_sampled)} non-cliffs")

    else:  # FULL mode
        print("FULL MODE: Processing all pairs")
        df_cliff_sampled = df_cliff
        df_non_sampled = df_non

    return df_cliff_sampled, df_non_sampled

def get_fragment_names() -> List[str]:
    """Get all 85 RDKit fr_* fragment descriptor names in sorted order."""
    # Get all fr_* attributes from Descriptors module
    fragment_names = []
    for attr_name in dir(Descriptors):
        if attr_name.startswith('fr_'):
            fragment_names.append(attr_name)

    return sorted(fragment_names)

def get_ensemble():
    """Get or create the ensemble model."""
    global _ensemble
    if _ensemble is None:
        _ensemble = EnsembleRandomForestExplainer()
    return _ensemble

def predict_proba_rf(smiles: str) -> float:
    """Predict probability using ensemble Random Forest model."""
    ensemble = get_ensemble()
    return ensemble.predict(smiles)

def get_shap_explainer():
    """Get or create SHAP TreeExplainer for ensemble."""
    global _shap_explainer, _shap_background

    if _shap_explainer is not None:
        return _shap_explainer, _shap_background

    ensemble = get_ensemble()

    print("Initializing SHAP explainer...")

    # Use first model for SHAP (could average across ensemble if needed)
    _shap_explainer = shap.TreeExplainer(
        ensemble.models[0],
        data=ensemble.shap_background,
        feature_perturbation="interventional"
    )
    _shap_background = ensemble.shap_background

    print("SHAP explainer initialized")
    return _shap_explainer, _shap_background

def get_shap_prob(model, scaler, X_bg, X_raw):
    """Get SHAP values in probability space for a single sample."""
    # Scale input
    X_scaled = scaler.transform(X_raw.reshape(1, -1))

    try:
        # Try to get probability-space SHAP values directly
        explainer_prob = shap.TreeExplainer(model, data=X_bg,
                                          feature_perturbation="interventional",
                                          model_output="probability")
        shap_vals = explainer_prob.shap_values(X_scaled)
        # For binary classification, use positive class contributions
        if isinstance(shap_vals, list):
            return shap_vals[1][0]  # positive class, first sample
        else:
            return shap_vals[0]  # single output
    except:
        # Fallback: convert log-odds SHAP to probability contributions
        explainer = shap.TreeExplainer(model, data=X_bg, feature_perturbation="interventional")
        phi = explainer.shap_values(X_scaled)
        
        if isinstance(phi, list):
            phi_pos = phi[1][0]  # positive class, first sample
        else:
            phi_pos = phi[0]

        # Convert log-odds to probability space
        base_logit = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        f_logit = base_logit + phi_pos.sum()
        p = 1 / (1 + np.exp(-f_logit))
        p_base = 1 / (1 + np.exp(-base_logit))
        dp = p - p_base

        # Distribute probability change proportional to absolute SHAP values
        weights = np.abs(phi_pos) / (np.abs(phi_pos).sum() + 1e-12)
        return weights * dp

def _prepare_shap_background_for_scaler(scaler, training_df: pd.DataFrame, feature_names: List[str], n_background: int = 1000) -> np.ndarray:
    """Prepare SHAP background matrix scaled with the provided scaler."""
    X_train = training_df[training_df['group'] == 'training'][feature_names]
    n_bg = min(int(n_background), len(X_train)) if len(X_train) else 0
    if n_bg <= 0:
        return np.zeros((1, len(feature_names)), dtype=float)
    idx = np.random.choice(len(X_train), size=n_bg, replace=False)
    X_bg_raw = X_train.iloc[idx].values
    try:
        X_bg_scaled = scaler.transform(X_bg_raw)
    except Exception:
        X_bg_scaled = X_bg_raw
    return X_bg_scaled


def shap_explain_single(smiles: str) -> Dict[str, float]:
    """
    Run single SHAP explanation for functional group analysis.

    Returns:
        Dictionary mapping feature names to SHAP weights in probability space
    """
    ensemble = get_ensemble()
    explainer, X_bg = get_shap_explainer()

    # Prepare input
    descriptors = ensemble.calculate_descriptors(smiles)
    X_raw = np.array([descriptors[name] for name in ensemble.feature_names])

    # Get SHAP values from first model
    shap_values = get_shap_prob(ensemble.models[0], ensemble.scalers[0], X_bg, X_raw)

    # Convert to dictionary
    weights_dict = {}
    for i, feature_name in enumerate(ensemble.feature_names):
        if shap_values.ndim == 2:
            # Binary classifier: take class 1 SHAP values
            weights_dict[feature_name] = float(shap_values[i, 1])
        else:
            # Single output: use directly
            weights_dict[feature_name] = float(shap_values[i])

    return weights_dict


def shap_explain_single_for_model(smiles: str, model, scaler, training_df: pd.DataFrame, feature_names: List[str]) -> Dict[str, float]:
    """Compute probability-space SHAP for a single RF model and return feature dict."""
    # Prepare descriptors in feature order
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    descriptors = {}
    for name in feature_names:
        try:
            descriptor_func = getattr(Descriptors, name)
            descriptors[name] = int(descriptor_func(mol))
        except AttributeError:
            descriptors[name] = 0
    X_raw = np.array([descriptors[name] for name in feature_names])
    # Background for this scaler
    X_bg = _prepare_shap_background_for_scaler(scaler, training_df, feature_names)
    # SHAP in probability space
    shap_values = get_shap_prob(model, scaler, X_bg, X_raw)
    weights: Dict[str, float] = {}
    for i, fname in enumerate(feature_names):
        if getattr(shap_values, 'ndim', 1) == 2:
            weights[fname] = float(shap_values[i, 1])
        else:
            weights[fname] = float(shap_values[i])
    return weights

def count_feature_presence(mol_feature_hits: Dict[str, Any]) -> Dict[str, int]:
    """Convert feature hits to presence counts."""
    return {feature: int(count) for feature, count in mol_feature_hits.items()}

def compute_edit_sets(pres_active: Dict[str, int], pres_inactive: Dict[str, int]) -> Dict[str, List[str]]:
    """Identify lost and gained features between active/inactive compounds."""
    lost, gained = [], []
    all_features = set(pres_active) | set(pres_inactive)
    
    for feature in all_features:
        active_count = pres_active.get(feature, 0)
        inactive_count = pres_inactive.get(feature, 0)
        
        if active_count > 0 and inactive_count == 0:
            lost.append(feature)
        elif active_count == 0 and inactive_count > 0:
            gained.append(feature)
    
    return {"lost": lost, "gained": gained}

def rank_edit_drivers(pres_active: Dict[str, int], pres_inactive: Dict[str, int], 
                     shap_active: Dict[str, float], shap_inactive: Dict[str, float], 
                     topk: int = 5) -> List[Dict[str, Any]]:
    """Rank structural changes by their likelihood to cause activity cliff."""
    edits = compute_edit_sets(pres_active, pres_inactive)
    candidates = []

    # Lost features: high positive SHAP in active = likely cliff driver
    for feature in edits["lost"]:
        support = max(0.0, shap_active.get(feature, 0.0))
        if support > 0:
            candidates.append({
                "feature": feature,
                "direction": "lost_from_active",
                "support": round(support, 6)
            })

    # Gained features: high negative SHAP in inactive = likely cliff driver
    for feature in edits["gained"]:
        support = max(0.0, -shap_inactive.get(feature, 0.0))
        if support > 0:
            candidates.append({
                "feature": feature,
                "direction": "gained_in_inactive",
                "support": round(support, 6)
            })

    candidates.sort(key=lambda x: x["support"], reverse=True)
    return candidates[:topk]

def create_feature_level_scores(active_shap_weights, inactive_shap_weights, changed_functional_groups, k=5):
    """Create standardized feature-level scores for cross-model comparison."""
    feature_scores_active = active_shap_weights.copy()
    feature_scores_inactive = inactive_shap_weights.copy()

    # Calculate absolute deltas
    all_features = set(active_shap_weights.keys()) | set(inactive_shap_weights.keys())
    feature_delta_abs = {}

    for feature in all_features:
        active_score = active_shap_weights.get(feature, 0.0)
        inactive_score = inactive_shap_weights.get(feature, 0.0)
        feature_delta_abs[feature] = abs(active_score - inactive_score)

    # Top-k features by change
    sorted_features = sorted(feature_delta_abs.items(), key=lambda x: x[1], reverse=True)
    topk_features = [feat for feat, delta in sorted_features[:k] if delta > 0]

    return {
        'feature_scores_active': feature_scores_active,
        'feature_scores_inactive': feature_scores_inactive,
        'feature_delta_abs': feature_delta_abs,
        'topk_features': topk_features
    }

def list_changed_groups(active_counts: Dict[str, int], inactive_counts: Dict[str, int]) -> List[str]:
    """List functional groups that differ between active and inactive compounds."""
    all_features = set(active_counts.keys()) | set(inactive_counts.keys())
    
    changed_groups = []
    for feature in all_features:
        if active_counts.get(feature, 0) != inactive_counts.get(feature, 0):
            changed_groups.append(feature)

    return changed_groups

def map_changed_groups_to_atoms(mol, changed_groups: List[str]) -> set:
    """
    Map changed functional groups to atom indices using ensemble SMARTS patterns.
    Uses the comprehensive FG_SMARTS dictionary.
    """
    edit_atoms = set()
    ensemble = get_ensemble()

    for group in changed_groups:
        if group in ensemble.smarts_mapping:
            smarts = ensemble.smarts_mapping[group]
            pattern = Chem.MolFromSmarts(smarts)

            if pattern is not None:
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    edit_atoms.update(match)

    return edit_atoms

def shap_to_atom_attributions(mol, shap_weights: Dict[str, float]) -> np.ndarray:
    """
    Distribute SHAP descriptor weights to atoms using ensemble SMARTS patterns.
    Uses the comprehensive FG_SMARTS dictionary from the ensemble.
    """
    n_atoms = mol.GetNumAtoms()
    atom_attributions = np.zeros(n_atoms)
    ensemble = get_ensemble()

    for descriptor, weight in shap_weights.items():
        if descriptor in ensemble.smarts_mapping and weight != 0:
            smarts = ensemble.smarts_mapping[descriptor]
            pattern = Chem.MolFromSmarts(smarts)

            if pattern is not None:
                matches = mol.GetSubstructMatches(pattern)
                if matches:
                    # Flatten all matches to unique atom indices
                    matched_atoms = set()
                    for match in matches:
                        matched_atoms.update(match)

                    # Distribute weight equally among matched atoms
                    if matched_atoms:
                        weight_per_atom = weight / len(matched_atoms)
                        for atom_idx in matched_atoms:
                            atom_attributions[atom_idx] += weight_per_atom

    return atom_attributions

def distance_bins(mol, edit_atoms: set) -> Dict[int, int]:
    """
    Compute shortest path distance from each atom to nearest edit atom.
    Returns dict mapping atom_idx -> distance_bin (0 for core, 2 for distant).
    """
    if not edit_atoms:
        return {}

    n_atoms = mol.GetNumAtoms()
    distances = {}

    # Compute shortest paths using RDKit's molecular graph
    for atom_idx in range(n_atoms):
        if atom_idx in edit_atoms:
            distances[atom_idx] = 0  # Core
        else:
            min_distance = float('inf')
            for edit_atom in edit_atoms:
                try:
                    path = Chem.GetShortestPath(mol, atom_idx, edit_atom)
                    if path:
                        distance = len(path) - 1
                        min_distance = min(min_distance, distance)
                except:
                    continue

            # Classify as core (0) or distant (2)
            distances[atom_idx] = 0 if min_distance <= 1 else 2

    return distances

def compute_deltas_and_indices(active_attrs: np.ndarray, inactive_attrs: np.ndarray,
                             edit_atoms: set, unmapped_edit_mass: float) -> Dict[str, Any]:
    """
    Compute attribution deltas and indices.

    Returns dictionary with:
    - total_delta, core_delta, distant_delta, edit_delta
    - delta_by_distance, propagation_index, core_edit_ratio, edit_concentration_index
    """
    # Compute per-atom deltas
    atom_deltas = np.abs(active_attrs - inactive_attrs)
    total_mapped_delta = np.sum(atom_deltas)

    # Add unmapped mass to edit_delta
    edit_delta = unmapped_edit_mass
    total_delta = total_mapped_delta + edit_delta

    if not edit_atoms or len(active_attrs) != len(inactive_attrs):
        # No edit atoms or mismatched lengths
        core_delta = 0.0
        distant_delta = total_mapped_delta
        delta_by_distance = {0: core_delta, 2: distant_delta, 'edit': edit_delta}
    else:
        # Split by distance - simplified approach
        core_delta = 0.0
        distant_delta = 0.0

        # Simple split: atoms in edit_atoms are core, rest are distant
        for i, delta in enumerate(atom_deltas):
            if i in edit_atoms:
                core_delta += delta
            else:
                distant_delta += delta

        delta_by_distance = {0: core_delta, 2: distant_delta, 'edit': edit_delta}

    # Compute indices
    if total_delta > 1e-12:
        propagation_index = (core_delta + distant_delta) / total_delta
        edit_concentration_index = core_delta / total_delta
    else:
        propagation_index = 0.0
        edit_concentration_index = 0.0

    if (core_delta + edit_delta) > 1e-12:
        core_edit_ratio = core_delta / (core_delta + edit_delta)
    else:
        core_edit_ratio = 0.0

    return {
        'total_delta': float(total_delta),
        'core_delta': float(core_delta),
        'distant_delta': float(distant_delta),
        'edit_delta': float(edit_delta),
        'delta_by_distance': delta_by_distance,
        'propagation_index': float(propagation_index),
        'core_edit_ratio': float(core_edit_ratio),
        'edit_concentration_index': float(edit_concentration_index)
    }

def analyze_pair(row, pair_type: str) -> Dict[str, Any]:
    """
    Analyze a single pair and return row dictionary for output CSV.

    Args:
        row: DataFrame row from cliff or non-cliff data
        pair_type: "cliff" or "non_cliff"

    Returns:
        Dictionary with all required output columns
    """
    try:
        # ONLY UPDATE THIS SECTION - Extract SMILES and metadata based on pair type
        if pair_type == "cliff":
            active_smiles = row['active_smiles']
            inactive_smiles = row['inactive_smiles']
            pair_id = f"cliff_{row['active_compound_id']}_{row['inactive_compound_id']}"
            similarity = row['structural_similarity']
            antibiotic_class = row['class']

            # Activity difference - use new columns
            activity_difference = row.get('activity_difference_log', 0.0)
            fold_difference = row.get('fold_difference', 0.0)

            # Group information and TARGET ground truth
            active_group = row.get('active_group', 'unknown')
            inactive_group = row.get('inactive_group', 'unknown')
            group_combination = row.get('group_combination', 'unknown')
            try:
                active_target = int(row.get('active_TARGET', 1))
            except Exception:
                active_target = 1
            try:
                inactive_target = int(row.get('inactive_TARGET', 0))
            except Exception:
                inactive_target = 0

            # Compound IDs
            active_compound_id = row['active_compound_id']
            inactive_compound_id = row['inactive_compound_id']

        else:  # non_cliff
            active_smiles = row['compound1_smiles']
            inactive_smiles = row['compound2_smiles']
            pair_id = row['pair_id']
            similarity = row['structural_similarity']
            antibiotic_class = row['class']

            # Activity difference
            if pd.notna(row.get('compound1_logMIC')) and pd.notna(row.get('compound2_logMIC')):
                activity_difference = abs(row['compound1_logMIC'] - row['compound2_logMIC'])
            else:
                activity_difference = 0.0

            fold_difference = row.get('fold_difference', 1.0)

            # Group information and TARGET ground truth
            active_group = row.get('compound1_group', 'unknown')
            inactive_group = row.get('compound2_group', 'unknown')
            group_combination = row.get('group_combination', 'unknown')
            try:
                active_target = int(row.get('compound1_TARGET', row.get('TARGET', 1)))
            except Exception:
                active_target = 1
            try:
                inactive_target = int(row.get('compound2_TARGET', row.get('TARGET', 1)))
            except Exception:
                inactive_target = 1

            # Compound IDs
            active_compound_id = row['compound1_id']
            inactive_compound_id = row['compound2_id']

        # Get predictions
        active_pred_prob = predict_proba_rf(active_smiles)
        inactive_pred_prob = predict_proba_rf(inactive_smiles)
        prediction_difference = active_pred_prob - inactive_pred_prob

        # Classifications
        active_class = 'Active' if active_pred_prob >= THRESHOLD else 'Inactive'
        inactive_class = 'Active' if inactive_pred_prob >= THRESHOLD else 'Inactive'

        # Get fragment counts using ensemble
        ensemble = get_ensemble()
        active_counts = ensemble.calculate_descriptors(active_smiles)
        inactive_counts = ensemble.calculate_descriptors(inactive_smiles)

        # Convert to vectors for compatibility
        active_vector = np.array([active_counts[name] for name in ensemble.feature_names])
        inactive_vector = np.array([inactive_counts[name] for name in ensemble.feature_names])

        # Feature statistics
        active_features_count = np.sum(active_vector > 0)
        inactive_features_count = np.sum(inactive_vector > 0)
        common_features_count = np.sum((active_vector > 0) & (inactive_vector > 0))

        feature_differences = {}
        for name in active_counts.keys():
            diff = active_counts[name] - inactive_counts.get(name, 0)
            if diff != 0:
                feature_differences[name] = diff

        if feature_differences:
            max_feature_diff = max(abs(d) for d in feature_differences.values())
            mean_feature_diff = np.mean([abs(d) for d in feature_differences.values()])
        else:
            max_feature_diff = 0.0
            mean_feature_diff = 0.0

        # SHAP explanations for functional group analysis (supports ensemble averaging)
        attr_std = float('nan')
        ensemble = get_ensemble()
        if CURRENT_IS_ENSEMBLE and len(ensemble.models) > 1:
            active_list: List[Dict[str, float]] = []
            inactive_list: List[Dict[str, float]] = []
            per_model_arrays_a: List[np.ndarray] = []
            per_model_arrays_i: List[np.ndarray] = []
            for m, sc in zip(ensemble.models, ensemble.scalers):
                wa = shap_explain_single_for_model(active_smiles, m, sc, ensemble.training_data, ensemble.feature_names)
                wi = shap_explain_single_for_model(inactive_smiles, m, sc, ensemble.training_data, ensemble.feature_names)
                active_list.append(wa)
                inactive_list.append(wi)
                per_model_arrays_a.append(np.array([wa.get(fn, 0.0) for fn in ensemble.feature_names], dtype=float))
                per_model_arrays_i.append(np.array([wi.get(fn, 0.0) for fn in ensemble.feature_names], dtype=float))
            all_keys = set()
            for d in active_list + inactive_list:
                all_keys.update(d.keys())
            active_shap_weights = {}
            inactive_shap_weights = {}
            for key in all_keys:
                vals_a = [d.get(key, 0.0) for d in active_list]
                vals_i = [d.get(key, 0.0) for d in inactive_list]
                active_shap_weights[key] = float(np.mean(vals_a))
                inactive_shap_weights[key] = float(np.mean(vals_i))
            try:
                A = np.stack(per_model_arrays_a, axis=0)
                I = np.stack(per_model_arrays_i, axis=0)
                std_a = np.std(A, axis=0).mean()
                std_i = np.std(I, axis=0).mean()
                attr_std = float(0.5 * (std_a + std_i))
                # Variance warning if relative std > 50%
                mean_abs_a = float(np.mean(np.abs(np.mean(A, axis=0)))) + 1e-12
                mean_abs_i = float(np.mean(np.abs(np.mean(I, axis=0)))) + 1e-12
                rel_std = 0.5 * (std_a / mean_abs_a + std_i / mean_abs_i)
                if rel_std > 0.5:
                    print(f"WARNING: Ensemble attribution variance high (relative std ~ {rel_std:.2f})")
            except Exception:
                attr_std = float('nan')
            print(f"Ensemble attribution: averaged {len(ensemble.models)} models")
        else:
            active_shap_weights = shap_explain_single(active_smiles)
            inactive_shap_weights = shap_explain_single(inactive_smiles)

        # Visualization-ready atom-level attributions via SMARTS mapping
        viz_active = prepare_visualization_data_rf(active_smiles, active_shap_weights, FG_SMARTS)
        viz_inactive = prepare_visualization_data_rf(inactive_smiles, inactive_shap_weights, FG_SMARTS)

        # Changed functional groups
        changed_functional_groups = list_changed_groups(active_counts, inactive_counts)

        # Get feature presence from existing descriptor calculations
        presence_active = count_feature_presence(active_counts)
        presence_inactive = count_feature_presence(inactive_counts)

        # Feature-level scores for cross-model comparison
        feature_comparison = create_feature_level_scores(
            active_shap_weights,
            inactive_shap_weights,
            changed_functional_groups,
            k=5
        )

        # Edit-aware cliff driver analysis
        edit_candidates = rank_edit_drivers(
            presence_active,
            presence_inactive,
            active_shap_weights,
            inactive_shap_weights,
            topk=5
        )
        top_edit_driver = edit_candidates[0]["feature"] if edit_candidates else ""
        top_edit_driver_support = edit_candidates[0]["support"] if edit_candidates else 0.0
        rf_visible = len(changed_functional_groups) > 0
        edit_basis = "rf_functional_group" if rf_visible else "no_feature_change"

        # Visualization lists: positive/negative functional groups by sign (sorted by |weight|)
        def _pos_neg_lists(d: Dict[str, float]) -> Tuple[List[str], List[str]]:
            try:
                pos = [(k, v) for k, v in d.items() if float(v) > 0]
                neg = [(k, v) for k, v in d.items() if float(v) < 0]
                pos = [k for k, _ in sorted(pos, key=lambda kv: -abs(kv[1]))]
                neg = [k for k, _ in sorted(neg, key=lambda kv: -abs(kv[1]))]
                return pos, neg
            except Exception:
                return [], []
        pos_a_list, neg_a_list = _pos_neg_lists(active_shap_weights)
        pos_i_list, neg_i_list = _pos_neg_lists(inactive_shap_weights)

        # Unified comparator fields: SHAP→atom masses and delta_core_align
        try:
            mol_a = Chem.MolFromSmiles(active_smiles)
            mol_i = Chem.MolFromSmiles(inactive_smiles)
        except Exception:
            mol_a, mol_i = None, None

        atom_attr_active_u: List[float] = []
        atom_attr_inactive_u: List[float] = []
        atom_mass_active = 0.0
        atom_mass_inactive = 0.0
        core_a_mass = 0.0
        core_i_mass = 0.0
        edit_mass = 0.0
        context_mass = 0.0
        delta_core_align = 0.0

        try:
            if mol_a is not None:
                aa = shap_to_atom_attributions(mol_a, active_shap_weights)
                atom_attr_active_u = aa.astype(float).tolist() if hasattr(aa, 'astype') else list(aa)
                atom_mass_active = float(np.sum(np.abs(aa)))
            if mol_i is not None:
                ii = shap_to_atom_attributions(mol_i, inactive_shap_weights)
                atom_attr_inactive_u = ii.astype(float).tolist() if hasattr(ii, 'astype') else list(ii)
                atom_mass_inactive = float(np.sum(np.abs(ii)))

            # Map changed groups to atoms and compute r1 core masses (distance bin 0)
            edit_atoms_a = map_changed_groups_to_atoms(mol_a, changed_functional_groups) if mol_a is not None else set()
            edit_atoms_i = map_changed_groups_to_atoms(mol_i, changed_functional_groups) if mol_i is not None else set()
            if mol_a is not None and atom_attr_active_u:
                bins_a = distance_bins(mol_a, edit_atoms_a)
                core_idx_a = [idx for idx, b in bins_a.items() if b == 0]
                if core_idx_a:
                    core_a_mass = float(np.sum(np.abs(np.asarray(atom_attr_active_u)[core_idx_a])))
            if mol_i is not None and atom_attr_inactive_u:
                bins_i = distance_bins(mol_i, edit_atoms_i)
                core_idx_i = [idx for idx, b in bins_i.items() if b == 0]
                if core_idx_i:
                    core_i_mass = float(np.sum(np.abs(np.asarray(atom_attr_inactive_u)[core_idx_i])))
            distant_a_mass = max(0.0, atom_mass_active - core_a_mass)
            distant_i_mass = max(0.0, atom_mass_inactive - core_i_mass)
            edit_mass = core_a_mass + core_i_mass
            context_mass = distant_a_mass + distant_i_mass
            denom_mass = edit_mass + context_mass
            delta_core_align = float(edit_mass / (denom_mass + 1e-12)) if denom_mass > 0 else 0.0
        except Exception:
            pass

        # Attribution analysis - Feature-space SHAP (edit vs context)
        if rf_visible:
            # Feature-space SHAP aggregation (Option A)
            # dphi: SHAP prob-space deltas (active - inactive) per fr_* feature
            dphi = np.array([active_shap_weights.get(name, 0.0) - inactive_shap_weights.get(name, 0.0)
                           for name in ensemble.feature_names])

            # Get edit vs context feature indices
            name_to_idx = {f: i for i, f in enumerate(ensemble.feature_names)}
            edit_idx = np.array([name_to_idx[f] for f in changed_functional_groups if f in name_to_idx], dtype=int)
            all_idx = np.arange(len(dphi), dtype=int)

            context_mask = np.ones_like(all_idx, dtype=bool)
            if edit_idx.size > 0:
                context_mask[edit_idx] = False

            # Compute feature-space deltas
            edit_delta = float(np.sum(np.abs(dphi[edit_idx]))) if edit_idx.size else 0.0
            context_delta = float(np.sum(np.abs(dphi[context_mask])))
            total_delta = float(np.sum(np.abs(dphi)))

            # Feature-space metrics
            propagation_index = context_delta / (edit_delta + context_delta + 1e-12)
            core_edit_ratio = edit_delta / (edit_delta + 1e-12)  # "all-edit" ratio in SHAP mode
            edit_concentration_index = edit_delta / (total_delta + 1e-12)

            delta_by_distance = {"edit": edit_delta, "context": context_delta}

            deltas = {
                'total_delta': total_delta,
                'core_delta': 0.0,  # no atom-distance in feature space
                'distant_delta': context_delta,  # rename to context_delta conceptually
                'edit_delta': edit_delta,
                'delta_by_distance': delta_by_distance,
                'propagation_index': propagation_index,
                'core_edit_ratio': core_edit_ratio,
                'edit_concentration_index': edit_concentration_index
            }
        else:
            # No feature changes
            total_delta = abs(active_pred_prob - inactive_pred_prob)  # Use prediction difference as proxy
            deltas = {
                'total_delta': total_delta,
                'core_delta': 0.0,
                'distant_delta': 0.0,
                'edit_delta': 0.0,
                'delta_by_distance': {0: total_delta},
                'propagation_index': 0.0,
                'core_edit_ratio': 0.0,
                'edit_concentration_index': 0.0
            }

        # Pair classification logic using TARGET ground truth
        # Classification based on TARGET and prediction threshold:
        # TARGET=1, pred≥0.5 → TP
        # TARGET=1, pred<0.5 → FN
        # TARGET=0, pred<0.5 → TN
        # TARGET=0, pred≥0.5 → FP

        if active_target == 1:
            active_classification = 'TP' if active_pred_prob >= THRESHOLD else 'FN'
        else:  # active_target == 0
            active_classification = 'TN' if active_pred_prob < THRESHOLD else 'FP'

        if inactive_target == 1:
            inactive_classification = 'TP' if inactive_pred_prob >= THRESHOLD else 'FN'
        else:  # inactive_target == 0
            inactive_classification = 'TN' if inactive_pred_prob < THRESHOLD else 'FP'

        # Pair classification
        both_correct = active_classification in ['TP', 'TN'] and inactive_classification in ['TP', 'TN']
        one_correct = (active_classification in ['TP', 'TN']) != (inactive_classification in ['TP', 'TN'])

        if both_correct:
            pair_classification = "BothCorrect"
        elif one_correct:
            pair_classification = "OneCorrect"
        else:
            pair_classification = "BothWrong"

        # UPDATE ONLY the result dictionary to include new column names
        result = {
            'pair_id': pair_id,
            'pair_type': pair_type,
            'antibiotic_class': antibiotic_class,  # Updated column name (was 'class')
            'active_compound_id': active_compound_id,  # New column
            'inactive_compound_id': inactive_compound_id,  # New column
            'active_smiles': active_smiles,
            'inactive_smiles': inactive_smiles,
            'similarity': similarity,
            'activity_difference': activity_difference,
            'fold_difference': fold_difference,  # New column
            'active_group': active_group,  # New column
            'inactive_group': inactive_group,  # New column
            'group_combination': group_combination,  # New column
            'active_target_ground_truth': int(active_target),
            'inactive_target_ground_truth': int(inactive_target),

            # Unified comparator fields (atom-level masses and delta)
            'attr_mode': 'shap',
            'atom_attr_active': json.dumps([float(v) for v in atom_attr_active_u]) if atom_attr_active_u else json.dumps([]),
            'atom_attr_inactive': json.dumps([float(v) for v in atom_attr_inactive_u]) if atom_attr_inactive_u else json.dumps([]),
            'atom_mass_active': float(atom_mass_active),
            'atom_mass_inactive': float(atom_mass_inactive),
            'core_align_active': float(core_a_mass),
            'core_align_inactive': float(core_i_mass),
            'edit_mass': float(edit_mass),
            'context_mass': float(context_mass),
            'delta_core_align': float(delta_core_align),

            # Feature-level cross-model comparison columns
            'model_type': 'RF',
            'xai_method': 'TreeSHAP',
            'feature_scores_active': json.dumps(feature_comparison['feature_scores_active']),
            'feature_scores_inactive': json.dumps(feature_comparison['feature_scores_inactive']),
            'feature_delta_abs': json.dumps(feature_comparison['feature_delta_abs']),
            'topk_features_changed': json.dumps(feature_comparison['topk_features']),
            'feature_presence_active': json.dumps(presence_active),
            'feature_presence_inactive': json.dumps(presence_inactive),
            'feature_changes': json.dumps(compute_edit_sets(presence_active, presence_inactive)),
            'edit_driver_candidates': json.dumps(edit_candidates),
            'top_edit_driver': top_edit_driver,
            'edit_driver_support': top_edit_driver_support,
            'pos_features_active': json.dumps(pos_a_list),
            'neg_features_active': json.dumps(neg_a_list),
            'pos_features_inactive': json.dumps(pos_i_list),
            'neg_features_inactive': json.dumps(neg_i_list),

            # KEEP ALL EXISTING RF-SPECIFIC COLUMNS EXACTLY AS THEY ARE:
            'active_pred_prob': active_pred_prob,
            'inactive_pred_prob': inactive_pred_prob,
            'prediction_difference': prediction_difference,
            'active_class': active_class,
            'inactive_class': inactive_class,
            'pair_classification': pair_classification,
            'active_classification': active_classification,
            'inactive_classification': inactive_classification,
            'common_features_count': int(common_features_count),
            'active_features_count': int(active_features_count),
            'inactive_features_count': int(inactive_features_count),
            'feature_differences': str(feature_differences),
            'max_feature_diff': max_feature_diff,
            'mean_feature_diff': mean_feature_diff,
            'propagation_index': deltas['propagation_index'],
            'core_edit_ratio': deltas['core_edit_ratio'],
            'delta_by_distance': str(deltas['delta_by_distance']),
            'edit_concentration_index': deltas['edit_concentration_index'],
            'total_delta': deltas['total_delta'],
            'distant_delta': deltas['distant_delta'],
            'core_delta': deltas['core_delta'],
            'edit_delta': deltas['edit_delta'],
            'edit_basis': edit_basis,
            'changed_functional_groups': str(changed_functional_groups),
            'rf_visible': rf_visible,
            'xai_visible': rf_visible
        }

        # Attach visualization fields and pred-attr alignment
        # Prediction–attribution alignment heuristic based on sign mass
        def _pred_attr_alignment(prob: float, atom_attr: List[float]) -> Tuple[bool, float]:
            """Simple prediction-attribution alignment based on sign distribution."""
            arr = np.asarray(atom_attr, dtype=float)
            if arr.size == 0 or not np.isfinite(prob):
                return False, 0.0
            
            positive_fraction = float(np.mean(arr > 0))
            alignment = bool((prob >= THRESHOLD and positive_fraction >= 0.5) or 
                           (prob < THRESHOLD and positive_fraction <= 0.5))
            return alignment, positive_fraction
        def weighted_pred_attr_alignment(prob: float, atom_attr: List[float], threshold: float = THRESHOLD):
            """Magnitude-weighted prediction-attribution alignment."""
            arr = np.asarray(atom_attr, dtype=float)
            if arr.size == 0 or not np.isfinite(prob):
                return False, 0.0, {'method': 'weighted', 'valid': False}
            
            weights = np.abs(arr)
            total_weight = float(weights.sum())
            if total_weight < 1e-12:
                return False, 0.0, {'method': 'weighted', 'valid': False, 'reason': 'zero_weight'}
            
            positive_weight = float((weights * (arr > 0)).sum())
            fraction = positive_weight / total_weight
            # One-time debug print for weighted mismatch sanity
            if not hasattr(weighted_pred_attr_alignment, '_logged'):
                print(f"[RF-QC-DEBUG] Sample weights: total={total_weight:.6f} positive={positive_weight:.6f} fraction={fraction:.3f}")
                weighted_pred_attr_alignment._logged = True
            alignment = bool((prob >= threshold and fraction >= 0.5) or (prob < threshold and fraction < 0.5))
            
            metrics = {
                'method': 'weighted', 'valid': True,
                'total_weight': total_weight, 'positive_weight': positive_weight,
                'negative_weight': total_weight - positive_weight, 'weighted_pos_fraction': fraction,
                'num_atoms': int(arr.size), 'num_positive_atoms': int((arr > 0).sum()),
                'mean_abs_attribution': float(np.mean(weights))
            }
            return alignment, fraction, metrics
        align_a, posfrac_a = _pred_attr_alignment(active_pred_prob, viz_active.get('atom_attributions', []))
        align_i, posfrac_i = _pred_attr_alignment(inactive_pred_prob, viz_inactive.get('atom_attributions', []))
        pair_align = bool(align_a and align_i)
        # Weighted alignment
        w_align_a, w_frac_a, w_metrics_a = weighted_pred_attr_alignment(active_pred_prob, viz_active.get('atom_attributions', []))
        w_align_i, w_frac_i, w_metrics_i = weighted_pred_attr_alignment(inactive_pred_prob, viz_inactive.get('atom_attributions', []))
        pair_align_weighted = bool(w_align_a and w_align_i)

        result.update({
            'viz_active_atom_attr': json.dumps(viz_active.get('atom_attributions', [])),
            'viz_active_atom_colors': json.dumps(viz_active.get('atom_colors', [])),
            'viz_active_positive_atoms': json.dumps(viz_active.get('positive_atoms', [])),
            'viz_active_negative_atoms': json.dumps(viz_active.get('negative_atoms', [])),
            'viz_inactive_atom_attr': json.dumps(viz_inactive.get('atom_attributions', [])),
            'viz_inactive_atom_colors': json.dumps(viz_inactive.get('atom_colors', [])),
            'viz_inactive_positive_atoms': json.dumps(viz_inactive.get('positive_atoms', [])),
            'viz_inactive_negative_atoms': json.dumps(viz_inactive.get('negative_atoms', [])),
            'pred_attr_alignment_active': bool(align_a),
            'pred_attr_alignment_inactive': bool(align_i),
            'pred_attr_alignment_pair': bool(pair_align),
            'pred_attr_positive_fraction_active': float(posfrac_a),
            'pred_attr_positive_fraction_inactive': float(posfrac_i),
            'pred_attr_mismatch_active': bool(not align_a),
            'pred_attr_mismatch_inactive': bool(not align_i),
            'pred_attr_mismatch_pair': bool(not pair_align),
            # Weighted alignment outputs
            'pred_attr_alignment_active_weighted': bool(w_align_a),
            'pred_attr_alignment_inactive_weighted': bool(w_align_i),
            'pred_attr_alignment_pair_weighted': bool(pair_align_weighted),
            'pred_attr_positive_fraction_active_weighted': float(w_frac_a),
            'pred_attr_positive_fraction_inactive_weighted': float(w_frac_i),
            'pred_attr_mismatch_active_weighted': bool(not w_align_a),
            'pred_attr_mismatch_inactive_weighted': bool(not w_align_i),
            'pred_attr_mismatch_pair_weighted': bool(not pair_align_weighted),
            'weighted_alignment_metrics_active': json.dumps(w_metrics_a),
            'weighted_alignment_metrics_inactive': json.dumps(w_metrics_i),
            'attribution_std_across_models': float(attr_std),
        })

        # RF operates in 85-fragment feature space; atom-level pharmacophore SMARTS not applicable (N/A in this pass)
        result.update({
            'pharm_recognized_active': json.dumps([]),
            'pharm_missed_active': json.dumps([]),
            'pharm_recognition_score_active': float('nan'),
            'pharm_recognition_rate_active': float('nan'),
            'pharm_recognized_inactive': json.dumps([]),
            'pharm_missed_inactive': json.dumps([]),
            'pharm_recognition_score_inactive': float('nan'),
            'pharm_recognition_rate_inactive': float('nan'),
            'pharm_recognition_score_pair': float('nan'),
            'pharmacophore_consistency_score': float('nan'),
            'pharmacophore_inconsistent_flag': False,
        })
        # Annotate model context and pair type
        result['pair_type'] = pair_type
        result['model_id'] = CURRENT_MODEL_ID
        result['is_ensemble'] = bool(CURRENT_IS_ENSEMBLE)

        # Aliases for non-cliff pairs to use compound1/compound2 naming
        if pair_type == 'non_cliff':
            try:
                result['compound1_id'] = result.get('active_compound_id', row.get('compound1_id'))
                result['compound2_id'] = result.get('inactive_compound_id', row.get('compound2_id'))
                result['compound1_pred_prob'] = float(result.get('active_pred_prob', np.nan))
                result['compound2_pred_prob'] = float(result.get('inactive_pred_prob', np.nan))
                # Feature scores aliases
                if 'feature_scores_active' in result:
                    result['feature_scores_compound1'] = result['feature_scores_active']
                if 'feature_scores_inactive' in result:
                    result['feature_scores_compound2'] = result['feature_scores_inactive']
                # Positive/negative feature aliases
                if 'pos_features_active' in result:
                    result['pos_features_compound1'] = result['pos_features_active']
                if 'neg_features_active' in result:
                    result['neg_features_compound1'] = result['neg_features_active']
                if 'pos_features_inactive' in result:
                    result['pos_features_compound2'] = result['pos_features_inactive']
                if 'neg_features_inactive' in result:
                    result['neg_features_compound2'] = result['neg_features_inactive']
            except Exception:
                pass

        return result

    except Exception as e:
        print(f"Error analyzing pair {pair_type} row {getattr(row, 'name', 'unknown')}: {e}")
        traceback.print_exc()
        return None

def visibility_aware_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute visibility-aware statistics and second-generation p-values."""

    # Coverage table
    coverage_stats = {}
    for pair_type in ['cliff', 'non_cliff']:
        subset = df[df['pair_type'] == pair_type]
        rf_visible_n = len(subset[subset['rf_visible'] == True])
        total_n = len(subset)
        coverage_pct = (rf_visible_n / total_n * 100) if total_n > 0 else 0

        coverage_stats[pair_type] = {
            'rf_visible_n': rf_visible_n,
            'total_n': total_n,
            'coverage_%': coverage_pct
        }

    # Visibility-aware means (only rf_visible == True)
    rf_visible_df = df[df['rf_visible'] == True]

    means_stats = {}
    for pair_type in ['cliff', 'non_cliff']:
        subset = rf_visible_df[rf_visible_df['pair_type'] == pair_type]
        if len(subset) > 0:
            means_stats[pair_type] = {
                'propagation_index_mean': subset['propagation_index'].mean(),
                'core_edit_ratio_mean': subset['core_edit_ratio'].mean()
            }
        else:
            means_stats[pair_type] = {
                'propagation_index_mean': 0.0,
                'core_edit_ratio_mean': 0.0
            }

    # Context sensitivity threshold τ
    rf_visible_non_cliffs = rf_visible_df[rf_visible_df['pair_type'] == 'non_cliff']
    if len(rf_visible_non_cliffs) > 0:
        tau_noncliff_P95 = np.percentile(rf_visible_non_cliffs['total_delta'], 95)
    else:
        tau_noncliff_P95 = 0.0

    rf_visible_cliffs = rf_visible_df[rf_visible_df['pair_type'] == 'cliff']
    if len(rf_visible_cliffs) > 0:
        context_sensitive_rate = np.mean(rf_visible_cliffs['total_delta'] > tau_noncliff_P95)
    else:
        context_sensitive_rate = 0.0

    # Bootstrap analysis and second-generation p-values
    bootstrap_stats = {}

    for metric in ['propagation_index', 'core_edit_ratio']:
        cliff_values = rf_visible_df[rf_visible_df['pair_type'] == 'cliff'][metric].values
        non_cliff_values = rf_visible_df[rf_visible_df['pair_type'] == 'non_cliff'][metric].values

        if len(cliff_values) > 0 and len(non_cliff_values) > 0:
            # Bootstrap the mean difference
            def stat_func(cliff_sample, non_cliff_sample):
                return np.mean(cliff_sample) - np.mean(non_cliff_sample)

            # Prepare data for bootstrap
            rng = np.random.default_rng(42)
            n_bootstrap = 3000

            bootstrap_diffs = []
            for _ in range(n_bootstrap):
                cliff_sample = rng.choice(cliff_values, size=len(cliff_values), replace=True)
                non_cliff_sample = rng.choice(non_cliff_values, size=len(non_cliff_values), replace=True)
                diff = stat_func(cliff_sample, non_cliff_sample)
                bootstrap_diffs.append(diff)

            bootstrap_diffs = np.array(bootstrap_diffs)

            # 95% confidence interval
            ci_lower = np.percentile(bootstrap_diffs, 2.5)
            ci_upper = np.percentile(bootstrap_diffs, 97.5)
            bootstrap_mean = np.mean(bootstrap_diffs)

            # Second-generation p-value with practical null interval [-0.01, +0.01]
            null_interval = [-0.01, 0.01]
            p_delta = np.mean((bootstrap_diffs >= null_interval[0]) &
                            (bootstrap_diffs <= null_interval[1]))

            bootstrap_stats[metric] = {
                'bootstrap_mean': bootstrap_mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_delta': p_delta
            }
        else:
            bootstrap_stats[metric] = {
                'bootstrap_mean': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'p_delta': 1.0
            }

    return {
        'coverage_stats': coverage_stats,
        'means_stats': means_stats,
        'tau_noncliff_P95': tau_noncliff_P95,
        'context_sensitive_rate': context_sensitive_rate,
        'bootstrap_stats': bootstrap_stats
    }

def main():
    """Main execution function."""
    global RUN_MODE, THRESHOLD, ACTIVITY_CSV, NONCLIFF_CSV
    parser = argparse.ArgumentParser(description='RF XAI Activity Pairs (TreeSHAP)')
    parser.add_argument('--per_model', action='store_true', help='Use a single CV/fold model as backbone')
    parser.add_argument('--ensemble', action='store_true', help='Use the full ensemble (default if --per_model not set)')
    parser.add_argument('--cv', type=int, default=None, help='Cross-validation index for --per_model')
    parser.add_argument('--fold', type=int, default=None, help='Fold index for --per_model')
    parser.add_argument('--full', action='store_true', help='Process full datasets; otherwise TEST mode sampling is used')
    parser.add_argument('--threshold', type=float, default=THRESHOLD, help='Classification threshold')
    parser.add_argument('--activity_csv', type=str, default=ACTIVITY_CSV, help='Path to activity cliff pairs CSV')
    parser.add_argument('--noncliff_csv', type=str, default=NONCLIFF_CSV, help='Path to non-cliff pairs CSV')
    args, _ = parser.parse_known_args()

    print("RF XAI Activity Pairs Analysis")
    print("=" * 50)

    # Override globals from CLI
    RUN_MODE = "FULL" if args.full else RUN_MODE
    THRESHOLD = float(args.threshold)
    ACTIVITY_CSV = args.activity_csv
    NONCLIFF_CSV = args.noncliff_csv

    print(f"Run mode: {RUN_MODE}")
    print(f"Classification threshold: {THRESHOLD}")
    print()

    # Initialize ensemble once at the beginning
    print("Initializing ensemble models...")
    global _ensemble, CURRENT_MODEL_ID, CURRENT_IS_ENSEMBLE
    _ensemble = EnsembleRandomForestExplainer()

    # Per-model restriction
    if args.per_model and (args.cv is not None and args.fold is not None):
        print(f"Restricting to single model: cv{args.cv}_fold{args.fold}")
        _ensemble.restrict_to_cv_fold(int(args.cv), int(args.fold))
        CURRENT_MODEL_ID = f"cv{int(args.cv)}_fold{int(args.fold)}"
        CURRENT_IS_ENSEMBLE = False
    elif args.per_model and (args.cv is None or args.fold is None):
        # Fallback: first available
        print("--per_model specified but --cv/--fold missing; using first available model")
        CURRENT_MODEL_ID = f"cv{_ensemble.best_models_info[0].get('cv')}_fold{_ensemble.best_models_info[0].get('fold')}"
        _ensemble.restrict_to_cv_fold(int(_ensemble.best_models_info[0].get('cv')), int(_ensemble.best_models_info[0].get('fold')))
        CURRENT_IS_ENSEMBLE = False
    else:
        CURRENT_MODEL_ID = "ensemble"
        CURRENT_IS_ENSEMBLE = True
    print()

    # Load data
    df_cliff, df_non = load_data()

    # Subsample for smoke test
    if RUN_MODE == "SMOKE":
        print(f"SMOKE TEST MODE: Processing {SMOKE_N_CLIFF} cliffs and {SMOKE_N_NON} non-cliffs")
        df_cliff = df_cliff.sample(n=min(SMOKE_N_CLIFF, len(df_cliff)), random_state=42).reset_index(drop=True)
        df_non = df_non.sample(n=min(SMOKE_N_NON, len(df_non)), random_state=42).reset_index(drop=True)
    else:
        print("FULL ANALYSIS MODE: Processing all pairs")

    print(f"Final dataset: {len(df_cliff)} cliffs, {len(df_non)} non-cliffs")
    print()

    # Process pairs
    all_results = []

    print("Processing activity cliff pairs...")
    for idx, row in tqdm(df_cliff.iterrows(), total=len(df_cliff), desc="Cliffs"):
        result = analyze_pair(row, "cliff")
        if result:
            all_results.append(result)

    print("Processing non-cliff pairs...")
    for idx, row in tqdm(df_non.iterrows(), total=len(df_non), desc="Non-cliffs"):
        result = analyze_pair(row, "non_cliff")
        if result:
            all_results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    print(f"\nProcessed {len(results_df)} pairs successfully")
    print(f"RF visible pairs: {len(results_df[results_df['rf_visible'] == True])}")

    # Verify balanced processing
    verify_balanced_processing(results_df)

    # Helper function to standardize schema for evaluation (parity with CNN/RGCN)
    def project_eval_schema(df: pd.DataFrame) -> pd.DataFrame:
        """Project RF results to standardized evaluation schema.

        For cliffs: compound_active_id/compound_inactive_id semantics
        For non-cliffs: compound1_id/compound2_id semantics

        RF uses feature_scores_active/inactive as equivalent to substruct_attr.
        """
        tmp = df.copy()

        # Determine if this is cliff or non-cliff based on pair_type
        is_cliff = False
        try:
            if 'pair_type' in tmp.columns and len(tmp) > 0:
                is_cliff = str(tmp['pair_type'].iloc[0]).strip() == 'cliff'
        except Exception:
            pass

        if is_cliff:
            # Cliff pairs: use active/inactive semantics
            # Rename ID columns
            for a_key in ['active_compound_id', 'compound1_id', 'active_id']:
                if a_key in tmp.columns and 'compound_active_id' not in tmp.columns:
                    tmp.rename(columns={a_key: 'compound_active_id'}, inplace=True)
                    break
            for b_key in ['inactive_compound_id', 'compound2_id', 'inactive_id']:
                if b_key in tmp.columns and 'compound_inactive_id' not in tmp.columns:
                    tmp.rename(columns={b_key: 'compound_inactive_id'}, inplace=True)
                    break

            # Rename probability columns
            for pa in ['active_pred_prob']:
                if pa in tmp.columns and 'compound_active_pred_prob' not in tmp.columns:
                    tmp.rename(columns={pa: 'compound_active_pred_prob'}, inplace=True)
                    break
            for pi in ['inactive_pred_prob']:
                if pi in tmp.columns and 'compound_inactive_pred_prob' not in tmp.columns:
                    tmp.rename(columns={pi: 'compound_inactive_pred_prob'}, inplace=True)
                    break

            # Add predicted class columns
            if 'compound_active_pred_prob' in tmp.columns:
                tmp['compound_active_pred_class'] = (tmp['compound_active_pred_prob'] >= 0.5).map({True: 'active', False: 'inactive'})
            if 'compound_inactive_pred_prob' in tmp.columns:
                tmp['compound_inactive_pred_class'] = (tmp['compound_inactive_pred_prob'] >= 0.5).map({True: 'active', False: 'inactive'})

            # Rename SMILES columns
            if 'active_smiles' in tmp.columns:
                tmp.rename(columns={'active_smiles': 'active_smiles'}, inplace=True)
            if 'inactive_smiles' in tmp.columns:
                tmp.rename(columns={'inactive_smiles': 'inactive_smiles'}, inplace=True)

            # RF-specific: feature_scores are equivalent to substruct_attr
            # Note: These stay as JSON strings, similar to CNN/RGCN substruct columns
            if 'feature_scores_active' in tmp.columns:
                tmp.rename(columns={'feature_scores_active': 'feature_attr_active'}, inplace=True)
            if 'feature_scores_inactive' in tmp.columns:
                tmp.rename(columns={'feature_scores_inactive': 'feature_attr_inactive'}, inplace=True)

            # Also rename pos/neg feature lists to match substruct naming pattern
            if 'pos_features_active' in tmp.columns:
                tmp.rename(columns={'pos_features_active': 'pos_features_active'}, inplace=True)
            if 'neg_features_active' in tmp.columns:
                tmp.rename(columns={'neg_features_active': 'neg_features_active'}, inplace=True)
            if 'pos_features_inactive' in tmp.columns:
                tmp.rename(columns={'pos_features_inactive': 'pos_features_inactive'}, inplace=True)
            if 'neg_features_inactive' in tmp.columns:
                tmp.rename(columns={'neg_features_inactive': 'neg_features_inactive'}, inplace=True)

            # Select only necessary columns for evaluation
            cols = [c for c in [
                'pair_type', 'antibiotic_class',
                'compound_active_id', 'compound_inactive_id',
                'compound_active_pred_prob', 'compound_inactive_pred_prob',
                'compound_active_pred_class', 'compound_inactive_pred_class',
                'feature_attr_active', 'feature_attr_inactive',
                'pos_features_active', 'neg_features_active',
                'pos_features_inactive', 'neg_features_inactive',
                'active_smiles', 'inactive_smiles',
                'model_type'
            ] if c in tmp.columns]
            return tmp[cols]

        else:
            # Non-cliff pairs: use compound1/compound2 semantics
            # Rename ID columns
            for a_key in ['active_compound_id', 'compound1_compound_id', 'active_id', 'compound1_id']:
                if a_key in tmp.columns:
                    tmp.rename(columns={a_key: 'compound1_id'}, inplace=True)
                    break
            for b_key in ['inactive_compound_id', 'compound2_compound_id', 'inactive_id', 'compound2_id']:
                if b_key in tmp.columns:
                    tmp.rename(columns={b_key: 'compound2_id'}, inplace=True)
                    break

            # Rename probability columns
            for pa in ['active_pred_prob', 'compound1_pred_prob']:
                if pa in tmp.columns:
                    tmp.rename(columns={pa: 'compound1_pred_prob'}, inplace=True)
                    break
            for pi in ['inactive_pred_prob', 'compound2_pred_prob']:
                if pi in tmp.columns:
                    tmp.rename(columns={pi: 'compound2_pred_prob'}, inplace=True)
                    break

            # Add predicted class columns
            if 'compound1_pred_prob' in tmp.columns:
                tmp['compound1_pred_class'] = (tmp['compound1_pred_prob'] >= 0.5).map({True: 'active', False: 'inactive'})
            if 'compound2_pred_prob' in tmp.columns:
                tmp['compound2_pred_class'] = (tmp['compound2_pred_prob'] >= 0.5).map({True: 'active', False: 'inactive'})

            # Rename SMILES columns
            if 'active_smiles' in tmp.columns:
                tmp.rename(columns={'active_smiles': 'compound1_smiles'}, inplace=True)
            if 'inactive_smiles' in tmp.columns:
                tmp.rename(columns={'inactive_smiles': 'compound2_smiles'}, inplace=True)

            # RF-specific: feature_scores to feature_attr
            if 'feature_scores_active' in tmp.columns:
                tmp.rename(columns={'feature_scores_active': 'feature_attr_compound1'}, inplace=True)
            if 'feature_scores_inactive' in tmp.columns:
                tmp.rename(columns={'feature_scores_inactive': 'feature_attr_compound2'}, inplace=True)

            # Rename feature lists
            if 'pos_features_active' in tmp.columns:
                tmp.rename(columns={'pos_features_active': 'pos_features_compound1'}, inplace=True)
            if 'neg_features_active' in tmp.columns:
                tmp.rename(columns={'neg_features_active': 'neg_features_compound1'}, inplace=True)
            if 'pos_features_inactive' in tmp.columns:
                tmp.rename(columns={'pos_features_inactive': 'pos_features_compound2'}, inplace=True)
            if 'neg_features_inactive' in tmp.columns:
                tmp.rename(columns={'neg_features_inactive': 'neg_features_compound2'}, inplace=True)

            # Select only necessary columns for evaluation
            base_cols = [c for c in [
                'pair_type', 'antibiotic_class',
                'compound1_id', 'compound2_id',
                'compound1_pred_prob', 'compound2_pred_prob',
                'compound1_pred_class', 'compound2_pred_class',
                'feature_attr_compound1', 'feature_attr_compound2',
                'pos_features_compound1', 'neg_features_compound1',
                'pos_features_compound2', 'neg_features_compound2',
                'compound1_smiles', 'compound2_smiles',
                'model_type'
            ] if c in tmp.columns]
            return tmp[base_cols]

    # Save results with variant-aware naming (per-model vs ensemble)
    # NOTE: legacy "balanced_detailed" files are disabled to avoid bloated schemas.
    mode_suffix = "test" if RUN_MODE == "TEST" else "full"
    variant = CURRENT_MODEL_ID if not bool(CURRENT_IS_ENSEMBLE) else "ensemble"
    # Also emit split convenience files by pair_type for parity with CNN/RGCN
    try:
        cliffs = results_df[results_df['pair_type'] == 'cliff'].copy()
        noncliffs = results_df[results_df['pair_type'] == 'non_cliff'].copy()

        # Apply standardized schema
        cliffs_std = project_eval_schema(cliffs)
        noncliffs_std = project_eval_schema(noncliffs)

        cliffs_path = f"rf_{variant}_cliffs.csv"
        noncliffs_path = f"rf_{variant}_non_cliffs.csv"
        cliffs_std.to_csv(cliffs_path, index=False)
        noncliffs_std.to_csv(noncliffs_path, index=False)
        print(f"Saved: {cliffs_path}")
        print(f"Saved: {noncliffs_path}")

        # Also save Parquet sidecar for nested JSON fidelity
        try:
            cliffs_std.to_parquet(cliffs_path.replace('.csv', '.parquet'), index=False)
            noncliffs_std.to_parquet(noncliffs_path.replace('.csv', '.parquet'), index=False)
        except Exception as e:
            print(f"Parquet save skipped: {e}")
        # Aliases rf_model{1..5}_cliffs.csv for per-model runs
        if not bool(CURRENT_IS_ENSEMBLE):
            try:
                # Parse CURRENT_MODEL_ID = cvX_foldY and find 1-based index in best_models.json
                parts = str(CURRENT_MODEL_ID).lower().replace('cv','').split('_fold')
                cv_id = int(parts[0]) if len(parts) == 2 else None
                fold_id = int(parts[1]) if len(parts) == 2 else None
                alias_idx = None
                if cv_id is not None and fold_id is not None:
                    for i, mi in enumerate(_ensemble.best_models_info, start=1):
                        if int(mi.get('cv')) == cv_id and int(mi.get('fold')) == fold_id:
                            alias_idx = i
                            break
                if alias_idx is not None:
                    alias_cliffs = f"rf_model{alias_idx}_cliffs.csv"
                    alias_noncliffs = f"rf_model{alias_idx}_non_cliffs.csv"
                    cliffs_std.to_csv(alias_cliffs, index=False)
                    noncliffs_std.to_csv(alias_noncliffs, index=False)
                    print(f"Saved: {alias_cliffs}")
                    print(f"Saved: {alias_noncliffs}")
                    # Parquet for aliases too
                    try:
                        cliffs_std.to_parquet(alias_cliffs.replace('.csv', '.parquet'), index=False)
                        noncliffs_std.to_parquet(alias_noncliffs.replace('.csv', '.parquet'), index=False)
                    except Exception:
                        pass
            except Exception as e:
                print(f"Warning: failed to write alias modelN CSVs: {e}")
    except Exception as e:
        print(f"Warning: failed to write split CSVs: {e}")

    # Also write compact, model-agnostic summary CSV for cross-model comparisons
    try:
        summary_cols = [
            'pair_id', 'pair_type', 'antibiotic_class',
            'active_compound_id', 'inactive_compound_id', 'similarity',
            'model_type', 'xai_method',
            'active_pred_prob', 'inactive_pred_prob', 'prediction_difference', 'pair_classification',
            'xai_visible', 'common_features_count',
            # raw deltas and indices (we will also add edit_mass/context_mass copies)
            'edit_delta', 'distant_delta', 'propagation_index', 'edit_concentration_index',
            'total_delta', 'core_delta', 'distant_delta',
            'topk_features_changed', 'top_edit_driver', 'edit_driver_support'
        ]

        # Build a friendly schema / rename to compact names
        def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
            tmp = df.copy()
            # Rename for compact schema (IDs only)
            rename_map = {
                'active_compound_id': 'active_id',
                'inactive_compound_id': 'inactive_id',
            }
            # Ensure required columns exist
            for c in ['xai_visible']:
                if c not in tmp.columns and 'rf_visible' in tmp.columns:
                    tmp[c] = tmp['rf_visible']

            # Select and rename
            # Filter available columns first to avoid KeyErrors
            available = [c for c in summary_cols if c in tmp.columns]
            out = tmp[available].rename(columns=rename_map)

            # Add duplicated fields for readability without losing raw columns
            if 'edit_delta' in out.columns and 'edit_mass' not in out.columns:
                out['edit_mass'] = out['edit_delta']
            if 'distant_delta' in out.columns and 'context_mass' not in out.columns:
                out['context_mass'] = out['distant_delta']

            # Reorder to final desired shape (only include those present)
            final_order = [
                'pair_id', 'pair_type', 'antibiotic_class', 'active_id', 'inactive_id', 'similarity',
                'model_type', 'xai_method',
                'active_pred_prob', 'inactive_pred_prob', 'prediction_difference', 'pair_classification',
                'xai_visible', 'common_features_count',
                'edit_mass', 'context_mass', 'propagation_index', 'edit_concentration_index',
                'total_delta', 'core_delta', 'distant_delta',
                'topk_features_changed', 'top_edit_driver', 'edit_driver_support'
            ]
            out = out[[c for c in final_order if c in out.columns]]
            return out

        summary_df = _build_summary(results_df)
        summary_path = 'rf_xai_pairs_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")
    except Exception as e:
        print(f"Warning: failed to write compact summary CSV: {e}")

    # Compute visibility-aware statistics
    print("\nComputing visibility-aware statistics...")
    stats = visibility_aware_stats(results_df)

    # Overwrite compact, model-agnostic summary CSV now that tau is known
    try:
        tau = stats.get('tau_noncliff_P95', None)
        def _build_summary_with_norms(df: pd.DataFrame, tau_val) -> pd.DataFrame:
            tmp = df.copy()
            # IDs and basic fields
            if 'xai_visible' not in tmp.columns and 'rf_visible' in tmp.columns:
                tmp['xai_visible'] = tmp['rf_visible']
            # Normalized masses
            denom = (tmp.get('edit_delta', 0) + tmp.get('distant_delta', 0)).replace(0, np.nan)
            tmp['edit_mass'] = tmp.get('edit_delta', 0)
            tmp['context_mass'] = tmp.get('distant_delta', 0)
            tmp['edit_mass_norm'] = (tmp['edit_mass'] / denom).fillna(0.0)
            tmp['context_mass_norm'] = (tmp['context_mass'] / denom).fillna(0.0)
            # Driver support normalized
            if 'edit_driver_support' not in tmp.columns:
                tmp['edit_driver_support'] = 0.0
            tmp['edit_driver_support_norm'] = (tmp['edit_driver_support'] / denom).fillna(0.0)
            # Context sensitivity flag (requires tau)
            if tau_val is not None:
                tmp['context_sensitive'] = tmp['total_delta'] > float(tau_val)
            else:
                tmp['context_sensitive'] = False
            # Both-correct boolean for quick accuracy filtering
            tmp['both_correct'] = tmp['pair_classification'].astype(str).str.lower().eq('both correct')
            # Rename IDs
            tmp = tmp.rename(columns={
                'active_compound_id': 'active_id',
                'inactive_compound_id': 'inactive_id'
            })
            # Select final columns (only those present)
            final_cols = [
                'pair_id', 'pair_type', 'antibiotic_class', 'active_id', 'inactive_id', 'similarity',
                'model_type', 'xai_method',
                'active_pred_prob', 'inactive_pred_prob', 'prediction_difference', 'pair_classification',
                'both_correct', 'xai_visible', 'common_features_count',
                'edit_mass', 'context_mass', 'edit_mass_norm', 'context_mass_norm',
                'propagation_index', 'edit_concentration_index',
                'total_delta', 'core_delta', 'distant_delta',
                'topk_features_changed', 'top_edit_driver', 'edit_driver_support', 'edit_driver_support_norm',
                'context_sensitive'
            ]
            available = [c for c in final_cols if c in tmp.columns]
            return tmp[available]

        summary_df2 = _build_summary_with_norms(results_df, tau)
        summary_path = 'rf_xai_pairs_summary.csv'
        summary_df2.to_csv(summary_path, index=False)
        print(f"Saved (updated with norms/tau): {summary_path}")
    except Exception as e:
        print(f"Warning: failed to write updated compact summary CSV: {e}")

    # Write summary report
    with open("rf_xai_summary.txt", "w", encoding='utf-8') as f:
        f.write("RF XAI Activity Pairs Analysis Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write("Coverage Table (RF Visibility):\n")
        f.write("pair_type, rf_visible_n, total_n, coverage_%\n")
        for pair_type, data in stats['coverage_stats'].items():
            f.write(f"{pair_type}, {data['rf_visible_n']}, {data['total_n']}, {data['coverage_%']:.1f}\n")
        f.write("\n")

        f.write("Visibility-aware Means (RF Visible Only):\n")
        for pair_type, data in stats['means_stats'].items():
            f.write(f"{pair_type}:\n")
            f.write(f"  Propagation Index: {data['propagation_index_mean']:.4f}\n")
            f.write(f"  Core Edit Ratio: {data['core_edit_ratio_mean']:.4f}\n")
        f.write("\n")

        f.write(f"Context Sensitivity Threshold (tau): {stats['tau_noncliff_P95']:.4f}\n")
        f.write(f"Context Sensitive Rate: {stats['context_sensitive_rate']:.4f}\n\n")

        f.write("Bootstrap 95% CIs and Second-Generation P-values:\n")
        for metric, data in stats['bootstrap_stats'].items():
            f.write(f"{metric}:\n")
            f.write(f"  Bootstrap Mean Difference: {data['bootstrap_mean']:.4f}\n")
            f.write(f"  95% CI: [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}]\n")
            f.write(f"  Second-gen p-value (p_delta): {data['p_delta']:.4f}\n\n")

        f.write(f"Analysis completed with SHAP TreeExplainer for fr_ functional group reliability testing\n")
        f.write(f"Random seed: 42\n")

    print("Saved: rf_xai_summary.txt")

    # Save machine-readable summary
    with open("rf_xai_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved: rf_xai_summary.json")

    # Print key results to console
    print("\n" + "=" * 50)
    print("KEY RESULTS:")
    print("=" * 50)

    for pair_type, data in stats['coverage_stats'].items():
        print(f"{pair_type}: {data['coverage_%']:.1f}% RF visible ({data['rf_visible_n']}/{data['total_n']})")

    print(f"\nContext sensitivity threshold: {stats['tau_noncliff_P95']:.4f}")
    print(f"Context sensitive rate: {stats['context_sensitive_rate']:.4f}")

    print(f"\nBootstrap results (cliff - non_cliff means):")
    for metric, data in stats['bootstrap_stats'].items():
        print(f"{metric}: {data['bootstrap_mean']:.4f} [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}], p_delta={data['p_delta']:.4f}")

    print("\nAnalysis complete!")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RF XAI Balanced Activity Pairs Analysis')
    parser.add_argument('--full', action='store_true',
                       help='Run full analysis on all pairs (default: test mode with 8 pairs each)')
    parser.add_argument('--pdf', action='store_true',
                       help='Generate plots and PDF report')
    parser.add_argument('--activity_csv', type=str, default='activity_cliff_pairs.csv',
                        help='Path to activity cliff pairs CSV')
    parser.add_argument('--noncliff_csv', type=str, default='non_cliff_pairs.csv',
                        help='Path to non-cliff pairs CSV')
    parser.add_argument('--per_model', action='store_true',
                        help='Run per-model analysis and write separate CSVs')
    parser.add_argument('--ensemble', action='store_true',
                        help='Run ensemble analysis (default)')
    parser.add_argument('--limit_per_class', type=int, default=None,
                        help='Optional limit on number of pairs per class for quick runs')
    return parser.parse_args()

def create_similarity_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add similarity bins to dataframe for stratified analysis."""
    if 'similarity' not in df.columns:
        df['similarity_bin'] = 'unknown'
        return df

    # Define similarity bins
    bins = [0, 0.3, 0.5, 0.7, 0.85, 1.0]
    labels = ['very_low', 'low', 'medium', 'high', 'very_high']

    # Create bins
    df['similarity_bin'] = pd.cut(df['similarity'], bins=bins, labels=labels, include_lowest=True)
    df['similarity_bin'] = df['similarity_bin'].astype(str).replace('nan', 'unknown')

    return df

def functional_group_enrichment(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze functional group changes across pairs.

    Robust to missing rf_visible and changed_functional_groups columns.
    """
    from collections import Counter

    # Filter to rf_visible when available; otherwise use all rows
    if 'rf_visible' in df.columns:
        try:
            rf_visible_df = df[df['rf_visible'] == True].copy()
        except Exception:
            rf_visible_df = df.copy()
    else:
        rf_visible_df = df.copy()
    if len(rf_visible_df) == 0:
        return {'error': 'No RF visible pairs found'}

    # Parse changed functional groups when available
    all_changed_groups = []
    cliff_groups = []
    non_cliff_groups = []

    has_changed = 'changed_functional_groups' in rf_visible_df.columns
    if not has_changed:
        # Graceful fallback: return coverage summary only
        return {
            'total_rf_visible': len(rf_visible_df),
            'total_changed_groups': 0,
            'top_changed_groups': {},
            'cliff_specific': {},
            'non_cliff_specific': {}
        }

    for _, row in rf_visible_df.iterrows():
        try:
            changed_groups = row['changed_functional_groups']
            if isinstance(changed_groups, str):
                # Legacy rows may store list-like as string
                try:
                    changed_groups = eval(changed_groups)
                except Exception:
                    changed_groups = []
            if isinstance(changed_groups, list):
                all_changed_groups.extend(changed_groups)
                if row.get('pair_type', '') == 'cliff':
                    cliff_groups.extend(changed_groups)
                else:
                    non_cliff_groups.extend(changed_groups)
        except Exception:
            continue

    # Count frequencies
    group_counts = Counter(all_changed_groups)
    cliff_counts = Counter(cliff_groups)
    non_cliff_counts = Counter(non_cliff_groups)

    return {
        'total_rf_visible': len(rf_visible_df),
        'total_changed_groups': len(group_counts),
        'top_changed_groups': dict(group_counts.most_common(15)),
        'cliff_specific': dict(cliff_counts.most_common(10)),
        'non_cliff_specific': dict(non_cliff_counts.most_common(10))
    }

def verify_balanced_processing(results_df: pd.DataFrame) -> bool:
    """Verify that balanced structure is maintained after processing."""
    balance_table = results_df.groupby(['pair_type', 'antibiotic_class']).size().unstack(fill_value=0)
    expected_count = TEST_SAMPLES_PER_CLASS if RUN_MODE == "TEST" else FULL_SAMPLES_PER_CLASS
    perfect_balance = all(balance_table.values.flatten() == expected_count)

    print(f"\nBalance Verification:")
    print(f"Expected count per cell: {expected_count}")
    print(f"Perfect balance achieved: {perfect_balance}")
    print("Actual counts:")
    print(balance_table)

    return perfect_balance

def generate_plots(df: pd.DataFrame, stats: Dict[str, Any]) -> None:
    """Generate comprehensive plots for the analysis."""
    if not GENERATE_PLOTS:
        return

    print("Generating plots...")

    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 20))

    # 1. RF Visibility Coverage by Pair Type
    ax1 = plt.subplot(4, 2, 1)
    coverage_data = []
    labels = []
    for pair_type, data in stats['coverage_stats'].items():
        coverage_data.append(data['coverage_%'])
        labels.append(f"{pair_type}\n({data['rf_visible_n']}/{data['total_n']})")

    bars = ax1.bar(labels, coverage_data, color=['skyblue', 'lightcoral'])
    ax1.set_ylabel('RF Visibility Coverage (%)')
    ax1.set_title('RF Visibility Coverage by Pair Type')
    ax1.set_ylim(0, 100)

    # Add value labels on bars
    for bar, value in zip(bars, coverage_data):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom')

    # 2. Propagation Index Distribution
    ax2 = plt.subplot(4, 2, 2)
    rf_visible_df = df[df['rf_visible'] == True]

    if len(rf_visible_df) > 0:
        cliff_prop = rf_visible_df[rf_visible_df['pair_type'] == 'cliff']['propagation_index']
        non_cliff_prop = rf_visible_df[rf_visible_df['pair_type'] == 'non_cliff']['propagation_index']

        ax2.hist(cliff_prop, alpha=0.7, label='Cliffs', bins=15, color='skyblue')
        ax2.hist(non_cliff_prop, alpha=0.7, label='Non-cliffs', bins=15, color='lightcoral')
        ax2.set_xlabel('Propagation Index')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Propagation Index Distribution')
        ax2.legend()

        # Add mean lines
        ax2.axvline(cliff_prop.mean(), color='blue', linestyle='--', alpha=0.8)
        ax2.axvline(non_cliff_prop.mean(), color='red', linestyle='--', alpha=0.8)

    # 3. Edit Concentration Index vs Propagation Index
    ax3 = plt.subplot(4, 2, 3)

    if len(rf_visible_df) > 0:
        cliff_data = rf_visible_df[rf_visible_df['pair_type'] == 'cliff']
        non_cliff_data = rf_visible_df[rf_visible_df['pair_type'] == 'non_cliff']

        ax3.scatter(cliff_data['edit_concentration_index'], cliff_data['propagation_index'],
                   alpha=0.7, label='Cliffs', color='skyblue')
        ax3.scatter(non_cliff_data['edit_concentration_index'], non_cliff_data['propagation_index'],
                   alpha=0.7, label='Non-cliffs', color='lightcoral')
        ax3.set_xlabel('Edit Concentration Index')
        ax3.set_ylabel('Propagation Index')
        ax3.set_title('Edit vs Context Attribution')
        ax3.legend()

        # Add context sensitivity threshold line
        tau = stats.get('tau_noncliff_P95', 0.5)
        ax3.axhline(y=tau, color='gray', linestyle=':', alpha=0.7,
                   label=f'τ = {tau:.3f}')
        ax3.legend()

    # 4. Bootstrap Confidence Intervals
    ax4 = plt.subplot(4, 2, 4)

    metrics = []
    means = []
    ci_lowers = []
    ci_uppers = []
    colors = []

    for metric, data in stats['bootstrap_stats'].items():
        if metric == 'core_edit_ratio':  # Skip in SHAP mode
            continue
        metrics.append(metric.replace('_', ' ').title())
        means.append(data['bootstrap_mean'])
        ci_lowers.append(data['ci_lower'])
        ci_uppers.append(data['ci_upper'])

        # Color based on significance
        p_val = data['p_delta']
        colors.append('red' if p_val < 0.05 else 'gray')

    if metrics:
        y_pos = range(len(metrics))
        ax4.errorbar(means, y_pos, xerr=[np.array(means) - np.array(ci_lowers),
                                        np.array(ci_uppers) - np.array(means)],
                    fmt='o', capsize=5, color='black')

        for i, (mean, color) in enumerate(zip(means, colors)):
            ax4.scatter(mean, i, color=color, s=100, zorder=3)

        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(metrics)
        ax4.set_xlabel('Difference (Cliff - Non-cliff)')
        ax4.set_title('Bootstrap Confidence Intervals\n(Red: p < 0.05)')
        ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # 5. Similarity Distribution by Pair Type
    ax5 = plt.subplot(4, 2, 5)

    if 'similarity' in df.columns:
        cliff_sim = df[df['pair_type'] == 'cliff']['similarity'].dropna()
        non_cliff_sim = df[df['pair_type'] == 'non_cliff']['similarity'].dropna()

        ax5.hist(cliff_sim, alpha=0.7, label='Cliffs', bins=20, color='skyblue')
        ax5.hist(non_cliff_sim, alpha=0.7, label='Non-cliffs', bins=20, color='lightcoral')
        ax5.set_xlabel('Structural Similarity')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Structural Similarity Distribution')
        ax5.legend()

    # 6. Top Changed Functional Groups
    ax6 = plt.subplot(4, 2, 6)

    enrichment = functional_group_enrichment(df)
    if 'top_changed_groups' in enrichment:
        top_groups = enrichment['top_changed_groups']
        groups = list(top_groups.keys())[:10]  # Top 10
        counts = [top_groups[g] for g in groups]

        bars = ax6.barh(range(len(groups)), counts, color='lightgreen')
        ax6.set_yticks(range(len(groups)))
        ax6.set_yticklabels([g.replace('fr_', '') for g in groups])
        ax6.set_xlabel('Frequency')
        ax6.set_title('Top 10 Changed Functional Groups')

        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax6.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    str(count), va='center')

    # 7. Prediction Performance
    ax7 = plt.subplot(4, 2, 7)

    # Confusion matrix-style visualization
    pair_classifications = df['pair_classification'].value_counts()
    if len(pair_classifications) > 0:
        ax7.pie(pair_classifications.values, labels=pair_classifications.index,
               autopct='%1.1f%%', startangle=90)
        ax7.set_title('Pair Classification Performance')

    # 8. Summary Statistics Table
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')

    # Create summary text
    summary_text = []
    summary_text.append(f"Analysis Summary (SHAP Mode)")
    summary_text.append(f"=" * 25)
    summary_text.append("")

    for pair_type, data in stats['coverage_stats'].items():
        summary_text.append(f"{pair_type.upper()}:")
        summary_text.append(f"  RF Visible: {data['rf_visible_n']}/{data['total_n']} ({data['coverage_%']:.1f}%)")
        if pair_type in stats['means_stats']:
            means = stats['means_stats'][pair_type]
            summary_text.append(f"  Prop. Index: {means['propagation_index_mean']:.3f}")
        summary_text.append("")

    summary_text.append(f"Context Sensitivity (τ): {stats['tau_noncliff_P95']:.3f}")
    summary_text.append(f"Context Sensitive Rate: {stats['context_sensitive_rate']:.3f}")
    summary_text.append("")

    # Add significance results
    summary_text.append("Significance Tests:")
    for metric, data in stats['bootstrap_stats'].items():
        if metric == 'core_edit_ratio':  # Skip in SHAP mode
            continue
        p_val = data['p_delta']
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        summary_text.append(f"  {metric}: p = {p_val:.4f} {sig}")

    ax8.text(0.05, 0.95, '\n'.join(summary_text), transform=ax8.transAxes,
             verticalalignment='top', fontsize=10, fontfamily='monospace')

    plt.tight_layout()

    # Save plots
    plt.savefig('rf_xai_plots.png', dpi=300, bbox_inches='tight')
    print("Saved: rf_xai_plots.png")

    if GENERATE_PLOTS:  # Save PDF if requested
        plt.savefig('rf_xai_plots.pdf', bbox_inches='tight')
        print("Saved: rf_xai_plots.pdf")

    plt.close()

def enhanced_summary_report(df: pd.DataFrame, stats: Dict[str, Any]) -> None:
    """Generate enhanced summary report with SHAP-specific insights.

    Reload all split outputs from RF_model/outputs so the summary reflects
    both cliffs and non-cliffs across per-model and ensemble runs.
    """

    # Reload all split outputs
    try:
        import glob
        out_dir = os.path.join('RF_model', 'outputs')
        cliff_files = glob.glob(os.path.join(out_dir, 'rf_*_cliff*.csv'))
        non_files   = glob.glob(os.path.join(out_dir, 'rf_*_non_cliff*.csv'))
        files = sorted(set(cliff_files + non_files))
        if files:
            frames = []
            for p in files:
                try:
                    frames.append(pd.read_csv(p))
                except Exception:
                    continue
            if frames:
                df_all = pd.concat(frames, ignore_index=True)
                df_all = df_all[df_all.get('pair_type','').isin(['cliff','non_cliff'])].copy()
                df = df_all
                # Recompute stats over the combined dataframe
                try:
                    stats = visibility_aware_stats(df)
                except Exception:
                    pass
    except Exception:
        pass

    # Add similarity binning
    df = create_similarity_bins(df)

    # Get functional group enrichment
    enrichment = functional_group_enrichment(df)

    # Write enhanced summary
    with open("rf_xai_summary_enhanced.txt", "w", encoding='utf-8') as f:
        f.write("RF XAI Activity Pairs Analysis - Enhanced SHAP Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write("This analysis uses SHAP TreeExplainer on Random Forest models to test\n")
        f.write("the reliability of fr_ functional group features for explaining bioactivity\n")
        f.write("differences in molecular matched pairs.\n\n")

        f.write("Key Finding: ")
        pi_diff = stats['bootstrap_stats']['propagation_index']['bootstrap_mean']
        pi_p = stats['bootstrap_stats']['propagation_index']['p_delta']
        if pi_p < 0.05:
            direction = "lower" if pi_diff < 0 else "higher"
            f.write(f"Cliffs show significantly {direction} propagation than non-cliffs ")
            f.write(f"(Δ = {pi_diff:.3f}, p = {pi_p:.4f})\n\n")
        else:
            f.write(f"No significant difference in propagation between cliffs and non-cliffs ")
            f.write(f"(p = {pi_p:.4f})\n\n")

        # Coverage Analysis
        f.write("1. RF VISIBILITY COVERAGE\n")
        f.write("-" * 25 + "\n")
        f.write("Pair Type    | RF Visible | Total | Coverage\n")
        f.write("-------------|------------|-------|----------\n")
        for pair_type, data in (stats.get('coverage_stats', {}) or {}).items():
            f.write(f"{pair_type:<12} | {data['rf_visible_n']:>10} | {data['total_n']:>5} | {data['coverage_%']:>7.1f}%\n")
        f.write("\n")

        # Similarity-binned analysis
        if 'similarity_bin' in df.columns:
            f.write("2. SIMILARITY-STRATIFIED ANALYSIS\n")
            f.write("-" * 32 + "\n")
            rf_visible_df = df[df['rf_visible'] == True]

            if len(rf_visible_df) > 0:
                sim_analysis = rf_visible_df.groupby(['pair_type', 'similarity_bin']).agg({
                    'propagation_index': ['count', 'mean', 'std'],
                    'edit_concentration_index': 'mean'
                }).round(3)

                f.write("Propagation Index by Similarity Bin:\n")
                f.write(str(sim_analysis))
                f.write("\n\n")

        # Statistical Results
        f.write("3. STATISTICAL ANALYSIS (SHAP Mode)\n")
        f.write("-" * 35 + "\n")
        f.write("In SHAP mode, we use feature-space aggregation:\n")
        f.write("- Edit Delta: SHAP attributions for changed fr_ groups\n")
        f.write("- Context Delta: SHAP attributions for unchanged fr_ groups\n")
        f.write("- Propagation Index: Context/(Edit + Context)\n")
        f.write("- Core Edit Ratio: Deprecated in SHAP mode (always ≈1)\n\n")

        f.write("Visibility-aware Means (RF Visible Only):\n")
        for pair_type, data in stats['means_stats'].items():
            f.write(f"{pair_type.upper()}:\n")
            f.write(f"  Propagation Index: {data['propagation_index_mean']:.4f}\n")
            f.write(f"  Edit Concentration: {1 - data['propagation_index_mean']:.4f}\n")
        f.write("\n")

        f.write(f"Context Sensitivity Threshold (τ): {stats['tau_noncliff_P95']:.4f}\n")
        f.write(f"Context Sensitive Rate: {stats['context_sensitive_rate']:.4f}\n")
        f.write("(Proportion of RF-visible pairs with propagation > τ)\n\n")

        f.write("Bootstrap 95% CIs and Second-Generation P-values:\n")
        for metric, data in stats['bootstrap_stats'].items():
            if metric == 'core_edit_ratio':
                continue  # Skip in SHAP mode
            sig_stars = "***" if data['p_delta'] < 0.001 else "**" if data['p_delta'] < 0.01 else "*" if data['p_delta'] < 0.05 else ""
            f.write(f"{metric}:\n")
            f.write(f"  Difference (Cliff - Non-cliff): {data['bootstrap_mean']:.4f} {sig_stars}\n")
            f.write(f"  95% CI: [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}]\n")
            f.write(f"  p-delta: {data['p_delta']:.4f}\n\n")

        # Functional Group Analysis
        if 'top_changed_groups' in enrichment:
            f.write("4. FUNCTIONAL GROUP ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total RF-visible pairs: {enrichment['total_rf_visible']}\n")
            f.write(f"Unique changed groups: {enrichment['total_changed_groups']}\n\n")

            f.write("Top 10 Most Frequently Changed Groups:\n")
            for group, count in list(enrichment['top_changed_groups'].items())[:10]:
                f.write(f"  {group:<25}: {count:>3} pairs\n")
            f.write("\n")

        # Technical Details
        f.write("5. TECHNICAL DETAILS\n")
        f.write("-" * 20 + "\n")
        f.write("Model: Random Forest ensemble (5 CV folds)\n")
        f.write("Features: 85 RDKit fr_ functional group descriptors\n")
        f.write("XAI Method: SHAP TreeExplainer (deterministic)\n")
        f.write("Aggregation: Feature-space edit vs context\n")
        f.write("Bootstrap: 1000 resamples for CI and p-values\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("Saved: rf_xai_summary_enhanced.txt")

def main():
    """Main analysis function with balanced dataset support."""
    global RUN_MODE, GENERATE_PLOTS

    # Parse command line arguments (update argument descriptions)
    args = parse_arguments()
    RUN_MODE = "FULL" if args.full else "TEST"  # Changed from SMOKE to TEST
    GENERATE_PLOTS = args.pdf
    global ACTIVITY_CSV, NONCLIFF_CSV, CURRENT_MODEL_ID, CURRENT_IS_ENSEMBLE, _ensemble, _shap_explainer, _shap_background
    ACTIVITY_CSV = args.activity_csv
    NONCLIFF_CSV = args.noncliff_csv

    print("RF XAI Activity Pairs Analysis - Balanced Dataset")
    print("=" * 60)
    print(f"Run mode: {RUN_MODE}")
    print(f"Strategic classes: {', '.join(STRATEGIC_CLASSES)}")
    print(f"Classification threshold: {THRESHOLD}")
    print()

    def run_once_split(df_to_process: pd.DataFrame, pair_type: str, out_csv: str, do_reports: bool = True):
        """Process a single split (cliff or non-cliff) and save results."""
        # Optional limit per class for speed
        if args.limit_per_class is not None:
            take = int(args.limit_per_class)
            df_to_process = df_to_process.sort_values('class').groupby('class', group_keys=False).head(take)

        # Process pairs with robust error logging
        all_results = []
        print(f"\nProcessing {pair_type} pairs...")
        # Error log in outputs directory
        err_log_dir = os.path.join("outputs")
        os.makedirs(err_log_dir, exist_ok=True)
        err_log_path = os.path.join(err_log_dir, f"rf_{pair_type}_errors.log")
        for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc=pair_type.capitalize()):
            try:
                result = analyze_pair(row, pair_type)
                if result:
                    all_results.append(result)
            except Exception as e:
                # Append a concise diagnostic per failing row
                try:
                    with open(err_log_path, 'a', encoding='utf-8') as ef:
                        ef.write(f"index={idx}, error={type(e).__name__}: {str(e)}\n")
                except Exception:
                    pass

        results_df = pd.DataFrame(all_results)
        print(f"\nProcessed {len(results_df)} {pair_type} pairs successfully")
        print(f"RF visible pairs: {len(results_df[results_df['rf_visible'] == True])}")

        # Standardize schema and emit both CSV and Parquet
        def project_eval_schema(df: pd.DataFrame) -> pd.DataFrame:
            tmp = df.copy()
            is_cliff = False
            try:
                if 'pair_type' in tmp.columns and len(tmp) > 0:
                    is_cliff = str(tmp['pair_type'].iloc[0]).strip() == 'cliff'
            except Exception:
                pass

            if is_cliff:
                # Active/inactive semantics for cliffs
                for a_key in ['active_compound_id','compound1_id']:
                    if a_key in tmp.columns and 'compound_active_id' not in tmp.columns:
                        tmp.rename(columns={a_key: 'compound_active_id'}, inplace=True)
                        break
                for b_key in ['inactive_compound_id','compound2_id']:
                    if b_key in tmp.columns and 'compound_inactive_id' not in tmp.columns:
                        tmp.rename(columns={b_key: 'compound_inactive_id'}, inplace=True)
                        break
                for pa in ['active_pred_prob','compound1_pred_prob']:
                    if pa in tmp.columns and 'compound_active_pred_prob' not in tmp.columns:
                        tmp.rename(columns={pa: 'compound_active_pred_prob'}, inplace=True)
                        break
                for pi in ['inactive_pred_prob','compound2_pred_prob']:
                    if pi in tmp.columns and 'compound_inactive_pred_prob' not in tmp.columns:
                        tmp.rename(columns={pi: 'compound_inactive_pred_prob'}, inplace=True)
                        break
                # Predicted classes
                if 'compound_active_pred_prob' in tmp.columns:
                    tmp['compound_active_pred_class'] = (tmp['compound_active_pred_prob'] >= 0.5).map({True: 'active', False: 'inactive'})
                if 'compound_inactive_pred_prob' in tmp.columns:
                    tmp['compound_inactive_pred_class'] = (tmp['compound_inactive_pred_prob'] >= 0.5).map({True: 'active', False: 'inactive'})
                # Rename feature_scores to feature_attr (RF equivalent of substruct_attr)
                if 'feature_scores_active' in tmp.columns:
                    tmp.rename(columns={'feature_scores_active': 'feature_attr_active'}, inplace=True)
                if 'feature_scores_inactive' in tmp.columns:
                    tmp.rename(columns={'feature_scores_inactive': 'feature_attr_inactive'}, inplace=True)
                cols = [c for c in [
                    'pair_type','antibiotic_class',
                    'compound_active_id','compound_inactive_id',
                    'compound_active_pred_prob','compound_inactive_pred_prob',
                    'compound_active_pred_class','compound_inactive_pred_class',
                    'feature_attr_active','feature_attr_inactive',
                    'pos_features_active','neg_features_active','pos_features_inactive','neg_features_inactive',
                    'active_smiles','inactive_smiles',
                    'model_type'
                ] if c in tmp.columns]
                return tmp[cols]
            else:
                # Non-cliffs: compound1/compound2
                for a_key in ['active_compound_id','compound1_id']:
                    if a_key in tmp.columns:
                        tmp.rename(columns={a_key: 'compound1_id'}, inplace=True)
                        break
                for b_key in ['inactive_compound_id','compound2_id']:
                    if b_key in tmp.columns:
                        tmp.rename(columns={b_key: 'compound2_id'}, inplace=True)
                        break
                for pa in ['active_pred_prob','compound1_pred_prob']:
                    if pa in tmp.columns:
                        tmp.rename(columns={pa: 'compound1_pred_prob'}, inplace=True)
                        break
                for pi in ['inactive_pred_prob','compound2_pred_prob']:
                    if pi in tmp.columns:
                        tmp.rename(columns={pi: 'compound2_pred_prob'}, inplace=True)
                        break
                # Only create pred_class if pred_prob exists and has valid values
                if 'compound1_pred_prob' in tmp.columns:
                    try:
                        tmp['compound1_pred_class'] = (pd.to_numeric(tmp['compound1_pred_prob'], errors='coerce') >= 0.5).map({True: 'active', False: 'inactive'})
                    except Exception:
                        tmp['compound1_pred_class'] = 'unknown'
                if 'compound2_pred_prob' in tmp.columns:
                    try:
                        tmp['compound2_pred_class'] = (pd.to_numeric(tmp['compound2_pred_prob'], errors='coerce') >= 0.5).map({True: 'active', False: 'inactive'})
                    except Exception:
                        tmp['compound2_pred_class'] = 'unknown'
                # Rename feature_scores to feature_attr (RF equivalent of substruct_attr)
                if 'feature_scores_active' in tmp.columns:
                    tmp.rename(columns={'feature_scores_active': 'feature_attr_compound1'}, inplace=True)
                if 'feature_scores_inactive' in tmp.columns:
                    tmp.rename(columns={'feature_scores_inactive': 'feature_attr_compound2'}, inplace=True)
                cols = [c for c in [
                    'pair_type','antibiotic_class','compound1_id','compound2_id',
                    'compound1_pred_prob','compound2_pred_prob',
                    'compound1_pred_class','compound2_pred_class',
                    'feature_attr_compound1','feature_attr_compound2',
                    'pos_features_compound1','neg_features_compound1','pos_features_compound2','neg_features_compound2',
                    'compound1_smiles','compound2_smiles',
                    'model_type'
                ] if c in tmp.columns]
                return tmp[cols]

        sdf = project_eval_schema(results_df)
        # Save CSV with standardized schema
        sdf.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
        # Write Parquet sidecar
        try:
            sdf.to_parquet(out_csv.replace('.csv', '.parquet'), index=False)
            print(f"Saved: {out_csv.replace('.csv', '.parquet')}")
        except Exception as e:
            print(f"Parquet save skipped ({pair_type}): {e}")
        # Fail fast if we could not write any rows for this split
        if len(results_df) == 0:
            which = "cliffs" if pair_type == "cliff" else "non-cliffs"
            raise RuntimeError(f"No {which} rows were processed. See error log at RF_model/outputs/rf_{pair_type}_errors.log for first failing row and exception.")

        if do_reports:
            # Reports (only for ensemble)
            print("\nComputing visibility-aware statistics...")
            stats = visibility_aware_stats(results_df)
            if GENERATE_PLOTS:
                generate_plots(results_df, stats)
            enhanced_summary_report(results_df, stats)

        return results_df

    def run_once(out_csv: str, do_reports: bool = True):
        """DEPRECATED: Use run_once_split instead"""
        # Load data
        print("\nLoading data files...")
        df_cliff, df_non = load_data()
        # Optional limit per class for speed
        if args.limit_per_class is not None:
            take = int(args.limit_per_class)
            df_cliff = df_cliff.sort_values('class').groupby('class', group_keys=False).head(take)
            df_non = df_non.sort_values('class').groupby('class', group_keys=False).head(take)
        # Sample
        df_cliff_sample, df_non_sample = sample_balanced_data(df_cliff, df_non)
        print(f"Final processing: {len(df_cliff_sample)} cliffs, {len(df_non_sample)} non-cliffs")

        # Process
        all_results = []
        print("\nProcessing activity cliff pairs...")
        for idx, row in tqdm(df_cliff_sample.iterrows(), total=len(df_cliff_sample), desc="Cliffs"):
            result = analyze_pair(row, "cliff")
            if result:
                all_results.append(result)
        print("Processing non-cliff pairs...")
        for idx, row in tqdm(df_non_sample.iterrows(), total=len(df_non_sample), desc="Non-cliffs"):
            result = analyze_pair(row, "non_cliff")
            if result:
                all_results.append(result)
        results_df = pd.DataFrame(all_results)
        print(f"\nProcessed {len(results_df)} pairs successfully")
        print(f"RF visible pairs: {len(results_df[results_df['rf_visible'] == True])}")
        verify_balanced_processing(results_df)

        # Standardize schema and emit both CSV and Parquet
        def project_eval_schema(df: pd.DataFrame) -> pd.DataFrame:
            tmp = df.copy()
            is_cliff = False
            try:
                if 'pair_type' in tmp.columns and len(tmp) > 0:
                    is_cliff = str(tmp['pair_type'].iloc[0]).strip() == 'cliff'
            except Exception:
                pass
            if is_cliff:
                for a_key in ['active_compound_id','compound1_id']:
                    if a_key in tmp.columns and 'compound_active_id' not in tmp.columns:
                        tmp.rename(columns={a_key: 'compound_active_id'}, inplace=True)
                        break
                for b_key in ['inactive_compound_id','compound2_id']:
                    if b_key in tmp.columns and 'compound_inactive_id' not in tmp.columns:
                        tmp.rename(columns={b_key: 'compound_inactive_id'}, inplace=True)
                        break
                for pa in ['active_pred_prob','compound1_pred_prob']:
                    if pa in tmp.columns and 'compound_active_pred_prob' not in tmp.columns:
                        tmp.rename(columns={pa: 'compound_active_pred_prob'}, inplace=True)
                        break
                for pi in ['inactive_pred_prob','compound2_pred_prob']:
                    if pi in tmp.columns and 'compound_inactive_pred_prob' not in tmp.columns:
                        tmp.rename(columns={pi: 'compound_inactive_pred_prob'}, inplace=True)
                        break
                if 'compound_active_pred_prob' in tmp.columns:
                    tmp['compound_active_pred_class'] = (tmp['compound_active_pred_prob'] >= 0.5).map({True: 'active', False: 'inactive'})
                if 'compound_inactive_pred_prob' in tmp.columns:
                    tmp['compound_inactive_pred_class'] = (tmp['compound_inactive_pred_prob'] >= 0.5).map({True: 'active', False: 'inactive'})
                # Rename feature_scores to feature_attr (RF equivalent of substruct_attr)
                if 'feature_scores_active' in tmp.columns:
                    tmp.rename(columns={'feature_scores_active': 'feature_attr_active'}, inplace=True)
                if 'feature_scores_inactive' in tmp.columns:
                    tmp.rename(columns={'feature_scores_inactive': 'feature_attr_inactive'}, inplace=True)
                cols = [c for c in [
                    'pair_type','antibiotic_class',
                    'compound_active_id','compound_inactive_id',
                    'compound_active_pred_prob','compound_inactive_pred_prob',
                    'compound_active_pred_class','compound_inactive_pred_class',
                    'feature_attr_active','feature_attr_inactive',
                    'pos_features_active','neg_features_active','pos_features_inactive','neg_features_inactive',
                    'active_smiles','inactive_smiles',
                    'model_type'
                ] if c in tmp.columns]
                return tmp[cols]
            else:
                for a_key in ['active_compound_id','compound1_id']:
                    if a_key in tmp.columns:
                        tmp.rename(columns={a_key: 'compound1_id'}, inplace=True)
                        break
                for b_key in ['inactive_compound_id','compound2_id']:
                    if b_key in tmp.columns:
                        tmp.rename(columns={b_key: 'compound2_id'}, inplace=True)
                        break
                for pa in ['active_pred_prob','compound1_pred_prob']:
                    if pa in tmp.columns:
                        tmp.rename(columns={pa: 'compound1_pred_prob'}, inplace=True)
                        break
                for pi in ['inactive_pred_prob','compound2_pred_prob']:
                    if pi in tmp.columns:
                        tmp.rename(columns={pi: 'compound2_pred_prob'}, inplace=True)
                        break
                # Only create pred_class if pred_prob exists and has valid values
                if 'compound1_pred_prob' in tmp.columns:
                    try:
                        tmp['compound1_pred_class'] = (pd.to_numeric(tmp['compound1_pred_prob'], errors='coerce') >= 0.5).map({True: 'active', False: 'inactive'})
                    except Exception:
                        tmp['compound1_pred_class'] = 'unknown'
                if 'compound2_pred_prob' in tmp.columns:
                    try:
                        tmp['compound2_pred_class'] = (pd.to_numeric(tmp['compound2_pred_prob'], errors='coerce') >= 0.5).map({True: 'active', False: 'inactive'})
                    except Exception:
                        tmp['compound2_pred_class'] = 'unknown'
                # Rename feature_scores to feature_attr (RF equivalent of substruct_attr)
                if 'feature_scores_active' in tmp.columns:
                    tmp.rename(columns={'feature_scores_active': 'feature_attr_compound1'}, inplace=True)
                if 'feature_scores_inactive' in tmp.columns:
                    tmp.rename(columns={'feature_scores_inactive': 'feature_attr_compound2'}, inplace=True)
                # Also fix renamed feature lists for non-cliffs
                if 'active_smiles' in tmp.columns:
                    tmp.rename(columns={'active_smiles': 'compound1_smiles'}, inplace=True)
                if 'inactive_smiles' in tmp.columns:
                    tmp.rename(columns={'inactive_smiles': 'compound2_smiles'}, inplace=True)
                if 'pos_features_active' in tmp.columns:
                    tmp.rename(columns={'pos_features_active': 'pos_features_compound1'}, inplace=True)
                if 'neg_features_active' in tmp.columns:
                    tmp.rename(columns={'neg_features_active': 'neg_features_compound1'}, inplace=True)
                if 'pos_features_inactive' in tmp.columns:
                    tmp.rename(columns={'pos_features_inactive': 'pos_features_compound2'}, inplace=True)
                if 'neg_features_inactive' in tmp.columns:
                    tmp.rename(columns={'neg_features_inactive': 'neg_features_compound2'}, inplace=True)
                cols = [c for c in [
                    'pair_type','antibiotic_class','compound1_id','compound2_id',
                    'compound1_pred_prob','compound2_pred_prob',
                    'compound1_pred_class','compound2_pred_class',
                    'feature_attr_compound1','feature_attr_compound2',
                    'pos_features_compound1','neg_features_compound1','pos_features_compound2','neg_features_compound2',
                    'compound1_smiles','compound2_smiles',
                    'model_type'
                ] if c in tmp.columns]
                return tmp[cols]

        sdf = project_eval_schema(results_df)
        sdf.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
        # Write Parquet sidecar
        try:
            sdf.to_parquet(out_csv.replace('.csv', '.parquet'), index=False)
            print(f"Saved: {out_csv.replace('.csv', '.parquet')}")
        except Exception as e:
            print(f"Parquet save skipped: {e}")
        # Fail fast if we could not write any rows for this split
        if len(results_df) == 0:
            which = "cliffs" if pair_type == "cliff" else "non-cliffs"
            raise RuntimeError(f"No {which} rows were processed. See error log at RF_model/outputs/rf_{pair_type}_errors.log for first failing row and exception.")
        if not do_reports:
            return
        # Reports
        print("\nComputing visibility-aware statistics...")
        stats = visibility_aware_stats(results_df)
        if GENERATE_PLOTS:
            generate_plots(results_df, stats)
        enhanced_summary_report(results_df, stats)
        with open("rf_xai_summary.txt", "w", encoding='utf-8') as f:
            f.write("RF XAI Activity Pairs Analysis Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write("Coverage Table (RF Visibility):\n")
            f.write("pair_type, rf_visible_n, total_n, coverage_%\n")
            for pair_type, data in stats['coverage_stats'].items():
                f.write(f"{pair_type}, {data['rf_visible_n']}, {data['total_n']}, {data['coverage_%']:.1f}\n")
            f.write("\n")

            f.write("Visibility-aware Means (RF Visible Only):\n")
            for pair_type, data in stats['means_stats'].items():
                f.write(f"{pair_type}:\n")
                f.write(f"  Propagation Index: {data['propagation_index_mean']:.4f}\n")
                # Hide core_edit_ratio in SHAP mode
                f.write(f"  Edit Concentration: {1 - data['propagation_index_mean']:.4f}\n")
            f.write("\n")

            f.write(f"Context Sensitivity Threshold (tau): {stats['tau_noncliff_P95']:.4f}\n")
            f.write(f"Context Sensitive Rate: {stats['context_sensitive_rate']:.4f}\n\n")

            f.write("Bootstrap 95% CIs and Second-Generation P-values:\n")
            for metric, data in stats['bootstrap_stats'].items():
                if metric == 'core_edit_ratio':  # Hide in SHAP mode
                    continue
                f.write(f"{metric}:\n")
                f.write(f"  Bootstrap Mean Difference: {data['bootstrap_mean']:.4f}\n")
                f.write(f"  95% CI: [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}]\n")
                f.write(f"  Second-gen p-value (p_delta): {data['p_delta']:.4f}\n\n")

            f.write(f"Analysis completed with SHAP TreeExplainer for fr_ functional group reliability testing\n")
            f.write(f"Random seed: {RANDOM_SEED}\n")

        print("Saved: rf_xai_summary.txt")
        with open("rf_xai_summary.json", "w") as f:
            json.dump(stats, f, indent=2)
        print("Saved: rf_xai_summary.json")
        print("\n" + "=" * 50)
        print("KEY RESULTS:")
        print("=" * 50)
        for pair_type, data in stats['coverage_stats'].items():
            print(f"{pair_type}: {data['coverage_%']:.1f}% RF visible ({data['rf_visible_n']}/{data['total_n']})")
        print(f"\nContext sensitivity threshold: {stats['tau_noncliff_P95']:.4f}")
        print(f"Context sensitive rate: {stats['context_sensitive_rate']:.4f}")
        print(f"\nBootstrap results (cliff - non_cliff means):")
        for metric, data in stats['bootstrap_stats'].items():
            if metric == 'core_edit_ratio':  # Hide in SHAP mode
                continue
            print(f"{metric}: {data['bootstrap_mean']:.4f} [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}], p_delta={data['p_delta']:.4f}")
        print("\nAnalysis complete!")

    # Determine run modes
    # When --full is specified, run BOTH per-model and ensemble to get all 12 output files
    # (5 per-model × 2 pair types + 1 ensemble × 2 pair types = 12 files total)
    if args.full:
        do_per_model = True  # Always run per-model in full mode
        do_ensemble = True   # Always run ensemble in full mode
        print("FULL MODE: Will generate all 12 output files (5 per-model + 1 ensemble, each with cliffs and non_cliffs)")
    else:
        # Manual mode selection
        do_per_model = args.per_model
        do_ensemble = args.ensemble or (not args.per_model)  # default to ensemble if nothing specified

    # Load data once (outside loops)
    print("\nLoading data files...")
    df_cliff, df_non = load_data()

    # Initialize full ensemble
    print("Initializing ensemble models...")
    _ensemble = None  # reset
    ensemble_all = get_ensemble()
    print("Enhanced RF Ensemble ready!")

    # Create output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    if do_per_model:
        # Loop over cv/fold (5 models)
        for model_idx, mi in enumerate(ensemble_all.best_models_info, start=1):
            cv = int(mi['cv']); fold = int(mi['fold'])
            # Build restricted instance
            _ensemble = EnsembleRandomForestExplainer()
            _ensemble.restrict_to_cv_fold(cv, fold)
            _shap_explainer = None; _shap_background = None
            CURRENT_MODEL_ID = f"cv{cv}_fold{fold}"
            CURRENT_IS_ENSEMBLE = False

            print(f"\n{'='*60}")
            print(f"Running per-model analysis: Model {model_idx} ({CURRENT_MODEL_ID})")
            print(f"{'='*60}")

            # Process cliffs
            cliff_output = os.path.join(output_dir, f"rf_model{model_idx}_cliffs.csv")
            run_once_split(df_cliff.copy(), "cliff", cliff_output, do_reports=False)

            # Process non-cliffs
            noncliff_output = os.path.join(output_dir, f"rf_model{model_idx}_non_cliffs.csv")
            run_once_split(df_non.copy(), "non_cliff", noncliff_output, do_reports=False)

    if do_ensemble:
        _ensemble = None  # reset
        _shap_explainer = None; _shap_background = None
        CURRENT_MODEL_ID = "ensemble"
        CURRENT_IS_ENSEMBLE = True

        print(f"\n{'='*60}")
        print("Running ensemble analysis...")
        print(f"{'='*60}")

        # Process cliffs
        cliff_output = os.path.join(output_dir, "rf_ensemble_cliffs.csv")
        run_once_split(df_cliff.copy(), "cliff", cliff_output, do_reports=True)

        # Process non-cliffs
        noncliff_output = os.path.join(output_dir, "rf_ensemble_non_cliffs.csv")
        run_once_split(df_non.copy(), "non_cliff", noncliff_output, do_reports=True)

if __name__ == "__main__":
    main()



