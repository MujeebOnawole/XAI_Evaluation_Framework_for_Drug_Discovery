CNN XAI Activity Pairs — Runbook (Windows/Anaconda)

This folder contains an explainability pipeline for the CNN model on molecular activity‑cliff pairs, plus analysis and pharmacophore validation tools. The pipeline mirrors the RF/RGCN analyses and writes compatible outputs so results are comparable across models.

1) Prerequisites

- OS: Windows 10/11
- Shell: Anaconda Prompt
- Python: 3.9–3.11 (recommended 3.10)
- GPU optional. Scripts run on CPU; GPU speeds up inference.

Create a conda environment

```
conda create -n cnn_xai python=3.10 -y
conda activate cnn_xai
```

Install dependencies

- Core: torch, pytorch-lightning, rdkit, pandas, numpy, matplotlib, scikit-learn

Suggested (CPU-only) install:

```
pip install torch==2.3.*
pip install pytorch-lightning==2.* pandas numpy matplotlib scikit-learn
conda install -c conda-forge rdkit -y
```

If you have CUDA, use matching torch wheels for your CUDA version.

2) Files and Folders

- Input CSVs
  - balanced_activity_cliff_pairs.csv
  - balanced_non_cliff_pairs.csv
- Model checkpoints
  - model_checkpoints\*.ckpt (CNN checkpoints)
  - The first checkpoint is used as the “backbone” for XAI ops; predictions can still use an ensemble average
- Scripts (CNN)
  - cnn_xai_activity_pairs.py — Main CNN XAI runner (Integrated Gradients)
  - analyze_cnn_xai_csv.py — Analysis & audit
  - cnn_pharmacophore_validation.py — Pharmacophore alignment
- Shared config
  - pharmacophores.json (shared with RGCN; includes strict + loose patterns)

3) Typical Run Order (CNN)

Change directory to this folder in Anaconda Prompt first:

```
cd "C:\Users\<you>\OneDrive - The University of Queensland\Desktop\automatic_XAI\CNN_model"
conda activate cnn_xai
```

3.1 Run CNN XAI (Integrated Gradients)

- Fast test (2 pairs per class):
```
python cnn_xai_activity_pairs.py --out_csv outputs\cnn_xai_balanced_full_detailed.csv
```
- Full dataset:
```
python cnn_xai_activity_pairs.py --full --out_csv outputs\cnn_xai_balanced_full_detailed.csv
```
- Useful flags:
  - `--ig_steps 64` (more steps = smoother IG; slower)
  - `--ig_baseline pad|zeros|meanpad` (default `pad`)
  - `--ig_noise 0.0` (add small noise for stability if needed)
  - `--ig_paths 1` (multi-path IG averaging)
  - `--map_mode span|interpolate` (default `span`)
  - `--fg_norm sum|mean|max` (default `sum`)
  - `--samples_per_class N` (when not using `--full`)
  - `--calibration_json path.json` (optional temperature scaling)
  - `--ckpt path.ckpt` (repeatable). If omitted, auto-discovers under `model_checkpoints\`; the first checkpoint is used as XAI backbone, ensemble average for predictions.

Output:
- `outputs\cnn_xai_balanced_full_detailed.csv`

3.2 Analyze CNN XAI CSV

```
python analyze_cnn_xai_csv.py --csv outputs\cnn_xai_balanced_full_detailed.csv --out_prefix cnn_xai_audit
```

Outputs:
- `cnn_xai_audit_class_diff.csv` — Cliff vs non-cliff deltas per class
- `cnn_xai_audit_correctness_stats.csv` — BothCorrect / OneCorrect / BothWrong
- `cnn_xai_audit_top_drivers.csv` — Top functional-group drivers
- `cnn_xai_audit_cnn_locality_fidelity_stats.csv` — CNN-native locality/fidelity/stability metrics
- Visuals: `cnn_xai_audit_*png`
- Text: `cnn_xai_audit_per_class_report.txt`

3.3 Pharmacophore Validation (CNN)

```
python cnn_pharmacophore_validation.py --xai_csv outputs\cnn_xai_balanced_full_detailed.csv --pharm_json pharmacophores.json --out_prefix cnn_xai
```

Outputs:
- `cnn_xai_pharm_pairs.csv` — Per-pair alignment/QC
  - Includes `pair_classification` (BothCorrect/OneCorrect/BothWrong) and mapping stats
- `cnn_xai_pharm_class_summary.csv` — Per-class summary (deltas, CIs, effect sizes)
- Optional: `cnn_xai_atom_attributions.csv` — Atom-level positive mass export
- If mirrored from RGCN validator, also coverage reports:
  - `cnn_xai_pharm_coverage_report.csv`
  - `cnn_xai_pharm_coverage_by_slice.csv`
  - `cnn_xai_pharm_coverage_report.txt`

4) Practical Notes

- Backbone checkpoint: For performance, IG and occlusion sanity use the first checkpoint (`backbone_ckpt` in the CSV); predictions use the ensemble average. You can pass a single `--ckpt` to make backbone = predictor.
- Performance: IG is compute‑heavy. Use defaults first; bump `--ig_steps` only if needed.
- Mapping mode: `span` is length‑preserving from tokens to atoms and robust across SMILES variations. `interpolate` is available if you want smoother distributions.
- RDKit install: Use conda‑forge on Windows (`conda install -c conda-forge rdkit`).
- Paths with spaces: Always wrap paths in quotes.

5) Quick Commands Summary

- Fast test:
```
python cnn_xai_activity_pairs.py --out_csv outputs\cnn_xai_balanced_full_detailed.csv
python analyze_cnn_xai_csv.py --csv outputs\cnn_xai_balanced_full_detailed.csv --out_prefix cnn_xai_audit
python cnn_pharmacophore_validation.py --xai_csv outputs\cnn_xai_balanced_full_detailed.csv --pharm_json pharmacophores.json --out_prefix cnn_xai
```

- Full run:
```
python cnn_xai_activity_pairs.py --full --out_csv outputs\cnn_xai_balanced_full_detailed.csv
```

- Choose backbone explicitly:
```
python cnn_xai_activity_pairs.py --ckpt model_checkpoints\CV1_Fold5_best-epoch=67-val_auroc=0.8465.ckpt --out_csv outputs\cnn_xai_balanced_full_detailed.csv
```

If you want a `requirements.txt` or Conda `environment.yml`, they can be generated from current usage to simplify setup.

