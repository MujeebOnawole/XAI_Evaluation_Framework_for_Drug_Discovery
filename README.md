# XAI Evaluation Framework for Drug Discovery

A comprehensive framework for training and evaluating explainable AI (XAI) methods in drug discovery using multiple machine learning architectures. This repository implements three distinct model architectures (CNN, Random Forest, and RGCN) and provides a hierarchical four-tier evaluation framework for assessing the quality and reliability of their explanations.

## 🎯 Overview

This project addresses a critical challenge in AI-driven drug discovery: **how do we evaluate whether model explanations are trustworthy?** The framework implements:

- **Three ML Model Architectures**: CNN (SMILES-based), Random Forest (descriptor-based), and RGCN (graph-based)
- **Hierarchical XAI Evaluation**: Four-tier framework (Scaffold Recognition, Model Independence, Context Sensitivity, Internal Consistency)
- **Activity Cliff Analysis**: Methods for analyzing structure-activity relationships and molecular activity cliffs
- **Pharmacophore Validation**: Tools for validating explanations against known pharmacophores

## Dataset
ChEMBL v34 source with 43,777 unique compounds against Staphylococcus aureus, MIC classification (active ≤ 64 µg/mL vs inactive > 64 µg/mL), 600 molecular pairs (300 activity cliffs and 300 non-cliffs), balanced across three antibiotic classes (100 pairs each): beta-lactams, fluoroquinolones, and oxazolidinones.

## 📁 Repository Structure

```
XAI_Evaluation_Framework_for_Drug_Discovery/

├── RF_model/                           # Random Forest model
│   ├── RF_CV.py                       # Cross-validation training
│   ├── descriptor.py                  # Molecular descriptor calculation
│   ├── RF_XAI_activity_pairs.py       # XAI analysis (TreeSHAP)
│   ├── RF_test_evaluation.py          # Test set evaluation
│   ├── RF_Visualizer.ipynb            # Visualization notebook
│   └── best_models.json               # Saved model configurations
│
├── CNN_model/                         # CNN-based SMILES model
│   ├── main.py                        # Main training pipeline
│   ├── model.py                       # CNN architecture
│   ├── data_preprocessing.py          # SMILES preprocessing
│   ├── cross_validation.py            # Cross-validation framework
│   ├── hyperparameter_opt.py          # Hyperparameter optimization
│   ├── cnn_xai_activity_pairs.py      # XAI analysis : Occlusion based (Token-mapping)
│   ├── CNN_Visualizer.ipynb           # Visualization notebook
│   └── README.md                      # CNN-specific documentation
│
│
├── RGCN_model/                         # Relational Graph Convolutional Network
│   ├── RGCN_CV.py                     # Cross-validation training
│   ├── model.py                       # RGCN architecture
│   ├── build_data.py                  # Graph construction
│   ├── data_module.py                 # PyTorch Lightning data module
│   ├── config.py                      # Model configuration
│   ├── rgcn_xai_activity_pairs.py     # XAI analysis (occlusion-based)
│   ├── RGCN_Visualizer.ipynb          # Visualization notebook
│   └── README.md                      # RGCN-specific documentation
│
└── XAI_evaluation_Framework_scripts/   # XAI evaluation framework
    ├── evaluate_SR.py                 # Tier 1: Scaffold Recognition
    ├── evaluate_CS.py                 # Tier 2: Context Sensitivity
    ├── evaluate_IC.py                 # Tier 3: Internal Consistency
    ├── evaluate_MI.py                 # Tier 4: Model Independence        
    └── README.md                      # Framework documentation
```

## 🧠 Model Architectures

### 1. CNN Model (Convolutional Neural Network)
- **Input**: SMILES strings (text-based molecular representation)
- **Architecture**: 1D CNN with token-level processing
- **XAI Method**: Occlusion based (Token-mapping) 
- **Key Features**:
  - SMILES tokenization with configurable mapping modes
  - GPU-accelerated training with PyTorch Lightning
  - Ensemble predictions across multiple folds
  - Token-to-atom attribution mapping

**Quick Start**:
```bash
cd CNN_model
# Fast test (2 pairs per class)
python cnn_xai_activity_pairs.py --out_csv outputs/cnn_xai_balanced_full_detailed.csv

# Full dataset
python cnn_xai_activity_pairs.py --full --out_csv outputs/cnn_xai_balanced_full_detailed.csv
```

### 2. Random Forest Model
- **Input**: Molecular descriptors (85 fragment-based features)
- **Architecture**: Ensemble of decision trees
- **XAI Method**: TreeSHAP (Shapley Additive Explanations)
- **Key Features**:
  - Fragment-based feature engineering
  - Fast training and inference
  - Interpretable feature importances
  - Present-feature filtering for accurate attribution

**Quick Start**:
```bash
cd RF_model
python RF_CV.py  # Cross-validation training
python RF_XAI_activity_pairs.py  # Generate explanations
```

### 3. RGCN Model (Relational Graph Convolutional Network)
- **Input**: Molecular graphs with typed edges
- **Architecture**: 3-edge RGCN (SINGLE, DOUBLE, TRIPLE bonds)
- **XAI Method**: Occlusion-based Substructure Masking 
- **Key Features**:
  - Graph-based molecular representation
  - Intentional 3-edge design for interpretability
  - Node feature-based aromaticity learning
  - PyTorch Geometric integration

**Design Philosophy**: Uses only 3 edge types (SINGLE=0, DOUBLE=1, TRIPLE=2) while aromatic bonds are preserved in graph topology but assigned undefined type (-1), forcing the model to learn aromaticity from node features for improved explainability.

**Quick Start**:
```bash
cd RGCN_model
python RGCN_CV.py  # Cross-validation training
python rgcn_xai_activity_pairs.py  # Generate explanations
```

## 🔍 XAI Evaluation Framework

The framework evaluates XAI methods across **four hierarchical tiers**, each addressing critical aspects of explanation quality:

### Tier 1: Scaffold Recognition (SR) - Essential
**Purpose**: Can the XAI method identify core drug scaffolds?

**Metrics**:
- RGCN/CNN: Top-K recognition (K=3) + Attribution magnitude >0.1
- RF: Top-K recognition (K=7, no magnitude threshold)

**Passing Criteria**: ≥90% complete recognition



### Tier 2: Model Independence (MI) - Deployment
**Purpose**: Are explanations consistent across different model instances?

**Metrics**:
- Jaccard similarity (feature overlap)
- Spearman correlation (ranking consistency)

**Passing Criteria**: ≥0.95 for deployment, ≥0.70 acceptable for research



### Tier 3: Context Sensitivity (CS) - Validation
**Purpose**: Does the method recognize that identical scaffolds contribute differently in different contexts?

**Metrics**:
- Directionality (35%): Paired t-test
- Context Awareness (35%): Levene's test
- Discrimination (30%): Binomial test



### Tier 4: Internal Consistency (IC) - Confidence
**Purpose**: Do explanations align with predictions?

**Metric**: Sign matching between net attribution and prediction direction
- Active prediction (≥0.5): mean(attributions) > 0
- Inactive prediction (<0.5): mean(attributions) < 0



### Running the Complete Framework

```bash
cd XAI_evaluation_Framework_scripts

# Run all evaluations in order
python evaluate_SR.py  # Tier 1: Scaffold Recognition
python evaluate_CS.py  # Tier 2: Context Sensitivity
python evaluate_IC.py  # Tier 3: Internal Consistency
python evaluate_MI.py  # Tier 4: Model Independence
```

## 📊 Key Features

### Activity Cliff Analysis
- Balanced activity cliff pairs analysis
- Non-cliff pairs for comparison
- Statistical validation of context sensitivity

### Pharmacophore Validation
- Strict SMARTS pattern matching
- Functional group mapping
- Coverage analysis across drug classes

### XAI Methods Integration
- **Occlusion based (Token-mapping)** (CNN): Gradient-based attribution
- **TreeSHAP** (RF): Shapley values for tree ensembles
- **Occlusion-based Substructure masking** (RGCN): Perturbation-based attribution

## 🔧 Installation

### Prerequisites
- Python 3.9-3.11 (3.10 recommended)
- Anaconda/Miniconda (recommended for RDKit)
- CUDA-capable GPU (optional, for faster training)

### Core Dependencies
```bash
# Create conda environment
conda create -n xai_drug_discovery python=3.10 -y
conda activate xai_drug_discovery

# Install RDKit (conda-forge recommended for Windows)
conda install -c conda-forge rdkit -y

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.3.* torchvision torchaudio

# Install deep learning frameworks
pip install pytorch-lightning==2.* torch-geometric

# Install scientific computing
pip install pandas numpy scipy scikit-learn matplotlib seaborn

# Install XAI libraries
pip install shap captum

# Install utilities
pip install joblib optuna
```

### Model-Specific Setup

**CNN Model**:
```bash
cd CNN_model
# Dependencies already covered above
```

**Random Forest**:
```bash
cd RF_model
# Ensure scikit-learn and RDKit are installed
```

**RGCN Model**:
```bash
cd RGCN_model
# Ensure PyTorch Geometric is installed
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

## 🚀 Usage Examples

### Training a Model

**CNN**:
```bash
cd CNN_model
python main.py --data_path path/to/data.csv --run_cv True
```

**Random Forest**:
```bash
cd RF_model
python RF_CV.py  # Uses SA_FG_fragments.csv
```

**RGCN**:
```bash
cd RGCN_model
python RGCN_CV.py
```

### Generating Explanations

**CNN (Occlusion based (Token-mapping))**:
```bash
cd CNN_model
python cnn_xai_activity_pairs.py \
    --full \
    --out_csv outputs/cnn_xai_results.csv \
    --ig_steps 64
```

**Random Forest (TreeSHAP)**:
```bash
cd RF_model
python RF_XAI_activity_pairs.py
```

**RGCN (Occlusion-based)**:
```bash
cd RGCN_model
python rgcn_xai_activity_pairs.py
```

### Visualizing Results

Each model includes a Jupyter notebook for visualization:
- `CNN_model/CNN_Visualizer.ipynb`
- `RF_model/RF_Visualizer.ipynb`
- `RGCN_model/RGCN_Visualizer.ipynb`
- Download model checkpoints from  https://zenodo.org/records/17678160

## 📈 Performance Comparison



## 🧪 Drug Classes Evaluated

The framework has been validated on three major antibiotic classes:

1. **Fluoroquinolones**: DNA gyrase inhibitors
   - Key scaffold: Bicyclic quinolone core + carboxylic acid
   
2. **Beta-lactams**: Cell wall synthesis inhibitors
   - Key scaffold: Beta-lactam ring
   
3. **Oxazolidinones**: Protein synthesis inhibitors
   - Key scaffold: Oxazolidinone ring

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{xai_evaluation_framework_drug_discovery,
  author = {Onawole, Mujeeb},
  title = {XAI Evaluation Framework for Drug Discovery},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/MujeebOnawole/XAI_Evaluation_Framework_for_Drug_Discovery}
}
```

## 📚 Additional Resources

### Model-Specific Documentation
- [CNN Model README](CNN_model/README.md) - Detailed CNN documentation
- [RGCN Model README](RGCN_model/README.md) - RGCN architecture details
- [XAI Framework README](XAI_evaluation_Framework_scripts/README.md) - Complete evaluation methodology

### Key Concepts

**Activity Cliffs**: Pairs of structurally similar molecules with large differences in biological activity. Critical for understanding structure-activity relationships.

**Scaffold Recognition**: The ability of an XAI method to identify the core structural motif responsible for a drug's activity.

**Model Independence**: The consistency of explanations across different trained instances of the same model architecture.

**Context Sensitivity**: The ability to recognize that the same structural feature can have different importance in different molecular contexts.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is available for academic and research purposes. Please contact the author for commercial use.

## 👤 Author

**Mujeeb Onawole**
- GitHub: [@MujeebOnawole](https://github.com/MujeebOnawole)

## 🙏 Acknowledgments

This work implements state-of-the-art XAI methods for drug discovery, building on research in:
- Explainable AI (XAI)
- Molecular machine learning
- Structure-activity relationship (SAR) analysis
- Pharmacophore modeling

---

**Last Updated**: November 2025  
**Status**: Active Development  
**Python Version**: 3.9-3.11 (3.10 recommended)
