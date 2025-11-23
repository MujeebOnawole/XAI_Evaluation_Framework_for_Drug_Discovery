# XAI Evaluation Framework for Drug Discovery

## Overview
A framework for evaluating explainable AI (XAI) methods in drug discovery using multiple machine learning architectures. This repository implements three distinct model architectures (CNN, Random Forest, and RGCN) and provides a hierarchical four-tier evaluation framework for assessing the quality and reliability of their explanations.  

## Dataset
ChEMBL v34 source with 43,777 unique compounds against Staphylococcus aureus, MIC classification (active ≤ 64 µg/mL vs inactive > 64 µg/mL), 600 molecular pairs (300 activity cliffs and 300 non-cliffs), balanced across three antibiotic classes (100 pairs each): beta-lactams, fluoroquinolones, and oxazolidinones.



## Model Architectures
1. Random Forest (simplest)
2. CNN (moderate)
3. RGCN (most complex)

- **Three ML Model Architectures**: Random Forest (fragment-based- functional groups), CNN (SMILES-based), and RGCN (graph-based)
- **Hierarchical XAI Evaluation**: Four-tier framework (Scaffold Recognition, Model Independence, Context Sensitivity, Internal Consistency)
- **Activity Cliff Analysis**: Methods for analyzing structure-activity relationships and molecular activity cliffs
- **Pharmacophore Validation**: Tools for validating explanations against known pharmacophores

## Evaluation Tiers
1. Tier 1: Basic Metrics
2. Tier 2: Statistical Significance Tests
3. Tier 3: Interpretability Analysis
4. Tier 4: End-user Evaluation

## Installation Instructions
To install the framework, use the following commands:

```bash
pip install -r requirements.txt
```

### 3. RGCN Model (Relational Graph Convolutional Network)
- **Input**: Molecular graphs with typed edges
- **XAI Method**: Occlusion-based attribution (Substructure Masking)
- **Key Features**:
  - Graph-based molecular representation
  - Intentional 3-edge design for interpretability
  - Node feature-based aromaticity learning
  - PyTorch Geometric integration

```python
import xai_framework
# Example usage code here
```

## 🔍 XAI Evaluation Framework

The framework evaluates XAI methods across **four hierarchical tiers**, each addressing critical aspects of explanation quality:

### Tier 1: Scaffold Recognition (SR) - Essential
**Purpose**: Can the XAI method identify core drug scaffolds?

**Metrics**:
- RGCN/CNN: Top-K recognition (K=3) + Attribution magnitude >0.1
- RF: Top-K recognition (K=7, no magnitude threshold)

**Passing Criteria**: ≥90% complete recognition





### Tier 2: Context Sensitivity (CS) - Validation
**Purpose**: Does the method recognize that identical scaffolds contribute differently in different contexts?

**Metrics**:
- Directionality (35%): Paired t-test
- Context Awareness (35%): Levene's test
- Discrimination (30%): Binomial test



### Tier 3: Internal Consistency (IC) - Confidence
**Purpose**: Do explanations align with predictions?

**Metric**: Sign matching between net attribution and prediction direction
- Active prediction (≥0.5): mean(attributions) > 0
- Inactive prediction (<0.5): mean(attributions) < 0




### Tier 4: Model Independence (MI) - Deployment
**Purpose**: Are explanations consistent across different model instances?

**Metrics**:
- Jaccard similarity (feature overlap)
- Spearman correlation (ranking consistency)

**Passing Criteria**: ≥0.95 for deployment, ≥0.70 acceptable for research


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
- **Occlusion-based Token Masking** (CNN): Gradient-based attribution
- **TreeSHAP** (RF): Shapley values for tree ensembles
- **Occlusion-based Substructure Masking** (RGCN): Perturbation-based attribution

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

**CNN (Occlusion based Token masking)**:
```bash
cd CNN_model
python cnn_xai_activity_pairs.py \
    --full \
    --out_csv outputs/cnn_xai_results.csv \

```

**Random Forest (TreeSHAP)**:
```bash
cd RF_model
python RF_XAI_activity_pairs.py
```

**RGCN (Occlusion-based Substructure masking)**:
```bash
cd RGCN_model
python rgcn_xai_activity_pairs.py
```

### Visualizing Results

Each model includes a Jupyter notebook for visualization:
- `CNN_model/CNN_Visualizer.ipynb`
- `RF_model/RF_Visualizer.ipynb`
- `RGCN_model/RGCN_Visualizer.ipynb`
- To download model checkpoints, visit https://zenodo.org/records/17678160



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

This work implements t XAI methods for drug discovery, building on research in:
- Explainable AI (XAI)
- Molecular machine learning
- Structure-activity relationship (SAR) analysis
- Pharmacophore modeling

## Citations
If you use this framework for your research, please cite:
> Author, Year. Title. Journal Name.

## Acknowledgments
We acknowledge the contributions of ...
