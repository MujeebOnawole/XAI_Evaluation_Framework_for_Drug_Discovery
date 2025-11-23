# XAI Evaluation Framework for Drug Discovery

## Overview
This framework aims to support the application of explainable artificial intelligence (XAI) methods in drug discovery.

## Dataset
ChEMBL v34 source with 43,777 unique compounds against Staphylococcus aureus, MIC classification (active ≤ 64 µg/mL vs inactive > 64 µg/mL), 600 molecular pairs (300 activity cliffs and 300 non-cliffs), balanced across three antibiotic classes (100 pairs each): beta-lactams, fluoroquinolones, and oxazolidinones.

## Model Architectures
1. Random Forest (simplest)
2. CNN (moderate)
3. RGCN (most complex)

## Drug Classes
1. Beta-lactams
2. Fluoroquinolones
3. Oxazolidinones

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

## Usage Examples
Below are examples of how to use the framework:

```python
import xai_framework
# Example usage code here
```

## Repository Structure
- `xai_framework/` - Contains the core framework modules.
- `tests/` - Includes unit tests for validation.

## Citations
If you use this framework for your research, please cite:
> Author, Year. Title. Journal Name.

## Acknowledgments
We acknowledge the contributions of ...