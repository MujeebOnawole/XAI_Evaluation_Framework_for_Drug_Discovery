# README

## Overview

This project aims to evaluate the effectiveness of various machine learning models in predicting antibiotic resistance.

### Dataset
- The dataset consists of compounds from ChEMBL v34 evaluated for activity against Staphylococcus aureus.
- Compounds classified by minimum inhibitory concentration (MIC): active (MIC ≤ 64 µg/mL) or inactive (MIC > 64 µg/mL).
- Total of 43,777 unique chemical compounds.
- Evaluations performed on 600 molecular pairs: 300 activity cliff pairs and 300 non-cliff pairs.
- Balanced across three antibiotic classes (100 pairs each): beta-lactams, fluoroquinolones, and oxazolidinones.

## Model Architectures
- Random Forest (RF)
- CNN (Convolutional Neural Network)
- RGCN (Relational Graph Convolutional Network)

## Drug Classes Evaluated
- Beta-lactams
- Fluoroquinolones
- Oxazolidinones

## Conclusion

These findings indicate strong potentials in model evaluation for drug discovery.