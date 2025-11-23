# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler  # ADD THIS IMPORT
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef
import os
import time
import json
import joblib

# Load the dataset
df = pd.read_csv('SA_FG_fragments.csv')

# Extract features (all columns starting with 'fr_')
feature_cols = [col for col in df.columns if col.startswith('fr_')]
X = df[feature_cols]  # Keep as DataFrame
y = df['TARGET']  # Keep as Series

# Get the training, test, and validation data
X_train = X[df['group'] == 'training'].values
y_train = y[df['group'] == 'training'].values

X_test = X[df['group'] == 'test'].values
y_test = y[df['group'] == 'test'].values

X_valid = X[df['group'] == 'valid'].values
y_valid = y[df['group'] == 'valid'].values

# Set up the cross-validation strategy
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create a results DataFrame to store metrics
results = []

# Create directory for saving models
os.makedirs('rf_models', exist_ok=True)

# Initialize checkpoint tracking
checkpoints = {}

# Outer loop: 5-fold CV
for cv_idx, (train_idx, test_idx) in enumerate(cv_outer.split(X_train, y_train)):
    # Get the training data for this outer fold
    X_outer_train, y_outer_train = X_train[train_idx], y_train[train_idx]
    X_outer_test, y_outer_test = X_train[test_idx], y_train[test_idx]
    
    # Inner loop: 5-fold CV
    for fold_idx, (inner_train_idx, inner_val_idx) in enumerate(cv_inner.split(X_outer_train, y_outer_train)):
        # Split the data for this inner fold
        X_inner_train = X_outer_train[inner_train_idx]
        y_inner_train = y_outer_train[inner_train_idx] 
        X_inner_val = X_outer_train[inner_val_idx]
        y_inner_val = y_outer_train[inner_val_idx]
        
        # ADD STANDARDIZATION HERE
        scaler = StandardScaler()
        X_inner_train_scaled = scaler.fit_transform(X_inner_train)
        X_inner_val_scaled = scaler.transform(X_inner_val)
        
        # Train the Random Forest model 
        start_time = time.time()
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42+fold_idx  # Add fold_idx to get different seeds
        )
        rf.fit(X_inner_train_scaled, y_inner_train)  # USE SCALED DATA
        training_time = time.time() - start_time
        
        # Get predictions on validation fold
        y_pred_proba = rf.predict_proba(X_inner_val_scaled)[:, 1]  # USE SCALED DATA
        y_pred = rf.predict(X_inner_val_scaled)  # USE SCALED DATA
        
        # Calculate metrics
        auc = roc_auc_score(y_inner_val, y_pred_proba)
        acc = accuracy_score(y_inner_val, y_pred)
        kappa = cohen_kappa_score(y_inner_val, y_pred)
        prec = precision_score(y_inner_val, y_pred)
        rec = recall_score(y_inner_val, y_pred)
        f1 = f1_score(y_inner_val, y_pred)
        mcc = matthews_corrcoef(y_inner_val, y_pred)  # Added MCC
        tn, fp, fn, tp = confusion_matrix(y_inner_val, y_pred).ravel()
        spec = tn / (tn + fp)  # Specificity
        
        # Save individual model and scaler
        fold_id = f"cv{cv_idx+1}_fold{fold_idx+1}"
        model_dir = f'rf_models/{fold_id}'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and scaler
        model_path = os.path.join(model_dir, 'model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        joblib.dump(rf, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Track checkpoint path and performance
        checkpoints[fold_id] = {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'performance': {
                'auc': auc,
                'f1': f1,
                'mcc': mcc,
                'acc': acc,
                'precision': prec,
                'recall': rec,
                'specificity': spec,
                'kappa': kappa
            }
        }
        
        # Store results
        results.append({
            'CV': cv_idx + 1,
            'Fold': fold_idx + 1,
            'Seed': 42 + fold_idx,
            'AUC': auc,
            'Acc': acc,
            'Kappa': kappa,
            'Prec': prec,
            'Recall': rec,
            'F1': f1,
            'MCC': mcc, 
            'Spec': spec,
            'Time': training_time
        })
        
        print(f"CV {cv_idx+1}, Fold {fold_idx+1} - AUC: {auc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv('RF_5x5_cv_results_standardized.csv', index=False)  # DIFFERENT FILENAME

# Save model checkpoints information
with open('rf_model_checkpoints.json', 'w') as f:
    json.dump(checkpoints, f, indent=4)

print(f"\nSaved {len(checkpoints)} individual RF models to rf_models/ directory")
print(f"Model checkpoints info saved to rf_model_checkpoints.json")

# Calculate average metrics
avg_metrics = results_df[['AUC', 'Acc', 'Kappa', 'Prec', 'Recall', 'F1', 'MCC', 'Spec']].mean()
print("\nAverage Performance Across 5X5 CV (with standardization):")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")

# Individual model evaluation approach - no final single model training
print(f"\n=== Random Forest 5x5 Cross-Validation Complete ===")
print(f"Individual models saved: {len(checkpoints)}")
print(f"Results saved to: RF_5x5_cv_results_standardized.csv")
print(f"Model checkpoints: rf_model_checkpoints.json")
print(f"Use the new RF_test_evaluation.py script for comprehensive evaluation")