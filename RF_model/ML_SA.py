import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)
from lazypredict.Supervised import LazyClassifier
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read the CSV file
df = pd.read_csv('SA_FG_fragments_filtered.csv')

# Separate features (X) and target (y)
X = df[[col for col in df.columns if col.startswith('fr_')]]
y = df['TARGET']

# Split the data based on the 'GROUP' column
X_train = X[df['GROUP'] == 'training']
X_valid = X[df['GROUP'] == 'valid']
X_test = X[df['GROUP'] == 'test']
y_train = y[df['GROUP'] == 'training']
y_valid = y[df['GROUP'] == 'valid']
y_test = y[df['GROUP'] == 'test']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Use LazyPredict to train and evaluate multiple classifiers
lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = lazy_clf.fit(X_train_scaled, X_valid_scaled, y_train, y_valid)

# Display all model results
print("\nAll model results:")
print(models)

# Save all models' metrics to CSV
models.to_csv('all_models_metrics.csv')
logger.info("Metrics for all models saved to 'all_models_metrics.csv'")

# Select the best model based on ROC AUC
best_model_name = models.sort_values('ROC AUC', ascending=False).index[0]
logger.info(f"Best model based on ROC AUC: {best_model_name}")

# Get the best model instance
best_model = lazy_clf.models[best_model_name]

# Combine training and validation sets for final training
X_train_valid_scaled = np.vstack((X_train_scaled, X_valid_scaled))
y_train_valid = pd.concat([y_train, y_valid])

# Refit the best model on the combined training and validation data
best_model.fit(X_train_valid_scaled, y_train_valid)

# Save the scaler and best model
joblib.dump(scaler, 'scaler.pkl')
model_filename = 'best_model.pkl'
joblib.dump(best_model, model_filename)
logger.info(f"Scaler saved to 'scaler.pkl' and best model saved to '{model_filename}'")

# Predict on the test set
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Calculate specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

# Calculate final metrics
metrics = {
    'Model': best_model_name,
    'Accuracy': accuracy_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'Specificity': specificity,
    'MCC': matthews_corrcoef(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_pred_proba)
}

# Save best model's test metrics to CSV
pd.DataFrame([metrics]).to_csv('best_model_test_metrics.csv', index=False)
logger.info("Best model's test metrics saved to 'best_model_test_metrics.csv'")

# Print final results
print("\nBest model performance on test set:")
print(pd.DataFrame([metrics]))