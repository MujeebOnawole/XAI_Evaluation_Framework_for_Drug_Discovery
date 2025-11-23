"""
Cross-validation implementation for CNN-based SMILES classification
Performs 5x5 cross-validation with best hyperparameters
"""

import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import logging
from typing import Dict, List, Any, Tuple, Optional
import warnings
from datetime import datetime
import json

from config import Config
from model import SMILESCNNModel
from data_loader import CrossValidationDataModule
from utils import (
    set_all_seeds, setup_logging, get_device, save_results,
    calculate_metrics_summary, save_dataframe_with_timestamp,
    explainability_test
)

# Enable Tensor Core optimization for faster matrix operations on V100/A100 GPUs
# This can provide 1.5-2x speedup for large models during long training runs
torch.set_float32_matmul_precision('medium')

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CrossValidationRunner:
    """Handles cross-validation experiments"""
    
    def __init__(
        self,
        features: np.ndarray,
        labels_df: pd.DataFrame,
        best_hyperparameters: Dict[str, Any],
        n_splits: int = Config.CV_FOLDS,
        n_repeats: int = Config.CV_REPEATS,
        max_epochs: int = Config.MAX_EPOCHS,
        patience: int = Config.EARLY_STOPPING_PATIENCE
    ):
        """
        Initialize cross-validation runner
        
        Args:
            features: Preprocessed features array
            labels_df: Labels dataframe
            best_hyperparameters: Best hyperparameters from optimization
            n_splits: Number of CV folds
            n_repeats: Number of CV repeats
            max_epochs: Maximum training epochs
            patience: Early stopping patience
        """
        self.features = features
        self.labels_df = labels_df
        self.best_hyperparameters = best_hyperparameters
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.max_epochs = max_epochs
        self.patience = patience
        
        # Setup CV data module
        self.cv_data_module = CrossValidationDataModule(
            features=features,
            labels_df=labels_df
        )
        
        # Store test data separately for final evaluation
        test_mask = labels_df['group'] == 'test'
        self.test_features = features[test_mask]
        self.test_labels_df = labels_df[test_mask].reset_index(drop=True)
        
        # Get data dimensions
        self.vocab_size = features.shape[-1]
        self.sequence_length = features.shape[1]
        
        # Storage for results
        self.cv_results = []
        self.trained_models = []
        self.fold_predictions = []
        
        logger.info(f"Cross-validation setup:")
        logger.info(f"  Total dataset: {features.shape[0]} samples")
        logger.info(f"  CV data: {len(self.cv_data_module.cv_features)} samples (augmented train+val)")
        logger.info(f"  Test data: {len(self.test_features)} samples (original, held out)")
        logger.info(f"  CV scheme: {n_repeats}x{n_splits} = {n_repeats * n_splits} models")
        logger.info(f"  Tensor Core optimization: {'enabled' if torch.get_float32_matmul_precision() == 'medium' else 'disabled'}")
        logger.info(f"  Best hyperparameters: {best_hyperparameters}")
        
        # Verify no test data contamination
        cv_groups = self.cv_data_module.cv_labels_df['group'].unique()
        test_groups = self.test_labels_df['group'].unique()
        logger.info(f"  CV groups: {cv_groups}")
        logger.info(f"  Test groups: {test_groups}")
        assert 'test' not in cv_groups, "ERROR: Test data found in CV - data leakage detected!"
        logger.info("  ✓ Data leakage check passed - test set properly separated")
    
    def run_single_fold(
        self,
        fold_id: str,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        save_model: bool = True
    ) -> Tuple[Dict[str, float], Optional[SMILESCNNModel]]:
        """
        Run training and evaluation for a single fold
        
        Args:
            fold_id: Unique identifier for this fold
            train_indices: Training data indices
            val_indices: Validation data indices
            save_model: Whether to save the trained model
            
        Returns:
            Tuple of (metrics_dict, trained_model)
        """
        logger.info(f"Starting fold {fold_id}")
        
        # Set seed for this fold
        fold_seed = Config.RANDOM_SEED + hash(fold_id) % 10000
        set_all_seeds(fold_seed)
        
        # Get data loaders for this fold
        train_loader, val_loader = self.cv_data_module.get_fold_dataloaders(
            train_indices, val_indices
        )
        
        # Get class weights for this fold
        class_weights = self.cv_data_module.get_class_weights(train_indices)
        
        # Create model
        model = SMILESCNNModel(
            vocab_size=self.vocab_size,
            sequence_length=self.sequence_length,
            class_weights=class_weights,
            **self.best_hyperparameters
        )
        
        # Setup callbacks
        early_stopping = EarlyStopping(
            monitor='val_auroc',
            mode='max',
            patience=self.patience,
            verbose=False
        )
        
        checkpoint_dir = os.path.join(Config.MODELS_DIR, 'cv_models', fold_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best-{epoch:02d}-{val_auroc:.4f}',
            monitor='val_auroc',
            mode='max',
            save_top_k=1,
            verbose=False
        )
        
        # Setup logger
        cv_logger = TensorBoardLogger(
            save_dir=Config.LOGS_DIR,
            name='cross_validation',
            version=fold_id
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator='auto',
            devices='auto',
            callbacks=[early_stopping, checkpoint_callback],
            logger=cv_logger,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=True,
            deterministic=True
        )
        
        # Train model
        trainer.fit(model, train_loader, val_loader)
        
        # Load best checkpoint
        best_model = SMILESCNNModel.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            vocab_size=self.vocab_size,
            sequence_length=self.sequence_length,
            class_weights=class_weights,
            **self.best_hyperparameters
        )
        
        # Evaluate on validation set
        trainer.test(best_model, val_loader, verbose=False)
        
        # Extract metrics from the model
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            if key.startswith('test_'):
                # Remove 'test_' prefix for consistency
                metric_name = key.replace('test_', '')
                metrics[metric_name] = float(value)
        
        # Add fold information
        metrics['fold_id'] = fold_id
        metrics['train_samples'] = len(train_indices)
        metrics['val_samples'] = len(val_indices)
        metrics['best_epoch'] = trainer.current_epoch
        
        # Get predictions for analysis
        predictions = trainer.predict(best_model, val_loader)
        fold_predictions = {
            'fold_id': fold_id,
            'val_indices': val_indices.tolist(),
            'predictions': self._extract_predictions(predictions)
        }
        
        logger.info(f"Fold {fold_id} completed. ROC-AUC: {metrics.get('auroc', 0):.4f}")
        
        # Clean up checkpoint if not saving models
        if not save_model:
            try:
                import shutil
                shutil.rmtree(checkpoint_dir)
            except Exception as e:
                logger.warning(f"Could not cleanup checkpoint for fold {fold_id}: {e}")
            return metrics, None
        
        return metrics, best_model, fold_predictions
    
    def _extract_predictions(self, predictions: List[Dict]) -> Dict[str, List]:
        """Extract predictions from model output"""
        all_probs = []
        all_preds = []
        all_logits = []
        all_compound_ids = []
        
        for batch_pred in predictions:
            all_probs.extend(batch_pred['probabilities'].tolist())
            all_preds.extend(batch_pred['predictions'].tolist())
            all_logits.extend(batch_pred['logits'].tolist())
            
            if 'compound_ids' in batch_pred:
                all_compound_ids.extend(batch_pred['compound_ids'])
        
        result = {
            'probabilities': all_probs,
            'predictions': all_preds,
            'logits': all_logits
        }
        
        if all_compound_ids:
            result['compound_ids'] = all_compound_ids
        
        return result
    
    def run_cross_validation(self, save_models: bool = True, job_part: Optional[int] = None) -> Dict[str, Any]:
        """
        Run complete cross-validation experiment
        
        Args:
            save_models: Whether to save trained models
            job_part: If specified (1 or 2), runs only subset of models for parallel processing
            
        Returns:
            Complete CV results dictionary
        """
        logger.info("Starting cross-validation experiment...")
        
        # Setup cross-validation
        cv_splitter = RepeatedStratifiedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=Config.RANDOM_SEED
        )
        
        # Get labels for stratification (only train/val data)
        cv_labels = self.cv_data_module.cv_labels_df['TARGET'].values
        
        # Run cross-validation
        model_counter = 0
        total_models = self.n_repeats * self.n_splits
        
        for repeat in range(self.n_repeats):
            logger.info(f"Starting repeat {repeat + 1}/{self.n_repeats}")
            
            # Create fold splitter for this repeat
            fold_splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=Config.RANDOM_SEED + repeat
            )
            
            for fold, (train_idx, val_idx) in enumerate(fold_splitter.split(
                np.arange(len(cv_labels)), cv_labels
            )):
                model_counter += 1
                
                # Skip models based on job_part
                if job_part == 1 and model_counter > 13:
                    continue
                if job_part == 2 and model_counter <= 13:
                    continue
                
                fold_id = f"cv{repeat + 1}_fold{fold + 1}"
                
                logger.info(f"Running model {model_counter}/{total_models}: {fold_id}")
                
                # Run single fold
                fold_results = self.run_single_fold(
                    fold_id=fold_id,
                    train_indices=train_idx,
                    val_indices=val_idx,
                    save_model=save_models
                )
                
                if save_models:
                    metrics, model, predictions = fold_results
                    self.trained_models.append({
                        'fold_id': fold_id,
                        'model': model
                    })
                    self.fold_predictions.append(predictions)
                else:
                    metrics, _ = fold_results
                
                # Add model counter and cv info to metrics
                metrics['model_id'] = model_counter
                metrics['cv_round'] = repeat + 1
                metrics['fold'] = fold + 1
                
                self.cv_results.append(metrics)
        
        if job_part:
            logger.info(f"Cross-validation part {job_part} completed!")
        else:
            logger.info("Cross-validation completed!")
        
        # Calculate summary statistics
        summary_stats = self._calculate_cv_summary()
        
        # Prepare final results
        final_results = {
            'cv_setup': {
                'n_splits': self.n_splits,
                'n_repeats': self.n_repeats,
                'total_folds': len(self.cv_results),
                'hyperparameters': self.best_hyperparameters,
                'job_part': job_part
            },
            'fold_results': self.cv_results,
            'summary_statistics': summary_stats,
            'predictions': self.fold_predictions if save_models else None
        }
        
        return final_results
    
    def _calculate_cv_summary(self) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics across all CV folds"""
        logger.info("Calculating cross-validation summary statistics...")
        
        # Extract metrics excluding non-numeric fields
        numeric_results = []
        for result in self.cv_results:
            numeric_result = {k: v for k, v in result.items() 
                            if isinstance(v, (int, float)) and k != 'fold_id'}
            numeric_results.append(numeric_result)
        
        summary = calculate_metrics_summary(numeric_results)
        
        # Log summary
        logger.info("Cross-validation summary:")
        for metric, stats in summary.items():
            logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                       f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
        
        return summary
    
    def save_results(self, filepath: Optional[str] = None, job_part: Optional[int] = None) -> str:
        """
        Save cross-validation results
        
        Args:
            filepath: Optional output file path
            job_part: Job part number for parallel execution
            
        Returns:
            Actual save path
        """
        if not self.cv_results:
            raise ValueError("No results to save. Run cross-validation first.")
        
        if filepath is None:
            if job_part:
                filepath = os.path.join(Config.OUTPUT_DIR, f'cv_part{job_part}_results.json')
            else:
                filepath = Config.CV_RESULTS_PATH
        
        # Prepare results for saving
        results_to_save = {
            'cv_setup': {
                'n_splits': self.n_splits,
                'n_repeats': self.n_repeats,
                'total_folds': len(self.cv_results),
                'hyperparameters': self.best_hyperparameters,
                'job_part': job_part
            },
            'fold_results': self.cv_results,
            'summary_statistics': self._calculate_cv_summary()
        }
        
        # Add model checkpoint paths if models were saved
        if self.trained_models:
            model_info = []
            for model_data in self.trained_models:
                fold_id = model_data['fold_id']
                checkpoint_dir = os.path.join(Config.MODELS_DIR, 'cv_models', fold_id)
                # Find the best checkpoint file
                try:
                    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('best-') and f.endswith('.ckpt')]
                    if checkpoint_files:
                        best_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
                        model_info.append({
                            'fold_id': fold_id,
                            'checkpoint_path': best_checkpoint
                        })
                except Exception as e:
                    logger.warning(f"Could not find checkpoint for {fold_id}: {e}")
            
            results_to_save['model_checkpoints'] = model_info
            logger.info(f"Saved {len(model_info)} model checkpoint paths")
        
        # Save JSON results
        save_results(results_to_save, filepath)
        
        # Save CSV for easy analysis
        cv_df = pd.DataFrame(self.cv_results)
        csv_path = filepath.replace('.json', '.csv')
        cv_df.to_csv(csv_path, index=False)
        logger.info(f"CV results CSV saved to {csv_path}")
        
        return filepath
    
    def get_best_model(self) -> SMILESCNNModel:
        """
        Get the best performing model from cross-validation
        
        Returns:
            Best model based on validation ROC-AUC
        """
        if not self.trained_models:
            raise ValueError("No trained models available. Set save_models=True in run_cross_validation.")
        
        # Find best model by ROC-AUC
        best_score = -1
        best_model = None
        
        for i, result in enumerate(self.cv_results):
            if result.get('auroc', 0) > best_score:
                best_score = result['auroc']
                best_model = self.trained_models[i]['model']
        
        logger.info(f"Best model ROC-AUC: {best_score:.4f}")
        return best_model
    
    def run_explainability_test(self, model: Optional[SMILESCNNModel] = None) -> Dict[str, Any]:
        """
        Run explainability test with the best model
        
        Args:
            model: Optional specific model to test. If None, uses best CV model.
            
        Returns:
            Explainability test results
        """
        if model is None:
            model = self.get_best_model()
        
        # We need a preprocessor for the explainability test
        # This is a simplified version - in practice, you'd load the actual preprocessor
        from data_preprocessing import SMILESPreprocessor
        preprocessor = SMILESPreprocessor()
        preprocessor.max_length = self.sequence_length
        
        return explainability_test(model, preprocessor)

def run_cross_validation_experiment(
    features: np.ndarray,
    labels_df: pd.DataFrame,
    best_hyperparameters: Dict[str, Any],
    save_models: bool = True,
    job_part: Optional[int] = None
) -> Dict[str, Any]:
    """
    Main function to run cross-validation experiment
    
    Args:
        features: Preprocessed features array
        labels_df: Labels dataframe
        best_hyperparameters: Best hyperparameters from optimization
        save_models: Whether to save trained models
        job_part: If specified (1 or 2), runs only subset of models for parallel processing
        
    Returns:
        Complete CV results
    """
    logger.info(f"Setting up cross-validation experiment (part {job_part})...")
    
    # Create CV runner
    cv_runner = CrossValidationRunner(
        features=features,
        labels_df=labels_df,
        best_hyperparameters=best_hyperparameters
    )
    
    # Run cross-validation
    results = cv_runner.run_cross_validation(save_models=save_models, job_part=job_part)
    
    # Save results with part-specific filename
    if job_part:
        results_path = os.path.join(Config.OUTPUT_DIR, f'cv_part{job_part}_results.json')
        csv_path = os.path.join(Config.OUTPUT_DIR, f'cv_part{job_part}_results.csv')
    else:
        results_path = Config.CV_RESULTS_PATH
        csv_path = Config.CV_RESULTS_PATH.replace('.json', '.csv')
    
    # Save results with model checkpoint info
    cv_runner.save_results(results_path, job_part)
    
    # Skip explainability test for parallel jobs
    if save_models and not job_part:
        try:
            explainability_results = cv_runner.run_explainability_test()
            results['explainability_test'] = explainability_results
            
            # Save explainability results separately
            explainability_path = os.path.join(Config.OUTPUT_DIR, 'explainability_test.json')
            save_results(explainability_results, explainability_path)
            
        except Exception as e:
            logger.warning(f"Explainability test failed: {e}")
    
    logger.info(f"Cross-validation experiment part {job_part} completed successfully!")
    
    return results

def main():
    """Main function with support for job part argument"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run CNN 5x5 Cross-Validation')
    parser.add_argument('--job-part', type=int, choices=[1, 2], 
                       help='Run specific job part (1: models 1-13, 2: models 14-25)')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Set seeds
    set_all_seeds()
    
    # Print system info
    from utils import print_system_info
    print_system_info()
    
    # Load preprocessed data
    try:
        # Use actual file names from your HPC outputs
        features_path = os.path.join(Config.OUTPUT_DIR, 'features.npy')
        labels_path = os.path.join(Config.OUTPUT_DIR, 'labels.csv')
        
        features = np.load(features_path)
        labels_df = pd.read_csv(labels_path)
        
        # Use optimized hyperparameters
        best_hyperparameters = {
            'layers': 3,
            'filters': 256,
            'kernel_size': 7,
            'dropout': 0.1992056469820396,
            'learning_rate': 0.00019566708363815042
        }
        
        logger.info(f"Loaded data: {features.shape} features, {len(labels_df)} labels")
        logger.info(f"Using best hyperparameters: {best_hyperparameters}")
        
        if args.job_part:
            logger.info(f"Running cross-validation job part {args.job_part}")
        
        # Run cross-validation experiment
        results = run_cross_validation_experiment(
            features=features,
            labels_df=labels_df,
            best_hyperparameters=best_hyperparameters,
            save_models=True,  # Save models for Grad-CAM explainability
            job_part=args.job_part
        )
        
        logger.info("Cross-validation completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"Required files not found: {e}")
        logger.info("Please ensure you have run data preprocessing first.")

if __name__ == "__main__":
    main()