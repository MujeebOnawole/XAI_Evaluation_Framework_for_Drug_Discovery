"""
Hyperparameter optimization using Optuna
Performs Bayesian optimization to find best CNN hyperparameters
"""

import optuna
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import warnings
import os
from datetime import datetime
import sqlite3
import json
from pathlib import Path

from config import Config
from model import SMILESCNNModel, create_model_from_trial
from data_loader import SMILESDataModule
from utils import set_all_seeds, setup_logging, get_device, save_results

# Enable Tensor Core optimization for faster matrix operations on V100/A100 GPUs
# This can provide 1.5-2x speedup for large models during long training runs
torch.set_float32_matmul_precision('medium')

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

class OptunaObjective:
    """Optuna objective function for hyperparameter optimization"""
    
    def __init__(
        self,
        data_module: SMILESDataModule,
        max_epochs: int = Config.MAX_EPOCHS,
        patience: int = Config.EARLY_STOPPING_PATIENCE,
        accelerator: str = 'auto',
        devices: str = 'auto'
    ):
        """
        Initialize objective function
        
        Args:
            data_module: Configured data module
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            accelerator: Training accelerator ('gpu', 'cpu', 'auto')
            devices: Number of devices to use
        """
        self.data_module = data_module
        self.max_epochs = max_epochs
        self.patience = patience
        self.accelerator = accelerator
        self.devices = devices
        
        # Get data statistics
        self.vocab_size = data_module.vocab_size
        self.sequence_length = data_module.sequence_length
        self.class_weights = data_module.class_weights
        
        logger.info(f"Objective initialized:")
        logger.info(f"  Vocab size: {self.vocab_size}")
        logger.info(f"  Sequence length: {self.sequence_length}")
        logger.info(f"  Max epochs: {self.max_epochs}")
        logger.info(f"  Patience: {self.patience}")
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation ROC-AUC score to maximize
        """
        try:
            # Set seeds for reproducibility
            trial_seed = Config.RANDOM_SEED + trial.number
            set_all_seeds(trial_seed)
            
            # Create model from trial
            model = create_model_from_trial(
                trial=trial,
                vocab_size=self.vocab_size,
                sequence_length=self.sequence_length,
                class_weights=self.class_weights
            )
            
            # Setup callbacks
            early_stopping = EarlyStopping(
                monitor='val_auroc',
                mode='max',
                patience=self.patience,
                verbose=False
            )
            
            # Create temporary checkpoint for this trial
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(Config.MODELS_DIR, f'trial_{trial.number}'),
                filename='best-{epoch:02d}-{val_auroc:.4f}',
                monitor='val_auroc',
                mode='max',
                save_top_k=1,
                verbose=False
            )
            
            # Setup logger for this trial
            trial_logger = TensorBoardLogger(
                save_dir=Config.LOGS_DIR,
                name=f'trial_{trial.number}',
                version='optuna'
            )
            
            # Create trainer
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                accelerator=self.accelerator,
                devices=self.devices,
                callbacks=[early_stopping, checkpoint_callback],
                logger=trial_logger,
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=True,
                deterministic=True
            )
            
            # Train model
            trainer.fit(model, self.data_module)
            
            # Get best validation score
            best_val_auroc = checkpoint_callback.best_model_score.item()
            
            # Log trial results
            logger.info(f"Trial {trial.number}: val_auroc={best_val_auroc:.4f}")
            logger.info(f"  Hyperparameters: {trial.params}")
            
            # Cleanup checkpoint directory to save space
            try:
                import shutil
                checkpoint_dir = os.path.join(Config.MODELS_DIR, f'trial_{trial.number}')
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)
            except Exception as e:
                logger.warning(f"Could not cleanup trial {trial.number} checkpoint: {e}")
            
            return best_val_auroc
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            # Return a low score for failed trials
            return 0.0

class HyperparameterOptimizer:
    """Main class for hyperparameter optimization with SQLite persistence"""
    
    def _find_existing_study(self, storage_dir: Path) -> Optional[str]:
        """Find existing study database"""
        pattern = "smiles_cnn_optimization_*.db"
        existing_dbs = list(storage_dir.glob(pattern))
        
        if existing_dbs:
            # Return the study name from the most recent database
            db_path = sorted(existing_dbs, key=lambda p: p.stat().st_mtime)[-1]
            return db_path.stem  # removes .db extension
        return None
    
    def __init__(
        self,
        data_module: SMILESDataModule,
        n_trials: int = Config.N_TRIALS,
        study_name: Optional[str] = None,
        direction: str = 'maximize',
        storage_dir: Optional[str] = None
    ):
        """
        Initialize hyperparameter optimizer
        
        Args:
            data_module: Configured data module
            n_trials: Number of optimization trials
            study_name: Optional study name
            direction: Optimization direction ('maximize' or 'minimize')
            storage_dir: Directory to store SQLite database (default: outputs/)
        """
        self.data_module = data_module
        self.n_trials = n_trials
        self.direction = direction
        
        # Setup storage directory first
        if storage_dir is None:
            storage_dir = Config.OUTPUT_DIR
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Check for existing study first, then create study name
        if study_name is None:
            existing_study = self._find_existing_study(self.storage_dir)
            if existing_study:
                study_name = existing_study
                logger.info(f"Found existing study: {study_name}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                study_name = f"smiles_cnn_optimization_{timestamp}"
                logger.info(f"Creating new study: {study_name}")
        
        self.study_name = study_name
        
        self.db_path = self.storage_dir / f"{self.study_name}.db"
        self.storage_url = f"sqlite:///{self.db_path}"
        
        # Create objective
        self.objective = OptunaObjective(data_module)
        
        # Initialize study
        self.study = None
        self.best_trial = None
        self.optimization_results = None
        self.is_resuming = False
        
        logger.info(f"Optimizer initialized:")
        logger.info(f"  Study name: {self.study_name}")
        logger.info(f"  Number of trials: {self.n_trials}")
        logger.info(f"  Direction: {self.direction}")
        logger.info(f"  Database path: {self.db_path}")
        logger.info(f"  Tensor Core optimization: {'enabled' if torch.get_float32_matmul_precision() == 'medium' else 'disabled'}")
    
    def _check_for_existing_study(self) -> bool:
        """
        Check if there's an existing study database to resume from
        
        Returns:
            True if existing study found, False otherwise
        """
        if not self.db_path.exists():
            return False
        
        try:
            # Try to load existing study
            test_study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage_url
            )
            n_existing_trials = len(test_study.trials)
            
            if n_existing_trials > 0:
                logger.info(f"Found existing study with {n_existing_trials} completed trials")
                return True
            
        except Exception as e:
            logger.warning(f"Could not load existing study: {e}")
            return False
        
        return False
    
    def _save_checkpoint(self, trial_number: int, trial_value: float, trial_params: Dict[str, Any]):
        """
        Save checkpoint information after each trial
        
        Args:
            trial_number: Trial number
            trial_value: Trial objective value
            trial_params: Trial hyperparameters
        """
        try:
            checkpoint_data = {
                'trial_number': trial_number,
                'trial_value': trial_value,
                'trial_params': trial_params,
                'timestamp': datetime.now().isoformat(),
                'study_name': self.study_name
            }
            
            checkpoint_path = self.storage_dir / f"{self.study_name}_checkpoint.json"
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            logger.debug(f"Checkpoint saved for trial {trial_number}")
            
        except Exception as e:
            logger.warning(f"Could not save checkpoint for trial {trial_number}: {e}")
    
    def optimize(self, timeout: Optional[int] = None) -> optuna.Study:
        """
        Run hyperparameter optimization with SQLite persistence and resume capability
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Completed Optuna study
        """
        # Check for existing study
        self.is_resuming = self._check_for_existing_study()
        
        if self.is_resuming:
            logger.info(f"Resuming optimization from existing database: {self.db_path}")
        else:
            logger.info(f"Starting fresh hyperparameter optimization")
            logger.info(f"Database will be created at: {self.db_path}")
        
        # Create or load study with SQLite storage
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage_url,
            load_if_exists=True,  # This enables resuming
            sampler=optuna.samplers.TPESampler(seed=Config.RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        )
        
        # Calculate remaining trials if resuming
        completed_trials = len(self.study.trials)
        remaining_trials = max(0, self.n_trials - completed_trials)
        
        if self.is_resuming:
            logger.info(f"Completed trials: {completed_trials}")
            logger.info(f"Remaining trials: {remaining_trials}")
            
            if remaining_trials == 0:
                logger.info("All trials already completed!")
                self.best_trial = self.study.best_trial
                return self.study
        
        # Custom objective wrapper for checkpoint saving
        def objective_with_checkpoint(trial):
            try:
                result = self.objective(trial)
                # Save checkpoint after successful trial
                self._save_checkpoint(trial.number, result, trial.params)
                return result
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                # Save failed trial checkpoint
                self._save_checkpoint(trial.number, None, trial.params)
                return 0.0
        
        # Run optimization
        logger.info(f"Running {remaining_trials} trials...")
        self.study.optimize(
            objective_with_checkpoint,
            n_trials=remaining_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Store best trial
        self.best_trial = self.study.best_trial
        
        # Log final results
        total_trials = len(self.study.trials)
        logger.info("Hyperparameter optimization completed!")
        logger.info(f"Total trials completed: {total_trials}")
        logger.info(f"Best trial: {self.best_trial.number}")
        logger.info(f"Best value (val_auroc): {self.best_trial.value:.4f}")
        logger.info(f"Best parameters: {self.best_trial.params}")
        logger.info(f"Study database saved to: {self.db_path}")
        
        return self.study
    
    def get_optimization_results(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization results
        
        Returns:
            Dictionary with optimization results
        """
        if self.study is None:
            raise ValueError("Optimization has not been run yet")
        
        # Basic study info
        results = {
            'study_name': self.study_name,
            'n_trials': len(self.study.trials),
            'direction': self.direction,
            'best_trial_number': self.best_trial.number,
            'best_value': self.best_trial.value,
            'best_params': self.best_trial.params,
            'database_path': str(self.db_path),
            'was_resumed': self.is_resuming
        }
        
        # All trial results
        trials_data = []
        for trial in self.study.trials:
            trial_data = {
                'trial_number': trial.number,
                'value': trial.value,
                'state': trial.state.name,
                'params': trial.params,
                'duration': trial.duration.total_seconds() if trial.duration else None
            }
            trials_data.append(trial_data)
        
        results['trials'] = trials_data
        
        # Parameter importance (if available)
        try:
            importance = optuna.importance.get_param_importances(self.study)
            results['param_importance'] = importance
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
            results['param_importance'] = {}
        
        # Optimization history
        results['optimization_history'] = [
            {'trial': i, 'value': trial.value} 
            for i, trial in enumerate(self.study.trials) 
            if trial.value is not None
        ]
        
        self.optimization_results = results
        return results
    
    def save_results(self, filepath: Optional[str] = None) -> str:
        """
        Save optimization results to file
        
        Args:
            filepath: Optional output file path
            
        Returns:
            Actual save path
        """
        if self.optimization_results is None:
            self.get_optimization_results()
        
        if filepath is None:
            filepath = Config.HYPEROPT_RESULTS_PATH
        
        save_results(self.optimization_results, filepath)
        
        # Also save as CSV for easy analysis
        trials_df = pd.DataFrame(self.optimization_results['trials'])
        csv_path = filepath.replace('.json', '.csv')
        trials_df.to_csv(csv_path, index=False)
        logger.info(f"Trials CSV saved to {csv_path}")
        
        return filepath
    
    def create_best_model(self) -> SMILESCNNModel:
        """
        Create model with best hyperparameters
        
        Returns:
            Model configured with best hyperparameters
        """
        if self.best_trial is None:
            raise ValueError("Optimization has not been run yet")
        
        model = SMILESCNNModel(
            vocab_size=self.data_module.vocab_size,
            sequence_length=self.data_module.sequence_length,
            layers=self.best_trial.params['layers'],
            filters=self.best_trial.params['filters'],
            kernel_size=self.best_trial.params['kernel_size'],
            dropout=self.best_trial.params['dropout'],
            learning_rate=self.best_trial.params['learning_rate'],
            class_weights=self.data_module.class_weights
        )
        
        logger.info("Best model created with optimized hyperparameters")
        return model
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history
        
        Args:
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.study is None:
                raise ValueError("Optimization has not been run yet")
            
            # Create optimization history plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Optimization history
            optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=ax1)
            ax1.set_title('Optimization History')
            
            # Parameter importance
            try:
                optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax2)
                ax2.set_title('Parameter Importance')
            except Exception:
                ax2.text(0.5, 0.5, 'Parameter importance\nnot available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Parameter Importance')
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = os.path.join(Config.OUTPUT_DIR, 'optimization_plots.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Optimization plots saved to {save_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available. Plots not generated.")
        except Exception as e:
            logger.error(f"Error creating plots: {e}")

def run_hyperparameter_optimization(
    features: np.ndarray,
    labels_df: pd.DataFrame,
    n_trials: int = Config.N_TRIALS,
    study_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to run hyperparameter optimization with persistence
    
    Args:
        features: Preprocessed features array
        labels_df: Labels dataframe
        n_trials: Number of optimization trials
        study_name: Optional study name for persistence
        
    Returns:
        Optimization results dictionary
    """
    logger.info("Setting up hyperparameter optimization...")
    
    # Setup data module
    data_module = SMILESDataModule(features=features, labels_df=labels_df)
    data_module.setup('fit')
    
    # Create optimizer with persistence
    optimizer = HyperparameterOptimizer(
        data_module=data_module,
        n_trials=n_trials,
        study_name=study_name
    )
    
    # Run optimization
    study = optimizer.optimize()
    
    # Get and save results
    results = optimizer.get_optimization_results()
    optimizer.save_results()
    
    # Create plots
    optimizer.plot_optimization_history()
    
    logger.info("Hyperparameter optimization completed successfully!")
    logger.info(f"Study database preserved at: {optimizer.db_path}")
    
    return results

def main():
    """Example usage of hyperparameter optimization"""
    # Setup logging
    logger = setup_logging()
    
    # Set seeds
    set_all_seeds()
    
    # Print system info
    from utils import print_system_info
    print_system_info()
    
    # Load preprocessed data (assuming it exists)
    try:
        features = np.load(Config.FEATURES_PATH)
        labels_df = pd.read_csv(Config.LABELS_PATH)
        
        logger.info(f"Loaded data: {features.shape} features, {len(labels_df)} labels")
        
        # Run optimization with fewer trials for testing
        results = run_hyperparameter_optimization(
            features=features,
            labels_df=labels_df,
            n_trials=10  # Use fewer trials for testing
        )
        
        logger.info("Hyperparameter optimization main completed successfully!")
        
    except FileNotFoundError:
        logger.error("Preprocessed data not found. Please run data preprocessing first.")
        logger.info("You can run: python data_preprocessing.py")

if __name__ == "__main__":
    main()