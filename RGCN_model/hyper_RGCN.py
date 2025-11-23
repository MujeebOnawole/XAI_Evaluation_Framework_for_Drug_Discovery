# hyper3.py

import os
import optuna
from optuna.storages import RDBStorage
from optuna.trial import TrialState
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
import pandas as pd
from datetime import datetime
import numpy as np
import logging
import random
from typing import Dict, Any
from model3 import BaseGNN  
from data_module import MoleculeDataModule
from logger import get_logger
from config3 import Configuration
from memory_tracker import MemoryTracker 
import time
import sqlalchemy
import glob
import gc

# Set PyTorch global settings upfront
torch.set_float32_matmul_precision('medium')
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialize logger at the module level
logger = get_logger(__name__)

from pytorch_lightning.callbacks import Callback

class BestMetricCallback(pl.Callback):
    """Enhanced callback to track the best metric during training."""
    
    def __init__(self, metric_name: str, mode: str, trial: optuna.Trial):
        super().__init__()
        self.metric_name = metric_name
        self.mode = mode
        self.trial = trial
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
    def on_validation_end(self, trainer, pl_module):
        current_metric = trainer.callback_metrics.get(self.metric_name)
        
        if current_metric is not None:
            current_metric = current_metric.item()
            
            improved = ((self.mode == 'min' and current_metric < self.best_metric) or
                       (self.mode == 'max' and current_metric > self.best_metric))
            
            if improved:
                self.best_metric = current_metric
                
            # Always update the trial with the best metric
            self.trial.set_user_attr('best_metric', self.best_metric)
            trainer.callback_metrics[f'best_{self.metric_name}'] = torch.tensor(self.best_metric)


class MemoryTrackingCallback(pl.Callback):
    """Optimized callback for tracking memory usage during training"""
    def __init__(self, memory_tracker, trial_number, is_classification, log_frequency=20):
        """
        Initialize the memory tracking callback.
        
        Args:
            memory_tracker: The memory tracker instance
            trial_number: The trial number being tracked
            is_classification: Whether this is a classification task
            log_frequency: How often to log memory stats (in epochs)
        """
        self.memory_tracker = memory_tracker
        self.trial_number = trial_number
        self.is_classification = is_classification
        self.log_frequency = log_frequency  # Log every N epochs
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Log memory stats only at fixed intervals"""
        current_epoch = trainer.current_epoch
        
        # Only log on specific conditions
        should_log = (
            current_epoch == 0 or  # First epoch
            current_epoch == trainer.max_epochs - 1 or  # Last epoch
            current_epoch % self.log_frequency == 0  # Every N epochs
        )
                
        if should_log:
            prefix = "FinalTraining" if getattr(trainer, "final_training", False) else f"Trial_{self.trial_number}"
            self.memory_tracker.log_memory_stats(
                f"{prefix}_Epoch_{current_epoch}"
            )
            
    def on_fit_start(self, trainer, pl_module):
        """Log memory stats at start of training"""
        prefix = "FinalTraining" if getattr(trainer, "final_training", False) else f"Trial_{self.trial_number}"
        self.memory_tracker.log_memory_stats(f"{prefix}_Start")
        
    def on_fit_end(self, trainer, pl_module):
        """Log memory stats at end of training"""
        prefix = "FinalTraining" if getattr(trainer, "final_training", False) else f"Trial_{self.trial_number}"
        self.memory_tracker.log_memory_stats(f"{prefix}_End")

class HyperparameterOptimization:
    def __init__(self, config: Configuration, n_trials: int = 50):
        self.config = config
        self.n_trials = n_trials  # Desired number of completed trials

        # Initialize memory tracker
        self.memory_tracker = MemoryTracker()
        self.memory_tracker.log_system_info()

        # Define study name and storage
        self.study_name = f"optimize_{config.task_name_with_suffix}_{config.task_type}"
        db_name = f"optuna_study_{config.task_name_with_suffix}_{config.task_type}.db"
        self.db_path = os.path.join(config.output_dir, db_name)

        self.storage = RDBStorage(
            url=f"sqlite:///{self.db_path}",
            engine_kwargs={"connect_args": {"timeout": 1800}}
        )

        self.logger = get_logger(__name__)
        self.logger.info(f"Optuna database created at: {self.db_path}")

        # Initialize other attributes
        self.study = None

        # Optimize configuration for hyperparameter search
        self.config.optimize_for_hyperparameter_search()

        # Trainer kwargs
        self.trainer_kwargs = {
            **self.config.get_training_kwargs(),
            'precision': '16',  # Use standard 16-bit precision
            'deterministic': False,  # Allow non-deterministic ops for better performance
            'benchmark': False,  # Keep benchmarking off for consistency
            'enable_checkpointing': False,  # Disable global checkpointing
        }
        
        # Track completed trials
        self.completed_trials = 0
        self.verification_pending = True
        
        # Initialize data module with proper cleanup
        self.data_module = None
        self._initialize_data_module()

        # Set optimization metrics based on task type

        if self.config.classification:
            self.optimization_metric = 'val_auc'
            self.optimization_mode_optuna = 'maximize'
            self.optimization_mode_lightning = 'max'  # Mapped value
            self.best_trial_value = float('-inf')
            self.metrics = [
                'val_accuracy', 'val_f1', 'val_precision', 'val_recall',
                'val_auc', 'val_specificity', 'val_mcc'
            ]
        else:
            # Change this section
            self.optimization_metric = 'val_weighted_rmse'  # Changed from 'val_rmse'
            self.optimization_mode_optuna = 'minimize'
            self.optimization_mode_lightning = 'min'  # Mapped value
            self.best_trial_value = float('inf')
            self.metrics = [
                'val_weighted_rmse', 'val_rmse', 'val_weighted_mse', 'val_mae', 'val_r2', 'val_mse'  # Reordered to put weighted metrics first
            ]

        self.best_trial_number = None

        # Load and validate metadata
        self._load_and_validate_metadata()

    def _initialize_data_module(self):
        """Initialize DataModule with proper error handling"""
        try:
            if self.data_module is not None:
                self.data_module.cleanup()
            
            self.data_module = MoleculeDataModule(self.config)
            self.data_module.setup()
        except Exception as e:
            self.logger.error(f"Failed to initialize DataModule: {str(e)}")
            raise

    def _load_and_validate_metadata(self):
        """Helper method to load and validate metadata."""
        meta_file_path = self.config.get_processed_file_path('meta', 'primary')
        self.logger.info(f"Looking for metadata file at: {meta_file_path}")

        if not os.path.exists(meta_file_path):
            raise ValueError(f"Metadata file not found at: {meta_file_path}")

        try:
            train_meta = pd.read_csv(meta_file_path)
            train_meta = train_meta[train_meta['group'] == 'training']
            self.logger.info(f"Successfully loaded metadata file. Columns: {train_meta.columns.tolist()}")
        except Exception as e:
            self.logger.error(f"Error loading metadata file: {str(e)}")
            raise

        # Verify label column exists
        if self.config.labels_name not in train_meta.columns:
            available_columns = train_meta.columns.tolist()
            error_msg = (
                f"Label column '{self.config.labels_name}' not found in metadata.\n"
                f"Available columns: {available_columns}\n"
                f"Please check that the labels_name in config matches one of the available columns."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        train_labels = train_meta[self.config.labels_name].values
        self.config.analyze_dataset(train_labels)
        self.config.set_train_labels(train_labels)

    def _load_study_status(self) -> Dict[str, int]:
        """Load study status directly from Optuna database."""
        try:
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage
            )

            trials = study.trials
            completed = len([t for t in trials if t.state == TrialState.COMPLETE])
            running = len([t for t in trials if t.state == TrialState.RUNNING])
            failed = len([t for t in trials if t.state == TrialState.FAIL])
            pruned = len([t for t in trials if t.state == TrialState.PRUNED])

            remaining = max(0, self.n_trials - completed)

            return {
                'total': len(trials),
                'completed': completed,
                'running': running,
                'failed': failed,
                'pruned': pruned,
                'remaining': remaining
            }
        except Exception as e:
            self.logger.error(f"Error loading study status: {str(e)}")
            return {
                'total': 0,
                'completed': 0,
                'running': 0,
                'failed': 0,
                'pruned': 0,
                'remaining': self.n_trials
            }

    def cleanup(self):
        """Enhanced cleanup method"""
        try:
            if self.data_module is not None:
                self.data_module.cleanup()
                self.data_module = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
            time.sleep(0.1)
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error in cleanup: {str(e)}")

    def _clean_running_trials(self):
        """Clean up stale running trials in the database."""
        try:
            engine = sqlalchemy.create_engine(f"sqlite:///{self.db_path}")
            with engine.connect() as conn:
                conn.execute(
                    sqlalchemy.text("UPDATE trials SET state = 'FAIL' WHERE state = 'RUNNING'")
                )
                conn.commit()
            self.logger.info("Cleaned up stale running trials")
        except Exception as e:
            self.logger.warning(f"Error cleaning running trials: {str(e)}")
            
    def __del__(self):
        """Enhanced destructor"""
        try:
            self.cleanup()
            self.config.restore_original_configuration()
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in destructor cleanup: {str(e)}")

    def run_optimization(self) -> None:
        """Run optimization without unnecessary best model training."""
        try:
            status = self._load_study_status()
            
            self.logger.info("\nCurrent Study Status:")
            self.logger.info(f"Total trials in database: {status['total']}")
            self.logger.info(f"Completed trials: {status['completed']}")
            self.logger.info(f"Running trials: {status['running']}")
            self.logger.info(f"Failed trials: {status['failed']}")
            self.logger.info(f"Pruned trials: {status['pruned']}")
            self.logger.info(f"Remaining trials needed: {status['remaining']}\n")

            if status['completed'] >= self.n_trials:
                self.logger.info(f"All {self.n_trials} trials already completed")
                self.study = optuna.load_study(
                    study_name=self.study_name,
                    storage=self.storage
                )
                # Save results and exit without training best model
                self.save_results(self.study)
                return

            # Clean any stale running trials
            if status['running'] > 0:
                self._clean_running_trials()
                status = self._load_study_status()

            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                load_if_exists=True,
                direction=self.optimization_mode_optuna,
                sampler=optuna.samplers.TPESampler(seed=42)
            )

            # Run exactly the number of remaining trials
            remaining_trials = status['remaining']
            self.logger.info(f"Running {remaining_trials} remaining trials...\n")
            
            self.study.optimize(
                func=self.objective,
                n_trials=remaining_trials,
                timeout=None
            )

            # Skip best model verification and training
            final_status = self._load_study_status()
            if final_status['completed'] >= self.n_trials:
                self.logger.info(f"\nDesired number of completed trials ({self.n_trials}) achieved.")
                # Save results and exit
                self.save_results(self.study)
            else:
                self.logger.warning(f"\nOnly achieved {final_status['completed']}/{self.n_trials} trials.")

        except Exception as e:
            self.logger.error(f"Error in optimization: {str(e)}")
            raise
        finally:
            self.config.restore_original_configuration()


    def objective(self, trial: optuna.Trial) -> float:
        """Modified objective function with improved metric tracking."""
        trial_id = trial.number
        
        try:
            self.logger.info(f"Initializing trial {trial_id} setup...")
            self.memory_tracker.log_memory_stats(f"TRIAL_START_{trial_id}")
            
            # Create model and add trial object
            model = self.create_model(trial)
            model.trial = trial
            
            # Define callbacks with increased patience
            callbacks = [
                EarlyStopping(
                    monitor=self.optimization_metric,
                    mode=self.optimization_mode_lightning,
                    patience=20,  # Increased patience
                    min_delta=1e-4,  # Add small threshold for improvements
                    verbose=True
                ),
                MemoryTrackingCallback(
                    self.memory_tracker,
                    trial_id,
                    self.config.classification
                ),
                BestMetricCallback(
                    metric_name=self.optimization_metric,
                    mode=self.optimization_mode_lightning,
                    trial=trial
                )
            ]
            
            # Initialize Trainer
            trainer = pl.Trainer(
                callbacks=callbacks,
                logger=pl.loggers.TensorBoardLogger(
                    save_dir=os.path.join(self.config.output_dir, 'tensorboard_logs'),
                    name=f'{self.config.task_name}_{self.config.task_type}',
                    version=f'trial_{trial_id}'
                ),
                **self.trainer_kwargs,
            )
            
            try:
                trainer.fit(model, self.data_module)
            except Exception as e:
                self.logger.error(f"Training failed for trial {trial_id}: {str(e)}")
                raise optuna.exceptions.TrialPruned()
            
            # Get the best metric value
            best_metric = trial.user_attrs.get('best_metric')
            if best_metric is None:
                # Fallback to final metric if best not found
                current_metric = trainer.callback_metrics.get(self.optimization_metric)
                if current_metric is not None:
                    best_metric = current_metric.item()
                else:
                    self.logger.warning(f"Trial {trial_id}: No {self.optimization_metric} found. Pruning trial.")
                    raise optuna.exceptions.TrialPruned()
            
            self.logger.info(f"Trial {trial_id} completed with best {self.optimization_metric}: {best_metric:.6f}")
            
            # Store other metrics for later analysis
            for metric_name in self.metrics:
                metric_value = trainer.callback_metrics.get(metric_name)
                if metric_value is not None:
                    trial.set_user_attr(f'final_{metric_name}', metric_value.item())
                    # Also store the best value if we have it
                    best_value = trainer.callback_metrics.get(f'best_{metric_name}')
                    if best_value is not None:
                        trial.set_user_attr(f'best_{metric_name}', best_value.item())
            
            return best_metric
            
        except optuna.exceptions.TrialPruned:
            self.logger.info(f"Trial {trial_id} pruned")
            raise
        except Exception as e:
            self.logger.error(f"Error in trial {trial_id}: {str(e)}")
            raise
        finally:
            # Clean up resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.memory_tracker.log_memory_stats(f"TRIAL_END_{trial_id}")

    def create_model(self, trial: optuna.Trial) -> BaseGNN:
        """Create model with sampled hyperparameters that are compatible with BaseGNN."""
        # Sample hyperparameters as per your existing implementation
        rgcn_hidden_feats_choice = trial.suggest_categorical(
            'rgcn_hidden_feats',
            self.config.rgcn_hidden_feats_choices
        )
        rgcn_hidden_feats = self.config.rgcn_hidden_feats_map[rgcn_hidden_feats_choice]

        ffn_hidden_feats_choice = trial.suggest_categorical(
            'ffn_hidden_feats',
            self.config.ffn_hidden_feats_choices
        )
        ffn_hidden_feats = self.config.ffn_hidden_feats_map[ffn_hidden_feats_choice]

        ffn_dropout = trial.suggest_categorical(
            'ffn_dropout',
            self.config.ffn_dropout_choices
        )
        rgcn_dropout = trial.suggest_categorical(
            'rgcn_dropout',
            self.config.rgcn_dropout_choices
        )

        lr = trial.suggest_float(
            'lr',
            self.config.lr_min,
            self.config.lr_max,
            log=True
        )
        weight_decay = trial.suggest_float(
            'weight_decay', 
            self.config.weight_decay_min, 
            self.config.weight_decay_max, 
            log=True
        )

        # Update config with new hyperparameters
        self.config.update_hyperparameters({
            'rgcn_hidden_feats': rgcn_hidden_feats,
            'ffn_hidden_feats': [ffn_hidden_feats],
            'ffn_dropout': ffn_dropout,
            'rgcn_dropout': rgcn_dropout,
            'lr': lr,
            'weight_decay': weight_decay
        })

        return BaseGNN(
            config=self.config,
            rgcn_hidden_feats=rgcn_hidden_feats,
            ffn_hidden_feats=ffn_hidden_feats,
            ffn_dropout=ffn_dropout,
            rgcn_dropout=rgcn_dropout,
            classification=self.config.classification
        )


    def save_results(self, study: optuna.Study) -> None:
        """Save optimization results handling both classification and regression metrics.
        
        Args:
            study (optuna.Study): The completed optimization study containing trials.
            
        Raises:
            Exception: If there's an error processing trials or saving results.
        """
        try:
            results = []
            
            # Define essential metrics based on task type
            if self.config.classification:
                essential_metrics = ['val_auc', 'val_accuracy', 'val_loss']
            else:
                essential_metrics = [
                    'val_weighted_rmse',  # Primary optimization metric
                    'val_rmse',
                    'val_weighted_mse',
                    'val_mse'
                ]
            
            metric_format = {k: '.6f' for k in essential_metrics}
    
            for trial in study.trials:
                if trial.state != TrialState.COMPLETE:
                    continue
    
                try:
                    # Basic trial information
                    row = {
                        'task_name': self.config.task_name,
                        'task_type': self.config.task_type,
                        'trial_number': trial.number,
                        'learning_rate': f"{trial.params.get('lr', 0.0):.6f}",
                        'weight_decay': f"{trial.params.get('weight_decay', 0.0):.6f}",
                        'rgcn_hidden_feats': trial.params.get('rgcn_hidden_feats', 'N/A'),
                        'ffn_hidden_feats': trial.params.get('ffn_hidden_feats', 'N/A'),
                        'ffn_dropout': f"{trial.params.get('ffn_dropout', 0.0):.6f}",
                        'rgcn_dropout': f"{trial.params.get('rgcn_dropout', 0.0):.6f}",
                    }
    
                    # Get all metrics from trial user_attrs
                    for metric in essential_metrics:
                        # Try both best and final values
                        best_value = trial.user_attrs.get(f'best_{metric}')
                        final_value = trial.user_attrs.get(f'final_{metric}')
                        
                        # If it's the optimization metric, also check trial.value
                        if metric == self.optimization_metric and trial.value is not None:
                            value = trial.value
                        else:
                            value = best_value if best_value is not None else final_value
                        
                        if value is not None and isinstance(value, (float, np.floating)):
                            row[metric] = format(float(value), metric_format.get(metric, '.6f'))
                        else:
                            row[metric] = 'N/A'
    
                    results.append(row)
    
                except Exception as e:
                    self.logger.error(f"Error processing trial {trial.number}: {str(e)}")
                    continue
    
            if not results:
                self.logger.warning("No completed trials found in the study")
                return
    
            # Define columns order
            columns = [
                'task_name', 'task_type', 'trial_number', 'learning_rate',
                'weight_decay', 'rgcn_hidden_feats', 'ffn_hidden_feats',
                'ffn_dropout', 'rgcn_dropout'
            ] + essential_metrics
    
            # Create DataFrame with specific column order
            df = pd.DataFrame(results)
            df = df[columns]  # Reorder columns
    
            # Sort by appropriate metric
            if self.optimization_mode_optuna == 'maximize':
                df = df.sort_values(self.optimization_metric, ascending=False)
            else:
                df = df.sort_values(self.optimization_metric)
    
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.config.output_dir,
                f'{self.config.task_name}_{self.config.task_type}_hyperparameter_results_{timestamp}.csv'
            )
    
            # Save results atomically using temporary file
            temp_file = output_file + '.tmp'
            try:
                df.to_csv(temp_file, index=False)
                os.replace(temp_file, output_file)
                self.logger.info(f"Results successfully saved to {output_file}")
                
                # Log summary statistics
                self.logger.info("\nOptimization Summary:")
                self.logger.info(f"Total completed trials: {len(results)}")
                self.logger.info(f"Task type: {self.config.task_type}")
                
                if len(results) > 0:
                    best_trial = df.iloc[0]  # First row after sorting
                    self.logger.info(f"Best trial number: {best_trial['trial_number']}")
                    self.logger.info(f"Best {self.optimization_metric}: {best_trial[self.optimization_metric]}")
                    
                    # Log other metrics for best trial
                    for metric in essential_metrics:
                        if metric != self.optimization_metric:
                            self.logger.info(f"Corresponding {metric}: {best_trial[metric]}")
                    
                    self.logger.info(f"Best trial hyperparameters:")
                    self.logger.info(f"  Learning rate: {best_trial['learning_rate']}")
                    self.logger.info(f"  Weight decay: {best_trial['weight_decay']}")
                    self.logger.info(f"  RGCN hidden features: {best_trial['rgcn_hidden_feats']}")
                    self.logger.info(f"  FFN hidden features: {best_trial['ffn_hidden_feats']}")
                    self.logger.info(f"  FFN dropout: {best_trial['ffn_dropout']}")
                    self.logger.info(f"  RGCN dropout: {best_trial['rgcn_dropout']}")
                    
            except Exception as e:
                self.logger.error(f"Error saving results to CSV: {str(e)}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                raise
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
        except Exception as e:
            self.logger.error(f"Error in save_results: {str(e)}")
            raise

    def _process_metrics(self, trainer) -> Dict[str, float]:
        """Extract and process metrics from trainer."""
        metrics = {}
        for metric in self.metrics:
            metric_value = trainer.callback_metrics.get(metric)
            if metric_value is not None:
                metrics[metric] = metric_value.item()
            else:
                # Set default values based on metric type
                if 'loss' in metric or 'mae' in metric or 'mse' in metric or 'rmse' in metric:
                    default = 0.0 if self.optimization_mode_optuna == 'maximize' else float('inf')
                elif 'r2' in metric:
                    default = float('-inf')
                else:
                    default = 0.0
                metrics[metric] = default

        return metrics

    def verify_best_metrics(self, study: optuna.Study) -> None:
        """Verify that the best metric corresponds to the best trial."""
        try:
            best_trial = study.best_trial
            best_metric = best_trial.value

            self.logger.info("\nFinal Verification:")
            self.logger.info(f"Task type: {'Classification' if self.config.classification else 'Regression'}")
            self.logger.info(f"Best trial number: {best_trial.number}")
            self.logger.info(f"Best {self.optimization_metric}: {best_metric:.6f}")

        except Exception as e:
            self.logger.error(f"Verification failed: {str(e)}")
            raise

    def _train_best_model(self, best_trial: optuna.Trial):
        """Train the best model based on the best trial's hyperparameters."""
        self.logger.info(f"Retraining best model for trial {best_trial.number}...")
        # Implement the logic to retrain the best model
        # This might involve creating the model with best_trial parameters and training it
        # For brevity, the implementation details are omitted
        pass

    def __del__(self):
        """Enhanced destructor"""
        try:
            self.cleanup()
            self.config.restore_original_configuration()
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error in destructor cleanup: {str(e)}")

def main():
    # Initialize Configuration
    config = Configuration()
    config.validate()
    config.set_seed(seed=42)

    max_retries = 3
    retry_delay = 60  # seconds

    for attempt in range(max_retries):
        try:
            optimizer = HyperparameterOptimization(config, n_trials=config.n_trials)
            optimizer.run_optimization()
            break  # If successful, break the retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
                raise

    logger.info("Hyperparameter optimization completed. Ready for statistical validation.")

if __name__ == "__main__":
    main()
