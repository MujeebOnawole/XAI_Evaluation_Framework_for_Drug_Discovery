"""
Main pipeline for CNN-based SMILES augmentation system
Runs the complete workflow: preprocessing -> optimization -> cross-validation
"""

import os
import sys
import argparse
import logging
import json
import torch
from datetime import datetime
from typing import Optional

# Enable Tensor Core optimization for faster matrix operations on V100/A100 GPUs
# This can provide 1.5-2x speedup for large models during long training runs
torch.set_float32_matmul_precision('medium')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_preprocessing import SMILESPreprocessor
from hyperparameter_opt import run_hyperparameter_optimization
from cross_validation import run_cross_validation_experiment
from utils import (
    setup_logging, set_all_seeds, print_system_info, 
    validate_input_data, save_results
)

def run_complete_pipeline(
    data_path: str,
    run_preprocessing: bool = True,
    run_hyperopt: bool = True,
    run_cv: bool = True,
    n_trials: int = Config.N_TRIALS,
    save_models: bool = True,
    output_dir: Optional[str] = None
) -> dict:
    """
    Run the complete CNN-based SMILES augmentation pipeline
    
    Args:
        data_path: Path to input CSV file
        run_preprocessing: Whether to run data preprocessing
        run_hyperopt: Whether to run hyperparameter optimization
        run_cv: Whether to run cross-validation
        n_trials: Number of hyperparameter optimization trials
        save_models: Whether to save CV models
        output_dir: Optional output directory override
        
    Returns:
        Dictionary with all results
    """
    logger = logging.getLogger(__name__)
    
    # Override output directory if provided
    if output_dir:
        Config.OUTPUT_DIR = output_dir
        Config.MODELS_DIR = os.path.join(output_dir, 'models')
        Config.LOGS_DIR = os.path.join(output_dir, 'logs')
        Config.create_directories()
    
    # Initialize results dictionary
    pipeline_results = {
        'pipeline_start_time': datetime.now().isoformat(),
        'config': {
            'data_path': data_path,
            'run_preprocessing': run_preprocessing,
            'run_hyperopt': run_hyperopt,
            'run_cv': run_cv,
            'n_trials': n_trials,
            'save_models': save_models
        }
    }
    
    logger.info("=" * 60)
    logger.info("CNN-BASED SMILES AUGMENTATION PIPELINE")
    logger.info("=" * 60)
    
    # Print system information
    print_system_info()
    
    # Validate input data
    logger.info("Validating input data...")
    if not validate_input_data(data_path):
        raise ValueError(f"Input data validation failed for {data_path}")
    
    # Phase 1: Data Preprocessing
    if run_preprocessing:
        logger.info("=" * 60)
        logger.info("PHASE 1: DATA PREPROCESSING & AUGMENTATION")
        logger.info("=" * 60)
        
        try:
            preprocessor = SMILESPreprocessor()
            features, labels_df = preprocessor.process_dataset(data_path)
            
            # Save processed data
            logger.info("Saving preprocessed data...")
            import numpy as np
            np.save(Config.FEATURES_PATH, features)
            labels_df.to_csv(Config.LABELS_PATH, index=False)
            
            preprocessing_results = {
                'status': 'completed',
                'features_shape': features.shape,
                'total_samples': len(labels_df),
                'vocab_size': preprocessor.vocab_size,
                'max_length': preprocessor.max_length,
                'class_distribution': labels_df['TARGET'].value_counts().to_dict(),
                'group_distribution': labels_df['group'].value_counts().to_dict()
            }
            
            pipeline_results['preprocessing'] = preprocessing_results
            logger.info("Data preprocessing completed successfully!")
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            pipeline_results['preprocessing'] = {'status': 'failed', 'error': str(e)}
            raise
    else:
        # Load existing preprocessed data
        logger.info("Loading existing preprocessed data...")
        import numpy as np
        features = np.load(Config.FEATURES_PATH)
        import pandas as pd
        labels_df = pd.read_csv(Config.LABELS_PATH)
        
        pipeline_results['preprocessing'] = {
            'status': 'skipped',
            'features_shape': features.shape,
            'total_samples': len(labels_df)
        }
    
    # Phase 2: Hyperparameter Optimization
    if run_hyperopt:
        logger.info("=" * 60)
        logger.info("PHASE 2: HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 60)
        
        try:
            hyperopt_results = run_hyperparameter_optimization(
                features=features,
                labels_df=labels_df,
                n_trials=n_trials
            )
            
            pipeline_results['hyperparameter_optimization'] = hyperopt_results
            best_hyperparameters = hyperopt_results['best_params']
            logger.info("Hyperparameter optimization completed successfully!")
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            pipeline_results['hyperparameter_optimization'] = {'status': 'failed', 'error': str(e)}
            raise
    else:
        # Load existing hyperparameter results or use defaults
        hyperopt_path = Config.HYPEROPT_RESULTS_PATH
        if os.path.exists(hyperopt_path):
            logger.info("Loading existing hyperparameter optimization results...")
            with open(hyperopt_path, 'r') as f:
                hyperopt_results = json.load(f)
            best_hyperparameters = hyperopt_results['best_params']
            pipeline_results['hyperparameter_optimization'] = {'status': 'loaded_existing'}
        else:
            logger.info("Using default hyperparameters...")
            best_hyperparameters = {
                'layers': 2,
                'filters': 128,
                'kernel_size': 5,
                'dropout': 0.3,
                'learning_rate': 1e-3
            }
            pipeline_results['hyperparameter_optimization'] = {
                'status': 'using_defaults',
                'best_params': best_hyperparameters
            }
    
    # Phase 3: Cross-Validation
    if run_cv:
        logger.info("=" * 60)
        logger.info("PHASE 3: CROSS-VALIDATION EVALUATION")
        logger.info("=" * 60)
        
        try:
            cv_results = run_cross_validation_experiment(
                features=features,
                labels_df=labels_df,
                best_hyperparameters=best_hyperparameters,
                save_models=save_models
            )
            
            pipeline_results['cross_validation'] = cv_results
            logger.info("Cross-validation completed successfully!")
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            pipeline_results['cross_validation'] = {'status': 'failed', 'error': str(e)}
            raise
    else:
        pipeline_results['cross_validation'] = {'status': 'skipped'}
    
    # Pipeline completion
    pipeline_results['pipeline_end_time'] = datetime.now().isoformat()
    pipeline_results['status'] = 'completed'
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    # Save complete pipeline results
    pipeline_results_path = os.path.join(Config.OUTPUT_DIR, 'pipeline_results.json')
    save_results(pipeline_results, pipeline_results_path)
    
    # Print summary
    print_pipeline_summary(pipeline_results)
    
    return pipeline_results

def print_pipeline_summary(results: dict):
    """Print a summary of pipeline results"""
    logger = logging.getLogger(__name__)
    
    logger.info("PIPELINE SUMMARY:")
    logger.info("-" * 40)
    
    # Preprocessing summary
    if 'preprocessing' in results:
        prep = results['preprocessing']
        if prep['status'] == 'completed':
            logger.info(f"Data preprocessing: ✓ COMPLETED")
            logger.info(f"  Total samples: {prep['total_samples']}")
            logger.info(f"  Features shape: {prep['features_shape']}")
            logger.info(f"  Vocabulary size: {prep['vocab_size']}")
            logger.info(f"  Max sequence length: {prep['max_length']}")
        else:
            logger.info(f"Data preprocessing: {prep['status'].upper()}")
    
    # Hyperparameter optimization summary
    if 'hyperparameter_optimization' in results:
        hyperopt = results['hyperparameter_optimization']
        if 'best_value' in hyperopt:
            logger.info(f"Hyperparameter optimization: ✓ COMPLETED")
            logger.info(f"  Best ROC-AUC: {hyperopt['best_value']:.4f}")
            logger.info(f"  Best parameters: {hyperopt['best_params']}")
        else:
            logger.info(f"Hyperparameter optimization: {hyperopt.get('status', 'UNKNOWN').upper()}")
    
    # Cross-validation summary
    if 'cross_validation' in results:
        cv = results['cross_validation']
        if 'summary_statistics' in cv:
            logger.info(f"Cross-validation: ✓ COMPLETED")
            stats = cv['summary_statistics']
            if 'auroc' in stats:
                auroc_stats = stats['auroc']
                logger.info(f"  ROC-AUC: {auroc_stats['mean']:.4f} ± {auroc_stats['std']:.4f}")
            if 'accuracy' in stats:
                acc_stats = stats['accuracy']
                logger.info(f"  Accuracy: {acc_stats['mean']:.4f} ± {acc_stats['std']:.4f}")
        else:
            logger.info(f"Cross-validation: {cv.get('status', 'UNKNOWN').upper()}")
    
    # Explainability test summary
    if 'cross_validation' in results and 'explainability_test' in results['cross_validation']:
        explainability = results['cross_validation']['explainability_test']
        if 'statistics' in explainability:
            stats = explainability['statistics']
            logger.info(f"Explainability test: ✓ COMPLETED")
            logger.info(f"  Probability range: {stats['probability_range']:.4f}")
        else:
            logger.info("Explainability test: FAILED")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="CNN-based SMILES Augmentation Pipeline for Binary Bioactivity Classification"
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        required=True,
        help='Path to input CSV file with columns: COMPOUND_ID, PROCESSED_SMILES, TARGET, group'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Output directory for results (default: outputs/)'
    )
    
    parser.add_argument(
        '--skip-preprocessing', 
        action='store_true',
        help='Skip data preprocessing (use existing processed data)'
    )
    
    parser.add_argument(
        '--skip-hyperopt', 
        action='store_true',
        help='Skip hyperparameter optimization (use existing results or defaults)'
    )
    
    parser.add_argument(
        '--skip-cv', 
        action='store_true',
        help='Skip cross-validation'
    )
    
    parser.add_argument(
        '--trials', 
        type=int, 
        default=Config.N_TRIALS,
        help=f'Number of hyperparameter optimization trials (default: {Config.N_TRIALS})'
    )
    
    parser.add_argument(
        '--no-save-models', 
        action='store_true',
        help='Do not save trained models (saves disk space)'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=Config.RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {Config.RANDOM_SEED})'
    )
    
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_level=args.log_level)
    
    # Log Tensor Core optimization status
    logger.info("=" * 60)
    logger.info("CNN-BASED SMILES AUGMENTATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Tensor Core optimization: {'enabled' if torch.get_float32_matmul_precision() == 'medium' else 'disabled'}")
    
    # Set seeds
    if args.seed != Config.RANDOM_SEED:
        Config.RANDOM_SEED = args.seed
    set_all_seeds(Config.RANDOM_SEED)
    
    try:
        # Run pipeline
        results = run_complete_pipeline(
            data_path=args.data,
            run_preprocessing=not args.skip_preprocessing,
            run_hyperopt=not args.skip_hyperopt,
            run_cv=not args.skip_cv,
            n_trials=args.trials,
            save_models=not args.no_save_models,
            output_dir=args.output
        )
        
        logger.info("Pipeline completed successfully! 🎉")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()