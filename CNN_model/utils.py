"""
Utility functions for the CNN-based SMILES augmentation system
Includes seed setting, logging configuration, and helper functions
"""

import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
import logging
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from rdkit import Chem
import warnings

from config import Config

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('rdkit').setLevel(logging.ERROR)

def set_all_seeds(seed: int = Config.RANDOM_SEED):
    """
    Set seeds for reproducibility across all libraries
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"All seeds set to {seed}")

def setup_logging(log_level: str = Config.LOG_LEVEL, 
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    
    # Configure logging
    log_format = Config.LOG_FORMAT
    log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(Config.LOGS_DIR, f"smiles_cnn_{timestamp}.log")
    
    handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=log_level_obj,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Level: {log_level}, File: {log_file}")
    
    return logger

def get_device() -> torch.device:
    """
    Get the best available device (CUDA, MPS, or CPU)
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        logging.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():  # Apple Silicon
        device = torch.device('mps')
        logging.info("Using MPS (Apple Silicon) device")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU device")
    
    return device

def save_results(results: Dict[str, Any], filepath: str):
    """
    Save results to JSON file
    
    Args:
        results: Dictionary of results to save
        filepath: Output file path
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert any numpy/torch objects to native Python types
    serializable_results = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logging.info(f"Results saved to {filepath}")

def convert_to_serializable(obj):
    """
    Convert numpy/torch objects to JSON-serializable types
    
    Args:
        obj: Object to convert
        
    Returns:
        Serializable object
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)

def load_checkpoint(checkpoint_path: str, model_class, **model_kwargs):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_class: Model class to instantiate
        **model_kwargs: Additional model arguments
        
    Returns:
        Loaded model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = model_class.load_from_checkpoint(checkpoint_path, **model_kwargs)
    logging.info(f"Model loaded from checkpoint: {checkpoint_path}")
    
    return model

def save_dataframe_with_timestamp(df: pd.DataFrame, base_path: str, 
                                 include_timestamp: bool = True) -> str:
    """
    Save dataframe with optional timestamp in filename
    
    Args:
        df: DataFrame to save
        base_path: Base file path
        include_timestamp: Whether to include timestamp in filename
        
    Returns:
        Actual save path
    """
    if include_timestamp:
        path_obj = Path(base_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = path_obj.parent / f"{path_obj.stem}_{timestamp}{path_obj.suffix}"
    else:
        save_path = base_path
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df.to_csv(save_path, index=False)
    logging.info(f"DataFrame saved to {save_path}")
    
    return str(save_path)

def calculate_metrics_summary(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate summary statistics for a list of metrics
    
    Args:
        metrics_list: List of metrics dictionaries
        
    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not metrics_list:
        return {}
    
    # Get all metric names
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())
    
    summary = {}
    for metric in all_metrics:
        values = [m.get(metric, np.nan) for m in metrics_list if metric in m]
        if values:
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
    
    return summary

def create_model_checkpoint_callback(monitor: str = 'val_auroc', 
                                   mode: str = 'max',
                                   save_top_k: int = 1,
                                   save_last: bool = True) -> pl.callbacks.ModelCheckpoint:
    """
    Create ModelCheckpoint callback for PyTorch Lightning
    
    Args:
        monitor: Metric to monitor
        mode: 'min' or 'max'
        save_top_k: Number of best models to save
        save_last: Whether to save last checkpoint
        
    Returns:
        ModelCheckpoint callback
    """
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Config.MODELS_DIR,
        filename='{epoch:02d}-{val_auroc:.4f}',
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=save_last,
        verbose=True
    )
    
    return checkpoint_callback

def create_early_stopping_callback(monitor: str = 'val_auroc',
                                  mode: str = 'max',
                                  patience: int = Config.EARLY_STOPPING_PATIENCE) -> pl.callbacks.EarlyStopping:
    """
    Create EarlyStopping callback for PyTorch Lightning
    
    Args:
        monitor: Metric to monitor
        mode: 'min' or 'max'
        patience: Number of epochs to wait
        
    Returns:
        EarlyStopping callback
    """
    early_stopping = pl.callbacks.EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        verbose=True
    )
    
    return early_stopping

def explainability_test(model, preprocessor, test_smiles: str = Config.TEST_MOLECULE_SMILES):
    """
    Test model explainability by comparing predictions for same molecule with different SMILES
    
    Args:
        model: Trained model
        preprocessor: SMILES preprocessor
        test_smiles: SMILES string to test
        
    Returns:
        Dictionary with test results
    """
    logging.info(f"Running explainability test with molecule: {test_smiles}")
    
    try:
        # Generate different SMILES representations
        variants = preprocessor.generate_smart_augmentation(test_smiles, target_count=5)
        
        results = {}
        for i, smiles in enumerate(variants):
            # Encode SMILES
            encoded = preprocessor.encode_smiles(smiles)
            padded = preprocessor.pad_sequences([encoded], max_length=preprocessor.max_length)
            one_hot = preprocessor.one_hot_encode(padded)
            
            # Convert to tensor
            features = torch.tensor(one_hot, dtype=torch.float32)
            
            # Get prediction
            model.eval()
            with torch.no_grad():
                logits = model(features)
                prob = torch.sigmoid(logits).item()
                pred = (prob > 0.5).item()
            
            results[f'variant_{i}'] = {
                'smiles': smiles,
                'probability': prob,
                'prediction': pred,
                'logit': logits.item()
            }
        
        # Calculate statistics
        probs = [r['probability'] for r in results.values()]
        results['statistics'] = {
            'mean_probability': np.mean(probs),
            'std_probability': np.std(probs),
            'min_probability': np.min(probs),
            'max_probability': np.max(probs),
            'probability_range': np.max(probs) - np.min(probs)
        }
        
        logging.info(f"Explainability test completed. Probability range: {results['statistics']['probability_range']:.4f}")
        
        return results
        
    except Exception as e:
        logging.error(f"Error in explainability test: {e}")
        return {'error': str(e)}

def print_system_info():
    """Print system information for debugging"""
    logging.info("=== SYSTEM INFORMATION ===")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"PyTorch Lightning version: {pl.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    logging.info(f"MPS available: {torch.backends.mps.is_available()}")
    logging.info(f"Number of CPU cores: {os.cpu_count()}")
    logging.info("========================")

def validate_input_data(csv_path: str) -> bool:
    """
    Validate input CSV data format
    
    Args:
        csv_path: Path to input CSV file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_columns = ['COMPOUND_ID', 'PROCESSED_SMILES', 'TARGET']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check data types and values
        if not df['TARGET'].isin([0, 1]).all():
            logging.error("TARGET column must contain only 0 and 1 values")
            return False
        
        # Check for missing values in critical columns
        for col in required_columns:
            if df[col].isnull().any():
                logging.warning(f"Found missing values in column: {col}")
        
        logging.info(f"Input data validation passed. Shape: {df.shape}")
        logging.info(f"Class distribution: {df['TARGET'].value_counts().to_dict()}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating input data: {e}")
        return False

class ProgressCallback(pl.Callback):
    """Custom callback for progress logging"""
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log progress at end of each epoch"""
        if trainer.current_epoch % 10 == 0:  # Log every 10 epochs
            metrics = trainer.callback_metrics
            val_auroc = metrics.get('val_auroc', 0)
            val_loss = metrics.get('val_loss', 0)
            logging.info(f"Epoch {trainer.current_epoch}: val_auroc={val_auroc:.4f}, val_loss={val_loss:.4f}")

def test_utils():
    """Test utility functions"""
    logger = setup_logging()
    
    # Test seed setting
    set_all_seeds(42)
    logger.info("Seeds set successfully")
    
    # Test device detection
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Test system info
    print_system_info()
    
    # Test serialization
    test_data = {
        'numpy_array': np.array([1, 2, 3]),
        'torch_tensor': torch.tensor([4, 5, 6]),
        'nested': {'value': np.float32(3.14)}
    }
    
    serializable = convert_to_serializable(test_data)
    logger.info(f"Serialization test: {serializable}")
    
    logger.info("Utils test completed successfully")

if __name__ == "__main__":
    test_utils()