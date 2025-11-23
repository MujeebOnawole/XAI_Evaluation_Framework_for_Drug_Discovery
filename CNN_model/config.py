"""
Configuration file for CNN-based SMILES augmentation system
Contains paths, hyperparameter ranges, and system settings
"""

import os
from typing import Dict, List, Any

class Config:
    # Data paths
    DATA_PATH = "data/compounds.csv"  # Input CSV with ['COMPOUND_ID', 'PROCESSED_SMILES', 'TARGET', 'group']
    OUTPUT_DIR = "outputs"
    MODELS_DIR = "models"
    LOGS_DIR = "logs"
    
    # Processed data outputs
    FEATURES_PATH = os.path.join(OUTPUT_DIR, "features.npy")
    LABELS_PATH = os.path.join(OUTPUT_DIR, "labels.csv")
    VOCAB_PATH = os.path.join(OUTPUT_DIR, "vocabulary.json")
    
    # Results outputs
    HYPEROPT_RESULTS_PATH = os.path.join(OUTPUT_DIR, "hyperparameter_results.csv")
    CV_RESULTS_PATH = os.path.join(OUTPUT_DIR, "cross_validation_results.csv")
    
    # Data preprocessing settings
    AUGMENTATION_COUNT = 3  # Exactly 3 unique SMILES variants per molecule
    MAX_MOLECULAR_WEIGHT = 700  # Da
    
    # Character vocabulary for SMILES encoding
    VOCAB_CHARS = [
        ' ', '#', '%', '(', ')', '+', '-', '.', '/', 
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '=', '@', 
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        '[', '\\', ']',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]
    
    # Data split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # Hyperparameter search space for Optuna
    HYPERPARAMETER_SPACE = {
        'layers': [1, 2, 3],
        'filters': [32, 64, 128, 256],
        'kernel_size': [3, 5, 7],
        'dropout': {'low': 0.1, 'high': 0.5},
        'learning_rate': {'low': 1e-5, 'high': 5e-3, 'log': True}
    }
    
    # Training settings
    N_TRIALS = 50  # Optuna optimization trials
    CV_FOLDS = 5   # 5x5 cross-validation
    CV_REPEATS = 5
    EARLY_STOPPING_PATIENCE = 10
    MAX_EPOCHS = 100
    BATCH_SIZE = 64
    
    # Model settings
    OPTIMIZER = 'Adam'
    LOSS_FUNCTION = 'BCEWithLogitsLoss'
    METRICS = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall', 'specificity', 'mcc']
    
    # System settings
    RANDOM_SEED = 42
    N_JOBS = -1  # Use all available cores
    DEVICE = 'cuda'  # Will fallback to 'cpu' if CUDA unavailable
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Memory optimization
    PIN_MEMORY = True
    NUM_WORKERS = 4
    
    # Class imbalance handling
    USE_CLASS_WEIGHTS = True
    POSITIVE_CLASS_RATIO = 0.67  # 67% active compounds
    
    # Explainability test molecule (ciprofloxacin)
    TEST_MOLECULE_SMILES = "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.OUTPUT_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def get_class_weights(cls):
        """Calculate class weights for imbalanced dataset"""
        pos_weight = (1 - cls.POSITIVE_CLASS_RATIO) / cls.POSITIVE_CLASS_RATIO
        return pos_weight
    
    @classmethod
    def get_vocab_size(cls):
        """Get vocabulary size"""
        return len(cls.VOCAB_CHARS)
    
    @classmethod
    def char_to_idx(cls):
        """Create character to index mapping"""
        return {char: idx for idx, char in enumerate(cls.VOCAB_CHARS)}
    
    @classmethod
    def idx_to_char(cls):
        """Create index to character mapping"""
        return {idx: char for idx, char in enumerate(cls.VOCAB_CHARS)}

# Create directories on import
Config.create_directories()