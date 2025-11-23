"""
PyTorch Dataset and DataModule for SMILES data
Handles loading, batching, and data management for training
"""

import torch
import pandas as pd
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional, Tuple, Dict, Any

from config import Config

logger = logging.getLogger(__name__)

class SMILESDataset(Dataset):
    """PyTorch Dataset for SMILES data"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, compound_ids: np.ndarray = None):
        """
        Initialize dataset
        
        Args:
            features: One-hot encoded SMILES features of shape (n_samples, seq_len, vocab_size)
            labels: Binary labels (0/1)
            compound_ids: Optional compound IDs for tracking
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.compound_ids = compound_ids
        
        logger.info(f"Dataset created with {len(self.features)} samples")
        logger.info(f"Features shape: {self.features.shape}")
        logger.info(f"Label distribution: {torch.bincount(self.labels.long())}")
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single item from dataset
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Dictionary containing features and labels
        """
        item = {
            'features': self.features[idx],
            'labels': self.labels[idx]
        }
        
        if self.compound_ids is not None:
            item['compound_id'] = self.compound_ids[idx]
            
        return item

class SMILESDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for SMILES data"""
    
    def __init__(
        self,
        features_path: str = None,
        labels_path: str = None,
        features: np.ndarray = None,
        labels_df: pd.DataFrame = None,
        batch_size: int = Config.BATCH_SIZE,
        num_workers: int = Config.NUM_WORKERS,
        pin_memory: bool = Config.PIN_MEMORY,
        use_class_weights: bool = Config.USE_CLASS_WEIGHTS
    ):
        """
        Initialize DataModule
        
        Args:
            features_path: Path to features numpy file
            labels_path: Path to labels CSV file
            features: Pre-loaded features array
            labels_df: Pre-loaded labels dataframe
            batch_size: Batch size for training
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            use_class_weights: Whether to compute class weights for imbalanced data
        """
        super().__init__()
        
        self.features_path = features_path or Config.FEATURES_PATH
        self.labels_path = labels_path or Config.LABELS_PATH
        self.features = features
        self.labels_df = labels_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_class_weights = use_class_weights
        
        # Will be populated in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None
        
        # Data statistics
        self.num_classes = 2
        self.vocab_size = None
        self.sequence_length = None
    
    def prepare_data(self):
        """Download or prepare data (called only on main process)"""
        # This method is typically used for downloading data
        # In our case, data should already be preprocessed
        pass
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage"""
        
        # Load data if not already provided
        if self.features is None:
            logger.info(f"Loading features from {self.features_path}")
            self.features = np.load(self.features_path)
        
        if self.labels_df is None:
            logger.info(f"Loading labels from {self.labels_path}")
            self.labels_df = pd.read_csv(self.labels_path)
        
        # Get data dimensions
        self.vocab_size = self.features.shape[-1]
        self.sequence_length = self.features.shape[1]
        
        logger.info(f"Data shape: {self.features.shape}")
        logger.info(f"Vocabulary size: {self.vocab_size}")
        logger.info(f"Sequence length: {self.sequence_length}")
        
        # Split data by group
        train_mask = self.labels_df['group'] == 'train'
        val_mask = self.labels_df['group'] == 'validation'
        test_mask = self.labels_df['group'] == 'test'
        
        # Create datasets
        if stage == "fit" or stage is None:
            train_features = self.features[train_mask]
            train_labels = self.labels_df.loc[train_mask, 'TARGET'].values
            train_ids = self.labels_df.loc[train_mask, 'COMPOUND_ID'].values
            
            val_features = self.features[val_mask]
            val_labels = self.labels_df.loc[val_mask, 'TARGET'].values
            val_ids = self.labels_df.loc[val_mask, 'COMPOUND_ID'].values
            
            self.train_dataset = SMILESDataset(train_features, train_labels, train_ids)
            self.val_dataset = SMILESDataset(val_features, val_labels, val_ids)
            
            # Compute class weights for training
            if self.use_class_weights:
                unique_labels = np.unique(train_labels)
                class_weights = compute_class_weight(
                    'balanced',
                    classes=unique_labels,
                    y=train_labels
                )
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
                logger.info(f"Class weights computed: {self.class_weights}")
            
            logger.info(f"Training set: {len(self.train_dataset)} samples")
            logger.info(f"Validation set: {len(self.val_dataset)} samples")
        
        if stage == "test" or stage is None:
            test_features = self.features[test_mask]
            test_labels = self.labels_df.loc[test_mask, 'TARGET'].values
            test_ids = self.labels_df.loc[test_mask, 'COMPOUND_ID'].values
            
            self.test_dataset = SMILESDataset(test_features, test_labels, test_ids)
            logger.info(f"Test set: {len(self.test_dataset)} samples")
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  # Drop last incomplete batch for consistent training
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction data loader (same as test)"""
        return self.test_dataloader()
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {
            'vocab_size': self.vocab_size,
            'sequence_length': self.sequence_length,
            'num_classes': self.num_classes,
        }
        
        if self.train_dataset:
            stats['train_size'] = len(self.train_dataset)
            train_labels = self.train_dataset.labels.numpy()
            stats['train_class_distribution'] = {
                str(int(label)): int(count) 
                for label, count in zip(*np.unique(train_labels, return_counts=True))
            }
        
        if self.val_dataset:
            stats['val_size'] = len(self.val_dataset)
            val_labels = self.val_dataset.labels.numpy()
            stats['val_class_distribution'] = {
                str(int(label)): int(count) 
                for label, count in zip(*np.unique(val_labels, return_counts=True))
            }
        
        if self.test_dataset:
            stats['test_size'] = len(self.test_dataset)
            test_labels = self.test_dataset.labels.numpy()
            stats['test_class_distribution'] = {
                str(int(label)): int(count) 
                for label, count in zip(*np.unique(test_labels, return_counts=True))
            }
        
        if self.class_weights is not None:
            stats['class_weights'] = self.class_weights.tolist()
        
        return stats

class CrossValidationDataModule:
    """Data module for cross-validation splits"""
    
    def __init__(
        self,
        features: np.ndarray,
        labels_df: pd.DataFrame,
        batch_size: int = Config.BATCH_SIZE,
        num_workers: int = Config.NUM_WORKERS,
        pin_memory: bool = Config.PIN_MEMORY
    ):
        """
        Initialize CV data module
        
        Args:
            features: Full features array
            labels_df: Full labels dataframe
            batch_size: Batch size for training
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
        """
        self.features = features
        self.labels_df = labels_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Extract only training and validation data for CV
        train_val_mask = self.labels_df['group'].isin(['train', 'validation'])
        self.cv_features = self.features[train_val_mask]
        self.cv_labels_df = self.labels_df[train_val_mask].reset_index(drop=True)
        
        logger.info(f"CV data prepared: {len(self.cv_features)} samples")
    
    def get_fold_dataloaders(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for a specific CV fold
        
        Args:
            train_indices: Indices for training data
            val_indices: Indices for validation data
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create training dataset
        train_features = self.cv_features[train_indices]
        train_labels = self.cv_labels_df.loc[train_indices, 'TARGET'].values
        train_ids = self.cv_labels_df.loc[train_indices, 'COMPOUND_ID'].values
        train_dataset = SMILESDataset(train_features, train_labels, train_ids)
        
        # Create validation dataset
        val_features = self.cv_features[val_indices]
        val_labels = self.cv_labels_df.loc[val_indices, 'TARGET'].values
        val_ids = self.cv_labels_df.loc[val_indices, 'COMPOUND_ID'].values
        val_dataset = SMILESDataset(val_features, val_labels, val_ids)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return train_loader, val_loader
    
    def get_class_weights(self, train_indices: np.ndarray) -> torch.Tensor:
        """
        Compute class weights for a specific fold
        
        Args:
            train_indices: Training indices for this fold
            
        Returns:
            Class weights tensor
        """
        train_labels = self.cv_labels_df.loc[train_indices, 'TARGET'].values
        unique_labels = np.unique(train_labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=train_labels
        )
        return torch.tensor(class_weights, dtype=torch.float32)

def test_datamodule():
    """Test function for the data module"""
    logging.basicConfig(level=logging.INFO, format=Config.LOG_FORMAT)
    
    # Create dummy data for testing
    n_samples = 1000
    seq_len = 50
    vocab_size = 65
    
    features = np.random.rand(n_samples, seq_len, vocab_size).astype(np.float32)
    
    labels_df = pd.DataFrame({
        'COMPOUND_ID': [f'COMP_{i}' for i in range(n_samples)],
        'TARGET': np.random.choice([0, 1], n_samples),
        'group': np.random.choice(['train', 'validation', 'test'], n_samples)
    })
    
    # Test DataModule
    dm = SMILESDataModule(features=features, labels_df=labels_df, batch_size=32)
    dm.setup()
    
    # Test data loaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    
    logger.info("DataModule test completed successfully")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Test batch
    batch = next(iter(train_loader))
    logger.info(f"Batch features shape: {batch['features'].shape}")
    logger.info(f"Batch labels shape: {batch['labels'].shape}")

if __name__ == "__main__":
    test_datamodule()