"""
CNN architecture for SMILES binary classification
PyTorch Lightning implementation with configurable hyperparameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall, Specificity, AUROC, MatthewsCorrCoef
import logging
from typing import Dict, Any, Optional

from config import Config

# Enable Tensor Core optimization for faster matrix operations on V100/A100 GPUs
# This can provide 1.5-2x speedup for large models during long training runs
torch.set_float32_matmul_precision('medium')

logger = logging.getLogger(__name__)

class SMILESCNNModel(pl.LightningModule):
    """CNN model for SMILES binary classification"""
    
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        layers: int = 2,
        filters: int = 128,
        kernel_size: int = 5,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Initialize CNN model
        
        Args:
            vocab_size: Size of character vocabulary
            sequence_length: Maximum sequence length
            layers: Number of CNN layers (1-3)
            filters: Number of filters in CNN layers
            kernel_size: Kernel size for convolution
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            class_weights: Optional class weights for imbalanced data
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.layers = layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Set up loss function with class weights
        if class_weights is not None:
            pos_weight = class_weights[1] / class_weights[0]  # Weight for positive class
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Build model architecture
        self._build_model()
        
        # Initialize metrics
        self._init_metrics()
        
        logger.info(f"Model initialized with parameters:")
        logger.info(f"  Vocab size: {vocab_size}")
        logger.info(f"  Sequence length: {sequence_length}")
        logger.info(f"  Layers: {layers}")
        logger.info(f"  Tensor Core optimization: {'enabled' if torch.get_float32_matmul_precision() == 'medium' else 'disabled'}")
        logger.info(f"  Filters: {filters}")
        logger.info(f"  Kernel size: {kernel_size}")
        logger.info(f"  Dropout: {dropout}")
        logger.info(f"  Learning rate: {learning_rate}")
    
    def _build_model(self):
        """Build the CNN architecture"""
        
        # CNN layers
        conv_layers = []
        
        # First layer: input channels = vocab_size
        conv_layers.append(
            nn.Conv1d(
                in_channels=self.vocab_size,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2
            )
        )
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Dropout(self.dropout))
        
        # Additional layers if specified
        for i in range(1, self.layers):
            conv_layers.append(
                nn.Conv1d(
                    in_channels=self.filters,
                    out_channels=self.filters,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(self.dropout))
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layer
        self.output_layer = nn.Linear(self.filters, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def _init_metrics(self):
        """Initialize torchmetrics"""
        # Training metrics
        self.train_accuracy = Accuracy(task='binary')
        self.train_f1 = F1Score(task='binary')
        self.train_precision = Precision(task='binary')
        self.train_recall = Recall(task='binary')
        self.train_specificity = Specificity(task='binary')
        self.train_auroc = AUROC(task='binary')
        self.train_mcc = MatthewsCorrCoef(task='binary')
        
        # Validation metrics
        self.val_accuracy = Accuracy(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.val_precision = Precision(task='binary')
        self.val_recall = Recall(task='binary')
        self.val_specificity = Specificity(task='binary')
        self.val_auroc = AUROC(task='binary')
        self.val_mcc = MatthewsCorrCoef(task='binary')
        
        # Test metrics
        self.test_accuracy = Accuracy(task='binary')
        self.test_f1 = F1Score(task='binary')
        self.test_precision = Precision(task='binary')
        self.test_recall = Recall(task='binary')
        self.test_specificity = Specificity(task='binary')
        self.test_auroc = AUROC(task='binary')
        self.test_mcc = MatthewsCorrCoef(task='binary')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, vocab_size)
            
        Returns:
            Output logits of shape (batch_size, 1)
        """
        # Transpose input for Conv1d: (batch, vocab_size, seq_len)
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        x = self.conv_layers(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # Shape: (batch, filters, 1)
        x = x.squeeze(-1)  # Shape: (batch, filters)
        
        # Output layer
        logits = self.output_layer(x)  # Shape: (batch, 1)
        
        return logits.squeeze(-1)  # Shape: (batch,)
    
    def _compute_loss_and_metrics(self, batch, stage: str):
        """Compute loss and metrics for a batch"""
        features = batch['features']
        labels = batch['labels']
        
        # Forward pass
        logits = self(features)
        
        # Compute loss
        loss = self.loss_fn(logits, labels)
        
        # Get predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        # Update metrics based on stage
        if stage == 'train':
            self.train_accuracy(preds, labels.int())
            self.train_f1(preds, labels.int())
            self.train_precision(preds, labels.int())
            self.train_recall(preds, labels.int())
            self.train_specificity(preds, labels.int())
            self.train_auroc(probs, labels.int())
            self.train_mcc(preds, labels.int())
        elif stage == 'val':
            self.val_accuracy(preds, labels.int())
            self.val_f1(preds, labels.int())
            self.val_precision(preds, labels.int())
            self.val_recall(preds, labels.int())
            self.val_specificity(preds, labels.int())
            self.val_auroc(probs, labels.int())
            self.val_mcc(preds, labels.int())
        elif stage == 'test':
            self.test_accuracy(preds, labels.int())
            self.test_f1(preds, labels.int())
            self.test_precision(preds, labels.int())
            self.test_recall(preds, labels.int())
            self.test_specificity(preds, labels.int())
            self.test_auroc(probs, labels.int())
            self.test_mcc(preds, labels.int())
        
        return loss, probs, preds
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        loss, probs, preds = self._compute_loss_and_metrics(batch, 'train')
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Log training metrics
        self.log('train_accuracy_epoch', self.train_accuracy.compute())
        self.log('train_f1_epoch', self.train_f1.compute())
        self.log('train_precision_epoch', self.train_precision.compute())
        self.log('train_recall_epoch', self.train_recall.compute())
        self.log('train_specificity_epoch', self.train_specificity.compute())
        self.log('train_auroc_epoch', self.train_auroc.compute())
        self.log('train_mcc_epoch', self.train_mcc.compute())
        
        # Reset metrics
        self.train_accuracy.reset()
        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_specificity.reset()
        self.train_auroc.reset()
        self.train_mcc.reset()
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss, probs, preds = self._compute_loss_and_metrics(batch, 'val')
        
        # Log loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Log validation metrics
        self.log('val_accuracy', self.val_accuracy.compute(), prog_bar=True)
        self.log('val_f1', self.val_f1.compute())
        self.log('val_precision', self.val_precision.compute())
        self.log('val_recall', self.val_recall.compute())
        self.log('val_specificity', self.val_specificity.compute())
        self.log('val_auroc', self.val_auroc.compute(), prog_bar=True)
        self.log('val_mcc', self.val_mcc.compute())
        
        # Reset metrics
        self.val_accuracy.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_specificity.reset()
        self.val_auroc.reset()
        self.val_mcc.reset()
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        loss, probs, preds = self._compute_loss_and_metrics(batch, 'test')
        
        # Log loss
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        return loss
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch"""
        # Log test metrics
        metrics = {
            'test_accuracy': self.test_accuracy.compute(),
            'test_f1': self.test_f1.compute(),
            'test_precision': self.test_precision.compute(),
            'test_recall': self.test_recall.compute(),
            'test_specificity': self.test_specificity.compute(),
            'test_auroc': self.test_auroc.compute(),
            'test_mcc': self.test_mcc.compute()
        }
        
        for name, value in metrics.items():
            self.log(name, value)
        
        # Reset metrics
        self.test_accuracy.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_specificity.reset()
        self.test_auroc.reset()
        self.test_mcc.reset()
        
        return metrics
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step"""
        features = batch['features']
        logits = self(features)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        result = {
            'logits': logits,
            'probabilities': probs,
            'predictions': preds
        }
        
        if 'compound_id' in batch:
            result['compound_ids'] = batch['compound_id']
        
        return result
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5  # Small L2 regularization
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auroc',
                'frequency': 1
            }
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = {
            'architecture': 'CNN',
            'layers': self.layers,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'vocab_size': self.vocab_size,
            'sequence_length': self.sequence_length
        }
        
        return summary

def create_model_from_trial(trial, vocab_size: int, sequence_length: int, 
                          class_weights: Optional[torch.Tensor] = None) -> SMILESCNNModel:
    """
    Create model instance from Optuna trial
    
    Args:
        trial: Optuna trial object
        vocab_size: Vocabulary size
        sequence_length: Maximum sequence length
        class_weights: Optional class weights
        
    Returns:
        Configured model instance
    """
    # Sample hyperparameters
    layers = trial.suggest_categorical('layers', Config.HYPERPARAMETER_SPACE['layers'])
    filters = trial.suggest_categorical('filters', Config.HYPERPARAMETER_SPACE['filters'])
    kernel_size = trial.suggest_categorical('kernel_size', Config.HYPERPARAMETER_SPACE['kernel_size'])
    dropout = trial.suggest_float('dropout', 
                                 Config.HYPERPARAMETER_SPACE['dropout']['low'],
                                 Config.HYPERPARAMETER_SPACE['dropout']['high'])
    learning_rate = trial.suggest_float('learning_rate',
                                       Config.HYPERPARAMETER_SPACE['learning_rate']['low'],
                                       Config.HYPERPARAMETER_SPACE['learning_rate']['high'],
                                       log=Config.HYPERPARAMETER_SPACE['learning_rate']['log'])
    
    model = SMILESCNNModel(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        layers=layers,
        filters=filters,
        kernel_size=kernel_size,
        dropout=dropout,
        learning_rate=learning_rate,
        class_weights=class_weights
    )
    
    return model

def test_model():
    """Test function for the model"""
    logging.basicConfig(level=logging.INFO, format=Config.LOG_FORMAT)
    
    # Test parameters
    vocab_size = 65
    sequence_length = 100
    batch_size = 32
    
    # Create dummy data
    features = torch.randn(batch_size, sequence_length, vocab_size)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    # Create model
    model = SMILESCNNModel(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        layers=2,
        filters=128,
        kernel_size=5,
        dropout=0.3,
        learning_rate=1e-3
    )
    
    # Test forward pass
    with torch.no_grad():
        logits = model(features)
        logger.info(f"Input shape: {features.shape}")
        logger.info(f"Output shape: {logits.shape}")
        logger.info(f"Output range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Test training step
    batch = {'features': features, 'labels': labels}
    loss = model.training_step(batch, 0)
    logger.info(f"Training loss: {loss:.4f}")
    
    # Print model summary
    summary = model.get_model_summary()
    logger.info("Model Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Model test completed successfully")

if __name__ == "__main__":
    test_model()