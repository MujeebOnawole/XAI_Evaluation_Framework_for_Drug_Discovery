"""
Data preprocessing module for SMILES augmentation and encoding
Handles SMILES cleaning, augmentation, and one-hot encoding
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import List, Tuple, Dict, Set
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
import warnings

from config import Config

# Suppress RDKit warnings
warnings.filterwarnings('ignore')
logging.getLogger('rdkit').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

class SMILESPreprocessor:
    """Handles SMILES processing, augmentation, and encoding"""
    
    def __init__(self):
        self.char_to_idx = Config.char_to_idx()
        self.idx_to_char = Config.idx_to_char()
        self.vocab_size = Config.get_vocab_size()
        self.max_length = None
        
    def is_valid_molecule(self, smiles: str) -> bool:
        """
        Check if SMILES represents a valid molecule meeting filtering criteria
        
        Args:
            smiles: SMILES string
            
        Returns:
            True if molecule is valid, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Check molecular weight
            mw = Descriptors.MolWt(mol)
            if mw > Config.MAX_MOLECULAR_WEIGHT:
                return False
                
            # Check for carbon atoms (exclude inorganic compounds)
            has_carbon = any(atom.GetSymbol() == 'C' for atom in mol.GetAtoms())
            if not has_carbon:
                return False
                
            # Check if it's a mixture (contains '.')
            if '.' in smiles:
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Error validating SMILES {smiles}: {e}")
            return False
    
    def canonicalize_smiles(self, smiles: str) -> str:
        """
        Convert SMILES to canonical form
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Canonical SMILES string or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            logger.warning(f"Error canonicalizing SMILES {smiles}: {e}")
            return None
    
    def generate_smart_augmentation(self, smiles: str, target_count: int = 3) -> List[str]:
        """
        Generate exactly 3 unique SMILES variants per molecule
        
        Args:
            smiles: Original SMILES string
            target_count: Number of variants to generate (default: 3)
            
        Returns:
            List of exactly target_count unique SMILES variants
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES for augmentation: {smiles}")
                return [smiles] * target_count  # Return original if invalid
            
            variants = set()
            
            # Always include original SMILES
            variants.add(smiles)
            
            # Add canonical SMILES
            canonical = Chem.MolToSmiles(mol, canonical=True)
            variants.add(canonical)
            
            # Generate random variants until we have exactly target_count unique variants
            max_attempts = 50  # Prevent infinite loop
            attempts = 0
            
            while len(variants) < target_count and attempts < max_attempts:
                try:
                    random_smiles = Chem.MolToSmiles(mol, doRandom=True)
                    if random_smiles:  # Only add if valid
                        variants.add(random_smiles)
                except Exception as e:
                    logger.warning(f"Error generating random SMILES: {e}")
                attempts += 1
            
            # Convert to list and ensure exactly target_count elements
            result = list(variants)[:target_count]
            
            # If we couldn't generate enough unique variants, pad with existing ones
            while len(result) < target_count:
                result.append(result[0])  # Use original SMILES as fallback
                
            return result
            
        except Exception as e:
            logger.error(f"Error in smart augmentation for {smiles}: {e}")
            return [smiles] * target_count
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataset by removing invalid molecules and canonicalizing SMILES
        
        Args:
            df: Input dataframe with columns ['COMPOUND_ID', 'PROCESSED_SMILES', 'TARGET', 'group']
            
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Starting dataset cleaning with {len(df)} compounds")
        
        # Remove rows with missing SMILES
        df = df.dropna(subset=['PROCESSED_SMILES'])
        logger.info(f"After removing missing SMILES: {len(df)} compounds")
        
        # Validate molecules
        valid_mask = df['PROCESSED_SMILES'].apply(self.is_valid_molecule)
        df = df[valid_mask].copy()
        logger.info(f"After filtering invalid molecules: {len(df)} compounds")
        
        # Canonicalize SMILES
        df['CANONICAL_SMILES'] = df['PROCESSED_SMILES'].apply(self.canonicalize_smiles)
        
        # Remove rows where canonicalization failed
        df = df.dropna(subset=['CANONICAL_SMILES'])
        logger.info(f"After canonicalization: {len(df)} compounds")
        
        # Remove duplicates based on canonical SMILES
        initial_count = len(df)
        df = df.drop_duplicates(subset=['CANONICAL_SMILES'])
        logger.info(f"Removed {initial_count - len(df)} duplicate compounds")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        logger.info(f"Dataset cleaning completed. Final dataset: {len(df)} compounds")
        logger.info(f"Class distribution: {df['TARGET'].value_counts().to_dict()}")
        
        return df
    
    def augment_smiles(self, df: pd.DataFrame, augment_test: bool = False) -> pd.DataFrame:
        """
        Augment SMILES data for training/validation (not for test set by default)
        
        Args:
            df: Input dataframe
            augment_test: Whether to augment test set (default: False)
            
        Returns:
            Augmented dataframe
        """
        augmented_data = []
        
        for _, row in df.iterrows():
            smiles = row['PROCESSED_SMILES']
            
            # For test set, keep original SMILES only unless explicitly requested
            if row['group'] == 'test' and not augment_test:
                augmented_data.append({
                    'COMPOUND_ID': row['COMPOUND_ID'],
                    'ORIGINAL_SMILES': smiles,
                    'AUGMENTED_SMILES': smiles,
                    'TARGET': row['TARGET'],
                    'group': row['group'],
                    'augmentation_id': 0
                })
            else:
                # Generate augmented variants for train/val
                variants = self.generate_smart_augmentation(smiles, Config.AUGMENTATION_COUNT)
                
                for aug_id, variant in enumerate(variants):
                    augmented_data.append({
                        'COMPOUND_ID': row['COMPOUND_ID'],
                        'ORIGINAL_SMILES': smiles,
                        'AUGMENTED_SMILES': variant,
                        'TARGET': row['TARGET'],
                        'group': row['group'],
                        'augmentation_id': aug_id
                    })
        
        augmented_df = pd.DataFrame(augmented_data)
        
        logger.info(f"Augmentation completed:")
        logger.info(f"Original dataset: {len(df)} compounds")
        logger.info(f"Augmented dataset: {len(augmented_df)} samples")
        
        # Log augmentation statistics by group
        for group in augmented_df['group'].unique():
            group_data = augmented_df[augmented_df['group'] == group]
            logger.info(f"{group} set: {len(group_data)} samples")
        
        return augmented_df
    
    def encode_smiles(self, smiles: str) -> List[int]:
        """
        Convert SMILES string to list of character indices
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of character indices
        """
        try:
            return [self.char_to_idx.get(char, 0) for char in smiles]  # Use 0 for unknown chars
        except Exception as e:
            logger.error(f"Error encoding SMILES {smiles}: {e}")
            return [0]  # Return single padding character if error
    
    def pad_sequences(self, sequences: List[List[int]], max_length: int = None) -> np.ndarray:
        """
        Pad sequences to same length
        
        Args:
            sequences: List of encoded sequences
            max_length: Maximum length to pad to (if None, use longest sequence)
            
        Returns:
            Padded sequences as numpy array
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        self.max_length = max_length
        
        padded = np.zeros((len(sequences), max_length), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            padded[i, :length] = seq[:length]
        
        return padded
    
    def one_hot_encode(self, padded_sequences: np.ndarray) -> np.ndarray:
        """
        Convert padded sequences to one-hot encoding
        
        Args:
            padded_sequences: Padded sequences array
            
        Returns:
            One-hot encoded array of shape (n_samples, seq_len, vocab_size)
        """
        n_samples, seq_len = padded_sequences.shape
        one_hot = np.zeros((n_samples, seq_len, self.vocab_size), dtype=np.float32)
        
        for i in range(n_samples):
            for j in range(seq_len):
                char_idx = padded_sequences[i, j]
                if 0 <= char_idx < self.vocab_size:
                    one_hot[i, j, char_idx] = 1.0
        
        return one_hot
    
    def split_data(self, df: pd.DataFrame, stratify_col: str = 'TARGET', 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets with stratification
        
        Args:
            df: Input dataframe
            stratify_col: Column to use for stratification
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=Config.TEST_RATIO,
            stratify=df[stratify_col],
            random_state=random_state
        )
        
        # Second split: separate train and validation
        val_ratio_adjusted = Config.VAL_RATIO / (Config.TRAIN_RATIO + Config.VAL_RATIO)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            stratify=train_val_df[stratify_col],
            random_state=random_state
        )
        
        # Add group labels
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        train_df['group'] = 'train'
        val_df['group'] = 'validation'
        test_df['group'] = 'test'
        
        logger.info(f"Data split completed:")
        logger.info(f"Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def process_dataset(self, csv_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Complete processing pipeline: load, clean, split, augment, and encode data
        
        Args:
            csv_path: Path to input CSV file
            
        Returns:
            Tuple of (features_array, labels_dataframe)
        """
        logger.info("Starting complete dataset processing pipeline")
        
        # Load data
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} compounds")
        
        # Clean data
        df = self.clean_dataset(df)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # Combine splits
        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        # Augment SMILES (test set won't be augmented)
        augmented_df = self.augment_smiles(combined_df)
        
        # Encode SMILES
        logger.info("Encoding SMILES sequences")
        encoded_sequences = [
            self.encode_smiles(smiles) 
            for smiles in augmented_df['AUGMENTED_SMILES']
        ]
        
        # Pad sequences
        logger.info("Padding sequences")
        padded_sequences = self.pad_sequences(encoded_sequences)
        
        # One-hot encode
        logger.info("One-hot encoding sequences")
        features = self.one_hot_encode(padded_sequences)
        
        logger.info(f"Final features shape: {features.shape}")
        logger.info(f"Max sequence length: {self.max_length}")
        
        # Save vocabulary information
        vocab_info = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length
        }
        
        with open(Config.VOCAB_PATH, 'w') as f:
            json.dump(vocab_info, f, indent=2)
        
        logger.info("Dataset processing completed successfully")
        
        return features, augmented_df

def main():
    """Example usage of the SMILESPreprocessor"""
    logging.basicConfig(level=logging.INFO, format=Config.LOG_FORMAT)
    
    # Initialize preprocessor
    preprocessor = SMILESPreprocessor()
    
    # Process dataset
    features, labels_df = preprocessor.process_dataset(Config.DATA_PATH)
    
    # Save processed data
    logger.info("Saving processed data")
    np.save(Config.FEATURES_PATH, features)
    labels_df.to_csv(Config.LABELS_PATH, index=False)
    
    logger.info("Data preprocessing completed and saved")

if __name__ == "__main__":
    main()