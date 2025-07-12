# src/utils/data_utils.py
"""Data handling utilities for the CHF project."""

import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
import logging

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seeds set to {seed}")


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device: Device string ('cuda', 'cpu', or None for auto)
        
    Returns:
        torch.device instance
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device_obj = torch.device(device)
    
    if device == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("Using CPU")
    
    return device_obj


def create_data_loader(data: pd.DataFrame,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 0,
                      pin_memory: bool = True) -> torch.utils.data.DataLoader:
    """
    Create PyTorch DataLoader from DataFrame.
    
    Args:
        data: DataFrame with features and target
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU transfer
        
    Returns:
        DataLoader instance
    """
    # Assume last column is target
    X = data.iloc[:, :-1].values.astype(np.float32)
    y = data.iloc[:, -1].values.astype(np.float32)
    
    # Create TensorDataset
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X),
        torch.tensor(y)
    )
    
    # Create DataLoader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available()
    )
    
    return loader


class DataPreprocessor:
    """Handle data preprocessing and scaling."""
    
    def __init__(self, 
                 scaler_type: str = 'standard',
                 feature_columns: Optional[List[str]] = None,
                 target_column: Optional[str] = None):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            feature_columns: List of feature column names
            target_column: Target column name
        """
        self.scaler_type = scaler_type
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Initialize scalers
        self.feature_scaler = self._create_scaler(scaler_type)
        self.target_scaler = self._create_scaler(scaler_type)
        
        self.is_fitted = False
    
    def _create_scaler(self, scaler_type: str):
        """Create scaler based on type."""
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit preprocessor on data.
        
        Args:
            data: Training data
            
        Returns:
            Self
        """
        # Identify columns
        if self.feature_columns is None:
            self.feature_columns = list(data.columns[:-1])
        if self.target_column is None:
            self.target_column = data.columns[-1]
        
        # Fit scalers
        X = data[self.feature_columns]
        y = data[self.target_column].values.reshape(-1, 1)
        
        self.feature_scaler.fit(X)
        self.target_scaler.fit(y)
        
        self.is_fitted = True
        logger.info(f"Preprocessor fitted on {len(data)} samples")
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Create copy
        transformed = data.copy()
        
        # Transform features
        if self.feature_columns:
            X_scaled = self.feature_scaler.transform(data[self.feature_columns])
            transformed[self.feature_columns] = X_scaled
        
        # Transform target if present
        if self.target_column and self.target_column in data.columns:
            y_scaled = self.target_scaler.transform(
                data[self.target_column].values.reshape(-1, 1)
            )
            transformed[self.target_column] = y_scaled.flatten()
        
        return transformed
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(data).transform(data)
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform target values.
        
        Args:
            y: Scaled target values
            
        Returns:
            Original scale target values
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted")
        
        y_reshaped = y.reshape(-1, 1)
        return self.target_scaler.inverse_transform(y_reshaped).flatten()
    
    def save(self, path: Path) -> None:
        """Save preprocessor state."""
        import pickle
        
        state = {
            'scaler_type': self.scaler_type,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'DataPreprocessor':
        """Load preprocessor from file."""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            scaler_type=state['scaler_type'],
            feature_columns=state['feature_columns'],
            target_column=state['target_column']
        )
        
        preprocessor.feature_scaler = state['feature_scaler']
        preprocessor.target_scaler = state['target_scaler']
        preprocessor.is_fitted = state['is_fitted']
        
        logger.info(f"Preprocessor loaded from {path}")
        
        return preprocessor


def create_cross_validation_splits(data: pd.DataFrame,
                                 n_splits: int = 5,
                                 stratify: bool = False,
                                 random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create cross-validation splits.
    
    Args:
        data: Input data
        n_splits: Number of splits
        stratify: Use stratified splits (for classification)
        random_state: Random seed
        
    Returns:
        List of (train_idx, val_idx) tuples
    """
    if stratify:
        # Assume last column is target
        y = data.iloc[:, -1]
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(data, y))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(data))
    
    logger.info(f"Created {n_splits} cross-validation splits")
    
    return splits


def load_data_with_validation(train_path: Path,
                            test_path: Optional[Path] = None,
                            validation_split: float = 0.2,
                            random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Load data with train/val/test splits.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data (optional)
        validation_split: Fraction for validation
        random_state: Random seed
        
    Returns:
        Dictionary with 'train', 'val', and optionally 'test' DataFrames
    """
    # Load training data
    train_data = pd.read_csv(train_path)
    logger.info(f"Loaded training data: {train_data.shape}")
    
    # Create validation split
    from sklearn.model_selection import train_test_split
    
    train_df, val_df = train_test_split(
        train_data,
        test_size=validation_split,
        random_state=random_state
    )
    
    logger.info(f"Split into train: {train_df.shape}, val: {val_df.shape}")
    
    data_dict = {
        'train': train_df,
        'val': val_df
    }
    
    # Load test data if provided
    if test_path and test_path.exists():
        test_df = pd.read_csv(test_path)
        data_dict['test'] = test_df
        logger.info(f"Loaded test data: {test_df.shape}")
    
    return data_dict


def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Check data quality and return report.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'shape': data.shape,
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
        'duplicates': data.duplicated().sum(),
        'numerical_stats': {},
        'categorical_stats': {}
    }
    
    # Numerical column statistics
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        report['numerical_stats'][col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'skew': data[col].skew(),
            'kurtosis': data[col].kurtosis()
        }
    
    # Categorical column statistics
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        report['categorical_stats'][col] = {
            'unique_values': data[col].nunique(),
            'most_common': data[col].value_counts().head().to_dict()
        }
    
    # Check for potential issues
    issues = []
    
    # High missing values
    high_missing = [col for col, pct in report['missing_percentage'].items() if pct > 20]
    if high_missing:
        issues.append(f"High missing values (>20%): {high_missing}")
    
    # Low variance
    for col, stats in report['numerical_stats'].items():
        if stats['std'] < 1e-6:
            issues.append(f"Low variance in {col}")
    
    # Outliers (using IQR method)
    for col in numerical_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            issues.append(f"{outliers} outliers in {col}")
    
    report['issues'] = issues
    
    return report


def save_data_splits(data_dict: Dict[str, pd.DataFrame], 
                    output_dir: Path) -> None:
    """
    Save data splits to directory.
    
    Args:
        data_dict: Dictionary with data splits
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, df in data_dict.items():
        output_path = output_dir / f"{split_name}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {split_name} data to {output_path}")


def load_feature_names(data_path: Path) -> List[str]:
    """
    Load feature names from data file.
    
    Args:
        data_path: Path to data file
        
    Returns:
        List of feature column names
    """
    # Read just the header
    df_header = pd.read_csv(data_path, nrows=0)
    
    # Assume last column is target
    feature_names = list(df_header.columns[:-1])
    
    return feature_names