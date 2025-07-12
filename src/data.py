# src/data.py
"""
Data handling module for the CHF project.

This module provides functions for:
- Downloading datasets from Kaggle
- Data preprocessing and encoding
- Train/test splitting
- Feature engineering
"""

import os
import shutil
import stat
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

# Try to import kagglehub
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    
from src.utils import get_logger
from src.utils.data_utils import check_data_quality, load_data_with_validation

logger = get_logger(__name__)


def secure_kaggle_json() -> None:
    """Set restricted permissions on kaggle.json (Windows compatible)."""
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_file = os.path.join(kaggle_dir, 'kaggle.json')
    
    if os.path.exists(kaggle_file):
        try:
            os.chmod(kaggle_file, stat.S_IREAD | stat.S_IWRITE)
            logger.debug("Kaggle API token secured")
        except Exception as e:
            logger.warning(f"Could not set permissions on kaggle.json: {e}")
    else:
        logger.warning(
            "kaggle.json not found. Please create ~/.kaggle/kaggle.json with:\n"
            '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}'
        )


def download_dataset(force_download: bool = False) -> bool:
    """
    Download the CHF dataset from Kaggle.
    
    Args:
        force_download: Force re-download even if data exists
        
    Returns:
        True if successful, False otherwise
    """
    if not KAGGLEHUB_AVAILABLE:
        logger.error(
            "kagglehub not installed. Install with: pip install kagglehub"
        )
        return False
    
    # Configuration
    DATA_DIR = Path('data/raw')
    DATASET_ID = "saurabhshahane/predicting-heat-flux"
    KAGGLE_FILENAME = "Data_CHF_Zhao_2020_ATE.csv"
    OUR_FILENAME = "heat_flux_data.csv"
    
    # Setup paths
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    final_path = DATA_DIR / OUR_FILENAME
    
    # Check if already downloaded
    if final_path.exists() and not force_download:
        logger.info(f"✓ Dataset already exists at {final_path}")
        
        # Verify it's valid
        try:
            df = pd.read_csv(final_path, nrows=5)
            logger.info(f"✓ Dataset verified: {len(df.columns)} columns")
            return True
        except Exception as e:
            logger.warning(f"Existing dataset appears corrupted: {e}")
            logger.info("Attempting re-download...")
    
    logger.info("📥 Downloading CHF dataset from Kaggle...")
    
    try:
        # Secure API credentials
        secure_kaggle_json()
        
        # Download dataset
        download_path = Path(kagglehub.dataset_download(
            DATASET_ID,
            force_download=True
        ))
        
        logger.debug(f"Downloaded to: {download_path}")
        
        # Find the CSV file
        csv_files = list(download_path.glob('*.csv'))
        
        if not csv_files:
            logger.error(f"No CSV files found in {download_path}")
            logger.debug(f"Contents: {list(download_path.glob('*'))}")
            return False
        
        # Find the correct file
        source_file = None
        for csv_file in csv_files:
            if KAGGLE_FILENAME in csv_file.name or 'CHF' in csv_file.name:
                source_file = csv_file
                break
        
        if source_file is None:
            # Use the first CSV if specific file not found
            source_file = csv_files[0]
            logger.warning(f"Expected file not found, using: {source_file.name}")
        
        # Move and rename
        shutil.move(str(source_file), str(final_path))
        logger.info(f"✓ Dataset saved as: {final_path}")
        
        # Verify download
        df = pd.read_csv(final_path, nrows=5)
        logger.info(f"✓ Download successful: {len(df.columns)} columns, {len(pd.read_csv(final_path))} rows")
        
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        logger.exception("Detailed traceback:")
        return False


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the CHF dataframe:
    - Remove unnecessary columns
    - Encode categorical variables
    - Clean column names
    - Handle missing values
    
    Args:
        df: Raw dataframe
        
    Returns:
        Preprocessed dataframe
    """
    logger.info("Preprocessing data...")
    
    # Create a copy
    df_processed = df.copy()
    
    # Log initial state
    logger.debug(f"Initial shape: {df_processed.shape}")
    logger.debug(f"Initial columns: {list(df_processed.columns)}")
    
    # Remove author column if present
    if 'author' in df_processed.columns:
        df_processed = df_processed.drop('author', axis=1)
        logger.info("✓ Removed 'author' column")
    
    # Handle categorical columns
    categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_columns:
        if col == 'geometry':
            # Special handling for geometry
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            df_processed = df_processed.drop(col, axis=1)
            
            # Log encoding mapping
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            logger.info(f"✓ Encoded '{col}': {mapping}")
        else:
            # For other categorical columns, decide on strategy
            unique_values = df_processed[col].nunique()
            
            if unique_values <= 10:
                # One-hot encode if few categories
                df_processed = pd.get_dummies(
                    df_processed, 
                    columns=[col], 
                    prefix=col,
                    drop_first=True
                )
                logger.info(f"✓ One-hot encoded '{col}' ({unique_values} categories)")
            else:
                # Drop if too many categories
                df_processed = df_processed.drop(col, axis=1)
                logger.warning(f"⚠ Dropped '{col}' (too many categories: {unique_values})")
    
    # Clean column names
    df_processed.columns = (df_processed.columns
                          .str.replace(' ', '_')
                          .str.replace('[', '')
                          .str.replace(']', '')
                          .str.replace('-', '_')
                          .str.replace('/', '_'))
    
    # Ensure target column is last
    target_columns = [col for col in df_processed.columns if 'chf_exp' in col.lower()]
    
    if target_columns:
        target_col = target_columns[0]
        cols = [col for col in df_processed.columns if col != target_col]
        cols.append(target_col)
        df_processed = df_processed[cols]
        logger.info(f"✓ Moved target column '{target_col}' to end")
    else:
        logger.warning("⚠ No 'chf_exp' column found - target column unclear")
    
    # Handle missing values
    missing_before = df_processed.isnull().sum().sum()
    if missing_before > 0:
        # Strategy: drop rows with missing target, impute features
        if target_columns:
            df_processed = df_processed.dropna(subset=[target_col])
        
        # Impute numerical features with median
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != target_col and df_processed[col].isnull().any():
                median_value = df_processed[col].median()
                df_processed[col].fillna(median_value, inplace=True)
                logger.debug(f"Imputed {col} with median: {median_value}")
        
        missing_after = df_processed.isnull().sum().sum()
        logger.info(f"✓ Handled missing values: {missing_before} → {missing_after}")
    
    # Remove any remaining rows with missing values
    if df_processed.isnull().any().any():
        df_processed = df_processed.dropna()
        logger.info(f"✓ Dropped remaining rows with missing values")
    
    # Check for duplicate rows
    n_duplicates = df_processed.duplicated().sum()
    if n_duplicates > 0:
        df_processed = df_processed.drop_duplicates()
        logger.info(f"✓ Removed {n_duplicates} duplicate rows")
    
    # Final checks
    logger.info(f"✓ Preprocessing complete: {df_processed.shape}")
    
    # Verify all columns are numeric
    non_numeric = df_processed.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning(f"⚠ Non-numeric columns remain: {non_numeric}")
    else:
        logger.info("✓ All columns are numeric")
    
    return df_processed


def split_data(test_size: float = 0.2, 
               random_state: int = 42,
               stratify: bool = False) -> Tuple[Path, Path]:
    """
    Split data into train/test sets with preprocessing.
    
    Args:
        test_size: Fraction for test set
        random_state: Random seed
        stratify: Whether to stratify split
        
    Returns:
        Tuple of (train_path, test_path)
    """
    # Path setup
    raw_path = Path('data/raw/heat_flux_data.csv')
    processed_dir = Path('data/processed')
    train_path = processed_dir / 'train.csv'
    test_path = processed_dir / 'test.csv'
    
    # Create directory
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already split and valid
    if train_path.exists() and test_path.exists():
        try:
            # Verify splits are valid
            train_df = pd.read_csv(train_path, nrows=5)
            test_df = pd.read_csv(test_path, nrows=5)
            
            # Check if preprocessing was applied
            if 'author' not in train_df.columns and 'geometry' not in train_df.columns:
                logger.info("✓ Valid train/test splits already exist")
                return train_path, test_path
            else:
                logger.info("⚠ Existing splits need reprocessing")
        except Exception as e:
            logger.warning(f"Error reading existing splits: {e}")
    
    # Load raw data
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")
    
    logger.info("Creating train/test split...")
    
    # Load and preprocess
    df = pd.read_csv(raw_path)
    logger.info(f"✓ Loaded raw data: {df.shape}")
    
    # Preprocess
    df_processed = preprocess_data(df)
    
    # Split data
    if stratify:
        # For regression, we can stratify on binned target values
        target_col = df_processed.columns[-1]
        y_binned = pd.qcut(df_processed[target_col], q=10, labels=False)
        
        train_df, test_df = train_test_split(
            df_processed,
            test_size=test_size,
            random_state=random_state,
            stratify=y_binned
        )
        logger.info("✓ Created stratified split")
    else:
        train_df, test_df = train_test_split(
            df_processed,
            test_size=test_size,
            random_state=random_state
        )
        logger.info("✓ Created random split")
    
    # Save splits
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"\n✓ Created splits:")
    logger.info(f"  - Train: {train_path} ({train_df.shape})")
    logger.info(f"  - Test:  {test_path} ({test_df.shape})")
    
    # Display sample
    logger.info("\n📊 Sample of training data:")
    logger.info(f"Columns: {list(train_df.columns)}")
    logger.debug("\n" + train_df.head().to_string())
    
    # Data quality report
    quality_report = check_data_quality(train_df)
    if quality_report['issues']:
        logger.warning(f"⚠ Data quality issues: {quality_report['issues']}")
    
    return train_path, test_path


def get_feature_names(data_path: Optional[Path] = None) -> List[str]:
    """
    Get feature names from processed data.
    
    Args:
        data_path: Path to data file (defaults to train.csv)
        
    Returns:
        List of feature column names
    """
    if data_path is None:
        data_path = Path('data/processed/train.csv')
    
    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        # Return default feature names based on expected structure
        return [
            'id', 'pressure_MPa', 'mass_flux_kg_m2_s', 'x_e_out__',
            'D_e_mm', 'D_h_mm', 'length_mm', 'geometry_encoded'
        ]
    
    # Read just the header
    df = pd.read_csv(data_path, nrows=1)
    
    # All columns except the last one (target)
    feature_names = list(df.columns[:-1])
    
    return feature_names


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional engineered features.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with additional features
    """
    df_features = df.copy()
    
    # Example feature engineering
    if 'D_h_mm' in df.columns and 'D_e_mm' in df.columns:
        # Diameter ratio
        df_features['diameter_ratio'] = df['D_h_mm'] / (df['D_e_mm'] + 1e-6)
    
    if 'length_mm' in df.columns and 'D_h_mm' in df.columns:
        # Length to diameter ratio
        df_features['L_D_ratio'] = df['length_mm'] / (df['D_h_mm'] + 1e-6)
    
    if 'mass_flux_kg_m2_s' in df.columns and 'pressure_MPa' in df.columns:
        # Flow intensity metric
        df_features['flow_intensity'] = (
            df['mass_flux_kg_m2_s'] * df['pressure_MPa']
        )
    
    logger.info(f"✓ Created {len(df_features.columns) - len(df.columns)} engineered features")
    
    return df_features


# Re-export for backward compatibility
from src.plotting import create_eda_plots

__all__ = [
    'download_dataset',
    'preprocess_data',
    'split_data',
    'get_feature_names',
    'create_features',
    'create_eda_plots',
    'check_data_quality',
    'load_data_with_validation'
]