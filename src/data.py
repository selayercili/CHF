"""
This will handle the organization of the data
"""

import os
import shutil
import stat
import kagglehub
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def secure_kaggle_json():
    """Set restricted permissions on kaggle.json (Windows compatible)"""
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_file = os.path.join(kaggle_dir, 'kaggle.json')
    
    if os.path.exists(kaggle_file):
        os.chmod(kaggle_file, stat.S_IREAD | stat.S_IWRITE)
        print("Kaggle API token secured")
    else:
        print("Warning: kaggle.json not found")

def download_dataset():

    # Configuration
    DATA_DIR = Path('data/raw')
    DATASET_ID = "saurabhshahane/predicting-heat-flux"
    KAGGLE_FILENAME = "Data_CHF_Zhao_2020_ATE.csv"
    OUR_FILENAME = "heat_flux_data.csv"
    
    # Setup paths
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    final_path = DATA_DIR / OUR_FILENAME
    
    # Skip if already downloaded
    if final_path.exists():
        print(f"✓ Dataset exists at {final_path}")
        return True
    
    print("↓ Downloading dataset...")
    try:
        secure_kaggle_json()
        
        # Force fresh download (KaggleHub caches aggressively)
        download_path = Path(kagglehub.dataset_download(
            DATASET_ID,
            force_download=True  # Critical for re-downloads
        ))
        
        # Find the downloaded file
        source_file = next(download_path.glob('*.csv'), None)
        if not source_file:
            print(f"× No CSV found in {download_path}")
            print("Contents:", list(download_path.glob('*')))
            return False
        
        # Move and rename
        shutil.move(str(source_file), str(final_path))
        print(f"✓ Saved as {final_path}")
        return True
        
    except Exception as e:
        print(f"× Download failed: {str(e)}")
        return False
    
def split_data(test_size=0.2, random_state=42):
    """
    Splits data into train/test sets if not already split
    Returns: (train_path, test_path) if split exists or was created
    """
    # Path setup
    raw_path = Path('data/raw/heat_flux_data.csv')
    train_path = Path('data/processed/train.csv')
    test_path = Path('data/processed/test.csv')
    
    # Create processed directory
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already split
    if train_path.exists() and test_path.exists():
        print("✓ Train/test splits already exist")
        return train_path, test_path
    
    # Verify raw data exists
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")
    
    print("Splitting data...")
    try:
        # Load and split
        df = pd.read_csv(raw_path)
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Save splits
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"✓ Created:\n- {train_path}\n- {test_path}")
        return train_path, test_path
        
    except Exception as e:
        print(f"× Splitting failed: {str(e)}")
        raise