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
from sklearn.preprocessing import LabelEncoder

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
    """Download the heat flux dataset from Kaggle"""
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

def preprocess_data(df):
    """
    Preprocess the dataframe:
    - Remove 'author' column
    - Convert 'geometry' to numeric (categorical encoding)
    - Clean column names (remove special characters)
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Remove the 'author' column
        df_processed = df_processed.drop('author', axis=1)
        print("✓ Removed 'author' column")
    
    # Handle categorical 'geometry' column
    if 'geometry' in df_processed.columns:
        # Option 1: Label encoding (tube=0, annulus=1)
        le = LabelEncoder()
        df_processed['geometry_encoded'] = le.fit_transform(df_processed['geometry'])
        df_processed = df_processed.drop('geometry', axis=1)
        print(f"✓ Encoded 'geometry' column: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Option 2: One-hot encoding (uncomment if preferred)
        # df_processed = pd.get_dummies(df_processed, columns=['geometry'], prefix='geometry')
        # print("✓ One-hot encoded 'geometry' column")
    
    # Clean column names - remove spaces and special characters
    df_processed.columns = df_processed.columns.str.replace(' ', '_')
    df_processed.columns = df_processed.columns.str.replace('[', '')
    df_processed.columns = df_processed.columns.str.replace(']', '')
    df_processed.columns = df_processed.columns.str.replace('-', '_')
    df_processed.columns = df_processed.columns.str.replace('/', '_')
    
    # Ensure the target column (chf_exp) is the last column
    if 'chf_exp_MW_m2' in df_processed.columns:
        # Move target to the end
        cols = [col for col in df_processed.columns if col != 'chf_exp_MW_m2']
        cols.append('chf_exp_MW_m2')
        df_processed = df_processed[cols]
    
    print(f"✓ Preprocessed data shape: {df_processed.shape}")
    print(f"✓ Columns: {list(df_processed.columns)}")
    
    # Check for any remaining non-numeric columns
    non_numeric_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print(f"⚠ Warning: Non-numeric columns remaining: {non_numeric_cols}")
    
    return df_processed
    
def split_data(test_size=0.2, random_state=42):
    """
    Splits data into train/test sets with preprocessing
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
        
        # Verify they don't contain string columns
        train_sample = pd.read_csv(train_path, nrows=5)
        if 'author' in train_sample.columns or 'geometry' in train_sample.columns:
            print("⚠ Existing splits contain string columns - regenerating...")
        else:
            return train_path, test_path
    
    # Verify raw data exists
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")
    
    print("Splitting and preprocessing data...")
    try:
        # Load raw data
        df = pd.read_csv(raw_path)
        print(f"✓ Loaded raw data: {df.shape}")
        
        # Preprocess the data
        df_processed = preprocess_data(df)
        
        # Split the preprocessed data
        train_df, test_df = train_test_split(
            df_processed, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Save splits
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"\n✓ Created splits:")
        print(f"  - Train: {train_path} ({train_df.shape})")
        print(f"  - Test:  {test_path} ({test_df.shape})")
        
        # Display sample of the processed data
        print("\n📊 Sample of processed training data:")
        print(train_df.head())
        
        # Verify all columns are numeric
        print("\n📊 Data types in processed data:")
        print(train_df.dtypes)
        
        return train_path, test_path
        
    except Exception as e:
        print(f"× Splitting failed: {str(e)}")
        raise

def get_feature_names():
    """
    Return the feature names after preprocessing
    Useful for model interpretability
    """
    train_path = Path('data/processed/train.csv')
    if train_path.exists():
        df = pd.read_csv(train_path, nrows=1)
        # All columns except the last one (target)
        return list(df.columns[:-1])
    else:
        # Default feature names based on the raw data structure
        return [
            'id', 'pressure_MPa', 'mass_flux_kg_m2_s', 'x_e_out__', 
            'D_e_mm', 'D_h_mm', 'length_mm', 'geometry_encoded'
        ]

if __name__ == "__main__":
    # Run the data pipeline
    print("🚀 Starting data pipeline...\n")
    
    # Download dataset
    if download_dataset():
        # Split and preprocess data
        train_path, test_path = split_data()
        
        print("\n✅ Data pipeline completed successfully!")
        print(f"\n📊 Feature names: {get_feature_names()}")
    else:
        print("\n❌ Data pipeline failed!")