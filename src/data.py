# src/data.py
"""
Data handling module for the CHF project.

This module provides functions for:
- Downloading datasets from Kaggle
- Data preprocessing and encoding
- Train/test splitting
- Feature engineering
- Clustering analysis
"""

import os
import shutil
import stat
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
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
    - Remove unnecessary columns (author, id, D_e [mm])
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
    
    # Remove unnecessary columns
    columns_to_drop = ['author', 'id', 'D_e [mm]']
    for col in columns_to_drop:
        if col in df_processed.columns:
            df_processed = df_processed.drop(col, axis=1)
            logger.info(f"✓ Removed '{col}' column")
    
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


def perform_clustering(data_path: Optional[Path] = None, 
                      n_clusters_range: Tuple[int, int] = (2, 10),
                      save_results: bool = True) -> Dict[str, Any]:
    """
    Perform clustering analysis on the CHF data.
    
    Args:
        data_path: Path to processed data (defaults to train.csv)
        n_clusters_range: Range of clusters to test for K-means
        save_results: Whether to save clustering results
        
    Returns:
        Dictionary containing clustering results and metrics
    """
    logger.info("Starting clustering analysis...")
    
    # Setup paths
    if data_path is None:
        data_path = Path('data/processed/train.csv')
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded data: {df.shape}")
    
    # Separate features and target
    target_columns = [col for col in df.columns if 'chf_exp' in col.lower()]
    if target_columns:
        target_col = target_columns[0]
        X = df.drop(target_col, axis=1)
        y = df[target_col]
    else:
        # If no clear target, use all columns for clustering
        X = df
        y = None
        logger.info("No target column found, using all features for clustering")
    
    logger.info(f"✓ Features for clustering: {X.shape[1]} columns, {X.shape[0]} samples")
    
    # Scale features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("✓ Features scaled for clustering")
    
    # Results storage
    clustering_results = {
        'data_shape': X.shape,
        'feature_names': list(X.columns),
        'algorithms': {}
    }
    
    # 1. K-Means clustering with elbow method
    logger.info("\n=== K-Means Clustering ===")
    kmeans_results = {}
    inertias = []
    silhouette_scores = []
    
    min_clusters, max_clusters = n_clusters_range
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
        
        inertias.append(inertia)
        silhouette_scores.append(silhouette_avg)
        
        kmeans_results[k] = {
            'inertia': inertia,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'labels': cluster_labels
        }
        
        logger.info(f"K={k}: Inertia={inertia:.2f}, Silhouette={silhouette_avg:.3f}")
    
    # Find optimal K using silhouette score
    optimal_k = min_clusters + np.argmax(silhouette_scores)
    logger.info(f"✓ Optimal K-Means clusters (by silhouette): {optimal_k}")
    
    clustering_results['algorithms']['kmeans'] = {
        'results': kmeans_results,
        'optimal_k': optimal_k,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }
    
    # 2. DBSCAN clustering
    logger.info("\n=== DBSCAN Clustering ===")
    # Try different eps values
    eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    dbscan_results = {}
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        cluster_labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        if n_clusters > 1:
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
            davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
        else:
            silhouette_avg = -1
            calinski_harabasz = -1
            davies_bouldin = float('inf')
        
        dbscan_results[eps] = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'labels': cluster_labels
        }
        
        logger.info(f"eps={eps}: Clusters={n_clusters}, Noise={n_noise}, Silhouette={silhouette_avg:.3f}")
    
    # Find best DBSCAN parameters
    valid_dbscan = {k: v for k, v in dbscan_results.items() if v['n_clusters'] > 1}
    if valid_dbscan:
        best_eps = max(valid_dbscan.keys(), key=lambda x: valid_dbscan[x]['silhouette_score'])
        logger.info(f"✓ Best DBSCAN eps: {best_eps}")
    else:
        best_eps = None
        logger.warning("⚠ No valid DBSCAN clustering found")
    
    clustering_results['algorithms']['dbscan'] = {
        'results': dbscan_results,
        'best_eps': best_eps
    }
    
    # 3. Hierarchical clustering
    logger.info("\n=== Hierarchical Clustering ===")
    hierarchical_results = {}
    
    for k in range(min_clusters, min(max_clusters + 1, 8)):  # Limit for computational efficiency
        hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
        cluster_labels = hierarchical.fit_predict(X_scaled)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
        
        hierarchical_results[k] = {
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'labels': cluster_labels
        }
        
        logger.info(f"K={k}: Silhouette={silhouette_avg:.3f}")
    
    # Find optimal hierarchical clustering
    hierarchical_scores = [v['silhouette_score'] for v in hierarchical_results.values()]
    optimal_hierarchical_k = min_clusters + np.argmax(hierarchical_scores)
    logger.info(f"✓ Optimal Hierarchical clusters: {optimal_hierarchical_k}")
    
    clustering_results['algorithms']['hierarchical'] = {
        'results': hierarchical_results,
        'optimal_k': optimal_hierarchical_k
    }
    
    # Summary
    logger.info("\n=== Clustering Summary ===")
    logger.info(f"K-Means optimal clusters: {optimal_k}")
    if best_eps:
        logger.info(f"DBSCAN optimal eps: {best_eps} ({dbscan_results[best_eps]['n_clusters']} clusters)")
    logger.info(f"Hierarchical optimal clusters: {optimal_hierarchical_k}")
    
    # Save results if requested
    if save_results:
        results_dir = Path('data/clustering')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cluster assignments for best models
        cluster_assignments = pd.DataFrame({
            'sample_id': range(len(X)),
            'kmeans_clusters': kmeans_results[optimal_k]['labels'],
            'hierarchical_clusters': hierarchical_results[optimal_hierarchical_k]['labels']
        })
        
        if best_eps:
            cluster_assignments['dbscan_clusters'] = dbscan_results[best_eps]['labels']
        
        # Add original features and target if available
        cluster_assignments = pd.concat([cluster_assignments, X.reset_index(drop=True)], axis=1)
        if y is not None:
            cluster_assignments[target_col] = y.reset_index(drop=True)
        
        assignments_path = results_dir / 'cluster_assignments.csv'
        cluster_assignments.to_csv(assignments_path, index=False)
        logger.info(f"✓ Cluster assignments saved: {assignments_path}")
        
        # Save clustering metrics
        import json
        metrics_path = results_dir / 'clustering_metrics.json'
        
        # Create a deep copy and properly convert all data types for JSON serialization
        def make_json_serializable(obj):
            """Recursively convert object to JSON-serializable format."""
            if isinstance(obj, dict):
                return {key: make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            else:
                # For any other type, try to convert to string
                return str(obj)
        
        # Convert clustering results for JSON
        results_for_json = make_json_serializable(clustering_results.copy())
        
        # Remove cluster labels from JSON (too large and not needed for metrics)
        for alg_name, alg_data in results_for_json['algorithms'].items():
            if 'results' in alg_data:
                for k, v in alg_data['results'].items():
                    if 'labels' in v:
                        del v['labels']
        
        try:
            with open(metrics_path, 'w') as f:
                json.dump(results_for_json, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Clustering metrics saved: {metrics_path}")
        except Exception as e:
            logger.error(f"Failed to save clustering metrics: {e}")
            # Save a simplified version with just the key metrics
            simplified_metrics = {
                'data_shape': list(clustering_results['data_shape']),
                'feature_names': clustering_results['feature_names'],
                'kmeans_optimal_k': clustering_results['algorithms']['kmeans']['optimal_k'],
                'kmeans_best_silhouette': float(max(clustering_results['algorithms']['kmeans']['silhouette_scores'])),
                'hierarchical_optimal_k': clustering_results['algorithms']['hierarchical']['optimal_k'],
                'dbscan_best_eps': clustering_results['algorithms']['dbscan']['best_eps']
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(simplified_metrics, f, indent=2)
            logger.info(f"✓ Simplified clustering metrics saved: {metrics_path}")
    
    logger.info("✓ Clustering analysis completed")
    return clustering_results


def apply_selected_clustering(algorithm='kmeans', data_path=None, save_to_original=True):
    """
    Apply the selected clustering algorithm and add cluster labels to training data.
    
    Args:
        algorithm: 'kmeans', 'hierarchical', or 'dbscan'
        data_path: Path to training data (defaults to train.csv)
        save_to_original: Whether to save cluster labels back to original train.csv
        
    Returns:
        DataFrame with cluster labels added
    """
    logger.info(f"Applying {algorithm} clustering to training data...")
    
    # Setup paths
    if data_path is None:
        data_path = Path('data/processed/train.csv')
    
    clustering_dir = Path('data/clustering')
    cluster_assignments_path = clustering_dir / 'cluster_assignments.csv'
    
    if not cluster_assignments_path.exists():
        raise FileNotFoundError(f"Cluster assignments not found. Run perform_clustering() first.")
    
    # Load original training data and cluster assignments
    train_df = pd.read_csv(data_path)
    cluster_df = pd.read_csv(cluster_assignments_path)
    
    # Select the appropriate cluster column
    cluster_column_map = {
        'kmeans': 'kmeans_clusters',
        'hierarchical': 'hierarchical_clusters', 
        'dbscan': 'dbscan_clusters'
    }
    
    if algorithm not in cluster_column_map:
        raise ValueError(f"Algorithm must be one of: {list(cluster_column_map.keys())}")
    
    cluster_col = cluster_column_map[algorithm]
    
    if cluster_col not in cluster_df.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in cluster assignments")
    
    # Add cluster labels to training data
    train_df_with_clusters = train_df.copy()
    train_df_with_clusters['cluster_label'] = cluster_df[cluster_col].values
    
    logger.info(f"✓ Added {algorithm} cluster labels to training data")
    logger.info(f"  Cluster distribution: {dict(pd.Series(cluster_df[cluster_col]).value_counts().sort_index())}")
    
    # Save back to original file if requested
    if save_to_original:
        # Create backup first
        backup_path = data_path.parent / f"{data_path.stem}_backup.csv"
        train_df.to_csv(backup_path, index=False)
        logger.info(f"✓ Created backup: {backup_path}")
        
        # Save with cluster labels
        train_df_with_clusters.to_csv(data_path, index=False)
        logger.info(f"✓ Updated training data with cluster labels: {data_path}")
    
    return train_df_with_clusters


def prepare_data_for_smote(algorithm='kmeans', target_column=None):
    """
    Prepare data for SMOTE by adding cluster-aware sampling.
    
    Args:
        algorithm: Clustering algorithm to use
        target_column: Name of target column (auto-detected if None)
        
    Returns:
        Tuple of (X, y, cluster_labels) ready for SMOTE
    """
    logger.info("Preparing data for SMOTE with cluster awareness...")
    
    # Apply clustering
    train_df_clustered = apply_selected_clustering(algorithm=algorithm, save_to_original=False)
    
    # Identify target column
    if target_column is None:
        target_columns = [col for col in train_df_clustered.columns if 'chf_exp' in col.lower()]
        if target_columns:
            target_column = target_columns[0]
        else:
            raise ValueError("Could not identify target column. Please specify target_column parameter.")
    
    # Separate features, target, and clusters
    cluster_labels = train_df_clustered['cluster_label']
    X = train_df_clustered.drop([target_column, 'cluster_label'], axis=1)
    y = train_df_clustered[target_column]
    
    logger.info(f"✓ Prepared data for SMOTE:")
    logger.info(f"  Features: {X.shape}")
    logger.info(f"  Target: {y.shape}")
    logger.info(f"  Clusters: {len(cluster_labels.unique())} unique clusters")
    
    # Show cluster-target relationship
    logger.info("\n📊 Cluster-Target Analysis:")
    for cluster_id in sorted(cluster_labels.unique()):
        cluster_mask = cluster_labels == cluster_id
        cluster_target_mean = y[cluster_mask].mean()
        cluster_target_std = y[cluster_mask].std()
        cluster_size = cluster_mask.sum()
        
        logger.info(f"  Cluster {cluster_id}: {cluster_size} samples, "
                   f"target mean={cluster_target_mean:.3f} ±{cluster_target_std:.3f}")
    
    return X, y, cluster_labels


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


def apply_cluster_aware_smote(X, y, cluster_labels, sampling_strategy='auto', random_state=42):
    """
    Apply SMOTE within each cluster separately for better synthetic sample generation.
    
    Args:
        X: Feature matrix
        y: Target values (will be binned for SMOTE)
        cluster_labels: Cluster assignments
        sampling_strategy: SMOTE sampling strategy
        random_state: Random seed
        
    Returns:
        Tuple of (X_resampled, y_resampled, cluster_labels_resampled)
    """
    try:
        from imblearn.over_sampling import SMOTE
        from sklearn.preprocessing import KBinsDiscretizer
    except ImportError:
        raise ImportError("Install imbalanced-learn: pip install imbalanced-learn")
    
    logger.info("Applying cluster-aware SMOTE...")
    
    # Convert continuous target to discrete classes for SMOTE
    # Use quantile-based binning
    n_bins = min(10, len(np.unique(y)))  # Max 10 bins, less if few unique values
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_binned = discretizer.fit_transform(y.values.reshape(-1, 1)).ravel().astype(int)
    
    logger.info(f"✓ Binned continuous target into {n_bins} classes")
    logger.info(f"  Original target range: {y.min():.3f} to {y.max():.3f}")
    
    # Store results
    X_resampled_list = []
    y_resampled_list = []
    cluster_resampled_list = []
    
    # Apply SMOTE within each cluster
    for cluster_id in sorted(cluster_labels.unique()):
        cluster_mask = cluster_labels == cluster_id
        X_cluster = X[cluster_mask]
        y_cluster_binned = y_binned[cluster_mask]
        y_cluster_original = y[cluster_mask]
        
        logger.info(f"\n🎯 Processing Cluster {cluster_id}:")
        logger.info(f"  Original samples: {len(X_cluster)}")
        
        # Check if cluster has enough diversity for SMOTE
        unique_classes = np.unique(y_cluster_binned)
        if len(unique_classes) < 2:
            logger.warning(f"  ⚠️ Cluster {cluster_id} has only 1 class, skipping SMOTE")
            X_resampled_list.append(X_cluster)
            y_resampled_list.append(y_cluster_original)
            cluster_resampled_list.append(np.full(len(X_cluster), cluster_id))
            continue
        
        # Apply SMOTE to this cluster
        try:
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
            X_cluster_resampled, y_cluster_binned_resampled = smote.fit_resample(X_cluster, y_cluster_binned)
            
            # Map back to original continuous values using cluster statistics
            y_cluster_mean = y_cluster_original.mean()
            y_cluster_std = y_cluster_original.std()
            
            # Generate synthetic continuous targets with same distribution
            n_synthetic = len(X_cluster_resampled) - len(X_cluster)
            synthetic_targets = np.random.normal(y_cluster_mean, y_cluster_std, n_synthetic)
            
            # Combine original and synthetic targets
            y_cluster_resampled = np.concatenate([
                y_cluster_original.values,
                synthetic_targets
            ])
            
            logger.info(f"  ✓ SMOTE applied: {len(X_cluster)} → {len(X_cluster_resampled)} samples")
            logger.info(f"    Added {n_synthetic} synthetic samples")
            
            X_resampled_list.append(X_cluster_resampled)
            y_resampled_list.append(y_cluster_resampled)
            cluster_resampled_list.append(np.full(len(X_cluster_resampled), cluster_id))
            
        except Exception as e:
            logger.warning(f"  ⚠️ SMOTE failed for cluster {cluster_id}: {e}")
            logger.warning(f"    Using original data for this cluster")
            X_resampled_list.append(X_cluster)
            y_resampled_list.append(y_cluster_original)
            cluster_resampled_list.append(np.full(len(X_cluster), cluster_id))
    
    # Combine all clusters
    X_resampled = np.vstack(X_resampled_list)
    y_resampled = np.concatenate(y_resampled_list)
    cluster_labels_resampled = np.concatenate(cluster_resampled_list)
    
    logger.info(f"\n✅ Cluster-aware SMOTE completed:")
    logger.info(f"  Original data: {X.shape[0]} samples")
    logger.info(f"  Resampled data: {X_resampled.shape[0]} samples")
    logger.info(f"  Synthetic samples added: {X_resampled.shape[0] - X.shape[0]}")
    
    return X_resampled, y_resampled, cluster_labels_resampled


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
            'pressure_MPa', 'mass_flux_kg_m2_s', 'x_e_out__',
            'D_h_mm', 'length_mm', 'geometry_encoded'
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
    if 'D_h_mm' in df.columns:
        # Length to diameter ratio
        if 'length_mm' in df.columns:
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
    'perform_clustering',
    'create_eda_plots',
    'check_data_quality',
    'load_data_with_validation'
    'apply_selected_clustering',       
    'prepare_data_for_smote',          
    'apply_cluster_aware_smote',     
]