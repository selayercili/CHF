# src/smote.py
"""
SMOTE module for the CHF project.

This module provides functions for:
- Preparing data for SMOTE with cluster awareness
- Applying SMOTE for regression problems
- Managing resampled datasets
"""

from pathlib import Path
from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import logging
from src.utils import get_logger

logger = get_logger(__name__)

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
    from src.cluster import apply_selected_clustering
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

def apply_cluster_aware_smote_regression(X, y, cluster_labels, sampling_strategy='auto', random_state=42):
    """
    Apply SMOTE for regression within each cluster using SMOTEN or alternative approaches.
    
    Args:
        X: Feature matrix (numpy array or pandas DataFrame)
        y: Target values (continuous)
        cluster_labels: Cluster assignments
        sampling_strategy: SMOTE sampling strategy
        random_state: Random seed
        
    Returns:
        Tuple of (X_resampled, y_resampled, cluster_labels_resampled)
    """
    try:
        from imblearn.over_sampling import SMOTEN, BorderlineSMOTE
        from sklearn.preprocessing import KBinsDiscretizer
        import numpy as np
        import pandas as pd
    except ImportError:
        raise ImportError("Install imbalanced-learn: pip install imbalanced-learn")
    
    logger.info("Applying cluster-aware SMOTE for regression...")
    
    # Convert to numpy arrays if pandas DataFrames
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        feature_names = X.columns.tolist()
    else:
        X_array = X
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    if isinstance(cluster_labels, pd.Series):
        cluster_labels_array = cluster_labels.values
    else:
        cluster_labels_array = cluster_labels
    
    # Strategy 1: Use quantile-based binning but preserve more target information
    n_bins = min(20, len(np.unique(y_array)) // 2)  # More bins for better granularity
    n_bins = max(3, n_bins)  # At least 3 bins
    
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_binned = discretizer.fit_transform(y_array.reshape(-1, 1)).ravel().astype(int)
    
    logger.info(f"✓ Binned continuous target into {n_bins} classes for SMOTE")
    logger.info(f"  Original target range: {y_array.min():.3f} to {y_array.max():.3f}")
    
    # Store results
    X_resampled_list = []
    y_resampled_list = []
    cluster_resampled_list = []
    
    # Apply SMOTE within each cluster
    for cluster_id in sorted(np.unique(cluster_labels_array)):
        cluster_mask = cluster_labels_array == cluster_id
        X_cluster = X_array[cluster_mask]
        y_cluster_binned = y_binned[cluster_mask]
        y_cluster_original = y_array[cluster_mask]
        
        logger.info(f"\n🎯 Processing Cluster {cluster_id}:")
        logger.info(f"  Original samples: {len(X_cluster)}")
        
        # Check class distribution in cluster
        unique_classes, class_counts = np.unique(y_cluster_binned, return_counts=True)
        min_class_size = min(class_counts) if len(class_counts) > 0 else 0
        
        logger.info(f"  Classes in cluster: {len(unique_classes)}")
        logger.info(f"  Min class size: {min_class_size}")
        
        # Skip clusters with insufficient diversity
        if len(unique_classes) < 2 or min_class_size < 2:
            logger.warning(f"  ⚠️ Cluster {cluster_id} has insufficient diversity for SMOTE")
            X_resampled_list.append(X_cluster)
            y_resampled_list.append(y_cluster_original)
            cluster_resampled_list.append(np.full(len(X_cluster), cluster_id))
            continue
        
        # Apply SMOTE with error handling
        try:
            # Try BorderlineSMOTE first (often better for regression-like problems)
            try:
                smote = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
                X_cluster_resampled, y_cluster_binned_resampled = smote.fit_resample(X_cluster, y_cluster_binned)
                smote_method = "BorderlineSMOTE"
            except:
                # Fallback to regular SMOTE
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
                X_cluster_resampled, y_cluster_binned_resampled = smote.fit_resample(X_cluster, y_cluster_binned)
                smote_method = "SMOTE"
            
            # Generate synthetic continuous targets using improved method
            n_original = len(X_cluster)
            n_synthetic = len(X_cluster_resampled) - n_original
            
            if n_synthetic > 0:
                # Method 1: Use class-aware target generation
                synthetic_targets = []
                
                # Get the synthetic samples and their classes
                synthetic_classes = y_cluster_binned_resampled[n_original:]
                
                for synthetic_class in synthetic_classes:
                    # Find original samples in the same class
                    class_mask = y_cluster_binned == synthetic_class
                    class_targets = y_cluster_original[class_mask]
                    
                    if len(class_targets) > 1:
                        # Sample from the class distribution with some noise
                        class_mean = np.mean(class_targets)
                        class_std = np.std(class_targets)
                        # Add some noise but keep within reasonable bounds
                        noise_factor = 0.1  # 10% noise
                        synthetic_target = np.random.normal(
                            class_mean, 
                            max(class_std * noise_factor, class_std * 0.05)  # At least 5% of std
                        )
                        # Clip to reasonable range based on class bounds
                        min_class_val = np.min(class_targets)
                        max_class_val = np.max(class_targets)
                        range_extension = (max_class_val - min_class_val) * 0.1  # Allow 10% extension
                        synthetic_target = np.clip(
                            synthetic_target,
                            min_class_val - range_extension,
                            max_class_val + range_extension
                        )
                    else:
                        # Single sample in class, use it directly with small noise
                        base_value = class_targets[0]
                        noise = np.random.normal(0, abs(base_value) * 0.05)  # 5% relative noise
                        synthetic_target = base_value + noise
                    
                    synthetic_targets.append(synthetic_target)
                
                synthetic_targets = np.array(synthetic_targets)
                
                # Combine original and synthetic targets
                y_cluster_resampled = np.concatenate([y_cluster_original, synthetic_targets])
            else:
                y_cluster_resampled = y_cluster_original
            
            logger.info(f"  ✓ {smote_method} applied: {len(X_cluster)} → {len(X_cluster_resampled)} samples")
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
    if X_resampled_list:
        X_resampled = np.vstack(X_resampled_list)
        y_resampled = np.concatenate(y_resampled_list)
        cluster_labels_resampled = np.concatenate(cluster_resampled_list)
    else:
        # Fallback if no clusters processed
        X_resampled = X_array
        y_resampled = y_array
        cluster_labels_resampled = cluster_labels_array
    
    logger.info(f"\n✅ Cluster-aware SMOTE completed:")
    logger.info(f"  Original data: {X_array.shape[0]} samples")
    logger.info(f"  Resampled data: {X_resampled.shape[0]} samples")
    logger.info(f"  Synthetic samples added: {X_resampled.shape[0] - X_array.shape[0]}")
    
    # Convert back to original format if needed
    if isinstance(X, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=feature_names)
    if isinstance(y, pd.Series):
        y_resampled = pd.Series(y_resampled, name=y.name)
    if isinstance(cluster_labels, pd.Series):
        cluster_labels_resampled = pd.Series(cluster_labels_resampled, name=cluster_labels.name)
    
    return X_resampled, y_resampled, cluster_labels_resampled

def _apply_cluster_aware_smote(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    clusters: Union[pd.Series, np.ndarray],
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal function to apply SMOTE within each cluster.
    
    Args:
        X: Feature matrix
        y: Target values
        clusters: Cluster assignments
        random_state: Random seed
        n_jobs: Number of CPU cores to use
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    try:
        from imblearn.over_sampling import SMOTENC
        from sklearn.preprocessing import KBinsDiscretizer
    except ImportError:
        logger.error("SMOTE requires imbalanced-learn. Install with: pip install imbalanced-learn")
        raise
    
    # Convert to numpy if pandas objects
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(clusters, pd.Series):
        clusters = clusters.values
    
    # Discretize target for SMOTE (required for regression)
    n_bins = max(3, min(20, len(np.unique(y)) // 2))
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel().astype(int)
    
    # Combine clusters with features (SMOTENC needs to know categorical features)
    # We treat cluster labels as categorical
    X_with_clusters = np.column_stack([X, clusters])
    categorical_features = [X.shape[1]]  # Index of cluster column
    
    # Apply SMOTENC (SMOTE for Numerical and Categorical)
    smote = SMOTENC(
        categorical_features=categorical_features,
        sampling_strategy='auto',
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    X_resampled, y_binned_resampled = smote.fit_resample(X_with_clusters, y_binned)
    
    # Convert back to original continuous target values
    # For synthetic samples, we use the median of their bin
    y_resampled = np.zeros_like(y_binned_resampled, dtype=float)
    
    for bin_idx in np.unique(y_binned):
        # Original samples in this bin
        original_mask = (y_binned == bin_idx)
        original_values = y[original_mask]
        
        # Synthetic samples in this bin
        synthetic_mask = (y_binned_resampled == bin_idx) & (np.arange(len(y_binned_resampled)) >= len(y))
        y_resampled[synthetic_mask] = np.median(original_values)
        
        # Keep original values for non-synthetic samples
        original_resampled_mask = (y_binned_resampled == bin_idx) & (np.arange(len(y_binned_resampled)) < len(y))
        y_resampled[original_resampled_mask] = y[original_mask]
    
    # Remove cluster column from features
    X_resampled = X_resampled[:, :-1]
    
    logger.info(
        f"Applied cluster-aware SMOTE: {len(X)} → {len(X_resampled)} samples "
        f"(added {len(X_resampled)-len(X)} synthetic samples)"
    )
    
    return X_resampled, y_resampled

def apply_smote_to_train_data(
    algorithm: str = 'kmeans',
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = -1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply cluster-aware SMOTE to the training data.
    
    Args:
        algorithm: Clustering algorithm used ('kmeans', 'hierarchical', 'dbscan')
        test_size: Size of test set (only used if need to recreate splits)
        random_state: Random seed for reproducibility
        n_jobs: Number of CPU cores to use (-1 for all)
        
    Returns:
        Tuple of (resampled_train_df, original_test_df)
    """
    # Load the clustered training data
    train_path = Path('data/processed/train.csv')
    test_path = Path('data/processed/test.csv')
    
    if not train_path.exists():
        logger.warning("train_with_clusters.csv not found, creating new split...")
        from src.data import split_data
        split_data(test_size=test_size, random_state=random_state)
        train_path = Path('data/processed/train.csv')  # Fall back to non-clustered version
    
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        logger.error(f"Failed to load train/test data: {e}")
        raise
    
    # Identify target column
    target_col = [col for col in train_df.columns if 'chf_exp' in col.lower()][0]
    
    # Prepare data for SMOTE
    X = train_df.drop(columns=[target_col, 'cluster_label'])
    y = train_df[target_col]
    clusters = train_df['cluster_label']
    
    # Apply SMOTE
    X_resampled, y_resampled = _apply_cluster_aware_smote(
        X, y, clusters,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    # Combine back into DataFrame
    resampled_train_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_train_df[target_col] = y_resampled
    resampled_train_df['cluster_label'] = clusters  # Keep original cluster labels
    
    # Save the resampled data
    resampled_path = Path('data/processed/train_resampled.csv')
    resampled_train_df.to_csv(resampled_path, index=False)
    logger.info(f"✓ SMOTE-applied training data saved to {resampled_path}")
    
    return resampled_train_df, test_df

__all__ = [
    'prepare_data_for_smote',
    'apply_cluster_aware_smote_regression',
    'apply_smote_to_train_data'
]