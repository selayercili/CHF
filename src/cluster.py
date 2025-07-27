# src/cluster.py
"""
Clustering module for the CHF project.

This module provides functions for:
- K-means clustering with elbow method
- DBSCAN clustering
- Hierarchical clustering
- Clustering evaluation metrics
- Cluster assignment and analysis
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging

from src.utils import get_logger

logger = get_logger(__name__)


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


__all__ = [
    'perform_clustering',
    'apply_selected_clustering'
]