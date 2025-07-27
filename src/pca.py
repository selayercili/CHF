# src/pca.py
"""
PCA module for the CHF project.

This module provides functions for:
- Principal Component Analysis with automatic component selection
- Dimensionality reduction and feature importance analysis
- PCA visualization and explained variance analysis
- Integration with clustered and SMOTE-enhanced data
- PCA model saving and loading for consistent transformations
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any, Union
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
import joblib
import json

from src.utils import get_logger

logger = get_logger(__name__)


def determine_optimal_components(
    X: Union[pd.DataFrame, np.ndarray],
    variance_threshold: float = 0.95,
    max_components: Optional[int] = None,
    method: str = 'cumulative_variance'
) -> int:
    """
    Determine optimal number of PCA components using various methods.
    
    Args:
        X: Feature matrix
        variance_threshold: Cumulative variance threshold (0.95 = 95%)
        max_components: Maximum number of components to consider
        method: Method to use ('cumulative_variance', 'elbow', 'kaiser')
        
    Returns:
        Optimal number of components
    """
    logger.info(f"Determining optimal PCA components using {method} method...")
    
    # Convert to numpy if pandas
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        n_features = X.shape[1]
    else:
        X_array = X
        n_features = X.shape[1]
    
    # Set max components
    if max_components is None:
        max_components = min(n_features, X_array.shape[0] - 1)
    else:
        max_components = min(max_components, n_features, X_array.shape[0] - 1)
    
    # Fit PCA with all possible components
    pca_full = PCA(n_components=max_components)
    pca_full.fit(X_array)
    
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    if method == 'cumulative_variance':
        # Find first component where cumulative variance exceeds threshold
        optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        logger.info(f"✓ Components for {variance_threshold*100:.1f}% variance: {optimal_components}")
        
    elif method == 'elbow':
        # Use elbow method - find point of maximum curvature
        # Calculate second derivative to find elbow
        if len(explained_variance_ratio) < 3:
            optimal_components = len(explained_variance_ratio)
        else:
            # Calculate differences
            first_diff = np.diff(explained_variance_ratio)
            second_diff = np.diff(first_diff)
            
            # Find elbow (maximum second derivative in absolute terms)
            elbow_idx = np.argmax(np.abs(second_diff)) + 2  # +2 because of double diff
            optimal_components = min(elbow_idx, max_components)
        
        logger.info(f"✓ Elbow method suggests: {optimal_components} components")
        
    elif method == 'kaiser':
        # Kaiser criterion - keep components with eigenvalue > 1
        eigenvalues = pca_full.explained_variance_
        optimal_components = np.sum(eigenvalues > 1.0)
        if optimal_components == 0:
            optimal_components = 1
        
        logger.info(f"✓ Kaiser criterion suggests: {optimal_components} components")
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Ensure we have at least 1 component and not more than available
    optimal_components = max(1, min(optimal_components, max_components))
    
    # Log variance explained by optimal number
    variance_explained = cumulative_variance[optimal_components - 1]
    logger.info(f"✓ {optimal_components} components explain {variance_explained:.3f} ({variance_explained*100:.1f}%) of variance")
    
    return optimal_components


def apply_pca_analysis(
    data_path: Optional[Path] = None,
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95,
    save_results: bool = True,
    use_smote_data: bool = True
) -> Dict[str, Any]:
    """
    Apply PCA analysis to the CHF data with comprehensive evaluation.
    
    Args:
        data_path: Path to data file (auto-selects based on use_smote_data)
        n_components: Number of components (auto-determined if None)
        variance_threshold: Variance threshold for auto-selection
        save_results: Whether to save PCA models and results
        use_smote_data: Whether to use SMOTE-enhanced data
        
    Returns:
        Dictionary containing PCA results and analysis
    """
    logger.info("Starting PCA analysis...")
    
    # Determine data path
    if data_path is None:
        if use_smote_data:
            data_path = Path('data/processed/train_resampled.csv')
            if not data_path.exists():
                logger.warning("SMOTE data not found, falling back to regular training data")
                data_path = Path('data/processed/train.csv')
        else:
            data_path = Path('data/processed/train.csv')
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"✓ Loaded data: {df.shape}")
    
    # Identify target and cluster columns
    target_columns = [col for col in df.columns if 'chf_exp' in col.lower()]
    cluster_columns = [col for col in df.columns if 'cluster' in col.lower()]
    
    # Separate features from target and clusters
    exclude_columns = target_columns + cluster_columns
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    X = df[feature_columns]
    target_col = target_columns[0] if target_columns else None
    y = df[target_col] if target_col else None
    
    logger.info(f"✓ Features for PCA: {X.shape[1]} columns, {X.shape[0]} samples")
    logger.info(f"  Feature columns: {feature_columns}")
    
    # Scale features before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("✓ Features scaled for PCA")
    
    # Determine optimal components if not specified
    if n_components is None:
        n_components = determine_optimal_components(
            X_scaled, 
            variance_threshold=variance_threshold,
            method='cumulative_variance'
        )
        
        # Also calculate other methods for comparison
        elbow_components = determine_optimal_components(X_scaled, method='elbow')
        kaiser_components = determine_optimal_components(X_scaled, method='kaiser')
        
        logger.info(f"\n📊 Component Selection Comparison:")
        logger.info(f"  Cumulative Variance ({variance_threshold*100:.1f}%): {n_components}")
        logger.info(f"  Elbow Method: {elbow_components}")
        logger.info(f"  Kaiser Criterion: {kaiser_components}")
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    logger.info(f"✓ PCA applied: {X.shape[1]} → {n_components} dimensions")
    
    # Create results dictionary
    pca_results = {
        'original_shape': X.shape,
        'pca_shape': X_pca.shape,
        'n_components': n_components,
        'feature_names': feature_columns,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'singular_values': pca.singular_values_.tolist(),
        'components': pca.components_.tolist(),
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    
    # Calculate total variance explained
    total_variance = sum(pca.explained_variance_ratio_)
    logger.info(f"✓ Total variance explained: {total_variance:.3f} ({total_variance*100:.1f}%)")
    
    # Feature importance analysis
    logger.info("\n🔍 Feature Importance Analysis:")
    
    # Calculate feature contributions to each component
    feature_importance = np.abs(pca.components_).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    pca_results['feature_importance'] = feature_importance_df.to_dict('records')
    
    # Display top contributing features
    logger.info("Top 10 most important features:")
    for i, row in feature_importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Component analysis
    logger.info(f"\n📈 Principal Component Analysis:")
    for i in range(min(5, n_components)):  # Show first 5 components
        variance_pct = pca.explained_variance_ratio_[i] * 100
        cumulative_pct = sum(pca.explained_variance_ratio_[:i+1]) * 100
        
        # Find top features for this component
        component_contributions = np.abs(pca.components_[i])
        top_feature_indices = np.argsort(component_contributions)[-3:][::-1]
        top_features = [feature_columns[idx] for idx in top_feature_indices]
        
        logger.info(f"  PC{i+1}: {variance_pct:.1f}% variance (cumulative: {cumulative_pct:.1f}%)")
        logger.info(f"       Top features: {', '.join(top_features)}")
    
    # Create PCA-transformed dataframes
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)
    
    # Add target and cluster columns back if they exist
    if target_col:
        X_pca_df[target_col] = y
    
    if cluster_columns:
        for cluster_col in cluster_columns:
            X_pca_df[cluster_col] = df[cluster_col]
    
    pca_results['pca_data'] = X_pca_df
    
    # Save results if requested
    if save_results:
        results_dir = Path('data/pca')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PCA model and scaler
        pca_model_path = results_dir / 'pca_model.joblib'
        scaler_path = results_dir / 'pca_scaler.joblib'
        
        joblib.dump(pca, pca_model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"✓ PCA model saved: {pca_model_path}")
        logger.info(f"✓ Scaler saved: {scaler_path}")
        
        # Save PCA-transformed data
        pca_data_path = results_dir / 'pca_transformed_data.csv'
        X_pca_df.to_csv(pca_data_path, index=False)
        logger.info(f"✓ PCA-transformed data saved: {pca_data_path}")
        
        # Save feature importance
        feature_importance_path = results_dir / 'feature_importance.csv'
        feature_importance_df.to_csv(feature_importance_path, index=False)
        logger.info(f"✓ Feature importance saved: {feature_importance_path}")
        
        # Save PCA analysis results (without the large data)
        analysis_results = pca_results.copy()
        del analysis_results['pca_data']  # Remove large dataframe
        
        analysis_path = results_dir / 'pca_analysis.json'
        try:
            with open(analysis_path, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            logger.info(f"✓ PCA analysis results saved: {analysis_path}")
        except Exception as e:
            logger.warning(f"Could not save analysis results as JSON: {e}")
    
    logger.info("✓ PCA analysis completed")
    return pca_results


def apply_pca_to_test_data() -> pd.DataFrame:
    """
    Apply the saved PCA transformation to test data.
    
    Returns:
        PCA-transformed test dataframe
    """
    logger.info("Applying PCA transformation to test data...")
    
    # Load test data
    test_path = Path('data/processed/test.csv')
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    test_df = pd.read_csv(test_path)
    logger.info(f"✓ Loaded test data: {test_df.shape}")
    
    # Load PCA model and scaler
    pca_dir = Path('data/pca')
    pca_model_path = pca_dir / 'pca_model.joblib'
    scaler_path = pca_dir / 'pca_scaler.joblib'
    
    if not pca_model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "PCA model or scaler not found. Run apply_pca_analysis() first."
        )
    
    pca = joblib.load(pca_model_path)
    scaler = joblib.load(scaler_path)
    logger.info("✓ Loaded PCA model and scaler")
    
    # Identify columns (same logic as training)
    target_columns = [col for col in test_df.columns if 'chf_exp' in col.lower()]
    cluster_columns = [col for col in test_df.columns if 'cluster' in col.lower()]
    exclude_columns = target_columns + cluster_columns
    feature_columns = [col for col in test_df.columns if col not in exclude_columns]
    
    X_test = test_df[feature_columns]
    target_col = target_columns[0] if target_columns else None
    y_test = test_df[target_col] if target_col else None
    
    # Apply same preprocessing and PCA transformation
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Create PCA-transformed dataframe
    pca_columns = [f'PC{i+1}' for i in range(X_test_pca.shape[1])]
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_columns, index=test_df.index)
    
    # Add target and cluster columns back
    if target_col:
        X_test_pca_df[target_col] = y_test
    
    if cluster_columns:
        for cluster_col in cluster_columns:
            X_test_pca_df[cluster_col] = test_df[cluster_col]
    
    # Save transformed test data
    pca_test_path = pca_dir / 'pca_test_data.csv'
    X_test_pca_df.to_csv(pca_test_path, index=False)
    logger.info(f"✓ PCA-transformed test data saved: {pca_test_path}")
    
    logger.info(f"✓ Test data transformed: {X_test.shape} → {X_test_pca.shape}")
    
    return X_test_pca_df


def get_pca_feature_names(n_components: Optional[int] = None) -> List[str]:
    """
    Get PCA feature names (PC1, PC2, etc.).
    
    Args:
        n_components: Number of components (auto-detected if None)
        
    Returns:
        List of PCA feature names
    """
    if n_components is None:
        # Try to load from saved PCA model
        pca_model_path = Path('data/pca/pca_model.joblib')
        if pca_model_path.exists():
            pca = joblib.load(pca_model_path)
            n_components = pca.n_components_
        else:
            logger.warning("PCA model not found, defaulting to 10 components")
            n_components = 10
    
    return [f'PC{i+1}' for i in range(n_components)]


def analyze_pca_impact_on_clusters() -> Dict[str, Any]:
    """
    Analyze how PCA transformation affects cluster separation.
    
    Returns:
        Dictionary with cluster analysis results
    """
    logger.info("Analyzing PCA impact on cluster separation...")
    
    # Load original and PCA-transformed data
    pca_data_path = Path('data/pca/pca_transformed_data.csv')
    original_data_path = Path('data/processed/train_resampled.csv')
    
    if not pca_data_path.exists():
        logger.warning("PCA data not found, falling back to regular training data")
        original_data_path = Path('data/processed/train.csv')
    
    if not pca_data_path.exists() or not original_data_path.exists():
        raise FileNotFoundError("Required data files not found for cluster analysis")
    
    pca_df = pd.read_csv(pca_data_path)
    original_df = pd.read_csv(original_data_path)
    
    # Check if cluster labels exist
    cluster_columns = [col for col in pca_df.columns if 'cluster' in col.lower()]
    if not cluster_columns:
        logger.warning("No cluster labels found, skipping cluster analysis")
        return {'error': 'No cluster labels found'}
    
    cluster_col = cluster_columns[0]
    clusters = pca_df[cluster_col]
    
    # Calculate cluster separation metrics
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    # For original data (excluding target and cluster columns)
    target_columns = [col for col in original_df.columns if 'chf_exp' in col.lower()]
    exclude_cols = target_columns + cluster_columns
    original_features = [col for col in original_df.columns if col not in exclude_cols]
    X_original = original_df[original_features]
    
    # Standardize original features
    scaler_original = StandardScaler()
    X_original_scaled = scaler_original.fit_transform(X_original)
    
    # For PCA data
    pca_feature_cols = [col for col in pca_df.columns if col.startswith('PC')]
    X_pca = pca_df[pca_feature_cols]
    
    # Calculate metrics
    try:
        # Original data metrics
        silhouette_original = silhouette_score(X_original_scaled, clusters)
        calinski_original = calinski_harabasz_score(X_original_scaled, clusters)
        davies_bouldin_original = davies_bouldin_score(X_original_scaled, clusters)
        
        # PCA data metrics
        silhouette_pca = silhouette_score(X_pca, clusters)
        calinski_pca = calinski_harabasz_score(X_pca, clusters)
        davies_bouldin_pca = davies_bouldin_score(X_pca, clusters)
        
        results = {
            'original_dimensions': X_original.shape[1],
            'pca_dimensions': X_pca.shape[1],
            'dimension_reduction': (X_original.shape[1] - X_pca.shape[1]) / X_original.shape[1],
            'metrics': {
                'original': {
                    'silhouette_score': silhouette_original,
                    'calinski_harabasz_score': calinski_original,
                    'davies_bouldin_score': davies_bouldin_original
                },
                'pca': {
                    'silhouette_score': silhouette_pca,
                    'calinski_harabasz_score': calinski_pca,
                    'davies_bouldin_score': davies_bouldin_pca
                }
            }
        }
        
        # Calculate improvements
        silhouette_improvement = (silhouette_pca - silhouette_original) / abs(silhouette_original)
        calinski_improvement = (calinski_pca - calinski_original) / calinski_original
        davies_bouldin_improvement = (davies_bouldin_original - davies_bouldin_pca) / davies_bouldin_original  # Lower is better
        
        results['improvements'] = {
            'silhouette_improvement_pct': silhouette_improvement * 100,
            'calinski_improvement_pct': calinski_improvement * 100,
            'davies_bouldin_improvement_pct': davies_bouldin_improvement * 100
        }
        
        logger.info(f"\n📊 PCA Impact on Cluster Separation:")
        logger.info(f"  Dimensions: {X_original.shape[1]} → {X_pca.shape[1]} "
                   f"({results['dimension_reduction']*100:.1f}% reduction)")
        logger.info(f"  Silhouette Score: {silhouette_original:.3f} → {silhouette_pca:.3f} "
                   f"({silhouette_improvement*100:+.1f}%)")
        logger.info(f"  Calinski-Harabasz: {calinski_original:.1f} → {calinski_pca:.1f} "
                   f"({calinski_improvement*100:+.1f}%)")
        logger.info(f"  Davies-Bouldin: {davies_bouldin_original:.3f} → {davies_bouldin_pca:.3f} "
                   f"({davies_bouldin_improvement*100:+.1f}%)")
        
        return results
        
    except Exception as e:
        logger.error(f"Error calculating cluster metrics: {e}")
        return {'error': str(e)}


def create_pca_pipeline(
    variance_threshold: float = 0.95,
    use_smote_data: bool = True,
    apply_to_test: bool = True
) -> Dict[str, Any]:
    """
    Complete PCA pipeline: analysis, transformation, and evaluation.
    
    Args:
        variance_threshold: Variance threshold for component selection
        use_smote_data: Whether to use SMOTE-enhanced data
        apply_to_test: Whether to also transform test data
        
    Returns:
        Dictionary with complete pipeline results
    """
    logger.info("🚀 Starting complete PCA pipeline...")
    
    pipeline_results = {}
    
    # Step 1: PCA Analysis
    logger.info("\n=== Step 1: PCA Analysis ===")
    pca_results = apply_pca_analysis(
        variance_threshold=variance_threshold,
        use_smote_data=use_smote_data,
        save_results=True
    )
    pipeline_results['pca_analysis'] = pca_results
    
    # Step 2: Transform test data
    if apply_to_test:
        logger.info("\n=== Step 2: Transform Test Data ===")
        try:
            test_pca_df = apply_pca_to_test_data()
            pipeline_results['test_transformation'] = {
                'success': True,
                'shape': test_pca_df.shape
            }
        except Exception as e:
            logger.error(f"Test data transformation failed: {e}")
            pipeline_results['test_transformation'] = {
                'success': False,
                'error': str(e)
            }
    
    # Step 3: Cluster impact analysis
    logger.info("\n=== Step 3: Cluster Impact Analysis ===")
    try:
        cluster_analysis = analyze_pca_impact_on_clusters()
        pipeline_results['cluster_analysis'] = cluster_analysis
    except Exception as e:
        logger.warning(f"Cluster analysis failed: {e}")
        pipeline_results['cluster_analysis'] = {'error': str(e)}
    
    # Summary
    logger.info("\n🎉 PCA Pipeline Summary:")
    logger.info(f"  Original dimensions: {pca_results['original_shape'][1]}")
    logger.info(f"  PCA dimensions: {pca_results['pca_shape'][1]}")
    logger.info(f"  Variance explained: {sum(pca_results['explained_variance_ratio']):.3f}")
    logger.info(f"  Dimension reduction: {(1 - pca_results['pca_shape'][1]/pca_results['original_shape'][1])*100:.1f}%")
    
    if 'cluster_analysis' in pipeline_results and 'error' not in pipeline_results['cluster_analysis']:
        cluster_results = pipeline_results['cluster_analysis']
        logger.info(f"  Cluster separation impact: "
                   f"Silhouette {cluster_results['improvements']['silhouette_improvement_pct']:+.1f}%, "
                   f"Davies-Bouldin {cluster_results['improvements']['davies_bouldin_improvement_pct']:+.1f}%")
    
    logger.info("✅ PCA pipeline completed successfully!")
    
    return pipeline_results


__all__ = [
    'determine_optimal_components',
    'apply_pca_analysis',
    'apply_pca_to_test_data',
    'get_pca_feature_names',
    'analyze_pca_impact_on_clusters',
    'create_pca_pipeline'
]