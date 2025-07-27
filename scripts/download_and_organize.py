#!/usr/bin/env python3
# scripts/download_and_organize.py
"""
Data Download and Organization Script

This script handles the complete data pipeline:
1. Downloads CHF dataset from Kaggle
2. Performs train/test split with preprocessing
3. Creates EDA plots
4. Performs clustering analysis
5. Applies selected clustering to training data
6. Applies cluster-aware SMOTE
7. Applies PCA for dimensionality reduction
8. Saves processed data

Usage:
    python scripts/download_and_organize.py [--force-download] [--test-size TEST_SIZE] [--skip-smote] [--skip-pca]
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Imports after path setup
from src.utils import setup_logging, get_logger
from src.data import (
    download_dataset, 
    preprocess_data,
)
from src.cluster import (
    perform_clustering,
    apply_selected_clustering,
)
from src.smote import (
    apply_cluster_aware_smote_regression
)
from src.pca import (
    create_pca_pipeline,
    apply_pca_analysis,
)
from src.plotting import create_eda_plots
from src.utils.data_utils import check_data_quality

def main():
    """Main function for data pipeline."""
    parser = argparse.ArgumentParser(
        description="Download and organize CHF dataset with SMOTE and PCA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard pipeline with SMOTE and PCA
  python scripts/download_and_organize.py
  
  # Skip SMOTE but apply PCA
  python scripts/download_and_organize.py --skip-smote
  
  # Skip both SMOTE and PCA
  python scripts/download_and_organize.py --skip-smote --skip-pca
  
  # Custom test size, force download, and specific variance threshold
  python scripts/download_and_organize.py --test-size 0.3 --force-download --variance-threshold 0.90
        """
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download even if data exists'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2)'
    )
    
    parser.add_argument(
        '--skip-viz',
        action='store_true',
        help='Skip visualization step'
    )
    
    parser.add_argument(
        '--skip-clustering',
        action='store_true',
        help='Skip clustering analysis step'
    )
    
    parser.add_argument(
        '--skip-smote',
        action='store_true',
        help='Skip SMOTE application step'
    )
    
    parser.add_argument(
        '--skip-pca',
        action='store_true',
        help='Skip PCA application step'
    )
    
    parser.add_argument(
        '--clustering-algorithm',
        type=str,
        choices=['kmeans', 'hierarchical', 'dbscan'],
        default='kmeans',
        help='Clustering algorithm to use (default: kmeans)'
    )
    
    parser.add_argument(
        '--variance-threshold',
        type=float,
        default=0.95,
        help='PCA variance threshold (default: 0.95 for 95%% variance)'
    )
    
    parser.add_argument(
        '--pca-components',
        type=int,
        default=None,
        help='Fixed number of PCA components (overrides variance threshold)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.variance_threshold <= 0 or args.variance_threshold > 1:
        parser.error("variance-threshold must be between 0 and 1")
    
    if args.pca_components is not None and args.pca_components <= 0:
        parser.error("pca-components must be positive")
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else 'INFO'
    setup_logging(log_level=log_level)
    logger = get_logger(__name__)
    
    logger.info("="*70)
    logger.info("CHF Data Pipeline with SMOTE and PCA")
    logger.info("="*70)
    
    # Step 1: Download data
    logger.info("\n=== Step 1: Data Download ===")
    download_success = download_dataset(force_download=args.force_download)
    
    if not download_success:
        logger.error("Data download failed!")
        sys.exit(1)
    
    # Step 2: Load and preprocess full dataset
    logger.info("\n=== Step 2: Data Preprocessing ===")
    try:
        raw_path = Path('data/raw/heat_flux_data.csv')
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found at {raw_path}")
        
        # Load raw data
        df_raw = pd.read_csv(raw_path)
        logger.info(f"✓ Loaded raw data: {df_raw.shape}")
        
        # Preprocess the full dataset
        from src.data import preprocess_data
        df_processed = preprocess_data(df_raw)
        logger.info(f"✓ Data preprocessed: {df_processed.shape}")
        
        # Save preprocessed full dataset temporarily
        processed_dir = Path('data/processed')
        processed_dir.mkdir(parents=True, exist_ok=True)
        full_processed_path = processed_dir / 'full_processed.csv'
        df_processed.to_csv(full_processed_path, index=False)
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        sys.exit(1)
    
    # Step 3: Data quality check
    logger.info("\n=== Step 3: Data Quality Check ===")
    quality_report = check_data_quality(df_processed)
    
    logger.info(f"Data shape: {quality_report['shape']}")
    logger.info(f"Missing values: {sum(quality_report['missing_values'].values())}")
    logger.info(f"Duplicate rows: {quality_report['duplicates']}")
    
    if quality_report['issues']:
        logger.warning("Data quality issues found:")
        for issue in quality_report['issues']:
            logger.warning(f"  - {issue}")
    else:
        logger.info("✓ No major data quality issues found")
    
    # Step 4: Create visualizations
    if not args.skip_viz:
        logger.info("\n=== Step 4: Creating EDA Plots ===")
        try:
            create_eda_plots()
            logger.info("✓ EDA plots created successfully")
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
    else:
        logger.info("\n=== Step 4: Skipping Visualization ===")
    
    # Step 5: Clustering analysis on full dataset
    clustering_completed = False
    if not args.skip_clustering:
        logger.info("\n=== Step 5: Clustering Analysis on Full Dataset ===")
        try:
            clustering_results = perform_clustering(
                data_path=full_processed_path,
                n_clusters_range=(2, 8),
                save_results=True
            )
            logger.info("✓ Clustering analysis completed successfully")
            
            # Apply selected clustering to full dataset
            df_with_clusters = apply_selected_clustering(
                algorithm=args.clustering_algorithm,
                data_path=full_processed_path,
                save_to_original=False  # Don't overwrite, we'll split next
            )
            
            clustering_completed = True
            logger.info(f"✓ Applied {args.clustering_algorithm} clustering to full dataset")
            
        except Exception as e:
            logger.error(f"Clustering analysis failed: {str(e)}")
            logger.exception("Detailed traceback:")
    else:
        logger.info("\n=== Step 5: Skipping Clustering Analysis ===")
        df_with_clusters = df_processed.copy()
    
    # Step 6: Split data into train/test (preserving cluster distribution if available)
    logger.info("\n=== Step 6: Train/Test Split ===")
    try:
        from sklearn.model_selection import train_test_split
        
        # Identify target column
        target_columns = [col for col in df_with_clusters.columns if 'chf_exp' in col.lower()]
        if not target_columns:
            raise ValueError("No target column found (looking for 'chf_exp')")
        target_col = target_columns[0]
        
        # Stratified split based on clusters if available
        stratify_col = None
        if 'cluster_label' in df_with_clusters.columns and clustering_completed:
            stratify_col = df_with_clusters['cluster_label']
            logger.info("Using cluster-stratified split")
        else:
            logger.info("Using random split")
        
        train_df, test_df = train_test_split(
            df_with_clusters,
            test_size=args.test_size,
            random_state=42,
            stratify=stratify_col
        )
        
        # Save splits
        train_path = processed_dir / 'train.csv'
        test_path = processed_dir / 'test.csv'
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"✓ Data split completed")
        logger.info(f"  Train: {train_path} ({train_df.shape})")
        logger.info(f"  Test: {test_path} ({test_df.shape})")
        
        if clustering_completed:
            logger.info("  Cluster distribution in train:")
            cluster_dist = train_df['cluster_label'].value_counts().sort_index()
            for cluster_id, count in cluster_dist.items():
                logger.info(f"    Cluster {cluster_id}: {count} samples")
        
    except Exception as e:
        logger.error(f"Data splitting failed: {str(e)}")
        sys.exit(1)
    
    # Step 7: Apply SMOTE to training data only
    smote_completed = False
    if not args.skip_smote and clustering_completed:
        logger.info("\n=== Step 7: Applying Cluster-Aware SMOTE to Training Data ===")
        try:
            # Separate features and target from training data
            X_train = train_df.drop([target_col, 'cluster_label'], axis=1)
            y_train = train_df[target_col]
            cluster_labels_train = train_df['cluster_label']
            
            # Apply cluster-aware SMOTE
            X_resampled, y_resampled, cluster_labels_resampled = apply_cluster_aware_smote_regression(
                X_train, y_train, cluster_labels_train,
                sampling_strategy='auto',
                random_state=42
            )
            
            # Create resampled training DataFrame
            resampled_train_df = pd.DataFrame(X_resampled, columns=X_train.columns)
            resampled_train_df[target_col] = y_resampled
            resampled_train_df['cluster_label'] = cluster_labels_resampled
            
            # Save resampled training data
            resampled_path = processed_dir / 'train_resampled.csv'
            resampled_train_df.to_csv(resampled_path, index=False)
            
            logger.info(f"✓ SMOTE applied successfully")
            logger.info(f"  Original train size: {len(train_df)}")
            logger.info(f"  Resampled train size: {len(resampled_train_df)}")
            logger.info(f"  Synthetic samples added: {len(resampled_train_df) - len(train_df)}")
            logger.info(f"  Resampled data saved to: {resampled_path}")
            
            # Show cluster distribution after SMOTE
            logger.info("  Cluster distribution after SMOTE:")
            cluster_dist_smote = resampled_train_df['cluster_label'].value_counts().sort_index()
            for cluster_id, count in cluster_dist_smote.items():
                logger.info(f"    Cluster {cluster_id}: {count} samples")
            
            smote_completed = True
            
        except Exception as e:
            logger.error(f"SMOTE application failed: {str(e)}")
            logger.exception("Detailed traceback:")
    else:
        skip_reason = "skipped by user" if args.skip_smote else "clustering not completed"
        logger.info(f"\n=== Step 7: Skipping SMOTE Application ({skip_reason}) ===")
    
    # Step 8: Apply PCA for dimensionality reduction
    pca_completed = False
    if not args.skip_pca:
        logger.info("\n=== Step 8: Applying PCA for Dimensionality Reduction ===")
        try:
            # Determine which data to use for PCA
            use_smote_data = smote_completed
            
            if use_smote_data:
                logger.info("Using SMOTE-enhanced data for PCA")
            else:
                logger.info("Using original training data for PCA")
            
            # Apply PCA pipeline
            if args.pca_components is not None:
                # Use fixed number of components
                logger.info(f"Using fixed {args.pca_components} PCA components")
                pca_results = apply_pca_analysis(
                    n_components=args.pca_components,
                    use_smote_data=use_smote_data,
                    save_results=True
                )
            else:
                # Use complete PCA pipeline with variance threshold
                logger.info(f"Using PCA pipeline with {args.variance_threshold*100:.1f}% variance threshold")
                pca_results = create_pca_pipeline(
                    variance_threshold=args.variance_threshold,
                    use_smote_data=use_smote_data,
                    apply_to_test=True
                )
                
                # Extract the main PCA results from pipeline
                if 'pca_analysis' in pca_results:
                    pca_analysis = pca_results['pca_analysis']
                else:
                    pca_analysis = pca_results
            
            # Log PCA results
            if args.pca_components is not None:
                pca_analysis = pca_results
            
            original_dims = pca_analysis['original_shape'][1]
            pca_dims = pca_analysis['pca_shape'][1]
            variance_explained = sum(pca_analysis['explained_variance_ratio'])
            dimension_reduction = (original_dims - pca_dims) / original_dims
            
            logger.info("✓ PCA applied successfully")
            logger.info(f"  Original dimensions: {original_dims}")
            logger.info(f"  PCA dimensions: {pca_dims}")
            logger.info(f"  Dimension reduction: {dimension_reduction*100:.1f}%")
            logger.info(f"  Variance explained: {variance_explained:.3f} ({variance_explained*100:.1f}%)")
            
            # Show top contributing features
            if 'feature_importance' in pca_analysis:
                logger.info("  Top 5 most important features:")
                for i, feature_info in enumerate(pca_analysis['feature_importance'][:5]):
                    logger.info(f"    {i+1}. {feature_info['feature']}: {feature_info['importance']:.4f}")
            
            # Log cluster impact if available
            if args.pca_components is None and 'cluster_analysis' in pca_results:
                cluster_analysis = pca_results['cluster_analysis']
                if 'error' not in cluster_analysis:
                    silhouette_improvement = cluster_analysis['improvements']['silhouette_improvement_pct']
                    davies_bouldin_improvement = cluster_analysis['improvements']['davies_bouldin_improvement_pct']
                    logger.info(f"  Cluster separation impact:")
                    logger.info(f"    Silhouette score: {silhouette_improvement:+.1f}%")
                    logger.info(f"    Davies-Bouldin score: {davies_bouldin_improvement:+.1f}%")
            
            pca_completed = True
            
        except Exception as e:
            logger.error(f"PCA application failed: {str(e)}")
            logger.exception("Detailed traceback:")
    else:
        logger.info("\n=== Step 8: Skipping PCA Application ===")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("Pipeline Summary")
    logger.info("="*70)
    logger.info("✓ Data downloaded and organized")
    logger.info(f"✓ Train/test split created (test size: {args.test_size})")
    logger.info("✓ Data preprocessed and encoded")
    
    if not args.skip_viz:
        logger.info("✓ EDA plots generated")
    
    if clustering_completed:
        logger.info(f"✓ {args.clustering_algorithm.upper()} clustering performed")
    
    if smote_completed:
        logger.info("✓ Cluster-aware SMOTE applied")
    
    if pca_completed:
        logger.info("✓ PCA dimensionality reduction applied")
    
    # Final recommendations
    logger.info("\n" + "="*60)
    logger.info("Next Steps & Available Data")
    logger.info("="*60)
    
    # List available data files
    available_files = []
    
    if pca_completed:
        pca_train_path = Path('data/pca/pca_transformed_data.csv')
        pca_test_path = Path('data/pca/pca_test_data.csv')
        
        if pca_train_path.exists() and pca_test_path.exists():
            available_files.append("🎯 PCA-transformed data (recommended for modeling):")
            available_files.append(f"   • Training: {pca_train_path}")
            available_files.append(f"   • Test: {pca_test_path}")
            available_files.append("")
            available_files.append("🚀 To load PCA data in your model:")
            available_files.append("   train_df = pd.read_csv('data/pca/pca_transformed_data.csv')")
            available_files.append("   test_df = pd.read_csv('data/pca/pca_test_data.csv')")
            available_files.append("")
    
    if smote_completed:
        available_files.append("💡 SMOTE-enhanced data (if you prefer full dimensionality):")
        available_files.append("   • Training: data/processed/train_resampled.csv")
        available_files.append("   • Test: data/processed/test.csv")
        available_files.append("")
    
    if clustering_completed:
        available_files.append("📊 Original clustered data:")
        available_files.append("   • Training: data/processed/train.csv")
        available_files.append("   • Test: data/processed/test.csv")
        available_files.append("")
    
    if available_files:
        for line in available_files:
            logger.info(line)
    
    # Provide recommendations based on what was completed
    logger.info("📋 Recommendations:")
    
    if pca_completed:
        logger.info("   1. Use PCA-transformed data for faster training and potentially better generalization")
        logger.info("   2. Check feature importance rankings in: data/pca/feature_importance.csv")
        logger.info("   3. Review PCA analysis results in: data/pca/pca_analysis.json")
    elif smote_completed:
        logger.info("   1. Consider running PCA for dimensionality reduction:")
        logger.info(f"      python scripts/download_and_organize.py --clustering-algorithm {args.clustering_algorithm}")
        logger.info("   2. Current SMOTE-enhanced data is ready for modeling")
    elif clustering_completed:
        logger.info("   1. Consider running with SMOTE and PCA:")
        logger.info(f"      python scripts/download_and_organize.py --clustering-algorithm {args.clustering_algorithm}")
        logger.info("   2. Current clustered data is ready for basic modeling")
    else:
        logger.info("   1. Run full pipeline for optimal results:")
        logger.info("      python scripts/download_and_organize.py --clustering-algorithm kmeans")
    
    # Additional data files available
    other_files = []
    
    if clustering_completed:
        other_files.append("🔍 Analysis files available:")
        other_files.append("   • Clustering: data/clustering/cluster_assignments.csv")
        other_files.append("   • Clustering metrics: data/clustering/clustering_metrics.json")
    
    if pca_completed:
        other_files.append("   • PCA models: data/pca/pca_model.joblib, data/pca/pca_scaler.joblib")
    
    if not args.skip_viz:
        other_files.append("   • Visualizations: data/plots/ (various EDA plots)")
    
    if other_files:
        logger.info("")
        for line in other_files:
            logger.info(line)
    
    logger.info(f"\n✅ Data pipeline completed successfully!")
    
    # Final statistics
    final_stats = []
    if pca_completed:
        try:
            pca_train_df = pd.read_csv('data/pca/pca_transformed_data.csv')
            final_stats.append(f"📈 Final dataset ready for modeling: {pca_train_df.shape}")
            
            # Count features vs target/cluster columns
            feature_cols = [col for col in pca_train_df.columns 
                          if not any(x in col.lower() for x in ['chf_exp', 'cluster'])]
            final_stats.append(f"   • Features: {len(feature_cols)} (PCA components)")
            
            if any('chf_exp' in col.lower() for col in pca_train_df.columns):
                final_stats.append("   • Target: 1 (CHF values)")
            
            if any('cluster' in col.lower() for col in pca_train_df.columns):
                final_stats.append("   • Cluster labels: 1 (for analysis)")
            
        except:
            pass
    
    if final_stats:
        logger.info("")
        for stat in final_stats:
            logger.info(stat)

if __name__ == "__main__":
    main()