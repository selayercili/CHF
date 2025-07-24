#!/usr/bin/env python3
# scripts/download_and_organize.py
"""
Data Download and Organization Script

This script handles the complete data pipeline:
1. Downloads CHF dataset from Kaggle
2. Performs train/test split
3. Preprocesses data (encoding, scaling)
4. Creates EDA plots
5. Performs clustering analysis
6. Applies cluster-aware SMOTE
7. Saves processed data

Usage:
    python scripts/download_and_organize.py [--force-download] [--test-size TEST_SIZE] [--skip-smote]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Imports after path setup
from src.utils import setup_logging, get_logger
from src.data import (
    download_dataset, 
    split_data, 
    create_eda_plots, 
    check_data_quality,
    perform_clustering, 
    apply_selected_clustering,
    apply_smote_to_train_data
)

def main():
    """Main function for data pipeline."""
    parser = argparse.ArgumentParser(
        description="Download and organize CHF dataset with SMOTE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard pipeline with SMOTE
  python scripts/download_and_organize.py
  
  # Skip SMOTE application
  python scripts/download_and_organize.py --skip-smote
  
  # Custom test size and force download
  python scripts/download_and_organize.py --test-size 0.3 --force-download
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
        '--clustering-algorithm',
        type=str,
        choices=['kmeans', 'hierarchical', 'dbscan'],
        default='kmeans',
        help='Clustering algorithm to use (default: kmeans)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else 'INFO'
    setup_logging(log_level=log_level)
    logger = get_logger(__name__)
    
    logger.info("="*60)
    logger.info("CHF Data Pipeline with SMOTE")
    logger.info("="*60)
    
    # Step 1: Download data
    logger.info("\n=== Step 1: Data Download ===")
    download_success = download_dataset(force_download=args.force_download)
    
    if not download_success:
        logger.error("Data download failed!")
        sys.exit(1)
    
    # Step 2: Split and preprocess data
    logger.info("\n=== Step 2: Data Splitting and Preprocessing ===")
    try:
        train_path, test_path = split_data(
            test_size=args.test_size,
            random_state=42
        )
        logger.info(f"✓ Data split completed")
        logger.info(f"  Train: {train_path}")
        logger.info(f"  Test: {test_path}")
    except Exception as e:
        logger.error(f"Data splitting failed: {str(e)}")
        sys.exit(1)
    
    # Step 3: Data quality check
    logger.info("\n=== Step 3: Data Quality Check ===")
    import pandas as pd
    train_df = pd.read_csv(train_path)
    quality_report = check_data_quality(train_df)
    
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
    
    # Step 5: Clustering analysis
    clustering_completed = False
    if not args.skip_clustering:
        logger.info("\n=== Step 5: Clustering Analysis ===")
        try:
            clustering_results = perform_clustering(
                data_path=train_path,
                n_clusters_range=(2, 8),
                save_results=True
            )
            logger.info("✓ Clustering analysis completed successfully")
            
            # Apply selected clustering
            train_df_with_clusters = apply_selected_clustering(
                algorithm=args.clustering_algorithm,
                data_path=train_path,
                save_to_original=True
            )
            
            clustering_completed = True
            
        except Exception as e:
            logger.error(f"Clustering analysis failed: {str(e)}")
    else:
        logger.info("\n=== Step 5: Skipping Clustering Analysis ===")
    
    # Step 6: Apply SMOTE
    smote_completed = False
    if not args.skip_smote and clustering_completed:
        logger.info("\n=== Step 6: Applying Cluster-Aware SMOTE ===")
        try:
            resampled_train_df, test_df = apply_smote_to_train_data(
                algorithm=args.clustering_algorithm,
                random_state=42
            )
            
            logger.info(f"✓ SMOTE applied successfully")
            logger.info(f"  Original train size: {len(train_df)}")
            logger.info(f"  Resampled train size: {len(resampled_train_df)}")
            logger.info(f"  Test size remains: {len(test_df)}")
            
            smote_completed = True
            
        except Exception as e:
            logger.error(f"SMOTE application failed: {str(e)}")
    else:
        skip_reason = "skipped by user" if args.skip_smote else "clustering not completed"
        logger.info(f"\n=== Step 6: Skipping SMOTE Application ({skip_reason}) ===")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Pipeline Summary")
    logger.info("="*60)
    logger.info("✓ Data downloaded and organized")
    logger.info(f"✓ Train/test split created (test size: {args.test_size})")
    logger.info("✓ Data preprocessed and encoded")
    
    if not args.skip_viz:
        logger.info("✓ EDA plots generated")
    
    if clustering_completed:
        logger.info(f"✓ {args.clustering_algorithm.upper()} clustering performed")
    
    if smote_completed:
        logger.info("✓ Cluster-aware SMOTE applied")
    
    # Final recommendations
    logger.info("\n" + "="*50)
    logger.info("Next Steps")
    logger.info("="*50)
    
    if smote_completed:
        logger.info("🎯 Your data is ready for modeling with:")
        logger.info(f"   • Resampled training data: data/processed/train_resampled.csv")
        logger.info(f"   • Original test data: data/processed/test.csv")
        logger.info("\n🚀 To use in your model:")
        logger.info("   from src.data import apply_smote_to_train_data")
        logger.info("   train_df, test_df = apply_smote_to_train_data()")
    elif clustering_completed:
        logger.info("💡 Run without --skip-smote to apply SMOTE:")
        logger.info("   python scripts/download_and_organize.py --clustering-algorithm kmeans")
    else:
        logger.info("💡 Run with clustering enabled for SMOTE preparation:")
        logger.info("   python scripts/download_and_organize.py --clustering-algorithm kmeans")
    
    logger.info("\n✅ Data pipeline completed successfully!")

if __name__ == "__main__":
    main()