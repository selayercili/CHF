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
6. Saves processed data

Usage:
    python scripts/download_and_organize.py [--force-download] [--test-size TEST_SIZE]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Imports after path setup
from src.utils import setup_logging, get_logger
from src.data import download_dataset, split_data, create_eda_plots, check_data_quality, perform_clustering
from src.utils.data_utils import save_data_splits


def main():
    """Main function for data pipeline."""
    parser = argparse.ArgumentParser(
        description="Download and organize CHF dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard pipeline
  python scripts/download_and_organize.py
  
  # Force re-download
  python scripts/download_and_organize.py --force-download
  
  # Custom test size
  python scripts/download_and_organize.py --test-size 0.3
  
  # Skip visualization
  python scripts/download_and_organize.py --skip-viz
  
  # Skip clustering
  python scripts/download_and_organize.py --skip-clustering
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
    logger.info("CHF Data Pipeline")
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
            # Don't exit - visualization is optional
    else:
        logger.info("\n=== Step 4: Skipping Visualization ===")
    
    # Step 5: Clustering analysis
    if not args.skip_clustering:
        logger.info("\n=== Step 5: Clustering Analysis ===")
        try:
            clustering_results = perform_clustering(
                data_path=train_path,
                n_clusters_range=(2, 8),
                save_results=True
            )
            logger.info("✓ Clustering analysis completed successfully")
            
            # Log summary of clustering results
            kmeans_optimal = clustering_results['algorithms']['kmeans']['optimal_k']
            hierarchical_optimal = clustering_results['algorithms']['hierarchical']['optimal_k']
            logger.info(f"  - K-Means optimal clusters: {kmeans_optimal}")
            logger.info(f"  - Hierarchical optimal clusters: {hierarchical_optimal}")
            
            dbscan_best_eps = clustering_results['algorithms']['dbscan']['best_eps']
            if dbscan_best_eps:
                dbscan_clusters = clustering_results['algorithms']['dbscan']['results'][dbscan_best_eps]['n_clusters']
                logger.info(f"  - DBSCAN optimal: eps={dbscan_best_eps}, clusters={dbscan_clusters}")
            else:
                logger.info("  - DBSCAN: No valid clustering found")
            
        except Exception as e:
            logger.error(f"Clustering analysis failed: {str(e)}")
            # Don't exit - clustering is optional
    else:
        logger.info("\n=== Step 5: Skipping Clustering Analysis ===")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Pipeline Summary")
    logger.info("="*60)
    logger.info("✓ Data downloaded and organized")
    logger.info("✓ Train/test split created")
    logger.info("✓ Data preprocessed and encoded")
    if not args.skip_viz:
        logger.info("✓ EDA plots generated")
    if not args.skip_clustering:
        logger.info("✓ Clustering analysis performed")
    
    logger.info("\n✅ Data pipeline completed successfully!")
    logger.info("Ready for model training - run: python scripts/train.py")


if __name__ == "__main__":
    main()