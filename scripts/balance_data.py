#!/usr/bin/env python3
# scripts/enhance_data.py
"""
Data Enhancement Script

This script handles:
1. Performs clustering analysis
2. Applies selected clustering
3. Applies cluster-aware SMOTE
4. Saves enhanced datasets

Usage:
    python scripts/enhance_data.py [--clustering-algorithm ALGORITHM] [--skip-smote] [--debug]
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
    apply_selected_clustering,
    apply_smote_to_train_data
)
from src.cluster import perform_clustering

def main():
    """Main function for data enhancement pipeline."""
    parser = argparse.ArgumentParser(
        description="Enhance CHF data with clustering and SMOTE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard enhancement with KMeans and SMOTE
  python scripts/enhance_data.py --clustering-algorithm kmeans
  
  # Skip SMOTE application
  python scripts/enhance_data.py --skip-smote
  
  # Use hierarchical clustering
  python scripts/enhance_data.py --clustering-algorithm hierarchical
        """
    )
    
    parser.add_argument(
        '--clustering-algorithm',
        type=str,
        choices=['kmeans', 'hierarchical', 'dbscan'],
        default='kmeans',
        help='Clustering algorithm to use (default: kmeans)'
    )
    
    parser.add_argument(
        '--skip-smote',
        action='store_true',
        help='Skip SMOTE application step'
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
    logger.info("CHF Data Enhancement Pipeline")
    logger.info("="*60)
    
    # Step 1: Clustering analysis
    logger.info("\n=== Step 1: Clustering Analysis ===")
    try:
        train_path = Path('data/processed/train.csv')
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
        sys.exit(1)
    
    # Step 2: Apply SMOTE
    smote_completed = False
    if not args.skip_smote and clustering_completed:
        logger.info("\n=== Step 2: Applying Cluster-Aware SMOTE ===")
        try:
            resampled_train_df, test_df = apply_smote_to_train_data(
                algorithm=args.clustering_algorithm,
                random_state=42
            )
            
            logger.info(f"✓ SMOTE applied successfully")
            logger.info(f"  Original train size: {len(train_df_with_clusters)}")
            logger.info(f"  Resampled train size: {len(resampled_train_df)}")
            logger.info(f"  Test size remains: {len(test_df)}")
            
            smote_completed = True
            
        except Exception as e:
            logger.error(f"SMOTE application failed: {str(e)}")
    else:
        skip_reason = "skipped by user" if args.skip_smote else "clustering not completed"
        logger.info(f"\n=== Step 2: Skipping SMOTE Application ({skip_reason}) ===")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Pipeline Summary")
    logger.info("="*60)
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
    else:
        logger.info("💡 Run without --skip-smote to apply SMOTE:")
        logger.info(f"   python scripts/enhance_data.py --clustering-algorithm {args.clustering_algorithm}")
    
    logger.info("\n✅ Data enhancement completed successfully!")

if __name__ == "__main__":
    main()