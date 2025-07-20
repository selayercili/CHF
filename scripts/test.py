#!/usr/bin/env python3
# scripts/test.py
"""
Model Testing and Evaluation Script

This script loads trained models and evaluates them on test data.
It saves predictions and metrics for further analysis.

Usage:
    python scripts/test.py [--config CONFIG_PATH] [--models MODEL_NAMES] [--weights-type WEIGHTS_TYPE]
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Imports after path setup
from src.utils import (
    setup_logging, get_logger, load_config,
    CheckpointManager, MetricsTracker, ConfigManager
)
from src.models import model_registry
from src.utils.metrics import calculate_metrics

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class ModelTester:
    """Handles model testing and evaluation pipeline."""
    
    def __init__(self, config_path: Optional[Path] = None, debug: bool = False):
        """
        Initialize the ModelTester.
        
        Args:
            config_path: Path to configuration file
            debug: Enable debug logging
        """
        # Setup logging
        log_level = 'DEBUG' if debug else 'INFO'
        setup_logging(log_level=log_level)
        self.logger = get_logger(__name__)
        
        # Load configurations
        self.config_manager = ConfigManager(config_path or Path("configs"))
        self.model_config = self.config_manager.get('model_configs')
        
        # Setup directories
        self.dirs = {
            'weights': Path("weights"),
            'results': Path("results"),
            'data': Path("data/processed")
        }
        
        # Create results directory
        self.dirs['results'].mkdir(exist_ok=True)
        
        self.logger.info("="*60)
        self.logger.info("Model Tester Initialized")
        self.logger.info(f"Config path: {config_path}")
        self.logger.info("="*60)
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data."""
        self.logger.info("Loading test data...")
        
        test_path = self.dirs['data'] / "test.csv"
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        test_df = pd.read_csv(test_path)
        self.logger.info(f"Loaded test data: {test_df.shape}")
        
        # Data quality check
        n_missing = test_df.isnull().sum().sum()
        if n_missing > 0:
            self.logger.warning(f"Test data contains {n_missing} missing values")
        
        return test_df
    
    def get_model_instance(self, model_name: str) -> Any:
        """Get model instance from registry."""
        if model_name not in model_registry:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = model_registry[model_name]
        model_config = self.model_config.get(model_name, {})
        init_params = model_config.get('init_params', {})
        
        # Special handling for neural networks
        if model_name in ['neural_network', 'pinn']:
            if 'architecture' in model_config:
                arch_config = model_config['architecture']
                if 'hidden_layers' in arch_config:
                    init_params['hidden_layers'] = arch_config['hidden_layers']
        
        return model_class(**init_params)
    
    def find_best_weights(self, model_name: str, weights_type: str = "best") -> Optional[Path]:
        """
        Find the best weights file for a model.
        
        Args:
            model_name: Name of the model
            weights_type: Type of weights to load ("best" or "latest")
            
        Returns:
            Path to weights file or None
        """
        model_weights_dir = self.dirs['weights'] / model_name
        
        if not model_weights_dir.exists():
            self.logger.warning(f"No weights directory found for {model_name}")
            return None
        
        if weights_type == "best":
            # Look for best_model.pth
            best_path = model_weights_dir / "best_model.pth"
            if best_path.exists():
                return best_path
            else:
                self.logger.warning(f"No best_model.pth found for {model_name}, trying latest")
                weights_type = "latest"
        
        if weights_type == "latest":
            # Find the latest epoch
            epoch_files = list(model_weights_dir.glob("epoch_*.pth"))
            if not epoch_files:
                self.logger.warning(f"No epoch files found for {model_name}")
                return None
            
            # Sort by epoch number
            epoch_files.sort(key=lambda x: int(x.stem.split('_')[1]))
            return epoch_files[-1]
        
        return None
    
    def test_model(self, model_name: str, test_data: pd.DataFrame, 
                   weights_type: str = "best") -> Optional[Dict[str, Any]]:
        """
        Test a single model.
        
        Args:
            model_name: Name of the model to test
            test_data: Test dataframe
            weights_type: Type of weights to load
            
        Returns:
            Dictionary containing predictions and metrics
        """
        self.logger.info(f"\nTesting {model_name}...")
        
        # Find weights
        weights_path = self.find_best_weights(model_name, weights_type)
        if weights_path is None:
            self.logger.error(f"No weights found for {model_name}")
            return None
        
        self.logger.info(f"Loading weights from: {weights_path}")
        
        # Create model instance
        try:
            model = self.get_model_instance(model_name)
        except Exception as e:
            self.logger.error(f"Failed to create model {model_name}: {str(e)}")
            return None
        
        # Load weights
        try:
            if hasattr(model, 'load'):
                # Model has custom load method
                metadata = model.load(weights_path)
                self.logger.info(f"Loaded model from epoch: {model.epoch if hasattr(model, 'epoch') else 'unknown'}")
            else:
                # Generic loading
                import torch    
                checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
                
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    model = checkpoint['model']
                else:
                    self.logger.error(f"Unknown checkpoint format for {model_name}")
                    return None
                
                metadata = checkpoint.get('metadata', {})
        
        except Exception as e:
            self.logger.error(f"Failed to load weights for {model_name}: {str(e)}")
            return None
        
        # Make predictions
        try:
            # Timer for inference
            import time
            start_time = time.time()
            
            # Get predictions
            predictions = model.predict(test_data)
            
            inference_time = time.time() - start_time
            
            # Get true values (last column)
            y_true = test_data.iloc[:, -1].values
            
            # Calculate metrics
            metrics = calculate_metrics(y_true, predictions, task='regression')
            metrics['inference_time'] = inference_time
            metrics['samples_per_second'] = len(test_data) / inference_time
            
            # Additional metrics
            errors = y_true - predictions
            percentiles = [5, 25, 50, 75, 95]
            for p in percentiles:
                metrics[f'error_p{p}'] = np.percentile(np.abs(errors), p)
            
            # Prepare results
            results = {
                'model_name': model_name,
                'weights_path': str(weights_path),
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'predictions': predictions,
                'actual': y_true,
                'test_size': len(test_data),
                'model_config': self.model_config.get(model_name, {})
            }
            
            # Add feature importance if available
            if hasattr(model, 'get_feature_importance'):
                try:
                    results['feature_importance'] = model.get_feature_importance()
                except:
                    self.logger.warning(f"Failed to get feature importance for {model_name}")
            
            # Log summary metrics
            self.logger.info(f"Test metrics for {model_name}:")
            self.logger.info(f"  RMSE: {metrics['rmse']:.6f}")
            self.logger.info(f"  MAE: {metrics['mae']:.6f}")
            self.logger.info(f"  R²: {metrics['r2']:.6f}")
            self.logger.info(f"  Inference time: {inference_time:.3f}s ({metrics['samples_per_second']:.1f} samples/s)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during testing {model_name}: {str(e)}")
            self.logger.exception("Detailed traceback:")
            return None
    
    def save_results(self, results: Dict[str, Any], model_name: str) -> None:
        """Save test results and predictions."""
        if results is None:
            return
        
        # Create model results directory
        model_results_dir = self.dirs['results'] / model_name
        model_results_dir.mkdir(exist_ok=True)
        
        # Save predictions as CSV
        predictions_df = pd.DataFrame({
            'actual': results['actual'],
            'predicted': results['predictions'],
            'error': results['actual'] - results['predictions'],
            'abs_error': np.abs(results['actual'] - results['predictions']),
            'percent_error': 100 * np.abs(results['actual'] - results['predictions']) / results['actual']
        })
        
        predictions_path = model_results_dir / 'predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        self.logger.info(f"Saved predictions to: {predictions_path}")
        
        # Save metrics as JSON
        metrics_data = {
            'model_name': results['model_name'],
            'weights_path': results['weights_path'],
            'timestamp': results['timestamp'],
            'metrics': results['metrics'],
            'test_size': results['test_size']
        }
        
        metrics_path = model_results_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        self.logger.info(f"Saved metrics to: {metrics_path}")
        
        # Save feature importance if available
        if 'feature_importance' in results and results['feature_importance'] is not None:
            importance_path = model_results_dir / 'feature_importance.csv'
            results['feature_importance'].to_csv(importance_path, index=False)
            self.logger.info(f"Saved feature importance to: {importance_path}")
        
        # Save full results as pickle
        results_pickle_path = model_results_dir / 'full_results.pkl'
        with open(results_pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary statistics
        summary_stats = {
            'prediction_stats': {
                'mean': float(predictions_df['predicted'].mean()),
                'std': float(predictions_df['predicted'].std()),
                'min': float(predictions_df['predicted'].min()),
                'max': float(predictions_df['predicted'].max())
            },
            'error_stats': {
                'mean_error': float(predictions_df['error'].mean()),
                'mean_abs_error': float(predictions_df['abs_error'].mean()),
                'mean_percent_error': float(predictions_df['percent_error'].mean()),
                'std_error': float(predictions_df['error'].std())
            }
        }
        
        summary_path = model_results_dir / 'summary_stats.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
    
    def test_all_models(self, model_names: Optional[List[str]] = None,
                       weights_type: str = "best") -> Dict[str, Any]:
        """
        Test all specified models.
        
        Args:
            model_names: List of model names to test (None for all)
            weights_type: Type of weights to load
            
        Returns:
            Dictionary of all results
        """
        # Load test data once
        test_data = self.load_test_data()
        
        # Get list of models to test
        if model_names is None:
            # Find models with saved weights
            model_names = []
            for model_dir in self.dirs['weights'].iterdir():
                if model_dir.is_dir() and model_dir.name in model_registry:
                    model_names.append(model_dir.name)
        
        self.logger.info(f"\nFound {len(model_names)} models to test: {model_names}")
        
        # Test each model
        all_results = {}
        
        for model_name in tqdm(model_names, desc="Testing models"):
            try:
                results = self.test_model(model_name, test_data, weights_type)
                if results:
                    all_results[model_name] = results
                    self.save_results(results, model_name)
            except KeyboardInterrupt:
                self.logger.warning("Testing interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Failed to test {model_name}: {str(e)}")
                continue
        
        # Save comparison summary
        self.save_comparison_summary(all_results)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("TESTING COMPLETED")
        self.logger.info("="*60)
        
        return all_results
    
    def save_comparison_summary(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Save summary comparison of all models."""
        if not all_results:
            return
        
        # Create comparison data
        comparison_data = []
        
        for model_name, results in all_results.items():
            row = {
                'model': model_name,
                'timestamp': results['timestamp'],
                **results['metrics']
            }
            comparison_data.append(row)
        
        # Save as CSV
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = self.dirs['results'] / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        self.logger.info(f"Saved model comparison to: {comparison_path}")
        
        # Save as markdown
        markdown_path = self.dirs['results'] / 'model_comparison.md'
        with open(markdown_path, 'w') as f:
            f.write("# Model Comparison Results\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Sort by RMSE
            comparison_df = comparison_df.sort_values('rmse')
            
            # Key metrics table
            f.write("## Key Metrics\n\n")
            key_metrics = ['rmse', 'mae', 'r2', 'max_error']
            available_metrics = [m for m in key_metrics if m in comparison_df.columns]
            
            if available_metrics:
                subset_df = comparison_df[['model'] + available_metrics]
                f.write(subset_df.to_markdown(index=False))
            
            f.write("\n\n## Best Performing Models\n\n")
            f.write(f"- **Lowest RMSE**: {comparison_df.iloc[0]['model']} ({comparison_df.iloc[0]['rmse']:.6f})\n")
            
            if 'mae' in comparison_df.columns:
                best_mae = comparison_df.loc[comparison_df['mae'].idxmin()]
                f.write(f"- **Lowest MAE**: {best_mae['model']} ({best_mae['mae']:.6f})\n")
            
            if 'r2' in comparison_df.columns:
                best_r2 = comparison_df.loc[comparison_df['r2'].idxmax()]
                f.write(f"- **Highest R²**: {best_r2['model']} ({best_r2['r2']:.6f})\n")
        
        # Print summary to console
        self.logger.info("\nModel Comparison Summary:")
        print(comparison_df[['model', 'rmse', 'mae', 'r2']].to_string(index=False))


def main():
    """Main entry point for the testing script."""
    parser = argparse.ArgumentParser(
        description="Test and evaluate trained CHF models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all models with best weights
  python scripts/test.py
  
  # Test specific models
  python scripts/test.py --models xgboost lightgbm
  
  # Use latest weights instead of best
  python scripts/test.py --weights-type latest
  
  # Debug mode
  python scripts/test.py --debug
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs',
        help='Path to configuration directory'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        type=str,
        help='Specific models to test (default: all with weights)'
    )
    
    parser.add_argument(
        '--weights-type',
        type=str,
        choices=['best', 'latest'],
        default='best',
        help='Type of weights to load'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    config_path = Path(args.config)
    tester = ModelTester(config_path=config_path, debug=args.debug)
    
    # Run testing
    try:
        tester.test_all_models(
            model_names=args.models,
            weights_type=args.weights_type
        )
    except KeyboardInterrupt:
        tester.logger.warning("\nTesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        tester.logger.error(f"Testing failed: {str(e)}")
        if args.debug:
            tester.logger.exception("Detailed traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()