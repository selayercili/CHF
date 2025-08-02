#!/usr/bin/env python3
# scripts/test.py
"""
Model Testing and Evaluation Script with SMOTE/Regular Data Support

This script loads trained models and evaluates them on test data.
It saves predictions and metrics for further analysis.

Usage:
    python scripts/test.py [--config CONFIG_PATH] [--models MODEL_NAMES] [--data-type DATA_TYPE] [--weights-type WEIGHTS_TYPE]
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
import torch

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
    
    def __init__(self, config_path: Optional[Path] = None, debug: bool = False, data_type: str = 'both'):
        """
        Initialize the ModelTester.
        
        Args:
            config_path: Path to configuration file
            debug: Enable debug logging
            data_type: Type of models to test ('smote', 'regular', or 'both')
        """
        # Setup logging
        log_level = 'DEBUG' if debug else 'INFO'
        setup_logging(log_level=log_level)
        self.logger = get_logger(__name__)
        
        # Load configurations
        self.config_manager = ConfigManager(config_path or Path("configs"))
        self.model_config = self.config_manager.get('model_configs')
        
        # Store data type
        self.data_type = data_type
        
        # Setup directories based on data type
        self.setup_directories()
        
        self.logger.info("="*60)
        self.logger.info("Model Tester Initialized")
        self.logger.info(f"Config path: {config_path}")
        self.logger.info(f"Data type: {data_type}")
        self.logger.info("="*60)
    
    def setup_directories(self):
        """Setup directories based on data type."""
        if self.data_type == 'both':
            self.weights_dirs = {
                'smote': Path("weights_smote"),
                'regular': Path("weights_regular")
            }
            self.results_dirs = {
                'smote': Path("results_smote"),
                'regular': Path("results_regular")
            }
        elif self.data_type == 'smote':
            self.weights_dirs = {'smote': Path("weights_smote")}
            self.results_dirs = {'smote': Path("results_smote")}
        else:  # regular
            self.weights_dirs = {'regular': Path("weights_regular")}
            self.results_dirs = {'regular': Path("results_regular")}
        
        # Create results directories
        for results_dir in self.results_dirs.values():
            results_dir.mkdir(exist_ok=True)
        
        # Data directory
        self.data_dir = Path("data/processed")
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data."""
        self.logger.info("Loading test data...")
        
        test_path = self.data_dir / "test.csv"
        
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
    
    def find_best_weights(self, model_name: str, weights_dir: Path, weights_type: str = "best") -> Optional[Path]:
        """
        Find the best weights file for a model.
        
        Args:
            model_name: Name of the model
            weights_dir: Directory containing weights
            weights_type: Type of weights to load ("best" or "latest")
            
        Returns:
            Path to weights file or None
        """
        model_weights_dir = weights_dir / model_name
        
        if not model_weights_dir.exists():
            self.logger.warning(f"No weights directory found for {model_name} in {weights_dir}")
            return None
        
        # Look for different file extensions that models might use
        possible_extensions = ['.pth', '.pkl', '.joblib', '.json', '.txt']
        
        if weights_type == "best":
            # Look for best_model with various extensions
            for ext in possible_extensions:
                best_path = model_weights_dir / f"best_model{ext}"
                if best_path.exists():
                    self.logger.debug(f"Found best weights: {best_path}")
                    return best_path
            
            # Also look for just "best" without extension
            best_path = model_weights_dir / "best_model"
            if best_path.exists():
                self.logger.debug(f"Found best weights: {best_path}")
                return best_path
            
            self.logger.warning(f"No best_model file found for {model_name}, trying latest")
            weights_type = "latest"
        
        if weights_type == "latest":
            # Find the latest epoch files
            epoch_files = []
            for ext in possible_extensions:
                epoch_files.extend(list(model_weights_dir.glob(f"epoch_*{ext}")))
            
            # Also check for epoch files without extension
            epoch_files.extend(list(model_weights_dir.glob("epoch_*")))
            
            if not epoch_files:
                # If no epoch files, look for any model files
                all_files = []
                for ext in possible_extensions:
                    all_files.extend(list(model_weights_dir.glob(f"*{ext}")))
                
                if all_files:
                    # Sort by modification time and take the latest
                    all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    self.logger.warning(f"No epoch files found, using latest file: {all_files[0]}")
                    return all_files[0]
                else:
                    self.logger.warning(f"No model files found for {model_name}")
                    return None
            
            # Sort epoch files by epoch number
            def extract_epoch_num(path):
                try:
                    # Extract number from epoch_XX.ext or epoch_XX
                    stem = path.stem
                    if stem.startswith('epoch_'):
                        return int(stem.split('_')[1])
                    return 0
                except (ValueError, IndexError):
                    return 0
            
            epoch_files.sort(key=extract_epoch_num, reverse=True)
            self.logger.debug(f"Found latest weights: {epoch_files[0]}")
            return epoch_files[0]
        
        return None
    
    def test_model(self, model_name: str, test_data: pd.DataFrame, 
               weights_dir: Path, results_dir: Path, data_type_label: str,
               weights_type: str = "best") -> Optional[Dict[str, Any]]:
        """
        Test a single model.
        
        Args:
            model_name: Name of the model to test
            test_data: Test dataframe
            weights_dir: Directory containing model weights
            results_dir: Directory to save results
            data_type_label: Label for data type ('smote' or 'regular')
            weights_type: Type of weights to load
            
        Returns:
            Dictionary containing predictions and metrics
        """
        self.logger.info(f"\nTesting {model_name} (trained on {data_type_label.upper()} data)...")
        
        # Find weights
        weights_path = self.find_best_weights(model_name, weights_dir, weights_type)
        if weights_path is None:
            self.logger.error(f"No weights found for {model_name} in {weights_dir}")
            return None
        
        self.logger.info(f"Loading weights from: {weights_path}")
        
        # Create model instance
        try:
            model = self.get_model_instance(model_name)
        except Exception as e:
            self.logger.error(f"Failed to create model {model_name}: {str(e)}")
            return None
        
        # Load weights - Always use custom load method since all models have it
        try:
            self.logger.debug(f"Using custom load method for {model_name}")
            metadata = model.load(weights_path)
            self.logger.info(f"Successfully loaded model weights")
            
            # Log epoch info if available
            if hasattr(model, 'epoch'):
                self.logger.info(f"Model epoch: {model.epoch}")
            elif isinstance(metadata, dict) and 'epoch' in metadata:
                self.logger.info(f"Model epoch: {metadata['epoch']}")
        
        except AttributeError as e:
            self.logger.error(f"Model {model_name} doesn't have a load method: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load weights for {model_name}: {str(e)}")
            self.logger.exception("Detailed error:")
            return None
        
        # Set device for PyTorch models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(model, 'to'):
            model.to(device)
        
        # Make predictions
        try:
            import time
            start_time = time.time()

            # Prepare test data - keep as DataFrame to preserve column names
            X_test = test_data.iloc[:, :-1]
            y_true = test_data.iloc[:, -1].values

            self.logger.debug(f"Test data shape: {X_test.shape}")
            self.logger.debug(f"Target shape: {y_true.shape}")

            # Make predictions
            predictions = model.predict(X_test)
            
            # Ensure predictions are numpy array
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)

            inference_time = time.time() - start_time

            self.logger.info(f"Predictions completed in {inference_time:.2f}s")
            self.logger.info(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

            # Calculate metrics
            metrics = calculate_metrics(y_true, predictions, task='regression')
            metrics['inference_time'] = inference_time
            metrics['samples_per_second'] = len(test_data) / inference_time

            # Additional error statistics
            errors = y_true - predictions
            percentiles = [5, 25, 50, 75, 95]
            for p in percentiles:
                metrics[f'error_p{p}'] = np.percentile(np.abs(errors), p)

            # Create results dictionary
            results = {
                'model_name': model_name,
                'data_type': data_type_label,
                'weights_path': str(weights_path),
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'predictions': predictions,
                'actual': y_true,
                'test_size': len(test_data),
                'model_config': self.model_config.get(model_name, {}),
                'metadata': metadata if isinstance(metadata, dict) else {}
            }
            
            self.logger.info(f"Testing completed successfully for {model_name}")
            self.logger.info(f"RMSE: {metrics.get('rmse', 'N/A'):.6f}, MAE: {metrics.get('mae', 'N/A'):.6f}, R²: {metrics.get('r2', 'N/A'):.6f}")
            
            return results

        except Exception as e:
            self.logger.error(f"Error during prediction for {model_name}: {str(e)}")
            self.logger.exception("Detailed traceback:")
            return None
    
    def save_results(self, results: Dict[str, Any], model_name: str, results_dir: Path) -> None:
        """Save test results and predictions."""
        if results is None:
            return
        
        # Create model results directory
        model_results_dir = results_dir / model_name
        model_results_dir.mkdir(exist_ok=True)

        epsilon = 1e-8  # to avoid division by zero
        predictions_df = pd.DataFrame({
            'actual': results['actual'],
            'predicted': results['predictions'],
            'error': results['actual'] - results['predictions'],
            'abs_error': np.abs(results['actual'] - results['predictions']),
            'percent_error': 100 * np.abs(results['actual'] - results['predictions']) / (np.abs(results['actual']) + epsilon)
        })

        predictions_path = model_results_dir / 'predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        self.logger.info(f"Saved predictions to: {predictions_path}")
        
        # Save metrics as JSON
        metrics_data = {
            'model_name': results['model_name'],
            'data_type': results['data_type'],
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
            'data_type': results['data_type'],
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
        
        # Test each model for each data type
        all_results = {}
        
        for data_type_label, weights_dir in self.weights_dirs.items():
            results_dir = self.results_dirs[data_type_label]
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Testing models trained on {data_type_label.upper()} data")
            self.logger.info(f"{'='*60}")
            
            # Get list of models to test
            if model_names is None:
                # Find models with saved weights
                found_models = []
                if weights_dir.exists():
                    for model_dir in weights_dir.iterdir():
                        if model_dir.is_dir() and model_dir.name in model_registry:
                            found_models.append(model_dir.name)
                model_names_to_test = found_models
            else:
                model_names_to_test = model_names
            
            self.logger.info(f"Found {len(model_names_to_test)} models to test: {model_names_to_test}")
            
            # Test each model
            for model_name in tqdm(model_names_to_test, desc=f"Testing {data_type_label} models"):
                try:
                    results = self.test_model(
                        model_name, test_data, weights_dir, results_dir, 
                        data_type_label, weights_type
                    )
                    if results:
                        # Key includes both model name and data type
                        result_key = f"{model_name}_{data_type_label}"
                        all_results[result_key] = results
                        self.save_results(results, model_name, results_dir)
                except KeyboardInterrupt:
                    self.logger.warning("Testing interrupted by user")
                    break
                except Exception as e:
                    self.logger.error(f"Failed to test {model_name}: {str(e)}")
                    continue
            
            # Save comparison summary for this data type
            self.save_comparison_summary(
                {k: v for k, v in all_results.items() if k.endswith(f"_{data_type_label}")},
                results_dir,
                data_type_label
            )
        
        # If testing both, create combined comparison
        if self.data_type == 'both':
            self.save_combined_comparison(all_results)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("TESTING COMPLETED")
        self.logger.info("="*60)
        
        return all_results
    
    def save_comparison_summary(self, results: Dict[str, Dict[str, Any]], 
                               results_dir: Path, data_type_label: str) -> None:
        """Save summary comparison of models for a specific data type."""
        if not results:
            return
        
        # Create comparison data
        comparison_data = []
        
        for result_key, result in results.items():
            row = {
                'model': result['model_name'],
                'data_type': data_type_label,
                'timestamp': result['timestamp'],
                **result['metrics']
            }
            comparison_data.append(row)
        
        # Save as CSV
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = results_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        self.logger.info(f"Saved {data_type_label} model comparison to: {comparison_path}")
        
        # Save as markdown
        markdown_path = results_dir / 'model_comparison.md'
        with open(markdown_path, 'w') as f:
            f.write(f"# Model Comparison Results - {data_type_label.upper()} Data\n\n")
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
        self.logger.info(f"\nModel Comparison Summary ({data_type_label.upper()}):")
        print(comparison_df[['model', 'rmse', 'mae', 'r2']].to_string(index=False))
    
    def save_combined_comparison(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Save combined comparison of all models across data types."""
        if not all_results:
            return
        
        # Create combined results directory
        combined_dir = Path("results") / "combined_comparison"
        combined_dir.mkdir(exist_ok=True)
        
        # Create comparison data
        comparison_data = []
        
        for result_key, result in all_results.items():
            row = {
                'model': result['model_name'],
                'data_type': result['data_type'],
                'timestamp': result['timestamp'],
                **result['metrics']
            }
            comparison_data.append(row)
        
        # Save as CSV
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = combined_dir / 'all_models_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        self.logger.info(f"Saved combined comparison to: {comparison_path}")
        
        # Create comparison report
        report_path = combined_dir / 'comparison_report.md'
        with open(report_path, 'w') as f:
            f.write("# Combined Model Comparison - SMOTE vs Regular Data\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall best model
            f.write("## Overall Best Model\n\n")
            best_overall = comparison_df.loc[comparison_df['rmse'].idxmin()]
            f.write(f"**{best_overall['model']} ({best_overall['data_type'].upper()})** - RMSE: {best_overall['rmse']:.6f}\n\n")
            
            # Comparison by model
            f.write("## Model Performance: SMOTE vs Regular\n\n")
            
            for model_name in comparison_df['model'].unique():
                f.write(f"### {model_name}\n\n")
                model_data = comparison_df[comparison_df['model'] == model_name]
                
                if len(model_data) == 2:  # Both SMOTE and regular
                    smote_data = model_data[model_data['data_type'] == 'smote'].iloc[0]
                    regular_data = model_data[model_data['data_type'] == 'regular'].iloc[0]
                    
                    # Calculate improvements
                    rmse_improvement = (regular_data['rmse'] - smote_data['rmse']) / regular_data['rmse'] * 100
                    r2_improvement = (smote_data['r2'] - regular_data['r2']) / abs(regular_data['r2']) * 100
                    
                    f.write("| Metric | Regular | SMOTE | Improvement |\n")
                    f.write("|--------|---------|-------|-------------|\n")
                    f.write(f"| RMSE | {regular_data['rmse']:.6f} | {smote_data['rmse']:.6f} | {rmse_improvement:+.1f}% |\n")
                    f.write(f"| MAE | {regular_data['mae']:.6f} | {smote_data['mae']:.6f} | - |\n")
                    f.write(f"| R² | {regular_data['r2']:.6f} | {smote_data['r2']:.6f} | {r2_improvement:+.1f}% |\n")
                    f.write("\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            summary = comparison_df.groupby('data_type')[['rmse', 'mae', 'r2']].agg(['mean', 'std'])
            f.write(summary.to_markdown())


def main():
    """Main entry point for the testing script."""
    parser = argparse.ArgumentParser(
        description="Test and evaluate trained CHF models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all models (both SMOTE and regular)
  python scripts/test.py --data-type both
  
  # Test only SMOTE-trained models
  python scripts/test.py --data-type smote
  
  # Test only regular-trained models
  python scripts/test.py --data-type regular
  
  # Test specific models
  python scripts/test.py --models xgboost lightgbm --data-type both
  
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
        '--data-type',
        type=str,
        choices=['smote', 'regular', 'both'],
        default='both',
        help='Test models trained on SMOTE, regular, or both data types (default: both)'
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
    tester = ModelTester(
        config_path=config_path, 
        debug=args.debug,
        data_type=args.data_type
    )
    
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