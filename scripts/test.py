#!/usr/bin/env python3
"""
1) Import the test data
2) Loop through the models
3) Evaluate the models - the weights are located in the models directory
4) Save the model predictions in the results directory - this will be in the results/ folder

from model_1 import model_1
from model_2 import model_2

model_names = [model_1, model_2]
config_args = yaml.safe_load(configs/model_config.yaml)

for model_name in model_names:
    model = model_1(train, test, config_args[model_name])
    model.load_weights(f"./weights/{model_name}/{epoch}.pth")
    model.test()

Model Testing and Evaluation Script

This script loads trained models and evaluates them on test data.
It saves predictions and metrics for further analysis.

Usage:
    python scripts/test_models.py [--config CONFIG_PATH] [--weights-type WEIGHTS_TYPE]
"""

import os
import sys
import argparse
import logging
import importlib
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelTester:
    """Handles model testing and evaluation pipeline."""
    
    def __init__(self, config_path: str = "configs/model_configs.yaml"):
        """
        Initialize the ModelTester.
        
        Args:
            config_path: Path to the model configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.weights_dir = Path("./weights")
        self.results_dir = Path("./results")
        self.data_dir = Path("data/processed")
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        logger = logging.getLogger("ModelTester")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_dir / f"testing_{timestamp}.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data.
        
        Returns:
            Test dataframe
        """
        self.logger.info("Loading test data...")
        
        test_path = self.data_dir / "test.csv"
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        test_df = pd.read_csv(test_path)
        self.logger.info(f"Loaded test data: {test_df.shape}")
        
        return test_df
    
    def _get_model_class(self, model_name: str):
        """
        Dynamically import and return model class.
        
        Args:
            model_name: Name of the model to import
            
        Returns:
            Model class
        """
        try:
            # Convert model name to module path
            module_path = f"src.models.{model_name}"
            module = importlib.import_module(module_path)
            
            # Get the model class
            model_class_name = ''.join(word.capitalize() for word in model_name.split('_'))
            model_class = getattr(module, model_class_name)
            
            return model_class
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to import model {model_name}: {e}")
            raise
    
    def _find_best_weights(self, model_name: str, weights_type: str = "best") -> Path:
        """
        Find the best weights file for a model.
        
        Args:
            model_name: Name of the model
            weights_type: Type of weights to load ("best" or "latest")
            
        Returns:
            Path to weights file
        """
        model_weights_dir = self.weights_dir / model_name
        
        if not model_weights_dir.exists():
            raise FileNotFoundError(f"No weights directory found for {model_name}")
        
        if weights_type == "best":
            # Look for best_model.pth
            best_path = model_weights_dir / "best_model.pth"
            if best_path.exists():
                return best_path
            else:
                self.logger.warning(f"No best_model.pth found for {model_name}, using latest")
                weights_type = "latest"
        
        if weights_type == "latest":
            # Find the latest epoch
            epoch_files = list(model_weights_dir.glob("epoch_*.pth"))
            if not epoch_files:
                raise FileNotFoundError(f"No epoch files found for {model_name}")
            
            # Sort by epoch number
            epoch_files.sort(key=lambda x: int(x.stem.split('_')[1]))
            return epoch_files[-1]
        
        raise ValueError(f"Unknown weights type: {weights_type}")
    
    def test_model(self, model_name: str, test_data: pd.DataFrame, 
                   weights_type: str = "best") -> Dict[str, Any]:
        """
        Test a single model.
        
        Args:
            model_name: Name of the model to test
            test_data: Test dataframe
            weights_type: Type of weights to load
            
        Returns:
            Dictionary containing predictions and metrics
        """
        self.logger.info(f"Testing {model_name}...")
        
        # Get model configuration
        if model_name not in self.config:
            self.logger.error(f"No configuration found for {model_name}")
            return None
        
        model_config = self.config[model_name]
        
        # Get model class
        try:
            model_class = self._get_model_class(model_name)
        except Exception as e:
            self.logger.error(f"Skipping {model_name} due to import error: {e}")
            return None
        
        # Find weights file
        try:
            weights_path = self._find_best_weights(model_name, weights_type)
            self.logger.info(f"Loading weights from: {weights_path}")
        except FileNotFoundError as e:
            self.logger.error(f"No weights found for {model_name}: {e}")
            return None
        
        # Initialize model
        try:
            model = model_class(**model_config.get('init_params', {}))
        except Exception as e:
            self.logger.error(f"Failed to initialize {model_name}: {e}")
            return None
        
        # Load weights - still broken
        try:
            # Check if this is a PyTorch model with custom load method (NN or PINN)
            if model_name.lower() in ['neural_network', 'pinn']:
                # Use the custom load method for PyTorch models
                if hasattr(model, 'load'):
                    metadata = model.load(weights_path)
                    self.logger.info(f"Loaded PyTorch model from epoch: {getattr(model, 'epoch', 'unknown')}")
                else:
                    self.logger.error(f"Model {model_name} doesn't have a load method")
                    return None
            else:
                # For other models (TensorFlow/Keras models)
                if hasattr(model, 'load'):
                    metadata = model.load(weights_path)
                    self.logger.info(f"Loaded model from epoch: {metadata.get('epoch', 'unknown')}")
                else:
                    # Fallback for generic PyTorch models
                    import torch
                    checkpoint = torch.load(weights_path, map_location='cpu',weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'model' in checkpoint:
                        model = checkpoint['model']
                    else:
                        self.logger.error(f"Unknown checkpoint format for {model_name}")
                        return None
        except Exception as e:
            self.logger.error(f"Failed to load weights for {model_name}: {e}")
            return None
        
        # Make predictions
        try:
            # Get features (all columns except last)
            X_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1].values
            
            # Predict
            predictions = model.predict(X_test)
            
            # Calculate metrics
            if hasattr(model, 'validate'):
                metrics = model.validate(test_data)
            else:
                # Calculate basic metrics
                if model_config.get('task_type', 'regression') == 'regression':
                    mse = np.mean((predictions - y_test) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(predictions - y_test))
                    r2 = 1 - (np.sum((y_test - predictions) ** 2) / 
                             np.sum((y_test - y_test.mean()) ** 2))
                    metrics = {
                        'mse': float(mse),
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'r2': float(r2)
                    }
                else:
                    accuracy = np.mean(predictions == y_test)
                    metrics = {'accuracy': float(accuracy)}
            
            # Prepare results
            results = {
                'model_name': model_name,
                'weights_path': str(weights_path),
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'predictions': predictions,
                'actual': y_test,
                'test_size': len(test_data)
            }
            
            # Add feature importance if available
            if hasattr(model, 'get_feature_importance'):
                try:
                    results['feature_importance'] = model.get_feature_importance()
                except:
                    pass
            
            self.logger.info(f"Test metrics for {model_name}: {metrics}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during testing {model_name}: {e}")
            return None
    
    def save_results(self, results: Dict[str, Any], model_name: str) -> None:
        """Save test results and predictions."""
        if results is None:
            return
        
        # Create model results directory
        model_results_dir = self.results_dir / model_name
        model_results_dir.mkdir(exist_ok=True)
        
        # Save predictions as CSV
        predictions_df = pd.DataFrame({
            'actual': results['actual'],
            'predicted': results['predictions']
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
        
        # Save full results as pickle for later analysis
        results_pickle_path = model_results_dir / 'full_results.pkl'
        with open(results_pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
    def test_all_models(self, weights_type: str = "best") -> None:
        """
        Test all models specified in the configuration.
        
        Args:
            weights_type: Type of weights to load ("best" or "latest")
        """
        # Load test data once
        test_data = self.load_test_data()
        
        # Get list of models to test
        model_names = list(self.config.keys())
        self.logger.info(f"Found {len(model_names)} models to test: {model_names}")
        
        # Test each model
        all_results = {}
        for model_name in tqdm(model_names, desc="Testing models"):
            try:
                results = self.test_model(model_name, test_data, weights_type)
                if results:
                    all_results[model_name] = results
                    self.save_results(results, model_name)
            except Exception as e:
                self.logger.error(f"Failed to test {model_name}: {e}")
                continue
        
        # Save summary of all results
        self._save_summary(all_results)
        
        self.logger.info("Testing pipeline completed!")
    
    def _save_summary(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Save summary of all model results."""
        summary_data = []
        
        for model_name, results in all_results.items():
            summary_entry = {
                'model': model_name,
                'timestamp': results['timestamp'],
                **results['metrics']
            }
            summary_data.append(summary_entry)
        
        # Save as CSV
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.results_dir / 'model_comparison.csv'
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"Saved model comparison to: {summary_path}")
        
        # Also save as formatted markdown
        markdown_path = self.results_dir / 'model_comparison.md'
        with open(markdown_path, 'w') as f:
            f.write("# Model Comparison Results\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(summary_df.to_markdown(index=False))
        
        # Print summary to console
        self.logger.info("\n" + "="*50)
        self.logger.info("MODEL COMPARISON SUMMARY")
        self.logger.info("="*50)
        print(summary_df.to_string(index=False))


def main():
    """Main entry point for the testing script."""
    parser = argparse.ArgumentParser(
        description="Test and evaluate trained models"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_configs.yaml',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--weights-type',
        type=str,
        choices=['best', 'latest'],
        default='best',
        help='Type of weights to load (best or latest)'
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ModelTester(config_path=args.config)
    
    # Run testing pipeline
    try:
        tester.test_all_models(weights_type=args.weights_type)
    except KeyboardInterrupt:
        tester.logger.info("Testing interrupted by user")
    except Exception as e:
        tester.logger.error(f"Testing failed: {e}")
        raise


if __name__ == "__main__":
    main()