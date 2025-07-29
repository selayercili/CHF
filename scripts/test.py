#!/usr/bin/env python3
"""
Robust Model Testing Script that handles both PyTorch and non-PyTorch models
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'max_error': np.max(np.abs(y_true - y_pred)),
        'mean_error': np.mean(y_true - y_pred),
        'std_error': np.std(y_true - y_pred)
    }

class RobustModelTester:
    """Test script that can handle both PyTorch and traditional ML models."""
    
    def __init__(self, data_type: str = 'both', debug: bool = False):
        self.data_type = data_type
        self.debug = debug
        
        self.setup_paths()
        self.load_dependencies()
    
    def setup_paths(self):
        """Setup directory paths."""
        if self.data_type == 'both':
            self.test_configs = [
                {'data_type': 'smote', 'weights_dir': Path('weights_smote'), 'results_dir': Path('results_smote')},
                {'data_type': 'regular', 'weights_dir': Path('weights_regular'), 'results_dir': Path('results_regular')}
            ]
        elif self.data_type == 'smote':
            self.test_configs = [{'data_type': 'smote', 'weights_dir': Path('weights_smote'), 'results_dir': Path('results_smote')}]
        else:
            self.test_configs = [{'data_type': 'regular', 'weights_dir': Path('weights_regular'), 'results_dir': Path('results_regular')}]
        
        for config in self.test_configs:
            config['results_dir'].mkdir(exist_ok=True)
        
        self.data_dir = Path("data/processed")
    
    def load_dependencies(self):
        """Load model registry and configurations."""
        try:
            from src.models import model_registry
            self.model_registry = model_registry
            print(f"✅ Model registry loaded: {list(model_registry.keys())}")
        except Exception as e:
            print(f"❌ Failed to load model registry: {e}")
            self.model_registry = {}
        
        try:
            from src.utils import ConfigManager
            self.config_manager = ConfigManager(Path("configs"))
            self.model_config = self.config_manager.get('model_configs')
            print(f"✅ Config loaded")
        except Exception as e:
            print(f"⚠️  Config loading failed, using empty config: {e}")
            self.model_config = {}
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data."""
        test_path = self.data_dir / "test.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        test_df = pd.read_csv(test_path)
        print(f"📊 Test data loaded: {test_df.shape}")
        return test_df
    
    def find_model_weights(self, model_dir: Path) -> Optional[Tuple[Path, str]]:
        """Find the best weights file for a model, handling different formats."""
        
        # Priority order for finding weights
        candidates = [
            (model_dir / "best_model.pkl", "best_pickle"),
            (model_dir / "best_model.pth", "best_pytorch"), 
        ]
        
        # Add epoch files
        epoch_pkl_files = list(model_dir.glob("epoch_*.pkl"))
        epoch_pth_files = list(model_dir.glob("epoch_*.pth"))
        
        if epoch_pkl_files:
            # Sort by epoch number and get latest
            epoch_pkl_files.sort(key=lambda x: int(x.stem.split('_')[1]))
            candidates.append((epoch_pkl_files[-1], "latest_pickle"))
        
        if epoch_pth_files:
            # Sort by epoch number and get latest  
            epoch_pth_files.sort(key=lambda x: int(x.stem.split('_')[1]))
            candidates.append((epoch_pth_files[-1], "latest_pytorch"))
        
        # Return first existing candidate
        for weight_path, weight_type in candidates:
            if weight_path.exists():
                return weight_path, weight_type
        
        return None
    
    def load_model_with_weights(self, model_name: str, weight_path: Path, weight_type: str):
        """Load model and weights with automatic format detection."""
        
        print(f"      🏋️  Loading {weight_type}: {weight_path.name}")
        
        # Create model instance first
        model_class = self.model_registry[model_name]
        model_config = self.model_config.get(model_name, {})
        init_params = model_config.get('init_params', {})
        
        # Handle neural network architectures
        if model_name in ['neural_network', 'pinn'] and 'architecture' in model_config:
            arch_config = model_config['architecture']
            if 'hidden_layers' in arch_config:
                init_params['hidden_layers'] = arch_config['hidden_layers']
        
        # Load the weights/model
        if 'pickle' in weight_type:
            # Pickle format - load entire checkpoint
            with open(weight_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                # Checkpoint format
                model = checkpoint['model']
                print(f"         ✅ Loaded from pickle checkpoint")
            else:
                # Direct model
                model = checkpoint
                print(f"         ✅ Loaded direct pickle model")
                
        elif 'pytorch' in weight_type:
            # PyTorch format
            try:
                checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
                
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        # Full model in checkpoint
                        model = checkpoint['model']
                        print(f"         ✅ Loaded full model from PyTorch checkpoint")
                    else:
                        # Need to load state dict
                        model = model_class(**init_params)
                        if 'model_state_dict' in checkpoint and hasattr(model, 'load_state_dict'):
                            model.load_state_dict(checkpoint['model_state_dict'])
                            print(f"         ✅ Loaded state dict to new model instance")
                        else:
                            print(f"         ⚠️  Could not load state dict, using new instance")
                else:
                    # Direct model
                    model = checkpoint
                    print(f"         ✅ Loaded direct PyTorch model")
                    
            except Exception as e:
                print(f"         ❌ PyTorch loading failed: {e}")
                # Fallback: try as pickle even though extension is .pth
                try:
                    with open(weight_path, 'rb') as f:
                        checkpoint = pickle.load(f)
                    
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        model = checkpoint['model']
                    else:
                        model = checkpoint
                    print(f"         ✅ Loaded as pickle despite .pth extension")
                except:
                    raise e
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
        
        # Move PyTorch models to appropriate device
        if hasattr(model, 'to') and hasattr(model, 'parameters'):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            print(f"         📱 Moved to {device}")
        
        return model
    
    def make_predictions(self, model, X_test):
        """Make predictions with different model types."""
        
        if hasattr(model, 'predict'):
            # Standard sklearn-like interface
            return model.predict(X_test)
            
        elif hasattr(model, 'forward') or hasattr(model, '__call__'):
            # PyTorch model
            model.eval() if hasattr(model, 'eval') else None
            
            with torch.no_grad():
                # Convert to tensor if needed
                if isinstance(X_test, pd.DataFrame):
                    X_tensor = torch.FloatTensor(X_test.values)
                else:
                    X_tensor = torch.FloatTensor(X_test)
                
                # Move to same device as model
                if hasattr(model, 'parameters') and next(model.parameters(), None) is not None:
                    device = next(model.parameters()).device
                    X_tensor = X_tensor.to(device)
                
                # Make prediction
                if hasattr(model, 'forward'):
                    predictions = model.forward(X_tensor)
                else:
                    predictions = model(X_tensor)
                
                # Convert back to numpy
                predictions = predictions.cpu().numpy()
                
                # Handle different output shapes
                if predictions.ndim > 1:
                    predictions = predictions.squeeze()
                
                return predictions
        else:
            raise AttributeError(f"Model has no predict or forward method: {type(model)}")
    
    def test_single_model(self, model_name: str, test_data: pd.DataFrame,
                         weights_dir: Path, results_dir: Path, data_type_label: str) -> Optional[Dict[str, Any]]:
        """Test a single model."""
        print(f"   🧪 Testing {model_name}...")
        
        model_dir = weights_dir / model_name
        if not model_dir.exists():
            print(f"      ❌ Model directory not found: {model_dir}")
            return None
        
        # Find weights
        weight_info = self.find_model_weights(model_dir)
        if weight_info is None:
            print(f"      ❌ No weights found in {model_dir}")
            if self.debug:
                files = list(model_dir.iterdir())
                print(f"         Available files: {[f.name for f in files]}")
            return None
        
        weight_path, weight_type = weight_info
        
        try:
            # Load model
            model = self.load_model_with_weights(model_name, weight_path, weight_type)
            
            # Prepare test data
            X_test = test_data.iloc[:, :-1]
            y_true = test_data.iloc[:, -1].values
            
            # Make predictions
            start_time = datetime.now()
            predictions = self.make_predictions(model, X_test)
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate metrics
            metrics = calculate_metrics(y_true, predictions)
            metrics['inference_time'] = inference_time
            metrics['samples_per_second'] = len(test_data) / inference_time
            
            print(f"      ✅ RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}")
            
            return {
                'model_name': model_name,
                'data_type': data_type_label,
                'weight_type': weight_type,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                'actual': y_true.tolist(),
                'test_size': len(test_data)
            }
            
        except Exception as e:
            print(f"      ❌ Failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def save_results(self, results: Dict[str, Any], results_dir: Path):
        """Save test results."""
        if not results:
            return
        
        model_name = results['model_name']
        model_results_dir = results_dir / model_name
        model_results_dir.mkdir(exist_ok=True)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'actual': results['actual'],
            'predicted': results['predictions'],
            'error': np.array(results['actual']) - np.array(results['predictions']),
            'abs_error': np.abs(np.array(results['actual']) - np.array(results['predictions']))
        })
        
        predictions_df.to_csv(model_results_dir / 'predictions.csv', index=False)
        
        # Save metrics
        metrics_data = {
            'model_name': results['model_name'],
            'data_type': results['data_type'],
            'weight_type': results['weight_type'],
            'timestamp': results['timestamp'],
            'metrics': results['metrics'],
            'test_size': results['test_size']
        }
        
        with open(model_results_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def test_all_models(self, model_names: Optional[List[str]] = None):
        """Test all available models."""
        test_data = self.load_test_data()
        all_results = {}
        
        for config in self.test_configs:
            data_type = config['data_type']
            weights_dir = config['weights_dir']
            results_dir = config['results_dir']
            
            print(f"\n{'='*60}")
            print(f"Testing {data_type.upper()} models from {weights_dir}")
            print(f"{'='*60}")
            
            if not weights_dir.exists():
                print(f"⚠️  Weights directory not found: {weights_dir}")
                continue
            
            # Find available models
            model_dirs = [d for d in weights_dir.iterdir() if d.is_dir()]
            available_models = [d.name for d in model_dirs if d.name in self.model_registry]
            
            if model_names:
                available_models = [m for m in available_models if m in model_names]
            
            if not available_models:
                print(f"⚠️  No models found")
                continue
            
            print(f"📋 Testing {len(available_models)} models: {available_models}")
            
            # Test each model
            for model_name in tqdm(available_models, desc=f"Testing {data_type} models"):
                try:
                    results = self.test_single_model(
                        model_name, test_data, weights_dir, results_dir, data_type
                    )
                    
                    if results:
                        key = f"{model_name}_{data_type}"
                        all_results[key] = results
                        self.save_results(results, results_dir)
                        
                except KeyboardInterrupt:
                    print("\n⚠️  Testing interrupted")
                    break
                except Exception as e:
                    print(f"   ❌ Error testing {model_name}: {e}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
            
            # Save comparison
            self.save_comparison(
                {k: v for k, v in all_results.items() if k.endswith(f"_{data_type}")},
                results_dir
            )
        
        if self.data_type == 'both':
            self.save_combined_comparison(all_results)
        
        return all_results
    
    def save_comparison(self, results: Dict[str, Any], results_dir: Path):
        """Save model comparison."""
        if not results:
            return
        
        comparison_data = []
        for key, result in results.items():
            row = {
                'model': result['model_name'],
                'data_type': result['data_type'],
                'weight_type': result['weight_type'],
                **result['metrics']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('rmse')
        
        comparison_path = results_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"\n📊 Results saved to: {comparison_path}")
        print(comparison_df[['model', 'rmse', 'mae', 'r2', 'weight_type']].to_string(index=False))
    
    def save_combined_comparison(self, all_results: Dict[str, Any]):
        """Save combined comparison."""
        combined_dir = Path("results") / "combined_comparison"
        combined_dir.mkdir(exist_ok=True)
        
        comparison_data = []
        for key, result in all_results.items():
            row = {
                'model': result['model_name'],
                'data_type': result['data_type'],
                'weight_type': result['weight_type'],
                **result['metrics']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = combined_dir / 'all_models_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"\n🏆 Combined results saved to: {comparison_path}")

def main():
    parser = argparse.ArgumentParser(description="Test trained models with robust format handling")
    parser.add_argument('--models', nargs='+', help='Specific models to test')
    parser.add_argument('--data-type', choices=['smote', 'regular', 'both'], 
                       default='both', help='Data type to test')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("🧪 STARTING ROBUST MODEL TESTING")
    print("=" * 60)
    
    try:
        tester = RobustModelTester(data_type=args.data_type, debug=args.debug)
        results = tester.test_all_models(model_names=args.models)
        
        print(f"\n✅ Testing completed! Tested {len(results)} model configurations.")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()