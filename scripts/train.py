#!/usr/bin/env python3
# scripts/train.py
"""
Model Training Pipeline Script with SMOTE/Regular Data Support

This script handles the training of multiple models with configurable parameters.
It loads data, trains models, saves checkpoints, and logs progress.

Usage:
    python scripts/train.py [--config CONFIG_PATH] [--models MODEL_NAMES] [--data-type DATA_TYPE] [--debug]
"""

import os
import sys
import argparse
import logging
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Imports after path setup
from src.utils import (
    setup_logging, get_logger, load_config, merge_configs,
    CheckpointManager, MetricsTracker, EarlyStopping,
    set_random_seeds, get_device, ConfigManager
)
from src.models import model_registry
from src.data import load_data_with_validation

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ModelTrainer:
    """Orchestrates model training pipeline for multiple models."""
    
    def __init__(self, config_path: Optional[Path] = None, debug: bool = False, data_type: str = 'smote'):
        """
        Initialize the ModelTrainer.
        
        Args:
            config_path: Path to configuration file
            debug: Enable debug logging
            data_type: Type of training data ('smote' or 'regular')
        """
        # Setup logging first
        log_level = 'DEBUG' if debug else 'INFO'
        setup_logging(log_level=log_level)
        self.logger = get_logger(__name__)
        
        # Store data type
        self.data_type = data_type
        
        # Load configurations
        self.config_manager = ConfigManager(config_path or Path("configs"))
        self.model_config = self.config_manager.get('model_configs')
        self.global_config = self.model_config.get('global_settings', {})
        
        # Set random seeds
        set_random_seeds(self.global_config.get('random_seed', 42))
        
        # Setup device
        self.device = get_device()
        
        # Setup directories with data type suffix
        self.setup_directories()
        
        self.logger.info("="*60)
        self.logger.info("Model Trainer Initialized")
        self.logger.info(f"Config path: {config_path}")
        self.logger.info(f"Debug mode: {debug}")
        self.logger.info(f"Data type: {data_type}")
        self.logger.info("="*60)
    
    def setup_directories(self) -> None:
        """Create necessary directories."""
        # Add suffix based on data type
        suffix = ""
        if self.data_type == "smote":
            suffix = "_smote"
        elif self.data_type == "regular":
            suffix = "_regular"
        
        self.dirs = {
            'weights': Path(f"weights{suffix}"),
            'logs': Path(f"logs{suffix}"),
            'results': Path(f"results{suffix}"),
            'data': Path("data/processed")
        }
        
        for dir_name, dir_path in self.dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Directory ready: {dir_path}")
    
    def load_data(self) -> Dict[str, Any]:
        """Load and prepare training data based on data type."""
        self.logger.info(f"Loading {self.data_type} training data...")
        
        # Select data file based on type
        if self.data_type == 'smote':
            train_path = self.dirs['data'] / "train_resampled.csv"
            # Fallback to regular if SMOTE doesn't exist
            if not train_path.exists():
                self.logger.warning("SMOTE data not found, falling back to regular training data")
                train_path = self.dirs['data'] / "train.csv"
                self.data_type = 'regular'  # Update data type
        else:  # regular
            train_path = self.dirs['data'] / "train.csv"
        
        test_path = self.dirs['data'] / "test.csv"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        # Load data with validation split
        data_dict = load_data_with_validation(
            train_path=train_path,
            test_path=test_path if test_path.exists() else None,
            validation_split=0.2,
            random_state=self.global_config.get('random_seed', 42)
        )
        
        # Log data statistics
        self.logger.info(f"Data type: {self.data_type.upper()}")
        for split_name, df in data_dict.items():
            self.logger.info(f"{split_name.capitalize()} data shape: {df.shape}")
            
        # Log if using SMOTE data
        if self.data_type == 'smote' and 'train' in data_dict:
            train_df = data_dict['train']
            if 'cluster_label' in train_df.columns:
                cluster_dist = train_df['cluster_label'].value_counts().sort_index()
                self.logger.info("Cluster distribution in training data:")
                for cluster_id, count in cluster_dist.items():
                    self.logger.info(f"  Cluster {cluster_id}: {count} samples")
        
        return data_dict
    
    def get_model_instance(self, model_name: str, config: Dict[str, Any]) -> Any:
        """
        Get model instance from registry.
        
        Args:
            model_name: Name of the model
            config: Model configuration
            
        Returns:
            Model instance
        """
        if model_name not in model_registry:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = model_registry[model_name]
        init_params = config.get('init_params', {})
        
        # Special handling for neural networks
        if model_name in ['neural_network', 'pinn']:
            # Architecture configuration
            if 'architecture' in config:
                arch_config = config['architecture']
                if 'hidden_layers' in arch_config:
                    # Convert layer list to parameters
                    init_params['hidden_layers'] = arch_config['hidden_layers']
        
        # Create model instance
        model = model_class(**init_params)
        
        # Add tuning parameters if available
        if 'tuning' in config:
            model.tuning_params = config['tuning']
        
        return model
    
    def save_training_results(self, model_name: str, results: Dict[str, Any], 
                            model: Any, metrics_tracker: MetricsTracker) -> None:
        """
        Save training results to the results directory.
        
        Args:
            model_name: Name of the model
            results: Training results dictionary
            model: Trained model instance
            metrics_tracker: Metrics tracker instance
        """
        # Create model-specific results directory
        model_results_dir = self.dirs['results'] / model_name
        model_results_dir.mkdir(exist_ok=True)
        
        # Save results summary as JSON
        results_file = model_results_dir / f"training_results_{self.data_type}.json"
        
        # Prepare serializable results
        serializable_results = {
            'model_name': results['model_name'],
            'data_type': results['data_type'],
            'epochs_trained': results['epochs_trained'],
            'best_val_loss': float(results['best_val_loss']) if results['best_val_loss'] != float('inf') else None,
            'training_time': results['training_time'],
            'training_time_formatted': f"{results['training_time']/60:.2f} minutes",
            'timestamp': datetime.now().isoformat(),
            'config_used': self.model_config.get(model_name, {}),
            'data_shapes': results.get('data_shapes', {}),
            'final_metrics': results.get('final_metrics', {})
        }
        
        # Add metrics summary if available
        if results.get('metrics_summary'):
            # Convert any numpy types to native Python types
            metrics_summary = {}
            for key, value in results['metrics_summary'].items():
                if hasattr(value, 'item'):  # numpy scalar
                    metrics_summary[key] = value.item()
                elif isinstance(value, (list, tuple)):
                    metrics_summary[key] = [v.item() if hasattr(v, 'item') else v for v in value]
                else:
                    metrics_summary[key] = value
            serializable_results['metrics_summary'] = metrics_summary
        
        # Save JSON results
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to: {results_file}")
        
        # Save training history as CSV if available
        history_data = metrics_tracker.get_history()
        if history_data:
            history_file = model_results_dir / f"training_history_{self.data_type}.csv"
            
            # Convert to DataFrame and save
            try:
                df = pd.DataFrame(history_data)
                df.to_csv(history_file, index=False)
                self.logger.info(f"Training history saved to: {history_file}")
            except Exception as e:
                self.logger.warning(f"Could not save training history: {e}")
        
        # Save model parameters/info
        model_info_file = model_results_dir / f"model_info_{self.data_type}.json"
        model_info = {
            'model_class': model.__class__.__name__,
            'model_type': model_name,
            'data_type': self.data_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # Try to get model-specific info
        try:
            if hasattr(model, 'get_params'):
                model_info['parameters'] = model.get_params()
            if hasattr(model, 'feature_importances_'):
                # Convert numpy arrays to lists for JSON serialization
                model_info['feature_importances'] = model.feature_importances_.tolist()
            if hasattr(model, 'n_features_in_'):
                model_info['n_features'] = int(model.n_features_in_)
        except Exception as e:
            self.logger.debug(f"Could not extract model info: {e}")
        
        with open(model_info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        self.logger.info(f"Model info saved to: {model_info_file}")
    
    def train_model(self, model_name: str, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            data_dict: Dictionary with train/val/test data
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Training {model_name} on {self.data_type.upper()} data")
        self.logger.info(f"{'='*60}")
        
        # Get model configuration
        if model_name not in self.model_config:
            self.logger.error(f"No configuration found for {model_name}")
            return None
        
        model_config = self.model_config[model_name]
        
        # Create model instance
        try:
            model = self.get_model_instance(model_name, model_config)
            self.logger.info(f"Model {model_name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize {model_name}: {str(e)}")
            return None
        
        # Setup training components
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.dirs['weights'],
            model_name=model_name,
            max_checkpoints=5,
            save_best_only=False
        )
        
        metrics_tracker = MetricsTracker(
            save_dir=self.dirs['logs'] / model_name,
            window_size=100
        )
        
        # Add data type to metadata
        metrics_tracker.update({'data_type': self.data_type})
        
        # Training parameters
        training_config = model_config.get('training', {})
        epochs = training_config.get('epochs', model_config.get('epochs', 10))
        batch_size = training_config.get('batch_size', 32)
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=training_config.get('early_stopping_patience', 10),
            mode='min',
            restore_best_weights=True
        )
        
        # Training data
        train_data = data_dict['train']
        val_data = data_dict.get('val')
        
        # Training loop
        best_val_loss = float('inf')
        training_start_time = datetime.now()
        final_metrics = {}
        
        self.logger.info(f"Starting training for {epochs} epochs...")
        self.logger.info(f"Training samples: {len(train_data)}")
        if val_data is not None:
            self.logger.info(f"Validation samples: {len(val_data)}")
        
        for epoch in range(epochs):
            epoch_start_time = datetime.now()
            
            # Train epoch
            try:
                metrics_tracker.start_timer('train_epoch')
                train_metrics = model.train_epoch(
                    train_data,
                    batch_size=batch_size
                )
                train_time = metrics_tracker.stop_timer('train_epoch')
                
                # Update metrics
                metrics_tracker.update(train_metrics, prefix='train/')
                
                # Validation
                val_metrics = {}
                if val_data is not None:
                    metrics_tracker.start_timer('validation')
                    val_metrics = model.validate(val_data)
                    val_time = metrics_tracker.stop_timer('validation')
                    metrics_tracker.update(val_metrics, prefix='val/')
                
                # Store final metrics
                final_metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
                
                # Log epoch results
                epoch_time = (datetime.now() - epoch_start_time).total_seconds()
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_metrics.get('loss', 0):.6f} - "
                    f"Val Loss: {val_metrics.get('loss', 0):.6f} - "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Save checkpoint
                current_loss = val_metrics.get('loss', train_metrics.get('loss', 0))
                is_best = current_loss < best_val_loss
                
                if is_best:
                    best_val_loss = current_loss
                
                # Add data type to checkpoint metadata
                checkpoint_metadata = {
                    **train_metrics, 
                    **{f'val_{k}': v for k, v in val_metrics.items()},
                    'data_type': self.data_type
                }
                
                checkpoint_manager.save_checkpoint(
                    model=model,
                    epoch=epoch + 1,
                    metrics=checkpoint_metadata,
                    is_best=is_best
                )
                
                # Early stopping check
                if val_data is not None and early_stopping(current_loss, model):
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
            except Exception as e:
                self.logger.error(f"Error during epoch {epoch + 1}: {str(e)}")
                self.logger.exception("Detailed traceback:")
                continue
        
        # Training completed
        training_time = (datetime.now() - training_start_time).total_seconds()
        
        # Save final metrics
        metrics_tracker.save()
        
        # Get summary
        summary = metrics_tracker.get_summary()
        
        # Prepare results
        results = {
            'model_name': model_name,
            'data_type': self.data_type,
            'epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'metrics_summary': summary,
            'final_metrics': final_metrics,
            'data_shapes': {split: data.shape for split, data in data_dict.items()}
        }
        
        # Save results to files
        self.save_training_results(model_name, results, model, metrics_tracker)
        
        self.logger.info(f"\nTraining completed for {model_name} on {self.data_type.upper()} data")
        self.logger.info(f"Total time: {training_time/60:.2f} minutes")
        self.logger.info(f"Best validation loss: {best_val_loss:.6f}")
        
        return results
    
    def train_all_models(self, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train all specified models.
        
        Args:
            model_names: List of model names to train (None for all)
            
        Returns:
            Dictionary of training results
        """
        # Load data once
        data_dict = self.load_data()
        
        # Get list of models to train
        if model_names is None:
            model_names = [name for name in self.model_config.keys() 
                          if name != 'global_settings']
        
        self.logger.info(f"\nWill train {len(model_names)} models on {self.data_type.upper()} data: {model_names}")
        
        # Train each model
        all_results = {}
        
        for model_name in model_names:
            try:
                results = self.train_model(model_name, data_dict)
                if results:
                    all_results[model_name] = results
            except KeyboardInterrupt:
                self.logger.warning("Training interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {str(e)}")
                self.logger.exception("Detailed traceback:")
                continue
        
        # Save overall summary
        self.save_overall_summary(all_results)
        
        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info(f"TRAINING SUMMARY ({self.data_type.upper()} data)")
        self.logger.info("="*60)
        
        for model_name, results in all_results.items():
            self.logger.info(
                f"{model_name}: "
                f"Epochs={results['epochs_trained']}, "
                f"Best Loss={results['best_val_loss']:.6f}, "
                f"Time={results['training_time']/60:.1f}min"
            )
        
        return all_results
    
    def save_overall_summary(self, all_results: Dict[str, Any]) -> None:
        """
        Save overall training summary.
        
        Args:
            all_results: Dictionary of all training results
        """
        summary_file = self.dirs['results'] / f"training_summary_{self.data_type}.json"
        
        summary = {
            'data_type': self.data_type,
            'timestamp': datetime.now().isoformat(),
            'total_models': len(all_results),
            'successful_models': len([r for r in all_results.values() if r.get('best_val_loss', float('inf')) != float('inf')]),
            'models': {}
        }
        
        # Add individual model summaries
        for model_name, results in all_results.items():
            summary['models'][model_name] = {
                'epochs_trained': results['epochs_trained'],
                'best_val_loss': float(results['best_val_loss']) if results['best_val_loss'] != float('inf') else None,
                'training_time_minutes': results['training_time'] / 60,
                'success': results.get('best_val_loss', float('inf')) != float('inf')
            }
        
        # Find best model
        if summary['successful_models'] > 0:
            best_model = min(all_results.keys(), 
                           key=lambda m: all_results[m].get('best_val_loss', float('inf')))
            summary['best_model'] = {
                'name': best_model,
                'loss': float(all_results[best_model]['best_val_loss'])
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Overall summary saved to: {summary_file}")


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train CHF prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models on SMOTE data
  python scripts/train.py --data-type smote
  
  # Train all models on regular data
  python scripts/train.py --data-type regular
  
  # Train specific models on regular data
  python scripts/train.py --models xgboost lightgbm --data-type regular
  
  # Use custom config
  python scripts/train.py --config configs/custom_config.yaml
  
  # Debug mode
  python scripts/train.py --debug
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs',
        help='Path to configuration directory or file'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        type=str,
        help='Specific models to train (default: all)'
    )
    
    parser.add_argument(
        '--data-type',
        type=str,
        choices=['smote', 'regular'],
        default='smote',
        help='Type of training data to use (default: smote)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume training from checkpoint'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    config_path = Path(args.config)
    trainer = ModelTrainer(
        config_path=config_path, 
        debug=args.debug,
        data_type=args.data_type
    )
    
    # Run training
    try:
        trainer.train_all_models(model_names=args.models)
    except KeyboardInterrupt:
        trainer.logger.warning("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        trainer.logger.error(f"Training failed: {str(e)}")
        if args.debug:
            trainer.logger.exception("Detailed traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()