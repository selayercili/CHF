#!/usr/bin/env python3
"""
1) Import in train and test data
2) Loop through the models
3) Train the models
4) Save the model weights in a standard location ./weights/{model_name}/{epoch}.pth

from model_1 import model_1
from model_2 import model_2

model_names = [model_1, model_2]
config_args = yaml.safe_load(configs/model_config.yaml)

for model_name in model_names:
    model = model_1(train, test, config_args[model_name])
    model.load_weights(f"./weights/{model_name}/{epoch}.pth")
    model.test()

Model Training Pipeline Script

This script handles the training of multiple models with configurable parameters.
It loads data, trains models, saves checkpoints, and logs progress.

Usage:
    python scripts/train_models.py [--config CONFIG_PATH] [--debug]
"""


import warnings
from sklearn.exceptions import FutureWarning

# Gets rid of noisy warnings from sklearn to keep output clean
warnings.filterwarnings("ignore", category=FutureWarning, 
                        message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.")

import os
import sys
import argparse
import logging
import importlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import yaml
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelTrainer:
    """Handles model training pipeline for multiple models."""
    
    def __init__(self, config_path: str = "configs/model_configs.yaml"):
        """
        Initialize the ModelTrainer.
        
        Args:
            config_path: Path to the model configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.weights_dir = Path("./weights")
        self.data_dir = Path("data/processed")
        
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
        logger = logging.getLogger("ModelTrainer")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_dir / f"training_{timestamp}.log")
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
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        self.logger.info("Loading data...")
        
        train_path = self.data_dir / "train.csv"
        test_path = self.data_dir / "test.csv"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        train_df = pd.read_csv(train_path)
        self.logger.info(f"Loaded training data: {train_df.shape}")
        
        # Load test data if it exists
        test_df = None
        if test_path.exists():
            test_df = pd.read_csv(test_path)
            self.logger.info(f"Loaded test data: {test_df.shape}")
        else:
            self.logger.warning("Test data not found. Proceeding without test set.")
        
        return train_df, test_df
    
    def _get_model_class(self, model_name: str):
        """
        Dynamically import and return model class.
        
        Args:
            model_name: Name of the model to import
            
        Returns:
            Model class
        """
        try:
            # Convert model name to module path (e.g., "xgboost" -> "src.models.xgboost")
            module_path = f"src.models.{model_name}"
            module = importlib.import_module(module_path)
            
            # Get the model class (assuming it has the same name as the module)
            # You might need to adjust this based on your naming convention
            model_class_name = ''.join(word.capitalize() for word in model_name.split('_'))
            model_class = getattr(module, model_class_name)
            
            return model_class
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Failed to import model {model_name}: {e}")
            raise
    
    def _create_checkpoint_dir(self, model_name: str) -> Path:
        """Create directory for model checkpoints."""
        checkpoint_dir = self.weights_dir / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir
    
    def train_model(self, model_name: str, train_data: pd.DataFrame, 
                   test_data: pd.DataFrame = None) -> None:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            train_data: Training dataframe
            test_data: Test dataframe (optional)
        """
        self.logger.info(f"Training {model_name}...")
        
        # Get model configuration
        if model_name not in self.config:
            self.logger.error(f"No configuration found for {model_name}")
            return
        
        model_config = self.config[model_name]
        
        # Get model class
        try:
            model_class = self._get_model_class(model_name)
        except Exception as e:
            self.logger.error(f"Skipping {model_name} due to import error: {e}")
            return
        
        # Create checkpoint directory
        checkpoint_dir = self._create_checkpoint_dir(model_name)
        
        # Initialize model
        try:
            model = model_class(**model_config.get('init_params', {}))
            self.logger.info(f"Initialized {model_name} with config: {model_config}")
        except Exception as e:
            self.logger.error(f"Failed to initialize {model_name}: {e}")
            return
        
        # Training parameters
        epochs = model_config.get('epochs', 10)
        batch_size = model_config.get('batch_size', 32)
        learning_rate = model_config.get('learning_rate', 0.001)
        
        # Setup optimizer if model uses PyTorch
        if hasattr(model, 'parameters'):
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=learning_rate
            )
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            try:
                # Train for one epoch
                train_metrics = model.train_epoch(
                    train_data, 
                    batch_size=batch_size,
                    optimizer=optimizer if hasattr(model, 'parameters') else None
                )
                
                # Validate if test data is available
                val_metrics = {}
                if test_data is not None:
                    val_metrics = model.validate(test_data)
                
                # Log metrics
                self.logger.info(
                    f"Train metrics: {train_metrics} | "
                    f"Val metrics: {val_metrics}"
                )
                
                # Save checkpoint
                checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pth"
                self._save_checkpoint(
                    model, 
                    checkpoint_path, 
                    epoch + 1,
                    train_metrics,
                    val_metrics
                )
                
                # Save best model
                current_loss = val_metrics.get('loss', train_metrics.get('loss', float('inf')))
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_checkpoint_path = checkpoint_dir / "best_model.pth"
                    self._save_checkpoint(
                        model,
                        best_checkpoint_path,
                        epoch + 1,
                        train_metrics,
                        val_metrics,
                        is_best=True
                    )
                    self.logger.info(f"Saved best model with loss: {best_loss:.4f}")
                    
            except Exception as e:
                self.logger.error(f"Error during epoch {epoch + 1}: {e}")
                continue
        
        self.logger.info(f"Completed training for {model_name}")
    
    def _save_checkpoint(self, model, checkpoint_path: Path, epoch: int,
                        train_metrics: Dict, val_metrics: Dict,
                        is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Handle different model types
        if hasattr(model, 'state_dict'):
            # PyTorch model
            checkpoint['model_state_dict'] = model.state_dict()
            torch.save(checkpoint, checkpoint_path)
        elif hasattr(model, 'save'):
            # Custom save method
            model.save(checkpoint_path, metadata=checkpoint)
        else:
            # Fallback: pickle the entire model
            import pickle
            checkpoint['model'] = model
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
        
        log_msg = f"Saved checkpoint: {checkpoint_path}"
        if is_best:
            log_msg = f"Saved best model: {checkpoint_path}"
        self.logger.info(log_msg)
    
    def train_all_models(self) -> None:
        """Train all models specified in the configuration."""
        # Load data once
        train_data, test_data = self.load_data()
        
        # Get list of models to train
        model_names = list(self.config.keys())
        self.logger.info(f"Found {len(model_names)} models to train: {model_names}")
        
        # Train each model
        for model_name in model_names:
            try:
                self.train_model(model_name, train_data, test_data)
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        self.logger.info("Training pipeline completed!")


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train multiple models with configuration"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_configs.yaml',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(config_path=args.config)
    
    # Set debug logging if requested
    if args.debug:
        trainer.logger.setLevel(logging.DEBUG)
    
    # Run training pipeline
    try:
        trainer.train_all_models()
    except KeyboardInterrupt:
        trainer.logger.info("Training interrupted by user")
    except Exception as e:
        trainer.logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
