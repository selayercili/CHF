"""
Neural Network Model Implementation

This module implements a neural network model interface for the training pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pickle

class NeuralNetwork:
    """Neural network model wrapper for the training pipeline."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1, 
                 learning_rate: float = 0.001, **kwargs):
        """
        Initialize neural network model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layers
            output_size: Size of output layer
            learning_rate: Learning rate for optimizer
        """
        self.model = self._build_model(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.is_fitted = False
        self.epoch = 0
        
    def _build_model(self, input_size, hidden_size, output_size) -> nn.Module:
        """Build the neural network architecture"""
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for training/validation.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple of (features, target) tensors
        """
        # Assume last column is target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        
        return X_tensor, y_tensor
    
    def train_epoch(self, train_data: pd.DataFrame, 
                    batch_size: int = 32,
                    optimizer: Any = None) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_data: Training dataframe
            batch_size: Batch size for training
            optimizer: Optimizer to use (will use self.optimizer if None)
            
        Returns:
            Dictionary of training metrics
        """
        # Prepare data
        X_train, y_train = self._prepare_data(train_data)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set model to training mode
        self.model.train()
        epoch_loss = 0.0
        
        # Use class optimizer if none provided
        optimizer = optimizer or self.optimizer
        
        for inputs, targets in dataloader:
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Calculate average loss
        avg_loss = epoch_loss / len(dataloader)
        self.epoch += 1
        self.is_fitted = True
        
        return {'loss': avg_loss, 'rmse': np.sqrt(avg_loss)}
    
    def validate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Validate the model on test data.
        
        Args:
            test_data: Test dataframe
            
        Returns:
            Dictionary of validation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before validation")
        
        # Prepare data
        X_test, y_test = self._prepare_data(test_data)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X_test, y_test)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        # Set model to evaluation mode
        self.model.eval()
        total_loss = 0.0
        mae = 0.0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, targets).item()
                mae += torch.nn.functional.l1_loss(outputs, targets).item()
        
        avg_loss = total_loss / len(dataloader)
        avg_mae = mae / len(dataloader)
        
        return {
            'loss': avg_loss,
            'rmse': np.sqrt(avg_loss),
            'mae': avg_mae
        }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        # If data has target column, remove it
        if data.shape[1] == self.model[0].in_features + 1:
            X = data.iloc[:, :-1].values
        else:
            X = data.values
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions.flatten()
    
    def save(self, path: Path, metadata: Dict[str, Any] = None):
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
            metadata: Additional metadata to save with the model
        """
        save_dict = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'is_fitted': self.is_fitted,
            'device': self.device.type
        }
        
        if metadata:
            save_dict['metadata'] = metadata
        
        torch.save(save_dict, path)
    
    def load(self, path: Path):
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epoch = checkpoint['epoch']
        self.is_fitted = checkpoint['is_fitted']
        
        return checkpoint.get('metadata', {})
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance using gradient-based method.
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Create dummy input
        dummy_input = torch.ones(1, self.model[0].in_features, device=self.device)
        dummy_input.requires_grad = True
        
        # Forward pass
        output = self.model(dummy_input)
        
        # Backward pass to get gradients
        output.backward()
        
        # Get absolute gradients
        gradients = torch.abs(dummy_input.grad).cpu().numpy().flatten()
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(len(gradients))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': gradients
        }).sort_values('importance', ascending=False)
        
        return importance_df