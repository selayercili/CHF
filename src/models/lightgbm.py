"""
LightGBM Model Implementation

This module implements the LightGBM model interface for the training pipeline.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

class Lightgbm:
    """LightGBM model wrapper for the training pipeline."""
    
    def __init__(self, **kwargs):
        """
        Initialize LightGBM model with given parameters.
        
        Args:
            **kwargs: Parameters passed to lgb.LGBMRegressor/LGBMClassifier
        """
        # Determine if it's a classification or regression task
        objective = kwargs.get('objective', 'regression')
        
        if 'binary' in objective or 'multi' in objective:
            self.model = lgb.LGBMClassifier(**kwargs)
            self.task_type = 'classification'
        else:
            self.model = lgb.LGBMRegressor(**kwargs)
            self.task_type = 'regression'
        
        self.is_fitted = False
        self.best_iteration = 0
        
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training/validation.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple of (features, target)
        """
        # Assume last column is target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        return X, y
    
    def train_epoch(self, train_data: pd.DataFrame, 
                   batch_size: int = None,
                   optimizer: Any = None) -> Dict[str, float]:
        """
        Train the model for one epoch (boosting round).
        
        Args:
            train_data: Training dataframe
            batch_size: Not used for LightGBM
            optimizer: Not used for LightGBM
            
        Returns:
            Dictionary of training metrics
        """
        X_train, y_train = self._prepare_data(train_data)
        
        # For first epoch: initialize model
        if not self.is_fitted:
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            self.best_iteration = 1
        else:
            # Continue training from current state
            self.model.fit(
                X_train, 
                y_train,
                init_model=self.model,
                callbacks=[lgb.reset_parameter(learning_rate=self.model.learning_rate_)],
                num_iteration=1  # Train just one more boosting round
            )
            self.best_iteration += 1
        
        # Calculate training metrics
        predictions = self.model.predict(X_train)
        
        if self.task_type == 'regression':
            mse = np.mean((predictions - y_train) ** 2)
            rmse = np.sqrt(mse)
            metrics = {'loss': mse, 'rmse': rmse}
        else:
            accuracy = np.mean(predictions == y_train)
            metrics = {'loss': 1 - accuracy, 'accuracy': accuracy}
        
        return metrics
    
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
        
        X_test, y_test = self._prepare_data(test_data)
        predictions = self.model.predict(X_test)
        
        if self.task_type == 'regression':
            mse = np.mean((predictions - y_test) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - y_test))
            metrics = {'loss': mse, 'rmse': rmse, 'mae': mae}
        else:
            accuracy = np.mean(predictions == y_test)
            metrics = {'loss': 1 - accuracy, 'accuracy': accuracy}
        
        return metrics
    
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
        if data.shape[1] == self.model.n_features_in_ + 1:
            X = data.iloc[:, :-1].values
        else:
            X = data.values
        
        return self.model.predict(X)
    
    def save(self, path: Path, metadata: Dict[str, Any] = None):
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
            metadata: Additional metadata to save with the model
        """
        save_dict = {
            'model': self.model,
            'task_type': self.task_type,
            'is_fitted': self.is_fitted,
            'best_iteration': self.best_iteration
        }
        
        if metadata:
            save_dict['metadata'] = metadata
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load(self, path: Path):
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
        """
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.model = save_dict['model']
        self.task_type = save_dict['task_type']
        self.is_fitted = save_dict['is_fitted']
        self.best_iteration = save_dict['best_iteration']
        
        return save_dict.get('metadata', {})
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: 'split' or 'gain'
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance = self.model.feature_importances_
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(len(importance))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df