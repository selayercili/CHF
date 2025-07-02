"""
Support Vector Machine (SVM) Model Implementation

This module implements the SVM model interface for the training pipeline.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class Svm:
    """SVM model wrapper for regression tasks."""
    
    def __init__(self, **kwargs):
        """
        Initialize SVM model with given parameters.
        
        Args:
            **kwargs: Parameters passed to sklearn.svm.SVR
        """
        self.model = SVR(**kwargs)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training/validation.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple of (scaled features, target)
        """
        # Assume last column is target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # Scale features (critical for SVM performance)
        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled, y
    
    def train_epoch(self, train_data: pd.DataFrame, 
                   batch_size: int = None,
                   optimizer: Any = None) -> Dict[str, float]:
        """
        Train the SVM model (full training in one epoch).
        
        Args:
            train_data: Training dataframe
            batch_size: Not used for SVM
            optimizer: Not used for SVM
            
        Returns:
            Dictionary of training metrics
        """
        # Prepare data
        X_train, y_train = self._prepare_data(train_data)
        
        # Train only once (SVM doesn't support incremental training)
        if not self.is_fitted:
            self.model.fit(X_train, y_train)
            self.is_fitted = True
        
        # Calculate training metrics
        predictions = self.model.predict(X_train)
        mse = np.mean((predictions - y_train) ** 2)
        rmse = np.sqrt(mse)
        
        return {'loss': mse, 'rmse': rmse}
    
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
        
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))
        
        return {'loss': mse, 'rmse': rmse, 'mae': mae}
    
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
        if data.shape[1] == self.scaler.n_features_in_ + 1:
            X = data.iloc[:, :-1].values
        else:
            X = data.values
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path: Path, metadata: Dict[str, Any] = None):
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
            metadata: Additional metadata to save with the model
        """
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
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
        self.scaler = save_dict['scaler']
        self.is_fitted = save_dict['is_fitted']
        
        return save_dict.get('metadata', {})
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance using model coefficients (linear kernel only).
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Only works for linear kernel
        if self.model.kernel != 'linear':
            raise NotImplementedError("Feature importance only available for linear kernel")
        
        coefficients = self.model.coef_[0]
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(len(coefficients))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(coefficients)
        }).sort_values('importance', ascending=False)
        
        return importance_df