"""
XGBoost Model Implementation

This module implements the XGBoost model interface for the training pipeline.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Optional, Tuple
import pickle
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV 

class Xgboost:
    """XGBoost model wrapper for the training pipeline."""
    
    def __init__(self, **kwargs):
        """
        Initialize XGBoost model with given parameters.
        
        Args:
            **kwargs: Parameters passed to xgb.XGBRegressor/XGBClassifier
        """
        self.logger = kwargs.pop('logger', None)  # <-- Add this
        self.params = kwargs
        self.tuning_params = kwargs.pop('tuning', {})  # Extract tuning config
        
        # Determine if it's a classification or regression task
        objective = kwargs.get('objective', 'reg:squarederror')
        
        if 'binary' in objective or 'multi' in objective:
            self.model = xgb.XGBClassifier(**kwargs)
            self.task_type = 'classification'
        else:
            self.model = xgb.XGBRegressor(**kwargs)
            self.task_type = 'regression'
        
        self.is_fitted = False

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
    
    def tune(self, train_data: pd.DataFrame):
        """Perform hyperparameter tuning"""
        X, y = self._prepare_data(train_data)
        
        # Create parameter grid
        param_grid = {k: v for k, v in self.params.items() if isinstance(v, list)}
        
        # Select tuning method
        if self.tuning_params.get('method', 'random') == 'grid':
            search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=self.tuning_params.get('cv', 5),
                scoring=self.tuning_params.get('scoring', 'neg_mean_squared_error'),
                verbose=1  # Reduced from 2 to 1
        )
        else:
            search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_grid,
                n_iter=self.tuning_params.get('n_iter', 10),
                cv=self.tuning_params.get('cv', 5),
                scoring=self.tuning_params.get('scoring', 'neg_mean_squared_error'),
                verbose=1
            )
        
        # Run search
        search.fit(X, y)
        
        # Log only the best parameters
        best_params = search.best_params_
        if self.logger:
            self.logger.info(f"Best parameters: {best_params}")
            self.logger.info(f"Best score (MSE): {-search.best_score_:.4f}")
        
        # Update model
        self.model = search.best_estimator_
        self.params.update(best_params)
        return best_params
    
    
    def train_epoch(self, train_data: pd.DataFrame, 
                   batch_size: int = None,
                   optimizer: Any = None) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_data: Training dataframe
            batch_size: Not used for XGBoost (tree-based)
            optimizer: Not used for XGBoost
            
        Returns:
            Dictionary of training metrics
        """
        
        X_train, y_train = self._prepare_data(train_data)
        
        if not self.is_fitted:
            # Tune if configured
            if hasattr(self, 'tuning_params') and self.tuning_params:
                best_params = self.tune(train_data)
                print(f"Tuned parameters: {best_params}")
        
            # Train full model
            self.model.fit(X_train, y_train)
            self.is_fitted = True
        else:
            # Continue training (add one tree)
            self.model.set_params(n_estimators=self.model.n_estimators + 1)
            self.model.fit(X_train, y_train, xgb_model=self.model.get_booster())
        
        # For XGBoost, we fit the entire model at once
        if not self.is_fitted:
            self.model.fit(X_train, y_train)
            self.is_fitted = True
        else:
            # For subsequent epochs, we can continue training
            # by using xgb_model parameter in a new instance
            self.model.fit(
                X_train, 
                y_train, 
                xgb_model=self.model.get_booster()
            )
        
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
        self.task_type = save_dict['task_type']
        self.is_fitted = save_dict['is_fitted']
        
        return save_dict.get('metadata', {})
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
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
