"""
Gaussian Process Regressor Model Implementation

This module implements the Gaussian Process model interface for heat flux prediction.
GPR is excellent for physical phenomena with uncertainty quantification capabilities.
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Tuple, List
import pickle
from pathlib import Path
import warnings


class GaussianProcess:
    """Gaussian Process model wrapper for the training pipeline."""
    
    def __init__(self, **kwargs):
        """
        Initialize Gaussian Process model with given parameters.
        
        Args:
            **kwargs: Parameters passed to GaussianProcessRegressor/GaussianProcessClassifier
        """
        # Extract special parameters
        self.logger = kwargs.pop('logger', None)
        self.tuning_params = kwargs.pop('tuning', {})
        
        # Store original parameters
        self.params = kwargs.copy()
        
        # Set default kernel if not provided
        if 'kernel' not in self.params:
            # Default: RBF kernel with automatic relevance determination
            length_scale = kwargs.get('length_scale', 1.0)
            self.params['kernel'] = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel()
        
        # Set other default parameters
        default_params = {
            'alpha': 1e-10,  # Nugget for numerical stability
            'n_restarts_optimizer': 5,  # Multiple optimization restarts
            'normalize_y': True,  # Normalize target values
            'random_state': 42
        }
        
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
        
        # Determine task type
        task_type = kwargs.get('task_type', 'regression')
        
        if task_type == 'classification':
            self.model = GaussianProcessClassifier(**self.params)
            self.task_type = 'classification'
        else:
            self.model = GaussianProcessRegressor(**self.params)
            self.task_type = 'regression'
        
        self.is_fitted = False
        self.epoch = 0
        self.feature_names = None
        self.feature_scaler = StandardScaler()
        self.train_X = None  # Store training data for feature importance
        self.train_y = None
        
        # Suppress GP warnings about optimization
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process")
    
    def _prepare_data(self, data: pd.DataFrame, fit_scaler: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training/validation.
        
        Args:
            data: DataFrame with features and target
            fit_scaler: Whether to fit the feature scaler
            
        Returns:
            Tuple of (features, target)
        """
        # Assume last column is target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        # Store feature names for importance analysis
        if not self.is_fitted:
            self.feature_names = list(data.columns[:-1])
        
        # Scale features for better GP performance
        if fit_scaler or not hasattr(self.feature_scaler, 'scale_'):
            X = self.feature_scaler.fit_transform(X)
        else:
            X = self.feature_scaler.transform(X)
        
        return X, y
    
    def _create_kernel_from_config(self, kernel_config: Dict[str, Any]):
        """Create kernel from configuration dictionary."""
        kernel_type = kernel_config.get('type', 'rbf')
        
        if kernel_type == 'rbf':
            length_scale = kernel_config.get('length_scale', 1.0)
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale)
        elif kernel_type == 'matern':
            length_scale = kernel_config.get('length_scale', 1.0)
            nu = kernel_config.get('nu', 1.5)
            kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=nu)
        elif kernel_type == 'rational_quadratic':
            length_scale = kernel_config.get('length_scale', 1.0)
            alpha = kernel_config.get('alpha', 1.0)
            kernel = ConstantKernel(1.0) * RationalQuadratic(length_scale=length_scale, alpha=alpha)
        else:
            # Default to RBF
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        
        # Add white noise kernel for numerical stability
        if kernel_config.get('add_white_noise', True):
            kernel = kernel + WhiteKernel()
        
        return kernel
    
    def tune(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            train_data: Training dataframe
            
        Returns:
            Dictionary of best parameters
        """
        X, y = self._prepare_data(train_data, fit_scaler=True)
        
        # Define parameter search space
        # Note: Kernel tuning is complex, so we'll focus on other parameters
        param_distributions = {
            'alpha': [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
            'n_restarts_optimizer': [0, 2, 5, 10],
            'normalize_y': [True, False]
        }
        
        # Use only parameters that are lists in current config
        param_grid = {}
        for param, values in param_distributions.items():
            if param in self.params and isinstance(self.params[param], list):
                param_grid[param] = self.params[param]
            elif param not in self.params or param == 'alpha':  # Always allow alpha tuning
                param_grid[param] = values
        
        if not param_grid:
            if self.logger:
                self.logger.info("No tuning parameters specified, skipping hyperparameter search")
            return {}
        
        # Select search method
        search_method = self.tuning_params.get('method', 'random')
        cv_folds = self.tuning_params.get('cv', 3)  # Use fewer folds for GP (computationally expensive)
        
        if search_method == 'grid':
            search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring='neg_mean_squared_error' if self.task_type == 'regression' else 'accuracy',
                n_jobs=1,  # GP doesn't parallelize well
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_grid,
                n_iter=min(self.tuning_params.get('n_iter', 10), 10),  # Limit iterations for GP
                cv=cv_folds,
                scoring='neg_mean_squared_error' if self.task_type == 'regression' else 'accuracy',
                n_jobs=1,
                random_state=42,
                verbose=1
            )
        
        # Perform search
        if self.logger:
            self.logger.info(f"Starting {search_method} search for Gaussian Process...")
        
        search.fit(X, y)
        
        # Update model with best parameters
        self.model = search.best_estimator_
        best_params = search.best_params_
        self.params.update(best_params)
        
        if self.logger:
            self.logger.info(f"Best parameters found: {best_params}")
            self.logger.info(f"Best CV score: {search.best_score_:.6f}")
        
        return best_params
    
    def train_epoch(self, train_data: pd.DataFrame, 
                   batch_size: int = None,
                   optimizer: Any = None) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_data: Training dataframe
            batch_size: Not used for Gaussian Process
            optimizer: Not used for Gaussian Process
            
        Returns:
            Dictionary of training metrics
        """
        print(f"🔧 Training Gaussian Process - Epoch {self.epoch + 1}")
        
        # Perform tuning if configured and not fitted yet
        if not self.is_fitted and self.tuning_params:
            print("🔍 Performing hyperparameter tuning...")
            self.tune(train_data)
            # Don't return early - continue with normal training
        
        X_train, y_train = self._prepare_data(train_data, fit_scaler=not self.is_fitted)
        
        print(f"📊 Training data shape: {X_train.shape}, Target shape: {y_train.shape}")
        
        # Store training data for feature importance calculation
        self.train_X = X_train
        self.train_y = y_train
        
        # For Gaussian Process, we fit the entire model at once
        if not self.is_fitted:
            print("🚀 Fitting Gaussian Process (this may take a while)...")
            if self.logger:
                self.logger.info("Fitting Gaussian Process (this may take a while)...")
            
            try:
                self.model.fit(X_train, y_train)
                self.is_fitted = True
                print("✅ Gaussian Process fitted successfully!")
            except Exception as e:
                print(f"❌ Error fitting Gaussian Process: {str(e)}")
                raise
        else:
            # For subsequent epochs, we refit (GP doesn't have incremental learning)
            print("🔄 Refitting Gaussian Process...")
            self.model.fit(X_train, y_train)
        
        self.epoch += 1
        
        # Calculate training metrics
        print("📈 Calculating training metrics...")
        predictions = self.model.predict(X_train)
        
        if self.task_type == 'regression':
            mse = np.mean((predictions - y_train) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - y_train))
            
            # Log marginal likelihood (model's internal score)
            log_likelihood = getattr(self.model, 'log_marginal_likelihood_value_', None)
            
            metrics = {'loss': mse, 'rmse': rmse, 'mae': mae}
            if log_likelihood is not None:
                metrics['log_marginal_likelihood'] = log_likelihood
                
            print(f"📊 Training metrics: RMSE={rmse:.6f}, MAE={mae:.6f}")
        else:
            accuracy = np.mean(predictions == y_train)
            metrics = {'loss': 1 - accuracy, 'accuracy': accuracy}
            print(f"📊 Training metrics: Accuracy={accuracy:.6f}")
        
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
        
        X_test, y_test = self._prepare_data(test_data, fit_scaler=False)
        
        if self.task_type == 'regression':
            # Get predictions with uncertainty
            predictions, std = self.model.predict(X_test, return_std=True)
            
            mse = np.mean((predictions - y_test) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - y_test))
            
            # Additional GP-specific metrics
            mean_uncertainty = np.mean(std)
            max_uncertainty = np.max(std)
            
            # R-squared
            ss_res = np.sum((y_test - predictions) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            metrics = {
                'loss': mse, 
                'rmse': rmse, 
                'mae': mae,
                'r2': r2,
                'max_error': np.max(np.abs(predictions - y_test)),
                'mean_uncertainty': mean_uncertainty,
                'max_uncertainty': max_uncertainty
            }
        else:
            predictions = self.model.predict(X_test)
            accuracy = np.mean(predictions == y_test)
            metrics = {'loss': 1 - accuracy, 'accuracy': accuracy}
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: DataFrame with features (no target column expected)
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        # For prediction, assume all columns are features
        X = data.values
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Return only mean predictions (not uncertainty)
        return self.model.predict(X_scaled)
    
    def predict_with_uncertainty(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Tuple of (predictions, standard_deviations)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        if self.task_type != 'regression':
            raise ValueError("Uncertainty prediction only available for regression")
        
        X = data.values
        X_scaled = self.feature_scaler.transform(X)
        
        return self.model.predict(X_scaled, return_std=True)
    
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
            'epoch': self.epoch,
            'params': self.params,
            'tuning_params': self.tuning_params,
            'feature_names': self.feature_names,
            'feature_scaler': self.feature_scaler,
            'train_X': self.train_X,
            'train_y': self.train_y
        }
        
        if metadata:
            save_dict['metadata'] = metadata
        
        # Ensure path has .pkl extension
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix('.pkl')
        
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(save_dict, f)
            
            print(f"✓ Saved Gaussian Process model to {path}")
            if self.logger:
                self.logger.info(f"Saved Gaussian Process model to {path}")
        except Exception as e:
            error_msg = f"Failed to save model to {path}: {str(e)}"
            print(f"✗ {error_msg}")
            if self.logger:
                self.logger.error(error_msg)
            raise
    
    def load(self, path: Path):
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Dictionary of metadata
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.model = save_dict['model']
        self.task_type = save_dict['task_type']
        self.is_fitted = save_dict['is_fitted']
        self.epoch = save_dict.get('epoch', 0)
        self.params = save_dict.get('params', {})
        self.tuning_params = save_dict.get('tuning_params', {})
        self.feature_names = save_dict.get('feature_names', None)
        self.feature_scaler = save_dict.get('feature_scaler', StandardScaler())
        self.train_X = save_dict.get('train_X', None)
        self.train_y = save_dict.get('train_y', None)
        
        return save_dict.get('metadata', {})
    
    def get_feature_importance(self, method: str = 'permutation') -> pd.DataFrame:
        """
        Get feature importance scores using permutation importance.
        
        Args:
            method: Method for calculating importance ('permutation' or 'kernel_gradients')
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.train_X is None or self.train_y is None:
            raise ValueError("Training data not available for importance calculation")
        
        if method == 'permutation':
            # Permutation importance
            from sklearn.inspection import permutation_importance
            
            # Use a subset of training data for efficiency
            n_samples = min(200, len(self.train_X))
            indices = np.random.choice(len(self.train_X), n_samples, replace=False)
            X_subset = self.train_X[indices]
            y_subset = self.train_y[indices]
            
            perm_importance = permutation_importance(
                self.model, X_subset, y_subset,
                n_repeats=5, random_state=42, scoring='neg_mean_squared_error'
            )
            
            importance_scores = perm_importance.importances_mean
            
        elif method == 'kernel_gradients':
            # Kernel-based importance (simplified approach)
            # This is a custom method specific to GP
            n_features = self.train_X.shape[1]
            importance_scores = np.zeros(n_features)
            
            # Use kernel hyperparameters as proxy for feature importance
            if hasattr(self.model.kernel_, 'get_params'):
                kernel_params = self.model.kernel_.get_params()
                
                # Look for length scale parameters
                if 'length_scale' in kernel_params:
                    length_scales = kernel_params['length_scale']
                    if hasattr(length_scales, '__len__') and len(length_scales) == n_features:
                        # Inverse of length scale indicates importance
                        importance_scores = 1.0 / (length_scales + 1e-10)
                    else:
                        # Single length scale - equal importance
                        importance_scores = np.ones(n_features)
                else:
                    # Fallback to permutation
                    return self.get_feature_importance(method='permutation')
        else:
            # Fallback to permutation
            return self.get_feature_importance(method='permutation')
        
        # Normalize importance scores
        importance_scores = importance_scores / np.sum(importance_scores)
        
        # Use stored feature names if available
        if self.feature_names:
            feature_names = self.feature_names
        else:
            feature_names = [f'feature_{i}' for i in range(len(importance_scores))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        info = {
            'status': 'fitted',
            'task_type': self.task_type,
            'kernel': str(self.model.kernel_),
            'n_features': len(self.feature_names) if self.feature_names else 'unknown',
            'epoch': self.epoch,
            'log_marginal_likelihood': getattr(self.model, 'log_marginal_likelihood_value_', None),
            'alpha': self.params.get('alpha', 'unknown')
        }
        
        return info
    
    def get_kernel_parameters(self) -> Dict[str, Any]:
        """
        Get the optimized kernel parameters.
        
        Returns:
            Dictionary of kernel parameters
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting kernel parameters")
        
        kernel_params = {}
        if hasattr(self.model, 'kernel_'):
            kernel_params = self.model.kernel_.get_params()
        
        return kernel_params