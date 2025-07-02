import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

class Lightgbm:
    
    def __init__(self, **kwargs):
        # Store original learning rate
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        
        # Determine task type
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
        """Prepare data for training/validation."""
        # Assume last column is target
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X, y
    
    def train_epoch(self, train_data: pd.DataFrame, 
                   batch_size: int = None,
                   optimizer: Any = None) -> Dict[str, float]:
        """Fixed training method with proper learning rate handling."""
        X_train, y_train = self._prepare_data(train_data)
        
        # For first epoch: initialize model
        if not self.is_fitted:
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            self.best_iteration = 1
        else:
            # Continue training with fixed learning rate
            try:
                # Try getting learning rate from booster
                current_lr = self.model.booster_.params.get('learning_rate', self.learning_rate)
            except AttributeError:
                current_lr = self.learning_rate
                
            self.model.fit(
                X_train, 
                y_train,
                init_model=self.model,
                callbacks=[lgb.reset_parameter(learning_rate=current_lr)],
                num_iteration=1
            )
            self.best_iteration += 1
        
        # Calculate training metrics
        return self._calculate_metrics(X_train, y_train)
    
    def _calculate_metrics(self, X, y) -> Dict[str, float]:
        """Calculate metrics based on task type."""
        predictions = self.model.predict(X)
        
        if self.task_type == 'regression':
            mse = np.mean((predictions - y) ** 2)
            rmse = np.sqrt(mse)
            return {'loss': mse, 'rmse': rmse}
        else:
            accuracy = np.mean(predictions == y)
            return {'loss': 1 - accuracy, 'accuracy': accuracy}
    
    def validate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Model must be trained before validation")
        
        X_test, y_test = self._prepare_data(test_data)
        predictions = self.model.predict(X_test)
        
        if self.task_type == 'regression':
            mse = np.mean((predictions - y_test) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - y_test))
            return {'loss': mse, 'rmse': rmse, 'mae': mae}
        else:
            accuracy = np.mean(predictions == y_test)
            return {'loss': 1 - accuracy, 'accuracy': accuracy}
    
    # ... (predict, save, load, get_feature_importance remain unchanged) ...