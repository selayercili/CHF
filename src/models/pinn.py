"""
Simplified and Fixed Physics-Informed Neural Network (PINN) Model

Key fixes:
1. Simplified data handling with consistent types
2. Better physics loss integration with proper weighting
3. Cleaner scaling and normalization
4. Improved error handling and debugging
5. More stable training procedure
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

class FixedPinn:
    """Simplified and more reliable PINN implementation."""
    
    def __init__(self, hidden_layers: list = None, learning_rate: float = 0.001, 
                 lambda_physics: float = 0.01, **kwargs):
        """
        Args:
            hidden_layers: List of hidden layer sizes [64, 64, 32]
            learning_rate: Optimizer learning rate
            lambda_physics: Weight for physics loss term (reduced default)
        """
        self.hidden_layers = hidden_layers or [128, 64, 32]  # Larger first layer
        self.learning_rate = learning_rate
        self.lambda_physics = lambda_physics
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted = False
        self.epoch = 0
        self.input_size = None
        
        # Simplified scaling - use StandardScaler for both
        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Training parameters
        self.debug = kwargs.get('debug', False)
        self.physics_warmup_epochs = kwargs.get('physics_warmup_epochs', 50)
        self.min_epochs_before_physics = 30  # Longer warmup
        
        # Store feature names for consistency
        self.feature_names = None
        
        # Training history
        self.loss_history = {'total': [], 'data': [], 'physics': [], 'epoch': []}
        
        print(f"Initialized FixedPINN with device: {self.device}")

    def _prepare_data(self, data: pd.DataFrame, is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simplified and robust data preparation."""
        
        if self.debug:
            print(f"_prepare_data - Input shape: {data.shape}, Training: {is_training}")
            print(f"_prepare_data - Columns: {list(data.columns)}")
        
        # Find target column more reliably
        target_col = None
        possible_targets = ['chf_exp', 'CHF_exp', 'CHF', 'target', 'chf']
        
        for col in data.columns:
            col_lower = col.lower()
            if any(target in col_lower for target in possible_targets):
                target_col = col
                break
        
        if target_col is None:
            # Use last column as fallback
            target_col = data.columns[-1]
            print(f"Warning: Using last column as target: {target_col}")
        
        # Get feature columns (all except target)
        feature_cols = [col for col in data.columns if col != target_col]
        
        if self.debug:
            print(f"Target column: {target_col}")
            print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
        
        # Store feature names on first training
        if is_training and self.feature_names is None:
            self.feature_names = feature_cols.copy()
            print(f"Stored feature names: {self.feature_names}")
        elif not is_training and self.feature_names is not None:
            # Check consistency
            if set(feature_cols) != set(self.feature_names):
                print(f"WARNING: Feature mismatch!")
                print(f"Training features: {self.feature_names}")
                print(f"Test features: {feature_cols}")
                # Use training feature order
                feature_cols = self.feature_names
        
        # Extract data with consistent types
        try:
            X = data[feature_cols].astype(np.float32).values
            y = data[target_col].astype(np.float32).values
        except KeyError as e:
            raise ValueError(f"Missing columns: {e}")
        except ValueError as e:
            raise ValueError(f"Data conversion failed: {e}")
        
        # Handle missing values
        if np.isnan(X).any() or np.isnan(y).any():
            print("Warning: Found NaN values, replacing with column means")
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
            y = np.nan_to_num(y, nan=np.nanmean(y))
        
        if self.debug:
            print(f"Raw data - X shape: {X.shape}, y shape: {y.shape}")
            print(f"Raw data - X range: [{X.min():.3f}, {X.max():.3f}]")
            print(f"Raw data - y range: [{y.min():.3f}, {y.max():.3f}]")
        
        # Scale data
        if is_training:
            X_scaled = self.input_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            if self.debug:
                print("Fitted scalers")
        else:
            X_scaled = self.input_scaler.transform(X)
            y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        # Initialize model if needed
        if self.model is None:
            self.input_size = X_scaled.shape[1]
            self.model = self._build_model(self.input_size)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=1e-4
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.7, patience=15, verbose=True
            )
            print(f"✓ Initialized model with {self.input_size} input features")
        
        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)
        
        return X_tensor, y_tensor

    def _build_model(self, input_size: int) -> nn.Module:
        """Build improved neural network."""
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(self.hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.GELU())  # Better activation than LeakyReLU
            if i < len(self.hidden_layers) - 1:
                layers.append(nn.Dropout(0.1))  # Reduced dropout
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        model = nn.Sequential(*layers).to(self.device)
        
        # Better weight initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model architecture: {[input_size] + self.hidden_layers + [1]}")
        print(f"Total parameters: {total_params}")
        
        return model

    def _physics_loss_simple(self, inputs: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """Simplified physics loss that's more stable."""
        try:
            # Convert to physical units
            pred_physical = self._to_physical_units(predictions)
            
            # Simple physics constraints based on domain knowledge
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # 1. CHF should be positive
            negative_penalty = torch.mean(torch.relu(-pred_physical) ** 2)
            
            # 2. CHF shouldn't be extremely high (> 50 MW/m²)
            extreme_penalty = torch.mean(torch.relu(pred_physical - 50.0) ** 2)
            
            # 3. Basic physical relationship: higher pressure -> higher CHF
            if inputs.shape[1] >= 2:
                pressure_scaled = inputs[:, 0]  # Assuming first column is pressure
                mass_flux_scaled = inputs[:, 1]  # Assuming second column is mass flux
                
                # Sort by pressure and check if CHF generally increases
                pressure_sorted, indices = torch.sort(pressure_scaled)
                chf_sorted = pred_physical[indices]
                
                # Penalize if CHF decreases too much with increasing pressure
                chf_diff = chf_sorted[1:] - chf_sorted[:-1]
                decreasing_penalty = torch.mean(torch.relu(-chf_diff - 0.5) ** 2)
            else:
                decreasing_penalty = torch.tensor(0.0, device=self.device)
            
            loss = loss + 0.1 * negative_penalty + 0.05 * extreme_penalty + 0.02 * decreasing_penalty
            
            if self.debug and torch.rand(1) < 0.01:  # Occasional debug
                print(f"Physics penalties - Negative: {negative_penalty.item():.6f}, "
                      f"Extreme: {extreme_penalty.item():.6f}, Decreasing: {decreasing_penalty.item():.6f}")
            
            return loss
            
        except Exception as e:
            if self.debug:
                print(f"Physics loss failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _to_physical_units(self, predictions: torch.Tensor) -> torch.Tensor:
        """Convert scaled predictions back to physical units."""
        pred_np = predictions.detach().cpu().numpy().reshape(-1, 1)
        pred_physical_np = self.target_scaler.inverse_transform(pred_np).flatten()
        return torch.tensor(pred_physical_np, device=predictions.device, requires_grad=True)

    def train_epoch(self, train_data: pd.DataFrame, batch_size: int = 64, **kwargs) -> Dict[str, float]:
        """Training with better loss balancing and stability."""
        
        X_train, y_train = self._prepare_data(train_data, is_training=True)
        
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        
        self.model.train()
        
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        n_batches = 0
        
        # Gradual physics weight increase
        if self.epoch < self.min_epochs_before_physics:
            physics_weight = 0.0
        elif self.epoch < self.physics_warmup_epochs:
            progress = (self.epoch - self.min_epochs_before_physics) / (self.physics_warmup_epochs - self.min_epochs_before_physics)
            physics_weight = self.lambda_physics * progress
        else:
            physics_weight = self.lambda_physics
        
        for inputs, targets in dataloader:
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs).squeeze()
            
            # Data loss
            data_loss = nn.functional.mse_loss(outputs, targets)
            
            # Physics loss (only after warmup)
            if physics_weight > 0:
                physics_loss = self._physics_loss_simple(inputs, outputs)
            else:
                physics_loss = torch.tensor(0.0, device=self.device)
            
            # Combined loss
            total_batch_loss = data_loss + physics_weight * physics_loss
            
            # Check for numerical issues
            if not torch.isfinite(total_batch_loss):
                print(f"Warning: Non-finite loss at epoch {self.epoch}")
                continue
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item()
            n_batches += 1
        
        # Calculate averages
        avg_total_loss = total_loss / n_batches if n_batches > 0 else 0
        avg_data_loss = total_data_loss / n_batches if n_batches > 0 else 0
        avg_physics_loss = total_physics_loss / n_batches if n_batches > 0 else 0
        
        # Learning rate scheduling
        if self.scheduler:
            self.scheduler.step(avg_total_loss)
        
        # Update history
        self.loss_history['epoch'].append(self.epoch)
        self.loss_history['total'].append(avg_total_loss)
        self.loss_history['data'].append(avg_data_loss)
        self.loss_history['physics'].append(avg_physics_loss)
        
        self.epoch += 1
        self.is_fitted = True
        
        # Metrics
        metrics = {
            'loss': avg_total_loss,
            'rmse': np.sqrt(avg_data_loss),
            'data_loss': avg_data_loss,
            'physics_loss': avg_physics_loss,
            'physics_weight': physics_weight,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        # Periodic logging
        if self.debug or self.epoch % 20 == 0:
            print(f"\nEpoch {self.epoch}:")
            print(f"  Total Loss: {avg_total_loss:.6f}")
            print(f"  Data Loss: {avg_data_loss:.6f}")
            print(f"  RMSE: {np.sqrt(avg_data_loss):.6f}")
            print(f"  Physics Loss: {avg_physics_loss:.6f}")
            print(f"  Physics Weight: {physics_weight:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        return metrics

    def validate(self, val_data: pd.DataFrame) -> Dict[str, float]:
        """Enhanced validation."""
        X_val, y_val = self._prepare_data(val_data, is_training=False)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_val).squeeze()
            
            # Basic losses
            mse = nn.functional.mse_loss(predictions, y_val)
            mae = nn.functional.l1_loss(predictions, y_val)
            
            # R² calculation
            y_mean = y_val.mean()
            ss_tot = torch.sum((y_val - y_mean) ** 2)
            ss_res = torch.sum((y_val - predictions) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0)
            
            # Convert to physical units for additional metrics
            pred_physical = self._to_physical_units(predictions)
            y_physical = self._to_physical_units(y_val)
            
            mse_physical = nn.functional.mse_loss(pred_physical, y_physical)
            mae_physical = nn.functional.l1_loss(pred_physical, y_physical)
        
        return {
            'loss': mse.item(),
            'rmse': np.sqrt(mse.item()),
            'mae': mae.item(),
            'r2': r2.item(),
            'rmse_physical': np.sqrt(mse_physical.item()),
            'mae_physical': mae_physical.item()
        }

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Simplified and reliable prediction."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.debug:
            print(f"Predict - Input shape: {data.shape}")
            print(f"Predict - Columns: {list(data.columns)}")
        
        # Find target column
        target_col = None
        possible_targets = ['chf_exp', 'CHF_exp', 'CHF', 'target', 'chf']
        
        for col in data.columns:
            col_lower = col.lower()
            if any(target in col_lower for target in possible_targets):
                target_col = col
                break
        
        # Get feature columns
        if target_col:
            feature_cols = [col for col in data.columns if col != target_col]
        else:
            feature_cols = list(data.columns)
        
        # Use stored feature names if available
        if self.feature_names is not None:
            # Check if we have all required features
            missing_features = set(self.feature_names) - set(feature_cols)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            feature_cols = self.feature_names
        
        try:
            # Extract and convert data
            X = data[feature_cols].astype(np.float32).values
            
            # Handle missing values
            if np.isnan(X).any():
                print("Warning: Found NaN values in prediction data")
                X = np.nan_to_num(X, nan=0.0)  # Simple imputation
            
            # Scale data
            X_scaled = self.input_scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                predictions_scaled = self.model(X_tensor).squeeze().cpu().numpy()
            
            # Convert back to physical units
            if predictions_scaled.ndim == 0:
                predictions_scaled = predictions_scaled.reshape(-1)
            
            predictions_scaled = predictions_scaled.reshape(-1, 1)
            predictions = self.target_scaler.inverse_transform(predictions_scaled).flatten()
            
            if self.debug:
                print(f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
            
            return predictions
            
        except Exception as e:
            print(f"Prediction error: {e}")
            print(f"Available columns: {list(data.columns)}")
            print(f"Required features: {self.feature_names}")
            raise

    def save(self, path: Path, metadata: Dict[str, Any] = None):
        """Save model with all components."""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        save_dict = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.epoch,
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'lambda_physics': self.lambda_physics,
            'is_fitted': self.is_fitted,
            'input_scaler': self.input_scaler,
            'target_scaler': self.target_scaler,
            'loss_history': self.loss_history,
            'feature_names': self.feature_names
        }
        
        if metadata:
            save_dict['metadata'] = metadata
        
        path = Path(path).with_suffix('.pth')
        torch.save(save_dict, path)
        
        print(f"✓ Saved FixedPINN model to {path}")
        print(f"  - Input size: {self.input_size}")
        print(f"  - Epoch: {self.epoch}")
        print(f"  - Features: {self.feature_names}")

    def load(self, path: Path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Restore configuration
        self.input_size = checkpoint['input_size']
        self.hidden_layers = checkpoint['hidden_layers']
        self.lambda_physics = checkpoint['lambda_physics']
        self.epoch = checkpoint['epoch']
        self.is_fitted = checkpoint['is_fitted']
        self.feature_names = checkpoint.get('feature_names')
        
        # Restore scalers
        self.input_scaler = checkpoint['input_scaler']
        self.target_scaler = checkpoint['target_scaler']
        
        # Restore history
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
        
        # Rebuild and load model
        self.model = self._build_model(self.input_size)
        self.model.load_state_dict(checkpoint['model_state'])
        
        # Rebuild optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        if checkpoint.get('optimizer_state'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Rebuild scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=15
        )
        if checkpoint.get('scheduler_state'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        print(f"✓ Loaded FixedPINN model from {path} (epoch {self.epoch})")
        return checkpoint.get('metadata', {})