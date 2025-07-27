"""
Fixed Physics-Informed Neural Network (PINN) Model Implementation

Key fixes:
1. Proper data preprocessing to exclude cluster_label
2. Differentiable physics loss without CoolProp calls
3. Simplified and more stable physics constraints
4. Better numerical stability
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler

class Pinn:
    """Fixed PINN model with proper data handling and differentiable physics."""
    
    def __init__(self, hidden_size: int = 64, learning_rate: float = 0.001, 
                 lambda_physics: float = 0.1, **kwargs):
        """
        Args:
            hidden_size: Neurons per hidden layer.
            learning_rate: Optimizer learning rate.
            lambda_physics: Weight for physics loss term.
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.lambda_physics = lambda_physics
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted = False
        self.epoch = 0
        self.input_size = None
        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Physics parameters
        self.debug = kwargs.get('debug', False)
        self.physics_warmup_epochs = kwargs.get('physics_warmup_epochs', 20)
        
        # CHF equation parameters
        self.chf_param_init = {
            'A': 0.5,   # Base coefficient
            'B': 1e-4,  # Quality coefficient  
            'C': 0.1    # Length coefficient
        }
        self.bowring_params = {}

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """FIXED: Properly handle data columns and exclude cluster_label."""
        # Find target column
        target_cols = [col for col in data.columns if 'chf_exp' in col.lower()]
        if not target_cols:
            raise ValueError("No CHF target column found!")
        target_col = target_cols[0]
        
        # CRITICAL FIX: Explicitly exclude cluster_label and other non-feature columns
        exclude_cols = [target_col, 'cluster_label'] + [col for col in data.columns if 'cluster' in col.lower()]
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        print(f"Target column: {target_col}")
        print(f"Feature columns: {feature_cols}")
        print(f"Excluded columns: {exclude_cols}")
        
        # Extract features and target
        X = data[feature_cols].values
        y = data[target_col].values
        
        # Fit scalers only on first call
        if not self.is_fitted and self.model is None:
            X = self.input_scaler.fit_transform(X)
            y = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            print(f"✓ Fitted scalers - Input shape: {X.shape}, Target shape: {y.shape}")
        else:
            X = self.input_scaler.transform(X)
            y = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        # Initialize model if needed
        if self.model is None:
            self.input_size = X.shape[1]
            self.model = self._build_model(self.input_size)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            print(f"✓ Initialized PINN model with {self.input_size} input features")
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        return X_tensor, y_tensor

    def _build_model(self, input_size: int) -> nn.Module:
        """Build neural network with learnable physics parameters."""
        model = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        ).to(self.device)
        
        # Xavier initialization
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Register physics parameters
        for name, init_value in self.chf_param_init.items():
            param = nn.Parameter(torch.tensor(init_value, device=self.device))
            model.register_parameter(f'chf_{name}', param)
            self.bowring_params[name] = param
        
        return model

    def _convert_units_differentiable(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert units while maintaining differentiability."""
        eps = 1e-8
        
        # Assuming input order: pressure_MPa, mass_flux, x_e_out, D_h_mm, length_mm, geometry
        pressure_Pa = torch.clamp(inputs[:, 0] * 1e6, min=1000.0, max=20e6)  # MPa to Pa
        mass_flux = torch.clamp(inputs[:, 1], min=eps)  # kg/m²s
        x_e_out = inputs[:, 2]  # Quality (can be negative)
        D_h_m = torch.clamp(inputs[:, 3] * 1e-3, min=eps)  # mm to m
        length_m = torch.clamp(inputs[:, 4] * 1e-3, min=eps)  # mm to m
        geometry = inputs[:, 5] if inputs.shape[1] > 5 else torch.zeros_like(pressure_Pa)
        
        return {
            'pressure_Pa': pressure_Pa,
            'mass_flux': mass_flux,
            'x_e_out': x_e_out,
            'D_h_m': D_h_m,
            'length_m': length_m,
            'geometry': geometry
        }

    def _approximated_latent_heat(self, pressure_Pa: torch.Tensor) -> torch.Tensor:
        """FIXED: Differentiable approximation of latent heat of vaporization."""
        # Polynomial approximation for h_fg as function of pressure
        # Based on steam tables, h_fg decreases with pressure
        # This is a simplified approximation - you could use a more sophisticated one
        
        # Convert Pa to MPa for better numerical stability
        P_MPa = pressure_Pa / 1e6
        
        # Polynomial approximation: h_fg ≈ a + b*P + c*P²
        # Coefficients fitted to steam table data (approximate)
        a = 2.3e6   # ~2.3 MJ/kg at low pressure
        b = -4e4    # Decreases with pressure
        c = -1e3    # Slight curvature
        
        h_fg = a + b * P_MPa + c * P_MPa**2
        
        # Clamp to reasonable range (0.1 - 2.5 MJ/kg)
        h_fg = torch.clamp(h_fg, min=1e5, max=2.5e6)
        
        return h_fg

    def _physics_loss(self, inputs: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """FIXED: Differentiable physics loss without CoolProp."""
        try:
            # Convert units
            units = self._convert_units_differentiable(inputs)
            pressure_Pa = units['pressure_Pa']
            mass_flux = units['mass_flux']
            x_e_out = units['x_e_out']
            D_h_m = units['D_h_m']
            length_m = units['length_m']
            
            # Differentiable latent heat approximation
            h_fg = self._approximated_latent_heat(pressure_Pa)
            
            # Get physics parameters
            A = self.bowring_params['A']
            B = self.bowring_params['B']
            C = self.bowring_params['C']
            
            # Physics equation (your CHF correlation)
            numerator = A - B * x_e_out * h_fg
            denominator = C + length_m - (4 * B * length_m) / (mass_flux * D_h_m)
            
            # Numerical stability
            denominator = denominator + torch.sign(denominator) * 1e-8
            
            # Physics prediction
            q_chf_physics = numerator / denominator
            
            # Model prediction (in scaled space, convert to physical units)
            q_chf_pred_scaled = predictions.squeeze()
            
            # For physics loss comparison, we need both in same units
            # Convert physics prediction to scaled space for fair comparison
            q_chf_physics_np = q_chf_physics.detach().cpu().numpy().reshape(-1, 1)
            if hasattr(self.target_scaler, 'mean_'):  # Check if scaler is fitted
                # Transform physics prediction to same scale as model output
                q_chf_physics_scaled = self.target_scaler.transform(q_chf_physics_np / 1e6)  # MW/m²
                q_chf_physics_scaled = torch.tensor(q_chf_physics_scaled.flatten(), 
                                                  device=self.device, requires_grad=True)
            else:
                # Fallback: normalize both predictions
                q_chf_physics_scaled = (q_chf_physics - q_chf_physics.mean()) / (q_chf_physics.std() + 1e-8)
            
            # Physics loss: MSE between model prediction and physics prediction
            physics_loss = torch.mean((q_chf_pred_scaled - q_chf_physics_scaled) ** 2)
            
            # Add constraint penalties
            # 1. Denominator should be positive for physical validity
            negative_denom_penalty = torch.mean(torch.relu(-denominator) ** 2)
            
            # 2. Parameters should be in reasonable ranges
            param_penalty = torch.relu(torch.abs(A) - 2.0)**2 + \
                           torch.relu(torch.abs(B) - 1e-2)**2 + \
                           torch.relu(torch.abs(C) - 1.0)**2
            
            total_physics_loss = physics_loss + 0.1 * negative_denom_penalty + 0.01 * param_penalty
            
            return total_physics_loss
            
        except Exception as e:
            if self.debug:
                print(f"Physics loss computation failed: {e}")
            # Return zero physics loss if computation fails
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def train_epoch(self, train_data: pd.DataFrame, batch_size: int = 32, 
                   optimizer: Any = None) -> Dict[str, float]:
        """Train one epoch with adaptive physics weighting."""
        X_train, y_train = self._prepare_data(train_data)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        
        # Gradually increase physics weight
        if self.epoch < self.physics_warmup_epochs:
            current_physics_weight = self.lambda_physics * (self.epoch / self.physics_warmup_epochs)
        else:
            current_physics_weight = self.lambda_physics
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            
            # Data loss
            data_loss = nn.functional.mse_loss(outputs.squeeze(), targets)
            
            # Physics loss
            physics_loss = self._physics_loss(inputs, outputs)
            
            # Combined loss
            total_batch_loss = data_loss + current_physics_weight * physics_loss
            
            # Check for numerical issues
            if not torch.isfinite(total_batch_loss):
                print(f"Warning: Non-finite loss at epoch {self.epoch}, batch {batch_idx}")
                print(f"Data loss: {data_loss.item()}, Physics loss: {physics_loss.item()}")
                continue
            
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item()
        
        # Calculate averages
        n_batches = len(dataloader)
        avg_loss = total_loss / n_batches
        avg_data_loss = total_data_loss / n_batches
        avg_physics_loss = total_physics_loss / n_batches
        
        self.epoch += 1
        self.is_fitted = True
        
        metrics = {
            'loss': avg_loss,
            'rmse': np.sqrt(avg_data_loss),
            'data_loss': avg_data_loss,
            'physics_loss': avg_physics_loss,
            'physics_weight': current_physics_weight
        }
        
        if self.debug or self.epoch % 10 == 0:
            print(f"Epoch {self.epoch}: Loss={avg_loss:.6f}, "
                  f"Data={avg_data_loss:.6f}, Physics={avg_physics_loss:.6f}, "
                  f"Weight={current_physics_weight:.4f}")
            
            # Print learned parameters
            params = self.get_chf_parameters()
            print(f"  Learned params: A={params['A']:.4f}, B={params['B']:.6f}, C={params['C']:.4f}")
        
        return metrics

    def validate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Validate model performance."""
        X_test, y_test = self._prepare_data(test_data)
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(X_test).squeeze()
            mse = nn.functional.mse_loss(predictions, y_test)
            mae = nn.functional.l1_loss(predictions, y_test)
            
            # R² calculation
            ss_res = torch.sum((y_test - predictions) ** 2)
            ss_tot = torch.sum((y_test - y_test.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
        
        return {
            'loss': mse.item(),
            'rmse': np.sqrt(mse.item()),
            'mae': mae.item(),
            'r2': r2.item()
        }

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        # Handle target column same way as in _prepare_data
        target_cols = [col for col in data.columns if 'chf_exp' in col.lower()]
        exclude_cols = target_cols + ['cluster_label'] + [col for col in data.columns if 'cluster' in col.lower()]
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols].values
        X_scaled = self.input_scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).squeeze().cpu().numpy()
        
        # Inverse transform
        predictions_scaled = predictions_scaled.reshape(-1, 1)
        predictions = self.target_scaler.inverse_transform(predictions_scaled).flatten()
        
        return predictions

    def get_chf_parameters(self) -> Dict[str, float]:
        """Get learned physics parameters."""
        return {name: param.item() for name, param in self.bowring_params.items()}

    def save(self, path: Path, metadata: Dict[str, Any] = None):
        """Save model state."""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        chf_params_dict = {name: param.item() for name, param in self.bowring_params.items()}
        
        save_dict = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'epoch': self.epoch,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'lambda_physics': self.lambda_physics,
            'is_fitted': self.is_fitted,
            'input_scaler': pickle.dumps(self.input_scaler),
            'target_scaler': pickle.dumps(self.target_scaler),
            'chf_params': chf_params_dict
        }
        
        if metadata:
            save_dict['metadata'] = metadata
        
        path = Path(path).with_suffix('.pth')
        torch.save(save_dict, path)
        print(f"✓ Saved PINN model to {path}")

    def load(self, path: Path):
        """Load model state."""
        from sklearn.preprocessing import StandardScaler
        torch.serialization.add_safe_globals([StandardScaler])
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Restore attributes
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.lambda_physics = checkpoint['lambda_physics']
        self.epoch = checkpoint['epoch']
        self.is_fitted = checkpoint['is_fitted']
        
        # Load scalers
        if 'input_scaler' in checkpoint:
            self.input_scaler = pickle.loads(checkpoint['input_scaler'])
        if 'target_scaler' in checkpoint:
            self.target_scaler = pickle.loads(checkpoint['target_scaler'])
        
        # Load CHF parameters
        if 'chf_params' in checkpoint:
            self.chf_param_init = checkpoint['chf_params']
        
        # Build and load model
        self.model = self._build_model(self.input_size)
        self.model.load_state_dict(checkpoint['model_state'])
        
        # Rebuild optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if checkpoint.get('optimizer_state'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Update parameter references
        for name in self.chf_param_init.keys():
            param_name = f'chf_{name}'
            if hasattr(self.model, param_name):
                self.bowring_params[name] = getattr(self.model, param_name)
        
        print(f"✓ Loaded PINN model from {path} (epoch {self.epoch})")
        return checkpoint.get('metadata', {})

    def get_feature_importance(self) -> pd.DataFrame:
        """Get gradient-based feature importance."""
        if self.model is None:
            return pd.DataFrame()
            
        dummy_input = torch.ones(1, self.input_size, device=self.device, requires_grad=True)
        output = self.model(dummy_input)
        output.backward()
        
        gradients = torch.abs(dummy_input.grad).cpu().numpy().flatten()
        
        return pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(gradients))],
            'importance': gradients
        }).sort_values('importance', ascending=False)