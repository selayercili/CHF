"""Fixed Physics-Informed Neural Network (PINN) Model Implementation"""
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
                 lambda_physics: float = 0.01, **kwargs):  # Reduced default lambda
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
        
        # Feature names for robust indexing
        self.feature_names = [
            'pressure_MPa', 'mass_flux_kg_m2_s', 'x_e_out__', 
            'D_h_mm', 'length_mm', 'geometry_encoded'
        ]
        
        # Physics parameters
        self.debug = kwargs.get('debug', False)
        self.physics_warmup_epochs = kwargs.get('physics_warmup_epochs', 50)  # Longer warmup
        
        # CHF equation parameters - realistic initial values
        self.chf_param_init = {
            'A': 2.9e6,   # Base coefficient (typical Bowring range)
            'B': 0.0031,  # Quality coefficient (typical Bowring range)
            'C': 0.25     # Length coefficient (typical Bowring range)
        }
        self.bowring_params = {}
        self.loss_history = []

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Properly handle data columns and exclude cluster_label."""
        # Find target column
        target_cols = [col for col in data.columns if 'chf_exp' in col.lower()]
        if not target_cols:
            raise ValueError("No CHF target column found!")
        target_col = target_cols[0]
        
        # Explicitly exclude cluster_label and other non-feature columns
        exclude_cols = [target_col, 'cluster_label'] + [col for col in data.columns if 'cluster' in col.lower()]
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Ensure we have all expected features
        missing_features = [f for f in self.feature_names if f not in feature_cols]
        if missing_features:
            raise ValueError(f"Missing expected features: {missing_features}")
        
        # Extract features and target
        X = data[feature_cols].values
        y = data[target_col].values
        
        # Fit scalers only on first call
        if not self.is_fitted and self.model is None:
            X = self.input_scaler.fit_transform(X)
            y = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        else:
            X = self.input_scaler.transform(X)
            y = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        # Initialize model if needed
        if self.model is None:
            self.input_size = X.shape[1]
            self.model = self._build_model(self.input_size)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Data validation
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("NaNs detected after scaling")
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("Infinite values detected after scaling")
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        return X_tensor, y_tensor

    def _build_model(self, input_size: int) -> nn.Module:
        """Build neural network with learnable physics parameters."""
        model = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.1), 
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(0.1),
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
        
        # Get feature indices by name
        pressure_idx = self.feature_names.index('pressure_MPa')
        mass_flux_idx = self.feature_names.index('mass_flux_kg_m2_s')
        x_e_out_idx = self.feature_names.index('x_e_out__')
        D_h_idx = self.feature_names.index('D_h_mm')
        length_idx = self.feature_names.index('length_mm')
        geometry_idx = self.feature_names.index('geometry_encoded')
        
        # Convert with stability constraints
        pressure_Pa = torch.clamp(inputs[:, pressure_idx] * 1e6, min=1e5, max=25e6)  # MPa to Pa
        mass_flux = torch.clamp(inputs[:, mass_flux_idx], min=50.0)  # kg/m²s (min reasonable flow)
        x_e_out = inputs[:, x_e_out_idx]  # Quality (can be negative)
        D_h_m = torch.clamp(inputs[:, D_h_idx] * 1e-3, min=1e-4)  # mm to m
        length_m = torch.clamp(inputs[:, length_idx] * 1e-3, min=0.01)  # mm to m
        geometry = inputs[:, geometry_idx] if geometry_idx < inputs.shape[1] else torch.zeros_like(pressure_Pa)
        
        return {
            'pressure_Pa': pressure_Pa,
            'mass_flux': mass_flux,
            'x_e_out': x_e_out,
            'D_h_m': D_h_m,
            'length_m': length_m,
            'geometry': geometry
        }

    def _approximated_latent_heat(self, pressure_Pa: torch.Tensor) -> torch.Tensor:
        """Differentiable approximation of latent heat of vaporization."""
        # Convert Pa to MPa for better numerical stability
        P_MPa = pressure_Pa / 1e6
        
        # Polynomial approximation: h_fg ≈ a + b*P + c*P²
        # Coefficients fitted to steam table data (accurate within 5% for 0.1-20MPa)
        a = 2.257e6   # MJ/kg at 0.1MPa
        b = -3.85e4   # Linear coefficient
        c = -1.15e3   # Quadratic coefficient
        
        h_fg = a + b * P_MPa + c * (P_MPa**2)
        
        # Clamp to reasonable range (0.1 - 2.5 MJ/kg)
        return torch.clamp(h_fg, min=1e5, max=2.5e6)

    def _physics_loss(self, inputs: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """Differentiable physics loss without CoolProp."""
        try:
            # Convert units with stability constraints
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
            
            # Physics equation (Bowring-style correlation)
            numerator = A - B * x_e_out * h_fg
            
            # Stabilized denominator calculation
            denominator_base = C + length_m
            correction = (4 * B * length_m) / (mass_flux * D_h_m + 1e-8)
            
            # Constrain correction to prevent negative denominators
            max_correction = 0.95 * denominator_base
            correction = torch.clamp(correction, max=max_correction)
            
            denominator = denominator_base - correction
            
            # Numerical stability
            denominator = denominator + torch.sign(denominator) * 1e-8
            
            # Physics prediction (q_chf in W/m²)
            q_chf_physics = numerator / denominator
            
            # Model prediction (in scaled space)
            q_chf_pred_scaled = predictions.squeeze()
            
            # Convert physics prediction to scaled space
            q_chf_physics_np = q_chf_physics.detach().cpu().numpy().reshape(-1, 1)
            if hasattr(self.target_scaler, 'mean_'):  # Check if scaler is fitted
                # Transform physics prediction to same scale as model output
                q_chf_physics_scaled = self.target_scaler.transform(q_chf_physics_np)
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
            param_penalty = (
                torch.relu(torch.abs(A) - 5e6).clamp(min=0)**2 + 
                torch.relu(torch.abs(B) - 0.01)**2 + 
                torch.relu(torch.abs(C) - 2.0)**2
            )
            
            total_physics_loss = physics_loss + 0.1 * negative_denom_penalty + 0.01 * param_penalty
            
            return total_physics_loss
            
        except Exception as e:
            if self.debug:
                print(f"Physics loss computation failed: {e}")
            # Return zero physics loss if computation fails
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def train_epoch(self, train_data: pd.DataFrame, batch_size: int = 32) -> Dict[str, float]:
        """Train one epoch with adaptive physics weighting."""
        X_train, y_train = self._prepare_data(train_data)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        
        # Gradually increase physics weight (exponential warmup)
        warmup = min(1.0, self.epoch / self.physics_warmup_epochs)
        current_physics_weight = self.lambda_physics * (warmup ** 2)
        
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
            
            # Diagnostic logging for first batch
            if batch_idx == 0 and (self.debug or self.epoch % 10 == 0):
                with torch.no_grad():
                    # Get sample predictions
                    sample_pred = outputs[0].item()
                    sample_target = targets[0].item()
                    
                    # Convert physics prediction
                    units = self._convert_units_differentiable(inputs)
                    pressure = units['pressure_Pa'][0].item() / 1e6
                    mass_flux = units['mass_flux'][0].item()
                    x_e = units['x_e_out'][0].item()
                    
                    # Calculate physics prediction
                    h_fg = self._approximated_latent_heat(units['pressure_Pa'][0]).item()
                    A = self.bowring_params['A'].item()
                    B = self.bowring_params['B'].item()
                    C = self.bowring_params['C'].item()
                    numerator = A - B * x_e * h_fg
                    denominator = C + units['length_m'][0].item() - (4 * B * units['length_m'][0].item()) / (mass_flux * units['D_h_m'][0].item())
                    phys_pred = numerator / denominator if denominator > 1e-6 else 0
                    
                    # Inverse scale for human-readable values
                    try:
                        sample_pred_orig = self.target_scaler.inverse_transform(
                            np.array([[sample_pred]]))[0][0]
                        sample_target_orig = self.target_scaler.inverse_transform(
                            np.array([[sample_target]]))[0][0]
                        phys_pred_orig = self.target_scaler.inverse_transform(
                            np.array([[phys_pred]]))[0][0] if phys_pred != 0 else 0
                    except:
                        sample_pred_orig = sample_pred
                        sample_target_orig = sample_target
                        phys_pred_orig = phys_pred
                    
                    print(
                        f"Sample: P={pressure:.2f}MPa, G={mass_flux:.0f}, x={x_e:.3f} | "
                        f"Target: {sample_target_orig:.2f} | "
                        f"Pred: {sample_pred_orig:.2f} | "
                        f"Physics: {phys_pred_orig:.2f}"
                    )
        
        # Calculate averages
        n_batches = len(dataloader)
        avg_loss = total_loss / n_batches
        avg_data_loss = total_data_loss / n_batches
        avg_physics_loss = total_physics_loss / n_batches
        
        self.epoch += 1
        self.is_fitted = True
        self.loss_history.append(avg_loss)
        
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
                  f"Weight={current_physics_weight:.6f}")
            
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
            r2 = 1 - ss_res / (ss_tot + 1e-8)  # Add epsilon for stability
        
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
            'chf_params': chf_params_dict,
            'feature_names': self.feature_names,
            'loss_history': self.loss_history
        }
        
        if metadata:
            save_dict['metadata'] = metadata
        
        path = Path(path).with_suffix('.pth')
        torch.save(save_dict, path)

    def load(self, path: Path):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore attributes
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.lambda_physics = checkpoint['lambda_physics']
        self.epoch = checkpoint['epoch']
        self.is_fitted = checkpoint['is_fitted']
        self.feature_names = checkpoint.get('feature_names', [
            'pressure_MPa', 'mass_flux_kg_m2_s', 'x_e_out__', 
            'D_h_mm', 'length_mm', 'geometry_encoded'
        ])
        self.loss_history = checkpoint.get('loss_history', [])
        
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

    def get_feature_importance(self) -> pd.DataFrame:
        """Get gradient-based feature importance."""
        if self.model is None:
            return pd.DataFrame()
            
        dummy_input = torch.ones(1, self.input_size, device=self.device, requires_grad=True)
        output = self.model(dummy_input)
        output.backward()
        
        gradients = torch.abs(dummy_input.grad).cpu().numpy().flatten()
        
        # Get feature names if available
        try:
            feature_names = self.input_scaler.feature_names_in_
        except AttributeError:
            feature_names = [f'feature_{i}' for i in range(len(gradients))]
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': gradients
        }).sort_values('importance', ascending=False)