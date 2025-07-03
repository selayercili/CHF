"""
Physics-Informed Neural Network (PINN) Model Implementation

Combines data-driven loss with physics-based constraints for heat flux prediction.
"""
import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple  # <-- Added Tuple import
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler

class Pinn:
    """PINN model wrapper with physics constraints for heat flux prediction."""
    
    def __init__(self, hidden_size: int = 64, learning_rate: float = 0.001, 
                 lambda_physics: float = 0.1, decay_rate: float = 0.995 ,**kwargs):
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
        self.decay_rate = decay_rate
        self.scheduler = None

    def _build_model(self, input_size: int) -> nn.Module:
        """3-layer MLP with ReLU activations."""
        return nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        ).to(self.device)

    def _physics_loss(self, inputs: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """
        Comprehensive physics loss for critical heat flux prediction.
        Combines Zuber's correlation with energy balance constraints.
        """
        # Ensure gradient tracking
        inputs = inputs.clone().requires_grad_(True)
        predictions = self.model(inputs)
        
        # ========================================================================
        # 1. Extract physical parameters from input features
        #    (MODIFY INDICES BASED ON YOUR ACTUAL DATA COLUMNS)
        # ========================================================================
        # Pressure [MPa] -> Pa conversion
        pressure = inputs[:, 0] * 1e6
        mass_flux = inputs[:, 1]        # [kg/m²s]
        quality = inputs[:, 2]           # [-]
        diameter = inputs[:, 3]          # [m] (hydraulic diameter)
        length = inputs[:, 4]            # [m] (heater length)
        T_sat = inputs[:, 5]             # [K] (saturation temperature)
        latent_heat = inputs[:, 6]       # [J/kg]
        liquid_density = inputs[:, 7]    # [kg/m³]
        vapor_density = inputs[:, 8]     # [kg/m³]
        surface_tension = inputs[:, 9]   # [N/m]
        cp_liquid = inputs[:, 10]        # [J/kg·K]
        viscosity = inputs[:, 11]        # [Pa·s]
        thermal_cond = inputs[:, 12]     # [W/m·K]
        
        # ========================================================================
        # 2. Calculate Critical Heat Flux using industry-standard correlations
        # ========================================================================
        g = 9.81  # Gravitational acceleration [m/s²]
        
        # A. Zuber's correlation (fundamental CHF for pool boiling)
        chf_zuber = (0.131 * latent_heat * vapor_density**0.5 * 
                    (surface_tension * g * (liquid_density - vapor_density))**0.25)
        
        # B. Groeneveld correlation (flow boiling - more accurate for pipes)
        #    CHF = 0.001 * (P^0.4) * (G^0.6) * (1 - x)^2
        chf_groeneveld = 1e-3 * (pressure/1e6)**0.4 * mass_flux**0.6 * (1 - quality)**2
        
        # C. Combine correlations with weighting factors
        pool_weight = torch.sigmoid(-mass_flux/100)  # Weight for pool boiling
        flow_weight = 1 - pool_weight                 # Weight for flow boiling
        
        # Final CHF prediction from physics
        chf_physics = pool_weight * chf_zuber + flow_weight * chf_groeneveld
        
        # ========================================================================
        # 3. Energy balance residual (convection + conduction = CHF)
        # ========================================================================
        # Compute temperature gradients (for conduction term)
        spatial_coords = inputs[:, 13:16]  # Spatial coordinates [x,y,z]
        T_pred = predictions
        
        # Compute first derivatives
        dT_dx = torch.autograd.grad(
            T_pred.sum(), 
            spatial_coords, 
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Conduction term (Fourier's Law)
        q_cond = -thermal_cond.unsqueeze(1) * dT_dx
        
        # Convection term (Newton's Law)
        # Simplified convection coefficient (Dittus-Boelter correlation)
        reynolds = mass_flux * diameter / viscosity
        prandtl = cp_liquid * viscosity / thermal_cond
        h_conv = 0.023 * thermal_cond / diameter * reynolds**0.8 * prandtl**0.4
        q_conv = h_conv * (T_pred.squeeze() - T_sat)
        
        # Energy balance residual
        energy_residual = (q_cond.norm(dim=1) + q_conv - chf_physics)   
        
        # ========================================================================
        # 4. Critical heat flux residual (main physics constraint)
        # ========================================================================
        chf_residual = predictions.squeeze() - chf_physics
        
        # ========================================================================
        # 5. Combine residuals with physics-based weighting
        # ========================================================================
        # Weighting factors (tunable)
        w_chf = 0.7   # Weight for CHF constraint
        w_energy = 0.3 # Weight for energy balance
        
        # Final loss
        loss_chf = torch.mean(chf_residual**2)
        loss_energy = torch.mean(energy_residual**2)
        total_loss = w_chf * loss_chf + w_energy * loss_energy
        
        return total_loss

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts DataFrame to tensors and initializes model if needed."""
        target_col = [col for col in data.columns if 'chf_exp' in col][0]
        X = data.drop(target_col, axis=1).values
        y = data[target_col].values
        
        # Add scaling:
        if not self.is_fitted:
            X = self.input_scaler.fit_transform(X)
            y = self.target_scaler.fit_transform(y.reshape(-1, 1))
        else:
            X = self.input_scaler.transform(X)
            y = self.target_scaler.transform(y.reshape(-1, 1))
        
        if self.model is None:
            self.input_size = X.shape[1]
            self.model = self._build_model(self.input_size)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        return X_tensor, y_tensor

    def train_epoch(self, train_data: pd.DataFrame, batch_size: int = 32, 
                   optimizer: Any = None) -> Dict[str, float]:
        """Trains for one epoch with composite loss (data + physics)."""
        X_train, y_train = self._prepare_data(train_data)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        total_loss = 0.0
        
        for inputs, targets in dataloader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Data loss (MSE)
            data_loss = torch.nn.functional.mse_loss(outputs, targets)
            
            # Physics loss
            physics_loss = self._physics_loss(inputs, outputs)
            
            # Combined loss
            loss = data_loss + self.lambda_physics * physics_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        if self.optimizer and self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, 
                gamma=self.decay_rate
        )
    
        if self.scheduler:
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Learning rate: {current_lr:.6f}")
        
        avg_loss = total_loss / len(dataloader)
        self.epoch += 1
        self.is_fitted = True
        return {'loss': avg_loss, 'rmse': np.sqrt(avg_loss)}

    def validate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Validate the model on test data with physics residual metric.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be trained before validation")
        
        # Prepare data
        X_test, y_test = self._prepare_data(test_data)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test)
        
        # Calculate standard regression metrics
        mse = torch.nn.functional.mse_loss(predictions, y_test)
        rmse = torch.sqrt(mse)
        mae = torch.nn.functional.l1_loss(predictions, y_test)
        
        # Initialize metrics dictionary
        metrics = {
            'loss': mse.item(),
            'rmse': rmse.item(),
            'mae': mae.item()
        }
        
        # Add physics residual metric
        try:
            physics_residual = self._physics_loss(X_test, predictions)
            metrics['physics_residual'] = physics_residual.item()
        except Exception as e:
            # Gracefully handle physics loss calculation failures
            metrics['physics_residual'] = float('nan')
            self.logger.warning(f"Physics residual calculation failed: {str(e)}")
        
        return metrics

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Makes predictions (same as other models)."""
        target_col = [col for col in data.columns if 'chf_exp' in col][0] if any('chf_exp' in col for col in data.columns) else None
        if target_col and target_col in data.columns:
            X = data.drop(target_col, axis=1).values
        else:
            X = data.values
            
        X = self.input_scaler.transform(X)  # Scale input
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
            
            
        # Inverse scale predictions:
        return self.target_scaler.inverse_transform(predictions_scaled).flatten()

    def save(self, path: Path, metadata: Dict[str, Any] = None):
        """Saves model state."""
        save_dict = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'lambda_physics': self.lambda_physics,
            'is_fitted': self.is_fitted
        }
        if metadata:
            save_dict['metadata'] = metadata
        torch.save(save_dict, path)

    def load(self, path: Path):
        """Loads model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.lambda_physics = checkpoint['lambda_physics']
        self.model = self._build_model(self.input_size)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epoch = checkpoint['epoch']
        self.is_fitted = checkpoint['is_fitted']
        return checkpoint.get('metadata', {})

    def get_feature_importance(self) -> pd.DataFrame:
        """Gradient-based feature importance (similar to NN)."""
        dummy_input = torch.ones(1, self.input_size, device=self.device, requires_grad=True)
        output = self.model(dummy_input)
        output.backward()
        gradients = torch.abs(dummy_input.grad).cpu().numpy().flatten()
        return pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(gradients))],
            'importance': gradients
        }).sort_values('importance', ascending=False)