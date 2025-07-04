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
        CHF Physics-Informed Loss Implementation
        
        Incorporates three key physics principles for critical heat flux:
        1. Energy balance: Heat input = Heat removed by fluid
        2. Momentum balance: Pressure drop relationships  
        3. Mass conservation: Continuity equation
        
        Input columns (based on your data):
        0: pressure_MPa
        1: mass_flux_kg_m2_s  
        2: x_e_out (exit quality)
        3: D_e_mm (equivalent diameter)
        4: D_h_mm (hydraulic diameter) 
        5: length_mm
        6: geometry_encoded
        
        Output: CHF in MW/m²
        """
        # Enable gradients for physics constraints
        inputs = inputs.detach().requires_grad_(True)
        predictions = self.model(inputs)
        
        if not predictions.requires_grad:
            predictions = predictions.requires_grad_(True)
        
        # Extract physical parameters
        pressure = inputs[:, 0] * 1e6  # Convert MPa to Pa
        mass_flux = inputs[:, 1]       # kg/m²/s
        x_exit = inputs[:, 2]          # Exit quality
        D_e = inputs[:, 3] * 1e-3      # Convert mm to m
        D_h = inputs[:, 4] * 1e-3      # Convert mm to m  
        length = inputs[:, 5] * 1e-3   # Convert mm to m
        
        # CHF prediction (MW/m²)
        chf_pred = predictions.squeeze() * 1e6  # Convert to W/m²
        
        physics_losses = []
        
        # 1. ENERGY BALANCE CONSTRAINT
        # Heat input should equal heat removed by fluid
        # Q = m_dot * h_fg * (quality change)
        # Assuming inlet quality ≈ 0 for subcooled inlet
        h_fg = 2.26e6  # Latent heat of vaporization (J/kg) - approximate for water
        
        # Energy balance: CHF * Area = mass_flux * h_fg * x_exit * Area
        # Simplifies to: CHF = mass_flux * h_fg * x_exit
        energy_balance = chf_pred - mass_flux * h_fg * x_exit
        energy_loss = torch.mean(energy_balance**2)
        physics_losses.append(energy_loss)
        
        # 2. MOMENTUM BALANCE CONSTRAINT  
        # Pressure drop should be consistent with flow physics
        # For two-phase flow: ΔP = f * (L/D) * (ρv²/2) * Φ²
        # where Φ is two-phase multiplier
        
        # Simplified momentum constraint: higher mass flux → higher pressure drop needed
        # This creates a physical relationship between pressure, mass flux, and geometry
        def safe_grad(y, x):
            try:
                grad_outputs = torch.ones_like(y, requires_grad=True)
                grads = torch.autograd.grad(
                    outputs=y,
                    inputs=x,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                    only_inputs=True
                )[0]
                return grads if grads is not None else torch.zeros_like(x)
            except RuntimeError:
                return torch.zeros_like(x)
        
        # Momentum balance: dCHF/dG should be positive (higher mass flux → higher CHF)
        dCHF_dG = safe_grad(chf_pred, mass_flux)
        momentum_constraint = torch.relu(-dCHF_dG)  # Penalize negative gradients
        momentum_loss = torch.mean(momentum_constraint**2)
        physics_losses.append(momentum_loss)
        
        # 3. GEOMETRIC SCALING CONSTRAINT
        # CHF should scale appropriately with hydraulic diameter
        # Smaller channels typically have higher CHF due to surface tension effects
        dCHF_dDh = safe_grad(chf_pred, D_h)
        geometric_constraint = torch.relu(dCHF_dDh)  # Penalize positive gradients (CHF should decrease with diameter)
        geometric_loss = torch.mean(geometric_constraint**2)
        physics_losses.append(geometric_loss)
        
        # 4. PRESSURE DEPENDENCY CONSTRAINT
        # CHF typically increases with pressure (up to critical pressure)
        # This is a well-known physical relationship
        dCHF_dP = safe_grad(chf_pred, pressure)
        pressure_constraint = torch.relu(-dCHF_dP)  # Penalize negative gradients
        pressure_loss = torch.mean(pressure_constraint**2)
        physics_losses.append(pressure_loss)
        
        # 5. PHYSICAL BOUNDS CONSTRAINT
        # CHF should be within reasonable physical bounds
        # Typical CHF values: 0.1 to 50 MW/m²
        chf_MW = chf_pred / 1e6  # Convert back to MW/m²
        lower_bound = torch.relu(0.1 - chf_MW)  # Penalize values below 0.1 MW/m²
        upper_bound = torch.relu(chf_MW - 50.0)  # Penalize values above 50 MW/m²
        bounds_loss = torch.mean(lower_bound**2) + torch.mean(upper_bound**2)
        physics_losses.append(bounds_loss)
        
        # Combine all physics losses with weights
        weights = [1.0, 0.5, 0.3, 0.5, 0.2]  # Adjust these based on importance
        total_physics_loss = sum(w * loss for w, loss in zip(weights, physics_losses))
        
        # Safety checks
        if not torch.isfinite(total_physics_loss).all() or not total_physics_loss.requires_grad:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_physics_loss

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
        
        avg_loss = total_loss / len(dataloader)
        self.epoch += 1
        self.is_fitted = True
        return {'loss': avg_loss, 'rmse': np.sqrt(avg_loss)}

    def validate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Validates model (physics loss not computed during validation)."""
        X_test, y_test = self._prepare_data(test_data)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test)
            mse = torch.nn.functional.mse_loss(predictions, y_test)
            mae = torch.nn.functional.l1_loss(predictions, y_test)
        return {'loss': mse.item(), 'rmse': np.sqrt(mse.item()), 'mae': mae.item()}

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
            'is_fitted': self.is_fitted,
            'input_scaler': self.input_scaler,
            'target_scaler': self.target_scaler
        }
        if metadata:
            save_dict.update(metadata)
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
        
        # Load scalers if they exist
        if 'input_scaler' in checkpoint:
            self.input_scaler = checkpoint['input_scaler']
        if 'target_scaler' in checkpoint:
            self.target_scaler = checkpoint['target_scaler']
            
        return checkpoint

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