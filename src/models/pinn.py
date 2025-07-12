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
                 lambda_physics: float = 0.01, **kwargs):  # Reduced physics weight
        """
        Args:
            hidden_size: Neurons per hidden layer.
            learning_rate: Optimizer learning rate.
            lambda_physics: Weight for physics loss term (reduced default).
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
        
        # Add debugging flags
        self.debug = kwargs.get('debug', False)
        self.physics_warmup_epochs = kwargs.get('physics_warmup_epochs', 10)  # Gradually introduce physics

    def _build_model(self, input_size: int) -> nn.Module:
        """Enhanced 4-layer MLP with better initialization and dropout."""
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
        
        # Better weight initialization
        for layer in model:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        
        return model

    def _physics_loss(self, inputs: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """
        Improved CHF Physics-Informed Loss with better balance and debugging
        """
        # Enable gradients for physics constraints
        inputs = inputs.detach().requires_grad_(True)
        predictions = self.model(inputs)
        
        if not predictions.requires_grad:
            predictions = predictions.requires_grad_(True)
        
        # Extract physical parameters (assuming your column order)
        # Adjust indices based on your actual column order after dropping target
        pressure = inputs[:, 0]        # pressure_MPa (already scaled)
        mass_flux = inputs[:, 1]       # mass_flux_kg_m2_s (already scaled)
        x_exit = inputs[:, 2]          # x_e_out (already scaled)
        D_e = inputs[:, 3]             # D_e_mm (already scaled)
        D_h = inputs[:, 4]             # D_h_mm (already scaled)
        length = inputs[:, 5]          # length_mm (already scaled)
        
        # CHF prediction (already scaled)
        chf_pred = predictions.squeeze()
        
        physics_losses = []
        
        # Safe gradient computation
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
        
        # 1. MONOTONICITY CONSTRAINTS (most important and stable)
        # CHF should increase with mass flux
        dCHF_dG = safe_grad(chf_pred, mass_flux)
        monotonic_mass_flux = torch.relu(-dCHF_dG)  # Penalize negative gradients
        physics_losses.append(torch.mean(monotonic_mass_flux**2))
        
        # CHF should increase with pressure (generally)
        dCHF_dP = safe_grad(chf_pred, pressure)
        monotonic_pressure = torch.relu(-dCHF_dP)  # Penalize negative gradients
        physics_losses.append(torch.mean(monotonic_pressure**2))
        
        # 2. SIMPLIFIED PHYSICAL BOUNDS (more lenient)
        # CHF should be positive and reasonable
        negative_chf = torch.relu(-chf_pred)  # Penalize negative predictions
        physics_losses.append(torch.mean(negative_chf**2))
        
        # 3. CONSISTENCY CONSTRAINTS (lighter weight)
        # Higher quality exit should relate to higher CHF capability
        dCHF_dx = safe_grad(chf_pred, x_exit)
        quality_consistency = torch.relu(-dCHF_dx)  # Should be positive relationship
        physics_losses.append(torch.mean(quality_consistency**2))
        
        # Combine physics losses with adaptive weights
        # Start with lighter physics constraints
        weights = [0.3, 0.2, 0.5, 0.1]  # Reduced weights
        total_physics_loss = sum(w * loss for w, loss in zip(weights, physics_losses))
        
        # Debug logging
        if self.debug and torch.rand(1).item() < 0.01:  # Log 1% of batches
            print(f"Physics losses: {[f'{loss.item():.6f}' for loss in physics_losses]}")
            print(f"Total physics loss: {total_physics_loss.item():.6f}")
        
        # Safety checks
        if not torch.isfinite(total_physics_loss).all() or not total_physics_loss.requires_grad:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_physics_loss

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts DataFrame to tensors and initializes model if needed."""
        target_col = [col for col in data.columns if 'chf_exp' in col][0]
        X = data.drop(target_col, axis=1).values
        y = data[target_col].values
        
        # Add scaling with better handling
        if not self.is_fitted:
            X = self.input_scaler.fit_transform(X)
            y = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        else:
            X = self.input_scaler.transform(X)
            y = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        if self.model is None:
            self.input_size = X.shape[1]
            self.model = self._build_model(self.input_size)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        return X_tensor, y_tensor

    def train_epoch(self, train_data: pd.DataFrame, batch_size: int = 32, 
                   optimizer: Any = None) -> Dict[str, float]:
        """Trains for one epoch with adaptive physics loss weighting."""
        X_train, y_train = self._prepare_data(train_data)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        
        # Adaptive physics weight (gradual introduction)
        current_physics_weight = self.lambda_physics
        if self.epoch < self.physics_warmup_epochs:
            # Gradually increase physics weight
            current_physics_weight = self.lambda_physics * (self.epoch / self.physics_warmup_epochs)
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Data loss (MSE)
            data_loss = torch.nn.functional.mse_loss(outputs.squeeze(), targets)
            
            # Physics loss (with adaptive weight)
            physics_loss = self._physics_loss(inputs, outputs)
            
            # Combined loss
            loss = data_loss + current_physics_weight * physics_loss
            
            # Check for NaN/Inf
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss at epoch {self.epoch}, batch {batch_idx}")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_data_loss = total_data_loss / len(dataloader)
        avg_physics_loss = total_physics_loss / len(dataloader)
        
        self.epoch += 1
        self.is_fitted = True
        
        # Enhanced metrics
        metrics = {
            'loss': avg_loss,
            'rmse': np.sqrt(avg_data_loss),
            'data_loss': avg_data_loss,
            'physics_loss': avg_physics_loss,
            'physics_weight': current_physics_weight
        }
        
        # Debug output
        if self.debug:
            print(f"Epoch {self.epoch}: Data Loss: {avg_data_loss:.6f}, "
                  f"Physics Loss: {avg_physics_loss:.6f}, "
                  f"Physics Weight: {current_physics_weight:.6f}")
        
        return metrics

    def validate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Validates model (physics loss not computed during validation)."""
        X_test, y_test = self._prepare_data(test_data)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test).squeeze()
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
            predictions_scaled = self.model(X_tensor).squeeze().cpu().numpy()
        
        # Inverse scale predictions
        predictions_scaled = predictions_scaled.reshape(-1, 1)
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
        """Loads model state with proper handling of sklearn objects."""
        from sklearn.preprocessing import StandardScaler
        
        # Add safe globals for sklearn objects
        torch.serialization.add_safe_globals([StandardScaler])
        
        # Load with weights_only=False since we have sklearn objects
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
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